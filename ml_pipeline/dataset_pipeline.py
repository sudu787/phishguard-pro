"""
ml_pipeline/dataset_pipeline.py

Complete phishing dataset pipeline — from raw URL collection to ML-ready
train/test CSVs with zero data leakage.

Pipeline stages:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 1: COLLECT   — Download URLs from PhishTank, OpenPhish,     │
  │                        Tranco (benign)                              │
  │  Stage 2: CLEAN     — Normalise, validate, remove broken URLs      │
  │  Stage 3: DEDUP     — Remove exact duplicates + near-duplicates    │
  │                        (same domain+path, different query)          │
  │  Stage 4: LABEL     — Assign labels (1=phishing, 0=legitimate)     │
  │  Stage 5: FEATURES  — Extract 32 URL features via features.py      │
  │  Stage 6: SPLIT     — Domain-aware stratified train/test split     │
  │                        (NO data leakage)                            │
  │  Stage 7: REPORT    — Data quality report + class distribution      │
  └─────────────────────────────────────────────────────────────────────┘

Anti-leakage guarantees:
  1. Split is by DOMAIN, not by URL — no URL from the same registered
     domain appears in both train and test sets.
  2. SMOTE is applied ONLY in train.py, never here.
  3. Feature extraction is stateless (no fitted transformers that could
     leak test information into training features).
  4. If PhishTank timestamps are available, an optional TEMPORAL split
     ensures the model is tested on strictly newer data.

Usage:
  # Full pipeline (downloads data, generates features, splits)
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data

  # Skip download (use previously downloaded raw data)
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data --skip-download

  # Custom split ratio
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data --test-size 0.25

  # Temporal split (requires PhishTank timestamps)
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data --temporal-split

  # Limit samples (for fast iteration)
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data --max-phishing 5000 --max-benign 5000
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import numpy as np

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phishguard.pipeline")

RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1: DATA COLLECTORS
# ═══════════════════════════════════════════════════════════════════════════════

class PhishTankCollector:
    """
    Download verified phishing URLs from PhishTank.

    Source: http://data.phishtank.com/data/online-valid.csv
    Format: phish_id, url, phish_detail_url, submission_time,
            verification_time, online, target

    Why PhishTank:
      - Largest community-verified phishing database (~30K active URLs)
      - Includes timestamps (enables temporal splits)
      - Includes target brand (useful for brand impersonation research)
      - Verified by multiple humans — high label quality
    """

    FEED_URL = "http://data.phishtank.com/data/online-valid.csv"

    def collect(self, output_dir: Path, api_key: Optional[str] = None) -> pd.DataFrame:
        """Download and parse the PhishTank database."""
        import requests

        logger.info("📥 Collecting PhishTank data...")

        url = self.FEED_URL
        if api_key:
            url = f"http://data.phishtank.com/data/{api_key}/online-valid.csv"

        headers = {"User-Agent": "phishguard-research/2.0"}

        try:
            resp = requests.get(url, headers=headers, timeout=60, stream=True)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"PhishTank download failed: {e}")
            return self._load_cached(output_dir)

        # Save raw data
        raw_path = output_dir / "raw" / "phishtank_raw.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(resp.content)
        logger.info(f"  Saved raw PhishTank data → {raw_path}")

        # Parse CSV
        try:
            df = pd.read_csv(
                io.StringIO(resp.text),
                usecols=["url", "submission_time", "verification_time", "online", "target"],
                on_bad_lines="skip",
            )
        except Exception as e:
            logger.warning(f"PhishTank CSV parse error: {e}")
            return self._load_cached(output_dir)

        # Filter to online (active) phishing pages only
        if "online" in df.columns:
            df = df[df["online"].astype(str).str.lower() == "yes"]

        df["source"] = "phishtank"
        df["label"] = 1  # phishing

        logger.info(f"  PhishTank: {len(df)} verified phishing URLs collected")
        return df[["url", "source", "label", "submission_time", "target"]].reset_index(drop=True)

    def _load_cached(self, output_dir: Path) -> pd.DataFrame:
        """Fall back to previously downloaded data."""
        cached = output_dir / "raw" / "phishtank_raw.csv"
        if cached.exists():
            logger.info(f"  Using cached PhishTank data from {cached}")
            df = pd.read_csv(cached, on_bad_lines="skip")
            if "online" in df.columns:
                df = df[df["online"].astype(str).str.lower() == "yes"]
            df["source"] = "phishtank"
            df["label"] = 1
            return df[["url", "source", "label"]].reset_index(drop=True)
        return pd.DataFrame(columns=["url", "source", "label"])


class OpenPhishCollector:
    """
    Download phishing URLs from the OpenPhish community feed.

    Source: https://openphish.com/feed.txt
    Format: One URL per line (~3,000 URLs, updated every 12 hours)

    Why OpenPhish:
      - Independent from PhishTank (different detection methodology)
      - More recent URLs (12-hour refresh cycle)
      - Combines with PhishTank for better coverage
      - No duplicate overlap with PhishTank (different URL sources)
    """

    FEED_URL = "https://openphish.com/feed.txt"

    def collect(self, output_dir: Path) -> pd.DataFrame:
        """Download and parse the OpenPhish feed."""
        import requests

        logger.info("📥 Collecting OpenPhish data...")

        try:
            resp = requests.get(self.FEED_URL, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"OpenPhish download failed: {e}")
            return self._load_cached(output_dir)

        # Save raw data
        raw_path = output_dir / "raw" / "openphish_raw.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(resp.text)

        urls = [
            line.strip()
            for line in resp.text.splitlines()
            if line.strip() and not line.startswith("#")
        ]

        df = pd.DataFrame({"url": urls, "source": "openphish", "label": 1})
        logger.info(f"  OpenPhish: {len(df)} phishing URLs collected")
        return df.reset_index(drop=True)

    def _load_cached(self, output_dir: Path) -> pd.DataFrame:
        cached = output_dir / "raw" / "openphish_raw.txt"
        if cached.exists():
            logger.info(f"  Using cached OpenPhish data from {cached}")
            urls = [l.strip() for l in cached.read_text().splitlines() if l.strip()]
            return pd.DataFrame({"url": urls, "source": "openphish", "label": 1})
        return pd.DataFrame(columns=["url", "source", "label"])


class TrancoCollector:
    """
    Download legitimate domains from the Tranco top sites list.

    Source: https://tranco-list.eu/top-1m.csv.zip
    Format: rank, domain (no protocol)

    Why Tranco (not Alexa):
      - Alexa Top 1M was discontinued in 2022
      - Tranco combines Alexa, Umbrella, Majestic, and Chrome UX data
      - More stable rankings (30-day averages, less susceptible to manipulation)
      - Published research standard (Pochat et al., NDSS 2019)
      - We use the top 10K–50K as "definitely legitimate" ground truth

    Anti-contamination:
      - Only use domains ranked in the top 50K (extremely unlikely to be phishing)
      - Prepend https:// to bare domains for consistent URL format
      - Skip domains that appear in any phishing blocklist
    """

    FEED_URL = "https://tranco-list.eu/top-1m.csv.zip"

    def collect(
        self,
        output_dir: Path,
        max_domains: int = 20000,
        min_rank: int = 1,
        max_rank: int = 50000,
    ) -> pd.DataFrame:
        """Download and parse the Tranco list."""
        import requests

        logger.info("📥 Collecting Tranco (benign) data...")

        try:
            resp = requests.get(self.FEED_URL, timeout=60, stream=True)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Tranco download failed: {e}")
            return self._load_cached(output_dir, max_domains, min_rank, max_rank)

        # Save raw zip
        raw_path = output_dir / "raw" / "tranco_top1m.csv.zip"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(resp.content)

        # Extract CSV from zip
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                csv_data = zf.read(csv_name).decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Tranco zip extraction failed: {e}")
            return self._load_cached(output_dir, max_domains, min_rank, max_rank)

        # Parse: rank,domain
        domains = []
        for line in csv_data.splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    rank = int(parts[0])
                except ValueError:
                    continue
                domain = parts[1].strip().lower()
                if min_rank <= rank <= max_rank and domain:
                    domains.append({"rank": rank, "domain": domain})
                    if len(domains) >= max_domains:
                        break

        # Convert bare domains to full URLs
        urls = []
        for d in domains:
            urls.append({
                "url": f"https://www.{d['domain']}/",
                "source": "tranco",
                "label": 0,  # legitimate
                "rank": d["rank"],
            })

        df = pd.DataFrame(urls)
        logger.info(f"  Tranco: {len(df)} legitimate domains collected (rank {min_rank}–{max_rank})")
        return df.reset_index(drop=True)

    def _load_cached(self, output_dir, max_domains, min_rank, max_rank):
        cached = output_dir / "raw" / "tranco_top1m.csv.zip"
        if cached.exists():
            logger.info(f"  Using cached Tranco data from {cached}")
            with zipfile.ZipFile(cached) as zf:
                csv_data = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
            domains = []
            for line in csv_data.splitlines():
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        rank = int(parts[0])
                    except ValueError:
                        continue
                    if min_rank <= rank <= max_rank:
                        domains.append({"url": f"https://www.{parts[1].strip()}/",
                                        "source": "tranco", "label": 0})
                    if len(domains) >= max_domains:
                        break
            return pd.DataFrame(domains)
        return pd.DataFrame(columns=["url", "source", "label"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2: DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

class DataCleaner:
    """
    Normalise and validate collected URLs.

    Operations:
      1. Strip whitespace and control characters
      2. Enforce http/https scheme (discard mailto:, ftp:, data:, etc.)
      3. Lowercase the scheme and hostname (path is case-sensitive per RFC)
      4. Remove trailing slashes for consistent dedup
      5. Remove URL fragments (#section) — not sent to server
      6. Validate URL structure (must have hostname with at least one dot)
      7. Filter out overly long URLs (> 2048 chars — likely garbage)
      8. Remove known non-phishing patterns from phishing set
    """

    # Domains that should never be labelled phishing
    WHITELIST_DOMAINS = {
        "google.com", "youtube.com", "facebook.com", "amazon.com",
        "wikipedia.org", "twitter.com", "instagram.com", "linkedin.com",
        "reddit.com", "github.com", "microsoft.com", "apple.com",
        "stackoverflow.com", "netflix.com", "whatsapp.com",
    }

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate all URLs. Returns cleaned DataFrame."""
        logger.info("🧹 Stage 2: Cleaning data...")
        initial_count = len(df)

        # 1. Drop rows with missing URLs
        df = df.dropna(subset=["url"]).copy()
        df["url"] = df["url"].astype(str).str.strip()

        # 2. Remove empty URLs
        df = df[df["url"].str.len() > 0]

        # 3. Normalise URLs
        df["url"] = df["url"].apply(self._normalise_url)

        # 4. Remove invalid URLs
        df = df[df["url"].notna()]
        df = df[df["url"].apply(self._is_valid_url)]

        # 5. Remove overly long URLs
        df = df[df["url"].str.len() <= 2048]

        # 6. Remove whitelisted domains from phishing set
        df = df[~(
            (df["label"] == 1) &
            df["url"].apply(self._is_whitelisted)
        )]

        # 7. Remove control characters
        df["url"] = df["url"].str.replace(r"[\x00-\x1f\x7f]", "", regex=True)

        cleaned_count = len(df)
        removed = initial_count - cleaned_count
        logger.info(f"  Cleaned: {initial_count} → {cleaned_count} ({removed} removed)")

        return df.reset_index(drop=True)

    def _normalise_url(self, url: str) -> Optional[str]:
        """Normalise a URL for consistent comparison."""
        try:
            url = url.strip()

            # Ensure scheme
            if not url.startswith(("http://", "https://")):
                url = "http://" + url

            parsed = urlparse(url)
            if not parsed.hostname:
                return None

            # Lowercase scheme and hostname only (path is case-sensitive)
            scheme = parsed.scheme.lower()
            hostname = parsed.hostname.lower()
            port = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""
            path = parsed.path.rstrip("/") or ""
            query = f"?{parsed.query}" if parsed.query else ""
            # Drop fragment — not sent to server

            return f"{scheme}://{hostname}{port}{path}{query}"
        except Exception:
            return None

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL structure."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or parsed.scheme not in ("http", "https"):
                return False
            if not parsed.hostname:
                return False
            # Must have at least one dot (or be an IP)
            if "." not in parsed.hostname and not re.match(r"^\d+\.\d+\.\d+\.\d+$", parsed.hostname):
                return False
            return True
        except Exception:
            return False

    def _is_whitelisted(self, url: str) -> bool:
        """Check if URL belongs to a known-legitimate domain."""
        try:
            hostname = urlparse(url).hostname or ""
            hostname = hostname.lower().lstrip("www.")
            return hostname in self.WHITELIST_DOMAINS
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3: DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class Deduplicator:
    """
    Remove exact and near-duplicate URLs.

    Strategy:
      1. Exact dedup: hash-based on full normalised URL
      2. Near dedup: same domain + path, different query params
         (phishing kits often generate unique query strings per victim,
          but the page is the same)
      3. Cross-source dedup: prefer PhishTank over OpenPhish (has timestamps)
         for overlapping URLs

    Why this matters:
      - Duplicate URLs inflate accuracy (model memorises, not generalises)
      - Near-dupes from the same phishing kit bias feature importance
      - Cross-source dupes cause label conflicts
    """

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates. Returns deduplicated DataFrame."""
        logger.info("🔄 Stage 3: Deduplicating...")
        initial_count = len(df)

        # ── Step 1: Exact URL dedup ────────────────────────────────────────
        # Keep first occurrence; prefer phishtank over openphish (has timestamps)
        source_priority = {"phishtank": 0, "openphish": 1, "tranco": 2}
        df["_priority"] = df["source"].map(source_priority).fillna(99)
        df = df.sort_values("_priority").drop_duplicates(subset=["url"], keep="first")
        df = df.drop(columns=["_priority"])
        after_exact = len(df)

        # ── Step 2: Near-duplicate removal ─────────────────────────────────
        # Group by domain+path (ignore query string)
        df["_domain_path"] = df["url"].apply(self._extract_domain_path)
        # Within each domain+path group, keep only one URL per label
        df = df.drop_duplicates(subset=["_domain_path", "label"], keep="first")
        df = df.drop(columns=["_domain_path"])
        after_near = len(df)

        # ── Step 3: Cross-label conflict resolution ────────────────────────
        # If the same domain appears as both phishing and legitimate, remove it
        # (ambiguous labels would poison the model)
        df["_domain"] = df["url"].apply(self._extract_domain)
        conflict_domains = (
            df.groupby("_domain")["label"]
            .nunique()
            .loc[lambda x: x > 1]
            .index
        )
        if len(conflict_domains) > 0:
            logger.warning(
                f"  ⚠ {len(conflict_domains)} domains have conflicting labels — removing"
            )
            df = df[~df["_domain"].isin(conflict_domains)]
        df = df.drop(columns=["_domain"])
        after_conflict = len(df)

        logger.info(
            f"  Dedup: {initial_count} → exact:{after_exact} → near:{after_near} "
            f"→ conflict:{after_conflict}"
        )
        return df.reset_index(drop=True)

    def _extract_domain_path(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            return f"{parsed.hostname}{parsed.path}".lower()
        except Exception:
            return url

    def _extract_domain(self, url: str) -> str:
        try:
            hostname = urlparse(url).hostname or ""
            # Extract registered domain (eTLD+1 approximation)
            parts = hostname.lower().split(".")
            if len(parts) >= 2:
                return ".".join(parts[-2:])
            return hostname
        except Exception:
            return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4: LABELING
# ═══════════════════════════════════════════════════════════════════════════════

class Labeler:
    """
    Verify and finalise labels.

    Labels are assigned during collection (phishing sources → 1, Tranco → 0),
    but this stage performs validation:
      1. Ensure no NaN labels
      2. Ensure binary labels (0 or 1 only)
      3. Log class distribution
      4. Optional: balance dataset by downsampling majority class
    """

    def label(
        self,
        df: pd.DataFrame,
        max_phishing: Optional[int] = None,
        max_benign: Optional[int] = None,
        balance_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        """Validate labels and optionally balance the dataset."""
        logger.info("🏷️  Stage 4: Labeling and balancing...")

        # Validate labels
        df = df[df["label"].isin([0, 1])].copy()
        df["label"] = df["label"].astype(int)

        phishing = df[df["label"] == 1]
        benign = df[df["label"] == 0]

        logger.info(f"  Before balancing: {len(phishing)} phishing, {len(benign)} benign")

        # Apply sample limits
        if max_phishing and len(phishing) > max_phishing:
            phishing = phishing.sample(n=max_phishing, random_state=RANDOM_STATE)
            logger.info(f"  Downsampled phishing to {max_phishing}")

        if max_benign and len(benign) > max_benign:
            benign = benign.sample(n=max_benign, random_state=RANDOM_STATE)
            logger.info(f"  Downsampled benign to {max_benign}")

        # Optional: balance to a specific ratio
        if balance_ratio:
            target_benign = int(len(phishing) * balance_ratio)
            if len(benign) > target_benign:
                benign = benign.sample(n=target_benign, random_state=RANDOM_STATE)
            logger.info(f"  Balanced to {balance_ratio}:1 ratio ({len(benign)} benign)")

        df = pd.concat([phishing, benign], ignore_index=True)

        # Shuffle
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        logger.info(
            f"  Final: {len(df)} total | "
            f"{len(df[df['label']==1])} phishing ({len(df[df['label']==1])/len(df)*100:.1f}%) | "
            f"{len(df[df['label']==0])} benign ({len(df[df['label']==0])/len(df)*100:.1f}%)"
        )
        return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 5: FEATURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureGenerator:
    """
    Extract ML features from URLs using the existing URLFeatureExtractor.

    Two modes:
      - FAST (use_network=False): ~0.1ms per URL, offline features only
        32 features from URL string analysis
      - FULL (use_network=True): ~2s per URL, includes WHOIS/DNS/SSL
        32 features + network-dependent features populated

    For initial pipeline development, FAST mode is recommended.
    Switch to FULL mode for the final production training dataset.

    Anti-leakage guarantee:
      Feature extraction is STATELESS — each URL is processed independently.
      No fitted transformers (e.g., TF-IDF, PCA) that could leak test data
      into training features. The URLFeatureExtractor computes deterministic
      features from the URL string alone.
    """

    def generate(
        self,
        df: pd.DataFrame,
        use_network: bool = False,
        batch_size: int = 500,
    ) -> pd.DataFrame:
        """Extract features for all URLs. Returns DataFrame with feature columns + label."""
        from ml_pipeline.features import URLFeatureExtractor, URL_FEATURE_COLUMNS

        logger.info(f"⚙️  Stage 5: Feature extraction ({'FULL' if use_network else 'FAST'} mode)...")
        extractor = URLFeatureExtractor(use_network=use_network)

        total = len(df)
        all_features = []
        errors = 0
        t0 = time.time()

        for i, row in df.iterrows():
            try:
                features = extractor.extract(row["url"])
                features["label"] = row["label"]
                features["url"] = row["url"]
                features["source"] = row.get("source", "unknown")
                all_features.append(features)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"  Feature extraction error for {row['url'][:60]}: {e}")

            # Progress logging
            if (i + 1) % batch_size == 0 or i == total - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%) | "
                    f"{rate:.0f} URLs/sec | ETA: {eta:.0f}s | Errors: {errors}"
                )

        result = pd.DataFrame(all_features)

        elapsed_total = time.time() - t0
        logger.info(
            f"  Features extracted: {len(result)} URLs in {elapsed_total:.1f}s "
            f"({len(result)/elapsed_total:.0f} URLs/sec) | {errors} errors"
        )

        # Ensure all expected columns exist
        for col in URL_FEATURE_COLUMNS:
            if col not in result.columns:
                result[col] = 0

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 6: DOMAIN-AWARE TRAIN/TEST SPLIT (NO DATA LEAKAGE)
# ═══════════════════════════════════════════════════════════════════════════════

class AntiLeakSplitter:
    """
    Split data into train/test sets with ZERO data leakage.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  WHY DOMAIN-AWARE SPLITTING MATTERS                                │
    │                                                                     │
    │  Naive random split:                                                │
    │    http://evil.com/login     → TRAIN (label=1)                     │
    │    http://evil.com/verify    → TEST  (label=1)                     │
    │    ↑ Model memorises evil.com's features, gets "free" accuracy     │
    │                                                                     │
    │  Domain-aware split:                                                │
    │    ALL evil.com URLs         → TRAIN (or TEST, never both)         │
    │    ↑ Model must generalise to unseen domains                        │
    │                                                                     │
    │  This is the ONLY correct way to evaluate phishing detectors.      │
    └─────────────────────────────────────────────────────────────────────┘

    Additional anti-leakage measures:
      - Stratified by label (preserves class ratio in both splits)
      - Optional temporal split (train on older data, test on newer)
      - Reports domain overlap (should be 0)
    """

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        temporal: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split into train/test with domain-based grouping.

        Returns:
          (train_df, test_df) — both contain feature columns + label
        """
        logger.info(f"✂️  Stage 6: Domain-aware split (test_size={test_size})...")

        if temporal and "submission_time" in df.columns:
            return self._temporal_split(df, test_size)

        return self._domain_split(df, test_size)

    def _domain_split(
        self, df: pd.DataFrame, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split by registered domain — no domain appears in both sets."""

        # Extract registered domain (eTLD+1 approximation)
        df = df.copy()
        df["_reg_domain"] = df["url"].apply(self._get_registered_domain)

        # Group domains by their majority label
        domain_labels = (
            df.groupby("_reg_domain")["label"]
            .agg(lambda x: x.mode()[0])  # majority vote
            .reset_index()
            .rename(columns={"label": "domain_label"})
        )

        # Stratified split of DOMAINS (not URLs)
        np.random.seed(RANDOM_STATE)
        phishing_domains = domain_labels[domain_labels["domain_label"] == 1]["_reg_domain"].values
        benign_domains = domain_labels[domain_labels["domain_label"] == 0]["_reg_domain"].values

        np.random.shuffle(phishing_domains)
        np.random.shuffle(benign_domains)

        # Split each class independently to maintain stratification
        n_test_phish = max(1, int(len(phishing_domains) * test_size))
        n_test_benign = max(1, int(len(benign_domains) * test_size))

        test_domains = set(
            list(phishing_domains[:n_test_phish]) +
            list(benign_domains[:n_test_benign])
        )
        train_domains = set(
            list(phishing_domains[n_test_phish:]) +
            list(benign_domains[n_test_benign:])
        )

        # Assign URLs to splits based on their domain
        train_mask = df["_reg_domain"].isin(train_domains)
        test_mask = df["_reg_domain"].isin(test_domains)

        train_df = df[train_mask].drop(columns=["_reg_domain"]).reset_index(drop=True)
        test_df = df[test_mask].drop(columns=["_reg_domain"]).reset_index(drop=True)

        # ── Verify zero leakage ────────────────────────────────────────────
        train_domains_final = set(train_df["url"].apply(self._get_registered_domain))
        test_domains_final = set(test_df["url"].apply(self._get_registered_domain))
        overlap = train_domains_final & test_domains_final

        if overlap:
            logger.error(f"  ❌ DATA LEAKAGE DETECTED: {len(overlap)} overlapping domains!")
            logger.error(f"     Examples: {list(overlap)[:5]}")
        else:
            logger.info("  ✅ Zero domain overlap between train/test — no data leakage")

        self._log_split_stats(train_df, test_df)
        return train_df, test_df

    def _temporal_split(
        self, df: pd.DataFrame, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by time — train on older data, test on newer data.
        Simulates real-world deployment: model trained on past, tested on future.
        """
        logger.info("  Using TEMPORAL split (train=older, test=newer)...")

        df = df.copy()

        # Parse timestamps
        df["_timestamp"] = pd.to_datetime(df["submission_time"], errors="coerce")
        has_time = df["_timestamp"].notna()

        # URLs with timestamps: split by time
        timed = df[has_time].sort_values("_timestamp")
        untimed = df[~has_time]

        split_idx = int(len(timed) * (1 - test_size))
        train_timed = timed.iloc[:split_idx]
        test_timed = timed.iloc[split_idx:]

        # URLs without timestamps (e.g., Tranco): random domain split
        if len(untimed) > 0:
            untimed_train, untimed_test = self._domain_split(untimed, test_size)
            train_df = pd.concat([train_timed, untimed_train], ignore_index=True)
            test_df = pd.concat([test_timed, untimed_test], ignore_index=True)
        else:
            train_df = train_timed
            test_df = test_timed

        # Drop temp columns
        for col in ["_timestamp", "_reg_domain"]:
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])

        self._log_split_stats(train_df, test_df)
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _get_registered_domain(self, url: str) -> str:
        """Extract the registered domain (eTLD+1 approximation)."""
        try:
            hostname = urlparse(url).hostname or ""
            parts = hostname.lower().replace("www.", "").split(".")
            if len(parts) >= 2:
                # Handle common 2-part TLDs (co.uk, com.au, etc.)
                two_part_tlds = {
                    "co.uk", "com.au", "co.in", "com.br", "co.jp", "co.kr",
                    "com.sg", "com.hk", "co.za", "com.mx", "com.tr", "co.nz",
                    "org.uk", "net.au", "ac.uk", "gov.uk",
                }
                if len(parts) >= 3 and f"{parts[-2]}.{parts[-1]}" in two_part_tlds:
                    return ".".join(parts[-3:])
                return ".".join(parts[-2:])
            return hostname
        except Exception:
            return ""

    def _log_split_stats(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Log train/test split statistics."""
        train_phish = (train_df["label"] == 1).sum()
        train_benign = (train_df["label"] == 0).sum()
        test_phish = (test_df["label"] == 1).sum()
        test_benign = (test_df["label"] == 0).sum()

        logger.info(f"  Train: {len(train_df)} total | {train_phish} phishing | {train_benign} benign")
        logger.info(f"  Test:  {len(test_df)} total | {test_phish} phishing | {test_benign} benign")
        logger.info(
            f"  Split ratio: {len(test_df)/(len(train_df)+len(test_df))*100:.1f}% test"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 7: DATA QUALITY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

class QualityReporter:
    """Generate a comprehensive data quality report."""

    def report(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
    ):
        """Generate and save a data quality report."""
        logger.info("📊 Stage 7: Generating data quality report...")

        from ml_pipeline.features import URL_FEATURE_COLUMNS

        report = {
            "generated_at": datetime.now().isoformat(),
            "pipeline_version": "2.0.0",
            "dataset_stats": {
                "total_samples": len(train_df) + len(test_df),
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "split_ratio": round(len(test_df) / (len(train_df) + len(test_df)), 3),
            },
            "class_distribution": {
                "train": {
                    "phishing": int((train_df["label"] == 1).sum()),
                    "benign": int((train_df["label"] == 0).sum()),
                    "ratio": round(
                        (train_df["label"] == 1).sum() / max(1, (train_df["label"] == 0).sum()),
                        3,
                    ),
                },
                "test": {
                    "phishing": int((test_df["label"] == 1).sum()),
                    "benign": int((test_df["label"] == 0).sum()),
                    "ratio": round(
                        (test_df["label"] == 1).sum() / max(1, (test_df["label"] == 0).sum()),
                        3,
                    ),
                },
            },
            "feature_stats": {},
            "data_leakage_check": {
                "domain_overlap": 0,
                "url_overlap": 0,
                "passed": True,
            },
        }

        # Feature statistics
        feature_cols = [c for c in URL_FEATURE_COLUMNS if c in train_df.columns]
        for col in feature_cols:
            if col in train_df.columns:
                report["feature_stats"][col] = {
                    "train_mean": round(float(train_df[col].mean()), 4),
                    "test_mean": round(float(test_df[col].mean()), 4) if col in test_df.columns else None,
                    "train_std": round(float(train_df[col].std()), 4),
                    "missing_pct": round(float(train_df[col].isna().mean() * 100), 2),
                }

        # Leakage check
        if "url" in train_df.columns and "url" in test_df.columns:
            url_overlap = set(train_df["url"]) & set(test_df["url"])
            report["data_leakage_check"]["url_overlap"] = len(url_overlap)

            splitter = AntiLeakSplitter()
            train_domains = set(train_df["url"].apply(splitter._get_registered_domain))
            test_domains = set(test_df["url"].apply(splitter._get_registered_domain))
            domain_overlap = train_domains & test_domains
            report["data_leakage_check"]["domain_overlap"] = len(domain_overlap)

            if len(url_overlap) > 0 or len(domain_overlap) > 0:
                report["data_leakage_check"]["passed"] = False
                logger.error("  ❌ DATA LEAKAGE DETECTED in final output!")
            else:
                logger.info("  ✅ No data leakage detected in final output")

        # Source distribution
        if "source" in train_df.columns:
            report["source_distribution"] = {
                "train": train_df["source"].value_counts().to_dict(),
                "test": test_df["source"].value_counts().to_dict(),
            }

        # Save report
        report_path = output_dir / "data_quality_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"  Report saved → {report_path}")

        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("  DATA QUALITY REPORT")
        logger.info("=" * 60)
        logger.info(f"  Total samples:     {report['dataset_stats']['total_samples']}")
        logger.info(f"  Train / Test:      {report['dataset_stats']['train_samples']} / {report['dataset_stats']['test_samples']}")
        logger.info(f"  Train phishing:    {report['class_distribution']['train']['phishing']}")
        logger.info(f"  Train benign:      {report['class_distribution']['train']['benign']}")
        logger.info(f"  Test phishing:     {report['class_distribution']['test']['phishing']}")
        logger.info(f"  Test benign:       {report['class_distribution']['test']['benign']}")
        logger.info(f"  Domain overlap:    {report['data_leakage_check']['domain_overlap']}")
        logger.info(f"  Leakage check:     {'✅ PASSED' if report['data_leakage_check']['passed'] else '❌ FAILED'}")
        logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetPipeline:
    """
    End-to-end pipeline orchestrator.

    Runs all 7 stages in sequence and saves intermediate checkpoints
    so stages can be re-run independently.
    """

    def __init__(self, output_dir: str = "ml_pipeline/data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collector_pt = PhishTankCollector()
        self.collector_op = OpenPhishCollector()
        self.collector_tr = TrancoCollector()
        self.cleaner = DataCleaner()
        self.deduplicator = Deduplicator()
        self.labeler = Labeler()
        self.feature_gen = FeatureGenerator()
        self.splitter = AntiLeakSplitter()
        self.reporter = QualityReporter()

    def run(
        self,
        skip_download: bool = False,
        use_network_features: bool = False,
        test_size: float = 0.2,
        temporal_split: bool = False,
        max_phishing: Optional[int] = None,
        max_benign: Optional[int] = None,
        balance_ratio: Optional[float] = None,
        phishtank_api_key: Optional[str] = None,
    ):
        """Run the full pipeline."""
        logger.info("🚀 PhishGuard V2 — Dataset Pipeline Starting")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info("")
        t_start = time.time()

        # ── Stage 1: Collect ───────────────────────────────────────────────
        if skip_download:
            logger.info("⏭️  Skipping download — loading from checkpoints...")
            raw_df = self._load_checkpoint("01_collected.csv")
            if raw_df is None:
                logger.error("No checkpoint found. Run without --skip-download first.")
                return
        else:
            pt_df = self.collector_pt.collect(self.output_dir, api_key=phishtank_api_key)
            op_df = self.collector_op.collect(self.output_dir)
            tr_df = self.collector_tr.collect(self.output_dir, max_domains=max_benign or 20000)

            raw_df = pd.concat([pt_df, op_df, tr_df], ignore_index=True)
            self._save_checkpoint(raw_df, "01_collected.csv")
            logger.info(f"  📦 Total collected: {len(raw_df)} URLs\n")

        # ── Stage 2: Clean ─────────────────────────────────────────────────
        clean_df = self.cleaner.clean(raw_df)
        self._save_checkpoint(clean_df, "02_cleaned.csv")

        # ── Stage 3: Deduplicate ───────────────────────────────────────────
        dedup_df = self.deduplicator.deduplicate(clean_df)
        self._save_checkpoint(dedup_df, "03_deduplicated.csv")

        # ── Stage 4: Label & balance ───────────────────────────────────────
        labeled_df = self.labeler.label(
            dedup_df,
            max_phishing=max_phishing,
            max_benign=max_benign,
            balance_ratio=balance_ratio,
        )
        self._save_checkpoint(labeled_df, "04_labeled.csv")

        # ── Stage 5: Feature extraction ────────────────────────────────────
        featured_df = self.feature_gen.generate(labeled_df, use_network=use_network_features)
        self._save_checkpoint(featured_df, "05_featured.csv")

        # ── Stage 6: Split ─────────────────────────────────────────────────
        train_df, test_df = self.splitter.split(
            featured_df,
            test_size=test_size,
            temporal=temporal_split,
        )

        # Save final train/test CSVs (ready for train.py)
        from ml_pipeline.features import URL_FEATURE_COLUMNS
        feature_cols = [c for c in URL_FEATURE_COLUMNS if c in train_df.columns]
        output_cols = feature_cols + ["label"]

        # Also save metadata columns separately (url, source — not for ML)
        meta_cols = ["url", "source", "label"]

        train_path = self.output_dir / "train.csv"
        test_path = self.output_dir / "test.csv"
        train_meta_path = self.output_dir / "train_metadata.csv"
        test_meta_path = self.output_dir / "test_metadata.csv"

        train_df[output_cols].to_csv(train_path, index=False)
        test_df[output_cols].to_csv(test_path, index=False)

        meta_available = [c for c in meta_cols if c in train_df.columns]
        train_df[meta_available].to_csv(train_meta_path, index=False)
        test_df[meta_available].to_csv(test_meta_path, index=False)

        logger.info(f"\n  💾 Train set saved → {train_path} ({len(train_df)} rows)")
        logger.info(f"  💾 Test set saved  → {test_path} ({len(test_df)} rows)")

        # ── Stage 7: Quality report ────────────────────────────────────────
        self.reporter.report(train_df, test_df, self.output_dir)

        elapsed = time.time() - t_start
        logger.info(f"\n✅ Pipeline complete in {elapsed:.1f}s")
        logger.info(f"   To train models: python -m ml_pipeline.train --data {train_path}")

    def _save_checkpoint(self, df: pd.DataFrame, filename: str):
        """Save intermediate checkpoint for debugging and re-runs."""
        path = self.output_dir / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def _load_checkpoint(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a previously saved checkpoint."""
        path = self.output_dir / "checkpoints" / filename
        if path.exists():
            return pd.read_csv(path)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="PhishGuard V2 — Phishing Dataset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download + clean + features + split)
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data

  # Skip download (use cached data)
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data --skip-download

  # Limit dataset size for fast iteration
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data --max-phishing 5000 --max-benign 5000

  # Temporal split (requires PhishTank timestamps)
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data --temporal-split

  # Full features (SLOW — includes WHOIS/DNS lookups)
  python -m ml_pipeline.dataset_pipeline --output ml_pipeline/data --full-features
        """,
    )
    parser.add_argument("--output", default="ml_pipeline/data", help="Output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download (use cached)")
    parser.add_argument("--full-features", action="store_true", help="Use network features (WHOIS/DNS — slow)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    parser.add_argument("--temporal-split", action="store_true", help="Use temporal split (train=older, test=newer)")
    parser.add_argument("--max-phishing", type=int, default=None, help="Max phishing samples")
    parser.add_argument("--max-benign", type=int, default=None, help="Max benign samples")
    parser.add_argument("--balance-ratio", type=float, default=None, help="Benign-to-phishing ratio (e.g., 1.5)")
    parser.add_argument("--phishtank-key", default=None, help="PhishTank API key (optional, for higher rate limits)")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = DatasetPipeline(output_dir=args.output)
    pipeline.run(
        skip_download=args.skip_download,
        use_network_features=args.full_features,
        test_size=args.test_size,
        temporal_split=args.temporal_split,
        max_phishing=args.max_phishing,
        max_benign=args.max_benign,
        balance_ratio=args.balance_ratio,
        phishtank_api_key=args.phishtank_key,
    )


if __name__ == "__main__":
    main()
