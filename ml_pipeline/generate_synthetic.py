"""
ml_pipeline/generate_synthetic.py

Generate a synthetic phishing dataset for local development and testing.

Why this exists:
  - PhishTank/OpenPhish require network access and rate limits
  - Researchers need a dataset to iterate on features/models quickly
  - CI/CD pipelines need deterministic test data
  - The synthetic data mimics real-world phishing URL distributions

The generator creates URLs with realistic feature distributions based on
published phishing research (Sahingoz 2019, Rao & Pais 2019).

Usage:
  python -m ml_pipeline.generate_synthetic --output ml_pipeline/data/synthetic.csv --count 10000
  python -m ml_pipeline.generate_synthetic --output ml_pipeline/data/synthetic.csv --count 10000 --phishing-ratio 0.4
"""

import argparse
import csv
import hashlib
import logging
import random
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phishguard.synthetic")

RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  Legitimate URL Generator
# ═══════════════════════════════════════════════════════════════════════════════

# Top 200 domains from Tranco (realistic benign distribution)
LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "github.com",
    "microsoft.com", "apple.com", "netflix.com", "stackoverflow.com", "whatsapp.com",
    "wordpress.com", "pinterest.com", "ebay.com", "dropbox.com", "tumblr.com",
    "cnn.com", "bbc.co.uk", "nytimes.com", "medium.com", "quora.com",
    "zoom.us", "office.com", "live.com", "bing.com", "yahoo.com",
    "adobe.com", "spotify.com", "paypal.com", "twitch.tv", "salesforce.com",
    "shopify.com", "stripe.com", "slack.com", "notion.so", "figma.com",
    "cloudflare.com", "aws.amazon.com", "heroku.com", "digitalocean.com",
    "gitlab.com", "bitbucket.org", "npmjs.com", "pypi.org", "docker.com",
    "ubuntu.com", "archlinux.org", "fedoraproject.org", "debian.org",
    "coursera.org", "udemy.com", "khanacademy.org", "edx.org",
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "economist.com",
    "github.io", "vercel.app", "netlify.app", "pages.dev",
    "docs.google.com", "drive.google.com", "mail.google.com", "maps.google.com",
    "play.google.com", "cloud.google.com", "firebase.google.com",
    "outlook.com", "teams.microsoft.com", "onedrive.live.com",
    "music.apple.com", "icloud.com", "developer.apple.com",
    "hdfcbank.com", "onlinesbi.com", "icicibank.com", "axisbank.com",
    "chase.com", "wellsfargo.com", "bankofamerica.com", "citibank.com",
]

LEGIT_PATHS = [
    "/", "/about", "/contact", "/help", "/support", "/blog", "/news",
    "/products", "/services", "/pricing", "/careers", "/team",
    "/docs", "/api", "/developer", "/resources", "/faq",
    "/search", "/explore", "/trending", "/popular",
    "/settings", "/account", "/profile", "/dashboard",
    "/privacy", "/terms", "/legal", "/security",
    "/en", "/en-us", "/en-gb", "/es", "/fr", "/de",
    "/2024/01/article-title", "/blog/post-name",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Phishing URL Generator
# ═══════════════════════════════════════════════════════════════════════════════

# Brands targeted by phishing attacks
TARGET_BRANDS = [
    "paypal", "google", "microsoft", "apple", "amazon", "netflix",
    "facebook", "instagram", "linkedin", "twitter", "chase", "wellsfargo",
    "bankofamerica", "hdfc", "sbi", "icici", "dropbox", "github",
    "whatsapp", "steam", "adobe", "yahoo", "outlook", "spotify",
]

# Suspicious TLDs used by phishing domains
PHISHING_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".buzz",
    ".club", ".work", ".click", ".link", ".info", ".support",
    ".review", ".stream", ".win", ".bid", ".download",
    ".loan", ".men", ".party", ".science", ".trade",
]

# Cheap hosting TLDs (semi-suspicious)
SEMI_SUSPICIOUS_TLDS = [
    ".com", ".net", ".org", ".site", ".online", ".space", ".fun",
    ".icu", ".monster", ".rest", ".cfd", ".sbs",
]

# Phishing keywords for paths
PHISHING_PATH_KEYWORDS = [
    "login", "signin", "sign-in", "log-in", "verify", "verification",
    "confirm", "update", "secure", "account", "banking", "password",
    "credential", "authenticate", "validate", "wallet", "recover",
    "reactivate", "suspend", "restrict", "unlock", "urgent",
]


class SyntheticGenerator:
    """Generate realistic synthetic phishing and legitimate URLs."""

    def __init__(self, seed: int = RANDOM_STATE):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_dataset(
        self,
        total_count: int = 10000,
        phishing_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """
        Generate a balanced synthetic dataset.

        Args:
            total_count: Total number of URLs to generate
            phishing_ratio: Fraction of URLs that are phishing (0.0–1.0)

        Returns:
            DataFrame with columns: url, label, source, phishing_technique
        """
        n_phishing = int(total_count * phishing_ratio)
        n_benign = total_count - n_phishing

        logger.info(f"Generating {total_count} synthetic URLs ({n_phishing} phishing, {n_benign} benign)")

        # Generate phishing URLs using different techniques
        phishing_urls = []
        techniques = [
            (self._gen_typosquatting, 0.25),     # 25% typosquatting
            (self._gen_subdomain_abuse, 0.20),   # 20% subdomain abuse
            (self._gen_ip_address, 0.10),        # 10% IP address
            (self._gen_keyword_stuffing, 0.15),  # 15% keyword stuffing
            (self._gen_path_mimicry, 0.10),      # 10% path mimicry
            (self._gen_url_shortener_style, 0.05),  # 5% shortener-like
            (self._gen_random_domain, 0.10),     # 10% random/DGA domains
            (self._gen_https_deception, 0.05),   # 5% HTTPS in domain
        ]

        for gen_func, fraction in techniques:
            count = int(n_phishing * fraction)
            for _ in range(count):
                url, technique = gen_func()
                phishing_urls.append({
                    "url": url,
                    "label": 1,
                    "source": "synthetic_phishing",
                    "phishing_technique": technique,
                })

        # Fill remaining with random technique mix
        while len(phishing_urls) < n_phishing:
            gen_func, _ = self.rng.choice(techniques)
            url, technique = gen_func()
            phishing_urls.append({
                "url": url,
                "label": 1,
                "source": "synthetic_phishing",
                "phishing_technique": technique,
            })

        # Generate legitimate URLs
        benign_urls = []
        for _ in range(n_benign):
            if self.rng.random() < 0.25:
                url = self._gen_search_engine_query()
            else:
                url = self._gen_legitimate()
            benign_urls.append({
                "url": url,
                "label": 0,
                "source": "synthetic_benign",
                "phishing_technique": "none",
            })

        # Combine and shuffle
        all_data = phishing_urls + benign_urls
        self.rng.shuffle(all_data)

        df = pd.DataFrame(all_data)
        logger.info(f"Generated {len(df)} URLs")

        # Log technique distribution
        if "phishing_technique" in df.columns:
            tech_dist = df[df["label"] == 1]["phishing_technique"].value_counts()
            logger.info(f"  Phishing techniques:\n{tech_dist.to_string()}")

        return df

    # ─── Legitimate URL generation ────────────────────────────────────────

    def _gen_legitimate(self) -> str:
        domain = self.rng.choice(LEGIT_DOMAINS)
        path = self.rng.choice(LEGIT_PATHS)
        scheme = "https"
        www = "www." if self.rng.random() > 0.4 else ""
        return f"{scheme}://{www}{domain}{path}"

    def _gen_search_engine_query(self) -> str:
        """Generate long, messy, but completely legitimate search engine URLs to fix false positives."""
        search_engines = ["google.com/search", "bing.com/search", "amazon.com/s", "youtube.com/results"]
        base = self.rng.choice(search_engines)
        query = self._random_word(4, 10) + "+" + self._random_word(4, 10)
        # Add lots of messy parameters common in safe tracking links
        params = f"?q={query}&rlz=1C1CHBF_enUS&oq={query}&aqs=chrome.0.69i59j0i512l9.3134j0j7&sourceid=chrome&ie=UTF-8"
        return f"https://www.{base}{params}"

    # ─── Phishing URL generation techniques ───────────────────────────────

    def _gen_typosquatting(self) -> tuple[str, str]:
        """
        Technique: Typosquatting (Levenshtein distance 1-2 from real brand)
        Example: paypa1.com, g00gle.com, arnazon.com
        Why attackers use it: Exploits typos and visual similarity
        """
        brand = self.rng.choice(TARGET_BRANDS)
        mutation = self._mutate_brand(brand)
        tld = self.rng.choice(PHISHING_TLDS + SEMI_SUSPICIOUS_TLDS)
        scheme = self.rng.choice(["http", "https"])
        path = self._gen_phishing_path()
        return f"{scheme}://{mutation}{tld}{path}", "typosquatting"

    def _gen_subdomain_abuse(self) -> tuple[str, str]:
        """
        Technique: Brand name in subdomain
        Example: paypal.com.secure-login.evil.xyz
        Why attackers use it: Left-to-right reading bias makes users
        think they're on the real site
        """
        brand = self.rng.choice(TARGET_BRANDS)
        fake_domain = self._random_word(6, 12) + self.rng.choice(PHISHING_TLDS)
        separators = ["-", ".", "-secure.", "-login.", ".verify."]
        sep = self.rng.choice(separators)
        subdomain = f"{brand}.com{sep}"
        scheme = self.rng.choice(["http", "https"])
        path = self._gen_phishing_path()
        return f"{scheme}://{subdomain}{fake_domain}{path}", "subdomain_abuse"

    def _gen_ip_address(self) -> tuple[str, str]:
        """
        Technique: Raw IP address as hostname
        Example: http://192.168.45.201/login
        Why attackers use it: No domain name = no reputation check
        """
        ip = f"{self.rng.randint(1,223)}.{self.rng.randint(0,255)}.{self.rng.randint(0,255)}.{self.rng.randint(1,254)}"
        port = self.rng.choice(["", ":8080", ":443", ":8443", ":3000"])
        path = self._gen_phishing_path()
        return f"http://{ip}{port}{path}", "ip_address"

    def _gen_keyword_stuffing(self) -> tuple[str, str]:
        """
        Technique: Multiple phishing keywords in URL
        Example: secure-login-verify-account.evil.xyz/update-password
        Why attackers use it: Keywords create false sense of legitimacy
        """
        keywords = self.rng.sample(PHISHING_PATH_KEYWORDS, k=self.rng.randint(2, 4))
        domain = "-".join(keywords) + self.rng.choice(PHISHING_TLDS)
        scheme = self.rng.choice(["http", "https"])
        path = "/" + "/".join(self.rng.sample(PHISHING_PATH_KEYWORDS, k=2))
        return f"{scheme}://{domain}{path}", "keyword_stuffing"

    def _gen_path_mimicry(self) -> tuple[str, str]:
        """
        Technique: Legitimate-looking path on suspicious domain
        Example: https://evil.xyz/paypal/login/verify.html
        Why attackers use it: The path looks official even on a random domain
        """
        brand = self.rng.choice(TARGET_BRANDS)
        domain = self._random_word(5, 10) + self.rng.choice(PHISHING_TLDS)
        actions = self.rng.sample(["login", "verify", "secure", "update", "confirm"], k=2)
        ext = self.rng.choice(["", ".html", ".php", ".asp"])
        path = f"/{brand}/{'/'.join(actions)}{ext}"
        return f"https://{domain}{path}", "path_mimicry"

    def _gen_url_shortener_style(self) -> tuple[str, str]:
        """
        Technique: Short, opaque URL hiding the real destination
        Example: https://bit.ly/3xK9mF2
        Why attackers use it: Hides the true destination domain
        """
        shorteners = ["bit.ly", "tinyurl.com", "t.co", "cutt.ly", "rb.gy", "is.gd"]
        shortener = self.rng.choice(shorteners)
        code = "".join(self.rng.choices(string.ascii_letters + string.digits, k=self.rng.randint(5, 8)))
        return f"https://{shortener}/{code}", "url_shortener"

    def _gen_random_domain(self) -> tuple[str, str]:
        """
        Technique: Algorithmically generated domain (DGA-style)
        Example: http://xk8f3m2p9q.xyz/login
        Why attackers use it: Cheap, disposable, hard to blocklist
        """
        length = self.rng.randint(8, 16)
        chars = string.ascii_lowercase + string.digits
        domain = "".join(self.rng.choices(chars, k=length))
        tld = self.rng.choice(PHISHING_TLDS)
        path = self._gen_phishing_path()
        return f"http://{domain}{tld}{path}", "random_dga"

    def _gen_https_deception(self) -> tuple[str, str]:
        """
        Technique: Security words in domain name
        Example: http://https-secure-paypal.login.xyz/
        Why attackers use it: "https" and "secure" in the domain
        psychologically reassure victims
        """
        brand = self.rng.choice(TARGET_BRANDS)
        prefix = self.rng.choice(["https-", "secure-", "ssl-", "safe-"])
        suffix = self.rng.choice(["-login", "-verify", "-secure", "-update"])
        tld = self.rng.choice(PHISHING_TLDS)
        path = self._gen_phishing_path()
        return f"http://{prefix}{brand}{suffix}{tld}{path}", "https_deception"

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _mutate_brand(self, brand: str) -> str:
        """Apply a random character mutation to a brand name."""
        mutations = [
            lambda b: b[:self.rng.randint(1, len(b)-1)] + self.rng.choice("0123456789") + b[self.rng.randint(1, len(b)-1)+1:],  # digit substitution
            lambda b: b.replace("l", "1") if "l" in b else b + "1",  # l→1
            lambda b: b.replace("o", "0") if "o" in b else b + "0",  # o→0
            lambda b: b.replace("a", "4") if "a" in b else b + "4",  # a→4
            lambda b: b + self.rng.choice(["l", "s", "x", "z"]),     # append char
            lambda b: b[:-1] if len(b) > 3 else b + "x",            # remove last char
            lambda b: b[:2] + b[2] + b[2:],                          # double a char
            lambda b: b + "-" + self.rng.choice(["secure", "login", "verify"]),  # append keyword
        ]
        return self.rng.choice(mutations)(brand)

    def _gen_phishing_path(self) -> str:
        """Generate a realistic phishing URL path."""
        patterns = [
            lambda: "/" + self.rng.choice(PHISHING_PATH_KEYWORDS),
            lambda: "/" + self.rng.choice(PHISHING_PATH_KEYWORDS) + ".html",
            lambda: "/" + self.rng.choice(PHISHING_PATH_KEYWORDS) + ".php",
            lambda: "/" + "/".join(self.rng.sample(PHISHING_PATH_KEYWORDS, k=2)),
            lambda: "/" + self.rng.choice(PHISHING_PATH_KEYWORDS) + "?" + "id=" + str(self.rng.randint(10000, 99999)),
            lambda: "/",
        ]
        return self.rng.choice(patterns)()

    def _random_word(self, min_len: int, max_len: int) -> str:
        length = self.rng.randint(min_len, max_len)
        return "".join(self.rng.choices(string.ascii_lowercase, k=length))


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Extraction (Integrated)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_featured_dataset(
    total_count: int = 10000,
    phishing_ratio: float = 0.5,
    output_path: str = "ml_pipeline/data/synthetic.csv",
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate a complete synthetic dataset with features extracted.

    This is the main entry point for creating a ready-to-train dataset
    without any network access.
    """
    from ml_pipeline.features import URLFeatureExtractor, URL_FEATURE_COLUMNS

    gen = SyntheticGenerator(seed=seed)
    raw_df = gen.generate_dataset(total_count, phishing_ratio)

    logger.info(f"Extracting features for {len(raw_df)} URLs...")
    extractor = URLFeatureExtractor()

    features_list = []
    errors = 0

    for i, row in raw_df.iterrows():
        try:
            features = extractor.extract(row["url"])
            features["label"] = row["label"]
            features["url"] = row["url"]
            features["source"] = row["source"]
            features_list.append(features)
        except Exception:
            errors += 1

        if (i + 1) % 1000 == 0:
            logger.info(f"  Progress: {i+1}/{len(raw_df)} ({errors} errors)")

    featured_df = pd.DataFrame(features_list)

    # Ensure all columns exist
    for col in URL_FEATURE_COLUMNS:
        if col not in featured_df.columns:
            featured_df[col] = 0

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full dataset (features + label)
    output_cols = URL_FEATURE_COLUMNS + ["label", "url"]
    featured_df[output_cols].to_csv(out_path, index=False)
    logger.info(f"💾 Saved featured dataset → {out_path} ({len(featured_df)} rows)")

    # Also do a domain-aware split
    from ml_pipeline.dataset_pipeline import AntiLeakSplitter
    splitter = AntiLeakSplitter()
    train_df, test_df = splitter.split(featured_df, test_size=0.2)

    train_path = out_path.parent / "synthetic_train.csv"
    test_path = out_path.parent / "synthetic_test.csv"
    train_df[output_cols].to_csv(train_path, index=False)
    test_df[output_cols].to_csv(test_path, index=False)

    logger.info(f"💾 Train split → {train_path} ({len(train_df)} rows)")
    logger.info(f"💾 Test split  → {test_path} ({len(test_df)} rows)")

    return featured_df


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic phishing dataset for development & testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10K URLs with features (default)
  python -m ml_pipeline.generate_synthetic

  # Custom size and ratio
  python -m ml_pipeline.generate_synthetic --count 50000 --phishing-ratio 0.4

  # Quick test dataset
  python -m ml_pipeline.generate_synthetic --count 1000 --output ml_pipeline/data/test_small.csv

  # Generate and immediately train
  python -m ml_pipeline.generate_synthetic --count 10000
  python -m ml_pipeline.train --data ml_pipeline/data/synthetic_train.csv --output ml_pipeline/models
        """,
    )
    parser.add_argument("--output", default="ml_pipeline/data/synthetic.csv", help="Output CSV path")
    parser.add_argument("--count", type=int, default=10000, help="Total number of URLs")
    parser.add_argument("--phishing-ratio", type=float, default=0.5, help="Fraction of phishing URLs (0.0–1.0)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")
    parser.add_argument("--raw-only", action="store_true", help="Generate URLs only (no feature extraction)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.raw_only:
        gen = SyntheticGenerator(seed=args.seed)
        df = gen.generate_dataset(args.count, args.phishing_ratio)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"💾 Raw URLs saved → {out}")
    else:
        generate_featured_dataset(
            total_count=args.count,
            phishing_ratio=args.phishing_ratio,
            output_path=args.output,
            seed=args.seed,
        )

    logger.info("✅ Synthetic dataset generation complete!")


if __name__ == "__main__":
    main()
