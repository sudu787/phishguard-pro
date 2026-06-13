"""
ml_pipeline/features.py

Advanced feature extraction combining:
  - 30 URL-string features (lexical, structural, DNS-based)
  - 26 DOM/HTML contextual features (extracted by content.js)

Both feature sets are documented in-line.
The URLFeatureExtractor class is used at both training time and inference time,
ensuring the exact same feature computation in all environments.
"""

import re
import math
import socket
import ssl
import logging
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional

import ipaddress

# ─── Lazy imports for network-dependent packages ─────────────────────────────
# These are only needed when use_network=True.  Importing them at module level
# causes crashes in offline environments (Chrome extension build, CI, etc.).
whois = None
requests = None
google_search = None

def _ensure_network_deps():
    """Import network-dependent packages on first use."""
    global whois, requests, google_search
    if whois is None:
        try:
            import whois as _whois
            whois = _whois
        except ImportError:
            logger.warning("python-whois not installed — WHOIS features disabled")
    if requests is None:
        try:
            import requests as _requests
            requests = _requests
        except ImportError:
            logger.warning("requests not installed — network features disabled")
    if google_search is None:
        try:
            from googlesearch import search as _gs
            google_search = _gs
        except ImportError:
            logger.warning("googlesearch-python not installed — Google index feature disabled")

logger = logging.getLogger("phishguard.features")

# ─── Canonical feature column orders ─────────────────────────────────────────
# These MUST match the training column order.  Change in both places or not at all.
URL_FEATURE_COLUMNS = [
    "having_ip",
    "have_at_sign",
    "url_length",
    "url_depth",
    "redirection",
    "https_in_domain",
    "tiny_url",
    "prefix_suffix",
    "sub_domains",
    "ssl_final_state",
    "domain_registration_length",
    "favicon",
    "port",
    "https_token",
    "request_url",
    "url_of_anchor",
    "links_in_tags",
    "sfh",
    "submitting_to_email",
    "abnormal_url",
    "redirect",
    "on_mouseover",
    "right_click",
    "popup_window",
    "iframe",
    "age_of_domain",
    "dns_record",
    "web_traffic",
    "page_rank",
    "google_index",
    "links_pointing_to_page",
    "statistical_report",
    "shannon_entropy",
    "vowel_consonant_ratio",
    "path_to_url_ratio",
    "digit_ratio",
    "special_char_ratio",
    "is_punycode",
    "keyword_match_score",
]

DOM_FEATURE_COLUMNS = [
    "num_forms",
    "has_external_form_action",
    "has_empty_form_action",
    "has_javascript_form_action",
    "num_password_inputs",
    "num_hidden_inputs",
    "num_suspicious_inputs",
    "num_links",
    "num_external_links",
    "num_null_links",
    "pct_external_links",
    "num_deceptive_links",
    "num_scripts",
    "num_external_scripts",
    "has_obfuscated_scripts",
    "num_iframes",
    "num_invisible_iframes",
    "favicon_from_external_domain",
    "has_meta_refresh",
    "disables_right_click",
    "has_empty_title",
    "has_copyright",
    "has_login_keywords",
    "has_status_bar_customization",
    "page_size_bytes",
    "dom_depth",
]

# ─── URL shortener patterns ───────────────────────────────────────────────────
SHORTENER_PATTERN = re.compile(
    r"bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|t\.co|is\.gd|buff\.ly|"
    r"short\.to|tr\.im|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|"
    r"url4\.eu|twit\.ac|su\.pr|snipurl\.com|bit\.do|db\.tt|lnkd\.in|"
    r"qr\.ae|adf\.ly|cur\.lv|po\.st|bc\.vc|buzurl\.com|cutt\.us|"
    r"yourls\.org|v\.gd|link\.zip\.net",
    re.IGNORECASE,
)


class URLFeatureExtractor:
    """
    Extracts all URL_FEATURE_COLUMNS from a raw URL string.

    Fast path (offline features, no network): always computed.
    Slow path (DNS, WHOIS, PageRank): computed when use_network=True (training).
    During inference in production, use_network=False for sub-20ms latency.
    """

    def __init__(self, use_network: bool = False):
        self.use_network = use_network
        if use_network:
            _ensure_network_deps()

    def extract(self, url: str) -> dict:
        """Return a dict mapping each URL_FEATURE_COLUMN to a numeric value."""
        features = {}
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or ""
            path = parsed.path or ""
            scheme = parsed.scheme or ""

            # ── Fast (offline) features ───────────────────────────────────
            features["having_ip"] = self._having_ip(domain)
            features["have_at_sign"] = 1 if "@" in url else 0
            features["url_length"] = self._url_length(url)
            features["url_depth"] = len([p for p in path.split("/") if p])
            features["redirection"] = self._redirection(url)
            features["https_in_domain"] = 1 if "https" in domain.lower() else 0
            features["tiny_url"] = 1 if SHORTENER_PATTERN.search(domain) else 0
            features["prefix_suffix"] = 1 if "-" in domain else 0
            features["sub_domains"] = self._count_subdomains(domain)
            features["https_token"] = 1 if scheme != "https" else -1

            # ── Lexical / NLP features (New for Phase 5) ──────────────────
            features["shannon_entropy"] = self._shannon_entropy(domain)
            features["vowel_consonant_ratio"] = self._vowel_consonant_ratio(domain)
            features["path_to_url_ratio"] = self._path_to_url_ratio(url, path)
            features["digit_ratio"] = self._digit_ratio(url)
            features["special_char_ratio"] = self._special_char_ratio(url)
            features["is_punycode"] = self._is_punycode(domain)
            features["keyword_match_score"] = self._keyword_match_score(url)

            # ── SSL certificate ───────────────────────────────────────────
            features["ssl_final_state"] = (
                self._ssl_state(domain) if self.use_network else 0
            )

            # ── Domain registration ───────────────────────────────────────
            features["domain_registration_length"] = (
                self._domain_reg_length(domain) if self.use_network else 0
            )
            features["age_of_domain"] = (
                self._age_of_domain(domain) if self.use_network else 0
            )

            # ── Placeholder features (require page content — set to 0 for URL-only) ──
            # These are populated by the DOM model; set to neutral for Tier 1.
            for col in [
                "favicon", "port", "request_url", "url_of_anchor",
                "links_in_tags", "sfh", "submitting_to_email",
                "abnormal_url", "redirect", "on_mouseover",
                "right_click", "popup_window", "iframe",
            ]:
                features[col] = 0

            # ── Network-dependent features ────────────────────────────────
            features["dns_record"] = self._dns_record(domain) if self.use_network else 0
            features["web_traffic"] = (
                self._web_traffic(domain) if self.use_network else 0
            )
            features["page_rank"] = 0  # PageRank API deprecated; use 0
            features["google_index"] = (
                self._google_index(url) if self.use_network else 0
            )
            features["links_pointing_to_page"] = 0
            features["statistical_report"] = 0

        except Exception as exc:
            logger.warning(f"Feature extraction error for {url!r}: {exc}")
            # Return neutral (0) for all features on error
            return {col: 0 for col in URL_FEATURE_COLUMNS}

        # Ensure all columns are present
        for col in URL_FEATURE_COLUMNS:
            features.setdefault(col, 0)

        return features

    # ── Individual feature methods ────────────────────────────────────────────

    def _having_ip(self, domain: str) -> int:
        """1 if domain is an IP address (phishing indicator), else 0."""
        clean = domain.split(":")[0]  # strip port
        for family in (ipaddress.IPv4Address, ipaddress.IPv6Address):
            try:
                family(clean)
                return 1
            except ValueError:
                pass
        return 0

    def _url_length(self, url: str) -> int:
        """Bucketed: 1 = long (>=75), 0 = medium (54–74), -1 = short (<54)."""
        n = len(url)
        if n >= 75:
            return 1
        if n >= 54:
            return 0
        return -1

    def _redirection(self, url: str) -> int:
        """1 if '//' appears in path (after scheme), phishing redirect pattern."""
        pos = url.rfind("//")
        return 1 if pos > 7 else 0

    def _count_subdomains(self, domain: str) -> int:
        """
        -1 = 1 subdomain (safe), 0 = 2 subdomains (suspicious), 1 = 3+ (phishing).
        Strips www. before counting.
        """
        domain = re.sub(r"^www\.", "", domain.split(":")[0])
        parts = domain.split(".")
        n = len(parts) - 2  # subtract SLD + TLD
        if n <= 1:
            return -1
        if n == 2:
            return 0
        return 1

    def _ssl_state(self, domain: str) -> int:
        """1 = trusted cert, 0 = untrusted/expired, -1 = no cert."""
        try:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(
                socket.socket(), server_hostname=domain.split(":")[0]
            ) as s:
                s.settimeout(3)
                s.connect((domain.split(":")[0], 443))
                cert = s.getpeercert()
                issuer = dict(x[0] for x in cert.get("issuer", []))
                trusted_issuers = ["DigiCert", "Let's Encrypt", "Comodo", "GeoTrust", "Sectigo"]
                org = issuer.get("organizationName", "")
                return 1 if any(t in org for t in trusted_issuers) else 0
        except ssl.SSLError:
            return 0
        except Exception:
            return -1

    def _domain_reg_length(self, domain: str) -> int:
        """1 if domain expires < 1 year (phishing), -1 if >= 1 year."""
        try:
            w = whois.whois(domain.split(":")[0])
            expiry = w.expiration_date
            if isinstance(expiry, list):
                expiry = expiry[0]
            if expiry and (expiry - datetime.now()).days < 365:
                return 1
            return -1
        except Exception:
            return -1

    def _age_of_domain(self, domain: str) -> int:
        """1 if domain < 6 months old (phishing), -1 if older."""
        try:
            w = whois.whois(domain.split(":")[0])
            created = w.creation_date
            if isinstance(created, list):
                created = created[0]
            if created and (datetime.now() - created).days < 180:
                return 1
            return -1
        except Exception:
            return -1

    def _shannon_entropy(self, string: str) -> float:
        """Calculate Shannon entropy for DGA (Domain Generation Algorithm) detection."""
        if not string:
            return 0.0
        prob = [float(string.count(c)) / len(string) for c in set(string)]
        entropy = -sum(p * math.log2(p) for p in prob)
        return float(entropy)

    def _vowel_consonant_ratio(self, string: str) -> float:
        """Calculate the ratio of vowels to consonants. Low ratios indicate random/DGA domains."""
        if not string:
            return 0.0
        s = string.lower()
        vowels = sum(1 for c in s if c in 'aeiou')
        consonants = sum(1 for c in s if c in 'bcdfghjklmnpqrstvwxyz')
        if consonants == 0:
            return float(vowels)
        return float(vowels) / float(consonants)

    def _path_to_url_ratio(self, url: str, path: str) -> float:
        """Ratio of path length to total URL length. Safe search URLs have long URLs but short paths."""
        if not url:
            return 0.0
        return float(len(path)) / float(len(url))

    def _dns_record(self, domain: str) -> int:
        """0 if DNS record found (legitimate), -1 if no record (phishing)."""
        try:
            socket.gethostbyname(domain.split(":")[0])
            return 0
        except socket.gaierror:
            return -1

    def _web_traffic(self, domain: str) -> int:
        """
        Placeholder for Alexa/SimilarWeb rank lookup.
        Returns 0 (neutral) — integrate your preferred traffic API.
        """
        return 0

    def _digit_ratio(self, url: str) -> float:
        if not url: return 0.0
        digits = sum(c.isdigit() for c in url)
        return digits / len(url)

    def _special_char_ratio(self, url: str) -> float:
        if not url: return 0.0
        specials = sum(1 for c in url if c in "-_=?&%")
        return specials / len(url)

    def _is_punycode(self, domain: str) -> int:
        return 1 if "xn--" in domain.lower() else 0

    def _keyword_match_score(self, url: str) -> int:
        keywords = ["login", "secure", "account", "update", "verify", "banking", "signin", "password", "auth", "confirm"]
        url_lower = url.lower()
        return sum(url_lower.count(k) for k in keywords)

    def _google_index(self, url: str) -> int:
        """1 if indexed by Google (legitimate), -1 if not."""
        try:
            results = list(google_search(f"site:{url}", num_results=1, sleep_interval=2))
            return 1 if results else -1
        except Exception:
            return 0


# ─── Combined feature extractor (training time) ───────────────────────────────
def extract_combined_features(url: str, dom_dict: Optional[dict] = None) -> dict:
    """
    Extract the full combined URL + DOM feature vector for training.
    dom_dict: pre-extracted DOM features (from content.js during labelled data collection)
              or None to use zero-fill for URL-only training.
    """
    extractor = URLFeatureExtractor(use_network=True)
    url_features = extractor.extract(url)

    if dom_dict is None:
        dom_features = {col: 0 for col in DOM_FEATURE_COLUMNS}
    else:
        dom_features = {col: dom_dict.get(col, 0) for col in DOM_FEATURE_COLUMNS}

    return {**url_features, **dom_features}
