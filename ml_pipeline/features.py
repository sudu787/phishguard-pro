"""
ml_pipeline/features.py

URL-string features (Tier 1) and the DOM feature column list (Tier 2).

URLFeatureExtractor is used at training time AND inference time, so every
feature here must be computable from the URL string alone, identically in both.

Normalisation (the important part):
  Training data mixes "https://www.x.com/a", "http://x.com/a/" and "x.com/a".
  If features see those differences, the model learns how each SOURCE DATASET
  formats URLs instead of what phishing looks like. So before extracting we:
    - add a scheme if missing (urlparse puts the whole host into .path otherwise)
    - drop the scheme from every feature (no http/https feature at all)
    - drop a leading "www." and a trailing "/"
"""

import difflib
import ipaddress
import logging
import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract

logger = logging.getLogger("phishguard.features")

# Offline: use tldextract's bundled public-suffix snapshot, never fetch it.
_TLD = tldextract.TLDExtract(suffix_list_urls=())
# Treats user-hosting suffixes (github.io, web.app, ...) as public: x.github.io is its own site.
_TLD_PRIVATE = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

# ─── Canonical feature column order ───────────────────────────────────────────
# MUST match the trained model. Retrain after any change here.
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
    "shannon_entropy",
    "vowel_consonant_ratio",
    "path_to_url_ratio",
    "digit_ratio",
    "special_char_ratio",
    "is_punycode",
    "keyword_match_score",
    "host_length",
    "tld_suspicious",
    "brand_outside_domain",
    "free_host",
    "brand_typo",
    "num_dots",
    "num_hyphens",
    "host_digits",
    "num_params",
    "has_port",
    "script_ext",
    "longest_token",
]
# Optional stacked input: char n-gram model score (train.py writes TEXT_MODEL_FILE
# next to the XGBoost model when it was used).
TEXT_FEATURE = "text_score"
TEXT_MODEL_FILE = "url_text_model.joblib"
# Optional fine-tuned transformer (finetune_bert.py) and its mix with XGBoost (blend.py).
BERT_DIR = "phishbert"
BERT_MAX_LEN = 96
BLEND_FILE = "blend.json"
# Tier 2 (train_page.py): page-text model, XGBoost on DOM features + its score, URL/page mix.
PAGE_TEXT_MODEL_FILE, PAGE_XGB_FILE, PAGE_BLEND_FILE = "page_text_model.joblib", "page_xgb.json", "page_blend.json"
PAGE_TEXT_SCORE = "page_text_score"
PAGE_BERT_DIR, PAGE_BERT_MAX_LEN = "pagebert", 256  # optional fine-tuned DistilBERT on page text (3rd mix input)

# ponytail: WHOIS/SSL/DNS features were removed — they were 0 at inference and
# non-zero at training (train/serve skew). Re-add only if computed in both paths.

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
PAGE_TEXT_CHARS = 2000  # title + visible text the extension sends for the page-text model

# Matched against the REGISTERED domain (a substring regex flagged microsoft.com as t.co).
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "short.to", "tr.im", "cli.gs", "tiny.cc", "bit.do", "db.tt", "lnkd.in",
    "qr.ae", "adf.ly", "cur.lv", "po.st", "bc.vc", "cutt.us", "v.gd",
    "rebrand.ly", "shorturl.at", "rb.gy", "t.ly", "s.id", "cutt.ly",
}

# TLDs heavily over-represented in phishing feeds (Spamhaus / Interisle reports).
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "site", "live",
    "icu", "buzz", "rest", "cyou", "shop", "work", "click", "link", "zip", "mov",
    "country", "kim", "loan", "men", "review", "support", "app", "info",
}

# Brands commonly impersonated. Seeing one OUTSIDE the registered domain
# (paypal.com.evil.xyz, evil.xyz/paypal/login) is a classic phishing pattern.
BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "onedrive",
    "google", "gmail", "amazon", "netflix", "facebook", "instagram", "whatsapp",
    "linkedin", "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
    "dhl", "fedex", "usps", "adobe", "dropbox", "docusign", "coinbase",
    "binance", "metamask", "steam", "ebay", "yahoo", "hdfc", "icici",
)

PHISH_KEYWORDS = (
    "login", "secure", "account", "update", "verify", "banking", "signin",
    "password", "auth", "confirm",
)

# Hosts where anyone can publish a page: free hosting, forms, file shares, tunnels.
# ponytail: hand-kept list; swap for a maintained feed if it starts missing hosts.
USER_CONTENT_HOSTS = SHORTENERS | {
    "sites.google.com", "docs.google.com", "drive.google.com", "forms.gle",
    "storage.googleapis.com", "firebasestorage.googleapis.com", "script.google.com",
    "forms.office.com", "sharepoint.com", "onedrive.live.com", "1drv.ms",
    "dropbox.com", "dropboxusercontent.com", "box.com", "wetransfer.com",
    "notion.site", "weebly.com", "wixsite.com", "webflow.io", "godaddysites.com",
    "square.site", "jotform.com", "typeform.com", "formstack.com", "glitch.me",
    "replit.dev", "repl.co", "ipfs.io", "dweb.link", "ngrok.io", "ngrok-free.app",
    "trycloudflare.com", "000webhostapp.com", "blogspot.com", "wordpress.com",
    "medium.com", "github.io", "gitlab.io", "pages.dev", "workers.dev", "web.app",
    "firebaseapp.com", "netlify.app", "vercel.app", "herokuapp.com",
    "azurewebsites.net", "appspot.com", "amazonaws.com", "cloudfront.net",
    "blob.core.windows.net", "r2.dev", "linktr.ee",
}

SCRIPT_EXTS = (".php", ".asp", ".aspx", ".cgi", ".jsp", ".exe", ".scr", ".zip", ".apk")
_LEET = str.maketrans("013457", "oleast")
_TOKEN = re.compile(r"[a-z0-9]+")


def normalise(url: str) -> tuple[str, str, str]:
    """Return (body, host, path): scheme-free, www-free, trailing-slash-free."""
    url = url.strip()
    if "://" not in url[:12]:
        url = "http://" + url
    body = url.split("://", 1)[1].removeprefix("www.").rstrip("/")
    try:
        p = urlparse(url)
        host, path = p.hostname or "", p.path
    except ValueError:  # junk in real feeds: unbalanced "[" in the host, NFKC-unsafe chars
        netloc = re.split(r"[/?#]", body, maxsplit=1)[0]
        path = body[len(netloc):].split("?")[0].split("#")[0]
        host = netloc.rsplit("@", 1)[-1].split(":")[0].lower()
    return body, host.removeprefix("www."), path.rstrip("/")


def registered_domain(url_or_host: str, private: bool = False) -> str:
    """eTLD+1 via the public suffix list (handles co.uk, ...).
    private=True also splits user-hosting suffixes: x.github.io → "x.github.io"."""
    host = url_or_host if "/" not in url_or_host else normalise(url_or_host)[1]
    ext = (_TLD_PRIVATE if private else _TLD)(host.split("@")[-1])
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


PLATFORMS_FILE = "platforms.json"  # written by train.py, read by the predictor's allowlist


def abused_platforms(urls, labels, min_hosts: int = 20) -> set[str]:
    """Sites that hand out subdomains and that phishers use: >= min_hosts distinct phishing
    hosts under one registered domain (gitbook.io, mybluehost.me, …). Learnt from the labels,
    so it keeps up with new platforms instead of relying on the USER_CONTENT_HOSTS hand list."""
    hosts: dict[str, set] = {}
    for u, y in zip(urls, labels):
        if y == 1:
            h = normalise(str(u))[1]
            if not any(h == x or h.endswith("." + x) for x in USER_CONTENT_HOSTS):  # docs.google.com etc.:
                hosts.setdefault(registered_domain(h, private=True), set()).add(h)  # already handled
    return {d for d, hs in hosts.items() if len(hs) >= min_hosts}


def abused_hosts(urls, labels, min_urls: int = 20, min_share: float = 0.5) -> set[str]:
    """Hosts that are mostly phishing in our labels (≥min_urls phishing URLs, ≥min_share of the host's URLs):
    path-hosted user content on top sites (new.express.adobe.com, share.hsforms.com, telegra.ph) that the
    allowlist must not wave through. PhreshPhish train by date: caught 209/372 later allowlisted phishing,
    5/23k later benign lost the allowlist (the model still scores them)."""
    n, bad = Counter(), Counter()
    for u, y in zip(urls, labels):
        h = normalise(str(u))[1]
        n[h] += 1
        bad[h] += int(y == 1)
    return {h for h, k in bad.items() if k >= min_urls and k >= min_share * n[h]}


def site_group(url: str, platforms: set[str]) -> str:
    """Train/test grouping key = one publisher. On platforms and user-content hosts that's the
    host (x.gitbook.io ≠ y.gitbook.io); grouping the whole platform put all of it in one split.
    On docs.google.com, bit.ly… the publisher is in the path: key = longest path/query token that
    isn't a file name (the doc/link/bucket id), so storage.googleapis.com/*/index.html isn't one group."""
    text, host, _ = normalise(str(url))
    if host in USER_CONTENT_HOSTS:
        return host + "/" + max(re.split(r"[/?&=#]", text[len(host):]), key=lambda t: ("." not in t, len(t)))
    reg = registered_domain(host, private=True)
    if reg in platforms or any(host.endswith("." + h) for h in USER_CONTENT_HOSTS):
        return host
    return reg


def url_text(url: str) -> str:
    """Input string for the character n-gram text model (same normalisation as the features)."""
    return normalise(str(url))[0].lower()


def url_tokens(text: str) -> list[str]:
    """Word unigrams + bigrams of a url_text() string for the text model.
    Host tokens are tagged "h:" so "paypal" in the host differs from "paypal" in the path.
    Module-level so the pickled text model can find it at load time."""
    host = re.split(r"[/?#]", text, maxsplit=1)[0]
    toks = [f"h:{t}" for t in _TOKEN.findall(host)] + _TOKEN.findall(text[len(host):])
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


class URLFeatureExtractor:
    """Extracts URL_FEATURE_COLUMNS from a raw URL string (offline, <1ms)."""

    def extract(self, url: str) -> dict:
        try:
            body, host, path = normalise(url)
            ext = _TLD(host)
            reg = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
            outside = (ext.subdomain + "/" + path).lower()  # query excluded: search URLs mention brands

            f = {
                "having_ip": _is_ip(host),
                "have_at_sign": int("@" in body),
                "url_length": len(body),
                "url_depth": len([s for s in path.split("/") if s]),
                "redirection": int("//" in body),
                "https_in_domain": int("http" in host),
                "tiny_url": int(reg in SHORTENERS),
                "prefix_suffix": int("-" in host),
                "sub_domains": len([s for s in ext.subdomain.split(".") if s]),
                "shannon_entropy": _entropy(host),
                "vowel_consonant_ratio": _vowel_consonant_ratio(host),
                "path_to_url_ratio": len(path) / len(body) if body else 0.0,
                "digit_ratio": sum(c.isdigit() for c in body) / len(body) if body else 0.0,
                "special_char_ratio": sum(c in "-_=?&%" for c in body) / len(body) if body else 0.0,
                "is_punycode": int("xn--" in host),
                "keyword_match_score": sum(body.lower().count(k) for k in PHISH_KEYWORDS),
                "host_length": len(host),
                "tld_suspicious": int(ext.suffix.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS),
                "brand_outside_domain": int(any(b in outside and b not in ext.domain for b in BRANDS)),
                "free_host": int(any(host == h or host.endswith("." + h) for h in USER_CONTENT_HOSTS)),
                "brand_typo": _brand_typo(ext.domain),
                "num_dots": host.count("."),
                "num_hyphens": host.count("-"),
                "host_digits": sum(c.isdigit() for c in host),
                "num_params": len([q for q in body.partition("?")[2].split("#")[0].split("&") if q]),
                "has_port": int(bool(re.search(r":\d+$", re.split(r"[/?#]", body, maxsplit=1)[0]))),
                "script_ext": int(path.lower().endswith(SCRIPT_EXTS)),
                "longest_token": max((len(t) for t in _TOKEN.findall(body.lower())), default=0),
            }
        except Exception as exc:
            logger.warning(f"Feature extraction error for {url!r}: {exc}")
            return {col: 0 for col in URL_FEATURE_COLUMNS}
        return f


def _brand_typo(domain: str) -> int:
    """A domain label imitating a brand without being it: paypa1, rnicrosoft, arnazon-login."""
    for label in domain.split("-"):
        fixed = label.translate(_LEET).replace("rn", "m").replace("vv", "w")
        for b in BRANDS:
            if label == b or len(b) < 4 or abs(len(label) - len(b)) > 2:
                continue
            if fixed == b or (len(b) >= 5 and difflib.SequenceMatcher(None, label, b).ratio() >= 0.85):
                return 1
    return 0


def _is_ip(host: str) -> int:
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))


def _vowel_consonant_ratio(s: str) -> float:
    vowels = sum(c in "aeiou" for c in s)
    consonants = sum(c in "bcdfghjklmnpqrstvwxyz" for c in s)
    return vowels / consonants if consonants else float(vowels)
