"""
app/services/predictor.py

Wraps the trained ML models and exposes predict_url() and predict_full().
Thread-safe: models are loaded once at startup and read-only thereafter.

Verdict thresholds come from `<model>_thresholds.json`, written by train.py
from out-of-fold scores at a target false-positive rate. The score returned
here must be the raw model probability those thresholds were fitted on — no
hand-weighted blending, or the thresholds stop meaning anything.

After the model, a top-sites allowlist (Tranco) can turn a non-safe verdict
into "safe". It is deliberately NOT a training feature: every benign source in
the training data comes from top-site lists, so as a feature it would teach
"not a top site ⇒ phishing" and flag every small legitimate site.
"""

import io
import itertools
import json
import logging
import math
import zipfile
from pathlib import Path
from typing import Any

import joblib
import xgboost as xgb

from app.core.config import settings
from ml_pipeline.features import (
    BERT_DIR, BERT_MAX_LEN, BLEND_FILE, DOM_FEATURE_COLUMNS, PAGE_BERT_DIR, PAGE_BERT_MAX_LEN, PAGE_BLEND_FILE,
    PAGE_TEXT_MODEL_FILE, PAGE_TEXT_SCORE,
    PAGE_XGB_FILE, PLATFORMS_FILE, TEXT_FEATURE, TEXT_MODEL_FILE, URL_FEATURE_COLUMNS,
    USER_CONTENT_HOSTS, URLFeatureExtractor, normalise, registered_domain, url_text,
)

logger = logging.getLogger("phishguard.predictor")

# USER_CONTENT_HOSTS (from features): anyone can publish there, so a top rank says nothing.


def load_top_domains(path: str, top_n: int) -> set[str]:
    """First `top_n` domains of a Tranco list (rank,domain per line; .csv or .csv.zip)."""
    p = Path(path)
    if top_n <= 0:
        return set()
    if not p.exists():
        logger.warning(f"Tranco list not found at {p} — top-sites allowlist disabled.")
        return set()
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as zf, zf.open(zf.namelist()[0]) as f:
            lines = list(itertools.islice(io.TextIOWrapper(f, "utf-8", errors="replace"), top_n))
    else:
        with p.open(encoding="utf-8", errors="replace") as f:
            lines = list(itertools.islice(f, top_n))
    top = {parts[1].strip().lower() for parts in (l.split(",") for l in lines) if len(parts) >= 2}
    logger.info(f"Top-sites allowlist: {len(top)} domains from {p.name}")
    return top


def top_site(url: str, top: set[str], platforms: set[str] = frozenset(), hosts: set[str] = frozenset()) -> str | None:
    """The URL's site if it's a top site that doesn't host user content, else None.
    From train.py's platforms.json: `platforms` = top sites that hand out subdomains phishers use,
    e.g. gitbook.io — only the site itself is allowlisted there, not x.gitbook.io; `hosts` = hosts
    mostly phishing in our labels (new.express.adobe.com) — never allowlisted."""
    if not top:
        return None
    host = normalise(url)[1]
    site = registered_domain(host, private=True)  # x.github.io stays x.github.io
    if site not in top:
        return None
    if any(host == h or host.endswith("." + h) for h in USER_CONTENT_HOSTS):
        return None
    if site in platforms and host != site or host in hosts:
        return None
    return site


def _load_thresholds(model_path: Path) -> dict:
    path = model_path.with_name(f"{model_path.stem}_thresholds.json")
    if path.exists():
        thr = json.loads(path.read_text())
        logger.info(f"Thresholds for {model_path.name}: phishing={thr['phishing']:.4f} suspicious={thr['suspicious']:.4f}")
        return thr
    logger.warning(f"{path.name} not found — using uncalibrated config thresholds. Retrain to fix.")
    return {"phishing": settings.PHISHING_THRESHOLD, "suspicious": settings.SUSPICIOUS_THRESHOLD}


def _bert_prob(bert: tuple, text: str, max_len: int) -> float:
    import torch
    tok, model = bert
    with torch.inference_mode():
        logits = model(**tok(text, truncation=True, max_length=max_len, return_tensors="pt")).logits
    return float(torch.softmax(logits, -1)[0, 1])


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _verdict(score: float, thr: dict) -> str:
    if score >= thr["phishing"]:
        return "phishing"
    if score >= thr["suspicious"]:
        return "suspicious"
    return "safe"


class PhishingPredictor:
    """
    url_model  → XGBoost on URL-string features (+ stacked text_score if trained with it)
    text_model → char n-gram logistic regression, loaded when the URL model expects it
    page model → optional Tier 2 (train_page.py): XGBoost on the extension's DOM features + stacked
                 page-text score, mixed with the URL score per page_blend.json. Loaded when shipped.
    """

    def __init__(self, url_model_path: str):
        self.url_model_path = Path(url_model_path)

        self._url_booster: xgb.Booster | None = None
        self._url_cols: list[str] = URL_FEATURE_COLUMNS
        self._text_model: Any = None
        self._bert: tuple | None = None   # (tokenizer, model) when blend.json is shipped
        self._blend: dict | None = None
        self._page: tuple | None = None   # (text_model, booster, page_blend) when train_page.py shipped Tier 2
        self._url_thr: dict = {}
        self._top: set[str] = set()
        self._platforms: set[str] = set(); self._abused_hosts: set[str] = set()

        self._url_extractor = URLFeatureExtractor()
        self.is_ready = False
        self.url_model_version = "url-xgb-v3"
        self.dom_model_version = "page-xgb-v1"

    def load(self):
        """Load models from disk. Called once at startup."""
        if not self.url_model_path.exists():
            raise FileNotFoundError(f"URL model not found: {self.url_model_path}")

        self._url_booster = xgb.Booster()
        self._url_booster.load_model(str(self.url_model_path))
        self._url_cols = list(self._url_booster.feature_names or URL_FEATURE_COLUMNS)
        self._url_thr = _load_thresholds(self.url_model_path)

        # Pickle-based loads below: only load artifacts you trained yourself.
        if TEXT_FEATURE in self._url_cols:
            path = self.url_model_path.with_name(TEXT_MODEL_FILE)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing — {self.url_model_path.name} was trained with it.")
            self._text_model = joblib.load(path)

        # Optional XGBoost + fine-tuned BERT mix (blend.py writes blend.json only if it helps).
        blend, bert = self.url_model_path.with_name(BLEND_FILE), self.url_model_path.with_name(BERT_DIR)
        if blend.exists() and bert.exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # heavy: only when used
            self._bert = (AutoTokenizer.from_pretrained(bert), AutoModelForSequenceClassification.from_pretrained(bert).eval())
            self._blend = json.loads(blend.read_text())
            self._url_thr = self._blend["thresholds"]  # thresholds belong to the blended score
            logger.info(f"BERT blend on: phishing={self._url_thr['phishing']:.4f} suspicious={self._url_thr['suspicious']:.4f}")

        page_files = [self.url_model_path.with_name(n) for n in (PAGE_TEXT_MODEL_FILE, PAGE_XGB_FILE, PAGE_BLEND_FILE)]
        if all(p.exists() for p in page_files):
            booster = xgb.Booster()
            booster.load_model(str(page_files[1]))
            mix = json.loads(page_files[2].read_text())
            page_bert = None
            if len(mix["coef"]) == 3:  # train_page.py --bert: fine-tuned page-text DistilBERT is the 3rd input
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                d = self.url_model_path.with_name(PAGE_BERT_DIR)
                page_bert = (AutoTokenizer.from_pretrained(d), AutoModelForSequenceClassification.from_pretrained(d).eval())
            self._page = (joblib.load(page_files[0]), booster, mix, page_bert)
            logger.info(f"Tier 2 page model on{' (+ page BERT)' if page_bert else ''}.")
        else:
            logger.warning("Tier 2 page model not found — /predict/full uses the URL score.")

        self._top = load_top_domains(settings.TRANCO_PATH, settings.TRANCO_ALLOWLIST_TOP)
        pf = self.url_model_path.with_name(PLATFORMS_FILE)
        abuse = json.loads(pf.read_text()) if pf.exists() else {}
        self._platforms, self._abused_hosts = set(abuse.get("subdomains", ())), set(abuse.get("hosts", ()))

        self.predict_url("http://example.com/login")  # warm-up
        self.is_ready = True
        logger.info("Models loaded and warmed up.")

    # ── Tier 1: URL only ──────────────────────────────────────────────────────
    def predict_url(self, url: str) -> dict:
        return self._apply_allowlist(url, self._score_url(url))

    # ── Tier 2: URL + DOM ─────────────────────────────────────────────────────
    def predict_full(self, url: str, dom_features: dict) -> dict:
        result = self._score_url(url)

        if self._page is not None:
            text_model, booster, mix, page_bert = self._page
            page_text = str(dom_features.get("page_text") or "")
            cols = DOM_FEATURE_COLUMNS + [PAGE_TEXT_SCORE]
            text_score = float(text_model.predict_proba([page_text])[0, 1])
            vector = [float(dom_features.get(c, 0) or 0) for c in DOM_FEATURE_COLUMNS] + [text_score]
            inputs = [result["score"], float(booster.predict(xgb.DMatrix([vector], feature_names=cols))[0])]
            if page_bert:
                inputs.append(_bert_prob(page_bert, page_text, mix.get("bert_max_len") or PAGE_BERT_MAX_LEN))
            z = mix["intercept"] + sum(w * _logit(p) for w, p in zip(mix["coef"], inputs))
            prob = 1 / (1 + math.exp(-z))
            result["score"] = prob
            result["verdict"] = _verdict(prob, mix["thresholds"])

        result = self._apply_allowlist(url, result)

        # UI hints only — they don't change the score.
        if result["verdict"] != "safe":
            if dom_features.get("num_password_inputs", 0) > 0:
                result["top_features"].insert(0, {"feature": "password_input", "description": "Password field detected on suspicious page", "value": 1})
            if dom_features.get("has_external_form_action"):
                result["top_features"].insert(0, {"feature": "external_form", "description": "Data sent to external domain", "value": 1})
        return result

    # ── internals ─────────────────────────────────────────────────────────────
    def _score_url(self, url: str) -> dict:
        f = self._url_extractor.extract(url)
        if self._text_model is not None:
            f[TEXT_FEATURE] = float(self._text_model.predict_proba([url_text(url)])[0, 1])
        dmatrix = xgb.DMatrix([[f[c] for c in self._url_cols]], feature_names=self._url_cols)
        prob = float(self._url_booster.predict(dmatrix)[0])
        if self._blend is not None:
            z = self._blend["intercept"] + sum(
                w * _logit(p) for w, p in zip(self._blend["coef"], (prob, _bert_prob(self._bert, url_text(url), BERT_MAX_LEN))))
            prob = 1 / (1 + math.exp(-z))
        return {
            "score": prob,  # unrounded: Tier 2 mixes its logit; _apply_allowlist rounds
            "verdict": _verdict(prob, self._url_thr),
            "top_features": _top_url_features(f),
        }


    def _apply_allowlist(self, url: str, result: dict) -> dict:
        result["score"] = round(result["score"], 4)
        site = top_site(url, self._top, self._platforms, self._abused_hosts)
        if site and result["verdict"] != "safe":
            result["verdict"] = "safe"
            result["top_features"].insert(0, {
                "feature": "top_site",
                "description": f"{site} is a top-{settings.TRANCO_ALLOWLIST_TOP:,} site (model score overridden)",
                "value": 1,
            })
        return result


# ─── Explanations for the popup UI ────────────────────────────────────────────
_FLAGS = {
    "brand_outside_domain": "Brand name used outside the real domain",
    "brand_typo": "Domain imitates a brand name (look-alike spelling)",
    "free_host": "Hosted on a free site/file-sharing service anyone can publish to",
    "having_ip": "IP address used instead of domain",
    "is_punycode": "Punycode (look-alike characters) in domain",
    "have_at_sign": "@ symbol in URL",
    "https_in_domain": "'http' written inside the domain name",
    "tiny_url": "URL shortener service",
    "redirection": "Double-slash redirection",
    "tld_suspicious": "Top-level domain often used for phishing",
    "prefix_suffix": "Hyphen in domain",
}


def _top_url_features(features: dict, n: int = 6) -> list[dict]:
    """Red flags that are set for this URL, most telling first."""
    out = []
    if features.get(TEXT_FEATURE, 0) >= 0.9:
        out.append({"feature": TEXT_FEATURE, "description": "URL text resembles known phishing URLs", "value": round(features[TEXT_FEATURE], 2)})
    out += [
        {"feature": k, "description": d, "value": 1}
        for k, d in _FLAGS.items() if features.get(k)
    ]
    if features.get("sub_domains", 0) >= 3:
        out.append({"feature": "sub_domains", "description": "Excessive subdomains", "value": int(features["sub_domains"])})
    if features.get("keyword_match_score", 0) > 0:
        out.append({"feature": "keyword_match_score", "description": "Login/verify-style keywords in URL", "value": int(features["keyword_match_score"])})
    return out[:n]
