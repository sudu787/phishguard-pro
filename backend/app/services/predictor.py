"""
app/services/predictor.py

Wraps the trained ML models and exposes predict_url() and predict_full().
Thread-safe: models are loaded once at startup and read-only thereafter.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb

from ml_pipeline.features import URLFeatureExtractor, DOM_FEATURE_COLUMNS, URL_FEATURE_COLUMNS

logger = logging.getLogger("phishguard.predictor")

PHISHING_THRESHOLD = 0.65
SUSPICIOUS_THRESHOLD = 0.40


def _score_to_verdict(score: float) -> str:
    if score >= PHISHING_THRESHOLD:
        return "phishing"
    if score >= SUSPICIOUS_THRESHOLD:
        return "suspicious"
    return "safe"


class PhishingPredictor:
    """
    Dual-model predictor + Optional Deep Learning Ensembling.

    url_model  → XGBoost booster trained on URL-string features (Tier 1)
    dom_model  → Random Forest trained on URL + DOM/HTML features (Tier 2)
    bert_model → Optional fine-tuned DistilBERT for raw URL string inference
    """

    def __init__(self, url_model_path: str, dom_model_path: str):
        self.url_model_path = Path(url_model_path)
        self.dom_model_path = Path(dom_model_path)
        self.bert_model_path = Path("../ml_pipeline/models/phishbert")
        
        self._url_booster: xgb.Booster | None = None
        self._dom_model: Any = None
        self._bert_model: Any = None
        self._bert_tokenizer: Any = None
        
        self._url_extractor = URLFeatureExtractor()
        self.is_ready = False
        self.url_model_version = "url-xgb-v2"
        self.dom_model_version = "dom-rf-v2"
        self.bert_model_version = "bert-v1"

    def load(self):
        """Load both models from disk. Called once at startup."""
        if not self.url_model_path.exists():
            raise FileNotFoundError(f"URL model not found: {self.url_model_path}")
        if not self.dom_model_path.exists():
            raise FileNotFoundError(f"DOM model not found: {self.dom_model_path}")

        self._url_booster = xgb.Booster()
        self._url_booster.load_model(str(self.url_model_path))

        self._dom_model = joblib.load(str(self.dom_model_path))

        # Try loading optional PhishBERT
        if self.bert_model_path.exists():
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                logger.info("PhishBERT found. Loading Deep Learning model...")
                self._bert_tokenizer = AutoTokenizer.from_pretrained(str(self.bert_model_path))
                self._bert_model = AutoModelForSequenceClassification.from_pretrained(str(self.bert_model_path))
                self._bert_model.eval()
                # Determine device
                self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                self._bert_model.to(self._device)
                logger.info(f"PhishBERT loaded on {self._device}.")
            except Exception as e:
                logger.error(f"Failed to load PhishBERT: {e}")

        # Warm-up inference (loads XGBoost's internal state into CPU caches)
        dummy_url = "http://example.com/login"
        _ = self.predict_url(dummy_url)

        self.is_ready = True
        logger.info("Both models loaded and warmed up.")

    # ── Tier 1: URL only ──────────────────────────────────────────────────────
    def predict_url(self, url: str) -> dict:
        url_features = self._url_extractor.extract(url)
        feature_vector = [url_features[f] for f in URL_FEATURE_COLUMNS]
        dmatrix = xgb.DMatrix([feature_vector], feature_names=URL_FEATURE_COLUMNS)
        xgb_prob = float(self._url_booster.predict(dmatrix)[0])
        
        final_prob = xgb_prob
        top_features = _top_url_features(url_features, xgb_prob)

        # Ensemble with PhishBERT if available
        if self._bert_model:
            import torch
            inputs = self._bert_tokenizer(url, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._bert_model(**inputs)
                bert_prob = torch.softmax(outputs.logits, dim=1)[0][1].item()
            
            # Smart Ensemble: BERT is prone to false positives on short/standard URLs (e.g. google.com)
            # because the dataset mostly contains benign URLs starting with 'www.'.
            # XGBoost is much more robust. We weight XGBoost higher.
            final_prob = (xgb_prob * 0.85) + (bert_prob * 0.15)
            
            # If BERT is absolutely certain it's phishing, but XGBoost says it's completely safe,
            # we penalize BERT's hallucination.
            if xgb_prob < 0.10 and bert_prob > 0.8:
                final_prob = xgb_prob * 1.5  # Keep it safe, slight bump
                
            # Add BERT as a feature reason only if it meaningfully contributed
            if bert_prob >= 0.75 and final_prob >= 0.5:
                top_features.insert(0, {"feature": "bert_nlp", "description": f"Semantic NLP Analysis (BERT Score: {bert_prob:.2f})", "value": 1})

        return {
            "score": round(final_prob, 4),
            "verdict": _score_to_verdict(final_prob),
            "top_features": top_features,
        }

    # ── Tier 2: URL + DOM ─────────────────────────────────────────────────────
    def predict_full(self, url: str, dom_features: dict) -> dict:
        # Get the highly-tuned BERT + XGBoost ensemble score first
        base_result = self.predict_url(url)
        base_prob = base_result["score"]
        top_features = base_result["top_features"]

        # Build the full feature vector exactly as the model was trained
        from ml_pipeline.features import DOM_FEATURE_COLUMNS, URL_FEATURE_COLUMNS
        url_features = self._url_extractor.extract(url)
        feature_vector = []
        for col in URL_FEATURE_COLUMNS:
            feature_vector.append(url_features.get(col, 0))
        for col in DOM_FEATURE_COLUMNS:
            feature_vector.append(dom_features.get(col, 0))
            
        try:
            # Get the Random Forest probability
            rf_prob = float(self._dom_model.predict_proba([feature_vector])[0][1])
            
            # The Random Forest is powerful but can over-penalize legitimate login pages
            # We trust the BERT + XGB ensemble (base_prob) slightly more, but blend them.
            final_prob = (base_prob * 0.75) + (rf_prob * 0.25)
            
            # Massive protection for legitimate tech domains (like google.com / github.com)
            # If the NLP model + XGBoost thinks it's completely safe (< 0.15), 
            # we IGNORE the DOM model's paranoia about password inputs!
            if base_prob < 0.15 and rf_prob < 0.60:
                final_prob = base_prob
                
        except Exception as e:
            logger.error(f"Random Forest inference failed: {e}")
            final_prob = base_prob

        # Add visual feature indicators for the UI
        if dom_features.get("num_password_inputs", 0) > 0 and final_prob > 0.40:
            top_features.insert(0, {"feature": "password_input", "description": "Password field detected on suspicious page", "value": 1})
        if dom_features.get("has_external_form_action") and final_prob > 0.40:
            top_features.insert(0, {"feature": "external_form", "description": "Data sent to external domain", "value": 1})

        return {
            "score": round(min(0.99, final_prob), 4),
            "verdict": _score_to_verdict(final_prob),
            "top_features": top_features,
        }


# ─── Feature importance helpers ───────────────────────────────────────────────
def _top_url_features(features: dict, prob: float, n: int = 6) -> list[dict]:
    """
    Return top N URL features sorted by their absolute phishing-indicative value.
    For interpretability in the popup UI.
    """
    phishing_indicators = {
        "having_ip": "IP address used instead of domain",
        "have_at_sign": "@ symbol in URL",
        "url_length": "Unusually long URL",
        "url_depth": "Deep URL path",
        "redirection": "Double-slash redirection",
        "https_in_domain": "HTTPS in domain name (deceptive)",
        "tiny_url": "URL shortener service",
        "prefix_suffix": "Hyphen in domain",
        "sub_domains": "Excessive subdomains",
        "domain_registration_length": "Short domain registration",
        "dns_record": "No DNS record found",
        "web_traffic": "Very low web traffic",
        "google_index": "Not indexed by Google",
    }

    results = []
    for key, description in phishing_indicators.items():
        val = features.get(key, 0)
        if val != 0:
            results.append({"feature": key, "description": description, "value": int(val)})
        if len(results) >= n:
            break
    return results


def _top_rf_features(columns: list, importances: np.ndarray, values: dict, n: int = 8) -> list[dict]:
    """Return top N features by Random Forest importance, with current values."""
    ranked = sorted(
        zip(columns, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {"feature": name, "importance": round(float(imp), 4), "value": values.get(name, 0)}
        for name, imp in ranked[:n]
    ]
