import sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append('.')
from app.services.predictor import PhishingPredictor

predictor = PhishingPredictor(
    url_model_path="../ml_pipeline/models/xgboost.json",
    dom_model_path="../ml_pipeline/models/random_forest.joblib"
)
predictor.load()
print(f"BERT MODEL LOADED: {predictor._bert_model is not None}")

test_urls = [
    "http://example.com/login",
    "https://google.com",
    "http://secure-update-paypal-account.com/login.php",
    "https://amazon-support-ticket-update.net/auth",
]

for url in test_urls:
    print(f"URL: {url}")
    print(f"Prediction: {predictor.predict_url(url)}")
    print("-" * 50)
