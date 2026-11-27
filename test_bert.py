"""Manual smoke check of the backend predictor. Run from backend/: python ../test_bert.py"""
import sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append('.')
sys.path.append('..')
from app.services.predictor import PhishingPredictor

predictor = PhishingPredictor(
    url_model_path="../ml_pipeline/models/xgboost.json"
)
predictor.load()

for url in [
    "http://example.com/login",
    "https://google.com",
    "http://secure-update-paypal-account.com/login.php",
    "https://amazon-support-ticket-update.net/auth",
]:
    print(url, "→", predictor.predict_url(url))
