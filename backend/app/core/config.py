"""
app/core/config.py
Centralised settings — reads from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    DEBUG: bool = False
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "0.0.0.0"]
    CORS_ORIGINS: List[str] = [
        # Chrome Extension origins (replace with your actual extension ID)
        "chrome-extension://YOUR_EXTENSION_ID_HERE",
        # Local development
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    MAX_REQUEST_BYTES: int = 32_768  # 32 KB

    # ── Model paths ───────────────────────────────────────────────────────────
    URL_MODEL_PATH: str = "../ml_pipeline/models/xgboost.json"

    # ── Scoring thresholds ────────────────────────────────────────────────────
    # Fallback only: train.py writes calibrated ones to <model>_thresholds.json.
    PHISHING_THRESHOLD: float = 0.65
    SUSPICIOUS_THRESHOLD: float = 0.40

    # ── Top-sites allowlist (Tranco list: rank,domain — .csv or .csv.zip) ───────
    TRANCO_PATH: str = "../ml_pipeline/data/raw/tranco_top1m.csv.zip"
    TRANCO_ALLOWLIST_TOP: int = 10_000   # 0 disables the allowlist

    # ── Threat intel ──────────────────────────────────────────────────────────
    OPENPHISH_FEED_URL: str = "https://openphish.com/feed.txt"
    URLHAUS_API_URL: str = "https://urlhaus-api.abuse.ch/v1/url/"
    INTEL_CACHE_TTL: int = 3600  # Refresh blocklist cache hourly

    # ── AI SOC Agent ──────────────────────────────────────────────────────────
    GEMINI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None


settings = Settings()
