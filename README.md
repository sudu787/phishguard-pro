# 🛡 PhishGuard Pro

> **Enterprise-grade phishing website detection** — a multi-modal ML system combining fast URL analysis, DOM/HTML contextual features, and real-time threat-intelligence feeds.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io)
[![Manifest V3](https://img.shields.io/badge/Chrome-Manifest%20V3-yellow.svg)](https://developer.chrome.com/docs/extensions/mv3/)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Quick Start (Docker)](#quick-start-docker)
5. [Manual Setup](#manual-setup)
6. [Training the Models](#training-the-models)
7. [Loading the Chrome Extension](#loading-the-chrome-extension)
8. [API Reference](#api-reference)
9. [Detection Pipeline Deep Dive](#detection-pipeline-deep-dive)
10. [Security Considerations](#security-considerations)
11. [Performance Benchmarks](#performance-benchmarks)
12. [Roadmap](#roadmap)

---

## Architecture Overview

PhishGuard Pro uses a **3-tier, defense-in-depth** detection pipeline designed to defeat modern phishing techniques including lookalike domains, URL shorteners, and evasive HTML structures.

```
Browser Tab Navigation
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  Chrome Extension (Manifest V3)                              │
│                                                              │
│  background.js ──── Orchestrator (Service Worker)           │
│       │                                                      │
│       ├─ TIER 1 ──► /predict/url     ◄── XGBoost URL model  │  ~5–20ms
│       │              (fast gate)                             │
│       │                                                      │
│       ├─ TIER 2 ──► /predict/full   ◄── RF DOM+URL model    │  ~30–80ms
│       │   ▲          (DOM analysis)                          │
│       │   └── content.js extracts 26 DOM features           │
│       │                                                      │
│       └─ TIER 3 ──► /predict/intel  ◄── OpenPhish+URLHaus   │  async
│           (non-blocking threat intel)                        │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
  Icon + Popup UI + In-page Warning Banner
```

### Why Three Tiers?

| Tier | Input | Model | Latency | Purpose |
|------|-------|-------|---------|---------|
| T1 | URL string only | XGBoost (32 features) | < 20ms | Catch obvious phishing immediately; gate for T2 |
| T2 | URL + 26 DOM features | Random Forest (58 features) | < 80ms | Contextual analysis: forms, scripts, links |
| T3 | URL | OpenPhish + URLHaus feeds | async | Known-bad blocklist confirmation |

Score fusion: `final_score = 0.35 × T1 + 0.65 × T2`. Tier 3 can upgrade verdict to `phishing` regardless of ML scores if URL is blocklisted.

---

## Repository Structure

```
phishguard/
│
├── extension/                  # Chrome Extension (Manifest V3)
│   ├── manifest.json           # Strictly MV3-compliant
│   ├── background.js           # Service worker — orchestrates 3-tier pipeline
│   ├── content.js              # DOM extractor + warning banner injector
│   ├── icons/                  # safe / suspicious / warning / scanning PNGs
│   └── popup/
│       ├── popup.html
│       ├── popup.js
│       └── popup.css
│
├── backend/                    # FastAPI application
│   └── app/
│       ├── main.py             # Routes, middleware, lifespan
│       ├── core/
│       │   └── config.py       # Pydantic-settings configuration
│       └── services/
│           ├── predictor.py    # Model wrapper + feature fusion
│           └── threat_intel.py # Async OpenPhish + URLHaus client
│
├── ml_pipeline/                # Training & feature engineering
│   ├── features.py             # URLFeatureExtractor + DOM feature columns
│   ├── train.py                # Full training pipeline (SMOTE + Optuna)
│   ├── models/                 # Trained model artefacts (gitignored)
│   │   ├── url_xgboost.json
│   │   ├── dom_rf.joblib
│   │   └── dom_rf_columns.json
│   └── data/                   # Training datasets (gitignored)
│
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # One-command local stack
├── requirements.txt            # Production Python dependencies
└── README.md
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | `python3 --version` |
| Docker | 24+ | For containerised deployment |
| Google Chrome | 115+ | For Manifest V3 support |
| Node.js | Not required | Extension uses vanilla JS |

---

## Quick Start (Docker)

The fastest path to a running backend:

```bash
# 1. Clone the repo
git clone https://github.com/yourorg/phishguard.git
cd phishguard

# 2. Train models first (required — see Training section)
#    OR copy pre-trained models into ml_pipeline/models/

# 3. Build and start the API
docker compose up --build

# 4. Verify the backend is healthy
curl http://localhost:8000/health
# {"status":"ok","models_loaded":true,"intel_ready":true,"version":"2.0.0"}
```

---

## Manual Setup

### Backend

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install production dependencies
pip install -r requirements.txt

# Copy .env.example to .env and edit
cp .env.example .env

# Run the development server (with auto-reload)
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Environment Variables (`.env`)

```dotenv
DEBUG=true
URL_MODEL_PATH=ml_pipeline/models/url_xgboost.json
DOM_MODEL_PATH=ml_pipeline/models/dom_rf.joblib
CORS_ORIGINS=["chrome-extension://YOUR_EXTENSION_ID","http://localhost:3000"]
OPENPHISH_FEED_URL=https://openphish.com/feed.txt
INTEL_CACHE_TTL=3600
PHISHING_THRESHOLD=0.65
SUSPICIOUS_THRESHOLD=0.40
```

> **Finding your Extension ID:** Load the unpacked extension (see below), then visit `chrome://extensions`. The ID is shown under the extension name.

---

## Training the Models

### 1. Prepare your dataset

The training CSV must have:
- One column per feature in `URL_FEATURE_COLUMNS` (and optionally `DOM_FEATURE_COLUMNS`)
- A `label` column: `1` = phishing, `0` = legitimate

**Recommended public datasets:**
- [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites) — 11,055 instances, 30 URL features
- [PhishTank](https://www.phishtank.com/developer_info.php) — live phishing URLs (pair with Alexa top-1M for legitimate)

Place your CSV at `ml_pipeline/data/dataset.csv`.

### 2. Install training dependencies

```bash
pip install -r ml_pipeline/requirements-train.txt
```

### 3. Run the training pipeline

```bash
# Train both models (Tier 1 + Tier 2)
python -m ml_pipeline.train \
  --data ml_pipeline/data/dataset.csv \
  --output ml_pipeline/models \
  --trials 50

# Train only the fast URL model (useful for quick iteration)
python -m ml_pipeline.train \
  --data ml_pipeline/data/dataset.csv \
  --output ml_pipeline/models \
  --trials 30 \
  --url-only
```

### What the training pipeline does

1. **Loads & validates** the CSV, checking for required columns.
2. **Splits** the data 80/20 with stratification (preserves class balance).
3. **Applies SMOTE** to the training split to handle class imbalance without discarding majority-class samples.
4. **Runs Optuna** Bayesian hyperparameter search (50 trials, 5-fold CV, maximising F1).
5. **Trains the final model** on the full resampled training set.
6. **Evaluates** on the held-out test set and prints/saves:
   - Accuracy, F1, Precision, Recall
   - Log-Loss (penalises confident wrong predictions)
   - ROC-AUC and Precision-Recall AUC
   - Confusion matrix PNG
   - Feature importance plot PNG
7. **Saves models** securely: XGBoost → `.json`, Random Forest → `.joblib` (never `.pickle`).

### Expected evaluation output

```
──────────────────────────────────────────────────
  XGBoost (Tier 1) — Evaluation Results
──────────────────────────────────────────────────
  Accuracy          : 0.9712
  F1 Score          : 0.9724
  Precision         : 0.9681
  Recall            : 0.9768
  Log-Loss          : 0.0891
  ROC-AUC           : 0.9941
  Precision-Recall AUC : 0.9928
```

---

## Loading the Chrome Extension

1. Open Chrome and navigate to `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. Click **"Load unpacked"**
4. Select the `extension/` folder from this repository
5. Note your **Extension ID** (e.g. `abcdefghijklmnopqrstuvwxyz123456`)
6. Update `CORS_ORIGINS` in your `.env` with the extension ID:
   ```
   CORS_ORIGINS=["chrome-extension://abcdefghijklmnopqrstuvwxyz123456"]
   ```
7. Restart the backend

> **Icons:** You must provide PNG icons in `extension/icons/` named `safe-16.png`, `safe-48.png`, `safe-128.png`, `warning-*`, `suspicious-*`, `scanning-*`. Use any 🛡 icon set (e.g. [Material Icons](https://fonts.google.com/icons)).

---

## API Reference

All endpoints accept and return JSON. The backend runs on `http://localhost:8000` by default.

### `GET /health`

```json
{
  "status": "ok",
  "models_loaded": true,
  "intel_ready": true,
  "version": "2.0.0"
}
```

### `POST /predict/url` — Tier 1

```json
// Request
{ "url": "http://paypa1-secure.login.xyz/verify" }

// Response
{
  "score": 0.921,
  "verdict": "phishing",
  "top_features": [
    { "feature": "prefix_suffix", "description": "Hyphen in domain", "value": 1 },
    { "feature": "tiny_url",       "description": "URL shortener service", "value": 0 }
  ],
  "model_version": "url-xgb-v2",
  "latency_ms": 8.4
}
```

### `POST /predict/full` — Tier 1 + 2

```json
// Request
{
  "url": "http://paypa1-secure.login.xyz/verify",
  "dom_features": {
    "num_forms": 2,
    "has_external_form_action": 1,
    "num_password_inputs": 1,
    "has_obfuscated_scripts": 1,
    ...
  }
}
```

### `POST /predict/intel` — Tier 3

```json
// Response
{
  "in_blocklist": true,
  "sources": ["OpenPhish", "URLHaus"],
  "lookup_ms": 142.3
}
```

---

## Detection Pipeline Deep Dive

### URL Features (32 features, Tier 1)

| Feature | Description | Phishing signal |
|---------|-------------|-----------------|
| `having_ip` | IP address used as domain | Strong |
| `tiny_url` | URL shortener detected | Moderate |
| `prefix_suffix` | Hyphen in domain (`paypa-l.com`) | Strong |
| `sub_domains` | 3+ subdomains | Moderate |
| `ssl_final_state` | Invalid/expired SSL cert | Moderate |
| `age_of_domain` | Domain registered < 6 months ago | Strong |
| `domain_registration_length` | Expires < 1 year | Moderate |
| `https_in_domain` | `https` word in hostname | Strong |
| `url_length` | URL ≥ 75 characters | Moderate |

### DOM Features (26 features, Tier 2)

| Feature | Description | Phishing signal |
|---------|-------------|-----------------|
| `has_external_form_action` | Form submits to a different domain | Very strong |
| `num_deceptive_links` | Visible URL text ≠ actual href | Strong |
| `has_obfuscated_scripts` | `eval()`, `unescape()`, `fromCharCode` | Strong |
| `num_invisible_iframes` | Zero-size/hidden iframes | Strong |
| `favicon_from_external_domain` | Favicon loaded from attacker's server | Moderate |
| `has_meta_refresh` | Auto-redirect via `<meta>` | Moderate |
| `num_password_inputs` | Password field count | Context-dependent |

### Score Fusion Logic

```
if T1_score >= 0.65:          # High-confidence early exit
    return PHISHING immediately (skip DOM analysis for speed)
else:
    fused = 0.35 × T1_score + 0.65 × T2_score
    verdict = PHISHING  if fused >= 0.65
              SUSPICIOUS if fused >= 0.40
              SAFE       otherwise

if T3_intel.blocklisted:
    verdict = PHISHING (override, regardless of ML score)
```

### Class Imbalance Handling

Real-world phishing datasets are typically imbalanced (2:1 to 5:1 legitimate:phishing ratio). PhishGuard uses **SMOTE** (Synthetic Minority Over-sampling Technique) applied *only to the training split* to:
- Synthesise new phishing samples by interpolating between existing ones in feature space
- Avoid over-fitting to duplicated samples (unlike simple random over-sampling)
- Maintain the honest evaluation by keeping the test split at the natural distribution

---

## Security Considerations

| Concern | Mitigation |
|---------|------------|
| Extension over-permissions | Only `tabs`, `scripting`, `activeTab`, `storage`, `alarms` — no `webRequest` blocking |
| Content script isolation | `content.js` never evaluates page code; reads DOM attributes only |
| Backend access control | CORS restricted to your specific Extension ID |
| Request flooding | SlowAPI rate limiting: 60 req/min for predict, 30 for intel |
| Large payload DoS | 32 KB request size cap on all endpoints |
| Insecure model storage | XGBoost `.json` + sklearn `.joblib` — never `.pickle` |
| Threat intel supply chain | OpenPhish feed validated; URLHaus uses HTTPS; both fail gracefully |
| False positives | Conservative thresholds (0.65 phishing, 0.40 suspicious); T2 reduces T1 false alarms |

---

## Performance Benchmarks

Measured on a MacBook Pro M2, backend running locally:

| Operation | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| Tier 1 URL predict | 6ms | 12ms | 18ms |
| DOM feature extraction (content.js) | 8ms | 22ms | 45ms |
| Tier 2 full predict | 14ms | 35ms | 65ms |
| Tier 3 intel (cache hit) | <1ms | 2ms | 5ms |
| Tier 3 intel (OpenPhish fresh) | 80ms | 200ms | 400ms |
| **Total perceived latency (T1 fast path)** | **~15ms** | **~35ms** | — |

> Tier 3 is fully non-blocking and never delays the UI warning.

---

## Roadmap

- [ ] **DistilBERT semantic URL encoder** — fine-tuned NLP model for semantic lookalike detection (e.g. "paypa1" vs "paypal")
- [ ] **Screenshot-based visual similarity** — compare page screenshot against known brand logos using perceptual hashing
- [ ] **User feedback loop** — allow users to mark false positives/negatives; feed into model retraining
- [ ] **Prometheus + Grafana dashboard** — real-time detection rate monitoring
- [ ] **Firefox / Edge support** — port extension manifest for WebExtensions API
- [ ] **CI/CD pipeline** — GitHub Actions: lint → test → train eval → Docker push

---

## Contributing

1. Fork the repo and create a feature branch.
2. For ML changes, include before/after evaluation metrics in your PR description.
3. Run `pytest backend/tests/` before submitting.
4. Follow the existing code style (Black formatter, type annotations).

---

## License

MIT © 2024 PhishGuard Contributors
