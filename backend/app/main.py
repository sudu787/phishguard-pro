"""
PhishGuard Pro — FastAPI Backend
app/main.py

Endpoints:
  GET  /health                → Liveness + readiness probe
  POST /predict/url           → Tier 1: fast URL-only scoring
  POST /predict/full          → Tier 1 + Tier 2: URL + DOM combined scoring
  POST /predict/intel         → Tier 3: async threat-intelligence lookup

Security hardening:
  - Strict CORS (configurable origins list)
  - SlowAPI rate limiting (60 req/min per IP for predict, 120 for health)
  - Pydantic v2 request validation with field constraints
  - Request size limit (32 KB) to prevent DoS via large DOM payloads
  - Trusted-host middleware
  - All errors return opaque 4xx/5xx — no stack traces in production
"""

import asyncio
import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.services.predictor import PhishingPredictor
from app.services.threat_intel import ThreatIntelService
from app.services.domain_intel import DomainIntelService
from app.services.soc_agent import SOCAgentService

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("phishguard")

# ─── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ─── App lifecycle (model loading) ────────────────────────────────────────────
predictor: PhishingPredictor = None
threat_intel: ThreatIntelService = None
domain_intel: DomainIntelService = None
soc_agent: SOCAgentService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and warm up services on startup; clean up on shutdown."""
    global predictor, threat_intel, domain_intel, soc_agent

    logger.info("Loading ML models...")
    t0 = time.perf_counter()
    predictor = PhishingPredictor(url_model_path=settings.URL_MODEL_PATH)
    predictor.load()
    logger.info(f"Models loaded in {(time.perf_counter()-t0)*1000:.1f}ms")

    logger.info("Initialising threat-intel service...")
    threat_intel = ThreatIntelService(
        cache_ttl_seconds=settings.INTEL_CACHE_TTL,
        openphish_url=settings.OPENPHISH_FEED_URL,
    )
    await threat_intel.warm_up()
    logger.info("Threat-intel service ready.")

    logger.info("Initialising domain-intel service...")
    domain_intel = DomainIntelService(
        dns_cache_ttl=3600,
        whois_cache_ttl=86400,
    )
    logger.info("Domain-intel service ready.")

    logger.info("Initialising SOC agent service...")
    soc_agent = SOCAgentService()
    logger.info("SOC agent service ready.")

    yield  # ← Application runs here

    logger.info("Shutting down...")
    await threat_intel.close()

# ─── App factory ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="PhishGuard Pro API",
    version="2.0.0",
    description="Multi-modal phishing detection backend",
    docs_url="/docs" if settings.DEBUG else None,   # Hide Swagger in prod
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# State for SlowAPI
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── Middleware stack ─────────────────────────────────────────────────────────
# 1. Trusted host (prevents Host-header injection)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

# 2. CORS — restrict to the Chrome Extension origin only
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
    max_age=600,
)

# 3. Request size limiter (prevent oversized DOM payloads)
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_REQUEST_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Request payload too large"})
    return await call_next(request)

# 4. Timing header (useful for debugging latency)
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response

# ─── Request / Response schemas ───────────────────────────────────────────────
class URLRequest(BaseModel):
    url: str = Field(..., min_length=4, max_length=2048, description="The URL to analyse")

    @field_validator("url")
    @classmethod
    def url_must_have_scheme(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v.strip()


class DOMFeatures(BaseModel):
    """Flat feature map extracted by content.js — all values numeric."""
    num_forms: int = Field(0, ge=0)
    has_external_form_action: int = Field(0, ge=0, le=1)
    has_empty_form_action: int = Field(0, ge=0, le=1)
    has_javascript_form_action: int = Field(0, ge=0, le=1)
    num_password_inputs: int = Field(0, ge=0)
    num_hidden_inputs: int = Field(0, ge=0)
    num_suspicious_inputs: int = Field(0, ge=0)
    num_links: int = Field(0, ge=0)
    num_external_links: int = Field(0, ge=0)
    num_null_links: int = Field(0, ge=0)
    pct_external_links: int = Field(0, ge=0, le=100)
    num_deceptive_links: int = Field(0, ge=0)
    num_scripts: int = Field(0, ge=0)
    num_external_scripts: int = Field(0, ge=0)
    has_obfuscated_scripts: int = Field(0, ge=0, le=1)
    num_iframes: int = Field(0, ge=0)
    num_invisible_iframes: int = Field(0, ge=0)
    favicon_from_external_domain: int = Field(0, ge=0, le=1)
    has_meta_refresh: int = Field(0, ge=0, le=1)
    disables_right_click: int = Field(0, ge=0, le=1)
    has_empty_title: int = Field(0, ge=0, le=1)
    has_copyright: int = Field(0, ge=0, le=1)
    has_login_keywords: int = Field(0, ge=0, le=1)
    has_status_bar_customization: int = Field(0, ge=0, le=1)
    page_size_bytes: int = Field(0, ge=0)
    dom_depth: int = Field(0, ge=0)
    page_text: str = Field("", max_length=4000)  # title | visible text, first 2,000 chars (content.js)


class FullRequest(BaseModel):
    url: str = Field(..., min_length=4, max_length=2048)
    dom_features: DOMFeatures

    @field_validator("url")
    @classmethod
    def url_must_have_scheme(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v.strip()


class PredictionResponse(BaseModel):
    score: float
    verdict: str            # "safe" | "suspicious" | "phishing"
    top_features: list[dict]
    model_version: str
    latency_ms: float


class ThreatIntelResponse(BaseModel):
    in_blocklist: bool = Field(..., description="True if URL is in a known blocklist")
    sources: list[str] = Field(..., description="Names of the blocklists that flagged this URL")


class DomainIntelResponse(BaseModel):
    domain: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    signals: list[str]
    whois: dict
    dns: dict
    asn: dict


class SOCRequest(BaseModel):
    url: str
    url_analysis: dict
    domain_intel: dict
    html_analysis: dict
    vision_analysis: dict


class SOCReportResponse(BaseModel):
    risk_score: int
    reasons: list[str]
    user_warning: str


class ReportRequest(BaseModel):
    url: str
    correct_label: int = Field(..., description="0 for benign, 1 for phishing")


# ─── Endpoints ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
@limiter.limit("120/minute")
async def health(request: Request):
    """Liveness + basic readiness probe. Returns 503 if models aren't loaded."""
    if predictor is None or not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "ok",
        "models_loaded": predictor.is_ready,
        "intel_ready": threat_intel.is_ready if threat_intel else False,
        "version": app.version,
    }


@app.post("/predict/url", response_model=PredictionResponse, tags=["predict"])
@limiter.limit("60/minute")
async def predict_url(request: Request, body: URLRequest):
    """
    Tier 1: URL-string-only analysis.
    Target latency: < 20ms (XGBoost inference on pre-computed features).
    """
    if not predictor or not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    t0 = time.perf_counter()
    try:
        result = predictor.predict_url(body.url)
    except Exception as exc:
        logger.error(f"Tier1 prediction error for {body.url!r}: {exc}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    latency = (time.perf_counter() - t0) * 1000
    logger.info(f"[Tier1] {body.url[:60]} → {result['verdict']} ({result['score']:.3f}) in {latency:.1f}ms")

    return PredictionResponse(
        score=result["score"],
        verdict=result["verdict"],
        top_features=result.get("top_features", []),
        model_version=predictor.url_model_version,
        latency_ms=round(latency, 2),
    )


@app.post("/predict/full", response_model=PredictionResponse, tags=["predict"])
@limiter.limit("60/minute")
async def predict_full(request: Request, body: FullRequest):
    """
    Tier 1 + Tier 2: Combined URL + DOM analysis.
    Fuses both model scores for higher accuracy.
    """
    if not predictor or not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    t0 = time.perf_counter()
    try:
        result = predictor.predict_full(body.url, body.dom_features.model_dump())
    except Exception as exc:
        logger.error(f"Tier2 prediction error: {exc}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    latency = (time.perf_counter() - t0) * 1000
    logger.info(f"[Tier2] {body.url[:60]} → {result['verdict']} ({result['score']:.3f}) in {latency:.1f}ms")

    return PredictionResponse(
        score=result["score"],
        verdict=result["verdict"],
        top_features=result.get("top_features", []),
        model_version=predictor.dom_model_version,
        latency_ms=round(latency, 2),
    )

@app.post("/predict/intel", response_model=ThreatIntelResponse, tags=["predict"])
@limiter.limit("60/minute")
async def predict_intel(request: Request, body: URLRequest):
    """
    Tier 3: Asynchronous threat-intelligence lookup.
    Checks the URL against local and remote blocklists.
    """
    if not threat_intel:
        raise HTTPException(status_code=503, detail="Intel service not ready")
        
    try:
        result = await threat_intel.check(body.url)
        return ThreatIntelResponse(**result)
    except Exception as e:
        logger.error(f"Threat intel error: {e}", exc_info=True)
        # Graceful degradation — return negative result rather than hard failure
        return ThreatIntelResponse(in_blocklist=False, sources=[])


@app.post("/predict/domain-intel", response_model=DomainIntelResponse, tags=["predict"])
@limiter.limit("60/minute")
async def predict_domain_intel(request: Request, body: URLRequest):
    """
    Tier 3.5: Advanced Domain Infrastructure Intelligence.
    Performs async WHOIS and DNS lookups to detect Fast-Flux, NRDs, missing MX records, and risky ASNs.
    """
    if not domain_intel:
        raise HTTPException(status_code=503, detail="Domain Intel service not ready")
        
    try:
        result = await domain_intel.analyze(body.url)
        if "error" in result and len(result) <= 2:
            raise HTTPException(status_code=400, detail=result.get("error", "Domain analysis failed"))
        return DomainIntelResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Domain intel error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during domain analysis")

@app.post("/predict/soc-report", response_model=SOCReportResponse, tags=["predict"])
@limiter.limit("30/minute")
async def predict_soc_report(request: Request, body: SOCRequest):
    """
    Tier 4: AI SOC Analyst Agent.
    Fuses outputs from all other modules and queries an LLM to produce an explainable risk report.
    """
    if not soc_agent:
        raise HTTPException(status_code=503, detail="SOC Agent service not ready")
        
    try:
        payload = body.model_dump()
        result = await soc_agent.analyze(payload)
        return SOCReportResponse(**result)
    except Exception as e:
        logger.error(f"SOC Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during SOC analysis")


@app.post("/predict/scan", response_model=SOCReportResponse, tags=["predict"])
@limiter.limit("30/minute")
async def predict_scan(request: Request, body: FullRequest):
    """
    Master endpoint: Orchestrates URL ML, Domain Intel, and SOC Agent.
    """
    if not predictor or not domain_intel or not soc_agent:
        raise HTTPException(status_code=503, detail="Services not ready")
        
    try:
        # 1. Get ML scores (URL + DOM)
        ml_result = predictor.predict_full(body.url, body.dom_features.model_dump())
        
        # 2. Get Domain Intel
        # Graceful handling if domain lookup fails
        domain_result = {"domain": body.url, "risk_score": 0.0, "signals": []}
        try:
            domain_result = await domain_intel.analyze(body.url)
        except Exception as de:
            logger.warning(f"Domain intel lookup failed, continuing without it: {de}")
        
        # 3. Construct SOC payload
        soc_payload = {
            "url": body.url,
            "url_analysis": {
                "score": ml_result["score"],
                "verdict": ml_result["verdict"],
                "top_features": ml_result.get("top_features", [])
            },
            "domain_intel": domain_result,
            "html_analysis": body.dom_features.model_dump(exclude={"page_text"}),  # page text never leaves for the LLM
            "vision_analysis": {}
        }
        
        # 4. Call SOC Agent LLM
        soc_result = await soc_agent.analyze(soc_payload)
        return SOCReportResponse(**soc_result)
        
    except Exception as e:
        logger.error(f"Orchestration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during scan orchestration")


@app.post("/feedback/report", tags=["feedback"])
@limiter.limit("20/minute")
async def report_feedback(request: Request, body: ReportRequest):
    """
    Phase 5 Feedback Loop:
    Accepts user corrections from the Chrome extension and appends them
    directly to the ML pipeline's feedback dataset for future retraining.
    """
    import csv
    from pathlib import Path
    
    file_path = Path("../ml_pipeline/data/user_reports.csv")
    file_exists = file_path.exists()
    
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['url', 'label'])
            writer.writerow([body.url, body.correct_label])
        logger.info(f"[Feedback] Appended {body.url} as label {body.correct_label}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to write report: {e}")
        raise HTTPException(status_code=500, detail="Failed to save report")

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        access_log=True,
    )
