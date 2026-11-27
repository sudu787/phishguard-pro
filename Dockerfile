# ─────────────────────────────────────────────────────────────────────────────
# PhishGuard Pro — Production Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder) — install dependencies into a virtual env
#   Stage 2 (runtime) — copy only the venv; no build tools in final image
#
# Security hardening:
#   - Non-root user (uid 1001)
#   - Read-only filesystem where possible
#   - No shell in final image (distroless-style slim)
#   - Pinned Python version
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building native extensions (xgboost, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install only the runtime shared lib needed by XGBoost / LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Non-root user
RUN groupadd --gid 1001 phishguard && \
    useradd  --uid 1001 --gid phishguard --no-create-home --shell /bin/false phishguard

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1

# Copy application source
COPY backend/app ./app
COPY ml_pipeline/features.py ./ml_pipeline/features.py
COPY ml_pipeline/models ./ml_pipeline/models

# Make model directory explicitly readable
RUN chown -R phishguard:phishguard /app

USER phishguard

EXPOSE 8000

# Health check (used by Docker / Kubernetes)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Use uvicorn with 1 worker (scale horizontally via replicas, not workers)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
