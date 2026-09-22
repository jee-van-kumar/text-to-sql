# --- Build stage: install deps into a venv so the runtime image stays lean ---
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim AS runtime

# libgomp1 is needed by torch's OpenMP threading
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY frontend/ ./frontend/

# Adapter/merged weights are mounted or pulled at runtime, not baked into
# the image (keeps the image small and lets you swap models without a
# rebuild). See README "Deploying" section.
ENV MODEL_ARTIFACTS_DIR=/app/artifacts
VOLUME ["/app/artifacts"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
