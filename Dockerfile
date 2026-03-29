# ── AnomalyVision — Hugging Face Spaces (Docker SDK) ─────────────────────────
# HF Spaces requires port 7860

FROM python:3.10-slim

LABEL maintainer="AnomalyVision"
LABEL description="Real-Time Surveillance Video Anomaly Detection"

# System packages required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py       .
COPY src/         ./src/
COPY scripts/     ./scripts/

# Streamlit configuration — port 7860 required by HF Spaces
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=7860", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
