# ── AnomalyVision Docker Image ────────────────────────────────────────────────
# Build:  docker build -t anomalyvision .
# Run:    docker run -p 8501:8501 -v $(pwd)/model_anomaly_detection.h5:/app/model_anomaly_detection.h5 anomalyvision

FROM python:3.10-slim

# Metadata
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

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py       .
COPY src/         ./src/
COPY scripts/     ./scripts/

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
