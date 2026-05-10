# ROBUST-6G WP6 PHY Demonstrator — API server image.
#
# Build:   docker build -t robust6g/wp6-phy:0.1.0 .
# Run:     docker run --rm -p 8000:8000 robust6g/wp6-phy:0.1.0
# Health:  curl http://localhost:8000/api/v1/health
#
# The image bundles the CSI dataset (~80 MB) under /app/dataset and the
# skg_robust6G package under /app/skg_robust6G. Override paths with the
# ROBUST6G_DATASET and ROBUST6G_SKG_PKG env vars if you mount your own.

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies. matplotlib needs libfreetype/libpng at runtime;
# scipy/numpy wheels are self-contained on glibc-based slim images.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libfreetype6 \
        libpng16-16 \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so they cache when source changes.
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the application. The expected layout is:
#   /app
#   ├── api_server.py
#   ├── models/
#   │   ├── jamming_detector_glrt.py
#   │   ├── spoof_detector.py
#   │   └── skg_engine.py
#   ├── skg_robust6G/        (the upstream reconciliation package)
#   ├── dataset/
#   │   └── data_ULA_skg.npz
#   └── (optional) main.py + assets/  for the NiceGUI demonstrator
COPY . /app

# Run as non-root.
RUN useradd --create-home --uid 1000 robust6g \
 && chown -R robust6g:robust6g /app
USER robust6g

EXPOSE 8000

# Container healthcheck hits the readiness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "api_server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
