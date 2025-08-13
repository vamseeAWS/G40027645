# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps:
# - build-essential: compilers for some wheels (e.g., shap)
# - libgomp1: OpenMP runtime (LightGBM)
# - fonts-dejavu-core: fonts so matplotlib/pycaret plots render
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        fonts-dejavu-core && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps (cache-friendly)
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# App code (compose can still bind-mount over this in dev)
COPY . .

# Ensure artifact dirs exist
RUN mkdir -p /app/models /app/data

EXPOSE 8501
# CMD is provided by docker-compose.yml

