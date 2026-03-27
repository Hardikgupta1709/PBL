# =============================================================================
# Dockerfile — Regime-Adaptive Pairs Trading Reproducibility Container
# =============================================================================
# Build:  docker build -t pairs-trading .
# Run:    docker run --rm -v $(pwd)/output:/app/output pairs-trading
# Quick:  docker run --rm pairs-trading python Research/reproduce.py --quick
# =============================================================================

FROM python:3.12.9-slim

LABEL maintainer="Hardik <hardik@pbl>"
LABEL description="Full reproducibility container for pairs trading paper"

# System deps (for scipy/numpy compilation and matplotlib backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ gfortran \
        libopenblas-dev liblapack-dev \
        libfreetype6-dev libpng-dev \
        git curl \
    && rm -rf /var/lib/apt/lists/*

# Use non-interactive matplotlib backend
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42

WORKDIR /app

# Install pinned Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create output directories
RUN mkdir -p Research/results Paper/figures Paper/tables output

# Default: run the full reproduction pipeline
CMD ["python", "Research/reproduce.py"]
