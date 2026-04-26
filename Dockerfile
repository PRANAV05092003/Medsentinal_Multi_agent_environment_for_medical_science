# ═══════════════════════════════════════════════════════════
# MedSentinel — HuggingFace Spaces Dockerfile
# ═══════════════════════════════════════════════════════════
#
# Strategy: Multi-stage build
#   Stage 1 (node-builder): Build React UI → static files
#   Stage 2 (runtime):      Python FastAPI serves static + API
#
# Single container, single port 7860.
# UI is unchanged — same React app, just served as static files.
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Build the React UI ─────────────────────────────
FROM node:20-slim AS node-builder

WORKDIR /ui-build

# Copy UI package files first (better Docker layer caching)
COPY ui/package.json ui/package-lock.json* ui/bun.lockb* ./

# Install dependencies (use npm, works everywhere)
RUN npm install --legacy-peer-deps

# Copy rest of UI source
COPY ui/ .

# Build for production — outputs to /ui-build/dist
RUN npm run build

# ── Stage 2: Python runtime ──────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — split for better caching
# Install heavy deps first (torch/transformers separate for layer caching)
COPY requirements.hf.txt .
RUN pip install --no-cache-dir -r requirements.hf.txt

# Copy project source
COPY agents/      ./agents/
COPY env/         ./env/
COPY server/      ./server/
COPY tools/       ./tools/
COPY data/        ./data/
COPY medsentinel_weights_to_share/ ./medsentinel_weights_to_share/
COPY models.py    .
COPY openenv.yaml .
COPY openenv_client.py .
COPY app_hf.py    .

# Copy built React UI static files from stage 1
COPY --from=node-builder /ui-build/dist ./ui_static/

# HuggingFace Spaces runs as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "app_hf.py"]
