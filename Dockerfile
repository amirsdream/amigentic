# Dockerfile for Magentic
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Frontend Builder
# ============================================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend files and install ALL dependencies (including devDependencies for build)
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ============================================================================
# Stage 2: Python Backend
# ============================================================================
FROM python:3.10-slim AS backend

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and set up directories
RUN useradd --create-home --shell /bin/bash magentic && \
    mkdir -p /app/data /app/rag_data && \
    chown -R magentic:magentic /app

USER magentic

# Copy requirements and install
COPY --chown=magentic:magentic requirements.txt .
RUN pip install --user -r requirements.txt

# Copy application code
COPY --chown=magentic:magentic src/ ./src/
COPY --chown=magentic:magentic alembic/ ./alembic/
COPY --chown=magentic:magentic alembic.ini .

# Copy frontend build
COPY --from=frontend-builder --chown=magentic:magentic /app/frontend/dist ./frontend/dist

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
