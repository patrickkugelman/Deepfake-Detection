# Use Python 3.11 (latest stable, avoiding 3.9 deprecation warnings)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CRITICAL FIX: Create proper Python package structure
# This allows 'from agents.coordinator.mcp_protocol import ...' to work
RUN mkdir -p /app/agents/coordinator

# Copy source files into the package structure
# (wildcard, not an explicit list — a previous explicit list silently fell out of
# sync as new modules were added: rate_limiter.py, usage_tracking.py, and
# security_headers.py were all missing and only worked locally because the dev
# volume mount papers over a missing COPY; a from-scratch build without that
# mount would crash at import time)
COPY *.py /app/agents/coordinator/
COPY tools /app/agents/coordinator/tools/

# Create __init__.py files to make it a proper package
RUN touch /app/agents/__init__.py
RUN touch /app/agents/coordinator/__init__.py

# Set Python path to include /app
ENV PYTHONPATH=/app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Change to the coordinator directory for execution
WORKDIR /app/agents/coordinator

# Start uvicorn (now api.py is in correct location)
# --reload removed for production stability (use volume mount + docker-compose for dev)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]