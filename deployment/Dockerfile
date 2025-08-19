# Advanced Data Analytics Portfolio Dockerfile
# Multi-stage build for optimized production deployment

# Base stage with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose ports
EXPOSE 8000 8501

# Development command
CMD ["streamlit", "run", "dashboard/main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Copy source code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose ports
EXPOSE 8000

# Production command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# API-only stage
FROM base as api

# Copy API code only
COPY api/ ./api/
COPY requirements.txt .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# API command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Dashboard-only stage
FROM base as dashboard

# Copy dashboard code only
COPY dashboard/ ./dashboard/
COPY requirements.txt .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Dashboard command
CMD ["streamlit", "run", "dashboard/main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
