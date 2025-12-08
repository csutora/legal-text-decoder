# Legal Text Decoder - Dockerfile
# Multi-stage build for efficient container with GPU support

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY notebook/ notebook/
COPY run.sh run.sh

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/log

# Make run script executable
RUN chmod +x /app/run.sh

# Expose port for Gradio
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Default command: run full pipeline
# Override with specific commands as needed:
#   docker run ... python src/04-inference.py
#   docker run ... python src/app.py
CMD ["bash", "/app/run.sh"]


# GPU-enabled variant
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install Python dependencies (GPU version of PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY notebook/ notebook/
COPY run.sh run.sh

# Create directories
RUN mkdir -p /app/data /app/models /app/log

# Make run script executable
RUN chmod +x /app/run.sh

# Expose port for Gradio
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["bash", "/app/run.sh"]
