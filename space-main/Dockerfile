# Gradio face-verification demo (build from this directory: docker build -t face-verification-demo .)
FROM python:3.11-slim

LABEL description="Face Verification — Research Demo (Gradio)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

COPY requirements.txt .

# build-essential: insightface ships Cython extensions that compile on install (esp. arm64)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 7860

# app.py reads GRADIO_SERVER_PORT or PORT (HF Spaces sets PORT in Docker Spaces)
CMD ["python", "app.py"]
