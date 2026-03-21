# Face liveness + DeepFace anti-spoofing (build from this directory: docker build -t face-liveness-demo .)
FROM python:3.11-slim

LABEL description="Face Liveness — DeepFace anti-spoofing demo (Gradio)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

COPY requirements.txt .

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

CMD ["python", "app.py"]
