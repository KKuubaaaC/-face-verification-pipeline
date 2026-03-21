FROM python:3.11-slim AS base

LABEL maintainer="Jakub <jakub@example.com>"
LABEL description="NASK Face Recognition — Reproducible Research Environment"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

COPY pyproject.toml ./
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install -e ".[deepface]"

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
