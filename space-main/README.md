---
title: Face Verification — Research Demo
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
---

# Face Verification — Research Demo

Gradio application that compares two face images side by side. You choose a backbone (**ArcFace**, **ViT-B/16**, or **SwinFace**); the UI reports **cosine distance** between L2-normalized embeddings and shows an **XAI saliency map** (Grad-CAM style) to highlight regions that influenced the score.

**Live demo (Hugging Face):** [https://huggingface.co/spaces/KKUBBAACC/face_verification](https://huggingface.co/spaces/KKUBBAACC/face_verification)

![Demo](screenshot.png)

## Models available

| Model | Backbone | Embedding dimension | Notes |
| --- | --- | --- | --- |
| ArcFace | ResNet-50 (`buffalo_l`, InsightFace) | 512 | Weights download to `./models/models/` on first run |
| ViT | ViT-B/16 (torchvision, ImageNet-21k) | 768 | Classifier head removed; L2-normalized features |
| SwinFace | Swin Transformer–T + multitask heads (Liu et al., 2023) | 512 | Large checkpoint; use HF Model repo + `SWINFACE_HF_REPO` on Spaces (see below) |

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Sample inputs live in `examples/`.

### Run with Docker

Build and run from this directory (`space-main/`). The app listens on port **7860** (see `GRADIO_SERVER_PORT` in `app.py`).

```bash
docker build -t face-verification-demo .
docker run --rm -p 7860:7860 face-verification-demo
```

Open `http://127.0.0.1:7860`. On Hugging Face Docker Spaces, the platform usually sets `PORT`; `app.py` picks up `PORT` or `GRADIO_SERVER_PORT` automatically.

## Deploy to Hugging Face Spaces

Official guide: [Spaces documentation](https://huggingface.co/docs/hub/spaces) (creating a Space, `sdk: gradio`, secrets, and hardware).

**SwinFace (~842 MB)** usually exceeds the default Space storage cap if committed. Do not commit `models/swinface.pt`. Instead: upload the file to a Hugging Face **Model** repository, then in the Space set **Settings → Variables and secrets** with `SWINFACE_HF_REPO` (and optionally `SWINFACE_HF_FILENAME`). The app downloads the checkpoint on first use.

InsightFace `buffalo_l` and torchvision ViT weights are fetched automatically when needed.
