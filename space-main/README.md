---
title: Face Verification — 3 Model Comparison
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.9.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Face Verification Research App

Compare face embeddings across three architectures using cosine distance, with built-in explainability (XAI) visualizations.

| Model | Embedding dim | XAI method |
|-------|--------------|------------|
| ArcFace (insightface `buffalo_l`) | 512-D | — |
| ViT-B/16 (ImageNet-pretrained) | 768-D | Attention Rollout |
| SwinFace Swin-T (multitask) | 512-D | Global Feature Map |

## How it works

1. Upload two face images (or capture via webcam).
2. Select a model architecture.
3. The app detects and aligns faces, extracts embeddings, computes cosine distance, and shows whether the pair is a match.
4. For ViT and SwinFace, an attention/feature heatmap is overlaid on the aligned crops.

## Model loading

- **InsightFace `buffalo_l`** — downloaded automatically on first run from the official GitHub release (~275 MB). Cached between restarts.
- **ViT-B/16** — downloaded by `torchvision` on first use (ImageNet-1K weights).
- **SwinFace** — checkpoint fetched from [`KKUBBAACC/swinface-weights`](https://huggingface.co/KKUBBAACC/swinface-weights) via `huggingface_hub` (~842 MB). Falls back to a local `models/swinface.pt` if present.

## Project structure

```
app.py                  # Gradio entry point
src/
  main.py               # UI layout and callbacks
  pipeline.py           # VerificationPipeline (detector → PAD → embedder → verify)
  vision/
    detector.py          # Face detection + alignment (insightface 2d106det)
    embedder.py          # ArcFace / ViT embedding extractor
    swinface_embedder.py # SwinFace wrapper (multitask: age, gender, expression)
  pad/
    liveness.py          # Blink detection (EAR) + Moiré / LBP / specular analysis
    physical_features.py # LBP histogram + FFT spectrum feature extractor
  xai/
    explainability.py    # Attention Rollout (ViT) and feature map (SwinFace)
third_party/
  swinface/              # Swin-T backbone and analysis modules (inference only)
```

## Development

Python **3.10–3.11**. Dependencies are declared in `pyproject.toml` (including **DeepFace**, **pandas**, and **tf-keras** for `src/gradio_app.py` and the `static_test` / `live_cam` helpers). The `requirements.txt` file is a pip-compatible export for Hugging Face Spaces.

```bash
uv sync --all-extras
uv run ruff check src
uv run ruff format src
```

To regenerate `requirements.txt` from `pyproject.toml`:

```bash
uv export --no-dev --no-hashes --no-editable -o requirements.txt
```

## License

This project is intended for research and educational purposes.
