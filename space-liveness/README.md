---
title: Face verification & attack detection
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Face verification & attack detection

**Live Space:** [KKUBBAACC/attack-detection](https://huggingface.co/spaces/KKUBBAACC/attack-detection)

Gradio app that checks **who you are** (match against a small face gallery) and **whether the face looks live** (anti-spoofing). It runs two image sets side by side for demos and benchmarks.

## What you see in the UI

- **First run** can take on the order of **10-15 seconds** while models load.
- **Set** (radio): switch between the **reference** photos (should match the gallery) and the **attack** photos (spoof / impostor-style probes).
- **File**: pick an image from that set.
- **Run example**: left = input image, right = same image with a **status label** and **face bounding box** (color encodes outcome).
- **Run benchmark**: walks **all** images in both folders, fills a **results table**, prints a **short accuracy summary**, and shows a **gallery of up to 8** annotated images (first rows from each set).

### Table columns

| Column | Meaning |
|--------|---------|
| **set** | Which folder the image came from (reference vs attack). |
| **file** | Filename. |
| **status** | What the pipeline returned for that image. |
| **expected** | Ground truth for scoring the benchmark. |
| **match** | Whether **status** satisfied the expectation (yes/no). |

### Status values

| Status | Meaning |
|--------|---------|
| **VERIFIED_MATCH** | Anti-spoof passed (or skipped) and **DeepFace.find** found a match in the gallery (label + distance on the overlay). |
| **SPOOF_DETECTED** | Anti-spoofing flagged a presentation attack (e.g. screen / print). |
| **UNAUTHORIZED** | Live-looking face but **no** match in the gallery. |
| **NO_FACE_FOUND** | No usable face after trying several detectors. |
| **ERROR** | Processing failed (e.g. exception during search). |

**Benchmark scoring:** reference images expect **VERIFIED_MATCH**. Attack images count as correct if status is **UNAUTHORIZED** **or** **SPOOF_DETECTED** (both mean “did not wrongly accept” the probe).

## How it works (technical)

Entry point: **`app.py`** → **`src/gradio_app.py`** → **`src/ui/hf_layout.py`** / **`src/ui/hf_handlers.py`**.

1. **Face crop:** **DeepFace** with detector fallback (**RetinaFace** → **OpenCV** → **MTCNN**), **anti-spoofing** when the runtime supports it (otherwise it continues without spoof scores).
2. If spoofing says “fake” → **SPOOF_DETECTED** (no identity search).
3. Else **DeepFace.find** with **Facenet512** against the enrolled folder **`db/ja/`**.
4. Match / no-match → **VERIFIED_MATCH** or **UNAUTHORIZED**.

Put enrollment photos in **`db/ja/`** (one or more faces per identity). Put attack / spoof-style probes in **`db/attack/`**. Optional sample assets can sit in **`examples/`** and be copied into `db/` as needed.

## Data layout

```
db/ja/       # Reference gallery used by DeepFace.find
db/attack/   # Images used only in the “attack” set for benchmarks / demos
examples/    # Optional sample images
```

## Project layout (Space-relevant)

```
app.py
requirements.txt
src/
  gradio_app.py
  ui/
    hf_strings.py    # UI copy
    hf_layout.py     # Gradio layout
    hf_handlers.py   # DeepFace verification + benchmark
```

Other files under `src/` (extra scripts, tests, alternate pipelines) are **not** what the Hugging Face Space runs by default.

## Development

Python **3.10-3.11**, dependencies in **`pyproject.toml`**. For Spaces, refresh pinned deps:

```bash
uv export --no-dev --no-hashes --no-editable -o requirements.txt
```

Remove a standalone `.` line from the export if it appears. Run from the repo root so `import src.*` resolves.

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest
```
## License

This project is intended for **research and educational** purposes.
