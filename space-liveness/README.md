---
title: Face Liveness — Anti-Spoofing Demo
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
---

# Face Liveness — Anti-Spoofing Demo

Gradio application for **face liveness verification** using **DeepFace** with **anti-spoofing** enabled (`anti_spoofing=True`). You upload a **single face image**; the pipeline runs spoof detection first, then (if the face is accepted as live) **Facenet512**-based identity search against a reference gallery. The UI surfaces a discrete outcome:

| User-facing label | Internal status | Meaning |
| --- | --- | --- |
| **VERIFIED** | `VERIFIED_MATCH` | Live face matches a reference identity in the database |
| **SPOOF** | `SPOOF_DETECTED` | Anti-spoofing model rejected the frame (attack / presentation) |
| **UNAUTHORIZED** | `UNAUTHORIZED` | Live face detected but no matching identity in the reference set |

Additional cases (e.g. no face, errors) are reported in the UI and logged; see `src/gradio_app.py` for the full list.

**Live demo (Hugging Face):** [https://huggingface.co/spaces/KKUBBAACC/adversarial-attack-detection](https://huggingface.co/spaces/KKUBBAACC/adversarial-attack-detection)

![Demo](screenshot.png)

## Models available

| Component | Role |
| --- | --- |
| **DeepFace anti-spoofing** | Presentation attack detection (real vs spoof score before verification) |
| **Facenet512 + DeepFace.find** | Face embedding and nearest-neighbour match against `db/ja/` (RetinaFace detector) |

Reference faces are read from `db/ja/` (see below). If the folder is empty, verification will usually return **UNAUTHORIZED** for live images.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Sample images are under `examples/`. Add reference photos to `db/ja/` for identity **Kuba** (or extend the app as needed).

### Run with Docker

Build and run from this directory (`space-liveness/`). Default port **7860**.

```bash
docker build -t face-liveness-demo .
docker run --rm -p 7860:7860 face-liveness-demo
```

Open `http://127.0.0.1:7860`. On Hugging Face, `PORT` or `GRADIO_SERVER_PORT` is set automatically; `app.py` reads both.

## Deploy to Hugging Face Spaces

Official guide: [Spaces documentation](https://huggingface.co/docs/hub/spaces) (Gradio SDK, secrets, hardware).

**Note:** This stack pulls **TensorFlow** (via `tf-keras` / DeepFace) and **PyTorch**; the image is large and cold starts can be slow. For production Spaces, consider a **Docker Space** and enough CPU/RAM, or trim dependencies if you fork a minimal build.
