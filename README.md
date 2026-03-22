# Face Verification Research Pipeline

End-to-end research workflow for face verification on AgeDB-30 and LFW-style protocols: detection, alignment, embedding extraction, ROC/EER evaluation, explainability, and presentation attack detection.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Architecture

```
Input Image
  → Face Detection (RetinaFace / 2d106det landmarks)
  → Alignment (5-point similarity / affine warp to 112×112)
  → Embedding (ArcFace 512-D, or GhostFaceNetV2 / ViT / SwinFace where configured)
  → Cosine distance (1 − cosine similarity on L2-normalized vectors)
  → Decision (threshold at EER or fixed FAR operating point)
```

## Results

Verification metrics from `results/evaluation_metrics.csv` (produced by `notebooks/04_evaluate.ipynb`; balanced 6k-pair protocols). TAR@FAR is reported at **FAR = 10⁻³** (0.1%).

| Model | Dataset | AUC | EER | TAR@FAR=10⁻³ |
| --- | --- | --- | --- | --- |
| ArcFace (buffalo_l / w600k_r50) | AgeDB-30 | 0.9906 | 2.50% | 94.60% |
| ArcFace (buffalo_l / w600k_r50) | LFW | 0.9995 | 0.33% | 99.73% |
| GhostFaceNetV2 | AgeDB-30 | 0.9894 | 3.32% | 91.23% |
| GhostFaceNetV2 | LFW | 0.9988 | 0.57% | 98.67% |

## Presentation Attack Detection

Multi-layer PAD stack evaluated in `notebooks/07_PAD_analysis.ipynb` (summary metrics in `notebooks/pad_output/pad_summary_metrics.csv`):

1. **Blink detection (EAR)** — eye aspect ratio over sequential frames to detect natural blink dynamics vs. static spoof.
2. **FFT moiré** — frequency-domain cues for screen-replay artefacts.
3. **LBP texture** — local binary patterns for print/texture inconsistencies.
4. **Specular highlights** — highlight patterns inconsistent with real skin reflectance.
5. **MobileNetV2 + Grad-CAM** — weakly supervised region importance for CNN-based liveness scoring.
6. **Physical features + SVM** — hand-crafted geometry/photometric features with a linear SVM; on the held-out protocol in this repo: **accuracy 95.44%**, **AUC 0.979** (see `pad_summary_metrics.csv`).

## Project structure

```
.
├── data/                 # AgeDB / LFW-style trees (local only — not in Git; see .gitignore)
├── eda_output/           # EDA tables, figures, per-dataset CSV caches (e.g. AgeDB/LFW)
├── notebooks/            # Numbered research notebooks (detection → PAD)
├── results/              # Parsed pairs, distance caches, evaluation_metrics.csv
├── logs/                 # Optional detection/parse logs
├── scripts/              # parse_pairs.py — img.list/pair.list → CSV
├── space-main/           # Gradio demo (ArcFace, ViT, SwinFace) + PAD / XAI hooks
├── space-liveness/       # Alternate Space bundle (same stack pattern)
├── Pipeline/             # Full end-to-end pipeline notebook
├── third_party/          # SwinFace sources — see scripts/bootstrap_swinface.sh (or your own copy)
├── models/               # Local checkpoints (e.g. SwinFace); large weights not always committed
├── Dockerfile            # Reproducible env (uv + Jupyter on port 8888)
├── pyproject.toml        # Package metadata and dependencies (MIT)
└── uv.lock               # Locked dependency versions (uv)
```

## Quick start

```bash
uv sync

# Step 1: parse protocols to CSV
python scripts/parse_pairs.py

# Steps 2–5: run notebooks 01–04 (detection → evaluation)
jupyter notebook notebooks/

# Optional Gradio demo (from repo root)
cd space-main && python app.py
```

**Hugging Face Spaces:** [face verification (space-main)](https://huggingface.co/spaces/KKUBBAACC/verification) · [liveness / anti-spoofing (space-liveness)](https://huggingface.co/spaces/KKUBBAACC/attack-detection)

### SwinFace sources (`third_party/swinface`)

SwinFace is loaded from upstream [lxq1000/SwinFace](https://github.com/lxq1000/SwinFace) (`swinface_project/`). After cloning this repo, if `third_party/swinface/swinface_project` is missing, run:

```bash
bash scripts/bootstrap_swinface.sh
```

That shallow-clones into `third_party/swinface`, `space-main/third_party/swinface`, and `space-liveness/third_party/swinface` when those paths are empty. If you already keep your own tree there (without a nested `.git`), the script skips it. To track SwinFace as a submodule instead, use `git submodule` against the same URL.

## Docker

```bash
docker build -t face-verification .
docker run -p 8888:8888 face-verification
```

The image starts Jupyter Notebook on `0.0.0.0:8888` (see `Dockerfile` `CMD`). Mount data or copy artifacts into the container as needed for your runs.

## Notebooks

Exploratory EDA is in `00_EDA.ipynb`. The seven numbered pipeline notebooks below cover detection through PAD.

| Notebook | Purpose |
| --- | --- |
| `01_detect_align.ipynb` | Face detection and landmark-based alignment to canonical crops |
| `02_extract_embeddings.ipynb` | 512-D embeddings (ArcFace, GhostFaceNetV2) and caches |
| `03_compare_pairs.ipynb` | Pairwise scores and distance distributions |
| `04_evaluate.ipynb` | ROC, AUC, EER, TAR@FAR; writes `results/evaluation_metrics.csv` |
| `05_xai_gradcam.ipynb` | Siamese Grad-CAM explanations for verification pairs |
| `06_SOTA_ViT_Evaluation.ipynb` | ViT-B/16 and comparison against strong baselines |
| `07_PAD_analysis.ipynb` | Presentation attack detection layers, metrics, and plots |

## Models

| Model | Architecture | Embedding dim | Weights / source |
| --- | --- | --- | --- |
| ArcFace | ResNet-50 (buffalo_l), ArcFace loss | 512 | InsightFace `w600k_r50` / `buffalo_l` |
| GhostFaceNetV2 | GhostNet-style backbone (via DeepFace) | 512 | DeepFace `GhostFaceNet` (downloads on first use) |
| SwinFace | Swin-T + FAM + multitask heads | 512 | Local `models/swinface.pt` or HF (see `space-main` README) |
| ViT-B/16 | torchvision ViT, classifier head removed | 768 | torchvision ImageNet-21k weights |

## XAI

**Siamese Grad-CAM** (`05_xai_gradcam.ipynb`): class-discriminative saliency on the Siamese verification network to show which regions drive the match score. **Attention rollout** (where used with ViT): propagation of attention weights across layers for a coarse spatial attribution map (Abnar & Zuidema, 2020).

## References

- Deng, Guo, Xue, Zafeiriou. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR, 2019.
- Liu et al. *SwinFace: A Multi-task Transformer for Face Recognition.* arXiv:2308.11509, 2023.
- Soukupová & Čech. *Real-Time Eye Blink Detection Using Facial Landmarks.* CVWW, 2016 (EAR).
- Abnar & Zuidema. *Quantifying Attention Flow in Transformers.* ACL, 2020 (attention rollout).

## License: MIT

This project is licensed under the MIT License (see `pyproject.toml`).
