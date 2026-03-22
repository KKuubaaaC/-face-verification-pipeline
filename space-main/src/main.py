"""
Gradio UI for face verification with liveness detection.

Tabs:
  1. Live verification: webcam, PAD, EAR visualization, verification outcome.
  2. Research comparison: upload two images, cosine distance, embedding comparison.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

# Ensure repo root is on PYTHONPATH for `src.*` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import VerificationPipeline
from src.vision.embedder import VERIFICATION_THRESHOLD, FaceEmbedder
from src.vision.swinface_embedder import SwinFaceAnalysis, SwinFaceEmbedder
from src.xai.explainability import FaceXAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- UI constants ---
_OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
_COLOR_OK = (50, 200, 50)  # BGR green
_COLOR_FAIL = (50, 50, 220)  # BGR red

# Eager pipeline init (models load before the UI starts)
logger.info("Loading InsightFace models; please wait...")
_pipeline = VerificationPipeline()
logger.info("Models ready. Starting UI...")


def _get_pipeline() -> VerificationPipeline:
    return _pipeline


# Lazy-loaded research models (first use in the UI)
_vit_embedder: FaceEmbedder | None = None
_swinface_embedder: SwinFaceEmbedder | None = None
_xai: FaceXAI | None = None


def _get_vit_embedder() -> FaceEmbedder:
    global _vit_embedder
    if _vit_embedder is None:
        logger.info("Loading FaceEmbedder[vit]...")
        _vit_embedder = FaceEmbedder(model_type="vit")
    return _vit_embedder


def _get_swinface_embedder() -> SwinFaceEmbedder:
    global _swinface_embedder
    if _swinface_embedder is None:
        logger.info("Loading SwinFaceEmbedder...")
        _swinface_embedder = SwinFaceEmbedder()
    return _swinface_embedder


_xai_vit: FaceXAI | None = None
_xai_swinface: FaceXAI | None = None


def _get_xai(model_type: str) -> FaceXAI:
    """Return FaceXAI for the given model type (lazy init, separate instances)."""
    global _xai_vit, _xai_swinface
    if model_type == "vit":
        if _xai_vit is None:
            _xai_vit = FaceXAI(_vit_model=_get_vit_embedder()._vit)
        return _xai_vit
    else:  # swinface
        if _xai_swinface is None:
            _xai_swinface = FaceXAI(_swinface_model=_get_swinface_embedder()._model)
        return _xai_swinface


# =============================================================================
#  Research comparison (two images)
# =============================================================================


def analyze_two_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    model_choice: str = "ArcFace (Baseline)",
) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Compare two images: extract embeddings and cosine distance.

    Returns (result markdown, side-by-side comparison image, side-by-side XAI map).
    """
    blank_small = _blank_frame(400, 200)

    if img_a is None or img_b is None:
        return "Upload both images.", blank_small, blank_small

    use_vit = model_choice.startswith("Vision Transformer")
    use_swinface = model_choice.startswith("SwinFace")
    if use_swinface:
        gr.Info("Loading SwinFace model (first run may take 10-15s)...")
    elif use_vit:
        gr.Info("Loading ViT model...")

    detector = _get_pipeline()._detector

    bgr_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
    bgr_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR)

    face_a = detector.get_largest_face(bgr_a)
    face_b = detector.get_largest_face(bgr_b)

    if face_a is None:
        return "No face detected in Image A.", blank_small, blank_small
    if face_b is None:
        return "No face detected in Image B.", blank_small, blank_small

    # Backend selection
    use_arcface = model_choice.startswith("ArcFace")
    use_vit = model_choice.startswith("Vision Transformer")
    use_swinface = model_choice.startswith("SwinFace")

    try:
        if use_arcface:
            embedder = _get_pipeline()._embedder
            emb_a = embedder.embed(face_a.aligned_crop)
            emb_b = embedder.embed(face_b.aligned_crop)
            result = embedder.verify(emb_a, emb_b)
            model_label = "ArcFace buffalo_l (512-D)"
            threshold_note = f"EER threshold: `{VERIFICATION_THRESHOLD}`"
            multitask_text = ""

        elif use_vit:
            embedder = _get_vit_embedder()
            emb_a = embedder.embed(face_a.aligned_crop)
            emb_b = embedder.embed(face_b.aligned_crop)
            result = embedder.verify(emb_a, emb_b)
            model_label = "ViT-B/16 (768-D)"
            threshold_note = "no EER threshold for ViT — indicative result"
            multitask_text = ""

        else:  # SwinFace
            sf = _get_swinface_embedder()
            analysis_a = sf.analyze(face_a.aligned_crop)
            analysis_b = sf.analyze(face_b.aligned_crop)
            result = sf.verify(analysis_a.embedding, analysis_b.embedding)
            model_label = "SwinFace Swin-T (512-D)"
            threshold_note = (
                f"ArcFace EER threshold: `{VERIFICATION_THRESHOLD}` (indicative for SwinFace)"
            )
            multitask_text = _format_swinface_multitask(analysis_a, analysis_b)

    except Exception as exc:
        logger.error("analyze_two_images error: %s", exc, exc_info=True)
        return f"Error: {exc}", blank_small, blank_small

    verdict = "MATCH" if result.is_match else "NO MATCH"
    confidence = max(0.0, 1.0 - result.cosine_distance / 0.5)

    text = (
        f"**Result:** {verdict}\n\n"
        f"**Model:** `{model_label}`\n\n"
        f"**Cosine distance:** `{result.cosine_distance:.4f}`  "
        f"({threshold_note})\n\n"
        f"**Confidence estimate:** `{confidence * 100:.1f}%`\n\n"
        f"**Det. score A:** `{face_a.det_score:.3f}` | "
        f"**Det. score B:** `{face_b.det_score:.3f}`"
        + (f"\n\n---\n{multitask_text}" if multitask_text else "")
    )

    comparison = _make_comparison_image(
        face_a.aligned_crop,
        face_b.aligned_crop,
        result.cosine_distance,
        result.is_match,
    )
    comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)

    # XAI ────────────────────────────────────────────────────────────────────
    xai_img = _make_xai_image(
        face_a.aligned_crop,
        face_b.aligned_crop,
        use_vit=use_vit,
        use_swinface=use_swinface,
    )

    return text, comparison_rgb, xai_img


# =============================================================================
#  Visualization helpers
# =============================================================================


def _make_comparison_image(
    crop_a: np.ndarray,
    crop_b: np.ndarray,
    distance: float,
    is_match: bool,
) -> np.ndarray:
    """Build a 2x112px comparison image with labels and distance bar."""
    size = 112
    pad = 16
    bar_h = 36
    total_w = size * 2 + pad * 3
    total_h = size + bar_h + pad * 2

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (45, 45, 45)

    a = cv2.resize(crop_a, (size, size))
    b = cv2.resize(crop_b, (size, size))

    x_a = pad
    x_b = pad * 2 + size
    y0 = pad

    canvas[y0 : y0 + size, x_a : x_a + size] = a
    canvas[y0 : y0 + size, x_b : x_b + size] = b

    cv2.putText(
        canvas,
        "Image A",
        (x_a + 24, y0 + size + 14),
        _OVERLAY_FONT,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Image B",
        (x_b + 24, y0 + size + 14),
        _OVERLAY_FONT,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Distance bar: green toward 0.0, red toward 0.5+
    color = _COLOR_OK if is_match else _COLOR_FAIL
    bar_y = y0 + size + 20
    bar_fill = int(np.clip(distance / 0.5, 0.0, 1.0) * (total_w - pad * 2))
    cv2.rectangle(canvas, (pad, bar_y), (pad + bar_fill, bar_y + 10), color, -1)
    cv2.rectangle(canvas, (pad, bar_y), (total_w - pad, bar_y + 10), (180, 180, 180), 1)

    # Linia progu
    threshold_x = int((VERIFICATION_THRESHOLD / 0.5) * (total_w - pad * 2)) + pad
    cv2.line(canvas, (threshold_x, bar_y - 2), (threshold_x, bar_y + 12), (0, 165, 255), 2)

    # Verdict label
    verdict = "MATCH" if is_match else "NO MATCH"
    cv2.putText(
        canvas,
        f"{verdict}  d={distance:.4f}",
        (pad, bar_y + 28),
        _OVERLAY_FONT,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    return canvas


def _format_swinface_multitask(
    a: SwinFaceAnalysis,
    b: SwinFaceAnalysis,
) -> str:
    """Format SwinFace multitask results as a Markdown table."""

    def _yn(v: bool) -> str:
        return "Yes" if v else "No"

    rows = [
        ("Age (regression)", f"`{a.age:.1f}`", f"`{b.age:.1f}`"),
        (
            "Gender",
            f"{a.gender} ({a.gender_conf * 100:.0f}%)",
            f"{b.gender} ({b.gender_conf * 100:.0f}%)",
        ),
        (
            "Expression",
            f"{a.expression} ({a.expression_conf * 100:.0f}%)",
            f"{b.expression} ({b.expression_conf * 100:.0f}%)",
        ),
        ("Smiling", _yn(a.smiling), _yn(b.smiling)),
        ("Eyeglasses", _yn(a.eyeglasses), _yn(b.eyeglasses)),
    ]

    lines = [
        "**SwinFace Multitask Analysis**\n",
        "| Attribute | Image A | Image B |",
        "|---|---|---|",
    ]
    for name, va, vb in rows:
        lines.append(f"| {name} | {va} | {vb} |")

    return "\n".join(lines)


def _make_xai_image(
    crop_a: np.ndarray,
    crop_b: np.ndarray,
    use_vit: bool = False,
    use_swinface: bool = False,
) -> np.ndarray:
    """
    Build side-by-side XAI visualization (heatmap A | heatmap B).

    For ArcFace, returns an informational placeholder (no XAI path).
    ``crop_a`` / ``crop_b``: (112, 112, 3) BGR aligned crops.
    Returns (H, W, 3) RGB uint8.
    """
    xai_size = 224  # pixels per heatmap tile

    if use_vit:
        xai_type = "vit"
    elif use_swinface:
        xai_type = "swinface"
    else:
        # ArcFace: no attention XAI
        w = xai_size * 2 + 32
        h = xai_size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)
        cv2.putText(
            canvas,
            "XAI not available for ArcFace (CNN)",
            (w // 2 - 190, h // 2 - 10),
            _OVERLAY_FONT,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Select ViT or SwinFace",
            (w // 2 - 120, h // 2 + 20),
            _OVERLAY_FONT,
            0.55,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
        return canvas

    xai = _get_xai(xai_type)
    map_a = xai.generate_attention_map(crop_a, model_type=xai_type)  # (H,W,3) RGB
    map_b = xai.generate_attention_map(crop_b, model_type=xai_type)  # (H,W,3) RGB

    map_a = cv2.resize(map_a, (xai_size, xai_size))
    map_b = cv2.resize(map_b, (xai_size, xai_size))

    sep = np.zeros((xai_size, 8, 3), dtype=np.uint8)
    sep[:] = (60, 60, 60)

    side_by_side = np.concatenate([map_a, sep, map_b], axis=1)  # (224, 456, 3)

    header_h = 24
    total_w = side_by_side.shape[1]
    out = np.zeros((xai_size + header_h, total_w, 3), dtype=np.uint8)
    out[header_h:, :] = side_by_side
    out[:header_h, :] = (30, 30, 30)

    col_b_x = xai_size + 8
    cv2.putText(out, "A", (4, 17), _OVERLAY_FONT, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(out, "B", (col_b_x, 17), _OVERLAY_FONT, 0.5, (210, 210, 210), 1, cv2.LINE_AA)

    return out


def _blank_frame(w: int = 640, h: int = 480) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(
        img, "No image", (w // 2 - 50, h // 2), _OVERLAY_FONT, 0.7, (120, 120, 120), 1, cv2.LINE_AA
    )
    return img


# =============================================================================
#  Gradio UI
# =============================================================================


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Face Verification — Embedding Comparison") as demo:
        with gr.Tabs():
            with gr.Tab("Face comparison"):
                gr.Markdown(
                    "### Compare two face images\n"
                    "Upload two photos — the system computes cosine distance between "
                    f"face embeddings and compares against EER threshold `{VERIFICATION_THRESHOLD}` (ArcFace)."
                )

                with gr.Row():
                    img_input_a = gr.Image(
                        label="Image A",
                        sources=["upload", "webcam"],
                        type="numpy",
                        height=280,
                    )
                    img_input_b = gr.Image(
                        label="Image B",
                        sources=["upload", "webcam"],
                        type="numpy",
                        height=280,
                    )

                model_radio = gr.Radio(
                    choices=["ArcFace (Baseline)", "Vision Transformer (ViT)", "SwinFace (Swin-T)"],
                    value="ArcFace (Baseline)",
                    label="Embedding model",
                )

                compare_btn = gr.Button("Compare faces", variant="primary")

                with gr.Row():
                    research_text = gr.Markdown(label="Results")
                    comparison_img = gr.Image(
                        label="Aligned crops + distance",
                        type="numpy",
                        height=200,
                        interactive=False,
                    )

                xai_img = gr.Image(
                    label="Model explainability (Attention Map / XAI)",
                    type="numpy",
                    height=260,
                    interactive=False,
                )

                compare_btn.click(
                    fn=analyze_two_images,
                    inputs=[img_input_a, img_input_b, model_radio],
                    outputs=[research_text, comparison_img, xai_img],
                )

    return demo


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css="",
    )
