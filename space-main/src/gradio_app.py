"""
Face Verification & Anti-Spoofing — Gradio web interface.

Architecture
------------
- Business logic  : process_verification_frame()
- Visualization   : draw_bounding_box(), draw_label()
- UI layer        : build_ui(), launch()

Webcam handling
---------------
Photo capture/upload uses gr.Image(sources=["upload", "webcam"])
and a button trigger (snapshot mode, no live stream processing).

Status codes
------------
VERIFIED_MATCH    — real face, identity confirmed in db/ja/
UNAUTHORIZED      — real face, identity NOT in db/ja/
SPOOF_DETECTED    — anti-spoofing model rejected the frame
NO_FACE_FOUND     — detector found no face
ERROR             — unexpected exception
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from deepface import DeepFace

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DB_REAL: str = str(_PROJECT_ROOT / "db" / "ja")

# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------
STATUS_VERIFIED: str = "VERIFIED_MATCH"
STATUS_UNAUTH: str = "UNAUTHORIZED"
STATUS_SPOOF: str = "SPOOF_DETECTED"
STATUS_NO_FACE: str = "NO_FACE_FOUND"
STATUS_ERROR: str = "ERROR"

# ---------------------------------------------------------------------------
# Visualization constants  (BGR for OpenCV)
# ---------------------------------------------------------------------------
COLOR_VERIFIED: tuple[int, int, int] = (34, 177, 76)  # green
COLOR_UNAUTH: tuple[int, int, int] = (0, 140, 255)  # orange
COLOR_SPOOF: tuple[int, int, int] = (0, 0, 220)  # red
COLOR_ERROR: tuple[int, int, int] = (160, 160, 160)  # grey

_STATUS_COLOR: dict[str, tuple[int, int, int]] = {
    STATUS_VERIFIED: COLOR_VERIFIED,
    STATUS_UNAUTH: COLOR_UNAUTH,
    STATUS_SPOOF: COLOR_SPOOF,
    STATUS_NO_FACE: COLOR_ERROR,
    STATUS_ERROR: COLOR_ERROR,
}

FONT: int = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.55
FONT_THICKNESS: int = 1
LABEL_PADDING: int = 6
BOX_THICKNESS: int = 2

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def draw_bounding_box(
    image: np.ndarray,
    region: dict[str, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a rectangle on *image* in-place using *region* (x, y, w, h)."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, BOX_THICKNESS)


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    bg_alpha: float = 0.70,
) -> None:
    """
    Draw *text* with a semi-transparent filled background rectangle.

    The background guarantees readability on any image brightness.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    x, y = origin
    pad = LABEL_PADDING
    h_img, w_img = image.shape[:2]

    x1 = max(x - pad, 0)
    y1 = max(y - text_h - pad, 0)
    x2 = min(x + text_w + pad, w_img)
    y2 = min(y + baseline + pad, h_img)

    roi = image[y1:y2, x1:x2]
    if roi.size > 0:
        bg = np.zeros_like(roi)
        image[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - bg_alpha, bg, bg_alpha, 0)

    cv2.putText(
        image,
        text,
        (x, y),
        FONT,
        FONT_SCALE,
        color,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------


def process_verification_frame(
    image: np.ndarray,
    db_path: str = DB_REAL,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """
    Run anti-spoofing then face verification on a single BGR frame.

    Parameters
    ----------
    image   : BGR uint8 ndarray.
    db_path : Path to the reference face database directory.

    Returns
    -------
    annotated  : BGR uint8 ndarray with bbox and label.
    status     : One of the STATUS_* constants.
    metadata   : Dict with diagnostic metrics:
                   processing_time_ms, antispoof_score, is_real,
                   face_area_px, identity, distance, error.
    """
    annotated: np.ndarray = image.copy()
    t_start: float = time.perf_counter()

    metadata: dict[str, Any] = {
        "processing_time_ms": None,
        "antispoof_score": None,
        "is_real": None,
        "face_area_px": None,
        "identity": None,
        "distance": None,
        "error": None,
    }

    # -- Step 1: face extraction + anti-spoofing ----------------------------
    try:
        faces: list[dict[str, Any]] = DeepFace.extract_faces(
            img_path=image,
            detector_backend="retinaface",
            enforce_detection=True,
            anti_spoofing=True,
        )
    except ValueError:
        logger.info("NO_FACE_FOUND — detector returned no face.")
        _stamp_time(metadata, t_start)
        _annotate(annotated, STATUS_NO_FACE, region=None)
        return annotated, STATUS_NO_FACE, metadata
    except Exception as exc:
        logger.error("extract_faces raised: %s", exc)
        metadata["error"] = str(exc)
        _stamp_time(metadata, t_start)
        _annotate(annotated, STATUS_ERROR, region=None)
        return annotated, STATUS_ERROR, metadata

    if not faces:
        logger.info("NO_FACE_FOUND — empty result list.")
        _stamp_time(metadata, t_start)
        _annotate(annotated, STATUS_NO_FACE, region=None)
        return annotated, STATUS_NO_FACE, metadata

    face: dict[str, Any] = faces[0]
    region: dict[str, int] | None = face.get("facial_area")
    is_real: bool | None = face.get("is_real")
    antispoof_score: float = float(face.get("antispoof_score", 0.0))

    metadata["is_real"] = is_real
    metadata["antispoof_score"] = round(antispoof_score, 4)

    if region:
        metadata["face_area_px"] = int(region["w"]) * int(region["h"])

    logger.info(
        "Anti-spoofing: is_real=%s score=%.4f face_area_px=%s",
        is_real,
        antispoof_score,
        metadata["face_area_px"],
    )

    if not is_real:
        logger.warning("SPOOF_DETECTED — score=%.4f", antispoof_score)
        _stamp_time(metadata, t_start)
        _annotate(annotated, STATUS_SPOOF, region)
        return annotated, STATUS_SPOOF, metadata

    # -- Step 2: identity verification --------------------------------------
    try:
        results: list[pd.DataFrame] = DeepFace.find(
            img_path=image,
            db_path=db_path,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False,
            silent=True,
        )
        df: pd.DataFrame = results[0] if results else pd.DataFrame()
    except Exception as exc:
        logger.error("DeepFace.find raised: %s", exc)
        metadata["error"] = str(exc)
        _stamp_time(metadata, t_start)
        _annotate(annotated, STATUS_ERROR, region)
        return annotated, STATUS_ERROR, metadata

    if df.empty:
        logger.info("UNAUTHORIZED — no match found in %s", db_path)
        _stamp_time(metadata, t_start)
        _annotate(annotated, STATUS_UNAUTH, region)
        return annotated, STATUS_UNAUTH, metadata

    best = df.iloc[0]
    identity = Path(str(best["identity"])).stem
    distance = round(float(best.get("distance", best.iloc[-1])), 4)

    metadata["identity"] = identity
    metadata["distance"] = distance

    logger.info("VERIFIED_MATCH — identity=%s distance=%.4f", identity, distance)
    _stamp_time(metadata, t_start)
    _annotate(
        annotated,
        STATUS_VERIFIED,
        region,
        detail=f"{identity}  dist={distance:.3f}",
    )
    return annotated, STATUS_VERIFIED, metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stamp_time(metadata: dict[str, Any], t_start: float) -> None:
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    metadata["processing_time_ms"] = round(elapsed_ms, 1)


def _annotate(
    image: np.ndarray,
    status: str,
    region: dict[str, int] | None,
    detail: str = "",
) -> None:
    """Overlay bbox + status label (+ optional detail line) on *image* in-place."""
    color = _STATUS_COLOR.get(status, COLOR_ERROR)

    if region:
        draw_bounding_box(image, region, color)
        label_x = region["x"]
        label_y = max(region["y"] - LABEL_PADDING, 18)
    else:
        label_x, label_y = 10, 28

    draw_label(image, status, (label_x, label_y), color)

    if detail:
        draw_label(image, detail, (label_x, label_y + 22), color)


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------


def _empty_meta() -> dict[str, Any]:
    return {
        "processing_time_ms": None,
        "antispoof_score": None,
        "is_real": None,
        "face_area_px": None,
        "identity": None,
        "distance": None,
        "error": None,
    }


def on_upload_verify(
    image_rgb: np.ndarray | None,
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    """Callback for the Upload tab — triggered by button click."""
    if image_rgb is None:
        return None, STATUS_NO_FACE, _empty_meta()

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr, status, meta = process_verification_frame(bgr)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), status, meta


# ---------------------------------------------------------------------------
# UI definition
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Face Verification & Anti-Spoofing") as demo:
        gr.Markdown(
            "### Weryfikacja twarzy (snapshot)\n"
            "Wgraj zdjęcie lub zrób fotografię kamerą, a następnie kliknij "
            "**Run verification**. Tryb live stream jest wyłączony."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input photo",
                    sources=["upload", "webcam"],
                    type="numpy",
                    height=380,
                )
                verify_btn = gr.Button("Run verification", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Annotated result",
                    type="numpy",
                    interactive=False,
                    height=380,
                )

        with gr.Row():
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                max_lines=1,
            )

        with gr.Row():
            metrics_json = gr.JSON(label="Metrics")

        verify_btn.click(
            fn=on_upload_verify,
            inputs=[input_image],
            outputs=[output_image, status_box, metrics_json],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch() -> None:
    logger.info("Starting Face Verification & Anti-Spoofing.")
    logger.info("Reference DB : %s", DB_REAL)
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    launch()
