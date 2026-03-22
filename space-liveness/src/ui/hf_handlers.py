"""Hugging Face Gradio demo: DeepFace verification, visualization, and benchmark logic."""

from __future__ import annotations

import logging
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from deepface import DeepFace

from src.ui import hf_strings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DB_REAL: str = str(_PROJECT_ROOT / "db" / "ja")
DB_ATTACK: str = str(_PROJECT_ROOT / "db" / "attack")
UI_PERSONA_REFERENCE: str = "Reference"
UI_PERSONA_ATTACK: str = "Attack"
GALLERY_MAX_IMAGES: int = 8
DETECTION_BACKENDS: tuple[str, ...] = ("retinaface", "opencv", "mtcnn")
_MIN_FACE_SIZE_PX: int = 32
_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _is_skipped_asset_filename(filename: str) -> bool:
    stem = unicodedata.normalize("NFC", Path(filename).name.strip()).lower()
    return (
        stem.startswith("zrzut ekranu")
        or "zrzut ekranu" in stem
        or "screenshot" in stem
        or stem.startswith("screen shot")
    )


def _folder_for_persona(persona: str) -> Path:
    if persona == UI_PERSONA_REFERENCE:
        return Path(DB_REAL)
    return Path(DB_ATTACK)


def _sorted_demo_filenames(folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    return sorted(
        p.name
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTS
        and not _is_skipped_asset_filename(p.name)
    )


STATUS_VERIFIED: str = "VERIFIED_MATCH"
STATUS_UNAUTH: str = "UNAUTHORIZED"
STATUS_SPOOF: str = "SPOOF_DETECTED"
STATUS_NO_FACE: str = "NO_FACE_FOUND"
STATUS_ERROR: str = "ERROR"

COLOR_VERIFIED: tuple[int, int, int] = (34, 177, 76)
COLOR_UNAUTH: tuple[int, int, int] = (0, 140, 255)
COLOR_SPOOF: tuple[int, int, int] = (0, 0, 220)
COLOR_ERROR: tuple[int, int, int] = (160, 160, 160)

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


def draw_bounding_box(
    image: np.ndarray,
    region: dict[str, int],
    color: tuple[int, int, int],
) -> None:
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


def process_verification_frame(
    image: np.ndarray,
    db_path: str = DB_REAL,
) -> tuple[np.ndarray, str, dict[str, Any]]:
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

    face, backend_used, antispoof_available = _extract_face_with_fallback(image)
    metadata["detector_backend"] = backend_used
    metadata["antispoof_available"] = antispoof_available

    if face is None:
        logger.info("NO_FACE_FOUND — all detector backends failed.")
        _stamp_time(metadata, t_start)
        _annotate(annotated, STATUS_NO_FACE, region=None)
        return annotated, STATUS_NO_FACE, metadata

    region: dict[str, int] | None = face.get("facial_area")
    is_real: bool | None = face.get("is_real")
    antispoof_score: float = float(face.get("antispoof_score", 0.0))

    metadata["is_real"] = is_real
    metadata["antispoof_score"] = round(antispoof_score, 4)

    if region:
        metadata["face_area_px"] = int(region["w"]) * int(region["h"])

    logger.info(
        "Anti-spoofing: backend=%s available=%s is_real=%s score=%.4f face_area_px=%s image_shape=%s",
        backend_used,
        antispoof_available,
        is_real,
        antispoof_score,
        metadata["face_area_px"],
        image.shape,
    )

    if antispoof_available and (is_real is False):
        logger.warning("SPOOF_DETECTED — score=%.4f", antispoof_score)
        _stamp_time(metadata, t_start)
        _annotate(annotated, STATUS_SPOOF, region)
        return annotated, STATUS_SPOOF, metadata

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


def _stamp_time(metadata: dict[str, Any], t_start: float) -> None:
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    metadata["processing_time_ms"] = round(elapsed_ms, 1)


def _annotate(
    image: np.ndarray,
    status: str,
    region: dict[str, int] | None,
    detail: str = "",
) -> None:
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


def _file_choices_for_persona(persona: str) -> list[str]:
    return _sorted_demo_filenames(_folder_for_persona(persona))


def _refresh_file_dropdown(persona: str):
    choices = _file_choices_for_persona(persona)
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def run_example_verification(
    persona: str,
    filename: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not filename:
        return None, None

    image_path = _folder_for_persona(persona) / filename
    if not image_path.exists():
        return None, None

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None, None

    annotated_bgr, _status, _meta = process_verification_frame(bgr)
    input_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return input_rgb, output_rgb


def _benchmark_summary_markdown(summary: dict[str, Any]) -> str:
    total = int(summary.get("total") or 0)
    if total == 0:
        return hf_strings.BENCHMARK_NO_FILES

    ok = int(summary.get("ok") or 0)
    acc = summary.get("accuracy")
    acc_pct = f"{100.0 * float(acc):.1f}%" if acc is not None else "—"

    k_t = int(summary.get("reference_total") or 0)
    k_ok = int(summary.get("reference_ok") or 0)
    k_a = summary.get("reference_accuracy")
    k_pct = f"{100.0 * float(k_a):.1f}%" if k_a is not None else "—"

    i_t = int(summary.get("attack_total") or 0)
    i_ok = int(summary.get("attack_ok") or 0)
    i_a = summary.get("attack_accuracy")
    i_pct = f"{100.0 * float(i_a):.1f}%" if i_a is not None else "—"

    return (
        f"**Overall:** {ok}/{total} matched expectation (**{acc_pct}**).\n\n"
        f"- **Reference:** {k_ok}/{k_t} ({k_pct}) — expected: **{STATUS_VERIFIED}**\n"
        f"- **Attack:** {i_ok}/{i_t} ({i_pct}) — expected: **{STATUS_UNAUTH}** "
        f"(or **{STATUS_SPOOF}** as correct rejection)"
    )


def _process_folder_for_benchmark(
    label: str,
    folder: Path,
    expected: str,
    rows: list[dict[str, Any]],
    gallery_rgbs: list[np.ndarray],
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTS
        and not _is_skipped_asset_filename(p.name)
    ):
        bgr = cv2.imread(str(file_path))
        if bgr is None:
            rows.append(
                {
                    "group": label,
                    "file": file_path.name,
                    "status": STATUS_ERROR,
                    "expected": expected,
                    "ok": False,
                    "distance": None,
                    "antispoof_available": None,
                    "detector_backend": None,
                    "error": "Cannot read image",
                }
            )
            continue

        annotated_bgr, status, meta = process_verification_frame(bgr)
        if expected == STATUS_VERIFIED:
            ok = status == expected
        else:
            ok = status in (expected, STATUS_SPOOF)

        rows.append(
            {
                "group": label,
                "file": file_path.name,
                "status": status,
                "expected": expected,
                "ok": bool(ok),
                "distance": meta.get("distance"),
                "antispoof_available": meta.get("antispoof_available"),
                "detector_backend": meta.get("detector_backend"),
                "error": meta.get("error"),
            }
        )
        if len(gallery_rgbs) < GALLERY_MAX_IMAGES:
            gallery_rgbs.append(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))


def run_benchmark_all_examples() -> tuple[pd.DataFrame, str, list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    gallery_rgbs: list[np.ndarray] = []

    _process_folder_for_benchmark(
        UI_PERSONA_REFERENCE, Path(DB_REAL), STATUS_VERIFIED, rows, gallery_rgbs
    )
    _process_folder_for_benchmark(
        UI_PERSONA_ATTACK, Path(DB_ATTACK), STATUS_UNAUTH, rows, gallery_rgbs
    )

    df = pd.DataFrame(rows)
    _bench_cols_en = {
        "group": "set",
        "file": "file",
        "status": "status",
        "expected": "expected",
        "ok": "match",
    }

    if df.empty:
        summary = {
            "total": 0,
            "ok": 0,
            "accuracy": None,
            "reference_total": 0,
            "reference_ok": 0,
            "reference_accuracy": None,
            "attack_total": 0,
            "attack_ok": 0,
            "attack_accuracy": None,
        }
        empty = pd.DataFrame(columns=list(_bench_cols_en.values()))
        return empty, _benchmark_summary_markdown(summary), []

    total = len(df)
    ok_count = int(df["ok"].sum())
    kub = df[df["group"] == UI_PERSONA_REFERENCE]
    imp = df[df["group"] == UI_PERSONA_ATTACK]
    k_ok = int(kub["ok"].sum()) if not kub.empty else 0
    i_ok = int(imp["ok"].sum()) if not imp.empty else 0

    summary = {
        "total": total,
        "ok": ok_count,
        "accuracy": round(ok_count / total, 4) if total else None,
        "reference_total": len(kub),
        "reference_ok": k_ok,
        "reference_accuracy": round(k_ok / len(kub), 4) if len(kub) else None,
        "attack_total": len(imp),
        "attack_ok": i_ok,
        "attack_accuracy": round(i_ok / len(imp), 4) if len(imp) else None,
    }
    public_df = df[list(_bench_cols_en.keys())].rename(columns=_bench_cols_en)
    return public_df, _benchmark_summary_markdown(summary), gallery_rgbs


def _extract_face_with_fallback(
    image: np.ndarray,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    antispoof_available = True

    for backend in DETECTION_BACKENDS:
        try:
            faces: list[dict[str, Any]] = DeepFace.extract_faces(
                img_path=image,
                detector_backend=backend,
                enforce_detection=False,
                anti_spoofing=True,
            )
        except Exception as exc:
            err = str(exc)
            if "install torch" in err.lower():
                antispoof_available = False
                logger.warning(
                    "Anti-spoofing unavailable (backend=%s): %s. Retrying without anti-spoofing.",
                    backend,
                    err,
                )
                try:
                    faces = DeepFace.extract_faces(
                        img_path=image,
                        detector_backend=backend,
                        enforce_detection=False,
                        anti_spoofing=False,
                    )
                except Exception as retry_exc:
                    logger.warning(
                        "extract_faces fallback failed for backend=%s: %s",
                        backend,
                        retry_exc,
                    )
                    continue
            else:
                logger.warning("extract_faces failed for backend=%s: %s", backend, exc)
                continue

        if not faces:
            continue

        best_face = _pick_largest_valid_face(faces)
        if best_face is not None:
            return best_face, backend, antispoof_available

    return None, None, antispoof_available


def _pick_largest_valid_face(faces: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for face in faces:
        region = face.get("facial_area")
        if not region:
            continue

        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
        if w < _MIN_FACE_SIZE_PX or h < _MIN_FACE_SIZE_PX:
            continue

        candidates.append((w * h, face))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
