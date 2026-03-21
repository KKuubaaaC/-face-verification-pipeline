"""
Face Verification & Anti-Spoofing — Gradio web interface.

Architecture
------------
- Business logic  : process_verification_frame()
- Visualization   : draw_bounding_box(), draw_label()
- UI layer        : build_ui(), launch()

UI
--
Demo: persona **Kuba** vs **Impostor** (ścieżki na dysku nie są pokazywane w interfejsie), benchmark + galeria do 8 wyników.

Status codes
------------
VERIFIED_MATCH    — tożsamość zgodna z referencją
UNAUTHORIZED      — twarz wykryta, brak zgodności z referencją
SPOOF_DETECTED    — anti-spoofing model rejected the frame
NO_FACE_FOUND     — detector found no face
ERROR             — unexpected exception
"""

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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_HF_SPACE_URL: str = "https://huggingface.co/spaces/KKUBBAACC/adversarial-attack-detection"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
# Ścieżki wewnętrzne — nie wyświetlamy ich w UI.
DB_REAL: str = str(_PROJECT_ROOT / "db" / "ja")
DB_ATTACK: str = str(_PROJECT_ROOT / "db" / "attack")
UI_PERSONA_KUBA: str = "Kuba"
UI_PERSONA_IMPOSTOR: str = "Impostor"
GALLERY_MAX_IMAGES: int = 8
DETECTION_BACKENDS: tuple[str, ...] = ("retinaface", "opencv", "mtcnn")
_MIN_FACE_SIZE_PX: int = 32
_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _is_skipped_asset_filename(filename: str) -> bool:
    """
    Wyklucza zrzuty ekranu i screenshoty — nadal mogą leżeć na dysku, ale nie trafiają do listy ani benchmarku.
    Używa normalizacji Unicode (np. różne warianty „ó”) oraz sprawdzenia początku nazwy.
    """
    stem = unicodedata.normalize("NFC", Path(filename).name.strip()).lower()
    if stem.startswith("zrzut ekranu"):
        return True
    if "zrzut ekranu" in stem:
        return True
    if "screenshot" in stem or stem.startswith("screen shot"):
        return True
    return False


def _folder_for_persona(persona: str) -> Path:
    if persona == UI_PERSONA_KUBA:
        return Path(DB_REAL)
    return Path(DB_ATTACK)


def _sorted_demo_filenames(folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    return sorted(
        p.name
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS and not _is_skipped_asset_filename(p.name)
    )


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
    """Tylko podgląd wejścia / wyjścia — szczegóły są na obrazie wynikowym."""
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
    """Krótkie podsumowanie — bez ścieżek dyskowych w tekście."""
    total = int(summary.get("total") or 0)
    if total == 0:
        return "**Benchmark:** brak plików do przetworzenia (po odfiltrowaniu m.in. zrzutów ekranu)."

    ok = int(summary.get("ok") or 0)
    acc = summary.get("accuracy")
    acc_pct = f"{100.0 * float(acc):.1f}%" if acc is not None else "—"

    k_t = int(summary.get("kuba_total") or 0)
    k_ok = int(summary.get("kuba_ok") or 0)
    k_a = summary.get("kuba_accuracy")
    k_pct = f"{100.0 * float(k_a):.1f}%" if k_a is not None else "—"

    i_t = int(summary.get("impostor_total") or 0)
    i_ok = int(summary.get("impostor_ok") or 0)
    i_a = summary.get("impostor_accuracy")
    i_pct = f"{100.0 * float(i_a):.1f}%" if i_a is not None else "—"

    return (
        f"**Łącznie:** {ok}/{total} zgodnych z oczekiwaniem (**{acc_pct}**).\n\n"
        f"- **Kuba:** {k_ok}/{k_t} ({k_pct}) — oczekiwane: **{STATUS_VERIFIED}**\n"
        f"- **Impostor:** {i_ok}/{i_t} ({i_pct}) — oczekiwane: **{STATUS_UNAUTH}** "
        f"(lub **{STATUS_SPOOF}** jako poprawne odrzucenie)"
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
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS and not _is_skipped_asset_filename(p.name)
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
            ok = status == expected or status == STATUS_SPOOF

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
    """Benchmark: Kuba + Impostor; galeria — pierwsze GALLERY_MAX_IMAGES wyników (najpierw Kuba, potem Impostor)."""
    rows: list[dict[str, Any]] = []
    gallery_rgbs: list[np.ndarray] = []

    _process_folder_for_benchmark(UI_PERSONA_KUBA, Path(DB_REAL), STATUS_VERIFIED, rows, gallery_rgbs)
    _process_folder_for_benchmark(UI_PERSONA_IMPOSTOR, Path(DB_ATTACK), STATUS_UNAUTH, rows, gallery_rgbs)

    df = pd.DataFrame(rows)
    _bench_cols_pl = {
        "group": "zestaw",
        "file": "plik",
        "status": "status",
        "expected": "oczekiwanie",
        "ok": "zgodność",
    }

    if df.empty:
        summary = {
            "total": 0,
            "ok": 0,
            "accuracy": None,
            "kuba_total": 0,
            "kuba_ok": 0,
            "kuba_accuracy": None,
            "impostor_total": 0,
            "impostor_ok": 0,
            "impostor_accuracy": None,
        }
        empty = pd.DataFrame(columns=list(_bench_cols_pl.values()))
        return empty, _benchmark_summary_markdown(summary), []

    total = int(len(df))
    ok_count = int(df["ok"].sum())
    kub = df[df["group"] == UI_PERSONA_KUBA]
    imp = df[df["group"] == UI_PERSONA_IMPOSTOR]
    k_ok = int(kub["ok"].sum()) if not kub.empty else 0
    i_ok = int(imp["ok"].sum()) if not imp.empty else 0

    summary = {
        "total": total,
        "ok": ok_count,
        "accuracy": round(ok_count / total, 4) if total else None,
        "kuba_total": int(len(kub)),
        "kuba_ok": k_ok,
        "kuba_accuracy": round(k_ok / len(kub), 4) if len(kub) else None,
        "impostor_total": int(len(imp)),
        "impostor_ok": i_ok,
        "impostor_accuracy": round(i_ok / len(imp), 4) if len(imp) else None,
    }
    public_df = df[list(_bench_cols_pl.keys())].rename(columns=_bench_cols_pl)
    return public_df, _benchmark_summary_markdown(summary), gallery_rgbs


def _extract_face_with_fallback(
    image: np.ndarray,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """
    Try multiple detector backends to improve robustness on CPU-only HF Spaces.
    Returns best face candidate, backend name and anti-spoof availability.
    """
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


# ---------------------------------------------------------------------------
# UI definition
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Demo: weryfikacja Kuba vs Impostor") as demo:
        gr.Markdown(
            "### Demo: weryfikacja tożsamości i ochrona przed spoofingiem\n"
            "**Kuba** to zdjęcia zgodne z referencją. **Impostor** to próby obejścia systemu. "
            "Wybierz plik i kliknij **Uruchom przykład**. Po prawej zobaczysz ten sam kadr z adnotacją, "
            "czyli status oraz ramkę wokół twarzy.\n\n"
            "**Uruchom benchmark** przetwarza po kolei oba zestawy, buduje tabelę "
            "z kolumnami *zestaw*, *plik*, *status*, *oczekiwanie* i *zgodność*, a także galerię do **8 obrazów** "
            "(najpierw Kuba, potem Impostor).\n\n"
            f"---\n[Space na Hugging Face]({_HF_SPACE_URL})"
        )

        _files0 = _file_choices_for_persona(UI_PERSONA_KUBA)
        with gr.Row():
            persona_radio = gr.Radio(
                choices=[UI_PERSONA_KUBA, UI_PERSONA_IMPOSTOR],
                value=UI_PERSONA_KUBA,
                label="Zestaw",
            )
            example_file = gr.Dropdown(
                choices=_files0,
                value=(_files0[0] if _files0 else None),
                label="Plik",
            )
            run_example_btn = gr.Button("Uruchom przykład", variant="primary")

        with gr.Row():
            example_input_image = gr.Image(
                label="Wejście",
                type="numpy",
                interactive=False,
                height=360,
            )
            example_output_image = gr.Image(
                label="Wynik (z adnotacją)",
                type="numpy",
                interactive=False,
                height=360,
            )

        gr.Markdown("---")
        run_benchmark_btn = gr.Button("Uruchom benchmark", variant="secondary")
        benchmark_table = gr.Dataframe(
            headers=["zestaw", "plik", "status", "oczekiwanie", "zgodność"],
            datatype=["str", "str", "str", "str", "bool"],
            interactive=False,
            wrap=True,
            label="Wyniki benchmarku",
        )
        benchmark_summary = gr.Markdown(
            value="*Kliknij **Uruchom benchmark**, aby zobaczyć podsumowanie i galerię.*",
            label="Podsumowanie",
        )

        gr.Markdown("#### Galeria wyników (max 8 zdjęć)")
        result_gallery = gr.Gallery(
            label="Adnotowane obrazy z pierwszych pozycji zestawu",
            columns=4,
            height=480,
            object_fit="contain",
            show_label=True,
        )

        persona_radio.change(
            fn=_refresh_file_dropdown,
            inputs=[persona_radio],
            outputs=[example_file],
        )
        run_example_btn.click(
            fn=run_example_verification,
            inputs=[persona_radio, example_file],
            outputs=[example_input_image, example_output_image],
        )
        run_benchmark_btn.click(
            fn=run_benchmark_all_examples,
            outputs=[benchmark_table, benchmark_summary, result_gallery],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch() -> None:
    import os

    logger.info("Starting Face Verification & Anti-Spoofing.")
    logger.info("Reference DB : %s", DB_REAL)
    demo = build_ui()
    if not os.environ.get("SPACE_ID"):
        port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
        demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)


if __name__ == "__main__":
    launch()
