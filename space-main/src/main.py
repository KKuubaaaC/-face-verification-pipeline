"""
Gradio UI dla systemu weryfikacji twarzy z liveness detection.

Zakładki:
  1. „Weryfikacja Live"    — kamera na żywo, PAD, wizualizacja EAR, wynik weryfikacji
  2. „Analiza Badawcza"   — wgranie dwóch zdjęć, dystans kosinusowy, porównanie embeddingów
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

# ── ścieżka do src/ na PYTHONPATH ────────────────────────────────────────────
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

_HF_SPACE_URL: str = "https://huggingface.co/spaces/KKUBBAACC/face_verification"

# ─── Stałe UI ────────────────────────────────────────────────────────────────
_OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
_COLOR_OK = (50, 200, 50)  # BGR zielony
_COLOR_FAIL = (50, 50, 220)  # BGR czerwony

# ─── Lazy init pipeline (HF Spaces) ───────────────────────────────────────────
# Eager load przy imporcie blokuje start: pobieranie buffalo_l trwa minuty → watchdog
# ubija kontener zanim Gradio zdąży odpowiedzieć na healthcheck. Pierwsze kliknięcie
# w UI lub pierwsze użycie ArcFace załaduje modele.
_pipeline: VerificationPipeline | None = None


def _get_pipeline() -> VerificationPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Ładowanie modeli insightface — proszę czekać...")
        _pipeline = VerificationPipeline()
        logger.info("Modele insightface gotowe.")
    return _pipeline


# Lazy-loaded modele badawcze — ładowane przy pierwszym wyborze w UI
_vit_embedder: FaceEmbedder | None = None
_swinface_embedder: SwinFaceEmbedder | None = None
_xai: FaceXAI | None = None


def _get_vit_embedder() -> FaceEmbedder:
    global _vit_embedder
    if _vit_embedder is None:
        logger.info("Ładowanie FaceEmbedder[vit]...")
        _vit_embedder = FaceEmbedder(model_type="vit")
    return _vit_embedder


def _get_swinface_embedder() -> SwinFaceEmbedder:
    global _swinface_embedder
    if _swinface_embedder is None:
        logger.info("Ładowanie SwinFaceEmbedder...")
        _swinface_embedder = SwinFaceEmbedder()
    return _swinface_embedder


_xai_vit: FaceXAI | None = None
_xai_swinface: FaceXAI | None = None


def _get_xai(model_type: str) -> FaceXAI:
    """Zwraca FaceXAI dla danego modelu (lazy-init, osobne instancje)."""
    global _xai_vit, _xai_swinface
    if model_type == "vit":
        if _xai_vit is None:
            _xai_vit = FaceXAI(_vit_model=_get_vit_embedder()._vit)
        return _xai_vit
    else:  # swinface
        if _xai_swinface is None:
            _xai_swinface = FaceXAI(_swinface_model=_get_swinface_embedder()._model)
        return _xai_swinface


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALIZA BADAWCZA
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_two_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    model_choice: str = "ArcFace (Baseline)",
) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Porównuje dwa zdjęcia: ekstrahuje embeddingi, liczy dystans kosinusowy.
    Zwraca: (tekst wyników, obraz porównania side-by-side, mapa XAI side-by-side)
    """
    blank_small = _blank_frame(400, 200)

    if img_a is None or img_b is None:
        return "Wgraj oba zdjęcia.", blank_small, blank_small

    use_arcface = model_choice.startswith("ArcFace")
    use_vit = model_choice.startswith("Vision Transformer")
    use_swinface = model_choice.startswith("SwinFace")

    # Pokaż info o ładowaniu modelu (SwinFace/ViT mogą się długo ładować)
    if use_swinface:
        gr.Info("Ładowanie modelu SwinFace (pierwsze uruchomienie może potrwać 10-15s)...")
    elif use_vit:
        gr.Info("Ładowanie modelu ViT...")

    detector = _get_pipeline()._detector

    bgr_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
    bgr_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR)

    face_a = detector.get_largest_face(bgr_a)
    face_b = detector.get_largest_face(bgr_b)

    if face_a is None:
        return "✗ Nie wykryto twarzy na Zdjęciu A.", blank_small, blank_small
    if face_b is None:
        return "✗ Nie wykryto twarzy na Zdjęciu B.", blank_small, blank_small

    # Wybór backendu ────────────────────────────────────────────────────────────
    try:
        if use_arcface:
            embedder = _get_pipeline()._embedder
            emb_a = embedder.embed(face_a.aligned_crop)
            emb_b = embedder.embed(face_b.aligned_crop)
            result = embedder.verify(emb_a, emb_b)
            model_label = "ArcFace buffalo_l (512-D)"
            threshold_note = f"próg EER: `{VERIFICATION_THRESHOLD}`"
            multitask_text = ""

        elif use_vit:
            embedder = _get_vit_embedder()
            emb_a = embedder.embed(face_a.aligned_crop)
            emb_b = embedder.embed(face_b.aligned_crop)
            result = embedder.verify(emb_a, emb_b)
            model_label = "ViT-B/16 (768-D)"
            threshold_note = "brak progu EER dla ViT — wynik orientacyjny"
            multitask_text = ""

        else:  # SwinFace
            sf = _get_swinface_embedder()
            analysis_a = sf.analyze(face_a.aligned_crop)
            analysis_b = sf.analyze(face_b.aligned_crop)
            result = sf.verify(analysis_a.embedding, analysis_b.embedding)
            model_label = "SwinFace Swin-T (512-D)"
            threshold_note = f"próg EER ArcFace: `{VERIFICATION_THRESHOLD}` (orientacyjny dla SwinFace)"
            multitask_text = _format_swinface_multitask(analysis_a, analysis_b)

    except Exception as exc:
        logger.error("Błąd analyze_two_images: %s", exc, exc_info=True)
        return f"✗ Błąd: {exc}", blank_small, blank_small

    verdict = "MATCH ✓" if result.is_match else "NO MATCH ✗"
    confidence = max(0.0, 1.0 - result.cosine_distance / 0.5)

    text = (
        f"**Wynik:** {verdict}\n\n"
        f"**Model:** `{model_label}`\n\n"
        f"**Dystans kosinusowy:** `{result.cosine_distance:.4f}`  "
        f"({threshold_note})\n\n"
        f"**Orientacyjna zgodność:** `{confidence * 100:.1f}%`\n\n"
        f"**Det. score A:** `{face_a.det_score:.3f}` | "
        f"**Det. score B:** `{face_b.det_score:.3f}`" + (f"\n\n---\n{multitask_text}" if multitask_text else "")
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


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS WIZUALIZACJI
# ═══════════════════════════════════════════════════════════════════════════════


def _make_comparison_image(
    crop_a: np.ndarray,
    crop_b: np.ndarray,
    distance: float,
    is_match: bool,
) -> np.ndarray:
    """Tworzy obraz porównania 2×112px z etykietami i paskiem dystansu."""
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

    # Etykiety pod cropami
    cv2.putText(canvas, "Zdj. A", (x_a + 28, y0 + size + 14), _OVERLAY_FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Zdj. B", (x_b + 28, y0 + size + 14), _OVERLAY_FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Pasek dystansu (0.0 → zielony, 0.5+ → czerwony)
    color = _COLOR_OK if is_match else _COLOR_FAIL
    bar_y = y0 + size + 20
    bar_fill = int(np.clip(distance / 0.5, 0.0, 1.0) * (total_w - pad * 2))
    cv2.rectangle(canvas, (pad, bar_y), (pad + bar_fill, bar_y + 10), color, -1)
    cv2.rectangle(canvas, (pad, bar_y), (total_w - pad, bar_y + 10), (180, 180, 180), 1)

    # Linia progu
    threshold_x = int((VERIFICATION_THRESHOLD / 0.5) * (total_w - pad * 2)) + pad
    cv2.line(canvas, (threshold_x, bar_y - 2), (threshold_x, bar_y + 12), (0, 165, 255), 2)

    # Wynik
    verdict = "MATCH" if is_match else "NO MATCH"
    cv2.putText(canvas, f"{verdict}  d={distance:.4f}", (pad, bar_y + 28), _OVERLAY_FONT, 0.5, color, 1, cv2.LINE_AA)

    return canvas


def _format_swinface_multitask(
    a: SwinFaceAnalysis,
    b: SwinFaceAnalysis,
) -> str:
    """Formatuje wyniki multitask SwinFace jako Markdown (dwie kolumny: A i B)."""

    def _yn(v: bool) -> str:
        return "tak" if v else "nie"

    rows = [
        ("Wiek (regresja)", f"`{a.age:.1f}`", f"`{b.age:.1f}`"),
        ("Płeć", f"{a.gender} ({a.gender_conf * 100:.0f}%)", f"{b.gender} ({b.gender_conf * 100:.0f}%)"),
        (
            "Ekspresja",
            f"{a.expression} ({a.expression_conf * 100:.0f}%)",
            f"{b.expression} ({b.expression_conf * 100:.0f}%)",
        ),
        ("Uśmiech", _yn(a.smiling), _yn(b.smiling)),
        ("Okulary", _yn(a.eyeglasses), _yn(b.eyeglasses)),
    ]

    lines = ["**Analiza multitask SwinFace**\n", "| Cecha | Zdjęcie A | Zdjęcie B |", "|---|---|---|"]
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
    Generuje obraz XAI side-by-side (mapa cieplna A | mapa cieplna B).

    Dla ArcFace — plansza informacyjna (brak XAI).
    crop_a / crop_b : (112, 112, 3) BGR aligned crops
    Zwraca          : (H, W, 3) RGB uint8
    """
    xai_size = 224  # rozmiar każdej mapy

    if use_vit:
        xai_type = "vit"
        xai_label = "Attention Rollout (ViT-B/16)"
    elif use_swinface:
        xai_type = "swinface"
        xai_label = "Global Feature Map (SwinFace Stage-4)"
    else:
        # ArcFace — brak XAI
        w = xai_size * 2 + 32
        h = xai_size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)
        cv2.putText(
            canvas,
            "XAI niedostepne dla ArcFace (CNN)",
            (w // 2 - 180, h // 2 - 10),
            _OVERLAY_FONT,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Wybierz ViT lub SwinFace",
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

    # Etykiety nad mapami
    out = np.zeros((xai_size + 28, side_by_side.shape[1], 3), dtype=np.uint8)
    out[28:, :] = side_by_side
    out[:28, :] = (30, 30, 30)
    cv2.putText(out, f"Zdj. A — {xai_label}", (4, 19), _OVERLAY_FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(out, f"Zdj. B — {xai_label}", (xai_size + 12, 19), _OVERLAY_FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    return out


def _blank_frame(w: int = 640, h: int = 480) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, "Brak obrazu", (w // 2 - 60, h // 2), _OVERLAY_FONT, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  BUDOWA INTERFEJSU GRADIO
# ═══════════════════════════════════════════════════════════════════════════════


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Analiza Badawcza — Embeddingi") as demo:  # noqa: SIM117
        with gr.Tabs():
            with gr.Tab("🔬 Analiza Badawcza — Embeddingi"):
                gr.Markdown(
                    "### Porównanie dwóch zdjęć\n"
                    "Wgraj dwa zdjęcia twarzy. Obliczymy dystans kosinusowy między nimi "
                    f"i zestawimy go z progiem EER `{VERIFICATION_THRESHOLD}`.\n\n"
                    f"---\n[Space na Hugging Face]({_HF_SPACE_URL})"
                )

                with gr.Row():
                    img_input_a = gr.Image(
                        label="Zdjęcie A",
                        sources=["upload", "webcam"],
                        type="numpy",
                        height=280,
                    )
                    img_input_b = gr.Image(
                        label="Zdjęcie B",
                        sources=["upload", "webcam"],
                        type="numpy",
                        height=280,
                    )

                model_radio = gr.Radio(
                    choices=["ArcFace (Baseline)", "Vision Transformer (ViT)", "SwinFace (Swin-T)"],
                    value="ArcFace (Baseline)",
                    label="Wybór architektury ekstrakcji cech",
                )

                compare_btn = gr.Button("Porównaj twarze", variant="primary")

                with gr.Row():
                    research_text = gr.Markdown(label="Wyniki")
                    comparison_img = gr.Image(
                        label="Aligned crops + dystans",
                        type="numpy",
                        height=200,
                        interactive=False,
                    )

                xai_img = gr.Image(
                    label="Wyjaśnialność modelu (Attention Map / XAI)",
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


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

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
