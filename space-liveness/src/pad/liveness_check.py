"""
verify_liveness — główna funkcja weryfikacji żywotności twarzy (single-image PAD).

Pipeline (zgodnie z 01_detect_align.pdf + spec modułów):

  1. Detekcja i wyrównanie (FaceDetector / buffalo_l + 2d106det)
     Wynik: aligned_crop (112×112 BGR).  Jeśli brak twarzy → FAIL.

  2. Analiza fizyczna (PhysicalFeatureExtractor):
       LBP uniform (skimage, P=8, R=1) → entropia histogramu
         Naturalna skóra: dużo równomiernych wzorców → wyższa entropia
         Wydruk / ekran: dominacja kilku wzorców → niższa entropia
       FFT log-amplituda → energia wysokich częstotliwości
         (kalibrowana przez MoireDetector z istniejącymi progami)

  3. AntiSpoofingModel (MobileNetV2GradCAM):
       Grad-CAM z warstwy out_relu (features[-1]) → mapa (7×7 → 112×112)
       Region scoring (z PDF §9):
         Real score  ↑: Eyes (×0.5) + Mouth (×0.3) + Nose (×0.2)
         Spoof score ↑: krawędzie (Top/Bottom/Left/Right)

  4. Łączenie score'ów:
       physical_score = 0.50·LBP_entropy + 0.35·moire_live + 0.15·specular_live
       dl_score       = gcam.real_score  (∈ [0, 1])
       liveness_score = 0.40·physical + 0.60·dl

  5. Decyzja:
       liveness_score >= threshold → is_live=True  (Live)
       liveness_score <  threshold → is_live=False (Security Alert)

Kalibracja empiryczna (LFW twarze 112×112):
  LBP entropy_norm ≈ 0.46  |  moire_score ≈ 0.54 (próg 0.72)  |  specular ≈ 0.076
  → physical_score ≈ 0.41  |  dl_score ≈ 0.4–0.6 (zależnie od twarzy)
  → liveness_score ≈ 0.40 (typowa żywa twarz przy threshold=0.1625)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.pad.liveness import (
    MOIRE_FFT_THRESHOLD,
    SPECULAR_RATIO_MAX,
    MoireDetector,
)
from src.pad.mobilenet_gradcam import MobileNetV2GradCAM
from src.pad.physical_features import PhysicalFeatureExtractor
from src.vision.detector import FaceDetector

# Indeksy 5 punktów kluczowych z modelu 2d106det (identyczne z face_detector_batch.py)
_KPS5_INDICES = [38, 88, 86, 52, 61]

logger = logging.getLogger(__name__)

# ─── Progi i wagi ─────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD: float = 0.1625  # EER threshold z projektu NASK

_W_PHYSICAL: float = 0.40  # waga komponentu fizycznego (LBP + FFT)
_W_DL: float = 0.60  # waga komponentu DL (MobileNetV2 Grad-CAM)

# Wagi wewnątrz physical_score
_W_LBP: float = 0.50
_W_MOIRE: float = 0.35
_W_SPECULAR: float = 0.15

# ─── Komunikaty ───────────────────────────────────────────────────────────────
_MSG_LIVE = "Liveness Verified: Live face detected."
_MSG_SPOOF = "Security Alert: Presentation Attack Detected"
_MSG_NO_FACE = "No face detected in the image."
_MSG_UNCERTAIN = "Liveness check inconclusive — score near threshold."


# ─── Dataclass wynikowy ───────────────────────────────────────────────────────


@dataclass
class LivenessResult:
    """
    Pełny wynik weryfikacji żywotności.

    Atrybuty
    --------
    is_live          : True = żywa twarz, False = atak prezentacyjny
    liveness_score   : ∈ [0, 1] — wyższy = bardziej live
    physical_score   : komponent LBP + FFT
    dl_score         : komponent MobileNetV2 Grad-CAM (real_score)
    gradcam_overlay  : (112, 112, 3) RGB uint8 — heatmapa Grad-CAM na cropsie
    aligned_crop     : (112, 112, 3) BGR lub None (brak twarzy)
    message          : czytelny komunikat do Gradio
    details          : słownik szczegółowych metryk diagnostycznych
    """

    is_live: bool
    liveness_score: float
    physical_score: float
    dl_score: float
    gradcam_overlay: np.ndarray  # (H, W, 3) RGB uint8
    aligned_crop: np.ndarray | None  # (112, 112, 3) BGR
    message: str
    details: dict[str, float] = field(default_factory=dict)


# ─── Lazy singletons (reuse między wywołaniami Gradio) ───────────────────────

_detector: FaceDetector | None = None
_feat_ext: PhysicalFeatureExtractor | None = None
_gcam: MobileNetV2GradCAM | None = None
_moire_det: MoireDetector = MoireDetector()  # bezstanowy


def _get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector(model_pack="buffalo_l", det_size=(640, 640))
    return _detector


def _get_feat_ext() -> PhysicalFeatureExtractor:
    global _feat_ext
    if _feat_ext is None:
        _feat_ext = PhysicalFeatureExtractor()
    return _feat_ext


def _get_gcam() -> MobileNetV2GradCAM:
    global _gcam
    if _gcam is None:
        _gcam = MobileNetV2GradCAM()
    return _gcam


# ─── Główna funkcja ───────────────────────────────────────────────────────────


def verify_liveness(
    image: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    return_details: bool = True,
) -> LivenessResult:
    """
    Weryfikuje żywotność twarzy na pojedynczym obrazie.

    Parametry
    ---------
    image     : (H, W, 3) BGR uint8  — obraz wejściowy (dowolna rozdzielczość)
    threshold : próg decyzyjny [0, 1]; domyślnie EER = 0.1625
                liveness_score >= threshold → Live
    return_details : czy zwracać szczegółowy słownik diagnostyczny

    Zwraca
    ------
    LivenessResult z:
      - is_live, liveness_score, physical_score, dl_score
      - gradcam_overlay (112×112 RGB) — gotowy do gr.Image w Gradio
      - aligned_crop (112×112 BGR)
      - message (string)
      - details (dict z metrykami)
    """
    if image is None or image.size == 0:
        logger.warning("verify_liveness: pusty obraz wejściowy.")
        return _error_result(_MSG_NO_FACE, np.zeros((112, 112, 3), dtype=np.uint8))

    # ── Krok 1: Detekcja i wyrównanie ─────────────────────────────────────────
    #
    # Strategia dwuetapowa (zgodna z 01_detect_align.pdf):
    #   a) Pełna detekcja (det_10g + 2d106det) — dla normalnych zdjęć z kamerą
    #   b) Fallback proxy-align (2d106det bez detekcji) — dla pre-cropped 112×112
    #      Treat full image as bbox → 2d106det wyznacza 106 landmarków bezpośrednio
    #
    detector = _get_detector()
    face = detector.get_largest_face(image)

    if face is not None:
        aligned_crop: np.ndarray = face.aligned_crop  # (112, 112, 3) BGR
        det_score: float = face.det_score
        logger.debug("Twarz wykryta (detector): det_score=%.3f  bbox=%s", det_score, face.bbox)
    else:
        # Fallback: proxy-align przez 2d106det (dla pre-cropped / małych obrazów)
        logger.info("verify_liveness: detector nie wykrył twarzy — próba proxy-align (2d106det).")
        aligned_crop, det_score = _proxy_align_fallback(image)
        if aligned_crop is None:
            logger.info("verify_liveness: proxy-align nie powiódł się — brak twarzy.")
            return _error_result(_MSG_NO_FACE, image)

        # Sprawdź czy aligned_crop ma wystarczającą zawartość wizualną.
        # Wariancja Laplasjanu < 5 → jednorodny obraz (czerń, biel, szum bez krawędzi)
        gray_check = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray_check, cv2.CV_64F).var())
        if lap_var < 5.0:
            logger.info(
                "verify_liveness: brak krawędzi (Laplacian var=%.2f) — odrzucam jako pusty obraz.",
                lap_var,
            )
            return _error_result(_MSG_NO_FACE, image)

    # ── Krok 2: Analiza fizyczna (PhysicalFeatureExtractor + MoireDetector) ────
    physical_score, phy_details = _compute_physical_score(aligned_crop)

    # ── Krok 3: DL AntiSpoofingModel (MobileNetV2 Grad-CAM) ───────────────────
    gcam = _get_gcam()
    gcam_result = gcam.analyze(aligned_crop)
    dl_score = gcam_result.real_score  # ∈ [0, 1]

    # ── Krok 4: Liveness score — średnia ważona ────────────────────────────────
    liveness_score = float(_W_PHYSICAL * physical_score + _W_DL * dl_score)

    logger.info(
        "Liveness: physical=%.3f  dl=%.3f  combined=%.3f  threshold=%.4f",
        physical_score,
        dl_score,
        liveness_score,
        threshold,
    )

    # ── Krok 5: Decyzja ────────────────────────────────────────────────────────
    is_live = liveness_score >= threshold

    if is_live:
        message = _MSG_LIVE
    else:
        message = _MSG_SPOOF

    # ── Wynik diagnostyczny ────────────────────────────────────────────────────
    details: dict[str, float] = {}
    if return_details:
        details = {
            "det_score": det_score,
            "liveness_score": liveness_score,
            "physical_score": physical_score,
            "dl_score": dl_score,
            "threshold": threshold,
            # Physical breakdown
            **phy_details,
            # Grad-CAM region means
            **{f"gradcam_{k}": v for k, v in gcam_result.region_means.items()},
            "gradcam_spoof_score": gcam_result.spoof_score,
            "gradcam_real_score": gcam_result.real_score,
        }

    return LivenessResult(
        is_live=is_live,
        liveness_score=liveness_score,
        physical_score=physical_score,
        dl_score=dl_score,
        gradcam_overlay=gcam_result.overlay,  # (112,112,3) RGB uint8
        aligned_crop=aligned_crop,
        message=message,
        details=details,
    )


# ─── Obliczenie physical_score ────────────────────────────────────────────────


def _compute_physical_score(
    aligned_crop: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """
    Oblicza fizyczny komponent liveness score z aligned_crop (112×112 BGR).

    Składniki:
      LBP (uniform, P=8, R=1):
        Entropia znormalizowanego histogramu Shannon'a.
        Naturalna skóra ma wyższą entropię (równomierniejszy rozkład wzorców)
        niż wydruk / ekran (dominacja kilku periodycznych wzorców).
        Zakres: [0, 1], wyższy = bardziej live.

      FFT (Moiré):
        Energia poza centrum widma / energia całkowita.
        Kalibrowane progiem MOIRE_FFT_THRESHOLD = 0.72.
        Konwertowane na live_score: 0 = powyżej progu, 1 = daleko poniżej progu.

      Specular:
        Udział prześwietlonych pikseli (V > 240 w HSV).
        Ekrany szklane i wydruki w warunkach oświetleniowych dają wyższe wartości.
        Konwertowane na live_score: 0 = ekran, 1 = brak speculars.

    Zwraca
    ------
    (physical_score ∈ [0,1], dict z metrykami)
    """
    feat_ext = _get_feat_ext()

    # LBP — uniform variant (skimage)
    lbp_hist = feat_ext.get_lbp_hist(aligned_crop)  # (59,) float32, sum=1
    lbp_entropy = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-12)))
    lbp_entropy_norm = float(lbp_entropy / np.log2(len(lbp_hist)))  # ∈ [0,1]

    # FFT + Specular — MoireDetector (kalibrowane progi)
    moire_score, lbp_var_256, specular = _moire_det.analyze(aligned_crop)

    moire_live = float(max(0.0, 1.0 - moire_score / MOIRE_FFT_THRESHOLD))
    specular_live = float(max(0.0, 1.0 - specular / SPECULAR_RATIO_MAX))

    physical_score = _W_LBP * lbp_entropy_norm + _W_MOIRE * moire_live + _W_SPECULAR * specular_live

    details: dict[str, float] = {
        "lbp_entropy_norm": lbp_entropy_norm,
        "lbp_entropy_raw": lbp_entropy,
        "moire_score": moire_score,
        "moire_live": moire_live,
        "lbp_var_256bin": lbp_var_256,
        "specular_ratio": specular,
        "specular_live": specular_live,
    }

    logger.debug(
        "Physical: lbp_entr=%.3f  moire=%.3f(live=%.3f)  specular=%.3f(live=%.3f) → %.3f",
        lbp_entropy_norm,
        moire_score,
        moire_live,
        specular,
        specular_live,
        physical_score,
    )

    return float(physical_score), details


# ─── Helper: proxy-align (fallback dla pre-cropped obrazów) ──────────────────


def _proxy_align_fallback(
    image: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """
    Fallback alignment dla pre-cropped / małych obrazów (np. 112×112 z LFW).

    Stosuje trick _FaceProxy z face_detector_batch.py:
      1. Traktuje cały obraz jako bbox twarzy (bez detekcji det_10g)
      2. Uruchamia model 2d106det bezpośrednio na całym obrazie
      3. Wybiera 5 kluczowych punktów (indeksy [38,88,86,52,61])
      4. Stosuje norm_crop (affine transform → 112×112)

    Zgodne z 01_detect_align.pdf — identyczna metoda jak w face_detector_batch.py.

    Zwraca (aligned_crop BGR, det_score=0.0) lub (None, 0.0) przy niepowodzeniu.
    """
    try:
        import insightface.app.common as common  # noqa: PLC0415
        from insightface.utils.face_align import norm_crop  # noqa: PLC0415

        detector = _get_detector()
        h, w = image.shape[:2]

        # Konstruujemy sztuczny obiekt Face z bbox = cały obraz
        proxy = common.Face(
            bbox=np.array([0, 0, w - 1, h - 1], dtype=np.float32),
            kps=None,
            det_score=1.0,
            landmark_3d_68=None,
            pose=None,
            landmark_2d_106=None,
            gender=None,
            age=None,
            embedding=None,
        )

        # Szukamy modelu 2d106det w załadowanym FaceAnalysis
        lm_model = None
        for model in detector._app.models.values():
            if "landmark_2d_106" in str(getattr(model, "taskname", "")):
                lm_model = model
                break

        if lm_model is None:
            logger.warning("proxy_align: brak modelu 2d106det.")
            # Ostatni fallback: resize do 112×112 bez alignment
            resized = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
            return resized, 0.0

        lm_model.get(image, proxy)
        lm106 = getattr(proxy, "landmark_2d_106", None)

        if lm106 is None:
            logger.warning("proxy_align: 2d106det nie zwrócił landmarków.")
            resized = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
            return resized, 0.0

        kps5 = lm106[_KPS5_INDICES].astype(np.float32)  # (5, 2)
        aligned = norm_crop(image, kps5, image_size=112, mode="arcface")
        return aligned, 0.0

    except Exception as exc:
        logger.error("proxy_align_fallback błąd: %s", exc)
        # Ostateczny fallback: resize bez alignment
        try:
            return cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR), 0.0
        except Exception:
            return None, 0.0


# ─── Helper: wynik błędu ──────────────────────────────────────────────────────


def _error_result(message: str, image: np.ndarray) -> LivenessResult:
    """Zwraca LivenessResult oznaczający błąd (brak twarzy / pusty obraz)."""
    h, w = image.shape[:2] if image.ndim == 3 else (112, 112)
    rgb = (
        cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2RGB)
        if image.ndim == 3
        else np.zeros((112, 112, 3), dtype=np.uint8)
    )

    # Rysujemy komunikat na overlay
    overlay = rgb.copy()
    cv2.putText(
        overlay,
        "No face detected",
        (5, overlay.shape[0] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 50, 50),
        1,
        cv2.LINE_AA,
    )

    return LivenessResult(
        is_live=False,
        liveness_score=0.0,
        physical_score=0.0,
        dl_score=0.0,
        gradcam_overlay=overlay,
        aligned_crop=None,
        message=message,
        details={},
    )


# ─── Formatowanie dla Gradio ──────────────────────────────────────────────────


def format_liveness_report(result: LivenessResult) -> str:
    """
    Formatuje LivenessResult jako czytelny markdown dla gr.Markdown w Gradio.
    """
    status = "✅ LIVE" if result.is_live else "🚨 SPOOF DETECTED"
    lines = [
        f"## {status}",
        f"**{result.message}**",
        "",
        "| Metryka | Wartość |",
        "|---------|---------|",
        f"| Liveness Score | `{result.liveness_score:.4f}` |",
        f"| Threshold | `{result.details.get('threshold', DEFAULT_THRESHOLD):.4f}` |",
        f"| Physical Score (LBP+FFT) | `{result.physical_score:.4f}` |",
        f"| DL Score (Grad-CAM) | `{result.dl_score:.4f}` |",
    ]

    if result.details:
        lines += [
            "",
            "### Szczegóły fizyczne",
            "| Składnik | Wartość |",
            "|----------|---------|",
            f"| LBP Entropy (uniform) | `{result.details.get('lbp_entropy_norm', 0):.4f}` |",
            f"| FFT Moiré score | `{result.details.get('moire_score', 0):.4f}` |",
            f"| Specular ratio | `{result.details.get('specular_ratio', 0):.4f}` |",
            "",
            "### Grad-CAM (region aktywacji)",
            "| Region | Aktywacja |",
            "|--------|-----------|",
        ]
        for region in ["Eyes", "Mouth", "Nose", "Forehead", "Chin/Jaw"]:
            key = f"gradcam_{region}"
            val = result.details.get(key, 0.0)
            mark = " ⬆ biometryczne" if region in ("Eyes", "Mouth") else ""
            lines.append(f"| {region} | `{val:.4f}`{mark} |")
        edge_keys = [k for k in result.details if k.startswith("gradcam_Edge_")]
        if edge_keys:
            edge_mean = np.mean([result.details[k] for k in edge_keys])
            lines.append(f"| Edge (mean) | `{edge_mean:.4f}` ← spoof sygnał |")

    return "\n".join(lines)
