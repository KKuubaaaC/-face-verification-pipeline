"""
Single-image face liveness scoring (distinct from sequential :class:`PADPipeline`).

Pipeline overview:

1. **Detect + align** with :class:`src.vision.detector.FaceDetector` (``buffalo_l`` / 2d106),
   yielding a 112×112 BGR ``aligned_crop`` (or proxy-align fallback for tight crops).

2. **Physical cues** via :class:`PhysicalFeatureExtractor` and :class:`MoireDetector`:
   LBP histogram entropy, FFT moiré score, specular ratio.

3. **MobileNetV2 Grad-CAM** branch (optional deep cue) with region scoring
   (eyes / mouth / nose vs edges).

4. **Fusion:** weighted blend of physical and deep scores → ``liveness_score`` in ``[0, 1]``.

5. **Decision:** ``liveness_score >= threshold`` ⇒ ``is_live=True``.

Empirical notes in code comments refer to sample stats on aligned LFW-style crops; tune
thresholds for your deployment.
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

# Five keypoint indices from 2d106det (same as face_detector_batch.py)
_KPS5_INDICES = [38, 88, 86, 52, 61]

logger = logging.getLogger(__name__)

# --- Thresholds and weights ---
DEFAULT_THRESHOLD: float = 0.1625  # project EER-style operating point

_W_PHYSICAL: float = 0.40  # physical branch (LBP + FFT)
_W_DL: float = 0.60  # MobileNetV2 Grad-CAM branch

# Weights inside physical_score
_W_LBP: float = 0.50
_W_MOIRE: float = 0.35
_W_SPECULAR: float = 0.15

# --- User-facing messages ---
_MSG_LIVE = "Liveness Verified: Live face detected."
_MSG_SPOOF = "Security Alert: Presentation Attack Detected"
_MSG_NO_FACE = "No face detected in the image."
_MSG_UNCERTAIN = "Liveness check inconclusive — score near threshold."


# --- Result dataclass ---


@dataclass
class LivenessResult:
    """
    Output of :func:`verify_liveness`.

    Attributes
    ----------
    is_live
        ``True`` if the fused score passes the threshold (live presentation).
    liveness_score
        Fused score in ``[0, 1]`` (higher → more likely live).
    physical_score / dl_score
        Physical (LBP/FFT/specular) vs Grad-CAM branch contributions.
    gradcam_overlay
        RGB visualization (e.g. 112×112) for UI.
    aligned_crop
        BGR aligned face patch or ``None`` if no face.
    message
        Short status string for UIs.
    details
        Flat dict of diagnostic floats (when requested).
    """

    is_live: bool
    liveness_score: float
    physical_score: float
    dl_score: float
    gradcam_overlay: np.ndarray  # (H, W, 3) RGB uint8
    aligned_crop: np.ndarray | None  # (112, 112, 3) BGR
    message: str
    details: dict[str, float] = field(default_factory=dict)


# --- Lazy singletons (reuse across Gradio calls) ---

_detector: FaceDetector | None = None
_feat_ext: PhysicalFeatureExtractor | None = None
_gcam: MobileNetV2GradCAM | None = None
_moire_det: MoireDetector = MoireDetector()  # stateless


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


# --- Main API ---


def verify_liveness(
    image: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    return_details: bool = True,
) -> LivenessResult:
    """
    Run single-image liveness on a BGR frame.

    Parameters
    ----------
    image
        ``(H, W, 3)`` BGR ``uint8`` at any resolution.
    threshold
        Decision threshold in ``[0, 1]`` (default ``0.1625`` matches project EER note).
    return_details
        When ``True``, populate ``details`` with per-cue metrics.

    Returns
    -------
    LivenessResult
        Scores, optional ``aligned_crop``, RGB ``gradcam_overlay``, and ``message``.
    """
    if image is None or image.size == 0:
        logger.warning("verify_liveness: empty input image.")
        return _error_result(_MSG_NO_FACE, np.zeros((112, 112, 3), dtype=np.uint8))

    # --- Step 1: detection + alignment ---
    #
    # Two-stage strategy:
    #   a) Full detector (det_10g + 2d106det) for normal camera frames
    #   b) Proxy-align fallback (2d106det only) for tight 112×112 crops
    #      Treat full image as bbox so 2d106det predicts 106 landmarks directly
    #
    detector = _get_detector()
    face = detector.get_largest_face(image)

    if face is not None:
        aligned_crop: np.ndarray = face.aligned_crop  # (112, 112, 3) BGR
        det_score: float = face.det_score
        logger.debug("Face detected (detector): det_score=%.3f  bbox=%s", det_score, face.bbox)
    else:
        # Fallback: proxy-align via 2d106det for pre-cropped / small images
        logger.info("verify_liveness: no face from detector — trying proxy-align (2d106det).")
        aligned_crop, det_score = _proxy_align_fallback(image)
        if aligned_crop is None:
            logger.info("verify_liveness: proxy-align failed — no face.")
            return _error_result(_MSG_NO_FACE, image)

        # Reject uniform / edge-less crops (blank or noise)
        # Laplacian variance < 5 → flat image
        gray_check = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray_check, cv2.CV_64F).var())
        if lap_var < 5.0:
            logger.info(
                "verify_liveness: no edges (Laplacian var=%.2f) — rejecting as empty.",
                lap_var,
            )
            return _error_result(_MSG_NO_FACE, image)

    # --- Step 2: physical cues (PhysicalFeatureExtractor + MoireDetector) ---
    physical_score, phy_details = _compute_physical_score(aligned_crop)

    # --- Step 3: MobileNetV2 Grad-CAM ---
    gcam = _get_gcam()
    gcam_result = gcam.analyze(aligned_crop)
    dl_score = gcam_result.real_score  # ∈ [0, 1]

    # --- Step 4: fused liveness score ---
    liveness_score = float(_W_PHYSICAL * physical_score + _W_DL * dl_score)

    logger.info(
        "Liveness: physical=%.3f  dl=%.3f  combined=%.3f  threshold=%.4f",
        physical_score,
        dl_score,
        liveness_score,
        threshold,
    )

    # --- Step 5: decision ---
    is_live = liveness_score >= threshold

    if is_live:
        message = _MSG_LIVE
    else:
        message = _MSG_SPOOF

    # --- Diagnostics ---
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


# --- physical_score computation ---


def _compute_physical_score(
    aligned_crop: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """
    Physical branch of the liveness score from aligned_crop (112×112 BGR).

    Components:
      LBP (uniform P=8, R=1): Shannon entropy of the normalized histogram.
        Live skin tends to higher entropy than print/screen textures.

      FFT (moiré): off-center spectrum energy / total energy vs MOIRE_FFT_THRESHOLD.

      Specular: saturated HSV-V pixels; screens and glare raise this cue.

    Returns
    -------
    (physical_score in [0, 1], metrics dict)
    """
    feat_ext = _get_feat_ext()

    # LBP — uniform variant (scikit-image)
    lbp_hist = feat_ext.get_lbp_hist(aligned_crop)  # (59,) float32, sum=1
    lbp_entropy = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-12)))
    lbp_entropy_norm = float(lbp_entropy / np.log2(len(lbp_hist)))  # ∈ [0,1]

    # FFT + specular — MoireDetector (thresholded)
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


# --- Helper: proxy-align for pre-cropped images ---


def _proxy_align_fallback(
    image: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """
    Proxy alignment for tight crops (e.g. 112×112 LFW cells), same as face_detector_batch.

    Steps:
      1. Treat the full image as the face bounding box (skip det_10g)
      2. Run 2d106det on the whole image
      3. Take five keypoints [38,88,86,52,61]
      4. norm_crop → 112×112 ArcFace-style patch

    Returns (aligned BGR, det_score=0.0) or (None, 0.0) on failure.
    """
    try:
        import insightface.app.common as common  # noqa: PLC0415
        from insightface.utils.face_align import norm_crop  # noqa: PLC0415

        detector = _get_detector()
        h, w = image.shape[:2]

        # Synthetic Face with bbox = full image
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

        # Locate 2d106det inside the loaded FaceAnalysis stack
        lm_model = None
        for model in detector.face_analysis.models.values():
            if "landmark_2d_106" in str(getattr(model, "taskname", "")):
                lm_model = model
                break

        if lm_model is None:
            logger.warning("proxy_align: 2d106det model missing.")
            # Last resort: plain resize to 112×112
            resized = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
            return resized, 0.0

        lm_model.get(image, proxy)
        lm106 = getattr(proxy, "landmark_2d_106", None)

        if lm106 is None:
            logger.warning("proxy_align: 2d106det returned no landmarks.")
            resized = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
            return resized, 0.0

        kps5 = lm106[_KPS5_INDICES].astype(np.float32)  # (5, 2)
        aligned = norm_crop(image, kps5, image_size=112, mode="arcface")
        return aligned, 0.0

    except Exception as exc:
        logger.error("proxy_align_fallback error: %s", exc)
        # Final fallback: resize only
        try:
            return cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR), 0.0
        except Exception:
            return None, 0.0


# --- Helper: error result ---


def _error_result(message: str, image: np.ndarray) -> LivenessResult:
    """Build a LivenessResult for missing face / empty input."""
    h, w = image.shape[:2] if image.ndim == 3 else (112, 112)
    rgb = (
        cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2RGB)
        if image.ndim == 3
        else np.zeros((112, 112, 3), dtype=np.uint8)
    )

    # Draw status on overlay
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


# --- Gradio markdown formatting ---


def format_liveness_report(result: LivenessResult) -> str:
    """Render ``LivenessResult`` as markdown for ``gr.Markdown``."""
    status = "LIVE" if result.is_live else "SPOOF DETECTED"
    lines = [
        f"## {status}",
        f"**{result.message}**",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Liveness score | `{result.liveness_score:.4f}` |",
        f"| Threshold | `{result.details.get('threshold', DEFAULT_THRESHOLD):.4f}` |",
        f"| Physical score (LBP+FFT) | `{result.physical_score:.4f}` |",
        f"| DL score (Grad-CAM) | `{result.dl_score:.4f}` |",
    ]

    if result.details:
        lines += [
            "",
            "### Physical breakdown",
            "| Component | Value |",
            "|-----------|-------|",
            f"| LBP entropy (uniform) | `{result.details.get('lbp_entropy_norm', 0):.4f}` |",
            f"| FFT moiré score | `{result.details.get('moire_score', 0):.4f}` |",
            f"| Specular ratio | `{result.details.get('specular_ratio', 0):.4f}` |",
            "",
            "### Grad-CAM (region means)",
            "| Region | Activation |",
            "|--------|------------|",
        ]
        for region in ["Eyes", "Mouth", "Nose", "Forehead", "Chin/Jaw"]:
            key = f"gradcam_{region}"
            val = result.details.get(key, 0.0)
            mark = " (biometric)" if region in ("Eyes", "Mouth") else ""
            lines.append(f"| {region} | `{val:.4f}`{mark} |")
        edge_keys = [k for k in result.details if k.startswith("gradcam_Edge_")]
        if edge_keys:
            edge_mean = np.mean([result.details[k] for k in edge_keys])
            lines.append(f"| Edge (mean) | `{edge_mean:.4f}` (spoof cue) |")

    return "\n".join(lines)
