"""
Presentation Attack Detection (PAD) for live face capture.

Two complementary cues:

1. **BlinkDetector** — Eye Aspect Ratio (EAR) on 68-point iBUG landmarks (from InsightFace).
   Detects a blink sequence → mitigates static photo / print replay.
2. **MoireDetector** — FFT moiré energy, OpenCV LBP variance, and HSV specular ratio.
   Flags phone / monitor presentation.

Facade: ``PADPipeline.process_frame(landmarks, face_crop)`` → :class:`PADResult`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- Landmark indices (InsightFace 1k3d68 → iBUG 68) ---
# We use landmark_3d_68 (not 2d106det) for documented index semantics.
# Right eye (observer view): 36–41
# Left eye  (observer view): 42–47
# Order per eye: outer corner, upper outer, upper inner,
#                inner corner, lower inner, lower outer
_RIGHT_EYE_IDX: tuple[int, ...] = (36, 37, 38, 39, 40, 41)
_LEFT_EYE_IDX: tuple[int, ...] = (42, 43, 44, 45, 46, 47)

# --- Operational thresholds ---
EAR_BLINK_THRESHOLD: float = 0.20  # below → eye considered closed
EAR_CONSEC_FRAMES: int = 2  # min consecutive closed-eye frames = one blink

# FFT: normalized energy outside spectrum center (0.0 natural, 1.0 screen-like)
# Typical webcam: ~0.60–0.68; face on phone/monitor: ~0.75–0.92
MOIRE_FFT_THRESHOLD: float = 0.72

# Blink timeout: face visible N frames with no blink → static replay attack
# At ~12 fps, 90 frames ~ 7.5 s.
BLINK_TIMEOUT_FRAMES: int = 90
# Inner radius of FFT low-pass mask (% of min image dimension)
_FFT_LOWPASS_RATIO: float = 0.10

# LBP: histogram variance (×10^6) — skin low, pixel-grid screens high
# Webcam compression often raises baseline ~150–200
LBP_VAR_THRESHOLD: float = 350.0

# Specular: max share of saturated pixels (V > 240 in HSV) on face crop
SPECULAR_RATIO_MAX: float = 0.20


# --- Data structures ---


@dataclass
class PADResult:
    """Aggregated PAD outcome for one frame (blink state + texture cues)."""

    liveness_passed: bool
    blink_detected: bool
    blink_timed_out: bool  # True if face visible >BLINK_TIMEOUT_FRAMES without blink
    ear_current: float  # mean EAR this frame
    moire_score: float  # higher → more screen-like (FFT cue)
    lbp_variance: float
    specular_ratio: float
    reason: str = ""  # human-readable rejection reason (empty if OK)


# --- Module 1: BlinkDetector ---


class BlinkDetector:
    """
    Frame-to-frame eyelid state; registers one blink per verification session.

    State is cumulative until :meth:`reset` (call at the start of each session).
    """

    def __init__(
        self,
        ear_threshold: float = EAR_BLINK_THRESHOLD,
        consec_frames: int = EAR_CONSEC_FRAMES,
        timeout_frames: int = BLINK_TIMEOUT_FRAMES,
    ) -> None:
        self._threshold = ear_threshold
        self._consec = consec_frames
        self._timeout = timeout_frames
        self._counter: int = 0
        self._total_frames: int = 0  # frames with face present
        self._blink_detected: bool = False

    # --- Public API ---

    def reset(self) -> None:
        self._counter = 0
        self._total_frames = 0
        self._blink_detected = False

    @property
    def blink_detected(self) -> bool:
        return self._blink_detected

    @property
    def timed_out(self) -> bool:
        """True if a face was visible for >``timeout_frames`` without any blink."""
        return (not self._blink_detected) and (self._total_frames >= self._timeout)

    def update(self, landmarks: np.ndarray) -> float:
        """
        Update the blink FSM from one frame's 68-point landmarks.

        ``landmarks``: (68, 3) float32 from InsightFace ``landmark_3d_68``.
        Returns the current average EAR (for UI / debugging).
        """
        self._total_frames += 1
        ear = self._avg_ear(landmarks)

        if ear < self._threshold:
            self._counter += 1
        else:
            if self._counter >= self._consec:
                self._blink_detected = True
                logger.debug(
                    "Blink detected (EAR=%.3f, frames=%d)",
                    ear,
                    self._counter,
                )
            self._counter = 0

        return ear

    # --- EAR computation ---

    def _avg_ear(self, lm: np.ndarray) -> float:
        ear_l = _eye_aspect_ratio(lm, _LEFT_EYE_IDX)
        ear_r = _eye_aspect_ratio(lm, _RIGHT_EYE_IDX)
        return (ear_l + ear_r) / 2.0


# --- Module 2: MoireDetector ---


class MoireDetector:
    """
    Texture-based PAD cues using OpenCV + NumPy only.

    Stateless: each :meth:`analyze` call is independent.
    """

    def analyze(
        self,
        face_crop: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        ``face_crop``: BGR face patch (e.g. 112×112 ``aligned_crop``).

        Returns ``(moire_score, lbp_variance, specular_ratio)``.
        """
        if face_crop is None or face_crop.size == 0:
            logger.warning("MoireDetector.analyze: empty face_crop.")
            return 0.0, 0.0, 0.0

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        moire = self._fft_moire_score(gray)
        lbp_var = self._lbp_variance(gray)
        specular = self._specular_ratio(face_crop)

        logger.debug(
            "PAD scores — moiré=%.3f  lbp_var=%.1f  specular=%.3f",
            moire,
            lbp_var,
            specular,
        )
        return moire, lbp_var, specular

    # --- A. FFT moiré ---

    @staticmethod
    def _fft_moire_score(gray: np.ndarray) -> float:
        """
        High-frequency energy / total energy in the amplitude spectrum.

        Digital displays add a regular pixel grid; when re-captured on camera this
        yields moiré — abnormal peaks away from the FFT center.

        Steps:
          1. DFT → fftshift → magnitude spectrum
          2. Circular mask removes the DC / low-frequency disk
          3. Ratio of energy outside the disk to total energy
        """
        f32 = gray.astype(np.float32)

        # OpenCV DFT is fast for small crops
        dft = cv2.dft(f32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)  # DC → centrum
        mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        h, w = mag.shape
        cy, cx = h // 2, w // 2
        r_inner = int(min(h, w) * _FFT_LOWPASS_RATIO)

        # Ring mask: True outside r_inner (high frequencies)
        y_idx, x_idx = np.ogrid[:h, :w]
        dist_sq = (y_idx - cy) ** 2 + (x_idx - cx) ** 2
        high_freq_mask = dist_sq > r_inner**2

        total_energy = float(mag.sum()) + 1e-9
        high_energy = float(mag[high_freq_mask].sum())

        return high_energy / total_energy

    # --- B. LBP (local binary patterns) ---

    @staticmethod
    def _lbp_variance(gray: np.ndarray) -> float:
        """
        Variance of the LBP histogram (OpenCV-only 8-neighbor LBP).

        Skin tends to a smooth texture → low, broad histogram.
        Screen pixel grids → repetitive pattern → high histogram variance.

        Per pixel: compare 8 neighbors to center, pack into one byte, histogram.
        """
        # 8 neighbor offsets
        _neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        h, w = gray.astype(np.float32).shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        center = gray.astype(np.int16)

        for bit, (dy, dx) in enumerate(_neighbors):
            shifted = np.roll(np.roll(center, dy, axis=0), dx, axis=1)
            lbp |= (shifted >= center).astype(np.uint8) << bit

        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / (hist.sum() + 1e-9)
        # Scale ×1e6 for a readable numeric range
        return float(np.var(hist_norm) * 1e6)

    # --- C. Specular highlights ---

    @staticmethod
    def _specular_ratio(bgr: np.ndarray) -> float:
        """
        Fraction of pixels with V > 240 in HSV (specular glare on glass screens).
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        bright_pixels = int(np.sum(v_channel > 240))
        return bright_pixels / float(v_channel.size)


# --- Facade: PADPipeline ---


class PADPipeline:
    """
    Entry point combining blink-based and texture-based PAD.

    Typical usage::

        pad = PADPipeline()
        pad.reset()  # start of each session
        while streaming:
            result = pad.process_frame(landmarks, face_crop)
            if result.liveness_passed:
                ...
    """

    def __init__(
        self,
        ear_threshold: float = EAR_BLINK_THRESHOLD,
        ear_consec_frames: int = EAR_CONSEC_FRAMES,
        moire_threshold: float = MOIRE_FFT_THRESHOLD,
        lbp_threshold: float = LBP_VAR_THRESHOLD,
        specular_threshold: float = SPECULAR_RATIO_MAX,
    ) -> None:
        self._blink = BlinkDetector(
            ear_threshold=ear_threshold,
            consec_frames=ear_consec_frames,
        )
        self._moire = MoireDetector()
        self._moire_t = moire_threshold
        self._lbp_t = lbp_threshold
        self._spec_t = specular_threshold

    def reset(self) -> None:
        """Reset blink FSM; call at the start of each session."""
        self._blink.reset()

    def process_frame(
        self,
        landmarks: np.ndarray,
        face_crop: np.ndarray,
    ) -> PADResult:
        """
        Process one camera frame.

        landmarks : (68, 3) float — face.landmark_3d_68 (iBUG 68)
        face_crop : (H, W, 3) BGR — aligned face patch (e.g. aligned_crop)
        """
        # Layer 1: EAR / blink + timeout
        ear = self._blink.update(landmarks)
        timed_out = self._blink.timed_out

        # Layer 2: texture / screen cues
        moire, lbp_var, specular = self._moire.analyze(face_crop)

        # Screen attack if any texture cue fires
        screen_attack = moire > self._moire_t or lbp_var > self._lbp_t or specular > self._spec_t

        # Liveness OK: blink seen, no screen attack, no blink timeout
        liveness_ok = self._blink.blink_detected and not screen_attack and not timed_out

        reason = _build_reason(
            blink_ok=self._blink.blink_detected,
            timed_out=timed_out,
            moire=moire,
            moire_t=self._moire_t,
            lbp_var=lbp_var,
            lbp_t=self._lbp_t,
            specular=specular,
            spec_t=self._spec_t,
        )

        return PADResult(
            liveness_passed=liveness_ok,
            blink_detected=self._blink.blink_detected,
            blink_timed_out=timed_out,
            ear_current=ear,
            moire_score=moire,
            lbp_variance=lbp_var,
            specular_ratio=specular,
            reason=reason,
        )


# --- Helpers ---


def _eye_aspect_ratio(lm: np.ndarray, idx: tuple[int, ...]) -> float:
    """
    Eye aspect ratio for one eye — Soukupová & Čech (2016).

    EAR = (||P1–P5|| + ||P2–P4||) / (2 · ||P0–P3||)

    iBUG 68 mapping (right 36–41, left 42–47):
        idx[0] outer corner    idx[3] inner corner
        idx[1] upper outer     idx[4] lower outer
        idx[2] upper inner     idx[5] lower inner

    lm  : (68, 3) or (68, 2) — uses columns 0,1 (x,y)
    idx : length-6 index tuple
    """
    p = lm[list(idx), :2]  # (6, 2) use x,y only (ignore z if present)

    vert_a = float(np.linalg.norm(p[1] - p[5]))
    vert_b = float(np.linalg.norm(p[2] - p[4]))
    horiz = float(np.linalg.norm(p[0] - p[3]))

    if horiz < 1e-6:
        logger.warning("_eye_aspect_ratio: zero eye width — returning 0.0")
        return 0.0

    return (vert_a + vert_b) / (2.0 * horiz)


def _build_reason(
    *,
    blink_ok: bool,
    timed_out: bool,
    moire: float,
    moire_t: float,
    lbp_var: float,
    lbp_t: float,
    specular: float,
    spec_t: float,
) -> str:
    """Human-readable rejection reason (empty when liveness would pass)."""
    if timed_out:
        return "ATTACK: no blink for >7s — likely static photo or pre-recorded video."
    if not blink_ok:
        return "Please blink at the camera."

    issues: list[str] = []
    if moire > moire_t:
        issues.append(f"moiré cue (FFT={moire:.3f} > {moire_t})")
    if lbp_var > lbp_t:
        issues.append(f"suspicious texture (LBP={lbp_var:.1f} > {lbp_t})")
    if specular > spec_t:
        issues.append(f"screen glare (specular={specular:.3f} > {spec_t})")

    if issues:
        return "Presentation attack: " + "; ".join(issues)

    return ""
