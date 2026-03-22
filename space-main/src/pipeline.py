"""
Live face verification pipeline facade.

Per camera frame the stages are:
  1. Face detection + alignment (FaceDetector -> DetectedFace)
  2. PAD / liveness (PADPipeline -> PADResult)
  3. Embedding from aligned crop (FaceEmbedder -> np.ndarray)
  4. Match against reference (FaceEmbedder.verify -> VerificationResult)

Step 4 runs only when ``liveness_passed`` is True. Register the reference embedding once
via ``register_reference()`` from a document or enrollment image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from src.pad.liveness import (
    LBP_VAR_THRESHOLD,
    MOIRE_FFT_THRESHOLD,
    SPECULAR_RATIO_MAX,
    PADPipeline,
    PADResult,
)
from src.vision.detector import DetectedFace, FaceDetector
from src.vision.embedder import VERIFICATION_THRESHOLD, FaceEmbedder, VerificationResult

logger = logging.getLogger(__name__)


# --- Session states ---


class SessionState(Enum):
    """Lifecycle states for one verification session."""

    WAITING_FOR_REFERENCE = auto()  # no reference embedding yet
    WAITING_FOR_BLINK = auto()  # reference set, waiting for blink
    VERIFIED = auto()  # liveness OK + identity match
    REJECTED_LIVENESS = auto()  # liveness failed
    REJECTED_IDENTITY = auto()  # liveness OK but no match


# --- Per-frame aggregate result ---


@dataclass
class FrameResult:
    """Aggregated result for one camera frame."""

    state: SessionState

    # Whether any face was detected on the frame
    face_detected: bool = False

    # PAD outcome (None if no face)
    pad: PADResult | None = None

    # Verification outcome (None if liveness failed or no reference)
    verification: VerificationResult | None = None

    # Current EAR for UI even when verification is skipped
    ear: float = 0.0

    # Human-readable message for Gradio
    message: str = ""


# --- Main class ---


class VerificationPipeline:
    """
    Models load once in ``__init__``; session state is mutable (``reset_session()`` clears PAD
    between users).

    Typical flow::
        pipeline = VerificationPipeline()
        pipeline.register_reference(reference_image_bgr)
        pipeline.reset_session()
        for frame in camera_stream:
            result = pipeline.process_frame(frame)
            if result.state == SessionState.VERIFIED:
                break
    """

    def __init__(
        self,
        verification_threshold: float = VERIFICATION_THRESHOLD,
        providers: list[str] | None = None,
    ) -> None:
        _providers = providers or ["CPUExecutionProvider"]

        logger.info("Initializing VerificationPipeline...")
        self._detector = FaceDetector(providers=_providers)

        # Embedder shares FaceAnalysis with the detector (single model bundle)
        self._embedder = FaceEmbedder(_app=self._detector._app)

        self._pad = PADPipeline()
        self._threshold = verification_threshold

        self._reference_embedding: np.ndarray | None = None
        self._session_state: SessionState = SessionState.WAITING_FOR_REFERENCE

        logger.info("VerificationPipeline ready.")

    def register_reference(self, reference_image: np.ndarray) -> bool:
        """
        Extract and store the embedding from a reference image (e.g. ID photo).

        ``reference_image``: BGR, any size; detection performs alignment.

        Returns True if registration succeeded, False if no face was found.
        """
        face = self._detector.get_largest_face(reference_image)
        if face is None:
            logger.warning("register_reference: no face detected on reference image.")
            return False

        try:
            self._reference_embedding = self._embedder.embed(face.aligned_crop)
        except ValueError as exc:
            logger.error("register_reference: embedding error: %s", exc)
            return False

        logger.info("Reference registered (det_score=%.3f).", face.det_score)

        if self._session_state == SessionState.WAITING_FOR_REFERENCE:
            self._session_state = SessionState.WAITING_FOR_BLINK

        return True

    def reset_session(self) -> None:
        """
        Reset PAD session state (blink counter, automaton).
        Does not clear the reference embedding until ``register_reference()`` is called again.
        """
        self._pad.reset()
        if self._reference_embedding is not None:
            self._session_state = SessionState.WAITING_FOR_BLINK
        else:
            self._session_state = SessionState.WAITING_FOR_REFERENCE
        logger.debug("Session reset.")

    @property
    def state(self) -> SessionState:
        return self._session_state

    @property
    def has_reference(self) -> bool:
        return self._reference_embedding is not None

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process one BGR camera frame.

        Returns ``FrameResult`` with current session state and diagnostics.

        If the session already finished (VERIFIED / REJECTED_*), further frames are ignored
        and the terminal message is returned without recomputation.
        """
        if self._session_state in (
            SessionState.VERIFIED,
            SessionState.REJECTED_LIVENESS,
            SessionState.REJECTED_IDENTITY,
        ):
            return FrameResult(
                state=self._session_state,
                message=_state_message(self._session_state),
            )

        if self._reference_embedding is None:
            return FrameResult(
                state=SessionState.WAITING_FOR_REFERENCE,
                message="Register a reference photo first.",
            )

        face: DetectedFace | None = self._detector.get_largest_face(frame)

        if face is None:
            return FrameResult(
                state=self._session_state,
                face_detected=False,
                message="No face detected. Center your face in the frame.",
            )

        pad_result: PADResult = self._pad.process_frame(
            landmarks=face.landmarks_68,
            face_crop=face.aligned_crop,
        )

        if not pad_result.liveness_passed:
            if pad_result.blink_timed_out:
                self._session_state = SessionState.REJECTED_LIVENESS
                logger.warning("Session rejected: blink timeout (possible static replay attack).")
            return FrameResult(
                state=self._session_state,
                face_detected=True,
                pad=pad_result,
                ear=pad_result.ear_current,
                message=pad_result.reason or "Please blink at the camera.",
            )

        try:
            probe_embedding = self._embedder.embed(face.aligned_crop)
        except ValueError as exc:
            logger.error("process_frame: probe embedding error: %s", exc)
            return FrameResult(
                state=self._session_state,
                face_detected=True,
                pad=pad_result,
                ear=pad_result.ear_current,
                message="Face feature extraction failed. Please try again.",
            )

        verification: VerificationResult = self._embedder.verify(
            embedding_probe=probe_embedding,
            embedding_reference=self._reference_embedding,
            threshold=self._threshold,
        )

        if verification.is_match:
            self._session_state = SessionState.VERIFIED
            logger.info(
                "Verification positive (d=%.4f <= %.4f).",
                verification.cosine_distance,
                self._threshold,
            )
        else:
            self._session_state = SessionState.REJECTED_IDENTITY
            logger.info(
                "Verification negative (d=%.4f > %.4f).",
                verification.cosine_distance,
                self._threshold,
            )

        return FrameResult(
            state=self._session_state,
            face_detected=True,
            pad=pad_result,
            verification=verification,
            ear=pad_result.ear_current,
            message=_verification_message(verification),
        )


# --- Helpers ---


def _is_screen_attack(pad: PADResult) -> bool:
    """
    True when PAD flags a screen-style attack before the blink phase completes.

    Used for fast rejection during the liveness phase.
    """
    return (
        pad.moire_score > MOIRE_FFT_THRESHOLD
        or pad.lbp_variance > LBP_VAR_THRESHOLD
        or pad.specular_ratio > SPECULAR_RATIO_MAX
    )


def _state_message(state: SessionState) -> str:
    return {
        SessionState.VERIFIED: "Verification completed successfully.",
        SessionState.REJECTED_LIVENESS: "Rejected: presentation attack detected.",
        SessionState.REJECTED_IDENTITY: "Rejected: identity not confirmed.",
        SessionState.WAITING_FOR_BLINK: "Please blink at the camera.",
        SessionState.WAITING_FOR_REFERENCE: "No reference photo registered.",
    }.get(state, "")


def _verification_message(result: VerificationResult) -> str:
    if result.is_match:
        return f"MATCH: cosine distance {result.cosine_distance:.4f}"
    return f"NO MATCH: cosine distance {result.cosine_distance:.4f} (threshold: {result.threshold})"
