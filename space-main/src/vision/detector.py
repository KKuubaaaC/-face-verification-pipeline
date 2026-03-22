"""
Face detector and aligner using InsightFace ``buffalo_l`` (2d106det).

Responsibilities:
  - Detect faces in a frame
  - Extract 106 2D landmarks (2d106det)
  - Affine alignment from eye / mouth keypoints to the ArcFace 112×112 template
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# Docelowy rozmiar aligned crop wymagany przez ArcFace
_ARCFACE_INPUT_SIZE: int = 112

# Referencyjne koordynaty 5 keypoints dla arcface_112_v2
# (left eye, right eye, nose, left mouth corner, right mouth corner)
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class DetectedFace:
    """Detection result for a single face."""

    bbox: np.ndarray  # (4,)     [x1, y1, x2, y2]
    landmarks_106: np.ndarray  # (106, 2) raw 2d106det coordinates
    landmarks_68: np.ndarray  # (68, 3)  iBUG 68 (1k3d68 from InsightFace)
    kps: np.ndarray  # (5, 2)   five keypoints for alignment
    det_score: float
    aligned_crop: np.ndarray  # (112, 112, 3) BGR, ready for the embedder


class FaceDetector:
    """
    Thin wrapper around InsightFace :class:`FaceAnalysis` (``buffalo_l``).

    Models load once in ``__init__``; ``process_frame`` is stateless.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
        root: str = "./models",
    ) -> None:
        _providers = providers or ["CPUExecutionProvider"]
        logger.info("Loading InsightFace models (%s) from %s...", model_pack, root)
        self._app = FaceAnalysis(
            name=model_pack,
            providers=_providers,
            root=root,
        )
        self._app.prepare(ctx_id=0, det_size=det_size)
        logger.info("FaceDetector ready.")

    @property
    def face_analysis(self) -> FaceAnalysis:
        """Shared InsightFace ``FaceAnalysis`` instance (detector + recognition models)."""
        return self._app

    def process_frame(self, frame: np.ndarray) -> list[DetectedFace]:
        """
        Detect all faces in ``frame`` and return :class:`DetectedFace` instances.

        ``frame``: BGR ``uint8`` array (H×W×3). Returns an empty list if none found.
        """
        if frame is None or frame.size == 0:
            logger.warning("process_frame: empty image.")
            return []

        faces = self._app.get(frame)
        if not faces:
            return []

        results: list[DetectedFace] = []
        for face in faces:
            lm106 = getattr(face, "landmark_2d_106", None)
            if lm106 is None:
                logger.warning("Missing landmark_2d_106 for a face — skipping.")
                continue

            lm68 = getattr(face, "landmark_3d_68", None)
            if lm68 is None:
                logger.warning("Missing landmark_3d_68 — blink detection unavailable.")
                lm68 = np.zeros((68, 3), dtype=np.float32)

            kps = face.kps  # (5, 2)
            try:
                crop = _align_face(frame, kps)
            except Exception as exc:
                logger.error("Face alignment error: %s", exc)
                continue

            results.append(
                DetectedFace(
                    bbox=face.bbox.astype(np.float32),
                    landmarks_106=lm106.astype(np.float32),
                    landmarks_68=lm68.astype(np.float32),
                    kps=kps.astype(np.float32),
                    det_score=float(face.det_score),
                    aligned_crop=crop,
                )
            )

        return results

    def get_largest_face(self, frame: np.ndarray) -> DetectedFace | None:
        """
        Return the face with the largest bounding-box area (closest / dominant).

        Returns ``None`` if no face is detected.
        """
        faces = self.process_frame(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: _bbox_area(f.bbox))


# ─── Helpers ─────────────────────────────────────────────────────────────────


def align_face_from_keypoints(
    image: np.ndarray,
    kps: np.ndarray,
    output_size: int = _ARCFACE_INPUT_SIZE,
) -> np.ndarray:
    """
    Affine-align a face region from five keypoints to the ArcFace 112×112 template.

    Public wrapper around the internal alignment used by :class:`FaceDetector`.
    """
    return _align_face(image, kps, output_size)


def _align_face(
    image: np.ndarray,
    kps: np.ndarray,
    output_size: int = _ARCFACE_INPUT_SIZE,
) -> np.ndarray:
    """
    Affine warp from five keypoints to ``_ARCFACE_DST`` (scaled to ``output_size``).

    Returns a BGR crop of shape (output_size, output_size, 3).
    """
    dst = _ARCFACE_DST.copy()
    # scale dst to requested output size (default 112×112)
    scale = output_size / 112.0
    dst *= scale

    transform = cv2.estimateAffinePartial2D(
        kps.astype(np.float32),
        dst,
        method=cv2.LMEDS,
    )[0]

    if transform is None:
        raise ValueError("estimateAffinePartial2D returned None — cannot align face.")

    aligned = cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
