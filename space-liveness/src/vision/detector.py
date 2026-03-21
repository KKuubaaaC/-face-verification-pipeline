"""
Face detector and aligner using insightface buffalo_l (2d106det).

Odpowiedzialność:
  - Detekcja twarzy na klatce z kamery
  - Ekstrakcja 106 landmarków (2d106det)
  - Transformacja afiniczna (alignment) na podstawie punktów oczu i ust
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
# (lewe oko, prawe oko, nos, lewy kąt ust, prawy kąt ust)
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
    """Wynik detekcji jednej twarzy."""

    bbox: np.ndarray  # (4,)    — [x1, y1, x2, y2]
    landmarks_106: np.ndarray  # (106, 2) — surowe współrzędne 2d106det
    landmarks_68: np.ndarray  # (68, 3)  — standardowe iBUG 68-punktowe (1k3d68)
    kps: np.ndarray  # (5, 2)  — 5 kluczowych punktów (do alignmentu)
    det_score: float
    aligned_crop: np.ndarray  # (112, 112, 3) BGR — gotowy do embeddingu


class FaceDetector:
    """
    Wrapper wokół insightface FaceAnalysis (buffalo_l).

    Ładuje modele raz przy __init__; process_frame() jest bezstanowe.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
        root: str = "./models",
    ) -> None:
        _providers = providers or ["CPUExecutionProvider"]
        logger.info("Ładowanie modeli insightface (%s) z %s...", model_pack, root)
        self._app = FaceAnalysis(
            name=model_pack,
            providers=_providers,
            root=root,
        )
        self._app.prepare(ctx_id=0, det_size=det_size)
        logger.info("FaceDetector gotowy.")

    def process_frame(self, frame: np.ndarray) -> list[DetectedFace]:
        """
        Wykrywa twarze w klatce i zwraca listę DetectedFace.

        frame: BGR image (np.ndarray HxWx3)
        Zwraca pustą listę jeśli żadna twarz nie została wykryta.
        """
        if frame is None or frame.size == 0:
            logger.warning("process_frame: otrzymano pusty obraz.")
            return []

        faces = self._app.get(frame)
        if not faces:
            return []

        results: list[DetectedFace] = []
        for face in faces:
            lm106 = getattr(face, "landmark_2d_106", None)
            if lm106 is None:
                logger.warning("Brak landmark_2d_106 dla wykrytej twarzy — pomijam.")
                continue

            lm68 = getattr(face, "landmark_3d_68", None)
            if lm68 is None:
                logger.warning("Brak landmark_3d_68 — blink detection niedostępny.")
                lm68 = np.zeros((68, 3), dtype=np.float32)

            kps = face.kps  # (5, 2)
            try:
                crop = _align_face(frame, kps)
            except Exception as exc:
                logger.error("Błąd alignmentu twarzy: %s", exc)
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
        Zwraca twarz o największej powierzchni bbox (najbliższa kamera).
        Zwraca None jeśli żadna twarz nie wykryta.
        """
        faces = self.process_frame(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: _bbox_area(f.bbox))


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _align_face(
    image: np.ndarray,
    kps: np.ndarray,
    output_size: int = _ARCFACE_INPUT_SIZE,
) -> np.ndarray:
    """
    Transformacja afiniczna na podstawie 5 kluczowych punktów (kps).

    Normalizuje odległość IPD i orientację twarzy względem _ARCFACE_DST.
    Zwraca: (output_size, output_size, 3) BGR crop.
    """
    dst = _ARCFACE_DST.copy()
    # skalujemy dst do żądanego rozmiaru (domyślnie 1:1 dla 112px)
    scale = output_size / 112.0
    dst *= scale

    transform = cv2.estimateAffinePartial2D(
        kps.astype(np.float32),
        dst,
        method=cv2.LMEDS,
    )[0]

    if transform is None:
        raise ValueError("estimateAffinePartial2D zwróciło None — nie można wyrównać twarzy.")

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
