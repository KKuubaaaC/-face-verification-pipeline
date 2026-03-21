"""
Face embedding extractor — wzorzec Factory dla ArcFace i ViT.

Obsługiwane model_type:
  "arcface" — insightface buffalo_l (ResNet-50 ArcFace, 512-D)
              próg EER: 0.1625 (cosine distance, wyznaczony na AgeDB-30)
  "vit"     — torchvision ViT-B/16 bez głowicy klasyfikacyjnej (768-D)
              próg należy wyznaczyć empirycznie dla danego zastosowania

Interfejs publiczny jest identyczny dla obu backendów:
  embedder.embed(aligned_crop)  →  np.ndarray (D,) float32, znormalizowany L2
  embedder.verify(e1, e2)       →  VerificationResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
from insightface.app import FaceAnalysis
from torchvision import models, transforms

logger = logging.getLogger(__name__)

VERIFICATION_THRESHOLD: float = 0.1625

# Normalizacja ImageNet używana przez ViT (pre-trained na ImageNet-21k)
_VIT_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_VIT_INPUT_SIZE: int = 224


@dataclass
class VerificationResult:
    """Wynik porównania dwóch embeddingów."""

    is_match: bool
    cosine_distance: float
    threshold: float = VERIFICATION_THRESHOLD


# ─── Factory ─────────────────────────────────────────────────────────────────


class FaceEmbedder:
    """
    Factory embeddera twarzy. Wybór backendu przez model_type w konstruktorze.

    Przykłady:
        embedder = FaceEmbedder(model_type="arcface")   # domyślny, produkcyjny
        embedder = FaceEmbedder(model_type="vit")       # badawczy, wymaga GPU/CPU torch
    """

    def __init__(
        self,
        model_type: Literal["arcface", "vit"] = "arcface",
        model_pack: str = "buffalo_l",
        providers: list[str] | None = None,
        _app: FaceAnalysis | None = None,
        root: str = "./models",
    ) -> None:
        self._model_type = model_type
        self._root = root

        if model_type == "arcface":
            self._init_arcface(model_pack, providers, _app)
        elif model_type == "vit":
            self._init_vit()
        else:
            raise ValueError(f"Nieznany model_type='{model_type}'. Dozwolone: 'arcface', 'vit'.")

    # ── Inicjalizacja backendów ────────────────────────────────────────────

    def _init_arcface(
        self,
        model_pack: str,
        providers: list[str] | None,
        _app: FaceAnalysis | None,
    ) -> None:
        if _app is not None:
            self._app = _app
            logger.debug("FaceEmbedder[arcface]: używa przekazanej instancji FaceAnalysis.")
        else:
            _providers = providers or ["CPUExecutionProvider"]
            logger.info("Ładowanie insightface (%s) z %s...", model_pack, self._root)
            self._app = FaceAnalysis(name=model_pack, providers=_providers, root=self._root)
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("FaceEmbedder[arcface] gotowy.")

    def _init_vit(self) -> None:
        logger.info("Ładowanie ViT-B/16 (torchvision, pretrained=ImageNet)...")
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # Usuwamy głowicę klasyfikacyjną — zostawiamy tylko feature extractor
        vit.heads = nn.Identity()
        self._vit: nn.Module = vit.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vit = self._vit.to(self._device)
        logger.info("FaceEmbedder[vit] gotowy (device=%s).", self._device)

    # ── Publiczne API ──────────────────────────────────────────────────────

    def embed(self, aligned_crop: np.ndarray) -> np.ndarray:
        """
        Ekstrahuje embedding z aligned crop i normalizuje go L2.

        aligned_crop : (112, 112, 3) BGR uint8 — wyjście z FaceDetector
        Zwraca       : (D,) float32 znormalizowany L2
                       D=512 dla arcface, D=768 dla vit
        """
        if aligned_crop is None or aligned_crop.size == 0:
            raise ValueError("embed: otrzymano pusty aligned_crop.")

        if self._model_type == "arcface":
            return self._embed_arcface(aligned_crop)
        else:
            return self._embed_vit(aligned_crop)

    def verify(
        self,
        embedding_probe: np.ndarray,
        embedding_reference: np.ndarray,
        threshold: float = VERIFICATION_THRESHOLD,
    ) -> VerificationResult:
        """
        Porównuje dwa znormalizowane L2 embeddingi dystansem kosinusowym.

        d = 1.0 - dot(v1, v2)   →  d ≤ threshold = MATCH
        """
        _assert_normalized(embedding_probe, "embedding_probe")
        _assert_normalized(embedding_reference, "embedding_reference")

        distance = float(np.clip(1.0 - np.dot(embedding_probe, embedding_reference), 0.0, 2.0))
        return VerificationResult(
            is_match=distance <= threshold,
            cosine_distance=distance,
            threshold=threshold,
        )

    # ── ArcFace backend ────────────────────────────────────────────────────

    def _embed_arcface(self, aligned_crop: np.ndarray) -> np.ndarray:
        faces = self._app.get(aligned_crop)

        if not faces:
            raw = self._embed_arcface_direct(aligned_crop)
        else:
            raw = faces[0].embedding

        if raw is None:
            raise ValueError("Model ArcFace nie zwrócił embeddingu.")

        return _l2_normalize(raw.astype(np.float32))

    def _embed_arcface_direct(self, aligned_crop: np.ndarray) -> np.ndarray:
        """Fallback: bezpośrednie wywołanie modelu rec gdy detekcja nie znalazła twarzy."""
        rec_model = None
        for model in self._app.models.values():
            if hasattr(model, "get_feat"):
                rec_model = model
                break

        if rec_model is None:
            raise ValueError("Nie znaleziono modelu rekognicji w FaceAnalysis.")

        return rec_model.get_feat([aligned_crop]).flatten()

    # ── ViT backend ────────────────────────────────────────────────────────

    def _embed_vit(self, aligned_crop: np.ndarray) -> np.ndarray:
        """
        Preprocessing aligned_crop (112×112 BGR) → ViT-B/16 → 768-D L2-normalized.

        Pipeline:
          1. BGR → RGB
          2. resize 112 → 224 (wymóg ViT-B/16)
          3. HWC uint8 → CHW float32 / 255.0
          4. Normalizacja ImageNet (mean/std)
          5. Forward pass przez ViT bez głowicy
          6. Normalizacja L2
        """
        # BGR → RGB, resize do 224×224
        rgb = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (_VIT_INPUT_SIZE, _VIT_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        # HWC → CHW, [0,255] → [0,1]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, 224, 224)
        tensor = _VIT_NORMALIZE(tensor).unsqueeze(0).to(self._device)  # (1, 3, 224, 224)

        with torch.no_grad():
            features = self._vit(tensor)  # (1, 768) — po usunięciu heads

        vec = features.squeeze(0).cpu().numpy().astype(np.float32)  # (768,)
        return _l2_normalize(vec)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalizacja L2: vec / ||vec||₂  (wymagana przez SKILLS.md)."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        logger.warning("_l2_normalize: wektor bliski zeru — zwracam bez normalizacji.")
        return vec
    return vec / norm


def _assert_normalized(vec: np.ndarray, name: str) -> None:
    """Rzuca ValueError jeśli wektor nie jest znormalizowany L2 (tolerancja 1e-5)."""
    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"{name} nie jest znormalizowany L2 (||v|| = {norm:.6f}). Wywołaj embed() przed verify().")
