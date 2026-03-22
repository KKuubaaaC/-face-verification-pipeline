"""
Face embedding factory for ArcFace (InsightFace) and ViT (torchvision).

Supported ``model_type`` values:

- ``"arcface"`` — InsightFace ``buffalo_l`` ResNet-50 ArcFace, 512-D, L2-normalized.
  Default cosine-distance threshold ``0.1625`` is documented as an EER-oriented
  operating point (AgeDB-30 context in project notes).
- ``"vit"`` — ViT-B/16 without the classification head (768-D). Thresholds must be
  tuned for your use case.

Both backends expose the same API:
  ``embed(aligned_crop)`` → ``(D,)`` float32 L2-normalized
  ``verify(e1, e2)`` → :class:`VerificationResult`
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

# ImageNet normalization used by ViT (pretrained on ImageNet-21k)
_VIT_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_VIT_INPUT_SIZE: int = 224


@dataclass
class VerificationResult:
    """Cosine-distance check between two L2-normalized embeddings."""

    is_match: bool
    cosine_distance: float
    threshold: float = VERIFICATION_THRESHOLD


# ─── Factory ─────────────────────────────────────────────────────────────────


class FaceEmbedder:
    """
    Face embedder with pluggable backend (``model_type``).

    Examples::

        embedder = FaceEmbedder(model_type="arcface")  # production default
        embedder = FaceEmbedder(model_type="vit")      # research ViT backend
    """

    def __init__(
        self,
        model_type: Literal["arcface", "vit"] = "arcface",
        model_pack: str = "buffalo_l",
        providers: list[str] | None = None,
        shared_face_analysis: FaceAnalysis | None = None,
        root: str = "./models",
    ) -> None:
        self._model_type = model_type
        self._root = root

        if model_type == "arcface":
            self._init_arcface(model_pack, providers, shared_face_analysis)
        elif model_type == "vit":
            self._init_vit()
        else:
            raise ValueError(f"Unknown model_type='{model_type}'. Allowed: 'arcface', 'vit'.")

    # --- Backend initialization ---

    def _init_arcface(
        self,
        model_pack: str,
        providers: list[str] | None,
        shared_face_analysis: FaceAnalysis | None,
    ) -> None:
        if shared_face_analysis is not None:
            self._app = shared_face_analysis
            logger.debug("FaceEmbedder[arcface]: reusing shared FaceAnalysis instance.")
        else:
            _providers = providers or ["CPUExecutionProvider"]
            logger.info("Loading InsightFace (%s) from %s...", model_pack, self._root)
            self._app = FaceAnalysis(name=model_pack, providers=_providers, root=self._root)
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("FaceEmbedder[arcface] ready.")

    def _init_vit(self) -> None:
        logger.info("Loading ViT-B/16 (torchvision, ImageNet weights)...")
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # Drop classification head — keep feature extractor only
        vit.heads = nn.Identity()
        self._vit: nn.Module = vit.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vit = self._vit.to(self._device)
        logger.info("FaceEmbedder[vit] ready (device=%s).", self._device)

    # ── Publiczne API ──────────────────────────────────────────────────────

    def embed(self, aligned_crop: np.ndarray) -> np.ndarray:
        """
        Extract an L2-normalized embedding from an aligned BGR crop.

        ``aligned_crop``: (112, 112, 3) ``uint8`` from :class:`FaceDetector`.
        Returns ``(D,)`` float32 with D=512 (ArcFace) or 768 (ViT).
        """
        if aligned_crop is None or aligned_crop.size == 0:
            raise ValueError("embed: empty aligned_crop.")

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
        Cosine distance ``d = 1 - dot(v1, v2)`` on L2-normalized vectors.

        ``d <= threshold`` implies a match.
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
            raise ValueError("ArcFace model returned no embedding.")

        return _l2_normalize(raw.astype(np.float32))

    def _embed_arcface_direct(self, aligned_crop: np.ndarray) -> np.ndarray:
        """Call the recognition head directly when ``get`` finds no face on the crop."""
        rec_model = None
        for model in self._app.models.values():
            if hasattr(model, "get_feat"):
                rec_model = model
                break

        if rec_model is None:
            raise ValueError("No recognition model found in FaceAnalysis.")

        return rec_model.get_feat([aligned_crop]).flatten()

    # ── ViT backend ────────────────────────────────────────────────────────

    def _embed_vit(self, aligned_crop: np.ndarray) -> np.ndarray:
        """
        ViT-B/16 forward on a 112×112 BGR crop (resized to 224×224), L2-normalized 768-D.

        Steps: BGR→RGB, resize, ImageNet normalize, forward without classification head, L2 norm.
        """
        # BGR → RGB, resize to 224×224
        rgb = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (_VIT_INPUT_SIZE, _VIT_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        # HWC → CHW, scale to [0, 1]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, 224, 224)
        tensor = _VIT_NORMALIZE(tensor).unsqueeze(0).to(self._device)  # (1, 3, 224, 224)

        with torch.no_grad():
            features = self._vit(tensor)  # (1, 768) after removing heads

        vec = features.squeeze(0).cpu().numpy().astype(np.float32)  # (768,)
        return _l2_normalize(vec)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize ``vec`` to unit length (required before ``verify``)."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        logger.warning("_l2_normalize: near-zero vector — returning unchanged.")
        return vec
    return vec / norm


def _assert_normalized(vec: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` if ``vec`` is not L2-normalized (tolerance 1e-5)."""
    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(
            f"{name} is not L2-normalized (||v|| = {norm:.6f}). Call embed() before verify()."
        )
