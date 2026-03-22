"""
SwinFace embedder — thin wrapper around the SwinFace Swin Transformer stack.

Architecture (SwinFace, Liu et al. 2023 — arxiv:2308.11509):
  backbone : Swin-T (img_size=112, patch_size=2, embed_dim=96, depths=[2,2,6,2])
  FAM      : Feature Attention Module (CBAM, split 3×3 conv)
  TSS      : Task-Specific Subnets (11 branches)
  OM       : Output Module → 42 heads + recognition embedding

Input:  (112, 112, 3) BGR uint8 aligned crop from FaceDetector
Output: 512-D L2-normalized recognition embedding
Multitask: age regression, gender, expression (7 RAF-DB classes), 40 face attributes

Weights: models/swinface.pt
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.vision.embedder import VERIFICATION_THRESHOLD, VerificationResult, _l2_normalize

logger = logging.getLogger(__name__)

# Paths ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SWINFACE_SRC = _PROJECT_ROOT / "third_party" / "swinface" / "swinface_project"
_CHECKPOINT = _PROJECT_ROOT / "models" / "swinface.pt"

# RAF-DB expression labels (7 classes) ---
_EXPRESSION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

_SWINFACE_BOOTSTRAPPED = False


def _bootstrap_swinface_imports() -> None:
    """
    Load only the SwinFace modules needed for inference, skipping analysis/__init__.py
    (which pulls mxnet/tensorboard training deps).

    Files are loaded via importlib.util.spec_from_file_location without package __init__.
    """
    global _SWINFACE_BOOTSTRAPPED
    if _SWINFACE_BOOTSTRAPPED:
        return

    def _load_file(module_name: str, file_path: Path) -> types.ModuleType:
        """Load a single .py file as a module."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = module_name.rsplit(".", 1)[0] if "." in module_name else ""
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    _a = _SWINFACE_SRC / "analysis"
    _b = _SWINFACE_SRC / "backbones"

    # --- analysis package (empty __init__ + real submodules) ---
    analysis_pkg = types.ModuleType("analysis")
    analysis_pkg.__path__ = [str(_a)]
    analysis_pkg.__package__ = "analysis"
    sys.modules["analysis"] = analysis_pkg

    cbam_mod = _load_file("analysis.cbam", _a / "cbam.py")
    subnets_mod = _load_file("analysis.subnets", _a / "subnets.py")
    analysis_pkg.cbam = cbam_mod  # type: ignore[attr-defined]
    analysis_pkg.subnets = subnets_mod  # type: ignore[attr-defined]

    # ── backbones package ─────────────────────────────────────────────────────
    backbones_pkg = types.ModuleType("backbones")
    backbones_pkg.__path__ = [str(_b)]
    backbones_pkg.__package__ = "backbones"
    sys.modules["backbones"] = backbones_pkg

    swin_mod = _load_file("backbones.swin", _b / "swin.py")
    backbones_pkg.swin = swin_mod  # type: ignore[attr-defined]

    # get_model inline (subset of backbones/__init__.py)
    def _get_model(name: str, **kwargs):  # type: ignore[override]
        if name == "swin_t":
            from backbones.swin import SwinTransformer  # noqa: PLC0415

            return SwinTransformer(num_classes=kwargs.get("num_features", 512))
        raise ValueError(f"Unknown backbone: {name}")

    backbones_pkg.get_model = _get_model  # type: ignore[attr-defined]

    _SWINFACE_BOOTSTRAPPED = True


# --- Model config (matches upstream inference.py) ---


class _SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size = 3
    fam_in_chans = 2112
    fam_conv_shared = False
    fam_conv_mode = "split"
    fam_channel_attention = "CBAM"
    fam_spatial_attention = None
    fam_pooling = "max"
    fam_la_num_list = [2 for _ in range(11)]
    fam_feature = "all"
    embedding_size = 512


# --- Multitask output ---


@dataclass
class SwinFaceAnalysis:
    """SwinFace multitask outputs for one crop."""

    embedding: np.ndarray  # (512,) float32 L2-normalized
    age: float  # regression output
    gender: str  # "Male" / "Female"
    gender_conf: float  # confidence [0, 1]
    expression: str  # RAF-DB label
    expression_conf: float  # confidence [0, 1]
    smiling: bool
    eyeglasses: bool
    attrs: dict[str, float] = field(default_factory=dict)
    # optional binary attribute heads -> P(True)


# --- Embedder ---


class SwinFaceEmbedder:
    """
    Eager-load SwinFace weights in __init__.

        emb = SwinFaceEmbedder()
        vec = emb.embed(aligned_crop)
        analysis = emb.analyze(aligned_crop)
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    # --- Weight loading ---

    def _load_model(self) -> None:
        if not _CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Missing SwinFace checkpoint: {_CHECKPOINT}\n"
                "Download from: https://drive.google.com/drive/folders/"
                "1NjVN3Kp_Tmwt17hWCIWgHpuWzkHYaman"
            )

        # Inference-only imports (skip analysis/__init__.py training stack).
        _bootstrap_swinface_imports()

        from analysis.subnets import (  # noqa: PLC0415
            FeatureAttentionModule,
            ModelBox,
            OutputModule,
            TaskSpecificSubnets,
        )
        from backbones import get_model  # noqa: PLC0415

        logger.info("SwinFace: loading weights (device=%s)...", self._device)
        cfg = _SwinFaceCfg()

        # Inline build_model (avoid importing model.py / analysis __init__)
        backbone = get_model(cfg.network, num_features=cfg.embedding_size)
        fam = FeatureAttentionModule(
            in_chans=cfg.fam_in_chans,
            kernel_size=cfg.fam_kernel_size,
            conv_shared=cfg.fam_conv_shared,
            conv_mode=cfg.fam_conv_mode,
            channel_attention=cfg.fam_channel_attention,
            spatial_attention=cfg.fam_spatial_attention,
            pooling=cfg.fam_pooling,
            la_num_list=cfg.fam_la_num_list,
        )
        tss = TaskSpecificSubnets()
        om = OutputModule()
        model = ModelBox(backbone=backbone, fam=fam, tss=tss, om=om, feature=cfg.fam_feature)

        ckpt = torch.load(_CHECKPOINT, map_location=self._device, weights_only=False)
        model.backbone.load_state_dict(ckpt["state_dict_backbone"])
        model.fam.load_state_dict(ckpt["state_dict_fam"])
        model.tss.load_state_dict(ckpt["state_dict_tss"])
        model.om.load_state_dict(ckpt["state_dict_om"])

        self._model = model.eval().to(self._device)
        logger.info("SwinFace: model ready.")

    # --- Public API ---

    def embed(self, aligned_crop: np.ndarray) -> np.ndarray:
        """
        Recognition embedding.

        aligned_crop : (112, 112, 3) BGR uint8
        Returns      : (512,) float32 L2-normalized
        """
        output = self._forward(aligned_crop)
        raw = output["Recognition"][0].cpu().numpy().astype(np.float32)
        return _l2_normalize(raw)

    def analyze(self, aligned_crop: np.ndarray) -> SwinFaceAnalysis:
        """Run every SwinFace head for one aligned crop."""
        output = self._forward(aligned_crop)

        embedding = _l2_normalize(output["Recognition"][0].cpu().numpy().astype(np.float32))

        # Age — regression (batch=1)
        age_raw = float(output["Age"][0].cpu().item())

        # Gender — 2-way softmax (0=Female, 1=Male per OM)
        gender_logits = output["Gender"][0].cpu().float()
        gender_probs = torch.softmax(gender_logits, dim=0).numpy()
        gender_idx = int(np.argmax(gender_probs))
        gender_label = "Male" if gender_idx == 1 else "Female"
        gender_conf = float(gender_probs[gender_idx])

        # Expression — 7-class RAF-DB
        expr_logits = output["Expression"][0].cpu().float()
        expr_probs = torch.softmax(expr_logits, dim=0).numpy()
        expr_idx = int(np.argmax(expr_probs))
        expr_label = _EXPRESSION_LABELS[expr_idx]
        expr_conf = float(expr_probs[expr_idx])

        # Smiling / Eyeglasses — binary (0=No, 1=Yes)
        smiling_prob = _binary_prob(output["Smiling"][0])
        eyeglasses_prob = _binary_prob(output["Eyeglasses"][0])

        # Optional binary attribute heads
        attr_names = [
            "Attractive",
            "Heavy Makeup",
            "Pale Skin",
            "Young",
            "Bangs",
            "Wearing Hat",
            "Eyeglasses",
            "Arched Eyebrows",
            "Bags Under Eyes",
            "Big Nose",
            "High Cheekbones",
            "Wearing Earrings",
            "No Beard",
            "Wearing Necklace",
        ]
        attrs: dict[str, float] = {}
        for name in attr_names:
            if name in output:
                attrs[name] = _binary_prob(output[name][0])

        return SwinFaceAnalysis(
            embedding=embedding,
            age=age_raw,
            gender=gender_label,
            gender_conf=gender_conf,
            expression=expr_label,
            expression_conf=expr_conf,
            smiling=smiling_prob >= 0.5,
            eyeglasses=eyeglasses_prob >= 0.5,
            attrs=attrs,
        )

    def verify(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        threshold: float = VERIFICATION_THRESHOLD,
    ) -> VerificationResult:
        """Cosine-distance verification (same convention as FaceEmbedder.verify)."""
        distance = float(np.clip(1.0 - np.dot(emb_a, emb_b), 0.0, 2.0))
        return VerificationResult(
            is_match=distance <= threshold,
            cosine_distance=distance,
            threshold=threshold,
        )

    # --- Internal ---

    def _forward(self, bgr_crop: np.ndarray) -> dict:
        """BGR crop → forward() → raw head dict."""
        assert self._model is not None
        tensor = _preprocess(bgr_crop, self._device)
        with torch.no_grad():
            output = self._model(tensor)
        return output


# --- Preprocessing ---


def preprocess_swinface(bgr_crop: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    BGR uint8 (112×112) → tensor (1, 3, 112, 112) float32 na device.
    Normalizacja: (x/255 - 0.5) / 0.5  →  [-1, 1]
    """
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float()
    t = t.div_(255).sub_(0.5).div_(0.5).unsqueeze(0).to(device)
    return t


# internal alias
_preprocess = preprocess_swinface


# --- Helper ---


def _binary_prob(logits: torch.Tensor) -> float:
    """Probability of the positive class for a 2-logit head."""
    probs = torch.softmax(logits.cpu().float(), dim=0).numpy()
    return float(probs[1]) if len(probs) >= 2 else 0.0
