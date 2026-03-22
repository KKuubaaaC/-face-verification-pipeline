"""
MobileNetV2 Grad-CAM for Presentation Attack Detection (PAD).

Architecture:
  encoder  : MobileNetV2 (ImageNet, torchvision) feature extractor
  target   : features[-1][-1] = ReLU6 (out_relu) — last conv layer
             for 224×224 input → feature map (1280, 7, 7)
  gradient : backprop from the L1 norm of globally averaged features,
             asking which spatial regions most affect the embedding representation.

Grad-CAM (Selvaraju et al. 2017, adapted per internal PDF §5):
  1. Forward pass → capture activations A in (1, 1280, 7, 7)
  2. score = sum |GAP(A)| — scalar “energy” of the embedding path
  3. score.backward() → capture gradients d(score)/dA
  4. alpha_k = (1/HW) sum_ij d(score)/dA_k_ij  — gradient GAP
  5. L = ReLU(sum_k alpha_k * A_k)             — weighted map
  6. Normalize → resize to 112×112

PAD logic (region-based scoring — PDF §9):
  Real score  rises when the heatmap focuses on Eyes, Mouth, Nose
              (live-face biometric cues)
  Spoof score rises when the heatmap focuses on image borders/frames
              (phone bezel, print edge, presentation artifacts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# --- ImageNet preprocessing ---
_MOBILENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_MOBILENET_INPUT_SIZE: int = 224  # 224×224 → 7×7 feature map


# --- Anatomical regions (PDF §9, on 112×112 heatmap) ---
# Coordinates: (row_slice, col_slice)
FACE_REGIONS: dict[str, tuple[slice, slice]] = {
    "Forehead": (slice(0, 30), slice(20, 92)),
    "Eyes": (slice(30, 55), slice(10, 102)),  # main REAL cue
    "Nose": (slice(45, 80), slice(35, 77)),
    "Mouth": (slice(75, 100), slice(25, 87)),  # second REAL cue
    "Chin/Jaw": (slice(95, 112), slice(15, 97)),
}

# Image borders (SPOOF cue: phone frame / print edge)
_EDGE_BAND: int = 12  # px from edge of 112-px map
EDGE_REGIONS: dict[str, tuple[slice, slice]] = {
    "Top": (slice(0, _EDGE_BAND), slice(0, 112)),
    "Bottom": (slice(112 - _EDGE_BAND, 112), slice(0, 112)),
    "Left": (slice(0, 112), slice(0, _EDGE_BAND)),
    "Right": (slice(0, 112), slice(112 - _EDGE_BAND, 112)),
}

# Region weights for real score
_REAL_WEIGHTS: dict[str, float] = {
    "Eyes": 0.50,
    "Mouth": 0.30,
    "Nose": 0.20,
}

# Activation thresholds
_SPOOF_EDGE_THRESHOLD: float = 0.25  # strong spoof if mean(edge) > this
_REAL_FACE_THRESHOLD: float = 0.30  # strong real if mean(eyes+mouth) > this


# --- Result ---


@dataclass
class GradCAMResult:
    """Grad-CAM + PAD scores for one face crop."""

    heatmap: np.ndarray  # (112, 112) float32 [0, 1]
    overlay: np.ndarray  # (112, 112, 3) RGB uint8
    real_score: float  # [0, 1] biometric regions
    spoof_score: float  # [0, 1] border/frame activation
    region_means: dict[str, float] = field(default_factory=dict)
    # {region_name -> mean_activation}
    verdict: str = ""  # "REAL" / "SPOOF" / "UNCERTAIN"


class MobileNetV2GradCAM:
    """
    MobileNetV2 Grad-CAM with PAD-style region scoring.

    Lazy-loads weights on first use.

        gcam = MobileNetV2GradCAM()
        result = gcam.analyze(aligned_crop_bgr)
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def analyze(self, bgr_crop: np.ndarray) -> GradCAMResult:
        """
        Full PAD-style analysis for one aligned face crop.

        bgr_crop : (H, W, 3) BGR uint8 aligned crop (e.g. 112×112)
        Returns  : GradCAMResult with heatmap, scores, and verdict
        """
        heatmap = self.compute_gradcam(bgr_crop)
        orig_rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        overlay = _apply_heatmap(orig_rgb, heatmap)
        real, spoof, region_means = self.compute_pad_scores(heatmap)
        verdict = _verdict(real, spoof)

        return GradCAMResult(
            heatmap=heatmap,
            overlay=overlay,
            real_score=real,
            spoof_score=spoof,
            region_means=region_means,
            verdict=verdict,
        )

    def compute_gradcam(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Grad-CAM heatmap from out_relu (features[-1]).

        Single-image variant (PDF §5):
          - No cosine similarity (no Siamese pair)
          - Backprop from L1 norm of GAP(features[-1]):
              score = sum |GlobalAvgPool(A)|
            Asks which spatial locations drive global embedding “energy”.
          - Same CAM recipe: alpha_k = GAP(dA_k), CAM = ReLU(sum alpha_k * A_k)

        bgr_crop : (H, W, 3) BGR uint8
        Returns  : (112, 112) float32 [0, 1]
        """
        model = self._get_model()
        tensor = _preprocess(bgr_crop, self._device)  # (1, 3, 224, 224)

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def _fwd_hook(module: nn.Module, inp, out: torch.Tensor) -> None:
            activations["out"] = out  # (1, 1280, 7, 7)

        def _bwd_hook(module: nn.Module, grad_in, grad_out: tuple) -> None:
            gradients["out"] = grad_out[0]  # (1, 1280, 7, 7)

        target_layer = model.features[-1]
        h_fwd = target_layer.register_forward_hook(_fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(_bwd_hook)

        try:
            with torch.enable_grad():
                model.zero_grad()
                feat_out = model.features(tensor)  # (1, 1280, 7, 7)

                gap = F.adaptive_avg_pool2d(feat_out, (1, 1)).squeeze()  # (1280,)
                score = gap.abs().sum()
                score.backward()
        finally:
            h_fwd.remove()
            h_bwd.remove()

        if "out" not in activations or "out" not in gradients:
            logger.error("Grad-CAM: hooks did not capture activations or gradients.")
            return np.zeros((112, 112), dtype=np.float32)

        act = activations["out"].detach()  # (1, 1280, 7, 7)
        grad = gradients["out"].detach()  # (1, 1280, 7, 7)

        weights = grad.mean(dim=[2, 3], keepdim=True)

        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()  # (7, 7) float32

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam.astype(np.float32), (112, 112), interpolation=cv2.INTER_CUBIC)
        return np.clip(cam, 0.0, 1.0).astype(np.float32)

    def compute_pad_scores(
        self,
        heatmap: np.ndarray,
    ) -> tuple[float, float, dict[str, float]]:
        """
        Spatial analysis of the heatmap -> real / spoof scores.

        Logic (extends PDF §9):
          Real score  = weighted Eyes (0.5) + Mouth (0.3) + Nose (0.2)
          Spoof score = mean activation on four image edges

        Both scores are independent (not constrained to sum to 1) and clipped to [0,1].
        Final verdict compares real_score vs spoof_score with confidence thresholds.

        heatmap : (112, 112) float32 [0, 1]
        Returns : (real_score, spoof_score, region_means)
        """
        region_means: dict[str, float] = {}

        for name, (rs, cs) in FACE_REGIONS.items():
            region_means[name] = float(heatmap[rs, cs].mean())

        edge_vals: list[float] = []
        for name, (rs, cs) in EDGE_REGIONS.items():
            v = float(heatmap[rs, cs].mean())
            region_means[f"Edge_{name}"] = v
            edge_vals.append(v)

        real_raw = sum(_REAL_WEIGHTS[k] * region_means[k] for k in _REAL_WEIGHTS)
        spoof_raw = float(np.mean(edge_vals))

        real_score = float(np.clip(real_raw, 0.0, 1.0))
        spoof_score = float(np.clip(spoof_raw, 0.0, 1.0))

        logger.debug(
            "PAD Grad-CAM: real=%.3f (eyes=%.3f mouth=%.3f nose=%.3f)  spoof=%.3f (edge_mean=%.3f)",
            real_score,
            region_means.get("Eyes", 0.0),
            region_means.get("Mouth", 0.0),
            region_means.get("Nose", 0.0),
            spoof_score,
            spoof_raw,
        )
        return real_score, spoof_score, region_means

    def _get_model(self) -> nn.Module:
        if self._model is None:
            logger.info("MobileNetV2 Grad-CAM: loading ImageNet weights...")
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            m.eval()
            self._model = m.to(self._device)
            logger.info(
                "MobileNetV2 ready (device=%s) target=features[-1][-1]=ReLU6",
                self._device,
            )
        return self._model


def _preprocess(bgr_crop: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    BGR uint8 -> (1, 3, 224, 224) float32 tensor with ImageNet normalization.

    224×224 input yields a 7×7 map at features[-1] (vs 4×4 for 112×112).
    """
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(
        rgb, (_MOBILENET_INPUT_SIZE, _MOBILENET_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
    )
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = _MOBILENET_NORMALIZE(t).unsqueeze(0).to(device)
    return t


def _apply_heatmap(
    orig_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.50,
) -> np.ndarray:
    """
    Blend a JET colormap heatmap over RGB.

    orig_rgb  : (H, W, 3) uint8 RGB
    heatmap   : (H, W)    float32 [0, 1]
    Returns   : (H, W, 3) uint8 RGB
    """
    h, w = orig_rgb.shape[:2]
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0.0, 1.0)
    hm_u8 = (hm * 255).astype(np.uint8)
    jet_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(
        jet_rgb.astype(np.float32),
        alpha,
        orig_rgb.astype(np.float32),
        1.0 - alpha,
        0.0,
    ).astype(np.uint8)
    return blended


def overlay_region_boxes(
    overlay_rgb: np.ndarray,
    region_means: dict[str, float],
    *,
    show_face: bool = True,
    show_edges: bool = True,
) -> np.ndarray:
    """
    Draw analysis region rectangles on the overlay (debug / Gradio).

    overlay_rgb  : (H, W, 3) uint8 RGB
    region_means : output of compute_pad_scores()[2]
    Returns      : (H, W, 3) uint8 RGB with rectangles
    """
    img = overlay_rgb.copy()

    if show_face:
        for name, (rs, cs) in FACE_REGIONS.items():
            mean_act = region_means.get(name, 0.0)
            g = int(mean_act * 255)
            color = (0, g, 255 - g)
            r0, c0, r1, c1 = rs.start, cs.start, rs.stop, cs.stop
            cv2.rectangle(img, (c0, r0), (c1, r1), color, 1)
            cv2.putText(
                img,
                f"{name[:3]} {mean_act:.2f}",
                (c0 + 2, r0 + 10),
                cv2.FONT_HERSHEY_PLAIN,
                0.65,
                color,
                1,
                cv2.LINE_AA,
            )

    if show_edges:
        for name, (rs, cs) in EDGE_REGIONS.items():
            mean_act = region_means.get(f"Edge_{name}", 0.0)
            r = int(mean_act * 255)
            color = (r, 0, 0)
            r0, c0 = rs.start, cs.start
            r1, c1 = rs.stop, cs.stop
            cv2.rectangle(img, (c0, r0), (c1, r1), color, 1)

    return img


def _verdict(real: float, spoof: float) -> str:
    """
    PAD verdict from scores.

      - If real_score > REAL_THRESHOLD and real > spoof  -> REAL
      - If spoof_score > SPOOF_THRESHOLD and spoof > real -> SPOOF
      - Else -> UNCERTAIN
    """
    if real > _REAL_FACE_THRESHOLD and real > spoof:
        return "REAL"
    if spoof > _SPOOF_EDGE_THRESHOLD and spoof > real:
        return "SPOOF"
    return "UNCERTAIN"
