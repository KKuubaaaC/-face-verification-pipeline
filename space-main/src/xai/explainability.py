"""
Explainability helpers for face embedding models.

Supported paths:
  model_type="vit"      -> Attention Rollout (Abnar & Zuidema, 2020)
                           Propagates attention weights through ViT blocks,
                           builds a heatmap, and blends it over the crop.

  model_type="swinface" -> Global feature map (last Swin-T stage)
                           Forward hooks capture global_features
                           (B, 768, 7×7) -> channel mean -> 7×7 map.
                           Interpret as spatial activity before FAM/TSS (CAM-like).

Attention Rollout vs. Grad-CAM:
  Grad-CAM relies on gradients of the last conv layer; ViT has no such layer,
  so Grad-CAM is unstable. Attention Rollout is gradient-free: it composes
  attention matrices across layers with residual “skip” mixing (each token
  retains 0.5 self-weight).

  SwinFace uses window attention without a global CLS token, so we average
  the last-stage feature map instead.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

logger = logging.getLogger(__name__)

_VIT_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_VIT_INPUT_SIZE: int = 224


class FaceXAI:
    """
    Stateless per-image XAI for face crops; ViT weights load on first use if not injected.

    Share a ViT instance with FaceEmbedder via `_vit_model`.
    """

    def __init__(
        self,
        _vit_model: nn.Module | None = None,
        _swinface_model: nn.Module | None = None,
    ) -> None:
        self._vit: nn.Module | None = None
        self._swinface: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        if _vit_model is not None:
            self._vit = _vit_model.to(self._device)
            logger.debug("FaceXAI: using injected ViT model.")
        if _swinface_model is not None:
            self._swinface = _swinface_model.to(self._device)
            logger.debug("FaceXAI: using injected SwinFace model.")

    def generate_attention_map(
        self,
        image_crop: np.ndarray,
        model_type: str = "vit",
    ) -> np.ndarray:
        """
        Build an attention / feature map and blend it over the crop.

        image_crop  : (H, W, 3) BGR uint8 aligned crop
        model_type  : "vit" | "swinface"
        Returns     : (H, W, 3) RGB uint8 overlay
        """
        try:
            if model_type == "vit":
                return self._attention_rollout(image_crop)
            if model_type == "swinface":
                return self._swinface_feature_map(image_crop)
            raise ValueError(f"Unsupported model_type='{model_type}'.")
        except Exception as exc:
            logger.error("generate_attention_map failed: %s", exc, exc_info=True)
            return _error_frame(image_crop)

    def _attention_rollout(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Attention Rollout for ViT-B/16.

        Algorithm (Abnar & Zuidema, 2020):
          1. Forward ViT while recording attention A_l per layer.
          2. Per layer add identity skip: A_l' = 0.5 * A_l + 0.5 * I
          3. Row-normalize to sum 1.
          4. Multiply layers: rollout = A_1' @ A_2' @ ... @ A_L'
          5. First column = patch influence on CLS.
          6. Reshape to patch grid, upsample, blend.
        """
        vit = self._get_vit()
        tensor, orig_rgb = self._preprocess(bgr_crop)

        attention_maps: list[torch.Tensor] = []
        hook_handles: list[torch.utils.hooks.RemovableHook] = []

        def _make_hook(storage: list[torch.Tensor]):
            def _hook(module: nn.MultiheadAttention, inp, output) -> None:
                # Calling module() inside the hook recurses; call the functional path.
                with torch.no_grad():
                    q = inp[0]
                    if getattr(module, "batch_first", False):
                        q = q.transpose(0, 1)
                    _, attn = F.multi_head_attention_forward(
                        q,
                        q,
                        q,
                        embed_dim_to_check=module.embed_dim,
                        num_heads=module.num_heads,
                        in_proj_weight=module.in_proj_weight,
                        in_proj_bias=module.in_proj_bias,
                        bias_k=module.bias_k,
                        bias_v=module.bias_v,
                        add_zero_attn=module.add_zero_attn,
                        dropout_p=0.0,
                        out_proj_weight=module.out_proj.weight,
                        out_proj_bias=module.out_proj.bias,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    if attn is not None:
                        storage.append(attn.detach().cpu())

            return _hook

        for block in vit.encoder.layers:
            h = block.self_attention.register_forward_hook(_make_hook(attention_maps))
            hook_handles.append(h)

        try:
            with torch.no_grad():
                _ = vit(tensor)
        finally:
            for h in hook_handles:
                h.remove()

        if not attention_maps:
            raise RuntimeError("No attention maps captured; hooks did not fire.")

        heatmap = _compute_rollout(attention_maps)

        grid_size = 14
        heatmap_2d = heatmap.reshape(grid_size, grid_size)

        overlay = _apply_heatmap(orig_rgb, heatmap_2d)
        return overlay

    def _swinface_feature_map(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        SwinFace XAI via the last-stage global feature tensor.

          1. Forward SwinFace backbone with a hook.
          2. Capture global_features: (B, 768, 7, 7).
          3. Channel mean -> spatial map.
          4. Normalize to [0, 1] and blend as JET.
        """
        model = self._get_swinface()
        orig_rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

        rgb112 = cv2.resize(orig_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb112).permute(2, 0, 1).float()
        t = t.div_(255).sub_(0.5).div_(0.5).unsqueeze(0).to(self._device)

        captured: list[torch.Tensor] = []

        def _hook(module: nn.Module, inp, output) -> None:
            global_features = output[1]
            captured.append(global_features.detach().cpu())

        handle = model.backbone.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                _ = model(t)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("SwinFace XAI: hook did not capture features.")

        gf = captured[0][0]
        heatmap_2d = gf.mean(dim=0).numpy().astype(np.float32)

        a = heatmap_2d
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)

        overlay = _apply_heatmap(orig_rgb, a)
        return overlay

    def _get_swinface(self) -> nn.Module:
        """Return injected SwinFace module (no lazy download)."""
        if self._swinface is None:
            raise RuntimeError(
                "FaceXAI: SwinFace model was not injected. "
                "Pass FaceXAI(_swinface_model=emb._model) after SwinFaceEmbedder loads."
            )
        return self._swinface

    def _get_vit(self) -> nn.Module:
        """Lazy-load ViT-B/16 on first use unless injected."""
        if self._vit is None:
            logger.info("XAI: loading ViT-B/16...")
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            vit.heads = nn.Identity()
            self._vit = vit.eval().to(self._device)
            logger.info("XAI: ViT-B/16 ready (device=%s).", self._device)
        return self._vit

    def _preprocess(self, bgr_crop: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """
        BGR crop -> ViT tensor and original RGB for blending.
        Returns (tensor (1,3,224,224), orig_rgb uint8).
        """
        orig_rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            orig_rgb, (_VIT_INPUT_SIZE, _VIT_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )

        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        t = _VIT_NORMALIZE(t).unsqueeze(0).to(self._device)
        return t, orig_rgb


def _compute_rollout(attention_maps: list[torch.Tensor]) -> np.ndarray:
    """
    Compose per-layer attention into an Attention Rollout vector.

    attention_maps : list of (1, heads, seq_len, seq_len)
    Returns        : (seq_len - 1,) float32 patch weights for CLS
                     (drop CLS when reshaping to the patch grid)
    """
    rollout: torch.Tensor | None = None

    for attn in attention_maps:
        avg = attn[0].mean(dim=0)
        seq_len = avg.shape[0]

        identity = torch.eye(seq_len, device=avg.device)
        aug = 0.5 * avg + 0.5 * identity

        row_sums = aug.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        aug = aug / row_sums

        rollout = aug if rollout is None else torch.matmul(aug, rollout)

    if rollout is None:
        raise RuntimeError("_compute_rollout: empty attention map list.")

    cls_attention = rollout[0, 1:]

    a = cls_attention.float().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return a.astype(np.float32)


def _apply_heatmap(
    orig_rgb: np.ndarray,
    heatmap_2d: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend JET heatmap over RGB.

    orig_rgb   : (H, W, 3) uint8 RGB
    heatmap_2d : (grid, grid) float32 [0, 1]
    alpha      : heatmap weight in the blend
    Returns    : (H, W, 3) uint8 RGB
    """
    h, w = orig_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(
        heatmap_rgb.astype(np.float32),
        alpha,
        orig_rgb.astype(np.float32),
        1.0 - alpha,
        0.0,
    ).astype(np.uint8)

    return blended


def _error_frame(bgr_crop: np.ndarray) -> np.ndarray:
    """Return RGB crop with an 'XAI Error' label when visualization fails."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB).copy()
    cv2.putText(
        rgb,
        "XAI Error",
        (10, rgb.shape[0] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (220, 50, 50),
        2,
        cv2.LINE_AA,
    )
    return rgb
