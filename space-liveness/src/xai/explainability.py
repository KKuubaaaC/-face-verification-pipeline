"""
Explainability module — wizualizacja decyzji modeli embeddingów twarzy.

Obsługiwane metody:
  model_type="vit"      →  Attention Rollout (Abnar & Zuidema, 2020)
                            Propaguje wagi atencji przez wszystkie warstwy Transformera,
                            tworzy mapę cieplną i nakłada ją na oryginalny crop.

  model_type="swinface" →  Global Feature Map (ostatni stage Swin-T)
                            Hooki na backbone.forward przechwytują global_features
                            (B, 768, 7×7) → uśrednienie po kanałach → mapa 7×7.
                            Interpretacja: aktywność przestrzenna ostatniego etapu
                            przed FAM/TSS, analogiczna do Class Activation Map.

Attention Rollout vs. Grad-CAM:
  Grad-CAM opiera się na gradientach ostatniej warstwy konwolucyjnej — w ViT
  nie ma takich warstw, więc metoda jest niestabilna. Attention Rollout jest
  metodą bezgradientową: agreguje macierze atencji ze wszystkich warstw
  uwzględniając "skip connections" (każdy token widzi sam siebie z wagą 0.5).

  Dla Swin Transformer window-based attention nie ma globalnego tokenu CLS,
  więc używamy uśrednionej mapy cech z ostatniego stage (global_features).
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
    Generuje mapy wyjaśnialności (XAI) dla embeddingów twarzy.

    Instancja jest bezstanowa względem obrazów — modele ładowane raz w __init__.
    Współdzielenie modelu z FaceEmbedder jest możliwe przez parametr _vit_model.

    Użycie:
        xai = FaceXAI()
        overlay = xai.generate_attention_map(aligned_crop, model_type="vit")
        # overlay: (H, W, 3) RGB uint8 — gotowy do wyświetlenia w Gradio
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
            logger.debug("FaceXAI: używa przekazanego modelu ViT.")
        if _swinface_model is not None:
            self._swinface = _swinface_model.to(self._device)
            logger.debug("FaceXAI: używa przekazanego modelu SwinFace.")
        # Lazy-load: modele ładowane dopiero przy pierwszym wywołaniu generate_attention_map

    # ── Publiczne API ──────────────────────────────────────────────────────

    def generate_attention_map(
        self,
        image_crop: np.ndarray,
        model_type: str = "vit",
    ) -> np.ndarray:
        """
        Generuje mapę atencji i nakłada ją na image_crop.

        image_crop  : (H, W, 3) BGR uint8 — aligned crop z FaceDetector
        model_type  : "vit" (jedyna obsługiwana metoda)
        Zwraca      : (H, W, 3) RGB uint8 — blend oryginału z heatmapą
        """
        try:
            if model_type == "vit":
                return self._attention_rollout(image_crop)
            elif model_type == "swinface":
                return self._swinface_feature_map(image_crop)
            else:
                raise ValueError(f"Nieobsługiwany model_type='{model_type}'.")
        except Exception as exc:
            logger.error("generate_attention_map błąd: %s", exc, exc_info=True)
            return _error_frame(image_crop)

    # ── Attention Rollout (ViT) ────────────────────────────────────────────

    def _attention_rollout(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Attention Rollout dla ViT-B/16.

        Algorytm (Abnar & Zuidema, 2020):
          1. Przesuń obraz przez ViT rejestrując macierze atencji A_l każdej warstwy.
          2. Dla każdej warstwy dodaj macierz jednostkową (skip connection):
             A_l' = 0.5 * A_l + 0.5 * I
          3. Znormalizuj wiersze do sumy 1.
          4. Złóż (mnożenie macierzowe) przez wszystkie warstwy:
             rollout = A_1' @ A_2' @ ... @ A_L'
          5. Pierwsza kolumna rollout (indeks 0) to wpływ każdego patcha na token CLS.
          6. Reshape do siatki patchów → interpoluj do rozmiaru wejścia → heatmapa.
        """
        vit = self._get_vit()
        tensor, orig_rgb = self._preprocess(bgr_crop)

        # Zbieramy macierze atencji ze wszystkich bloków TransformerEncoder
        attention_maps: list[torch.Tensor] = []
        hook_handles: list[torch.utils.hooks.RemovableHook] = []

        def _make_hook(storage: list[torch.Tensor]):
            def _hook(module: nn.MultiheadAttention, inp, output) -> None:
                # Wywołanie module() wewnątrz hooka powoduje rekurencję.
                # Zamiast tego wołamy F.multi_head_attention_forward bezpośrednio
                # — omija zarejestrowane hooki i zwraca wagi atencji.
                with torch.no_grad():
                    q = inp[0]
                    # torchvision ViT używa batch_first=True → (batch, seq, embed).
                    # F.multi_head_attention_forward wymaga (seq, batch, embed).
                    if getattr(module, "batch_first", False):
                        q = q.transpose(0, 1)  # (seq_len, batch, embed_dim)
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
                        average_attn_weights=False,  # (batch, heads, seq, seq)
                    )
                    if attn is not None:
                        storage.append(attn.detach().cpu())

            return _hook

        # Rejestruj hook na każdym bloku self-attention w ViT-B/16
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
            raise RuntimeError("Nie zebrano żadnych map atencji — hooki nie zadziałały.")

        heatmap = _compute_rollout(attention_maps)  # (num_patches,) float32

        # Reshape do siatki patchów ViT-B/16: 224/16 = 14 × 14
        grid_size = 14
        heatmap_2d = heatmap.reshape(grid_size, grid_size)

        overlay = _apply_heatmap(orig_rgb, heatmap_2d)
        return overlay

    # ── SwinFace Feature Map ──────────────────────────────────────────────

    def _swinface_feature_map(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        XAI dla SwinFace — Global Feature Map z ostatniego stage Swin-T.

        Algorytm:
          1. Forward pass przez SwinFace backbone z hookiem na wyjściu.
          2. Przechwytujemy global_features: (B, 768, 7, 7).
          3. Uśredniamy po kanałach → (7, 7) mapa przestrzenna.
          4. Normalizujemy do [0, 1] i nakładamy jako heatmapę JET.
        """
        model = self._get_swinface()
        orig_rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

        # Preprocessing SwinFace: BGR→RGB, 112×112, (x/255-0.5)/0.5
        rgb112 = cv2.resize(orig_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb112).permute(2, 0, 1).float()
        t = t.div_(255).sub_(0.5).div_(0.5).unsqueeze(0).to(self._device)

        captured: list[torch.Tensor] = []

        def _hook(module: nn.Module, inp, output) -> None:
            # backbone.forward zwraca (local_features, global_features, embedding)
            # global_features: (B, 768, H, W) — ostatni stage, przed FAM
            global_features = output[1]
            captured.append(global_features.detach().cpu())

        handle = model.backbone.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                _ = model(t)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("SwinFace XAI: hook nie przechwycił cech.")

        gf = captured[0][0]  # (768, H, W) — np. (768, 7, 7)
        heatmap_2d = gf.mean(dim=0).numpy().astype(np.float32)  # (H, W)

        # Normalizacja do [0, 1]
        a = heatmap_2d
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)

        overlay = _apply_heatmap(orig_rgb, a)
        return overlay

    def _get_swinface(self) -> nn.Module:
        """Zwraca model SwinFace (przekazany w __init__ lub błąd — brak lazy-load)."""
        if self._swinface is None:
            raise RuntimeError(
                "FaceXAI: model SwinFace nie został przekazany. "
                "Utwórz FaceXAI(_swinface_model=emb._model) po załadowaniu SwinFaceEmbedder."
            )
        return self._swinface

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_vit(self) -> nn.Module:
        """Lazy-load ViT-B/16 przy pierwszym użyciu."""
        if self._vit is None:
            logger.info("XAI: Ładowanie ViT-B/16...")
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            vit.heads = nn.Identity()
            self._vit = vit.eval().to(self._device)
            logger.info("XAI: ViT-B/16 gotowy (device=%s).", self._device)
        return self._vit

    def _preprocess(self, bgr_crop: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """
        BGR crop → tensor do ViT + oryginalny RGB (zachowany do blendowania).
        Zwraca: (tensor (1,3,224,224), orig_rgb (H,W,3) uint8)
        """
        orig_rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(orig_rgb, (_VIT_INPUT_SIZE, _VIT_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        t = _VIT_NORMALIZE(t).unsqueeze(0).to(self._device)
        return t, orig_rgb


# ─── Rollout i wizualizacja ────────────────────────────────────────────────────


def _compute_rollout(attention_maps: list[torch.Tensor]) -> np.ndarray:
    """
    Złożenie macierzy atencji ze wszystkich warstw (Attention Rollout).

    attention_maps : lista tensorów (1, heads, seq_len, seq_len)
    Zwraca         : (seq_len - 1,) float32 — wagi patchów dla tokenu CLS
                     (seq_len - 1 bo odejmujemy token CLS z siatki patchów)
    """
    # Uśredniamy po głowicach → (seq_len, seq_len)
    rollout: torch.Tensor | None = None

    for attn in attention_maps:
        # attn: (batch=1, heads, seq_len, seq_len)
        avg = attn[0].mean(dim=0)  # (seq_len, seq_len)
        seq_len = avg.shape[0]

        # Skip connection: A' = 0.5*A + 0.5*I
        identity = torch.eye(seq_len, device=avg.device)
        aug = 0.5 * avg + 0.5 * identity

        # Normalizacja wierszowa
        row_sums = aug.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        aug = aug / row_sums

        rollout = aug if rollout is None else torch.matmul(aug, rollout)

    if rollout is None:
        raise RuntimeError("_compute_rollout: pusta lista map atencji.")

    # Pierwsza kolumna rollout = wpływ każdego tokenu na CLS (indeks 0)
    cls_attention = rollout[0, 1:]  # (num_patches,) — pomijamy token CLS

    # Normalizuj do [0, 1]
    a = cls_attention.float().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return a.astype(np.float32)


def _apply_heatmap(
    orig_rgb: np.ndarray,
    heatmap_2d: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Nakłada heatmapę JET na oryginalny obraz (blend).

    orig_rgb   : (H, W, 3) uint8 RGB
    heatmap_2d : (grid, grid) float32 [0, 1]
    alpha      : waga heatmapy w blendzie (0 = tylko oryginał, 1 = tylko heatmapa)
    Zwraca     : (H, W, 3) uint8 RGB
    """
    h, w = orig_rgb.shape[:2]

    # Interpoluj siatkę patchów do rozmiaru oryginału
    heatmap_resized = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # float [0,1] → uint8 [0,255] → colormap JET (BGR)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend: alpha * heatmap + (1-alpha) * original
    blended = cv2.addWeighted(
        heatmap_rgb.astype(np.float32),
        alpha,
        orig_rgb.astype(np.float32),
        1.0 - alpha,
        0.0,
    ).astype(np.uint8)

    return blended


def _error_frame(bgr_crop: np.ndarray) -> np.ndarray:
    """Zwraca oryginalny obraz (RGB) z napisem 'XAI Error' gdy coś pójdzie nie tak."""
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
