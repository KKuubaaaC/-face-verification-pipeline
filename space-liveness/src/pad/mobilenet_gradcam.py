"""
MobileNetV2 Grad-CAM dla Presentation Attack Detection (PAD).

Architektura:
  encoder  : MobileNetV2 (ImageNet, torchvision) — ekstraktor cech
  target   : features[-1][-1] = ReLU6 (out_relu) — ostatnia warstwa splotowa
             dla wejścia 224×224 → mapa (1280, 7, 7)
  gradient : backpropagacja od normy L1 globalnie uśrednionych cech,
             co odpowiada pytaniu: „które regiony przestrzenne najbardziej
             wpływają na reprezentację embeddingu?"

Algorytm Grad-CAM (Selvaraju et al. 2017, adaptacja do PDF §5):
  1. Forward pass → przechwycenie aktywacji A ∈ (1, 1280, 7, 7)
  2. score = Σ |GAP(A)| — skalar opisujący „energię" embeddingu
  3. score.backward() → przechwycenie gradientów ∂score/∂A
  4. α_k = (1/H·W) Σ_i Σ_j (∂score/∂A_k_ij)  — GAP gradientów
  5. L = ReLU(Σ_k α_k · A_k)                  — mapa ważona
  6. Normalizacja → resize do 112×112

Logika PAD (region-based scoring — PDF §9):
  Real score  ↑  gdy heatmapa skupia się na: Eyes, Mouth, Nose
              (cechy biometryczne żywej twarzy)
  Spoof score ↑  gdy heatmapa skupia się na: krawędziach/ramkach obrazu
              (ramka telefonu, brzeg wydruku, artefakt prezentacji)
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

# ─── Preprocessing (ImageNet) ─────────────────────────────────────────────────
_MOBILENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_MOBILENET_INPUT_SIZE: int = 224  # 224×224 → 7×7 feature map


# ─── Regiony anatomiczne (z PDF §9, na 112×112 heatmapie) ────────────────────
# Współrzędne: (slice_rows, slice_cols)
FACE_REGIONS: dict[str, tuple[slice, slice]] = {
    "Forehead": (slice(0, 30), slice(20, 92)),
    "Eyes": (slice(30, 55), slice(10, 102)),  # główny indykator REAL
    "Nose": (slice(45, 80), slice(35, 77)),
    "Mouth": (slice(75, 100), slice(25, 87)),  # drugi indykator REAL
    "Chin/Jaw": (slice(95, 112), slice(15, 97)),
}

# Krawędzie obrazu (indykator SPOOF: ramka telefonu / brzeg wydruku)
_EDGE_BAND: int = 12  # px od krawędzi 112-px obrazu
EDGE_REGIONS: dict[str, tuple[slice, slice]] = {
    "Top": (slice(0, _EDGE_BAND), slice(0, 112)),
    "Bottom": (slice(112 - _EDGE_BAND, 112), slice(0, 112)),
    "Left": (slice(0, 112), slice(0, _EDGE_BAND)),
    "Right": (slice(0, 112), slice(112 - _EDGE_BAND, 112)),
}

# Wagi regionów dla Real score
_REAL_WEIGHTS: dict[str, float] = {
    "Eyes": 0.50,
    "Mouth": 0.30,
    "Nose": 0.20,
}

# Próg odchylenia aktywacji od centrum → wzmocnienie spoof score
_SPOOF_EDGE_THRESHOLD: float = 0.25  # jeśli mean(edge) > this → silny sygnał spoof
_REAL_FACE_THRESHOLD: float = 0.30  # jeśli mean(eyes+mouth) > this → silny sygnał real


# ─── Wynik ────────────────────────────────────────────────────────────────────


@dataclass
class GradCAMResult:
    """Wynik analizy Grad-CAM dla jednego obrazu twarzy."""

    heatmap: np.ndarray  # (112, 112) float32 [0, 1]
    overlay: np.ndarray  # (112, 112, 3) RGB uint8
    real_score: float  # [0, 1] — aktywacja w regionach biometrycznych
    spoof_score: float  # [0, 1] — aktywacja na krawędziach/ramkach
    region_means: dict[str, float] = field(default_factory=dict)
    # {nazwa_regionu → mean_activation} dla wszystkich regionów
    verdict: str = ""  # "REAL" / "SPOOF" / "UNCERTAIN"


# ─── Główna klasa ─────────────────────────────────────────────────────────────


class MobileNetV2GradCAM:
    """
    MobileNetV2 Grad-CAM + PAD scoring.

    Lazy-load modelu przy pierwszym wywołaniu.

    Użycie::

        gcam = MobileNetV2GradCAM()
        result = gcam.analyze(aligned_crop_bgr)
        # result.heatmap   : (112, 112) float32
        # result.overlay   : (112, 112, 3) RGB uint8 — do Gradio
        # result.real_score : float ∈ [0, 1]
        # result.spoof_score: float ∈ [0, 1]
        # result.verdict   : "REAL" / "SPOOF" / "UNCERTAIN"
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Publiczne API ──────────────────────────────────────────────────────────

    def analyze(self, bgr_crop: np.ndarray) -> GradCAMResult:
        """
        Pełna analiza PAD na jednym cropsie twarzy.

        bgr_crop : (H, W, 3) BGR uint8 — aligned crop (np. 112×112)
        Zwraca   : GradCAMResult z heatmapą, score'ami i werdyktem
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
        Generuje heatmapę Grad-CAM z warstwy out_relu (features[-1]).

        Algorytm (PDF §5, adaptacja do pojedynczego obrazu):
          - Nie używamy podobieństwa kosinusowego (brak pary Siamese)
          - Backpropagujemy od L1-normy GAP(features[-1]):
              score = Σ |GlobalAvgPool(A)|
            Pytamy: „które lokalizacje przestrzenne mają największy wpływ
            na globalną energię embeddingu?"
          - Reszta identyczna: α_k = GAP(∇A_k), CAM = ReLU(Σ α_k·A_k)

        bgr_crop : (H, W, 3) BGR uint8
        Zwraca   : (112, 112) float32 [0, 1]
        """
        model = self._get_model()
        tensor = _preprocess(bgr_crop, self._device)  # (1, 3, 224, 224)

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def _fwd_hook(module: nn.Module, inp, out: torch.Tensor) -> None:
            activations["out"] = out  # (1, 1280, 7, 7)

        def _bwd_hook(module: nn.Module, grad_in, grad_out: tuple) -> None:
            gradients["out"] = grad_out[0]  # (1, 1280, 7, 7)

        # Podpinamy hooki na features[-1] (cały blok Conv+BN+ReLU6 = out_relu)
        target_layer = model.features[-1]
        h_fwd = target_layer.register_forward_hook(_fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(_bwd_hook)

        try:
            with torch.enable_grad():
                model.zero_grad()
                feat_out = model.features(tensor)  # (1, 1280, 7, 7)

                # Score: L1-norma globalnie uśrednionych cech
                # → pytamy o przestrzenne źródło energii embeddingu
                gap = F.adaptive_avg_pool2d(feat_out, (1, 1)).squeeze()  # (1280,)
                score = gap.abs().sum()
                score.backward()
        finally:
            h_fwd.remove()
            h_bwd.remove()

        if "out" not in activations or "out" not in gradients:
            logger.error("Grad-CAM: hooki nie przechwyciły danych.")
            return np.zeros((112, 112), dtype=np.float32)

        act = activations["out"].detach()  # (1, 1280, 7, 7)
        grad = gradients["out"].detach()  # (1, 1280, 7, 7)

        # α_k = GAP(∂score/∂A_k)  →  (1, 1280, 1, 1)
        weights = grad.mean(dim=[2, 3], keepdim=True)

        # L = ReLU(Σ_k α_k · A_k)  →  (1, 1, 7, 7)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()  # (7, 7) float32

        # Normalizacja do [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize do 112×112 (interpolacja kubiczna — gładsze krawędzie)
        cam = cv2.resize(cam.astype(np.float32), (112, 112), interpolation=cv2.INTER_CUBIC)
        return np.clip(cam, 0.0, 1.0).astype(np.float32)

    def compute_pad_scores(
        self,
        heatmap: np.ndarray,
    ) -> tuple[float, float, dict[str, float]]:
        """
        Analiza przestrzenna heatmapy → Real / Spoof score.

        Logika (rozszerzenie PDF §9 o logikę PAD):
          Real score  = ważona aktywacja w Eyes (×0.5) + Mouth (×0.3) + Nose (×0.2)
          Spoof score = mean aktywacji na 4 krawędziach obrazu (Top/Bottom/Left/Right)

          Oba score'y są niezależne (nie sumują się do 1) i normalizowane do [0,1].
          Końcowy werdykt: real_score vs spoof_score + próg pewności.

        heatmap : (112, 112) float32 [0, 1]
        Zwraca  : (real_score, spoof_score, region_means)
        """
        region_means: dict[str, float] = {}

        # ── Regiony anatomiczne ────────────────────────────────────────────
        for name, (rs, cs) in FACE_REGIONS.items():
            region_means[name] = float(heatmap[rs, cs].mean())

        # ── Krawędzie ─────────────────────────────────────────────────────
        edge_vals: list[float] = []
        for name, (rs, cs) in EDGE_REGIONS.items():
            v = float(heatmap[rs, cs].mean())
            region_means[f"Edge_{name}"] = v
            edge_vals.append(v)

        # ── Obliczenia score'ów ───────────────────────────────────────────
        # Real: weighted sum biometric regions
        real_raw = sum(_REAL_WEIGHTS[k] * region_means[k] for k in _REAL_WEIGHTS)
        # Spoof: mean edge activations
        spoof_raw = float(np.mean(edge_vals))

        # Normalizacja każdego score'u niezależnie do [0, 1]
        # używając znanych zakresów: aktywacje [0,1], weighted mean też [0,1]
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

    # ── Lazy model load ────────────────────────────────────────────────────────

    def _get_model(self) -> nn.Module:
        if self._model is None:
            logger.info("MobileNetV2 Grad-CAM: ładowanie modelu (ImageNet)…")
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            # Usuwamy classifier — używamy tylko features (enkoder)
            m.eval()
            self._model = m.to(self._device)
            logger.info(
                "MobileNetV2 gotowy (device=%s)  target=features[-1][-1]=ReLU6",
                self._device,
            )
        return self._model


# ─── Preprocessing ─────────────────────────────────────────────────────────────


def _preprocess(bgr_crop: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    BGR uint8 → tensor (1, 3, 224, 224) float32 z normalizacją ImageNet.

    Resize do 224×224 zapewnia mapę 7×7 na wyjściu features[-1]
    (w porównaniu do 4×4 dla wejścia 112×112).
    """
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_MOBILENET_INPUT_SIZE, _MOBILENET_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = _MOBILENET_NORMALIZE(t).unsqueeze(0).to(device)
    return t


# ─── Wizualizacja ──────────────────────────────────────────────────────────────


def _apply_heatmap(
    orig_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.50,
) -> np.ndarray:
    """
    Nakłada heatmapę JET na obraz RGB.

    orig_rgb  : (H, W, 3) uint8 RGB
    heatmap   : (H, W)    float32 [0, 1]
    Zwraca    : (H, W, 3) uint8 RGB
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
    Rysuje prostokąty regionów analizy na obrazie overlay (debug/Gradio).

    overlay_rgb  : (H, W, 3) uint8 RGB
    region_means : wynik compute_pad_scores()[2]
    Zwraca       : (H, W, 3) uint8 RGB z prostokątami
    """
    img = overlay_rgb.copy()

    if show_face:
        for name, (rs, cs) in FACE_REGIONS.items():
            mean_act = region_means.get(name, 0.0)
            # zielony dla wysokiej aktywacji, niebieski dla niskiej
            g = int(mean_act * 255)
            color = (0, g, 255 - g)  # BGR→RGB swap
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
            color = (r, 0, 0)  # czerwony = spoof
            r0, c0 = rs.start, cs.start
            r1, c1 = rs.stop, cs.stop
            cv2.rectangle(img, (c0, r0), (c1, r1), color, 1)

    return img


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _verdict(real: float, spoof: float) -> str:
    """
    Wyznacza werdykt PAD na podstawie score'ów.

    Logika:
      - Jeśli real_score > REAL_THRESHOLD  I  real > spoof  →  REAL
      - Jeśli spoof_score > SPOOF_THRESHOLD I  spoof > real →  SPOOF
      - W przeciwnym razie                                   →  UNCERTAIN
    """
    if real > _REAL_FACE_THRESHOLD and real > spoof:
        return "REAL"
    if spoof > _SPOOF_EDGE_THRESHOLD and spoof > real:
        return "SPOOF"
    return "UNCERTAIN"
