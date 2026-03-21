"""
SwinFace embedder — wrapper wokół modelu SwinFace (Swin Transformer).

Architektura (SwinFace, Liu et al. 2023 — arxiv:2308.11509):
  backbone : Swin-T (img_size=112, patch_size=2, embed_dim=96, depths=[2,2,6,2])
  FAM      : Feature Attention Module (CBAM channel attention, split 3×3 conv)
  TSS      : Task-Specific Subnets (11 gałęzi)
  OM       : Output Module → 42 zadania + Recognition (embedding)

Wejście:  (112, 112, 3) BGR uint8  — aligned crop z FaceDetector
Wyjście:  512-D embedding L2-normalized  (Recognition head)
Multitask: Age (regresja), Gender, Expression (7 klas), 40 atrybutów twarzy

Checkpoint: models/swinface.pt
"""

from __future__ import annotations

import importlib.util
import logging
import os
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

# Ścieżki ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SWINFACE_SRC = _PROJECT_ROOT / "third_party" / "swinface" / "swinface_project"
_CHECKPOINT = _PROJECT_ROOT / "models" / "swinface.pt"

# Gdy w Space nie ustawisz SWINFACE_HF_REPO, spróbujemy tego repo (publiczne wagi).
_DEFAULT_HF_WEIGHTS_REPO = "KKUBBAACC/swinface-weights"

# Etykiety ekspresji (RAF-DB: 7 klas) ─────────────────────────────────────────
_EXPRESSION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

_SWINFACE_BOOTSTRAPPED = False


def _ensure_swinface_checkpoint(checkpoint: Path) -> None:
    """
    Jeśli brak lokalnego pliku, pobierz z Hugging Face Hub (np. osobne repo modelu).

    Zmienne środowiskowe (HF Space → Settings → Variables and secrets):
      SWINFACE_HF_REPO   — np. ``KKUBBAACC/swinface-weights`` (repo typu model)
      SWINFACE_HF_FILENAME — domyślnie ``swinface.pt``
    """
    if checkpoint.exists():
        return

    raw_repo = os.environ.get("SWINFACE_HF_REPO", "")
    repo_id = raw_repo.strip()
    if not repo_id:
        # Często Variables w Space nie są widoczne przy pierwszym deployu albo brak restartu.
        repo_id = _DEFAULT_HF_WEIGHTS_REPO
        logger.warning(
            "SWINFACE_HF_REPO puste w środowisku — używam domyślnego repo %s. "
            "Ustaw zmienną w Space → Settings → Variables and secrets, jeśli używasz innego modelu.",
            repo_id,
        )
    filename = (os.environ.get("SWINFACE_HF_FILENAME") or "swinface.pt").strip()

    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "SwinFace: brak lokalnego checkpointu — pobieram z Hub: %s / %s",
        repo_id,
        filename,
    )
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(checkpoint.parent),
        local_dir_use_symlinks=False,
    )
    downloaded = checkpoint.parent / filename
    if downloaded.exists() and downloaded.resolve() != checkpoint.resolve():
        downloaded.replace(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pobranie SwinFace z Hub nie utworzyło pliku: {checkpoint}")


def _bootstrap_swinface_imports() -> None:
    """
    Ładuje wyłącznie moduły SwinFace potrzebne do inferencji, omijając
    analysis/__init__.py (który wymaga mxnet, tensorboard — zależności treningowe).

    Używamy importlib.util.spec_from_file_location żeby załadować poszczególne
    pliki .py bezpośrednio, bez uruchamiania __init__.py pakietów.
    """
    global _SWINFACE_BOOTSTRAPPED
    if _SWINFACE_BOOTSTRAPPED:
        return

    def _load_file(module_name: str, file_path: Path) -> types.ModuleType:
        """Ładuje pojedynczy plik .py jako moduł o podanej nazwie."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = module_name.rsplit(".", 1)[0] if "." in module_name else ""
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    _a = _SWINFACE_SRC / "analysis"
    _b = _SWINFACE_SRC / "backbones"

    # ── analysis package (stub __init__ + realne submoduły) ─────────────────
    # Tworzymy pusty pakiet 'analysis' — unikamy mxnet/tensorboard w __init__.py
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

    # get_model inline (odpowiednik backbones/__init__.py, tylko swin_t)
    def _get_model(name: str, **kwargs):  # type: ignore[override]
        if name == "swin_t":
            from backbones.swin import SwinTransformer  # noqa: PLC0415

            return SwinTransformer(num_classes=kwargs.get("num_features", 512))
        raise ValueError(f"Nieznana architektura: {name}")

    backbones_pkg.get_model = _get_model  # type: ignore[attr-defined]

    _SWINFACE_BOOTSTRAPPED = True


# ─── Konfiguracja modelu (identyczna jak w inference.py) ─────────────────────


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


# ─── Wynik multitask ─────────────────────────────────────────────────────────


@dataclass
class SwinFaceAnalysis:
    """Wyniki wszystkich zadań SwinFace dla jednego obrazu."""

    embedding: np.ndarray  # (512,) float32 L2-normalized
    age: float  # wartość regresji (przybliżony wiek)
    gender: str  # "Male" / "Female"
    gender_conf: float  # pewność [0, 1]
    expression: str  # etykieta ekspresji (RAF-DB)
    expression_conf: float  # pewność [0, 1]
    smiling: bool
    eyeglasses: bool
    attrs: dict[str, float] = field(default_factory=dict)
    # wybrane atrybuty binarne → prawdopodobieństwo klasy True [0,1]


# ─── Embedder ────────────────────────────────────────────────────────────────


class SwinFaceEmbedder:
    """
    Lazy-load SwinFace przy pierwszym wywołaniu.

    Użycie:
        emb = SwinFaceEmbedder()
        vec = emb.embed(aligned_crop)          # → (512,) L2-normalized
        analysis = emb.analyze(aligned_crop)   # → SwinFaceAnalysis
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    # ── Ładowanie modelu ──────────────────────────────────────────────────

    def _load_model(self) -> None:
        _ensure_swinface_checkpoint(_CHECKPOINT)

        # Ładujemy tylko moduły potrzebne do inferencji, omijając __init__.py
        # pakietu 'analysis' (który importuje mxnet, tensorboard itp. — tylko trening).
        _bootstrap_swinface_imports()

        from analysis.subnets import (  # noqa: PLC0415
            FeatureAttentionModule,
            ModelBox,
            OutputModule,
            TaskSpecificSubnets,
        )
        from backbones import get_model  # noqa: PLC0415

        logger.info("SwinFace: ładowanie modelu (device=%s)...", self._device)
        cfg = _SwinFaceCfg()

        # Odtwarzamy build_model inline (nie importujemy model.py żeby uniknąć
        # ponownego triggeru analysis/__init__.py)
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
        logger.info("SwinFace: model gotowy.")

    # ── Publiczne API ─────────────────────────────────────────────────────

    def embed(self, aligned_crop: np.ndarray) -> np.ndarray:
        """
        Ekstrakcja embeddingu rozpoznawania twarzy.

        aligned_crop : (112, 112, 3) BGR uint8
        Zwraca       : (512,) float32 L2-normalized
        """
        output = self._forward(aligned_crop)
        raw = output["Recognition"][0].cpu().numpy().astype(np.float32)
        return _l2_normalize(raw)

    def analyze(self, aligned_crop: np.ndarray) -> SwinFaceAnalysis:
        """
        Pełna analiza multitask dla jednego cropa.

        Zwraca SwinFaceAnalysis z embeddingiem, wiekiem, płcią,
        ekspresją i wybranymi atrybutami.
        """
        output = self._forward(aligned_crop)

        embedding = _l2_normalize(output["Recognition"][0].cpu().numpy().astype(np.float32))

        # Age — regresja (batch=1 → scalar)
        age_raw = float(output["Age"][0].cpu().item())

        # Gender — softmax na 2 klasach: 0=Female, 1=Male (kolejność z OM)
        gender_logits = output["Gender"][0].cpu().float()
        gender_probs = torch.softmax(gender_logits, dim=0).numpy()
        gender_idx = int(np.argmax(gender_probs))
        gender_label = "Male" if gender_idx == 1 else "Female"
        gender_conf = float(gender_probs[gender_idx])

        # Expression — 7 klas RAF-DB
        expr_logits = output["Expression"][0].cpu().float()
        expr_probs = torch.softmax(expr_logits, dim=0).numpy()
        expr_idx = int(np.argmax(expr_probs))
        expr_label = _EXPRESSION_LABELS[expr_idx]
        expr_conf = float(expr_probs[expr_idx])

        # Smiling i Eyeglasses — binary (2 klasy: 0=No, 1=Yes)
        smiling_prob = _binary_prob(output["Smiling"][0])
        eyeglasses_prob = _binary_prob(output["Eyeglasses"][0])

        # Wybrane atrybuty binarne → dict {name: prob_True}
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
        """Porównanie dystansem kosinusowym (identyczne z FaceEmbedder.verify)."""
        distance = float(np.clip(1.0 - np.dot(emb_a, emb_b), 0.0, 2.0))
        return VerificationResult(
            is_match=distance <= threshold,
            cosine_distance=distance,
            threshold=threshold,
        )

    # ── Wewnętrzne ────────────────────────────────────────────────────────

    def _forward(self, bgr_crop: np.ndarray) -> dict:
        """BGR crop → forward pass → słownik wyników SwinFace."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        tensor = _preprocess(bgr_crop, self._device)
        with torch.no_grad():
            output = self._model(tensor)
        return output


# ─── Preprocessing ───────────────────────────────────────────────────────────


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


# alias wewnętrzny
_preprocess = preprocess_swinface


# ─── Helper ──────────────────────────────────────────────────────────────────


def _binary_prob(logits: torch.Tensor) -> float:
    """
    Zwraca prawdopodobieństwo klasy '1' (True) z tensorka (2,) logitów.
    """
    probs = torch.softmax(logits.cpu().float(), dim=0).numpy()
    return float(probs[1]) if len(probs) >= 2 else 0.0
