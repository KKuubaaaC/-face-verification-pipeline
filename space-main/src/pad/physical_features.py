"""
Physical texture descriptors for presentation-attack cues (print / screen).

**LBP (uniform, R=1, P=8)** — rotation- and illumination-robust micro-texture histogram
(59 bins) useful for print artifacts.

**FFT log-magnitude spectrum** — high-frequency energy patterns linked to moiré from
camera vs display pixel grids; resized to 64×64 then flattened by default.

**Combined feature vector:** concatenation ``[LBP histogram | FFT flatten]`` for downstream
classical models.

Inputs may be BGR ``(H,W,3)`` or grayscale ``(H,W)`` ``uint8``; internally converted to
float32 grayscale.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)

# --- LBP constants ---
_LBP_RADIUS: int = 1
_LBP_POINTS: int = 8
_LBP_METHOD: str = "uniform"
# Bin count for uniform LBP: P*(P-1)+3 -> 8*7+3 = 59
_LBP_N_BINS: int = _LBP_POINTS * (_LBP_POINTS - 1) + 3  # 59

# --- FFT constants ---
# Spectrum map size before flattening to a feature vector.
# 64×64 = 4096 dims — enough for SVM/MLP; shrink for CNN heads.
_FFT_MAP_SIZE: int = 64


# --- Main class ---


class PhysicalFeatureExtractor:
    """
    Stateless extractor of LBP + FFT physical descriptors for a face patch.

    Example::

        ext = PhysicalFeatureExtractor()
        hist = ext.get_lbp_hist(face_crop)           # (59,) float32
        spec = ext.get_fft_spectrum(face_crop)       # (64, 64) float32
        feats = ext.get_combined_features(face_crop)  # (4155,) float32
    """

    def __init__(
        self,
        lbp_radius: int = _LBP_RADIUS,
        lbp_points: int = _LBP_POINTS,
        fft_map_size: int = _FFT_MAP_SIZE,
    ) -> None:
        self._r = lbp_radius
        self._p = lbp_points
        self._n_bins = lbp_points * (lbp_points - 1) + 3
        self._fft_sz = fft_map_size

    # --- LBP ---

    def get_lbp_hist(self, image: np.ndarray) -> np.ndarray:
        """
        Normalized LBP histogram (uniform, R=1, P=8).

        Parameters
        ----------
        image : (H, W, 3) BGR uint8  or  (H, W) uint8

        Returns
        -------
        hist : (59,) float32 — sums to 1.0, robust to rotation / lighting
        """
        gray = _to_gray(image)

        # skimage local_binary_pattern -> float labels in [0, P+1]
        lbp_map = local_binary_pattern(
            gray,
            P=self._p,
            R=self._r,
            method=self._lbp_method,
        )

        hist, _ = np.histogram(
            lbp_map.ravel(),
            bins=self._n_bins,
            range=(0, self._n_bins),
            density=False,
        )

        # L1 normalize -> probability mass (float32)
        total = float(hist.sum())
        hist_norm = (hist.astype(np.float32)) / np.float32(total + 1e-9)

        return hist_norm  # (59,) float32

    @property
    def _lbp_method(self) -> str:
        return _LBP_METHOD

    # --- FFT ---

    def get_fft_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        Log-amplitude spectrum (2D FFT with DC shifted to center).

        Steps:
          1. float32 grayscale
          2. numpy.fft.fft2 — full-image DFT
          3. numpy.fft.fftshift — move DC to center
          4. magnitude |F|
          5. log(1 + |F|) dampens DC and highlights high-frequency moiré peaks
          6. resize to (fft_map_size × fft_map_size)

        Parameters
        ----------
        image : (H, W, 3) BGR uint8  or  (H, W) uint8

        Returns
        -------
        spectrum : (fft_map_size, fft_map_size) float32
        """
        gray = _to_gray(image).astype(np.float32)

        F = np.fft.fft2(gray)
        F_shifted = np.fft.fftshift(F)
        magnitude = np.abs(F_shifted)
        log_spectrum = np.log1p(magnitude).astype(np.float32)

        if log_spectrum.shape != (self._fft_sz, self._fft_sz):
            log_spectrum = cv2.resize(
                log_spectrum,
                (self._fft_sz, self._fft_sz),
                interpolation=cv2.INTER_LINEAR,
            )

        return log_spectrum

    # --- Combined feature vector ---

    def get_combined_features(self, image: np.ndarray) -> np.ndarray:
        """
        Concatenated physical feature vector for classical classifiers.

        Contents:
          - LBP histogram        :  59-D   (print texture)
          - flattened FFT map    :  fft_map_size²-D  (screen grid / moiré)

        Default: 59 + 64*64 = 4155 float32 values.

        LBP uses L1 normalization; FFT map uses min-max so both live in ~[0, 1].

        Parameters
        ----------
        image : (H, W, 3) BGR uint8  or  (H, W) uint8

        Returns
        -------
        features : (59 + fft_map_size², ) float32
        """
        lbp_hist = self.get_lbp_hist(image)

        fft_map = self.get_fft_spectrum(image)
        fft_norm = _minmax_normalize(fft_map).ravel()

        return np.concatenate([lbp_hist, fft_norm], axis=0).astype(np.float32)

    # --- Dimension helpers ---

    @property
    def feature_dim(self) -> int:
        """Total dimension of ``get_combined_features()``."""
        return self._n_bins + self._fft_sz * self._fft_sz

    @property
    def lbp_dim(self) -> int:
        return self._n_bins

    @property
    def fft_dim(self) -> int:
        return self._fft_sz * self._fft_sz


# --- Visualization ---


def visualize_lbp(image: np.ndarray) -> np.ndarray:
    """
    LBP map as uint8 grayscale for debugging.

    image : (H, W, 3) BGR uint8  or  (H, W) uint8
    Returns (H, W) uint8 — raw LBP scaled to [0, 255]
    """
    gray = _to_gray(image)
    lbp_map = local_binary_pattern(gray, P=_LBP_POINTS, R=_LBP_RADIUS, method=_LBP_METHOD)
    lbp_u8 = cv2.normalize(lbp_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    return lbp_u8


def visualize_fft(image: np.ndarray, map_size: int = _FFT_MAP_SIZE) -> np.ndarray:
    """
    FFT spectrum as a BGR JET heatmap for Gradio.

    image    : (H, W, 3) BGR uint8  or  (H, W) uint8
    Returns  : (map_size, map_size, 3) BGR uint8
    """
    ext = PhysicalFeatureExtractor(fft_map_size=map_size)
    spectrum = ext.get_fft_spectrum(image)
    norm = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return heatmap


# --- Helpers ---


def _to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary input to uint8 grayscale.

    Supports BGR/RGB 3-channel, single channel, and float inputs.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    if gray.dtype != np.uint8:
        if gray.max() <= 1.0:
            gray = (gray * 255).clip(0, 255).astype(np.uint8)
        else:
            gray = gray.clip(0, 255).astype(np.uint8)

    return gray


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize matrix to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)
