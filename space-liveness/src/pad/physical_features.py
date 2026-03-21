"""
PhysicalFeatureExtractor — deskryptory fizyczne do wykrywania ataków prezentacyjnych.

Dwie komplementarne metody:

  LBP (Local Binary Patterns):
    Wariant ``uniform`` (Ojala et al. 2002) jest oporny na rotację i oświetlenie.
    Wykrywa artefakty druku (papier, wydruk) przez charakterystyczny rozkład
    wzorców tekstury — regularną siatkę tonera lub atramentu.
    Promień R=1, P=8 punktów → 10 klas uniform + 1 non-uniform = 59 binów histogramu.

  FFT Magnitude Spectrum (logarytmiczne widmo amplitudy):
    Efekt mory (Moiré) pochodzi z interferencji siatki pikseli ekranu z matrycą
    sensora. Objawia się jako symetryczne piki wysokich częstotliwości w widmie.
    np.fft.fft2 + fftshift → log(1 + |F|) → spłaszczone widmo do wektora cech.

Połączony wektor cech:
    features = [lbp_hist (59-D) | fft_flat (N*M-D)] → do klasyfikatora lub SVM.
    Domyślnie fft_flat pochodzi z mapy 64×64 (4096-D) po resize.

Kształty wejść:
    image: (H, W, 3) BGR uint8  lub  (H, W) szaroodcieniowy uint8
    Wewnętrznie konwertujemy do float32 grayscale.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)

# ─── Stałe LBP ───────────────────────────────────────────────────────────────
_LBP_RADIUS: int = 1
_LBP_POINTS: int = 8
_LBP_METHOD: str = "uniform"
# Liczba binów: P*(P-1)+3 dla uniform → 8*7+3 = 59
_LBP_N_BINS: int = _LBP_POINTS * (_LBP_POINTS - 1) + 3  # 59

# ─── Stałe FFT ───────────────────────────────────────────────────────────────
# Rozmiar mapy widmowej przed spłaszczeniem do wektora cech.
# 64×64 = 4096 wymiarów — wystarczy dla SVM/MLP; zmniejsz dla CNN heads.
_FFT_MAP_SIZE: int = 64


# ─── Główna klasa ─────────────────────────────────────────────────────────────


class PhysicalFeatureExtractor:
    """
    Bezstanowy ekstraktor deskryptorów fizycznych obrazu twarzy.

    Użycie::

        ext = PhysicalFeatureExtractor()

        hist  = ext.get_lbp_hist(face_crop)          # (59,)  float32
        spec  = ext.get_fft_spectrum(face_crop)       # (64,64) float32  [wizualizacja]
        feats = ext.get_combined_features(face_crop)  # (4155,) float32  [do klasyfikatora]
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

    # ── LBP ───────────────────────────────────────────────────────────────────

    def get_lbp_hist(self, image: np.ndarray) -> np.ndarray:
        """
        Znormalizowany histogram LBP (uniform, R=1, P=8).

        Parametry
        ---------
        image : (H, W, 3) BGR uint8  lub  (H, W) uint8

        Zwraca
        ------
        hist : (59,) float32 — suma = 1.0, oporny na rotację i zmianę oświetlenia
        """
        gray = _to_gray(image)

        # skimage.feature.local_binary_pattern → wartości float w [0, P+1]
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

        # Normalizacja L1 → rozkład prawdopodobieństwa (float32 przez cały czas)
        total = float(hist.sum())
        hist_norm = (hist.astype(np.float32)) / np.float32(total + 1e-9)

        return hist_norm  # (59,) float32

    @property
    def _lbp_method(self) -> str:
        return _LBP_METHOD

    # ── FFT ───────────────────────────────────────────────────────────────────

    def get_fft_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        Logarytmiczne widmo amplitudowe (2D FFT z przesunięciem DC do centrum).

        Algorytm:
          1. Konwersja do float32 grayscale.
          2. numpy.fft.fft2 — DFT 2D na całym obrazie.
          3. numpy.fft.fftshift — przesuwa składową stałą (DC) do centrum mapy.
          4. Amplituda: |F(u,v)| = sqrt(Re² + Im²).
          5. Skala logarytmiczna: log(1 + |F|) tłumi dominację DC
             i uwydatnia subtelne piki wysokich częstotliwości (efekt mory).
          6. Resize do (fft_map_size × fft_map_size) dla standaryzacji wymiarów.

        Parametry
        ---------
        image : (H, W, 3) BGR uint8  lub  (H, W) uint8

        Zwraca
        ------
        spectrum : (fft_map_size, fft_map_size) float32 — wartości w [0, ~log(max+1)]
                   gotowe do wizualizacji lub spłaszczenia jako wektor cech
        """
        gray = _to_gray(image).astype(np.float32)

        # 2D DFT
        F = np.fft.fft2(gray)

        # DC → centrum
        F_shifted = np.fft.fftshift(F)

        # Logarytmiczne widmo amplitudy
        magnitude = np.abs(F_shifted)
        log_spectrum = np.log1p(magnitude).astype(np.float32)

        # Standaryzacja do (fft_map_size × fft_map_size)
        if log_spectrum.shape != (self._fft_sz, self._fft_sz):
            log_spectrum = cv2.resize(
                log_spectrum,
                (self._fft_sz, self._fft_sz),
                interpolation=cv2.INTER_LINEAR,
            )

        return log_spectrum  # (fft_map_size, fft_map_size) float32

    # ── Połączony wektor cech ─────────────────────────────────────────────────

    def get_combined_features(self, image: np.ndarray) -> np.ndarray:
        """
        Połączony wektor cech fizycznych do klasyfikatora.

        Skład:
          - LBP histogram        :  59-D   (tekstura, artefakty druku)
          - FFT spectrum (flat)  : (fft_map_size²)-D  (siatka ekranu, mora)

        Domyślnie: 59 + 64*64 = 4155 wymiarów, dtype float32.

        Oba składniki są znormalizowane niezależnie (LBP: L1, FFT: min-max),
        dzięki czemu mają porównywalną skalę wartości ∈ [0, 1].

        Parametry
        ---------
        image : (H, W, 3) BGR uint8  lub  (H, W) uint8

        Zwraca
        ------
        features : (59 + fft_map_size², ) float32
        """
        lbp_hist = self.get_lbp_hist(image)  # (59,) ∈ [0,1]

        fft_map = self.get_fft_spectrum(image)  # (sz, sz) float32
        fft_norm = _minmax_normalize(fft_map).ravel()  # (sz²,) ∈ [0,1]

        return np.concatenate([lbp_hist, fft_norm], axis=0).astype(np.float32)  # (4155,)

    # ── Opisy wymiarów ────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        """Całkowita liczba wymiarów wektora get_combined_features()."""
        return self._n_bins + self._fft_sz * self._fft_sz

    @property
    def lbp_dim(self) -> int:
        return self._n_bins

    @property
    def fft_dim(self) -> int:
        return self._fft_sz * self._fft_sz


# ─── Wizualizacja ─────────────────────────────────────────────────────────────


def visualize_lbp(image: np.ndarray) -> np.ndarray:
    """
    Zwraca mapę LBP jako uint8 grayscale do podglądu.

    image : (H, W, 3) BGR uint8  lub  (H, W) uint8
    Zwraca: (H, W) uint8 — surowe wartości LBP przeskalowane do [0, 255]
    """
    gray = _to_gray(image)
    lbp_map = local_binary_pattern(gray, P=_LBP_POINTS, R=_LBP_RADIUS, method=_LBP_METHOD)
    lbp_u8 = cv2.normalize(lbp_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return lbp_u8


def visualize_fft(image: np.ndarray, map_size: int = _FFT_MAP_SIZE) -> np.ndarray:
    """
    Zwraca widmo FFT jako heatmapę BGR do podglądu w Gradio.

    Normalizuje do [0, 255] i nakłada colormap JET.

    image    : (H, W, 3) BGR uint8  lub  (H, W) uint8
    Zwraca   : (map_size, map_size, 3) BGR uint8
    """
    ext = PhysicalFeatureExtractor(fft_map_size=map_size)
    spectrum = ext.get_fft_spectrum(image)  # float32
    norm = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)  # (H, W, 3) BGR
    return heatmap


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _to_gray(image: np.ndarray) -> np.ndarray:
    """
    Konwertuje obraz do uint8 grayscale niezależnie od wejścia.

    Obsługuje: BGR (3-kanałowy), RGB (3-kanałowy), jednokanałowy, float.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Nieoczekiwany kształt obrazu: {image.shape}")

    if gray.dtype != np.uint8:
        # float [0,1] lub float [0,255] → uint8
        if gray.max() <= 1.0:
            gray = (gray * 255).clip(0, 255).astype(np.uint8)
        else:
            gray = gray.clip(0, 255).astype(np.uint8)

    return gray


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalizuje macierz do [0, 1] metodą min-max."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)
