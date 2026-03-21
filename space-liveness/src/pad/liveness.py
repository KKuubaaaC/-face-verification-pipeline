"""
Presentation Attack Detection (PAD) module.

Dwie niezależne warstwy ochrony:
  1. BlinkDetector  — EAR (Eye Aspect Ratio) z 106-punktowych landmarków insightface.
                      Wykrywa mrugnięcie → obrona przed zdjęciami/wydrukami.
  2. MoireDetector  — FFT (Moiré) + LBP + Specular highlights z OpenCV.
                      Wykrywa ekrany telefonów/tabletów.

Fasada: PADPipeline.process_frame(landmarks, face_crop) → PADResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Indeksy landmarków (insightface 1k3d68 — standard iBUG 68-punktowy) ────
# Używamy landmark_3d_68 zamiast 2d106det — indeksy są oficjalnie udokumentowane.
# Prawe oko (z perspektywy obserwatora): 36–41
# Lewe  oko (z perspektywy obserwatora): 42–47
# Kolejność w każdej grupie: kąt_lewy, górna_zewn, górna_wewn,
#                             kąt_prawy, dolna_wewn, dolna_zewn
_RIGHT_EYE_IDX: tuple[int, ...] = (36, 37, 38, 39, 40, 41)
_LEFT_EYE_IDX: tuple[int, ...] = (42, 43, 44, 45, 46, 47)

# ─── Progi operacyjne ────────────────────────────────────────────────────────
EAR_BLINK_THRESHOLD: float = 0.20  # poniżej → oko zamknięte  (SKILLS.md §2)
EAR_CONSEC_FRAMES: int = 2  # min. klatek z zamkniętym okiem = 1 mrugnięcie

# FFT: znormalizowana energia poza centrum widma (0.0 = naturalne, 1.0 = ekran)
# Normalna kamera internetowa: ~0.60–0.68
# Twarz na ekranie telefonu/monitora: ~0.75–0.92 (siatka subpikseli + JPEG re-kompresja)
MOIRE_FFT_THRESHOLD: float = 0.72

# Timeout blink: jeśli twarz widoczna przez N klatek bez mrugnięcia → atak fotograficzny
# Przy ~12fps: 90 klatek ≈ 7.5 sekundy. Żywa osoba ZAWSZE mrugnęła w ciągu 10s.
BLINK_TIMEOUT_FRAMES: int = 90
# Promień wewnętrzny filtra dolnoprzepustowego (% min. wymiaru obrazu)
_FFT_LOWPASS_RATIO: float = 0.10

# LBP: wariancja histogramu (×10⁶) — skóra niska, pikselowa siatka ekranu wysoka
# Kamery internetowe kompresują obraz co podnosi LBP baseline do ~150–200
LBP_VAR_THRESHOLD: float = 350.0

# Specular: max. udział prześwietlonych pikseli (V > 240 w HSV) na obszarze twarzy
SPECULAR_RATIO_MAX: float = 0.20


# ─── Struktury danych ────────────────────────────────────────────────────────


@dataclass
class PADResult:
    liveness_passed: bool
    blink_detected: bool
    blink_timed_out: bool  # True gdy twarz widoczna >BLINK_TIMEOUT_FRAMES bez mrugnięcia
    ear_current: float  # aktualny EAR na tej klatce
    moire_score: float  # [0, 1] — wyższy → bardziej podejrzane
    lbp_variance: float
    specular_ratio: float
    reason: str = ""  # czytelny powód odrzucenia (pusty gdy OK)


# ─── Moduł 1: BlinkDetector ──────────────────────────────────────────────────


class BlinkDetector:
    """
    Śledzi stan powiek na kolejnych klatkach i rejestruje mrugnięcie.

    Stan jest kumulatywny w obrębie sesji weryfikacji — wywołaj reset()
    na początku każdej nowej sesji.
    """

    def __init__(
        self,
        ear_threshold: float = EAR_BLINK_THRESHOLD,
        consec_frames: int = EAR_CONSEC_FRAMES,
        timeout_frames: int = BLINK_TIMEOUT_FRAMES,
    ) -> None:
        self._threshold = ear_threshold
        self._consec = consec_frames
        self._timeout = timeout_frames
        self._counter: int = 0
        self._total_frames: int = 0  # klatki z wykrytą twarzą
        self._blink_detected: bool = False

    # ── API publiczne ──────────────────────────────────────────────────────

    def reset(self) -> None:
        self._counter = 0
        self._total_frames = 0
        self._blink_detected = False

    @property
    def blink_detected(self) -> bool:
        return self._blink_detected

    @property
    def timed_out(self) -> bool:
        """True gdy twarz widoczna przez >timeout_frames bez ani jednego mrugnięcia."""
        return (not self._blink_detected) and (self._total_frames >= self._timeout)

    def update(self, landmarks: np.ndarray) -> float:
        """
        Aktualizuje stan automatu na podstawie landmarków jednej klatki.

        landmarks: (68, 3) float — face.landmark_3d_68 z insightface
        Zwraca: aktualny avg_EAR (przydatny do debugowania w UI)
        """
        self._total_frames += 1
        ear = self._avg_ear(landmarks)

        if ear < self._threshold:
            self._counter += 1
        else:
            if self._counter >= self._consec:
                self._blink_detected = True
                logger.debug(
                    "Mrugnięcie wykryte (EAR=%.3f, klatki=%d)",
                    ear,
                    self._counter,
                )
            self._counter = 0

        return ear

    # ── Obliczenia EAR ────────────────────────────────────────────────────

    def _avg_ear(self, lm: np.ndarray) -> float:
        ear_l = _eye_aspect_ratio(lm, _LEFT_EYE_IDX)
        ear_r = _eye_aspect_ratio(lm, _RIGHT_EYE_IDX)
        return (ear_l + ear_r) / 2.0


# ─── Moduł 2: MoireDetector ──────────────────────────────────────────────────


class MoireDetector:
    """
    Trójwarstwowa analiza obrazu twarzy (wyłącznie OpenCV + NumPy).

    Bezstanowy — każde wywołanie analyze() jest niezależne.
    """

    def analyze(
        self,
        face_crop: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        face_crop: BGR obraz wyciętej twarzy (np. 112×112 z aligned_crop).
        Zwraca: (moire_score, lbp_variance, specular_ratio)
        """
        if face_crop is None or face_crop.size == 0:
            logger.warning("MoireDetector.analyze: pusty face_crop.")
            return 0.0, 0.0, 0.0

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        moire = self._fft_moire_score(gray)
        lbp_var = self._lbp_variance(gray)
        specular = self._specular_ratio(face_crop)

        logger.debug(
            "PAD scores — moiré=%.3f  lbp_var=%.1f  specular=%.3f",
            moire,
            lbp_var,
            specular,
        )
        return moire, lbp_var, specular

    # ── A. FFT Moiré ──────────────────────────────────────────────────────

    @staticmethod
    def _fft_moire_score(gray: np.ndarray) -> float:
        """
        Energia w wysokich częstotliwościach / energia całkowita.

        Ekrany cyfrowe mają regularną siatkę pikseli, która przy nagraniu
        kamerą tworzy efekt Moiré — nienaturalne piki poza centrum widma FFT.

        Algorytm:
          1. DFT → przesunięcie centrum → moduł widma amplitudowego
          2. Maska kołowa: odrzuca centrum (niskie częstotliwości, DC)
          3. Stosunek energii poza centrum do energii całkowitej
        """
        f32 = gray.astype(np.float32)

        # DFT przez OpenCV (szybsze niż np.fft dla małych obrazów)
        dft = cv2.dft(f32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)  # DC → centrum
        mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        h, w = mag.shape
        cy, cx = h // 2, w // 2
        r_inner = int(min(h, w) * _FFT_LOWPASS_RATIO)

        # Maska kołowa: True poza promieniem r_inner (wysokie częstotliwości)
        y_idx, x_idx = np.ogrid[:h, :w]
        dist_sq = (y_idx - cy) ** 2 + (x_idx - cx) ** 2
        high_freq_mask = dist_sq > r_inner**2

        total_energy = float(mag.sum()) + 1e-9
        high_energy = float(mag[high_freq_mask].sum())

        return high_energy / total_energy

    # ── B. LBP (Local Binary Patterns) ───────────────────────────────────

    @staticmethod
    def _lbp_variance(gray: np.ndarray) -> float:
        """
        Wariancja histogramu LBP — implementacja przez morfologię OpenCV.

        Skóra → płynna tekstura → niski, szeroki histogram.
        Siatka pikseli ekranu → powtarzalny wzór → wysoka wariancja histogramu.

        Uproszczony LBP 3×3 bez scikit-image:
          - Dla każdego piksela porównujemy 8 sąsiadów z centrum
          - Budujemy wartość 8-bitową (0–255)
          - Histogram tej wartości jest sygnaturą tekstury
        """
        # Przesunięcia 8 sąsiadów wokół centrum piksela
        _neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        h, w = gray.astype(np.float32).shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        center = gray.astype(np.int16)

        for bit, (dy, dx) in enumerate(_neighbors):
            shifted = np.roll(np.roll(center, dy, axis=0), dx, axis=1)
            lbp |= (shifted >= center).astype(np.uint8) << bit

        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / (hist.sum() + 1e-9)
        # Skalujemy ×10⁶ aby uzyskać czytelny zakres liczbowy
        return float(np.var(hist_norm) * 1e6)

    # ── C. Specular Highlights ────────────────────────────────────────────

    @staticmethod
    def _specular_ratio(bgr: np.ndarray) -> float:
        """
        Udział prześwietlonych pikseli (V > 240) w obszarze twarzy.

        Ekrany szklane odbijają światło nienaturalnie intensywnie.
        Kanał V (Value) w przestrzeni HSV wykrywa te blaski.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        bright_pixels = int(np.sum(v_channel > 240))
        return bright_pixels / float(v_channel.size)


# ─── Fasada: PADPipeline ─────────────────────────────────────────────────────


class PADPipeline:
    """
    Główny punkt wejścia dla logiki PAD.

    Użycie typowe:
        pad = PADPipeline()
        pad.reset()                              # na początku każdej sesji
        while streaming:
            result = pad.process_frame(landmarks, face_crop)
            if result.liveness_passed:
                # kontynuuj weryfikację tożsamości
    """

    def __init__(
        self,
        ear_threshold: float = EAR_BLINK_THRESHOLD,
        ear_consec_frames: int = EAR_CONSEC_FRAMES,
        moire_threshold: float = MOIRE_FFT_THRESHOLD,
        lbp_threshold: float = LBP_VAR_THRESHOLD,
        specular_threshold: float = SPECULAR_RATIO_MAX,
    ) -> None:
        self._blink = BlinkDetector(
            ear_threshold=ear_threshold,
            consec_frames=ear_consec_frames,
        )
        self._moire = MoireDetector()
        self._moire_t = moire_threshold
        self._lbp_t = lbp_threshold
        self._spec_t = specular_threshold

    def reset(self) -> None:
        """Resetuje stan automatu mrugania. Wywołaj przed każdą nową sesją."""
        self._blink.reset()

    def process_frame(
        self,
        landmarks: np.ndarray,
        face_crop: np.ndarray,
    ) -> PADResult:
        """
        Przetwarza jedną klatkę z kamery.

        landmarks : (68, 3) float — face.landmark_3d_68 (iBUG 68-punktowy)
        face_crop : (H, W, 3) BGR  — wyciętý crop twarzy (np. aligned_crop)
        """
        # Warstwa 1: EAR / blink + timeout
        ear = self._blink.update(landmarks)
        timed_out = self._blink.timed_out

        # Warstwa 2: analiza tekstury / ekranu
        moire, lbp_var, specular = self._moire.analyze(face_crop)

        # Decyzja o ataku ekranowym (wystarczy jeden pozytywny sygnał)
        screen_attack = moire > self._moire_t or lbp_var > self._lbp_t or specular > self._spec_t

        # Liveness OK tylko gdy: mrugnięcie wykryte ORAZ brak ataku ekranowego ORAZ brak timeout
        liveness_ok = self._blink.blink_detected and not screen_attack and not timed_out

        reason = _build_reason(
            blink_ok=self._blink.blink_detected,
            timed_out=timed_out,
            moire=moire,
            moire_t=self._moire_t,
            lbp_var=lbp_var,
            lbp_t=self._lbp_t,
            specular=specular,
            spec_t=self._spec_t,
        )

        return PADResult(
            liveness_passed=liveness_ok,
            blink_detected=self._blink.blink_detected,
            blink_timed_out=timed_out,
            ear_current=ear,
            moire_score=moire,
            lbp_variance=lbp_var,
            specular_ratio=specular,
            reason=reason,
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _eye_aspect_ratio(lm: np.ndarray, idx: tuple[int, ...]) -> float:
    """
    EAR dla jednego oka — wzór Soukupová & Čech (2016).

    EAR = (||P1–P5|| + ||P2–P4||) / (2 · ||P0–P3||)

    Mapowanie dla iBUG 68-punktowego (prawe: 36-41, lewe: 42-47):
        idx[0] = kąt lewy       idx[3] = kąt prawy
        idx[1] = górna zewn.    idx[4] = dolna zewn.
        idx[2] = górna wewn.    idx[5] = dolna wewn.

    lm  : (68, 3) lub (68, 2) array — używamy tylko X i Y (kolumny 0, 1)
    idx : 6-elementowa krotka indeksów
    """
    p = lm[list(idx), :2]  # (6, 2) — bierzemy tylko X, Y (pomijamy Z)

    vert_a = float(np.linalg.norm(p[1] - p[5]))
    vert_b = float(np.linalg.norm(p[2] - p[4]))
    horiz = float(np.linalg.norm(p[0] - p[3]))

    if horiz < 1e-6:
        logger.warning("_eye_aspect_ratio: zerowa szerokość oka — zwracam 0.0")
        return 0.0

    return (vert_a + vert_b) / (2.0 * horiz)


def _build_reason(
    *,
    blink_ok: bool,
    timed_out: bool,
    moire: float,
    moire_t: float,
    lbp_var: float,
    lbp_t: float,
    specular: float,
    spec_t: float,
) -> str:
    """Buduje czytelny komunikat o przyczynie odrzucenia (lub pusty string gdy OK)."""
    if timed_out:
        return "✗ ATAK WYKRYTY: brak mrugnięcia przez >7s — prawdopodobnie zdjęcie lub nagranie."
    if not blink_ok:
        return "Proszę mrugnąć do kamery."

    issues: list[str] = []
    if moire > moire_t:
        issues.append(f"efekt Moiré (FFT={moire:.3f} > {moire_t})")
    if lbp_var > lbp_t:
        issues.append(f"podejrzana tekstura (LBP={lbp_var:.1f} > {lbp_t})")
    if specular > spec_t:
        issues.append(f"odbicia ekranu (specular={specular:.3f} > {spec_t})")

    if issues:
        return "Wykryto atak prezentacyjny: " + "; ".join(issues)

    return ""
