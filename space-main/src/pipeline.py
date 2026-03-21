"""
VerificationPipeline — główna fasada systemu weryfikacji twarzy.

Kolejność kroków dla każdej klatki z kamery:
  1. Detekcja twarzy + alignment (FaceDetector → DetectedFace)
  2. PAD / Liveness check (PADPipeline → PADResult)
  3. Ekstrakcja embeddingu z aligned crop (FaceEmbedder → np.ndarray)
  4. Porównanie z wektorem referencyjnym (FaceEmbedder.verify → VerificationResult)

Weryfikacja tożsamości (krok 4) uruchamiana jest TYLKO gdy liveness_passed == True.
Embedding referencyjny rejestrujemy raz przez register_reference() ze zdjęcia dokumentu.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from src.pad.liveness import PADPipeline, PADResult
from src.vision.detector import DetectedFace, FaceDetector
from src.vision.embedder import VERIFICATION_THRESHOLD, FaceEmbedder, VerificationResult

logger = logging.getLogger(__name__)


# ─── Stany sesji ─────────────────────────────────────────────────────────────


class SessionState(Enum):
    """Etapy życia jednej sesji weryfikacji."""

    WAITING_FOR_REFERENCE = auto()  # brak wektora referencyjnego
    WAITING_FOR_BLINK = auto()  # referencja zarejestrowana, czeka na mrugnięcie
    VERIFIED = auto()  # liveness OK + MATCH
    REJECTED_LIVENESS = auto()  # liveness nie przeszedł
    REJECTED_IDENTITY = auto()  # liveness OK, ale NO MATCH


# ─── Wynik przetworzenia klatki ───────────────────────────────────────────────


@dataclass
class FrameResult:
    """Zagregowany wynik dla jednej klatki z kamery."""

    state: SessionState

    # Czy twarz w ogóle wykryta na klatce
    face_detected: bool = False

    # Wynik PAD (None gdy brak twarzy)
    pad: PADResult | None = None

    # Wynik weryfikacji (None gdy liveness nie przeszedł lub brak referencji)
    verification: VerificationResult | None = None

    # Aktualny EAR — do wyświetlenia w UI nawet gdy brak wyniku weryfikacji
    ear: float = 0.0

    # Czytelny komunikat dla Gradio UI
    message: str = ""


# ─── Główna klasa ─────────────────────────────────────────────────────────────


class VerificationPipeline:
    """
    Bezstanowy względem modeli (modele ładowane raz w __init__),
    stanowy względem sesji (reset_session() czyści stan między użytkownikami).

    Typowy flow:
        pipeline = VerificationPipeline()
        pipeline.register_reference(reference_image_bgr)

        pipeline.reset_session()
        for frame in camera_stream:
            result = pipeline.process_frame(frame)
            if result.state == SessionState.VERIFIED:
                break
    """

    def __init__(
        self,
        verification_threshold: float = VERIFICATION_THRESHOLD,
        providers: list[str] | None = None,
    ) -> None:
        _providers = providers or ["CPUExecutionProvider"]

        logger.info("Inicjalizacja VerificationPipeline...")
        self._detector = FaceDetector(providers=_providers)

        # Embedder współdzieli FaceAnalysis z detektorem — jeden zestaw modeli
        self._embedder = FaceEmbedder(_app=self._detector._app)

        self._pad = PADPipeline()
        self._threshold = verification_threshold

        # Stan sesji
        self._reference_embedding: np.ndarray | None = None
        self._session_state: SessionState = SessionState.WAITING_FOR_REFERENCE

        logger.info("VerificationPipeline gotowy.")

    # ── Rejestracja referencji ─────────────────────────────────────────────

    def register_reference(self, reference_image: np.ndarray) -> bool:
        """
        Ekstrahuje i zapisuje embedding ze zdjęcia referencyjnego (np. dokumentu).

        reference_image: BGR image (dowolny rozmiar — detekcja wykona alignment)
        Zwraca True gdy referencja zarejestrowana pomyślnie, False gdy brak twarzy.
        """
        face = self._detector.get_largest_face(reference_image)
        if face is None:
            logger.warning("register_reference: nie wykryto twarzy na zdjęciu referencyjnym.")
            return False

        try:
            self._reference_embedding = self._embedder.embed(face.aligned_crop)
        except ValueError as exc:
            logger.error("register_reference: błąd embeddingu — %s", exc)
            return False

        logger.info("Referencja zarejestrowana (det_score=%.3f).", face.det_score)

        if self._session_state == SessionState.WAITING_FOR_REFERENCE:
            self._session_state = SessionState.WAITING_FOR_BLINK

        return True

    # ── Zarządzanie sesją ──────────────────────────────────────────────────

    def reset_session(self) -> None:
        """
        Resetuje stan sesji PAD (licznik mrugnięć, stan automatu).
        NIE kasuje wektora referencyjnego — ten pozostaje do następnego
        wywołania register_reference().
        """
        self._pad.reset()
        if self._reference_embedding is not None:
            self._session_state = SessionState.WAITING_FOR_BLINK
        else:
            self._session_state = SessionState.WAITING_FOR_REFERENCE
        logger.debug("Sesja zresetowana.")

    @property
    def state(self) -> SessionState:
        return self._session_state

    @property
    def has_reference(self) -> bool:
        return self._reference_embedding is not None

    # ── Przetwarzanie klatki ───────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Przetwarza jedną klatkę z kamery.

        frame: BGR image z kamery internetowej
        Zwraca FrameResult z aktualnym stanem sesji i metrykami diagnostycznymi.

        Gdy sesja już zakończona (VERIFIED / REJECTED_*), kolejne klatki są
        ignorowane i zwracany jest wynik końcowy bez ponownych obliczeń.
        """
        # Sesja już zakończona — nie przetwarzaj dalej
        if self._session_state in (
            SessionState.VERIFIED,
            SessionState.REJECTED_LIVENESS,
            SessionState.REJECTED_IDENTITY,
        ):
            return FrameResult(
                state=self._session_state,
                message=_state_message(self._session_state),
            )

        # Brak referencji
        if self._reference_embedding is None:
            return FrameResult(
                state=SessionState.WAITING_FOR_REFERENCE,
                message="Najpierw zarejestruj zdjęcie referencyjne.",
            )

        # ── Krok 1: detekcja ──────────────────────────────────────────────
        face: DetectedFace | None = self._detector.get_largest_face(frame)

        if face is None:
            return FrameResult(
                state=self._session_state,
                face_detected=False,
                message="Nie wykryto twarzy — ustaw twarz w kadrze.",
            )

        # ── Krok 2: PAD / liveness ────────────────────────────────────────
        pad_result: PADResult = self._pad.process_frame(
            landmarks=face.landmarks_68,  # iBUG 68-pkt — pewne indeksy oczu
            face_crop=face.aligned_crop,
        )

        # Liveness nie przeszedł — sprawdź czy to timeout (atak foto) czy tylko czekamy
        if not pad_result.liveness_passed:
            if pad_result.blink_timed_out:
                self._session_state = SessionState.REJECTED_LIVENESS
                logger.warning("Sesja odrzucona: timeout mrugnięcia — atak fotograficzny.")
            return FrameResult(
                state=self._session_state,
                face_detected=True,
                pad=pad_result,
                ear=pad_result.ear_current,
                message=pad_result.reason or "Proszę mrugnąć do kamery.",
            )

        # ── Krok 3: embedding probe ───────────────────────────────────────
        try:
            probe_embedding = self._embedder.embed(face.aligned_crop)
        except ValueError as exc:
            logger.error("process_frame: błąd embeddingu probe — %s", exc)
            return FrameResult(
                state=self._session_state,
                face_detected=True,
                pad=pad_result,
                ear=pad_result.ear_current,
                message="Błąd ekstrakcji cech twarzy — spróbuj ponownie.",
            )

        # ── Krok 4: weryfikacja tożsamości ────────────────────────────────
        verification: VerificationResult = self._embedder.verify(
            embedding_probe=probe_embedding,
            embedding_reference=self._reference_embedding,
            threshold=self._threshold,
        )

        if verification.is_match:
            self._session_state = SessionState.VERIFIED
            logger.info(
                "Weryfikacja POZYTYWNA (d=%.4f ≤ %.4f).",
                verification.cosine_distance,
                self._threshold,
            )
        else:
            self._session_state = SessionState.REJECTED_IDENTITY
            logger.info(
                "Weryfikacja NEGATYWNA (d=%.4f > %.4f).",
                verification.cosine_distance,
                self._threshold,
            )

        return FrameResult(
            state=self._session_state,
            face_detected=True,
            pad=pad_result,
            verification=verification,
            ear=pad_result.ear_current,
            message=_verification_message(verification),
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _state_message(state: SessionState) -> str:
    return {
        SessionState.VERIFIED: "✓ Weryfikacja zakończona pomyślnie.",
        SessionState.REJECTED_LIVENESS: "✗ Odrzucono: wykryto atak prezentacyjny.",
        SessionState.REJECTED_IDENTITY: "✗ Odrzucono: tożsamość nie potwierdzona.",
        SessionState.WAITING_FOR_BLINK: "Proszę mrugnąć do kamery.",
        SessionState.WAITING_FOR_REFERENCE: "Brak zdjęcia referencyjnego.",
    }.get(state, "")


def _verification_message(result: VerificationResult) -> str:
    if result.is_match:
        return f"✓ MATCH — dystans kosinusowy: {result.cosine_distance:.4f}"
    return f"✗ NO MATCH — dystans kosinusowy: {result.cosine_distance:.4f} (próg: {result.threshold})"
