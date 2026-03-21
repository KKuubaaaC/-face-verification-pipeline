"""
Static test: Face Verification + Anti-Spoofing na zdjęciach.

Użycie:
    uv run python src/static_test.py --probe ścieżka/do/zdjecia.jpg

Wyniki:
    - Czy twarz pasuje do bazy db/ja/ (DeepFace.find)
    - Czy zdjęcie jest live / spoof (DeepFace.analyze z anti-spoofing)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from deepface import DeepFace

logger = logging.getLogger(__name__)

DB_REAL = str(Path(__file__).parent.parent / "db" / "ja")
DB_ATTACK = str(Path(__file__).parent.parent / "db" / "attack")


def run(probe_path: str) -> None:
    probe = str(Path(probe_path).resolve())

    logger.info("\n%s", "=" * 55)
    logger.info("  Probe : %s", probe)
    logger.info("%s", "=" * 55)

    # ── 1. Weryfikacja tożsamości ─────────────────────────────
    logger.info("\n[1/2] Weryfikacja tożsamości (db/ja/)...")
    try:
        results: list[pd.DataFrame] = DeepFace.find(
            img_path=probe,
            db_path=DB_REAL,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False,
            silent=True,
        )
        df = results[0] if results else pd.DataFrame()
        if df.empty:
            logger.warning("  ✗ Nie znaleziono pasującej twarzy w db/ja/")
            verified = False
        else:
            best = df.iloc[0]
            dist = best.get("distance", best.iloc[-1])
            logger.info("  ✓ Dopasowanie: %s", Path(best["identity"]).name)
            logger.info("    dystans = %.4f", dist)
            verified = True
    except Exception as exc:
        logger.warning("  ✗ Błąd weryfikacji: %s", exc)
        verified = False

    # ── 2. Anti-Spoofing ──────────────────────────────────────
    logger.info("\n[2/2] Anti-Spoofing...")
    try:
        analysis = DeepFace.extract_faces(
            img_path=probe,
            anti_spoofing=True,
            detector_backend="retinaface",
            enforce_detection=False,
        )
        face = analysis[0]
        is_real = face.get("is_real", None)
        spoof_conf = face.get("antispoof_score", None)

        if is_real is True:
            label = "REAL ✓"
        elif is_real is False:
            label = "SPOOF ✗"
        else:
            label = "UNKNOWN"

        logger.info("  Wynik: %s", label)
        if spoof_conf is not None:
            logger.info("  Pewność: %.3f", spoof_conf)

    except Exception as exc:
        logger.warning("  ✗ Błąd anti-spoofing: %s", exc)
        is_real = None

    # ── Podsumowanie ──────────────────────────────────────────
    logger.info("\n%s", "─" * 55)
    status = "DOSTĘP PRZYZNANY ✓" if (verified and is_real) else "DOSTĘP ODMÓWIONY ✗"
    logger.info("  %s", status)
    logger.info("%s\n", "─" * 55)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, help="Ścieżka do testowanego zdjęcia")
    args = parser.parse_args()
    run(args.probe)
