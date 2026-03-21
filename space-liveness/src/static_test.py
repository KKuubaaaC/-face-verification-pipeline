"""
Static test: Face Verification + Anti-Spoofing na zdjęciach.

Użycie:
    uv run python src/static_test.py --probe ścieżka/do/zdjecia.jpg

Wyniki:
    - Czy twarz pasuje do bazy db/ja/ (DeepFace.find)
    - Czy zdjęcie jest live / spoof (DeepFace.analyze z anti-spoofing)
"""

import argparse
from pathlib import Path

import pandas as pd
from deepface import DeepFace

DB_REAL = str(Path(__file__).parent.parent / "db" / "ja")
DB_ATTACK = str(Path(__file__).parent.parent / "db" / "attack")


def run(probe_path: str) -> None:
    probe = str(Path(probe_path).resolve())

    print(f"\n{'=' * 55}")
    print(f"  Probe : {probe}")
    print(f"{'=' * 55}")

    # ── 1. Weryfikacja tożsamości ─────────────────────────────
    print("\n[1/2] Weryfikacja tożsamości (db/ja/)...")
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
            print("  ✗ Nie znaleziono pasującej twarzy w db/ja/")
            verified = False
        else:
            best = df.iloc[0]
            dist = best.get("distance", best.iloc[-1])
            print(f"  ✓ Dopasowanie: {Path(best['identity']).name}")
            print(f"    dystans = {dist:.4f}")
            verified = True
    except Exception as exc:
        print(f"  ✗ Błąd weryfikacji: {exc}")
        verified = False

    # ── 2. Anti-Spoofing ──────────────────────────────────────
    print("\n[2/2] Anti-Spoofing...")
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

        print(f"  Wynik: {label}")
        if spoof_conf is not None:
            print(f"  Pewność: {spoof_conf:.3f}")

    except Exception as exc:
        print(f"  ✗ Błąd anti-spoofing: {exc}")
        is_real = None

    # ── Podsumowanie ──────────────────────────────────────────
    print(f"\n{'─' * 55}")
    status = "DOSTĘP PRZYZNANY ✓" if (verified and is_real) else "DOSTĘP ODMÓWIONY ✗"
    print(f"  {status}")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, help="Ścieżka do testowanego zdjęcia")
    args = parser.parse_args()
    run(args.probe)
