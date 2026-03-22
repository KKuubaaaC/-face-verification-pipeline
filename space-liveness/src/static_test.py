"""
Static test: face verification + anti-spoofing on disk images.

Usage:
    uv run python src/static_test.py --probe path/to/image.jpg

Output:
    - Whether the face matches gallery db/ja/ (DeepFace.find)
    - Whether the image looks live vs spoof (DeepFace with anti-spoofing)
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

    # --- 1. Identity vs db/ja/ ---
    print("\n[1/2] Identity verification (db/ja/)...")
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
            print("  No matching face in db/ja/")
            verified = False
        else:
            best = df.iloc[0]
            dist = best.get("distance", best.iloc[-1])
            print(f"  Match: {Path(best['identity']).name}")
            print(f"    distance = {dist:.4f}")
            verified = True
    except Exception as exc:
        print(f"  Verification error: {exc}")
        verified = False

    # --- 2. Anti-spoofing ---
    print("\n[2/2] Anti-spoofing...")
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
            label = "REAL"
        elif is_real is False:
            label = "SPOOF"
        else:
            label = "UNKNOWN"

        print(f"  Result: {label}")
        if spoof_conf is not None:
            print(f"  Confidence: {spoof_conf:.3f}")

    except Exception as exc:
        print(f"  Anti-spoof error: {exc}")
        is_real = None

    # --- Summary ---
    print(f"\n{'─' * 55}")
    status = "ACCESS GRANTED" if (verified and is_real) else "ACCESS DENIED"
    print(f"  {status}")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, help="Path to probe image")
    args = parser.parse_args()
    run(args.probe)
