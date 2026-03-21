"""
Live cam: Face Verification + Anti-Spoofing z kamery w czasie rzeczywistym.

Użycie:
    uv run python src/live_cam.py

Sterowanie:
    q / ESC  — wyjście
    s        — zrzut ekranu do debug/

Logika (co ~1 sekundę):
    1. extract_faces(anti_spoofing=True)  → is_real / spoof
    2. Jeśli REAL → DeepFace.find() w db/ja/
    3. Nakłada wynik na klatkę (bbox + etykiety)
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

DB_REAL = str(Path(__file__).parent.parent / "db" / "ja")
DEBUG_DIR = Path(__file__).parent.parent / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

# Kolor BGR
_GREEN = (50, 200, 50)
_RED = (50, 50, 200)
_ORANGE = (30, 140, 220)
_WHITE = (230, 230, 230)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Minimalny interwał między kolejnymi wywołaniami DeepFace [s]
ANALYZE_INTERVAL = 1.0


def _draw_overlay(
    frame: np.ndarray,
    label: str,
    color: tuple,
    sub: str = "",
    region: dict | None = None,
) -> None:
    """Rysuje bbox + etykietę na klatce (in-place)."""
    h, w = frame.shape[:2]

    if region:
        x, y, rw, rh = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 2)

    # Pasek statusu na dole
    cv2.rectangle(frame, (0, h - 52), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, label, (10, h - 28), _FONT, 0.8, color, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (10, h - 8), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Nie można otworzyć kamery (index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Kamera uruchomiona. Naciśnij 'q' lub ESC aby wyjść, 's' aby zapisać zrzut.")

    last_time = 0.0
    last_label = "Inicjalizacja..."
    last_color = _WHITE
    last_sub = ""
    last_region = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        # ── Analiza co ANALYZE_INTERVAL sekund ───────────────
        if now - last_time >= ANALYZE_INTERVAL:
            last_time = now
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    anti_spoofing=True,
                    detector_backend="opencv",
                    enforce_detection=False,
                )
            except Exception as exc:
                last_label = f"Błąd: {exc}"
                last_color = _WHITE
                last_sub = ""
                last_region = None
                faces = []

            if not faces:
                last_label = "Brak twarzy"
                last_color = _WHITE
                last_sub = ""
                last_region = None
            else:
                face = faces[0]
                is_real = face.get("is_real", None)
                score = face.get("antispoof_score", 0.0)
                region = face.get("facial_area", None)

                if is_real is False:
                    last_label = "SPOOF ✗"
                    last_color = _RED
                    last_sub = f"antispoof_score={score:.3f}"
                    last_region = region
                elif is_real is True:
                    # Weryfikacja tożsamości
                    try:
                        res: list[pd.DataFrame] = DeepFace.find(
                            img_path=frame,
                            db_path=DB_REAL,
                            model_name="Facenet512",
                            detector_backend="opencv",
                            enforce_detection=False,
                            silent=True,
                        )
                        df = res[0] if res else pd.DataFrame()
                        if df.empty:
                            last_label = "REAL — Nieznana osoba"
                            last_color = _ORANGE
                            last_sub = f"Brak dopasowania w db/ja/  score={score:.3f}"
                        else:
                            best = df.iloc[0]
                            name = Path(best["identity"]).stem
                            dist = best.get("distance", best.iloc[-1])
                            last_label = f"REAL — {name} ✓"
                            last_color = _GREEN
                            last_sub = f"dystans={dist:.4f}  antispoof={score:.3f}"
                    except Exception as exc:
                        last_label = "REAL — błąd weryfikacji"
                        last_color = _ORANGE
                        last_sub = str(exc)[:60]
                    last_region = region
                else:
                    last_label = "Analiza..."
                    last_color = _WHITE
                    last_sub = ""
                    last_region = region

        # ── Nakładanie overlay ─────────────────────────────────
        display = frame.copy()
        _draw_overlay(display, last_label, last_color, last_sub, last_region)
        cv2.imshow("Face Verification + Anti-Spoofing  [q=wyjście]", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q lub ESC
            break
        if key == ord("s"):
            path = DEBUG_DIR / f"snapshot_{int(now)}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Zapisano: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Koniec.")


if __name__ == "__main__":
    main()
