"""
Live camera: face verification + anti-spoofing in real time.

Usage:
    uv run python src/live_cam.py

Keys:
    q / ESC  — quit
    s        — save snapshot to debug/

Roughly every second:
    1. extract_faces(anti_spoofing=True)  -> is_real / spoof
    2. If REAL -> DeepFace.find() in db/ja/
    3. Draw bbox + labels on frame
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

# BGR colors
_GREEN = (50, 200, 50)
_RED = (50, 50, 200)
_ORANGE = (30, 140, 220)
_WHITE = (230, 230, 230)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Minimum interval between DeepFace calls [s]
ANALYZE_INTERVAL = 1.0


def _draw_overlay(
    frame: np.ndarray,
    label: str,
    color: tuple,
    sub: str = "",
    region: dict | None = None,
) -> None:
    """Draw bbox + label on frame (in-place)."""
    h, w = frame.shape[:2]

    if region:
        x, y, rw, rh = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 2)

    cv2.rectangle(frame, (0, h - 52), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, label, (10, h - 28), _FONT, 0.8, color, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (10, h - 8), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera (index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera started. Press 'q' or ESC to quit, 's' for snapshot.")

    last_time = 0.0
    last_label = "Initializing..."
    last_color = _WHITE
    last_sub = ""
    last_region = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

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
                last_label = f"Error: {exc}"
                last_color = _WHITE
                last_sub = ""
                last_region = None
                faces = []

            if not faces:
                last_label = "No face"
                last_color = _WHITE
                last_sub = ""
                last_region = None
            else:
                face = faces[0]
                is_real = face.get("is_real", None)
                score = face.get("antispoof_score", 0.0)
                region = face.get("facial_area", None)

                if is_real is False:
                    last_label = "SPOOF"
                    last_color = _RED
                    last_sub = f"antispoof_score={score:.3f}"
                    last_region = region
                elif is_real is True:
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
                            last_label = "REAL — unknown identity"
                            last_color = _ORANGE
                            last_sub = f"No match in db/ja/  score={score:.3f}"
                        else:
                            best = df.iloc[0]
                            name = Path(best["identity"]).stem
                            dist = best.get("distance", best.iloc[-1])
                            last_label = f"REAL — {name}"
                            last_color = _GREEN
                            last_sub = f"distance={dist:.4f}  antispoof={score:.3f}"
                    except Exception as exc:
                        last_label = "REAL — verification error"
                        last_color = _ORANGE
                        last_sub = str(exc)[:60]
                    last_region = region
                else:
                    last_label = "Analyzing..."
                    last_color = _WHITE
                    last_sub = ""
                    last_region = region

        display = frame.copy()
        _draw_overlay(display, last_label, last_color, last_sub, last_region)
        cv2.imshow("Face verification + anti-spoofing  [q=quit]", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            path = DEBUG_DIR / f"snapshot_{int(now)}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
