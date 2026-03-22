"""
Batch face detection and alignment for LFW and AgeDB datasets.

Methodology (matching 01_detect_align.ipynb):
  - Landmark model: insightface 2d106det (106-point model from buffalo_l)
  - RetinaFace is NOT used — it achieves <1% detection on pre-cropped 112×112 images
  - Alignment: norm_crop() from insightface.utils.face_align using 5-point subset
    of 2d106det landmarks (indices [38, 88, 86, 52, 61] → eyes, nose, mouth corners)
  - _FaceProxy trick: treat full image as bbox so 2d106det localizes landmarks
    even on already-cropped inputs
  - MD5 deduplication: process unique images only, copy results for duplicates

Output:
  - Aligned crops → data/aligned/{lfw,agedb}/<basename>.jpg  (112×112 BGR)
  - detection_stats.csv → data/aligned/detection_stats.csv

Usage:
    uv run python -m src.vision.face_detector_batch \
        --lfw   "/path/to/lfw"   \
        --agedb "/path/to/agedb" \
        --out   "data/aligned"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 5-point indices within 2d106det 106-landmark set
# [left_eye, right_eye, nose_tip, mouth_left, mouth_right]
_KPS5_INDICES = [38, 88, 86, 52, 61]


# ─── Exceptions ──────────────────────────────────────────────────────────────


class MultipleFacesError(ValueError):
    """Raised when more than one face is detected in a single image."""

    def __init__(self, path: str, count: int) -> None:
        super().__init__(f"Detected {count} faces in image: {path}")
        self.path = path
        self.count = count


# ─── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class AlignResult:
    file_path: str
    face_detected: bool
    confidence_score: float  # det_score from 2d106det; 0.0 if not detected
    aligned_path: str | None = None
    error: str | None = None


# ─── Core detector ────────────────────────────────────────────────────────────


class FaceDetectorBatch:
    """
    Batch face detector / aligner based on insightface 2d106det.

    Loads models once at __init__; process_image() is stateless.
    """

    def __init__(self, model_pack: str = "buffalo_l") -> None:
        from insightface.app import FaceAnalysis  # noqa: PLC0415

        logger.info("Loading insightface models (%s)...", model_pack)
        self._app = FaceAnalysis(
            name=model_pack,
            allowed_modules=["detection", "landmark_2d_106"],
            providers=["CPUExecutionProvider"],
        )
        # det_size=(640,640) is standard; 2d106det runs on the full tensor so
        # 112×112 inputs still work after internal scaling.
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("FaceDetectorBatch ready.")

    # --- Public API ---

    def process_image(self, img_path: Path) -> tuple[np.ndarray | None, float]:
        """
        Detect face, align, return (aligned_crop, det_score).

        Raises MultipleFacesError if more than one face is found.
        Returns (None, 0.0) if no face.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        faces = self._app.get(img)

        if len(faces) > 1:
            raise MultipleFacesError(str(img_path), len(faces))

        if len(faces) == 0:
            # Fallback: treat whole image as face proxy
            aligned, score = self._proxy_align(img, img_path)
            return aligned, score

        face = faces[0]
        lm106 = getattr(face, "landmark_2d_106", None)
        if lm106 is None:
            logger.warning("Missing landmark_2d_106 for %s — using proxy.", img_path.name)
            return self._proxy_align(img, img_path)

        kps5 = lm106[_KPS5_INDICES].astype(np.float32)  # (5, 2)
        aligned = _norm_crop(img, kps5)
        return aligned, float(face.det_score)

    # --- Internal ---

    def _proxy_align(self, img: np.ndarray, img_path: Path) -> tuple[np.ndarray | None, float]:
        """
        _FaceProxy: treat the full image as a bbox and run landmark_2d_106 only.
        Works for pre-cropped 112×112 cells.
        """

        h, w = img.shape[:2]
        # Synthetic Face covering the whole frame
        import insightface.app.common as common  # noqa: PLC0415

        proxy = common.Face(
            bbox=np.array([0, 0, w - 1, h - 1], dtype=np.float32),
            kps=None,
            det_score=1.0,
            landmark_3d_68=None,
            pose=None,
            landmark_2d_106=None,
            gender=None,
            age=None,
            embedding=None,
        )

        lm_model = self._get_lm_model()
        if lm_model is None:
            logger.warning("2d106det model missing — returning resized crop.")
            resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            return resized, 0.0

        lm_model.get(img, proxy)
        lm106 = getattr(proxy, "landmark_2d_106", None)
        if lm106 is None:
            logger.warning("Proxy: no landmarks — returning resize.")
            resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            return resized, 0.0

        kps5 = lm106[_KPS5_INDICES].astype(np.float32)
        aligned = _norm_crop(img, kps5)
        return aligned, 0.0  # no det_score in proxy mode

    def _get_lm_model(self):
        """Return the 2d106det module from FaceAnalysis.models."""
        for model in self._app.models.values():
            if hasattr(model, "taskname") and "landmark_2d_106" in str(
                getattr(model, "taskname", "")
            ):
                return model
        # Fallback: search by module name
        for name, model in self._app.models.items():
            if "106" in str(name):
                return model
        return None


# ─── Alignment helper ────────────────────────────────────────────────────────


def _norm_crop(img: np.ndarray, kps5: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Thin wrapper around insightface.utils.face_align.norm_crop."""
    from insightface.utils.face_align import norm_crop  # noqa: PLC0415

    return norm_crop(img, kps5, image_size=image_size, mode="arcface")


# ─── MD5 deduplication ───────────────────────────────────────────────────────


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_dedup_map(
    paths: list[Path],
) -> tuple[list[Path], dict[Path, Path]]:
    """
    Return (unique_paths, duplicate_map).

    duplicate_map: {dup_path → canonical_path}
    """
    seen: dict[str, Path] = {}
    unique: list[Path] = []
    dups: dict[Path, Path] = {}

    for p in paths:
        h = _md5(p)
        if h in seen:
            dups[p] = seen[h]
        else:
            seen[h] = p
            unique.append(p)

    return unique, dups


# ─── Dataset loader ──────────────────────────────────────────────────────────


def _load_dataset(dataset_dir: Path) -> list[Path]:
    """
    Load image paths from img.list.

    Each line may be a long path; only the basename is used under <dataset_dir>/imgs/.
    """
    list_file = dataset_dir / "img.list"
    imgs_dir = dataset_dir / "imgs"

    if not list_file.exists():
        raise FileNotFoundError(f"Missing img.list in: {dataset_dir}")

    paths: list[Path] = []
    with open(list_file) as f:
        for line in f:
            name = Path(line.strip()).name
            p = imgs_dir / name
            if p.exists():
                paths.append(p)
            else:
                logger.warning("Missing file: %s", p)

    logger.info("Loaded %d paths from %s", len(paths), list_file)
    return paths


# --- Batch runner ---


def run_batch(
    datasets: dict[str, Path],
    out_dir: Path,
    *,
    stop_on_multiple_faces: bool = False,
) -> list[AlignResult]:
    """
    Process every image listed for the given datasets.

    datasets : {"lfw": Path(...), "agedb": Path(...)}
    out_dir  : output root (e.g. data/aligned)

    Returns AlignResult rows and writes detection_stats.csv.
    """
    detector = FaceDetectorBatch()
    all_results: list[AlignResult] = []

    for dataset_name, dataset_dir in datasets.items():
        logger.info("=== Dataset: %s ===", dataset_name)
        paths = _load_dataset(dataset_dir)

        unique_paths, dup_map = _build_dedup_map(paths)
        dup_count = len(dup_map)
        logger.info(
            "%s: %d images, %d unique, %d duplicates (%.1f%%)",
            dataset_name,
            len(paths),
            len(unique_paths),
            dup_count,
            100 * dup_count / max(len(paths), 1),
        )

        save_dir = out_dir / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- Process unique images ---
        unique_results: dict[Path, AlignResult] = {}

        for idx, img_path in enumerate(unique_paths, 1):
            if idx % 500 == 0:
                logger.info("  %s: %d / %d…", dataset_name, idx, len(unique_paths))

            rel = img_path.name
            out_p = save_dir / rel

            result = AlignResult(
                file_path=str(img_path),
                face_detected=False,
                confidence_score=0.0,
                aligned_path=None,
            )

            try:
                aligned, score = detector.process_image(img_path)
                if aligned is not None:
                    cv2.imwrite(str(out_p), aligned)
                    result.face_detected = True
                    result.confidence_score = score
                    result.aligned_path = str(out_p)
                else:
                    result.face_detected = False
                    result.confidence_score = 0.0

            except MultipleFacesError as exc:
                logger.warning("MultipleFaces: %s", exc)
                if stop_on_multiple_faces:
                    raise
                result.error = str(exc)

            except Exception as exc:
                logger.error("Processing error %s: %s", img_path, exc)
                result.error = str(exc)

            unique_results[img_path] = result
            all_results.append(result)

        # --- Copy aligned crops for duplicate hashes ---
        for dup_path, canonical_path in dup_map.items():
            canonical_result = unique_results.get(canonical_path)
            dup_out_p = save_dir / dup_path.name

            dup_result = AlignResult(
                file_path=str(dup_path),
                face_detected=canonical_result.face_detected if canonical_result else False,
                confidence_score=canonical_result.confidence_score if canonical_result else 0.0,
                aligned_path=None,
                error=canonical_result.error if canonical_result else "canonical not processed",
            )

            if canonical_result and canonical_result.aligned_path:
                shutil.copy2(canonical_result.aligned_path, dup_out_p)
                dup_result.aligned_path = str(dup_out_p)

            all_results.append(dup_result)

        detected = sum(1 for r in unique_results.values() if r.face_detected)
        logger.info(
            "%s: detected %d / %d unique (%.1f%%)",
            dataset_name,
            detected,
            len(unique_paths),
            100 * detected / max(len(unique_paths), 1),
        )

    # --- Write detection_stats.csv ---
    stats_path = out_dir / "detection_stats.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_path", "face_detected", "confidence_score", "aligned_path", "error"],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow(
                {
                    "file_path": r.file_path,
                    "face_detected": r.face_detected,
                    "confidence_score": f"{r.confidence_score:.4f}",
                    "aligned_path": r.aligned_path or "",
                    "error": r.error or "",
                }
            )

    logger.info("Wrote %d rows → %s", len(all_results), stats_path)
    return all_results


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch face detection + alignment for LFW / AgeDB")
    p.add_argument(
        "--lfw",
        type=Path,
        default=Path("/Users/jakub/Desktop/folder bez nazwy 2/lfw"),
        help="Path to LFW folder (contains img.list and imgs/)",
    )
    p.add_argument(
        "--agedb",
        type=Path,
        default=Path("/Users/jakub/Desktop/folder bez nazwy 2/agedb"),
        help="Path to AgeDB folder (contains img.list and imgs/)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/aligned"),
        help="Output directory for aligned crops and detection_stats.csv",
    )
    p.add_argument(
        "--stop-on-multiple-faces",
        action="store_true",
        help="Abort if >1 face is detected (default: log error and continue)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    datasets = {}
    if args.lfw.exists():
        datasets["lfw"] = args.lfw
    else:
        logger.warning("LFW folder missing: %s — skipping.", args.lfw)

    if args.agedb.exists():
        datasets["agedb"] = args.agedb
    else:
        logger.warning("AgeDB folder missing: %s — skipping.", args.agedb)

    if not datasets:
        logger.error("No datasets available. Pass --lfw / --agedb.")
        sys.exit(1)

    results = run_batch(
        datasets,
        args.out,
        stop_on_multiple_faces=args.stop_on_multiple_faces,
    )

    detected = sum(1 for r in results if r.face_detected)
    errors = sum(1 for r in results if r.error)
    print(
        f"\nSummary: {len(results)} images, "
        f"{detected} faces ({100 * detected / max(len(results), 1):.1f}%), "
        f"{errors} errors."
    )
    print(f"CSV: {args.out / 'detection_stats.csv'}")


if __name__ == "__main__":
    main()
