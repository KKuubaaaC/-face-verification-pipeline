"""
Parser plików pair.list i img.list do zunifikowanego formatu CSV.

Format wejściowy pair.list: idx1 idx2 label
  - label 1 = genuine (ta sama osoba)
  - label 0 = impostor (różne osoby)

Format wyjściowy CSV: img1_path, img2_path, true_label
"""

import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "agedb": PROJECT_ROOT / "data" / "agedb",
    "lfw": PROJECT_ROOT / "data" / "lfw",
}

RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("parse_pairs")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_img_list(img_list_path: Path, logger: logging.Logger) -> dict[int, Path]:
    """Wczytuje img.list → {index: absolute_path}."""
    index_to_path: dict[int, Path] = {}
    with open(img_list_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # ścieżka może być absolutna lub relatywna względem PROJECT_ROOT
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            index_to_path[line_no] = candidate
    logger.debug("img.list: załadowano %d wpisów z %s", len(index_to_path), img_list_path)
    return index_to_path


def parse_pairs(
    dataset_name: str,
    dataset_dir: Path,
    logger: logging.Logger,
) -> tuple[list[dict], list[str]]:
    """
    Parsuje pair.list i img.list dla jednego datasetu.

    Zwraca:
        rows          — lista słowników gotowych do zapisu CSV
        missing_files — lista ścieżek nieistniejących na dysku
    """
    img_list_path = dataset_dir / "img.list"
    pair_list_path = dataset_dir / "pair.list"

    if not img_list_path.exists():
        logger.error("[%s] Brak pliku img.list: %s", dataset_name, img_list_path)
        return [], []
    if not pair_list_path.exists():
        logger.error("[%s] Brak pliku pair.list: %s", dataset_name, pair_list_path)
        return [], []

    index_to_path = load_img_list(img_list_path, logger)
    rows: list[dict] = []
    missing_files: list[str] = []

    genuine_count = 0
    impostor_count = 0
    bad_label = 0
    self_pair = 0

    with open(pair_list_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                logger.warning("[%s] Linia %d: nieprawidłowy format '%s' — pomijam", dataset_name, line_no, line)
                continue

            try:
                idx1, idx2, label = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                logger.warning("[%s] Linia %d: nie można sparsować jako int: '%s'", dataset_name, line_no, line)
                continue

            # walidacja etykiety
            if label not in (0, 1):
                logger.warning("[%s] Linia %d: nieznana etykieta %d", dataset_name, line_no, label)
                bad_label += 1
                continue

            # walidacja self-pair
            if idx1 == idx2:
                logger.warning("[%s] Linia %d: self-pair (idx=%d)", dataset_name, line_no, idx1)
                self_pair += 1
                continue

            # rozwiązanie ścieżek
            path1 = index_to_path.get(idx1)
            path2 = index_to_path.get(idx2)

            if path1 is None:
                logger.warning("[%s] Linia %d: idx1=%d nie istnieje w img.list", dataset_name, line_no, idx1)
                missing_files.append(f"[{dataset_name}] MISSING_INDEX idx1={idx1} (linia {line_no})")
                continue
            if path2 is None:
                logger.warning("[%s] Linia %d: idx2=%d nie istnieje w img.list", dataset_name, line_no, idx2)
                missing_files.append(f"[{dataset_name}] MISSING_INDEX idx2={idx2} (linia {line_no})")
                continue

            # walidacja istnienia pliku na dysku
            if not path1.exists():
                missing_files.append(str(path1))
                logger.debug("[%s] MISSING FILE: %s", dataset_name, path1)
            if not path2.exists():
                missing_files.append(str(path2))
                logger.debug("[%s] MISSING FILE: %s", dataset_name, path2)

            rows.append(
                {
                    "img1_path": str(path1),
                    "img2_path": str(path2),
                    "true_label": label,
                }
            )

            if label == 1:
                genuine_count += 1
            else:
                impostor_count += 1

    logger.info(
        "[%s] Sparsowano %d par: genuine=%d, impostor=%d | bad_label=%d, self_pair=%d | missing_files=%d",
        dataset_name,
        len(rows),
        genuine_count,
        impostor_count,
        bad_label,
        self_pair,
        len(missing_files),
    )
    return rows, missing_files


def save_csv(rows: list[dict], output_path: Path, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img1_path", "img2_path", "true_label"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Zapisano %d wierszy → %s", len(rows), output_path)


def save_missing_log(missing: list[str], log_path: Path, logger: logging.Logger) -> None:
    if not missing:
        logger.info("Brak brakujących plików — log nie zostanie zapisany.")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    unique_missing = sorted(set(missing))
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in unique_missing:
            f.write(entry + "\n")
    logger.warning("Zapisano %d brakujących ścieżek → %s", len(unique_missing), log_path)


def main() -> None:
    log_file = LOGS_DIR / "parse_pairs.log"
    logger = setup_logging(log_file)
    logger.info("START parse_pairs.py — PROJECT_ROOT=%s", PROJECT_ROOT)

    all_missing: list[str] = []

    for dataset_name, dataset_dir in DATASETS.items():
        logger.info("=== Dataset: %s ===", dataset_name)
        rows, missing = parse_pairs(dataset_name, dataset_dir, logger)

        if rows:
            output_csv = RESULTS_DIR / f"pairs_{dataset_name}.csv"
            save_csv(rows, output_csv, logger)

        all_missing.extend(missing)

    missing_log = LOGS_DIR / "missing_files.log"
    save_missing_log(all_missing, missing_log, logger)

    logger.info("DONE")


if __name__ == "__main__":
    main()
