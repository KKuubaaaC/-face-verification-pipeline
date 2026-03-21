#!/usr/bin/env bash
# Clone or update upstream SwinFace (https://github.com/lxq1000/SwinFace) into
# every path this repo expects: repo root and space-main (HF Space build context).
#
# If you already have a local copy of swinface_project/ (without .git), this script
# leaves it untouched — only empty paths or existing git clones are filled/updated.
#
# Usage:
#   bash scripts/bootstrap_swinface.sh
# Optional:
#   SWINFACE_REPO=https://github.com/lxq1000/SwinFace.git

set -euo pipefail

SWINFACE_REPO="${SWINFACE_REPO:-https://github.com/lxq1000/SwinFace.git}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

clone_one() {
  local dest="$1"
  # Vendored tree present: do not require removal
  if [[ -d "$dest/swinface_project" ]] && [[ ! -d "$dest/.git" ]]; then
    echo "==> skip (already have swinface_project/, not a git clone): $dest"
    return 0
  fi
  if [[ -d "$dest/.git" ]]; then
    echo "==> pull: $dest"
    git -C "$dest" pull --ff-only
  elif [[ ! -e "$dest" ]] || { [[ -d "$dest" ]] && [[ -z "$(ls -A "$dest" 2>/dev/null)" ]]; }; then
    echo "==> clone: $dest"
    git clone --depth 1 "$SWINFACE_REPO" "$dest"
  else
    echo "error: $dest exists but has no swinface_project/ and is not a git clone — fix or remove, then re-run" >&2
    exit 1
  fi
  if [[ ! -d "$dest/swinface_project" ]]; then
    echo "error: $dest has no swinface_project/ — wrong repo?" >&2
    exit 1
  fi
}

clone_one "$ROOT/third_party/swinface"
clone_one "$ROOT/space-main/third_party/swinface"

if [[ -d "$ROOT/space-liveness" ]]; then
  clone_one "$ROOT/space-liveness/third_party/swinface"
fi

echo "SwinFace OK — swinface_project present under third_party/swinface (and Space copies)."
