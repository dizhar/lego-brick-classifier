#!/usr/bin/env bash
#
# Download model weights from GCS into ./models so the app can run.
#
# Weights are NOT tracked in git (see .gitignore) — this script is the
# source of truth for provisioning them. Idempotent: files that already
# exist are skipped unless you pass --force.
#
# Usage:
#   ./scripts/download_models.sh           # download missing weights
#   ./scripts/download_models.sh --force   # re-download everything
#
set -euo pipefail

BUCKET="gs://briqvision-training/training_weights"

# Map each local destination to its source object in GCS.
# Tier 1: YOLO11m, 43 broad categories.  Tier 2: EfficientNet-B0.
declare -a MODELS=(
  "models/best.pt|$BUCKET/yolo11m_43cat_v1/best.pt"
  "models/efficientnet_b0_tier2_v1/best.pt|$BUCKET/efficientnet_b0_tier2_v1/best.pt"
)

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# Fail early with a helpful message if gcloud is missing.
if ! command -v gcloud >/dev/null 2>&1; then
  echo "✗ gcloud CLI not found. Install it: https://cloud.google.com/sdk/docs/install" >&2
  exit 1
fi

# Repo root = parent of this script's directory, so it works from anywhere.
cd "$(dirname "$0")/.."

for entry in "${MODELS[@]}"; do
  dest="${entry%%|*}"
  src="${entry##*|}"

  if [[ -f "$dest" && "$FORCE" -eq 0 ]]; then
    echo "✓ $dest already present — skipping (use --force to re-download)"
    continue
  fi

  echo "↓ $src"
  mkdir -p "$(dirname "$dest")"
  gcloud storage cp "$src" "$dest"
done

echo "✓ Models ready in ./models"
