#!/usr/bin/env bash
# Train CILF on Physion kinetic-transfer manifests with precomputed object tracks.
#
# Tracks must exist under data/kinetic_transfer/tracks/ (one JSON per stim_id).
# Precompute if needed:
#   python scripts/precompute_yolo_tracks.py \
#     --manifest data/kinetic_transfer/manifest_kinetic_train.jsonl \
#     --tracks-dir data/kinetic_transfer/tracks \
#     --detector physion
#
# Usage:
#   scripts/train_physion_tracks.sh
#   MAX_STEPS=500 scripts/train_physion_tracks.sh
#   STAGE=jepa_pretrain MAX_STEPS=400 scripts/train_physion_tracks.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MAX_STEPS="${MAX_STEPS:-2000}"
STAGE="${STAGE:-cilf_fusion}"
CKPT="${CKPT:-}"
CONFIG="${CONFIG:-}"

if [[ -z "${CONFIG}" ]]; then
  if [[ -f models/siglip-base-patch16-224/model.safetensors ]]; then
    CONFIG="configs/cilf_physion.yaml"
  else
    echo "SigLIP weights missing — using office config (trainable conv encoder + tracks)."
    echo "  Run: scripts/setup_models_brew.sh  (or CONFIG=configs/cilf_physion.yaml after download)"
    CONFIG="configs/cilf_physion_office.yaml"
  fi
fi

ARGS=(--config "${CONFIG}" --override "training.stage=${STAGE}" --override "training.max_steps=${MAX_STEPS}")
if [[ -n "${CKPT}" ]]; then
  ARGS+=(--override "training.checkpoint_path=${CKPT}")
fi

echo "=== CILF train with detector tracks ==="
echo "stage=${STAGE} max_steps=${MAX_STEPS}"
python -m cilf.train "${ARGS[@]}"
