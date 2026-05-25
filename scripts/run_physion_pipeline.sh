#!/usr/bin/env bash
# End-to-end Physion pipeline: download -> build manifests -> precompute tracks
# -> train CILF with object-centric dynamics -> evaluate zero-shot causal transfer.
#
# Override the defaults via env vars, eg.
#   CONFIG=configs/cilf_physion.yaml MAX_STEPS=500 PRECOMPUTE_LIMIT=200 \
#     scripts/run_physion_pipeline.sh
#
# By default we use a small subset so the pipeline finishes in roughly one
# afternoon on a single MacBook.  Set ``MAX_STEPS=0`` to skip training and only
# run the data + detector stages.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG=${CONFIG:-configs/cilf_physion.yaml}
SPLIT=${SPLIT:-test}                  # test | train | both
PHYSION_DIR=${PHYSION_DIR:-data/physion}
PHYSION_LINK_DIR="$PHYSION_DIR/PhysionTest-Core/Physion"
MANIFEST_DIR=${MANIFEST_DIR:-data/kinetic_transfer}
TRACKS_DIR=${TRACKS_DIR:-data/kinetic_transfer/tracks}
DETECTOR=${DETECTOR:-physion}         # physion | yolo
PRECOMPUTE_LIMIT=${PRECOMPUTE_LIMIT:-0}   # 0 = no limit
MAX_STEPS=${MAX_STEPS:-300}
EXTRA_TRAIN_OVERRIDES=${EXTRA_TRAIN_OVERRIDES:-}

echo "[1/4] downloading Physion ($SPLIT)..."
python scripts/download_physion.py --split "$SPLIT" --dest "$PHYSION_DIR"

# When the test split is downloaded as a bare "Physion/" we also expose a
# PhysionTest-Core/Physion symlink so the manifest builder finds the videos.
if [[ ! -d "$PHYSION_LINK_DIR" && -d "$PHYSION_DIR/Physion" ]]; then
  mkdir -p "$PHYSION_DIR/PhysionTest-Core"
  ln -snf "../Physion" "$PHYSION_LINK_DIR"
fi

echo "[2/4] building kinetic-transfer manifests..."
PHYSION_ROOT="$PHYSION_LINK_DIR"
[[ -d "$PHYSION_ROOT" ]] || PHYSION_ROOT="$PHYSION_DIR/Physion"
python scripts/build_kinetic_transfer_manifests.py \
  --physion-root "$PHYSION_ROOT" \
  --output-dir "$MANIFEST_DIR"

echo "[3/4] precomputing object tracks via $DETECTOR..."
PRECOMPUTE_ARGS=(
  --manifest "$MANIFEST_DIR/manifest_kinetic_train.jsonl"
  --tracks-dir "$TRACKS_DIR"
  --detector "$DETECTOR"
)
if [[ "$PRECOMPUTE_LIMIT" -gt 0 ]]; then
  PRECOMPUTE_ARGS+=(--limit "$PRECOMPUTE_LIMIT")
fi
python scripts/precompute_yolo_tracks.py "${PRECOMPUTE_ARGS[@]}"

python scripts/precompute_yolo_tracks.py \
  --manifest "$MANIFEST_DIR/manifest_kinetic_val.jsonl" \
  --tracks-dir "$TRACKS_DIR" \
  --detector "$DETECTOR"

python scripts/precompute_yolo_tracks.py \
  --manifest "$MANIFEST_DIR/manifest_kinetic_transfer.jsonl" \
  --tracks-dir "$TRACKS_DIR" \
  --detector "$DETECTOR"

if [[ "$MAX_STEPS" -gt 0 ]]; then
  echo "[4/4] training CILF on Physion + detector tracks..."
  TRAIN_OVERRIDES=(--override "max_steps=$MAX_STEPS")
  if [[ -n "$EXTRA_TRAIN_OVERRIDES" ]]; then
    TRAIN_OVERRIDES+=(--override "$EXTRA_TRAIN_OVERRIDES")
  fi
  python -m cilf.train --config "$CONFIG" "${TRAIN_OVERRIDES[@]}"
  echo "[done] evaluating zero-shot causal transfer..."
  python scripts/evaluate_causal_transfer.py \
    --config "$CONFIG" \
    --manifest "$MANIFEST_DIR/manifest_kinetic_transfer.jsonl"
else
  echo "[skip] training disabled (MAX_STEPS=0)."
fi
