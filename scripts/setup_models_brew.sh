#!/usr/bin/env bash
# Download CILF vision weights on macOS via Homebrew's `hf` CLI (office-friendly).
#
# Install once:
#   brew install hf git-lfs
#   brew link --overwrite certifi   # if brew reports certifi link conflicts
#
# Optional (faster / fewer rate limits on corporate networks):
#   export HF_TOKEN=hf_...
#
# Usage:
#   scripts/setup_models_brew.sh
#   scripts/setup_models_brew.sh distilgpt2   # also cache the default LLM

set -euo pipefail
cd "$(dirname "$0")/.."

HF_BIN="${HF_BIN:-/opt/homebrew/bin/hf}"
if [[ ! -x "${HF_BIN}" ]]; then
  echo "Homebrew hf not found. Run: brew install hf git-lfs"
  exit 1
fi

export PATH="/opt/homebrew/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

SIGLIP_DIR="models/siglip-base-patch16-224"
mkdir -p "${SIGLIP_DIR}"

echo "=== Download SigLIP → ${SIGLIP_DIR} ==="
if [[ -f "${SIGLIP_DIR}/model.safetensors" ]]; then
  echo "Already present: ${SIGLIP_DIR}/model.safetensors"
else
  # Resume-friendly; use brew hf (not venv shadowed `hf` on PATH).
  "${HF_BIN}" download google/siglip-base-patch16-224 \
    --local-dir "${SIGLIP_DIR}" \
    --max-workers 2
fi

for extra in "$@"; do
  echo "=== Download ${extra} ==="
  "${HF_BIN}" download "${extra}" --local-dir "models/$(basename "${extra}")"
done

echo "=== Verify load (project venv) ==="
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
from transformers import AutoModel

p = Path("models/siglip-base-patch16-224")
if not (p / "model.safetensors").exists():
    raise SystemExit(
        f"Missing {p}/model.safetensors — download may be blocked on office Wi‑Fi.\n"
        "Try: phone hotspot, VPN, or copy the folder from another machine."
    )
m = AutoModel.from_pretrained(str(p), local_files_only=True)
print("SigLIP OK:", sum(x.numel() for x in m.parameters()))
PY

echo "Done. Train with: scripts/train_physion_tracks.sh"
