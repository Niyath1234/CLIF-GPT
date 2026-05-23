#!/usr/bin/env bash
# Create .venv and install project dependencies only inside that venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "$ROOT/.venv" ]]; then
  echo "Refusing to install: another venv is active (${VIRTUAL_ENV})."
  echo "Deactivate it first, then run: bash scripts/setup_venv.sh"
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .

echo ""
echo "Setup complete. Activate before training:"
echo "  source .venv/bin/activate"
