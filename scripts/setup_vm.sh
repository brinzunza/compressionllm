#!/usr/bin/env bash
# One-shot setup for a fresh GPU VM: venv, dependencies, data, tests.
#   git clone https://github.com/brinzunza/compressionllm.git && cd compressionllm && scripts/setup_vm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"      # on Linux the default torch wheel is the CUDA build
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
scripts/get_data.sh
python -m pytest -q
echo "Setup OK. Next: source .venv/bin/activate && cllm run configs/smoke.yaml"
