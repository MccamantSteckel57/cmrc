#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
LAYER="${1:-all}"
shift || true

python -m coupled_monotone_risk.cli.evaluate --config configs/study/main.yaml --layer "${LAYER}" "$@"
