#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
STUDY="${1:-configs/study/main.yaml}"
shift || true

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  -m coupled_monotone_risk.cli.train --config "${STUDY}" "$@"
