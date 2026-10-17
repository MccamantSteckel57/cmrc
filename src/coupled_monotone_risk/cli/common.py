"""Shared command-line helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from coupled_monotone_risk.utils.atomic import atomic_write_json
from coupled_monotone_risk.utils.config import load_config
from coupled_monotone_risk.utils.logging import configure


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", action="append", default=[], help="YAML file, repeatable")
    parser.add_argument("--set", action="append", default=[], help="dotted override key=value")
    parser.add_argument("--output", type=Path, default=None, help="JSON artefact path")
    parser.add_argument("--log-level", default="INFO")


def resolve_config(arguments: argparse.Namespace) -> dict[str, Any]:
    configure(str(arguments.log_level))
    paths = [Path(item) for item in arguments.config]
    return load_config(paths, list(arguments.set))


def emit(payload: dict[str, Any], destination: Path | None) -> None:
    text = json.dumps(payload, indent=2)
    if destination is None:
        print(text)
    else:
        atomic_write_json(destination, payload)
