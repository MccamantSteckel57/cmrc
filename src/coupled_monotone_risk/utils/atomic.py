"""Atomic writes and digests for checkpoints, artefacts and manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    return path


def atomic_write_json(path: Path, payload: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=indent, sort_keys=False) + "\n")


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
    os.replace(temporary, path)
    return path


def atomic_torch_load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path.name} does not contain a mapping")
    return payload
