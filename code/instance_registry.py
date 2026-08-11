"""Strict loader for frozen Minimum Set Cover instances.

New numerical scripts should use :func:`load_instance` instead of invoking an
extractor or generator.  The loader verifies both the full-file hash from the
manifest and the canonical hash of the mathematical payload before returning
the incidence matrix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "instances" / "manifest.json"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_instance(
    instance_id: str,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    require_status: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load and validate one frozen instance.

    Returns ``(A, costs, metadata)`` with ``A`` in the bases-by-tasks
    convention. ``costs`` is ``None`` for unweighted records.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    matches = [r for r in manifest["records"] if r["instance_id"] == instance_id]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one manifest record for {instance_id!r}")
    entry = matches[0]
    if require_status is not None and entry["status"] != require_status:
        raise ValueError(f"{instance_id}: status {entry['status']!r}, required {require_status!r}")

    root = manifest_path.parent.parent
    record_path = root / entry["path"]
    raw = record_path.read_bytes()
    if _sha256(raw) != entry["file_sha256"]:
        raise ValueError(f"{instance_id}: frozen-file SHA-256 mismatch")
    record = json.loads(raw)
    if record["instance_id"] != instance_id:
        raise ValueError(f"{instance_id}: record ID mismatch")
    if record["payload_sha256"] != entry["payload_sha256"]:
        raise ValueError(f"{instance_id}: manifest/record payload hash mismatch")
    if _sha256(_canonical_bytes(record["payload"])) != record["payload_sha256"]:
        raise ValueError(f"{instance_id}: canonical payload SHA-256 mismatch")

    strings = record["payload"]["incidence"]
    if len(strings) != record["n_B"] or any(len(row) != record["n_T"] for row in strings):
        raise ValueError(f"{instance_id}: declared dimensions do not match incidence strings")
    if any(set(row) - {"0", "1"} for row in strings):
        raise ValueError(f"{instance_id}: incidence strings are not binary")
    A = np.asarray([[char == "1" for char in row] for row in strings], dtype=np.int8)
    if not A.any(axis=0).all():
        raise ValueError(f"{instance_id}: at least one task is uncovered")
    raw_costs = record["payload"].get("costs")
    costs = None if raw_costs is None else np.asarray(raw_costs, dtype=float)
    if costs is not None and costs.shape != (record["n_B"],):
        raise ValueError(f"{instance_id}: cost vector has the wrong length")
    return A, costs, record


def primary_instance_ids(manifest_path: str | Path = DEFAULT_MANIFEST) -> list[str]:
    """Return sorted IDs admitted to the first reconstruction."""
    manifest = json.loads(Path(manifest_path).read_text())
    return sorted(r["instance_id"] for r in manifest["records"] if r["status"] == "candidate_primary")
