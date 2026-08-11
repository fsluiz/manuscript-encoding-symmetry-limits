#!/usr/bin/env python3
"""Create and validate the immutable instance registry used by the rewrite.

This command deliberately does *not* update any legacy result in ``data/``.
It freezes the mathematical inputs that future calculations may consume and
records enough provenance to reconstruct every matrix from its parent source.

Usage from the repository root::

    python3 code/freeze_instances.py --write
    python3 code/freeze_instances.py --check

``--write`` is non-destructive: an existing file is accepted only when its
bytes are identical.  A differing existing file is an error; there is no
force/overwrite option by design.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from orlib_parse import parse_scp
from orlib_subextract import extract_subinstance
from spectral_gap_study import grid_instance, random_msc_instance


ROOT = Path(__file__).resolve().parent.parent
ORLIB_DIR = ROOT / "orlib_instances"
REGISTRY_DIR = ROOT / "instances" / "frozen"
MANIFEST_PATH = ROOT / "instances" / "manifest.json"
REPORT_PATH = ROOT / "reports" / "instance_audit.md"

SCHEMA_VERSION = 1
EXTRACTOR_ID = "orlib_subextract.extract_subinstance@working-tree-2026-08-10"


# Instances intended for the first symmetry/accessibility reconstruction.
ORLIB_PRIMARY = [
    ("scpe1", 8, 30, 0),
    ("scpe2", 8, 30, 0),
    ("scpe3", 8, 30, 0),
    ("scpe4", 8, 30, 0),
    ("scpe5", 8, 30, 0),
    ("scpe1", 8, 20, 0),
    ("scpe2", 8, 20, 0),
    ("scpe3", 8, 20, 0),
]

# Inputs used only by appendices/experiments in the superseded manuscript.
ORLIB_LEGACY = [(f"scpe{i}", 16, 20, 0) for i in range(1, 6)]
WEIGHTED_LEGACY = [("scp41", 8, 15, 0)]

GRID_SPECS = [(2, 4), (3, 3), (2, 5), (2, 6), (3, 4), (2, 7), (3, 5), (4, 4)]

# These are the exact random matrices named in data/gap_results.json.  The
# compact mechanism aliases in spectral_taxonomy_results.json refer to seeds
# 2 and 14 below.
RANDOM_SPECS = [
    (8, 7, 0.50, 0),
    (8, 7, 0.50, 2),
    (8, 7, 0.50, 4),
    (7, 7, 0.45, 14),
    (8, 8, 0.42, 9),
    (8, 8, 0.42, 14),
    (8, 7, 0.30, 19),
    (8, 8, 0.22, 1),
    (8, 8, 0.22, 4),
    (8, 8, 0.22, 9),
]


def canonical_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON representation used for content hashes."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def exact_minimum_covers(A: np.ndarray) -> tuple[int, list[tuple[int, ...]]]:
    """Return the exact unweighted optimum and all optimal covers.

    All registry instances have at most 16 bases, so exhaustive enumeration is
    transparent, dependency-free, and fast enough for registry validation.
    """
    n_b, n_t = A.shape
    full = (1 << n_t) - 1
    masks: list[int] = []
    for row in A.astype(bool):
        mask = 0
        for t, covered in enumerate(row):
            if covered:
                mask |= 1 << t
        masks.append(mask)
    for k in range(1, n_b + 1):
        covers = [c for c in itertools.combinations(range(n_b), k) if _union_mask(c, masks) == full]
        if covers:
            return k, covers
    raise ValueError("infeasible set-cover instance")


def _union_mask(indices: Iterable[int], masks: list[int]) -> int:
    out = 0
    for index in indices:
        out |= masks[index]
    return out


def exact_weighted_optimum(A: np.ndarray, costs: np.ndarray) -> tuple[float, list[tuple[int, ...]]]:
    """Return exact minimum cost and all minimum-cost covers."""
    n_b, n_t = A.shape
    full = (1 << n_t) - 1
    masks = []
    for row in A.astype(bool):
        mask = 0
        for t, covered in enumerate(row):
            if covered:
                mask |= 1 << t
        masks.append(mask)
    best = float("inf")
    covers: list[tuple[int, ...]] = []
    for k in range(1, n_b + 1):
        for combo in itertools.combinations(range(n_b), k):
            if _union_mask(combo, masks) != full:
                continue
            cost = float(sum(float(costs[b]) for b in combo))
            if cost < best:
                best, covers = cost, [combo]
            elif cost == best:
                covers.append(combo)
    if not covers:
        raise ValueError("infeasible weighted set-cover instance")
    return best, covers


def matrix_payload(A: np.ndarray, costs: np.ndarray | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "matrix_convention": "bases_by_tasks",
        "incidence": ["".join("1" if x else "0" for x in row) for row in A.astype(bool)],
    }
    if costs is not None:
        payload["costs"] = [int(x) if float(x).is_integer() else float(x) for x in costs]
    return payload


def make_record(
    instance_id: str,
    A: np.ndarray,
    provenance: dict[str, Any],
    status: str,
    aliases: list[str],
    costs: np.ndarray | None = None,
) -> dict[str, Any]:
    A = np.asarray(A, dtype=np.int8)
    if A.ndim != 2 or not np.isin(A, [0, 1]).all():
        raise ValueError(f"{instance_id}: incidence matrix is not binary and two-dimensional")
    if not A.any(axis=0).all():
        raise ValueError(f"{instance_id}: at least one task is uncovered")
    k_star, covers = exact_minimum_covers(A)
    payload = matrix_payload(A, costs)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "status": status,
        "aliases": sorted(set(aliases)),
        "n_B": int(A.shape[0]),
        "n_T": int(A.shape[1]),
        "k_star": k_star,
        "n_optimal_cardinality_covers": len(covers),
        "optimal_cardinality_covers": [list(c) for c in covers],
        "provenance": provenance,
        "payload": payload,
        "payload_sha256": sha256_bytes(canonical_bytes(payload)),
    }
    if costs is not None:
        optimum, weighted_covers = exact_weighted_optimum(A, np.asarray(costs))
        record["weighted_optimum"] = {
            "cost": int(optimum) if optimum.is_integer() else optimum,
            "minimum_cardinality_among_optima": min(map(len, weighted_covers)),
            "covers": [list(c) for c in weighted_covers],
        }
    return record


def freeze_orlib(parent: str, n_target: int, m_target: int, seed: int, status: str) -> dict[str, Any]:
    source = ORLIB_DIR / f"{parent}.txt"
    parsed = parse_scp(source)
    A_sub, rows, cols = extract_subinstance(parsed, n_target=n_target, m_target=m_target, seed=seed)
    A = A_sub.T.astype(np.int8)
    n_t = int(A.shape[1])
    instance_id = f"{parent}-b{n_target}-t{n_t}-s{seed}-v1"
    requested_alias = f"{parent}-n{n_target}m{m_target}s{seed}"
    retained_alias = f"{parent}-n{n_target}m{n_t}s{seed}"
    provenance = {
        "kind": "orlib_subinstance",
        "parent_file": str(source.relative_to(ROOT)),
        "parent_sha256": sha256_file(source),
        "parent_dimensions": {"n_B": int(parsed.n), "n_T": int(parsed.m)},
        "extractor": EXTRACTOR_ID,
        "parameters": {"n_target": n_target, "m_target": m_target, "seed": seed},
        "selected_parent_task_indices_zero_based": rows.astype(int).tolist(),
        "selected_parent_base_indices_zero_based": cols.astype(int).tolist(),
        "requested_n_T": m_target,
        "retained_n_T": n_t,
    }
    return make_record(instance_id, A, provenance, status, [requested_alias, retained_alias])


def freeze_weighted(parent: str, n_target: int, m_target: int, seed: int) -> dict[str, Any]:
    source = ORLIB_DIR / f"{parent}.txt"
    parsed = parse_scp(source)
    A_sub, rows, cols = extract_subinstance(parsed, n_target=n_target, m_target=m_target, seed=seed)
    A = A_sub.T.astype(np.int8)
    costs = parsed.costs[cols]
    n_t = int(A.shape[1])
    provenance = {
        "kind": "orlib_weighted_subinstance",
        "parent_file": str(source.relative_to(ROOT)),
        "parent_sha256": sha256_file(source),
        "parent_dimensions": {"n_B": int(parsed.n), "n_T": int(parsed.m)},
        "extractor": EXTRACTOR_ID,
        "parameters": {"n_target": n_target, "m_target": m_target, "seed": seed},
        "selected_parent_task_indices_zero_based": rows.astype(int).tolist(),
        "selected_parent_base_indices_zero_based": cols.astype(int).tolist(),
        "requested_n_T": m_target,
        "retained_n_T": n_t,
    }
    return make_record(
        f"{parent}-weighted-b{n_target}-t{n_t}-s{seed}-v1",
        A,
        provenance,
        "legacy_only",
        [f"{parent}-n{n_target}m{m_target}s{seed}", f"{parent}-n{n_target}m{n_t}s{seed}"],
        costs=costs,
    )


def freeze_grid(rows: int, cols: int) -> dict[str, Any]:
    A = grid_instance(rows, cols).astype(np.int8)
    legacy = f"grid_{rows}x{cols}"
    status = "candidate_primary" if (rows, cols) == (2, 4) else "legacy_only"
    return make_record(
        f"grid-{rows}x{cols}-v1",
        A,
        {
            "kind": "generated_grid",
            "generator": "spectral_gap_study.grid_instance",
            "parameters": {"rows": rows, "cols": cols},
        },
        status,
        [legacy],
    )


def freeze_random(n_b: int, n_t: int, p: float, seed: int) -> dict[str, Any]:
    A = random_msc_instance(n_b, n_t, p, seed).astype(np.int8)
    pcode = f"{round(100 * p):02d}"
    full_alias = f"rand_nB{n_b}_nT{n_t}_p{pcode}_s{seed:03d}"
    aliases = [full_alias]
    status = "legacy_only"
    if (n_b, n_t, p, seed) == (8, 7, 0.50, 2):
        aliases.append("rand_nB8_p50_s002")
        status = "candidate_primary"
    if (n_b, n_t, p, seed) == (8, 8, 0.42, 14):
        aliases.append("rand_nB8_p42_s014")
        status = "candidate_primary"
    return make_record(
        f"random-b{n_b}-t{n_t}-p{pcode}-s{seed:03d}-v1",
        A,
        {
            "kind": "generated_random",
            "generator": "spectral_gap_study.random_msc_instance",
            "rng": "numpy.random.default_rng",
            "parameters": {"n_B": n_b, "n_T": n_t, "p": p, "seed": seed},
        },
        status,
        aliases,
    )


def build_records() -> list[dict[str, Any]]:
    records = [freeze_orlib(*spec, "candidate_primary") for spec in ORLIB_PRIMARY]
    records += [freeze_orlib(*spec, "legacy_only") for spec in ORLIB_LEGACY]
    records += [freeze_weighted(*spec) for spec in WEIGHTED_LEGACY]
    records += [freeze_grid(*spec) for spec in GRID_SPECS]
    records += [freeze_random(*spec) for spec in RANDOM_SPECS]
    ids = [r["instance_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate frozen instance ID")
    return sorted(records, key=lambda r: r["instance_id"])


def encoded_file(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def build_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for record in records:
        content = encoded_file(record)
        entries.append({
            "instance_id": record["instance_id"],
            "path": f"instances/frozen/{record['instance_id']}.json",
            "status": record["status"],
            "n_B": record["n_B"],
            "n_T": record["n_T"],
            "k_star": record["k_star"],
            "payload_sha256": record["payload_sha256"],
            "file_sha256": sha256_bytes(content),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": "Future numerical claims must cite an instance_id and payload_sha256 from this manifest.",
        "records": entries,
    }


def build_report(records: list[dict[str, Any]]) -> str:
    primary_orlib = [r for r in records if r["status"] == "candidate_primary" and r["provenance"]["kind"] == "orlib_subinstance"]
    weighted = next(r for r in records if r["provenance"]["kind"] == "orlib_weighted_subinstance")
    lines = [
        "# Instance audit for the symmetry/accessibility rewrite",
        "",
        "This report freezes the current mathematical inputs. It does **not** certify that superseded legacy JSON files were generated from these exact matrices. Those legacy files are intentionally omitted from the archival artifact and are not accepted as evidence for the current manuscript.",
        "",
        "## Current OR-Library extraction",
        "",
        "| Frozen ID | requested tasks | retained tasks | bases | exact `k*` | payload SHA-256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in primary_orlib:
        p = r["provenance"]
        lines.append(f"| `{r['instance_id']}` | {p['requested_n_T']} | {r['n_T']} | {r['n_B']} | {r['k_star']} | `{r['payload_sha256'][:16]}…` |")
    lines += [
        "",
        "The registry records the selected zero-based parent task/base indices and the SHA-256 of each OR-Library parent file. Requested and retained task counts are separate fields because the current extractor drops sampled tasks left uncovered by its selected bases.",
        "",
        "## Reproducibility discrepancies found",
        "",
        "1. `scpe4`: the current extraction retains 29 tasks. `data/orlib_experiment_results.json` labels it as `m30`, whereas later files label it as `m29`.",
        "2. `scpe5`: the current extraction retains all 30 tasks, but `data/orlib_experiment_results.json` records `n_T=29` under an `m30` label, and the manuscript/later files use an `m29` label. The old 29-task matrix and selected indices were not stored, so that historical input cannot be reconstructed from the repository.",
        "3. `scpe3` with 30 requested/retained tasks: exact enumeration on the current matrix gives `k*=5`, while every legacy table/result records `k*=4`. Consequently, the stored spectrum cannot be attributed to the currently reconstructible matrix.",
        "4. The extractor docstring describes a contiguous/first-row procedure, while the implementation samples rows uniformly with a seeded NumPy generator. It also does not test connectedness, despite the old README calling the outputs connected.",
        f"5. Weighted `scp41`: the current extraction is `{weighted['instance_id']}`, with costs `{weighted['payload']['costs']}` and weighted optimum {weighted['weighted_optimum']['cost']}. The legacy file reports costs `[3, 4, 12, 24, 73, 77, 96, 100]` and optimum 196, so it came from a different, unrecoverable slice.",
        "6. Legacy result files generally store labels and dimensions but not the input matrix, selected parent indices, source hash, code revision, or residual-based numerical acceptance data. Equality of a label is therefore not proof of equality of the input.",
        "",
        "## Editorial disposition",
        "",
        "- `candidate_primary`: candidate inputs for the new full-H symmetry/accessibility calculations. This status is not a numerical endorsement; results must be recomputed.",
        "- `legacy_only`: retained to explain the old manuscript and prevent provenance loss, but excluded from the first reconstruction unless promoted explicitly.",
        "- The weighted, Fiedler, ADAPT, QAOA, and fixed-walk experiments are not part of the new primary claim and will not be carried into the rewritten main text without a separate justification.",
        "- No legacy JSON is accepted as a source of a new table. New outputs must embed both `instance_id` and `payload_sha256`.",
        "",
        "## Superseded artifacts",
        "",
        "The superseded result files used to diagnose the discrepancies above are not part of this archival repository. Their omission is deliberate: registry validation must depend only on the immutable source instances, deterministic generators, and files listed in `instances/manifest.json`.",
        "",
        "## Validation commands",
        "",
        "```bash",
        "python3 code/freeze_instances.py --check",
        "python3 code/freeze_instances.py --write   # only for a clean initial creation",
        "```",
        "",
        "The checker reconstructs every instance, recomputes exact minimum covers, validates binary dimensions and coverage, and compares canonical bytes and hashes. It fails rather than overwriting a divergent frozen file.",
        "",
    ]
    return "\n".join(lines)


def expected_outputs(records: list[dict[str, Any]]) -> dict[Path, bytes]:
    outputs = {REGISTRY_DIR / f"{r['instance_id']}.json": encoded_file(r) for r in records}
    manifest = build_manifest(records)
    outputs[MANIFEST_PATH] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    outputs[REPORT_PATH] = build_report(records).encode()
    return outputs


def write_outputs(outputs: dict[Path, bytes]) -> None:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite divergent file: {path.relative_to(ROOT)}")
        if not path.exists():
            path.write_bytes(content)


def check_outputs(outputs: dict[Path, bytes]) -> None:
    failures = []
    for path, content in outputs.items():
        if not path.exists():
            failures.append(f"missing: {path.relative_to(ROOT)}")
        elif path.read_bytes() != content:
            failures.append(f"divergent: {path.relative_to(ROOT)}")
    if failures:
        raise RuntimeError("registry validation failed:\n  " + "\n  ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="create missing frozen files without overwriting")
    mode.add_argument("--check", action="store_true", help="reconstruct and byte-compare the registry")
    args = parser.parse_args()

    records = build_records()
    outputs = expected_outputs(records)
    if args.write:
        write_outputs(outputs)
        action = "created/verified"
    else:
        check_outputs(outputs)
        action = "verified"
    primary = sum(r["status"] == "candidate_primary" for r in records)
    print(f"Registry {action}: {len(records)} instances ({primary} candidate_primary).")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"Audit:    {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
