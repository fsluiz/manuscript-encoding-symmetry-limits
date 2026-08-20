#!/usr/bin/env python3
"""Independent shell-resolved audit of the protocol-cyclic space.

The manuscript reconstruction closes the initial state under ``H_init`` and
``H_prob``.  This audit uses the equivalent generators

    M = sum_r |e><e|_r,   V = lambda H_cov + mu H_excl,

and exploits the exact integer-valued spectral shells of ``V``.  At every
iteration the new transport directions are projected back into every shell;
this is the joint-invariant closure required by ``alg*(M,V,I)``.  Running the
calculation at several tolerances tests whether a reported cyclic deficit is
a numerical rank artefact.

The script never modifies the frozen registry.  Use ``--write`` to record the
audit, or ``--check`` to compare a fresh calculation byte-for-byte with the
recorded JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from accessibility_prototype import (
    joint_orbits,
    orbit_isometry,
    quotient,
    single_uniform_driver,
)
from instance_registry import load_instance, primary_instance_ids
from orbit_cyclic import penalty_on_orbits, shell_cyclic_basis
from spectral_gap_study import build_hamiltonian
from symmetry_analysis import find_base_automorphisms


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "orbit_cyclic_audit.json"
REFERENCE = ROOT / "data" / "accessibility_table_reconstructed.json"
LAM, MU, NU = 5.0, 10.0, 50.0
TOLERANCES = (1e-9, 1e-11, 1e-13)


def reference_dimensions() -> dict[str, int]:
    document = json.loads(REFERENCE.read_text())
    return {
        row["instance_id"]: int(row["dimensions"]["cyclic"])
        for row in document["instances"]
    }


def analyse_instance(instance_id: str, expected_dimension: int) -> dict[str, Any]:
    incidence, costs, record = load_instance(
        instance_id, require_status="candidate_primary"
    )
    if costs is not None:
        raise ValueError(f"{instance_id}: weighted instance is outside this audit")
    n_b, n_t = map(int, incidence.shape)
    k = int(record["k_star"])
    full_dimension = n_b**k
    group = find_base_automorphisms(incidence)
    orbits = joint_orbits(n_b, k, group)
    isometry = orbit_isometry(orbits, full_dimension)

    h_init = single_uniform_driver(n_b, k)
    h_prob, _ = build_hamiltonian(incidence, k, LAM, MU, NU)
    h_init_q = quotient(h_init, isometry)
    h_prob_q = quotient(h_prob, isometry)
    fixed_dimension = len(orbits)
    identity = np.eye(fixed_dimension)
    transport = n_b * (k * identity - h_init_q)
    walk = (transport - k * identity) / (n_b - 1)
    potential_from_hamiltonian = h_prob_q - walk
    penalties = penalty_on_orbits(incidence, k, orbits, LAM, MU)
    diagonal_potential = np.diag(penalties.astype(float))

    uniform = np.ones(full_dimension, dtype=float) / math.sqrt(full_dimension)
    initial = np.asarray(isometry.T @ uniform).ravel()
    tolerance_records = []
    bases: dict[float, np.ndarray] = {}
    for tolerance in TOLERANCES:
        basis, audit = shell_cyclic_basis(
            transport, penalties, initial, tolerance
        )
        bases[tolerance] = basis
        tolerance_records.append(audit)

    dimensions = [entry["dimension"] for entry in tolerance_records]
    if len(set(dimensions)) != 1:
        status = "tolerance-dependent"
    elif dimensions[0] != expected_dimension:
        status = "disagrees-with-reference"
    else:
        status = "confirmed"

    frozen_path = ROOT / "instances" / "frozen" / f"{instance_id}.json"
    return {
        "instance_id": instance_id,
        "payload_sha256": record["payload_sha256"],
        "frozen_file_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "n_B": n_b,
        "n_T": n_t,
        "k": k,
        "base_group_order": len(group),
        "full_dimension": full_dimension,
        "joint_fixed_dimension": fixed_dimension,
        "number_of_penalty_shells": int(len(np.unique(penalties))),
        "reference_cyclic_dimension": expected_dimension,
        "potential_identity_residual": float(
            np.linalg.norm(potential_from_hamiltonian - diagonal_potential, ord=2)
        ),
        "tolerance_audits": tolerance_records,
        "status": status,
    }


def calculate(instance_ids: list[str]) -> dict[str, Any]:
    expected = reference_dimensions()
    rows = []
    for position, instance_id in enumerate(instance_ids, start=1):
        print(f"[{position}/{len(instance_ids)}] {instance_id}", flush=True)
        row = analyse_instance(instance_id, expected[instance_id])
        rows.append(row)
        dims = [entry["dimension"] for entry in row["tolerance_audits"]]
        print(
            f"  fixed={row['joint_fixed_dimension']} reference={row['reference_cyclic_dimension']} "
            f"shell_closures={dims} status={row['status']}",
            flush=True,
        )
    return {
        "schema_version": 1,
        "method": "shell-projector closure of alg*(M,V,I) in the exact joint orbit quotient",
        "tolerances": list(TOLERANCES),
        "instances": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", action="append", dest="instances")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    instance_ids = args.instances or primary_instance_ids()
    document = calculate(instance_ids)
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != rendered:
            raise SystemExit(f"{OUTPUT.relative_to(ROOT)} is absent or stale")
        print("Orbit-cyclic audit verified byte-for-byte.")
    elif args.write:
        OUTPUT.write_text(rendered)
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    else:
        print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
