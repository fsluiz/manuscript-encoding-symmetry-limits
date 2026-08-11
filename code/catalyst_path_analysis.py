#!/usr/bin/env python3
"""Symmetry-preserving two-register catalyst for the frozen D4 prototype.

The endpoint Hamiltonians are unchanged.  The catalyst

    C2 = - sum_{r<s} W_r W_s

is multiplied by 4*gamma*s*(1-s), so it is absent at s=0,1 but supplies
off-diagonal couplings when the one-register hopping cancels.  This script
tests robustness over a fixed gamma grid and resolves global dark crossings
for gamma=2.  Results are finite-instance numerical evidence, not an
asymptotic gap theorem.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp

from accessibility_prototype import (
    INSTANCE_ID,
    LAM,
    MU,
    NU,
    joint_orbits,
    orbit_isometry,
    quotient,
    single_uniform_driver,
)
from instance_registry import load_instance
from spectral_gap_study import build_hamiltonian
from symmetry_analysis import find_base_automorphisms


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "catalyst_path_results.json"
GAMMA_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
SELECTED_GAMMA = 2.0


def two_register_catalyst(n_b: int, k: int) -> sp.csr_matrix:
    W = (np.ones((n_b, n_b)) - np.eye(n_b)) / (n_b - 1)
    W = sp.csr_matrix(W)
    ident = sp.eye(n_b, format="csr")
    out = sp.csr_matrix((n_b**k, n_b**k), dtype=float)
    for r in range(k):
        for t in range(r + 1, k):
            term = sp.eye(1, format="csr")
            for position in range(k):
                term = sp.kron(term, W if position in (r, t) else ident, format="csr")
            out -= term
    return out


def path_matrix(H0: np.ndarray, H1: np.ndarray, C2: np.ndarray, gamma: float, s: float) -> np.ndarray:
    return (1 - s) * H0 + s * H1 + 4 * gamma * s * (1 - s) * C2


def lowest_gap(H0: np.ndarray, H1: np.ndarray, C2: np.ndarray, gamma: float, s: float) -> float:
    values = la.eigh(path_matrix(H0, H1, C2, gamma, s), subset_by_index=[0, 1], eigvals_only=True)
    return float(max(0.0, values[1] - values[0]))


def minimize_gap(H0: np.ndarray, H1: np.ndarray, C2: np.ndarray, gamma: float, cancellation_s: float) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 2001)
    gaps = np.asarray([lowest_gap(H0, H1, C2, gamma, float(s)) for s in grid])
    candidates = [0, len(grid) - 1]
    candidates.extend(i for i in range(1, len(grid) - 1) if gaps[i] <= gaps[i - 1] and gaps[i] <= gaps[i + 1])
    best_index = int(np.argmin(gaps))
    best_s, best_gap = float(grid[best_index]), float(gaps[best_index])
    for index in candidates:
        left = float(grid[max(0, index - 1)])
        right = float(grid[min(len(grid) - 1, index + 1)])
        if left == right:
            continue
        result = opt.minimize_scalar(
            lambda s: lowest_gap(H0, H1, C2, gamma, float(s)),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-14},
        )
        if result.fun < best_gap:
            best_s, best_gap = float(result.x), float(result.fun)
    # Include the exact uncatalysed cancellation point in every comparison.
    cancellation_gap = lowest_gap(H0, H1, C2, gamma, cancellation_s)
    if cancellation_gap < best_gap:
        best_s, best_gap = cancellation_s, cancellation_gap
    return {
        "gamma": gamma,
        "minimum_A1_gap": best_gap,
        "at_s": best_s,
        "gap_at_uncatalysed_cancellation_s": cancellation_gap,
        "grid_points": len(grid),
    }


def analyse() -> dict[str, Any]:
    A, _, record = load_instance(INSTANCE_ID, require_status="candidate_primary")
    n_b = A.shape[0]
    k = int(record["k_star"])
    dim = n_b**k
    Hprob, _ = build_hamiltonian(A, k, LAM, MU, NU)
    H0 = single_uniform_driver(n_b, k)
    C2 = two_register_catalyst(n_b, k)

    group = find_base_automorphisms(A)
    orbits = joint_orbits(n_b, k, group)
    S = orbit_isometry(orbits, dim)
    H0a, Hpa, C2a = quotient(H0, S), quotient(Hprob, S), quotient(C2, S)
    cancellation_s = (n_b - 1) / (2 * n_b - 1)
    robustness = [minimize_gap(H0a, Hpa, C2a, gamma, cancellation_s) for gamma in GAMMA_GRID]

    # Resolve the complement of A1 for the selected catalyst.  Any crossing
    # found here is symmetry-dark because every path term is group invariant.
    R = la.null_space(S.T.toarray())
    H0o = R.T @ H0 @ R
    Hpo = R.T @ Hprob @ R
    C2o = R.T @ C2 @ R

    def floor(matrix: np.ndarray) -> float:
        return float(la.eigh(matrix, subset_by_index=[0, 0], eigvals_only=True)[0])

    def branch_difference(s: float) -> float:
        return floor(path_matrix(H0o, Hpo, C2o, SELECTED_GAMMA, s)) - floor(
            path_matrix(H0a, Hpa, C2a, SELECTED_GAMMA, s)
        )

    scan = np.linspace(0.0, 1.0, 401)
    differences = np.asarray([branch_difference(float(s)) for s in scan])
    crossings = []
    for left, right, fleft, fright in zip(scan[:-1], scan[1:], differences[:-1], differences[1:]):
        if fleft == 0 or fleft * fright < 0:
            root = float(opt.brentq(branch_difference, float(left), float(right), xtol=1e-13, rtol=1e-13))
            if crossings and abs(root - crossings[-1]["s"]) < 1e-9:
                continue
            derivative = Hprob - H0 + 4 * SELECTED_GAMMA * (1 - 2 * root) * C2
            crossings.append({
                "s": root,
                "energy": floor(path_matrix(H0a, Hpa, C2a, SELECTED_GAMMA, root)),
                "A1_gap": lowest_gap(H0a, Hpa, C2a, SELECTED_GAMMA, root),
                "branch_difference": branch_difference(root),
                "cross_sector_derivative_coupling_norm": float(np.linalg.norm(R.T @ derivative @ S, ord=2)),
            })

    endpoint_catalyst_norms = {
        "s0": float(np.linalg.norm(4 * SELECTED_GAMMA * 0 * (1 - 0) * C2a, ord=2)),
        "s1": float(np.linalg.norm(4 * SELECTED_GAMMA * 1 * (1 - 1) * C2a, ord=2)),
    }
    return {
        "schema_version": 1,
        "scope": "finite D4 catalyst diagnostic; no asymptotic or interval-certified gap bound",
        "instance_id": INSTANCE_ID,
        "payload_sha256": record["payload_sha256"],
        "parameters": {"k": k, "lambda": LAM, "mu": MU, "nu": NU},
        "catalyst": {
            "definition": "C2 = -sum_{r<s} W_r W_s",
            "envelope": "4 gamma s(1-s)",
            "preserves": ["S_k", "G_B"],
            "selected_gamma": SELECTED_GAMMA,
            "endpoint_term_norms": endpoint_catalyst_norms,
        },
        "dimensions": {"full_valid": dim, "A1_joint_fixed": S.shape[1]},
        "robustness_grid": robustness,
        "selected_path": {
            "minimum_A1_gap": next(row for row in robustness if row["gamma"] == SELECTED_GAMMA),
            "global_A1_vs_nontrivial_crossings": crossings,
        },
    }


def main() -> None:
    result = analyse()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for row in result["robustness_grid"]:
        print(f"gamma={row['gamma']:>4}: min A1 gap={row['minimum_A1_gap']:.9g} at s={row['at_s']:.9f}")
    print("dark crossings:")
    for crossing in result["selected_path"]["global_A1_vs_nontrivial_crossings"]:
        print(crossing)


if __name__ == "__main__":
    main()
