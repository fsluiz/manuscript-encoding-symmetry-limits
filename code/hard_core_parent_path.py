#!/usr/bin/env python3
"""Hard-core Johnson parent for the analytic cycle family.

This is the primary analytically controlled path in the symmetry rewrite.  Its
initial state is the uniform superposition over k-subsets rather than the
product-register state.  The beta=0 parent is the Johnson-graph Laplacian at
rate 1/n.  The endpoint Gibbs weight is n^{-a U} with a=7.

Removing duplicate occupations isolates whether the intermediate-temperature
gap problem is intrinsic to coverage or caused by the product-register
embedding.
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cycle_family_exploration import ROOT, dihedral_group, orbit_isometry, subset_orbits


OUTPUT = ROOT / "data" / "hard_core_parent_path_results.json"
SIZES = [6, 8, 10, 12, 14, 16, 18]
ENDPOINT_EXPONENT = 7.0
GRID_POINTS = 31


def analyse(n: int) -> dict[str, Any]:
    k = n // 2
    states = list(itertools.combinations(range(n), k))
    index = {state: i for i, state in enumerate(states)}
    uncovered = np.empty(len(states), dtype=int)
    for i, state in enumerate(states):
        selected = set(state)
        uncovered[i] = sum(
            vertex not in selected and (vertex + 1) % n not in selected
            for vertex in range(n)
        )

    source: list[int] = []
    target: list[int] = []
    delta_u: list[int] = []
    universe = set(range(n))
    for source_index, state in enumerate(states):
        selected = set(state)
        for removed in state:
            for added in universe - selected:
                target_state = tuple(sorted((selected - {removed}) | {added}))
                target_index = index[target_state]
                source.append(source_index)
                target.append(target_index)
                delta_u.append(int(uncovered[target_index] - uncovered[source_index]))
    source_array = np.asarray(source, dtype=np.int64)
    target_array = np.asarray(target, dtype=np.int64)
    delta_array = np.asarray(delta_u, dtype=float)

    orbits = subset_orbits(states, dihedral_group(n))
    isometry = orbit_isometry(orbits, len(states))

    def quotient(exponent: float) -> sp.csr_matrix:
        offdiag = -np.exp(-0.5 * exponent * math.log(n) * np.abs(delta_array)) / n
        accepted = np.exp(-exponent * math.log(n) * np.maximum(delta_array, 0.0)) / n
        diagonal = np.bincount(source_array, weights=accepted, minlength=len(states))
        matrix = sp.csr_matrix(
            (offdiag, (target_array, source_array)), shape=(len(states), len(states))
        ) + sp.diags(diagonal, format="csr")
        return (isometry.T @ matrix @ isometry).tocsr()

    def gap(exponent: float) -> float:
        matrix = quotient(exponent)
        if matrix.shape[0] <= 3:
            values = np.linalg.eigvalsh(matrix.toarray())
        else:
            values = np.sort(
                spla.eigsh(matrix, k=3, which="SM", return_eigenvectors=False, tol=1e-10)
            )
        return float(values[1] - values[0])

    exponent_grid = np.linspace(0.0, ENDPOINT_EXPONENT, GRID_POINTS)
    gaps = np.asarray([gap(float(exponent)) for exponent in exponent_grid])
    minimum_index = int(np.argmin(gaps))
    endpoint_weights = np.exp(-ENDPOINT_EXPONENT * math.log(n) * uncovered)
    cover_probability = float(np.sum(endpoint_weights[uncovered == 0]) / np.sum(endpoint_weights))
    return {
        "n_B": n,
        "k": k,
        "hard_core_states": len(states),
        "joint_fixed_orbits": len(orbits),
        "initial_gap": float(gaps[0]),
        "endpoint_gap": float(gaps[-1]),
        "minimum_grid_gap": float(gaps[minimum_index]),
        "minimum_grid_exponent": float(exponent_grid[minimum_index]),
        "minimum_at_endpoint": minimum_index == len(exponent_grid) - 1,
        "endpoint_cover_probability": cover_probability,
        "grid_gaps": [
            {"energy_exponent": float(exponent), "gap": float(value)}
            for exponent, value in zip(exponent_grid, gaps)
        ],
    }


def main() -> None:
    results = [analyse(n) for n in SIZES]
    output = {
        "schema_version": 1,
        "path": "hard-core Johnson Metropolis parent in the D_n-fixed quotient",
        "endpoint_energy_exponent": ENDPOINT_EXPONENT,
        "qualification": (
            "primary analytic path; it changes the initial state from the product-register "
            "uniform state to the uniform injective/Dicke state; the uniform gap is proved "
            "analytically by zero-range comparison and a log-concave three-box coupling, "
            "not by the finite-size grid"
        ),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for row in results:
        print(
            f"C_{row['n_B']:2d}: gap0={row['initial_gap']:.8g}, "
            f"gap_min={row['minimum_grid_gap']:.8g} at a={row['minimum_grid_exponent']:.3g}, "
            f"gap7={row['endpoint_gap']:.8g}"
        )


if __name__ == "__main__":
    main()
