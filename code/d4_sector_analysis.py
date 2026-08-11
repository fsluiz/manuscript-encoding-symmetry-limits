#!/usr/bin/env python3
"""Irrep-resolved initial-state comparison for the frozen 2x4 grid.

The calculation first quotients by register permutations S_k, then resolves
the resulting 120-dimensional space into the five D4 isotypic components.
It determines which alternative symmetry-adapted initial sectors can contain
the final global ground state and whether the uncatalysed linear path remains
isolated at the exact hopping-cancellation point.
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
    cover_mask,
    eigensystem,
    joint_orbits,
    level_summary,
    orbit_isometry,
    quotient,
    single_uniform_driver,
)
from instance_registry import load_instance
from spectral_gap_study import build_hamiltonian
from symmetry_analysis import build_base_perm, find_base_automorphisms


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "d4_sector_comparison.json"


def compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Permutation composition left o right."""
    return tuple(left[right[i]] for i in range(len(left)))


def inverse(perm: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(perm)
    for i, image in enumerate(perm):
        out[image] = i
    return tuple(out)


def power(perm: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    out = tuple(range(len(perm)))
    for _ in range(exponent):
        out = compose(perm, out)
    return out


def order(perm: tuple[int, ...]) -> int:
    identity = tuple(range(len(perm)))
    value = identity
    for exponent in range(1, 100):
        value = compose(perm, value)
        if value == identity:
            return exponent
    raise ValueError("permutation order exceeded defensive limit")


def identify_d4(group: list[tuple[int, ...]]) -> tuple[tuple[int, ...], tuple[int, ...], dict[tuple[int, ...], tuple[str, int]]]:
    """Choose generators r,s with r^4=s^2=e and srs=r^-1."""
    rotations = [g for g in group if order(g) == 4]
    for r in rotations:
        rotation_set = {power(r, a) for a in range(4)}
        for s in group:
            if s in rotation_set or order(s) != 2:
                continue
            if compose(compose(s, r), s) != inverse(r):
                continue
            coordinates: dict[tuple[int, ...], tuple[str, int]] = {}
            for a in range(4):
                coordinates[power(r, a)] = ("r", a)
                coordinates[compose(s, power(r, a))] = ("sr", a)
            if set(coordinates) == set(group):
                return r, s, coordinates
    raise ValueError("automorphism group did not admit the expected D4 presentation")


def d4_characters(coordinates: dict[tuple[int, ...], tuple[str, int]]) -> dict[str, dict[tuple[int, ...], float]]:
    chars = {name: {} for name in ["A1", "A2", "B1", "B2", "E"]}
    for g, (kind, a) in coordinates.items():
        reflection = kind == "sr"
        chars["A1"][g] = 1.0
        chars["A2"][g] = -1.0 if reflection else 1.0
        chars["B1"][g] = float((-1) ** a)
        chars["B2"][g] = float(-((-1) ** a) if reflection else (-1) ** a)
        chars["E"][g] = 0.0 if reflection or a % 2 else (2.0 if a == 0 else -2.0)
    return chars


def projector_basis(projector: np.ndarray) -> tuple[np.ndarray, float]:
    values, vectors = np.linalg.eigh((projector + projector.T) / 2)
    basis = vectors[:, values > 0.5]
    residual = float(np.linalg.norm(projector @ projector - projector, ord=2))
    return basis, residual


def spectral_record(H: np.ndarray, Pcover: np.ndarray) -> dict[str, Any]:
    values, vectors, residuals = eigensystem(H)
    summary = level_summary(values, float(np.linalg.norm(H, ord=2)), residuals)
    g0 = summary["ground_rank"]
    ground = vectors[:, :g0]
    cover_effect = ground.T @ Pcover @ ground
    probabilities = np.linalg.eigvalsh((cover_effect + cover_effect.T) / 2)
    return {
        "E0": float(values[0]),
        **summary,
        "ground_cover_probability_min": float(probabilities[0]),
        "ground_cover_probability_max": float(probabilities[-1]),
        "max_eigenpair_residual": float(residuals.max()),
    }


def simple_ground_gap_diagnostic(H0: np.ndarray, H1: np.ndarray, exact_records: dict[str, Any]) -> dict[str, Any]:
    """Numerically minimize E1-E0; meaningful for a rank-one ground branch."""
    if H0.shape[0] < 2:
        return {"minimum_E1_minus_E0": None, "qualification": "one-dimensional sector"}

    def gap(s: float) -> float:
        values = la.eigh((1 - s) * H0 + s * H1, subset_by_index=[0, 1], eigvals_only=True)
        return float(max(0.0, values[1] - values[0]))

    exact_degenerate = [name for name, rec in exact_records.items() if rec["ground_rank"] > 1]
    if exact_degenerate:
        point_map = {"s0": 0.0, "s_cancel": 7.0 / 15.0, "s1": 1.0}
        return {
            "minimum_E1_minus_E0": 0.0,
            "at_s": point_map[exact_degenerate[0]],
            "rank_one_at_all_three_exact_points": False,
            "qualification": "exact degeneracy at one or more audited path points",
        }

    grid = np.linspace(0.0, 1.0, 2001)
    gaps = np.asarray([gap(float(s)) for s in grid])
    candidates = [0, len(grid) - 1]
    candidates.extend(i for i in range(1, len(grid) - 1) if gaps[i] <= gaps[i - 1] and gaps[i] <= gaps[i + 1])
    best_s, best_gap = float(grid[int(np.argmin(gaps))]), float(gaps.min())
    for index in candidates:
        left = float(grid[max(0, index - 1)])
        right = float(grid[min(len(grid) - 1, index + 1)])
        if left == right:
            continue
        result = opt.minimize_scalar(gap, bounds=(left, right), method="bounded", options={"xatol": 1e-14})
        if result.fun < best_gap:
            best_s, best_gap = float(result.x), float(result.fun)
    return {
        "minimum_E1_minus_E0": best_gap,
        "at_s": best_s,
        "rank_one_at_all_three_exact_points": True,
        "grid_points": len(grid),
        "qualification": "scale-aware endpoint checks plus numerical one-dimensional minimization; not interval arithmetic",
    }


def analyse() -> dict[str, Any]:
    A, _, record = load_instance(INSTANCE_ID, require_status="candidate_primary")
    n_b = A.shape[0]
    k = int(record["k_star"])
    dim = n_b**k
    Hprob, _ = build_hamiltonian(A, k, LAM, MU, NU)
    H0 = single_uniform_driver(n_b, k)

    # First quotient only by S_k.  D4 continues to act faithfully here.
    identity = tuple(range(n_b))
    sk_orbits = joint_orbits(n_b, k, [identity])
    T = orbit_isometry(sk_orbits, dim)
    H0s, Hps = quotient(H0, T), quotient(Hprob, T)
    mask = cover_mask(A, k)
    Pcover_s = np.asarray((T.T @ sp.diags(mask.astype(float)) @ T).toarray())

    group = find_base_automorphisms(A)
    r, s, coordinates = identify_d4(group)
    characters = d4_characters(coordinates)
    representations = {
        g: np.asarray((T.T @ build_base_perm(g, k, n_b, n_b) @ T).toarray())
        for g in group
    }
    dimensions = {"A1": 1, "A2": 1, "B1": 1, "B2": 1, "E": 2}

    sector_bases: dict[str, np.ndarray] = {}
    projector_checks: dict[str, Any] = {}
    projector_sum = np.zeros_like(H0s)
    for name, char in characters.items():
        projector = sum(char[g] * representations[g] for g in group) * (dimensions[name] / len(group))
        basis, idempotence = projector_basis(projector)
        sector_bases[name] = basis
        projector_sum += projector
        projector_checks[name] = {
            "isotypic_rank": int(basis.shape[1]),
            "irrep_dimension": dimensions[name],
            "multiplicity": int(basis.shape[1] // dimensions[name]),
            "projector_idempotence_residual": idempotence,
        }

    cancellation_s = (n_b - 1) / (2 * n_b - 1)
    points = {"s0": 0.0, "s_cancel": cancellation_s, "s1": 1.0}
    sectors: dict[str, Any] = {}
    for name, basis in sector_bases.items():
        H0a = basis.T @ H0s @ basis
        Hpa = basis.T @ Hps @ basis
        Pa = basis.T @ Pcover_s @ basis
        cover_rank = int(np.sum(np.linalg.eigvalsh((Pa + Pa.T) / 2) > 0.5))
        records = {}
        for point_name, path_s in points.items():
            records[point_name] = spectral_record((1 - path_s) * H0a + path_s * Hpa, Pa)
        sectors[name] = {
            **projector_checks[name],
            "cover_subspace_rank": cover_rank,
            "path_points": records,
            "simple_ground_gap_diagnostic": simple_ground_gap_diagnostic(H0a, Hpa, records),
            "final_is_global_ground_sector": False,
        }

    final_floor = min(sectors[name]["path_points"]["s1"]["E0"] for name in sectors)
    final_tol = 1e-10
    for name in sectors:
        sectors[name]["final_is_global_ground_sector"] = bool(
            abs(sectors[name]["path_points"]["s1"]["E0"] - final_floor) < final_tol
        )

    return {
        "schema_version": 1,
        "scope": "D4 isotypic comparison inside the S_k-trivial sector",
        "instance_id": INSTANCE_ID,
        "payload_sha256": record["payload_sha256"],
        "parameters": {"k": k, "lambda": LAM, "mu": MU, "nu": NU},
        "dimensions": {"full_valid": dim, "S_k_trivial": T.shape[1]},
        "D4": {
            "order": len(group),
            "rotation_generator": list(r),
            "reflection_generator": list(s),
            "projector_completeness_residual": float(np.linalg.norm(projector_sum - np.eye(T.shape[1]), ord=2)),
        },
        "sector_results": sectors,
        "conclusion_rule": (
            "An alternative initial irrep can target the final global ground sector, "
            "but an adiabatic rank-one/rank-d guarantee still fails if its ground rank "
            "increases at s_cancel."
        ),
    }


def main() -> None:
    result = analyse()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for name, sector in result["sector_results"].items():
        c = sector["path_points"]["s_cancel"]
        f = sector["path_points"]["s1"]
        print(
            f"{name}: rank={sector['isotypic_rank']:3d}, cover_rank={sector['cover_subspace_rank']:2d}, "
            f"cancel_g0={c['ground_rank']:2d}, final_E0={f['E0']:+.9f}, "
            f"final_g0={f['ground_rank']}, global={sector['final_is_global_ground_sector']}"
        )


if __name__ == "__main__":
    main()
