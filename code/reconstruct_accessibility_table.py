#!/usr/bin/env python3
"""Reconstruct the finite-instance accessibility table from frozen inputs.

This script does not consume any legacy spectral JSON. For every
candidate-primary instance it verifies and loads the frozen incidence
matrix, constructs the full Hamiltonian and the exact joint fixed-space
quotient, closes the cyclic space, and applies residual-aware spectral
tests. The output carries the payload hash and numerical residuals needed
to trace every table entry.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from accessibility_prototype import (
    eigensystem,
    joint_orbits,
    orbit_isometry,
    quotient,
    single_uniform_driver,
)
from instance_registry import load_instance, primary_instance_ids
from spectral_gap_study import build_hamiltonian, _decode_bases
from symmetry_analysis import find_base_automorphisms


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "accessibility_table_reconstructed.json"
TABLE_OUTPUT = ROOT / "sections" / "generated_accessibility_tables.tex"
LAM, MU, NU = 5.0, 10.0, 50.0
RESIDUAL_FACTOR = 50.0


def matrix_scale(H: sp.spmatrix | np.ndarray) -> float:
    """Return a deterministic upper bound on the spectral norm."""
    if sp.issparse(H):
        return float(np.asarray(abs(H).sum(axis=1)).max(initial=0.0))
    return float(np.linalg.norm(H, ord=np.inf))


def cyclic_basis(
    generators: list[np.ndarray],
    initial: np.ndarray,
    relative_tol: float = 1e-10,
) -> tuple[np.ndarray, list[int]]:
    """Close a cyclic space by incrementally expanding only its frontier.

    If Q_j is already invariant under all generators on every column added
    before the latest frontier, only the latest columns can add directions at
    step j+1. Double orthogonalisation controls roundoff; the small SVD acts
    on the frontier residual rather than on the entire accumulated candidate
    matrix.
    """
    basis = initial[:, None] / np.linalg.norm(initial)
    frontier = basis
    dimensions = [1]
    while frontier.shape[1]:
        candidates = np.column_stack([generator @ frontier for generator in generators])
        for _ in range(2):
            candidates -= basis @ (basis.T @ candidates)
        left, singular, _ = np.linalg.svd(candidates, full_matrices=False)
        if singular.size == 0:
            break
        threshold = relative_tol * max(1.0, float(singular[0]))
        added = int(np.sum(singular > threshold))
        if added == 0:
            break
        frontier = left[:, :added]
        basis = np.column_stack([basis, frontier])
        dimensions.append(int(basis.shape[1]))
        if basis.shape[1] == basis.shape[0]:
            break
    return basis, dimensions


def residual_level_summary(
    values: np.ndarray,
    residuals: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    """Classify the lowest manifold using eigenpair residuals and scale."""
    eps_term = np.finfo(float).eps * max(1.0, scale)
    ground_rank = 1
    thresholds = []
    for i in range(1, len(values)):
        threshold = RESIDUAL_FACTOR * (residuals[0] + residuals[i] + eps_term)
        thresholds.append(float(threshold))
        if abs(values[i] - values[0]) <= threshold:
            ground_rank += 1
        else:
            break
    gap = None if ground_rank == len(values) else float(values[ground_rank] - values[0])
    delta_mult = 0.0 if ground_rank > 1 else gap
    return {
        "E0": float(values[0]),
        "ground_rank": int(ground_rank),
        "excitation_gap": gap,
        "multiplicity_gap": delta_mult,
        "comparison_thresholds": thresholds,
        "scale_bound": float(scale),
    }


def integer_partitions(total: int, maximum: int | None = None) -> list[tuple[int, ...]]:
    """Return partitions in lexicographically decreasing order."""
    if total == 0:
        return [()]
    maximum = total if maximum is None else min(maximum, total)
    answer: list[tuple[int, ...]] = []
    for first in range(maximum, 0, -1):
        for tail in integer_partitions(total - first, first):
            answer.append((first,) + tail)
    return answer


def specht_dimension(partition: tuple[int, ...]) -> int:
    """Dimension from the hook-length formula."""
    hooks = 1
    for row, length in enumerate(partition):
        for column in range(length):
            below = sum(int(other > column) for other in partition[row + 1 :])
            hooks *= length - column + below
    return math.factorial(sum(partition)) // hooks


def transposition_class_scalar(partition: tuple[int, ...]) -> int:
    """Content sum, the scalar of the transposition class sum."""
    return sum(column - row for row, length in enumerate(partition) for column in range(length))


def class_sum_apply(vector: np.ndarray, n_b: int, k: int) -> np.ndarray:
    """Apply the sum of all register transpositions."""
    tensor = vector.reshape((n_b,) * k)
    result = np.zeros_like(tensor)
    for left in range(k):
        for right in range(left + 1, k):
            axes = list(range(k))
            axes[left], axes[right] = axes[right], axes[left]
            result += np.transpose(tensor, axes)
    return result.reshape(-1)


def class_project(
    vector: np.ndarray,
    target: int,
    scalars: list[int],
    n_b: int,
    k: int,
) -> np.ndarray:
    """Apply the exact polynomial projector associated with one class scalar."""
    projected = vector.copy()
    for other in scalars:
        if other == target:
            continue
        projected = (
            class_sum_apply(projected, n_b, k) - other * projected
        ) / (target - other)
    return projected


def register_permute(
    vector: np.ndarray,
    axes: tuple[int, ...],
    n_b: int,
    k: int,
) -> np.ndarray:
    """Apply one register permutation to a state vector."""
    return np.transpose(vector.reshape((n_b,) * k), axes).reshape(-1)


def specht_orbit_basis(
    representative: np.ndarray,
    expected_dimension: int,
    n_b: int,
    k: int,
    existing: np.ndarray | None = None,
) -> np.ndarray:
    """Generate one Specht multiplet from a representative eigenvector."""
    columns = []
    fixed = (
        np.empty((representative.size, 0), dtype=float)
        if existing is None
        else existing
    )
    for axes in itertools.permutations(range(k)):
        candidate = register_permute(representative, axes, n_b, k)
        for _ in range(2):
            if fixed.shape[1]:
                candidate -= fixed @ (fixed.T @ candidate)
            if columns:
                block = np.column_stack(columns)
                candidate -= block @ (block.T @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-7:
            columns.append(candidate / norm)
            if len(columns) == expected_dimension:
                break
    if len(columns) != expected_dimension:
        raise RuntimeError(
            f"register orbit has rank {len(columns)}, expected {expected_dimension}"
        )
    return np.column_stack(columns)


def dense_low_eigensystem(
    H: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the full dense spectrum for the small cases."""
    dim = H.shape[0]
    dense = H.toarray()
    values, vectors = np.linalg.eigh(dense)
    residuals = np.linalg.norm(dense @ vectors - vectors * values[None, :], axis=0)
    summary = residual_level_summary(values, residuals, matrix_scale(dense))
    summary["solver"] = "numpy.linalg.eigh(full dense spectrum)"
    summary["reported_eigenpairs"] = int(dim)
    return values, vectors, residuals, summary


def dense_ground_irreps(
    vectors: np.ndarray,
    ground_rank: int,
    n_b: int,
    k: int,
) -> list[dict[str, Any]]:
    """Classify a dense ground manifold by central S_k projectors."""
    ground = vectors[:, :ground_rank]
    partitions = integer_partitions(k)
    scalars = [transposition_class_scalar(partition) for partition in partitions]
    records = []
    total_rank = 0
    for partition, scalar in zip(partitions, scalars):
        projected = np.column_stack([
            class_project(ground[:, column], scalar, scalars, n_b, k)
            for column in range(ground_rank)
        ])
        singular = np.linalg.svd(projected, compute_uv=False)
        rank = int(np.sum(singular > 1e-8))
        if rank == 0:
            continue
        dimension = specht_dimension(partition)
        if rank % dimension:
            raise RuntimeError(
                f"dense ground rank {rank} is incompatible with S_{k} irrep {partition}"
            )
        records.append({
            "partition": list(partition),
            "specht_dimension": dimension,
            "multiplicity_space_ground_rank": rank // dimension,
        })
        total_rank += rank
    if total_rank != ground_rank:
        raise RuntimeError(
            f"dense S_{k} ground classification has rank {total_rank}, expected {ground_rank}"
        )
    return records


def sector_resolved_low_spectrum(
    H: sp.csr_matrix,
    n_b: int,
    k: int,
    initial_support: np.ndarray,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Resolve the low spectrum in every S_k isotypic sector.

    The transposition class sum has a different integer scalar for every
    partition of k when k <= 5. A polynomial in that class sum therefore
    supplies an exact central projector without constructing character
    tables. Lanczos starts in each projected sector separately; class-sum
    residuals verify that numerical drift did not change sectors.
    """
    partitions = integer_partitions(k)
    scalars = [transposition_class_scalar(partition) for partition in partitions]
    if len(set(scalars)) != len(scalars):
        raise ValueError(f"transposition class does not separate all S_{k} irreps")
    scale = matrix_scale(H)
    rng = np.random.default_rng(20260811)
    seed = np.zeros(H.shape[0], dtype=float)
    seed[initial_support] = rng.standard_normal(int(initial_support.sum()))
    sector_records = []
    retained_vectors: list[np.ndarray] = []

    def make_sector_operator(scalar: int) -> spla.LinearOperator:
        complement_shift = scale + 1.0

        def sector_matvec(vector: np.ndarray) -> np.ndarray:
            flat = np.asarray(vector).reshape(-1)
            sector_part = class_project(flat, scalar, scalars, n_b, k)
            return H @ sector_part + complement_shift * (flat - sector_part)

        return spla.LinearOperator(
            H.shape, matvec=sector_matvec, rmatvec=sector_matvec, dtype=float
        )

    def solve_sector(
        partition: tuple[int, ...],
        scalar: int,
        projected: np.ndarray,
        count: int,
    ) -> tuple[list[dict[str, float]], list[np.ndarray]]:
        sector_operator = make_sector_operator(scalar)
        values, vectors = spla.eigsh(
            sector_operator,
            k=count,
            which="SA",
            v0=projected,
            tol=1e-8,
            maxiter=200_000,
            ncv=min(H.shape[0], max(20, 4 * count + 1)),
        )
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        residuals = np.linalg.norm(H @ vectors - vectors * values[None, :], axis=0)
        class_residuals = np.asarray([
            np.linalg.norm(class_sum_apply(vectors[:, i], n_b, k) - scalar * vectors[:, i])
            for i in range(count)
        ])
        accepted = np.where(class_residuals <= 1e-6)[0]
        if len(accepted) < 1:
            raise RuntimeError(
                f"S_{k} irrep {partition}: only {len(accepted)} sector-pure eigenpairs"
            )
        distinct = []
        distinct_vectors = []
        for index in accepted:
            if distinct:
                tolerance = RESIDUAL_FACTOR * (
                    distinct[-1]["residual"]
                    + float(residuals[index])
                    + np.finfo(float).eps * max(1.0, scale)
                )
                if abs(float(values[index]) - distinct[-1]["energy"]) <= tolerance:
                    continue
            distinct.append({
                "energy": float(values[index]),
                "residual": float(residuals[index]),
                "class_sum_residual": float(class_residuals[index]),
            })
            distinct_vectors.append(vectors[:, index].copy())
        return distinct, distinct_vectors

    # First pass: only the lowest level in every sector.  The global
    # excitation can only be the lowest level of a non-ground sector or the
    # second distinct level of a ground sector, so computing several levels
    # in every high sector is unnecessary.
    projected_seeds: dict[tuple[int, ...], np.ndarray] = {}
    lowest_vectors: dict[tuple[int, ...], np.ndarray] = {}
    for partition, scalar in zip(partitions, scalars):
        projected = class_project(seed, scalar, scalars, n_b, k)
        projection_norm = float(np.linalg.norm(projected))
        if projection_norm < 1e-10:
            raise RuntimeError(f"zero initial projection for S_{k} irrep {partition}")
        projected /= projection_norm
        projected_seeds[partition] = projected
        distinct, vectors = solve_sector(partition, scalar, projected, 1)
        lowest_vectors[partition] = vectors[0]
        sector_records.append({
            "partition": list(partition),
            "dimension": specht_dimension(partition),
            "transposition_class_scalar": scalar,
            "initial_projection_norm": projection_norm,
            "levels": distinct,
        })

    ground_energy = min(record["levels"][0]["energy"] for record in sector_records)
    ground_candidates = []
    for record in sector_records:
        first = record["levels"][0]
        tolerance = RESIDUAL_FACTOR * (
            first["residual"] + np.finfo(float).eps * max(1.0, scale)
        )
        if abs(first["energy"] - ground_energy) <= tolerance:
            ground_candidates.append(record)

    # Second pass only in sectors attaining the global ground energy.  The
    # register orbit of one representative gives the full Specht multiplet.
    # Deflating that orbit then detects an additional cospectral copy, if one
    # exists, without relying on solver-selected vectors in a degenerate block.
    ground_certifications = []
    for record in ground_candidates:
        partition = tuple(record["partition"])
        scalar = int(record["transposition_class_scalar"])
        specht_dim = int(record["dimension"])
        e0 = float(record["levels"][0]["energy"])
        ground_basis = specht_orbit_basis(
            lowest_vectors[partition], specht_dim, n_b, k
        )
        multiplicity = 1
        next_energy = None
        next_residual = None
        next_class_residual = None
        while multiplicity <= 4:
            base_operator = make_sector_operator(scalar)
            deflation_penalty = scale + 1.0

            def deflated_matvec(vector: np.ndarray) -> np.ndarray:
                flat = np.asarray(vector).reshape(-1)
                return (
                    base_operator @ flat
                    + deflation_penalty * ground_basis @ (ground_basis.T @ flat)
                )

            deflated = spla.LinearOperator(
                H.shape,
                matvec=deflated_matvec,
                rmatvec=deflated_matvec,
                dtype=float,
            )
            trial = projected_seeds[partition].copy()
            for _ in range(2):
                trial -= ground_basis @ (ground_basis.T @ trial)
            if np.linalg.norm(trial) < 1e-10:
                trial = class_project(
                    np.random.default_rng(20260811 + multiplicity).standard_normal(
                        H.shape[0]
                    ),
                    scalar,
                    scalars,
                    n_b,
                    k,
                )
                for _ in range(2):
                    trial -= ground_basis @ (ground_basis.T @ trial)
            trial /= np.linalg.norm(trial)
            _, vectors = spla.eigsh(
                deflated,
                k=1,
                which="SA",
                v0=trial,
                tol=1e-8,
                maxiter=200_000,
                ncv=min(H.shape[0], 20),
            )
            candidate = vectors[:, 0]
            energy = float(candidate @ (H @ candidate))
            residual = float(np.linalg.norm(H @ candidate - energy * candidate))
            class_residual = float(
                np.linalg.norm(
                    class_sum_apply(candidate, n_b, k) - scalar * candidate
                )
            )
            tolerance = RESIDUAL_FACTOR * (
                record["levels"][0]["residual"]
                + residual
                + np.finfo(float).eps * max(1.0, scale)
            )
            if abs(energy - e0) > tolerance:
                next_energy = energy
                next_residual = residual
                next_class_residual = class_residual
                break
            new_multiplet = specht_orbit_basis(
                candidate, specht_dim, n_b, k, existing=ground_basis
            )
            ground_basis = np.column_stack([ground_basis, new_multiplet])
            multiplicity += 1
        if next_energy is None:
            raise RuntimeError(
                f"S_{k} ground irrep {partition}: multiplicity exceeded audit cap"
            )
        ground_residuals = np.linalg.norm(
            H @ ground_basis - ground_basis * e0,
            axis=0,
        )
        ground_class_residuals = np.asarray([
            np.linalg.norm(
                class_sum_apply(ground_basis[:, column], n_b, k)
                - scalar * ground_basis[:, column]
            )
            for column in range(ground_basis.shape[1])
        ])
        certification = {
            "E0": e0,
            "ground_rank": int(ground_basis.shape[1]),
            "excitation_gap": float(next_energy - e0),
            "multiplicity_gap": (
                0.0 if ground_basis.shape[1] > 1 else float(next_energy - e0)
            ),
            "partition": list(partition),
            "specht_dimension": specht_dim,
            "multiplicity_space_ground_rank": multiplicity,
            "max_ground_residual": float(ground_residuals.max()),
            "next_level_residual": next_residual,
            "max_reported_residual": float(
                max(ground_residuals.max(), next_residual)
            ),
            "max_class_sum_residual": float(
                max(ground_class_residuals.max(), next_class_residual)
            ),
            "deflated_multiplets": multiplicity,
        }
        ground_certifications.append(certification)
        retained_vectors.extend(
            ground_basis[:, column].copy()
            for column in range(ground_basis.shape[1])
        )

    ground_energy = min(record["E0"] for record in ground_certifications)
    ground_rank = sum(record["ground_rank"] for record in ground_certifications)
    ground_labels = [
        {
            "partition": record["partition"],
            "specht_dimension": record["specht_dimension"],
            "multiplicity_space_ground_rank": record["multiplicity_space_ground_rank"],
        }
        for record in ground_certifications
    ]

    excitation_candidates = []
    for record in sector_records:
        first = record["levels"][0]["energy"]
        if first - ground_energy > 0:
            excitation_candidates.append(first)
    excitation_candidates.extend(
        record["E0"] + record["excitation_gap"]
        for record in ground_certifications
        if record["excitation_gap"] is not None
    )
    energy_tolerance = RESIDUAL_FACTOR * (
        max(record["max_reported_residual"] for record in ground_certifications)
        + np.finfo(float).eps * max(1.0, scale)
    )
    next_energy = min(
        energy
        for energy in excitation_candidates
        if energy - ground_energy > energy_tolerance
    )
    summary = {
        "E0": ground_energy,
        "ground_rank": int(ground_rank),
        "ground_irreps": ground_labels,
        "excitation_gap": float(next_energy - ground_energy),
        "multiplicity_gap": 0.0 if ground_rank > 1 else float(next_energy - ground_energy),
        "scale_bound": scale,
        "solver": (
            "S_k sectors separated by the transposition class sum; "
            "scipy.sparse.linalg.eigsh in each isotypic sector"
        ),
        "spectral_strategy": (
            "one lowest Ritz value in every sector; S_k-orbit multiplet "
            "construction and deflation in every ground-energy sector"
        ),
        "sector_spectra": sector_records,
        "ground_sector_certifications": ground_certifications,
        "max_ground_residual": max(
            record["max_ground_residual"] for record in ground_certifications
        ),
        "max_reported_residual": max(
            record["max_reported_residual"] for record in ground_certifications
        ),
        "max_class_sum_residual": max(
            record["max_class_sum_residual"] for record in ground_certifications
        ),
    }
    return summary, retained_vectors


def ground_projector_probability(
    vectors: np.ndarray,
    ground_rank: int,
    diagonal_projector: np.ndarray,
) -> tuple[float, float]:
    """Return min/max projector probability over a numerical ground manifold."""
    basis, _ = np.linalg.qr(vectors[:, :ground_rank])
    compressed = basis.T @ (diagonal_projector[:, None] * basis)
    eigenvalues = np.linalg.eigvalsh((compressed + compressed.T) / 2)
    return float(eigenvalues[0]), float(eigenvalues[-1])


def ordered_cover_mask(A: np.ndarray, k: int) -> np.ndarray:
    n_b = A.shape[0]
    decoded = _decode_bases(n_b**k, k, n_b)
    distinct = np.asarray([len(set(map(int, row))) == k for row in decoded])
    covered = np.asarray([A[row].any(axis=0).all() for row in decoded])
    return distinct & covered


def cover_orbit_count(
    covers: list[list[int]],
    base_group: list[tuple[int, ...]],
) -> int:
    remaining = {tuple(sorted(map(int, cover))) for cover in covers}
    count = 0
    while remaining:
        representative = next(iter(remaining))
        orbit = {
            tuple(sorted(g[index] for index in representative))
            for g in base_group
        }
        remaining.difference_update(orbit)
        count += 1
    return count


def clean_nonnegative(value: float, tolerance: float) -> float:
    if value < 0 and abs(value) <= tolerance:
        return 0.0
    return value


def analyse_instance(instance_id: str) -> dict[str, Any]:
    A, costs, record = load_instance(instance_id, require_status="candidate_primary")
    if costs is not None:
        raise ValueError(f"{instance_id}: weighted records are outside this table")
    n_b, n_t = map(int, A.shape)
    k = int(record["k_star"])
    if n_b & (n_b - 1):
        raise ValueError(f"{instance_id}: this reconstruction expects no padding")

    Hprob, _ = build_hamiltonian(A, k, LAM, MU, NU)
    H0 = single_uniform_driver(n_b, k)
    dim = n_b**k
    group = find_base_automorphisms(A)
    print(f"  group: |G_B|={len(group)}", flush=True)
    orbits = joint_orbits(n_b, k, group)
    isometry = orbit_isometry(orbits, dim)
    H0q = quotient(H0, isometry)
    Hpq = quotient(Hprob, isometry)
    print(f"  quotient: {dim} -> {len(orbits)}", flush=True)

    uniform = np.ones(dim, dtype=float) / math.sqrt(dim)
    uniform_q = np.asarray(isometry.T @ uniform).ravel()
    cyclic, growth = cyclic_basis([H0q, Hpq], uniform_q)
    print(f"  cyclic closure: {growth}", flush=True)
    cyclic_residuals = {
        "H0": float(np.linalg.norm((np.eye(len(orbits)) - cyclic @ cyclic.T) @ H0q @ cyclic, ord=2)),
        "Hprob": float(np.linalg.norm((np.eye(len(orbits)) - cyclic @ cyclic.T) @ Hpq @ cyclic, ord=2)),
    }
    quotient_residuals = {
        "H0": float(sp.linalg.norm(H0 @ isometry - isometry @ sp.csr_matrix(H0q))),
        "Hprob": float(sp.linalg.norm(Hprob @ isometry - isometry @ sp.csr_matrix(Hpq))),
    }

    mask = ordered_cover_mask(A, k)
    orbit_cover = np.asarray([bool(mask[orbit[0]]) for orbit in orbits], dtype=float)

    if dim <= 768:
        full_values, full_vectors, full_residuals, full_summary = dense_low_eigensystem(
            Hprob
        )
        full_rank = int(full_summary["ground_rank"])
        full_pmin, full_pmax = ground_projector_probability(
            full_vectors, full_rank, mask.astype(float)
        )
        full_summary["cover_probability_min"] = full_pmin
        full_summary["cover_probability_max"] = full_pmax
        full_summary["ground_irreps"] = dense_ground_irreps(
            full_vectors, full_rank, n_b, k
        )
        full_summary["max_ground_residual"] = float(full_residuals[:full_rank].max())
        full_summary["max_reported_residual"] = float(full_residuals.max())
        print(
            f"  full dense spectrum: {full_summary['reported_eigenpairs']} eigenpairs",
            flush=True,
        )
    else:
        if len(group) != 1:
            raise ValueError(
                f"{instance_id}: sparse sector solver also requires resolving nontrivial G_B"
            )
        full_summary, ground_representatives = sector_resolved_low_spectrum(
            Hprob, n_b, k, mask
        )
        ground_matrix = np.column_stack(ground_representatives)
        full_pmin, full_pmax = ground_projector_probability(
            ground_matrix, ground_matrix.shape[1], mask.astype(float)
        )
        full_summary["cover_probability_min"] = full_pmin
        full_summary["cover_probability_max"] = full_pmax
        print(
            f"  full sector spectrum: {len(full_summary['sector_spectra'])} S_k irreps",
            flush=True,
        )

    Hacc = cyclic.T @ Hpq @ cyclic
    acc_values, acc_vectors, acc_residuals = eigensystem(Hacc)
    acc_summary = residual_level_summary(acc_values, acc_residuals, matrix_scale(Hacc))
    acc_rank = int(acc_summary["ground_rank"])
    projector_acc = cyclic.T @ (orbit_cover[:, None] * cyclic)
    acc_basis, _ = np.linalg.qr(acc_vectors[:, :acc_rank])
    acc_projected = acc_basis.T @ projector_acc @ acc_basis
    acc_probabilities = np.linalg.eigvalsh((acc_projected + acc_projected.T) / 2)
    acc_summary["cover_probability_min"] = float(acc_probabilities[0])
    acc_summary["cover_probability_max"] = float(acc_probabilities[-1])
    acc_summary["max_ground_residual"] = float(acc_residuals[:acc_rank].max())
    acc_summary["max_reported_residual"] = float(acc_residuals.max())
    acc_summary["solver"] = "exact orbit quotient + cyclic restriction + numpy.linalg.eigh"

    energy_tolerance = RESIDUAL_FACTOR * (
        full_summary["max_ground_residual"]
        + acc_summary["max_ground_residual"]
        + np.finfo(float).eps
        * max(full_summary["scale_bound"], acc_summary["scale_bound"], 1.0)
    )
    delta_accessible = clean_nonnegative(
        float(acc_summary["E0"] - full_summary["E0"]),
        float(energy_tolerance),
    )

    cancellation_s = (n_b - 1) / (2 * n_b - 1)
    Hcancel = (1 - cancellation_s) * H0q + cancellation_s * Hpq
    cancel_values, _, cancel_residuals = eigensystem(Hcancel)
    cancel_summary = residual_level_summary(
        cancel_values, cancel_residuals, matrix_scale(Hcancel)
    )
    cancel_rank = int(cancel_summary["ground_rank"])
    cancel_summary["max_ground_residual"] = float(cancel_residuals[:cancel_rank].max())
    cancel_summary["off_diagonal_norm"] = float(
        np.linalg.norm(Hcancel - np.diag(np.diag(Hcancel)), ord=2)
    )

    frozen_path = ROOT / "instances" / "frozen" / f"{instance_id}.json"
    optimal_covers = record["optimal_cardinality_covers"]
    n_cover_orbits = cover_orbit_count(optimal_covers, group)
    return {
        "instance_id": instance_id,
        "payload_sha256": record["payload_sha256"],
        "frozen_file_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "n_B": n_b,
        "n_T": n_t,
        "k_star": k,
        "parameters": {"lambda": LAM, "mu": MU, "nu": NU},
        "represented_group": {
            "register_group": f"S_{k}",
            "base_group_order": len(group),
            "base_permutations_zero_based": [list(g) for g in group],
        },
        "dimensions": {
            "full_valid": dim,
            "symmetric_before_GB": math.comb(n_b + k - 1, k),
            "joint_fixed": len(orbits),
            "cyclic": int(cyclic.shape[1]),
        },
        "optimal_covers": {
            "count": int(record["n_optimal_cardinality_covers"]),
            "G_B_orbit_count": n_cover_orbits,
        },
        "cyclic_growth": growth,
        "checks": {
            "orbit_isometry_error": float(
                np.linalg.norm(
                    (isometry.T @ isometry).toarray() - np.eye(len(orbits)),
                    ord=2,
                )
            ),
            "quotient_invariance_residual": quotient_residuals,
            "cyclic_invariance_residual": cyclic_residuals,
        },
        "problem_endpoint": {
            "full": full_summary,
            "accessible": acc_summary,
            "delta_accessible": delta_accessible,
            "delta_accessible_tolerance": float(energy_tolerance),
        },
        "linear_path_cancellation": {
            "s": cancellation_s,
            "joint_fixed": cancel_summary,
            "ground_rank_matches_optimal_cover_orbits": bool(
                cancel_summary["ground_rank"] == n_cover_orbits
            ),
        },
    }


def fmt_float(value: float | None) -> str:
    if value is None:
        return "--"
    if value == 0:
        return "0"
    return f"{value:.3g}"


def latex_label(instance_id: str) -> str:
    label = instance_id.removesuffix("-v1")
    if label.startswith("scpe"):
        label = label.removesuffix("-s0")
    return label.replace("random-", "rnd-")


def latex_partition_content(records: list[dict[str, Any]]) -> str:
    pieces = []
    for record in records:
        partition = "[" + ",".join(map(str, record["partition"])) + "]"
        multiplicity = int(record["multiplicity_space_ground_rank"])
        pieces.append(partition if multiplicity == 1 else f"{multiplicity}{partition}")
    return "$" + r"\oplus".join(pieces) + "$"


def render_tables(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Generated by code/reconstruct_accessibility_table.py; do not edit.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Exact structural reduction for the frozen finite instances. "
        r"All dimensions refer to the physical, unpadded address space.}",
        r"\label{tab:finite-structure}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"instance & $k$ & $|G_B|$ & $\dim\mathcal H_{\rm valid}$ & "
        r"$\dim\mathcal H_{\rm sym}$ & $\dim\mathcal K_u$ & cover orbits\\",
        r"\hline",
    ]
    for row in rows:
        dims = row["dimensions"]
        lines.append(
            f"{latex_label(row['instance_id'])} & {row['k_star']} & "
            f"{row['represented_group']['base_group_order']} & "
            f"{dims['full_valid']} & {dims['joint_fixed']} & {dims['cyclic']} & "
            f"{row['optimal_covers']['G_B_orbit_count']}\\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Problem-endpoint spectra reconstructed from the full Hamiltonian "
        r"and the exact cyclic restriction. Here $\Delta$ is the gap above the "
        r"entire ground manifold, and $p_{\rm opt}^{\min}$ is minimized over "
        r"normalized accessible ground states. The last column is the joint-fixed "
        r"ground rank at the analytic hopping-cancellation point.}",
        r"\label{tab:finite-endpoints}",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\hline",
        r"instance & ground $S_k$ content & $g_0^{\rm full}$ & $\Delta_{\rm full}$ & $\delta_{\rm acc}$ & "
        r"$g_0^{\rm acc}$ & $\Delta_{\rm acc}$ & $p_{\rm opt}^{\min}$ & "
        r"$g_0^{\rm sym}(s_c)$\\",
        r"\hline",
    ])
    for row in rows:
        endpoint = row["problem_endpoint"]
        full = endpoint["full"]
        acc = endpoint["accessible"]
        cancel = row["linear_path_cancellation"]["joint_fixed"]
        lines.append(
            f"{latex_label(row['instance_id'])} & "
            f"{latex_partition_content(full['ground_irreps'])} & "
            f"{full['ground_rank']} & "
            f"{fmt_float(full['excitation_gap'])} & "
            f"{fmt_float(endpoint['delta_accessible'])} & "
            f"{acc['ground_rank']} & {fmt_float(acc['excitation_gap'])} & "
            f"{acc['cover_probability_min']:.4f} & {cancel['ground_rank']}\\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance",
        action="append",
        dest="instances",
        help="reconstruct only this candidate-primary instance (repeatable)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and require byte-identical JSON and LaTeX outputs",
    )
    args = parser.parse_args()
    instance_ids = args.instances or primary_instance_ids()
    rows = []
    for position, instance_id in enumerate(instance_ids, start=1):
        print(f"[{position}/{len(instance_ids)}] {instance_id}", flush=True)
        row = analyse_instance(instance_id)
        rows.append(row)
        endpoint = row["problem_endpoint"]
        print(
            f"  dims={row['dimensions']}  "
            f"g_full={endpoint['full']['ground_rank']}  "
            f"Delta_full={fmt_float(endpoint['full']['excitation_gap'])}  "
            f"delta_acc={fmt_float(endpoint['delta_accessible'])}  "
            f"g_acc={endpoint['accessible']['ground_rank']}  "
            f"Delta_acc={fmt_float(endpoint['accessible']['excitation_gap'])}",
            flush=True,
        )

    document = {
        "schema_version": 1,
        "scope": "frozen candidate-primary unweighted instances",
        "method": (
            "full sparse/dense Hamiltonian plus exact S_k x G_B orbit quotient; "
            "no PHP or Lowdin values"
        ),
        "residual_factor": RESIDUAL_FACTOR,
        "instances": rows,
    }
    json_text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    table_text = render_tables(rows)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != json_text:
            raise SystemExit(f"{OUTPUT.relative_to(ROOT)} is absent or stale")
        if not TABLE_OUTPUT.exists() or TABLE_OUTPUT.read_text() != table_text:
            raise SystemExit(f"{TABLE_OUTPUT.relative_to(ROOT)} is absent or stale")
        print("Reconstruction outputs verified byte-for-byte.")
    else:
        OUTPUT.write_text(json_text)
        TABLE_OUTPUT.write_text(table_text)
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        print(f"Wrote {TABLE_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
