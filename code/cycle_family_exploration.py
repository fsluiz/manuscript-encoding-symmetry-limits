#!/usr/bin/env python3
"""Joint-orbit exploration of the even-cycle vertex-cover MSC family.

For C_{2m}, bases are vertices and tasks are edges; a base covers its two
incident edges.  The minimum cover size is m and the two alternating optimum
covers form one orbit under D_{2m}.  The script works directly in the
S_k-symmetric occupation-number basis and then quotients by D_{2m}; it never
constructs the n_B^k ordered-register Hilbert space.

This is exploratory finite-size evidence for an asymptotic theorem, not the
theorem itself.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "cycle_family_results.json"
LAM, MU = 5.0, 10.0
EVEN_SIZES = [4, 6, 8, 10, 12]
HARD_CORE_EXTENSION_SIZES = [14, 16]
SPARSE_HARD_CORE_EXTENSION_SIZES = [18]


def weak_compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    """Yield all length-parts nonnegative tuples summing to total."""
    for separators in itertools.combinations(range(total + parts - 1), parts - 1):
        boundaries = (-1,) + separators + (total + parts - 1,)
        yield tuple(boundaries[i + 1] - boundaries[i] - 1 for i in range(parts))


def cycle_incidence(n: int) -> np.ndarray:
    """A[b,t]=1 when vertex b is incident to cycle edge t=(t,t+1)."""
    A = np.zeros((n, n), dtype=np.int8)
    for t in range(n):
        A[t, t] = 1
        A[(t + 1) % n, t] = 1
    return A


def dihedral_group(n: int) -> list[tuple[int, ...]]:
    elements = set()
    for shift in range(n):
        elements.add(tuple((b + shift) % n for b in range(n)))
        elements.add(tuple((shift - b) % n for b in range(n)))
    return sorted(elements)


def symmetric_matrices(n: int, k: int, lam: float, mu: float) -> tuple[list[tuple[int, ...]], sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Build H0, Hprob and the cover mask in the bosonic occupation basis."""
    states = list(weak_compositions(k, n))
    index = {state: i for i, state in enumerate(states)}
    rows: list[int] = []
    cols: list[int] = []
    walk_data: list[float] = []
    driver_data: list[float] = []
    diag_penalty = np.zeros(len(states))
    cover = np.zeros(len(states), dtype=bool)

    for source, state in enumerate(states):
        uncovered = sum(state[t] == 0 and state[(t + 1) % n] == 0 for t in range(n))
        duplicates = sum(c * (c - 1) // 2 for c in state)
        diag_penalty[source] = lam * uncovered + mu * duplicates
        cover[source] = uncovered == 0 and duplicates == 0
        for b, population in enumerate(state):
            if population == 0:
                continue
            for a in range(n):
                if a == b:
                    continue
                target_state = list(state)
                target_state[b] -= 1
                target_state[a] += 1
                target = index[tuple(target_state)]
                amplitude = math.sqrt(population * (state[a] + 1))
                rows.append(target)
                cols.append(source)
                walk_data.append(amplitude / (n - 1))
                driver_data.append(-amplitude / n)

    shape = (len(states), len(states))
    Hwalk = sp.csr_matrix((walk_data, (rows, cols)), shape=shape)
    off_driver = sp.csr_matrix((driver_data, (rows, cols)), shape=shape)
    H0 = off_driver + sp.eye(len(states), format="csr") * (k * (1 - 1 / n))
    Hprob = Hwalk + sp.diags(diag_penalty, format="csr")
    return states, H0, Hprob, cover


def occupancy_orbits(states: list[tuple[int, ...]], group: list[tuple[int, ...]]) -> list[list[int]]:
    index = {state: i for i, state in enumerate(states)}
    assigned = np.zeros(len(states), dtype=bool)
    orbits: list[list[int]] = []
    for source, state in enumerate(states):
        if assigned[source]:
            continue
        orbit = set()
        for g in group:
            image = [0] * len(state)
            for b, count in enumerate(state):
                image[g[b]] = count
            orbit.add(index[tuple(image)])
        members = sorted(orbit)
        assigned[members] = True
        orbits.append(members)
    if not assigned.all():
        raise AssertionError("occupancy orbits failed to partition the basis")
    return orbits


def hard_core_matrices(n: int, k: int, lam: float) -> tuple[list[tuple[int, ...]], sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Projected distinct-base model on k-subsets (Johnson-graph hopping)."""
    states = list(itertools.combinations(range(n), k))
    index = {state: i for i, state in enumerate(states)}
    rows, cols, data = [], [], []
    penalty = np.zeros(len(states))
    cover = np.zeros(len(states), dtype=bool)
    universe = set(range(n))
    for source, state in enumerate(states):
        selected = set(state)
        uncovered = sum(t not in selected and (t + 1) % n not in selected for t in range(n))
        penalty[source] = lam * uncovered
        cover[source] = uncovered == 0
        for removed in state:
            for added in universe - selected:
                target = tuple(sorted((selected - {removed}) | {added}))
                rows.append(index[target])
                cols.append(source)
                data.append(1 / (n - 1))
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(len(states), len(states)))
    H0 = -(n - 1) / n * adjacency + sp.eye(len(states), format="csr") * (k * (1 - 1 / n))
    Hprob = adjacency + sp.diags(penalty, format="csr")
    return states, H0, Hprob, cover


def hard_core_spin_identity(
    n: int,
    states: list[tuple[int, ...]],
    H0: sp.csr_matrix,
    Hprob: sp.csr_matrix,
    lam: float,
) -> dict[str, Any]:
    """Certify the exact Johnson/collective-XY plus cycle-Ising identity.

    In the half-filled hard-core space, U=N_00=(n+sum_i z_i z_{i+1})/4.
    The unweighted hopping matrix is the Johnson adjacency A_J.  Hence

      H(s)=c(s)I-tau(s)A_J+s*lam/4*(n I+sum_i Z_i Z_{i+1}).

    The reported residual is a direct matrix-level check at several s values.
    """
    k = n // 2
    uncovered = np.asarray(
        [
            sum(t not in state and (t + 1) % n not in state for t in range(n))
            for state in states
        ],
        dtype=float,
    )
    zz_cycle = np.empty(len(states), dtype=float)
    for index, state in enumerate(states):
        occupied = set(state)
        z = np.asarray([1.0 if site in occupied else -1.0 for site in range(n)])
        zz_cycle[index] = float(sum(z[t] * z[(t + 1) % n] for t in range(n)))

    potential_identity_residual = float(np.max(np.abs(uncovered - (n + zz_cycle) / 4)))
    U = sp.diags(uncovered, format="csr")
    AJ = (n - 1) * (Hprob - lam * U)
    checks = []
    for s in (0.0, 1 / 6, (n - 1) / (2 * n - 1), 1.0):
        c = (1 - s) * k * (1 - 1 / n)
        tau = (1 - s) / n - s / (n - 1)
        lhs = (1 - s) * H0 + s * Hprob
        rhs = (
            sp.eye(len(states), format="csr") * (c + s * lam * n / 4)
            - tau * AJ
            + sp.diags(s * lam * zz_cycle / 4, format="csr")
        )
        residual = lhs - rhs
        max_abs = float(np.max(np.abs(residual.data))) if residual.nnz else 0.0
        checks.append({"s": s, "tau": tau, "max_abs_matrix_residual": max_abs})

    return {
        "identity": "H_hc(s)=c(s)I-tau(s)A_J+s*lambda/4*(nI+sum_i Z_i Z_{i+1})",
        "c_of_s": "(1-s)k(1-1/n)",
        "tau_of_s": "(1-s)/n-s/(n-1)",
        "magnetization_sector": "sum_i Z_i=0",
        "johnson_adjacency": "A_J of J(n,n/2), equivalently collective XY hopping",
        "potential_identity_max_abs_residual": potential_identity_residual,
        "matrix_checks": checks,
    }


def subset_orbits(states: list[tuple[int, ...]], group: list[tuple[int, ...]]) -> list[list[int]]:
    index = {state: i for i, state in enumerate(states)}
    assigned = np.zeros(len(states), dtype=bool)
    orbits: list[list[int]] = []
    for source, state in enumerate(states):
        if assigned[source]:
            continue
        members = sorted({index[tuple(sorted(g[b] for b in state))] for g in group})
        assigned[members] = True
        orbits.append(members)
    if not assigned.all():
        raise AssertionError("subset orbits failed to partition the hard-core basis")
    return orbits


def orbit_isometry(orbits: list[list[int]], dimension: int) -> sp.csr_matrix:
    rows, cols, data = [], [], []
    for column, orbit in enumerate(orbits):
        value = 1 / math.sqrt(len(orbit))
        rows.extend(orbit)
        cols.extend([column] * len(orbit))
        data.extend([value] * len(orbit))
    return sp.csr_matrix((data, (rows, cols)), shape=(dimension, len(orbits)))


def cyclic_basis(generators: list[np.ndarray], initial: np.ndarray, relative_tol: float = 1e-11) -> tuple[np.ndarray, list[int]]:
    """Smallest numerically resolved common invariant space containing initial."""
    basis = initial[:, None] / np.linalg.norm(initial)
    growth = [1]
    while True:
        candidates = np.column_stack([basis] + [G @ basis for G in generators])
        U, singular, _ = np.linalg.svd(candidates, full_matrices=False)
        rank = int(np.sum(singular > relative_tol * singular[0]))
        enlarged = U[:, :rank]
        growth.append(rank)
        if rank == basis.shape[1]:
            return enlarged, growth
        basis = enlarged


def lowest_two(H: np.ndarray) -> np.ndarray:
    return la.eigh(H, subset_by_index=[0, 1], eigvals_only=True)


def minimize_gap(H0: np.ndarray, H1: np.ndarray, cancellation_s: float) -> dict[str, Any]:
    def gap(s: float) -> float:
        values = lowest_two((1 - s) * H0 + s * H1)
        return float(max(0.0, values[1] - values[0]))

    grid = np.linspace(0.0, 1.0, 401)
    gaps = np.asarray([gap(float(s)) for s in grid])
    candidates = [0, len(grid) - 1, int(np.argmin(gaps))]
    candidates.extend(i for i in range(1, len(grid) - 1) if gaps[i] <= gaps[i - 1] and gaps[i] <= gaps[i + 1])
    best_s, best_gap = float(grid[int(np.argmin(gaps))]), float(gaps.min())
    for i in sorted(set(candidates)):
        left, right = float(grid[max(0, i - 1)]), float(grid[min(len(grid) - 1, i + 1)])
        if left == right:
            continue
        result = opt.minimize_scalar(gap, bounds=(left, right), method="bounded", options={"xatol": 1e-13})
        if result.fun < best_gap:
            best_s, best_gap = float(result.x), float(result.fun)
    cancellation_gap = gap(cancellation_s)
    if cancellation_gap < best_gap:
        best_s, best_gap = cancellation_s, cancellation_gap
    return {
        "minimum_accessible_gap": best_gap,
        "at_s": best_s,
        "gap_at_hopping_cancellation": cancellation_gap,
        "grid_points": len(grid),
    }


def minimize_gap_sparse(H0: sp.csr_matrix, H1: sp.csr_matrix, cancellation_s: float) -> dict[str, Any]:
    """Locate the lowest gap without densifying a larger orbit quotient."""
    dimension = H0.shape[0]
    v0 = np.ones(dimension) / math.sqrt(dimension)

    def gap(s: float) -> float:
        values = sp.linalg.eigsh(
            (1 - s) * H0 + s * H1,
            k=2,
            which="SA",
            return_eigenvectors=False,
            tol=2e-11,
            maxiter=max(5000, 20 * dimension),
            v0=v0,
        )
        values.sort()
        return float(max(0.0, values[1] - values[0]))

    # The previously observed minimum lies in this window.  Endpoints and the
    # analytic cancellation point are still checked explicitly.
    grid = np.linspace(0.10, 0.25, 61)
    gaps = np.asarray([gap(float(s)) for s in grid])
    i = int(np.argmin(gaps))
    left, right = float(grid[max(0, i - 1)]), float(grid[min(len(grid) - 1, i + 1)])
    result = opt.minimize_scalar(gap, bounds=(left, right), method="bounded", options={"xatol": 2e-10})
    candidates = [
        (float(result.x), float(result.fun)),
        (0.0, gap(0.0)),
        (1.0, gap(1.0)),
        (cancellation_s, gap(cancellation_s)),
    ]
    best_s, best_gap = min(candidates, key=lambda pair: pair[1])
    return {
        "minimum_accessible_gap": best_gap,
        "at_s": best_s,
        "gap_at_hopping_cancellation": next(value for s, value in candidates if s == cancellation_s),
        "grid_points": len(grid),
        "search_window": [float(grid[0]), float(grid[-1])],
        "method": "sparse eigsh plus bounded local minimization",
    }


def analyse_size(n: int) -> dict[str, Any]:
    if n % 2:
        raise ValueError("family requires an even cycle")
    k = n // 2
    A = cycle_incidence(n)
    states, H0s, Hps, cover = symmetric_matrices(n, k, LAM, MU)
    group = dihedral_group(n)
    orbits = occupancy_orbits(states, group)
    S = orbit_isometry(orbits, len(states))
    H0q = np.asarray((S.T @ H0s @ S).toarray())
    Hpq = np.asarray((S.T @ Hps @ S).toarray())
    Pq = np.asarray((S.T @ sp.diags(cover.astype(float)) @ S).toarray())
    uniform_q = []
    for orbit in orbits:
        state = states[orbit[0]]
        multinomial = math.factorial(k)
        for count in state:
            multinomial //= math.factorial(count)
        uniform_q.append(math.sqrt(len(orbit) * multinomial / (n**k)))
    K, cyclic_growth = cyclic_basis([H0q, Hpq], np.asarray(uniform_q))
    H0k, Hpk, Pk = K.T @ H0q @ K, K.T @ Hpq @ K, K.T @ Pq @ K
    cancellation_s = (n - 1) / (2 * n - 1)
    Hc = (1 - cancellation_s) * H0q + cancellation_s * Hpq
    values_c = np.linalg.eigvalsh(Hc)
    cancellation_rank = int(np.sum(np.abs(values_c - values_c[0]) < 1e-9))
    Hc_k = (1 - cancellation_s) * H0k + cancellation_s * Hpk
    values_c_k = np.linalg.eigvalsh(Hc_k)
    cancellation_rank_k = int(np.sum(np.abs(values_c_k - values_c_k[0]) < 1e-9))
    final_values, final_vectors = np.linalg.eigh(Hpk)
    final_ground = final_vectors[:, 0]
    final_cover_probability = float(final_ground @ Pk @ final_ground)
    gap_result = minimize_gap(H0k, Hpk, cancellation_s)
    minimum_s = gap_result["at_s"]
    minimum_values, minimum_vectors = np.linalg.eigh((1 - minimum_s) * H0k + minimum_s * Hpk)
    minimum_ground = minimum_vectors[:, 0]
    distinct_mask = np.asarray([all(count <= 1 for count in state) for state in states], dtype=float)
    duplicate_pairs = np.asarray(
        [sum(count * (count - 1) // 2 for count in state) for state in states], dtype=float
    )
    uncovered_edges = np.asarray(
        [sum(state[t] == 0 and state[(t + 1) % n] == 0 for t in range(n)) for state in states],
        dtype=float,
    )

    def quotient_observable(diagonal: np.ndarray) -> np.ndarray:
        observable_q = np.asarray((S.T @ sp.diags(diagonal, format="csr") @ S).toarray())
        return K.T @ observable_q @ K

    minimum_distinct_probability = float(minimum_ground @ quotient_observable(distinct_mask) @ minimum_ground)
    minimum_cover_probability = float(minimum_ground @ Pk @ minimum_ground)
    minimum_expected_duplicate_pairs = float(minimum_ground @ quotient_observable(duplicate_pairs) @ minimum_ground)
    minimum_expected_uncovered_edges = float(minimum_ground @ quotient_observable(uncovered_edges) @ minimum_ground)
    m_penalty = min(LAM, MU)
    post_cancel_bound_at_sc = m_penalty * cancellation_s
    post_cancel_bound_at_s1 = m_penalty - k / (n - 1)
    post_cancel_uniform_bound = min(post_cancel_bound_at_sc, post_cancel_bound_at_s1)
    pre_cancel_threshold = (k * (n - 1) / n) / (m_penalty + k + k * (n - 1) / n)
    perturbation_norm_bound = 2 * k + LAM * n + MU * math.comb(k, 2)
    initial_interval_endpoint = 1 / (4 * perturbation_norm_bound)

    # Diagnostic restriction to distinct-base configurations. This is not the
    # original Hamiltonian, but identifies whether duplicate states control the
    # observed low gap.
    subsets, H0hc, Hphc, cover_hc = hard_core_matrices(n, k, LAM)
    hard_core_identity = hard_core_spin_identity(n, subsets, H0hc, Hphc, LAM)
    subset_orbit_list = subset_orbits(subsets, group)
    Shc = orbit_isometry(subset_orbit_list, len(subsets))
    H0hc_q = np.asarray((Shc.T @ H0hc @ Shc).toarray())
    Hphc_q = np.asarray((Shc.T @ Hphc @ Shc).toarray())
    hard_core_gap = minimize_gap(H0hc_q, Hphc_q, cancellation_s)
    invariance = {
        "H0": float(sp.linalg.norm(H0s @ S - S @ sp.csr_matrix(H0q))),
        "Hprob": float(sp.linalg.norm(Hps @ S - S @ sp.csr_matrix(Hpq))),
    }
    return {
        "n_B": n,
        "n_T": n,
        "k_star": k,
        "n_unordered_optimal_covers": 2,
        "n_cover_orbits_under_Dn": 1,
        "global_ground_rank_at_cancellation": 2 * math.factorial(k),
        "dimensions": {
            "ordered": str(n**k),
            "S_k_trivial": len(states),
            "joint_fixed": len(orbits),
            "cyclic": int(K.shape[1]),
        },
        "joint_fixed_ground_rank_at_cancellation": cancellation_rank,
        "cyclic_ground_rank_at_cancellation": cancellation_rank_k,
        "cyclic_growth": cyclic_growth,
        "final_accessible_cover_probability": final_cover_probability,
        "final_accessible_gap": float(final_values[1] - final_values[0]),
        "path": gap_result,
        "ground_state_at_minimum_gap": {
            "ground_energy": float(minimum_values[0]),
            "cover_probability": minimum_cover_probability,
            "distinct_base_probability": minimum_distinct_probability,
            "expected_duplicate_pairs": minimum_expected_duplicate_pairs,
            "expected_uncovered_edges": minimum_expected_uncovered_edges,
        },
        "rigorous_post_cancellation_bound": {
            "interval": [cancellation_s, 1.0],
            "at_s_cancel": post_cancel_bound_at_sc,
            "at_s1": post_cancel_bound_at_s1,
            "uniform_lower_bound": post_cancel_uniform_bound,
            "method": "rank-one cover block, Q-block kinetic floor, and Cauchy interlacing",
        },
        "rigorous_pre_cancellation_block_bound": {
            "interval": [pre_cancel_threshold, cancellation_s],
            "threshold_where_bound_vanishes": pre_cancel_threshold,
            "linear_bound": "s*m+s*k-(1-s)*k*(n-1)/n",
            "at_s_cancel": post_cancel_bound_at_sc,
            "method": "rank-one cover block, pre-cancellation kinetic floor, and Cauchy interlacing",
            "qualification": "positive only above the stated threshold; not uniform at its lower endpoint",
        },
        "rigorous_initial_perturbative_bound": {
            "interval": [0.0, initial_interval_endpoint],
            "gap_lower_bound": 0.5,
            "norm_bound_Hprob_minus_H0": perturbation_norm_bound,
            "method": "Weyl inequality from the unit H0 gap",
        },
        "hard_core_diagnostic": {
            "subset_dimension": len(subsets),
            "joint_fixed_dimension": len(subset_orbit_list),
            "path": hard_core_gap,
            "qualification": "projected distinct-base model, not the full Hamiltonian",
            "exact_spin_reduction": hard_core_identity,
        },
        "quotient_invariance_residual": invariance,
    }


def analyse_hard_core_only(n: int) -> dict[str, Any]:
    k = n // 2
    states, H0, H1, _ = hard_core_matrices(n, k, LAM)
    group = dihedral_group(n)
    orbits = subset_orbits(states, group)
    S = orbit_isometry(orbits, len(states))
    H0q = np.asarray((S.T @ H0 @ S).toarray())
    H1q = np.asarray((S.T @ H1 @ S).toarray())
    cancellation_s = (n - 1) / (2 * n - 1)
    identity = hard_core_spin_identity(n, states, H0, H1, LAM)
    return {
        "n_B": n,
        "k_star": k,
        "subset_dimension": len(states),
        "joint_fixed_dimension": len(orbits),
        "path": minimize_gap(H0q, H1q, cancellation_s),
        "exact_spin_reduction": identity,
        "qualification": "hard-core diagnostic only; exclusion sector omitted",
    }


def analyse_hard_core_only_sparse(n: int) -> dict[str, Any]:
    k = n // 2
    states, H0, H1, _ = hard_core_matrices(n, k, LAM)
    group = dihedral_group(n)
    orbits = subset_orbits(states, group)
    S = orbit_isometry(orbits, len(states))
    H0q = (S.T @ H0 @ S).tocsr()
    H1q = (S.T @ H1 @ S).tocsr()
    cancellation_s = (n - 1) / (2 * n - 1)
    identity = hard_core_spin_identity(n, states, H0, H1, LAM)
    return {
        "n_B": n,
        "k_star": k,
        "subset_dimension": len(states),
        "joint_fixed_dimension": len(orbits),
        "path": minimize_gap_sparse(H0q, H1q, cancellation_s),
        "exact_spin_reduction": identity,
        "qualification": "hard-core sparse diagnostic only; exclusion sector omitted",
    }


def main() -> None:
    results = [analyse_size(n) for n in EVEN_SIZES]
    hard_core_extension = [analyse_hard_core_only(n) for n in HARD_CORE_EXTENSION_SIZES]
    hard_core_extension.extend(analyse_hard_core_only_sparse(n) for n in SPARSE_HARD_CORE_EXTENSION_SIZES)
    hard_core_points = [
        (row["n_B"], row["hard_core_diagnostic"]["path"]["minimum_accessible_gap"])
        for row in results
    ] + [
        (row["n_B"], row["path"]["minimum_accessible_gap"])
        for row in hard_core_extension
    ]
    fit_n = np.asarray([point[0] for point in hard_core_points], dtype=float)
    fit_gap = np.asarray([point[1] for point in hard_core_points], dtype=float)
    slope_all, intercept_all = np.polyfit(np.log(fit_n), np.log(fit_gap), 1)
    slope_tail, intercept_tail = np.polyfit(np.log(fit_n[-4:]), np.log(fit_gap[-4:]), 1)
    output = {
        "schema_version": 1,
        "family": "minimum vertex cover on even cycles encoded as set cover",
        "definition": "bases=vertices, tasks=cycle edges, each base covers its two incident edges",
        "symmetry": "S_k x D_n",
        "parameters": {"lambda": LAM, "mu": MU},
        "qualification": (
            "finite-size full-path exploration with rigorous initial and post-cancellation interval bounds; "
            "the intermediate asymptotic minimum-gap bound remains open"
        ),
        "results": results,
        "hard_core_extension": hard_core_extension,
        "hard_core_scaling_diagnostic": {
            "all_sizes_power_law_exponent": float(slope_all),
            "all_sizes_prefactor": float(np.exp(intercept_all)),
            "last_four_power_law_exponent": float(slope_tail),
            "last_four_prefactor": float(np.exp(intercept_tail)),
            "n_to_three_halves_times_gap": [
                {"n_B": int(n), "value": float((n ** 1.5) * gap)} for n, gap in hard_core_points
            ],
            "qualification": "finite-size fit; n^(-3/2) is a conjectural asymptote, not a bound",
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for row in results:
        print(
            f"C_{row['n_B']:2d}: k={row['k_star']}, dims={row['dimensions']['ordered']}/"
            f"{row['dimensions']['S_k_trivial']}/{row['dimensions']['joint_fixed']}/"
            f"{row['dimensions']['cyclic']}, "
            f"min_gap={row['path']['minimum_accessible_gap']:.6g} at s={row['path']['at_s']:.5f}, "
            f"cancel_gap={row['path']['gap_at_hopping_cancellation']:.6g}, "
            f"p={row['final_accessible_cover_probability']:.6f}"
        )
    for row in hard_core_extension:
        print(
            f"C_{row['n_B']:2d} hard-core only: dims={row['subset_dimension']}/"
            f"{row['joint_fixed_dimension']}, min_gap={row['path']['minimum_accessible_gap']:.6g} "
            f"at s={row['path']['at_s']:.5f}"
        )


if __name__ == "__main__":
    main()
