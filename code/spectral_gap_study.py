"""
Spectral gap scaling study for the Multi-Register Quantum Walk (Minimum Set Cover).
Goal: resolve whether Δ(n_B) ~ n_B^{-α} (polynomial) or exp(-β n_B) (exponential).

Reference: multi_register_quantum_walk.md
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import curve_fit, milp, LinearConstraint, Bounds

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------


def partition_instance(n_groups: int, bases_per_group: int) -> np.ndarray:
    """Build a partition-structure adjacency matrix.

    There are ``n_groups`` tasks and ``n_groups * bases_per_group`` bases.
    Each base belongs to exactly one group and covers exactly that group's task.
    The minimum set cover requires one base from each group (k* = n_groups).
    Multiple distinct optimal covers exist (bases_per_group^n_groups choices),
    so single-hop transitions connect different covers in the valid subspace,
    giving a non-zero projected gap for Path A eigsh.

    Parameters
    ----------
    n_groups : int
        Number of groups (= n_T = k*).
    bases_per_group : int
        Number of interchangeable bases per group.

    Returns
    -------
    np.ndarray
        Adjacency matrix of shape (n_B, n_T) with n_B = n_groups * bases_per_group,
        n_T = n_groups.
    """
    n_B = n_groups * bases_per_group
    n_T = n_groups
    A = np.zeros((n_B, n_T), dtype=np.int32)
    for g in range(n_groups):
        for b_in_g in range(bases_per_group):
            A[g * bases_per_group + b_in_g, g] = 1
    return A


def random_msc_instance(
    n_B: int,
    n_T: int,
    p: float,
    seed: int,
) -> np.ndarray:
    """Erdos-Renyi bipartite coverage matrix.

    Each edge (base b, task t) exists independently with probability p.
    Guarantees: every task covered by >=1 base, every base covers >=1 task.

    Parameters
    ----------
    n_B : int
        Number of bases.
    n_T : int
        Number of tasks.
    p : float
        Edge probability (controls density / expected k*).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_B, n_T), dtype int32.
    """
    rng = np.random.default_rng(seed)
    A = (rng.random((n_B, n_T)) < p).astype(np.int32)
    # Ensure every task has at least one covering base
    for t in range(n_T):
        if not A[:, t].any():
            A[rng.integers(0, n_B), t] = 1
    # Ensure every base covers at least one task
    for b in range(n_B):
        if not A[b, :].any():
            A[b, rng.integers(0, n_T)] = 1
    return A


def solve_milp_msc(
    A: np.ndarray,
    costs: np.ndarray | None = None,
) -> tuple[int, list[tuple[int, ...]], float, float]:
    """Solve Minimum (or minimum-cost) Set Cover exactly via ILP.

    Unweighted ILP: min sum(x_b)  s.t.  A^T x >= 1_{n_T}, x in {0,1}^{n_B}.
    Weighted ILP:   min sum(c_b x_b) with the same constraints, when ``costs``
    is supplied.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of shape (n_B, n_T).
    costs : np.ndarray or None
        Per-base costs of length n_B.  If None, unit costs are assumed and the
        minimum-cardinality cover is returned (legacy behaviour).

    Returns
    -------
    k_star : int
        Cardinality of an optimal cover.
    covers : list of tuples
        All distinct optimal covers — i.e. covers attaining the optimal
        objective.  When ``costs`` is None this means minimum-cardinality
        covers; when ``costs`` is given, all minimum-cost covers (their
        cardinalities may differ from k_star, but every entry has the same
        total cost).
    milp_time : float
        Wall-clock seconds for the MILP solve.
    optimal_cost : float
        Optimal objective value (= k_star for unit costs).
    """
    n_B, n_T = A.shape
    t0 = time.perf_counter()

    if costs is None:
        c = np.ones(n_B, dtype=np.float64)
    else:
        c = np.asarray(costs, dtype=np.float64).copy()
        if c.shape != (n_B,):
            raise ValueError(
                f"costs must have shape ({n_B},); got {c.shape}"
            )

    # Coverage: -A^T x <= -1  <=>  A^T x >= 1
    constr = LinearConstraint(-A.T.astype(float), -np.inf, -np.ones(n_T))
    bounds = Bounds(0, 1)
    integrality = np.ones(n_B)

    result = milp(c, constraints=constr, integrality=integrality, bounds=bounds)
    milp_time = time.perf_counter() - t0

    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    optimal_cost = float(result.fun)
    sol_indices = tuple(sorted(int(i) for i, v in enumerate(result.x) if v > 0.5))
    k_star = len(sol_indices)

    # Enumerate all optimal covers via brute-force (fast for n_B <= 40).
    covers: list[tuple[int, ...]] = []
    if n_B <= 40:
        all_tasks = set(range(n_T))
        if costs is None:
            # Minimum-cardinality covers: enumerate by combinations of size k_star.
            for combo in itertools.combinations(range(n_B), k_star):
                covered = set(
                    int(x) for x in np.where(A[list(combo), :].any(axis=0))[0]
                )
                if covered == all_tasks:
                    covers.append(combo)
        else:
            # Minimum-cost covers: enumerate covers up to size n_B and keep
            # those whose total cost equals the optimum (within 1e-9).  We
            # iterate by size up to a safe upper bound (>= k_star) since a
            # minimum-cost cover may use more bases when individual costs vary.
            # In practice for n_B <= 40 and reasonable cost dispersion this
            # remains tractable; we cap the search at min(n_B, k_star + 4).
            tol = 1e-9
            k_max_search = min(n_B, k_star + 4)
            for k_try in range(1, k_max_search + 1):
                for combo in itertools.combinations(range(n_B), k_try):
                    covered = set(
                        int(x) for x in np.where(A[list(combo), :].any(axis=0))[0]
                    )
                    if covered != all_tasks:
                        continue
                    cost_combo = float(c[list(combo)].sum())
                    if abs(cost_combo - optimal_cost) < tol:
                        covers.append(combo)
            # Always include the MILP-returned solution (defensive)
            if sol_indices not in covers:
                covers.append(sol_indices)
    else:
        # For large n_B just return the single MILP solution.
        covers = [sol_indices]

    return k_star, covers, milp_time, optimal_cost


def compute_cover_graph_metrics(
    covers: list[tuple[int, ...]],
) -> dict[str, Any]:
    """Compute cover-graph topology metrics.

    Nodes are distinct optimal covers; two covers are adjacent iff their
    symmetric difference has exactly 2 elements (one-base swap).

    Parameters
    ----------
    covers : list of tuples
        Distinct optimal covers (same length k).

    Returns
    -------
    dict with keys: n_covers, n_edges, avg_degree, algebraic_connectivity
        (Fiedler value λ₂ of Laplacian; 0 if disconnected), is_connected.
    """
    n = len(covers)
    if n == 0:
        return {
            "n_covers": 0, "n_edges": 0, "avg_degree": 0.0,
            "algebraic_connectivity": 0.0, "is_connected": False,
        }
    if n == 1:
        return {
            "n_covers": 1, "n_edges": 0, "avg_degree": 0.0,
            "algebraic_connectivity": 0.0, "is_connected": True,
        }

    cover_sets = [frozenset(c) for c in covers]
    degree = np.zeros(n, dtype=np.float64)
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if len(cover_sets[i].symmetric_difference(cover_sets[j])) == 2:
                adj[i, j] = adj[j, i] = 1.0
                degree[i] += 1.0
                degree[j] += 1.0

    n_edges = int(degree.sum() / 2)
    avg_degree = float(degree.mean())
    L = np.diag(degree) - adj
    vals = np.linalg.eigvalsh(L)
    lam2 = float(vals[1])  # Fiedler value: > 0 iff connected
    is_connected = lam2 > 1e-10

    return {
        "n_covers": n,
        "n_edges": n_edges,
        "avg_degree": avg_degree,
        "algebraic_connectivity": lam2,
        "is_connected": is_connected,
    }


def generate_study_instances(
    configs: list[tuple[int, int, float]],
    n_seeds: int = 5,
    target_k: int | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Generate a list of (name, A) instances for the scaling study.

    Parameters
    ----------
    configs : list of (n_B, n_T, p)
        Erdos-Renyi parameters to sample from.
    n_seeds : int
        Number of random seeds per config.
    target_k : int or None
        If given, only keep instances where k* == target_k.

    Returns
    -------
    list of (name, A) where name encodes parameters and seed.
    """
    instances: list[tuple[str, np.ndarray]] = []
    for n_B, n_T, p in configs:
        kept = 0
        for seed in range(n_seeds * 10):  # oversample to hit target_k
            A = random_msc_instance(n_B, n_T, p, seed=seed)
            try:
                k, _, _, _ = solve_milp_msc(A)
            except Exception:
                continue
            if target_k is not None and k != target_k:
                continue
            name = f"rand_nB{n_B}_nT{n_T}_p{int(p * 100):02d}_s{seed:03d}"
            instances.append((name, A))
            kept += 1
            if kept >= n_seeds:
                break
    return instances


def grid_instance(rows: int, cols: int) -> np.ndarray:
    """Build adjacency matrix for a 2-D grid with von Neumann neighbourhood.

    Base b covers task t iff their Manhattan distance is <= 1.
    Both bases and tasks are indexed by (row, col) in row-major order.

    Parameters
    ----------
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.

    Returns
    -------
    np.ndarray
        Boolean adjacency matrix of shape (n_B, n_T) with n_B = n_T = rows * cols.
    """
    n = rows * cols
    A = np.zeros((n, n), dtype=np.int32)
    for b in range(n):
        br, bc = divmod(b, cols)
        for t in range(n):
            tr, tc = divmod(t, cols)
            if abs(br - tr) + abs(bc - tc) <= 1:
                A[b, t] = 1
    return A


# ---------------------------------------------------------------------------
# Optimal k
# ---------------------------------------------------------------------------


def find_optimal_k(A: np.ndarray) -> int:
    """Find the minimum set cover size k* for the given instance.

    Uses exact brute-force for n_B <= 20 (via itertools.combinations) and a
    greedy upper bound for larger instances.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of shape (n_B, n_T).

    Returns
    -------
    int
        Minimum number of bases required to cover all tasks.
    """
    n_B, n_T = A.shape
    all_tasks = set(range(n_T))

    if n_B <= 20:
        for k in range(1, n_B + 1):
            for combo in itertools.combinations(range(n_B), k):
                covered = set(np.where(A[list(combo), :].any(axis=0))[0])
                if covered == all_tasks:
                    return k
        return n_B  # fallback (should not happen for valid instances)

    # For n_B > 20 use MILP (faster and exact)
    try:
        k_star, _, _, _ = solve_milp_msc(A)
        return k_star
    except Exception:
        pass

    # Greedy fallback
    uncovered = set(all_tasks)
    selected: list[int] = []
    available = list(range(n_B))
    while uncovered and available:
        best = max(available, key=lambda b: len(set(np.where(A[b, :])[0]) & uncovered))
        newly = set(np.where(A[best, :])[0]) & uncovered
        if not newly:
            break
        uncovered -= newly
        selected.append(best)
        available.remove(best)
    return len(selected)


# ---------------------------------------------------------------------------
# Basis-state decoding (shared utility)
# ---------------------------------------------------------------------------


def _decode_bases(dim: int, k: int, reg_dim: int) -> np.ndarray:
    """Decode every computational basis index into its k register values.

    bases[idx, r] = (idx // reg_dim**r) % reg_dim

    Parameters
    ----------
    dim : int
        Total Hilbert-space dimension.
    k : int
        Number of registers.
    reg_dim : int
        Dimension of each register (= 2**q).

    Returns
    -------
    np.ndarray
        Integer array of shape (dim, k) with dtype int32.
    """
    idx_vec = np.arange(dim, dtype=np.int64)
    bases = np.empty((dim, k), dtype=np.int32)
    for r in range(k):
        bases[:, r] = (idx_vec // int(reg_dim**r)) % reg_dim
    return bases


# ---------------------------------------------------------------------------
# Penalty diagonals (vectorised)
# ---------------------------------------------------------------------------


def build_penalties(
    A: np.ndarray,
    k: int,
    bases: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct diagonal penalty vectors for all Hilbert-space basis states.

    Decodes every computational basis state |idx> -> (j_1, ..., j_k) in a
    single vectorised pass.  If ``bases`` is provided (precomputed by
    ``_decode_bases``), decoding is skipped to avoid redundant work.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of shape (n_B, n_T).
    k : int
        Number of registers.
    bases : np.ndarray or None
        Precomputed register indices of shape (dim, k).  When None the array
        is decoded from scratch.

    Returns
    -------
    diag_cov : np.ndarray
        Coverage penalty diagonal (number of uncovered tasks per state).
    diag_excl : np.ndarray
        Exclusion penalty diagonal (number of repeated-base pairs per state).
    diag_inv : np.ndarray
        Invalid-state penalty diagonal (number of registers in padding zone).
    """
    n_B, n_T = A.shape
    q = math.ceil(math.log2(n_B)) if n_B > 1 else 1
    reg_dim = 2**q
    dim = reg_dim**k

    # Memory guard: dim * k * 4 bytes
    mem_bytes = dim * k * 4
    mem_gb = mem_bytes / (1024**3)
    if mem_gb > 4.0:
        raise MemoryError(
            f"bases array would require {mem_gb:.2f} GB (> 4 GB limit). "
            f"dim={dim}, k={k}."
        )

    if bases is None:
        logger.info(
            "  Decoding %d states x %d registers (%.1f MB)...",
            dim, k, mem_gb * 1024,
        )
        bases = _decode_bases(dim, k, reg_dim)

    # H_inv: count registers where j_r >= n_B
    diag_inv = np.sum(bases >= n_B, axis=1).astype(np.float64)

    # H_excl: count pairs (r1, r2) with r1 < r2 where j_r1 == j_r2
    diag_excl = np.zeros(dim, dtype=np.float64)
    for r1, r2 in itertools.combinations(range(k), 2):
        diag_excl += (bases[:, r1] == bases[:, r2]).astype(np.float64)

    # H_cov: fully vectorised across all tasks simultaneously.
    # A_bool[b, t] = True iff base b covers task t.
    # valid_bases clips padding so A indexing is always safe; valid_mask
    # zeros out contributions from invalid (padded) registers.
    valid_bases = bases.clip(0, n_B - 1)  # (dim, k)
    valid_mask = bases < n_B              # (dim, k) bool

    # covered[idx, r, t] = True if register r of state idx covers task t
    # A_bool has shape (n_B, n_T); A_bool[valid_bases] -> (dim, k, n_T)
    A_bool = A.astype(bool)
    covered = A_bool[valid_bases]         # (dim, k, n_T)
    covered &= valid_mask[:, :, np.newaxis]  # zero invalid registers

    # task_covered[idx, t] = True if any register covers task t
    task_covered = covered.any(axis=1)   # (dim, n_T)
    diag_cov = (~task_covered).sum(axis=1).astype(np.float64)

    return diag_cov, diag_excl, diag_inv


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------


def build_hamiltonian(
    A: np.ndarray,
    k: int,
    lam: float,
    mu: float,
    nu: float,
    costs: np.ndarray | None = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build the total sparse Hamiltonian H = H_walk + λ H_cov + μ H_excl + ν H_inv
    (+ H_cost if ``costs`` is given for weighted SCP).

    Also returns the decoded register-index array so callers can reuse it for
    ``compute_p_opt`` without re-decoding.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of shape (n_B, n_T).
    k : int
        Number of registers.
    lam : float
        Coverage penalty weight λ.
    mu : float
        Exclusion penalty weight μ.
    nu : float
        Invalid-state penalty weight ν.
    costs : np.ndarray or None
        Optional per-base costs of length n_B for the weighted SCP.  When
        provided, an additive H_cost = Σ_r I⊗…⊗C⊗…⊗I term is included, with
        C = diag(c_0, …, c_{n_B-1}, 0, …, 0) padded to size reg_dim.  The
        ground state of the resulting Hamiltonian then corresponds to the
        minimum-cost cover (assuming the penalty weights are large enough to
        keep the ground state inside the cover subspace).

    Returns
    -------
    H : sp.csr_matrix
        Sparse Hamiltonian matrix of shape (dim, dim).
    bases : np.ndarray
        Register index array of shape (dim, k) with dtype int32.
    """
    n_B = A.shape[0]
    q = math.ceil(math.log2(n_B)) if n_B > 1 else 1
    reg_dim = 2**q
    dim = reg_dim**k

    # Decode basis states once; reuse in both build_penalties and compute_p_opt.
    mem_bytes = dim * k * 4
    mem_gb = mem_bytes / (1024**3)
    logger.info(
        "  Decoding %d states x %d registers (%.1f MB)...",
        dim, k, mem_gb * 1024,
    )
    bases = _decode_bases(dim, k, reg_dim)

    # --- H_hop: complete graph on valid states, normalised by spectral norm ---
    # Spectral norm of the complete-graph all-ones-minus-identity matrix on n_B
    # vertices is exactly (n_B - 1) (its largest eigenvalue), so dividing by it
    # makes the per-register kinetic bandwidth independent of n_B and lets us
    # compare absolute gap values across instances with different n_B.
    W = np.zeros((reg_dim, reg_dim), dtype=np.float64)
    for i in range(n_B):
        for j in range(n_B):
            if i != j:
                W[i, j] = 1.0
    spec = float(n_B - 1)  # spectral norm of complete-graph hopping = n_B-1
    if spec > 0.0:
        W /= spec

    H_hop_sp = sp.csr_matrix(W)
    I_reg = sp.eye(reg_dim, format="csr")

    # H_walk = sum_r  I^{⊗r} ⊗ H_hop ⊗ I^{⊗(k-1-r)}
    H_walk = sp.csr_matrix((dim, dim), dtype=np.float64)
    for r in range(k):
        term = sp.eye(1, format="csr")
        for s in range(k):
            if s == r:
                term = sp.kron(term, H_hop_sp, format="csr")
            else:
                term = sp.kron(term, I_reg, format="csr")
        H_walk = H_walk + term

    # --- Diagonal penalties (pass precomputed bases) ---
    diag_cov, diag_excl, diag_inv = build_penalties(A, k, bases=bases)

    diag_total = lam * diag_cov + mu * diag_excl + nu * diag_inv

    # --- Optional H_cost (weighted SCP) ---
    if costs is not None:
        c_vec = np.asarray(costs, dtype=np.float64)
        if c_vec.shape != (n_B,):
            raise ValueError(
                f"costs must have shape ({n_B},); got {c_vec.shape}"
            )
        # Vectorised diagonal H_cost = Σ_r c_{j_r} (zero in padding registers).
        # bases (dim, k) holds register values; clip to keep indexing safe and
        # mask out padded entries.
        valid_bases = bases.clip(0, n_B - 1)
        valid_mask = bases < n_B
        # cost_per_register[idx, r] = c_{j_r} if valid, else 0
        cost_per_reg = c_vec[valid_bases] * valid_mask.astype(np.float64)
        diag_total = diag_total + cost_per_reg.sum(axis=1)

    H_penalty = sp.diags(diag_total, format="csr")

    H = (H_walk + H_penalty).tocsr()
    return H, bases


# ---------------------------------------------------------------------------
# Symmetric-sector spectral gap
# ---------------------------------------------------------------------------


def _compute_gap_sym_sector(
    A: np.ndarray,
    k: int,
    reg_dim: int,
    lam: float,
    mu: float,
    nu: float,
) -> tuple[float, float, float]:
    """Exact spectral gap via the fully-symmetric S_k sector.

    H commutes with all register permutations (S_k symmetry), so the ground
    state lives in the totally-symmetric sector.  Its basis is the set of
    *unordered* k-tuples of distinct values in {0, …, reg_dim-1}:
    C(reg_dim, k) states.

    The effective Hamiltonian on this basis is:
      H_sym[C1, C2] = -1   if |C1 ∩ C2| = k-1   (adjacent, single-element swap)
      H_sym[C1, C2] =  0   otherwise
      H_sym[C, C]   = λ * n_uncovered(C) + ν * n_invalid(C)
                        (H_excl = 0 since all elements are distinct)

    Full dense diagonalisation via numpy.linalg.eigvalsh.  The dimension is
    C(reg_dim, k), which is at most:
      k=3, reg_dim=16: 560   k=4, reg_dim=16: 1820   k=5, reg_dim=8: 56

    Returns
    -------
    E0, E1, gap : float
    """
    n_B, n_T = A.shape
    A_bool = A.astype(bool)
    all_states = list(itertools.combinations(range(reg_dim), k))
    n_sym = len(all_states)
    idx = {s: i for i, s in enumerate(all_states)}

    H_sym = np.zeros((n_sym, n_sym), dtype=np.float64)
    for i, C in enumerate(all_states):
        C_set = set(C)

        # Diagonal: coverage and invalid penalties
        valid_B = [b for b in C if b < n_B]
        if valid_B:
            covered = np.zeros(n_T, dtype=bool)
            for b in valid_B:
                covered |= A_bool[b]
            n_uncov = int(n_T - covered.sum())
        else:
            n_uncov = n_T
        n_inv = sum(1 for b in C if b >= n_B)
        H_sym[i, i] = lam * n_uncov + nu * n_inv

        # Off-diagonal: single-element swap neighbours
        for drop in C:
            for add in range(reg_dim):
                if add in C_set:
                    continue
                nb = tuple(sorted((C_set - {drop}) | {add}))
                j = idx.get(nb)
                if j is not None and j > i:
                    H_sym[i, j] = -1.0
                    H_sym[j, i] = -1.0

    vals = np.linalg.eigvalsh(H_sym)
    E0, E1 = float(vals[0]), float(vals[1])
    return E0, E1, E1 - E0


# ---------------------------------------------------------------------------
# Spectral gap
# ---------------------------------------------------------------------------


def compute_gap(
    H: sp.csr_matrix,
    dim: int,
    n_B: int,
    k: int,
    valid_idx: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Compute the two lowest eigenvalues, the spectral gap, and the ground state.

    Uses a two-path hybrid strategy:

    Path A (projected, for dim > 32768):
        The ground state lives almost entirely in the zero-penalty subspace
        (P_opt ≈ 1), and the within-subspace gap equals the full gap when
        optimal covers are structurally heterogeneous (grid_3x4, 29 covers,
        confirmed Δ=1.26e-4).  ARPACK eigsh on the sparse projected matrix
        is used; requests k=50 eigenvalues to escape degenerate E0 manifolds.

    Path B-L (Löwdin H_eff, for dim ≤ 32768):
        Löwdin partitioning gives an effective H on the zero-penalty ("cover")
        subspace P, avoiding the full H entirely:

            H_eff = H_PP + H_PQ diag(1/(E0−E_Q)) H_QP

        Faster than shift-invert SuperLU on this machine (17 MFLOPS BLAS),
        and immune to the ARPACK non-determinism that afflicts shift-invert
        when the gap is small relative to the distance from sigma to E0 (seen
        for k*=4 instances with n_valid≈300).  Cost: ~0.2 s for dim=32768,
        <0.05 s for dim≤4096.  Covers both the old "Path B" (dim≤8192) and
        "Path B-L" (8192<dim≤32768) — now unified under one code path.

    Parameters
    ----------
    H : sp.csr_matrix
        Sparse Hamiltonian matrix.
    dim, n_B, k : int
        Hilbert dimension, number of bases, number of registers.

    Returns
    -------
    E0, E1, gap, gs
    """
    diag = H.diagonal()
    if valid_idx is None:
        valid_idx = np.where(diag < 1e-10)[0]
    n_valid = len(valid_idx)
    logger.info("    dim=%d, n_valid=%d", dim, n_valid)

    # Sparse submatrix in the valid-cover subspace (needed by both paths).
    H_proj_sp = sp.csr_matrix(H[np.ix_(valid_idx, valid_idx)])

    if dim > 32768:
        # --- Path A: projected subspace ---
        # Δ_proj = Δ_phys only when the cover graph is strongly connected AND
        # covers are heterogeneous (grid_3x4: 29 covers, confirmed Δ=1.26e-4).
        # For k*=5 instances (dim=32768) the k!=120 permutation copies of each
        # cover make E0=E1 in H_proj (eigsh k=2 returns the degenerate ground
        # manifold), giving an apparent zero gap.  Those instances must use
        # Path B.  Path A is reserved for dim > 32768 where Path B (SuperLU
        # factorisation of the full H) is too slow on this machine.
        logger.info("    Path A (projected eigsh)")
        if H_proj_sp.nnz == 0:
            logger.info(
                "    H_proj is zero (isolated covers); gap_proj = 0. "
                "Physical gap requires full-H solver — not available for dim=%d.", dim
            )
            E0, E1 = 0.0, 0.0
            gs = np.zeros(dim)
            if n_valid > 0:
                gs[valid_idx] = 1.0 / math.sqrt(n_valid)
        else:
            # Request enough eigenvalues to escape a possibly degenerate E0
            # (e.g. grid_2x6 has 21-fold degenerate E0 due to grid symmetry).
            k_req = min(50, n_valid - 1)
            vals_proj, vecs_proj = spla.eigsh(
                H_proj_sp, k=k_req, which="SA", tol=1e-12, maxiter=200_000
            )
            vals_proj = np.sort(vals_proj)
            E0 = float(vals_proj[0])
            # First eigenvalue distinctly above E0 (tolerance 1e-8)
            above = np.where(vals_proj > E0 + 1e-8)[0]
            E1 = float(vals_proj[above[0]]) if len(above) > 0 else E0
            degen = above[0] if len(above) > 0 else k_req
            gs = np.zeros(dim)
            gs[valid_idx] = vecs_proj[:, 0]
            gs /= np.linalg.norm(gs)
            logger.info(
                "    Path A gap_proj=%.4e  E0_degen=%d  (physical gap if covers connected)",
                E1 - E0, degen,
            )

    else:
        # --- Path B-L: Löwdin-partitioned H_eff for dim ≤ 32768 ---
        #
        # Replaces the old shift-invert SuperLU "Path B" for dim≤8192.
        # Shift-invert is non-deterministic on this machine when the gap is
        # small relative to |sigma−E0| (seen for k*=4 instances, n_valid~300).
        # Löwdin H_eff is deterministic, faster (~0.05–0.2 s), and gives the
        # same within-cover-manifold gap.
        logger.info("    Path B-L (Löwdin H_eff, dim=%d)", dim)

        if n_valid < 2:
            E0 = E1 = 0.0
            gs = np.zeros(dim)
            if n_valid == 1:
                gs[valid_idx[0]] = 1.0
        elif H_proj_sp.nnz == 0 and n_valid <= 50:
            # Covers are fully disconnected with a tiny valid subspace.
            # Löwdin 2nd-order gives gap=0 by symmetry; the physical gap comes
            # from high-order virtual hops.  Shift-invert is reliable here
            # because n_valid is small and ARPACK degeneracy issues are absent.
            logger.info(
                "    Path dense-SI fallback: H_proj.nnz=0 + n_valid=%d", n_valid
            )
            sigma_fallback = -0.1
            vals_si, vecs_si = spla.eigsh(
                H, k=2, which="LM", sigma=sigma_fallback,
                tol=1e-10, maxiter=50_000, ncv=min(40, dim - 1),
            )
            order_si = np.argsort(vals_si)
            E0, E1 = float(vals_si[order_si[0]]), float(vals_si[order_si[1]])
            gs = vecs_si[:, order_si[0]]
            gs /= np.linalg.norm(gs)
            logger.info("    gap=%.4e  E0=%.6f  E1=%.6f", E1 - E0, E0, E1)
        elif dim <= 4096:
            # Dense exact diagonalization: feasible for dim≤4096 (~12 s).
            # Avoids ARPACK non-determinism and Löwdin 2nd-order error (~60%
            # overestimate observed for grid_3x3 due to degenerate H_proj E0).
            # Use eigh (not eigvalsh) to recover the actual ground-state
            # eigenvector — the previous hand-crafted uniform superposition
            # produced vacuous p_opt=1 by construction.
            logger.info("    Path dense (exact eigh, dim=%d)", dim)
            vals_d, vecs_d = np.linalg.eigh(H.toarray())
            E0, E1 = float(vals_d[0]), float(vals_d[1])
            gs = vecs_d[:, 0]
            gs /= np.linalg.norm(gs)
            logger.info("    gap=%.4e  E0=%.6f  E1=%.6f", E1 - E0, E0, E1)
        else:
            # Estimate E0 from H_proj ground state (used in Löwdin denominator)
            if H_proj_sp.nnz > 0:
                try:
                    rough, vecs_r = spla.eigsh(
                        H_proj_sp, k=1, which="SA", tol=1e-4,
                        maxiter=10 * n_valid,
                    )
                    E0_proj = float(rough[0])
                    gs_proj = np.zeros(dim)
                    gs_proj[valid_idx] = vecs_r[:, 0]
                    gs_proj /= np.linalg.norm(gs_proj)
                except Exception:
                    E0_proj = 0.0
                    gs_proj = np.zeros(dim)
                    gs_proj[valid_idx] = 1.0 / math.sqrt(n_valid)
            else:
                E0_proj = 0.0
                gs_proj = np.zeros(dim)
                gs_proj[valid_idx] = 1.0 / math.sqrt(n_valid)

            # Build H_eff = H_PP + H_PQ diag(D_Q) H_QP via sparse ops.
            # H_PQ = H[valid_idx, Q] where Q = non-valid-cover states.
            # D_Q[s] = 1 / (E0_proj - E_Q[s]),  |denom| clamped to ≥ 0.01.
            diag_H = H.diagonal()
            H_PP_dense = H_proj_sp.toarray()  # n_valid × n_valid

            # Build sparse H_PQ: slice P rows, zero out P columns
            H_PQ = H[valid_idx, :]  # n_valid × dim (CSR)
            p_col_mask = np.zeros(dim, dtype=bool)
            p_col_mask[valid_idx] = True
            H_PQ = H_PQ.multiply(~p_col_mask[np.newaxis, :]).tocsr()

            # D vector over all Q states (P entries stay 0 after multiply).
            # In Löwdin partitioning the denominator (E0 − E_Q) should be safely
            # bounded away from zero because Q-states have E ≥ λ ≥ 2 while
            # E0 ≈ 0. We assert this rather than silently clamping; a violation
            # would indicate the Löwdin approximation is invalid for this case.
            E_all = diag_H.copy()
            diffs_Q = (E0_proj - E_all)[~p_col_mask]
            min_abs = float(np.abs(diffs_Q).min()) if diffs_Q.size > 0 else float("inf")
            assert min_abs > 0.01, (
                f"Löwdin denominator near zero: min|E0-E_Q|={min_abs:.4e} "
                f"(E0_proj={E0_proj:.4e}); 2nd-order Löwdin not valid here."
            )
            D_all = np.zeros(dim)
            D_all[~p_col_mask] = 1.0 / (E0_proj - E_all[~p_col_mask])

            # correction = H_PQ @ diag(D_all) @ H_PQ.T  (sparse ops)
            H_PQ_D = H_PQ.multiply(D_all[np.newaxis, :])  # n_valid × dim
            correction = (H_PQ_D @ H_PQ.T).toarray()      # n_valid × n_valid

            H_eff = H_PP_dense + correction

            vals_eff = np.linalg.eigvalsh(H_eff)
            E0 = float(vals_eff[0])
            E1 = float(vals_eff[1])
            gs = np.zeros(dim)
            gs[valid_idx] = gs_proj[valid_idx]
            gs /= np.linalg.norm(gs)

            logger.info(
                "    Path B-L gap=%.4e  E0_eff=%.6f  E1_eff=%.6f",
                E1 - E0, E0, E1,
            )

    return E0, E1, E1 - E0, gs


# ---------------------------------------------------------------------------
# Ground-state probability on optimal solutions
# ---------------------------------------------------------------------------


def compute_p_opt(
    A: np.ndarray,
    k: int,
    gs: np.ndarray,
    bases: np.ndarray | None = None,
) -> float:
    """Compute the probability of measuring an optimal cover in the ground state.

    A state |j_1, ..., j_k> is an optimal cover iff:
      - all j_r < n_B (no padding states),
      - all j_r are distinct,
      - the union of rows A[j_r, :] covers all tasks.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of shape (n_B, n_T).
    k : int
        Number of registers.
    gs : np.ndarray
        Normalised ground-state vector of length dim (from ``compute_gap``).
    bases : np.ndarray or None
        Precomputed register-index array of shape (dim, k).  When None it is
        decoded from scratch.

    Returns
    -------
    float
        Sum of |amplitude|^2 on optimal-cover states.
    """
    n_B, n_T = A.shape
    q = math.ceil(math.log2(n_B)) if n_B > 1 else 1
    reg_dim = 2**q
    dim = len(gs)

    gs = gs / np.linalg.norm(gs)

    if bases is None:
        bases = _decode_bases(dim, k, reg_dim)

    # All registers must be valid
    all_valid = (bases < n_B).all(axis=1)

    # All registers must select distinct bases
    sorted_bases = np.sort(bases, axis=1)
    has_dup = (np.diff(sorted_bases, axis=1) == 0).any(axis=1)
    all_distinct_mask = ~has_dup

    # All tasks must be covered by at least one register
    valid_bases = bases.clip(0, n_B - 1)
    A_bool = A.astype(bool)
    covers_any_task = A_bool[valid_bases].any(axis=1)  # (dim, n_T)
    full_cover = covers_any_task.all(axis=1)

    optimal_mask = all_valid & all_distinct_mask & full_cover
    return float(np.sum(np.abs(gs[optimal_mask]) ** 2))


# ---------------------------------------------------------------------------
# Main study loop
# ---------------------------------------------------------------------------

# Each entry: (name, instance_factory).
# Grid instances with von Neumann neighbourhood; k* grows slowly with grid size.
#
# Path selection (see compute_gap):
#   dim <=  32 768 → Path B (full-H shift-invert, exact Δ_phys)
#   dim >   32 768 → Path A (projected eigsh, Δ_proj ≈ Δ_phys only for
#                    heterogeneous, richly-connected covers like grid_3x4)
#
#   name        n_B  k*   q   dim
#   grid_2x4      8   3   3       512   (Path B)
#   grid_3x3      9   3   4     4 096   (Path B, validated Δ=1.61e-3)
#   grid_2x5     10   3   4     4 096   (Path B)
#   rand k*=4     7   4   3     2 187   (Path B)   ← q=ceil(log2(7))=3
#   rand k*=4     8   4   3     4 096   (Path B)
#   rand k*=5     8   5   3    32 768   (Path B, ~90 s SuperLU)
#   grid_2x6     12   4   4    65 536   (Path A)
#   grid_3x4     12   4   4    65 536   (Path A, validated Δ=1.27e-4)
#   grid_2x7     14   4   4    65 536   (Path A)
#   grid_3x5     15   4   4    65 536   (Path A)
#   grid_4x4     16   4   4    65 536   (Path A)

# Grid instances (fixed geometry)
_GRID_INSTANCES: list[tuple[str, object]] = [
    # k*=3 instances (dim<=4096 → Path B, exact Δ_phys via shift-invert)
    ("grid_2x4", lambda: grid_instance(2, 4)),   # n_B=8,  k*=3, dim=512
    ("grid_3x3", lambda: grid_instance(3, 3)),   # n_B=9,  k*=3, dim=4096
    ("grid_2x5", lambda: grid_instance(2, 5)),   # n_B=10, k*=3, dim=4096
    # k*=4 instances (dim=65536 → Path A)
    ("grid_2x6", lambda: grid_instance(2, 6)),   # n_B=12, k*=4, dim=65536
    ("grid_3x4", lambda: grid_instance(3, 4)),   # n_B=12, k*=4, dim=65536 (validated)
    ("grid_2x7", lambda: grid_instance(2, 7)),   # n_B=14, k*=4, dim=65536
    ("grid_3x5", lambda: grid_instance(3, 5)),   # n_B=15, k*=4, dim=65536
    ("grid_4x4", lambda: grid_instance(4, 4)),   # n_B=16, k*=4, dim=65536
]

# Random k*=3: n_B=8 (q=3, dim=512) → Path B, exact Δ_phys, very fast.
# p=0.50 with n_T=7 yields ~k*=3 instances with ≥5 covers (controlled n_B=8 study).
_RANDOM_CONFIGS_K3 = [(8, 7, 0.50)]

# Random k*=4: n_B=7 (q=3, dim=2187) and n_B=8 (q=3, dim=4096) → Path B, exact Δ_phys
_RANDOM_CONFIGS_K4 = [(7, 7, 0.45), (8, 8, 0.50), (8, 8, 0.42)]

# Random k*=5: n_B=8 (q=3, dim=32768) → Path B-L, Löwdin H_eff (~0.2s per instance)
# Density chosen to produce k*=5 while allowing multiple covers (2–8 covers typical).
# Physical gap arises from second-order virtual hops through invalid states.
_RANDOM_CONFIGS_K5 = [(8, 7, 0.30), (8, 8, 0.22), (8, 6, 0.35)]


def _build_instances() -> list[tuple[str, object]]:
    """Build the full instance list: grid + random (k*=3, k*=4, and k*=5)."""
    instances: list[tuple[str, object]] = list(_GRID_INSTANCES)
    for name, A in generate_study_instances(_RANDOM_CONFIGS_K3, n_seeds=3, target_k=3):
        A_copy = A.copy()
        instances.append((name, lambda _A=A_copy: _A))
    for name, A in generate_study_instances(_RANDOM_CONFIGS_K4, n_seeds=3, target_k=4):
        A_copy = A.copy()
        instances.append((name, lambda _A=A_copy: _A))
    for name, A in generate_study_instances(_RANDOM_CONFIGS_K5, n_seeds=3, target_k=5):
        A_copy = A.copy()
        instances.append((name, lambda _A=A_copy: _A))
    return instances


INSTANCES: list[tuple[str, object]] = _build_instances()

PARAM_SETS: list[dict[str, float]] = [
    {"lam": 2.0, "mu": 5.0, "nu": 20.0},
    {"lam": 5.0, "mu": 10.0, "nu": 50.0},
]

DIM_LIMIT = 50_000_000


def run_study(
    output_path: str = str(DATA_DIR / "gap_results.json"),
) -> list[dict[str, Any]]:
    """Run the spectral gap study across all instances and parameter sets.

    Results are saved as JSON and returned as a list of records.

    Parameters
    ----------
    output_path : str
        Destination path for the JSON output file.

    Returns
    -------
    list[dict[str, Any]]
        List of result dictionaries, one per (instance, param_set) combination.
    """
    results: list[dict[str, Any]] = []

    for inst_name, inst_fn in INSTANCES:
        A = inst_fn()
        n_B, n_T = A.shape
        k = find_optimal_k(A)
        q = math.ceil(math.log2(n_B)) if n_B > 1 else 1
        reg_dim = 2**q
        dim = reg_dim**k
        n_qubits = k * q

        logger.info(
            "Instance %s: n_B=%d, n_T=%d, k*=%d, q=%d, dim=%d, n_qubits=%d",
            inst_name, n_B, n_T, k, q, dim, n_qubits,
        )

        if dim > DIM_LIMIT:
            logger.info("  Skipping %s: dim=%d > %d limit.", inst_name, dim, DIM_LIMIT)
            results.append({
                "instance": inst_name,
                "n_B": n_B,
                "n_T": n_T,
                "k_opt": k,
                "dim": dim,
                "n_qubits": n_qubits,
                "skipped": True,
                "skip_reason": f"dim={dim} > {DIM_LIMIT}",
            })
            continue

        # Memory pre-check for bases array
        mem_bytes = dim * k * 4
        mem_gb = mem_bytes / (1024**3)
        if mem_gb > 4.0:
            logger.warning(
                "  Skipping %s: bases array would need %.2f GB.", inst_name, mem_gb
            )
            results.append({
                "instance": inst_name,
                "n_B": n_B,
                "n_T": n_T,
                "k_opt": k,
                "dim": dim,
                "n_qubits": n_qubits,
                "skipped": True,
                "skip_reason": f"bases array {mem_gb:.2f} GB > 4 GB",
            })
            continue

        # MILP: get k*, covers, and cover-graph metrics for this instance
        try:
            k_milp, covers, milp_t, _ = solve_milp_msc(A)
            cg = compute_cover_graph_metrics(covers)
        except Exception as exc:
            logger.warning("  MILP failed for %s: %s", inst_name, exc)
            k_milp, covers, milp_t = k, [], 0.0
            cg = {
                "n_covers": 0, "n_edges": 0, "avg_degree": 0.0,
                "algebraic_connectivity": 0.0, "is_connected": None,
            }

        for params in PARAM_SETS:
            lam, mu, nu = params["lam"], params["mu"], params["nu"]
            logger.info("  Params: lam=%.1f, mu=%.1f, nu=%.1f", lam, mu, nu)

            t0 = time.perf_counter()
            try:
                H, bases = build_hamiltonian(A, k, lam, mu, nu)
                E0, E1, gap, gs = compute_gap(H, dim, n_B, k)
                p_opt = compute_p_opt(A, k, gs, bases=bases)
                elapsed = time.perf_counter() - t0
                logger.info(
                    "    E0=%.6f  E1=%.6f  gap=%.6e  P_opt=%.4f  (%.1fs)",
                    E0, E1, gap, p_opt, elapsed,
                )
                if dim <= 4096:
                    gap_type = "phys"    # dense exact diagonalization
                elif dim <= 32768:
                    gap_type = "lowdin"  # Löwdin 2nd-order (~20% error for k*=5)
                else:
                    gap_type = "proj"    # H_proj projected gap
                results.append({
                    "instance": inst_name,
                    "n_B": n_B,
                    "n_T": n_T,
                    "k_opt": k,
                    "dim": dim,
                    "n_qubits": n_qubits,
                    "lam": lam,
                    "mu": mu,
                    "nu": nu,
                    "E0": E0,
                    "E1": E1,
                    "gap": gap,
                    "gap_type": gap_type,
                    "p_opt": p_opt,
                    "elapsed_s": elapsed,
                    "milp_time_s": milp_t,
                    "n_covers": cg["n_covers"],
                    "n_edges": cg["n_edges"],
                    "avg_degree": cg["avg_degree"],
                    "algebraic_connectivity": cg["algebraic_connectivity"],
                    "covers_connected": cg["is_connected"],
                    "skipped": False,
                })
            except MemoryError as exc:
                logger.warning("  MemoryError for %s: %s", inst_name, exc)
                results.append({
                    "instance": inst_name,
                    "n_B": n_B,
                    "n_T": n_T,
                    "k_opt": k,
                    "dim": dim,
                    "n_qubits": n_qubits,
                    "lam": lam,
                    "mu": mu,
                    "nu": nu,
                    "skipped": True,
                    "skip_reason": str(exc),
                })
            except Exception as exc:
                logger.error(
                    "  Unexpected error for %s: %s", inst_name, exc, exc_info=True
                )
                results.append({
                    "instance": inst_name,
                    "n_B": n_B,
                    "n_T": n_T,
                    "k_opt": k,
                    "dim": dim,
                    "n_qubits": n_qubits,
                    "lam": lam,
                    "mu": mu,
                    "nu": nu,
                    "skipped": True,
                    "skip_reason": f"error: {exc}",
                })

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to %s", output_path)
    return results


# ---------------------------------------------------------------------------
# Scaling analysis
# ---------------------------------------------------------------------------


def _poly_model(n_B: np.ndarray, a: float, alpha: float) -> np.ndarray:
    """Polynomial decay model: Δ = a * n_B^{-α}."""
    return a * np.power(n_B, -alpha)


def _exp_model(n_B: np.ndarray, a: float, beta: float) -> np.ndarray:
    """Exponential decay model: Δ = a * exp(-β * n_B)."""
    return a * np.exp(-beta * n_B)


def analyze_scaling(
    results: list[dict[str, Any]],
    lam: float = 5.0,
    mu: float = 10.0,
) -> None:
    """Fit polynomial and exponential models to Δ(n_B) and compare residuals.

    Prints fit parameters, RSS values, the preferred model, and gap projections
    for n_B = 50, 100, 133.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Output of run_study().
    lam : float
        Coverage penalty weight to filter on.
    mu : float
        Exclusion penalty weight to filter on.
    """
    records = [
        r for r in results
        if not r.get("skipped", False)
        and abs(r.get("lam", 0) - lam) < 1e-9
        and abs(r.get("mu", 0) - mu) < 1e-9
        and r.get("gap", 0) > 1e-12
    ]
    # Separate exact gaps (phys/lowdin) from projected approximations.
    phys_records = [r for r in records if r.get("gap_type") in ("phys", "lowdin")]

    if len(records) < 2:
        logger.warning(
            "Not enough data points for scaling analysis (lam=%.1f, mu=%.1f). "
            "Need at least 2, found %d.",
            lam, mu, len(records),
        )
        return

    records_sorted = sorted(records, key=lambda r: r["n_B"])
    # Fit only on physical gaps; fall back to all records if too few phys points.
    fit_records = phys_records if len(phys_records) >= 2 else records
    fit_sorted = sorted(fit_records, key=lambda r: r["n_B"])
    n_B_arr = np.array([r["n_B"] for r in fit_sorted], dtype=np.float64)
    gap_arr = np.array([r["gap"] for r in fit_sorted], dtype=np.float64)
    if fit_records is phys_records:
        print(f"  Fitting on {len(fit_records)} physical-gap data points.")
    else:
        print(f"  WARNING: <2 phys-gap points; fitting on all {len(fit_records)} records.")

    print(f"\n=== Scaling analysis (lam={lam}, mu={mu}) ===")
    print(f"{'n_B':>6}  {'k*':>4}  {'gap':>14}  {'type':>5}  instance")
    for r in records_sorted:
        gtype = r.get("gap_type", "?")
        print(
            f"{r['n_B']:>6}  {r['k_opt']:>4}  {r['gap']:>14.6e}  {gtype:>5}  {r['instance']}"
        )

    # --- Polynomial fit in log-log space ---
    log_nB = np.log(n_B_arr)
    log_gap = np.log(gap_arr)
    try:
        poly_popt, _ = curve_fit(
            lambda x, log_a, alpha: log_a - alpha * x,
            log_nB, log_gap,
            p0=[0.0, 2.0],
        )
        log_a_poly, alpha_poly = poly_popt
        a_poly = float(np.exp(log_a_poly))
        alpha_poly = float(alpha_poly)
        gap_poly_pred = _poly_model(n_B_arr, a_poly, alpha_poly)
        rss_poly = float(np.sum((gap_arr - gap_poly_pred) ** 2))
        poly_ok = True
    except Exception as exc:
        logger.warning("Polynomial fit failed: %s", exc)
        poly_ok = False
        rss_poly = float("inf")
        a_poly = alpha_poly = float("nan")

    # --- Exponential fit ---
    try:
        exp_popt, _ = curve_fit(
            _exp_model, n_B_arr, gap_arr,
            p0=[1.0, 0.1],
            maxfev=10_000,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
        )
        a_exp, beta_exp = float(exp_popt[0]), float(exp_popt[1])
        gap_exp_pred = _exp_model(n_B_arr, a_exp, beta_exp)
        rss_exp = float(np.sum((gap_arr - gap_exp_pred) ** 2))
        exp_ok = True
    except Exception as exc:
        logger.warning("Exponential fit failed: %s", exc)
        exp_ok = False
        rss_exp = float("inf")
        a_exp = beta_exp = float("nan")

    print("\n--- Polynomial fit: Delta = a * n_B^{-alpha} ---")
    if poly_ok:
        print(f"  a     = {a_poly:.6e}")
        print(f"  alpha = {alpha_poly:.4f}")
        print(f"  RSS   = {rss_poly:.6e}")
    else:
        print("  Fit failed.")

    print("\n--- Exponential fit: Delta = a * exp(-beta * n_B) ---")
    if exp_ok:
        print(f"  a    = {a_exp:.6e}")
        print(f"  beta = {beta_exp:.6f}")
        print(f"  RSS  = {rss_exp:.6e}")
    else:
        print("  Fit failed.")

    print("\n--- Model comparison ---")
    if poly_ok and exp_ok:
        if rss_poly <= rss_exp:
            print(
                f"  Polynomial model preferred "
                f"(RSS_poly={rss_poly:.3e} <= RSS_exp={rss_exp:.3e})."
            )
            print(
                f"  Interpretation: Delta ~ n_B^(-{alpha_poly:.2f}) "
                f"=> adiabatic T ~ n_B^({2 * alpha_poly:.2f}) (polynomial)."
            )
        else:
            print(
                f"  Exponential model preferred "
                f"(RSS_exp={rss_exp:.3e} < RSS_poly={rss_poly:.3e})."
            )
            print(
                f"  Interpretation: Delta ~ exp(-{beta_exp:.4f} * n_B) "
                f"=> exponential closing."
            )
    elif poly_ok:
        print("  Only polynomial fit succeeded.")
    elif exp_ok:
        print("  Only exponential fit succeeded.")
    else:
        print("  Both fits failed — insufficient data or degenerate inputs.")

    print("\n--- Gap projections ---")
    print(f"{'n_B':>6}  {'Poly Delta':>14}  {'Exp Delta':>14}")
    for n_proj in [50, 100, 133]:
        n_f = float(n_proj)
        val_poly = (
            _poly_model(np.array([n_f]), a_poly, alpha_poly)[0]
            if poly_ok else float("nan")
        )
        val_exp = (
            _exp_model(np.array([n_f]), a_exp, beta_exp)[0]
            if exp_ok else float("nan")
        )
        print(f"{n_proj:>6}  {val_poly:>14.4e}  {val_exp:>14.4e}")

    # --- Gap vs k* (primary scaling) ---
    # Physical hypothesis: Delta ~ exp(-beta_k * k*) because the quantum walk
    # must tunnel through k* registers; each extra register adds a penalty
    # suppression factor.  Only use physical-gap (Path B) records for the fit.
    by_kstar: dict[int, list[float]] = {}
    for r in records_sorted:
        if r.get("gap_type") != "phys":
            continue
        kv = r.get("k_opt", 0)
        by_kstar.setdefault(kv, []).append(r["gap"])

    print("\n--- Gap vs k* (median, physical-gap records only) ---")
    print(f"  {'k*':>4}  {'median gap':>14}  {'n_pts':>6}")
    medians_by_k: dict[int, float] = {}
    for k_val in sorted(by_kstar):
        gaps = [g for g in by_kstar[k_val] if g > 1e-12]
        if not gaps:
            continue
        median_g = float(np.median(gaps))
        medians_by_k[k_val] = median_g
        print(f"  {k_val:>4}  {median_g:>14.4e}  {len(gaps):>6}")

    if len(medians_by_k) >= 2:
        ks = sorted(medians_by_k)
        k_arr = np.array(ks, dtype=np.float64)
        g_arr = np.array([medians_by_k[k] for k in ks], dtype=np.float64)
        try:
            popt_k, _ = curve_fit(
                lambda k, log_a, bk: log_a - bk * k,
                k_arr, np.log(g_arr), p0=[0.0, 1.0],
            )
            a_k = float(np.exp(popt_k[0]))
            beta_k = float(popt_k[1])
            print(f"\n  Exponential fit  Delta ~ {a_k:.4e} * exp(-{beta_k:.3f} * k*)")
            print(f"  Ratio per +1 k*: {math.exp(beta_k):.2f}x")
            print(f"  Adiabatic cost T ~ 1/Delta^2 ~ exp({2*beta_k:.3f} * k*)")
        except Exception as exc:
            # Fallback: two-point estimate
            g0, g1 = medians_by_k[ks[0]], medians_by_k[ks[-1]]
            dk = ks[-1] - ks[0]
            beta_k = math.log(g0 / g1) / dk
            print(f"\n  Two-point estimate: Delta ~ exp(-{beta_k:.2f} * k*)")
            print(f"  Gap reduction per unit k*: ~{math.exp(beta_k):.1f}x")

    # MILP vs quantum comparison
    milp_records = [r for r in records_sorted if "milp_time_s" in r and r["milp_time_s"] > 0]
    if milp_records:
        print("\n--- MILP time vs 1/gap^2 (adiabatic cost proxy) ---")
        print(f"  {'instance':>30}  {'k*':>4}  {'MILP (s)':>10}  {'1/gap^2':>14}  connected")
        for r in milp_records:
            g = r.get("gap", 0)
            inv_gap2 = 1.0 / (g * g) if g > 1e-20 else float("inf")
            print(
                f"  {r['instance']:>30}  {r['k_opt']:>4}  {r['milp_time_s']:>10.4f}"
                f"  {inv_gap2:>14.3e}  {r.get('covers_connected')}"
            )

    print()


def analyze_cover_graph(
    results: list[dict[str, Any]],
    lam: float = 5.0,
    mu: float = 10.0,
) -> None:
    """Print cover-graph metrics table and correlate with spectral gap.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Output of run_study().
    lam : float
        Coverage penalty weight to filter on.
    mu : float
        Exclusion penalty weight to filter on.
    """
    from scipy.stats import pearsonr

    records = [
        r for r in results
        if not r.get("skipped", False)
        and abs(r.get("lam", 0) - lam) < 1e-9
        and abs(r.get("mu", 0) - mu) < 1e-9
        and r.get("algebraic_connectivity") is not None
    ]

    if not records:
        logger.warning(
            "analyze_cover_graph: no valid records for lam=%.1f, mu=%.1f.", lam, mu
        )
        return

    records_sorted = sorted(records, key=lambda r: (r["k_opt"], r["n_B"], r["instance"]))

    print(f"\n=== Cover-graph metrics (lam={lam}, mu={mu}) ===")
    print(
        f"{'instance':>35}  {'k*':>4}  {'n_B':>4}  {'n_cov':>6}"
        f"  {'avg_deg':>7}  {'λ₂(L)':>10}  {'gap':>12}"
    )
    for r in records_sorted:
        print(
            f"{r['instance']:>35}  {r['k_opt']:>4}  {r['n_B']:>4}  {r['n_covers']:>6}"
            f"  {r['avg_degree']:>7.3f}  {r['algebraic_connectivity']:>10.4e}"
            f"  {r['gap']:>12.4e}"
        )

    # Pearson correlation on records with non-trivial gap
    nonzero = [r for r in records if r.get("gap", 0) > 1e-12]
    if len(nonzero) < 3:
        print(f"\n  Too few non-zero gap records ({len(nonzero)}) for correlation.")
        return

    log_gaps = np.array([math.log(r["gap"]) for r in nonzero])
    alg_conns = np.array([r["algebraic_connectivity"] for r in nonzero])

    if alg_conns.std() < 1e-12:
        print("\n  algebraic_connectivity is constant — Pearson r undefined.")
        return

    r_val, p_val = pearsonr(log_gaps, alg_conns)
    print(
        f"\n  Pearson r(log(Δ), λ₂) = {r_val:.4f}  (p={p_val:.3e}, n={len(nonzero)})"
    )
    direction = "positive" if r_val > 0 else "negative"
    print(
        f"  {'Strong' if abs(r_val) > 0.7 else 'Weak'} {direction} correlation: "
        f"{'larger' if r_val > 0 else 'smaller'} Fiedler value → "
        f"{'larger' if r_val > 0 else 'smaller'} gap."
    )

    # Per-k* summary
    print(f"\n  {'k*':>4}  {'n_pts':>6}  {'median λ₂':>12}  {'median gap':>12}")
    for kstar in sorted({r["k_opt"] for r in nonzero}):
        sub = [r for r in nonzero if r["k_opt"] == kstar]
        med_lam2 = float(np.median([r["algebraic_connectivity"] for r in sub]))
        med_gap = float(np.median([r["gap"] for r in sub]))
        print(f"  {kstar:>4}  {len(sub):>6}  {med_lam2:>12.4e}  {med_gap:>12.4e}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    results = run_study()

    for params in PARAM_SETS:
        analyze_scaling(results, lam=params["lam"], mu=params["mu"])
        analyze_cover_graph(results, lam=params["lam"], mu=params["mu"])
