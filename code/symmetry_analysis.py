"""
Symmetry analysis for the Multi-Register Quantum Walk (Minimum Set Cover).

Goal: explain why some instances have *exact* ground-state degeneracy
(gap = 0) even when the cover graph is connected (lambda_2 > 0).

Approach:
  1. Decompose the degenerate E0 subspace by S_k (register-permutation) irreps.
  2. Look for additional Hamiltonian symmetries that come from instance-level
     base permutations leaving the adjacency matrix A invariant.
  3. For grid_2x4, scan lambda to find the level crossing between lam=2 and
     lam=5 that turns a small physical gap into an exact gap=0.

Reference: spectral_gap_study.py (build_hamiltonian, random_msc_instance,
solve_milp_msc, grid_instance, _decode_bases).
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_gap_study import (
    build_hamiltonian,
    grid_instance,
    random_msc_instance,
    solve_milp_msc,
    compute_cover_graph_metrics,
    _decode_bases,
)


# ---------------------------------------------------------------------------
# Permutation matrices for S_k acting on register labels
# ---------------------------------------------------------------------------


def build_register_perm(perm: tuple[int, ...], k: int, reg_dim: int) -> sp.csr_matrix:
    """Build the unitary that permutes register labels by ``perm``.

    A computational state |j_1, ..., j_k> with index
        idx = sum_r j_r * reg_dim**r
    is mapped to |j_{perm^{-1}(1)}, ..., j_{perm^{-1}(k)}>.

    Equivalently: P_perm @ |idx> = |idx_new>, where the new register r holds
    the old register perm[r].

    Parameters
    ----------
    perm : tuple of int
        ``perm[r] = s`` means new register r takes its value from old register s.
        Equivalently, applying perm twice with ``perm o perm`` composes naturally.
    k : int
    reg_dim : int

    Returns
    -------
    sp.csr_matrix of shape (dim, dim).
    """
    dim = reg_dim ** k
    bases = _decode_bases(dim, k, reg_dim)  # (dim, k)
    # New bases: new_bases[idx, r] = bases[idx, perm[r]]
    new_bases = bases[:, list(perm)]
    # Re-encode: new_idx = sum_r new_bases[idx, r] * reg_dim**r
    powers = np.array([reg_dim ** r for r in range(k)], dtype=np.int64)
    new_idx = (new_bases.astype(np.int64) * powers[np.newaxis, :]).sum(axis=1)
    rows = new_idx
    cols = np.arange(dim, dtype=np.int64)
    data = np.ones(dim, dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))


def build_base_perm(
    sigma: tuple[int, ...], k: int, n_B: int, reg_dim: int
) -> sp.csr_matrix:
    """Build the unitary that relabels base values inside each register.

    A computational state |j_1, ..., j_k> is mapped to
    |sigma(j_1), ..., sigma(j_k)>, where sigma is a permutation of {0,...,n_B-1}
    extended trivially over the padding range [n_B, reg_dim).
    """
    dim = reg_dim ** k
    bases = _decode_bases(dim, k, reg_dim)  # (dim, k)
    sigma_arr = np.arange(reg_dim, dtype=np.int64)
    sigma_arr[: n_B] = np.array(sigma, dtype=np.int64)
    new_bases = sigma_arr[bases]
    powers = np.array([reg_dim ** r for r in range(k)], dtype=np.int64)
    new_idx = (new_bases.astype(np.int64) * powers[np.newaxis, :]).sum(axis=1)
    rows = new_idx
    cols = np.arange(dim, dtype=np.int64)
    data = np.ones(dim, dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))


# ---------------------------------------------------------------------------
# S_k irrep projectors via group averaging
# ---------------------------------------------------------------------------


def _sk_irreps_characters(k: int) -> dict[str, dict[tuple[int, ...], float]]:
    """Character table of S_k for k in {3, 4}.

    Returns a dict of {irrep_name: {permutation: chi(perm)}}.
    Uses cycle-type characters for the standard irreps.
    """
    perms = list(itertools.permutations(range(k)))

    def cycle_type(p: tuple[int, ...]) -> tuple[int, ...]:
        seen = [False] * k
        cycles = []
        for i in range(k):
            if seen[i]:
                continue
            j = i
            length = 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                length += 1
            cycles.append(length)
        return tuple(sorted(cycles, reverse=True))

    if k == 3:
        # Cycle types: (1,1,1) -> id (1 elt), (2,1) -> transpositions (3 elts),
        #              (3,) -> 3-cycles (2 elts).
        # Characters from standard S_3 character table:
        chars = {
            "trivial": {(1, 1, 1): 1, (2, 1): 1, (3,): 1},
            "sign": {(1, 1, 1): 1, (2, 1): -1, (3,): 1},
            "standard": {(1, 1, 1): 2, (2, 1): 0, (3,): -1},
        }
    elif k == 4:
        # Cycle types of S_4 and their sizes:
        #   (1,1,1,1) -> id (1), (2,1,1) -> 6, (2,2) -> 3, (3,1) -> 8, (4,) -> 6
        # 5 irreps: trivial, sign, standard (3-dim), standard x sign (3-dim),
        #           2-dim (E).
        chars = {
            "trivial":     {(1, 1, 1, 1): 1, (2, 1, 1): 1, (2, 2): 1, (3, 1): 1, (4,): 1},
            "sign":        {(1, 1, 1, 1): 1, (2, 1, 1): -1, (2, 2): 1, (3, 1): 1, (4,): -1},
            "standard":    {(1, 1, 1, 1): 3, (2, 1, 1): 1, (2, 2): -1, (3, 1): 0, (4,): -1},
            "standard_x_sign": {(1, 1, 1, 1): 3, (2, 1, 1): -1, (2, 2): -1, (3, 1): 0, (4,): 1},
            "two_dim":     {(1, 1, 1, 1): 2, (2, 1, 1): 0, (2, 2): 2, (3, 1): -1, (4,): 0},
        }
    else:
        raise ValueError(f"Character table not implemented for k={k}")

    out: dict[str, dict[tuple[int, ...], float]] = {}
    for name, ctype_chi in chars.items():
        out[name] = {p: float(ctype_chi[cycle_type(p)]) for p in perms}
    return out


def build_irrep_projectors(
    k: int, reg_dim: int
) -> dict[str, sp.csr_matrix]:
    """Build orthogonal projectors onto each S_k irrep sector.

    For an irrep alpha with character chi_alpha and dimension d_alpha,
        P_alpha = (d_alpha / |G|) sum_{g in G} chi_alpha(g) * rho(g).
    """
    chars = _sk_irreps_characters(k)
    perms = list(itertools.permutations(range(k)))
    G = math.factorial(k)

    perm_mats: dict[tuple[int, ...], sp.csr_matrix] = {
        p: build_register_perm(p, k, reg_dim) for p in perms
    }

    projectors: dict[str, sp.csr_matrix] = {}
    identity = tuple(range(k))
    for name, chi in chars.items():
        d_alpha = chi[identity]  # dimension = chi(identity)
        coeff = d_alpha / G
        P = sp.csr_matrix(perm_mats[identity].shape, dtype=np.float64)
        for p in perms:
            P = P + coeff * chi[p] * perm_mats[p]
        projectors[name] = P.tocsr()
    return projectors


# ---------------------------------------------------------------------------
# Cover-set automorphisms: which base permutations leave A invariant?
# ---------------------------------------------------------------------------


def find_base_automorphisms(A: np.ndarray, max_perms: int = 100_000) -> list[tuple[int, ...]]:
    """Return all base permutations that are automorphisms of the bipartite
    incidence graph of A.

    Build the bipartite graph G with task nodes 0..n_T-1 and base nodes
    n_T..n_T+n_B-1; an edge (t, n_T+b) exists iff A[t, b] is nonzero. We then
    enumerate all graph automorphisms of G that preserve the bipartition (left
    nodes map to left, right to right) using networkx VF2, and project onto
    the base-side permutation. This catches *joint* base-task symmetries
    (e.g. rotating a 2×4 grid permutes both bases AND tasks simultaneously),
    which the previous row-signature grouping missed.

    Note: A is passed with shape (n_B, n_T) elsewhere in this codebase.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix, shape (n_B, n_T) — rows are bases, columns tasks.
    max_perms : int
        Soft limit on the number of automorphisms returned (kept as a
        defensive cap; VF2 is fine for n_B <= ~16).

    Returns
    -------
    list of base permutations as tuples of length n_B.
    """
    n_B, n_T = A.shape

    # Build bipartite incidence graph.
    G = nx.Graph()
    G.add_nodes_from(range(n_T), bipartite=0)               # task nodes
    G.add_nodes_from(range(n_T, n_T + n_B), bipartite=1)    # base nodes
    for b in range(n_B):
        for t in range(n_T):
            if A[b, t]:
                G.add_edge(t, n_T + b)

    # VF2 isomorphism search: only allow task<->task and base<->base maps.
    GM = nx.algorithms.isomorphism.GraphMatcher(
        G, G,
        node_match=lambda u, v: u.get("bipartite") == v.get("bipartite"),
    )

    base_perms: set[tuple[int, ...]] = set()
    for iso in GM.isomorphisms_iter():
        # iso maps node -> node. Project onto the base side: new register r is
        # filled with base iso[n_T + r] - n_T from the original instance.
        base_perm = tuple(iso[n_T + b] - n_T for b in range(n_B))
        base_perms.add(base_perm)
        if len(base_perms) >= max_perms:
            break

    if not base_perms:
        # Fallback (should never happen — identity is always an automorphism).
        base_perms.add(tuple(range(n_B)))
    return sorted(base_perms)


# ---------------------------------------------------------------------------
# Eigensystem and degeneracy analysis
# ---------------------------------------------------------------------------


def analyse_instance(
    name: str,
    A: np.ndarray,
    k: int,
    lam: float,
    mu: float,
    nu: float,
    deg_tol: float = 1e-6,
) -> dict[str, Any]:
    """Run the full symmetry analysis on a single instance.

    Returns a dict with E0, the degeneracy of E0, the irrep projection weights
    of the degenerate subspace, the cover graph metrics, and the size of the
    base-automorphism group.
    """
    print(f"\n{'='*72}")
    print(f"INSTANCE: {name}")
    print(f"  n_B={A.shape[0]}, n_T={A.shape[1]}, k={k}, lam={lam}, mu={mu}, nu={nu}")

    n_B = A.shape[0]
    q = math.ceil(math.log2(n_B)) if n_B > 1 else 1
    reg_dim = 2 ** q
    dim = reg_dim ** k
    print(f"  q={q}, reg_dim={reg_dim}, dim={dim}")

    # Optimal covers and cover-graph metrics
    k_milp, covers, _, _ = solve_milp_msc(A)
    cg = compute_cover_graph_metrics(covers)
    print(f"  k*={k_milp}, n_covers={cg['n_covers']}, "
          f"lambda_2={cg['algebraic_connectivity']:.4f}, "
          f"connected={cg['is_connected']}")

    # Build H and exact spectrum
    H, _ = build_hamiltonian(A, k, lam, mu, nu)
    H_dense = H.toarray()
    vals, vecs = np.linalg.eigh(H_dense)
    E0 = float(vals[0])
    print(f"  E0 = {E0:.10f}")

    # Find degenerate ground subspace
    deg_mask = np.abs(vals - E0) < deg_tol
    deg_count = int(deg_mask.sum())
    print(f"  Degeneracy of E0 (|E_i - E0| < {deg_tol}): {deg_count}")
    print(f"  Lowest 6 eigenvalues:")
    for i in range(min(6, len(vals))):
        marker = " <-- E0" if deg_mask[i] else ""
        print(f"    E_{i} = {vals[i]:.10f}{marker}")
    if deg_count < len(vals):
        E_next = float(vals[deg_count])
        print(f"  First eigenvalue above the degenerate manifold: "
              f"E_{deg_count} = {E_next:.10f}  (gap to manifold = {E_next - E0:.4e})")

    # Project degenerate subspace onto S_k irreps via group averaging
    projectors = build_irrep_projectors(k, reg_dim)
    V_deg = vecs[:, deg_mask]  # (dim, deg_count)
    print(f"  Irrep weights of the E0 subspace (sum of |<P_alpha psi_i>|^2):")
    irrep_weights: dict[str, float] = {}
    for name_alpha, P_alpha in projectors.items():
        # weight_i = || P_alpha v_i ||^2; total weight summed over basis of V_deg
        Pv = P_alpha @ V_deg  # (dim, deg_count)
        weight = float(np.sum(np.abs(Pv) ** 2))
        irrep_weights[name_alpha] = weight
        # Also compute dimension of irrep sector restricted to the degenerate
        # manifold: trace(V_deg^T P_alpha V_deg).
        gram = V_deg.T @ (P_alpha @ V_deg)
        sector_dim = float(np.trace(gram))
        print(f"    {name_alpha:>20}: weight = {weight:.6f}   "
              f"sector_dim_in_E0 = {sector_dim:.4f}")

    # Base-automorphism group
    auts = find_base_automorphisms(A)
    n_auts = len(auts)
    print(f"  Base-automorphism group |Aut_base(A)| = {n_auts}  "
          f"(within-row-signature-class permutations)")
    if n_auts > 1 and n_auts <= 12:
        print(f"  Sample automorphisms (non-identity):")
        shown = 0
        for sigma in auts:
            if sigma == tuple(range(n_B)):
                continue
            print(f"    sigma = {sigma}")
            shown += 1
            if shown >= 4:
                break

    # Detect "block-equivalent" cover groups: covers that are images of each
    # other under a base automorphism collapse to one orbit.
    cover_set = [frozenset(c) for c in covers]
    orbit_map: dict[int, int] = {}
    next_orbit = 0
    for i, c in enumerate(cover_set):
        if i in orbit_map:
            continue
        orbit_map[i] = next_orbit
        for sigma in auts:
            c_img = frozenset(sigma[b] for b in c)
            if c_img in cover_set:
                j = cover_set.index(c_img)
                if j not in orbit_map:
                    orbit_map[j] = next_orbit
        next_orbit += 1
    n_orbits = len(set(orbit_map.values()))
    print(f"  Cover orbits under Aut_base: {n_orbits} orbits over {len(covers)} covers")

    # Extra diagnostic: does the lowest-energy subspace of H_proj already
    # contain the degenerate manifold? Compare H_proj eigenvalues directly.
    diag = H_dense.diagonal()
    valid_idx = np.where(diag < 1e-10)[0]
    n_valid = len(valid_idx)
    if n_valid >= 2:
        H_proj = H_dense[np.ix_(valid_idx, valid_idx)]
        vals_proj = np.linalg.eigvalsh(H_proj)
        # Look for the equivalent degenerate manifold inside H_proj
        e0_proj = float(vals_proj[0])
        deg_proj = int(np.sum(np.abs(vals_proj - e0_proj) < deg_tol))
        print(f"  H_proj subspace: n_valid={n_valid}, E0_proj={e0_proj:.6f}, "
              f"deg_proj={deg_proj}")
        print(f"  H_proj 6 lowest: {[f'{v:.6f}' for v in vals_proj[:6]]}")

    return {
        "name": name,
        "k": k,
        "n_B": A.shape[0],
        "n_T": A.shape[1],
        "E0": E0,
        "degeneracy": deg_count,
        "lowest_eigs": vals[:6].tolist(),
        "irrep_weights": irrep_weights,
        "n_aut_base": n_auts,
        "n_cover_orbits": n_orbits,
        "n_covers": cg["n_covers"],
        "lambda_2": cg["algebraic_connectivity"],
    }


# ---------------------------------------------------------------------------
# Level-crossing scan for grid_2x4
# ---------------------------------------------------------------------------


def scan_lambda_grid_2x4(
    mu: float = 10.0,
    nu: float = 50.0,
    lam_values: list[float] | None = None,
    n_eigs: int = 6,
) -> list[dict[str, float]]:
    """Scan lambda for grid_2x4 and report the lowest eigenvalues.

    Identifies the level crossing between lam=2 (gap = 2.93e-4) and lam=5
    (gap = 0).
    """
    if lam_values is None:
        # Coarse scan + fine sweep across the suspected crossing window
        lam_values = sorted(set(
            [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0]
            + list(np.round(np.linspace(3.50, 3.80, 13), 4).tolist())
        ))

    A = grid_instance(2, 4)
    k = 3
    print(f"\n{'='*72}")
    print(f"LEVEL CROSSING SCAN: grid_2x4, mu={mu}, nu={nu}")
    print(f"  lambda values: {lam_values}")

    rows: list[dict[str, float]] = []
    print(f"\n  {'lam':>6}  " + "  ".join(f"E_{i}".rjust(13) for i in range(n_eigs))
          + f"  {'gap':>11}")
    for lam in lam_values:
        H, _ = build_hamiltonian(A, k, lam, mu, nu)
        vals = np.linalg.eigvalsh(H.toarray())
        E0 = float(vals[0])
        E1 = float(vals[1])
        gap = E1 - E0
        eig_str = "  ".join(f"{vals[i]:>13.7f}" for i in range(n_eigs))
        print(f"  {lam:>6.2f}  {eig_str}  {gap:>11.4e}")
        rows.append({"lam": lam, "E0": E0, "E1": E1, "gap": gap,
                     "eigs": vals[:n_eigs].tolist()})
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    lam, mu, nu = 5.0, 10.0, 50.0

    print("="*72)
    print("SYMMETRY ANALYSIS OF MULTI-REGISTER QUANTUM WALK MSC HAMILTONIAN")
    print("="*72)
    print(f"Parameters: lam={lam}, mu={mu}, nu={nu}")

    # gap=0 instances
    instances: list[tuple[str, np.ndarray, int]] = []
    instances.append(
        ("rand_nB8_p50_s002 (gap=0)", random_msc_instance(8, 7, 0.50, seed=2), 3)
    )
    instances.append(
        ("rand_nB8_p42_s014 (gap=0)", random_msc_instance(8, 8, 0.42, seed=14), 4)
    )
    instances.append(
        ("grid_2x4 (gap=0 at lam=5)", grid_instance(2, 4), 3)
    )
    # gap > 0 controls
    instances.append(
        ("rand_nB8_p42_s009 (gap>0)", random_msc_instance(8, 8, 0.42, seed=9), 4)
    )
    instances.append(
        ("rand_nB8_p50_s000 (gap>0)", random_msc_instance(8, 7, 0.50, seed=0), 3)
    )

    summaries: list[dict[str, Any]] = []
    for name, A, k in instances:
        try:
            summary = analyse_instance(name, A, k, lam, mu, nu)
            summaries.append(summary)
        except Exception as exc:
            print(f"\n  !! Analysis failed for {name}: {exc}")
            import traceback
            traceback.print_exc()

    # Level-crossing scan for grid_2x4
    scan_rows = scan_lambda_grid_2x4(mu=mu, nu=nu)

    # ----- Summary table -----
    print(f"\n{'='*72}")
    print("SUMMARY TABLE")
    print("="*72)
    print(f"{'instance':>32}  {'k':>2}  {'n_cov':>6}  "
          f"{'deg(E0)':>8}  {'|Aut|':>6}  {'orbits':>7}  "
          f"{'top irrep':>16}")
    for s in summaries:
        top_irrep = max(s["irrep_weights"].items(), key=lambda x: x[1])
        print(f"{s['name']:>32}  {s['k']:>2}  {s['n_covers']:>6}  "
              f"{s['degeneracy']:>8}  {s['n_aut_base']:>6}  {s['n_cover_orbits']:>7}  "
              f"{top_irrep[0]:>10} ({top_irrep[1]:>4.2f})")

    # ----- grid_2x4 level crossing -----
    if scan_rows:
        gap_arr = np.array([r["gap"] for r in scan_rows])
        lam_arr = np.array([r["lam"] for r in scan_rows])
        i_min = int(np.argmin(gap_arr))
        print(f"\nLevel crossing for grid_2x4:")
        print(f"  Minimum gap = {gap_arr[i_min]:.4e} at lambda = {lam_arr[i_min]:.3f}")
        # Look for sign change in (E1 - E2)
        for i in range(len(scan_rows) - 1):
            row_a = scan_rows[i]
            row_b = scan_rows[i + 1]
            d_a = row_a["eigs"][1] - row_a["eigs"][2]
            d_b = row_b["eigs"][1] - row_b["eigs"][2]
            if d_a * d_b < 0:
                print(f"  Sign change in (E1 - E2) between lam={row_a['lam']:.2f} "
                      f"and lam={row_b['lam']:.2f}")


if __name__ == "__main__":
    main()
