"""
OR-lib SCP parser and instance analyzer.

Format (Beasley):
    m n
    cost_1 cost_2 ... cost_n        (one cost per column)
    for i = 1..m:
        n_cols_covering_row_i
        col_1 col_2 ... col_{n_i}   (1-indexed columns)

This module:
- parses OR-lib .txt files
- computes density, row coverage stats, column degree stats
- computes implied Hilbert-space dimension for the multi-register quantum walk
  given a target cover size k

For unit-cost benchmarking against our algorithm, group E (scpe1..scpe5) is
the only OR-lib family with all costs = 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SCPInstance:
    name: str
    m: int                          # number of rows (tasks, n_T)
    n: int                          # number of columns (bases, n_B)
    costs: np.ndarray               # shape (n,)
    A: np.ndarray                   # shape (m, n) bool, A[i,j]=1 if col j covers row i
    unicost: bool = False
    density: float = 0.0
    col_degree: np.ndarray = field(default_factory=lambda: np.array([]))
    row_degree: np.ndarray = field(default_factory=lambda: np.array([]))


def parse_scp(path: str | Path) -> SCPInstance:
    """Parse an OR-lib SCP file (Beasley format)."""
    path = Path(path)
    toks = path.read_text().split()
    p = 0
    m = int(toks[p]); p += 1
    n = int(toks[p]); p += 1
    costs = np.array([int(toks[p + j]) for j in range(n)], dtype=np.int32)
    p += n
    A = np.zeros((m, n), dtype=bool)
    for i in range(m):
        ni = int(toks[p]); p += 1
        cols = [int(toks[p + j]) - 1 for j in range(ni)]  # 1-indexed -> 0-indexed
        p += ni
        A[i, cols] = True
    inst = SCPInstance(
        name=path.stem,
        m=m, n=n,
        costs=costs,
        A=A,
        unicost=bool(np.all(costs == costs[0])),
        density=float(A.mean()),
        col_degree=A.sum(axis=0),   # how many rows each column covers
        row_degree=A.sum(axis=1),   # how many columns cover each row
    )
    return inst


def hilbert_dim(n_B: int, k: int) -> tuple[int, int, int]:
    """Return (q, kq, dim) = (qubits per register, total qubits, Hilbert dim)."""
    q = max(1, math.ceil(math.log2(max(n_B, 2))))
    kq = q * k
    dim = 1 << kq
    return q, kq, dim


def feasibility(dim: int) -> str:
    """Categorize feasibility under the 10-minute / 17 MFLOPS budget."""
    if dim <= 4096:
        return f"DENSE FEASIBLE (~12s, exact)"
    if dim <= 32768:
        return f"LOWDIN FEASIBLE (~0.2s, eff. Hamiltonian)"
    if dim <= 1 << 20:
        return f"ARPACK BORDERLINE (~10s-min, projected gap)"
    return f"INFEASIBLE (dim={dim:.2e})"


def analyse(inst: SCPInstance, k_assumed: int | None = None) -> dict:
    """Compute structural and quantum-resource statistics."""
    info = {
        "name": inst.name,
        "m (rows / tasks)": inst.m,
        "n (cols / bases)": inst.n,
        "unicost": inst.unicost,
        "density": round(inst.density, 4),
        "col_degree mean/median/max": (
            round(float(inst.col_degree.mean()), 2),
            int(np.median(inst.col_degree)),
            int(inst.col_degree.max()),
        ),
        "row_degree mean/median/min": (
            round(float(inst.row_degree.mean()), 2),
            int(np.median(inst.row_degree)),
            int(inst.row_degree.min()),
        ),
    }
    if k_assumed is None:
        # Beasley-paper unit-cost optima are unknown a priori; show a range.
        ks = [3, 4, 5, 6]
    else:
        ks = [k_assumed]
    info["dim_table"] = []
    for k in ks:
        q, kq, dim = hilbert_dim(inst.n, k)
        info["dim_table"].append({
            "k": k, "q": q, "total_qubits": kq, "dim": dim,
            "feasibility": feasibility(dim),
        })
    # One-hot QAOA qubit count for comparison
    info["onehot_qubits_QAOA"] = inst.n
    return info


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    ORLIB_DIR = HERE.parent / "orlib_instances"
    files = sorted(ORLIB_DIR.glob("scp*.txt"))
    for f in files:
        try:
            inst = parse_scp(f)
        except Exception as exc:
            print(f"SKIP {f.name}: {exc}")
            continue
        info = analyse(inst)
        print(f"\n=== {info['name']} ===")
        for kk, vv in info.items():
            if kk == "dim_table":
                continue
            print(f"  {kk}: {vv}")
        for row in info["dim_table"]:
            print(
                f"  k={row['k']}: q={row['q']}, total={row['total_qubits']} qubits, "
                f"dim={row['dim']:>15,}  {row['feasibility']}"
            )
        print(f"  one-hot QAOA qubits: {info['onehot_qubits_QAOA']}")
