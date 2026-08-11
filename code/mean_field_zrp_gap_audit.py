#!/usr/bin/env python3
"""Audit the mean-field zero-range theorem at finite size.

The script does not certify the all-size inequality.  It evaluates the
three-box conditional operator entering the Caputo--Sasada reduction and,
independently, the complete-graph zero-range gap at small sizes.  The theorem
uses an Efron monotonicity coupling; this output is only a regression test.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Iterator

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cycle_family_exploration import ROOT


OUTPUT = ROOT / "data" / "mean_field_zrp_gap_audit.json"
TAUS = [1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-5, 1e-7]
MAX_THREE_BOX_TOTAL = 80
MEAN_FIELD_SIZES = range(2, 9)
MEAN_FIELD_TAUS = [1.0, 0.1, 1e-3, 1e-7]


def conditional_operator_spectrum(total: int, tau: float) -> np.ndarray:
    """Spectrum of K_{m,tau} in its reversible one-site marginal."""
    weights = np.ones(total + 1, dtype=float)
    weights[0] = tau
    marginal = np.empty(total + 1, dtype=float)
    operator = np.zeros((total + 1, total + 1), dtype=float)
    for first in range(total + 1):
        remainder = total - first
        conditional = np.asarray(
            [weights[j] * weights[remainder - j] for j in range(remainder + 1)]
        )
        normalization = float(np.sum(conditional))
        operator[first, : remainder + 1] = conditional / normalization
        marginal[first] = weights[first] * normalization
    marginal /= np.sum(marginal)
    root = np.sqrt(marginal)
    symmetric = root[:, None] * operator / root[None, :]
    symmetric = 0.5 * (symmetric + symmetric.T)
    return np.linalg.eigvalsh(symmetric)


def three_box_audit(tau: float) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []
    for total in range(1, MAX_THREE_BOX_TOTAL + 1):
        spectrum = conditional_operator_spectrum(total, tau)
        mu_min = float(spectrum[0])
        mu_max_nonconstant = float(spectrum[-2])
        heat_bath_gap = min(
            (2.0 + mu_min) / 3.0,
            (2.0 - 2.0 * mu_max_nonconstant) / 3.0,
        )
        rows.append(
            {
                "total": total,
                "mu_min": mu_min,
                "mu_max_nonconstant": mu_max_nonconstant,
                "three_box_heat_bath_gap": heat_bath_gap,
            }
        )
    worst_mu = max(rows, key=lambda row: float(row["mu_max_nonconstant"]))
    worst_gap = min(rows, key=lambda row: float(row["three_box_heat_bath_gap"]))
    return {
        "tau": tau,
        "max_total_checked": MAX_THREE_BOX_TOTAL,
        "largest_nonconstant_positive_eigenvalue": worst_mu,
        "smallest_three_box_heat_bath_gap": worst_gap,
        "target_mu_upper_bound": 0.5,
        "target_gap_lower_bound": 1.0 / 3.0,
    }


def weak_compositions(boxes: int, particles: int) -> Iterator[tuple[int, ...]]:
    if boxes == 1:
        yield (particles,)
        return
    for first in range(particles + 1):
        for tail in weak_compositions(boxes - 1, particles - first):
            yield (first,) + tail


def mean_field_zrp_gap(k: int, tau: float) -> float:
    states = list(weak_compositions(k, k))
    index = {state: i for i, state in enumerate(states)}
    stationary = np.asarray(
        [tau ** sum(value == 0 for value in state) for state in states], dtype=float
    )
    stationary /= np.sum(stationary)
    row: list[int] = []
    column: list[int] = []
    value: list[float] = []
    for source, state in enumerate(states):
        diagonal = 0.0
        for donor, receiver in itertools.permutations(range(k), 2):
            occupancy = state[donor]
            rate = 0.0 if occupancy == 0 else tau if occupancy == 1 else 1.0
            if rate == 0.0:
                continue
            target = list(state)
            target[donor] -= 1
            target[receiver] += 1
            transition = rate / k
            row.append(source)
            column.append(index[tuple(target)])
            value.append(transition)
            diagonal -= transition
        row.append(source)
        column.append(source)
        value.append(diagonal)
    generator = sp.csr_matrix((value, (row, column)), shape=(len(states), len(states)))
    root = np.sqrt(stationary)
    discriminant = sp.diags(root) @ (-generator) @ sp.diags(1.0 / root)
    eigenvalues = np.sort(
        spla.eigsh(
            discriminant,
            k=min(3, len(states) - 1),
            which="SM",
            return_eigenvectors=False,
            tol=1e-10,
        )
    )
    return float(eigenvalues[1])


def main() -> None:
    three_box = [three_box_audit(tau) for tau in TAUS]
    mean_field: list[dict[str, float | int]] = []
    for tau in MEAN_FIELD_TAUS:
        for k in MEAN_FIELD_SIZES:
            gap = mean_field_zrp_gap(k, tau)
            conditional_bound = 2.0 * tau / (k**3)
            mean_field.append(
                {
                    "tau": tau,
                    "boxes_and_particles": k,
                    "gap": gap,
                    "analytic_lower_bound": conditional_bound,
                    "gap_to_analytic_bound_ratio": gap / conditional_bound,
                }
            )
    output = {
        "schema_version": 1,
        "qualification": (
            "finite-size audit only; the all-total gap theorem is proved separately by "
            "Efron monotonicity, bounded split coupling, and Caputo--Sasada recursion"
        ),
        "three_box": three_box,
        "mean_field_zero_range": mean_field,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for row in three_box:
        mu = row["largest_nonconstant_positive_eigenvalue"]
        gap = row["smallest_three_box_heat_bath_gap"]
        print(
            f"tau={row['tau']:.3g}: max mu={mu['mu_max_nonconstant']:.10g} "
            f"at m={mu['total']}; min gap3={gap['three_box_heat_bath_gap']:.10g} "
            f"at m={gap['total']}"
        )


if __name__ == "__main__":
    main()
