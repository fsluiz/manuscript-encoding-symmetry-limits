#!/usr/bin/env python3
"""Reproduce the sector-localisation certificate of Proposition 5.

The audit works in the exact register-symmetric quotient, resolves the five
``D4`` isotypic components of the frozen grid instance, and also evaluates
the register-symmetric sector of the frozen asymmetric random instance.  It
reports every term entering the proposition, rather than inferring leakage
from an effective Hamiltonian.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from accessibility_prototype import (
    LAM,
    MU,
    NU,
    cover_mask,
    eigensystem,
    joint_orbits,
    level_summary,
    orbit_isometry,
    quotient,
)
from d4_sector_analysis import d4_characters, identify_d4, projector_basis
from instance_registry import load_instance
from spectral_gap_study import build_hamiltonian
from symmetry_analysis import build_base_perm, find_base_automorphisms


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "sector_localization_audit.json"
TABLE_OUTPUT = ROOT / "sections" / "generated_sector_localization_table.tex"
GRID_ID = "grid-2x4-v1"
RANDOM_ID = "random-b8-t7-p50-s002-v1"


def _symmetric_problem(
    instance_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return S_k-trivial H, H_walk, P, isometry, and incidence data."""
    incidence, costs, record = load_instance(
        instance_id, require_status="candidate_primary"
    )
    if costs is not None:
        raise AssertionError("sector audit is restricted to unweighted instances")
    n_b = int(incidence.shape[0])
    k = int(record["k_star"])
    if 2 ** math.ceil(math.log2(n_b)) != n_b:
        raise ValueError("audit expects no padding in the selected frozen rows")

    h_problem, _ = build_hamiltonian(incidence, k, LAM, MU, NU)
    h_walk, _ = build_hamiltonian(incidence, k, 0.0, 0.0, 0.0)
    identity = tuple(range(n_b))
    orbits = joint_orbits(n_b, k, [identity])
    isometry = orbit_isometry(orbits, n_b**k)
    h_problem_sym = quotient(h_problem, isometry)
    h_walk_sym = quotient(h_walk, isometry)
    p_full = sp.diags(cover_mask(incidence, k).astype(float), format="csr")
    p_sym = np.asarray((isometry.T @ p_full @ isometry).toarray())
    return (
        h_problem_sym,
        h_walk_sym,
        p_sym,
        np.asarray(isometry.toarray()),
        incidence,
        record,
    )


def _projector_split(projector: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    projector = (projector + projector.T) / 2
    values, vectors = np.linalg.eigh(projector)
    p_basis = vectors[:, values > 0.5]
    q_basis = vectors[:, values <= 0.5]
    residuals = {
        "projector_idempotence_residual": float(
            np.linalg.norm(projector @ projector - projector, ord=2)
        ),
        "projector_spectral_residual": float(
            np.max(np.minimum(np.abs(values), np.abs(values - 1.0)))
        ),
    }
    return p_basis, q_basis, residuals


def _sector_certificate(
    label: str,
    sector_basis: np.ndarray,
    h_problem_sym: np.ndarray,
    h_walk_sym: np.ndarray,
    p_sym: np.ndarray,
    n_b: int,
    k: int,
) -> dict[str, Any]:
    """Evaluate the exact and sufficient sector-localisation bounds."""
    h_problem = sector_basis.T @ h_problem_sym @ sector_basis
    h_walk = sector_basis.T @ h_walk_sym @ sector_basis
    p_sector = sector_basis.T @ p_sym @ sector_basis
    p_basis, q_basis, projector_residuals = _projector_split(p_sector)

    if p_basis.shape[1] == 0:
        return {
            "label": label,
            "sector_dimension": int(sector_basis.shape[1]),
            "feasible_rank": 0,
            "invalid_rank": int(q_basis.shape[1]),
            "qualification": "Proposition hypothesis P_gamma != 0 is not met",
            **projector_residuals,
        }

    eta = float(np.linalg.eigvalsh(p_basis.T @ h_walk @ p_basis)[0])
    coupling = q_basis.T @ h_walk @ p_basis
    coupling_norm = float(np.linalg.norm(coupling, ord=2)) if q_basis.shape[1] else 0.0

    values, vectors, residuals = eigensystem(h_problem)
    summary = level_summary(values, float(np.linalg.norm(h_problem, ord=2)), residuals)
    ground_rank = int(summary["ground_rank"])
    ground = vectors[:, :ground_rank]
    cover_effect = ground.T @ p_sector @ ground
    cover_probabilities = np.linalg.eigvalsh((cover_effect + cover_effect.T) / 2)
    observed_leakage = float(max(0.0, 1.0 - cover_probabilities[0]))

    kappa = k / (n_b - 1)
    m = min(LAM, MU)
    g_sufficient = float(m - kappa - eta)
    naive_separation = float(m - kappa)

    if q_basis.shape[1] == 0:
        gamma = math.inf
        exact_bound = 0.0
        sufficient_bound = 0.0
        q_floor = math.inf
    else:
        q_floor = float(
            np.linalg.eigvalsh(q_basis.T @ h_problem @ q_basis)[0]
        )
        gamma = float(q_floor - values[0])
        if gamma <= 0:
            raise AssertionError(f"{label}: compressed separation is not positive")
        exact_bound = float(coupling_norm**2 / (gamma**2 + coupling_norm**2))
        sufficient_bound = (
            float(coupling_norm**2 / (g_sufficient**2 + coupling_norm**2))
            if g_sufficient > 0
            else None
        )

    scale = max(1.0, float(np.linalg.norm(h_problem, ord=2)))
    tolerance = 500.0 * np.finfo(float).eps * scale
    if observed_leakage > exact_bound + tolerance:
        raise AssertionError(f"{label}: observed leakage exceeds exact certificate")
    if gamma < g_sufficient - tolerance:
        raise AssertionError(f"{label}: Gamma_gamma lower bound failed")

    return {
        "label": label,
        "sector_dimension": int(sector_basis.shape[1]),
        "feasible_rank": int(p_basis.shape[1]),
        "invalid_rank": int(q_basis.shape[1]),
        "ground_rank": ground_rank,
        "E0_gamma": float(values[0]),
        "eta_gamma": eta,
        "kappa": float(kappa),
        "m": float(m),
        "naive_m_minus_kappa": naive_separation,
        "naive_quantity_is_not_a_sector_certificate": True,
        "g_gamma_sufficient": g_sufficient,
        "invalid_compressed_floor": q_floor,
        "Gamma_gamma": gamma,
        "B_gamma_norm": coupling_norm,
        "observed_worst_ground_leakage": observed_leakage,
        "exact_Gamma_bound": exact_bound,
        "sufficient_kinetic_floor_bound": sufficient_bound,
        "maximum_eigenpair_residual": float(residuals.max()),
        "certificate_roundoff_tolerance": float(tolerance),
        "Gamma_minus_sufficient_floor": float(gamma - g_sufficient),
        "exact_bound_minus_observed_leakage": float(
            exact_bound - observed_leakage
        ),
        **projector_residuals,
    }


def _grid_audit() -> dict[str, Any]:
    h_problem, h_walk, p_sym, isometry, incidence, record = _symmetric_problem(
        GRID_ID
    )
    n_b = int(incidence.shape[0])
    k = int(record["k_star"])
    group = find_base_automorphisms(incidence)
    _, _, coordinates = identify_d4(group)
    characters = d4_characters(coordinates)
    dimensions = {"A1": 1, "A2": 1, "B1": 1, "B2": 1, "E": 2}
    representations = {
        g: isometry.T
        @ np.asarray(build_base_perm(g, k, n_b, n_b).toarray())
        @ isometry
        for g in group
    }

    sectors: dict[str, Any] = {}
    for name, character in characters.items():
        projector = sum(character[g] * representations[g] for g in group)
        projector *= dimensions[name] / len(group)
        basis, residual = projector_basis(projector)
        result = _sector_certificate(
            f"S_{k}-trivial x D4-{name}",
            basis,
            h_problem,
            h_walk,
            p_sym,
            n_b,
            k,
        )
        result["isotypic_projector_idempotence_residual"] = residual
        sectors[name] = result

    return {
        "instance_id": GRID_ID,
        "payload_sha256": record["payload_sha256"],
        "group": "S_3 x D4, with D4 of order 8",
        "sectors": sectors,
    }


def _random_audit() -> dict[str, Any]:
    h_problem, h_walk, p_sym, _, incidence, record = _symmetric_problem(RANDOM_ID)
    n_b = int(incidence.shape[0])
    k = int(record["k_star"])
    basis = np.eye(h_problem.shape[0])
    return {
        "instance_id": RANDOM_ID,
        "payload_sha256": record["payload_sha256"],
        "group": "S_3 (effective base group is trivial)",
        "sectors": {
            "trivial": _sector_certificate(
                "S_3-trivial",
                basis,
                h_problem,
                h_walk,
                p_sym,
                n_b,
                k,
            )
        },
    }


def analyse() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "proposition": "sector localisation with kinetic floor",
        "parameters": {"lambda": LAM, "mu": MU, "nu_padding": NU},
        "instances": [_grid_audit(), _random_audit()],
        "interpretation": (
            "The exact Gamma certificate and its kinetic-floor lower bound "
            "are checked against the worst leakage in each sector ground "
            "manifold. The quantity m-kappa alone is recorded only to show "
            "what the sector proof would omit; it is not asserted as a bound."
        ),
    }


def write_table(result: dict[str, Any]) -> None:
    """Write the compact appendix table directly from the audited values."""
    def display(value: float) -> str:
        if abs(value) < 5e-13:
            return "0"
        return f"{value:.6g}"

    lines = [
        "% Generated by code/sector_localization_audit.py; do not edit.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Direct audit of Proposition~\\ref{prop:sector-localization-rewrite}. "
        "The reported leakage is the worst value in the sector ground manifold; "
        "$b_{\\Gamma}$ and $b_g$ are the right-hand side of "
        "Eq.~\\eqref{eq:sector-bound-rewrite} using the exact separation "
        "$\\Gamma_\\gamma$ and the sufficient kinetic-floor separation "
        "$g_\\gamma=m-\\kappa-\\eta_\\gamma$, respectively.}",
        "\\label{tab:sector-localization-audit}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\hline",
        "instance & sector $\\gamma$ & $d_\\gamma$ & $\\operatorname{rank}P_\\gamma$ & "
        "$\\eta_\\gamma$ & $\\Gamma_\\gamma$ & $\\|B_\\gamma\\|$ & leakage & "
        "$b_{\\Gamma}$ & $b_g$\\\\",
        "\\hline",
    ]
    for instance in result["instances"]:
        short_name = (
            "grid-2x4"
            if instance["instance_id"] == GRID_ID
            else "rnd-b8-t7-p50-s002"
        )
        for name, sector in instance["sectors"].items():
            if sector["feasible_rank"] == 0:
                continue
            sector_name = f"$[3]\\times {name}$" if short_name == "grid-2x4" else "$[3]$"
            lines.append(
                f"{short_name} & {sector_name} & {sector['sector_dimension']} & "
                f"{sector['feasible_rank']} & {display(sector['eta_gamma'])} & "
                f"{display(sector['Gamma_gamma'])} & {display(sector['B_gamma_norm'])} & "
                f"{display(sector['observed_worst_ground_leakage'])} & "
                f"{display(sector['exact_Gamma_bound'])} & "
                f"{display(sector['sufficient_kinetic_floor_bound'])}\\\\"
            )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    result = analyse()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_table(result)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {TABLE_OUTPUT.relative_to(ROOT)}")
    for instance in result["instances"]:
        for name, sector in instance["sectors"].items():
            if sector["feasible_rank"] == 0:
                print(f"{instance['instance_id']} {name}: no feasible sector")
                continue
            print(
                f"{instance['instance_id']} {name}: "
                f"eta={sector['eta_gamma']:+.6g}, "
                f"Gamma={sector['Gamma_gamma']:.6g}, "
                f"||B||={sector['B_gamma_norm']:.6g}, "
                f"leak={sector['observed_worst_ground_leakage']:.6g}, "
                f"bound={sector['exact_Gamma_bound']:.6g}"
            )


if __name__ == "__main__":
    main()
