"""Shell-resolved cyclic closure in a joint orbit quotient."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spectral_gap_study import _decode_bases


def penalty_on_orbits(
    incidence: np.ndarray,
    k: int,
    orbits: list[list[int]],
    lam: float,
    mu: float,
) -> np.ndarray:
    """Return the exact diagonal penalty on every joint orbit."""
    n_b = int(incidence.shape[0])
    decoded = _decode_bases(n_b**k, k, n_b)
    penalties: list[float] = []
    for orbit in orbits:
        labels = [int(value) for value in decoded[orbit[0]]]
        uncovered = int((~incidence[labels].any(axis=0)).sum())
        repetitions = sum(
            int(labels[left] == labels[right])
            for left in range(k)
            for right in range(left + 1, k)
        )
        penalties.append(lam * uncovered + mu * repetitions)
    return np.asarray(penalties, dtype=float)


def shell_cyclic_basis(
    transport: np.ndarray,
    penalties: np.ndarray,
    initial: np.ndarray,
    relative_tol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Close ``initial`` under transport and all penalty-shell projectors.

    Columns belonging to distinct shells have disjoint support. Maintaining
    one orthonormal block per shell prevents exact penalty degeneracies from
    being mixed by a global numerical rank decision.
    """
    dimension = int(transport.shape[0])
    scale = float(np.linalg.norm(transport, ord=2))
    normalized_transport = transport / max(1.0, scale)
    values = np.unique(penalties)
    indices = [np.flatnonzero(penalties == value) for value in values]

    blocks: list[np.ndarray] = []
    frontier_columns: list[np.ndarray] = []
    for shell_indices in indices:
        local = initial[shell_indices].astype(float, copy=True)
        norm = float(np.linalg.norm(local))
        if norm == 0.0:
            blocks.append(np.empty((len(shell_indices), 0), dtype=float))
            continue
        local /= norm
        blocks.append(local[:, None])
        column = np.zeros(dimension, dtype=float)
        column[shell_indices] = local
        frontier_columns.append(column)

    frontier = np.column_stack(frontier_columns)
    growth = [int(frontier.shape[1])]
    min_kept = math.inf
    max_discarded = 0.0
    iterations = 0

    while frontier.shape[1] and growth[-1] < dimension:
        transported = normalized_transport @ frontier
        new_columns: list[np.ndarray] = []
        for shell_number, shell_indices in enumerate(indices):
            candidates = transported[shell_indices, :].copy()
            block = blocks[shell_number]
            for _ in range(2):
                if block.shape[1]:
                    candidates -= block @ (block.T @ candidates)
            if candidates.size == 0:
                continue
            left, singular, _ = np.linalg.svd(candidates, full_matrices=False)
            if singular.size == 0:
                continue
            threshold = relative_tol * max(1.0, float(singular[0]))
            added = int(np.sum(singular > threshold))
            if added:
                min_kept = min(min_kept, float(singular[added - 1]))
                local_new = left[:, :added]
                for _ in range(2):
                    if block.shape[1]:
                        local_new -= block @ (block.T @ local_new)
                local_new, _ = np.linalg.qr(local_new, mode="reduced")
                blocks[shell_number] = np.column_stack([block, local_new])
                for column_number in range(local_new.shape[1]):
                    column = np.zeros(dimension, dtype=float)
                    column[shell_indices] = local_new[:, column_number]
                    new_columns.append(column)
            if added < singular.size:
                max_discarded = max(max_discarded, float(singular[added]))

        iterations += 1
        if not new_columns:
            break
        frontier = np.column_stack(new_columns)
        growth.append(growth[-1] + int(frontier.shape[1]))

    global_columns: list[np.ndarray] = []
    for block, shell_indices in zip(blocks, indices):
        for column_number in range(block.shape[1]):
            column = np.zeros(dimension, dtype=float)
            column[shell_indices] = block[:, column_number]
            global_columns.append(column)
    basis = np.column_stack(global_columns)
    gram_error = float(
        np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]), ord=2)
    )
    invariance_residual = float(
        np.linalg.norm(
            transport @ basis - basis @ (basis.T @ transport @ basis), ord=2
        )
    )
    record = {
        "relative_tolerance": relative_tol,
        "dimension": int(basis.shape[1]),
        "growth": growth,
        "iterations": iterations,
        "transport_scale": scale,
        "gram_error": gram_error,
        "transport_invariance_residual": invariance_residual,
        "minimum_kept_singular_value": None if math.isinf(min_kept) else min_kept,
        "maximum_discarded_singular_value": max_discarded,
    }
    return basis, record
