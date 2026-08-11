# Instance audit for the symmetry/accessibility rewrite

This report freezes the current mathematical inputs. It does **not** certify that superseded legacy JSON files were generated from these exact matrices. Those legacy files are intentionally omitted from the archival artifact and are not accepted as evidence for the current manuscript.

## Current OR-Library extraction

| Frozen ID | requested tasks | retained tasks | bases | exact `k*` | payload SHA-256 |
|---|---:|---:|---:|---:|---|
| `scpe1-b8-t20-s0-v1` | 20 | 20 | 8 | 4 | `fb8bb96d53181586…` |
| `scpe1-b8-t30-s0-v1` | 30 | 30 | 8 | 5 | `8713960484f56292…` |
| `scpe2-b8-t20-s0-v1` | 20 | 20 | 8 | 4 | `dc6ae8ffa2a9956d…` |
| `scpe2-b8-t30-s0-v1` | 30 | 30 | 8 | 5 | `11ab5d33546d30f4…` |
| `scpe3-b8-t20-s0-v1` | 20 | 20 | 8 | 3 | `83cacaeed67f9818…` |
| `scpe3-b8-t30-s0-v1` | 30 | 30 | 8 | 5 | `efa65316413a0912…` |
| `scpe4-b8-t29-s0-v1` | 30 | 29 | 8 | 5 | `7b439e341169e756…` |
| `scpe5-b8-t30-s0-v1` | 30 | 30 | 8 | 5 | `d4c71a4c6bacde6f…` |

The registry records the selected zero-based parent task/base indices and the SHA-256 of each OR-Library parent file. Requested and retained task counts are separate fields because the current extractor drops sampled tasks left uncovered by its selected bases.

## Reproducibility discrepancies found

1. `scpe4`: the current extraction retains 29 tasks. `data/orlib_experiment_results.json` labels it as `m30`, whereas later files label it as `m29`.
2. `scpe5`: the current extraction retains all 30 tasks, but `data/orlib_experiment_results.json` records `n_T=29` under an `m30` label, and the manuscript/later files use an `m29` label. The old 29-task matrix and selected indices were not stored, so that historical input cannot be reconstructed from the repository.
3. `scpe3` with 30 requested/retained tasks: exact enumeration on the current matrix gives `k*=5`, while every legacy table/result records `k*=4`. Consequently, the stored spectrum cannot be attributed to the currently reconstructible matrix.
4. The extractor docstring describes a contiguous/first-row procedure, while the implementation samples rows uniformly with a seeded NumPy generator. It also does not test connectedness, despite the old README calling the outputs connected.
5. Weighted `scp41`: the current extraction is `scp41-weighted-b8-t10-s0-v1`, with costs `[4, 7, 12, 33, 41, 72, 73, 77]` and weighted optimum 157. The legacy file reports costs `[3, 4, 12, 24, 73, 77, 96, 100]` and optimum 196, so it came from a different, unrecoverable slice.
6. Legacy result files generally store labels and dimensions but not the input matrix, selected parent indices, source hash, code revision, or residual-based numerical acceptance data. Equality of a label is therefore not proof of equality of the input.

## Editorial disposition

- `candidate_primary`: candidate inputs for the new full-H symmetry/accessibility calculations. This status is not a numerical endorsement; results must be recomputed.
- `legacy_only`: retained to explain the old manuscript and prevent provenance loss, but excluded from the first reconstruction unless promoted explicitly.
- The weighted, Fiedler, ADAPT, QAOA, and fixed-walk experiments are not part of the new primary claim and will not be carried into the rewritten main text without a separate justification.
- No legacy JSON is accepted as a source of a new table. New outputs must embed both `instance_id` and `payload_sha256`.

## Superseded artifacts

The superseded result files used to diagnose the discrepancies above are not part of this archival repository. Their omission is deliberate: registry validation must depend only on the immutable source instances, deterministic generators, and files listed in `instances/manifest.json`.

## Validation commands

```bash
python3 code/freeze_instances.py --check
python3 code/freeze_instances.py --write   # only for a clean initial creation
```

The checker reconstructs every instance, recomputes exact minimum covers, validates binary dimensions and coverage, and compares canonical bytes and hashes. It fails rather than overwriting a divergent frozen file.
