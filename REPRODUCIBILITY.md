# Reproducing the current manuscript

This document applies to `main_rewrite.tex` and excludes the superseded
algorithmic experiments retained for provenance.  The current paper uses only
NumPy, SciPy, NetworkX, and Matplotlib; Qiskit is not required.

## Environment

The reference environment is Python 3.11.2 with the exact package versions in
`requirements-repro.txt`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-repro.txt
```

All commands below must be run from the repository root with single-threaded
BLAS so that sparse eigensolves do not oversubscribe the machine:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Frozen inputs

The machine-readable registry is `instances/manifest.json`.  Every manuscript
row cites an `instance_id`, a frozen-file SHA-256 digest, and a payload SHA-256
digest.  Verify the registry before any eigensolve:

```bash
python3 code/freeze_instances.py --check
```

The check is read-only and fails if a frozen instance differs from its recorded
hash.

## Current numerical claims

| Manuscript result | Command | Machine-readable output |
|---|---|---|
| Tables 1 and 2 | `python3 code/reconstruct_accessibility_table.py` | `data/accessibility_table_reconstructed.json`, `sections/generated_accessibility_tables.tex` |
| Shell-resolved cyclic-space audit | `python3 code/orbit_cyclic_audit.py --write` | `data/orbit_cyclic_audit.json` |
| Finite joint-symmetry audit | `python3 code/d4_sector_analysis.py` | `data/d4_sector_comparison.json` |
| Endpoint-vanishing catalyst | `python3 code/catalyst_path_analysis.py` | `data/catalyst_path_results.json` |
| Exact even-cycle quotients | `python3 code/hard_core_parent_path.py` | `data/hard_core_parent_path_results.json` |
| Mean-field/three-box numerical audit | `python3 code/mean_field_zrp_gap_audit.py` | `data/mean_field_zrp_gap_audit.json` |

The table reconstruction is the expensive step.  It computes the full
Hamiltonian spectra reported in the manuscript rather than reading legacy
spectral JSON.  Numerical degeneracy decisions are scale- and residual-aware;
the residuals and thresholds used for every reported row are included in the
output JSON.

Run the complete workflow with:

```bash
bash code/reproduce_manuscript.sh
```

For a fast structural check that omits the full 11-instance table
reconstruction:

```bash
bash code/reproduce_manuscript.sh --quick
```

## Manuscript build

```bash
pdflatex -interaction=nonstopmode -halt-on-error main_rewrite.tex
bibtex main_rewrite
pdflatex -interaction=nonstopmode -halt-on-error main_rewrite.tex
pdflatex -interaction=nonstopmode -halt-on-error main_rewrite.tex
```

The archived release must contain the generated PDF, all frozen inputs, the
six current output JSON files listed above, the generated table source, and
the exact source revision. It must also retain the top-level dual-license
notice and both files in `LICENSES/`. The GitHub release tag and
version-specific Zenodo DOI must be recorded in the final Code and data
availability statement.
