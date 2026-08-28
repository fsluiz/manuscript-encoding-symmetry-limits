# Orbit-resolved spectra and dynamical accessibility in multi-register covering Hamiltonians

Reproducibility artifact for a theory-centred study of which part of a
multi-register covering-Hamiltonian spectrum is dynamically relevant when the
initial state and interpolation preserve joint symmetries.

The current manuscript contributes:

- orbit-invariant penalties, P\'olya--Redfield counts, the exact joint orbit
  quotient, and the distinction between $\mathcal H_{\rm sym}$, the common
  invariant cyclic envelope $\mathcal K_u$, and the smaller trajectory set
  of any specified schedule;
- a sector-resolved Schur-complement localisation theorem containing the
  sector kinetic floor;
- an orbit-resolved hopping-cancellation theorem and a separate stability
  theorem for transverse symmetry-dark crossings;
- an even-cycle parent path with endpoint cover probability
  $1-O(n^{-5})$ and the explicit uniform cyclic-envelope-gap lower bound
  $1024n^{-13}$; and
- a frozen 11-instance reconstruction using full Hamiltonians,
  residual-aware degeneracy tests, and exact cyclic restrictions.

No quantum-speedup, NISQ-feasibility, or total hardware-resource claim is
made.

## Repository layout

- `main_rewrite.tex` and `main_rewrite.pdf` — current manuscript source and
  compiled PDF;
- `CITATION.cff` and `.zenodo.json` — GitHub citation metadata and
  scope-aware Zenodo deposit metadata;
- `sections/` and `refs.bib` — manuscript sections and bibliography;
- `quantumarticle.cls` and `quantum.bst` — the exact official Quantum class
  and bibliography style used by the archived build;
- `code/` — the exact reconstruction and analytic-audit programs;
- `instances/` — immutable, hashed input registry;
- `orlib_instances/` — only the OR-Library parent files needed to reconstruct
  the frozen registry;
- `data/` — the seven machine-readable outputs used by the current manuscript;
- `examples/` — a standalone MathJax worked example that expands a five-set
  Hamiltonian and applies the symmetry, cyclic-envelope, localisation,
  and gap analysis;
- `reports/` — provenance and full-reproduction audit reports; and
- `REPRODUCIBILITY.md` — pinned environment and command-level instructions.

Superseded manuscript versions, exploratory experiments, local notes, and
unused numerical outputs are intentionally excluded from this archival
repository.

## Reproduce the manuscript

The reference environment is CPython 3.11.2 on Linux x86-64 with the three
pinned scientific packages in `requirements-repro.txt`. The archived programs
do not require Matplotlib, Qiskit, or their dependency trees.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-repro.txt
bash code/reproduce_manuscript.sh --full
```

The full command verifies all 32 frozen records, reconstructs the 11 primary
finite-instance rows, independently audits the shell-resolved cyclic closure
and the sector-localisation certificate, regenerates the four other auxiliary
numerical audits and
the LaTeX tables and finite-diagnostic figure, and compiles the manuscript.
A faster structural test is:

```bash
bash code/reproduce_manuscript.sh --quick
```

After a successful manuscript build, create and independently compile the
minimal arXiv source package with:

```bash
python3 code/build_arxiv_bundle.py --verify
```

The resulting deterministic archive is written to
`dist/arxiv_source_candidate.tar.gz`. It contains only the recursive TeX
dependency closure, generated `.bbl`, bibliography provenance, and the exact
Quantum class/style; repository code, data, notes, and the compiled PDF are
excluded.

See `REPRODUCIBILITY.md` for individual commands, output names, numerical
acceptance rules, and the reference hashes.

### Numerical correction recorded for the revised manuscript

An earlier draft reported
$\dim\mathcal K_u=326$ for `random-b8-t8-p42-s014-v1`.  Closing the cyclic
space independently inside every exact potential shell corrects this value
to $324$.  The result is unchanged at relative rank tolerances
$10^{-9}$, $10^{-11}$, and $10^{-13}$; the complete conditioning record is
stored in `data/orbit_cyclic_audit.json`.  This is a correction of the
previous closure calculation, not a change to the frozen instance.

## Numerical scope

The current manuscript uses these machine-readable outputs:

| Result | Program | Output |
|---|---|---|
| Finite structural and endpoint tables | `code/reconstruct_accessibility_table.py` | `data/accessibility_table_reconstructed.json` |
| Independent shell-resolved cyclic closure | `code/orbit_cyclic_audit.py` | `data/orbit_cyclic_audit.json` |
| Sector-localisation certificate | `code/sector_localization_audit.py` | `data/sector_localization_audit.json` |
| $D_4$ sector audit | `code/d4_sector_analysis.py` | `data/d4_sector_comparison.json` |
| Endpoint-vanishing catalyst | `code/catalyst_path_analysis.py` | `data/catalyst_path_results.json` |
| Even-cycle orbit quotients | `code/hard_core_parent_path.py` | `data/hard_core_parent_path_results.json` |
| Mean-field zero-range audit | `code/mean_field_zrp_gap_audit.py` | `data/mean_field_zrp_gap_audit.json` |

Finite numerical results are diagnostics, not scaling evidence. The
asymptotic claim rests on the analytic theorem in the manuscript.

## License

Original software in `code/` is released under the MIT License. Original
manuscript material, documentation, generated results, and project-authored
data are released under CC BY 4.0. See `LICENSE` and `LICENSES/` for the
precise scope.

The repository does not relicense third-party material. In particular, the
official `quantumarticle.cls` and `quantum.bst` package files remain under
LPPL 1.3c, while OR-Library source files and their incidence/cost values
retain J. E. Beasley's MIT notice, reproduced in
`LICENSES/ORLIB-MIT.txt`.
Project-authored provenance, selection metadata, and numerical analyses are
covered as described in `LICENSE`.

## Citation

Citation metadata are provided in `CITATION.cff`; `.zenodo.json` records the
scope-specific licensing required by the mixed software/data archive.  The
associated preprint is
[arXiv:2608.11503](https://arxiv.org/abs/2608.11503).  The reproducibility
artifact is preserved as
[GitHub release v1.0.0](https://github.com/fsluiz/manuscript-encoding-symmetry-limits/releases/tag/v1.0.0)
and [Zenodo record 10.5281/zenodo.22142799](https://doi.org/10.5281/zenodo.22142799).
