# Joint symmetry and dynamical accessibility in compact Hamiltonian encodings of set cover

Reproducibility artifact for a theory-centred study of which part of a compact
Minimum Set Cover Hamiltonian spectrum is dynamically relevant when the
initial state and interpolation preserve joint symmetries.

The current manuscript contributes:

- the effective group $S_k\times G_B$ and the distinction between the
  symmetry-allowed space $\mathcal H_{\rm sym}$ and the protocol-generated
  cyclic space $\mathcal K_u$;
- a sector-resolved Schur-complement localisation theorem containing the
  sector kinetic floor;
- exact orbit quotients, symmetry-dark crossings, and a stability theorem;
- an even-cycle parent path with endpoint cover probability
  $1-O(n^{-5})$ and a uniform accessible-gap certificate
  $\Omega(n^{-13})$; and
- a frozen 11-instance reconstruction using full Hamiltonians,
  residual-aware degeneracy tests, and exact cyclic restrictions.

No quantum-speedup, NISQ-feasibility, or total hardware-resource claim is
made.

## Repository layout

- `main_rewrite.tex` and `main_rewrite.pdf` — current manuscript source and
  compiled PDF;
- `sections/` and `refs.bib` — manuscript sections and bibliography;
- `code/` — the exact reconstruction and analytic-audit programs;
- `instances/` — immutable, hashed input registry;
- `orlib_instances/` — only the OR-Library parent files needed to reconstruct
  the frozen registry;
- `data/` — the five machine-readable outputs used by the current manuscript;
- `reports/` — provenance and full-reproduction audit reports; and
- `REPRODUCIBILITY.md` — pinned environment and command-level instructions.

Superseded manuscript versions, exploratory experiments, local notes, and
unused numerical outputs are intentionally excluded from this archival
repository.

## Reproduce the manuscript

The reference environment is Python 3.11.2 with the exact packages in
`requirements-repro.txt`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-repro.txt
bash code/reproduce_manuscript.sh --full
```

The full command verifies all 32 frozen records, reconstructs the 11 primary
finite-instance rows, regenerates the four auxiliary numerical audits and
LaTeX tables, and compiles the manuscript. A faster structural test is:

```bash
bash code/reproduce_manuscript.sh --quick
```

See `REPRODUCIBILITY.md` for individual commands, output names, numerical
acceptance rules, and the reference hashes.

## Numerical scope

The current manuscript uses these machine-readable outputs:

| Result | Program | Output |
|---|---|---|
| Finite structural and endpoint tables | `code/reconstruct_accessibility_table.py` | `data/accessibility_table_reconstructed.json` |
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

The repository does not relicense third-party material. In particular,
`quantumarticle.cls` remains under LPPL 1.3c or later, while OR-Library source
files and their incidence/cost values retain the terms of their source.
Project-authored provenance, selection metadata, and numerical analyses are
covered as described in `LICENSE`.

## Citation

Citation metadata are provided in `CITATION.cff`. The public release will add
the arXiv identifier, GitHub release URL, and version-specific Zenodo DOI.
