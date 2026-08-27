# Full reproduction audit — 2026-08-27

Command executed from the repository root:

```bash
bash code/reproduce_manuscript.sh --full
```

The command exited with status 0. It verified all 32 frozen instance records,
reconstructed the 11 primary finite-instance rows, confirmed every cyclic
dimension independently at three rank tolerances, regenerated the finite
joint-symmetry, sector-localisation, catalyst, parent-path, and zero-range
audits, and compiled `main_rewrite.pdf` through the complete
LaTeX/BibTeX sequence.

## Environment

```text
Python 3.11.2
numpy 1.24.2
scipy 1.10.1
networkx 2.8.8
```

This matches `requirements-repro.txt`. BLAS thread counts were fixed to one,
and the deterministic PDF environment used
`SOURCE_DATE_EPOCH=1786449600`, `FORCE_SOURCE_DATE=1`, and `TZ=UTC`.

## Output SHA-256 digests

```text
518a57c7e6232f2d5f631e5eb0e65d20d797d5ce5d02b474b4cb5cc9bb0b1c54  data/accessibility_table_reconstructed.json
b3fc388369d6086fd086aaafccd52dc9a35e3fe76bbd2a88495ff99455507032  data/orbit_cyclic_audit.json
fc3d53501a978860e2dbded7a9b45d776bc5fb1ce8c4f7e38763d52c06265cae  data/d4_sector_comparison.json
4b5c003495ea77ac7eca79bb9e83bcecbafd5816abd462d6fdd9accea5fa9b4e  data/sector_localization_audit.json
4c11493c3b4c55f2b1bf9f0ab7bb3456a1f7ea8bffe7199414fa338f19e84507  data/catalyst_path_results.json
26c0767d38842c3d4fdd9cb333e871c3cee40d60345f22b7a4b2989c27556e50  data/hard_core_parent_path_results.json
19ce63b4c3e7949e4b6c31037c7a514808f82417cbaade8dd4f09227718fd62f  data/mean_field_zrp_gap_audit.json
6f5c824fd050a37befa7dfa0df3d2bed1974ab0204e91ddb051e0ba5946ae827  sections/generated_accessibility_tables.tex
b0c7fe7b31ca3efc0948c03bf434ffecf406c28838717352f5aac5b99cbd0e87  sections/generated_accessibility_figure.tex
ae500c4bdb3aa1f7d2f9153cc5748a7b3c7f6cf37fd5f1928ddf8095a6d32e13  sections/generated_sector_localization_table.tex
b49287060f99fa87520fe1d53363f3dff4968b091fb07516059566c6091bc048  main_rewrite.pdf
```

The final PDF has 20 pages and 699988 bytes. Its final log contains no
undefined citation/reference, multiply-defined label, overfull box, or
duplicate PDF destination warning. Remaining underfull-box messages are
non-fatal two-column line-breaking diagnostics.

The independently compiled 21-file arXiv source bundle also passed and has
SHA-256
`52c00cb4fb580ef3a82789a467f695a106688e34d480e0c9c4e1df46c4e21a73`.
Its clean-build PDF hash is identical to the manuscript PDF hash above.

## New sector-localisation gate

The sector audit evaluates the full Hamiltonian rather than an effective
model. In the `grid-2x4` $S_3$-trivial $\times D_4$-$E$ sector it obtains
$\eta_\gamma=-2/7$, $\Gamma_\gamma=4.88189$,
$\lVert B_\gamma\rVert=0.889694$, worst observed leakage $0.00357428$,
exact bound $0.0321452$, and kinetic-floor bound $0.0324629$. Every audited
sector satisfies both the exact certificate and
$\Gamma_\gamma\ge m-\kappa-\eta_\gamma$ within the recorded roundoff
tolerance.
