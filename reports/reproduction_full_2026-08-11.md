# Full reproduction audit — 2026-08-11

Command executed from the repository root:

```bash
bash code/reproduce_manuscript.sh --full
```

The command exited with status 0. It verified all 32 frozen instance records,
reconstructed the 11 primary finite-instance rows, regenerated the four
auxiliary numerical audits and both LaTeX tables, and compiled
`main_rewrite.pdf` through the complete LaTeX/BibTeX sequence.

## Environment

```text
Python 3.11.2
numpy 1.24.2
scipy 1.10.1
networkx 2.8.8
matplotlib 3.6.3
```

This matches `requirements-repro.txt`. BLAS thread counts were fixed to one.

## Output SHA-256 digests

```text
5dc053145fd6a8429b9e96d4f37bbc1090f496e23f282a5a0e49ea76b46d3f75  data/accessibility_table_reconstructed.json
fc3d53501a978860e2dbded7a9b45d776bc5fb1ce8c4f7e38763d52c06265cae  data/d4_sector_comparison.json
7490b42730b42f69465ce0b0f918acca4364721b97648dade25a56eaba529d38  data/catalyst_path_results.json
26c0767d38842c3d4fdd9cb333e871c3cee40d60345f22b7a4b2989c27556e50  data/hard_core_parent_path_results.json
19ce63b4c3e7949e4b6c31037c7a514808f82417cbaade8dd4f09227718fd62f  data/mean_field_zrp_gap_audit.json
efd9cbeb85737bb1d27156dbea2a8faba3365c233066ffafbd41f880f5dbf507  sections/generated_accessibility_tables.tex
f11af52bdcec3940fce6dfa1489b4b98e5e21bb92b19b723f97ed55a55638d7a  main_rewrite.pdf
```

The final PDF has 11 pages and 511910 bytes. Its final log contains no
undefined citation/reference, multiply-defined label, overfull box, or
`pdfTeX` destination warning. The remaining underfull-box messages are
non-fatal two-column line-breaking diagnostics.
