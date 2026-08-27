# Revision execution record — 2026-08-27

This record fixes the baseline and acceptance gates for the post-arXiv
revision of `main_rewrite.tex`.

## Preserved baseline

- Branch: `quantum-reader-ready`
- Baseline commit: `88c1458f6540c6efede7a16ea9cbf3c1d93ce61d`
- SHA-256 of the complete pre-existing binary Git diff:
  `f4f09af37ee16a804a25791d44c60cd77cdcad49286521e70e67da9dfeeb5e9d`
- Pre-existing modified files: `main_rewrite.pdf`, `refs.bib`,
  `sections/app_reader_representation.tex`,
  `sections/sec7_analytic_family.tex`,
  `sections/sec_rewrite_framework.tex`, `sections/sec_rewrite_intro.tex`, and
  `sections/sec_rewrite_localization.tex`.

The pre-existing source changes define the exact-cardinality scope, explain
the local retained walk, cite the colour-preserving automorphism and
Johnson--Metropolis background, and repair notation. They are part of the
revision and must not be reverted.

## Acceptance gates

1. Distinguish the common invariant cyclic envelope from states actually
   reached by a specified time schedule.
2. Position the result against the closest effective-gap, symmetry-sector,
   and time-dependent Krylov literature.
3. Reproduce at least one nontrivial sector-localisation certificate,
   including the kinetic floor and observed leakage.
4. State the separation between the retained-walk and parent protocols in a
   compact comparison visible in the main text.
5. Rebuild every generated datum and the manuscript in the pinned
   environment with no unresolved references or overfull boxes.
6. Re-audit terminology, hypotheses, exact-cardinality scope, numerical
   provenance, and archive metadata before any public release.

Public GitHub/Zenodo release and an arXiv replacement are downstream actions,
not substitutes for these gates.

## Gate closure

All six gates passed on 2026-08-27.  The final full reproduction command
exited with status 0; all 32 frozen records and 11 primary rows were verified;
the sector-localisation inequalities passed in every audited sector; and the
20-page PDF contains no unresolved reference or citation, multiply-defined
label, overfull box, or duplicate PDF destination.  The final terminology
distinguishes the common invariant cyclic envelope from the smaller trajectory
set of any prescribed schedule, and the retained-walk and Johnson--Metropolis
protocols are compared explicitly in the main text.
