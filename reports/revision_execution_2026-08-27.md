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

## Public archival closure

Commit `6805c67e10355a98fadb6a96d34a1b963493182b` was published as GitHub
release `v1.0.0`.  After enabling the release webhook, Zenodo archived that
release as version DOI `10.5281/zenodo.22142799` (concept DOI
`10.5281/zenodo.22142798`).  The release assets retain the verified PDF and
arXiv-source hashes recorded in the full reproduction report.

After inserting the DOI into the manuscript, the post-archive journal
candidate was rebuilt in quick verification mode and its independent arXiv
bundle was compiled successfully.  Their SHA-256 digests are, respectively,
`a8dc37fca490ed34ac4b6580d78df1fec10debdf2f766e8db077b29eea2d74e5`
and
`b920051d6dd6e57607d6ca0c4d1a2672c5285904071c6b5b9294886e49b00d6b`.
