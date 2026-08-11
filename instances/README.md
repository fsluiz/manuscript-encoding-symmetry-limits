# Frozen instance registry

This directory is the source of truth for numerical inputs used by the
symmetry/accessibility rewrite. Each file in `frozen/` contains:

- the incidence matrix in the `bases_by_tasks` convention;
- the exact minimum-cardinality covers;
- full deterministic provenance (source hashes and selected indices, or
  generator parameters);
- a canonical SHA-256 of the mathematical payload.

`manifest.json` additionally hashes every complete instance file. Future
result files must carry the registry `instance_id` and `payload_sha256`; a
legacy label such as `scpe5-n8m29s0` is not sufficient provenance.

## Licensing and source rights

Project-generated grid and random instances, together with the registry,
project-authored provenance, annotations, and computed optima, are made
available under CC BY 4.0. For frozen records whose `provenance.kind` begins
with `orlib_`, the underlying incidence and cost values originate from the
OR-Library and are not relicensed by this repository. The selection indices,
hashes, annotations, and derived numerical results produced by this project
are covered by CC BY 4.0 to the extent rights subsist in them. See the
top-level `LICENSE` for the controlling scope statement.

Create the registry once and validate it subsequently with:

```bash
python3 code/freeze_instances.py --write
python3 code/freeze_instances.py --check
```

The command never overwrites divergent frozen files. See
`reports/instance_audit.md` for the audit of the superseded manuscript data.
