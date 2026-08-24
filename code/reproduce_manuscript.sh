#!/usr/bin/env bash
set -euo pipefail

repro_mode="${1:---full}"
if [[ "${repro_mode}" != "--full" && "${repro_mode}" != "--quick" ]]; then
  echo "usage: bash code/reproduce_manuscript.sh [--full|--quick]" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Make the compiled PDF byte-reproducible.  The default epoch is noon UTC on
# the manuscript date; callers may override it when archiving a later version.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1786449600}"
export FORCE_SOURCE_DATE="${FORCE_SOURCE_DATE:-1}"
export TZ="UTC"

python3 code/freeze_instances.py --check

if [[ "${repro_mode}" == "--full" ]]; then
  python3 code/reconstruct_accessibility_table.py
  python3 code/orbit_cyclic_audit.py --write
fi

python3 code/d4_sector_analysis.py
python3 code/catalyst_path_analysis.py
python3 code/hard_core_parent_path.py
python3 code/mean_field_zrp_gap_audit.py

pdflatex -interaction=nonstopmode -halt-on-error main_rewrite.tex
bibtex main_rewrite
pdflatex -interaction=nonstopmode -halt-on-error main_rewrite.tex
pdflatex -interaction=nonstopmode -halt-on-error main_rewrite.tex

echo "Reproduction completed in ${repro_mode} mode."
