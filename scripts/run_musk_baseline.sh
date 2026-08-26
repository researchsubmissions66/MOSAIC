#!/usr/bin/env bash
# MUSK evaluated alone, on the 384px grid where it is the only encoder.
#
# Uses BOTH MIL heads, matching the rest of the study: every reported bar
# averages ABMIL and TransMIL. The results currently in results/musk_baseline/
# predate this and are ABMIL-only, so they are NOT strictly comparable to the
# main table's two-head averages -- re-run this script to fix that.
# MUSK single-encoder downstream baselines. MUSK is alone on the 384px grid, so
# it cannot enter any paired analysis, but the single-encoder condition needs no
# pairing and covers all 2296 CPTAC slides.
set -u
export MPLBACKEND=Agg
PY=${PY:-python}
OUT=${OUT:-results/musk_baseline}
for T in cptac_luad_tp53 cptac_luad_kras cptac_luad_stk11 \
         cptac_brca_pik3ca cptac_brca_gata3 cptac_brca_map3k1 \
         cptac_coad_tp53 cptac_coad_kras cptac_coad_pik3ca; do
  echo "=== $T ==="
  $PY scripts/downstream_mil.py --task "$T" --group cptac_benchmark/10x_384px \
      --encoders musk --conditions single --mil abmil transmil --epochs 80 \
      --max-patches 4000 --out "$OUT/$T" || echo "FAILED: $T"
done
echo "MUSK BASELINES DONE"
