#!/usr/bin/env bash
# The magnification ablation for every available series.
set -u
export MPLBACKEND=Agg
PY=${PY:-python}
OUT=${OUT:-results/magnification}

for S in cptac_benchmark/256px master_benchmark/256px cptac_benchmark/224px \
         master_benchmark/224px cptac_benchmark/512px; do
  LABEL=$(echo "$S" | tr '/' '_')
  echo "=============================================================="
  echo "SERIES $S -> $OUT/$LABEL"
  echo "=============================================================="
  $PY scripts/magnification_ablation.py --series "$S" --out "$OUT/$LABEL" \
      --n-patches 50000 --max-slides 500 --max-samples 8000 --latent-dim 64 \
      --alignment joint_pca gcca procrustes || echo "FAILED: $S"
done
echo "ALL MAGNIFICATION SERIES DONE"
