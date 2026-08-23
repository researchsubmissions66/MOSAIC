#!/usr/bin/env bash
# Layer-wise analysis for both pooling modes and both encoder grids.
# GigaPath is excluded: trident's loader pins timm==0.9.16.
set -u
export MPLBACKEND=Agg
PY=${PY:-python}
OUT=${OUT:-results/layerwise}

for POOL in cls mean; do
  $PY scripts/layerwise_alignment.py --encoders uni_v2 conch_v1 ctranspath resnet50 \
      --group master_benchmark/10x_256px --n-patches 512 --batch-size 16 \
      --pool "$POOL" --out "$OUT/tcga_10x_256_$POOL" || echo "FAILED: 256px $POOL"
done
echo "ALL LAYERWISE DONE"
