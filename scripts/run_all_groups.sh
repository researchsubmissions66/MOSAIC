#!/usr/bin/env bash
# Every remaining patch-encoder group analysis, run sequentially.
# The 10x/256px CPTAC group is already covered by results/full_run/analysis.
set -u
export MPLBACKEND=Agg
PY=${PY:-python}
OUT=${OUT:-results/groups}
STAGES="similarity alignment transfer retrieval"

run () {  # run <group> <label>
  echo "=============================================================="
  echo "GROUP $1  ->  $OUT/$2"
  echo "=============================================================="
  $PY scripts/run_study.py --out "$OUT/$2" --preset full --group "$1" --stages $STAGES \
    || echo "FAILED: $1"
}

run master_benchmark/10x_256px tcga_10x_256    # conch_v1 ctranspath gigapath resnet50 uni_v2
run cptac_benchmark/20x_224px  cptac_20x_224   # gpfm hoptimus0 virchow virchow2
run master_benchmark/20x_224px tcga_20x_224    # gpfm hoptimus0 virchow virchow2
run cptac_benchmark/20x_512px  cptac_20x_512   # conch_v1 conch_v15
run cptac_benchmark/20x_256px  cptac_20x_256   # ctranspath gigapath keep resnet50 uni_v2

echo "ALL GROUPS DONE"
