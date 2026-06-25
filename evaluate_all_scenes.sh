#!/bin/bash

# =============================================================================
# evaluate_all_scenes.sh
# =============================================================================
#
# Script to print metrics for all scenes processed in v7_final.sh.
# It computes metrics for aeroplane, toy, still3, cycle, bike, buddha,
# and firehydrant using eval_unsparse_scenes.py.
# =============================================================================

set -e

REPO="/home/suresh/3dgs/SplatSuRe"
DATASET="/home/suresh/3dgs/HAT_REALx4"
OUTPUT="/home/suresh/3dgs/OUTPUTS/Splatsure_v2"
PRED="$OUTPUT/submissions"

cd "$REPO"

echo "============================================================"
echo " Evaluating Metrics for All Scenes..."
echo "============================================================"

# List all the scenes processed in the v7_final script.
SCENES="aeroplane toy still3 cycle bike buddha firehydrant"

python eval_unsparse_scenes.py \
    --dataset-root "$DATASET" \
    --pred-root    "$PRED" \
    --scenes       $SCENES \
    --lpips \
    --save-csv     "$OUTPUT/metrics_summary.csv"

echo "============================================================"
echo " Metrics evaluation complete. Summary saved at:"
echo " ${OUTPUT}/metrics_summary.csv"
echo "============================================================"
