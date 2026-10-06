#!/bin/bash
# =============================================================================
# Plain SplatSuRe — No Pseudo-Depth Regularization
# Uses metric_eval/SplatSuRe as the repo root
# =============================================================================

set -e
SAVE_NAME="Splatsure_depthv5"
REPO="/home/suresh/3dgs/SplatSuRe"
DATASET="/home/suresh/3dgs/HAT_REALx4"
OUTPUT="/home/suresh/3dgs/OUTPUTS/$SAVE_NAME"
PRED="$OUTPUT/submissions"

cd "$REPO"
mkdir -p "$OUTPUT"
cp "$REPO/run-all.sh"  "$OUTPUT/script.sh"

# SCENES="aeroplane toy still3 cycle bike buddha firehydrant face"
SCENES="aeroplane" 
# Run SplatSuRe for all scenes
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes       $SCENES \
  --lr-iterations  15000 \
  --sr-iterations  30000 \
  --ratio-threshold 1.05 \
  --upscale 4 \
  --sr-folder images_sr \
  --extra-train-args "--opacity_reset_interval 3000 \
                      --sh_degree 3 \
                      --densify_grad_threshold 0.0002 \
                      --antialiasing \
                      --white_background" \
  --extra-lr-args "--pseudo_view_weight 0.07 \
                   --pseudo_view_interval 10 \
                   --pseudo_view_start 1000 \
                   --pseudo_view_end 10000 \
                   --pseudo_view_model_size large \
                   --pseudo_view_model_path true \
                   --densify_until_iter 12000 \
                   --position_lr_max_steps 15000 " \
  --extra-sr-args "--gamma 0.45 \
                   --lambda_dssim 0.35 \
                   --densify_until_iter 27000 \
                   --position_lr_max_steps 30000 " 


echo "============================================================"
echo "  Plain SplatSuRe complete. Output: $OUTPUT"
echo "============================================================"

echo "============================================================"
echo " Evaluating Metrics for All Scenes..."
echo "============================================================"

# List all the scenes processed in the v7_final script.

echo "Saving to $SAVE_NAME"

python eval_unsparse_scenes.py \
    --dataset-root "$DATASET" \
    --pred-root    "$PRED" \
    --scenes       $SCENES \
    --gt-folder    "images_sr" \
    --save-csv     "$OUTPUT/metrics_summary.csv" > "$OUTPUT/result.txt"
echo " With train-test-exp" >> "$OUTPUT/result.txt"
echo "============================================================"
echo " Metrics evaluation complete. Summary saved at:"
echo " ${OUTPUT}/metrics_summary.csv"
echo "============================================================"