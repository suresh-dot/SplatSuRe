#!/bin/bash
# =============================================================================
# v9_classical_sr.sh
# =============================================================================

set -e

REPO="/home/suresh/3dgs/SplatSuRe"
DATASET="/home/suresh/3dgs/ADAx4"
OUTPUT='/home/suresh/3dgs/OUTPUTS/with_depth_v1'
PRED="$OUTPUT/submissions"

cd "$REPO"

#  PASS 1A: Thin + reliable depth  
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes aeroplane toy \
  --lr-iterations  15000 \
  --sr-iterations  40000 \
  --ratio-threshold 1.05 \
  --upscale 4 \
  --sr-folder images_sr \
  --extra-train-args "--opacity_reset_interval 1500 \
                      --sh_degree 3 \
                      --densify_grad_threshold 0.00015 \
                      --antialiasing \
                      --random_background" \
  --extra-lr-args "--pseudo_view_weight 0.03 \
                   --pseudo_view_interval 10 \
                   --pseudo_view_start 1000 \
                   --pseudo_view_end 8000 \
                   --pseudo_view_model_size small \
                   --depth_l1_weight_init 0.5 \
                   --depth_l1_weight_final 0.05 \
                   --densify_until_iter 12000 \
                   --position_lr_max_steps 15000" \
  --extra-sr-args "--gamma 0.45 \
                   --lambda_dssim 0.35 \
                   --densify_until_iter 21000 \
                   --position_lr_max_steps 40000 \
                   --densify_grad_threshold 0.0002 \
                   --densification_interval 50 \
                   --percent_dense 0.005 \
                   --depth_l1_weight_init 0.5 \
                   --depth_l1_weight_final 0.05 \
                   --opacity_lr 0.05 \
                   --scaling_lr 0.002" 

#  PASS 2B-face: Complex background scene ─
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes face \
  --lr-iterations  15000 \
  --sr-iterations  50000 \
  --ratio-threshold 1.1 \
  --upscale 4 \
  --sr-folder images_sr \
  --extra-train-args "--opacity_reset_interval 3000 \
                      --sh_degree 3 \
                      --densify_grad_threshold 0.0003 \
                      --depth_l1_weight_init 0 \
                      --depth_l1_weight_final 0 \
                      --antialiasing \
                      --random_background" \
  --extra-lr-args "--pseudo_view_weight 0.0 \
                   --densify_until_iter 12000 \
                   --position_lr_max_steps 15000" \
  --extra-sr-args "--gamma 0.4 \
                   --lambda_dssim 0.4 \
                   --densify_until_iter 30000 \
                   --position_lr_max_steps 50000 \
                   --densify_grad_threshold 0.0004 \
                   --opacity_lr 0.05 \
                   --scaling_lr 0.003"

#  PASS 2B-still3: Flat wall + foreground objects ─
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes still3 \
  --lr-iterations  15000 \
  --sr-iterations  50000 \
  --ratio-threshold 1.2 \
  --upscale 4 \
  --sr-folder images_sr \
  --extra-train-args "--opacity_reset_interval 3000 \
                      --sh_degree 3 \
                      --densify_grad_threshold 0.0003 \
                      --depth_l1_weight_init 0 \
                      --depth_l1_weight_final 0 \
                      --antialiasing \
                      --random_background" \
  --extra-lr-args "--pseudo_view_weight 0.0 \
                   --densify_until_iter 12000 \
                   --position_lr_max_steps 15000" \
  --extra-sr-args "--gamma 0.4 \
                   --lambda_dssim 0.35 \
                   --densify_until_iter 30000 \
                   --position_lr_max_steps 50000 \
                   --densify_grad_threshold 0.001 \
                   --opacity_lr 0.05 \
                   --scaling_lr 0.003"

#  PASS 1B: Thin + depth + pseudo-view  
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes cycle \
  --lr-iterations  15000 \
  --sr-iterations  40000 \
  --ratio-threshold 1.5 \
  --upscale 4 \
  --sr-folder images_sr \
  --extra-train-args "--opacity_reset_interval 2000 \
                      --sh_degree 3 \
                      --densify_grad_threshold 0.0002 \
                      --antialiasing \
                      --random_background" \
  --extra-lr-args "--pseudo_view_weight 0.02 \
                   --pseudo_view_interval 15 \
                   --pseudo_view_start 1000 \
                   --pseudo_view_end 8000 \
                   --pseudo_view_model_size small \
                   --depth_l1_weight_init 0.5 \
                   --depth_l1_weight_final 0.05 \
                   --densify_until_iter 12000 \
                   --position_lr_max_steps 15000" \
  --extra-sr-args "--gamma 0.5 \
                   --lambda_dssim 0.35 \
                   --densify_until_iter 26000 \
                   --position_lr_max_steps 40000 \
                   --densify_grad_threshold 0.0002 \
                   --depth_l1_weight_init 0.2 \
                   --depth_l1_weight_final 0.01 \
                   --opacity_lr 0.05 \
                   --scaling_lr 0.003"

#  PASS BIKE: Thin structure + clean grey bg + exceptional depth ─
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes bike \
  --lr-iterations  15000 \
  --sr-iterations  40000 \
  --ratio-threshold 1.3 \
  --upscale 4 \
  --sr-folder images_sr \
  --extra-train-args "--opacity_reset_interval 2500 \
                      --sh_degree 3 \
                      --densify_grad_threshold 0.0002 \
                      --antialiasing \
                      --random_background" \
  --extra-lr-args "--pseudo_view_weight 0.01 \
                   --pseudo_view_interval 15 \
                   --pseudo_view_start 1000 \
                   --pseudo_view_end 8000 \
                   --pseudo_view_model_size small \
                   --depth_l1_weight_init 0.8 \
                   --depth_l1_weight_final 0.05 \
                   --densify_until_iter 10000 \
                   --position_lr_max_steps 15000" \
  --extra-sr-args "--gamma 0.55 \
                   --lambda_dssim 0.35 \
                   --densify_until_iter 25000 \
                   --position_lr_max_steps 40000 \
                   --densify_grad_threshold 0.00015 \
                   --depth_l1_weight_init 0.2 \
                   --depth_l1_weight_final 0.01 \
                   --opacity_lr 0.05 \
                   --scaling_lr 0.004"

#  PASS BUDDHA: Bright close-up statues, great depth, large camera swings 
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes buddha \
  --lr-iterations  15000 \
  --sr-iterations  40000 \
  --ratio-threshold 1.4 \
  --upscale 4 \
  --sr-folder images_sr \
  --extra-train-args "--opacity_reset_interval 1500 \
                      --sh_degree 4 \
                      --densify_grad_threshold 0.00015 \
                      --antialiasing \
                      --random_background" \
  --extra-lr-args "--pseudo_view_weight 0.05 \
                   --pseudo_view_interval 10 \
                   --pseudo_view_start 1000 \
                   --pseudo_view_end 10000 \
                   --pseudo_view_model_size small \
                   --depth_l1_weight_init 1.0 \
                   --depth_l1_weight_final 0.05 \
                   --densify_until_iter 9000 \
                   --position_lr_max_steps 15000" \
  --extra-sr-args "--gamma 0.45 \
                   --lambda_dssim 0.35 \
                   --densify_until_iter 27000 \
                   --position_lr_max_steps 50000 \
                   --densify_grad_threshold 0.00015 \
                   --depth_l1_weight_init 0.3 \
                   --depth_l1_weight_final 0.01 \
                   --opacity_lr 0.05 \
                   --scaling_lr 0.003"

#  PASS FIREHYDRANT: Outdoor scene, rich background, near-perfect depth 
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes firehydrant \
  --lr-iterations  15000 \
  --sr-iterations  40000 \
  --ratio-threshold 1.1 \
  --upscale 4 \
  --sr-folder images_sr \
  --extra-train-args "--opacity_reset_interval 1500 \
                      --sh_degree 3 \
                      --densify_grad_threshold 0.0003 \
                      --antialiasing \
                      --random_background" \
  --extra-lr-args "--pseudo_view_weight 0.03 \
                   --pseudo_view_interval 10 \
                   --pseudo_view_start 1000 \
                   --pseudo_view_end 10000 \
                   --pseudo_view_model_size small \
                   --depth_l1_weight_init 1.0 \
                   --depth_l1_weight_final 0.05 \
                   --densify_until_iter 12000 \
                   --position_lr_max_steps 15000" \
  --extra-sr-args "--gamma 0.4 \
                   --lambda_dssim 0.35 \
                   --densify_until_iter 27000 \
                   --position_lr_max_steps 40000 \
                   --densify_grad_threshold 0.0003 \
                   --depth_l1_weight_init 0.3 \
                   --depth_l1_weight_final 0.01 \
                   --opacity_lr 0.05 \
                   --scaling_lr 0.003"

echo "============================================================"
echo "  v9 Classical SR - Strict Constraints complete."
echo "============================================================"