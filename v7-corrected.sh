#!/bin/bash
# =============================================================================
# v7_final.sh — Universal Dense Init + Unified Dataset Root (LR Optimized)
# =============================================================================

set -e

REPO="/home/suresh/3dgs/SplatSuRe"
DATASET="/home/suresh/3dgs/ADAx4"
OUTPUT="/home/suresh/3dgs/OUTPUTS/with_depth_v1"
PRED="$OUTPUT/submissions"

cd "$REPO"
# mkdir $OUTPUT
# touch "$OUTPUT/script.sh"
# cat "/home/suresh/3dgs/SplatSuRe/v7-corrected.sh" > "$OUTPUT/script.sh"
#── PASS 1A: Thin + reliable depth (aeroplane, toy) ──────────────────────────
# Active depth signal (ρ=0.855/0.911) + Universal MVS Init
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes aeroplane  \
  --lr-iterations  26000 \
  --sr-iterations  32900 \
  --ratio-threshold 1.1 \
  --upscale 4 \
  --extra-train-args "--opacity_reset_interval 3000 --densify_grad_threshold 0.0003" \
  --extra-lr-args "--pseudo_view_weight 0.05 \
                 --pseudo_view_interval 10 \
                 --pseudo_view_start 3000 \
                 --pseudo_view_end 20000 \
                 --pseudo_view_model_size small \
                 --densify_until_iter 20000 \
                 --position_lr_max_steps 26000" \
  --extra-sr-args "--gamma 0.4  \
                 --densify_until_iter 32000"

# # ── PASS 2B: Smooth + no depth (face, still3) ────────────────────────────────
# # Depth Loss ZERO + Universal MVS Init + Conservative ratio
# python train_splatsure_competition.py \
#   --repo-root    "$REPO" \
#   --dataset-root "$DATASET" \
#   --output-root  "$OUTPUT" \
#   --pred-root    "$PRED" \
#   --scenes still3 face \
#   --lr-iterations  14900 \
#   --sr-iterations  40400 \
#   --ratio-threshold 1.3 \
#   --upscale 4 \
#   --extra-train-args "--opacity_reset_interval 1500 --densify_grad_threshold 0.0003 --depth_l1_weight_init 0 --depth_l1_weight_final 0" \
#   --extra-lr-args "--pseudo_view_weight 0.0 \
#                  --densify_until_iter 13000 \
#                  --position_lr_max_steps 14000" \
#   --extra-sr-args "--gamma 0.4 \
#                  --densify_until_iter 35000"

# # # ── PASS 1B: Thin + no depth (cycle, bike) ───────────────────────────────────
# # # Depth Loss ZERO + Universal MVS Init
# python train_splatsure_competition.py \
#   --repo-root    "$REPO" \
#   --dataset-root "$DATASET" \
#   --output-root  "$OUTPUT" \
#   --pred-root    "$PRED" \
#   --scenes cycle bike \
#   --lr-iterations  25000 \
#   --sr-iterations  40000 \
#   --ratio-threshold 1.1 \
#   --upscale 4 \
#   --extra-train-args "--opacity_reset_interval 1500 --densify_grad_threshold 0.0003 --depth_l1_weight_init 0 --depth_l1_weight_final 0" \
#   --extra-lr-args "--pseudo_view_weight 0.0 \
#                  --densify_until_iter 20000 \
#                  --position_lr_max_steps 25000" \
#   --extra-sr-args "--gamma 0.4 \
#                  --densify_until_iter 35000"

# # # ── PASS 2A: Smooth + reliable depth (buddha, firehydrant) ───────────────────
# # # Active depth signal (ρ=0.890/0.985) + Universal MVS Init
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes buddha firehydrant \
  --lr-iterations  25000 \
  --sr-iterations  40000 \
  --ratio-threshold 1.5 \
  --upscale 4 \
  --extra-train-args "--opacity_reset_interval 1500 --densify_grad_threshold 0.0003" \
  --extra-lr-args "--pseudo_view_weight 0.05 \
                 --pseudo_view_interval 10 \
                 --pseudo_view_start 3000 \
                 --pseudo_view_end 20000 \
                 --pseudo_view_model_size small \
                 --densify_until_iter 20000 \
                 --position_lr_max_steps 25000" \
  --extra-sr-args "--gamma 0.4 \
                 --densify_until_iter 35000"



# echo "============================================================"
# echo "  v7_final complete. Initialization: Universal fused.ply"
# echo "============================================================"