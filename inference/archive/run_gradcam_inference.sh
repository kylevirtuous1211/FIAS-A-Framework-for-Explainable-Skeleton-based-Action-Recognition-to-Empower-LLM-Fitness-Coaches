#!/bin/bash
# This script automates running the Grad-CAM inference pipeline for four different
# ST-GCN++ models. It finds the best checkpoint for each model and passes all
# necessary paths to the Python script.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# ❗️ Please verify these paths match your project structure.
MMACTION2_BASE="/home/cvlab123/mmaction2"
MMPOSE_BASE="/home/cvlab123/mmpose"
INFERENCE_BASE="/home/cvlab123/inference"

# --- Video to be Analyzed ---
# ❗️ CHANGE THIS to the video you want to process.
VIDEO_PATH="${INFERENCE_BASE}/test_video/pushup_IMG_1709 - 何羿秀.mov"

# --- Pose Model (RTMPose-S) ---
# POSE_CONFIG="${MMPOSE_BASE}/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py"
# POSE_CHECKPOINT="${MMPOSE_BASE}/checkpoints/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth"

# --- Pose Model (RTMPose-L) ---
POSE_CONFIG="/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py"
POSE_CHECKPOINT="/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth"


# --- Helper Function to Run a Single Inference ---
run_inference() {
    # Arguments passed to the function
    local action_config_name=$1
    local work_dir_suffix=$2
    local model_description=$3
    local modality_key=$4

    echo "--- Preparing to run inference for ${model_description} model ---"

    # Construct full paths for the action recognition model
    local action_config="${MMACTION2_BASE}/configs/skeleton/custom_stgcnpp/${action_config_name}"
    local work_dir="${MMACTION2_BASE}/work_dirs/${work_dir_suffix}"

    # Check if the work directory exists
    if [ ! -d "$work_dir" ]; then
        echo "❌ ERROR: Work directory not found at ${work_dir}"
        return 1
    fi

    # Find the MOST RECENT checkpoint file starting with 'best_acc_top1'
    local action_checkpoint
    action_checkpoint=$(find "${work_dir}" -name "best_acc_top1*.pth" -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)

    # Check if a checkpoint file was found
    if [ -z "$action_checkpoint" ]; then
        echo "❌ ERROR: No 'best_acc_top1*.pth' checkpoint found in ${work_dir}"
        return 1
    fi

    echo "✅ Found newest checkpoint: ${action_checkpoint}"
    echo "🚀 Running inference for ${model_description}..."

    # Execute the Python script with all required arguments
    python3 "${INFERENCE_BASE}/inference_new_gradcam.py" \
        --pose-config "${POSE_CONFIG}" \
        --pose-checkpoint "${POSE_CHECKPOINT}" \
        --action-config "${action_config}" \
        --action-checkpoint "${action_checkpoint}" \
        --video-path "${VIDEO_PATH}" \
        --modality-key "${modality_key}" \
        --work_dir "${work_dir_suffix}"

    echo "--- Inference for ${model_description} finished ---"
    echo "" # Add a blank line for better readability
}

# --- Main Execution ---
# The script will now call the run_inference function for each of your four models.

## 記得看現在有沒有 idle class，要去 inference script 調整 label

run_inference "stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py" "1101_CL_reweighting_add_idle_2D_joint" "Joint" "joint"
run_inference "stgcnpp_8xb16-joint-motion-u100-80e_OurDataset-xsub-keypoint-2d.py" "1101_CL_reweighting_add_idle_2D_joint_motion" "Joint Motion" "joint_motion"
run_inference "stgcnpp_8xb16-bone-u100-80e_OurDataset-xsub-keypoint-2d.py" "1101_CL_reweighting_add_idle_2D_bone" "Bone" "bone"
run_inference "stgcnpp_8xb16-bone-motion-u100-80e_OurDataset-xsub-keypoint-2d.py" "1101_CL_reweighting_add_idle_2D_bone_motion" "Bone Motion" "bone_motion"

echo "✅ All inference jobs completed successfully!"