# Inference & Demo Scripts
## TODO
All of these demo or inference tests requires to update the action recognition checkpoint path, mmaction2 config file to get the correct inference result. For example, using a different modality or different architecture.

From my experience, pose estimation config and its checkpoints won't need to update, unless you want to finetune or are doing experiments on pose estimation.
```
# --- Config ---
path_prefix = Path('/home/cvlab123') 
POSE_2D_CONFIG="/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py"
POSE_2D_CHECKPOINT="/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth"
ACTION_REC_CONFIG = path_prefix / 'mmaction2/configs/skeleton/custom_stgcnpp/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py'
ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/1027_all_plus_idle100_2D_joint/best_acc_top1_epoch_8.pth'

```
## 1. Interactive Streamlit Demos

These scripts launch web-based applications using Streamlit for interactive demonstrations.

### `demo_final.py`

Launches a file-based demo. Users can upload a video file, and the application will run the full analysis pipeline (Pose -> Action -> Grad-CAM -> LLM) and display the results, including the predicted action, confidence, most critical joint, and AI coach feedback.

**Usage:**
```bash
# Ensure you are in the project's root directory
streamlit run inference/demo_final.py
```

### `demo.py`

Launches a real-time demo using the user's webcam. It provides continuous action recognition feedback, overlaying the predicted action and score on the live video stream. It is optimized for responsiveness in a live setting.

**Usage:**
```bash
streamlit run inference/demo.py
```

### `demo_log_output.py`

An extension of the real-time webcam demo (`demo.py`) that includes performance logging. It allows the user to start/stop logging and download a CSV file containing frame-by-frame metrics like processing time for pose and action recognition.

**Usage:**
```bash
streamlit run inference/demo_log_output.py
```
This directory contains Python scripts for running inference and demonstrating the capabilities of the skeleton-based action recognition framework. The scripts are categorized into two main types: command-line inference scripts for analysis and Streamlit-based interactive demos.

---

## 2. Command-Line Inference Scripts

These scripts are designed for offline processing of video files to perform action recognition, performance benchmarking, and explainable AI (XAI) analysis.

### `inference_new_gradcam.py`

This is the primary script for a complete, offline analysis pipeline. It processes a video to extract skeletons, performs action recognition, runs a Grad-CAM analysis to identify critical joints, and generates a text report with feedback from a Large Language Model (LLM).

**Usage:**
```bash
# Ensure you are in the project's root directory
python inference/inference_new_gradcam.py \
    --video /path/to/your/exercise.mp4 \
    --output_folder /path/to/save/results \
    --action_name "squat_correct"
```

### `only_inference.py`

A script focused on batch inference and performance analysis. It processes videos using a sliding window, aggregates predictions, and outputs performance metrics (like FPS) and top-k predictions to a JSON file. It's ideal for evaluating model accuracy and speed on a dataset.

**Usage:**
```bash
python inference/only_inference.py \
    --video_folder /path/to/your/videos \
    --output_folder /path/to/save/results \
    --window_size 30
```

### `inference_rtmpose_grad_LLM.py`

An earlier version of the full XAI pipeline. It generates a video with the Grad-CAM heatmap overlaid on the skeleton and saves LLM feedback to a text file.

**Usage:**
*(Note: This script may have hardcoded paths and requires modification before use.)*
```bash
python inference/inference_rtmpose_grad_LLM.py
```

---
