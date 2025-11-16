# Data Preprocessing Pipeline

This directory contains the scripts necessary to process raw video data into a format suitable for training the ST-GCN++ action recognition model. The preprocessing pipeline consists of two main stages.

## Folder Structure

*   `json_keypoints_generation/`: Scripts for extracting 2D and 3D skeleton keypoints from videos.
*   `pickle_annotation_generation/`: Scripts for converting the generated keypoints into `.pkl` annotation files for model training.

---

## Pipeline Workflow

The end-to-end data preparation process is as follows:

### 1. Generate Keypoints from Videos

First, use the scripts in the `json_keypoints_generation/` directory to process your video dataset. This step uses pose estimation models (RTMPose for 2D and VideoPose3D for 3D) to extract skeletal data for each frame and saves it as a `.json` file for each video.

➡️ **See instructions in: `json_keypoints_generation/readme.md`**

### 2. Generate Pickle Annotations

Next, use the scripts in the `pickle_annotation_generation/` directory. These scripts take the `.json` keypoint files and package them into `.pkl` (pickle) files. This is the final format required by the ST-GCN++ model for training and evaluation.

➡️ **See instructions in: `pickle_annotation_generation/readme.md`**