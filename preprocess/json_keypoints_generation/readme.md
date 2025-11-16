# JSON Keypoint Generation

The scripts in this directory are used to extract 2D or 3D skeleton keypoints from video files and save them as `.json` files. This is the first step in the data preprocessing pipeline.

### Prerequisites

Ensure you have installed all the necessary dependencies for `mmpose` and `videopose3d`. Refer to the main project documentation for installation instructions.

---

### 2D Keypoint Generation

The `get_keypoints_rtmpose.py` script uses the RTMPose model from MMPose to extract 2D keypoints for each frame of the input videos.

**Usage:**

The script is configured to process videos from a specified input directory and save the corresponding `.json` files to an output directory. You may need to modify the paths inside the script before running.

```bash
# Navigate to the project root directory
python preprocess/json_keypoints_generation/get_keypoints_rtmpose.py
```

---

### 3D Keypoint Generation

The `get_keypoints_rtmpose_videopose3d.py` script first generates 2D keypoints and then lifts them to 3D using the VideoPose3D model.

**Usage:**

This script also reads from an input video directory and saves the resulting 3D keypoints to an output directory. You may need to modify the paths inside the script before running.

```bash
# Navigate to the project root directory
python preprocess/json_keypoints_generation/get_keypoints_rtmpose_videopose3d.py
```
