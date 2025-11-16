import os
import sys
import json
import torch
from pathlib import Path
import numpy as np
import copy # Needed for deep copying pose data
import random
from tqdm import tqdm

try:
    import cv2
    from mmpose.apis import MMPoseInferencer
    # Import necessary structures for creating PoseDataSample objects
    from mmpose.structures import PoseDataSample
    from mmengine.structures import InstanceData
except ImportError:
    print("[!] Required libraries not found. Please ensure mmpose and its dependencies are installed.")
    sys.exit(1)

# --- Configuration ---
# Configuration and checkpoint paths for 2D pose estimation
pose_config = '/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py'
pose_checkpoint = '/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'

# Configuration and checkpoint paths for 3D pose lifting
pose_3d_config = '/home/cvlab123/mmpose/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m.py'
pose_3d_checkpoint = '/home/cvlab123/mmpose/checkpoints/videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth'

# Input and output directories
INPUT_VIDEO_DIR = '/home/cvlab123/data/OOD_test/'
OUTPUT_JSON_DIR = '/home/cvlab123/data/json/json_OOD_test_3D'

# Supported video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.MOV']


def process_video(inferencer_2d, inferencer_3d, video_path: Path, output_path: Path):

    print(f"Processing video: {video_path}")
    try:
        # --- Pass 1: Perform 2D pose estimation for all frames ---
        print("Starting 2D pose estimation...")
        all_2d_poses = []
        last_valid_pose = None
        
        num_keypoints = len(inferencer_2d.inferencer.model.dataset_meta.get('keypoint_info', {}))
        if num_keypoints == 0: num_keypoints = 17 # Fallback for COCO

        results_2d_gen = inferencer_2d(str(video_path), show_progress=True)
        for frame_idx, frame_results_2d in enumerate(results_2d_gen):
            poses_in_frame = []
            if frame_idx % 2 == 1:
                continue
            if frame_results_2d['predictions'] and frame_results_2d['predictions'][0]:
                person_data_2d = frame_results_2d['predictions'][0][0]
                keypoints_2d = np.array(person_data_2d['keypoints'], dtype=np.float32)
                keypoint_scores_2d = np.array(person_data_2d['keypoint_scores'], dtype=np.float32)
                
                instance_data = InstanceData(keypoints=keypoints_2d, keypoint_scores=keypoint_scores_2d)
                pose_data_sample = PoseDataSample(pred_instances=instance_data)
                
                last_valid_pose = pose_data_sample
                poses_in_frame.append(pose_data_sample)
            else:
                print(f"Warning: No person detected in frame {frame_idx}. Re-using last valid pose.")
                if last_valid_pose:
                    poses_in_frame.append(copy.deepcopy(last_valid_pose))
                else:
                    placeholder_kpts = np.zeros((num_keypoints, 2), dtype=np.float32)
                    placeholder_scores = np.zeros(num_keypoints, dtype=np.float32)
                    placeholder_instance = InstanceData(keypoints=placeholder_kpts, keypoint_scores=placeholder_scores)
                    poses_in_frame.append(PoseDataSample(pred_instances=placeholder_instance))
            
            all_2d_poses.append(poses_in_frame)
        
        # --- Pass 2: Perform 3D pose lifting frame by frame with a sliding window ---
        print("Starting 3D pose lifting...")
        video_all_frames_data = []
        num_frames = len(all_2d_poses)
        # The 3D model expects a sequence of 27 frames
        window_size = 27
        center_frame_idx = window_size // 2

        for i in range(num_frames):
            # Create a sliding window of 2D poses centered around the current frame 'i'
            start = i - center_frame_idx
            end = i + center_frame_idx + 1
            
            window_2d_poses = []
            for j in range(start, end):
                # Pad with the first frame if the window goes before the start of the video
                actual_idx = max(0, j)
                # Pad with the last frame if the window goes past the end of the video
                actual_idx = min(actual_idx, num_frames - 1)
                window_2d_poses.append(all_2d_poses[actual_idx])

            # The 3D model will predict the pose for the center frame of the window
            results_3d_gen = inferencer_3d(
                inputs=[str(video_path)], # Still needs a reference input
                pose_results_2d=window_2d_poses,
                show_progress=False # Disable progress bar for inner loop
            )
            
            # The generator will yield one result for the center frame of the window
            frame_results_3d = next(results_3d_gen)

            people_data_for_current_frame = []
            if frame_results_3d['predictions'] and frame_results_3d['predictions'][0]:
                person_data_3d = frame_results_3d['predictions'][0][0]
                keypoints_3d = person_data_3d.get('keypoints')
                keypoint_scores_3d = person_data_3d.get('keypoint_scores')

                if keypoints_3d is not None and keypoint_scores_3d is not None:
                    pose_keypoints_3d_flat = []
                    for kp_3d, score_3d in zip(keypoints_3d, keypoint_scores_3d):
                        pose_keypoints_3d_flat.extend([float(kp_3d[0]), float(kp_3d[1]), float(kp_3d[2]), float(score_3d)])
                    
                    people_data_for_current_frame.append({'pose_keypoints_3d': pose_keypoints_3d_flat})
            
            video_all_frames_data.append({
                'frame_index': i,
                'people': people_data_for_current_frame
            })

        print("3D pose lifting completed.")

        final_output = {'version': '1.0', 'frames': video_all_frames_data}

        print(f"Saving processed video data to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(final_output, f, indent=4)

        print("Processed video data saved successfully.")
        return True

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        inferencer_2d = MMPoseInferencer(
            pose2d=pose_config,
            pose2d_weights=pose_checkpoint,
            device=device,
        )
        print("MMPoseInferencer (2D) initialized successfully.")

        inferencer_3d = MMPoseInferencer(
            pose3d=pose_3d_config,
            pose3d_weights=pose_3d_checkpoint,
            device=device,
        )
        print("MMPoseInferencer (3D) initialized successfully.")

    except Exception as e:
        print(f"Error initializing MMPoseInferencer: {e}")
        sys.exit(1)


    # Create output directories if they don't exist
    Path(OUTPUT_JSON_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create the same subfolder structure in the output directory
    for root, dirs, files in os.walk(INPUT_VIDEO_DIR):
        for dir_name in dirs:
            rel_path = os.path.relpath(os.path.join(root, dir_name), INPUT_VIDEO_DIR)
            os.makedirs(os.path.join(OUTPUT_JSON_DIR, rel_path), exist_ok=True)

    all_video_paths = []
    for ext in VIDEO_EXTENSIONS:
        all_video_paths.extend(Path(INPUT_VIDEO_DIR).rglob(f'*{ext}'))
    
    for video_path in tqdm(all_video_paths, desc="Processing Videos"):
        rel_path = video_path.relative_to(INPUT_VIDEO_DIR)
        output_subfolder = os.path.join(OUTPUT_JSON_DIR, rel_path.parent)
        
        # We'll use a unique name based on the video to avoid counter resets
        output_json_path = Path(output_subfolder) / f"{video_path.stem}.json"
        
        if os.path.exists(output_json_path):
            # print(f"Output file {output_json_path} already exists. Skipping.")
            continue
            
        process_video(inferencer_2d, inferencer_3d, video_path, output_json_path)
            
    print("\nAll videos processed successfully.")

  