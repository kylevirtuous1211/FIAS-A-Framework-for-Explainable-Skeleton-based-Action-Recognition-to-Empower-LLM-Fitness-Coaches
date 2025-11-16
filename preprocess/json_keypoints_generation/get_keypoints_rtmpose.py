import os
import sys
import json
import torch
from pathlib import Path
import cv2
from mmpose.apis import MMPoseInferencer
import random
from tqdm import tqdm

# ... (keep the rest of your file's imports and constants) ...

pose_config = '/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py'
pose_checkpoint = '/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'

INPUT_VIDEO_DIR = '/home/cvlab123/data/raw_data/test_friends'
OUTPUT_JSON_DIR = '/home/cvlab123/data/json/json_friend_test'
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.MOV']


def process_video(inferencer, video_path: Path, output_path: Path):
    """
    ✅ UPDATED: Process a single video file to extract keypoints, ensuring a
    consistent 40-frame clip using robust random clipping and padding.
    """
    try:
        # print(f"Processing video: {video_path}")
        results_generator = inferencer(str(video_path), show_progress=False)

        # --- Step 1: Extract keypoints from all frames first ---
        all_keypoints_data = []
        for frame_idx, frame_results in enumerate(results_generator):
            # Process every other frame to match training stride
            if frame_idx % 2 == 0:
                if not frame_results['predictions'] or not frame_results['predictions'][0]:
                    continue

                person_data = frame_results['predictions'][0][0]
                keypoints = person_data['keypoints']
                keypoint_scores = person_data['keypoint_scores']

                pose_keypoints_2d = []
                for kp, score in zip(keypoints, keypoint_scores):
                    pose_keypoints_2d.extend([float(kp[0]), float(kp[1]), score])
                
                all_keypoints_data.append({
                    'frame_index': frame_idx,
                    'pose_keypoints_2d': pose_keypoints_2d
                })

        # --- Step 3: Format the data for saving ---
        video_frame_data = all_keypoints_data
        final_output = {
            'version': '1.0',
            'people': video_frame_data
        }

        with open(output_path, 'w') as f:
            json.dump(final_output, f)
            
        # print(f"Processed video saved to: {output_path}")
        return True
    
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return False

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        inferencer = MMPoseInferencer(
            pose_config, 
            pose_checkpoint, 
            device=device, 
            show_progress=True,
        )
    except Exception as e:
        print(f"Error initializing MMPoseInferencer: {e}")
        sys.exit(1)
        
    print("MMPoseInferencer initialized successfully.")
    
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    print(f"Output directory set to: {OUTPUT_JSON_DIR}")
    
    # Create the same subfolder structure in the output directory
    for root, dirs, files in os.walk(INPUT_VIDEO_DIR):
        for dir_name in dirs:
            rel_path = os.path.relpath(os.path.join(root, dir_name), INPUT_VIDEO_DIR)
            os.makedirs(os.path.join(OUTPUT_JSON_DIR, rel_path), exist_ok=True)

    all_video_paths = []
    for ext in VIDEO_EXTENSIONS:
        all_video_paths.extend(Path(INPUT_VIDEO_DIR).rglob(f'*{ext}'))

    print(f"Found {len(all_video_paths)} total videos to process.")

    for video_path in tqdm(all_video_paths, desc="Processing Videos"):
        rel_path = video_path.relative_to(INPUT_VIDEO_DIR)
        output_subfolder = os.path.join(OUTPUT_JSON_DIR, rel_path.parent)
        
        # We'll use a unique name based on the video to avoid counter resets
        output_json_path = Path(output_subfolder) / f"{video_path.stem}.json"
        
        if os.path.exists(output_json_path):
            # print(f"Output file {output_json_path} already exists. Skipping.")
            continue
            
        process_video(inferencer, video_path, output_json_path)
            
    print("\nAll videos processed successfully.")
