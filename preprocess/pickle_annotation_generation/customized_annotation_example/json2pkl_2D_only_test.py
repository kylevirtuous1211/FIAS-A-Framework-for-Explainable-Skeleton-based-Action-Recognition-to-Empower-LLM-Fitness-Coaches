import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import random
import argparse
from collections import defaultdict

def load_rtmpose_keypoints(file_path):

    with open(file_path) as f:
        data = json.load(f)
    
    all_frames_data = data["people"]
    
    T = len(all_frames_data) // 2 # Total number of frames
    if (T == 0):
        return np.zeros((1, 0, 17, 2)), np.zeros((1, 0, 17)), 0
    V = 17  # rtmpose keypoints
    M = 1   # Number of persons
    C = 2   # (x, y, score) coordinates

    keypoint = np.zeros((M, T, V, C), dtype=np.float32)
    keypoint_score = np.zeros((M, T, V), dtype=np.float32)

    for t in range (T):
        pose_data = all_frames_data[t * 2 + 1]
        
        pose = pose_data['pose_keypoints_2d']
        
        for v in range(V):
            x = pose[v * 3]
            y = pose[v * 3 + 1]
            s = pose[v * 3 + 2]
            
            keypoint[0, t, v, 0] = x
            keypoint[0, t, v, 1] = y
            keypoint_score[0, t, v] = s

    return keypoint, keypoint_score, T

def build_annotation(rtmpose_root, label_map):
    """
    Crawls the directory structure to find all samples, processes them,
    and builds the annotation dictionary with train, val, and test splits.
    """
    annotations = []
    # Initialize train, val, and test splits with "xsub" naming
    split = {'xsub_train': [], 'xsub_val': [], 'xsub_test': []}
    
    # --- Discover all sample paths ---
    sample_by_class = defaultdict(list)
    print("🔎 Finding all samples...")
    for exercise_type in sorted(os.listdir(rtmpose_root)):
        exercise_path = os.path.join(rtmpose_root, exercise_type)
        if not os.path.isdir(exercise_path):
            continue
        
        for class_name in sorted(os.listdir(exercise_path)):
            class_path = os.path.join(exercise_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            label_key = f"{exercise_type}_{class_name}"
            
            for sequence_file in sorted(os.listdir(class_path)):
                # Process only .json files
                if sequence_file.endswith('.json'):
                    sequence_path = os.path.join(class_path, sequence_file)
                    base_name = os.path.splitext(sequence_file)[0]
                    
                    unique_id = f"{label_key}_{base_name}"
                    
                    sample_by_class[label_key].append({
                        'path': sequence_path,
                        'unique_id': unique_id,
                        'label_key': label_key
                    })

    # --- Shuffle samples for a random train/val/test split ---

    print("📊 Performing stratified split for each class...")
    all_samples_for_processing = []
    
    for label_key, samples in sample_by_class.items():
        class_test = samples
        
        for sample in class_test:
            split['xsub_test'].append(sample['unique_id'])
        all_samples_for_processing.extend(samples)
            
    # --- Process each sample ---
    print("Processing all samples to build annotations...")
    for sample in tqdm(all_samples_for_processing, desc="Processing Samples"):
        folder_path = sample['path']
        unique_id = sample['unique_id']
        label_key = sample['label_key'] 
        
        label = label_map.get(label_key)
        if label is None:
            print(f"[Warning] Unknown label key: {label_key} for file {folder_path}")
            continue

        keypoint, keypoint_score, T = load_rtmpose_keypoints(folder_path)
        if T == 0:
             continue

        annotation = {
            'frame_dir': unique_id,
            'total_frames': T,
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'label': label,
            'keypoint': keypoint,
            'keypoint_score': keypoint_score,
        }
        annotations.append(annotation)

    return {'annotations': annotations, 'split': split}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate MMAction-compatible pickle file from OpenPose JSONs")

    parser.add_argument('--json_root', default='/home/cvlab123/data/json/json_OOD_test_2D', help='Root directory of JSON files (e.g., data/json)')
    parser.add_argument('--out_pkl', default='../data/pickle/json_OOD_test_2D_without_idle.pkl', help='Output path for the .pkl file (e.g., data/custom_data.pkl)')
    
    args = parser.parse_args()
    random.seed(42)

    # --- Dynamically generate the label map from the folder structure ---
    print("🗺️  Generating label map from directory structure...")
    # label_map = {}
    # current_label_idx = 0
    
    # root_dir = args.json_root
    # for exercise_type in sorted(os.listdir(root_dir)):
    #     exercise_path = os.path.join(root_dir, exercise_type)
    #     if not os.path.isdir(exercise_path):
    #         continue
    #     for class_name in sorted(os.listdir(exercise_path)):
    #         class_path = os.path.join(exercise_path, class_name)
    #         if not os.path.isdir(class_path):
    #             continue
            
    #         label_key = f"{exercise_type}_{class_name}"
    #         if label_key not in label_map:
    #             label_map[label_key] = current_label_idx
    #             current_label_idx += 1
                
    # print("--- Generated Label Map ---")
    # print(json.dumps(label_map, indent=4))
    # print("--------------------------")
    print("🗺️  Using predefined label map...")
    label_map = {
        "lunge_correct": 0,
        "lunge_knee_pass_toe": 1,
        "lunge_too_high": 2,
        "push_up_arched_back": 3,
        "push_up_correct": 4,
        "push_up_elbow": 5,
        "squat_correct": 6,
        "squat_feet_too_close": 7,
        "squat_knees_inward": 8
    }
    
    out_dir = os.path.dirname(args.out_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    anno_data = build_annotation(args.json_root, label_map)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(anno_data, f)
        
    print(f"✅ Annotation file successfully generated at: {args.out_pkl}")
