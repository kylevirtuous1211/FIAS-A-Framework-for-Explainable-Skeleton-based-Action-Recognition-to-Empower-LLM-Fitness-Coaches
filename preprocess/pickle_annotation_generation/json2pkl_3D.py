import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import random
import argparse
from collections import defaultdict

def load_3d_keypoints(file_path):
    """
    Loads 3D keypoints from a JSON file for a single video sequence.
    This version creates a 'keypoint' array of shape (M, T, V, 3) containing
    only (x, y, z) and discards the confidence score.
    """
    with open(file_path) as f:
        data = json.load(f)
    
    all_frames_data = data.get("frames", [])
    
    T = len(all_frames_data)
    if T == 0:
        return np.zeros((1, 0, 17, 3)), 0

    V = 17
    M = 1
    C_coords = 3
    C_source = 4

    keypoint = np.zeros((M, T, V, C_coords), dtype=np.float32)

    for t, frame_data in enumerate(all_frames_data):
        people_in_frame = frame_data.get('people', [])
        if not people_in_frame:
            continue
            
        person_data = people_in_frame[0]
        pose = person_data.get('pose_keypoints_3d', [])
        
        if not pose or len(pose) < V * C_source:
            continue

        for v in range(V):
            base_idx = v * C_source
            x = pose[base_idx]
            y = pose[base_idx + 1]
            z = pose[base_idx + 2]
            
            keypoint[0, t, v, 0] = x
            keypoint[0, t, v, 1] = y
            keypoint[0, t, v, 2] = z

    return keypoint, T

def build_annotation(rtmpose_root, label_map):
    """
    Crawls the directory structure, groups samples by class, and performs
    a stratified split to build the annotation dictionary.
    --- THIS FUNCTION HAS BEEN UPDATED WITH STRATIFIED SAMPLING LOGIC ---
    """
    annotations = []
    split = {'xsub_train': [], 'xsub_val': [], 'xsub_test': []}
    
    # --- Step 1: Discover all sample paths and group by class ---
    sample_by_class = defaultdict(list)
    print("🔎 Finding and grouping samples by class...")
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
                if sequence_file.endswith('.json'):
                    sequence_path = os.path.join(class_path, sequence_file)
                    base_name = os.path.splitext(sequence_file)[0]
                    unique_id = f"{label_key}_{base_name}"
                    
                    sample_by_class[label_key].append({
                        'path': sequence_path,
                        'unique_id': unique_id,
                        'label_key': label_key
                    })

    # --- Step 2: Perform stratified split for each class ---
    print("📊 Performing stratified split for each class...")
    all_samples_for_processing = []
    
    for label_key, samples in sample_by_class.items():
        random.shuffle(samples)
        
        n_samples = len(samples)
        
        # Handle very small classes to avoid empty training sets
        if n_samples < 3:
            print(f"[Warning] Class '{label_key}' has only {n_samples} samples. Putting all into training set.")
            for sample in samples:
                split['xsub_train'].append(sample['unique_id'])
            all_samples_for_processing.extend(samples)
            continue

        n_val = max(1, n_samples // 10)
        n_test = max(1, n_samples // 10)
        
        class_val = samples[:n_val]
        class_test = samples[n_val : n_val + n_test]
        class_train = samples[n_val + n_test:]
        
        for sample in class_val:
            split['xsub_val'].append(sample['unique_id'])
        for sample in class_test:
            split['xsub_test'].append(sample['unique_id'])
        for sample in class_train:
            split['xsub_train'].append(sample['unique_id'])
        
        all_samples_for_processing.extend(samples)
            
    # --- Step 3: Process all samples to build the 'annotations' list ---
    print("Processing all samples to build annotations...")
    for sample in tqdm(all_samples_for_processing, desc="Processing Samples"):
        folder_path = sample['path']
        unique_id = sample['unique_id']
        label_key = sample['label_key']
        
        label = label_map.get(label_key)
        if label is None:
            print(f"[Warning] Unknown label key: {label_key} for file {folder_path}")
            continue

        # Use the 3D keypoint loader
        keypoint, T = load_3d_keypoints(folder_path)
        if T == 0:
            continue

        # Create annotation dictionary for 3D data (no keypoint_score)
        annotation = {
            'frame_dir': unique_id,
            'total_frames': T,
            'img_shape': (1080, 1920),  # Placeholder, not used for 3D
            'original_shape': (1080, 1920), # Placeholder, not used for 3D
            'label': label,
            'keypoint': keypoint,
        }
        annotations.append(annotation)

    return {'annotations': annotations, 'split': split}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate MMAction-compatible pickle file from 3D JSONs")
    parser.add_argument('--json_root', default='data/json_rtmpose_all_view_3D', help='Root directory of JSON files (e.g., data/json)')
    # parser.add_argument('--json_root', default='data/json_rtmpose_front_3D', help='Root directory of JSON files (e.g., data/json)')
    # parser.add_argument('--json_root', default='data/json_rtmpose_back_view_3D', help='Root directory of JSON files (e.g., data/json)')
    parser.add_argument('--out_pkl', default='data/pickle/rtmpose_3D_all.pkl', help='Output path for the .pkl file (e.g., data/custom_data.pkl)')
    # parser.add_argument('--out_pkl', default='data/pickle/rtmpose_3D_front.pkl', help='Output path for the .pkl file (e.g., data/custom_data.pkl)')
    # parser.add_argument('--out_pkl', default='data/pickle/rtmpose_3D_back.pkl', help='Output path for the .pkl file')
    args = parser.parse_args()
    random.seed(42)

    # --- Dynamically generate the label map from the folder structure ---
    print("🗺️  Generating label map from directory structure...")
    label_map = {}
    current_label_idx = 0
    
    root_dir = args.json_root
    for exercise_type in sorted(os.listdir(root_dir)):
        exercise_path = os.path.join(root_dir, exercise_type)
        if not os.path.isdir(exercise_path):
            continue
        for class_name in sorted(os.listdir(exercise_path)):
            class_path = os.path.join(exercise_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            label_key = f"{exercise_type}_{class_name}"
            if label_key not in label_map:
                label_map[label_key] = current_label_idx
                current_label_idx += 1
                
    print("--- Generated Label Map ---")
    print(json.dumps(label_map, indent=4))
    print("--------------------------")

    out_dir = os.path.dirname(args.out_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    anno_data = build_annotation(args.json_root, label_map)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(anno_data, f)
        
    print(f"✅ Annotation file successfully generated at: {args.out_pkl}")