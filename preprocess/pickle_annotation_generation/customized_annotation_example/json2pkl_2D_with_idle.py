import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import random
import argparse
from collections import defaultdict

MAX_IDLE_FOR_TRAINING = 200
# (load_rtmpose_keypoints function remains the same as you provided)
def load_rtmpose_keypoints(file_path):
    with open(file_path) as f:
        data = json.load(f)
    all_frames_data = data["people"]
    T = len(all_frames_data) // 2
    if (T == 0):
        return np.zeros((1, 0, 17, 2)), np.zeros((1, 0, 17)), 0
    V = 17
    M = 1
    C = 2
    keypoint = np.zeros((M, T, V, C), dtype=np.float32)
    keypoint_score = np.zeros((M, T, V), dtype=np.float32)
    for t in range (T):
        pose_data = all_frames_data[t * 2 + 1]
        pose = pose_data['pose_keypoints_2d']
        for v in range(V):
            x, y, s = pose[v * 3], pose[v * 3 + 1], pose[v * 3 + 2]
            keypoint[0, t, v, 0] = x
            keypoint[0, t, v, 1] = y
            keypoint_score[0, t, v] = s
    return keypoint, keypoint_score, T

# --- NEW: A robust function to find all samples ---
def discover_samples(root_dir):
    """
    Uses os.walk to find all .json samples, handling any nesting depth.
    It determines the label for each sample based on its path.
    """
    print("🔎 Finding all samples using os.walk()...")
    samples_by_class = defaultdict(list)
    
    for dirpath, _, filenames in os.walk(root_dir):
        json_files = [f for f in filenames if f.endswith('.json')]
        if not json_files:
            continue

        # Determine the label based on the relative folder path
        relative_path = os.path.relpath(dirpath, root_dir)
        path_parts = relative_path.split(os.sep)
        
        label_key = None
        # --- Problem 1 solved: Handle 'idle' class ---
        if path_parts[0] == 'idle':
            label_key = 'idle'
        # --- Problem 2 solved: Handle nested classes ---
        elif len(path_parts) >= 2:
            # Assumes the first two folders determine the class, e.g., "lunge/correct"
            exercise_type = path_parts[0]
            class_name = path_parts[1]
            label_key = f"{exercise_type}_{class_name}"

        if label_key is None:
            continue

        for json_file in json_files:
            sequence_path = os.path.join(dirpath, json_file)
            base_name = os.path.splitext(json_file)[0]
            unique_id = f"{label_key}_{base_name}"
            
            samples_by_class[label_key].append({
                'path': sequence_path,
                'unique_id': unique_id,
                'label_key': label_key
            })
            
    return samples_by_class

# --- MODIFIED: This function now uses discover_samples ---
def build_annotation(sample_by_class, label_map):
    annotations = []
    split = {'xsub_train': [], 'xsub_val': [], 'xsub_test': []}
    all_samples_for_processing = []
    
    # ✅ NEW: Dictionary to store final training counts
    train_sample_counts = defaultdict(int)

    print("📊 Performing stratified split for each class...")
    for label_key, samples in sample_by_class.items():
        if label_key not in label_map:
             print(f"[Warning] Skipping class '{label_key}' as it's not in the provided label_map.")
             continue
             
        random.shuffle(samples)
        n_samples = len(samples)
        
        if (label_key == 'idle' and n_samples > MAX_IDLE_FOR_TRAINING):
            samples = samples[:MAX_IDLE_FOR_TRAINING] 
            n_samples = len(samples) 
            print(f"Class 'idle' undersampled to {n_samples} total samples.")
                
        if n_samples < 3: 
            print(f"[Warning] Class '{label_key}' has only {n_samples} samples. Adding all to training set.")
            split['xsub_train'].extend([s['unique_id'] for s in samples])
            train_sample_counts[label_key] = n_samples # ✅ Log count
            all_samples_for_processing.extend(samples)
            continue
            
        n_val = max(1, n_samples // 10) 
        n_test = max(1, n_samples // 10) 
        class_val = samples[:n_val]
        class_test = samples[n_val:n_val+n_test]
        class_train = samples[n_val+n_test:]
        
        split['xsub_val'].extend([s['unique_id'] for s in class_val])
        split['xsub_test'].extend([s['unique_id'] for s in class_test])
        split['xsub_train'].extend([s['unique_id'] for s in class_train])
        train_sample_counts[label_key] = len(class_train) # ✅ Log count
        all_samples_for_processing.extend(samples)
    
    # --- ✅ NEW: Print the final training counts for re-weighting ---
    print("\n" + "="*60)
    print("📊 Final Training Sample Counts (for Class Weighting)")
    print("="*60)
    
    # Sort the counts by the label index for consistency with the model
    sorted_counts = sorted(train_sample_counts.items(), key=lambda item: label_map.get(item[0], 99))
    
    weights_list_for_config = [0.0] * len(label_map)
    print(f"{'Class Name':<25} | {'Label ID'} | {'Train Samples'}")
    print("-" * 50)
    
    for label_key, count in sorted_counts:
        label_id = label_map.get(label_key, -1)
        if label_id != -1:
            print(f"{label_key:<25} | {label_id:<8} | {count}")
            if label_id < len(weights_list_for_config):
                 weights_list_for_config[label_id] = float(count)

    print("-" * 50)
    print("Copy this into your MMAction2 config file's 'loss_cls' section:")
    
    total_samples = sum(weights_list_for_config)
    num_classes = len(weights_list_for_config)
    
    if total_samples == 0 or num_classes == 0:
        print("Error: No training samples found. Cannot calculate weights.")
    else:
        # Calculate weights: higher weight for smaller classes
        class_weights = [round(total_samples / (num_classes * count), 2) if count > 0 else 0 
                         for count in weights_list_for_config]
        
        print(f"class_weight = {class_weights}")
    print("="*60 + "\n")
    # --- End of new print block ---
    
    print("Processing all samples to build annotations...")
    for sample in tqdm(all_samples_for_processing, desc="Processing Samples"):
        folder_path, unique_id, label_key = sample['path'], sample['unique_id'], sample['label_key']
        label = label_map.get(label_key)

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
    parser = argparse.ArgumentParser(description="Generate MMAction-compatible pickle file")
    parser.add_argument('--json_root', default='../data/json/json_all_view_plus_idle', help='Root directory of JSON files')
    parser.add_argument('--out_pkl', default=f'../data/pickle/1101_Ourdataset_with_idle.pkl', help='Output path for the .pkl file')
    args = parser.parse_args()
    random.seed(42)

    # --- MODIFIED: Use discover_samples to find all data first ---
    all_samples = discover_samples(args.json_root)
    
    # --- MODIFIED: Generate label map from the discovered samples ---
    print("🗺️  Generating label map from discovered samples...")
    label_map = {}
    # Prioritize 'idle' as label 0
    if 'idle' in all_samples:
        label_map['idle'] = 0
    # Add other labels
    for label_key in sorted(all_samples.keys()):
        if label_key not in label_map:
            label_map[label_key] = len(label_map)

    print("--- Generated Label Map ---")
    print(json.dumps(label_map, indent=4))
    print("--------------------------")

    out_dir = os.path.dirname(args.out_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Pass the discovered samples directly to the build function
    anno_data = build_annotation(all_samples, label_map)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(anno_data, f)
        
    print(f"✅ Annotation file successfully generated at: {args.out_pkl}")
