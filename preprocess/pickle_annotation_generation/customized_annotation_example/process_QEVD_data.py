import json
import os
import shutil
from tqdm import tqdm
from collections import defaultdict
import random

# ==============================================================================
# 1. Configuration - Define Mappings and Paths
# ==============================================================================
# --- Mapping rules from QEVD labels to your custom class names ---
EXERCISE_MAPPING = {
    # Push-ups
    "pushups - 90 degrees": ("push_up", "correct"),
    "pushups - shallow": ("push_up", "shallow"),
    "pushups - very wide": ("push_up", "elbow"),
    # Squats
    "squats - narrow": ("squat", "feet_too_close"),
    "squats - back not straight": ("squat", "back_not_straight"),

    "lunges - depth=1": ("lunge", "too_high"), 
    "lunges - depth=2": ("lunge", "too_high"), 
    "front knee going forward over toes": ("lunge", "knee_pass_toe") 
}

# List of keywords that will map to the 'idle' class
IDLE_KEYWORDS = [
    "bobbing head", "neck-warm-up", "boxing bounce-steps", "nodding head",
    "catching your breath", "open and drink", "picking up the camera",
    "plank preparation", "towel off sweat", "stretching", "adjusting",
    "waving", "pointing", "looking around"
]

# --- Sampling Parameters ---
# ✅ NEW: Define the training split ratio (90% for train, 10% for test)
TRAIN_SPLIT_RATIO = 0.9 

# --- Base Paths ---
ROOT_DIR = "../data"
QEVD_METADATA_PATH = os.path.join(ROOT_DIR, 'QEVD', 'fine_grained_labels.json')
QEVD_VIDEO_DIRS = {
    "Part-1": os.path.join(ROOT_DIR, 'QEVD', 'QEVD-FIT-300k-Part-1'),
    "Part-2": os.path.join(ROOT_DIR, 'QEVD', 'QEVD-FIT-300k-Part-2'),
    "Part-3": os.path.join(ROOT_DIR, 'QEVD', 'QEVD-FIT-300k-Part-3'),
    "Part-4": os.path.join(ROOT_DIR, 'QEVD', 'QEVD-FIT-300k-Part-4')
}

# ✅ NEW: Define separate Train and Test directories
OOD_TRAIN_DIR = os.path.join(ROOT_DIR, 'OOD_TRAIN/OOD_train')
OOD_TEST_DIR = os.path.join(ROOT_DIR, 'OOD_TRAIN/OOD_test')


# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def find_video_path(video_filename):
    """Finds the full path of a video file across the QEVD part-directories."""
    for part_dir in QEVD_VIDEO_DIRS.values():
        potential_path = os.path.join(part_dir, video_filename)
        if os.path.exists(potential_path):
            return potential_path
    return None

def get_mapped_class(labels):
    """
    Maps a list of QEVD labels to a single custom class.
    This version is stricter and only returns a mapping if it is completely unambiguous.
    """
    potential_mappings = set()
    labels_set = set(labels)

    # --- Strict check for 'squat_correct' ---
    # It must contain all three specified labels to be considered correct.
    correct_squat_labels = {"squats - shoulder-width", "squats - 90 degrees", "squats - no obvious issue"}
    if correct_squat_labels.issubset(labels_set):
        potential_mappings.add(("squat", "correct"))
        
        
    correct_lunge_labels = { "alternating forward lunges - depth=4", "alternating forward lunges - no obvious issue"}
    correct_lunge_labels5 = { "alternating forward lunges - depth=5", "alternating forward lunges - no obvious issue"}
    if correct_lunge_labels.issubset(labels_set) or correct_lunge_labels5.issubset(labels_set):
        potential_mappings.add(("lunge", "correct"))
    
    # -----------------------------------------

    for label in labels:
        for keyword, mapping in EXERCISE_MAPPING.items():
            if keyword in label:
                potential_mappings.add(mapping)
        
        for keyword in IDLE_KEYWORDS:
            if keyword in label:
                potential_mappings.add(("idle", "general"))

    if not potential_mappings:
        return None, None
    
    exercise_types = {m[0] for m in potential_mappings if m[0] != 'idle'}
    if len(exercise_types) > 1:
        return None, None
        
    if len(potential_mappings) > 1:
        return None, None
        
    return potential_mappings.pop()

def copy_files(video_list, base_dir, pbar_desc):
    """Copies a list of video_info dicts to the target base_dir."""
    print(f"\n{pbar_desc}")
    copy_count = 0
    for video_info in tqdm(video_list, desc="Copying files"):
        exercise_type = video_info['type']
        class_name = video_info['class']
        src_path = video_info['src']
        
        dest_dir = None
        if exercise_type == 'idle':
            dest_dir = os.path.join(base_dir, "idle")
        else:
            dest_dir = os.path.join(base_dir, exercise_type, class_name)
        
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        
        if not os.path.exists(dest_path):
            shutil.copy(src_path, dest_path)
            copy_count += 1
    return copy_count

# ==============================================================================
# 3. Main Script Logic
# ==============================================================================

def main():
    print("Starting QEVD dataset processing...")

    # ✅ NEW: Create both train and test directories
    os.makedirs(OOD_TRAIN_DIR, exist_ok=True)
    os.makedirs(OOD_TEST_DIR, exist_ok=True)

    print(f"Loading metadata from {QEVD_METADATA_PATH}")
    with open(QEVD_METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    print("Pass 1: Identifying and collecting potential video files...")
    # ✅ NEW: Use one dictionary for ALL classes, including idle
    all_videos_by_class = defaultdict(list)

    for entry in tqdm(metadata, desc="Scanning metadata"):
        video_filename = entry.get("video_path", "").lstrip("./")
        labels = entry.get("labels", [])
        
        # ✅ NEW: We don't need the original 'split' since we are creating our own.
        if not video_filename or not labels:
            continue

        exercise_type, class_name = get_mapped_class(labels)

        if not exercise_type:
            continue

        src_path = find_video_path(video_filename)
        if not src_path:
            continue
        
        # ✅ NEW: Add all classes to the same dictionary
        full_class_name = f"{exercise_type}/{class_name}"
        all_videos_by_class[full_class_name].append({
            'src': src_path,
            'type': exercise_type,
            'class': class_name
        })

    print("\n--- Initial Scan Results ---")
    for class_name, videos in all_videos_by_class.items():
        print(f"Found {len(videos)} total videos for class: {class_name}")

    # --- ✅ NEW: Step 2: Split ALL classes into 90/10 Train/Test sets ---
    print(f"\nPass 2: Splitting videos into {TRAIN_SPLIT_RATIO*100:.0f}/{100-TRAIN_SPLIT_RATIO*100:.0f} Train/Test sets...")
    
    train_videos_to_copy = []
    test_videos_to_copy = []

    for full_class_name, videos in all_videos_by_class.items():
        # Shuffle the list of videos for this class to ensure a random split
        random.shuffle(videos)
        
        # Find the split index
        split_index = int(len(videos) * TRAIN_SPLIT_RATIO)
        
        # Split the list
        train_set = videos[:split_index]
        test_set = videos[split_index:]
        
        # Add the videos to their respective copy lists
        train_videos_to_copy.extend(train_set)
        test_videos_to_copy.extend(test_set)
        
        print(f"  > Class '{full_class_name}': {len(train_set)} train, {len(test_set)} test")

    # --- ✅ NEW: Step 3: Copy files to their final destinations ---
    
    train_count = copy_files(
        train_videos_to_copy, 
        OOD_TRAIN_DIR, 
        "Pass 3: Copying TRAINING files..."
    )
    
    test_count = copy_files(
        test_videos_to_copy, 
        OOD_TEST_DIR, 
        "Pass 4: Copying TESTING files..."
    )

    print("\n--- Processing Complete ---")
    print(f"✅ Copied {train_count} new video files to: {OOD_TRAIN_DIR}")
    print(f"✅ Copied {test_count} new video files to: {OOD_TEST_DIR}")
    print(f"\nNext step: Generate annotation files for your new OOD_train and OOD_test datasets.")


if __name__ == "__main__":
    main()