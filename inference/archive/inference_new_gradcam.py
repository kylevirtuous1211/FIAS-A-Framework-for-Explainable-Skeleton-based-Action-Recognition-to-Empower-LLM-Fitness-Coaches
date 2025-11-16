import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import time
import sys
import copy
from collections import deque, Counter
from typing import List, Dict, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import argparse

# MMPose Imports
from mmpose.apis import MMPoseInferencer
from mmpose.structures import PoseDataSample

# MMAction and MMEngine Imports
from mmaction.apis import init_recognizer
from mmengine.registry import init_default_scope, DefaultScope
from mmengine.structures import InstanceData
from mmengine.dataset import Compose, pseudo_collate

from tqdm import tqdm
from module.llm import get_ai_coach_feedback
from module.generate_prompt import generate_llm_prompt_en

import os.path as osp
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
import fias_custom_loss.custom_loss  # <-- FORCE THE IMPORT

# ==============================================================================
# 0. Keypoint Name Definitions
# ==============================================================================
KEYPOINT_NAMES = {
    0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
    5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
    9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
    13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
}

# ==============================================================================
# 1. GradCAM Core Class
# ==============================================================================
class STGCNGradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.device = next(model.parameters()).device
        try:
            self.target_layer = self._get_target_layer()
        except Exception as e:
            print(f"ERROR: Could not find target layer '{target_layer_name}'. Check model architecture.")
            sys.exit(1)
            
        self.feature_maps = {}
        self.gradients = {}
        self.handlers = []
        self._register_hooks()

    def _get_target_layer(self):
        module = self.model
        for name in self.target_layer_name.split('.'):
            module = module[int(name)] if name.isdigit() else getattr(module, name)
        return module

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps[self.target_layer] = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients[self.target_layer] = grad_out[0].detach()
        
        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_backward_hook(backward_hook))

    def _calculate_localization_map(self, feature_maps, grads):
        if grads is None or torch.all(grads == 0):
            print(f"\nWARNING: Gradients for target layer '{self.target_layer_name}' are zero. Cannot generate heatmap.")
            return np.zeros(feature_maps.shape[2:], dtype=np.float32)

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(feature_maps * weights, dim=1)
        cam = F.relu(cam)
        return cam.cpu().numpy()

    def __call__(self, inputs, index=-1):
        self.model.zero_grad()
        
        cfg = self.model.cfg
        test_pipeline_cfg = cfg.get('test_pipeline', cfg.get('val_pipeline'))

        with DefaultScope.overwrite_default_scope('mmaction'):
            test_pipeline = Compose(test_pipeline_cfg)
        
        data = test_pipeline(inputs.copy())
        data = pseudo_collate([data])
        
        # The test_step function handles device placement and data unpacking internally.

        # Forward pass for prediction without gradients
        with torch.no_grad():
            results_for_pred = self.model.test_step(data)[0]
        
        scores = results_for_pred.pred_score
        if index == -1:
            index = scores.argmax().item()
        pred_score = scores[index].item()

        # ✅ [FIX] Use test_step again for the gradient pass. 
        # Calling it outside a `no_grad` context allows gradients to be computed.
        # This is simpler and more robust than calling model.forward() directly.
        results_for_grad = self.model.test_step(data)[0]
        score_for_backward = results_for_grad.pred_score[index]
        
        score_for_backward.backward()
        
        feature_maps = self.feature_maps.get(self.target_layer)
        grads = self.gradients.get(self.target_layer)

        if feature_maps is None or grads is None:
             raise RuntimeError("Hooks did not capture features or gradients.")

        localization_map = self._calculate_localization_map(feature_maps, grads)
        final_map = localization_map[0] if len(localization_map.shape) > 2 else localization_map
        
        return final_map, index, pred_score

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()

# ==============================================================================
# 2. Visualization Helper Function & Main Pipeline Class
# ==============================================================================
coco_skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                 [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                 [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                 [3, 5], [4, 6]]

def visualize_gradcam_on_frame(frame, keypoints, gradcam_scores, skeleton_conn, frame_size):
    fig, ax = plt.subplots(figsize=(frame_size[0] / 100, frame_size[1] / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    cmap = plt.get_cmap('jet')
    
    colors = np.array([[0.2, 0.2, 1.0]] * len(keypoints)) 
    if gradcam_scores is not None and np.any(gradcam_scores):
        enhanced_scores = np.nan_to_num(gradcam_scores**0.5)
        colors = cmap(enhanced_scores)[:, :3]
        
    ax.scatter(keypoints[:, 0], keypoints[:, 1], c=colors, s=150, zorder=2, alpha=0.9)
    
    lines, line_colors = [], []
    for p1_idx, p2_idx in skeleton_conn:
        if p1_idx < len(keypoints) and p2_idx < len(keypoints) and \
           keypoints[p1_idx, 0] > 0 and keypoints[p2_idx, 0] > 0:
            lines.append([keypoints[p1_idx], keypoints[p2_idx]])
            line_colors.append((colors[p1_idx] + colors[p2_idx]) / 2)
            
    lc = LineCollection(lines, colors=line_colors, linewidths=8, zorder=1, alpha=0.85)
    ax.add_collection(lc)
    
    fig.canvas.draw()
    buf = fig.canvas.tostring_argb()
    img_argb = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_rgba = np.roll(img_argb, 3, axis=2)
    
    plt.close(fig)
    return img_rgba[:, :, :3]

class FitnessAnalysisPipeline:
    def __init__(self, pose_config, pose_checkpoint, action_config, action_checkpoint, workdir_name):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading models...")
        self.workdir_name = workdir_name
        self.action_model = init_recognizer(action_config, action_checkpoint, device=self.device)
        self.pose_inferencer = MMPoseInferencer(pose2d=pose_config, pose2d_weights=pose_checkpoint, device=self.device)
        print("Models loaded successfully.\n")

        # label_map = {
        #     "lunge_correct": 0, "lunge_knee_pass_toe": 1, "lunge_too_high": 2,
        #     "push_up_arched_back": 3, "push_up_correct": 4, "push_up_elbow": 5,
        #     "squat_correct": 6, "squat_feet_too_close": 7, "squat_knees_inward": 8
        # }
        # with idle class
        label_map = {
            "idle": 0,
            "lunge_correct": 1,
            "lunge_knee_pass_toe": 2,
            "lunge_too_high": 3,
            "push_up_arched_back": 4,
            "push_up_correct": 5,
            "push_up_elbow": 6,
            "squat_correct": 7,
            "squat_feet_too_close": 8,
            "squat_knees_inward": 9
        }
        self.idx_to_label = [""] * len(label_map)
        for label, idx in label_map.items():
            self.idx_to_label[idx] = label

    def run(self, video_path, frame_stride, pose_batch_size, window_size, stride, modality_key):
        total_start_time = time.time()
        
        # --- 1. Video Loading & Pose Estimation ---
        strided_frames, keypoints_list = self._extract_skeletons(video_path, frame_stride, pose_batch_size)
        if not strided_frames:
            return
        
        total_frames = len(keypoints_list)
        frame_h, frame_w, _ = strided_frames[0].shape

        # --- 2. Grad-CAM and Prediction Analysis ---
        final_gradcam_map, smoothed_result, all_valid_predictions = self._run_gradcam_analysis(keypoints_list, window_size, stride, (frame_h, frame_w), frame_stride)
        
        # --- 3. Visualization ---
        # self._generate_visualization(strided_frames, keypoints_list, final_gradcam_map, smoothed_result, video_path, frame_stride, modality_key)
        
        # --- 4. LLM Feedback ---
        self._generate_llm_report(final_gradcam_map, smoothed_result, all_valid_predictions, total_frames, video_path, modality_key, self.workdir_name)

        print("\n" + "="*80)
        print(f"🎉 All processing complete! Total time: {time.time() - total_start_time:.2f} seconds")

    def _extract_skeletons(self, video_path, frame_stride, pose_batch_size):
        start_time = time.time() 
        print(f"Extracting skeletons from {Path(video_path).name}...")
        cap = cv2.VideoCapture(video_path)
        video_frames = [frame for success, frame in iter(lambda: cap.read(), (False, None))]
        cap.release()
        
        if not video_frames:
            print(f"ERROR: Could not read video '{video_path}' or video is empty.")
            return None, None

        print(f"Original frame count: {len(video_frames)}. Downsampling by a stride of {frame_stride}...")
        strided_frames = video_frames[::frame_stride]
        print(f"New frame count after downsampling: {len(strided_frames)}.")
        
        results_generator = self.pose_inferencer(strided_frames, show_progress=True, batch_size=pose_batch_size)
        
        keypoints_list = []
        for p in results_generator:
            if p['predictions'] and p['predictions'][0]:
                keypoints_data = p['predictions'][0][0]
                keypoints = np.array(keypoints_data['keypoints'])
                scores = np.array(keypoints_data['keypoint_scores'])
                keypoints_with_scores = np.hstack([keypoints, scores[:, None]])
                keypoints_list.append(keypoints_with_scores)
            else:
                keypoints_list.append(np.zeros((17, 3)))
                
        print(f"Skeleton extraction complete.")
        print(f"--- ⏱️ Video read & skeleton extraction time: {time.time() - start_time:.2f} seconds ---\n")
        return strided_frames, keypoints_list

    def _run_gradcam_analysis(self, keypoints_list: List[np.ndarray], window_size: int, stride: int, frame_shape: tuple, frame_stride: int):
        start_time = time.time()
        print("Calculating Grad-CAM and predictions with sliding window...")
        target_layer_name = 'backbone.gcn.9.gcn'
        gradcam = STGCNGradCAM(self.action_model, target_layer_name)

        # --- Buffers for Aggregation ---
        all_valid_predictions = []
        CONFIDENCE_THRESHOLD = 0.5

        total_frames = len(keypoints_list)
        full_gradcam_map = np.zeros((total_frames, 17))
        overlap_counter = np.zeros((total_frames, 17))

        # Ensure there are enough frames for at least one window before proceeding.
        if total_frames < window_size:
            print(f"WARNING: Video has only {total_frames} frames, which is less than the required window size of {window_size}.")
            print("No predictions will be made.")
            # Return empty/default values to allow the script to complete without error.
            smoothed_result = {"name": "Not Detected", "score": 0.0}
            return np.zeros((total_frames, 17)), smoothed_result, []

        for i in tqdm(range(0, total_frames - window_size + 1, stride), desc="Sliding Window Progress"):
            window_keypoints = np.array(keypoints_list[i : i + window_size])
            anno = {
                'keypoint': window_keypoints[np.newaxis, ..., :2],
                'keypoint_score': window_keypoints[np.newaxis, ..., 2],
                'total_frames': window_size,
                'img_shape': frame_shape
            }
            try:
                gradcam_map, pred_idx, pred_score = gradcam(anno)
                
                target_shape = (17, window_size)
                resized_gradcam_map = cv2.resize(gradcam_map, target_shape, interpolation=cv2.INTER_LINEAR)
                
                full_gradcam_map[i : i + window_size] += resized_gradcam_map
                overlap_counter[i : i + window_size] += 1

                if pred_score >= CONFIDENCE_THRESHOLD:
                    pred_name = self.idx_to_label[pred_idx]
                    all_valid_predictions.append({
                        "action": pred_name,
                        "score": pred_score,
                        "start_frame": i * frame_stride
                    })

            except Exception as e:
                print(f"\nError processing window {i}-{i+window_size}: {e}")
                continue

        gradcam.remove_hooks()
        
        # --- New Aggregation Logic for Final Prediction ---
        smoothed_result = {"name": "Not Detected", "score": 0.0}
        if all_valid_predictions:
            # Step 1: Group predictions by action type
            action_scores = {}
            for pred in all_valid_predictions:
                action = pred['action']
                if action not in action_scores:
                    action_scores[action] = []
                action_scores[action].append(pred['score'])

            # Step 2: Calculate a weighted score for each action
            # This score considers both how often an action was detected and its average confidence.
            final_action_scores = {}
            for action, scores in action_scores.items():
                count = len(scores)
                avg_score = np.mean(scores)
                # The final score is a product of count and average confidence.
                # This rewards actions that are both frequent and confident.
                final_action_scores[action] = count * avg_score

            # Step 3: Find the action with the highest final score
            best_action = max(final_action_scores, key=final_action_scores.get)
            smoothed_result["name"] = best_action
            smoothed_result["score"] = np.mean(action_scores[best_action]) # Report the average confidence

        print(f"\nSmoothed Final Prediction: '{smoothed_result['name']}' (Score: {smoothed_result['score']:.2%})")
        print(f"Total valid predictions logged: {len(all_valid_predictions)}")

        overlap_counter[overlap_counter == 0] = 1
        final_gradcam_map = full_gradcam_map / overlap_counter
        print(f"--- ⏱️ Analysis time: {time.time() - start_time:.2f} seconds ---\n")
        
        return final_gradcam_map, smoothed_result, all_valid_predictions

    def _generate_visualization(self, strided_frames, keypoints_list, final_gradcam_map, results, video_path, frame_stride, modality_key):
        start_time = time.time()
        print("Generating visualization video...")
        
        video_stem = Path(video_path).stem
        # output_path = Path("gradcam_video_output") / f"{video_stem}_{modality_key}_visualized.mp4"
        
        frame_h, frame_w, _ = strided_frames[0].shape
        video_fps = 30 / frame_stride
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, (frame_w, frame_h))

        for frame_idx, frame in enumerate(tqdm(strided_frames, desc="Generating video")):
            kps = keypoints_list[frame_idx][:, :2]
            frame_scores = final_gradcam_map[frame_idx]
            
            scores_for_frame = None
            if frame_scores.max() > frame_scores.min():
                scores_for_frame = (frame_scores - frame_scores.min()) / (frame_scores.max() - frame_scores.min() + 1e-6)
            
            vis_frame_rgb = visualize_gradcam_on_frame(frame, kps, scores_for_frame, coco_skeleton, (frame_w, frame_h))
            vis_frame_bgr = cv2.cvtColor(vis_frame_rgb, cv2.COLOR_RGB2BGR)
            
            text = f'Pred: {results["name"]} ({results["score"]:.2%})'
            cv2.putText(vis_frame_bgr, text, (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
            # video_writer.write(vis_frame_bgr)
            
        # video_writer.release()
        # print(f"✅ Grad-CAM visualization video saved to: {output_path}")
        print(f"--- ⏱️ Visualization video generation time: {time.time() - start_time:.2f} seconds ---\n")

    def _generate_llm_report(self, final_gradcam_map, smoothed_result, all_predictions, total_frames, video_path, modality_key, workdir_name):
        print("\n" + "="*80)
        print("🤖 Generating Prompt and Feedback for LLM...")
        
        prediction_log = "No consistent action detected."
        if all_predictions:
            prediction_log_lines = [f"- At frame {p['start_frame']}: Detected '{p['action']}' (Score: {p['score']:.1%})" for p in all_predictions]
            prediction_log = "\n".join(prediction_log_lines)

        # TODO: To use the full prediction log, update the `generate_llm_prompt_en` function
        # in `module/generate_prompt.py` to accept a sixth argument, e.g., `prediction_log=None`.
        llm_prompt = generate_llm_prompt_en(
            final_gradcam_map, 
            smoothed_result["name"], 
            smoothed_result["score"], 
            KEYPOINT_NAMES, 
            total_frames
            # prediction_log # This is the 6th argument that caused the error
        )
        print("\n--- LLM PROMPT ---")
        print(llm_prompt)
        
        llm_response = get_ai_coach_feedback(llm_prompt)
        print("\n----- AI Coach Feedback: -----")
        print(llm_response)

        video_stem = Path(video_path).stem
        output_path = Path("gradcam_text_output") / f"{video_stem}_{modality_key}_{workdir_name}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("--- SMOOTHED FINAL PREDICTION ---\n")
            f.write(f"{smoothed_result['name']} ({smoothed_result['score']:.2%})\n")
            f.write("\n--- FULL PREDICTION LOG ---\n")
            f.write(prediction_log)
            f.write("\n\n--- LLM PROMPT ---\n")
            f.write(llm_prompt)
            f.write("\n\n--- AI COACH FEEDBACK ---\n")
            f.write(llm_response)
        print(f"✅ Full analysis and feedback saved to: {output_path}")

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fitness Analysis Pipeline with Grad-CAM")
    parser.add_argument('--pose-config', required=True, help='Path to the MMPose model config file.')
    parser.add_argument('--pose-checkpoint', required=True, help='Path to the MMPose model checkpoint file.')
    parser.add_argument('--action-config', required=True, help='Path to the MMAction2 model config file.')
    parser.add_argument('--action-checkpoint', required=True, help='Path to the MMAction2 model checkpoint file.')
    parser.add_argument('--video-path', required=True, help='Path to the input video file.')
    parser.add_argument('--modality-key', required=True, help='A short key for the model modality (e.g., "joint", "bone_motion") for output filenames.')
    parser.add_argument('--work_dir', required=True, help='workdir information is detail information of the training')
    
    
    args = parser.parse_args()

    # --- Pipeline Parameters ---
    WINDOW_SIZE = 40
    STRIDE = 1
    FRAME_STRIDE = 2
    POSE_BATCH_SIZE = 16
    
    # Create output directories if they don't exist
    Path("gradcam_video_output").mkdir(exist_ok=True)
    Path("gradcam_text_output").mkdir(exist_ok=True)

    try:
        pipeline = FitnessAnalysisPipeline(
            pose_config=args.pose_config,
            pose_checkpoint=args.pose_checkpoint,
            action_config=args.action_config,
            action_checkpoint=args.action_checkpoint,
            workdir_name=args.work_dir
        )
        pipeline.run(
            video_path=args.video_path,
            frame_stride=FRAME_STRIDE,
            pose_batch_size=POSE_BATCH_SIZE,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            modality_key=args.modality_key
        )
    except Exception as e:
        print(f"\n--- A CRITICAL ERROR OCCURRED ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
