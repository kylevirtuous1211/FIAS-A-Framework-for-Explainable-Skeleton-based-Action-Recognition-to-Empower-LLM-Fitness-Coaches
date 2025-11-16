import torch
import cv2
import numpy as np
import time
from pathlib import Path
import copy
import json
import os

# MMPose Imports
from mmpose.apis import MMPoseInferencer
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData

# MMAction Imports
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.structures import ActionDataSample


class PoseActionRecognition:
    """
    A class to perform pose estimation (2D), optional 3D lifting, and skeleton-based action recognition.
    """
    def __init__(self, action_rec_config: str, action_rec_checkpoint: str,
                 pose_2d_config: str, pose_2d_checkpoint: str,
                 pose_3d_config: str, pose_3d_checkpoint: str):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize models
        print("Initializing 2D Pose Estimator...")
        self.pose_inferencer_2d = MMPoseInferencer(pose2d=pose_2d_config, pose2d_weights=pose_2d_checkpoint, device=self.device)
        print("Initializing 3D Pose Lifter...")
        self.pose_inferencer_3d = MMPoseInferencer(pose3d=pose_3d_config, pose3d_weights=pose_3d_checkpoint, device=self.device)
        print("Initializing Action Recognizer...")
        self.action_model = init_recognizer(action_rec_config, action_rec_checkpoint, device=self.device)

        self.is_3d_action_model = '3d' in action_rec_config.lower()
        self.label_map = {
            "lunge_correct": 0, "lunge_knee_pass_toe": 1, "lunge_too_high": 2,
            "push_up_arched_back": 3, "push_up_correct": 4, "push_up_elbow": 5,
            "squat_correct": 6, "squat_feet_too_close": 7, "squat_knees_inward": 8
        }
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.warmup()

    def warmup(self):
        """Warms up the models to prevent latency on the first inference."""
        print("Warming up models...")
        dummy_image = np.zeros((256, 192, 3), dtype=np.uint8)
        _ = next(self.pose_inferencer_2d(dummy_image))
        
        dummy_kpts_2d = np.zeros((17, 2), dtype=np.float32)
        dummy_scores_2d = np.ones(17, dtype=np.float32)
        instance_2d = InstanceData(keypoints=dummy_kpts_2d, keypoint_scores=dummy_scores_2d)
        pose_sample_2d = PoseDataSample(pred_instances=instance_2d)
        _ = next(self.pose_inferencer_3d(inputs=[dummy_image], pose_results_2d=[[pose_sample_2d]]))
        
        dummy_anno_2d = {
            'keypoint': np.zeros((1, 40, 17, 2), dtype=np.float32),
            'keypoint_score': np.ones((1, 40, 17), dtype=np.float32),
            'total_frames': 40, 'img_shape': (256, 192)
        }
        _ = inference_recognizer(self.action_model, dummy_anno_2d)
        print("Warmup complete. 🔥\n")
        
    def get_top_predictions(self, result: ActionDataSample, k: int = 3) -> list:
        if result is None: return []
        all_scores = result.pred_score.cpu().numpy()
        sorted_indices = np.argsort(all_scores)[::-1]
        top_k = [{'action': self.idx_to_label[idx], 'score': float(all_scores[idx])} 
                 for idx in sorted_indices[:min(k, len(sorted_indices))]]
        return top_k

    def run_inference(self, video_path: str, frame_stride: int, pose_batch_size: int, 
                      window_size: int = 40, window_stride: int = 1) -> dict:
        print(f"--- Starting Inference on {Path(video_path).name} ---")
        pipeline_timers = {}
        start_pipeline_time = time.time()
        
        # Determine if the action model is 2D or 3D from its config
        is_3d_action_model = '3d' in self.action_model.cfg.filename.lower()
        dimension = 3 if is_3d_action_model else 2
        print(f"Action model detected as {dimension}D.")

        # === 1. Video Loading and Frame Extraction ===
        start_block_time = time.time()
        cap = cv2.VideoCapture(video_path)
        video_frames = [frame for success, frame in iter(lambda: cap.read(), (False, None))]
        if not video_frames:
            raise ValueError(f"Could not read frames from video: {video_path}")
        frame_height, frame_width, _ = video_frames[0].shape
        cap.release()
        strided_frames = video_frames[::frame_stride]
        pipeline_timers['1_Video_Loading'] = time.time() - start_block_time
        print(f"[TIMER] 1. Video Loading & Striding: {pipeline_timers['1_Video_Loading']:.4f}s")

        # === 2. 2D Pose Estimation ===
        print("\nRunning 2D pose estimation...")
        start_block_time = time.time()
        results_2d_list = list(self.pose_inferencer_2d(strided_frames, pose_batch_size=pose_batch_size, show_progress=True))
        pipeline_timers['2_2D_Pose_Inference'] = time.time() - start_block_time
        print(f"[TIMER] 2. 2D Pose Inference: {pipeline_timers['2_2D_Pose_Inference']:.4f}s")

        # === 3. Data Formatting ===
        start_block_time = time.time()
        all_2d_poses_for_lift = []
        keypoints_for_action = []
        scores_for_action = []
        last_valid_pose = None
        num_keypoints = 17

        for frame_results_2d in results_2d_list:
            if frame_results_2d['predictions'] and frame_results_2d['predictions'][0]:
                person_data = frame_results_2d['predictions'][0][0]
                keypoints = np.array(person_data['keypoints'], dtype=np.float32)
                scores = np.array(person_data['keypoint_scores'], dtype=np.float32)
                instance_data = InstanceData(keypoints=keypoints, keypoint_scores=scores)
                pose_data_sample = PoseDataSample(pred_instances=instance_data)
                last_valid_pose = pose_data_sample
                all_2d_poses_for_lift.append([pose_data_sample])
            else: # If no person is detected, use a placeholder
                placeholder_kpts = np.zeros((num_keypoints, 2), dtype=np.float32)
                placeholder_scores = np.zeros(num_keypoints, dtype=np.float32)
                instance = InstanceData(keypoints=placeholder_kpts, keypoint_scores=placeholder_scores)
                all_2d_poses_for_lift.append([PoseDataSample(pred_instances=instance)])

        # === 4. 3D Lifting (if needed) or 2D Data Prep ===
        if is_3d_action_model:
            print("\nRunning 3D pose lifting...")
            results_3d_list = list(self.pose_inferencer_3d(inputs=strided_frames, pose_results_2d=all_2d_poses_for_lift, show_progress=True))
            for frames_results_3d in results_3d_list:
                if frames_results_3d['predictions'] and frames_results_3d['predictions'][0]:
                    keypoints_3d = frames_results_3d['predictions'][0][0].get('keypoints')
                    keypoints_for_action.append(keypoints_3d)
                else:
                    keypoints_for_action.append(np.zeros((num_keypoints, 3), dtype=np.float32))
        else: # Prepare 2D data
            print("\nPreparing 2D keypoints for action recognition...")
            for frame_poses in all_2d_poses_for_lift:
                pred_instance = frame_poses[0].pred_instances
                keypoints_for_action.append(pred_instance.keypoints)
                scores_for_action.append(pred_instance.keypoint_scores)
        
        pipeline_timers['3_Data_Formatting_and_Lifting'] = time.time() - start_block_time
        print(f"[TIMER] 3. Data Formatting & 3D Lift (if any): {pipeline_timers['3_Data_Formatting_and_Lifting']:.4f}s")

        if not keypoints_for_action:
            raise RuntimeError("No keypoints were extracted from the video.")

        # === 5. Sliding Window Action Recognition ===
        print(f"\nRunning Action Recognition with sliding window (size: {window_size}, stride: {window_stride})...")
        start_block_time = time.time()

        keypoints_array = np.array(keypoints_for_action)
        scores_array = np.array(scores_for_action) if not is_3d_action_model else None
        
        total_frames = keypoints_array.shape[0]
        if total_frames < window_size:
            raise ValueError(f"Total frames after striding ({total_frames}) is less than window size ({window_size}).")

        all_window_scores = []
        for i in range(0, total_frames - window_size + 1, window_stride):
            window_keypoints = keypoints_array[i : i + window_size]
            
            anno = {
                'keypoint': window_keypoints[np.newaxis, ...],
                'total_frames': window_size,
                'img_shape': (frame_height, frame_width)
            }
            if not is_3d_action_model:
                window_scores = scores_array[i : i + window_size]
                anno['keypoint_score'] = window_scores[np.newaxis, ...]
            
            # Run inference on the window
            result = inference_recognizer(self.action_model, anno)
            all_window_scores.append(result.pred_score.cpu().numpy())

        # Aggregate results by averaging scores across all windows
        if not all_window_scores:
            raise RuntimeError("No windows were processed for action recognition.")
            
        avg_scores = np.mean(all_window_scores, axis=0)
        
        # Create a final ActionDataSample with the averaged scores
        final_result = ActionDataSample()
        final_result.pred_score = torch.from_numpy(avg_scores)

        pipeline_timers['4_Action_Inference'] = time.time() - start_block_time
        print(f"[TIMER] 4. Action Recognition (Sliding Window): {pipeline_timers['4_Action_Inference']:.4f}s")
        
        total_time = time.time() - start_pipeline_time
        
        print(f"\n--- Performance Summary ---")
        for name, duration in pipeline_timers.items():
            print(f"{name:<35}: {duration:.4f} seconds")
        print("---------------------------------------")
        print(f"{'Total Pipeline Time':<35}: {total_time:.4f} seconds")
        print("---------------------------------------\n")
        
        return {'total_time': total_time, 'result': final_result}

def main():
    # ==============================================================================
    # --- ✏️ 1. EDIT YOUR CONFIGURATION HERE ---
    # ==============================================================================
    
    # --- Action Recognition Model ---
    # The config file for the action recognition model you want to test.
    ACTION_REC_CONFIG = '/home/cvlab123/mmaction2/configs/skeleton/custom_stgcnpp/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py'
    ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/916_rtmpose_all_2D_joint/best_acc_top1_epoch_13.pth'
    
    # --- Pose Estimation Models (usually don't need to change) ---
    POSE_2D_CONFIG = '/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py'
    POSE_2D_CHECKPOINT = '/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'
    POSE_3D_CONFIG = '/home/cvlab123/mmpose/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m.py'
    POSE_3D_CHECKPOINT = '/home/cvlab123/mmpose/checkpoints/videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth'
    
    # --- Input Video ---
    VIDEO_PATH = '/home/cvlab123/inference/test_video/feet_too_close_Kyle.MOV'
    
    # --- Output ---
    # Directory to save the results.
    OUTPUT_DIR = 'single_inference_result'
    
    # ==============================================================================
    # --- 2. END OF CONFIGURATION ---
    # ==============================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Initialize the full pipeline
        pipeline = PoseActionRecognition(
            action_rec_config=ACTION_REC_CONFIG,
            action_rec_checkpoint=ACTION_REC_CHECKPOINT,
            pose_2d_config=POSE_2D_CONFIG,
            pose_2d_checkpoint=POSE_2D_CHECKPOINT,
            pose_3d_config=POSE_3D_CONFIG,
            pose_3d_checkpoint=POSE_3D_CHECKPOINT
        )
        
        # Run inference
        metrics = pipeline.run_inference(
            video_path=VIDEO_PATH, 
            frame_stride=2, 
            pose_batch_size=16,
            window_size=40  # Using a 40-frame window
        )

        # Process and print results
        print("--- Final Results ---")
        top_preds = pipeline.get_top_predictions(metrics['result'], k=3)
        
        if top_preds:
            print(f"Top 3 Predictions for '{Path(VIDEO_PATH).name}':")
            for pred in top_preds:
                print(f"  - Action: {pred['action']}, Score: {pred['score']:.4f}")
        else:
            print("No action was recognized.")

        # Save results to a JSON file
        output_data = {
            'video_path': VIDEO_PATH,
            'total_inference_time': metrics['total_time'],
            'top_3_predictions': top_preds
        }
        output_filename = f"{Path(VIDEO_PATH).stem}_result.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"\n✅ Full results saved to: {output_path}")

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"\n--- An error occurred ---")
        print(f"Error: {e}")
        print("Please check your file paths and configurations.")

if __name__ == "__main__":
    main()
