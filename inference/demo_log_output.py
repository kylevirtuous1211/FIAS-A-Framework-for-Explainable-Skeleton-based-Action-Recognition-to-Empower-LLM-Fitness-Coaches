# Usage (at /home/cvlab123): streamlit run demo.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import threading
from collections import deque
import numpy as np
import torch
from pathlib import Path
import time
import pandas as pd

# imports for Hierarchical Loss
import sys
import os.path as osp
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
import fias_custom_loss.custom_loss  # <-- FORCE THE IMPORT


# --- MMPose & MMAction Imports ---
try:
    from mmpose.apis import MMPoseInferencer
    from mmaction.apis import inference_recognizer, init_recognizer
except ImportError:
    st.error("ImportError")
    st.stop()


# --- Page Setup ---
st.set_page_config(page_title="FITS Demo", layout="wide")
st.title("FIAS Demo")

# --- Config ---
path_prefix = Path('/home/cvlab123') 
POSE_2D_CONFIG="/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py"
POSE_2D_CHECKPOINT="/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth"
ACTION_REC_CONFIG = path_prefix / 'mmaction2/configs/skeleton/custom_stgcnpp/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py'
ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/1101_CL_reweighting_add_idle_2D_joint/best_acc_top1_epoch_10.pth'

# label_map = {
#     0: "lunge_correct", 1: "lunge_knee_pass_toe", 2: "lunge_too_high",
#     3: "push_up_arched_back", 4: "push_up_correct", 5: "push_up_elbow",
#     6: "squat_correct", 7: "squat_feet_too_close", 8: "squat_knees_inward"
# }

label_map = {
    0: "idle",
    1: "lunge_correct",
    2: "lunge_knee_pass_toe",
    3: "lunge_too_high",
    4: "push_up_arched_back",
    5: "push_up_correct",
    6: "push_up_elbow",
    7: "squat_correct",
    8: "squat_feet_too_close",
    9: "squat_knees_inward" # F FINETUNE ON QEVD, THERE IS NO KNEES INWARD
}

@st.cache_resource
def load_models():
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        st.info(f"Loading models on {device}...")
        
        pose_inferencer = MMPoseInferencer(
            pose2d=str(POSE_2D_CONFIG),
            pose2d_weights=str(POSE_2D_CHECKPOINT),
            device=device
        )
        
        action_model = init_recognizer(
            str(ACTION_REC_CONFIG),
            str(ACTION_REC_CHECKPOINT),
            device=device
        )
        
        st.success("Models loaded successfully.")
        return pose_inferencer, action_model

    except Exception as e:
        st.error(f"Model loading failed.: {e}")
        return None, None

pose_inferencer, action_model = load_models()

# for thread safety
class VideoProcessor:
    def __init__(self, pose_inferencer, action_model, label_map):
        self.lock = threading.Lock()
        self.pose_inferencer = pose_inferencer
        self.action_model = action_model
        self.label_map = label_map
        
        # Custom frame buffer
        self.keypoints_buffer = deque(maxlen=40)
        self.scores_history = deque(maxlen=5)
        
        self.latest_action = "N/A"
        self.latest_score = 0.0
        
        self.is_logging = False
        self.logs = []
        self.first_timestamp = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # <--- MODIFICATION: Start total frame timer
        start_time = time.time()
        with self.lock:
            logging_enabled = self.is_logging
            
        try:
            img = frame.to_ndarray(format="bgr24")

            # <--- MODIFICATION: Time the pose inference
            pose_start = time.time()
            results_generator = self.pose_inferencer(img, show=False, return_vis=True)
            result = next(results_generator)
            pose_time_ms = (time.time() - pose_start) * 1000  # Pose time in milliseconds
            
            # ... (keypoints extraction) ...
            if result['predictions'] and result['predictions'][0]:
                person_data = result['predictions'][0][0]
                keypoints = np.array(person_data['keypoints'], dtype=np.float32)
                keypoint_scores = np.array(person_data['keypoint_scores'], dtype=np.float32)
            else:
                keypoints = np.zeros((17, 2), dtype=np.float32)
                keypoint_scores = np.zeros(17, dtype=np.float32)

            self.keypoints_buffer.append({
                "keypoints": keypoints,
                "keypoint_scores": keypoint_scores
            })
            
            # <--- MODIFICATION: Initialize metrics for this frame
            action_time_ms = 0.0
            new_prediction = None
            new_confidence = 0.0
            
            action_to_update, score_to_update = None, None
            if len(self.keypoints_buffer) == self.keypoints_buffer.maxlen:
                keypoints_list = [item['keypoints'] for item in self.keypoints_buffer]
                keypoint_scores_list = [item['keypoint_scores'] for item in self.keypoints_buffer]
                
                anno = {
                    'keypoint': np.array(keypoints_list)[np.newaxis, ...],
                    'keypoint_score': np.array(keypoint_scores_list)[np.newaxis, ...],
                    'total_frames': len(keypoints_list),
                    'img_shape': img.shape[:2]
                }
                
                # <--- MODIFICATION: Time the action recognition
                action_start = time.time()
                action_result = inference_recognizer(self.action_model, anno)
                action_time_ms = (time.time() - action_start) * 1000  # Action time in milliseconds
                
                current_scores = action_result.pred_score.cpu().numpy()
                self.scores_history.append(current_scores)
                
                top_idx = -1
                top_score = 0.0
                
                if self.scores_history:
                    avg_scores = np.mean(self.scores_history, axis=0)
                    top_idx = avg_scores.argmax()
                    top_score = avg_scores[top_idx]
                
                CONFIDENCE_THRESHOLD = 0.5
                if top_score > CONFIDENCE_THRESHOLD:
                    action_to_update = self.label_map.get(top_idx, f"Unknown ({top_idx})")
                    # Use the averaged score
                    score_to_update = top_score
                else:
                    action_to_update = "Idle"
                    # Still log the score even if it's below threshold
                    score_to_update = top_score

                # <--- MODIFICATION: Store new prediction for logging
                new_prediction = action_to_update
                new_confidence = score_to_update

            # ... (your existing 'with self.lock' section) ...
            with self.lock:
                if action_to_update is not None:
                    self.latest_action = action_to_update
                    self.latest_score = score_to_update
                
                action_for_drawing = self.latest_action
                score_for_drawing = self.latest_score

            vis_img = result.get('visualization', [img])[0]
            
            # ... (your existing cv2.putText drawing code) ...
            cv2.putText(vis_img, f"Action: {action_for_drawing}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(vis_img, f"Action: {action_for_drawing}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(vis_img, f"Score: {score_for_drawing:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(vis_img, f"Score: {score_for_drawing:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            
            # <--- MODIFICATION: Calculate total time and append to log
            total_time_ms = (time.time() - start_time) * 1000
            
            if logging_enabled:
                log_entry = {
                    "timestamp_s": time.time() - self.first_timestamp,
                    "total_frame_time_ms": total_time_ms,
                    "pose_time_ms": pose_time_ms,
                    "action_time_ms": action_time_ms,  # Will be 0 if action didn't run
                    # "new_prediction": new_prediction,    # Will be None if action didn't run
                    # "new_confidence": new_confidence,  # Will be 0 if action didn't run
                    "displayed_prediction": action_for_drawing,
                    "displayed_confidence": score_for_drawing
                }

            with self.lock:
                self.logs.append(log_entry)
            
            return av.VideoFrame.from_ndarray(vis_img, format="rgb24")
        
        except Exception as e:
            # <--- MODIFICATION: Log errors as well
            if logging_enabled:
                 log_entry = {
                    "timestamp_ms": time.time() * 1000,
                    "error": str(e)
                 }
                 with self.lock:
                     self.logs.append(log_entry)

            error_img = frame.to_ndarray(format="bgr24")
            cv2.putText(error_img, "ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(error_img, format="rgb24")

if pose_inferencer is None or action_model is None:
    st.warning("No models loaded.")
else:
    if 'processor' not in st.session_state:
        st.session_state.processor = VideoProcessor(pose_inferencer, action_model, label_map)

    processor = st.session_state.processor

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        webrtc_streamer(
            key="camera-input-full-model",
            video_frame_callback=processor.recv,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # <--- MODIFICATION: Add logging controls and download button ---
    st.header("Logging & Metrics")
    
    with processor.lock:
        current_logging_status = processor.is_logging
        log_count = len(processor.logs)
    
    status_text = "Logging is ON" if processor.is_logging else "Logging is OFF"
    st.info(status_text)
    
    l_col1, l_col2, l_col3 = st.columns(3)
    
    if l_col1.button("Start Logging"):
        with processor.lock:
            processor.is_logging = True
        st.rerun()

    if l_col2.button("Stop Logging"):
        with processor.lock:
            processor.is_logging = False
        st.rerun()

    if l_col3.button("Clear Logs"):
        with processor.lock:
            processor.logs = []
        st.rerun()


    st.metric("Total Frames Logged", log_count)

    if log_count > 0:
        try:
            # Safely get a *copy* of the logs for processing
            with processor.lock:
                logs_copy = list(processor.logs)

            # Convert the copy to a DataFrame
            df = pd.DataFrame(logs_copy)
            
            # Show a preview
            st.subheader("Log Preview (Last 5 Frames)")
            st.dataframe(df.tail())
            
            # Create download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Logs as CSV",
                data=csv,
                file_name="fias_demo_logs.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Failed to process logs for display: {e}")
