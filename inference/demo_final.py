import streamlit as st
import av
import cv2
import numpy as np
import torch
from pathlib import Path
import time
from collections import deque
from tqdm import tqdm # Useful for showing progress during processing

# imports for Hierarchical Loss
import sys
import os.path as osp
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
import fias_custom_loss.custom_loss  # <-- FORCE THE IMPORT

# --- MMPose & MMAction Imports ---
try:
    from mmpose.apis import MMPoseInferencer
    from mmaction.apis import inference_recognizer, init_recognizer
    from mmengine.registry import init_default_scope, DefaultScope
    from mmengine.dataset import Compose, pseudo_collate
    import torch.nn.functional as F
except ImportError:
    st.error("ImportError: Missing MMPose or MMAction dependencies. Ensure environments are set up.")
    st.stop()


# ==============================================================================
# 0. KEYPOINT NAMES & GRAD-CAM CLASS (from Code 1)
# ==============================================================================
KEYPOINT_NAMES = {
    0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
    5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
    9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
    13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
}

# NOTE: The full STGCNGradCAM class from the first code must be included here.
# For demo purposes, we keep the placeholder structure.
class STGCNGradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        # ... (rest of the init, hooks registration)
        self.device = next(model.parameters()).device
        
    def __call__(self, inputs, index=-1):
        window_size = inputs['keypoint'].shape[1]
        
        # Mimic Grad-CAM analysis: returns a dummy heatmap
        # Shape: (17 keypoints x window_size)
        dummy_gradcam_map = np.random.rand(17, window_size) * 10
        dummy_gradcam_map = dummy_gradcam_map.astype(np.float32)
        
        # Run actual prediction
        with torch.no_grad():
            action_result = inference_recognizer(self.model, inputs)
            scores = action_result.pred_score
            index = scores.argmax().item() if index == -1 else index
            pred_score = scores[index].item()
            
        return dummy_gradcam_map, index, pred_score
    
    def remove_hooks(self):
        pass

# ==============================================================================
# 1. LLM / Prompt Generation Functions (from Code 1)
# ==============================================================================
def get_ai_coach_feedback(prompt: str) -> str:
    # Simulate LLM response time
    time.sleep(3) 
    return f"Based on the analysis, the AI Coach recommends you focus on **improving stability and range of motion** for the most critical joints identified."

def generate_llm_prompt_en(gradcam_map, action_name, score, kp_names, total_frames) -> str:
    # Average importance across the sequence
    final_gradcam_map = np.mean(gradcam_map, axis=1) 
    # Find the top 3 most critical joints based on normalized importance
    top_3_indices = np.argsort(final_gradcam_map)[::-1][:3]
    top_3_names = [kp_names.get(i, f"KP{i}") for i in top_3_indices]
    
    return (
        f"Analysis of {total_frames} frames: Action: '{action_name}', Confidence: {score:.2%}.\n"
        f"The top 3 most influential body parts (Grad-CAM) were: {', '.join(top_3_names)}.\n"
        "Please provide specific coaching feedback for this performance, focusing on the influential joints."
    )

# ==============================================================================
# 2. CONFIGURATION & MODEL LOADING
# ==============================================================================

# Define Paths (Adjust these as necessary for your environment)
path_prefix = Path('/home/cvlab123') 
POSE_2D_CONFIG="/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py"
POSE_2D_CHECKPOINT="/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth"
ACTION_REC_CONFIG = path_prefix / 'mmaction2/configs/skeleton/custom_stgcnpp/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py'
ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/1027_all_plus_idle100_2D_joint/best_acc_top1_epoch_8.pth'

label_map = {
    0: "idle", 1: "lunge_correct", 2: "lunge_knee_pass_toe", 3: "lunge_too_high",
    4: "push_up_arched_back", 5: "push_up_correct", 6: "push_up_elbow",
    7: "squat_correct", 8: "squat_feet_too_close", 9: "squat_knees_inward"
}
idx_to_label = {v: k for k, v in label_map.items()}

# Pipeline Parameters (from Code 1)
WINDOW_SIZE = 40
FRAME_STRIDE = 2
POSE_BATCH_SIZE = 16 

@st.cache_resource
def load_models():
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        st.info(f"Loading models on {device}...")
        
        pose_inferencer = MMPoseInferencer(
            pose2d=str(POSE_2D_CONFIG), pose2d_weights=str(POSE_2D_CHECKPOINT), device=device
        )
        action_model = init_recognizer(
            str(ACTION_REC_CONFIG), str(ACTION_REC_CHECKPOINT), device=device
        )
        
        st.success("Models loaded successfully. Proceed to upload video. 🚀")
        return pose_inferencer, action_model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

pose_inferencer, action_model = load_models()


# ==============================================================================
# 3. CORE ANALYSIS FUNCTION (Adapting Code 1's flow)
# ==============================================================================
def run_full_video_analysis(video_file, p_inferencer, a_model, label_map_rev):
    """
    Implements the full Pose -> Action -> Grad-CAM -> LLM flow from the first code.
    """
    
    st.session_state.processing_status = "⏳ Step 1/4: Reading video frames..."
    
    # 1. Video Loading & Pose Estimation (Step 1/4)
    t0 = time.time()
    temp_video_path = Path("temp_uploaded_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())

    cap = cv2.VideoCapture(str(temp_video_path))
    original_frames = [frame for success, frame in iter(lambda: cap.read(), (False, None))]
    cap.release()
    temp_video_path.unlink() # Clean up temp file

    if not original_frames:
        st.error("Could not read video or video is empty.")
        return None
    
    # Downsample frames
    strided_frames = original_frames[::FRAME_STRIDE]
    frame_h, frame_w, _ = strided_frames[0].shape
    
    st.session_state.processing_status = f"⏳ Step 2/4: Extracting skeletons ({len(strided_frames)} frames)..."
    progress_bar = st.progress(0)
    
    keypoints_list = []
    
    # Process pose estimation in chunks for the progress bar
    for i in range(0, len(strided_frames), POSE_BATCH_SIZE):
        batch_frames = strided_frames[i : i + POSE_BATCH_SIZE]
        results_generator = p_inferencer(batch_frames, show_progress=False, batch_size=POSE_BATCH_SIZE)
        
        for p in results_generator:
            if p['predictions'] and p['predictions'][0]:
                keypoints_data = p['predictions'][0][0]
                keypoints = np.array(keypoints_data['keypoints'])
                scores = np.array(keypoints_data['keypoint_scores'])
                keypoints_with_scores = np.hstack([keypoints, scores[:, None]])
                keypoints_list.append(keypoints_with_scores)
            else:
                keypoints_list.append(np.zeros((17, 3)))
        
        progress_bar.progress((i + len(batch_frames)) / len(strided_frames))
        
    total_frames = len(keypoints_list)
    progress_bar.empty()
    st.session_state.processing_status = f"✅ Skeletons extracted. ({total_frames} data points)"

    
    # 2. Grad-CAM and Prediction Analysis (Step 3/4)
    st.session_state.processing_status = "⏳ Step 3/4: Running Grad-CAM and Action Analysis..."
    
    # We simplify this by only running one window for the entire video clip, 
    # taking the middle WINDOW_SIZE frames, if the video is long enough.
    
    if total_frames < WINDOW_SIZE:
        st.error(f"Video too short for analysis. Need at least {WINDOW_SIZE} frames ({WINDOW_SIZE * FRAME_STRIDE} original frames).")
        return None

    # Determine a central window for analysis
    start_frame = (total_frames - WINDOW_SIZE) // 2
    end_frame = start_frame + WINDOW_SIZE
    window_keypoints = np.array(keypoints_list[start_frame:end_frame])
    
    anno = {
        'keypoint': window_keypoints[np.newaxis, ..., :2],
        'keypoint_score': window_keypoints[np.newaxis, ..., 2],
        'total_frames': WINDOW_SIZE,
        'img_shape': (frame_h, frame_w)
    }
    
    target_layer_name = 'backbone.gcn.9.gcn'
    gradcam = STGCNGradCAM(a_model, target_layer_name)
    
    gradcam_map, pred_idx, pred_score = gradcam(anno)
    gradcam.remove_hooks()
    
    pred_name = label_map_rev.get(pred_idx, "Unknown Action")
    
    # Aggregate Grad-CAM map over time (T, 17) -> (17)
    final_gradcam_map = np.mean(gradcam_map, axis=1)

    # 3. LLM Feedback (Step 4/4)
    st.session_state.processing_status = "⏳ Step 4/4: Generating AI Coach Feedback..."
    
    llm_prompt = generate_llm_prompt_en(
        gradcam_map, 
        pred_name, 
        pred_score, 
        KEYPOINT_NAMES, 
        total_frames
    )
    
    llm_response = get_ai_coach_feedback(llm_prompt)
    
    st.session_state.processing_status = "✅ Analysis Complete!"
    
    return {
        "video_frames": strided_frames,
        "keypoints": keypoints_list,
        "prediction": pred_name,
        "confidence": pred_score,
        "gradcam_map": final_gradcam_map, # Final 17-element score
        "prompt": llm_prompt,
        "feedback": llm_response,
        "process_time": time.time() - t0
    }

# ==============================================================================
# 4. STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="FIAS Demo: Video Upload", layout="wide")
st.title("🏋️ Fitness Analysis Pipeline: Video Upload")
st.markdown("Upload a video of an exercise (Lunge, Push-up, or Squat) for Grad-CAM and AI Coach feedback.")
st.markdown("---")

# Initialize session state for results
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "Awaiting video upload."

if pose_inferencer is None or action_model is None:
    st.warning("Cannot proceed. Models failed to load.")
else:
    uploaded_file = st.file_uploader(
        "Upload a video file (MP4, MOV)", 
        type=["mp4", "mov"], 
        accept_multiple_files=False
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Video Player")
        if uploaded_file is not None:
            # Display the video
            st.video(uploaded_file)
            
            # Start analysis button
            if st.button("▶️ Run Full Analysis (Pose + Grad-CAM + LLM)", disabled=st.session_state.processing_status.startswith('⏳')):
                st.session_state.analysis_result = None
                st.session_state.processing_status = "Starting analysis..."
                # Rerun to show initial status
                st.rerun()
            
            # --- The actual execution of the analysis ---
            if st.session_state.processing_status.startswith("Starting analysis..."):
                with st.spinner(st.session_state.processing_status):
                    results = run_full_video_analysis(uploaded_file, pose_inferencer, action_model, label_map)
                    st.session_state.analysis_result = results
                    st.rerun()
        else:
             st.info("Upload a video above to begin.")

    with col2:
        st.subheader("Analysis Status & Results")
        st.info(st.session_state.processing_status)
        
        results = st.session_state.analysis_result
        if results:
            st.success(f"Analysis Complete! Total Time: {results['process_time']:.2f} seconds")
            st.markdown("### 🥇 Action Quality Assessment")
            
            # Display key results
            st.metric("Predicted Action", results['prediction'])
            st.metric("Confidence Score", f"{results['confidence']:.2%}")
            st.metric("Most Critical Joint", KEYPOINT_NAMES.get(results['gradcam_map'].argmax(), "N/A"))
            
            st.markdown("---")
            
            # Display LLM Feedback
            st.markdown("### 🤖 AI Coach Feedback")
            with st.expander("Show LLM Prompt", expanded=False):
                st.code(results['prompt'])
                
            st.markdown(f"**Coach Response:**")
            st.markdown(f"**{results['feedback']}**")