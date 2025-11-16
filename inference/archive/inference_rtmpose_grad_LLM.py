import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import time
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
from mmpose.apis import MMPoseInferencer
from mmaction.apis import init_recognizer
from mmengine.registry import init_default_scope
from tqdm import tqdm
from module.llm import get_ai_coach_feedback
from module.generate_prompt import generate_llm_prompt_en

# ==============================================================================
# 0. 關節點名稱定義
# ==============================================================================
KEYPOINT_NAMES = {
    0: "鼻子 (Nose)", 1: "左眼 (Left Eye)", 2: "右眼 (Right Eye)", 3: "左耳 (Left Ear)", 4: "右耳 (Right Ear)",
    5: "左肩 (Left Shoulder)", 6: "右肩 (Right Shoulder)", 7: "左肘 (Left Elbow)", 8: "右肘 (Right Elbow)",
    9: "左腕 (Left Wrist)", 10: "右腕 (Right Wrist)", 11: "左髖 (Left Hip)", 12: "右髖 (Right Hip)",
    13: "左膝 (Left Knee)", 14: "右膝 (Right Knee)", 15: "左踝 (Left Ankle)", 16: "右踝 (Right Ankle)"
}

# ==============================================================================
# 1. GradCAM 核心類別 (已驗證)
# ==============================================================================
class STGCNGradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        try:
            self.target_layer = self._get_target_layer()
        except Exception as e:
            print(f"錯誤: 無法找到目標層 '{target_layer_name}'。請檢查模型架構。")
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
            print(f"\n警告: 目標層 '{self.target_layer_name}' 的梯度為零，無法生成熱圖。")
            return np.zeros(feature_maps.shape[2:], dtype=np.float32)

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(feature_maps * weights, dim=1)
        cam = F.relu(cam)
        return cam.cpu().numpy()

    def __call__(self, inputs, index=-1):
        self.model.zero_grad()
        
        from mmengine.dataset import Compose, pseudo_collate
        from mmengine.registry import DefaultScope

        cfg = self.model.cfg
        test_pipeline_cfg = cfg.get('test_pipeline', cfg.get('val_pipeline'))

        with DefaultScope.overwrite_default_scope('mmaction'):
            test_pipeline = Compose(test_pipeline_cfg)
        
        data = test_pipeline(inputs.copy())
        data = pseudo_collate([data])
        
        with torch.no_grad():
            results_for_pred = self.model.test_step(data)[0]
        
        scores = results_for_pred.pred_score
        if index == -1:
            index = scores.argmax().item()
        pred_score = scores[index].item()

        results_for_grad = self.model.test_step(data)[0]
        score_for_backward = results_for_grad.pred_score[index]
        
        score_for_backward.backward()
        
        feature_maps = self.feature_maps[self.target_layer]
        grads = self.gradients[self.target_layer]
        localization_map = self._calculate_localization_map(feature_maps, grads)
        
        final_map = localization_map[0] if len(localization_map.shape) > 2 else localization_map
        
        return final_map, index, pred_score

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()

# ==============================================================================
# 2. 視覺化輔助函式 (已驗證)
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
    colors = np.array([[0.2, 0.2, 1.]] * len(keypoints))
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
    img_rgb = img_rgba[:, :, :3]
    
    plt.close(fig)
    return img_rgb

# ==============================================================================
# 3. 主函式 (最終修正版)
# ==============================================================================
def main():
    total_start_time = time.time()

    init_default_scope('mmaction')
    os.makedirs("gradcam_video_output/", exist_ok=True)
    os.makedirs("gradcam_text_output/", exist_ok=True)
    # ACTION_REC_CONFIG = '/home/cvlab123/mmaction2/configs/skeleton/custom_stgcnpp_test/stgcnpp_8xb16-bone-motion-u100-80e_OurDataset-xsub-keypoint-2d.py'
    # # ACTION_REC_CONFIG = '/home/cvlab123/mmaction2/configs/skeleton/custom_stgcnpp/stgcnpp_8xb16-bone-u100-80e_OurDataset-xsub-keypoint-2d.py'
    # # ACTION_REC_CONFIG = '/home/cvlab123/mmaction2/configs/skeleton/custom_stgcnpp_test/stgcnpp_8xb16-joint-motion-u100-80e_OurDataset-xsub-keypoint-2d.py'
    # # ACTION_REC_CONFIG = '/home/cvlab123/mmaction2/configs/skeleton/custom_stgcnpp_test/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py'
    
    
    # ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/916_rtmpose_all_2D_bone_motion/best_acc_top1_epoch_16.pth'
    # # ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/916_rtmpose_all_2D_bone/best_acc_top1_epoch_9.pth'
    # # ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/916_rtmpose_all_2D_joint_motion/best_acc_top1_epoch_10.pth'
    # # ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/916_rtmpose_all_2D_joint/best_acc_top1_epoch_13.pth'
    # # --- Aligning with demo.py ---
    # Pose Estimation Model (RTMPose-S)
    POSE_CONFIG = '/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py'
    POSE_CHECKPOINT = '/home/cvlab123/mmpose/checkpoints/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'

    # # POSE_CONFIG = '/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py'
    # POSE_CONFIG = '/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py'
    # # POSE_CHECKPOINT = '/home/cvlab123/mmpose/checkpoints/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'
    # POSE_CHECKPOINT = '/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'
    # Action Recognition Model (STGCN++ Joint)
    ACTION_REC_CONFIG = '/home/cvlab123/mmaction2/configs/skeleton/custom_stgcnpp/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py'
    ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/916_rtmpose_all_2D_joint/best_acc_top1_epoch_13.pth'
    # --- End of Alignment ---
    
    VIDEO_PATH = '/home/cvlab123/data/test_data/20250627_172807.mp4'
    
    video_stem = Path(VIDEO_PATH).stem
    OUTPUT_VIDEO_DIR = Path("gradcam_video_output")
    OUTPUT_VIDEO_PATH = OUTPUT_VIDEO_DIR / f"{video_stem}_visualize.mp4"
    OUTPUT_TEXT_DIR = Path("gradcam_text_output")
    OUTPUT_TEXT_PATH = OUTPUT_TEXT_DIR / f"{video_stem}_llm_text.txt"
    
    WINDOW_SIZE = 40
    STRIDE = 1
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    print("正在載入模型...")
    action_model = init_recognizer(ACTION_REC_CONFIG, ACTION_REC_CHECKPOINT, device=device)
    pose_inferencer = MMPoseInferencer(pose2d=POSE_CONFIG, pose2d_weights=POSE_CHECKPOINT, device=device)
    print("模型載入完成！")
    print(f"--- ⏱️  模型載入耗時: {time.time() - start_time:.2f} 秒 ---\n")
    
    # [FIX] Use a list for labels to guarantee order. The index is the class ID.
    idx_to_label = [
        "lunge_correct", "lunge_knee_pass_toe", "lunge_too_high",
        "push_up_arched_back", "push_up_correct", "push_up_elbow",
        "squat_correct", "squat_feet_too_close", "squat_knees_inward"
    ]

    start_time = time.time()
    print(f"正在從影片 {VIDEO_PATH} 提取骨架...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_frames = [frame for success, frame in iter(lambda: cap.read(), (False, None))]
    cap.release()
    
    # [RECOMMENDED FIX] Add this line to match your training data sampling
    print(f"Original frame count: {len(video_frames)}. Downsampling by skipping every 2nd frame...")
    video_frames = video_frames[::2]
    print(f"New frame count after downsampling: {len(video_frames)}.")
    
    if not video_frames:
        print(f"錯誤：無法讀取影片 '{VIDEO_PATH}' 或影片為空。")
        return

    frame_h, frame_w, _ = video_frames[0].shape
    results_generator = pose_inferencer(video_frames, show_progress=True, batch_size=16)
    
    keypoints_list = [
        np.hstack([p['predictions'][0][0]['keypoints'], np.array(p['predictions'][0][0]['keypoint_scores'])[:, None]])
        if p['predictions'] and p['predictions'][0] else np.zeros((17, 3))
        for p in results_generator
    ]
    total_frames = len(keypoints_list)
    print(f"骨架提取完成，共 {total_frames} 幀。")    
    print(f"--- ⏱️  影片讀取與骨架提取耗時: {time.time() - start_time:.2f} 秒 ---\n")

    start_time = time.time()
    print("正在使用滑動窗口計算 Grad-CAM...")
    target_layer_name = 'backbone.gcn.9.gcn'
    gradcam = STGCNGradCAM(action_model, target_layer_name)
    
    full_gradcam_map = np.zeros((total_frames, 17))
    overlap_counter = np.zeros((total_frames, 17))
    
    overall_pred_name = "N/A"
    overall_pred_score = 0.0

    if total_frames < WINDOW_SIZE:
        print(f"錯誤: 影片總幀數 ({total_frames}) 小於窗口大小 ({WINDOW_SIZE})。無法處理。")
        return
    
    # 紀錄每個 class 的 score
    class_score_sum = {}
    class_pred_count = {}

    for i in tqdm(range(0, total_frames - WINDOW_SIZE + 1, STRIDE), desc="滑動窗口進度"):
        start_frame = i
        end_frame = i + WINDOW_SIZE
        window_keypoints = np.array(keypoints_list[start_frame:end_frame])
        
        # [FIX] Split keypoints and scores to match the training pipeline format
        keypoints_for_anno = window_keypoints[..., :2]  # Shape: (WINDOW_SIZE, 17, 2)
        scores_for_anno = window_keypoints[..., 2]    # Shape: (WINDOW_SIZE, 17)

        anno = {
            'keypoint': keypoints_for_anno[np.newaxis, ...],
            'keypoint_score': scores_for_anno[np.newaxis, ...],
            'total_frames': WINDOW_SIZE,
            'img_shape': (frame_h, frame_w)
        }
        try:
            gradcam_map, pred_idx, pred_score = gradcam(anno)
            
            # [最終修正] 將 gradcam_map (例如 shape 10,17) 放大回窗口大小 (48,17)
            # cv2.resize 的參數 dsize 是 (寬, 高)，對應我們的維度是 (V, T)
            target_shape = (17, WINDOW_SIZE) # (V, T)
            resized_gradcam_map = cv2.resize(gradcam_map, target_shape, interpolation=cv2.INTER_LINEAR)
            
            # resized_gradcam_map 的 shape 是 (48, 17)，可以直接與切片相加
            full_gradcam_map[start_frame:end_frame] += resized_gradcam_map
            overlap_counter[start_frame:end_frame] += 1

            pred_name = idx_to_label[pred_idx] if 0 <= pred_idx < len(idx_to_label) else "Unknown"
            class_score_sum[pred_name] = class_score_sum.get(pred_name, 0) + pred_score
            class_pred_count[pred_name] = class_pred_count.get(pred_name, 0) + 1

        except Exception as e:
            print(f"處理窗口 {start_frame}-{end_frame} 時發生錯誤: {e}")
            continue

    gradcam.remove_hooks()
    
    # After the loop and gradcam.remove_hooks()
    if class_score_sum:
        # --- Step 1: Calculate the average score for each class ---
        class_avg_scores = {
            name: class_score_sum[name] / class_pred_count[name] 
            for name in class_score_sum
        }
        
        # --- Step 2: Find the class with the highest average score ---
        best_pred_name = max(class_avg_scores, key=class_avg_scores.get)
        best_avg_score = class_avg_scores[best_pred_name]

        print(f"Aggregated average predictions: {class_avg_scores}")
        
        # --- Step 3: Apply the 0.5 threshold ---
        if best_avg_score >= 0.5:
            overall_pred_name = best_pred_name
            # IMPORTANT: Also update the score to be the more meaningful average score
            overall_pred_score = best_avg_score 
            print(f"Final Prediction: '{overall_pred_name}' (Avg Score: {overall_pred_score:.4f})")
        else:
            overall_pred_name = "Not Detected"
            overall_pred_score = best_avg_score # Still useful to know the score
            print(f"Best prediction '{best_pred_name}' score ({best_avg_score:.4f}) is below threshold.")

    else:
        # This handles the case where no predictions were made at all
        overall_pred_name = "Not Detected"
        overall_pred_score = 0.0
        print("No valid predictions were made across any window.")
        
    overlap_counter[overlap_counter == 0] = 1
    final_gradcam_map = full_gradcam_map / overlap_counter
    print(f"模型預測結果 (代表): {overall_pred_name} (分數: {overall_pred_score:.4f})")
    print(f"--- ⏱️  Grad-CAM 計算耗時: {time.time() - start_time:.2f} 秒 ---\n")

    start_time = time.time()
    print("正在生成視覺化影片...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 25, (frame_w, frame_h))

    for frame_idx, frame in enumerate(tqdm(video_frames, desc="生成影片")):
        kps = keypoints_list[frame_idx][:, :2]
        
        frame_scores = final_gradcam_map[frame_idx]
        scores_for_frame = None
        if frame_scores.max() > frame_scores.min():
            scores_for_frame = (frame_scores - frame_scores.min()) / (frame_scores.max() - frame_scores.min() + 1e-6)
        
        vis_frame_rgb = visualize_gradcam_on_frame(frame, kps, scores_for_frame, coco_skeleton, (frame_w, frame_h))
        vis_frame_bgr = cv2.cvtColor(vis_frame_rgb, cv2.COLOR_RGB2BGR)

        text = f'Pred: {overall_pred_name} ({overall_pred_score:.2f})'
        cv2.putText(vis_frame_bgr, text, (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        video_writer.write(vis_frame_bgr)
        
    video_writer.release()
    print(f"✅ Grad-CAM 視覺化影片已儲存至: {OUTPUT_VIDEO_PATH}")
    print(f"--- ⏱️  視覺化影片生成耗時: {time.time() - start_time:.2f} 秒 ---\n")

    # --- [新增] 生成並印出 LLM Prompt ---
    start_time = time.time()
    print("\n" + "="*80)
    print("🤖 正在生成給 LLM 的 Prompt...")
    print("="*80)
    
    llm_prompt = generate_llm_prompt_en(
        final_gradcam_map, 
        overall_pred_name, 
        overall_pred_score, 
        KEYPOINT_NAMES, 
        total_frames
    )
    print(llm_prompt)
    print("="*80)
    print(f"--- ⏱️  LLM Prompt 生成耗時: {time.time() - start_time:.2f} 秒 ---")
    

    print(f"✅ Prompt has been saved to {OUTPUT_TEXT_PATH}")

    start_time = time.time()
    llm_response = get_ai_coach_feedback(llm_prompt)
    print(f"----- AI Coach Feedback: -----")
    print(llm_response)
    print(f"--- ⏱️  LLM 推理耗時: {time.time() - start_time:.2f} 秒 ---\n")

    print("="*80)
    print(f"🎉 全部處理完成！總耗時: {time.time() - total_start_time:.2f} 秒")
    # Save to file
    with open(OUTPUT_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(llm_prompt)
        f.write(llm_response)
    
if __name__ == '__main__':
    # if not all(os.path.exists(f) for f in ['stgcnpp_8xb16-bone-motion-u100-80e_OurDataset-xsub-keypoint-2d.py', 'best_acc_top1_epoch_11.pth']):
    #     print("錯誤: 請確保設定檔和權重檔與本腳本在同一目錄下。")
    #     sys.exit(1)
    main()