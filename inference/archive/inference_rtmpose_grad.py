import torch
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from mmpose.apis import MMPoseInferencer
from mmaction.apis import init_recognizer
from mmaction.structures import ActionDataSample
from mmaction.datasets import transforms # 確保註冊
from mmengine.registry import DefaultScope
from mmengine.registry import init_default_scope

import torch.nn.functional as F

# ==============================================================================
# 1. GradCAM 類別 (最終修正版)
# ==============================================================================
class STGCNGradCAM:
    """
    完全獨立的 Grad-CAM 類別，專為 STGCN 類型的模型設計。
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.feature_maps = {}
        self.gradients = {}
        self.handlers = []
        self._register_hooks()

    def _get_target_layer(self):
        module = self.model
        for name in self.target_layer_name.split('.'):
            if name.isdigit():
                module = module[int(name)]
            else:
                module = getattr(module, name)
        return module

    def _register_hooks(self):
        target_layer = self._get_target_layer()
        def forward_hook(module, input, output):
            self.feature_maps[target_layer] = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients[target_layer] = grad_out[0].detach()
        self.handlers.append(target_layer.register_forward_hook(forward_hook))
        self.handlers.append(target_layer.register_backward_hook(backward_hook))

    def _calculate_localization_map(self, feature_maps, grads):
        N, C, T, V = feature_maps[0].shape
        weights = torch.mean(grads[0], dim=(2, 3), keepdim=True)
        cam = torch.sum(feature_maps[0] * weights, dim=1)

        # === 深入診斷程式碼 START ===
        print("\n--- Grad-CAM 內部數值診斷 ---")
        fm = feature_maps[0]
        gr = grads[0]
        print(f"特徵圖 (Activations) -> Shape: {fm.shape}, Max: {fm.max():.6f}, Min: {fm.min():.6f}, Mean: {fm.mean():.6f}, Std: {fm.std():.6f}")
        print(f"梯度 (Gradients)    -> Shape: {gr.shape}, Max: {gr.max():.6f}, Min: {gr.min():.6f}, Mean: {gr.mean():.6f}, Std: {gr.std():.6f}")
        print(f"權重 (Weights)      -> Shape: {weights.shape}, Max: {weights.max():.6f}, Min: {weights.min():.6f}, Mean: {weights.mean():.6f}")
        print(f"CAM (最終結果)      -> Shape: {cam.shape}, Max: {cam.max():.6f}, Min: {cam.min():.6f}, Mean: {cam.mean():.6f}")
        print("---------------------------------\n")
        # === 深入診斷程式碼 END ===
        
        # 保持註解，以觀察完整 CAM
        # cam = F.relu(cam) 
        
        return cam.detach().cpu().numpy()

    def __call__(self, inputs, index=-1):
            self.model.zero_grad()

            # 1. 手動建立一個 5D 張量 (N, M, T, V, C)。
            # 2. 再增加一個假的維度，變成 6D，以應對 RecognizerGCN 會移除第一維的問題。
            # 3. 將這個 "特製" 的 6D 張量直接傳遞給模型。
            
            # `inputs` 是我們在 main() 中建立的 anno 字典
            keypoint_tensor_4d = torch.from_numpy(inputs['keypoint']).float()
            
            # 建立 5D 張量 (N=1, M=1, T, V, C)
            input_tensor_5d = keypoint_tensor_4d.unsqueeze(0)
            # 再增加一個維度，變為 6D (1, N, M, T, V, C)
            device = next(self.model.parameters()).device
            input_tensor_6d = input_tensor_5d.unsqueeze(0).to(device)

            # DataSample 的處理不變
            data_sample = ActionDataSample()
            data_samples = [data_sample.to(input_tensor_6d.device)]
            
            # 前向傳播
            scores = self.model(input_tensor_6d, data_samples, mode='predict')[0].pred_score
            
            if index == -1:
                index = scores.argmax().item()
            score = scores[index].item()

            # 反向傳播
            scores[index].backward()

            feature_maps = [self.feature_maps[self._get_target_layer()]]
            grads = [self.gradients[self._get_target_layer()]]
            localization_map = self._calculate_localization_map(feature_maps, grads)
            
            return localization_map[0], index, score

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()

# ==============================================================================
# 2. 視覺化輔助函式
# ==============================================================================
coco_skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                 [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                 [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                 [3, 5], [4, 6]]

def visualize_gradcam_on_frame(frame, keypoints, gradcam_scores, skeleton_conn, frame_size):
    """
    在單一影格上繪製骨架及 Grad-CAM 熱圖。
    """
    fig, ax = plt.subplots(figsize=(frame_size[0] / 100, frame_size[1] / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # 移除白邊
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # [修正 1] 使用新版 API: plt.get_cmap()
    cmap = plt.get_cmap('jet')
    colors = cmap(gradcam_scores)[:, :3]

    ax.scatter(keypoints[:, 0], keypoints[:, 1], c=colors, s=80, zorder=2)
    
    lines, line_colors = [], []
    for p1_idx, p2_idx in skeleton_conn:
        if keypoints[p1_idx, 0] > 0 and keypoints[p2_idx, 0] > 0:
            lines.append([keypoints[p1_idx], keypoints[p2_idx]])
            line_colors.append((colors[p1_idx] + colors[p2_idx]) / 2)
    
    lc = LineCollection(lines, colors=line_colors, linewidths=3, zorder=1)
    ax.add_collection(lc)

    # [修正 2] 使用新版 API 將 figure 畫布轉換為 numpy 陣列
    fig.canvas.draw()
    img_rgba = np.array(fig.canvas.renderer.buffer_rgba())
    vis_frame = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    
    plt.close(fig)
    return vis_frame

# ==============================================================================
# 3. 主函式
# ==============================================================================
def main():
    # 強制設定 mmaction 為預設 scope
    init_default_scope('mmaction')

    # --- 3.1 模型設定 ---
    ACTION_REC_CONFIG = '/home/cvlab123/mmaction2/configs/skeleton/custom_stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_OurDataset-xsub-keypoint-2d.py'
    ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/rtmpose_all_class_ourdataset_joint_motion/best_acc_top1_epoch_11.pth'
    
    POSE_CONFIG = '/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py'
    POSE_CHECKPOINT = '/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'

    VIDEO_PATH = '/home/cvlab123/data/test_data/20250627_172807.mp4'
    OUTPUT_VIDEO_PATH = 'gradcam_output_video.mp4'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # --- 3.2 載入模型 ---
    print("正在載入模型...")
    action_model = init_recognizer(ACTION_REC_CONFIG, ACTION_REC_CHECKPOINT, device=device)
    pose_inferencer = MMPoseInferencer(pose2d=POSE_CONFIG, pose2d_weights=POSE_CHECKPOINT, device=device)
    print("模型載入完成！")

    label_map = {
        "lunge_correct": 0, "lunge_knee_pass_toe": 1, "lunge_too_high": 2,
        "push_up_arched_back": 3, "push_up_correct": 4, "push_up_elbow": 5,
        "squat_correct": 6, "squat_feet_too_close": 7, "squat_knees_inward": 8
    }
    idx_to_label = {v: k for k, v in label_map.items()}

    # --- 3.3 提取骨架關鍵點 ---
    print(f"正在從影片 {VIDEO_PATH} 提取骨架...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        video_frames.append(frame)
    cap.release()
    
    if not video_frames:
        print("錯誤：無法讀取影片或影片為空。")
        return

    frame_h, frame_w, _ = video_frames[0].shape
    
    results_generator = pose_inferencer(video_frames, show_progress=True)
    
    keypoints_list = []
    for frame_results in results_generator:
        predictions = frame_results['predictions'][0]
        if predictions:
            person_data = predictions[0]
            keypoints = person_data['keypoints']
            scores = person_data['keypoint_scores']
            keypoints_with_scores = np.hstack([keypoints, np.array(scores)[:, None]])
            keypoints_list.append(keypoints_with_scores)
        else:
            keypoints_list.append(np.zeros((17, 3)))
    
    keypoints_array = np.array(keypoints_list)
    print(f"骨架提取完成，共 {keypoints_array.shape[0]} 幀。")

    # --- 3.4 準備輸入資料 ---
    anno = {
        'keypoint': keypoints_array[np.newaxis, ...], # (1, T, V, 3) -> (M, T, V, C)
        'total_frames': keypoints_array.shape[0],
        'img_shape': (frame_h, frame_w)
    }

    # --- 3.5 初始化並執行 Grad-CAM ---
    print("正在計算 Grad-CAM...")
    target_layer_name = 'backbone.gcn.8'
    gradcam = STGCNGradCAM(action_model, target_layer_name)
    
    gradcam_map, pred_class_idx, pred_score = gradcam(anno) 
    pred_class_name = idx_to_label.get(pred_class_idx, "Unknown")
    
    print(f"模型預測結果: {pred_class_name} (分數: {pred_score:.4f})")
    print(f"正在為類別 '{pred_class_name}' 生成 Grad-CAM...")
    
    # --- 3.6 視覺化並儲存影片 ---
    print("正在生成視覺化影片...")
    gradcam_scores_normalized = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-6)

    # [偵錯步驟] 檢查正規化後的數值分佈
    print(f"正規化後 Grad-CAM -> Shape: {gradcam_scores_normalized.shape}, Max: {gradcam_scores_normalized.max():.6f}, Min: {gradcam_scores_normalized.min():.6f}, Mean: {gradcam_scores_normalized.mean():.6f}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 25, (frame_w, frame_h))

    for frame_idx, frame in enumerate(video_frames):
        if frame_idx >= len(keypoints_list): break
        
        kps = keypoints_list[frame_idx][:, :2]
        # 確保 gradcam_scores_normalized 的 frame_idx 不會超出範圍
        if frame_idx < gradcam_scores_normalized.shape[0]:
            scores = gradcam_scores_normalized[frame_idx]
            vis_frame_mpl = visualize_gradcam_on_frame(frame, kps, scores, coco_skeleton, (frame_w, frame_h))
            vis_frame_bgr = cv2.cvtColor(vis_frame_mpl, cv2.COLOR_RGB2BGR)
        else: # 如果 CAM map 比影片短，則後面的幀用原圖
            vis_frame_bgr = frame

        text = f'Pred: {pred_class_name} ({pred_score:.2f})'
        cv2.putText(vis_frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(vis_frame_bgr)
        
    video_writer.release()
    gradcam.remove_hooks() # 釋放 hook
    print(f"✅ Grad-CAM 視覺化影片已儲存至: {OUTPUT_VIDEO_PATH}")

if __name__ == '__main__':
    main()