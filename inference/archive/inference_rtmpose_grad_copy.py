import torch
import cv2
import numpy as np
import os

import matplotlib
matplotlib.use('Agg') # 使用非 GUI 後端，避免在伺服器上出錯
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from mmpose.apis import MMPoseInferencer
from mmaction.apis import init_recognizer
from mmaction.structures import ActionDataSample
from mmaction.datasets import transforms # 確保註冊
from mmengine.registry import DefaultScope, init_default_scope

import torch.nn.functional as F

# ==============================================================================
# 1. GradCAM 類別 (最終版)
# ==============================================================================
class STGCNGradCAM:
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
            module = module[int(name)] if name.isdigit() else getattr(module, name)
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
        if grads[0] is None or torch.all(grads[0] == 0):
            print("\n警告: 梯度為零，無法生成有意義的熱圖。請嘗試更換目標層 (如 'backbone.gcn.7')。\n")
            return np.zeros(feature_maps[0].shape[2:], dtype=np.float32)

        # 根據特徵圖的維度決定如何處理
        # 處理特徵 (4D): (N*M, C, T, V)
        if len(feature_maps[0].shape) == 4:
            N, C, T, V = feature_maps[0].shape
            weights = torch.mean(grads[0], dim=(2, 3), keepdim=True)
            cam = torch.sum(feature_maps[0] * weights, dim=1)
        # 處理原始關節點 (5D): (N*M, C, T, V) - 在STGCN中C和V可能位置不同，以實際為準
        else: 
            # 假設傳入的是5D的某個中間層，這裡提供一個通用處理
            # 這裡的維度需要根據實際情況調整
            g = grads[0]
            f = feature_maps[0]
            weights = torch.mean(g, dim=tuple(range(2, g.dim())), keepdim=True)
            cam = torch.sum(f * weights, dim=1)

        return cam.detach().cpu().numpy()

    def __call__(self, inputs, index=-1):
        self.model.zero_grad()
        
        # 導入 mmaction2 的標準工具
        from mmengine.dataset import Compose, pseudo_collate
        from mmengine.registry import DefaultScope

        cfg = self.model.cfg
        test_pipeline_cfg = cfg.get('test_pipeline', cfg.get('val_pipeline'))

        with DefaultScope.overwrite_default_scope('mmaction'):
            test_pipeline = Compose(test_pipeline_cfg)
        
        # 1. 執行官方的資料處理流程
        data = test_pipeline(inputs.copy())

        # 2. [釜底抽薪的最終修正] 加入被遺漏的 pseudo_collate 步驟
        #    這會將單一樣本打包成一個批次(batch)，並確保所有內部格式正確
        data = pseudo_collate([data])
        
        # 3. 呼叫模型內建的 test_step 函式來完成前向傳播
        results = self.model.test_step(data)
        scores = results[0].pred_score

        if index == -1:
            index = scores.argmax().item()
        
        # [最終修正] scores 是一個 1D 張量，所以我們用 1D 索引
        score_for_backward = scores[index]
        score = score_for_backward.item()

        # 4. 進行反向傳播
        score_for_backward.backward()
        
        # 5. 獲取特徵圖和梯度
        feature_maps = [self.feature_maps[self._get_target_layer()]]
        grads = [self.gradients[self._get_target_layer()]]
        localization_map = self._calculate_localization_map(feature_maps, grads)
        
        # 返回的 map 可能是 (T, V) 或 (M, T, V)，取第一個即可
        final_map = localization_map[0] if len(localization_map.shape) > 2 else localization_map
        
        return final_map, index, score

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

# ... (visualize_gradcam_on_frame 函式維持不變) ...
def visualize_gradcam_on_frame(frame, keypoints, gradcam_scores, skeleton_conn, frame_size):
    """
    在單一影格上繪製骨架及 Grad-CAM 熱圖。
    """
    fig, ax = plt.subplots(figsize=(frame_size[0] / 100, frame_size[1] / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    cmap = plt.get_cmap('jet')
    
    colors = np.array([[0, 0, 1.]] * len(keypoints)) # 預設為藍色
    if gradcam_scores is not None:
        # [新功能] 增強對比度 (0.5 可以調整，越小越紅)
        enhanced_scores = gradcam_scores**0.5
        colors = cmap(enhanced_scores)[:, :3]

    # [修改] 增大關節點尺寸
    ax.scatter(keypoints[:, 0], keypoints[:, 1], c=colors, s=120, zorder=2, alpha=0.8)
    
    lines, line_colors = [], []
    for p1_idx, p2_idx in coco_skeleton:
        if p1_idx < len(keypoints) and p2_idx < len(keypoints) and \
           keypoints[p1_idx, 0] > 0 and keypoints[p2_idx, 0] > 0:
            lines.append([keypoints[p1_idx], keypoints[p2_idx]])
            line_colors.append((colors[p1_idx] + colors[p2_idx]) / 2)
    
    # [修改] 增粗骨骼線條
    lc = LineCollection(lines, colors=line_colors, linewidths=5, zorder=1, alpha=0.8)
    ax.add_collection(lc)

    fig.canvas.draw()
    img_rgba = np.array(fig.canvas.renderer.buffer_rgba())
    vis_frame = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return vis_frame

def main():
    init_default_scope('mmaction')

    # ... (模型設定維持不變) ...
    ACTION_REC_CONFIG = '/home/cvlab123/mmaction2/configs/skeleton/custom_stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_OurDataset-xsub-keypoint-2d.py'
    ACTION_REC_CHECKPOINT = '/home/cvlab123/mmaction2/work_dirs/rtmpose_all_class_ourdataset_joint_motion/best_acc_top1_epoch_11.pth'
    POSE_CONFIG = '/home/cvlab123/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py'
    POSE_CHECKPOINT = '/home/cvlab123/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'
    VIDEO_PATH = '/home/cvlab123/data/test_data/20250627_172742.mp4'
    OUTPUT_VIDEO_PATH = 'gradcam_output_video.mp4'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
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

    print(f"正在從影片 {VIDEO_PATH} 提取骨架...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_frames = [frame for success, frame in iter(lambda: cap.read(), (False, None))]
    cap.release()
    
    if not video_frames:
        print("錯誤：無法讀取影片或影片為空。")
        return

    frame_h, frame_w, _ = video_frames[0].shape
    results_generator = pose_inferencer(video_frames, show_progress=True)
    
    keypoints_list = [
        np.hstack([p['predictions'][0][0]['keypoints'], np.array(p['predictions'][0][0]['keypoint_scores'])[:, None]])
        if p['predictions'] and p['predictions'][0] else np.zeros((17, 3))
        for p in results_generator
    ]
    
    keypoints_array = np.array(keypoints_list)
    print(f"骨架提取完成，共 {keypoints_array.shape[0]} 幀。")

    anno = {
        'keypoint': keypoints_array[np.newaxis, ...],
        'total_frames': keypoints_array.shape[0],
        'img_shape': (frame_h, frame_w)
    }

    print("正在計算 Grad-CAM...")
    target_layer_name = 'backbone.gcn.6'
    gradcam = STGCNGradCAM(action_model, target_layer_name)
    
    gradcam_map, pred_class_idx, pred_score = gradcam(anno) 
    pred_class_name = idx_to_label.get(pred_class_idx, "Unknown")
    
    print(f"模型預測結果: {pred_class_name} (分數: {pred_score:.4f})")
    
    # --- 3.6 視覺化並儲存影片 ---
    print("正在生成視覺化影片...")
    
    # [修改] 我們不再這裡進行全域正規化，而是移到下面的迴圈中逐幀處理
    # gradcam_scores_normalized = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-6)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 25, (frame_w, frame_h))

    for frame_idx, frame in enumerate(video_frames):
        if frame_idx >= len(keypoints_list): break
        
        kps = keypoints_list[frame_idx][:, :2]
        scores_normalized = None

        if frame_idx < gradcam_map.shape[0]:
            # [新功能] 逐幀正規化
            frame_scores = gradcam_map[frame_idx]
            if frame_scores.max() > frame_scores.min():
                scores_normalized = (frame_scores - frame_scores.min()) / (frame_scores.max() - frame_scores.min())
            else:
                scores_normalized = np.zeros_like(frame_scores)
        
        vis_frame_mpl = visualize_gradcam_on_frame(frame, kps, scores_normalized, coco_skeleton, (frame_w, frame_h))
        vis_frame_bgr = cv2.cvtColor(vis_frame_mpl, cv2.COLOR_RGB2BGR)

        text = f'Pred: {pred_class_name} ({pred_score:.2f})'
        cv2.putText(vis_frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(vis_frame_bgr)
        
    video_writer.release()
    gradcam.remove_hooks()
    print(f"✅ Grad-CAM 視覺化影片已儲存至: {OUTPUT_VIDEO_PATH}")

if __name__ == '__main__':
    main()