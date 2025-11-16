import numpy as np
import json

def generate_llm_prompt_en(gradcam_map, pred_name, pred_score, keypoint_names, total_frames):
    """
    Analyzes Grad-CAM data and generates a structured LLM prompt in English.
    This version now accepts a list of valid predictions to analyze the sequence of actions.
    """
    FACE_KEYPOINT_INDICES = {0, 1, 2, 3, 4}  # Nose, Left Eye, Right Eye, Left Ear, Right Ear
    body_gradcam_map = gradcam_map.copy()
    body_gradcam_map[:, list(FACE_KEYPOINT_INDICES)] = 0

    # --- New: Handle list of valid predictions ---
    if isinstance(pred_name, list) and pred_name:
        # This is the new format with a list of valid predictions
        prompt_parts = [
            "The AI model made the following sequence of high-confidence predictions throughout the video:",
            json.dumps(pred_name, indent=2),
            "\nBased on the model's attention (Grad-CAM) data across the video, here is a summary of key observations:"
        ]
        # For the final task description, we can use the most frequent action as the main subject
        from collections import Counter
        main_action = Counter(p['action'] for p in pred_name).most_common(1)[0][0]
    else:
        # Fallback for the old format (single prediction)
        prompt_parts = [
            f"The AI model's final prediction for the entire video is '{pred_name}' with an overall confidence of {pred_score:.2%}.",
            "Based on the model's attention (Grad-CAM) data across the video, here is a summary of key observations:"
        ]
        main_action = pred_name

    # --- 1. Overall Most Important Joints ---
    avg_joint_importance = np.mean(body_gradcam_map, axis=0)
    top_joint_indices = np.argsort(avg_joint_importance)[::-1][:3]
    top_joint_names = [keypoint_names.get(idx, f"Unknown Joint {idx}") for idx in top_joint_indices]
    
    prompt_parts.append(f"1. Overall Most Critical Body Parts: {', '.join(top_joint_names)}.")

    # --- 2. The Most Critical Moment ---
    peak_activation = np.max(body_gradcam_map)
    peak_frame_idx, peak_joint_idx = np.unravel_index(np.argmax(body_gradcam_map), body_gradcam_map.shape)
    peak_joint_name = keypoint_names.get(peak_joint_idx, f"Unknown Joint {peak_joint_idx}")

    prompt_parts.append(f"2. The Most Critical Moment: At frame {peak_frame_idx}, the model's peak attention ({peak_activation:.2f}) occurred at the '{peak_joint_name}'.")

    # --- 3. Attention Shift Analysis ---
    third = total_frames // 3
    
    map_part_1 = body_gradcam_map[0:third]
    top_joint_1 = keypoint_names.get(np.argmax(np.mean(map_part_1, axis=0))) if map_part_1.size > 0 else "None"

    map_part_2 = body_gradcam_map[third:2*third]
    top_joint_2 = keypoint_names.get(np.argmax(np.mean(map_part_2, axis=0))) if map_part_2.size > 0 else "None"
        
    map_part_3 = body_gradcam_map[2*third:]
    top_joint_3 = keypoint_names.get(np.argmax(np.mean(map_part_3, axis=0))) if map_part_3.size > 0 else "None"

    prompt_parts.extend([
        f"3. Attention Shift Over Time:",
        f"   - Early Phase (frames 0-{third}): Primarily focused on the '{top_joint_1}'.",
        f"   - Middle Phase (frames {third}-{2*third}): Primarily focused on the '{top_joint_2}'.",
        f"   - Late Phase (frames {2*third}-{total_frames}): Primarily focused on the '{top_joint_3}'."
    ])

    # --- 4. Task Instruction for the LLM ---
    prompt_parts.append(f"\n--- TASK FOR THE LANGUAGE MODEL ---")
    prompt_parts.append(f"You are an expert AI fitness coach. Based on the sequence of actions and the attention data for the main action '{main_action}', provide a concise analysis of the user's performance. Your analysis should:")
    prompt_parts.append("1. Briefly summarize the sequence of actions performed by the user. Note any changes or inconsistencies in form (e.g., 'The user started with a correct squat but then their form degraded into squat_knees_inward').")
    prompt_parts.append("2. Infer the reason for the model's attention patterns. For example, 'The model focused on the knees and hips, which is where form breakdown typically occurs in an incorrect squat.'")
    prompt_parts.append("3. Provide clear, actionable advice for correction. For instance, 'To correct this, focus on pushing your knees outward, as if you are trying to spread the floor apart with your feet.'")
    prompt_parts.append("4. Keep the tone encouraging and professional.")

    return "\n".join(prompt_parts)


def generate_llm_prompt_ch(gradcam_map, pred_name, pred_score, keypoint_names, total_frames):
    """
    分析 Grad-CAM 數據並生成結構化的 LLM Prompt。
    [新功能] 此版本會忽略臉部關節點，專注於身體動作分析。
    """
    # [核心修正 1] 定義要忽略的臉部關節點索引
    FACE_KEYPOINT_INDICES = {0, 1, 2, 3, 4}  # 鼻子, 左眼, 右眼, 左耳, 右耳

    # [核心修正 2] 創建一個新的熱圖副本，並將臉部關節點的權重設為0
    # 這樣它們就不會被後續的 argmax 或 argsort 選中
    body_gradcam_map = gradcam_map.copy()
    body_gradcam_map[:, list(FACE_KEYPOINT_INDICES)] = 0

    prompt_parts = []

    # --- Part 1: 模型的基本預測 ---
    prompt_parts.append(f"模型基本分析報告：")
    prompt_parts.append(f"1. 最終預測動作: {pred_name}")
    prompt_parts.append(f"2. 預測信賴度: {pred_score:.2%}")

    # --- Part 2: 整體最重要的關節 (空間分析) ---
    # [核心修正 3] 在過濾後的 body_gradcam_map 上進行計算
    avg_joint_importance = np.mean(body_gradcam_map, axis=0)
    top_k = 3
    top_joint_indices = np.argsort(avg_joint_importance)[::-1][:top_k]
    top_joint_names = [keypoint_names.get(idx, f"未知關節 {idx}") for idx in top_joint_indices]
    
    prompt_parts.append(f"\n模型注意力分析：")
    prompt_parts.append(f"1. 整體最關鍵的身體部位: {', '.join(top_joint_names)}")

    # --- Part 3: 最關鍵的瞬間 (時空分析) ---
    # [核心修正 4] 同樣在過濾後的 map 上尋找最大值
    peak_activation = np.max(body_gradcam_map)
    peak_frame_idx, peak_joint_idx = np.unravel_index(np.argmax(body_gradcam_map), body_gradcam_map.shape)
    peak_joint_name = keypoint_names.get(peak_joint_idx, f"未知關節 {peak_joint_idx}")

    prompt_parts.append(f"2. 最關鍵的瞬間: 在第 {peak_frame_idx} 幀，模型的注意力頂峰 ({peak_activation:.2f}) 出現在「{peak_joint_name}」。")

    # --- Part 4: 注意力轉移分析 (進階) ---
    third = total_frames // 3
    
    # [核心修正 5] 所有分段分析都使用過濾後的 map
    map_part_1 = body_gradcam_map[0:third]
    top_joint_1 = keypoint_names.get(np.argmax(np.mean(map_part_1, axis=0))) if map_part_1.size > 0 else "無"

    map_part_2 = body_gradcam_map[third:2*third]
    top_joint_2 = keypoint_names.get(np.argmax(np.mean(map_part_2, axis=0))) if map_part_2.size > 0 else "無"
        
    map_part_3 = body_gradcam_map[2*third:]
    top_joint_3 = keypoint_names.get(np.argmax(np.mean(map_part_3, axis=0))) if map_part_3.size > 0 else "無"

    prompt_parts.append(f"3. 注意力隨時間轉移的過程:")
    prompt_parts.append(f"   - 動作前期 (0-{third}幀): 主要關注「{top_joint_1}」")
    prompt_parts.append(f"   - 動作中期 ({third}-{2*third}幀): 主要關注「{top_joint_2}」")
    prompt_parts.append(f"   - 動作後期 ({2*third}-{total_frames}幀): 主要關注「{top_joint_3}」")

    # --- Part 5: 給 LLM 的任務指令 ---
    prompt_parts.append(f"\n給語言模型的任務：")
    prompt_parts.append(f"基於以上對身體主要關節的數據分析，請以專業健身教練的口吻，分析這個「{pred_name}」動作的可能執行情況。請推斷為什麼模型會關注這些特定的身體部位和時間點，並給出可能的修正建議。")

    return "\n".join(prompt_parts)