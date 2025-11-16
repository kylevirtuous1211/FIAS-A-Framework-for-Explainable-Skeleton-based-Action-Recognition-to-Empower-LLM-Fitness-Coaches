import os
from dotenv import load_dotenv

# --- LangChain Imports ---
# Use the standard ChatOpenAI class, which works with any OpenAI-compatible API
from langchain_openai import ChatOpenAI
# Import message types for structuring the conversation with the LLM
from langchain.schema.messages import HumanMessage, SystemMessage

def get_ai_coach_feedback(input_prompt: str) -> str:
    print("🤖 Initializing AI Fitness Coach...")
    # --- 1. Load Groq API Key ---
    try:
        load_dotenv(dotenv_path='/home/cvlab123/api_key/.env')
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY not found in the .env file. Please check the path and file content."
    except Exception as e:
        return f"Error loading .env file: {e}"

    # --- 2. Initialize the Language Model (LLM) ---
    try:
        llm = ChatOpenAI(
            model="llama-3.3-70b-versatile", 
            api_key=api_key, 
            base_url="https://api.groq.com/openai/v1"
        )
    except Exception as e:
        return f"Error initializing the LLM. Check your API key and network. Details: {e}"

    analysis_prompt = input_prompt

    # --- 4. Define the AI Coach's Persona and Task (System Prompt) ---
    # system_prompt = """
    #     # 角色
    #     您是一位頂尖的運動科學專家，正在解讀一份由 AI 視覺分析系統產出的生物力學報告。

    #     # 情境
    #     該系統分析了一位使用者做的伏地挺身，並將其動作判定為 `push_up_elbow` (意指手肘角度有問題的伏地挺身)，信賴度為 58.68%。以下是報告中的關鍵數據：

    #     * **判斷依據（關鍵身體部位）**: 系統的判斷主要基於「左腕」、「右肩」和「右髖」這三個部位的動態。
    #     * **關鍵時間點**: 動作的後段（第73幀）是判定的最高峰，尤其是在「左腕」。
    #     * **動態過程**: 動作初期、中期到後期，系統的判斷依據從腳踝轉移到手腕。

    #     # 任務指令
    #     您的任務不是複述「模型關注了哪裡」，而是要**將這些被標記出的「關鍵身體部位」，轉譯為對使用者實際動作的具體描述與診斷**。

    #     請根據以上數據，完成以下分析：

    #     1.  **動作診斷 (Movement Diagnosis)**: 綜合「手肘有問題」的分類結果以及被高亮的「手腕、肩膀、髖部」，請直接推斷使用者最可能犯的動作錯誤是什麼？（例如：手肘是否過度向外打開？身體是否未能維持直線？手腕角度是否不正確？）
    #     2.  **原因分析 (Causal Analysis)**: 以專業的生物力學角度，解釋為什麼這個錯誤的動作模式，會導致「手腕」、「肩膀」和「髖部」成為壓力或不穩定的焦點。
    #     3.  **修正建議 (Corrective Advice)**: 基於您的診斷，直接給予使用者清晰、可行的修正指令。

    #     # 輸出要求
    #     * **口吻**: 請直接以專業、親切的教練口吻對使用者說話。
    #     * **禁忌詞**: 在您的回覆中，**絕對不要**提及「模型」、「AI」、「信賴度」、「幀」、「注意力」或「分析報告」等任何技術詞彙。您就是專家，這份報告是您的專業判斷。
    # """
    
    system_prompt = """
    # Role
    You are a top-tier sports science expert interpreting a biomechanical report produced by an AI visual analysis system.

        # Context
        The system analyzed a user's push-up and classified the movement as `push_up_elbow` (meaning a push-up with problematic elbow angles), with a confidence of 58.68%. The key data from the report is as follows:

        * **Basis for Judgment (Key Body Parts)**: The system's classification was primarily based on the dynamics of the 'left wrist', 'right shoulder', and 'right hip'.
        * **Key Moment**: The latter part of the movement (frame 73) was the peak moment for the classification, especially concerning the 'left wrist'.
        * **Dynamic Process**: From the initial, middle, to late phases of the movement, the system's focus shifted from the ankles to the wrists.

        # Task Instructions
        Your task is not to repeat "where the model focused," but to **translate these highlighted "key body parts" into a concrete description and diagnosis of the user's actual movement**.

        Based on the data above, please complete the following analysis:

        1.  **Movement Diagnosis**: Synthesizing the "problematic elbow" classification with the highlighted "wrist, shoulder, and hip", directly infer the most likely movement error the user is making. (e.g., Are the elbows flaring out too much? Is the body failing to maintain a straight line? Is the wrist angle incorrect?)
        2.  **Causal Analysis**: From a professional biomechanical perspective, explain why this incorrect movement pattern would cause the "wrist", "shoulder", and "hip" to become points of stress or instability.
        3.  **Corrective Advice**: Based on your diagnosis, provide the user with clear, actionable corrective instructions.

        # Output Requirements
        * **Tone**: Speak directly to the user in the professional, friendly tone of a coach.
        * **Forbidden Words**: In your response, **absolutely do not** mention "model", "AI", "confidence", "frame", "attention", "analysis report", or any other technical jargon. You are the expert; this report is your professional judgment.
    """

    # --- 5. Invoke the LLM and Get Feedback ---
    print("💬 Asking the AI coach for feedback...")
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_prompt)
        ])
        feedback = response.content
        return feedback
    except Exception as e:
        return f"Error getting feedback from the AI model: {e}"