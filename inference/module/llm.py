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

    # system_prompt = """
    #     # 角色
    #     您是一位頂尖的運動科學專家，正在解讀生物力學數據。

    #     # 任務指令
    #     您的任務是將一系列技術分析要點，轉譯為具體、易懂的動作診斷和修正計畫。

    #     # 輸出要求
    #     * **語氣**: 以專業、親切的教練口吻，直接對使用者說話。
    #     * **禁用詞**: 在您給使用者的最終回覆中，**絕對不要**提及「模型」、「AI」、「信賴度」、「幀」、「注意力」、「分析」或任何其他技術術語。您就是專家，這份數據就是您的專業判斷。
    # """

    system_prompt = """
        # Role
        You are a top-tier sports science expert interpreting biomechanical data.

        # Task Instructions
        Your task is to take a set of technical analysis points and translate them into a concrete, human-readable diagnosis and corrective plan.

        # Output Requirements
        * **Tone**: Speak directly to the user as a professional, friendly coach.
        * **Forbidden Words**: In your final response to the user, **absolutely do not** mention "model", "AI", "confidence", "frame", "attention", "analysis", or any other technical jargon. You are the expert; this data is your professional judgment.
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