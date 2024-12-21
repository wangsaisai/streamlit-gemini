import os
import streamlit as st
import google.generativeai as genai
from PIL import Image

# 配置页面
st.set_page_config(
    page_title="Gemini AI 聊天助手",
)

# 设置 API 密钥
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("请设置 GOOGLE_API_KEY")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# 初始化 Gemini-Pro 模型
MODEL_OPTIONS = {
    "1.5-pro": "gemini-1.5-pro-latest",
    "1.5-flash": "gemini-1.5-flash-latest",
    "2.0-exp(gemini-exp-1206)": "gemini-exp-1206",
    "2.0-flash-exp(gemini-2.0-flash-exp)": "gemini-2.0-flash-exp",
    "2.0-thinking-exp(gemini-2.0-flash-thinking-exp)": "gemini-2.0-flash-thinking-exp",
}

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]

# 设置默认系统提示词
SYSTEM_PROMPT = """你是一个AI助手。请使用中文回答问题。"""

# 页面标题
st.title("Gemini AI 聊天助手")

# 添加侧边栏配置
with st.sidebar:
    st.header("参数设置")

    selected_model = st.selectbox(
        "选择模型",
        options=list(MODEL_OPTIONS.keys()),
        help="选择要使用的 Gemini 模型"
    )

    # 根据选择设置当前模型
    current_model_name = MODEL_OPTIONS[selected_model]
    model = genai.GenerativeModel(current_model_name)

    temperature = st.slider(
        "温度 (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="较高的值会使输出更加随机，较低的值会使其更加集中和确定"
    )
    
    max_tokens = st.number_input(
        "最大 Token 数量",
        min_value=128,
        max_value=8192,
        value=8192,
        help="生成文本的最大长度"
    )

    st.divider()
        
    # 添加新的复选框
    translate_enabled = st.checkbox("翻译模式", help="中英文互译")
    computer_expert = st.checkbox("计算机专家模式", help="使用计算机专家角色进行回答")
    careful_check = st.checkbox("仔细检查", help="更仔细地检查和验证回答")

    st.divider()
    
    upload_image = st.file_uploader("在此上传您的图片", accept_multiple_files=False, type = ['jpg', 'png'])
    
    if upload_image:
        image = Image.open(upload_image)
    else:
        image = None
    st.divider()

    if st.button("清除聊天历史"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
user_input = st.chat_input("Your Question")

if user_input:
    # 添加用户消息到历史记录
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 处理响应
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                # 构建历史消息
                chat = model.start_chat(history=[])
                # 首先发送系统提示词
                chat.send_message(SYSTEM_PROMPT)
                # 然后发送历史消息
                for message in st.session_state.messages[:-1]:  # 不包含最新的用户消息
                    chat.send_message(message["content"])

                # 如果启用翻译模式，先翻译用户输入
                if translate_enabled:
                    # 检测输入是否包含中文字符
                    def contains_chinese(text):
                        return any('\u4e00' <= char <= '\u9fff' for char in text)
                    
                    is_chinese = contains_chinese(user_input)
                    if is_chinese:
                        translation_prompt = f"请将以下中文文本翻译成英文：\n{user_input}"
                    else:
                        translation_prompt = f"请将以下英文文本翻译成中文：\n{user_input}"
                    
                    response = model.generate_content(translation_prompt)
                elif image:
                    # 检查模型是否支持图片处理
                    try:
                        model_info = model.list_model_capabilities()
                        supports_vision = "IMAGE" in [task.name for task in model_info.supported_tasks]
                        
                        if not supports_vision:
                            st.warning("当前选择的模型不支持图片处理，请选择 flash 模型")
                            st.stop()
                    except Exception as e:
                        st.error(f"检查模型能力时出错: {str(e)}")
                        st.stop()
                        
                    st.image(image, caption="上传的图片", use_column_width=True)
                    response = model.generate_content(
                        [user_input, image],
                        generation_config=generation_config
                    )
                else:
                    # 构建提示词
                    prompt_prefix = ""

                    # 如果启用计算机专家模式，添加相应提示词
                    if computer_expert:
                        prompt_prefix += """你是一位专业的计算机领域专家，特别擅长编程、算法、系统架构等技术问题的解答。"""

                    # 如果启用仔细检查模式，添加相应提示词
                    if careful_check:
                        prompt_prefix += """
                        在提供答案之前，请：
                        1. 检查答案的正确性和完整性
                        2. 考虑可能的边界情况和特殊情况
                        3. 确保解释清晰且易于理解
                        4. 如有必要，提供具体的示例
                        5. 如果不确定答案，请明确指出
                        """

                    # 纯文本对话，使用chat.send_message保持上下文
                    response = chat.send_message(
                        prompt_prefix + user_input,
                        generation_config=generation_config
                    )

                # 添加响应检查和处理
                if response.candidates and len(response.candidates) > 0:
                    if response.candidates[0].content:
                        response_text = response.candidates[0].content.parts[0].text
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    else:
                        st.error("模型未能生成有效响应")
                else:
                    st.error("未收到模型响应")
            
            except Exception as e:
                st.error(f"发生错误: {str(e)}")

