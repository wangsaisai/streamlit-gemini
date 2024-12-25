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
    "2.0-flash-exp(gemini-2.0-flash-exp)": "gemini-2.0-flash-exp",
    "2.0-exp(gemini-exp-1206)": "gemini-exp-1206",
    "2.0-thinking-exp(gemini-2.0-flash-thinking-exp)": "gemini-2.0-flash-thinking-exp",
    "1.5-pro": "gemini-1.5-pro-latest",
    "1.5-flash": "gemini-1.5-flash-latest",
}

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]

# 设置默认系统提示词
SYSTEM_PROMPT = """你是一个AI聊天助手。请使用中文回答用户问题。"""

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
        value=2048,
        help="生成文本的最大长度"
    )

    st.divider()
        
    # 添加新的复选框
    stream_enabled = st.checkbox("流式输出", value=True, help="开启后将实时显示AI响应")
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


# 创建统一的流式输出处理函数
def handle_streaming_response(response_iter):
    message_placeholder = st.empty()
    full_response = ""
    
    try:
        for chunk in response_iter:
            # 只处理文本内容
            if hasattr(chunk, 'text'):
                full_response += chunk.text
            elif hasattr(chunk, 'parts'):
                for part in chunk.parts:
                    if hasattr(part, 'text'):
                        full_response += part.text
            
            # 更新显示
            message_placeholder.markdown(full_response + "▌")
    except Exception as e:
        st.error(f"流式输出错误: {str(e)}")
        return None
    
    # 最终显示（移除光标）
    message_placeholder.markdown(full_response)
    return full_response

# 添加一个辅助函数来处理非流式响应
def handle_normal_response(response):
    if response.candidates and len(response.candidates) > 0:
        if response.candidates[0].content:
            parts = response.candidates[0].content.parts
            # 合并所有文本部分
            return ''.join(part.text for part in parts if hasattr(part, 'text'))
    return None


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

                # 处理普通对话模式
                if not translate_enabled and not image:
                    prompt_prefix = ""
                    if computer_expert:
                        prompt_prefix += """你是一位专业的计算机领域专家，特别擅长编程、算法、系统架构等技术问题的解答。"""
                    if careful_check:
                        prompt_prefix += """
                        在提供答案之前，请：
                        1. 检查答案的正确性和完整性
                        2. 考虑可能的边界情况和特殊情况
                        3. 确保解释清晰且易于理解
                        4. 如有必要，提供具体的示例
                        5. 如果不确定答案，请明确指出
                        """

                    # 构建完整的对话历史
                    history_messages = []
                    for msg in st.session_state.messages[:-1]:  # 不包含最新的用户消息
                        history_messages.append({"role": msg["role"], "parts": [msg["content"]]})

                    # 添加系统提示词到历史消息开头
                    full_messages = [
                        {"role": "user", "parts": [SYSTEM_PROMPT]},
                        *history_messages,
                        {"role": "user", "parts": [prompt_prefix + user_input]}
                    ]

                    try:
                        if stream_enabled:
                            response = model.generate_content(
                                full_messages,
                                generation_config=generation_config,
                                stream=True
                            )
                            response_text = handle_streaming_response(response)
                        else:
                            response = model.generate_content(
                                prompt_prefix + user_input,
                                generation_config=generation_config
                            )
                            response_text = handle_normal_response(response)
                    except Exception as e:
                        st.error(f"生成响应时出错: {str(e)}")
                        st.stop()

                # 处理翻译模式
                elif translate_enabled:
                    def contains_chinese(text):
                        return any('\u4e00' <= char <= '\u9fff' for char in text)
                    
                    is_chinese = contains_chinese(user_input)
                    if is_chinese:
                        translation_prompt = f"请将以下中文文本翻译成英文：\n{user_input}"
                    else:
                        translation_prompt = f"请将以下英文文本翻译成中文：\n{user_input}"
                    
                    if stream_enabled:
                        response_text = handle_streaming_response(
                            model.generate_content(
                                translation_prompt,
                                generation_config=generation_config,
                                stream=True
                            )
                        )
                    else:
                        response = model.generate_content(
                            translation_prompt,
                            generation_config=generation_config
                        )
                        response_text = handle_normal_response(response)

                # 处理图片模式
                else:  # image mode
                    # 检查模型是否支持图片处理
                    # try:
                    #     model_info = model.list_model_capabilities()
                    #     supports_vision = "IMAGE" in [task.name for task in model_info.supported_tasks]
                        
                    #     if not supports_vision:
                    #         st.warning("当前选择的模型不支持图片处理，请选择 flash 模型")
                    #         st.stop()
                    # except Exception as e:
                    #     st.error(f"检查模型能力时出错: {str(e)}")
                    #     st.stop()
                        
                    st.image(image, caption="上传的图片", use_column_width=True)
                    
                    if stream_enabled:
                        response_text = handle_streaming_response(
                            model.generate_content(
                                [user_input, image],
                                generation_config=generation_config,
                                stream=True
                            )
                        )
                    else:
                        response = model.generate_content(
                            [user_input, image],
                            generation_config=generation_config
                        )
                        response_text = handle_normal_response(response)


                # 保存响应到会话状态
                if response_text:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    if not stream_enabled:  # 如果不是流式输出，需要显示响应
                        st.markdown(response_text)
                else:
                    st.error("未能获取有效响应")
                    st.stop()

            except Exception as e:
                st.error(f"发生错误: {str(e)}")

