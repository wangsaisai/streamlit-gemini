import os
import streamlit as st
import logging
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, GenerateContentResponse
from PIL import Image

# 配置日志记录器
logging.basicConfig(
    filename='chat_log.txt',  # 日志文件名
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

# 配置页面
st.set_page_config(
    page_title="Gemini AI 聊天助手",
)

# 设置 API 密钥
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("请设置 GOOGLE_API_KEY")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# 初始化 Gemini-Pro 模型
MODEL_OPTIONS = {
    "2.5-preview": "gemini-2.5-pro-preview-03-25",
    "2.5-exp(gemini-2.5-pro-exp-03-25)": "gemini-2.5-pro-exp-03-25",
    "2.0-flash(gemini-2.0-flash)": "gemini-2.0-flash",
    "2.0-thinking-exp(gemini-2.0-flash-thinking-exp-01-21)": "gemini-2.0-flash-thinking-exp-01-21",
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

    temperature = st.slider(
        "温度 (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
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
    stream_enabled = st.checkbox("流式输出", value=True, help="开启后将实时显示AI响应")
    translate_enabled = st.checkbox("翻译模式", help="中英文互译")
    computer_expert = st.checkbox("计算机专家模式", help="使用计算机专家角色进行回答")
    careful_check = st.checkbox("仔细检查", help="更仔细地检查和验证回答")
    search_enabled = st.checkbox("启用搜索工具", value=False, help="使用Google搜索增强回答能力")

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
            if isinstance(chunk, GenerateContentResponse):
                # 处理多部分内容
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    full_response += part.text
            elif isinstance(chunk, str):
                full_response += chunk
            else:
                st.warning(f"未知类型的chunk: {type(chunk)}")
                continue
            
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
    # 记录用户输入
    logging.info(f"\n\n\nUser: {user_input}\n")
    # 添加用户消息到历史记录
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 处理响应
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 配置搜索工具
                google_search_tool = Tool(
                    google_search = GoogleSearch()
                )

                generation_config = genai.types.GenerateContentConfig(
                    tools=[google_search_tool] if search_enabled else None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                prompt_prefix = ""
                if computer_expert:
                    prompt_prefix += """
                    你是一位专业的计算机领域专家，特别擅长编程、算法、系统架构等技术问题的解答。

                    """
                if careful_check:
                    prompt_prefix += """
                    在提供答案之前，请：
                    1. 检查答案的正确性和完整性
                    2. 考虑可能的边界情况和特殊情况
                    3. 确保解释清晰且易于理解
                    4. 如有必要，提供具体的示例
                    5. 如果不确定答案，请明确指出

                    """

                # 构建正确的消息格式
                messages = [
                    {
                        "role": "user",
                        "parts": [{"text": SYSTEM_PROMPT}]
                    }
                ]

                # 添加历史消息
                for msg in st.session_state.messages[:-1]:
                    role = "model" if msg["role"] == "assistant" else "user"
                    messages.append({
                        "role": role,
                        "parts": [{"text": msg["content"]}]
                    })

                # 添加当前用户消息
                messages.append({
                    "role": "user",
                    "parts": [{"text": prompt_prefix + user_input}]
                })

                if search_enabled:
                    response = client.models.generate_content(
                        model=MODEL_OPTIONS[selected_model],
                        contents=messages,
                        config=generation_config,
                    )
                    
                    candidate = response.candidates[0]
                    if (hasattr(candidate, 'grounding_metadata') and 
                        candidate.grounding_metadata and 
                        hasattr(candidate.grounding_metadata, 'search_entry_point') and
                        candidate.grounding_metadata.search_entry_point and
                        hasattr(candidate.grounding_metadata.search_entry_point, 'rendered_content')):
                        with st.expander("搜索结果"):
                            st.markdown(candidate.grounding_metadata.search_entry_point.rendered_content, unsafe_allow_html=True)
                    
                    response_text = handle_normal_response(response)

                # 处理普通对话模式
                elif not translate_enabled and not image:
                    try:
                        if stream_enabled:
                            response = client.models.generate_content_stream(
                                model=MODEL_OPTIONS[selected_model],
                                contents=messages,
                                config=generation_config,
                            )
                            response_text = handle_streaming_response(response)
                        else:
                            response = client.models.generate_content(
                                model=MODEL_OPTIONS[selected_model],
                                contents=messages,
                                config=generation_config,
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
                            client.models.generate_content_stream(
                                contents=translation_prompt,
                                model=MODEL_OPTIONS[selected_model],
                                config=generation_config,
                            )
                        )
                    else:
                        response = client.models.generate_content(
                            contents=translation_prompt,
                            model=MODEL_OPTIONS[selected_model],
                            config=generation_config
                        )
                        response_text = handle_normal_response(response)

                # 处理图片模式
                else:  # image mode  
                    st.image(image, caption="上传的图片", use_column_width=True)
                    
                    if stream_enabled:
                        response_text = handle_streaming_response(
                            client.models.generate_content_stream(
                                contents=[user_input, image],
                                model=MODEL_OPTIONS[selected_model],
                                config=generation_config,
                            )
                        )
                    else:
                        response = client.models.generate_content(
                            contents=[user_input, image],
                            model=MODEL_OPTIONS[selected_model],
                            config=generation_config,
                        )
                        response_text = handle_normal_response(response)

                # 保存响应到会话状态
                if response_text:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    # 记录助手响应
                    logging.info(f"Assistant: {response_text}")
                    if not stream_enabled or search_enabled:  # 如果不是流式输出，需要显示响应
                        st.markdown(response_text)
                else:
                    st.error("未能获取有效响应")
                    st.stop()

            except Exception as e:
                st.error(f"发生错误: {str(e)}")

