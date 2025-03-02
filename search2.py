import os
import streamlit as st
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, GenerateContentResponse
from PIL import Image

def configure_page():
    st.set_page_config(page_title="Gemini AI 聊天助手")

def get_api_key():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        st.error("请设置 GOOGLE_API_KEY")
        st.stop()
    return api_key

def initialize_client(api_key):
    return genai.Client(api_key=api_key)

def initialize_model_options():
    return {
        "2.0-flash(gemini-2.0-flash)": "gemini-2.0-flash",
        "2.0-exp(gemini-2.0-pro-exp-02-05)": "gemini-2.0-pro-exp-02-05",
        "2.0-thinking-exp(gemini-2.0-flash-thinking-exp-01-21)": "gemini-2.0-flash-thinking-exp-01-21",
        "1.5-pro": "gemini-1.5-pro-latest",
        "1.5-flash": "gemini-1.5-flash-latest",
    }

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]

def display_sidebar(model_options):
    with st.sidebar:
        st.header("参数设置")

        selected_model = st.selectbox("选择模型", options=list(model_options.keys()), help="选择要使用的 Gemini 模型")
        current_model_name = model_options[selected_model]

        temperature = st.slider("温度 (Temperature)", min_value=0.0, max_value=1.0, value=0.3, step=0.1, help="较高的值会使输出更加随机，较低的值会使其更加集中和确定")
        max_tokens = st.number_input("最大 Token 数量", min_value=128, max_value=8192, value=2048, help="生成文本的最大长度")

        st.divider()

        stream_enabled = st.checkbox("流式输出", value=True, help="开启后将实时显示AI响应")
        translate_enabled = st.checkbox("翻译模式", help="中英文互译")
        computer_expert = st.checkbox("计算机专家模式", help="使用计算机专家角色进行回答")
        careful_check = st.checkbox("仔细检查", help="更仔细地检查和验证回答")
        search_enabled = st.checkbox("启用搜索工具", value=True, help="使用Google搜索增强回答能力, 仅gemini-2.0-flash支持")

        st.divider()

        upload_image = st.file_uploader("在此上传您的图片", accept_multiple_files=False, type=['jpg', 'png'])

        if upload_image:
            image = Image.open(upload_image)
        else:
            image = None

        st.divider()

        if st.button("清除聊天历史"):
            st.session_state.messages.clear()
            st.session_state["messages"] = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]

    return current_model_name, temperature, max_tokens, stream_enabled, translate_enabled, computer_expert, careful_check, search_enabled, image

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_streaming_response(response_iter):
    message_placeholder = st.empty()
    full_response = ""

    try:
        for chunk in response_iter:
            if isinstance(chunk, GenerateContentResponse):
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

            message_placeholder.markdown(full_response + "▌")
    except Exception as e:
        st.error(f"流式输出错误: {str(e)}")
        return None

    message_placeholder.markdown(full_response)
    return full_response

def handle_normal_response(response):
    if response.candidates and len(response.candidates) > 0:
        if response.candidates[0].content:
            parts = response.candidates[0].content.parts
            return ''.join(part.text for part in parts if hasattr(part, 'text'))
    return None

def generate_response(client, messages, model, generation_config, stream_enabled, search_enabled):
    if search_enabled:
        response = client.models.generate_content(model='gemini-2.0-flash', contents=messages, config=generation_config)
        candidate = response.candidates[0]
        if (hasattr(candidate, 'grounding_metadata') and 
            candidate.grounding_metadata and 
            hasattr(candidate.grounding_metadata, 'search_entry_point') and
            candidate.grounding_metadata.search_entry_point and
            hasattr(candidate.grounding_metadata.search_entry_point, 'rendered_content')):
            with st.expander("搜索结果"):
                st.markdown(candidate.grounding_metadata.search_entry_point.rendered_content, unsafe_allow_html=True)
        return handle_normal_response(response)
    else:
        try:
            if stream_enabled:
                response = client.models.generate_content_stream(model=model, contents=messages, config=generation_config)
                return handle_streaming_response(response)
            else:
                response = client.models.generate_content(model=model, contents=messages, config=generation_config)
                return handle_normal_response(response)
        except Exception as e:
            st.error(f"生成响应时出错: {str(e)}")
            st.stop()

def main():
    configure_page()
    api_key = get_api_key()
    client = initialize_client(api_key)
    model_options = initialize_model_options()
    initialize_session_state()
    st.title("Gemini AI 聊天助手")
    current_model_name, temperature, max_tokens, stream_enabled, translate_enabled, computer_expert, careful_check, search_enabled, image = display_sidebar(model_options)
    display_chat_history()
    
    user_input = st.chat_input("Your Question")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                google_search_tool = Tool(google_search=GoogleSearch())
                generation_config = genai.types.GenerateContentConfig(
                    tools=[google_search_tool] if search_enabled else None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                prompt_prefix = ""
                if computer_expert:
                    prompt_prefix += "你是一位专业的计算机领域专家，特别擅长编程、算法、系统架构等技术问题的解答。\n\n"
                if careful_check:
                    prompt_prefix += "在提供答案之前，请：\n1. 检查答案的正确性和完整性\n2. 考虑可能的边界情况和特殊情况\n3. 确保解释清晰且易于理解\n4. 如有必要，提供具体的示例\n5. 如果不确定答案，请明确指出\n\n"

                messages = [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}]
                for msg in st.session_state.messages[:-1]:
                    role = "model" if msg["role"] == "assistant" else "user"
                    messages.append({"role": role, "parts": [{"text": msg["content"]}]})
                messages.append({"role": "user", "parts": [{"text": prompt_prefix + user_input}]})

                response_text = generate_response(client, messages, current_model_name, generation_config, stream_enabled, search_enabled)
                if response_text:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    if not stream_enabled or search_enabled:
                        st.markdown(response_text)
                else:
                    st.error("未能获取有效响应")
                    st.stop()

if __name__ == "__main__":
    main()
