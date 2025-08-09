import os
import streamlit as st
import logging
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, GenerateContentResponse
from PIL import Image
import prompts # Import the new prompts module

# 配置日志记录器
logging.basicConfig(
    filename='chat_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    "2.5-pro": "gemini-2.5-pro",
    "2.5-flash": "gemini-2.5-flash",
    "2.0-flash": "gemini-2.0-flash",
    "2.0-thinking-exp": "gemini-2.0-flash-thinking-exp-01-21",
}

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]
if "prompt_used" not in st.session_state:
    st.session_state.prompt_used = False
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# 页面标题
st.title("Gemini AI 聊天助手")

# --- Helper Functions ---
def configure_sidebar():
    """Configures and displays the sidebar elements."""
    with st.sidebar:
        st.header("参数设置")

        selected_model_key = st.selectbox(
            "选择模型",
            options=list(MODEL_OPTIONS.keys()),
            help="选择要使用的 Gemini 模型"
        )
        current_model_name = MODEL_OPTIONS[selected_model_key]

        with st.expander("高级参数设置", expanded=False):
            temperature = st.slider(
                "温度 (Temperature)", 0.0, 1.0, 0.1, 0.1,
                help="较高的值会使输出更加随机，较低的值会使其更加集中和确定"
            )
            max_tokens = st.number_input(
                "最大 Token 数量", 128, 8192, 8192,
                help="生成文本的最大长度"
            )
        st.divider()

        st.subheader("输出与搜索设置")
        stream_enabled = st.checkbox("流式输出", value=True, help="开启后将实时显示AI响应")
        search_enabled = st.checkbox("启用搜索工具", value=True, help="使用Google搜索增强回答能力")
        st.divider()

        st.subheader("模式选择")
        translate_enabled = st.checkbox("翻译模式", help="中英文互译")
        computer_expert = st.checkbox("计算机专家模式", help="使用计算机专家角色进行回答")
        book_mode = st.checkbox("书籍模式", help="深入理解一本书")
        careful_check = st.checkbox("仔细检查", help="更仔细地检查和验证回答")
        st.divider()

        # Use a dynamic key to allow programmatic clearing of the uploader
        upload_image_file = st.file_uploader(
            "在此上传您的图片",
            accept_multiple_files=False,
            type=['jpg', 'png'],
            key=f"file_uploader_{st.session_state.uploader_key}"
        )

        # Synchronize the PIL image object with the file uploader's state
        if upload_image_file is None:
            st.session_state.uploaded_image = None
        else:
            st.session_state.uploaded_image = Image.open(upload_image_file)

        image = st.session_state.get("uploaded_image")

        if image and upload_image_file:
            st.info(f"已上传图片: {upload_image_file.name}，将在聊天中使用")

        st.divider()

        if st.button("清除聊天历史"):
            st.session_state.messages = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]
            st.session_state.prompt_used = False
            st.session_state.uploaded_image = None
            st.session_state.uploader_key += 1  # Increment key to reset file uploader
            st.rerun()

    return (current_model_name, temperature, max_tokens, stream_enabled,
            translate_enabled, computer_expert, book_mode, careful_check,
            search_enabled, image)

def display_chat_history():
    """Displays the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_streaming_response(response_iter):
    """Handles streaming responses from the API."""
    message_placeholder = st.empty()
    full_response = ""
    try:
        for chunk in response_iter:
            text_chunk = ""
            if isinstance(chunk, GenerateContentResponse):
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    text_chunk += part.text
            elif isinstance(chunk, str): # Fallback for direct string chunks if API changes
                text_chunk = chunk
            else:
                # st.warning(f"未知类型的chunk: {type(chunk)}") # Can be noisy
                continue
            
            full_response += text_chunk
            message_placeholder.markdown(full_response + "▌")
    except Exception as e:
        st.error(f"流式输出错误: {str(e)}")
        return None
    message_placeholder.markdown(full_response)
    return full_response

def handle_normal_response(response):
    """Handles non-streaming (normal) responses from the API."""
    if response and hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
            return ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
    return None

def build_prompt_prefix(computer_expert, careful_check, book_mode):
    """Builds the prefix for the prompt based on selected modes."""
    prefix = ""
    if not st.session_state.prompt_used:
        if computer_expert:
            prefix += prompts.COMPUTER_EXPERT_PROMPT
        if careful_check:
            prefix += prompts.CAREFUL_CHECK_PROMPT
        if book_mode:
            prefix += prompts.BOOK_MODE_PROMPT
            st.session_state.prompt_used = True # Mark prompt as used for book mode
    return prefix

def build_gemini_messages(system_prompt_text, history_messages, current_user_input_with_prefix):
    """Constructs the message list in the format expected by the Gemini API."""
    messages = [{"role": "user", "parts": [{"text": system_prompt_text}]}]
    for msg in history_messages:
        role = "model" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "parts": [{"text": msg["content"]}]})
    messages.append({"role": "user", "parts": [{"text": current_user_input_with_prefix}]})
    return messages

def generate_response_with_search(client_instance, model_name, messages_list, gen_config):
    """Generates a response using the search tool."""
    response = client_instance.models.generate_content(
        model=model_name,
        contents=messages_list,
        config=gen_config,
    )
    if (response and hasattr(response, 'candidates') and response.candidates and
        hasattr(response.candidates[0], 'grounding_metadata') and
        response.candidates[0].grounding_metadata and
        hasattr(response.candidates[0].grounding_metadata, 'search_entry_point') and
        response.candidates[0].grounding_metadata.search_entry_point and
        hasattr(response.candidates[0].grounding_metadata.search_entry_point, 'rendered_content')):
        with st.expander("搜索结果"):
            st.markdown(response.candidates[0].grounding_metadata.search_entry_point.rendered_content, unsafe_allow_html=True)
    return handle_normal_response(response)

def generate_standard_response(client_instance, model_name, messages_list, gen_config, stream_enabled_flag):
    """Generates a standard text response (streaming or normal)."""
    try:
        if stream_enabled_flag:
            response_iter = client_instance.models.generate_content_stream(
                model=model_name,
                contents=messages_list,
                config=gen_config,
            )
            return handle_streaming_response(response_iter)
        else:
            response = client_instance.models.generate_content(
                model=model_name,
                contents=messages_list,
                config=gen_config,
            )
            return handle_normal_response(response)
    except Exception as e:
        st.error(f"生成响应时出错: {str(e)}")
        return None

def generate_translation_response(client_instance, model_name, user_text, gen_config, stream_enabled_flag):
    """Generates a translation response."""
    is_chinese = any('\u4e00' <= char <= '\u9fff' for char in user_text)
    if is_chinese:
        translation_prompt_text = prompts.TRANSLATE_TO_ENGLISH_PROMPT_TEMPLATE.format(text=user_text)
    else:
        translation_prompt_text = prompts.TRANSLATE_TO_CHINESE_PROMPT_TEMPLATE.format(text=user_text)

    if stream_enabled_flag:
        return handle_streaming_response(
            client_instance.models.generate_content_stream(
                contents=translation_prompt_text, model=model_name, config=gen_config
            )
        )
    else:
        response = client_instance.models.generate_content(
            contents=translation_prompt_text, model=model_name, config=gen_config
        )
        return handle_normal_response(response)

def generate_image_response(client_instance, model_name, user_text, image_data, gen_config, stream_enabled_flag):
    """Generates a response for image-based input."""
    st.image(image_data, caption="上传的图片", use_container_width=True)
    contents = [user_text, image_data]
    if stream_enabled_flag:
        return handle_streaming_response(
            client_instance.models.generate_content_stream(
                contents=contents, model=model_name, config=gen_config
            )
        )
    else:
        response = client_instance.models.generate_content(
            contents=contents, model=model_name, config=gen_config
        )
        return handle_normal_response(response)

# --- Main App Logic ---
(current_model, temperature_setting, max_tokens_setting, stream_enabled_opt,
 translate_enabled_opt, computer_expert_opt, book_mode_opt, careful_check_opt,
 search_enabled_opt, image_data_opt) = configure_sidebar()

display_chat_history()

user_input = st.chat_input("Your Question")

if user_input:
    logging.info(f"\n\n\nUser: {user_input}\n")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                current_max_tokens = max_tokens_setting * 2 if book_mode_opt else max_tokens_setting

                generation_config_obj = genai.types.GenerateContentConfig(
                    tools=[Tool(google_search=GoogleSearch())] if search_enabled_opt else None,
                    temperature=temperature_setting,
                    max_output_tokens=current_max_tokens,
                )

                prompt_prefix_text = build_prompt_prefix(
                    computer_expert_opt, careful_check_opt, book_mode_opt
                )
                
                gemini_messages = build_gemini_messages(
                    prompts.SYSTEM_PROMPT,
                    st.session_state.messages[:-1], # History
                    prompt_prefix_text + user_input # Current input with prefix
                )
                
                response_text = None
                if image_data_opt:
                    response_text = generate_image_response(
                        client, current_model, user_input, image_data_opt, generation_config_obj, stream_enabled_opt
                    )
                elif search_enabled_opt:
                    response_text = generate_response_with_search(
                        client, current_model, gemini_messages, generation_config_obj
                    )
                elif translate_enabled_opt:
                    response_text = generate_translation_response(
                        client, current_model, user_input, generation_config_obj, stream_enabled_opt
                    )
                else: # Standard chat
                    response_text = generate_standard_response(
                        client, current_model, gemini_messages, generation_config_obj, stream_enabled_opt
                    )

                if response_text:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    logging.info(f"Assistant: {response_text}")
                    if not stream_enabled_opt or search_enabled_opt: # Non-streaming or search needs explicit markdown
                        st.markdown(response_text)
                else:
                    st.error("未能获取有效响应")
                    # st.stop() # Consider if stopping is always desired on no response

            except Exception as e:
                st.error(f"发生错误: {str(e)}")
                logging.error(f"Error during response generation: {str(e)}", exc_info=True)
