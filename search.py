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
    book_mode = st.checkbox("书籍模式", help="深入理解一本书")
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
                if book_mode:
                    prompt_prefix += """
                    我希望你扮演一个书籍分析专家，帮助我深入理解一本书。请针对文末提供的书籍信息，生成一份详细的分析报告。

                    请在报告中包含以下几个方面的内容，并尽可能详细和深刻：

                    1.  **基本信息：**
                        *   作者简介：介绍作者的背景、写作风格、所属领域、以及可能影响本书的其他重要作品或经历。
                        *   出版信息：首次出版时间，如有重要再版或修订版，也请提及。
                        *   书籍分类：明确本书属于哪个大的类别（如社科、小说、历史、哲学、科普、心理学、传记等），并尽可能细化（如社科中的社会学、人类学；小说中的科幻、推理、言情等）。

                    2.  **核心内容概述：**
                        *   详细阐述本书的主要内容、中心论点、故事情节（如果是小说）或探讨的核心问题。
                        *   如果可能，可以按章节或主要部分进行概括，梳理其逻辑结构或叙事脉络。

                    3.  **深刻洞见与主要论点/主题：**
                        *   分析并提炼书中提出的最核心、最具启发性的观点、见解或思想。
                        *   作者试图通过本书传达的关键信息是什么？有哪些独到的发现或深刻的思考？
                        *   （如果是小说）分析其核心主题、象征意义、人物塑造的深层含义等。

                    4.  **关键概念解读：**
                        *   识别并解释书中提出的重要的、独特的或反复出现的概念、术语、理论框架或模型。
                        *   解释这些概念的含义、来源（如果是引用或发展而来）及其在书中的作用和重要性。

                    5.  **书籍评价与影响：**
                        *   总结本书在学术界、评论界或读者群体中的普遍评价，包括正反两方面的观点。
                        *   本书自出版以来产生了哪些重要影响？（例如，在特定领域引发了讨论、启发了后续研究、改变了人们的认知、获得了重要奖项等）。
                        *   是否存在围绕本书的著名争议或批评？

                    6.  **阅读建议与方法：**
                        *   对于想要深入理解本书的读者，你有什么推荐的阅读方法？（例如：需要具备哪些背景知识？阅读时应关注哪些重点？是适合快速浏览还是需要精读细思？做笔记或思维导图是否有帮助？）
                        *   是否有推荐的辅助阅读材料（如其他相关书籍、纪录片、学术论文等）？

                    7.  **适合读者群体：**
                        *   这本书主要适合哪些类型的读者？（例如：专业研究人员、学生、对特定话题感兴趣的普通读者、特定行业的从业者、寻找特定情感体验的读者等）。
                        *   阅读这本书可能需要读者具备哪些基础知识或兴趣点？

                    8.  **同类书籍比较：**
                        *   是否存在探讨相似主题或内容的同类书籍？请列举一些。
                        *   这些同类书籍与本书相比，在观点、论证方式、结论或侧重点上有哪些主要的区别？请重点分析其不同之处。

                    请确保你的分析客观、深入、全面，并使用清晰、易于理解的语言进行阐述。

                    ---
                    **需要分析的书籍信息：**

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

