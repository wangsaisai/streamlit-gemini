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
    "2.5-preview-05-06": "gemini-2.5-pro-preview-05-06",
    "2.5-flash-preview-05-20": "gemini-2.5-flash-preview-05-20",
    "2.0-flash": "gemini-2.0-flash",
    "2.0-thinking-exp": "gemini-2.0-flash-thinking-exp-01-21",
}

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]
if "prompt_used" not in st.session_state:
    st.session_state.prompt_used = False

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
        st.session_state.prompt_used = False

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

                if book_mode:
                    max_tokens *= 2

                generation_config = genai.types.GenerateContentConfig(
                    tools=[google_search_tool] if search_enabled else None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                prompt_prefix = ""
                if not st.session_state.prompt_used:
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
                        我希望你扮演一位深刻的书籍分析专家，运用你的专业知识和洞察力，为我剖析一本书。请根据文末提供的书籍信息，生成一份极其详尽且富有深度的分析报告。

                        **报告的核心目标是：** 彻底挖掘并清晰阐释书籍的**核心洞见**与**主要观点**，同时全面覆盖其他重要分析维度。

                        请在报告中包含以下方面的内容，并确保分析的深度和广度：

                        1.  **基本信息：**
                            *   作者简介：深入介绍作者的学术/创作背景、知识体系、写作风格特点、所属流派或领域。提及可能深刻影响本书创作的其他关键作品、人生经历或思想转变。
                            *   出版信息：首次出版的确切时间与背景。
                            *   书籍分类：精准定位本书所属的大类（如社科、哲学、文学、历史、科普、心理学、传记等），并尽可能细化至具体子领域（如社科中的批判理论、人类学民族志；文学中的魔幻现实主义、成长小说等）。

                        2.  **核心内容精粹：**
                            *   **（重点）** 提纲挈领地概述全书探讨的核心议题、试图解答的关键问题或（若是小说）驱动情节发展的核心冲突与脉络。
                            *   梳理全书的宏观结构（如章节安排、逻辑递进关系、叙事框架），点明各主要部分的关键内容和功能。

                        3.  **【重中之重】核心洞见与主要观点深度剖析：**
                            *   **（极其重点）** **集中火力、不吝篇幅地**分析和提炼书中提出的**最核心、最具原创性、最具启发性的深刻洞见和关键论断**。作者通过本书究竟想向世界传达什么根本性的信息？
                            *   详细阐述这些核心观点是如何被论证的？作者运用了哪些证据、逻辑或叙事技巧来支撑它们？
                            *   这些观点的新颖性、颠覆性或深刻性体现在何处？它们挑战了哪些传统认知或流行观念？
                            *   （如果是小说）深入解读其核心主题（如爱、死亡、自由、正义等）、反复出现的象征意象、人物弧光背后揭示的人性或社会现实。

                        4.  **关键概念与理论框架解读：**
                            *   识别并透彻解释书中反复出现、或对理解核心观点至关重要的**独特概念、术语、理论模型或分析框架**。
                            *   阐明这些概念的精确内涵、来源（是作者原创、借用还是批判性发展？），以及它们在构建全书论证体系或叙事世界中的核心作用。

                        5.  **书中名言警句/精彩摘录：**
                            *   精选书中**最能体现核心思想、语言精辟、发人深省或极具代表性**的名言警句、经典段落。
                            *   摘录原文，并可选择性地附上简要的语境说明或意义解读，以展现其精华所在。

                        6.  **书籍评价、争议与深远影响：**
                            *   客观总结本书在学术界、评论界及不同读者群体中的主流评价，务必包含**赞誉和批评**两方面的主要声音。
                            *   本书自问世以来，在思想界、特定学科领域、社会文化层面或后续创作中引发了哪些具体而重要的影响？（例如：开创了新的研究范式、引发了重大社会讨论、成为某领域的奠基之作、被广泛引用、获得重大奖项等）。
                            *   是否存在围绕本书的著名争议、重要的学术辩论或持续的批评焦点？具体内容是什么？
                            *   时效性与现代审视： 根据最新的科学研究、学术进展或社会观念变迁，评估本书内容的时代局限性。明确指出书中是否有观点因后续发展而被认为过时、存在错误，或者需要进行补充、修正和批判性看待？

                        7.  **阅读策略与进阶建议：**
                            *   为渴望深度理解本书的读者提供具体的阅读方法建议：需要哪些学科背景或知识储备？阅读时应特别留意哪些线索或论证层次？适合快速把握脉络还是需要字斟句酌地精读？推荐采用何种笔记法（如思维导图、章节摘要、概念卡片）？
                            *   推荐哪些有助于加深理解的辅助阅读材料？（如：作者的其他著作、相关的学术论文、评论文章、纪录片、访谈、同一主题的其他经典书籍等）。

                        8.  **目标读者画像：**
                            *   清晰描绘本书最适合的读者群体特征：是专业研究者、高校学生、特定行业从业人员、对特定议题有浓厚兴趣的公众读者，还是寻求特定情感共鸣或人生启迪的读者？
                            *   阅读本书可能需要读者具备哪些先验的知识基础、思维能力或兴趣偏好？

                        9.  **同类书比较与独特定位：**
                            *   列举若干本探讨相似主题、领域或体裁的重要书籍。
                            *   **着重对比分析**：本书与这些同类书籍相比，在核心观点、研究方法、论证风格、叙事策略、材料选择、结论或整体基调上有哪些**显著的异同**？本书的独特性和不可替代的价值体现在哪里？

                        请确保你的分析报告展现出真正的专家水准：**洞察深刻、论证严谨、信息翔实、结构清晰、语言精练且富有启发性。**

                        ---
                        **需要分析的书籍信息：**
                        """
                    
                    st.session_state.prompt_used = True  # 标记 prompt 已使用

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

