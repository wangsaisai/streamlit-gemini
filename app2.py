import os
import streamlit as st
import google.generativeai as genai
from PIL import Image

# 配置页面
st.set_page_config(
    page_title="Gemini AI 聊天助手",
    page_icon="🤖",
    layout="wide"
)

# 设置 API 密钥
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("请设置 GOOGLE_API_KEY")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# 初始化 Gemini-Pro 模型
model = genai.GenerativeModel('gemini-1.5-pro-latest')
vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 页面标题
st.title("💬 Gemini AI 聊天助手")
st.caption("🚀 支持文本对话和图片分析的 AI 助手")

# 添加侧边栏配置
with st.sidebar:
    st.header("参数设置")
    temperature = st.slider(
        "温度 (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
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

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 文件上传
uploaded_file = st.file_uploader("上传图片进行分析", type=['png', 'jpg', 'jpeg'])
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)

# 用户输入
user_input = st.chat_input("请输入您的问题...")

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

                if image:
                    # 如果有图片，使用 vision 模型
                    response = vision_model.generate_content(
                        [user_input, image],
                        generation_config=generation_config
                    )
                else:
                    # 纯文本对话
                    response = model.generate_content(
                        user_input,
                        generation_config=generation_config
                    )
                
                response_text = response.text
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error(f"发生错误: {str(e)}")

# 添加清除按钮
if st.button("清除对话"):
    st.session_state.messages = []
    st.rerun()

# 添加页脚
st.markdown("---")
st.markdown("📝 Powered by Google Gemini & Streamlit")
