import streamlit as st
from PIL import Image
from io import BytesIO
import os

# For Text (Translation)
import google.generativeai as genga
# For Images (Client style)
from google import genai as genai_client
from google.genai import types as genai_client_types

from langdetect import detect, LangDetectException

# --- Streamlit App ---

st.set_page_config(page_title="文字生成图片", layout="wide")

st.title("🎨 文字生成图片 Web 服务")
st.caption("使用 Google Imagen 3 模型")

# API Key 输入
st.sidebar.header("配置")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.info("请输入您的 Google AI API Key 以开始使用。")
    st.sidebar.warning("API Key 未提供。")

# 用户输入
st.sidebar.header("图片参数")
prompt = st.sidebar.text_area("图片描述 (Prompt):", "一个宇航员在月球上骑着彩虹色的独角兽，背景是星空。", height=150)
# num_images = st.sidebar.slider("生成图片数量:", 1, 4, 4, help="目前代码固定为4张，此滑块仅为示例。")
# 根据用户提供的代码，固定生成4张图片
fixed_num_images = 4

# 生成按钮
generate_button = st.sidebar.button("✨ 生成图片", use_container_width=True, disabled=not api_key)

st.markdown("---")

if generate_button:
    if not api_key:
        st.error("❌ 请在侧边栏输入您的 Google AI API Key。")
    elif not prompt:
        st.error("❌ 请输入图片描述 (prompt)。")
    else:
        translated_prompt = prompt
        original_language = "en"

        try:
            # Detect language
            original_language = detect(prompt)
            st.info(f"检测到输入语言: {original_language}")

            if original_language != "en":
                st.info(f"输入语言非英文 ({original_language})，正在尝试翻译...")
                # Using gemini-1.5-flash for translation
                # Initialize the model directly
                # Ensure API key is configured for the text generation module
                genga.configure(api_key=api_key) # Configure API key globally using genga
                translation_model = genga.GenerativeModel("gemini-2.0-flash") # Use genga

                # Simple translation prompt
                translation_response = translation_model.generate_content(f"Translate the following text to English: {prompt}")

                if translation_response and translation_response.text:
                    translated_prompt = translation_response.text.strip()
                    st.success(f"翻译成功！翻译后的 Prompt: {translated_prompt}")
                else:
                    st.warning("⚠️ 翻译失败，将使用原始 Prompt 进行图片生成。")
                    translated_prompt = prompt # Fallback to original prompt
            else:
                st.info("输入语言为英文，无需翻译。")

        except LangDetectException:
            st.warning("⚠️ 语言检测失败，将使用原始 Prompt 进行图片生成。")
            translated_prompt = prompt # Fallback to original prompt
        except Exception as translation_err:
            st.error(f"翻译过程中发生错误: {translation_err}")
            st.warning("⚠️ 翻译失败，将使用原始 Prompt 进行图片生成。")
            translated_prompt = prompt # Fallback to original prompt


        try:
            # Using user provided code structure for image generation
            # API key is already configured globally

            with st.spinner(f"🧠 正在生成 {fixed_num_images} 张图片中，请稍候..."):
                # Instantiate the genai_client Client for images
                client = genai_client.Client(api_key=api_key)

                # Call client.generate_images (trying direct method on client)
                response = client.models.generate_image( # Changed from client.models.generate_images
                    model='imagen-3.0-generate-002', # This model name might need to be 'models/imagen-3.0-generate-002'
                    prompt=translated_prompt,
                    config=genai_client_types.GenerateImageConfig( # Use genai_client_types
                        number_of_images=fixed_num_images,
                    )
                )

            # Process the response assuming it has a 'generated_images' attribute
            # This structure is based on the user's snippet for client.models.generate_images
            # If client.generate_images has a different response structure, this may need adjustment
            if response and hasattr(response, 'generated_images') and response.generated_images:
                st.success(f"🎉 成功生成 {len(response.generated_images)} 张图片！")

                # Display images
                cols = st.columns(2)
                for i, generated_image in enumerate(response.generated_images):
                    try:
                        # Access image bytes as per the user's provided snippet and original code
                        image_bytes = generated_image.image.image_bytes
                        image = Image.open(BytesIO(image_bytes))
                        with cols[i % 2]:
                            st.image(image, caption=f"图片 {i+1} - {prompt[:30]}...", use_column_width=True)
                    except Exception as img_err:
                        st.error(f"处理图片 {i+1} 时出错: {img_err}")
            else:
                st.warning("⚠️ 未能生成图片。可能是提示词不当、模型配置问题、API 权限不足，或响应结构不符合预期。")

        except Exception as e:
            st.error(f"发生错误: {e}")
            

# 使用说明和脚注
st.sidebar.markdown("---")
st.sidebar.markdown("""
**使用说明:**
1. 该模型只支持英文的提示词。请使用英文写提示词，若使用其他语音，会自动翻译成英文后在生成图片
""")

st.markdown("---")
st.markdown("<p style='text-align: center;'>由 AI 助手基于您的代码构建</p>", unsafe_allow_html=True)
