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


# --- Helper Functions ---
def translate_text_to_english(text: str, api_key: str) -> tuple[str, str | None]:
    """
    Detects the language of the input text and translates it to English if it's not already English.
    Returns the translated text (or original if already English/translation failed) and the detected language.
    """
    original_language = "en"
    translated_text = text

    try:
        original_language = detect(text)
        st.info(f"检测到输入语言: {original_language}")

        if original_language != "en":
            st.info(f"输入语言非英文 ({original_language})，正在尝试翻译...")
            genga.configure(api_key=api_key)
            # Using a specific model known for good translation, adjust if needed
            translation_model = genga.GenerativeModel("gemini-1.5-flash-latest")
            
            # Optimized prompt for direct translation
            prompt_template = f"Translate the following text from {original_language} to English. Provide only the translated text, without any additional explanations or alternative translations:\n\n{text}"
            
            translation_response = translation_model.generate_content(prompt_template)

            if translation_response and translation_response.text:
                translated_text = translation_response.text.strip()
                st.success(f"翻译成功！翻译后的 Prompt: {translated_text}")
            else:
                st.warning("⚠️ 翻译失败，将使用原始 Prompt 进行图片生成。")
                # Fallback to original text if translation response is empty
        else:
            st.info("输入语言为英文，无需翻译。")

    except LangDetectException:
        st.warning("⚠️ 语言检测失败，将使用原始 Prompt 进行图片生成。")
        original_language = None # Indicate detection failure
    except Exception as translation_err:
        st.error(f"翻译过程中发生错误: {translation_err}")
        st.warning("⚠️ 翻译失败，将使用原始 Prompt 进行图片生成。")
        original_language = None # Indicate translation error
    
    return translated_text, original_language

def generate_images_from_prompt(prompt_text: str, num_images: int, api_key: str):
    """
    Generates images based on the provided prompt using Google Imagen.
    Displays images or error messages in Streamlit.
    """
    try:
        with st.spinner(f"🧠 正在生成 {num_images} 张图片中，请稍候..."):
            client = genai_client.Client(api_key=api_key)
            response = client.models.generate_images(
                model='imagen-3.0-generate-002',
                config=genai_client_types.GenerateImagesConfig(
                    number_of_images=num_images,
                ),
                prompt=prompt_text # Pass the (potentially translated) prompt here
            )

        if response and hasattr(response, 'generated_images') and response.generated_images:
            st.success(f"🎉 成功生成 {len(response.generated_images)} 张图片！")
            cols = st.columns(min(num_images, 2)) # Adjust columns based on number of images, max 2
            for i, generated_image in enumerate(response.generated_images):
                try:
                    image_bytes = generated_image.image.image_bytes
                    image = Image.open(BytesIO(image_bytes))
                    with cols[i % min(num_images, 2)]: # Ensure correct column assignment
                        st.image(image, caption=f"图片 {i+1}", use_container_width=True)
                except Exception as img_err:
                    st.error(f"处理图片 {i+1} 时出错: {img_err}")
        else:
            st.warning("⚠️ 未能生成图片。可能是提示词不当、模型配置问题、API 权限不足，或响应结构不符合预期。")
            st.warning(f"使用的 Prompt: {prompt_text}") # Show the prompt used

    except Exception as e:
        st.error(f"图片生成过程中发生错误: {e}")
        st.error(f"使用的 Prompt: {prompt_text}") # Show the prompt used in case of error

# --- Streamlit App UI ---
st.set_page_config(page_title="文字生成图片", layout="wide")

st.title("🎨 文字生成图片 Web 服务")
st.caption("使用 Google Imagen 3 模型")

# API Key Input
st.sidebar.header("配置")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Google AI API Key:", type="password", help="在此输入您的 Google AI API Key")

if not api_key:
    st.info("请输入您的 Google AI API Key 以开始使用。")
    st.sidebar.warning("API Key 未提供。")

# User Input - Moved to main area
st.header("🖼️ 图片生成参数")
prompt = st.text_area("图片描述 (Prompt):", "一个宇航员在月球上骑着彩虹色的独角兽，背景是星空。", height=200, key="main_prompt")

# Image count configuration
num_images_to_generate = st.sidebar.number_input("生成图片数量:", min_value=1, max_value=10, value=2, step=1) # Default to 2, max 10 for now

# Generate Button
generate_button = st.sidebar.button("✨ 生成图片", use_container_width=True, disabled=not api_key)

st.markdown("---")

if generate_button:
    if not api_key:
        st.error("❌ 请在侧边栏输入您的 Google AI API Key。")
    elif not prompt: # Check the main prompt area
        st.error("❌ 请输入图片描述 (prompt)。")
    else:
        # 1. Translate if necessary
        translated_prompt, _ = translate_text_to_english(prompt, api_key)
        
        # 2. Generate Images
        if translated_prompt: # Proceed only if we have a prompt (original or translated)
            generate_images_from_prompt(translated_prompt, num_images_to_generate, api_key)
        else:
            st.error("❌ 无法获取用于生成图片的有效 Prompt。")
            

# Usage Instructions and Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**使用说明:**
1. 该模型只支持英文的提示词。请使用英文写提示词，若使用其他语音，会自动翻译成英文后在生成图片
""")

st.markdown("---")
st.markdown("<p style='text-align: center;'>由 AI 助手基于您的代码构建</p>", unsafe_allow_html=True)
