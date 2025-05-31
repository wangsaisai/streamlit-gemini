import streamlit as st
from PIL import Image
from io import BytesIO
import os
import logging
from datetime import datetime

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
            translation_model = genga.GenerativeModel("gemini-2.0-flash")
            
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


# --- New Function for Image-to-Image Generation ---
def generate_image_from_image_and_prompt(uploaded_image_file, prompt_text: str, api_key: str):
    """
    Generates/edits an image based on an uploaded image and a text prompt using Google Gemini.
    Displays the new image, saves it, and logs the prompt and any text response.
    """
    logging.info(f"Attempting image-to-image generation with prompt: {prompt_text}")
    try:
        pil_image = Image.open(uploaded_image_file)
        
        # Optionally display the uploaded image for confirmation
        # st.image(pil_image, caption="您上传的图片 (用于编辑)", use_container_width=False, width=300)

        with st.spinner(f"🎨 正在根据您的图片和描述生成新图片..."):
            # Using the genai_client (google.genai) as per the example structure for Gemini multimodal
            client = genai_client.Client(api_key=api_key)
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation", # Specific model for image-to-image
                contents=[prompt_text, pil_image], # Order: text prompt, then image
                config=genai_client_types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'] # Expecting both text and image in response
                )
            )

        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            st.success("🎉 图片编辑/生成成功！")
            generated_image_data = None
            generated_text = None

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    generated_text = part.text
                    st.info(f"模型回复: {generated_text}")
                    logging.info(f"Image-to-image model text response: {generated_text}")
                elif part.inline_data is not None and part.inline_data.data:
                    generated_image_data = part.inline_data.data

            if generated_image_data:
                new_image = Image.open(BytesIO(generated_image_data))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Differentiate filename for image-to-image results
                image_filename = f"{IMAGE_OUTPUT_DIR}/img2img_{timestamp}.png"
                new_image.save(image_filename)
                logging.info(f"Saved image-to-image result: {image_filename}")
                st.image(new_image, caption=f"生成/编辑后的图片 (已保存至 {image_filename})", use_container_width=True)
            else:
                st.warning("⚠️ 未能从模型响应中提取生成的图片。")
                logging.warning("Failed to extract generated image from image-to-image response.")
            
            if not generated_image_data and not generated_text:
                 st.warning("⚠️ 模型未返回图片或文本。请检查您的提示词或上传的图片。")
                 logging.warning("Image-to-image model returned no image or text.")

        else:
            st.warning("⚠️ 图片编辑/生成失败。可能是提示词不当、模型配置问题、API 权限不足，或响应结构不符合预期。")
            st.warning(f"使用的 Prompt: {prompt_text}")
            if response:
                st.warning(f"原始响应详情: {response}") # Log or show more details if needed
            logging.warning(f"Image-to-image generation failed. Prompt used: {prompt_text}. Response: {response}")

    except Exception as e:
        st.error(f"图片编辑/生成过程中发生错误: {e}")
        st.error(f"使用的 Prompt: {prompt_text}")
        logging.error(f"Error during image-to-image generation with prompt '{prompt_text}': {e}", exc_info=True)


# --- Logging Setup ---
LOG_FILE = "image_generation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        # logging.StreamHandler() # Optionally, to also print to console
    ]
)

# --- Image Saving Configuration ---
IMAGE_OUTPUT_DIR = "generated_images"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)


def generate_images_from_prompt(prompt_text: str, num_images: int, api_key: str):
    """
    Generates images based on the provided prompt using Google Imagen.
    Displays images, saves them to a folder, and logs the prompt.
    """
    logging.info(f"Attempting to generate images with prompt: {prompt_text}")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, generated_image in enumerate(response.generated_images):
                try:
                    image_bytes = generated_image.image.image_bytes
                    image = Image.open(BytesIO(image_bytes))
                    
                    # Save the image
                    image_filename = f"{IMAGE_OUTPUT_DIR}/img_{timestamp}_{i+1}.png"
                    image.save(image_filename)
                    logging.info(f"Saved image: {image_filename}")

                    with cols[i % min(num_images, 2)]: # Ensure correct column assignment
                        st.image(image, caption=f"图片 {i+1} (已保存至 {image_filename})", use_container_width=True)
                except Exception as img_err:
                    st.error(f"处理或保存图片 {i+1} 时出错: {img_err}")
                    logging.error(f"Error processing/saving image {i+1}: {img_err}")
        else:
            st.warning("⚠️ 未能生成图片。可能是提示词不当、模型配置问题、API 权限不足，或响应结构不符合预期。")
            st.warning(f"使用的 Prompt: {prompt_text}") # Show the prompt used
            logging.warning(f"Failed to generate images. Prompt used: {prompt_text}")

    except Exception as e:
        st.error(f"图片生成过程中发生错误: {e}")
        st.error(f"使用的 Prompt: {prompt_text}") # Show the prompt used in case of error
        logging.error(f"Error during image generation with prompt '{prompt_text}': {e}")

# --- Streamlit App UI ---
st.set_page_config(page_title="AI 图片生成与编辑", layout="wide")

st.title("🎨 AI 图片生成与编辑服务")
st.caption("支持文字生成图片 (Imagen 3) 与图片编辑 (Gemini)")

# API Key Input
st.sidebar.header("⚙️ 配置")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Google AI API Key:", type="password", help="在此输入您的 Google AI API Key")

if not api_key:
    st.info("请输入您的 Google AI API Key 以开始使用。")
    st.sidebar.warning("API Key 未提供。")

# Mode Selection
generation_mode = st.sidebar.radio(
    "选择模式:",
    ("文生图 (Text-to-Image)", "图生图 (Image-to-Image)"),
    key="generation_mode_selector"
)

# User Input Area
st.header("🖼️ 生成参数")

uploaded_image_file = None
if generation_mode == "图生图 (Image-to-Image)":
    uploaded_image_file = st.file_uploader("上传一张图片进行编辑:", type=["png", "jpg", "jpeg", "webp"], key="image_uploader")
    if uploaded_image_file:
        st.image(uploaded_image_file, caption="您上传的图片", width=300) # Preview uploaded image

prompt_label = "图片描述 (Text-to-Image Prompt):"
default_prompt = "一个宇航员在月球上骑着彩虹色的独角兽，背景是星空。"
if generation_mode == "图生图 (Image-to-Image)":
    prompt_label = "编辑指令 (Image-to-Image Prompt):"
    default_prompt = "为图片中的主要对象戴上一顶派对帽。"

prompt = st.text_area(prompt_label, default_prompt, height=150, key="main_prompt")


# Image count configuration - only for text-to-image
num_images_to_generate = 1 # Default
if generation_mode == "文生图 (Text-to-Image)":
    num_images_to_generate = st.sidebar.number_input("生成图片数量:", min_value=1, max_value=10, value=2, step=1, key="num_images_input")


# Generate Button
generate_button_label = "✨ 生成图片"
if generation_mode == "图生图 (Image-to-Image)":
    generate_button_label = "🎨 编辑图片"
    
generate_button = st.sidebar.button(generate_button_label, use_container_width=True, disabled=not api_key)

st.markdown("---") # Main area separator

# Main logic when generate button is clicked
if generate_button:
    if not api_key:
        st.error("❌ 请在侧边栏输入您的 Google AI API Key。")
    elif not prompt:
        st.error(f"❌ 请输入{prompt_label.split('(')[0].strip()}。")
    else:
        # 1. Translate prompt if necessary (common for both modes)
        # For image-to-image, the prompt is an instruction, translation is still beneficial.
        translated_prompt, original_lang = translate_text_to_english(prompt, api_key)
        final_prompt_to_use = translated_prompt if translated_prompt else prompt
        
        log_message_intro = f"Original prompt ({original_lang})" if original_lang and original_lang != "en" and translated_prompt != prompt else "Prompt"
        if original_lang and original_lang != "en" and translated_prompt != prompt:
            logging.info(f"{log_message_intro}: {prompt}")
            logging.info(f"Translated prompt (en): {translated_prompt}")
        elif not original_lang: # Language detection failed
             logging.info(f"Prompt (language detection failed, using as is): {prompt}")
        else: # Already English or translation not needed/failed but using original
            logging.info(f"Prompt (en or using original): {prompt}")

        # 2. Generate based on mode
        if final_prompt_to_use:
            if generation_mode == "文生图 (Text-to-Image)":
                generate_images_from_prompt(final_prompt_to_use, num_images_to_generate, api_key)
            elif generation_mode == "图生图 (Image-to-Image)":
                if uploaded_image_file is None:
                    st.error("❌ 请上传一张图片以进行图生图编辑。")
                else:
                    generate_image_from_image_and_prompt(uploaded_image_file, final_prompt_to_use, api_key)
        else:
            st.error("❌ 无法获取用于操作的有效 Prompt。")
            logging.error("Could not obtain a valid prompt for image generation/editing.")
            

# Usage Instructions and Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**使用说明:**
1.  **通用**: 模型主要支持英文提示词。若使用其他语言，系统会自动尝试翻译成英文。API Key 在左侧配置。
2.  **文生图 (Text-to-Image)**: 在上方选择此模式，输入描述文字，选择生成数量，点击“生成图片”。
3.  **图生图 (Image-to-Image)**: 在上方选择此模式，上传一张图片，输入编辑指令 (例如："给图片中的猫加上一顶宇航员头盔")，点击“编辑图片”。
""")

st.markdown("---")
st.markdown("<p style='text-align: center;'>由 AI 助手基于您的代码构建与增强</p>", unsafe_allow_html=True)
