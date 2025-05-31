import streamlit as st
from PIL import Image
from io import BytesIO
import os
import logging
from datetime import datetime
import time # Added for video generation polling

# For Text (Translation)
import google.generativeai as genga
# For Images (Client style)
from google import genai as genai_client
from google.genai import types as genai_client_types

from langdetect import detect, LangDetectException

# --- Constants ---
# API Models
TRANSLATION_MODEL_NAME = "gemini-2.0-flash"
IMAGE_TO_IMAGE_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"
TEXT_TO_IMAGE_MODEL_NAME = 'imagen-3.0-generate-002'
VIDEO_MODEL_NAME = "veo-2.0-generate-001" # Added for video generation

# UI Texts
PAGE_TITLE = "AI 内容生成服务" # Updated title
APP_TITLE = "🎨 AI 内容生成服务" # Updated title
APP_CAPTION = "支持文字生成图片 (Imagen 3)、图片编辑 (Gemini) 及文字生成视频 (VEO)" # Updated caption
SIDEBAR_CONFIG_HEADER = "⚙️ 配置"
API_KEY_LABEL = "Google AI API Key:"
API_KEY_HELP = "在此输入您的 Google AI API Key"
MODE_SELECTION_LABEL = "选择模式:"
TEXT_TO_IMAGE_MODE = "文生图 (Text-to-Image)"
IMAGE_TO_IMAGE_MODE = "图生图 (Image-to-Image)"
TEXT_TO_VIDEO_MODE = "文生视频 (Text-to-Video)" # Added for video
GENERATE_PARAMS_HEADER = "🖼️ 生成参数" # Could be "媒体生成参数"
UPLOAD_IMAGE_LABEL = "上传一张图片进行编辑:"
UPLOADED_IMAGE_CAPTION = "您上传的图片"
TEXT_TO_IMAGE_PROMPT_LABEL = "图片描述 (Text-to-Image Prompt):"
TEXT_TO_IMAGE_DEFAULT_PROMPT = "一个宇航员在月球上骑着彩虹色的独角兽，背景是星空。"
IMAGE_TO_IMAGE_PROMPT_LABEL = "编辑指令 (Image-to-Image Prompt):"
IMAGE_TO_IMAGE_DEFAULT_PROMPT = "为图片中的主要对象戴上一顶派对帽。"
TEXT_TO_VIDEO_PROMPT_LABEL = "视频描述 (Text-to-Video Prompt):" # Added for video
TEXT_TO_VIDEO_DEFAULT_PROMPT = "一只小猫在阳光下睡觉的摇摄广角镜头。" # Added for video
NUM_IMAGES_LABEL = "生成图片数量:"
GENERATE_BUTTON_TEXT_TO_IMAGE = "✨ 生成图片"
GENERATE_BUTTON_TEXT_TO_IMAGE_EDIT = "🎨 编辑图片"
GENERATE_BUTTON_TEXT_TO_VIDEO = "🎬 生成视频" # Added for video
FOOTER_TEXT = "<p style='text-align: center;'>由 AI 助手基于您的代码构建与增强</p>"
USAGE_INSTRUCTIONS = """
**使用说明:**
1.  **通用**: 模型主要支持英文提示词。若使用其他语言，系统会自动尝试翻译成英文。API Key 在左侧配置。
2.  **文生图 (Text-to-Image)**: 在上方选择此模式，输入描述文字，选择生成数量，点击“生成图片”。
3.  **图生图 (Image-to-Image)**: 在上方选择此模式，上传一张图片，输入编辑指令 (例如："给图片中的猫加上一顶宇航员头盔")，点击“编辑图片”。
4.  **文生视频 (Text-to-Video)**: 在上方选择此模式，输入视频描述，点击“生成视频”。视频生成可能需要较长时间。
"""

# Configuration
LOG_FILE = "content_generation.log" # Renamed for broader scope
IMAGE_OUTPUT_DIR = "generated_images"
VIDEO_OUTPUT_DIR = "generated_videos" # Added for video


# --- Logging Setup ---
# Moved logging setup to be configured once, early.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        # logging.StreamHandler() # Optionally, to also print to console
    ]
)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True) # Added for video


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
            translation_model = genga.GenerativeModel(TRANSLATION_MODEL_NAME)
            
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
                model=IMAGE_TO_IMAGE_MODEL_NAME, # Specific model for image-to-image
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
                st.image(new_image, caption="生成/编辑后的图片", use_container_width=True)
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


# Note: Logging setup and IMAGE_OUTPUT_DIR creation moved to the top global scope.

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
                model=TEXT_TO_IMAGE_MODEL_NAME,
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
                        st.image(image, caption=f"图片 {i+1}", use_container_width=True)
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


# --- New Function for Text-to-Video Generation ---
def generate_video_from_prompt(prompt_text: str, api_key: str):
    """
    Generates a video based on the provided prompt using Google VEO.
    Displays the video, saves it, and logs the prompt.
    """
    logging.info(f"Attempting to generate video with prompt: {prompt_text}")
    try:
        with st.spinner(f"🎬 正在生成视频，这可能需要几分钟时间，请耐心等待..."):
            client = genai_client.Client(api_key=api_key)
            
            operation = client.models.generate_videos(
                model=VIDEO_MODEL_NAME,
                prompt=prompt_text,
                config=genai_client_types.GenerateVideosConfig(
                    person_generation="dont_allow",  # "dont_allow" or "allow_adult"
                    aspect_ratio="16:9",  # "16:9" or "9:16"
                ),
            )
            
            st.info("视频生成任务已提交，正在处理中... 您可以在下方看到进度更新。")
            progress_bar = st.progress(0)
            # Polling loop
            while not operation.done:
                time.sleep(20) # Poll every 20 seconds
                operation = client.operations.get(name=operation.operation.name) # Refresh operation status
                # Try to get progress if available in metadata
                if operation.metadata and hasattr(operation.metadata, 'progress_percentage'):
                     progress_bar.progress(int(operation.metadata.progress_percentage))
                elif operation.metadata and hasattr(operation.metadata, 'state_description'):
                     st.info(f"更新: {operation.metadata.state_description}")


            progress_bar.progress(100) # Mark as complete

        if operation.response and operation.response.generated_videos:
            st.success(f"🎉 成功生成 {len(operation.response.generated_videos)} 个视频！")
            for i, generated_video_obj in enumerate(operation.response.generated_videos):
                try:
                    video_file_resource = generated_video_obj.video # This is a File object
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"{VIDEO_OUTPUT_DIR}/video_{timestamp}_{i}.mp4"
                    
                    # Download the video file
                    # The File object itself doesn't have a .save(). We use genai.download_file()
                    genai_client.download_file(video_file_resource.name, video_filename)
                    logging.info(f"Saved video: {video_filename}")

                    st.video(video_filename)
                    st.caption(f"视频 {i+1} (已保存至 {video_filename})")

                except Exception as vid_err:
                    st.error(f"处理或保存视频 {i+1} 时出错: {vid_err}")
                    logging.error(f"Error processing/saving video {i+1} (name: {video_file_resource.name if 'video_file_resource' in locals() else 'unknown'}): {vid_err}")
        else:
            st.warning("⚠️ 未能生成视频。可能是提示词不当、模型配置问题、API 权限不足，或响应结构不符合预期。")
            st.warning(f"使用的 Prompt: {prompt_text}")
            logging.warning(f"Failed to generate video. Prompt used: {prompt_text}. Operation details: {operation}")

    except Exception as e:
        st.error(f"视频生成过程中发生错误: {e}")
        st.error(f"使用的 Prompt: {prompt_text}")
        logging.error(f"Error during video generation with prompt '{prompt_text}': {e}", exc_info=True)


# --- Streamlit UI Setup ---
def setup_sidebar():
    """Sets up the Streamlit sidebar elements."""
    st.sidebar.header(SIDEBAR_CONFIG_HEADER)
    api_key_input = os.getenv("GOOGLE_API_KEY")
    if not api_key_input:
        api_key_input = st.sidebar.text_input(API_KEY_LABEL, type="password", help=API_KEY_HELP)

    if not api_key_input:
        st.sidebar.warning("API Key 未提供。")
    
    selected_mode = st.sidebar.radio(
        MODE_SELECTION_LABEL,
        (TEXT_TO_IMAGE_MODE, IMAGE_TO_IMAGE_MODE, TEXT_TO_VIDEO_MODE), # Added video mode
        key="generation_mode_selector"
    )
    
    num_images = 4 # Default for text-to-image
    if selected_mode == TEXT_TO_IMAGE_MODE:
        num_images = st.sidebar.number_input(NUM_IMAGES_LABEL, min_value=1, max_value=10, value=4, step=1, key="num_images_input")
    elif selected_mode == IMAGE_TO_IMAGE_MODE:
        num_images = 1 # For image-to-image, only one image is processed/generated
    # No num_images input for video mode, as it typically generates one video.

    st.sidebar.markdown("---")
    st.sidebar.markdown(USAGE_INSTRUCTIONS)
    return api_key_input, selected_mode, num_images

def setup_main_interface(generation_mode: str):
    """Sets up the main interface elements based on the selected mode."""
    st.header(GENERATE_PARAMS_HEADER)
    
    uploaded_file = None
    current_prompt_label = TEXT_TO_IMAGE_PROMPT_LABEL
    current_default_prompt = TEXT_TO_IMAGE_DEFAULT_PROMPT
    generate_button_text = GENERATE_BUTTON_TEXT_TO_IMAGE

    if generation_mode == IMAGE_TO_IMAGE_MODE:
        uploaded_file = st.file_uploader(UPLOAD_IMAGE_LABEL, type=["png", "jpg", "jpeg", "webp"], key="image_uploader")
        if uploaded_file:
            st.image(uploaded_file, caption=UPLOADED_IMAGE_CAPTION, width=300)
        current_prompt_label = IMAGE_TO_IMAGE_PROMPT_LABEL
        current_default_prompt = IMAGE_TO_IMAGE_DEFAULT_PROMPT
        generate_button_text = GENERATE_BUTTON_TEXT_TO_IMAGE_EDIT
    elif generation_mode == TEXT_TO_VIDEO_MODE:
        # No file upload for text-to-video
        current_prompt_label = TEXT_TO_VIDEO_PROMPT_LABEL
        current_default_prompt = TEXT_TO_VIDEO_DEFAULT_PROMPT
        generate_button_text = GENERATE_BUTTON_TEXT_TO_VIDEO
        
    prompt_text_area = st.text_area(current_prompt_label, current_default_prompt, height=150, key="main_prompt")
        
    st.markdown("---")
    return uploaded_file, prompt_text_area, generate_button_text, current_prompt_label


def handle_generation_request(api_key: str, prompt: str, original_prompt_label: str,
                              generation_mode: str, num_images: int, uploaded_image_file):
    """Handles the logic for image generation or editing when the button is clicked."""
    if not api_key:
        st.error("❌ 请在侧边栏输入您的 Google AI API Key。")
        return
    if not prompt:
        st.error(f"❌ 请输入{original_prompt_label.split('(')[0].strip()}。")
        return

    translated_prompt, original_lang = translate_text_to_english(prompt, api_key)
    final_prompt_to_use = translated_prompt if translated_prompt else prompt
    
    log_message_intro = f"Original prompt ({original_lang})" if original_lang and original_lang != "en" and translated_prompt != prompt else "Prompt"
    if original_lang and original_lang != "en" and translated_prompt != prompt:
        logging.info(f"{log_message_intro}: {prompt}")
        logging.info(f"Translated prompt (en): {translated_prompt}")
    elif not original_lang:
        logging.info(f"Prompt (language detection failed, using as is): {prompt}")
    else:
        logging.info(f"Prompt (en or using original): {prompt}")

    if final_prompt_to_use:
        if generation_mode == TEXT_TO_IMAGE_MODE:
            generate_images_from_prompt(final_prompt_to_use, num_images, api_key)
        elif generation_mode == IMAGE_TO_IMAGE_MODE:
            if uploaded_image_file is None:
                st.error("❌ 请上传一张图片以进行图生图编辑。")
            else:
                generate_image_from_image_and_prompt(uploaded_image_file, final_prompt_to_use, api_key)
        elif generation_mode == TEXT_TO_VIDEO_MODE:
            generate_video_from_prompt(final_prompt_to_use, api_key)
    else:
        st.error("❌ 无法获取用于操作的有效 Prompt。")
        logging.error("Could not obtain a valid prompt for content generation.")


# --- Main Application ---
def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    # Sidebar setup
    api_key, generation_mode, num_images_to_generate = setup_sidebar()
    
    if not api_key: # If still no API key after sidebar input (and not from env)
        st.info("请输入您的 Google AI API Key 以开始使用。")
        # No need to return here, allow UI to render, button will be disabled.

    # Main interface setup
    uploaded_image_file, prompt_input, generate_button_label, current_prompt_label_for_error = setup_main_interface(generation_mode)

    # Generate Button
    if st.button(generate_button_label, use_container_width=True, disabled=not api_key, key="main_generate_button"):
        handle_generation_request(
            api_key=api_key,
            prompt=prompt_input,
            original_prompt_label=current_prompt_label_for_error,
            generation_mode=generation_mode,
            num_images=num_images_to_generate,
            uploaded_image_file=uploaded_image_file
        )

    st.markdown("---")
    st.markdown(FOOTER_TEXT, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
