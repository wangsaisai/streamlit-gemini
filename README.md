# Streamlit + Google Gemini 聊天程序

基于 Streamlit 和 Google Gemini 构建的智能聊天应用，支持文本对话和图片分析功能。

## 功能特点

- 💬 支持自然语言对话 (使用模型：`gemini-1.5-pro-latest`)
- 🖼️ 支持图片上传和分析 (使用模型：`gemini-1.5-flash-latest`)
- 🚀 简单易用的 Web 界面
- 💻 基于 Streamlit 快速部署

## 使用前准备

1. 获取 Google Gemini API Key
2. 安装必要的依赖包：
```python
pip install streamlit google-generativeai pillow
```

## 使用方法

1. 设置环境变量：
```bash
export GOOGLE_API_KEY="your-api-key"
```

也可以在app.py中直接修改GOOGLE_API_KEY：
```python
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY = "your-api-key"
```

2. 运行应用：
```bash
streamlit run app.py
```

**说明**
> app.py: 界面更加整洁 <br>
> app2.py: 页面上有美观的logo <br>
> 两者功能一致，按个人喜好进行选择


3. 在浏览器中打开显示的地址（默认为 http://localhost:8501）

## 注意事项

- 请确保有稳定的网络连接
- API Key 请妥善保管，不要上传到公共仓库
