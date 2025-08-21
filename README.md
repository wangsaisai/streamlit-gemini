# Streamlit + Google Gemini 聊天程序

基于 Streamlit 和 Google Gemini 构建的智能聊天应用，支持文本对话和图片分析功能。

## 🛠️ 技术栈 (Technical Stack)

### 核心框架 (Core Framework)
- **[Streamlit](https://streamlit.io/)** - Python Web 应用框架
  - 提供直观的用户界面组件
  - 支持实时交互和状态管理
  - 快速部署和原型开发

### AI/ML 集成 (AI/ML Integration)
- **[Google Gemini API](https://ai.google.dev/)** - 先进的大语言模型
  - `gemini-2.0-flash-exp` - 最新实验版本，速度快
  - `gemini-exp-1206` - 实验版本，功能增强
  - `gemini-2.0-flash-thinking-exp` - 思维链推理模型
  - `gemini-2.5-pro` / `gemini-2.5-flash` - 生产版本
  - `gemini-1.5-pro-latest` / `gemini-1.5-flash-latest` - 稳定版本

### Python 依赖库 (Python Dependencies)
```python
# 核心依赖
streamlit              # Web 应用框架
google-generativeai    # Google Gemini SDK (传统版本)
google-genai          # Google Gemini SDK (新版本)

# 图像处理
Pillow                # 图像处理库

# 语言处理
langdetect            # 语言检测，支持中英文自动识别

# 协议和数据
protobuf              # Protocol Buffers，Google API 通信协议
```

### 应用架构 (Application Architecture)

#### 多应用变体 (Multiple App Variants)
1. **`app.py`** - 基础版本
   - 简洁的用户界面
   - 基本的文本对话和图片分析功能

2. **`app2.py`** - 美化版本
   - 带有美观 Logo 的界面
   - 功能与 app.py 一致

3. **`app_computer.py`** - 增强版本
   - 计算机专家模式
   - 中英文翻译功能
   - 仔细检查模式

4. **`app_computer_multi_stream.py`** - 多模型流式版本
   - 支持多种 Gemini 模型选择
   - 流式输出响应
   - 增强的用户体验

5. **`search.py`** - 搜索增强版本
   - 集成 Google 搜索功能
   - 实时信息获取
   - 书籍分析模式

6. **`image.py`** - 多媒体生成版本
   - 文本生成图片 (Imagen 3)
   - 图片编辑和处理
   - 文本生成视频 (VEO)

### 核心功能模块 (Core Modules)

#### 对话管理 (Chat Management)
- 会话状态管理 (`st.session_state`)
- 历史记录保存和显示
- 多轮对话上下文保持

#### 模型配置 (Model Configuration)
- 温度参数调节 (Temperature)
- 最大 Token 数量控制
- 流式输出开关
- 模型动态切换

#### 图像处理 (Image Processing)
- 图片上传和预处理
- 多格式支持 (JPG, PNG)
- 图片分析和描述生成
- 图像编辑功能

#### 语言处理 (Language Processing)
- 自动语言检测
- 中英文双向翻译
- 多语言提示词管理

#### 搜索增强 (Search Enhancement)
- Google 搜索 API 集成
- 实时信息获取
- 搜索结果整合到对话中

### 系统要求 (System Requirements)
- **Python**: 3.8+ (推荐 3.9+)
- **内存**: 最小 2GB RAM (推荐 4GB+)
- **网络**: 稳定的互联网连接 (访问 Google API)
- **API**: Google AI API Key (Gemini 访问权限)

### 部署架构 (Deployment Architecture)
```
用户浏览器 (User Browser)
       ↓
Streamlit Web 服务器 (Streamlit Web Server)
       ↓
Google Gemini API (AI 模型服务)
       ↓
Google 搜索 API (可选，搜索功能)
```

### 技术特色 (Technical Features)
- **响应式设计**: 适配不同屏幕尺寸
- **实时流式输出**: 类似 ChatGPT 的打字机效果
- **状态管理**: 自动保存会话历史
- **错误处理**: 完善的异常捕获和用户提示
- **多模态支持**: 文本 + 图像 + 视频处理
- **国际化**: 支持中英文双语界面

### 技术选型说明 (Technology Choice Rationale)

#### 为什么选择 Streamlit?
- ✅ **快速开发**: 纯 Python 语法，无需前端知识
- ✅ **组件丰富**: 内置聊天、文件上传、侧边栏等组件
- ✅ **部署简单**: 一行命令即可启动 Web 应用
- ✅ **交互性强**: 支持实时更新和状态管理

#### 为什么选择 Google Gemini?
- ✅ **多模态能力**: 同时处理文本、图像、视频
- ✅ **模型多样**: 从快速响应到深度思考的多种选择
- ✅ **API 稳定**: Google 提供的企业级 API 服务
- ✅ **功能先进**: 支持最新的 AI 技术和功能

#### 为什么使用多个应用文件?
- ✅ **模块化设计**: 每个文件专注特定功能
- ✅ **易于维护**: 独立开发和测试不同功能
- ✅ **用户选择**: 根据需求选择合适的版本
- ✅ **学习友好**: 从简单到复杂的渐进式学习

## 功能特点

- 💬 支持自然语言对话
- 🤖 支持多种模型选择 (`gemini-2.0-flash`, `gemini-2.5-exp`, `gemini-2.5-flash`)
- 📝 支持流式输出回复
- 🖼️ 支持图片上传和分析,生成
- 🔍 支持 Google 搜索功能，对话时可获取最新的知识
- 🚀 简单易用的 Web 界面
- 💻 基于 Streamlit 快速部署

## 使用前准备

1. 获取 Google Gemini API Key
2. 安装必要的依赖包：
```python
pip install streamlit google-generativeai google-genai pillow
```

## 📁 项目结构 (Project Structure)

```
streamlit-gemini/
├── app.py                          # 基础聊天应用 (推荐新手使用)
├── app2.py                         # 美化版聊天应用 (带 Logo)
├── app_computer.py                 # 计算机专家版本
├── app_computer_multi.py           # 多模型支持版本
├── app_computer_multi_stream.py    # 多模型流式输出版本 (功能最全)
├── search.py                       # 搜索增强版本 (集成 Google 搜索)
├── image.py                        # 多媒体生成版本 (图片/视频生成)
├── prompts.py                      # 提示词模板管理
├── requirements.txt                # Python 依赖列表
├── README.md                       # 项目文档
├── .gitignore                      # Git 忽略文件配置
└── images/                         # 界面截图和演示图片
    ├── app-display.png
    ├── search-display1.png
    └── ...
```

### 文件功能说明 (File Descriptions)

| 文件名 | 主要功能 | 适用场景 |
|--------|----------|----------|
| `app.py` | 基础对话 + 图片分析 | 简单使用，界面简洁 |
| `app2.py` | 基础对话 + 美观界面 | 演示展示，界面美观 |
| `app_computer.py` | 专家模式 + 翻译功能 | 技术问答，多语言支持 |
| `app_computer_multi_stream.py` | 多模型 + 流式输出 | 高级用户，功能最全 |
| `search.py` | 搜索增强 + 实时信息 | 需要最新信息的对话 |
| `image.py` | 图片生成 + 视频创作 | 创意内容生成 |
| `prompts.py` | 提示词管理 | 所有应用的提示词模板 |

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

# reset port
streamlit run --server.port 8080 app.py
```

**说明**
> app.py: 界面更加整洁 <br>
> app2.py: 页面上有美观的logo <br>
> 两者功能一致，按个人喜好进行选择


3. 在浏览器中打开显示的地址（默认为 http://localhost:8501 ）

#### 效果图:
image.py展示 (体验地址：http://34.125.51.214:8086/)
![image.py展示1](images/image-display.png)
search.py展示
![search.py展示1](images/search-display1.png)
![search.py展示2](images/search-display2.png)
app_computer_multi_stream.py展示
![最新界面展示](images/app_computer_multi_stream-display.png)
app.py展示
![app.py展示](images/app-display.png)
app2.py展示
![app2.py展示](images/app2-display.png)

4. 体验地址
http://34.125.51.214:8082/ （免费，无广告，无需注册登录等）

## 注意事项

- 请确保有稳定的网络连接
- API Key 请妥善保管，不要上传到公共仓库
