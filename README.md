# RAG 知识库问答系统 (RAG Knowledge Base)

这是一个基于 RAG（检索增强生成）技术的本地知识库问答系统。项目实现了从文档上传、向量化存储到基于大模型智能回答的完整流程，并配备了简洁的前端交互界面。

## 核心功能

- **文档解析**：自动处理上传的文档，进行清洗与切片。
- **向量检索**：使用 ChromaDB 本地向量数据库，高效存储和检索知识片段。
- **智能问答**：结合通义千问（DashScope）大模型，基于检索到的上下文生成精准回答。
- **Web 交互**：提供基于原生 HTML/JS 的前端界面，操作便捷。

## 技术栈

- **后端框架**: Python, LangChain
- **向量数据库**: ChromaDB
- **大模型服务**: Alibaba DashScope (通义千问)
- **前端**: HTML5 / CSS3 / JavaScript

## 快速开始

### 1. 环境准备
确保已安装 Python 3.8+，并安装所需依赖：

```bash
pip install langchain chromadb openai dashscope python-dotenv
```

### 2. 配置 API Key（重要）
为了安全起见，API Key 不直接写在代码中。请在项目根目录创建 `.env` 文件，内容如下：

```env
DASHSCOPE_API_KEY=sk-your-api-key-here
```

注：`.env` 文件已被 `.gitignore` 忽略，不会上传到 GitHub。

### 3. 运行项目
执行主程序启动服务：

```bash
python main.py
```

启动后，在浏览器访问终端显示的地址（通常为 http://127.0.0.1:8000）即可开始使用。

## 项目结构说明

- `main.py`：入口文件，负责启动 Web 服务器。
- `rag_pipeline.py`：核心逻辑，包含 LLM 调用链、ChromaDB 初始化及检索逻辑。
- `document_processor.py`：数据处理，负责文档的加载、分割与向量化。
- `应用/静态/`：前端资源，包含用户交互界面（index.html）。
- `.gitignore`：安全配置，自动过滤敏感文件与缓存。

## 安全提示

本项目已配置 `.gitignore` 规则，请勿手动将 `.env` 文件或 `chroma_db` 数据库文件上传至公共仓库，以免泄露敏感信息。
