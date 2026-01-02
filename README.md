<div align="center">
  <h1>⚡ QuicklyRag</h1>
  <p>
    <strong>Python 项目的 AI “瑞士军刀” —— 一行代码，构建 RAG 知识库</strong>
  </p>

  <p>
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python">
    </a>
    <a href="https://docs.pydantic.dev/">
        <img src="https://img.shields.io/badge/Pydantic-v2.0+-e92063.svg?style=flat-square&logo=pydantic&logoColor=white" alt="Pydantic">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License">
    </a>
  </p>

  <p>
    <a href="#-核心特性">核心特性</a> •
    <a href="#-快速开始">快速开始</a> •
    <a href="#-项目架构">项目架构</a> •
  </p>
</div>

---

## 📖 简介

**QuicklyRag** 是一个轻量级、嵌入式的 RAG（检索增强生成）解决方案。

现有的 RAG 框架（如 n8n, Dify, Coze）往往过于庞大，或者作为独立服务部署，增加了系统的复杂度和内存开销。**QuicklyRag** 的设计灵感来源于 Java 的 `HuTool` —— **它不是一个框架，而是一个工具包**。

你只需要将源码文件夹拖入你的 Python 项目，即可通过极简的 API，为你的应用赋予构建本地知识库和智能对话的能力。

### 🎯 适用场景
* 现有 Python 项目急需接入 AI 对话能力，但不想引入复杂的微服务架构。
* 需要完全掌控 RAG 流程（数据清洗、切分、向量化），便于深度二次开发。
* 对内存占用敏感，追求极简的运行环境。

---

## ✨ 核心特性

- **🧩 零侵入嵌入式架构**：文件夹即组件，拒绝臃肿，即插即用。
- **🛡️ 极致的类型安全**：基于 `Pydantic` 重构，提供完整的代码提示与参数校验，开发体验丝般顺滑。
- **🔌 插件化服务提供者**：
    - **LLM**: 支持 OpenAI, SiliconFlow (硅基流动) 等多种模型。
    - **Vector Store**: 支持 Milvus, Chroma 等向量数据库。
    - **Embedding**: 支持多种嵌入模型切换。
- **⚡ 极简 API 设计**：封装了文档切分、向量化、混合检索（Hybrid Search）、重排（Rerank）等复杂链路，对外仅暴露清爽接口。

---

## 🚀 快速开始

### 1. 获取项目
```bash
git clone [https://github.com/zhangyuanyuan/QuicklyRag.git](https://github.com/zhangyuanyuan/QuicklyRag.git)
```

### 2. 集成到你的项目
将 quickly_rag 文件夹直接复制到你的项目根目录下即可。

### 3. 安装依赖
推荐使用现代化的 uv 包管理器（也可以使用 pip）：
```bash
# 使用 uv (推荐)
uv sync

# 或者使用 pip
pip install -r requirements.txt
```
### 4. 修改配置文件
配置文件位于 `quickly_rag/config/` 目录下。
建议全部阅读一下
### 5. Hello World
在你的代码中引入 API 即可开始使用：
```bash
# main.py
from quickly_rag import api

# 1. 直接调用方法即可, 第一次调用由于会初始化会稍微耗时
# 2. 上传文档建立知识库 (支持 PDF, MD, TXT 等)
# 系统会自动进行切分、向量化并存储
api.vectorize_file("./data/产品手册.pdf")

# 3. 开始流式对话 (支持上下文记忆)
# 系统会自动进行：问题重写 -> 混合检索 -> Rerank重排 -> LLM生成
response_generator = api.llm_chat(
    question="我们的产品有哪些核心优势？",
    session_id="user_123"
)

print("AI 回复: ", end="")
for chunk in response_generator:
    print(chunk, end="", flush=True)
```

### 🏗️ 项目架构
本项目采用现代化 Python 工程结构，严格遵循 Snake Case 命名规范与 Facade 门面模式。
```bash
├── chat.db       默认持久化 保存对话记录 sqlite 数据库
├── main.py       fastAPI 快速接入示例
├── pyproject.toml       python版本
├── quickly_rag       需要复制的项目文件夹
|  ├── api.py       api入口门面
|  ├── chat       对话相关功能模块
|  |  ├── chatRequest       对话请求处理模块
|  |  |  ├── chat_request_handler.py
|  |  ├── message       对话消息模块
|  |  |  ├── chat_message.py
|  |  |  ├── chat_message_manager.py
|  |  |  ├── chat_session_manager.py
|  |  └── prompt       提示词管理模块
|  |     ├── system.md
|  |     ├── system_prompt_manager.py
|  ├── config       项目整体配置模块
|  |  ├── chat_config.py
|  |  ├── document_config.py
|  |  ├── platform_config.py
|  |  ├── vector_config.py
|  ├── core       基类
|  |  ├── document_base.py
|  |  ├── Platform_base.py
|  |  ├── search_base.py
|  |  ├── Vector_base.py
|  ├── document       文档处理模块
|  |  ├── document_loader.py
|  |  ├── document_splitter.py
|  ├── enums       枚举模块
|  |  ├── platform_enum.py
|  |  ├── vector_enum.py
|  ├── provider       服务提供者模块
|  |  ├── chat_model_provider.py
|  |  ├── embedding_model_provider.py
|  |  ├── reranker_provider.py
|  |  ├── vector_store_provider.py
|  ├── vector       向量相关功能模块
|  |  ├── embedding       向量化模块
|  |  |  ├── vector_embedding.py
|  |  └── store       向量存储
|  |     ├── milvus_util.py
|  |     ├── vector_store.py
|  ├── __init__.py
├── README.md
├── requirements.txt       项目依赖
├── test.py       工具调用示例
├── uv.lock       uv包管理器依赖文件
```
### ⚙️ 配置指南
所有配置文件位于 quickly_rag/config/ 目录下。


|  配置文件 | 作用 |
|---|---|
| platform_config.py | LLM 平台配置 |
| vector_config.py | 向量数据库 |
| chat_config.py | 对话参数 |
| document_config.py | 文档处理 |

### 💡 提示: 
项目默认使用 `SQLite (chat.db)` 存储会话历史，默认使用 `Milvus` 作为向量存储。你可以在 provider/ 目录下轻松扩展其他实现。

### 🤝 贡献
QuicklyRag 秉持“简单至上”的原则。如果你有更好的想法，欢迎提交 Issue 或 Pull Request。

1. Fork 本仓库

2. 新建 Feat_xxx 分支

3. 提交代码

4. 新建 Pull Request
### 任何问题欢迎联系作者
<img src="./img/weixin.jpg" style="height: 300px"/>

<p align="center"> Made with ❤️ by QuicklyRag Team </p>
