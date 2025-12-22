import json
import time
import uuid
from pathlib import Path
from collections.abc import Iterator
from langchain.agents import create_agent
from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.utils import Output
from loguru import logger
from quicklyRag.baseClass.documentBase import RagDocumentInfo
from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType, PlatformChatModelType
from quicklyRag.chat.message.ChatSessionManager import ChatSessionManager
from quicklyRag.chat.message.chatMessage import convert_history_to_langchain_format
from quicklyRag.chat.prompt.SystemPromptManager import SystemPromptManager
from quicklyRag.baseEnum.VectorEnum import VectorStorageType
from quicklyRag.config.DocumentConfig import rag_document_info
from quicklyRag.config.PlatformConfig import default_embedding_use_platform, default_chat_model_use_platform
from quicklyRag.config.VectorConfig import default_embedding_database_type
from quicklyRag.document.loadDocument import load_document
from quicklyRag.document.spliterDocument import spliter_file
from quicklyRag.provider.QuicklyChatModelProvider import QuicklyChatModelProvider
from quicklyRag.vector.embedding.vectorEmbedding import embed_document
from quicklyRag.vector.store.vectorStore import store_vector_by_documents, search_by_scores, VectorSearchParams


# YIRC99 TODO 2025/12/20 目前添加向量化的方法写好了
# YIRC99 TODO 2025/12/20 1. 还需要对文档进行list查询, 获取向量数据库中的数据
# YIRC99 TODO 2025/12/20 2. 还需要对文档进行批量删除的方法

# 向量化文档 只需要传入一个 文件路径 即可把文档向量化到向量数据库中
def vectorize_file(
        file_path: str | Path,
        rag_config: RagDocumentInfo = rag_document_info,
        embedding_type: PlatformEmbeddingType = default_embedding_use_platform,
        vectorstore_type: VectorStorageType = default_embedding_database_type
) -> bool:
    """
    将指定文件向量化并存储到向量数据库中

    Args:
        file_path: 要处理的文件路径
        rag_config: 文档分割配置
        embedding_type: 嵌入模型类型
        vectorstore_type: 向量存储类型

    Returns:
        bool: 处理是否成功

    Raises:
        FileNotFoundError: 当指定文件不存在时
        Exception: 其他处理过程中的异常
    """
    try:
        # 验证文件路径
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error(f"文件不存在: {file_path}")
            return False

        logger.info(f"开始处理文件: {path_obj.absolute()}")
        # 1. 加载文档
        logger.info("正在加载文档...")
        document = load_document(file_path=file_path)
        logger.success(f"文档加载成功, 文档类型: {type(document)}, 文档数量: {len(document) if hasattr(document, '__len__') else 'N/A'}")

        # 2. 拆分文档
        logger.info("正在拆分文档...")
        documents = spliter_file(document, rag_config=rag_config)
        logger.success(f"文档拆分成功, 拆分段数: {len(documents)}")

        # 3. 向量化文档
        logger.info("正在进行文档向量化...")
        embeds = embed_document(documents, embedding_type)
        logger.success(f"向量化成功, 向量数量: {len(embeds)}, 向量维度: {len(embeds[0]) if embeds and len(embeds) > 0 else 'N/A'}")

        # 4. 存储向量
        logger.info("正在存储向量到数据库...")
        store_vector_by_documents(documents, vectorstore_type)
        logger.success(f"文档存储成功，共存储 {len(documents)} 个文档片段")

        logger.info(f"文件 {file_path} 处理完成")
        return True

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {file_path}, 错误详情: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时发生错误: {str(e)}")
        return False

def _format_sse(data: str | dict) -> str:
    """将数据格式化为 SSE 标准字符串"""
    if isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False)
    # SSE 规范: 以 data: 开头，双换行结尾
    return f"data: {data}\n\n"

# SSE模型流式对话
def llm_stream_chat(question: str,
                    session_id: str = None,
                    prompt_name: str = 'system',
                    search_params :VectorSearchParams = None,
                    platform_type: PlatformChatModelType = default_chat_model_use_platform) -> Iterator[Output]:
    if session_id is None:
        session_id = str(uuid.uuid4())

    # 1.查询对话记忆 先获取管理session 在根据userid获取管理器 在从管理器中获取全部对话记录
    session = ChatSessionManager(default_max_messages=100, ttl_seconds=3600 * 24 * 7)
    message_manager = session.get_session(session_id)
    logger.info(f'查询到的消息记录: {message_manager.list_messages()}')

    # 2. 转换历史记录为LangChain格式
    converted_history = convert_history_to_langchain_format(message_manager.list_messages())

    # 5. 向量检索对话 对话相关的资料
    if search_params is None:
        search_params = VectorSearchParams(query=question)
    scores = search_by_scores(search_params)
    context_str = "\n\n".join([f"<资料片段>\n\n: {res.text}\n\n<资料片段>\n\n" for res in scores])
    logger.info(f'向量检索结果: {context_str}')

    # 3.添加提示词 获取管理器对象
    # 然后默认读取系统提示词 需要给系统提示词一个名称 默认会读取SystemPromptManager类下的system.md当作系统提示词 也可以传入file_path
    system_prompt_manager = SystemPromptManager()
    system_prompt = system_prompt_manager.get_prompt(prompt_name)

    # 4.创建包含历史的提示模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "以下是检索到的参考资料，请基于这些资料回答问题：\n\n{context_str}"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    # 获取llm
    llm = QuicklyChatModelProvider(platform_type)
    chain = prompt_template | llm.chat_model | StrOutputParser()

    try:
        # 使用 yield from 返回生成器，让上层调用者可以流式接收
        full_response = ""
        for chunk in chain.stream({
            "chat_history": converted_history,
            "context_str": context_str,
            "question": question
        }):
            if chunk:
                full_response += chunk
                payload = {
                    "content": chunk,  # 当前片段
                    "session_id": session_id,
                    "status": "thinking"  # 标识正在生成
                }
                yield _format_sse(payload)

        # 8. 对话结束后，手动保存对话记录
        if full_response:
            message_manager.add_human_message(question)
            message_manager.add_ai_message(full_response)
            session.save_session(session_id)

        yield _format_sse({"content": "", "status": "done", "session_id": session_id})

    except Exception as e:
        logger.error(f"流式对话出错: {e}")
        yield f"出错啦: {e}"

if __name__ == '__main__':

    set_debug(True)  # 启用 LangChain 调试模式
    for token in llm_stream_chat(
            question='我喜欢什么你知道吗?',
            session_id='a8c67b91-f9c7-46d2-8610-bc1a8ea82e63',
    ):
        print(token, end='', flush=True)  # 实时打印每个token而不换行
    print()  # 最后换行以保证终端显示整洁

    for token in llm_stream_chat(
            question='江西省职业院校技能大赛高职组数字化设计与制造赛项有什么介绍?',
            session_id='a8c67b91-f9c7-46d2-8610-bc1a8ea82e63',
    ):
        print(token, end='', flush=True)  # 实时打印每个token而不换行
    print()  # 最后换行以保证终端显示整洁






