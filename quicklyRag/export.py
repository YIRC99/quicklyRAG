import time
import uuid
from pathlib import Path

from langchain.agents import create_agent
from loguru import logger

from quicklyRag.baseClass.documentBase import RagDocumentInfo
from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType, PlatformChatModelType
from quicklyRag.chat.message.ChatSessionManager import ChatSessionManager
from quicklyRag.chat.propmt.SystemPromptManager import SystemPromptManager
from quicklyRag.baseEnum.VectorEnum import VectorStorageType
from quicklyRag.config.DocumentConfig import rag_document_info
from quicklyRag.config.PlatformConfig import default_embedding_use_platform
from quicklyRag.config.VectorConfig import default_embedding_database_type
from quicklyRag.document.loadDocument import load_document
from quicklyRag.document.spliterDocument import spliter_file
from quicklyRag.provider.QuicklyChatModelProvider import QuicklyChatModelProvider


from quicklyRag.vector.embedding.vectorEmbedding import embed_document
from quicklyRag.vector.store.vectorStore import store_vector_by_documents


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




# TODO 模型流式对话
def llm_stream_chat(question: str,
                    user_id: str = uuid.uuid4(),
                    prompt_name: str = 'system'):
    # 1.查询对话记忆 先获取管理session 在根据userid获取管理器 在从管理器中获取全部对话记录
    session = ChatSessionManager(db_path=user_id + '.db', default_max_messages=100, ttl_seconds=3600 * 24 * 7)
    message_manager = session.get_session(user_id)
    logger.info(f'查询到的消息记录: {message_manager.list_messages()}')

    # 2.添加提示词 获取管理器对象
    # 然后默认读取系统提示词 需要给系统提示词一个名称 默认会读取SystemPromptManager类下的system.md当作系统提示词 也可以传入file_path
    system_prompt_manager = SystemPromptManager()
    system_prompt = system_prompt_manager.get_prompt(prompt_name)
    logger.info(f'系统提示词: {system_prompt}')

    # 将问题进行检索召回



    # llm = QuicklyChatModelProvider(PlatformChatModelType.SILICONFLOW)
    # 添加对话记忆
    # 创建llm
    # 流式对话返回
    # 实时流式返回后处理

    # agent = create_agent(model=llm.chat_model)
    # invoke = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # agent.invoke(invoke)
    # print(invoke)
    return user_id


if __name__ == '__main__':

    # print(llm_stream_chat(question='你好, 你是谁',
    #                       user_id='a8c67b91-f9c7-46d2-8610-bc1a8ea82e63',
    #                       prompt_name='system2'))
    print()
