# quicklyRag/api/__init__.py
from pathlib import Path
from typing import Iterator

from langchain_core.runnables.utils import Output

from quicklyRag.baseClass.documentBase import RagDocumentInfo
from quicklyRag.baseClass.searchBase import VectorSearchParams
from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType, PlatformChatModelType
from quicklyRag.baseEnum.VectorEnum import VectorStorageType
from quicklyRag.chat.chatRequest.chatRequestHandler import llm_stream_chat
from quicklyRag.config.DocumentConfig import rag_document_info
from quicklyRag.config.PlatformConfig import default_embedding_use_platform, default_chat_model_use_platform
from quicklyRag.config.VectorConfig import default_embedding_database_type
from quicklyRag.vector.embedding.vectorEmbedding import vectorize_file


class QuicklyRagAPI:
    """QuicklyRag对外API门面"""

    @staticmethod
    def vectorize_file(file_path: str | Path,rag_config: RagDocumentInfo = rag_document_info,embedding_type: PlatformEmbeddingType = default_embedding_use_platform,vectorstore_type: VectorStorageType = default_embedding_database_type
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
        return vectorize_file(file_path, rag_config, embedding_type, vectorstore_type)

    @staticmethod
    def llm_stream_chat(question: str,session_id: str = None,prompt_name: str = 'system',search_params: VectorSearchParams = None,platform_type: PlatformChatModelType = default_chat_model_use_platform) -> Iterator[Output]:
        """
        调用平台的对话模型进行对话(包括向量检索和重排序)
        :param question: 问题
        :param session_id: 对话id(用于保存对话上下文记忆)
        :param prompt_name: (系统提示词名称)
        :param search_params: (向量检索参数对象)
        :param platform_type: (指定对话使用的平台配置)
        :return: 返回SSE流式响应数据
        """
        return llm_stream_chat(question, session_id, prompt_name, search_params, platform_type)


    # 提供全局实例
    __all__ = ['vectorize_file','llm_stream_chat']