from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.config.PlatformConfig import MySiliconflowAiInfo, MyOllamaInfo


class QuicklyEmbeddingModelFactory(Embeddings):
    """
    封装不同平台的嵌入模型，提供统一的 embedding_text 接口。
    """

    def __init__(self, platform_type: PlatformEmbeddingType):
        self.platform_type = platform_type
        # 获取对应平台的原始 LangChain Embeddings 对象
        self._embeddings_model: Embeddings = self._get_embedding_model_instance(platform_type)

    # 新增：创建嵌入模型实例的统一入口
    def _get_embedding_model_instance(self, platform_type: PlatformEmbeddingType) -> Embeddings:
        if platform_type == PlatformEmbeddingType.SILICONFLOW:
            return self.__siliconflow_embed()
        elif platform_type == PlatformEmbeddingType.AZURE:
            raise NotImplementedError("Azure embedding model factory not implemented yet.")
        elif platform_type == PlatformEmbeddingType.OLLAMA:
            return self.__ollama_embed()
        else:
            raise ValueError(f"Unsupported platform type: {platform_type}")

    @staticmethod
    @lru_cache(maxsize=1)
    def __siliconflow_embed() -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=MySiliconflowAiInfo.embedding_model,
            base_url=MySiliconflowAiInfo.base_url,
            api_key=MySiliconflowAiInfo.key,
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def __ollama_embed() -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=MyOllamaInfo.embedding_model,
            base_url=MyOllamaInfo.base_url,
        )

    def embedding_text(self, texts: str | list[str] | list[Document]) -> list[list[float]]:
        """
        对输入文本进行向量化。
        Args:
            texts (str | list[str] | list[Document]): 单个文本字符串、文本列表或 Document 对象列表。
        Returns:
            list[list[float]]: 对应的嵌入向量列表。
        """
        try:
            processed_texts = []  # 用于存储最终要向量化的字符串

            if isinstance(texts, str):
                processed_texts = [texts]
            elif isinstance(texts, list):
                if not texts:
                    return []
                # 检查列表内容并提取文本
                for item in texts:
                    if isinstance(item, Document):
                        processed_texts.append(item.page_content)
                    elif isinstance(item, str):
                        processed_texts.append(item)
                    else:
                        raise TypeError(
                            f"All items in the list must be strings or Document objects. Found: {type(item)}")
            else:
                raise TypeError(
                    "Input 'texts' must be either a string, a list of strings, or a list of Document objects.")

            # 调用底层模型的 embed_documents 方法，传入处理好的字符串列表
            embeddings = self._embeddings_model.embed_documents(processed_texts)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding error: {e}", exc_info=True)
            raise  # 重新抛出异常，让调用者知道发生了错误

    # 实现 LangChain Embeddings 接口
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings_model.embed_documents(texts)

    # 实现 LangChain Embeddings 接口
    def embed_query(self, text: str) -> list[float]:
        return self._embeddings_model.embed_query(text)