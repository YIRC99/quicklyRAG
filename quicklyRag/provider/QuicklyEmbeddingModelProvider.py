import traceback
from functools import lru_cache
from httpx import ConnectError, TimeoutException
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from pydantic import BaseModel, Field

from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.config.PlatformConfig import MySiliconflowAiInfo, MyOllamaInfo


class QuicklyEmbeddingModelProvider(Embeddings,BaseModel):
    """
    封装不同平台的嵌入模型，提供统一的 embedding_text 接口。
    """
    platform_type: PlatformEmbeddingType = Field(..., description="指定要使用的嵌入模型平台类型")

    def __init__(self, platform_type: PlatformEmbeddingType, /, **data):
        super().__init__(platform_type=platform_type, **data)
        self._embeddings_model: Embeddings | None = None
        try:
            self._embeddings_model: Embeddings = self._get_embedding_model_instance(platform_type)
        except Exception as e:
             logger.error(f"Unexpected error during initialization for platform {platform_type.name}: {e}")
             raise

    def _get_embedding_model_instance(self, platform_type: PlatformEmbeddingType) -> Embeddings:
        try:
            if platform_type == PlatformEmbeddingType.SILICONFLOW:
                return self.__siliconflow_embed()
            elif platform_type == PlatformEmbeddingType.AZURE:
                raise NotImplementedError("Azure embedding model provider not implemented yet.")
            elif platform_type == PlatformEmbeddingType.OLLAMA:
                return self.__ollama_embed()
            else:
                raise ValueError(f"Unsupported platform type: {platform_type}")
        except NotImplementedError:
            raise
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model for {platform_type.name}: {e}")
            raise

    @staticmethod
    @lru_cache(maxsize=1)
    def __siliconflow_embed() -> OpenAIEmbeddings:
        try:
            model = OpenAIEmbeddings(
                model=MySiliconflowAiInfo.embedding_model,
                base_url=MySiliconflowAiInfo.base_url,
                api_key=MySiliconflowAiInfo.key,
            )
            logger.info("SiliconFlow embedding model initialized successfully.")
            return model
        except Exception as e:
            logger.error(f"Error creating SiliconFlow embedding model: {e}")
            raise

    @staticmethod
    @lru_cache(maxsize=1)
    def __ollama_embed() -> OllamaEmbeddings:
        try:
            model = OllamaEmbeddings(
                model=MyOllamaInfo.embedding_model,
                base_url=MyOllamaInfo.base_url,
            )
            logger.info("Ollama embedding model initialized successfully.")
            return model
        except Exception as e:
             error_msg = f"Error creating Ollama embedding model: {e}"
             if isinstance(e, (ConnectError, TimeoutException)):
                 logger.warning(error_msg)
             else:
                 logger.error(error_msg)
             raise

    def embedding_text(self, texts: str | list[str] | list[Document]) -> list[list[float]]:
        """
        对输入文本进行向量化。
        Args:
            texts (str | list[str] | list[Document]): 单个文本字符串、文本列表或 Document 对象列表。
        Returns:
            list[list[float]]: 对应的嵌入向量列表。
        """
        if not hasattr(self, '_embeddings_model') or self._embeddings_model is None:
             logger.error("Embedding model is not available.")
             raise RuntimeError("Embedding model failed to initialize or is unavailable.")

        try:
            processed_texts = []

            if isinstance(texts, str):
                processed_texts = [texts]
            elif isinstance(texts, list):
                if not texts:
                    return []
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

            # 对于 SiliconFlow 平台，需要分批处理，每批最多32个
            if self.platform_type == PlatformEmbeddingType.SILICONFLOW:
                batch_size = 256
                all_embeddings = []
                
                # 分批处理文档
                for i in range(0, len(processed_texts), batch_size):
                    batch_texts = processed_texts[i:i + batch_size]
                    logger.info(f"处理文档批次 {i//batch_size + 1}/{(len(processed_texts)-1)//batch_size + 1}，包含 {len(batch_texts)} 个文档")
                    batch_embeddings = self._embeddings_model.embed_documents(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    
                return all_embeddings
            else:
                # 其他平台按原有方式处理
                embeddings = self._embeddings_model.embed_documents(processed_texts)
                return embeddings
                
        except Exception as e:
            logger.error(f"Embedding error occurred: {str(e)}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not hasattr(self, '_embeddings_model') or self._embeddings_model is None:
             raise RuntimeError("Embedding model is not initialized.")
        return self._embeddings_model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if not hasattr(self, '_embeddings_model') or self._embeddings_model is None:
             raise RuntimeError("Embedding model is not initialized.")
        return self._embeddings_model.embed_query(text)
