from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.config.PlatformConfig import MySiliconflowAiInfo, MyOllamaInfo

class QuicklyEmbeddingModel:
    """
    封装不同平台的嵌入模型，提供统一的 embedding_text 接口。
    """
    def __init__(self, platform_type: PlatformEmbeddingType):
        self.platform_type = platform_type
        # 获取对应平台的原始 LangChain Embeddings 对象
        self._embeddings_model: Embeddings = self._get_embedding_model_instance(platform_type)

    # 新增：创建嵌入模型实例的统一入口
    def _get_embedding_model_instance(self,platform_type: PlatformEmbeddingType) -> Embeddings:
        if platform_type == PlatformEmbeddingType.SILICONFLOW:
            return self.__siliconflow_embed()
        elif platform_type == PlatformEmbeddingType.AZURE:
            raise NotImplementedError("Azure embedding model factory not implemented yet.")
        elif platform_type == PlatformEmbeddingType.OLLAMA:
            return self.__ollama_embed()
        else:
            raise ValueError(f"Unsupported platform type: {platform_type}")

    def __siliconflow_embed(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=MySiliconflowAiInfo.embedding_model,
            base_url=MySiliconflowAiInfo.base_url,
            api_key=MySiliconflowAiInfo.key,
        )

    def __ollama_embed(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=MyOllamaInfo.embedding_model,
            base_url=MyOllamaInfo.base_url,
        )

    def embedding_text(self, texts: str | list[str]) -> list[list[float]]:
        try:
            # LangChain 的 Embeddings.embed_documents 方法通常处理列表
            if isinstance(texts, str):
                embeddings = self._embeddings_model.embed_documents([texts])
                return embeddings[0] if embeddings else []
            elif isinstance(texts, list):
                if not all(isinstance(t, str) for t in texts):
                    raise TypeError("All items in the text list must be strings.")
                return self._embeddings_model.embed_documents(texts)
            else:
                raise TypeError("Input 'texts' must be either a string or a list of strings.")
        except Exception as e:
            print(f"Error occurred during embedding with {self.platform_type}: {e}")
            raise  # 重新抛出异常，让调用者知道发生了错误



