from functools import lru_cache
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.baseModel.embeddingBase import QuicklyEmbeddingModel
from quicklyRag.config.PlatformConfig import MySiliconflowAiInfo, MyAzureAiInfo, MyOllamaInfo


@lru_cache(maxsize=1)
def siliconflow_embed2() -> QuicklyEmbeddingModel:
    return QuicklyEmbeddingModel(PlatformEmbeddingType.SILICONFLOW)

@lru_cache(maxsize=1)
def siliconflow_embed() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=MySiliconflowAiInfo.embedding_model,
        base_url=MySiliconflowAiInfo.base_url,
        api_key=MySiliconflowAiInfo.key,
    )

@lru_cache(maxsize=1)
def ollama_embed() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=MyOllamaInfo.embedding_model,
        base_url=MyAzureAiInfo.base_url,
    )
@lru_cache(maxsize=1)
def ollama_embed2() -> QuicklyEmbeddingModel:
    return QuicklyEmbeddingModel(PlatformEmbeddingType.OLLAMA)
