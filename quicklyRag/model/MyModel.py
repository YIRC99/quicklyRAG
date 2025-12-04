import os
from functools import lru_cache
import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI

from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.baseModel.embeddingBase import QuicklyEmbeddingModel
from quicklyRag.config.PlatformConfig import MySiliconflowAiInfo, MyAzureAiInfo, MyOllamaInfo


@lru_cache(maxsize=1)
def siliconflow_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MySiliconflowAiInfo.chat_model,
        base_url=MySiliconflowAiInfo.base_url,
        api_key=MySiliconflowAiInfo.key,
    )

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
def azure_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        api_key=MyAzureAiInfo.key,
        azure_endpoint=MyAzureAiInfo.base_url,
        api_version=MyAzureAiInfo.api_version,
        model=MyAzureAiInfo.chat_model,
    )

@lru_cache(maxsize=1)
def ollama_llm() -> ChatOllama:
    return ChatOllama(
        model=MyOllamaInfo.chat_model,
        base_url=MyAzureAiInfo.base_url,
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
