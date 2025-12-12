import warnings
from functools import lru_cache
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from quicklyRag.config.PlatformConfig import MySiliconflowAiInfo, MyAzureAiInfo, MyOllamaInfo


@lru_cache(maxsize=1)
def siliconflow_llm() -> ChatOpenAI:
    warnings.warn("This method is deprecated. Use QuicklyChatModelProvider instead.", DeprecationWarning)
    return ChatOpenAI(
        model=MySiliconflowAiInfo.chat_model,
        base_url=MySiliconflowAiInfo.base_url,
        api_key=MySiliconflowAiInfo.key,
    )


@lru_cache(maxsize=1)
def azure_llm() -> AzureChatOpenAI:
    warnings.warn("This method is deprecated. Use QuicklyChatModelProvider instead.", DeprecationWarning)
    return AzureChatOpenAI(
        api_key=MyAzureAiInfo.key,
        azure_endpoint=MyAzureAiInfo.base_url,
        api_version=MyAzureAiInfo.api_version,
        model=MyAzureAiInfo.chat_model,
    )

@lru_cache(maxsize=1)
def ollama_llm() -> ChatOllama:
    warnings.warn("This method is deprecated. Use QuicklyChatModelProvider instead.", DeprecationWarning)
    return ChatOllama(
        model=MyOllamaInfo.chat_model,
        base_url=MyOllamaInfo.base_url,
    )