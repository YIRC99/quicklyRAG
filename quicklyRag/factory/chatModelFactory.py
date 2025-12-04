from functools import lru_cache
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from quicklyRag.config.PlatformConfig import MySiliconflowAiInfo, MyAzureAiInfo, MyOllamaInfo


@lru_cache(maxsize=1)
def siliconflow_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MySiliconflowAiInfo.chat_model,
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

