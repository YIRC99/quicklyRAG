import os
from functools import lru_cache
import dotenv
from langchain.chat_models import init_chat_model
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings

from quicklyRag.config.MyConfig import MySiliconflowAiInfo, MyAzureAiInfo, MyOllamaInfo


@lru_cache(maxsize=1)
def siliconflow_llm():
    return ChatOpenAI(
        model=MySiliconflowAiInfo.chat_model,
        base_url=MySiliconflowAiInfo.base_url,
        api_key=MySiliconflowAiInfo.key,
    )

@lru_cache(maxsize=1)
def azure_llm():
    return AzureChatOpenAI(
        api_key=MyAzureAiInfo.key,
        azure_endpoint=MyAzureAiInfo.base_url,
        api_version=MyAzureAiInfo.api_version,
        model=MyAzureAiInfo.chat_model,
        temperature=0.7,
        max_tokens=1000
    )

@lru_cache(maxsize=1)
def ollama_llm():
    return ChatOllama(
        model=MyOllamaInfo.chat_model,
        base_url=MyAzureAiInfo.base_url,
    )

