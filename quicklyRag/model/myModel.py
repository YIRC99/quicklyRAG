import os
from functools import lru_cache
import dotenv
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings

from quicklyRag.config.myConfig import MyAiConfig, MyAzureAiConfig


# 事实证明 下面的注解方法更加的简单 直接创建模型实例
@lru_cache(maxsize=1)
def qwen_llm():
    return ChatOpenAI(
        model=MyAiConfig.chat_model,
        base_url=MyAiConfig.base_url,
        api_key=MyAiConfig.key,
    )

@lru_cache(maxsize=1)
def gpt_llm():
    return AzureChatOpenAI(
        api_key=MyAzureAiConfig.key,
        azure_endpoint=MyAzureAiConfig.base_url,
        api_version="2025-01-01-preview",
        model="gpt-4.1",
        temperature=0.7,
        max_tokens=1000
    )


