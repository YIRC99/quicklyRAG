import os
from functools import lru_cache
import dotenv
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings

# 加载环境变量
dotenv.load_dotenv()

# 实时证明 下面的注解方法更加的简单 直接创建模型实例
@lru_cache(maxsize=1)
def qwen_llm(model='Qwen/Qwen3-Omni-30B-A3B-Instruct'):
    return ChatOpenAI(
        model_name=model,
        base_url=os.getenv("OPENAI_API_BASEURL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

@lru_cache(maxsize=1)
def gpt_llm():
    return AzureChatOpenAI(
        api_key="亚马逊的Key",
        azure_endpoint="https://99313-mafhdjdl-eastus2.cognitiveservices.azure.com",
        api_version="2025-01-01-preview",
        model="gpt-4.1",
        temperature=0.7,
        max_tokens=1000
    )

@lru_cache(maxsize=1)
def flow_embedding(model='BAAI/bge-m3'):
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASEURL")
    )

@lru_cache(maxsize=1)
def ollama_embeddings(model="qwen3-embedding:0.6b"):
    return OllamaEmbeddings(model=model)

@lru_cache(maxsize=1)
def my_milvus(is_delete=True):
    return Milvus(
            embedding_function=ollama_embeddings(),
            connection_args={
                # 老版本使用host:域名 新版本使用uri加http://域名
                "uri": "99999999999",
                "port": "19530",  # Milvus默认端口
                "user": "",  # 无用户名
                "password": ""  # 无密码
            },
            collection_name="study_langchain3",  # 指定集合名称
            auto_id=True,  # 是否自动生成ID
            # enable_dynamic_field=True,  # 启用动态字段以存储任意元数据
            drop_old=is_delete,  # 是否删除已有集合   删除现有集合并重新创建
            consistency_level="Strong",
            # 添加距离度量参数
            index_params={
                "metric_type": "COSINE",  # 使用余弦相似度
                "index_type": "IVF_FLAT"
            }
        )

