from quicklyRag.baseClass.PlatformBase import MySiliconflowAiConfig, MyAzureAiConfig, MyOllamaAiConfig
from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType

#默认使用的平台
default_embedding_use_platform = PlatformEmbeddingType.SILICONFLOW

# 硅基流动平台配置
MySiliconflowAiInfo = MySiliconflowAiConfig(
    base_url='https://api.siliconflow.cn/v1',
    chat_model='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    embedding_model='BAAI/bge-m3',
    key='硅基流动的key'
)

# 创建亚马逊平台gpt的配置
MyAzureAiInfo = MyAzureAiConfig(
    base_url='https://99313-mafhdjdl-eastus2.cognitiveservices.azure.com',
    chat_model='gpt-4.1',
    key='亚马逊的Key',
    api_version="2025-01-01-preview"
)

# 创建本地Ollama的配置 可以使用默认的配置
MyOllamaInfo = MyOllamaAiConfig(
    base_url='http://127.0.0.1:11434',
    chat_model='qwen3:0.6b',
    embedding_model='qwen3-embedding:0.6b'
)
