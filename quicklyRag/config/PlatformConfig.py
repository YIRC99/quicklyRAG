from quicklyRag.baseClass.PlatformBase import QuicklySiliconflowAiConfig, QuicklyAzureAiConfig, QuicklyOllamaAiConfig
from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType, PlatformChatModelType

#向量化默认使用的平台
default_embedding_use_platform = PlatformEmbeddingType.SILICONFLOW
# 聊天模型默认使用的平台
default_chat_model_use_platform = PlatformChatModelType.SILICONFLOW

# 硅基流动平台配置
MySiliconflowAiInfo = QuicklySiliconflowAiConfig(
    base_url='https://api.siliconflow.cn/v1',
    chat_model='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    embedding_model='Qwen/Qwen3-Embedding-8B',
    key='硅基流动的Key'
)

# 创建亚马逊平台gpt的配置
MyAzureAiInfo = QuicklyAzureAiConfig(
    base_url='https://99313-mafhdjdl-eastus2.cognitiveservices.azure.com',
    chat_model='gpt-4.1',
    key='亚马逊的Key',
    api_version="2025-01-01-preview"
)

# 创建本地Ollama的配置 可以使用默认的配置
MyOllamaInfo = QuicklyOllamaAiConfig(
    base_url='http://127.0.0.1:11434',
    chat_model='qwen3:0.6b',
    embedding_model='qwen3-embedding:0.6b',
    key=''
)

