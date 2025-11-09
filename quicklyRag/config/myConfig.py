from typing import Optional

class myAiConfig:
    def __init__(
        self,
        base_url: Optional[str]='https://api.siliconflow.cn/v1',
        chat_model: Optional[str]='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        embedding_model: Optional[str]='BAAI/bge-m3',
        key: Optional[str]='硅基流动的key'
    ):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.key = key

# 使用默认值创建实例  默认是轨迹流动
MyAiConfig = myAiConfig()

# 创建亚马逊平台gpt的配置
MyAzureAiConfig = myAiConfig(
    base_url='https://99313-mafhdjdl-eastus2.cognitiveservices.azure.com',
    chat_model='gpt-4.1',
    key='亚马逊的Key'
)

# 创建本地Ollama的配置
MyOllamaConfig = myAiConfig(
    base_url='http://127.0.0.1:11434',
    embedding_model='qwen3-embedding:0.6b'
)
