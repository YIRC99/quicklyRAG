from typing import Optional


class MyAiConfig:
    def __init__(
        self,
        base_url: Optional[str]='https://api.siliconflow.cn/v1',
        chat_model: Optional[str]='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        embedding_model: Optional[str]='BAAI/bge-m3',
        key: Optional[str]='硅基流动的key',
        **kwargs
    ):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.key = key
        # 动态设置额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)
