from typing import Optional


class MyOllamaAiConfig:
    def __init__(
        self,
        base_url: Optional[str]='http://127.0.0.1:11434',
        chat_model: Optional[str]='qwen3:0.6b',
        embedding_model: Optional[str]='qwen3-embedding:0.6b',
        **kwargs
    ):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        for key, value in kwargs.items():
            setattr(self, key, value)