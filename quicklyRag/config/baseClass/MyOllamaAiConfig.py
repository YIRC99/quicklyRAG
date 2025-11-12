from typing import Optional


class MyOllamaAiConfig:
    def __init__(
            self,
            base_url: Optional[str],
            chat_model: Optional[str],
            embedding_model: Optional[str],
            key: Optional[str] = None,
            **kwargs
    ):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.key = key
        for key, value in kwargs.items():
            setattr(self, key, value)
