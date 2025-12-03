from typing import Optional

class MySiliconflowAiConfig():
    def __init__(
            self,
            base_url: Optional[str],
            chat_model: Optional[str],
            key: Optional[str],
            embedding_model: Optional[str] = None,
            **kwargs
    ):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.key = key
        # 动态设置额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)

class MyAzureAiConfig():
    def __init__(
            self,
            base_url: Optional[str],
            chat_model: Optional[str],
            api_version: Optional[str],
            key: Optional[str],
            embedding_model: Optional[str] = None,
            **kwargs
    ):
        self.base_url = base_url
        self.chat_model = chat_model
        self.api_version = api_version
        self.embedding_model = embedding_model
        self.key = key
        # 动态设置额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)

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