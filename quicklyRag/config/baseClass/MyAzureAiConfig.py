from typing import Optional

from quicklyRag.config.baseClass.MyAiConfig import MyAiConfig


class MyAzureAiConfig(MyAiConfig):
    api_version: Optional[str] = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
