from functools import lru_cache
from httpx import ConnectError, TimeoutException
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from quicklyRag.baseEnum.PlatformEnum import PlatformChatModelType
from quicklyRag.config.PlatformConfig import MySiliconflowAiInfo, MyAzureAiInfo, MyOllamaInfo


class ChatModelInitializationError(Exception):
    """当聊天模型初始化失败时抛出的异常"""
    pass


class QuicklyChatModelProvider(BaseModel):
    """
    封装不同平台的聊天模型，提供统一的聊天模型接口。
    """
    platform_type: PlatformChatModelType = Field(..., description="指定要使用的聊天模型平台类型")

    def __init__(self, platform_type: PlatformChatModelType, /, **data):
        """
        初始化 QuicklyChatModelProvider 实例。
        Args:
            platform_type (PlatformChatModelType): 指定要使用的聊天模型平台类型。
        Raises:
            ChatModelInitializationError: 如果指定平台的聊天模型初始化失败。
            ValueError: 如果 platform_type 不受支持。
        """
        super().__init__(platform_type=platform_type, **data)
        self._chat_model: BaseChatModel | None = None
        try:
            self._chat_model = self._get_chat_model_instance(platform_type)
        except ChatModelInitializationError:
            logger.error(f"Critical failure initializing chat model for {platform_type.name}. Provider is unusable.")
            raise
        except Exception as e:
             logger.error(f"Unexpected error during initialization for platform {platform_type.name}: {e}")
             raise ChatModelInitializationError(f"Unexpected error initializing {platform_type.name}") from e

    def _get_chat_model_instance(self, platform_type: PlatformChatModelType) -> BaseChatModel:
        """根据平台类型获取对应的聊天模型实例"""
        try:
            if platform_type == PlatformChatModelType.SILICONFLOW:
                return self.__siliconflow_chat()
            elif platform_type == PlatformChatModelType.AZURE:
                return self.__azure_chat()
            elif platform_type == PlatformChatModelType.OLLAMA:
                return self.__ollama_chat()
            else:
                raise ValueError(f"Unsupported chat model platform type: {platform_type}")
        except ChatModelInitializationError:
            raise
        except ValueError:
            raise
        except Exception as e:
             logger.error(f"Unexpected error getting chat model instance for {platform_type.name}: {e}")
             raise ChatModelInitializationError(
                 f"Unexpected error while getting instance for {platform_type.name}"
             ) from e

    @staticmethod
    @lru_cache(maxsize=1)
    def __siliconflow_chat() -> ChatOpenAI:
        """【内部】创建并返回 SiliconFlow 聊天模型实例 (单例)"""
        try:
            model = ChatOpenAI(
                model=MySiliconflowAiInfo.chat_model,
                base_url=MySiliconflowAiInfo.base_url,
                api_key=MySiliconflowAiInfo.key,
            )
            logger.info("SiliconFlow chat model initialized successfully.")
            return model
        except Exception as e:
            logger.error(f"Error creating SiliconFlow chat model: {e}")
            raise ChatModelInitializationError("SiliconFlow chat model initialization failed.") from e

    @staticmethod
    @lru_cache(maxsize=1)
    def __azure_chat() -> AzureChatOpenAI:
        """【内部】创建并返回 Azure 聊天模型实例 (单例)"""
        try:
            model = AzureChatOpenAI(
                api_key=MyAzureAiInfo.key,
                azure_endpoint=MyAzureAiInfo.base_url,
                api_version=MyAzureAiInfo.api_version,
                model=MyAzureAiInfo.chat_model,
            )
            logger.info("Azure chat model initialized successfully.")
            return model
        except Exception as e:
            logger.error(f"Error creating Azure chat model: {e}")
            raise ChatModelInitializationError("Azure chat model initialization failed.") from e

    @staticmethod
    @lru_cache(maxsize=1)
    def __ollama_chat() -> ChatOllama:
        """【内部】创建并返回 Ollama 聊天模型实例 (单例)"""
        try:
            model = ChatOllama(
                model=MyOllamaInfo.chat_model,
                base_url=MyOllamaInfo.base_url,
            )
            logger.info("Ollama chat model initialized successfully.")
            return model
        except Exception as e:
             error_msg = f"Error creating Ollama chat model: {e}"
             if isinstance(e, (ConnectError, TimeoutException)):
                 logger.warning(error_msg)
             else:
                 logger.error(error_msg)
             raise ChatModelInitializationError("Ollama chat model initialization failed.") from e

    @property
    def chat_model(self) -> BaseChatModel:
        """
        获取底层的聊天模型实例对象 (SiliconFlow, Azure, Ollama 等)。
        Raises:
            RuntimeError: 如果聊天模型实例尚未成功初始化。
        """
        if self._chat_model is None:
            raise RuntimeError("Chat model is not available because its initialization failed.")
        return self._chat_model

    def is_available(self) -> bool:
        """检查聊天模型实例是否已成功初始化并可用。"""
        return self._chat_model is not None