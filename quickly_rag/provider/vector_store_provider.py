import os
from functools import lru_cache
from typing import Type, Any

from langchain_chroma import Chroma
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from loguru import logger
from pydantic import BaseModel, Field

from quickly_rag.config.vector_config import MyMilieusInfo, MyFaissInfo, MyChromaInfo
from quickly_rag.enums.vector_enum import VectorStorageType


class VectorStoreInitializationError(Exception):
    """当向量数据库初始化失败时抛出的异常"""
    pass

class QuicklyVectorStoreProvider(BaseModel):
    """
    统一向量数据库服务提供者。
    根据指定的 VectorStorageType 和预定义配置，提供并管理对应的 VectorStore 实例（单例）。
    该类负责处理连接、初始化错误，并提供对底层 VectorStore 实例的访问。
    """
    platform_type: VectorStorageType = Field(..., description="指定要使用的向量数据库平台类型")

    def __init__(self, platform_type: VectorStorageType, /, **data: Any):
        """
        初始化 QuicklyVectorStoreProvider 实例。
        Args:
            platform_type (VectorStorageType): 指定要使用的向量数据库平台类型。
        Raises:
            VectorStoreInitializationError: 如果指定平台的向量库初始化失败。
            ValueError: 如果 platform_type 不受支持。
        """
        super().__init__(platform_type=platform_type, **data)
        self._vector_store: VectorStore | None = None
        try:
            self._vector_store = self._get_vector_store_instance(platform_type)
        except VectorStoreInitializationError:
            logger.error(f"Critical failure initializing vector store for {platform_type.name}. Provider is unusable.")
            raise
        except Exception as e:
             logger.error(f"Unexpected error during initialization for platform {platform_type.name}: {e}")
             raise VectorStoreInitializationError(f"Unexpected error initializing {platform_type.name}") from e

    @staticmethod
    @lru_cache(maxsize=1)
    def __create_milvus_store() -> Milvus:
        """【内部】创建并返回 Milvus 向量库实例 (单例)"""
        try:
            milvus_instance = Milvus(
                embedding_function=MyMilieusInfo.embedding_model,
                connection_args={
                    "uri": MyMilieusInfo.uri,
                    "port": MyMilieusInfo.port,
                    "user": MyMilieusInfo.user,
                    "password": MyMilieusInfo.password
                },
                collection_name=MyMilieusInfo.collection_name,
                auto_id=True,
                enable_dynamic_field=False,
                drop_old=MyMilieusInfo.drop_old,
                consistency_level="Strong",
                index_params={
                    "metric_type": MyMilieusInfo.metric_type,
                    "index_type": MyMilieusInfo.index_type,
                    **MyMilieusInfo.index_params
                }
            )
            logger.info("Successfully connected to Milvus.")
            return milvus_instance
        except Exception as e:
            logger.error(f"Failed to connect to Milvus at {MyMilieusInfo.uri}:{MyMilieusInfo.port} : {e}")
            raise VectorStoreInitializationError("Milvus connection or initialization failed.") from e

    @staticmethod
    def __create_chroma_store() -> Chroma:
        """
        【内部】创建并返回 Chroma 向量库实例 (本地持久化)
        这就是任何人都能跑的关键！
        """
        try:
            # 确保数据存储目录存在
            persist_dir = MyChromaInfo.persist_dir
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)

            logger.info(f"Initializing ChromaDB at {persist_dir}...")

            chroma_instance = Chroma(
                collection_name=MyChromaInfo.collection_name,  # 集合名称
                embedding_function=MyChromaInfo.embedding_model,  # 使用你的 Embedding 模型
                persist_directory=persist_dir,  # 数据持久化到本地文件夹
            )
            logger.info("Successfully initialized ChromaDB.")
            return chroma_instance
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreInitializationError("ChromaDB initialization failed.") from e

    @staticmethod
    @lru_cache(maxsize=1)
    def __create_faiss_store() -> FAISS:
        """【内部】创建并返回 FAISS 向量库实例 (初始空状态, 单例)"""
        try:
            faiss_instance = FAISS.from_texts(
                texts=["dummy"],
                embedding=MyFaissInfo.embedding
            )
            faiss_instance.index_to_docstore_id = {}
            faiss_instance.docstore = InMemoryDocstore({})
            logger.info("FAISS instance created.")
            return faiss_instance
        except Exception as e:
            logger.error(f"Failed to create FAISS instance: {e}")
            raise VectorStoreInitializationError("FAISS initialization failed.") from e

    def _get_vector_store_instance(self, platform_type: VectorStorageType) -> VectorStore:
        """根据平台类型获取对应的向量库实例"""
        try:
            if platform_type == VectorStorageType.MILVUS:
                return self.__create_milvus_store()
            elif platform_type == VectorStorageType.CHROMA:
                return self.__create_chroma_store()
            # elif platform_type == VectorStorageType.FAISS:
            #     return self.__create_faiss_store()
            else:
                raise ValueError(f"Unsupported vector store platform type: {platform_type}")
        except VectorStoreInitializationError:
            raise
        except ValueError:
            raise
        except Exception as e:
             logger.error(f"Unexpected error getting vector store instance for {platform_type.name}: {e}")
             raise VectorStoreInitializationError(
                 f"Unexpected error while getting instance for {platform_type.name}"
             ) from e

    @property
    def vector_store(self) -> VectorStore:
        """
        获取底层的向量库实例对象 (Milvus, FAISS 等)。
        Raises:
            RuntimeError: 如果向量库实例尚未成功初始化。
        """
        if self._vector_store is None:
            raise RuntimeError("Vector store is not available because its initialization failed.")
        return self._vector_store

    def is_available(self) -> bool:
        """检查向量库实例是否已成功初始化并可用。"""
        return self._vector_store is not None