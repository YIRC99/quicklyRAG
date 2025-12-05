from functools import lru_cache

from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from loguru import logger
from quicklyRag.baseEnum.PlatformEnum import PlatformVectorStoreType
from quicklyRag.baseEnum.VectorEnum import VectorStorageType
from quicklyRag.config.VectorConfig import MyMilieusInfo, MyFaissInfo


class QuicklyVectorStoreProvider:
    """
    封装不同平台的向量数据库，提供统一的基础接口。
    注意：由于 Milvus 和 FAISS 的 API 和生命周期管理差异较大，
    此类主要提供一个获取实例的统一入口，并暴露底层实例供调用。
    """

    def __init__(self, platform_type: VectorStorageType):
        """
        初始化 QuicklyVectorStore 实例。
        Args:
            platform_type (PlatformVectorStoreType): 指定要使用的向量数据库平台类型。
        """
        self.platform_type = platform_type
        self._vector_store: VectorStore = self._get_vector_store_instance(platform_type)

    @staticmethod
    @lru_cache(maxsize=1)
    def __create_milvus_store() -> Milvus:  # 保持双下划线也可以，但作为静态方法更清晰
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
                drop_old=MyMilieusInfo.is_delete,
                consistency_level="Strong",
                index_params={
                    "metric_type": MyMilieusInfo.metric_type,
                    "index_type": MyMilieusInfo.index_type
                }
            )
            logger.info("Successfully connected to Milvus.")
            return milvus_instance
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    @staticmethod
    @lru_cache(maxsize=1)
    def __create_faiss_store() -> FAISS:
        """【内部】创建并返回 FAISS 向量库实例 (初始空状态, 单例)"""
        try:
            # Create a FAISS vector store with a dummy document to initialize properly
            faiss_instance = FAISS.from_texts(
                texts=["dummy"],
                embedding=MyFaissInfo.embedding
            )
            # Remove the dummy document
            faiss_instance.index_to_docstore_id = {}
            faiss_instance.docstore = InMemoryDocstore({})
            logger.info("FAISS instance created.")
            return faiss_instance
        # TODO保存 FAISS数据：如果你希望在程序关闭后保留FAISS的索引和文档，你需要在添加完文档后调用
        # FAISS.save_local(folder_path=..., index_name=...)方法。
        # 加载
        # FAISS
        # 数据：下次启动程序时，可以使用
        # FAISS.load_local(folder_path=..., embeddings=..., allow_dangerous_deserialization=True)
        # 来加载之前保存的数据。
        except Exception as e:
            logger.error(f"Failed to create FAISS instance: {e}")
            raise

    def _get_vector_store_instance(self, platform_type: VectorStorageType) -> VectorStore:
        """根据平台类型获取对应的向量库实例"""
        if platform_type == VectorStorageType.MILVUS:
            # 调用静态方法
            return self.__create_milvus_store()
        elif platform_type == VectorStorageType.FAISS:
            return self.__create_faiss_store()
        else:
            raise ValueError(f"Unsupported vector store platform type: {platform_type}")

    @property
    def vector_store(self):
        """
        获取底层的向量库实例对象 (Milvus, FAISS 等)。
        """
        return self._vector_store

    def aaa(self):
        print('11111')
