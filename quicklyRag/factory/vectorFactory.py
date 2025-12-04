from functools import lru_cache

from langchain_milvus import Milvus
from langchain_community.vectorstores import FAISS

from quicklyRag.baseEnum.PlatformEnum import PlatformVectorStoreType
from quicklyRag.baseModel.vectorBase import QuicklyVectorStoreModel
from quicklyRag.config.VectorConfig import MyMilieusInfo, MyFaissInfo

@lru_cache(maxsize=1)
def my_milvus() -> Milvus:
    return Milvus(
        embedding_function=MyMilieusInfo.embedding_model,
        connection_args={
            # 老版本使用host:域名 新版本使用uri加http://域名
            "uri": MyMilieusInfo.base_url,
            "port": MyMilieusInfo.port,  # Milvus默认端口
            "user": MyMilieusInfo.user,  # 无用户名
            "password": MyMilieusInfo.password  # 无密码
        },
        collection_name=MyMilieusInfo.collection_name,  # 指定集合名称
        auto_id=True,  # 是否自动生成ID
        enable_dynamic_field=False,  # 启用动态字段以存储任意元数据
        drop_old=MyMilieusInfo.is_delete,  # 是否删除已有集合   删除现有集合并重新创建
        consistency_level="Strong",
        # 添加距离度量参数
        index_params={
            "metric_type": MyMilieusInfo.metric_type,  # 使用余弦相似度
            "index_type": MyMilieusInfo.index_type
        }
    )

@lru_cache(maxsize=1)
def my_milvus2() -> QuicklyVectorStoreModel:
    return QuicklyVectorStoreModel(PlatformVectorStoreType.MILVUS)

@lru_cache(maxsize=1)
def my_faiss() -> FAISS:
    """
    使用FAISS进行向量存储，基于VectorConfig中的MyFaissInfo配置
    """
    return FAISS(
        embedding_function=MyFaissInfo.embedding,
        index=None,  # FAISS索引对象，初次创建时为None
        docstore=None,  # 文档存储，初次创建时为None
        index_to_docstore_id={}  # 索引到文档ID的映射
    )

@lru_cache(maxsize=1)
def my_faiss2() -> QuicklyVectorStoreModel:
    return QuicklyVectorStoreModel(PlatformVectorStoreType.FAISS)
