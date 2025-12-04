# 支持本地内存 redisSession Milvus 存储向量
import faiss
from langchain_community.docstore import InMemoryDocstore

from quicklyRag.baseClass.VectorBase import MyMilvusConfig, MyFaissConfig
from quicklyRag.baseEnum.PlatformEnum import PlatformVectorStoreType
from quicklyRag.baseEnum.VectorEnum import VectorStorageType, VectorMetricType, VectorIndexType
from quicklyRag.model.MyModel import siliconflow_embed

# 切换默认全局存储类型
default_embedding_database_type = VectorStorageType.MILVUS

MyMilieusInfo = MyMilvusConfig(
        uri='99999999999',
        user='',
        password='',
        port=19530,
        collection_name='quicklyRag',
        metric_type=VectorMetricType.COSINE.value,  # 默认为COSINE 余弦相似度算法
        index_type=VectorIndexType.HNSW.value,
        is_delete=True,
        auto_id=True,
        drop_old=True,
        embedding_model=siliconflow_embed(),
        enable_dynamic_field=False,
        index_params={'M': 16, 'efConstruction': 64},
        search_params={'ef': 64}
    )

# For FAISS, we initialize with None values which will be properly set up later
MyFaissInfo = MyFaissConfig(
        metric_type=VectorMetricType.COSINE,  # 默认为COSINE 余弦相似度算法
        embedding=siliconflow_embed(),
        docstore=InMemoryDocstore({}), # Initialize with empty dict
        index=None, # Initialize with None, will be created by the FAISS vector store
    )















