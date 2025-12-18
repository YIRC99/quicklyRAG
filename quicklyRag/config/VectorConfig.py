# 支持本地内存 redisSession Milvus 存储向量
import faiss
from langchain_community.docstore import InMemoryDocstore

from quicklyRag.baseClass.VectorBase import QuicklyMilvusConfig, MyFaissConfig
from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.baseEnum.VectorEnum import VectorStorageType, VectorMetricType, VectorIndexType
from quicklyRag.provider.QuicklyEmbeddingModelProvider import QuicklyEmbeddingModelProvider

# 向量存储 默认使用的向量库
default_embedding_database_type = VectorStorageType.MILVUS

MyMilieusInfo = QuicklyMilvusConfig(
        uri='99999999999',
        user='',
        password='',
        port=19530,
        collection_name='quicklyRag',
        metric_type=VectorMetricType.COSINE.value,  # 默认为COSINE 余弦相似度算法
        index_type=VectorIndexType.HNSW.value,
        auto_id=True,
        drop_old=False,
        embedding_model=QuicklyEmbeddingModelProvider(PlatformEmbeddingType.SILICONFLOW),
        enable_dynamic_field=False,
        index_params={'M': 32, 'efConstruction': 128},
        search_params={'ef': 128}
    )

MyFaissInfo = MyFaissConfig(
        metric_type=VectorMetricType.COSINE,  # 默认为COSINE 余弦相似度算法
        embedding=QuicklyEmbeddingModelProvider(PlatformEmbeddingType.SILICONFLOW),
        docstore=InMemoryDocstore({}),
        index=None,
    )















