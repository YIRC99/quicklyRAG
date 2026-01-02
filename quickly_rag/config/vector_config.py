# 支持本地内存 redisSession Milvus 存储向量
from langchain_community.docstore import InMemoryDocstore

from quickly_rag.config.platform_config import default_embedding_use_platform
from quickly_rag.core.Vector_base import QuicklyMilvusConfig, MyFaissConfig, QuicklyChromaConfig
from quickly_rag.enums.platform_enum import PlatformEmbeddingType
from quickly_rag.enums.vector_enum import VectorStorageType, VectorMetricType, VectorIndexType
from quickly_rag.provider.embedding_model_provider import QuicklyEmbeddingModelProvider

# 向量存储 默认使用的向量库
default_embedding_database_type = VectorStorageType.MILVUS


def get_embedding_model(
        embedding_type: PlatformEmbeddingType = default_embedding_use_platform) -> QuicklyEmbeddingModelProvider:
    return QuicklyEmbeddingModelProvider(embedding_type)


MyChromaInfo = QuicklyChromaConfig(
    embedding_model=get_embedding_model(default_embedding_use_platform),
    persist_dir="./quickly_rag_data/chroma_db",
    collection_name = "quickly_rag_chroma_collection",
)

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
    embedding_model=get_embedding_model(default_embedding_use_platform),
    enable_dynamic_field=False,
    index_params={'M': 32, 'efConstruction': 128},
    search_params={'ef': 128}
)

MyFaissInfo = MyFaissConfig(
    metric_type=VectorMetricType.COSINE,  # 默认为COSINE 余弦相似度算法
    embedding=get_embedding_model(default_embedding_use_platform),
    docstore=InMemoryDocstore({}),
    index=None,
)
