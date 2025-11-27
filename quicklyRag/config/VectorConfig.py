# 支持本地内存 redisSession Milvus 存储向量
from quicklyRag.config.baseClass.MyFaissConfig import MyFaissConfig
from quicklyRag.config.baseClass.MyMilvusConfig import MyMilvusConfig
from quicklyRag.config.baseEnum.VectorIndexTypeEnum import VectorIndexType
from quicklyRag.config.baseEnum.VectorMetricTypeEnum import VectorMetricType
from quicklyRag.config.baseEnum.VectorStoreTypeEnum import VectorStorageType
from quicklyRag.model.MyModel import siliconflow_embed

# 切换全局存储类型
type = VectorStorageType.REDIS

MyMilieusInfo = MyMilvusConfig(
        host='99999999999',
        user='',
        password='',
        port=19530,
        collection_name='quicklyRag',
        metric_type=VectorMetricType.COSINE,  # 默认为COSINE 余弦相似度算法
        index_type=VectorIndexType.HNSW,
        is_delete=True,
        enable_dynamic_field=False,
        index_params={'M': 16, 'efConstruction': 64},
        search_params={'ef': 64}
    )

MyFaissInfo = MyFaissConfig(
        metric_type=VectorMetricType.COSINE,  # 默认为COSINE 余弦相似度算法
        embedding=siliconflow_embed()
    )















