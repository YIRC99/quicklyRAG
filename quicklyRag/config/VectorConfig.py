# 支持本地内存 redisSession Milvus 存储向量
from quicklyRag.baseClass.VectorBase import MyMilvusConfig, MyFaissConfig
from quicklyRag.baseEnum.VectorEnum import VectorStorageType, VectorMetricType, VectorIndexType
from quicklyRag.model.MyModel import siliconflow_embed

# 切换默认全局存储类型
default_use_database_type = VectorStorageType.MILVUS

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
        embedding=siliconflow_embed(),
        # FAISS配置项说明：
        # collection_name: 向量集合名称，默认为"faiss_index"
        # auto_id: 是否自动生成ID，默认为True
        # drop_old: 是否删除已存在的索引，默认为False
        # enable_dynamic_field: 是否启用动态字段，默认为False
        # index_params: 索引参数，如{"nlist": 100}，默认为None
        # search_params: 搜索参数，如{"nprobe": 10}，默认为None
        # consistency_level: 一致性级别，默认为"Strong"
    )















