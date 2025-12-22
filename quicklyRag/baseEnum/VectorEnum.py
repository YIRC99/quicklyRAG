from enum import Enum

class VectorIndexType(Enum):
    """
    Milvus 支持的索引类型枚举
    """
    FLAT = "FLAT"                    # 精确搜索，默认索引类型
    IVF_FLAT = "IVF_FLAT"           # 基于量化的 IVF 索引
    IVF_SQ8 = "IVF_SQ8"             # 使用标量量化的 IVF 索引
    IVF_PQ = "IVF_PQ"               # 乘积量化的 IVF 索引
    HNSW = "HNSW"                   # 分层可导航小世界图索引
    ANNOY = "ANNOY"                 # 近似最近邻搜索算法
    RHNSW_FLAT = "RHNSW_FLAT"       # 基于 RHNSW 的 FLAT 索引
    RHNSW_SQ = "RHNSW_SQ"           # 基于 RHNSW 的标量量化索引
    RHNSW_PQ = "RHNSW_PQ"           # 基于 RHNSW 的乘积量化索引


class VectorMetricType(Enum):
    L2 = "L2"
    COSINE = "COSINE"

class ScoreField(Enum):
    RELEVANCE = "relevance_score"
    VECTOR = "score"
    AUTO = 'auto'


class VectorStorageType(Enum):
    # REDIS = "redis" 暂不支持
    MILVUS = "milvus"
    # FAISS = 'FAISS'暂不支持


