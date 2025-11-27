from enum import Enum


class VectorStorageType(Enum):
    MEMORY = "memory"
    REDIS = "redis"
    MILVUS = "milvus"