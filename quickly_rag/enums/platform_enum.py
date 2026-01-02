from enum import Enum


class PlatformEmbeddingType(Enum):
    SILICONFLOW = 'SILICONFLOW'
    # AZURE = 'AZURE'
    OLLAMA = 'OLLAMA'
    ALIYUN = 'ALIYUN'


class PlatformChatModelType(Enum):
    SILICONFLOW = 'SILICONFLOW'
    # AZURE = 'AZURE' 暂未完全适配
    OLLAMA = 'OLLAMA'
    ALIYUN = 'ALIYUN'


class PlatformVectorStoreType(Enum):
    MILVUS = 'MILVUS'
    # FAISS = 'FAISS'