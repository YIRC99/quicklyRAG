from enum import Enum


class PlatformEmbeddingType(Enum):
    SILICONFLOW = 'SILICONFLOW'
    AZURE = 'AZURE'
    OLLAMA = 'OLLAMA'


class PlatformChatModelType(Enum):
    SILICONFLOW = 'SILICONFLOW'
    AZURE = 'AZURE'
    OLLAMA = 'OLLAMA'


class PlatformVectorStoreType(Enum):
    MILVUS = 'MILVUS'
    FAISS = 'FAISS'