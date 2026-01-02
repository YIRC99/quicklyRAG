from pathlib import Path
from typing import Optional

import faiss
from langchain_community.docstore import InMemoryDocstore

from pydantic import BaseModel, Field

from quickly_rag.enums.vector_enum import VectorMetricType
from quickly_rag.provider.embedding_model_provider import QuicklyEmbeddingModelProvider

class QuicklyChromaConfig(BaseModel):
    embedding_model: QuicklyEmbeddingModelProvider = Field(..., description="嵌入模型实例")
    persist_dir: str | Path = Field(..., description="向量数据库持久化目录")
    collection_name: str = Field(..., description="向量数据库集合名称")
    pass


class QuicklyMilvusConfig(BaseModel):
    uri: str = Field(..., description="Milvus的服务地址")
    port: str | int = Field(..., description="Milvus的服务端口")
    user: str = Field(..., description="Milvus的用户名")
    password: str = Field(..., description="Milvus的密码")
    collection_name: str = Field(..., description="Milvus的集合名称")
    auto_id: bool = Field(..., description="是否自动生成ID")
    drop_old: bool = Field(..., description="是否删除已有集合   删除现有集合并重新创建")
    metric_type: str = Field(..., description="向量相似度计算类型")
    embedding_model: QuicklyEmbeddingModelProvider = Field(..., description="嵌入模型实例")
    index_type: str = Field(..., description="向量索引类型")
    enable_dynamic_field: bool = Field(..., description="是否启用动态字段")
    index_params: dict = Field(..., description="向量索引参数")
    search_params: dict = Field(..., description="向量搜索参数")




class MyFaissConfig:
    def __init__(
            self,
            metric_type: Optional[VectorMetricType],  # 向量相似度计算类型，默认为COSINE（余弦相似度）
            embedding: Optional[QuicklyEmbeddingModelProvider],  # 嵌入模型实例
            docstore: Optional[InMemoryDocstore],
            index: Optional[faiss.Index], # Fixed: Use proper faiss Index type
            **kwargs
    ):
        self.metric_type = metric_type
        self.embedding = embedding
        self.docstore = docstore
        self.index = index
        # 动态设置额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)