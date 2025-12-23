from pydantic import BaseModel, Field

from quickly_rag.config.document_config import default_top_k, default_vector_search_score, default_score_filter_strategy
from quickly_rag.config.vector_config import default_embedding_database_type
from quickly_rag.enums.vector_enum import ScoreField, VectorStorageType


# 向量搜索的查询参数类, 只有query是必须传入的
class VectorSearchParams(BaseModel):
    query: str = Field(..., description="查询内容")
    top_k: int = Field(default=default_top_k, description="返回的文档数量")
    score: float = Field(default=default_vector_search_score, description="文档过滤分数")
    filter_strategy: ScoreField = Field(default=default_score_filter_strategy, description="过滤策略")
    vectorstore_type: VectorStorageType = Field(default=default_embedding_database_type, description="向量存储库类型")

class VectorSearchResult(BaseModel):
    text: str = Field(description="文档内容")
    relevance_score: float = Field(description="重排模型分数")
    score: float = Field(description="向量搜索分数")