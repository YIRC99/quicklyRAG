from quickly_rag.core.document_base import RagDocumentInfo
from quickly_rag.enums.vector_enum import ScoreField

# 默认向量检索返回的文档段数
default_top_k = 10
# 默认的文件检索分数
default_vector_search_score = 0.3
# 默认的分数过滤策略
# 有三种分数过滤策略
# 1. auto 如果配置类重排模型, 那么就使用重排模型对分数进行过滤 否则使用向量检索的分数
# 2. vector_search_score 使用向量检索的分数
# 3. relevance_score 使用重排分数
default_score_filter_strategy = ScoreField.AUTO


# 文档拆分配置
rag_document_info = RagDocumentInfo(
    chunk_size=300, # 默认分块大小
    chunk_overlap=60  # 默认分块重叠大小
)













