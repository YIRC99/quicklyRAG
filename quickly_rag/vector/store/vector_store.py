from langchain_core.documents import Document
from loguru import logger

from quickly_rag.config.vector_config import default_embedding_database_type
from quickly_rag.core.search_base import VectorSearchResult, VectorSearchParams
from quickly_rag.enums.vector_enum import VectorStorageType
from quickly_rag.provider.reranker_provider import QuicklyRerankerProvider
from quickly_rag.provider.vector_store_provider import QuicklyVectorStoreProvider


# 将输入数据转换为文档对象列表
def _normalize_documents(documents: list[Document] | Document | str) -> list[Document]:
    """将输入数据转换为文档对象列表。"""
    if isinstance(documents, str):
        return [Document(page_content=documents)]
    elif isinstance(documents, Document):
        return [documents]
    elif isinstance(documents, list):
        return documents
    else:
        raise TypeError("documents must be a string, Document, or list of Documents")

# 获取向量存储库
def get_vectorstore_model(
        vectorstore_type: VectorStorageType = default_embedding_database_type) -> QuicklyVectorStoreProvider:
    return QuicklyVectorStoreProvider(vectorstore_type)


# 只用来存储向量 可以传入存储库的类型 默认使用默认的向量存储库
def store_vector_by_documents(documents: list[Document] | Document | str,
                              vectorstore_type: VectorStorageType = default_embedding_database_type) -> QuicklyVectorStoreProvider:
    vectorstore_model = get_vectorstore_model(vectorstore_type)
    normalized_docs = _normalize_documents(documents)

    # 对文档进行分批处理，每批最多32个文档，避免超过Milvus的限制
    batch_size = 256
    for i in range(0, len(normalized_docs), batch_size):
        batch_docs = normalized_docs[i:i + batch_size]
        vectorstore_model.vector_store.add_documents(batch_docs)
        print(
            f"已存储文档批次 {i // batch_size + 1}/{(len(normalized_docs) - 1) // batch_size + 1}，包含 {len(batch_docs)} 个文档")

    return vectorstore_model


# 将重排模型和向量检索的结果合并格式化
def format_vectorstore_result(is_ranker: bool, ranker_arr: list[dict], scores: list[tuple[Document, float]]) -> list[VectorSearchResult]:
    results = []
    if is_ranker:
        for i in ranker_arr:
            results.append(VectorSearchResult(text=i['document'].get('text'), relevance_score=i['relevance_score'],
                                              score=scores[int(i['index'])][1]))
    else:
        for i in scores:
            results.append(VectorSearchResult(text=i[0].page_content, relevance_score=0, score=i[1]))
    return results


# 根据传入的字段名(target_field)动态过滤结果
def filter_results_dynamic(results: list[VectorSearchResult],
                           threshold: float,
                           target_field: str) -> list[VectorSearchResult]:
    filtered_data = []
    for res in results:
        val = getattr(res, target_field, 0.0)
        if val >= threshold:
            filtered_data.append(res)
    return filtered_data


# 向量检索的方法, 但是因为直接检索效果不好, 但是用算法优化又会有其他的开销, 但是不优化了
def search_by_scores(search_params :VectorSearchParams) -> list[VectorSearchResult]:
    vectorstore_model = get_vectorstore_model(search_params.vectorstore_type)

    # 如果你想使用带有文本过滤的混合搜索，可以使用如下表达式：
    scores = vectorstore_model.vector_store.similarity_search_with_score(
        search_params.query,
        k=search_params.top_k,
        # expr='text like "%%大数据%%" and text like "%%应用%%" '  # 正确的LIKE语法  ---->暂时不用
    )

    # 使用重排模型查询 防止重排模型未配置时会查询报错
    is_ranker = False
    ranker_arr = []
    try:
        reranker = QuicklyRerankerProvider()
        documents = [doc.page_content for doc, score in scores]

        if len(documents) > 0:
            ranker_arr = reranker.rerank(search_params.query, documents, top_n=search_params.top_k)
            is_ranker = True
    except Exception as e:
        is_ranker = False
        logger.warning(f"重排模型查询出错-使用默认召回查询: {e}")

    all_results = format_vectorstore_result(is_ranker, ranker_arr, scores)

    if search_params.filter_strategy.value == "auto":
        # 如果重排成功，就用重排分过滤，否则用向量分
        if is_ranker:
            target_field = "relevance_score"
            logger.info(f"使用重排分数过滤 (阈值: {search_params.score})")
        else:
            target_field = "score"
            logger.info(f"重排未启用或失败，使用向量分数过滤 (阈值: {search_params.score})")
    else:
        # 强制指定了要过滤的字段
        target_field = search_params.filter_strategy.value

    # 3. 执行动态过滤
    final_results = filter_results_dynamic(all_results, search_params.score, target_field)

    # 格式化并且合并两种查询的结果
    return final_results


if __name__ == '__main__':
    parms = VectorSearchParams(query='财税融合大数据应用赛项是什么', score=0.4, filter_strategy=ScoreField.RELEVANCE)
    print(search_by_scores(parms))

