from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.baseEnum.VectorEnum import VectorStorageType
from quicklyRag.config.PlatformConfig import default_embedding_use_platform
from quicklyRag.config.VectorConfig import default_embedding_database_type
from quicklyRag.provider.QuicklyRerankerProvider import QuicklyRerankerProvider
from quicklyRag.provider.QuicklyVectorStoreProvider import QuicklyVectorStoreProvider
from quicklyRag.vector.embedding.vectorEmbedding import embed_document


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


def get_vectorstore_model(vectorstore_type: VectorStorageType = default_embedding_database_type) -> QuicklyVectorStoreProvider:
    return QuicklyVectorStoreProvider(vectorstore_type)

# 只用来存储向量 可以传入存储库的类型 默认使用默认的向量存储库
def store_vector_by_documents(documents: list[Document] | Document | str, vectorstore_type: VectorStorageType = default_embedding_database_type) -> QuicklyVectorStoreProvider:
    vectorstore_model = get_vectorstore_model(vectorstore_type)
    normalized_docs = _normalize_documents(documents)
    
    # 对文档进行分批处理，每批最多32个文档，避免超过Milvus的限制
    batch_size = 256
    for i in range(0, len(normalized_docs), batch_size):
        batch_docs = normalized_docs[i:i + batch_size]
        vectorstore_model.vector_store.add_documents(batch_docs)
        print(f"已存储文档批次 {i//batch_size + 1}/{(len(normalized_docs)-1)//batch_size + 1}，包含 {len(batch_docs)} 个文档")
    
    return vectorstore_model


def search_by_scores(query: str,
                     topK:int = 10,
                     vectorstore_type: VectorStorageType = default_embedding_database_type,
                     embedding_type: PlatformEmbeddingType = default_embedding_use_platform) -> list[Document]:
    vectorstore_model = get_vectorstore_model(vectorstore_type)
    # 使用 similarity_search_with_score 替代 similarity_search_with_relevance_scores
    scores = vectorstore_model.vector_store.similarity_search_with_score(query,k=topK)

    reranker = QuicklyRerankerProvider()
    documents = [doc.page_content for doc, score in scores]
    results = reranker.rerank(query, documents, top_n=10)


    # 详细打印检索结果
    print(f"查询语句: {query}")
    print(f"检索到 {len(scores)} 个相关文档:")
    print("-" * 80)
    for idx, (doc, score) in enumerate(scores):
        # 只显示前50个字符的内容
        content_preview = doc.page_content[:2200] + "..." if len(doc.page_content) > 50 else doc.page_content

        print(f"文档 {idx + 1}:")
        print(f"  内容预览: {content_preview}")
        print(f"  相似度分数: {score:.4f}")

        # 打印元数据（如果存在）
        if doc.metadata:
            print(f"  元数据: {doc.metadata}")
        else:
            print("  元数据: 无")
        print("-" * 80)

    vector = embed_document(query)
    return scores


if __name__ == '__main__':
    search_by_scores('江西省职业院校技能大赛中职组数字艺术设计赛项使用的技术')