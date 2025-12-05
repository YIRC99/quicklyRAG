from langchain_core.documents import Document
from quicklyRag.baseEnum.VectorEnum import VectorStorageType
from quicklyRag.config.VectorConfig import default_embedding_database_type
from quicklyRag.factory.QuicklyVectorStoreFactory import QuicklyVectorStoreFactory


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



def get_vectorstore_model(vectorstore_type: VectorStorageType = default_embedding_database_type) -> QuicklyVectorStoreFactory:
    return QuicklyVectorStoreFactory(vectorstore_type)

# 只用来存储向量 可以传入存储库的类型 默认使用默认的向量存储库
def store_vector_by_documents(documents: list[Document] | Document | str, vectorstore_type: VectorStorageType = default_embedding_database_type) -> QuicklyVectorStoreFactory:
    vectorstore_model = get_vectorstore_model(vectorstore_type)
    vectorstore_model.vector_store.add_documents(_normalize_documents(documents))
    return vectorstore_model
