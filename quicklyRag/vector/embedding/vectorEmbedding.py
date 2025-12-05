from langchain_core.documents import Document

from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.config.PlatformConfig import default_embedding_use_platform
from quicklyRag.factory.QuicklyEmbeddingModelFactory import QuicklyEmbeddingModelFactory


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


def get_embedding_model(embedding_type: PlatformEmbeddingType = default_embedding_use_platform)-> QuicklyEmbeddingModelFactory:
    return QuicklyEmbeddingModelFactory(embedding_type)

def embed_document(documents: str | list[str] | list[Document],embedding_type: PlatformEmbeddingType = default_embedding_use_platform) -> list[list[float]]:
    model = get_embedding_model(embedding_type)
    return model.embedding_text(_normalize_documents(documents))














