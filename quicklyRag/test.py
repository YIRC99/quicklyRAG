from pathlib import Path
from langchain_core.documents import Document
from loguru import logger

from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.baseEnum.VectorEnum import VectorStorageType
from quicklyRag.document.loadDocument import load_document
from quicklyRag.document.spliterDocument import spliter_file
from quicklyRag.provider.QuicklyEmbeddingModelProvider import QuicklyEmbeddingModelProvider
from quicklyRag.provider.QuicklyVectorStoreProvider import QuicklyVectorStoreProvider

from quicklyRag.vector.embedding.vectorEmbedding import embed_document
from quicklyRag.vector.store.vectorStore import store_vector_by_documents


if __name__ == '__main__':

    print('12')