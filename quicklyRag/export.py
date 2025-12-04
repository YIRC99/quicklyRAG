from pathlib import Path
from quicklyRag.baseClass.documentBase import RagDocumentInfo
from quicklyRag.config.DocumentConfig import rag_document_info
from quicklyRag.document.loadDocument import load_document


# 向量化文档 只需要传入一个 文件路径 即可把文档向量化到向量数据库中
def vectorize_file(file_path: str | Path, rag_config: RagDocumentInfo = rag_document_info) -> None:
    documents = load_document(file_path)
    pass



if __name__ == '__main__':
    vectorize_file('./document/testmd.md')