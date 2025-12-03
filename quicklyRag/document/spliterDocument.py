from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from quicklyRag.config.DocumentConfig import RagDocumentInfo, rag_document_info


# 文档拆分默认使用配置中的配置 但是也可以传入
def spliter_file(test: str | list[Document], rag_config: RagDocumentInfo = rag_document_info) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_config.chunk_size,  # 使用配置类的分块大小
        chunk_overlap=rag_config.chunk_overlap,  # 使用配置类的分块重叠
        add_start_index=True,
    )
    # 判断传入的是字符串还是文档列表
    if isinstance(test, str):
        # 如果是字符串，则创建包含该字符串的文档列表
        return text_splitter.create_documents([test])
    else:
        # 如果是文档列表，则直接分割文档
        return text_splitter.split_documents(test)