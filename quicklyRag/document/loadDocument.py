import os
from pathlib import Path
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PythonLoader

from quicklyRag.config.DocumentConfig import rag_document_info
from quicklyRag.document.spliterDocument import spliter_file
from quicklyRag.baseClass.documentBase import RagDocumentInfo

def get_file_extension(file_path: str | Path) -> str:
    return os.path.splitext(file_path)[1].lower()

def get_document_loader(file_path: str | Path) -> BaseLoader:
    ext = get_file_extension(file_path)
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".md":
        return UnstructuredMarkdownLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext == ".json":
        return JSONLoader(file_path, jq_schema=".", text_content=False)
    elif ext == ".html":
        return UnstructuredHTMLLoader(file_path)
    elif ext == ".py":
        return PythonLoader(file_path)
    else:
        return TextLoader(file_path, encoding="utf-8")

def load_document(file_path: str | Path, rag_config: RagDocumentInfo = rag_document_info) -> list[Document]:
    loader = get_document_loader(file_path)
    documents = loader.load()
    return spliter_file(documents, rag_config)
