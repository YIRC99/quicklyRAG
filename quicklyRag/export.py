from pathlib import Path

from loguru import logger

from quicklyRag.baseClass.documentBase import RagDocumentInfo
from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.baseEnum.VectorEnum import VectorStorageType
from quicklyRag.config.DocumentConfig import rag_document_info
from quicklyRag.config.PlatformConfig import default_embedding_use_platform
from quicklyRag.config.VectorConfig import default_embedding_database_type
from quicklyRag.document.loadDocument import load_document
from quicklyRag.document.spliterDocument import spliter_file
from quicklyRag.vector.embedding.vectorEmbedding import embed_document
from quicklyRag.vector.store.vectorStore import store_vector_by_documents


# 向量化文档 只需要传入一个 文件路径 即可把文档向量化到向量数据库中
def vectorize_file(
        file_path: str | Path,
        rag_config: RagDocumentInfo = rag_document_info,
        embedding_type: PlatformEmbeddingType = default_embedding_use_platform,
        vectorstore_type: VectorStorageType = default_embedding_database_type
) -> bool:
    """
    将指定文件向量化并存储到向量数据库中
    
    Args:
        file_path: 要处理的文件路径
        rag_config: 文档分割配置
        embedding_type: 嵌入模型类型
        vectorstore_type: 向量存储类型
        
    Returns:
        bool: 处理是否成功
        
    Raises:
        FileNotFoundError: 当指定文件不存在时
        Exception: 其他处理过程中的异常
    """
    try:
        # 验证文件路径
        path_obj = Path(file_path)
        # 如果是相对路径，尝试在当前目录查找
        if not path_obj.exists() and not path_obj.is_absolute():
            # 尝试在 quicklyRag/document 目录下查找
            alt_path = Path("quicklyRag") / "document" / path_obj
            if alt_path.exists():
                path_obj = alt_path
            else:
                # 尝试在项目根目录下查找
                alt_path = Path(".") / path_obj
                if alt_path.exists():
                    path_obj = alt_path
        
        if not path_obj.exists():
            logger.error(f"文件不存在: {file_path}")
            return False
            
        logger.info(f"开始处理文件: {path_obj.absolute()}")
        logger.info(f"文件大小: {path_obj.stat().st_size} 字节")
        
        # 1. 加载文档
        logger.info("正在加载文档...")
        document = load_document(file_path=path_obj)
        logger.success(f"文档加载成功, 文档类型: {type(document)}, 文档数量: {len(document) if hasattr(document, '__len__') else 'N/A'}")
        
        # 2. 拆分文档
        logger.info("正在拆分文档...")
        documents = spliter_file(document, rag_config=rag_config)
        logger.success(f"文档拆分成功, 拆分段数: {len(documents)}")
        
        # 3. 向量化文档
        logger.info("正在进行文档向量化...")
        embeds = embed_document(documents, embedding_type)
        logger.success(f"向量化成功, 向量数量: {len(embeds)}, 向量维度: {len(embeds[0]) if embeds and len(embeds) > 0 else 'N/A'}")
        
        # 4. 存储向量
        logger.info("正在存储向量到数据库...")
        store_vector_by_documents(documents, vectorstore_type)
        logger.success(f"文档存储成功，共存储 {len(documents)} 个文档片段")
        
        logger.info(f"文件 {path_obj} 处理完成")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {file_path}, 错误详情: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时发生错误: {str(e)}")
        logger.exception(e)  # 输出完整的异常堆栈
        return False


if __name__ == '__main__':
    success = vectorize_file('testmd.md')
    if success:
        logger.info("主程序执行成功")
    else:
        logger.error("主程序执行失败")