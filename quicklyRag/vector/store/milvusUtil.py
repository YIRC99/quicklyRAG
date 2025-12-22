from langchain_milvus import Milvus
from loguru import logger


# 1. 封装通用查询方法 (List)
def list_documents(store: Milvus,
                   offset: int = 0,
                   limit: int = 10,
                   filter_expr: str = "") -> list[dict]:
    """
    使用 store.client 直接查询文档
    """
    # 必须获取集合名称，LangChain 实例中存储了这个属性
    collection_name = store.collection_name

    # 构造过滤条件。MilvusClient 要求必须有 filter，如果为空，通常用 pk 占位
    # 如果你的 pk 是 int，用 pk >= 0；如果是 str，用 pk != ""
    # 这里假设是 auto_id (int)
    if not filter_expr:
        filter_expr = "pk >= 0"

    logger.info(f"正在查询集合 [{collection_name}], 条件: {filter_expr}")

    try:
        # 调用 MilvusClient.query
        res = store.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            offset=offset,
            limit=limit,
            output_fields=["*"]  # "*" 表示返回所有字段(包括向量和meta)，也可以指定 ["text", "source"]
        )
        return res
    except Exception as e:
        logger.error(f"查询出错: {e}")
        return []


# 2. 封装删除方法
def delete_documents(store: Milvus,
                     ids: list[str | int] = None,
                     filter_expr: str = None):
    """
    使用 store.client 删除文档
    支持按 ID 删除，也支持按条件(filter)删除 (MilvusClient 的优势)
    """
    collection_name = store.collection_name

    try:
        if ids:
            # 方式A: 按 ID 删除
            logger.info(f"正在从 [{collection_name}] 删除 ID: {ids}")
            store.client.delete(
                collection_name=collection_name,
                ids=ids
            )
        elif filter_expr:
            # 方式B: 按条件删除 (例如 source == 'xxx')
            logger.info(f"正在从 [{collection_name}] 删除, 条件: {filter_expr}")
            store.client.delete(
                collection_name=collection_name,
                filter=filter_expr
            )
        else:
            logger.warning("删除操作必须提供 ids 或 filter_expr")

        logger.success("删除操作执行完毕")

    except Exception as e:
        logger.error(f"删除失败: {e}")

# 根据pk查询方法
def get_document_by_id(store: Milvus, doc_id: int) -> dict | None:
    """
    根据 pk 获取数据
    注意: 你的 pk 是 Int64, 所以参数 doc_id 必须是 int 类型
    """
    try:
        res = store.client.get(
            collection_name=store.collection_name,
            ids=[doc_id],
            output_fields=["*"]  # 拿回 text, source, start_index, vector
        )
        if res:
            return res[0]
        return None
    except Exception as e:
        logger.error(f"查询失败: {e}")
        return None





