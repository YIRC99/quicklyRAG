from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from quicklyRag.baseEnum.VectorEnum import VectorStorageType
from quicklyRag.baseModel.vectorBase import QuicklyVectorStoreModel
from quicklyRag.config.PlatformConfig import default_embedding_use_platform

def _normalize_documents(documents: list[Document] | Document | str) -> list[Document]:
    """Normalize input to list of Document objects."""
    if isinstance(documents, str):
        return [Document(page_content=documents)]
    elif isinstance(documents, Document):
        return [documents]
    elif isinstance(documents, list):
        return documents
    else:
        raise TypeError("documents must be a string, Document, or list of Documents")



def get_vectorstore_model(vectorstore_type: VectorStorageType = default_embedding_use_platform) -> QuicklyVectorStoreModel:
    return QuicklyVectorStoreModel(vectorstore_type)

# 只用来存储向量 可以传入存储库的类型 默认使用默认的向量存储库
def store_vector(documents: list[Document] | Document | str, vectorstore_type: VectorStorageType = default_embedding_use_platform) -> QuicklyVectorStoreModel:
    vectorstore_model = get_vectorstore_model(vectorstore_type)
    documents = _normalize_documents(documents)
    vectorstore_model.vector_store.add_documents(documents)
    return vectorstore_model


if __name__ == '__main__':
    model = get_vectorstore_model(VectorStorageType.MILVUS)
    raw_documents = [
        Document(
            page_content="葡萄是一种常见的水果，属于葡萄科葡萄属植物。它的果实呈圆形或椭圆形，颜色有绿色、紫色、红色等多种。葡萄富含维生素C和抗氧化物质，可以直接食用或酿造成葡萄酒。",
            # 元数据 其实就是数据的其他字段 方便用来查询和其他操作的
            metadata={"source": "水果", "type": "植物"}
        ),
        Document(
            page_content="白菜是十字花科蔬菜，原产于中国北方。它的叶片层层包裹形成紧密的球状，口感清脆微甜。白菜富含膳食纤维和维生素K，常用于制作泡菜、炒菜或煮汤。",
            metadata={"source": "蔬菜", "type": "植物"}
        ),
        Document(
            page_content="狗是人类最早驯化的动物之一，属于犬科。它们具有高度社会性，能理解人类情绪，常被用作宠物、导盲犬或警犬。不同品种的狗在体型、毛色和性格上有很大差异。",
            metadata={"source": "动物", "type": "哺乳动物"}
        )]
    model.vector_store.add_documents(raw_documents)
