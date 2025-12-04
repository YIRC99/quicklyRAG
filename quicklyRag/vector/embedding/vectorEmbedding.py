from langchain_core.documents import Document

from quicklyRag.baseEnum.PlatformEnum import PlatformEmbeddingType
from quicklyRag.baseModel.embeddingBase import QuicklyEmbeddingModel
from quicklyRag.config.PlatformConfig import default_embedding_use_platform
from quicklyRag.model.MyModel import siliconflow_embed, siliconflow_embed2, ollama_embed2


def get_embedding_model(embedding_type: PlatformEmbeddingType)-> QuicklyEmbeddingModel:
    if embedding_type is PlatformEmbeddingType.SILICONFLOW:
        return siliconflow_embed2()
    elif embedding_type is PlatformEmbeddingType.OLLAMA:
        return ollama_embed2()
    else:
        return siliconflow_embed2()

def embed_document(documents: list[Document],embedding_type: PlatformEmbeddingType = default_embedding_use_platform) -> None:
    model = get_embedding_model(embedding_type)
    print(model)
    pass



if __name__ == '__main__':
    embed_document([Document(page_content="hello world")])












