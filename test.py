import time
from typing import Generic

from langchain.agents import create_agent
from langchain.agents.structured_output import SchemaT
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from quickly_rag import api
from quickly_rag.core.document_base import RagDocumentInfo
from quickly_rag.enums.platform_enum import PlatformChatModelType, PlatformEmbeddingType
from quickly_rag.enums.vector_enum import VectorStorageType
from quickly_rag.provider.chat_model_provider import QuicklyChatModelProvider
from quickly_rag.provider.embedding_model_provider import QuicklyEmbeddingModelProvider
from quickly_rag.provider.vector_store_provider import QuicklyVectorStoreProvider




if __name__ == '__main__':
    print(api.vectorize_file('./document_test/document.txt', RagDocumentInfo(chunk_size=500, chunk_overlap=100)))

    # embed = QuicklyEmbeddingModelProvider(PlatformEmbeddingType.ALIYUN)
    # print(embed.embedding_text("hello world"))


    #
    # llm = QuicklyChatModelProvider(PlatformChatModelType.ALIYUN)
    # # start_time = time.time()
    # # print(llm.invoke("请用中文回答, 你是谁?"))
    # # print(f"llm总耗时: {time.time() - start_time}")
    #
    # agent = create_agent(model=llm.chat_model)
    #
    # start_time = time.time()
    # print(agent.invoke(
    #     {"messages": [{"role": "user", "content": "介绍一下你自己"}]}
    # ))
    #
    # print(f"agent总耗时: {time.time() - start_time}")





