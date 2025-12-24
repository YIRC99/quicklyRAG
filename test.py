import time
from typing import Generic

from langchain.agents import create_agent
from langchain.agents.structured_output import SchemaT
from langchain_core.messages import HumanMessage

from quickly_rag import api
from quickly_rag.core.document_base import RagDocumentInfo
from quickly_rag.enums.platform_enum import PlatformChatModelType
from quickly_rag.enums.vector_enum import VectorStorageType
from quickly_rag.provider.chat_model_provider import QuicklyChatModelProvider
from quickly_rag.provider.vector_store_provider import QuicklyVectorStoreProvider




if __name__ == '__main__':
    # 图片类型的pdf效果并不好
    # api.vectorize_file('./document_test/document2.pdf',RagDocumentInfo(chunk_size=500,chunk_overlap=100))


    # start_time = time.time()
    # print(llm.invoke("请用中文回答, 你是谁?"))
    # print(f"llm总耗时: {time.time() - start_time}")

    llm = QuicklyChatModelProvider(PlatformChatModelType.SILICONFLOW).chat_model
    agent = create_agent(model=llm)

    start_time = time.time()
    print(agent.invoke({"messages": [{"请用中文回答, 你是谁?"}]}))

    print(f"agent总耗时: {time.time() - start_time}")




