import time
from typing import Generic
from loguru import logger
from langchain.agents import create_agent
from langchain.agents.structured_output import SchemaT
from langchain_core.globals import set_debug, set_verbose
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from quickly_rag import api
from quickly_rag.core.document_base import RagDocumentInfo
from quickly_rag.core.search_base import VectorSearchParams
from quickly_rag.enums.platform_enum import PlatformChatModelType, PlatformEmbeddingType
from quickly_rag.enums.vector_enum import VectorStorageType
from quickly_rag.provider.chat_model_provider import QuicklyChatModelProvider
from quickly_rag.provider.embedding_model_provider import QuicklyEmbeddingModelProvider
from quickly_rag.provider.vector_store_provider import QuicklyVectorStoreProvider

if __name__ == '__main__':
    # print(api.vectorize_file('./document_test/document.txt', RagDocumentInfo(chunk_size=500, chunk_overlap=100)))

    # embed = QuicklyEmbeddingModelProvider(PlatformEmbeddingType.ALIYUN)
    # print(embed.embedding_text("hello world"))

    #
    # llm = QuicklyChatModelProvider(PlatformChatModelType.SILICONFLOW)
    # # start_time = time.time()
    # # print(llm.invoke("请用中文回答, 你是谁?"))
    # # print(f"llm总耗时: {time.time() - start_time}")
    #
    # agent = create_agent(model=llm.chat_model)

    # print(agent.invoke(
    #     {"messages": [{"role": "user", "content": "介绍一下你自己"}]}
    # ))
    #

    # provider = QuicklyEmbeddingModelProvider(PlatformEmbeddingType.SILICONFLOW)
    # print(provider.embed_query("hello"))

    # set_debug(True)  # 开启调试模式
    # set_verbose(False)  # 可关闭详细模式避免重复输出
    # start_time = time.time()
    # logger.error(api.llm_chat("你知道我叫什么嘛",session_id='e0b16f06-ec48-42c9-8545-610f4b86a3c0'))
    # print(f"agent总耗时: {time.time() - start_time}")

    # for chunk in api.llm_stream_chat("能不能帮我介绍一下江西省职业院校技能大赛",session_id='e0b16f06-ec48-42c9-8545-610f4b86a3c0'):
    #     print(chunk)

    # provider = QuicklyVectorStoreProvider(VectorStorageType.CHROMA)
    # print(provider.vector_store.add_texts("hello"))

    # api.vectorize_file(file_path='./document_test/document.txt',
    #                    vectorstore_type=VectorStorageType.MILVUS)

    logger.error(api.llm_chat(
        search_params=VectorSearchParams(query='祝家俊擅长的技术是什么',
                                         score=0.1,
                                         vectorstore_type=VectorStorageType.MILVUS)))
