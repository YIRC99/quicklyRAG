import json
import uuid
from collections.abc import Iterator

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Other
from langchain_core.runnables.utils import Output
from loguru import logger

from quickly_rag.core.search_base import VectorSearchParams
from quickly_rag.enums.platform_enum import  PlatformChatModelType
from quickly_rag.chat.message.chat_message import convert_history_to_langchain_format
from quickly_rag.chat.message.chat_session_manager import ChatSessionManager
from quickly_rag.chat.prompt.system_prompt_manager import SystemPromptManager
from quickly_rag.config.platform_config import default_chat_model_use_platform
from quickly_rag.provider.chat_model_provider import QuicklyChatModelProvider
from quickly_rag.vector.store.vector_store import search_by_scores


def _format_sse(data: str | dict) -> str:
    """将数据格式化为 SSE 标准字符串"""
    if isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False)
    # SSE 规范: 以 data: 开头，双换行结尾
    return f"data: {data}\n\n"
# SSE模型流式对话
def llm_stream_chat(question: str,
                    session_id: str = None,
                    prompt_name: str = 'system',
                    search_params :VectorSearchParams = None,
                    platform_type: PlatformChatModelType = default_chat_model_use_platform) -> Iterator[Output]:
    if session_id is None:
        session_id = str(uuid.uuid4())

    # 1.查询对话记忆 先获取管理session 在根据userid获取管理器 在从管理器中获取全部对话记录
    session = ChatSessionManager(default_max_messages=100, ttl_seconds=3600 * 24 * 7)
    message_manager = session.get_session(session_id)
    logger.info(f'查询到的消息记录: {message_manager.list_messages()}')

    # 2. 转换历史记录为LangChain格式
    converted_history = convert_history_to_langchain_format(message_manager.list_messages())

    # 5. 向量检索对话 对话相关的资料
    if search_params is None:
        search_params = VectorSearchParams(query=question)
    scores = search_by_scores(search_params)
    context_str = "\n\n".join([f"<资料片段>\n\n: {res.text}\n\n<资料片段>\n\n" for res in scores])
    logger.info(f'向量检索结果: {context_str}')

    # 3.添加提示词 获取管理器对象
    # 然后默认读取系统提示词 需要给系统提示词一个名称 默认会读取SystemPromptManager类下的system.md当作系统提示词 也可以传入file_path
    system_prompt_manager = SystemPromptManager()
    system_prompt = system_prompt_manager.get_prompt(prompt_name)


    # 获取llm
    llm = QuicklyChatModelProvider(platform_type)

    try:
        # 使用 yield from 返回生成器，让上层调用者可以流式接收
        full_response = ""
        agent = create_agent(model=llm.chat_model, tools=[], system_prompt=system_prompt)

        # 4.创建包含历史的提示模板
        full_system_content = f"{system_prompt}\n\n以下是检索到的参考资料，请基于这些资料回答问题：\n\n{context_str}"
        input_messages = [SystemMessage(content=full_system_content)] + converted_history + [HumanMessage(content=question)]

        for msg, metadata in agent.stream({"messages": input_messages}, stream_mode="messages"):

            # 4. 过滤数据
            if isinstance(msg, AIMessageChunk) and msg.content:
                content = msg.content
                full_response += content

                payload = {
                    "content": content,
                    "session_id": session_id,
                    "status": "thinking"
                }
                yield _format_sse(payload)

        # 8. 对话结束后，手动保存对话记录
        if full_response:
            message_manager.add_human_message(question)
            message_manager.add_ai_message(full_response)
            session.save_session(session_id)

        yield _format_sse({"content": "", "status": "done", "session_id": session_id})

    except Exception as e:
        logger.error(f"流式对话出错: {e}")
        yield f"出错啦: {e}"


# 调用模型对话, 一次性返回
def llm_chat(question: str,
                    session_id: str = None,
                    prompt_name: str = 'system',
                    search_params :VectorSearchParams = None,
                    platform_type: PlatformChatModelType = default_chat_model_use_platform) -> dict[str, str] | str:
    if session_id is None:
        session_id = str(uuid.uuid4())

    # 1.查询对话记忆 先获取管理session 在根据userid获取管理器 在从管理器中获取全部对话记录
    session = ChatSessionManager(default_max_messages=100, ttl_seconds=3600 * 24 * 7)
    message_manager = session.get_session(session_id)
    logger.info(f'查询到的消息记录: {message_manager.list_messages()}')

    # 2. 转换历史记录为LangChain格式
    converted_history = convert_history_to_langchain_format(message_manager.list_messages())

    # 5. 向量检索对话 对话相关的资料
    if search_params is None:
        search_params = VectorSearchParams(query=question)
    scores = search_by_scores(search_params)
    context_str = "\n\n".join([f"<资料片段>\n\n: {res.text}\n\n<资料片段>\n\n" for res in scores])
    logger.info(f'向量检索结果: {context_str}')

    # 3.添加提示词 获取管理器对象
    # 然后默认读取系统提示词 需要给系统提示词一个名称 默认会读取SystemPromptManager类下的system.md当作系统提示词 也可以传入file_path
    system_prompt_manager = SystemPromptManager()
    system_prompt = system_prompt_manager.get_prompt(prompt_name)


    # 获取llm
    llm = QuicklyChatModelProvider(platform_type)
    # chain = prompt_template | llm.chat_model | StrOutputParser() 这个后处理会自动取出AI的回复内容
    try:

        agent = create_agent(model=llm.chat_model, tools=[], system_prompt=system_prompt)
        # 4.创建包含历史的提示模板
        full_system_content = f"{system_prompt}\n\n以下是检索到的参考资料，请基于这些资料回答问题：\n\n{context_str}"
        input_messages = [SystemMessage(content=full_system_content)] + converted_history + [HumanMessage(content=question)]
        result = agent.invoke({"messages": input_messages})
        response = result["messages"][-1]
        # 8. 对话结束后，手动保存对话记录
        if response:
            message_manager.add_human_message(question)
            message_manager.add_ai_message(response.content)
            session.save_session(session_id)
        #
        return {
            "content": response,
            "session_id": session_id,
        }

    except Exception as e:
        logger.error(f"对话出错: {e}")
        return f"出错啦: {e}"