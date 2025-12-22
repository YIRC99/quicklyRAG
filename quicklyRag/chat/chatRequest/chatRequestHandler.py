import json
import uuid
from collections.abc import Iterator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.utils import Output
from loguru import logger
from quicklyRag.baseEnum.PlatformEnum import  PlatformChatModelType
from quicklyRag.chat.message.ChatSessionManager import ChatSessionManager
from quicklyRag.chat.message.chatMessage import convert_history_to_langchain_format
from quicklyRag.chat.prompt.SystemPromptManager import SystemPromptManager
from quicklyRag.config.PlatformConfig import  default_chat_model_use_platform
from quicklyRag.provider.QuicklyChatModelProvider import QuicklyChatModelProvider
from quicklyRag.vector.store.vectorStore import  search_by_scores, VectorSearchParams



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

    # 4.创建包含历史的提示模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "以下是检索到的参考资料，请基于这些资料回答问题：\n\n{context_str}"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    # 获取llm
    llm = QuicklyChatModelProvider(platform_type)
    chain = prompt_template | llm.chat_model | StrOutputParser()

    try:
        # 使用 yield from 返回生成器，让上层调用者可以流式接收
        full_response = ""
        for chunk in chain.stream({
            "chat_history": converted_history,
            "context_str": context_str,
            "question": question
        }):
            if chunk:
                full_response += chunk
                payload = {
                    "content": chunk,  # 当前片段
                    "session_id": session_id,
                    "status": "thinking"  # 标识正在生成
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
                    platform_type: PlatformChatModelType = default_chat_model_use_platform) -> str:
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

    # 4.创建包含历史的提示模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "以下是检索到的参考资料，请基于这些资料回答问题：\n\n{context_str}"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    # 获取llm
    llm = QuicklyChatModelProvider(platform_type)
    chain = prompt_template | llm.chat_model | StrOutputParser()


    try:
        response = chain.invoke({"chat_history": converted_history, "context_str": context_str, "question": question})

        # 8. 对话结束后，手动保存对话记录
        if response:
            message_manager.add_human_message(question)
            message_manager.add_ai_message(response)
            session.save_session(session_id)

        return response

    except Exception as e:
        logger.error(f"对话出错: {e}")
        return f"出错啦: {e}"