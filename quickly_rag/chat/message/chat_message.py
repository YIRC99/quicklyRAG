from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from quickly_rag.chat.message.chat_message_manager import Message


def convert_history_to_langchain_format(history_list: list[Message]) -> list[BaseMessage]:
    """
    将自定义格式的对话历史转换为LangChain消息格式

    Args:
        history_list: 自定义格式的对话历史列表

    Returns:
        list: LangChain格式的消息列表
    """
    langchain_messages = []

    for message in history_list:
        role = message.get('role')
        content = message.get('content')

        if role == 'human':
            langchain_messages.append(HumanMessage(content=content))
        elif role == 'ai':
            langchain_messages.append(AIMessage(content=content))

    return langchain_messages
