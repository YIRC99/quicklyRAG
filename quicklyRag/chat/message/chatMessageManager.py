"""
对话消息管理类
用于管理系统消息、用户消息和AI消息
"""

from enum import Enum
from typing import List, Optional, Union
from datetime import datetime
from pydantic import BaseModel


# --- 基础定义保持不变 ---
class MessageType(Enum):
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"


class Message(BaseModel):
    role: MessageType
    content: str
    timestamp: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.now()



class ChatMessageManager:
    """聊天消息管理器"""

    def __init__(self, max_messages: int = 50):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.system_message: Optional[Message] = None
        # 新增：记录最后访问时间，用于TTL判断
        self.last_accessed: datetime = datetime.now()

    def _touch(self):
        """更新最后访问时间"""
        self.last_accessed = datetime.now()
    
    def set_system_message(self, content: str) -> None:
        """
        设置系统消息（只保留一条）
        
        Args:
            content: 系统消息内容
        """
        self.system_message = Message(role=MessageType.SYSTEM, content=content)
        self._touch()
    
    def add_human_message(self, content: str) -> None:
        """
        添加用户消息
        
        Args:
            content: 用户消息内容
        """
        self._add_message(Message(role=MessageType.HUMAN, content=content))
    
    def add_ai_message(self, content: str) -> None:
        """
        添加AI消息
        
        Args:
            content: AI消息内容
        """
        self._add_message(Message(role=MessageType.AI, content=content))
    
    def _add_message(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
        self._touch() # 每次添加消息都视为一次活跃访问
    
    def list_messages(self) -> List[Message]:
        """
        获取所有消息列表（包括系统消息）
        
        Returns:
            包含所有消息的列表，系统消息在第一位（如果有）
        """
        result = []
        if self.system_message:
            result.append(self.system_message)
        result.extend(self.messages)
        return result
    
    def get_messages_for_llm(self) -> List[dict]:
        """
        获取适用于LLM调用的消息格式
        
        Returns:
            适用于LLM调用的消息字典列表
        """
        messages = []
        if self.system_message:
            messages.append({
                "role": self.system_message.role.value,
                "content": self.system_message.content
            })
        
        for message in self.messages:
            messages.append({
                "role": message.role.value,
                "content": message.content
            })
        self._touch()  # 读取也算活跃
        return messages
    
    def get_messages_for_openai(self) -> List[dict]:
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message.content})
        for message in self.messages:
            role = "user" if message.role == MessageType.HUMAN else "assistant"
            messages.append({"role": role, "content": message.content})
        self._touch() # 读取也算活跃
        return messages
    
    def clear_messages(self) -> None:
        """清空所有非系统消息"""
        self.messages.clear()
    
    def clear_all(self) -> None:
        """清空所有消息（包括系统消息）"""
        self.messages.clear()
        self.system_message = None
    
    def get_message_count(self) -> int:
        """
        获取消息总数（不包括系统消息）
        
        Returns:
            消息数量
        """
        return len(self.messages)
    
    def get_total_count(self) -> int:
        """
        获取总消息数（包括系统消息）
        
        Returns:
            总消息数
        """
        count = len(self.messages)
        if self.system_message:
            count += 1
        return count

    def to_dict(self) -> dict:
        return {
            "max_messages": self.max_messages,
            "last_accessed": self.last_accessed.isoformat(),
            "system_message": self.system_message.model_dump(mode='json') if self.system_message else None,
            "messages": [m.model_dump(mode='json') for m in self.messages]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessageManager':
        """从字典恢复对象 (包含脏数据清洗逻辑)"""
        manager = cls(max_messages=data.get("max_messages", 50))

        # 1. 恢复时间
        if "last_accessed" in data:
            manager.last_accessed = datetime.fromisoformat(data["last_accessed"])

        # 2. 恢复 System Message (同样需要清洗 role)
        if data.get("system_message"):
            sys_msg_data = data["system_message"]
            cls._clean_role_data(sys_msg_data)  # 调用清洗辅助函数
            manager.system_message = Message(**sys_msg_data)

        # 3. 恢复聊天记录
        if data.get("messages"):
            manager.messages = data.get("messages")
        #     clean_messages = []
        #     for m in data["messages"]:
        #         # 调用清洗逻辑，将 'MessageType.HUMAN' 转换为 'human'
        #         cls._clean_role_data(m)
        #         clean_messages.append(Message(**m))
        #
        #     manager.messages = clean_messages
        return manager

    @staticmethod
    def _clean_role_data(msg_data: dict):
        """
        辅助函数：清洗消息数据中的 role 字段
        将 'MessageType.HUMAN' 格式清洗为 'human'
        """
        role = msg_data.get("role")
        # 检查是否是 'MessageType.XXX' 这种字符串格式
        if isinstance(role, str) and "MessageType." in role:
            try:
                # 提取 . 后面的部分，例如 'HUMAN'
                member_name = role.split(".")[-1]
                # 从 Enum 类获取真实的值，例如 MessageType['HUMAN'].value -> 'human'
                msg_data["role"] = MessageType[member_name].value
            except KeyError:
                # 如果解析失败，保持原样，让 Pydantic 抛出更具体的错误
                pass