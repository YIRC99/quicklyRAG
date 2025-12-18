"""
对话消息管理类
用于管理系统消息、用户消息和AI消息
"""

from enum import Enum
from typing import List, Optional, Union
from datetime import datetime
from pydantic import BaseModel


class MessageType(Enum):
    """消息类型枚举"""
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"


class Message(BaseModel):
    """消息模型"""
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
        """
        初始化聊天消息管理器
        
        Args:
            max_messages: 最大消息数量限制，默认50条
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.system_message: Optional[Message] = None
    
    def set_system_message(self, content: str) -> None:
        """
        设置系统消息（只保留一条）
        
        Args:
            content: 系统消息内容
        """
        self.system_message = Message(role=MessageType.SYSTEM, content=content)
    
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
        """
        内部方法：添加消息并维护消息数量限制
        
        Args:
            message: 消息对象
        """
        self.messages.append(message)
        # 如果超出最大消息数量，移除最早的消息（但保留至少一条）
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
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
        
        return messages
    
    def get_messages_for_openai(self) -> List[dict]:
        """
        获取适用于OpenAI API调用的消息格式
        
        Returns:
            适用于OpenAI API调用的消息字典列表
        """
        messages = []
        # 添加系统消息（如果存在）
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message.content
            })
        
        # 添加对话消息
        for message in self.messages:
            # 转换角色名称以匹配OpenAI API的要求
            role = "user" if message.role == MessageType.HUMAN else "assistant"
            messages.append({
                "role": role,
                "content": message.content
            })
        
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