from datetime import timedelta
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum

from loguru import logger
from pydantic import BaseModel

from quicklyRag.chat.message.chatMessageManager import ChatMessageManager


class ChatSessionManager:
    """
    支持 内存管理 + SQLite持久化 + TTL自动过期的会话管理器
    """

    def __init__(self, db_path: str = None, default_max_messages: int = 50, ttl_seconds: int = 3600):
        """
        Args:
            db_path: SQLite文件路径 (例如 'chat_history.db'). 如果为None，则纯内存运行.
            default_max_messages: 默认消息保留条数.
            ttl_seconds: 会话过期时间(秒)，默认1小时.
        """
        self.default_max_messages = default_max_messages
        self.ttl = timedelta(seconds=ttl_seconds)
        self.db_path = db_path

        # 内存缓存
        self._sessions: Dict[str, ChatMessageManager] = {}

        # 如果提供了db_path，初始化数据库
        if self.db_path:
            self._init_db()

    def _init_db(self):
        """初始化SQLite表结构"""
        with sqlite3.connect(self.db_path) as conn:
            # 创建一个简单的 key-value 风格的表存储 JSON 数据
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS sessions
                         (
                             session_id
                             TEXT
                             PRIMARY
                             KEY,
                             data
                             TEXT
                             NOT
                             NULL,
                             updated_at
                             TIMESTAMP
                             DEFAULT
                             CURRENT_TIMESTAMP
                         )
                         """)
            conn.commit()

    def _is_expired(self, manager: ChatMessageManager) -> bool:
        """检查某个管理器是否过期"""
        if datetime.now() - manager.last_accessed > self.ttl:
            return True
        return False

    def get_session(self, session_id: str) -> ChatMessageManager:
        """
        获取会话逻辑：
        1. 内存有且未过期 -> 返回
        2. 内存有但已过期 -> 清理并新建
        3. 内存无 -> 尝试从DB加载
        4. DB无 -> 新建
        """
        # 1. 尝试从内存获取
        if session_id in self._sessions:
            manager = self._sessions[session_id]
            if self._is_expired(manager):
                logger.info(f"[SessionManager] 会话 {session_id} 已过期 (内存)，正在清理...")
                del self._sessions[session_id]
                # 注意：如果过期了，也要从DB删掉吗？通常是的，或者新建一个覆盖
            else:
                return manager

        # 2. 尝试从 DB 加载 (如果启用了DB)
        if self.db_path:
            manager = self._load_from_db(session_id)
            if manager:
                # 加载后检查是否过期 (防止加载了半年前的陈旧会话)
                if self._is_expired(manager):
                    logger.info(f"[SessionManager] 会话 {session_id} 已过期 (磁盘)，忽略并新建...")
                else:
                    logger.info(f"[SessionManager] 从磁盘加载了会话: {session_id}")
                    self._sessions[session_id] = manager
                    return manager

        # 3. 新建会话
        logger.info(f"[SessionManager] 创建新会话: {session_id}")
        new_manager = ChatMessageManager(max_messages=self.default_max_messages)
        self._sessions[session_id] = new_manager
        return new_manager

    def save_session(self, session_id: str) -> bool:
        """
        【手动持久化】将指定会话保存到 SQLite
        """
        if not self.db_path:
            logger.info("[SessionManager] 未配置 db_path，无法保存")
            return False

        if session_id not in self._sessions:
            return False

        manager = self._sessions[session_id]
        data_json = json.dumps(manager.to_dict(), default=str, ensure_ascii=False)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, data, updated_at) VALUES (?, ?, ?)",
                    (session_id, data_json, datetime.now())
                )
            logger.info(f"[SessionManager] 会话 {session_id} 已保存到磁盘")
            return True
        except Exception as e:
            logger.info(f"[SessionManager] 保存失败: {e}")
            return False

    def _load_from_db(self, session_id: str) -> Optional[ChatMessageManager]:
        """内部方法：从DB读取并反序列化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    return ChatMessageManager.from_dict(data)
        except Exception as e:
            logger.info(f"[SessionManager] 读取失败: {e}")
        return None

    def cleanup_expired(self):
        """
        手动清理所有过期的会话（内存 + 数据库）
        建议定时任务调用
        """
        now = datetime.now()

        # 1. 清理内存
        expired_ids = [
            sid for sid, mgr in self._sessions.items()
            if self._is_expired(mgr)
        ]
        for sid in expired_ids:
            del self._sessions[sid]

        # 2. 清理数据库 (如果有)
        if self.db_path:
            # 计算过期的时间戳
            cutoff = now - self.ttl
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))

        logger.info(f"[SessionManager] 清理完成。移除内存会话: {len(expired_ids)}")
