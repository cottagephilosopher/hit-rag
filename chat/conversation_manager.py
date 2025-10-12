"""
对话会话管理器
负责对话会话的创建、存储、检索和上下文管理
"""

import sqlite3
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class ConversationManager:
    """对话会话管理器"""

    def __init__(self, db_path: str = None):
        """
        初始化对话管理器

        Args:
            db_path: 数据库路径，默认使用项目根目录下的 rag_preprocessor.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / ".dbs/rag_preprocessor.db"
        self.db_path = str(db_path)

    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_session(self, metadata: Dict[str, Any] = None) -> str:
        """
        创建新的对话会话

        Args:
            metadata: 会话元数据（用户信息、配置等）

        Returns:
            session_id: 会话ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO chat_sessions (session_id, created_at, last_activity, metadata, status)
                VALUES (?, ?, ?, ?, 'active')
            """, (session_id, now, now, json.dumps(metadata or {})))
            conn.commit()

        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = None,
        sources: List[Dict] = None,
        metadata: Dict = None
    ) -> int:
        """
        添加消息到对话历史

        Args:
            session_id: 会话ID
            role: 角色 ('user' | 'assistant' | 'system')
            content: 消息内容
            intent: 意图类型 ('question' | 'clarification' | 'chitchat')
            sources: 引用的文档片段列表
            metadata: 额外元数据

        Returns:
            message_id: 消息ID
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            # 添加消息
            cursor = conn.execute("""
                INSERT INTO chat_messages
                (session_id, role, content, intent, sources, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                role,
                content,
                intent,
                json.dumps(sources or []),
                json.dumps(metadata or {}),
                now
            ))

            message_id = cursor.lastrowid

            # 更新会话的最后活动时间
            conn.execute("""
                UPDATE chat_sessions
                SET last_activity = ?
                WHERE session_id = ?
            """, (now, session_id))

            conn.commit()

        return message_id

    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10,
        include_system: bool = False
    ) -> List[Dict[str, Any]]:
        """
        获取对话历史

        Args:
            session_id: 会话ID
            limit: 返回的最大消息数（最近的 N 条）
            include_system: 是否包含系统消息

        Returns:
            消息列表
        """
        with self._get_connection() as conn:
            if include_system:
                query = """
                    SELECT * FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
            else:
                query = """
                    SELECT * FROM chat_messages
                    WHERE session_id = ? AND role != 'system'
                    ORDER BY timestamp DESC
                    LIMIT ?
                """

            cursor = conn.execute(query, (session_id, limit))
            rows = cursor.fetchall()

        # 转换为字典列表并反转顺序（从旧到新）
        messages = []
        for row in reversed(rows):
            msg = dict(row)
            # 解析 JSON 字段
            msg['sources'] = json.loads(msg['sources']) if msg['sources'] else []
            msg['metadata'] = json.loads(msg['metadata']) if msg['metadata'] else {}
            messages.append(msg)

        return messages

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息字典，如果不存在则返回 None
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM chat_sessions
                WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()

        if row:
            session = dict(row)
            session['metadata'] = json.loads(session['metadata']) if session['metadata'] else {}
            return session
        return None

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话及其所有消息

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        with self._get_connection() as conn:
            # 由于有外键约束，删除会话会自动删除相关消息
            cursor = conn.execute("""
                DELETE FROM chat_sessions
                WHERE session_id = ?
            """, (session_id,))
            conn.commit()

        return cursor.rowcount > 0

    def archive_session(self, session_id: str) -> bool:
        """
        归档会话（不删除，只标记为非活跃）

        Args:
            session_id: 会话ID

        Returns:
            是否归档成功
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE chat_sessions
                SET status = 'archived'
                WHERE session_id = ?
            """, (session_id,))
            conn.commit()

        return cursor.rowcount > 0

    def get_active_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取活跃的对话会话列表

        Args:
            limit: 返回的最大数量

        Returns:
            会话列表
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM chat_sessions
                WHERE status = 'active'
                ORDER BY last_activity DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

        sessions = []
        for row in rows:
            session = dict(row)
            session['metadata'] = json.loads(session['metadata']) if session['metadata'] else {}
            sessions.append(session)

        return sessions

    def format_history_for_llm(
        self,
        session_id: str,
        limit: int = 5
    ) -> str:
        """
        格式化对话历史为 LLM 友好的文本格式

        Args:
            session_id: 会话ID
            limit: 包含的消息数量

        Returns:
            格式化的对话历史文本
        """
        messages = self.get_conversation_history(session_id, limit=limit)

        if not messages:
            return ""

        formatted = []
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'user':
                formatted.append(f"用户: {content}")
            elif role == 'assistant':
                formatted.append(f"助手: {content}")

        return "\n".join(formatted)
