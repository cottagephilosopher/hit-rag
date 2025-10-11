"""轻量级对话记忆缓存，供 DSPy 管道回退使用"""

from __future__ import annotations

import uuid
import logging
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


class ConversationMemoryCache:
    """维护最近几轮用户-助手对话，必要时作为检索回退使用"""

    def __init__(self, max_items: int = 3) -> None:
        self.max_items = max_items
        self._memory: Dict[str, List[Dict[str, object]]] = {}

    def add_exchange(self, session_id: Optional[str], user_query: str, assistant_response: str) -> None:
        """记录一轮对话，保留最近 N 条"""
        if not session_id:
            return

        trimmed_user = user_query.strip()
        trimmed_response = assistant_response.strip()
        if not trimmed_user and not trimmed_response:
            return

        content = f"用户: {trimmed_user}\n助手: {trimmed_response}"
        # 控制单条记忆长度，避免提示过长
        max_length = 1500
        if len(content) > max_length:
            content = content[:max_length] + "..."

        entry = {
            "chunk_id": f"mem-{uuid.uuid4()}",
            "content": content,
            "document": "conversation_memory",
            "score": 0.0,
            "metadata": {
                "type": "conversation_memory",
                "source": "assistant_reply"
            }
        }

        exchanges = self._memory.setdefault(session_id, [])
        exchanges.append(entry)
        logger.debug("memory_cache.add_exchange session=%s size=%s", session_id, len(exchanges))

        if len(exchanges) > self.max_items:
            # 只保留最近的 N 条
            self._memory[session_id] = exchanges[-self.max_items :]

    def get_context_chunks(self, session_id: Optional[str]) -> List[Dict[str, object]]:
        """以检索片段形式返回记忆"""
        if not session_id:
            return []
        exchanges = self._memory.get(session_id, [])
        logger.debug("memory_cache.get_context_chunks session=%s size=%s", session_id, len(exchanges))
        return [entry.copy() for entry in exchanges]

    def clear(self, session_id: Optional[str]) -> None:
        """清空指定会话的记忆"""
        if session_id and session_id in self._memory:
            logger.debug("memory_cache.clear session=%s", session_id)
            self._memory.pop(session_id, None)
