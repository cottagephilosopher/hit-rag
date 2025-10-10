"""
对话系统模块
使用 DSPy 实现智能 RAG 对话功能
"""

from .conversation_manager import ConversationManager
from .dspy_pipeline import DSPyRAGPipeline

__all__ = ['ConversationManager', 'DSPyRAGPipeline']
