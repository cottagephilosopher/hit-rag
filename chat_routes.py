"""
对话系统 API 路由
提供对话消息发送、历史查询等接口
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json

from chat.conversation_manager import ConversationManager
from chat.dspy_pipeline import DSPyRAGPipeline
from vector_db.vectorization_manager import VectorizationManager

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# 全局实例（懒加载）
conversation_manager = None
dspy_pipeline = None


def get_conversation_manager() -> ConversationManager:
    """获取对话管理器实例"""
    global conversation_manager
    if conversation_manager is None:
        conversation_manager = ConversationManager()
        logger.info("✅ ConversationManager initialized")
    return conversation_manager


def get_dspy_pipeline() -> DSPyRAGPipeline:
    """获取 DSPy Pipeline 实例"""
    global dspy_pipeline
    if dspy_pipeline is None:
        try:
            # 获取向量存储
            vectorization_manager = VectorizationManager()
            vector_store = vectorization_manager.vector_store

            # 初始化 DSPy Pipeline
            dspy_pipeline = DSPyRAGPipeline(
                vector_store=vector_store,
                llm_model="gpt-4o-mini",
                temperature=0.7,
                confidence_threshold=0.5
            )
            logger.info("✅ DSPy Pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DSPy Pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"DSPy初始化失败: {str(e)}")

    return dspy_pipeline


# ==================== Pydantic Models ====================

class ChatMessageRequest(BaseModel):
    """发送消息请求"""
    session_id: Optional[str] = None  # 如果为空，创建新会话
    message: str
    context: Optional[Dict[str, Any]] = None  # 文档过滤、标签过滤等


class ChatMessageResponse(BaseModel):
    """消息响应"""
    session_id: str
    response: Dict[str, Any]  # 包含 type, content, sources 等
    conversation_id: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """对话历史响应"""
    session_id: str
    messages: List[Dict[str, Any]]


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    created_at: str
    last_activity: str
    status: str
    metadata: Dict[str, Any]


# ==================== API Routes ====================

@router.post("/api/chat/message/stream")
async def send_message_stream(request: ChatMessageRequest):
    """
    发送消息并获取流式回复（SSE）

    处理流程（流式）：
    1. 创建或获取会话
    2. 保存用户消息
    3. 通过 DSPy Pipeline 流式处理查询
    4. 每个步骤完成时立即推送 SSE 事件
    5. 最后保存助手回复
    """
    try:
        manager = get_conversation_manager()
        pipeline = get_dspy_pipeline()

        # 1. 创建或获取会话
        if request.session_id:
            session_id = request.session_id
            session = manager.get_session_info(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="会话不存在")
        else:
            session_id = manager.create_session(metadata=request.context or {})
            logger.info(f"📝 Created new session: {session_id}")
        logger.debug("chat.message.stream using session %s", session_id)

        # 2. 保存用户消息
        manager.add_message(
            session_id=session_id,
            role="user",
            content=request.message
        )

        # 3. 获取对话历史
        conversation_history = manager.format_history_for_llm(session_id, limit=5)

        # 4. 准备过滤条件
        filters = None
        if request.context:
            pass  # TODO: 根据 context 构建过滤条件

        # 5. 定义 SSE 生成器
        async def event_generator():
            """SSE 事件生成器 - 符合前端标准格式"""
            try:
                final_content = ""
                final_sources = []

                # 流式处理查询
                async for chunk in pipeline.process_query_stream(
                    user_query=request.message,
                    conversation_history=conversation_history,
                    filters=filters,
                    session_id=session_id
                ):
                    # 按照标准 SSE 格式发送: data: {...}\n\n
                    chunk_type = chunk.get("type", "unknown")

                    # 保存最终内容用于存储到数据库
                    if chunk_type == "content":
                        final_content += chunk.get("content", "")
                    elif chunk_type == "files":
                        final_sources.append(chunk.get("content", {}))

                    # 发送标准 SSE 格式
                    event_data = json.dumps(chunk, ensure_ascii=False)
                    yield f"data: {event_data}\n\n"

                # 发送完成标记
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

                # 保存助手回复到数据库
                if final_content:
                    manager.add_message(
                        session_id=session_id,
                        role="assistant",
                        content=final_content,
                        intent="question",
                        sources=final_sources,
                        metadata={}
                    )

            except Exception as e:
                logger.error(f"❌ Stream error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
                yield f"event: error\ndata: {error_data}\n\n"

        # 返回 StreamingResponse
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"流式消息处理失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=f"流式消息处理失败: {str(e)}")


@router.post("/api/chat/message", response_model=ChatMessageResponse)
async def send_message(request: ChatMessageRequest):
    """
    发送消息并获取回复

    处理流程：
    1. 创建或获取会话
    2. 保存用户消息
    3. 通过 DSPy Pipeline 处理查询
    4. 保存助手回复
    5. 返回结果
    """
    try:
        manager = get_conversation_manager()
        pipeline = get_dspy_pipeline()

        # 1. 创建或获取会话
        if request.session_id:
            session_id = request.session_id
            # 验证会话是否存在
            session = manager.get_session_info(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="会话不存在")
        else:
            # 创建新会话
            session_id = manager.create_session(metadata=request.context or {})
            logger.info(f"📝 Created new session: {session_id}")
        logger.debug("chat.message using session %s", session_id)

        # 2. 保存用户消息
        manager.add_message(
            session_id=session_id,
            role="user",
            content=request.message
        )

        # 3. 获取对话历史
        conversation_history = manager.format_history_for_llm(session_id, limit=5)

        # 4. 通过 DSPy Pipeline 处理查询
        logger.info(f"🤖 Processing query for session: {session_id}")

        # 准备过滤条件
        filters = None
        if request.context:
            # TODO: 根据 context 构建 Milvus 过滤条件
            pass

        result = pipeline.process_query(
            user_query=request.message,
            conversation_history=conversation_history,
            filters=filters,
            session_id=session_id
        )

        # 5. 构建响应
        if result['type'] in {'answer', 'answer_with_clarification', 'web_search_answer'}:
            # 正常回答
            filtered_sources = [
                chunk for chunk in result.get('sources', [])
                if chunk.get('metadata', {}).get('type') not in {"conversation_history", "conversation_memory"}
            ]
            response_content = {
                "type": "answer",
                "answer_type": result['type'],
                "content": result['response'],
                "sources": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "document": chunk.get("document", ""),
                        "content": chunk.get("content", "")[:200] + "...",  # 截断显示
                        "score": chunk.get("score", 0.0),
                        "source_type": chunk.get("metadata", {}).get("type"),
                        "url": chunk.get("metadata", {}).get("url")
                    }
                    for chunk in filtered_sources[:5]
                ],
                "confidence": result.get('confidence', 0.0)
            }

            # 保存助手回复
            manager.add_message(
                session_id=session_id,
                role="assistant",
                content=result['response'],
                intent="question",
                sources=filtered_sources,
                metadata={
                    "confidence": result.get('confidence'),
                    "rewritten_query": result.get('rewrite', {}).get('rewritten_query')
                }
            )

        elif result['type'] == 'clarification':
            # 需要澄清
            response_content = {
                "type": "clarification",
                "question": result['question'],
                "options": result.get('options', []),
                "sources": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "document": chunk.get("document", ""),
                        "content": chunk.get("content", "")[:200] + "..."
                    }
                    for chunk in result.get('sources', [])[:3]
                ]
            }

            # 保存澄清消息
            manager.add_message(
                session_id=session_id,
                role="assistant",
                content=result['question'],
                intent="clarification",
                metadata={"options": result.get('options', [])}
            )

        elif result['type'] == 'chitchat':
            # 闲聊
            response_content = {
                "type": "chitchat",
                "content": result['response']
            }

            manager.add_message(
                session_id=session_id,
                role="assistant",
                content=result['response'],
                intent="chitchat"
            )

        else:
            # 无结果
            response_content = {
                "type": "no_results",
                "content": result.get('response', "抱歉，我没有找到相关信息。")
            }

            manager.add_message(
                session_id=session_id,
                role="assistant",
                content=response_content['content'],
                intent="question"
            )

        return ChatMessageResponse(
            session_id=session_id,
            response=response_content
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"消息处理失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=f"消息处理失败: {str(e)}")


@router.get("/api/chat/sessions/{session_id}/history", response_model=ChatHistoryResponse)
async def get_history(session_id: str, limit: int = 20):
    """获取对话历史"""
    try:
        manager = get_conversation_manager()

        # 验证会话是否存在
        session = manager.get_session_info(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        # 获取消息历史
        messages = manager.get_conversation_history(session_id, limit=limit)

        return ChatHistoryResponse(
            session_id=session_id,
            messages=messages
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史失败: {str(e)}")


@router.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    try:
        manager = get_conversation_manager()
        success = manager.delete_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")

        global dspy_pipeline
        try:
            if dspy_pipeline is not None:
                dspy_pipeline.memory_cache.clear(session_id)
        except Exception as exc:  # noqa: BLE001 - 仅记录缓存清理失败
            logger.warning("清理对话记忆缓存失败: %s", exc)

        return {"message": "会话已删除", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")


@router.get("/api/chat/sessions", response_model=List[SessionInfo])
async def list_sessions(limit: int = 10):
    """获取活跃会话列表"""
    try:
        manager = get_conversation_manager()
        sessions = manager.get_active_sessions(limit=limit)

        return [SessionInfo(**session) for session in sessions]

    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@router.post("/api/chat/sessions/{session_id}/archive")
async def archive_session(session_id: str):
    """归档会话"""
    try:
        manager = get_conversation_manager()
        success = manager.archive_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")

        return {"message": "会话已归档", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"归档会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"归档会话失败: {str(e)}")
