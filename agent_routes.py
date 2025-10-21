"""
Agent 对话路由
提供 Agent React、工具列表、助手列表等功能
"""

import json
import asyncio
import os
from typing import List, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from loguru import logger
from urllib.parse import quote

router = APIRouter()


# ==================== Pydantic Models ====================

class AgentReactRequest(BaseModel):
    """Agent React 请求"""
    messages: List[Dict[str, Any]]
    stream: bool = True


# ==================== API ====================

@router.get("/api/assistants")
async def get_assistants():
    """获取助手列表"""
    assistants = [
        {
            "id": "rag-assistant",
            "name": "RAG知识助手",
            "description": "基于文档的智能问答助手",
            "model": "dspy-rag",
            "capabilities": ["document_qa", "semantic_search", "clarification"],
            "status": "1",  # 1 表示激活状态
            "use-type": ["knowledge", "base", "semantic_search"]
        }
    ]

    return {
        "success": True,
        "data": {
            "assistants": assistants,
            "active": assistants[0]  # 默认第一个为活跃助手
        },
        "message": "获取助手信息成功"
    }


@router.get("/api/agent/tools")
async def get_agent_tools():
    """获取可用工具列表"""
    return {
        "tools": [
            {
                "name": "document_search",
                "description": "搜索文档内容",
                "status": "active",
                "type": "retrieval"
            },
            {
                "name": "semantic_search",
                "description": "语义相似度搜索",
                "status": "active",
                "type": "retrieval"
            },
            {
                "name": "tag_filter",
                "description": "标签过滤",
                "status": "active",
                "type": "filter"
            }
        ]
    }


@router.post("/api/agent/react")
async def agent_react(request: AgentReactRequest):
    """Agent React 接口（SSE 流式响应）- 真正调用 DSPy RAG"""
    from chat_routes import get_dspy_pipeline, get_conversation_manager

    def extract_text_content(content) -> str:
        """提取消息内容文本（支持字符串或数组格式）"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text' and 'text' in item:
                        texts.append(item['text'])
                    elif 'text' in item:
                        texts.append(str(item['text']))
            return ' '.join(texts)
        return str(content)

    async def event_stream():
        """生成 SSE 事件流"""
        try:
            # 提取最后一条用户消息
            user_messages = [msg for msg in request.messages if msg.get('role') == 'user']
            if not user_messages:
                yield f'data: {json.dumps({"type": "error", "content": "No user message found"})}\n\n'
                return

            last_message = user_messages[-1]
            query = extract_text_content(last_message.get('content', ''))

            logger.info(f"---- query extracted ---  : {query}")

            # === Phase 1: 初始化和意图识别 ===
            yield f'data: {json.dumps({"type": "reasoning", "content": "开始分析问题..."})}\n\n'
            await asyncio.sleep(0.05)

            # 工具状态：激活意图识别工具
            status_data = {"toolName": "intent_classifier", "status": "active"}
            canvas_content = {"canvas-type": "status", "canvas-source": status_data}
            yield f'data: {json.dumps({"type": "canvas", "content": canvas_content})}\n\n'
            await asyncio.sleep(0.05)

            # 获取 DSPy Pipeline 和对话管理器
            try:
                pipeline = get_dspy_pipeline()
                conv_manager = get_conversation_manager()
            except Exception as e:
                yield f'data: {json.dumps({"type": "error", "content": f"初始化失败: {str(e)}"})}\n\n'
                return

            # 构建对话历史
            history_messages = []
            for msg in request.messages[:-1]:
                role = msg.get('role', '')
                content = extract_text_content(msg.get('content', ''))
                if role in ['user', 'assistant'] and content:
                    history_messages.append(f"{role}: {content}")
            history = "\n".join(history_messages) if history_messages else ""

            # 调用 DSPy Pipeline 处理（先完成意图识别）
            result = pipeline.process_query(query, history)

            # 关闭意图识别工具状态
            status_data = {"toolName": "intent_classifier", "status": "inactive"}
            canvas_content = {"canvas-type": "status", "canvas-source": status_data}
            yield f'data: {json.dumps({"type": "canvas", "content": canvas_content})}\n\n'

            # === Phase 2: 意图识别结果 ===
            intent_info = result.get('intent')
            if intent_info:
                # 处理 intent 可能是字符串或字典的情况
                if isinstance(intent_info, dict):
                    intent_type = intent_info.get('intent', '未知')
                    confidence = intent_info.get('confidence', 0)
                    reasoning = intent_info.get('reasoning', '')

                    # 意图类型映射
                    intent_map = {
                        'question': '问答请求',
                        'chitchat': '💬 闲聊',
                        'clarification': '🤔 需要澄清',
                        'unknown': '未知意图'
                    }
                    intent_display = intent_map.get(intent_type, f'🔍 {intent_type}')

                    intent_text = f"识别意图: {intent_display}"
                    if confidence > 0:
                        intent_text += f" (置信度: {confidence*100:.0f}%)"

                    yield f'data: {json.dumps({"type": "reasoning", "content": intent_text})}\n\n'
                    await asyncio.sleep(0.05)

                    # 如果有推理说明，也显示出来（简短版本）
                    if reasoning and len(reasoning) <= 100:
                        reasoning_text = f"💭 {reasoning}"
                        yield f'data: {json.dumps({"type": "reasoning", "content": reasoning_text})}\n\n'
                        await asyncio.sleep(0.05)
                else:
                    # 如果是简单字符串
                    intent_text = f"识别意图: {intent_info}"
                    confidence = result.get('confidence', 0)
                    if confidence > 0:
                        intent_text += f" (置信度: {confidence*100:.0f}%)"
                    yield f'data: {json.dumps({"type": "reasoning", "content": intent_text})}\n\n'
                    await asyncio.sleep(0.05)

            # === Phase 3: 文档检索（仅当需要时）===
            sources = result.get('sources', [])
            result_type = result.get('type', 'answer')

            # 只有在需要文档检索的情况下才显示检索过程
            if sources or result_type in ['answer', 'clarification']:
                yield f'data: {json.dumps({"type": "reasoning", "content": "🔍 开始检索相关文档..."})}\n\n'
                await asyncio.sleep(0.05)

                # 工具状态：激活文档搜索
                status_data = {"toolName": "document_search", "status": "active"}
                canvas_content = {"canvas-type": "status", "canvas-source": status_data}
                yield f'data: {json.dumps({"type": "canvas", "content": canvas_content})}\n\n'
                await asyncio.sleep(0.05)

            # === Phase 4: 检索结果展示 ===
            if sources:
                yield f'data: {json.dumps({"type": "reasoning", "content": f"找到 {len(sources)} 个相关文档片段"})}\n\n'
                await asyncio.sleep(0.05)

                # 发送文件源信息（使用 files 类型）
                api_base_url = os.getenv("API_BASE_URL", f"http://localhost:{os.getenv('API_PORT', '8086')}")

                for i, source in enumerate(sources[:5]):  # 最多显示5个来源
                    doc_name = source.get('document', f'文档{i+1}')
                    chunk_db_id = source.get('chunk_db_id', '')  # 使用数据库主键ID

                    # 构造API导航URL（指向后端API，由后端重定向到前端）
                    if chunk_db_id and doc_name:
                        # URL编码文档名以处理特殊字符
                        encoded_doc_name = quote(doc_name)
                        file_url = f"{api_base_url}/api/view/document/{encoded_doc_name}/chunk/{chunk_db_id}"
                    else:
                        # 降级：没有chunk_db_id时使用传统路径
                        file_url = source.get('file_path', f'/docs/{doc_name}')

                    files_content = {
                        "fileName": doc_name,
                        "filePath": file_url,
                        "chunkDbId": chunk_db_id,  # 传递数据库主键ID
                        "sourceFile": doc_name  # 传递源文件名
                    }
                    yield f'data: {json.dumps({"type": "files", "content": files_content})}\n\n'
                    await asyncio.sleep(0.05)

                # 使用 canvas 展示详细的检索结果（markdown 格式）
                sources_markdown_parts = ["## 检索到的参考文档\n"]
                for i, s in enumerate(sources[:3]):  # 详细显示前3个
                    score = s.get('score', 0)
                    doc_name = s.get('document', 'Unknown')
                    content_preview = s.get('content', '')[:150]

                    sources_markdown_parts.append(f"### 📄 来源 {i+1}: {doc_name}\n")
                    sources_markdown_parts.append(f"**相似度**: {score*100:.1f}%\n\n")
                    sources_markdown_parts.append(f"**内容预览**:\n{content_preview}...\n")

                sources_markdown = "\n".join(sources_markdown_parts)
                canvas_content = {"canvas-type": "markdown", "canvas-source": sources_markdown}
                yield f'data: {json.dumps({"type": "canvas", "content": canvas_content})}\n\n'
                await asyncio.sleep(0.05)

            # 关闭文档搜索工具状态（如果打开了）
            if sources or result_type in ['answer', 'clarification']:
                status_data = {"toolName": "document_search", "status": "inactive"}
                canvas_content = {"canvas-type": "status", "canvas-source": status_data}
                yield f'data: {json.dumps({"type": "canvas", "content": canvas_content})}\n\n'

            # === Phase 5: 生成回答 ===
            # 根据不同类型显示不同的生成提示
            if result_type == 'chitchat':
                yield f'data: {json.dumps({"type": "reasoning", "content": "💬 生成回复..."})}\n\n'
                await asyncio.sleep(0.05)
            elif sources:
                yield f'data: {json.dumps({"type": "reasoning", "content": "💡 基于检索结果生成回答..."})}\n\n'
                await asyncio.sleep(0.05)

                # 工具状态：激活语义搜索
                status_data = {"toolName": "semantic_search", "status": "active"}
                canvas_content = {"canvas-type": "status", "canvas-source": status_data}
                yield f'data: {json.dumps({"type": "canvas", "content": canvas_content})}\n\n'
                await asyncio.sleep(0.05)
            else:
                yield f'data: {json.dumps({"type": "reasoning", "content": "💡 生成回答..."})}\n\n'
                await asyncio.sleep(0.05)

            # === Phase 6: 发送正式回复（流式） ===
            result_type = result.get('type', 'answer')
            response_text = ""

            if result_type == 'clarification':
                # 需要澄清的情况
                response_text = result.get('question', '抱歉，我需要更多信息。')
                yield f'data: {json.dumps({"type": "content", "content": response_text})}\n\n'
                await asyncio.sleep(0.05)

                if result.get('options'):
                    choice_text = "\n\n请选择："
                    yield f'data: {json.dumps({"type": "content", "content": choice_text})}\n\n'
                    await asyncio.sleep(0.05)
                    for i, opt in enumerate(result['options']):
                        option_text = f"\n{i+1}. {opt}"
                        yield f'data: {json.dumps({"type": "content", "content": option_text})}\n\n'
                        await asyncio.sleep(0.05)

            elif result_type == 'chitchat':
                response_text = result.get('response', '您好！有什么我可以帮您的吗？')
                # 分段发送（模拟流式）
                for chunk in [response_text[i:i+50] for i in range(0, len(response_text), 50)]:
                    yield f'data: {json.dumps({"type": "content", "content": chunk})}\n\n'
                    await asyncio.sleep(0.05)

            elif result_type == 'no_results':
                response_text = result.get('response', '抱歉，我没有找到相关的文档内容。')
                yield f'data: {json.dumps({"type": "content", "content": response_text})}\n\n'
                await asyncio.sleep(0.05)

                # 提供建议
                suggestions = [
                    "\n\n您可以尝试：",
                    "\n1. 换一种表述方式",
                    "\n2. 使用更具体的关键词",
                    "\n3. 查看可用的文档列表"
                ]
                for suggestion in suggestions:
                    yield f'data: {json.dumps({"type": "content", "content": suggestion})}\n\n'
                    await asyncio.sleep(0.05)

            else:  # answer type - 标准回答
                response_text = result.get('response', '抱歉，无法生成回答。')

                # 流式分段发送（按句子或换行符分割）
                sentences = response_text.split('\n')
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        prefix = "" if i == 0 else "\n"
                        content = prefix + sentence
                        yield f'data: {json.dumps({"type": "content", "content": content})}\n\n'
                        await asyncio.sleep(0.05)

            # 关闭语义搜索工具状态（如果打开了）
            if sources:
                status_data = {"toolName": "semantic_search", "status": "inactive"}
                canvas_content = {"canvas-type": "status", "canvas-source": status_data}
                yield f'data: {json.dumps({"type": "canvas", "content": canvas_content})}\n\n'

            # === Phase 7: 附加信息（可选）===
            # 如果有置信度低的情况，提供警告
            if result.get('confidence', 1.0) < 0.5:
                warning_text = "⚠️ 提示：回答的置信度较低，建议人工核实"
                yield f'data: {json.dumps({"type": "reasoning", "content": warning_text})}\n\n'
                await asyncio.sleep(0.05)

            # 如果需要，可以添加统计图表
            if sources and len(sources) > 1:
                # 创建一个简单的来源统计 HTML 表格
                stats_html = "<div style='padding: 10px; background: #f5f5f5; border-radius: 5px;'>"
                stats_html += "<h4 style='margin-top: 0;'>📊 检索统计</h4>"
                stats_html += f"<p><strong>检索到的文档数量：</strong> {len(sources)}</p>"
                if sources:
                    avg_score = sum(s.get('score', 0) for s in sources) / len(sources)
                    stats_html += f"<p><strong>平均相似度：</strong> {avg_score*100:.1f}%</p>"
                stats_html += "</div>"

                canvas_content = {"canvas-type": "html", "canvas-source": stats_html}
                yield f'data: {json.dumps({"type": "canvas", "content": canvas_content})}\n\n'
                await asyncio.sleep(0.05)

            # === Phase 8: 完成标记 ===
            yield f'data: {json.dumps({"type": "done"})}\n\n'

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            print(f"Agent react error: {error_detail}")

            # 发送错误消息
            error_msg = f"❌ 处理请求时发生错误: {str(e)}"
            yield f'data: {json.dumps({"type": "reasoning", "content": error_msg})}\n\n'
            error_content = "抱歉，处理您的请求时遇到了问题。请稍后再试。"
            yield f'data: {json.dumps({"type": "content", "content": error_content})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
            "Transfer-Encoding": "chunked"
        }
    )
