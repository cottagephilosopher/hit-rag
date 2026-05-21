"""
DSPy RAG Pipeline 实现
整合意图识别、查询改写、检索、评估和回复生成
"""

import dspy
import json
import os
import requests
import asyncio
import uuid
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

from .dspy_signatures import (
    IntentClassification,
    QueryRewrite,
    TagIdentification,
    ConfidenceEvaluation,
    ClarificationGeneration,
    ResponseGeneration
)
from .memory_cache import ConversationMemoryCache
from .web_search import WebSearchClient


class DSPyRAGPipeline:
    """DSPy RAG Pipeline 主类"""

    def __init__(
        self,
        vector_store=None,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        confidence_threshold: float = None
    ):
        """
        初始化 DSPy Pipeline

        Args:
            vector_store: 向量存储实例（Milvus）
            llm_model: LLM 模型名称
            temperature: 生成温度
            confidence_threshold: 置信度阈值（可选，默认从环境变量读取）
        """
        self.vector_store = vector_store

        # 读取闲聊模式配置
        self.enable_chat_mode = os.getenv('ENABLE_CHAT_MODE', 'false').lower() == 'true'
        self.chat_mode_threshold = float(os.getenv('CHAT_MODE_THRESHOLD', '0.7'))

        # 读取自动标签识别配置
        self.enable_auto_tag_filter = os.getenv('ENABLE_AUTO_TAG_FILTER', 'true').lower() == 'true'
        self.auto_tag_filter_threshold = float(os.getenv('AUTO_TAG_FILTER_THRESHOLD', '0.5'))

        # 读取 RAG Pipeline 阈值配置
        self.confidence_threshold = confidence_threshold or float(os.getenv('RAG_CONFIDENCE_THRESHOLD', '0.5'))

        # 相关性阈值
        self.rerank_score_threshold = float(os.getenv('RAG_RERANK_SCORE_THRESHOLD', '0.2'))
        self.l2_distance_threshold = float(os.getenv('RAG_L2_DISTANCE_THRESHOLD', '1.2'))

        # 检索质量阈值
        self.rerank_good_threshold = float(os.getenv('RAG_RERANK_GOOD_THRESHOLD', '0.2'))
        self.rerank_excellent_threshold = float(os.getenv('RAG_RERANK_EXCELLENT_THRESHOLD', '0.3'))
        self.l2_good_threshold = float(os.getenv('RAG_L2_GOOD_THRESHOLD', '1.2'))
        self.l2_excellent_threshold = float(os.getenv('RAG_L2_EXCELLENT_THRESHOLD', '1.0'))

        # 检索数量配置
        self.entity_top_k = int(os.getenv('RAG_ENTITY_TOP_K', '3'))
        self.multi_entity_dedup_limit = int(os.getenv('RAG_MULTI_ENTITY_DEDUP_LIMIT', '15'))
        self.rerank_top_n = int(os.getenv('RAG_RERANK_TOP_N', '8'))
        self.single_query_top_k = int(os.getenv('RAG_SINGLE_QUERY_TOP_K', '8'))
        self.files_display_limit = int(os.getenv('RAG_FILES_DISPLAY_LIMIT', '5'))
        self.web_search_provider = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo").lower()
        self.web_search_client = WebSearchClient(provider=self.web_search_provider)

        # 配置 DSPy LLM
        self._configure_dspy(llm_model, temperature)

        # 初始化各个模块
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.query_rewriter = dspy.ChainOfThought(QueryRewrite)
        self.tag_identifier = dspy.ChainOfThought(TagIdentification)
        self.confidence_evaluator = dspy.ChainOfThought(ConfidenceEvaluation)
        self.clarification_generator = dspy.ChainOfThought(ClarificationGeneration)
        self.response_generator = dspy.ChainOfThought(ResponseGeneration)
        memory_window = int(os.getenv('CHAT_MEMORY_MAX_ITEMS', '3'))
        self.memory_cache = ConversationMemoryCache(max_items=memory_window)

        # 缓存标签列表（避免每次查询都读数据库）
        self._available_tags_cache = None
        self._tags_cache_time = 0

        logger.info(f"✅ DSPy RAG Pipeline initialized with model: {llm_model}")
        logger.info(f"   闲聊模式: {'开启' if self.enable_chat_mode else '关闭'} (置信度阈值: {self.chat_mode_threshold})")
        logger.info(f"   自动标签筛选: {'开启' if self.enable_auto_tag_filter else '关闭'} (置信度阈值: {self.auto_tag_filter_threshold})")
        logger.info(f"   置信度阈值: {self.confidence_threshold}")
        logger.info(f"   相关性阈值: Rerank={self.rerank_score_threshold}, L2={self.l2_distance_threshold}")
        logger.info(f"   检索数量: 单实体={self.entity_top_k}, 去重={self.multi_entity_dedup_limit}, 重排={self.rerank_top_n}")
        logger.info(f"   对话记忆缓存窗口: {memory_window}")
        logger.info(f"   互联网检索Provider: {self.web_search_provider}")

    @staticmethod
    def _safe_parse_confidence(value: Any, default: float = 0.5) -> float:
        """
        安全地解析置信度值，容错处理各种格式

        Args:
            value: 待解析的值（可能是数字、字符串、或包含数字的文本）
            default: 解析失败时的默认值

        Returns:
            解析后的浮点数
        """
        try:
            # 如果已经是数字类型，直接返回
            if isinstance(value, (int, float)):
                return float(value)

            # 转换为字符串并清理
            str_value = str(value).strip()

            # 尝试直接转换
            try:
                return float(str_value)
            except ValueError:
                pass

            # 尝试从文本中提取数字（如 "0.75" 或 "High - 0.75" 或 "置信度: 0.75"）
            import re
            # 匹配 0.0 到 1.0 范围的小数
            match = re.search(r'\b([0-1]?\.\d+|0|1)\b', str_value)
            if match:
                confidence = float(match.group(1))
                # 确保在合理范围内
                if 0.0 <= confidence <= 1.0:
                    logger.debug(f"从文本 '{str_value[:50]}...' 中提取置信度: {confidence}")
                    return confidence

            # 尝试映射文本描述到数值
            str_lower = str_value.lower()
            if 'high' in str_lower or '高' in str_lower:
                return 0.8
            elif 'medium' in str_lower or '中' in str_lower:
                return 0.5
            elif 'low' in str_lower or '低' in str_lower:
                return 0.3

            logger.warning(f"无法解析置信度值: '{str_value[:100]}', 使用默认值 {default}")
            return default

        except Exception as e:
            logger.warning(f"解析置信度时出错: {e}, 使用默认值 {default}")
            return default

    @staticmethod
    def _build_history_chunk(conversation_history: str) -> Optional[Dict[str, Any]]:
        """将纯文本对话历史包装成检索片段，用于回退"""
        cleaned = (conversation_history or "").strip()
        if not cleaned:
            return None

        max_length = 2000
        if len(cleaned) > max_length:
            cleaned = cleaned[-max_length:]

        return {
            "chunk_id": f"history-{uuid.uuid4()}",
            "content": cleaned,
            "document": "conversation_history",
            "score": 0.0,
            "metadata": {
                "type": "conversation_history",
                "source": "conversation_manager"
            }
        }

    @staticmethod
    def _env_bool(name: str, default: str = "false") -> bool:
        """Read a boolean environment variable."""
        return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}

    def _get_numeric_config(self, key: str, default: float) -> float:
        """Read a numeric RAG config value from DB, falling back to environment defaults."""
        try:
            from database import get_rag_config

            config = get_rag_config(key)
            if config and config.get("config_value") is not None:
                return float(config["config_value"])
        except Exception as exc:
            logger.debug(f"Unable to load RAG config {key} from DB: {exc}")

        return default

    def _load_web_search_config(self) -> Dict[str, Any]:
        """Load web-search fallback config dynamically so UI changes take effect."""
        enabled_default = 1.0 if self._env_bool("ENABLE_WEB_SEARCH_FALLBACK", "false") else 0.0
        return {
            "enabled": self._get_numeric_config("ENABLE_WEB_SEARCH_FALLBACK", enabled_default) >= 0.5,
            "max_results": int(self._get_numeric_config(
                "WEB_SEARCH_MAX_RESULTS",
                float(os.getenv("WEB_SEARCH_MAX_RESULTS", "5")),
            )),
            "timeout": float(self._get_numeric_config(
                "WEB_SEARCH_TIMEOUT_SECONDS",
                float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "5")),
            )),
        }

    def _web_results_to_chunks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert web-search results into the same source shape used by RAG chunks."""
        chunks = []
        for index, result in enumerate(results, 1):
            title = (result.get("title") or "互联网搜索结果").strip()
            url = (result.get("url") or "").strip()
            snippet = (result.get("snippet") or "").strip()
            provider = result.get("provider") or self.web_search_provider

            content_parts = [f"标题: {title}"]
            if snippet:
                content_parts.append(f"摘要: {snippet}")
            if url:
                content_parts.append(f"链接: {url}")

            chunks.append({
                "chunk_id": f"web-{index}",
                "content": "\n".join(content_parts),
                "score": 0.0,
                "document": f"互联网搜索: {title}",
                "metadata": {
                    "type": "web_search",
                    "source": provider,
                    "url": url,
                    "title": title,
                },
            })
        return chunks

    def _answer_with_web_search(
        self,
        user_query: str,
        search_query: str,
        conversation_history: str = "",
        reason: str = "no_results",
    ) -> Optional[Dict[str, Any]]:
        """Use internet search as a fallback answer source when knowledge-base retrieval fails."""
        config = self._load_web_search_config()
        if not config["enabled"]:
            logger.info("  🌐 Web search fallback is disabled")
            return None

        query = (search_query or user_query or "").strip()
        if not query:
            return None

        logger.info(f"  🌐 Running web search fallback ({reason}) with query: {query[:100]}...")
        results = self.web_search_client.search(
            query=query,
            max_results=config["max_results"],
            timeout=config["timeout"],
        )
        if not results:
            logger.info("  🌐 Web search fallback returned no results")
            return None

        web_chunks = self._web_results_to_chunks(results)
        intent_note = (
            "知识库检索未能提供足够依据，以下资料来自互联网搜索结果。"
            "回答时请明确基于互联网来源，并提醒用户核验。"
        )
        response_result = self.generate_response(
            user_query=user_query,
            retrieved_chunks=web_chunks,
            conversation_history=conversation_history,
            clarification_hint="",
            intent_note=intent_note,
        )

        response_text = (
            f"{response_result['response']}\n\n---\n"
            "⚠️ **提示**：知识库未找到足够依据，以上内容来自互联网搜索结果，请注意核验。"
        )

        return {
            "type": "web_search_answer",
            "response": response_text,
            "confidence": response_result.get("confidence", 0.0),
            "sources": web_chunks,
            "source_ids": response_result.get("source_ids", [c["chunk_id"] for c in web_chunks]),
            "web_search": {
                "provider": self.web_search_provider,
                "query": query,
                "reason": reason,
            },
        }

    def _build_file_source_payload(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a stream file/source payload for local chunks and web-search chunks."""
        metadata = chunk.get("metadata", {}) or {}
        metadata_type = metadata.get("type")
        if metadata_type in {"conversation_history", "conversation_memory"}:
            return None

        if metadata_type == "web_search":
            title = metadata.get("title") or chunk.get("document", "互联网搜索结果")
            url = metadata.get("url") or ""
            if not url:
                return None
            return {
                "fileName": title,
                "filePath": url,
                "chunkDbId": chunk.get("chunk_id", ""),
                "sourceFile": title,
                "sourceType": "web_search",
            }

        import os
        from urllib.parse import quote

        api_base_url = os.getenv("API_BASE_URL", "http://localhost:8086")
        doc_name = chunk.get("document", "Unknown")
        chunk_db_id = chunk.get("chunk_db_id", "")

        if chunk_db_id and doc_name and doc_name != "Unknown":
            encoded_doc_name = quote(doc_name)
            file_path = f"{api_base_url}/api/view/document/{encoded_doc_name}/chunk/{chunk_db_id}"
        else:
            file_path = metadata.get("file_path", "") or f"#chunk-{chunk_db_id}"

        if not doc_name:
            return None

        return {
            "fileName": doc_name,
            "filePath": file_path,
            "chunkDbId": chunk_db_id,
            "sourceFile": doc_name,
        }

    def _build_web_search_stream_events(self, web_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build SSE chunks for a completed web-search fallback answer."""
        events = [
            {
                "type": "reasoning",
                "content": f"✓ 互联网检索完成\n  • 找到 {len(web_result.get('sources', []))} 条结果",
            },
            {"type": "content", "content": web_result["response"]},
        ]

        for chunk in web_result.get("sources", [])[:self.files_display_limit]:
            payload = self._build_file_source_payload(chunk)
            if payload:
                events.append({"type": "files", "content": payload})

        events.append({"type": "done"})
        return events

    def _configure_dspy(self, model: str, temperature: float):
        """配置 DSPy 的 LLM"""
        try:
            import os

            # 检查是否使用 Azure OpenAI
            llm_provider = os.getenv('LLM_PROVIDER', 'openai')

            if llm_provider == 'azure':
                # Azure OpenAI 配置
                azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                azure_key = os.getenv('AZURE_OPENAI_API_KEY')
                azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')
                azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')

                # 使用 azure/ 前缀格式
                model_name = f"azure/{azure_deployment}"

                lm = dspy.LM(
                    model=model_name,
                    api_base=azure_endpoint,
                    api_key=azure_key,
                    api_version=azure_api_version,
                    temperature=temperature,
                    max_tokens=2000
                )
                logger.info(f"✅ DSPy configured with Azure OpenAI: {azure_deployment}")

            elif llm_provider == 'dashscope':
                # DashScope 配置（阿里云灵积）
                # 使用 OpenAI 兼容模式，通过 api_base 指定 DashScope 端点
                dashscope_key = os.getenv('DASHSCOPE_API_KEY')
                dashscope_model = os.getenv('DASHSCOPE_MODEL', 'qwen-max')
                dashscope_endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1"

                # 使用 openai/ 前缀 + api_base 来调用 OpenAI 兼容端点
                model_name = f"openai/{dashscope_model}"

                lm = dspy.LM(
                    model=model_name,
                    api_key=dashscope_key,
                    api_base=dashscope_endpoint,
                    temperature=temperature,
                    max_tokens=2000
                )
                logger.info(f"✅ DSPy configured with DashScope (OpenAI-compatible): {dashscope_model}")

            else:
                # 标准 OpenAI 配置
                api_base = os.getenv('OPENAI_API_BASE') or os.getenv('API_BASE')
                api_key = os.getenv('OPENAI_API_KEY') or os.getenv('API_KEY')
                model = os.getenv('OPENAI_MODEL') or os.getenv('MODEL_NAME', 'gpt-4o')

                # 如果模型名不是以 openai/ 开头，添加前缀以避免被 LiteLLM 误识别
                # 例如 claude-sonnet-4-20250514 会被识别为 Anthropic，需要加 openai/ 前缀
                if not model.startswith('openai/'):
                    model = f'openai/{model}'
                    logger.info(f"🔄 添加 openai/ 前缀以避免模型名被误识别: {model}")

                lm = dspy.LM(model=model, api_base=api_base, api_key=api_key, temperature=temperature, max_tokens=2000)

                if api_base:
                    logger.info(f"✅ DSPy configured with OpenAI-compatible endpoint: {model} @ {api_base}")
                else:
                    logger.info(f"✅ DSPy configured with OpenAI: {model}")

            dspy.configure(lm=lm)

        except Exception as e:
            logger.error(f"❌ Failed to configure DSPy: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def classify_intent(
        self,
        user_query: str,
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """
        步骤1: 识别用户意图

        Args:
            user_query: 用户查询
            conversation_history: 对话历史

        Returns:
            意图分类结果
        """
        try:
            result = self.intent_classifier(
                conversation_history=conversation_history or "无历史对话",
                user_query=user_query
            )

            return {
                "intent": result.intent.lower(),
                "confidence": self._safe_parse_confidence(result.confidence),
                "business_relevance": result.business_relevance.lower(),
                "reasoning": result.reasoning
            }
        except Exception as e:
            logger.error(f"❌ Intent classification failed: {e}")
            # 默认认为是问答
            return {
                "intent": "question",
                "confidence": 0.5,
                "business_relevance": "medium",
                "reasoning": f"分类失败，默认为问答: {e}"
            }

    def rewrite_query(
        self,
        user_query: str,
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """
        步骤2: 改写查询

        Args:
            user_query: 用户查询
            conversation_history: 对话历史

        Returns:
            改写结果
        """
        try:
            result = self.query_rewriter(
                conversation_history=conversation_history or "无历史对话",
                user_query=user_query
            )

            # 解析 key_entities
            try:
                key_entities = json.loads(result.key_entities)
            except:
                key_entities = [result.key_entities]

            return {
                "rewritten_query": result.rewritten_query,
                "key_entities": key_entities,
                "search_strategy": result.search_strategy
            }
        except Exception as e:
            logger.error(f"❌ Query rewrite failed: {e}")
            return {
                "rewritten_query": user_query,
                "key_entities": [],
                "search_strategy": "semantic"
            }

    def _get_available_tags(self) -> List[str]:
        """
        获取系统可用标签列表（带缓存）
        缓存5分钟，减少数据库查询
        """
        import time
        current_time = time.time()

        # 缓存5分钟
        if self._available_tags_cache and (current_time - self._tags_cache_time) < 300:
            return self._available_tags_cache

        try:
            # 从数据库获取标签
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from database import get_connection

            with get_connection() as conn:
                # 获取 user_tags
                user_tags = conn.execute("""
                    SELECT DISTINCT user_tag FROM document_chunks
                    WHERE user_tag IS NOT NULL AND user_tag != ''
                """).fetchall()

                # 获取 content_tags
                content_tags_rows = conn.execute("""
                    SELECT DISTINCT content_tags FROM document_chunks
                    WHERE content_tags IS NOT NULL AND content_tags != '[]'
                """).fetchall()

                # 获取文档级标签
                doc_tags = conn.execute("""
                    SELECT DISTINCT tag_text FROM document_tags
                """).fetchall()

            tags = set()

            # 处理 user_tags
            for row in user_tags:
                if row['user_tag']:
                    tags.add(row['user_tag'].lstrip('@'))

            # 处理 content_tags
            for row in content_tags_rows:
                try:
                    tag_list = json.loads(row['content_tags'])
                    if isinstance(tag_list, list):
                        for tag in tag_list:
                            clean_tag = tag.lstrip('@') if isinstance(tag, str) else tag
                            if clean_tag:
                                tags.add(clean_tag)
                except:
                    continue

            # 处理文档级标签
            for row in doc_tags:
                if row['tag_text']:
                    tags.add(row['tag_text'].strip().lstrip('@'))

            tag_list = sorted(list(tags))
            self._available_tags_cache = tag_list
            self._tags_cache_time = current_time

            logger.debug(f"Loaded {len(tag_list)} available tags")
            return tag_list

        except Exception as e:
            logger.error(f"Failed to load tags: {e}")
            return []

    def identify_relevant_tags(self, user_query: str) -> Dict[str, Any]:
        """
        识别用户查询中的相关标签

        Args:
            user_query: 用户查询

        Returns:
            标签识别结果
        """
        try:
            available_tags = self._get_available_tags()

            if not available_tags:
                logger.warning("No available tags found")
                return {
                    "relevant_tags": [],
                    "confidence": 0.0,
                    "reasoning": "系统标签库为空"
                }

            # 调用 DSPy 标签识别
            tags_json = json.dumps(available_tags, ensure_ascii=False)

            result = self.tag_identifier(
                user_query=user_query,
                available_tags=tags_json
            )

            # 解析 relevant_tags
            try:
                relevant_tags = json.loads(result.relevant_tags)
                if not isinstance(relevant_tags, list):
                    relevant_tags = []
            except:
                relevant_tags = []

            return {
                "relevant_tags": relevant_tags,
                "confidence": self._safe_parse_confidence(result.confidence),
                "reasoning": result.reasoning
            }

        except Exception as e:
            logger.error(f"❌ Tag identification failed: {e}")
            return {
                "relevant_tags": [],
                "confidence": 0.0,
                "reasoning": f"标签识别失败: {e}"
            }

    def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        步骤3: 向量检索

        Args:
            query: 查询文本
            top_k: 返回前K个结果
            filters: 过滤条件

        Returns:
            检索到的文档片段列表
        """
        if not self.vector_store:
            logger.warning("⚠️  No vector store configured")
            return []

        try:
            # 使用 search_with_score 方法（返回带分数的结果）
            results = self.vector_store.search_with_score(
                query=query,
                k=top_k,
                filters=filters
            )

            # 格式化结果 - search_with_score 返回 [(Document, score), ...]
            # Document 是 LangChain Document，有 .page_content 和 .metadata
            chunks = []
            for doc, score in results:
                metadata = doc.metadata
                chunks.append({
                    "chunk_db_id": metadata.get("chunk_db_id") or metadata.get("pk"),  # 数据库主键ID
                    "chunk_sequence": metadata.get("chunk_sequence", 0),  # 文档内顺序编号
                    "content": metadata.get("original_content") or doc.page_content,
                    "score": score,
                    "document": metadata.get("source_file", "Unknown"),
                    "metadata": metadata
                })

            logger.info(f"✅ Retrieved {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去重并按分数排序（L2距离越小越好）

        Args:
            chunks: 检索到的文档片段列表

        Returns:
            去重后的文档片段列表，按分数升序排序
        """
        seen_ids = set()
        unique_chunks = []

        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
            elif not chunk_id:
                # 如果没有chunk_id，用content的hash作为标识
                content_hash = hash(chunk.get('content', ''))
                if content_hash not in seen_ids:
                    seen_ids.add(content_hash)
                    unique_chunks.append(chunk)

        # 按L2距离升序排序（分数越小越好）
        unique_chunks.sort(key=lambda x: x.get('score', float('inf')))
        return unique_chunks

    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        使用 DashScope 重排序检索结果

        Args:
            query: 原始用户查询
            chunks: 检索到的文档片段列表
            top_n: 返回前N个结果

        Returns:
            重排序后的文档片段列表
        """
        if not chunks:
            return chunks

        try:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                logger.warning("⚠️  DASHSCOPE_API_KEY not found, skipping rerank")
                return chunks[:top_n]

            # 准备文档列表
            documents = [chunk.get('content', '') for chunk in chunks]

            # 调用 DashScope rerank API
            url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gte-rerank-v2",
                "input": {
                    "query": query,
                    "documents": documents
                },
                "parameters": {
                    "return_documents": True,
                    "top_n": top_n
                }
            }

            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()

            result = response.json()

            # 解析重排序结果 - DashScope 成功时返回 output 字段
            if "output" in result and "results" in result["output"]:
                reranked_results = result["output"]["results"]
                reranked_chunks = []

                for item in reranked_results:
                    original_index = item.get("index")
                    relevance_score = item.get("relevance_score")

                    if original_index is not None and original_index < len(chunks):
                        chunk = chunks[original_index].copy()
                        chunk['rerank_score'] = relevance_score
                        reranked_chunks.append(chunk)

                logger.info(f"✅ Reranked {len(reranked_chunks)} chunks (scores: {[round(c.get('rerank_score', 0), 3) for c in reranked_chunks[:3]]}...)")
                return reranked_chunks
            else:
                logger.warning(f"⚠️  Rerank API returned unexpected format: {result}")
                return chunks[:top_n]

        except Exception as e:
            logger.error(f"❌ Rerank failed: {e}")
            # 降级：返回原始排序结果
            return chunks[:top_n]

    def evaluate_confidence(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        步骤4: 评估检索结果的充分性

        Args:
            user_query: 用户查询
            retrieved_chunks: 检索到的文档片段

        Returns:
            评估结果
        """
        try:
            # 格式化 chunks 为文本
            chunks_text = json.dumps(
                [{"content": c["content"], "source": c.get("document", "")}
                 for c in retrieved_chunks[:5]],  # 只取前3个避免太长
                ensure_ascii=False,
                indent=2
            )

            result = self.confidence_evaluator(
                user_query=user_query,
                retrieved_chunks=chunks_text
            )

            return {
                "is_sufficient": result.is_sufficient.lower() == "yes",
                "has_ambiguity": result.has_ambiguity.lower() == "yes",
                "confidence": self._safe_parse_confidence(result.confidence),
                "ambiguity_type": result.ambiguity_type,
                "clarification_hint": result.clarification_hint,
                "reasoning": result.reasoning
            }
        except Exception as e:
            logger.error(f"❌ Confidence evaluation failed: {e}")
            # 如果评估失败，保守地认为信息充分（避免过度反问）
            return {
                "is_sufficient": True,
                "has_ambiguity": False,
                "confidence": 0.5,
                "ambiguity_type": "none",
                "clarification_hint": "",
                "reasoning": f"评估失败: {e}"
            }

    def generate_clarification(
        self,
        user_query: str,
        missing_info: List[str],
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """
        步骤5: 生成澄清问题

        Args:
            user_query: 用户查询
            missing_info: 缺失的信息
            conversation_history: 对话历史

        Returns:
            澄清问题
        """
        try:
            missing_info_text = json.dumps(missing_info, ensure_ascii=False)

            result = self.clarification_generator(
                user_query=user_query,
                missing_info=missing_info_text,
                conversation_history=conversation_history or "无历史对话"
            )

            # 解析选项
            try:
                options = json.loads(result.suggested_options)
            except:
                options = []

            return {
                "question": result.clarification_question,
                "options": options
            }
        except Exception as e:
            logger.error(f"❌ Clarification generation failed: {e}")
            return {
                "question": f"抱歉，我需要更多信息。您能详细说明一下 {missing_info[0] if missing_info else '您的需求'} 吗？",
                "options": []
            }

    def generate_chitchat_response(
        self,
        user_query: str,
        conversation_history: str = ""
    ) -> str:
        """
        生成闲聊回复（使用 LLM）

        Args:
            user_query: 用户查询
            conversation_history: 对话历史

        Returns:
            生成的闲聊回复
        """
        try:
            # 使用 dspy.LM 直接生成回复
            lm = dspy.settings.lm

            system_prompt = """你是一个友好的文档助手。当用户进行闲聊时，请：
1. 保持友好、专业的态度
2. 简短回复，不要过于冗长
3. 适当引导用户提出文档相关问题
4. 如果用户问候，可以介绍你的功能"""

            # 构建完整的提示文本
            full_prompt = system_prompt + "\n\n"

            # 添加对话历史
            if conversation_history:
                full_prompt += f"对话历史:\n{conversation_history}\n\n"

            # 添加当前用户查询
            full_prompt += f"用户: {user_query}\n助手:"

            # 调用 LLM（直接传入文本而不是消息列表）
            response = lm(full_prompt)

            # 提取响应文本（如果是列表则取第一个元素）
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0]
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)

            return response_text

        except Exception as e:
            logger.error(f"❌ Chitchat response generation failed: {e}")
            # 降级到默认回复
            return "您好！我是文档助手，专门帮助您查找和理解文档内容。有什么我可以帮您的吗？"

    def generate_response(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: str = "",
        clarification_hint: str = "",
        intent_note: str = ""
    ) -> Dict[str, Any]:
        """
        步骤6: 生成最终回复

        Args:
            user_query: 用户查询
            retrieved_chunks: 检索到的文档片段
            conversation_history: 对话历史
            clarification_hint: 可选的澄清提示（用于"先答后问"）
            intent_note: 意图识别备注

        Returns:
            生成的回复
        """
        try:
            # 格式化 chunks
            chunks_text = json.dumps(
                [{
                    "id": c.get("chunk_id"),
                    "content": c["content"],
                    "source": c.get("document", ""),
                    "score": c.get("score", 0.0)
                } for c in retrieved_chunks],
                ensure_ascii=False,
                indent=2
            )

            result = self.response_generator(
                conversation_history=conversation_history or "无历史对话",
                user_query=user_query,
                retrieved_chunks=chunks_text,
                clarification_hint=clarification_hint,
                intent_note=intent_note
            )

            # 解析 source_ids
            try:
                source_ids = json.loads(result.source_ids)
            except:
                source_ids = [c.get("chunk_id") for c in retrieved_chunks[:3]]

            return {
                "response": result.response,
                "source_ids": source_ids,
                "confidence": self._safe_parse_confidence(result.confidence),
                "sources": retrieved_chunks
            }
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return {
                "response": "抱歉，我在生成回复时遇到了问题。请重新表述您的问题。",
                "source_ids": [],
                "confidence": 0.0,
                "sources": []
            }

    def process_query(
        self,
        user_query: str,
        conversation_history: str = "",
        filters: Dict = None,
        force_answer: bool = False,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        完整处理用户查询（主入口）

        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            filters: 检索过滤条件
            force_answer: 是否强制回答（不生成澄清问题）
            session_id: 会话ID，用于记忆缓存

        Returns:
            处理结果
        """
        logger.info(f"🔍 Processing query: {user_query[:100]}...")

        # 1. 意图识别
        intent_result = self.classify_intent(user_query, conversation_history)
        logger.info(f"  Intent: {intent_result['intent']} (confidence: {intent_result['confidence']})")

        # 如果是闲聊且置信度高，根据配置决定回复方式
        if intent_result['intent'] == 'chitchat' and intent_result['confidence'] >= self.chat_mode_threshold:
            # 检查是否开启闲聊模式
            if self.enable_chat_mode:
                logger.info(f"  闲聊模式已开启，置信度 {intent_result['confidence']:.2f} >= {self.chat_mode_threshold}，使用 LLM 回复")
                llm_response = self.generate_chitchat_response(user_query, conversation_history)
                # 在回复末尾添加明显的 AI 生成提醒
                llm_response_with_notice = f"{llm_response}\n\n---\n⚠️ **提示**：此回复由 AI 生成，仅供参考，请注意甄别。"
                self.memory_cache.add_exchange(session_id, user_query, llm_response_with_notice)
                return {
                    "type": "chitchat",
                    "response": llm_response_with_notice,
                    "intent": intent_result,
                    "chat_mode": "llm"
                }
            else:
                # 未开启闲聊模式，使用固定回复
                logger.info(f"  闲聊模式未开启，使用固定回复")
                fixed_reply = "您好！我是文档助手，专门帮助您查找和理解文档内容。有什么我可以帮您的吗？"
                self.memory_cache.add_exchange(session_id, user_query, fixed_reply)
                return {
                    "type": "chitchat",
                    "response": fixed_reply,
                    "intent": intent_result,
                    "chat_mode": "fixed"
                }

        # 置信度不足或非闲聊意图，继续进行知识库检索
        intent_note = ""
        if intent_result['intent'] == 'chitchat' and intent_result['confidence'] < self.chat_mode_threshold:
            logger.info(f"  闲聊意图置信度不足 ({intent_result['confidence']:.2f} < {self.chat_mode_threshold})，继续检索知识库")
            intent_note = "检测到您的问题可能偏向闲聊，但我尝试在知识库中为您查找相关内容"

        # 2. 并行执行：查询改写 + 标签识别（如果开启）
        if self.enable_auto_tag_filter:
            logger.info("  🔄 Parallel processing: Query rewrite + Tag identification...")
        else:
            logger.info("  🔄 Query rewrite...")

        import concurrent.futures
        import threading

        rewrite_result = None
        tag_result = None
        rewrite_error = None
        tag_error = None

        def do_rewrite():
            nonlocal rewrite_result, rewrite_error
            try:
                rewrite_result = self.rewrite_query(user_query, conversation_history)
            except Exception as e:
                rewrite_error = e

        def do_tag_identification():
            nonlocal tag_result, tag_error
            try:
                tag_result = self.identify_relevant_tags(user_query)
            except Exception as e:
                tag_error = e

        # 根据配置决定是否并行执行标签识别
        if self.enable_auto_tag_filter:
            # 使用线程池并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_rewrite = executor.submit(do_rewrite)
                future_tags = executor.submit(do_tag_identification)

                # 等待两个任务完成
                concurrent.futures.wait([future_rewrite, future_tags])
        else:
            # 只执行查询改写
            do_rewrite()
            tag_result = {
                "relevant_tags": [],
                "confidence": 0.0,
                "reasoning": "自动标签识别已关闭"
            }

        # 处理查询改写结果
        if rewrite_error:
            logger.error(f"❌ Query rewrite failed: {rewrite_error}")
            rewrite_result = {
                "rewritten_query": user_query,
                "key_entities": [],
                "search_strategy": "semantic"
            }

        logger.info(f"  ✅ Rewritten query: {rewrite_result['rewritten_query'][:100]}...")
        optimized_query = rewrite_result['rewritten_query']

        # 处理标签识别结果（仅当开启自动标签识别时）
        if self.enable_auto_tag_filter:
            if tag_error:
                logger.error(f"❌ Tag identification failed: {tag_error}")
                tag_result = {
                    "relevant_tags": [],
                    "confidence": 0.0,
                    "reasoning": f"标签识别失败: {tag_error}"
                }

            relevant_tags = tag_result.get('relevant_tags', [])
            tag_confidence = tag_result.get('confidence', 0.0)

            # 如果识别到标签且置信度足够，添加到 filters
            if relevant_tags and tag_confidence >= self.auto_tag_filter_threshold:
                logger.info(f"  🏷️  Identified tags: {relevant_tags} (confidence: {tag_confidence:.2f})")
                if not filters:
                    filters = {}
                # 使用 content_tags 筛选（因为它会同时匹配 user_tag 和 content_tags）
                filters['content_tags'] = relevant_tags
            elif relevant_tags:
                logger.info(f"  🏷️  Tags identified but low confidence: {relevant_tags} (confidence: {tag_confidence:.2f} < {self.auto_tag_filter_threshold}), not using for filtering")
            else:
                logger.info(f"  🏷️  No relevant tags identified")
        else:
            logger.debug("  🏷️  Auto tag filter is disabled")

        # 3. 向量检索 - 利用 key_entities 做混合检索
        key_entities = rewrite_result.get('key_entities', [])

        if key_entities and len(key_entities) > 1:
            # 多实体：对每个关键实体检索，然后合并去重
            logger.info(f"  🔍 Multi-entity search with: {key_entities}")
            all_chunks = []
            for entity in key_entities:
                entity_chunks = self.retrieve_chunks(entity, top_k=3, filters=filters)
                all_chunks.extend(entity_chunks)
                logger.info(f"    - Entity '{entity}': found {len(entity_chunks)} chunks")

            # 去重并按分数排序，取更多候选（用于重排）
            chunks = self._deduplicate_chunks(all_chunks)[:15]
            logger.info(f"  ✅ After deduplication: {len(chunks)} unique chunks")

            # 使用原始查询进行重排序
            logger.info(f"  🔄 Reranking with original query: {user_query[:50]}...")
            chunks = self.rerank_chunks(user_query, chunks, top_n=8)
        else:
            # 单实体或无实体：直接用改写后的查询
            logger.info(f"  🔍 Single query search: {optimized_query[:50]}...")
            chunks = self.retrieve_chunks(optimized_query, top_k=8, filters=filters)

        memory_fallback = False
        if not chunks:
            memory_chunks = self.memory_cache.get_context_chunks(session_id)
            logger.info(f"  Memory fallback check (no retrieval): {len(memory_chunks)} cached exchanges")
            if memory_chunks:
                logger.info("  🔁 No retrieval results, falling back to conversation memory")
                chunks = memory_chunks
                memory_fallback = True
            else:
                history_chunk = self._build_history_chunk(conversation_history)
                if history_chunk:
                    logger.info("  🔁 No retrieval results, falling back to formatted conversation history")
                    chunks = [history_chunk]
                    memory_fallback = True
                else:
                    web_result = self._answer_with_web_search(
                        user_query=user_query,
                        search_query=optimized_query,
                        conversation_history=conversation_history,
                        reason="no_results"
                    )
                    if web_result:
                        self.memory_cache.add_exchange(session_id, user_query, web_result['response'])
                        return web_result

                    return {
                        "type": "no_results",
                        "response": "抱歉，我没有找到相关的文档内容。您可以换个方式提问吗？",
                        "intent": intent_result,
                        "rewrite": rewrite_result,
                        "sources": []
                    }

        # 3.5. 检查相似度分数，过滤掉不相关的检索结果
        has_rerank = not memory_fallback and any('rerank_score' in c for c in chunks)
        best_score = 1.0

        if not memory_fallback:
            if has_rerank:
                best_score = max(c.get('rerank_score', 0) for c in chunks) if chunks else 0
                score_threshold = self.rerank_score_threshold
                is_relevant = best_score >= score_threshold
                score_type = "rerank"
            else:
                best_score = min(c.get('score', float('inf')) for c in chunks) if chunks else float('inf')
                score_threshold = self.l2_distance_threshold
                is_relevant = best_score <= score_threshold
                score_type = "L2"

            if not is_relevant:
                memory_chunks = self.memory_cache.get_context_chunks(session_id)
                logger.info(f"  Memory fallback check (low relevance): {len(memory_chunks)} cached exchanges")
                if memory_chunks:
                    logger.info(f"  {score_type} score below threshold ({best_score:.3f}); using conversation memory fallback")
                    chunks = memory_chunks
                    memory_fallback = True
                    has_rerank = False
                else:
                    history_chunk = self._build_history_chunk(conversation_history)
                    if history_chunk:
                        logger.info(f"  {score_type} score below threshold ({best_score:.3f}); using conversation history fallback")
                        chunks = [history_chunk]
                        memory_fallback = True
                        has_rerank = False
                    else:
                        logger.info(f"  {score_type} score indicates no relevant results: {best_score:.3f}")
                        web_result = self._answer_with_web_search(
                            user_query=user_query,
                            search_query=optimized_query,
                            conversation_history=conversation_history,
                            reason="low_relevance"
                        )
                        if web_result:
                            self.memory_cache.add_exchange(session_id, user_query, web_result['response'])
                            return web_result

                        return {
                            "type": "no_results",
                            "response": "抱歉，我的知识库中没有找到相关的文档内容。我只能回答与已有文档相关的问题。",
                            "intent": intent_result,
                            "rewrite": rewrite_result,
                            "sources": []
                        }

        if memory_fallback:
            best_score = 1.0
            eval_result = {
                "is_sufficient": True,
                "has_ambiguity": False,
                "confidence": 0.65,
                "ambiguity_type": "none",
                "clarification_hint": "",
                "reasoning": "使用对话记忆生成回复"
            }
            logger.info("  Using conversation memory, skip relevance evaluation")
        else:
            eval_result = self.evaluate_confidence(user_query, chunks)
            logger.info(f"  Evaluation: sufficient={eval_result['is_sufficient']}, ambiguity={eval_result['has_ambiguity']} ({eval_result['ambiguity_type']}), score: {best_score:.3f}")

        # 5. 判断检索质量
        if memory_fallback:
            retrieval_is_good = True
            retrieval_is_excellent = True
        elif has_rerank:
            retrieval_is_good = best_score >= self.rerank_good_threshold
            retrieval_is_excellent = best_score >= self.rerank_excellent_threshold
        else:
            retrieval_is_good = best_score < self.l2_good_threshold
            retrieval_is_excellent = best_score < self.l2_excellent_threshold

        # 6. 决定回答策略
        # 新策略：只要检索到内容，就基于内容回答（可带澄清）
        # - 检索到内容 + 有二义性 → 先答后问 (answer_with_clarification)
        # - 检索到内容 + 信息不完整 → 给出基础答案 + 引导补充信息
        # - 检索到内容 → 直接回答
        # - 业务相关性低 + 检索极差 → 超出范围

        business_relevance = intent_result.get('business_relevance', 'medium')

        # 只有业务相关性低且检索极差时才说超出范围
        if business_relevance == 'low' and not retrieval_is_good:
            logger.info(f"  Query has low business relevance and very poor retrieval, out of scope")
            web_result = self._answer_with_web_search(
                user_query=user_query,
                search_query=optimized_query,
                conversation_history=conversation_history,
                reason="out_of_scope"
            )
            if web_result:
                self.memory_cache.add_exchange(session_id, user_query, web_result['response'])
                return web_result

            return {
                "type": "out_of_scope",
                "response": "抱歉，这个问题似乎超出了我的知识范围。我主要帮助解答产品手册、安装文档、说明书等相关问题。您可以换个产品相关的问题试试？",
                "intent": intent_result,
                "sources": []
            }

        # 决定是否需要澄清提示
        clarification_hint = ""
        response_type = "answer"

        # 情况1：检索质量好 + 有二义性 → 先答后问
        if retrieval_is_good and eval_result['has_ambiguity'] and eval_result['clarification_hint']:
            logger.info(f"  Good retrieval + ambiguity ({eval_result['ambiguity_type']}) → answer with clarification")
            clarification_hint = eval_result['clarification_hint']
            response_type = "answer_with_clarification"

        # 情况2：检索质量一般 + 信息不完整 → 给基础答案并引导补充
        elif retrieval_is_good and not eval_result['is_sufficient'] and eval_result.get('clarification_hint'):
            logger.info(f"  Moderate retrieval + insufficient info → answer with hint for more details")
            clarification_hint = eval_result['clarification_hint']
            response_type = "answer_with_clarification"

        # 情况3：其他情况直接回答
        else:
            logger.info(f"  Retrieval quality: {'excellent' if retrieval_is_excellent else 'good'}, direct answer")

        if retrieval_is_good and (
            not eval_result['is_sufficient']
            or eval_result.get('confidence', 0.0) < self.confidence_threshold
        ):
            web_result = self._answer_with_web_search(
                user_query=user_query,
                search_query=optimized_query,
                conversation_history=conversation_history,
                reason="insufficient_context"
            )
            if web_result:
                self.memory_cache.add_exchange(session_id, user_query, web_result['response'])
                return web_result

        # 7. 生成回复（带或不带澄清提示）
        generation_chunks = chunks
        if not memory_fallback:
            history_chunk = self._build_history_chunk(conversation_history)
            if history_chunk:
                generation_chunks = chunks + [history_chunk]

        response_result = self.generate_response(
            user_query,
            generation_chunks,
            conversation_history,
            clarification_hint=clarification_hint,
            intent_note=intent_note
        )

        self.memory_cache.add_exchange(session_id, user_query, response_result['response'])

        return {
            "type": response_type,
            "response": response_result['response'],
            "confidence": response_result['confidence'],
            "sources": generation_chunks,
            "source_ids": response_result['source_ids'],
            "intent": intent_result,
            "rewrite": rewrite_result,
            "evaluation": eval_result
        }

    async def process_query_stream(
        self,
        user_query: str,
        conversation_history: str = "",
        filters: Dict = None,
        force_answer: bool = False,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户查询，每个步骤完成时立即 yield 结果

        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            filters: 检索过滤条件
            force_answer: 是否强制回答
            session_id: 会话ID，用于记忆缓存

        Yields:
            每个处理步骤的中间结果
        """
        logger.info(f"🔍 [STREAM] Processing query: {user_query[:100]}...")

        # 步骤1: 意图识别
        # 发送 reasoning 消息（符合前端规范）
        yield {"type": "reasoning", "content": "🔍 正在识别意图..."}
        await asyncio.sleep(0)

        # 在线程池中执行同步调用，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        intent_result = await loop.run_in_executor(
            None,
            self.classify_intent,
            user_query,
            conversation_history
        )
        logger.info(f"  Intent: {intent_result['intent']} (confidence: {intent_result['confidence']})")

        # 构建详细的意图识别结果消息
        intent_msg = f"✓ 意图识别完成\n"
        intent_msg += f"  • 意图类型: {intent_result['intent']}\n"
        intent_msg += f"  • 置信度: {intent_result['confidence']:.2f}\n"
        intent_msg += f"  • 业务相关性: {intent_result['business_relevance']}"

        yield {"type": "reasoning", "content": intent_msg}
        await asyncio.sleep(0)

        # 如果是闲聊且置信度高，根据配置决定回复方式
        if intent_result['intent'] == 'chitchat' and intent_result['confidence'] >= self.chat_mode_threshold:
            if self.enable_chat_mode:
                yield {"type": "reasoning", "content": "检测到闲聊意图，正在生成回复..."}
                await asyncio.sleep(0)

                llm_response = self.generate_chitchat_response(user_query, conversation_history)
                llm_response_with_notice = f"{llm_response}\n\n---\n⚠️ **提示**：此回复由 AI 生成，仅供参考，请注意甄别。"
                self.memory_cache.add_exchange(session_id, user_query, llm_response_with_notice)

                yield {"type": "content", "content": llm_response_with_notice}
                await asyncio.sleep(0)

                yield {"type": "done"}
                await asyncio.sleep(0)
                return
            else:
                fixed_reply = "您好！我是文档助手，专门帮助您查找和理解文档内容。有什么我可以帮您的吗？"

                yield {"type": "content", "content": fixed_reply}
                await asyncio.sleep(0)

                yield {"type": "done"}
                await asyncio.sleep(0)

                self.memory_cache.add_exchange(session_id, user_query, fixed_reply)
                return

        # 步骤2: 查询改写
        intent_note = ""
        if intent_result['intent'] == 'chitchat' and intent_result['confidence'] < self.chat_mode_threshold:
            logger.info(f"  闲聊意图置信度不足，继续检索知识库")
            intent_note = "检测到您的问题可能偏向闲聊，但我尝试在知识库中为您查找相关内容"

        yield {"type": "reasoning", "content": "📝 正在优化查询..."}
        await asyncio.sleep(0)

        rewrite_result = await loop.run_in_executor(
            None,
            self.rewrite_query,
            user_query,
            conversation_history
        )
        optimized_query = rewrite_result['rewritten_query']
        logger.info(f"  Rewritten query: {optimized_query[:100]}...")

        # 构建详细的查询改写结果消息
        rewrite_msg = f"✓ 查询优化完成\n"
        rewrite_msg += f"  • 原始查询: {user_query}\n"
        rewrite_msg += f"  • 优化后查询: {optimized_query}\n"
        key_entities = rewrite_result.get('key_entities', [])
        if key_entities:
            rewrite_msg += f"  • 关键实体: {', '.join(key_entities)}\n"
        rewrite_msg += f"  • 检索策略: {rewrite_result.get('search_strategy', 'semantic')}"

        yield {"type": "reasoning", "content": rewrite_msg}
        await asyncio.sleep(0)

        # 步骤3: 向量检索
        yield {"type": "reasoning", "content": "🔎 正在检索相关文档..."}
        await asyncio.sleep(0)

        key_entities = rewrite_result.get('key_entities', [])

        if key_entities and len(key_entities) > 1:
            logger.info(f"  🔍 Multi-entity search with: {key_entities}")
            all_chunks = []
            for entity in key_entities:
                entity_chunks = await loop.run_in_executor(
                    None,
                    self.retrieve_chunks,
                    entity,
                    self.entity_top_k,
                    filters
                )
                all_chunks.extend(entity_chunks)
                logger.info(f"    - Entity '{entity}': found {len(entity_chunks)} chunks")

            chunks = self._deduplicate_chunks(all_chunks)[:self.multi_entity_dedup_limit]
            logger.info(f"  ✅ After deduplication: {len(chunks)} unique chunks")

            yield {"type": "reasoning", "content": "🔄 正在重排序检索结果..."}
            await asyncio.sleep(0)
            chunks = await loop.run_in_executor(
                None,
                self.rerank_chunks,
                user_query,
                chunks,
                self.rerank_top_n
            )
        else:
            logger.info(f"  🔍 Single query search: {optimized_query[:50]}...")
            chunks = await loop.run_in_executor(
                None,
                self.retrieve_chunks,
                optimized_query,
                self.single_query_top_k,
                filters
            )

        memory_fallback = False
        if not chunks:
            memory_chunks = self.memory_cache.get_context_chunks(session_id)
            logger.info(f"  Memory fallback check (no retrieval): {len(memory_chunks)} cached exchanges")
            if memory_chunks:
                memory_fallback = True
                chunks = memory_chunks
                logger.info("  🔁 No retrieval results, falling back to conversation memory")
                yield {"type": "reasoning", "content": "📎 未检索到文档，改用最近的对话记忆继续回答"}
                await asyncio.sleep(0)
            else:
                history_chunk = self._build_history_chunk(conversation_history)
                if history_chunk:
                    memory_fallback = True
                    chunks = [history_chunk]
                    logger.info("  🔁 No retrieval results, falling back to formatted conversation history")
                    yield {"type": "reasoning", "content": "📎 未检索到文档，改用之前的对话内容继续回答"}
                    await asyncio.sleep(0)

        # 构建详细的检索结果消息
        if memory_fallback:
            retrieval_msg = f"✓ 未命中文档片段，使用最近 {len(chunks)} 条对话记忆继续推理"
        else:
            retrieval_msg = f"✓ 检索完成\n  • 找到 {len(chunks)} 个相关片段"
            if chunks:
                # 显示前3个片段的来源和分数
                for i, chunk in enumerate(chunks[:3], 1):
                    score = chunk.get('rerank_score') or chunk.get('score', 0)
                    score_type = 'rerank' if 'rerank_score' in chunk else 'L2'
                    doc_name = chunk.get('document', 'Unknown')[:40]
                    retrieval_msg += f"\n  • [{i}] {doc_name} ({score:.3f})"
                if len(chunks) > 3:
                    retrieval_msg += f"\n  • ... 还有 {len(chunks) - 3} 个片段"

        yield {"type": "reasoning", "content": retrieval_msg}
        await asyncio.sleep(0)

        # 发送检索结果的 canvas 表格可视化
        if chunks and len(chunks) > 0 and not memory_fallback:
            # 构建 Markdown 表格
            table_md = "## 📚 检索结果详情\n\n"
            table_md += "| 排名 | 文档来源 | 相关性 | 内容预览 |\n"
            table_md += "|------|----------|--------|----------|\n"

            for i, chunk in enumerate(chunks[:5], 1):
                score = chunk.get('rerank_score') or chunk.get('score', 0)
                score_type = 'rerank' if 'rerank_score' in chunk else 'L2'
                doc_name = chunk.get('document', 'Unknown')[:30]
                content_preview = chunk.get('content', '')[:60].replace('\n', ' ').replace('|', '\\|')
                table_md += f"| {i} | {doc_name} | {score:.3f} ({score_type}) | {content_preview}... |\n"

            yield {
                "type": "canvas",
                "content": {
                    "canvas-type": "markdown",
                    "canvas-source": table_md
                }
            }
            await asyncio.sleep(0)

        if not chunks:
            # 输出未找到任何内容的详细信息
            no_results_msg = f"❌ 检索结果: 未找到任何内容\n"
            no_results_msg += f"  • 使用的查询: {optimized_query}\n"
            if key_entities:
                no_results_msg += f"  • 尝试的关键词: {', '.join(key_entities)}\n"
            no_results_msg += f"  • 检索策略: {rewrite_result.get('search_strategy', 'semantic')}\n"
            no_results_msg += f"\n💡 可能的原因:\n"
            no_results_msg += f"  • 知识库中没有相关文档\n"
            no_results_msg += f"  • 关键词不匹配\n"
            no_results_msg += f"  • 问题超出文档范围"

            yield {"type": "reasoning", "content": no_results_msg}
            await asyncio.sleep(0)

            yield {"type": "reasoning", "content": "🌐 知识库未命中，检查是否可以使用互联网检索..."}
            await asyncio.sleep(0)
            web_result = await loop.run_in_executor(
                None,
                self._answer_with_web_search,
                user_query,
                optimized_query,
                conversation_history,
                "no_results"
            )
            if web_result:
                self.memory_cache.add_exchange(session_id, user_query, web_result['response'])
                for event in self._build_web_search_stream_events(web_result):
                    yield event
                    await asyncio.sleep(0)
                return

            yield {
                "type": "content",
                "content": "抱歉，我没有找到相关的文档内容。\n\n建议：\n- 尝试使用其他关键词\n- 简化或具体化您的问题\n- 确认问题是否属于文档涵盖的范围"
            }
            await asyncio.sleep(0)

            yield {"type": "done"}
            await asyncio.sleep(0)
            return

        # 步骤4: 评估相似度和置信度
        yield {"type": "reasoning", "content": "⚖️  正在评估检索结果质量..."}
        await asyncio.sleep(0)

        # 计算相似度
        has_rerank = not memory_fallback and any('rerank_score' in c for c in chunks)
        best_score = 1.0

        if not memory_fallback:
            if has_rerank:
                best_score = max(c.get('rerank_score', 0) for c in chunks) if chunks else 0
                score_threshold = self.rerank_score_threshold
                is_relevant = best_score >= score_threshold
                score_type = "rerank"
            else:
                best_score = min(c.get('score', float('inf')) for c in chunks) if chunks else float('inf')
                score_threshold = self.l2_distance_threshold
                is_relevant = best_score <= score_threshold
                score_type = "L2"

            if not is_relevant:
                memory_chunks = self.memory_cache.get_context_chunks(session_id)
                logger.info(f"  Memory fallback check (low relevance): {len(memory_chunks)} cached exchanges")
                if memory_chunks:
                    memory_fallback = True
                    chunks = memory_chunks
                    has_rerank = False
                    original_score = best_score
                    best_score = 1.0
                    logger.info(f"  {score_type} score below threshold ({original_score:.3f}); using conversation memory fallback")
                    yield {"type": "reasoning", "content": "📎 检索相关度偏低，改用最近的对话记忆继续回答"}
                    await asyncio.sleep(0)
                else:
                    history_chunk = self._build_history_chunk(conversation_history)
                    if history_chunk:
                        memory_fallback = True
                        chunks = [history_chunk]
                        has_rerank = False
                        best_score = 1.0
                        logger.info(f"  {score_type} score below threshold ({best_score:.3f}); using conversation history fallback")
                        yield {"type": "reasoning", "content": "📎 检索相关度偏低，改用之前的对话内容继续回答"}
                        await asyncio.sleep(0)
                    else:
                        logger.info(f"  {score_type} score indicates no relevant results: {best_score:.3f}")

                        # 输出相关度过低的详细信息
                        low_relevance_msg = f"⚠️  相关度评估: 未达到阈值\n"
                        low_relevance_msg += f"  • 找到 {len(chunks)} 个片段，但相关度过低\n"
                        low_relevance_msg += f"  • 最佳相关度分数: {best_score:.3f} ({score_type})\n"
                        low_relevance_msg += f"  • 阈值要求: {score_threshold:.3f}\n"
                        low_relevance_msg += f"  • 最相关的文档:\n"

                        for i, chunk in enumerate(chunks[:3], 1):
                            score = chunk.get('rerank_score') or chunk.get('score', 0)
                            doc_name = chunk.get('document', 'Unknown')[:40]
                            low_relevance_msg += f"    [{i}] {doc_name} (分数: {score:.3f})\n"

                        low_relevance_msg += f"\n💡 建议：请尝试换个方式提问，或提供更具体的关键词"

                        yield {"type": "reasoning", "content": low_relevance_msg}
                        await asyncio.sleep(0)

                        yield {"type": "reasoning", "content": "🌐 知识库相关度不足，检查是否可以使用互联网检索..."}
                        await asyncio.sleep(0)
                        web_result = await loop.run_in_executor(
                            None,
                            self._answer_with_web_search,
                            user_query,
                            optimized_query,
                            conversation_history,
                            "low_relevance"
                        )
                        if web_result:
                            self.memory_cache.add_exchange(session_id, user_query, web_result['response'])
                            for event in self._build_web_search_stream_events(web_result):
                                yield event
                                await asyncio.sleep(0)
                            return

                        # 发送最终回复
                        yield {
                            "type": "content",
                            "content": "抱歉，我的知识库中虽然找到了一些文档片段，但它们与您的问题相关度过低，无法给出可靠的回答。\n\n建议：\n- 尝试使用不同的关键词重新提问\n- 提供更具体的上下文信息\n- 确认问题是否在文档涵盖范围内"
                        }
                        await asyncio.sleep(0)

                        yield {"type": "done"}
                        await asyncio.sleep(0)
                        return

        if memory_fallback:
            eval_result = {
                "is_sufficient": True,
                "has_ambiguity": False,
                "confidence": 0.65,
                "ambiguity_type": "none",
                "clarification_hint": "",
                "reasoning": "使用对话记忆生成回复"
            }
            logger.info("  Using conversation memory, skip confidence evaluation")
        else:
            eval_result = await loop.run_in_executor(
                None,
                self.evaluate_confidence,
                user_query,
                chunks
            )
            logger.info(f"  Evaluation: sufficient={eval_result['is_sufficient']}, ambiguity={eval_result['has_ambiguity']}")

        # 构建详细的评估结果消息
        eval_msg = f"✓ 评估完成\n"
        eval_msg += f"  • 置信度: {eval_result['confidence']:.2f}\n"
        eval_msg += f"  • 信息充分性: {'充分' if eval_result['is_sufficient'] else '不充分'}\n"
        eval_msg += f"  • 是否有歧义: {'是' if eval_result['has_ambiguity'] else '否'}"
        if eval_result.get('ambiguity_type') and eval_result['ambiguity_type'] != 'none':
            eval_msg += f"\n  • 歧义类型: {eval_result['ambiguity_type']}"

        yield {"type": "reasoning", "content": eval_msg}
        await asyncio.sleep(0)

        # 步骤5: 判断回答策略
        if memory_fallback:
            retrieval_is_good = True
            retrieval_is_excellent = True
        elif has_rerank:
            retrieval_is_good = best_score >= self.rerank_good_threshold
            retrieval_is_excellent = best_score >= self.rerank_excellent_threshold
        else:
            retrieval_is_good = best_score < self.l2_good_threshold
            retrieval_is_excellent = best_score < self.l2_excellent_threshold

        business_relevance = intent_result.get('business_relevance', 'medium')

        if business_relevance == 'low' and not retrieval_is_good:
            yield {"type": "reasoning", "content": "🌐 问题不在知识库范围内，检查是否可以使用互联网检索..."}
            await asyncio.sleep(0)
            web_result = await loop.run_in_executor(
                None,
                self._answer_with_web_search,
                user_query,
                optimized_query,
                conversation_history,
                "out_of_scope"
            )
            if web_result:
                self.memory_cache.add_exchange(session_id, user_query, web_result['response'])
                for event in self._build_web_search_stream_events(web_result):
                    yield event
                    await asyncio.sleep(0)
                return

            yield {
                "type": "content",
                "content": "抱歉，这个问题似乎超出了我的知识范围。我主要帮助解答产品手册、安装文档、说明书等相关问题。您可以换个产品相关的问题试试？"
            }
            await asyncio.sleep(0)

            yield {"type": "done"}
            await asyncio.sleep(0)
            return

        clarification_hint = ""
        response_type = "answer"

        if retrieval_is_good and eval_result['has_ambiguity'] and eval_result['clarification_hint']:
            clarification_hint = eval_result['clarification_hint']
            response_type = "answer_with_clarification"
        elif retrieval_is_good and not eval_result['is_sufficient'] and eval_result.get('clarification_hint'):
            clarification_hint = eval_result['clarification_hint']
            response_type = "answer_with_clarification"

        if retrieval_is_good and (
            not eval_result['is_sufficient']
            or eval_result.get('confidence', 0.0) < self.confidence_threshold
        ):
            yield {"type": "reasoning", "content": "🌐 知识库依据不充分，检查是否可以使用互联网检索补充..."}
            await asyncio.sleep(0)
            web_result = await loop.run_in_executor(
                None,
                self._answer_with_web_search,
                user_query,
                optimized_query,
                conversation_history,
                "insufficient_context"
            )
            if web_result:
                self.memory_cache.add_exchange(session_id, user_query, web_result['response'])
                for event in self._build_web_search_stream_events(web_result):
                    yield event
                    await asyncio.sleep(0)
                return

        # 步骤6: 生成最终回复
        yield {"type": "reasoning", "content": "💡 正在生成回复..."}
        await asyncio.sleep(0)

        generation_chunks = chunks
        if not memory_fallback:
            history_chunk = self._build_history_chunk(conversation_history)
            if history_chunk:
                generation_chunks = chunks + [history_chunk]

        response_result = await loop.run_in_executor(
            None,
            self.generate_response,
            user_query,
            generation_chunks,
            conversation_history,
            clarification_hint,
            intent_note
        )

        self.memory_cache.add_exchange(session_id, user_query, response_result['response'])

        # 发送最终回复内容
        yield {"type": "content", "content": response_result['response']}
        await asyncio.sleep(0)

        # 发送文件源信息
        if not memory_fallback:
            for chunk in generation_chunks[:self.files_display_limit]:
                payload = self._build_file_source_payload(chunk)
                if payload:
                    yield {"type": "files", "content": payload}
                    await asyncio.sleep(0)
