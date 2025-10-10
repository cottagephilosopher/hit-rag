"""
DSPy RAG Pipeline 实现
整合意图识别、查询改写、检索、评估和回复生成
"""

import dspy
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from .dspy_signatures import (
    IntentClassification,
    QueryRewrite,
    ConfidenceEvaluation,
    ClarificationGeneration,
    ResponseGeneration
)

logger = logging.getLogger(__name__)


class DSPyRAGPipeline:
    """DSPy RAG Pipeline 主类"""

    def __init__(
        self,
        vector_store=None,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        confidence_threshold: float = 0.5
    ):
        """
        初始化 DSPy Pipeline

        Args:
            vector_store: 向量存储实例（Milvus）
            llm_model: LLM 模型名称
            temperature: 生成温度
            confidence_threshold: 置信度阈值
        """
        self.vector_store = vector_store
        self.confidence_threshold = confidence_threshold

        # 配置 DSPy LLM
        self._configure_dspy(llm_model, temperature)

        # 初始化各个模块
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.query_rewriter = dspy.ChainOfThought(QueryRewrite)
        self.confidence_evaluator = dspy.ChainOfThought(ConfidenceEvaluation)
        self.clarification_generator = dspy.ChainOfThought(ClarificationGeneration)
        self.response_generator = dspy.ChainOfThought(ResponseGeneration)

        logger.info(f"✅ DSPy RAG Pipeline initialized with model: {llm_model}")

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
            else:
                api_base = os.getenv('API_BASE')
                api_key = os.getenv('API_KEY')
                model = os.getenv('MODEL_NAME', 'gpt-4o')
                # 标准 OpenAI 配置
                lm = dspy.LM(model=model, api_base=api_base, api_key=api_key, temperature=temperature, max_tokens=2000)
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
                "confidence": float(result.confidence),
                "reasoning": result.reasoning
            }
        except Exception as e:
            logger.error(f"❌ Intent classification failed: {e}")
            # 默认认为是问答
            return {
                "intent": "question",
                "confidence": 0.5,
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
                    "chunk_id": metadata.get("chunk_db_id") or metadata.get("pk"),
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
                 for c in retrieved_chunks[:3]],  # 只取前3个避免太长
                ensure_ascii=False,
                indent=2
            )

            result = self.confidence_evaluator(
                user_query=user_query,
                retrieved_chunks=chunks_text
            )

            # 解析 missing_info
            try:
                missing_info = json.loads(result.missing_info)
            except:
                missing_info = [result.missing_info] if result.missing_info else []

            return {
                "is_sufficient": result.is_sufficient.lower() == "yes",
                "confidence": float(result.confidence),
                "missing_info": missing_info,
                "reasoning": result.reasoning
            }
        except Exception as e:
            logger.error(f"❌ Confidence evaluation failed: {e}")
            # 如果评估失败，保守地认为信息充分（避免过度反问）
            return {
                "is_sufficient": True,
                "confidence": 0.5,
                "missing_info": [],
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

    def generate_response(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """
        步骤6: 生成最终回复

        Args:
            user_query: 用户查询
            retrieved_chunks: 检索到的文档片段
            conversation_history: 对话历史

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
                retrieved_chunks=chunks_text
            )

            # 解析 source_ids
            try:
                source_ids = json.loads(result.source_ids)
            except:
                source_ids = [c.get("chunk_id") for c in retrieved_chunks[:3]]

            return {
                "response": result.response,
                "source_ids": source_ids,
                "confidence": float(result.confidence),
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
        force_answer: bool = False
    ) -> Dict[str, Any]:
        """
        完整处理用户查询（主入口）

        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            filters: 检索过滤条件
            force_answer: 是否强制回答（不生成澄清问题）

        Returns:
            处理结果
        """
        logger.info(f"🔍 Processing query: {user_query[:100]}...")

        # 1. 意图识别
        intent_result = self.classify_intent(user_query, conversation_history)
        logger.info(f"  Intent: {intent_result['intent']} (confidence: {intent_result['confidence']})")

        # 如果是闲聊，直接简单回复
        if intent_result['intent'] == 'chitchat':
            return {
                "type": "chitchat",
                "response": "您好！我是文档助手，专门帮助您查找和理解文档内容。有什么我可以帮您的吗？",
                "intent": intent_result
            }

        # 2. 查询改写
        rewrite_result = self.rewrite_query(user_query, conversation_history)
        optimized_query = rewrite_result['rewritten_query']
        logger.info(f"  Rewritten query: {optimized_query[:100]}...")

        # 3. 向量检索
        chunks = self.retrieve_chunks(optimized_query, top_k=5, filters=filters)

        if not chunks:
            return {
                "type": "no_results",
                "response": "抱歉，我没有找到相关的文档内容。您可以换个方式提问吗？",
                "intent": intent_result,
                "rewrite": rewrite_result,
                "sources": []
            }

        # 3.5. 检查相似度分数，过滤掉不相关的检索结果
        # Milvus 使用 L2 距离：值越小越相似（0=完全相同）
        # - 相关文档通常 < 1.0
        # - 不相关文档通常 > 1.2
        min_score = min(c.get('score', float('inf')) for c in chunks) if chunks else float('inf')

        # 设置阈值为 1.2，超过此值认为文档库中没有相关内容
        if min_score > 1.2:
            logger.info(f"  Min L2 distance too high: {min_score:.3f}, treating as no relevant results")
            return {
                "type": "no_results",
                "response": "抱歉，我的知识库中没有找到相关的文档内容。我只能回答与已有文档相关的问题。",
                "intent": intent_result,
                "rewrite": rewrite_result,
                "sources": []
            }

        # 4. 评估置信度
        eval_result = self.evaluate_confidence(user_query, chunks)
        logger.info(f"  Confidence: {eval_result['confidence']} (sufficient: {eval_result['is_sufficient']}), Min L2 distance: {min_score:.3f}")

        # 5. 判断是否需要澄清
        # 对于非常短的查询且检索分数较好（<1.1），优先直接回答而不澄清
        # L2距离: <0.8=很相关, <1.0=相关, <1.2=可能相关, >1.2=不相关

        # 判断查询是否简短：中文<=5字 或 英文<=3个单词
        query_text = user_query.strip()
        chinese_char_count = sum(1 for c in query_text if '\u4e00' <= c <= '\u9fff')
        word_count = len(query_text.split())

        query_is_short = (
            chinese_char_count > 0 and chinese_char_count <= 5  # 中文短查询
        ) or (
            chinese_char_count == 0 and word_count <= 3  # 英文短查询
        )

        retrieval_is_good = min_score < 1.1  # 放宽阈值以包括1.049这样的情况

        should_skip_clarification = (
            force_answer or
            (query_is_short and retrieval_is_good) or
            eval_result['is_sufficient']
        )

        if not should_skip_clarification and eval_result['missing_info']:
            clarification = self.generate_clarification(
                user_query,
                eval_result['missing_info'],
                conversation_history
            )
            logger.info(f"  Generating clarification (query_length={len(user_query)}, min_score={min_score:.3f})")
            return {
                "type": "clarification",
                "question": clarification['question'],
                "options": clarification['options'],
                "intent": intent_result,
                "rewrite": rewrite_result,
                "evaluation": eval_result,
                "sources": chunks[:3]
            }

        # 如果跳过澄清，记录原因
        if query_is_short and retrieval_is_good and not eval_result['is_sufficient']:
            logger.info(f"  Skipping clarification due to short query with good retrieval (query_length={len(user_query)}, min_score={min_score:.3f})")

        # 6. 生成回复
        response_result = self.generate_response(user_query, chunks, conversation_history)

        return {
            "type": "answer",
            "response": response_result['response'],
            "confidence": response_result['confidence'],
            "sources": chunks,
            "source_ids": response_result['source_ids'],
            "intent": intent_result,
            "rewrite": rewrite_result,
            "evaluation": eval_result
        }
