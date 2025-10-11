"""
DSPy RAG Pipeline 实现
整合意图识别、查询改写、检索、评估和回复生成
"""

import dspy
import json
import os
import requests
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple

from .dspy_signatures import (
    IntentClassification,
    QueryRewrite,
    ConfidenceEvaluation,
    ClarificationGeneration,
    ResponseGeneration
)


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

        # 读取闲聊模式配置
        self.enable_chat_mode = os.getenv('ENABLE_CHAT_MODE', 'false').lower() == 'true'
        self.chat_mode_threshold = float(os.getenv('CHAT_MODE_THRESHOLD', '0.7'))

        # 配置 DSPy LLM
        self._configure_dspy(llm_model, temperature)

        # 初始化各个模块
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.query_rewriter = dspy.ChainOfThought(QueryRewrite)
        self.confidence_evaluator = dspy.ChainOfThought(ConfidenceEvaluation)
        self.clarification_generator = dspy.ChainOfThought(ClarificationGeneration)
        self.response_generator = dspy.ChainOfThought(ResponseGeneration)

        logger.info(f"✅ DSPy RAG Pipeline initialized with model: {llm_model}")
        logger.info(f"   闲聊模式: {'开启' if self.enable_chat_mode else '关闭'} (置信度阈值: {self.chat_mode_threshold})")

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
                "confidence": float(result.confidence),
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

        # 如果是闲聊且置信度高，根据配置决定回复方式
        if intent_result['intent'] == 'chitchat' and intent_result['confidence'] >= self.chat_mode_threshold:
            # 检查是否开启闲聊模式
            if self.enable_chat_mode:
                logger.info(f"  闲聊模式已开启，置信度 {intent_result['confidence']:.2f} >= {self.chat_mode_threshold}，使用 LLM 回复")
                llm_response = self.generate_chitchat_response(user_query, conversation_history)
                # 在回复末尾添加明显的 AI 生成提醒
                llm_response_with_notice = f"{llm_response}\n\n---\n⚠️ **提示**：此回复由 AI 生成，仅供参考，请注意甄别。"
                return {
                    "type": "chitchat",
                    "response": llm_response_with_notice,
                    "intent": intent_result,
                    "chat_mode": "llm"
                }
            else:
                # 未开启闲聊模式，使用固定回复
                logger.info(f"  闲聊模式未开启，使用固定回复")
                return {
                    "type": "chitchat",
                    "response": "您好！我是文档助手，专门帮助您查找和理解文档内容。有什么我可以帮您的吗？",
                    "intent": intent_result,
                    "chat_mode": "fixed"
                }

        # 置信度不足或非闲聊意图，继续进行知识库检索
        intent_note = ""
        if intent_result['intent'] == 'chitchat' and intent_result['confidence'] < self.chat_mode_threshold:
            logger.info(f"  闲聊意图置信度不足 ({intent_result['confidence']:.2f} < {self.chat_mode_threshold})，继续检索知识库")
            intent_note = "检测到您的问题可能偏向闲聊，但我尝试在知识库中为您查找相关内容"

        # 2. 查询改写
        rewrite_result = self.rewrite_query(user_query, conversation_history)
        logger.info(f"  --- Rewritten query: {rewrite_result}")
        optimized_query = rewrite_result['rewritten_query']
        logger.info(f"  Rewritten query: {optimized_query[:100]}...")

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

        if not chunks:
            return {
                "type": "no_results",
                "response": "抱歉，我没有找到相关的文档内容。您可以换个方式提问吗？",
                "intent": intent_result,
                "rewrite": rewrite_result,
                "sources": []
            }

        # 3.5. 检查相似度分数，过滤掉不相关的检索结果
        # 优先使用重排分数（rerank_score），否则使用 L2 距离（score）
        # - rerank_score: 0.3+ 相关，0.5+ 很相关
        # - L2距离: <0.8 很相关, <1.0 相关, <1.2 可能相关, >1.2 不相关

        # 计算最佳相似度指标
        has_rerank = any('rerank_score' in c for c in chunks)

        if has_rerank:
            # 使用重排分数（越大越好）
            best_score = max(c.get('rerank_score', 0) for c in chunks) if chunks else 0
            score_threshold = 0.2  # rerank_score 低于 0.2 认为不相关
            is_relevant = best_score >= score_threshold
            score_type = "rerank"
        else:
            # 使用 L2 距离（越小越好）
            best_score = min(c.get('score', float('inf')) for c in chunks) if chunks else float('inf')
            score_threshold = 1.2  # L2 距离高于 1.2 认为不相关
            is_relevant = best_score <= score_threshold
            score_type = "L2"

        if not is_relevant:
            logger.info(f"  {score_type} score indicates no relevant results: {best_score:.3f}")
            return {
                "type": "no_results",
                "response": "抱歉，我的知识库中没有找到相关的文档内容。我只能回答与已有文档相关的问题。",
                "intent": intent_result,
                "rewrite": rewrite_result,
                "sources": []
            }

        # 4. 评估置信度和二义性
        eval_result = self.evaluate_confidence(user_query, chunks)
        logger.info(f"  Evaluation: sufficient={eval_result['is_sufficient']}, ambiguity={eval_result['has_ambiguity']} ({eval_result['ambiguity_type']}), score: {best_score:.3f}")

        # 5. 判断检索质量
        if has_rerank:
            # rerank_score: 0.2+ 可用，0.3+ 良好，0.5+ 优秀
            retrieval_is_good = best_score >= 0.2
            retrieval_is_excellent = best_score >= 0.3
        else:
            # L2 距离: <1.2 可用，<1.0 良好，<0.8 优秀
            retrieval_is_good = best_score < 1.2
            retrieval_is_excellent = best_score < 1.0

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

        # 7. 生成回复（带或不带澄清提示）
        response_result = self.generate_response(
            user_query,
            chunks,
            conversation_history,
            clarification_hint=clarification_hint,
            intent_note=intent_note
        )

        return {
            "type": response_type,
            "response": response_result['response'],
            "confidence": response_result['confidence'],
            "sources": chunks,
            "source_ids": response_result['source_ids'],
            "intent": intent_result,
            "rewrite": rewrite_result,
            "evaluation": eval_result
        }
