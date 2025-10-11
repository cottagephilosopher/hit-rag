"""
Milvus 向量存储
基于 LangChain 的 Milvus 集成
"""

import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional
from langchain_milvus import Milvus
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RAGVectorStore:
    """
    RAG 向量存储封装
    使用 LangChain 的 Milvus 集成
    """

    def __init__(self, embedding_service):
        """
        初始化 Milvus 向量存储

        Args:
            embedding_service: EmbeddingService 实例
        """
        # 从环境变量读取 Milvus 配置
        self.milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
        self.milvus_port = os.getenv("MILVUS_PORT", "19530")
        milvus_uri = f"http://{self.milvus_host}:{self.milvus_port}"

        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME", "knowledges")
        
        logger.info(f"Initializing Milvus vector store: {milvus_uri}")
        logger.info(f"Collection name: {self.collection_name}")

        try:
            # 使用 LangChain Milvus 集成
            self.vector_store = Milvus(
                embedding_function=embedding_service.embeddings,
                connection_args={"uri": milvus_uri},
                collection_name=self.collection_name,
                drop_old=False,  # 不删除已存在的 collection
                auto_id=False,  # 使用内容 hash 作为确定性 ID，实现自动去重
            )

            logger.info("✅ Milvus vector store initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Milvus vector store: {e}")
            raise

    @staticmethod
    def _compute_content_hash(content: str, user_tag: str = '') -> str:
        """
        计算内容的 hash 值作为确定性 ID

        使用 SHA256 生成 64 位十六进制字符串
        相同的内容（包括 user_tag）会产生相同的 ID，实现自动去重

        Args:
            content: 原始文本内容
            user_tag: 用户标签（影响向量化）

        Returns:
            64 位十六进制字符串
        """
        # 组合 user_tag 和 content，因为它们共同决定向量化的文本
        combined = f"{user_tag}||{content}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def add_chunks(self, chunks: List[Dict[str, Any]], document_tags: List[str] = None) -> List[str]:
        """
        批量添加 chunks 到向量库

        使用内容 hash 作为 ID，相同内容会自动覆盖（去重）

        Args:
            chunks: chunk 字典列表，每个 chunk 包含:
                - id: chunk 数据库 ID
                - content 或 edited_content: 文本内容
                - document_id, chunk_id: 文档和chunk编号
                - source_file: 源文件路径
                - user_tag, content_tags: 标签
                - is_atomic, atomic_type: ATOMIC 属性
                - token_count: token 数量
            document_tags: 文档级别的标签列表

        Returns:
            Milvus 向量 ID 列表（基于内容 hash）

        Raises:
            ValueError: 如果 chunks 为空或格式不正确
            Exception: 向量化或插入失败
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        logger.info(f"Adding {len(chunks)} chunks to Milvus (with deduplication)...")

        documents = []
        content_hashes = []
        for chunk in chunks:
            try:
                # 优先使用 edited_content，如果不存在则使用 content
                content = chunk.get('edited_content') or chunk.get('content', '')
                if not content:
                    logger.warning(f"Chunk {chunk.get('id')} has no content, skipping")
                    continue

                # 获取 user_tag（用于 hash 计算和向量化）
                user_tag = chunk.get('user_tag', '') or ''

                # 计算内容 hash 作为确定性 ID
                content_hash = self._compute_content_hash(content, user_tag)
                content_hashes.append(content_hash)

                # 构建元数据
                # 注意: Milvus 要求所有字段都必须有值，None 会导致数据不一致错误

                # 合并标签：document_tags + content_tags
                content_tags = chunk.get('content_tags', []) or []
                doc_tags = document_tags or []
                # 合并并去重（保持顺序：document_tags 在前，这样共性标签优先显示）
                merged_tags = list(dict.fromkeys(doc_tags + content_tags))

                metadata = {
                    'chunk_db_id': chunk['id'],
                    'document_id': chunk.get('document_id', 0),
                    'chunk_id': chunk.get('chunk_id', 0),
                    'source_file': chunk.get('source_file', '') or 'unknown',
                    'user_tag': user_tag or 'none',
                    'content_tags': json.dumps(merged_tags, ensure_ascii=False),  # 存储合并后的标签
                    'is_atomic': bool(chunk.get('is_atomic', False)),
                    'atomic_type': chunk.get('atomic_type') or 'none',  # 重要: 不能为空字符串或 None
                    'token_count': chunk.get('token_count', 0),
                    'vectorized_at': int(time.time()),
                    'original_content': content,
                    'content_hash': content_hash,  # 保存 hash 值用于追踪
                }

                # 创建向量化文本：user_tag + content
                # user_tag 作为标题，应该有更高的权重，所以重复3次
                if user_tag and user_tag != 'none':
                    # 将标题重复3次以提高其在向量中的权重
                    vectorize_text = f"{user_tag}\n{user_tag}\n{user_tag}\n\n{content}"
                else:
                    vectorize_text = content

                # 创建 LangChain Document
                doc = Document(
                    page_content=vectorize_text,
                    metadata=metadata
                )

                documents.append(doc)

            except Exception as e:
                logger.error(f"Failed to prepare chunk {chunk.get('id')}: {e}")
                continue

        if not documents:
            raise ValueError("No valid documents to add")

        try:
            # 使用 LangChain 的 add_documents 方法，传入确定性 ID
            # 相同的 ID 会自动覆盖，实现去重
            ids = self.vector_store.add_documents(documents, ids=content_hashes)

            logger.info(f"✅ Successfully added {len(ids)} chunks to Milvus (duplicates auto-replaced)")
            return ids

        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to add documents to Milvus: {e}")
            logger.error(traceback.format_exc())
            raise

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        语义搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 元数据过滤条件，例如:
                {'document_id': 1, 'is_atomic': True}

        Returns:
            Document 列表，包含 page_content 和 metadata
        """
        if not query:
            raise ValueError("Query cannot be empty")

        logger.info(f"Searching for: {query[:50]}... (k={k})")

        try:
            # 构建 Milvus 过滤表达式
            expr = self._build_filter_expr(filters) if filters else None

            # 使用 LangChain 的 similarity_search
            results = self.vector_store.similarity_search(
                query,
                k=k,
                expr=expr
            )

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_with_score(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        语义搜索（带相似度分数）

        Returns:
            [(Document, score), ...] 列表
        """
        try:
            expr = self._build_filter_expr(filters) if filters else None

            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                expr=expr
            )

            logger.info(f"Found {len(results)} results with scores")
            return results

        except Exception as e:
            logger.error(f"Search with score failed: {e}")
            raise

    def delete_by_ids(self, milvus_ids: List[str]):
        """
        删除指定的向量（按 Milvus pk）

        注意：不推荐使用此方法，建议使用 delete_by_chunk_db_ids，
        因为 chunk 可能有多个历史向量

        Args:
            milvus_ids: Milvus 向量 ID 列表（pk 值）
        """
        logger.warning("delete_by_ids is not recommended, consider using delete_by_chunk_db_ids")
        if not milvus_ids:
            logger.warning("No IDs provided for deletion")
            return

        logger.info(f"Deleting {len(milvus_ids)} vectors from Milvus...")
        logger.info(f"IDs to delete: {milvus_ids}")

        try:
            # 构建删除表达式：pk in [id1, id2, ...]
            # 注意：Milvus 的 pk 字段类型是 int64，不能带引号
            ids_str = ", ".join(str(id) for id in milvus_ids)
            expr = f"pk in [{ids_str}]"

            logger.info(f"Delete expression: {expr}")

            # 使用 LangChain 的 delete 方法（传递 expr 参数）
            delete_result = self.vector_store.delete(expr=expr)
            logger.info(f"Delete result: {delete_result}")

            if not delete_result:
                logger.warning("Delete returned False, but this may not indicate failure")

            # 等待一小段时间确保删除完全生效
            time.sleep(0.3)

            logger.info("✅ Vectors deleted successfully")

        except Exception as e:
            logger.error(f"❌ Failed to delete vectors: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def delete_by_chunk_db_ids(self, chunk_db_ids: List[int]):
        """
        根据 chunk_db_id 删除所有相关向量（推荐使用）
        这样可以删除同一个 chunk 的所有历史向量

        Args:
            chunk_db_ids: Chunk 数据库 ID 列表
        """
        if not chunk_db_ids:
            logger.warning("No chunk_db_ids provided for deletion")
            return

        logger.info(f"Deleting vectors for {len(chunk_db_ids)} chunks from Milvus...")
        logger.info(f"Chunk DB IDs to delete: {chunk_db_ids}")

        try:
            # 构建删除表达式：chunk_db_id in [id1, id2, ...]
            ids_str = ", ".join(str(id) for id in chunk_db_ids)
            expr = f"chunk_db_id in [{ids_str}]"

            logger.info(f"Delete expression: {expr}")

            # 使用 LangChain 的 delete 方法（传递 expr 参数）
            # 注意：不能使用 ids 参数，因为它会把 ID 当作字符串处理，导致类型错误
            delete_result = self.vector_store.delete(expr=expr)
            logger.info(f"Delete result: {delete_result}")

            if not delete_result:
                logger.warning("Delete returned False, but this may not indicate failure")

            # 等待一小段时间确保删除完全生效
            time.sleep(0.3)

            logger.info("✅ Vectors deleted successfully")

        except Exception as e:
            logger.error(f"❌ Failed to delete vectors: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _build_filter_expr(self, filters: Dict[str, Any]) -> str:
        """
        构建 Milvus 过滤表达式

        Args:
            filters: 过滤条件字典
                支持格式:
                - {'is_atomic': False}  # 布尔值
                - {'document_id': 1}    # 数字
                - {'source_file': 'xx.md'}  # 字符串
                - {'content_tags': ['tag1', 'tag2']}  # 数组（任意匹配）

        Returns:
            Milvus expr 字符串
        """
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if key == 'content_tags' and isinstance(value, list):
                # 标签数组过滤：content_tags 包含任意一个指定标签
                # content_tags 存储为 JSON 字符串，需要用 array_contains_any
                # 但 Milvus 不支持直接对 JSON 字符串数组操作
                # 改为使用 LIKE 匹配（不够精确但可用）
                tag_conditions = []
                for tag in value:
                    # 匹配 JSON 数组中的标签
                    tag_conditions.append(f'content_tags like "%\\"{tag}\\"%"')
                if tag_conditions:
                    conditions.append(f"({' or '.join(tag_conditions)})")
            elif isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
            elif isinstance(value, bool):
                conditions.append(f'{key} == {str(value).lower()}')
            elif isinstance(value, (int, float)):
                conditions.append(f'{key} == {value}')
            else:
                logger.warning(f"Unsupported filter type for {key}: {type(value)}")

        expr = " and ".join(conditions) if conditions else None
        logger.debug(f"Filter expression: {expr}")

        return expr


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    from .embedding_service import EmbeddingService

    try:
        # 初始化服务
        embedding_service = EmbeddingService()
        vector_store = RAGVectorStore(embedding_service)

        # 测试添加
        test_chunks = [
            {
                'id': 1,
                'content': '这是测试内容1',
                'document_id': 1,
                'chunk_id': 1,
                'source_file': 'test.md',
                'user_tag': '测试',
                'content_tags': ['测试', '样例'],
                'is_atomic': False,
                'atomic_type': None,
                'token_count': 10
            },
            {
                'id': 2,
                'content': '这是测试内容2，包含产品参数信息',
                'document_id': 1,
                'chunk_id': 2,
                'source_file': 'test.md',
                'user_tag': '产品',
                'content_tags': ['产品', '参数'],
                'is_atomic': True,
                'atomic_type': 'table',
                'token_count': 15
            }
        ]

        ids = vector_store.add_chunks(test_chunks, document_tags=['测试文档'])
        print(f"\n✅ Added chunks with IDs: {ids}")

        # 测试搜索
        results = vector_store.search("产品参数", k=2)
        print(f"\n✅ Search results: {len(results)}")
        for i, doc in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Content: {doc.page_content[:50]}...")
            print(f"    Metadata: {doc.metadata}")

        print("\n✅ Vector store test passed!")

    except Exception as e:
        print(f"\n❌ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
