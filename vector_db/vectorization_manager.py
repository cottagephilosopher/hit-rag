"""
向量化管理器
处理 chunks 的向量化业务逻辑，包括状态管理和日志记录
"""

import logging
from typing import List, Dict, Any, Callable, Optional
from .embedding_service import EmbeddingService
from .vector_store import RAGVectorStore

logger = logging.getLogger(__name__)


class VectorizationManager:
    """
    向量化管理器
    负责协调 embedding 和 vector store，管理向量化流程
    """

    def __init__(self):
        """初始化向量化管理器"""
        logger.info("Initializing VectorizationManager...")

        try:
            self.embedding_service = EmbeddingService()
            self.vector_store = RAGVectorStore(self.embedding_service)
            logger.info("✅ VectorizationManager initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize VectorizationManager: {e}")
            raise

    def vectorize_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        批量向量化 chunks

        Args:
            chunks: chunk 字典列表
            document_tags: 文档级别的标签
            progress_callback: 进度回调函数 (current, total)

        Returns:
            结果字典: {
                'success': [{'chunk_id': id, 'milvus_id': id}, ...],
                'failed': [{'chunk_id': id, 'reason': str}, ...],
                'skipped': [{'chunk_id': id, 'reason': str}, ...],
            }
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        logger.info(f"Starting vectorization for {len(chunks)} chunks...")

        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }

        # 过滤出可以向量化的 chunks
        valid_chunks = []
        for chunk in chunks:
            chunk_id = chunk.get('id')
            status = chunk.get('status')

            # 检查状态
            if status == -1:
                results['skipped'].append({
                    'chunk_id': chunk_id,
                    'reason': '已废弃 (status=-1)'
                })
                continue

            if status == 2:
                results['skipped'].append({
                    'chunk_id': chunk_id,
                    'reason': '已向量化 (status=2)'
                })
                continue

            valid_chunks.append(chunk)

        if not valid_chunks:
            logger.warning("No valid chunks to vectorize")
            return results

        logger.info(f"Vectorizing {len(valid_chunks)} valid chunks...")

        try:
            # 批量向量化
            milvus_ids = self.vector_store.add_chunks(valid_chunks, document_tags)

            # 记录成功
            for i, (chunk, milvus_id) in enumerate(zip(valid_chunks, milvus_ids)):
                results['success'].append({
                    'chunk_id': chunk['id'],
                    'milvus_id': milvus_id
                })

                # 调用进度回调
                if progress_callback:
                    try:
                        progress_callback(i + 1, len(valid_chunks))
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")

            logger.info(f"✅ Successfully vectorized {len(results['success'])} chunks")

        except Exception as e:
            logger.error(f"❌ Batch vectorization failed: {e}")
            # 将所有 valid_chunks 标记为失败
            for chunk in valid_chunks:
                results['failed'].append({
                    'chunk_id': chunk['id'],
                    'reason': str(e)
                })

        return results

    def vectorize_single_chunk(
        self,
        chunk: Dict[str, Any],
        document_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        向量化单个 chunk

        Args:
            chunk: chunk 字典
            document_tags: 文档级别的标签

        Returns:
            结果字典: {'success': bool, 'milvus_id': str or None, 'reason': str or None}
        """
        chunk_id = chunk.get('id')
        logger.info(f"Vectorizing single chunk: {chunk_id}")

        # 检查状态
        if chunk.get('status') == -1:
            return {
                'success': False,
                'milvus_id': None,
                'reason': '已废弃 (status=-1)'
            }

        if chunk.get('status') == 2:
            return {
                'success': False,
                'milvus_id': None,
                'reason': '已向量化 (status=2)'
            }

        try:
            # 向量化
            milvus_ids = self.vector_store.add_chunks([chunk], document_tags)

            logger.info(f"✅ Chunk {chunk_id} vectorized successfully: {milvus_ids[0]}")

            return {
                'success': True,
                'milvus_id': milvus_ids[0],
                'reason': None
            }

        except Exception as e:
            logger.error(f"❌ Failed to vectorize chunk {chunk_id}: {e}")
            return {
                'success': False,
                'milvus_id': None,
                'reason': str(e)
            }

    def search_chunks(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        with_score: bool = False
    ) -> List[Dict[str, Any]]:
        """
        语义搜索 chunks

        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 元数据过滤
            with_score: 是否返回相似度分数

        Returns:
            结果列表: [{
                'content': str,
                'metadata': dict,
                'score': float (if with_score=True)
            }, ...]
        """
        logger.info(f"Searching chunks: {query[:50]}...")

        try:
            if with_score:
                results = self.vector_store.search_with_score(query, k, filters)
                return [
                    {
                        'content': doc.metadata.get('original_content', doc.page_content),
                        'metadata': doc.metadata,
                        'score': float(score)
                    }
                    for doc, score in results
                ]
            else:
                results = self.vector_store.search(query, k, filters)
                return [
                    {
                        'content': doc.metadata.get('original_content', doc.page_content),
                        'metadata': doc.metadata
                    }
                    for doc in results
                ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def delete_chunk_vectors(self, milvus_ids: List[str]):
        """
        删除 chunk 向量

        Args:
            milvus_ids: Milvus 向量 ID 列表
        """
        if not milvus_ids:
            logger.warning("No Milvus IDs provided for deletion")
            return

        logger.info(f"Deleting {len(milvus_ids)} chunk vectors...")

        try:
            self.vector_store.delete_by_ids(milvus_ids)
            logger.info("✅ Chunk vectors deleted successfully")

        except Exception as e:
            logger.error(f"❌ Failed to delete chunk vectors: {e}")
            raise


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    try:
        manager = VectorizationManager()

        # 测试单个向量化
        test_chunk = {
            'id': 100,
            'content': '这是一个测试chunk，用于验证向量化功能',
            'document_id': 1,
            'chunk_id': 1,
            'source_file': 'test.md',
            'user_tag': '测试',
            'content_tags': ['测试', '验证'],
            'is_atomic': False,
            'atomic_type': None,
            'token_count': 20,
            'status': 0
        }

        result = manager.vectorize_single_chunk(test_chunk, document_tags=['测试文档'])
        print(f"\n✅ Single vectorization result: {result}")

        if result['success']:
            # 测试搜索
            search_results = manager.search_chunks(
                "测试功能",
                k=1,
                with_score=True
            )
            print(f"\n✅ Search results: {len(search_results)}")
            for r in search_results:
                print(f"  Content: {r['content'][:50]}...")
                print(f"  Score: {r['score']}")
                print(f"  Metadata: {r['metadata']}")

        print("\n✅ VectorizationManager test passed!")

    except Exception as e:
        print(f"\n❌ VectorizationManager test failed: {e}")
        import traceback
        traceback.print_exc()
