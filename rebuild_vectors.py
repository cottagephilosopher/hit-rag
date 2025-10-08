#!/usr/bin/env python3
"""
重建所有向量

删除现有向量并使用新的向量化策略（包含user_tag）重新向量化所有chunks
"""

import os
import sys
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from database import get_connection
from vector_db.vectorization_manager import VectorizationManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """重建所有向量"""
    print("\n" + "="*80)
    print("重建向量 - 删除旧向量并使用新策略重新向量化")
    print("="*80)

    print("\n⚠️  警告: 此操作将:")
    print("  1. 删除所有已向量化的chunk的向量")
    print("  2. 将所有chunk的状态重置为'initial'(0)")
    print("  3. 使用新的向量化策略(包含user_tag)重新向量化所有chunk")

    confirm = input("\n是否继续? (输入 'yes' 确认): ")
    if confirm.lower() != 'yes':
        print("❌ 操作已取消")
        return

    try:
        manager = VectorizationManager()

        with get_connection() as conn:
            # 步骤1: 获取所有已向量化的chunks
            print("\n" + "-"*80)
            print("步骤1: 查询所有已向量化的chunks...")
            print("-"*80)

            result = conn.execute("""
                SELECT id, document_id, chunk_id, source_file, content, edited_content,
                       user_tag, content_tags, is_atomic, atomic_type, token_count, milvus_id
                FROM document_chunks
                WHERE status = 2
                ORDER BY document_id, chunk_id
            """).fetchall()

            if not result:
                print("没有已向量化的chunks，退出")
                return

            print(f"✓ 找到 {len(result)} 个已向量化的chunks")

            # 转换为字典列表
            chunks = []
            for row in result:
                chunk = {
                    'id': row['id'],
                    'document_id': row['document_id'],
                    'chunk_id': row['chunk_id'],
                    'source_file': row['source_file'],
                    'content': row['content'],
                    'edited_content': row['edited_content'],
                    'user_tag': row['user_tag'],
                    'content_tags': row['content_tags'] or [],
                    'is_atomic': bool(row['is_atomic']),
                    'atomic_type': row['atomic_type'],
                    'token_count': row['token_count'],
                    'milvus_id': row['milvus_id']
                }
                chunks.append(chunk)

            # 步骤2: 删除Milvus中的向量
            print("\n" + "-"*80)
            print("步骤2: 删除Milvus中的向量...")
            print("-"*80)

            chunk_db_ids = [chunk['id'] for chunk in chunks]

            try:
                manager.vector_store.delete_by_chunk_db_ids(chunk_db_ids)
                print(f"✓ 成功删除 {len(chunk_db_ids)} 个chunk的向量")
            except Exception as e:
                logger.error(f"删除向量失败: {e}")
                print(f"⚠️  删除向量时出现错误: {e}")
                print("继续执行...")

            # 步骤3: 重置数据库状态
            print("\n" + "-"*80)
            print("步骤3: 重置chunks的向量化状态...")
            print("-"*80)

            conn.execute("""
                UPDATE document_chunks
                SET status = 0, milvus_id = NULL
                WHERE status = 2
            """)

            print(f"✓ 已重置 {len(chunks)} 个chunk的状态")

            # 步骤4: 按文档分组重新向量化
            print("\n" + "-"*80)
            print("步骤4: 重新向量化所有chunks...")
            print("-"*80)

            # 按文档分组
            docs_chunks = {}
            for chunk in chunks:
                doc_id = chunk['document_id']
                if doc_id not in docs_chunks:
                    docs_chunks[doc_id] = []
                docs_chunks[doc_id].append(chunk)

            print(f"✓ 共 {len(docs_chunks)} 个文档需要重新向量化")

            total_success = 0
            total_failed = 0

            for doc_id, doc_chunks in docs_chunks.items():
                print(f"\n处理文档 ID={doc_id}, {len(doc_chunks)} 个chunks...")

                # 获取文件名
                if doc_chunks:
                    source_file = doc_chunks[0]['source_file']
                    print(f"  文件: {source_file}")

                    # 获取文档标签
                    tag_result = conn.execute("""
                        SELECT tags FROM documents WHERE id = ?
                    """, (doc_id,)).fetchone()

                    document_tags = tag_result['tags'] if tag_result and tag_result['tags'] else []
                    print(f"  文档标签: {document_tags}")
                else:
                    document_tags = []

                # 向量化
                try:
                    result = manager.vectorize_chunks(doc_chunks, document_tags)

                    success = result.get('success', 0)
                    failed = result.get('failed', 0)

                    total_success += success
                    total_failed += failed

                    print(f"  ✓ 成功: {success}, 失败: {failed}")

                except Exception as e:
                    logger.error(f"向量化文档 {doc_id} 失败: {e}")
                    print(f"  ❌ 向量化失败: {e}")
                    total_failed += len(doc_chunks)

            # 步骤5: 总结
            print("\n" + "="*80)
            print("重建完成")
            print("="*80)
            print(f"\n总计:")
            print(f"  - 删除向量: {len(chunks)} 个")
            print(f"  - 重新向量化成功: {total_success} 个")
            print(f"  - 失败: {total_failed} 个")

            if total_failed == 0:
                print(f"\n✅ 所有向量已成功重建！")
                print(f"\n现在可以在搜索页面测试'安装&维护'查询，应该能看到正确的排序了。")
            else:
                print(f"\n⚠️  重建完成，但有 {total_failed} 个chunk向量化失败")
                print(f"   请检查日志了解详情")

    except Exception as e:
        logger.error(f"重建向量失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 重建失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
