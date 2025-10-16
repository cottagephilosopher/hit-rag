"""
数据库操作模块
提供 SQLite 数据库的连接、初始化和基础操作
"""

import sqlite3
import json
import threading
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 数据库文件路径
DB_FILE = Path(os.getenv("DB_FILE", ".dbs/rag_preprocessor.db"))

# 数据库锁（用于并发控制）
_DB_LOCK = threading.Lock()


def _json_dump(obj: Any) -> str:
    """将对象序列化为JSON字符串"""
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False)


def _json_load(text: Optional[str]) -> Any:
    """将JSON字符串反序列化为对象"""
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


@contextmanager
def get_connection():
    """获取数据库连接（上下文管理器）"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # 返回字典格式
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """初始化数据库（创建表和索引）"""
    # 执行文档相关表的 schema
    schema_file = Path(__file__).parent / ".dbs/schema.sql"
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_sql = f.read()

    # 执行对话相关表的 schema
    chat_schema_file = Path(__file__).parent / ".dbs/chat_schema.sql"
    chat_schema_sql = ""
    if chat_schema_file.exists():
        with open(chat_schema_file, 'r', encoding='utf-8') as f:
            chat_schema_sql = f.read()

    # 执行 RAG 配置表的 schema
    rag_config_schema_file = Path(__file__).parent / ".dbs/rag_config_schema.sql"
    rag_config_schema_sql = ""
    if rag_config_schema_file.exists():
        with open(rag_config_schema_file, 'r', encoding='utf-8') as f:
            rag_config_schema_sql = f.read()

    # 执行提示词配置表的 schema
    prompt_config_schema_file = Path(__file__).parent / ".dbs/prompt_config_schema.sql"
    prompt_config_schema_sql = ""
    if prompt_config_schema_file.exists():
        with open(prompt_config_schema_file, 'r', encoding='utf-8') as f:
            prompt_config_schema_sql = f.read()

    with _DB_LOCK:
        with get_connection() as conn:
            # 执行文档表 schema
            conn.executescript(schema_sql)
            # 执行对话表 schema（如果存在）
            if chat_schema_sql:
                conn.executescript(chat_schema_sql)
            # 执行 RAG 配置表 schema（如果存在）
            if rag_config_schema_sql:
                conn.executescript(rag_config_schema_sql)
            # 执行提示词配置表 schema（如果存在）
            if prompt_config_schema_sql:
                conn.executescript(prompt_config_schema_sql)

    print(f"✅ Database initialized at: {DB_FILE}")


# ============================================
# Document 操作
# ============================================

def create_document(
    filename: str,
    source_path: str,
    status: str = 'pending'
) -> int:
    """创建文档记录"""
    with _DB_LOCK:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (filename, source_path, status)
                VALUES (?, ?, ?)
                """,
                (filename, source_path, status)
            )
            return cursor.lastrowid


def get_document(document_id: int) -> Optional[Dict[str, Any]]:
    """根据ID获取文档"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,)
        ).fetchone()
        return dict(row) if row else None


def get_document_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    """根据文件名获取文档"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE filename = ?",
            (filename,)
        ).fetchone()
        return dict(row) if row else None


def update_document(
    document_id: int,
    *,
    status: Optional[str] = None,
    total_chunks: Optional[int] = None,
    total_tokens: Optional[int] = None,
    processed_at: Optional[datetime] = None,
    error_message: Optional[str] = None
):
    """更新文档信息"""
    fields = []
    params = []

    if status is not None:
        fields.append("status = ?")
        params.append(status)

    if total_chunks is not None:
        fields.append("total_chunks = ?")
        params.append(total_chunks)

    if total_tokens is not None:
        fields.append("total_tokens = ?")
        params.append(total_tokens)

    if processed_at is not None:
        fields.append("processed_at = ?")
        params.append(processed_at)

    if error_message is not None:
        fields.append("error_message = ?")
        params.append(error_message)

    if not fields:
        return

    params.append(document_id)

    with _DB_LOCK:
        with get_connection() as conn:
            conn.execute(
                f"UPDATE documents SET {', '.join(fields)} WHERE id = ?",
                params
            )


# ============================================
# Chunk 操作
# ============================================

def create_chunk(
    document_id: int,
    chunk_id: int,
    content: str,
    token_start: int,
    token_end: int,
    token_count: int,
    *,
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
    user_tag: Optional[str] = None,
    content_tags: Optional[List[str]] = None,
    is_atomic: bool = False,
    atomic_type: Optional[str] = None,
    status: int = 0
) -> int:
    """创建chunk记录"""
    with _DB_LOCK:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO document_chunks (
                    document_id, chunk_id, content,
                    token_start, token_end, token_count,
                    char_start, char_end, user_tag, content_tags,
                    is_atomic, atomic_type, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id, chunk_id, content,
                    token_start, token_end, token_count,
                    char_start, char_end, user_tag,
                    _json_dump(content_tags) if content_tags else None,
                    is_atomic, atomic_type, status
                )
            )
            return cursor.lastrowid


def get_chunk(chunk_id: int) -> Optional[Dict[str, Any]]:
    """根据ID获取chunk（包含source_file）"""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT c.*, d.filename as source_file
            FROM document_chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.id = ?
            """,
            (chunk_id,)
        ).fetchone()

        if not row:
            return None

        chunk = dict(row)
        chunk['content_tags'] = _json_load(chunk.get('content_tags'))
        return chunk


def get_chunk_by_chunk_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """根据 chunk_id（字符串格式）获取 chunk"""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT c.*, d.filename as source_file
            FROM document_chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.chunk_id = ?
            """,
            (chunk_id,)
        ).fetchone()

        if not row:
            return None

        chunk = dict(row)
        chunk['content_tags'] = _json_load(chunk.get('content_tags'))
        return chunk


def get_chunks_by_document(document_id: int) -> List[Dict[str, Any]]:
    """获取文档的所有chunks"""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_id ASC
            """,
            (document_id,)
        ).fetchall()

        chunks = []
        for row in rows:
            chunk = dict(row)
            chunk['content_tags'] = _json_load(chunk.get('content_tags'))

            # ���果有编辑内容，使用编辑内容；否则使用原始内容
            # 但为了保持兼容性，我们在返回时保留 content 字段
            if not chunk.get('edited_content'):
                chunk['edited_content'] = chunk['content']

            chunks.append(chunk)

        return chunks


def update_chunk(
    chunk_id: int,
    *,
    edited_content: Optional[str] = None,
    status: Optional[int] = None,
    content_tags: Optional[List[str]] = None,
    user_tag: Optional[str] = None,
    last_editor_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    更新chunk并自动增加版本号
    返回更新前的数据用于记录日志
    """
    # 先获取旧数据
    old_chunk = get_chunk(chunk_id)
    if not old_chunk:
        raise ValueError(f"Chunk {chunk_id} not found")

    fields = []
    params = []

    if edited_content is not None:
        fields.append("edited_content = ?")
        params.append(edited_content)

    if status is not None:
        fields.append("status = ?")
        params.append(status)

    if content_tags is not None:
        fields.append("content_tags = ?")
        params.append(_json_dump(content_tags))

    if user_tag is not None:
        fields.append("user_tag = ?")
        params.append(user_tag)

    if last_editor_id is not None:
        fields.append("last_editor_id = ?")
        params.append(last_editor_id)

    if not fields:
        return old_chunk

    # 自动增加版本号
    fields.append("version = COALESCE(version, 0) + 1")
    params.append(chunk_id)

    with _DB_LOCK:
        with get_connection() as conn:
            conn.execute(
                f"UPDATE document_chunks SET {', '.join(fields)} WHERE id = ?",
                params
            )

    return old_chunk


# ============================================
# Log 操作
# ============================================

def insert_log(
    *,
    document_id: int,
    action: str,
    message: Optional[str] = None,
    chunk_id: Optional[int] = None,
    user_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None
) -> int:
    """插入操作日志"""
    with _DB_LOCK:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO document_logs (
                    document_id, chunk_id, action, message, user_id, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk_id,
                    action,
                    message,
                    user_id,
                    _json_dump(payload)
                )
            )
            return cursor.lastrowid


def get_chunk_logs(chunk_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """获取chunk的变更历史"""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, document_id, chunk_id, action, message,
                   user_id, created_at, payload
            FROM document_logs
            WHERE chunk_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (chunk_id, limit)
        ).fetchall()

        logs = []
        for row in rows:
            log = dict(row)
            log['payload'] = _json_load(log.get('payload'))
            logs.append(log)

        return logs


def get_document_logs(document_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """获取文档的所有日志"""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, document_id, chunk_id, action, message,
                   user_id, created_at, payload
            FROM document_logs
            WHERE document_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (document_id, limit)
        ).fetchall()

        logs = []
        for row in rows:
            log = dict(row)
            log['payload'] = _json_load(log.get('payload'))
            logs.append(log)

        return logs


# ============================================
# 工具函数
# ============================================

def import_json_to_db(json_file: Path, filename: str) -> int:
    """
    从JSON文件导入数据到数据库
    返回document_id
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    chunks = data.get('chunks', [])

    # 检查文档是否已存在
    existing_doc = get_document_by_filename(filename)
    if existing_doc:
        document_id = existing_doc['id']
        # 更新文档状态
        update_document(
            document_id,
            status='completed',
            total_chunks=len(chunks),
            total_tokens=metadata.get('statistics', {}).get('total_tokens', 0),
            processed_at=datetime.utcnow()
        )
    else:
        # 创建新文档
        document_id = create_document(
            filename=filename,
            source_path=metadata.get('source_file', ''),
            status='completed'
        )
        update_document(
            document_id,
            total_chunks=len(chunks),
            total_tokens=metadata.get('statistics', {}).get('total_tokens', 0),
            processed_at=datetime.utcnow()
        )

    # 删除旧chunks（如果存在）
    with _DB_LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))

    # 插入所有chunks
    for chunk in chunks:
        db_chunk_id = create_chunk(
            document_id=document_id,
            chunk_id=chunk.get('chunk_id'),
            content=chunk.get('content', ''),
            token_start=chunk.get('token_start', 0),
            token_end=chunk.get('token_end', 0),
            token_count=chunk.get('token_count', 0),
            char_start=chunk.get('char_start'),
            char_end=chunk.get('char_end'),
            user_tag=chunk.get('user_tag'),
            content_tags=chunk.get('content_tags', []),
            is_atomic=chunk.get('is_atomic', False),
            atomic_type=chunk.get('atomic_type'),
            status=chunk.get('status', 0)
        )

        # 记录导入日志
        insert_log(
            document_id=document_id,
            chunk_id=db_chunk_id,
            action='create',
            message='从JSON导入',
            user_id='system',
            payload={'source': str(json_file)}
        )

    print(f"✅ Imported {len(chunks)} chunks from {json_file}")
    return document_id


# ============================================
# 标签管理相关函数
# ============================================

def get_document_tags(document_id: int) -> List[str]:
    """获取文档的所有标签"""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT tag_text FROM document_tags
            WHERE document_id = ?
            ORDER BY tag_order ASC, id ASC
            """,
            (document_id,)
        ).fetchall()
        return [row['tag_text'] for row in rows]


def add_document_tag(document_id: int, tag_text: str) -> bool:
    """添加文档标签"""
    tag_text = tag_text.strip()
    if not tag_text:
        return False

    with get_connection() as conn:
        try:
            # 获取当前最大的 tag_order
            max_order_row = conn.execute(
                "SELECT MAX(tag_order) as max_order FROM document_tags WHERE document_id = ?",
                (document_id,)
            ).fetchone()
            next_order = (max_order_row['max_order'] or 0) + 1

            conn.execute(
                """
                INSERT INTO document_tags (document_id, tag_text, tag_order)
                VALUES (?, ?, ?)
                """,
                (document_id, tag_text, next_order)
            )
            return True
        except sqlite3.IntegrityError:
            # 标签已存在
            return False


def remove_document_tag(document_id: int, tag_text: str) -> bool:
    """删除文档标签"""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM document_tags WHERE document_id = ? AND tag_text = ?",
            (document_id, tag_text)
        )
        return cursor.rowcount > 0


def get_tags_by_filename(filename: str) -> List[str]:
    """根据文件名获取标签"""
    doc = get_document_by_filename(filename)
    if not doc:
        return []
    return get_document_tags(doc['id'])


def add_tag_by_filename(filename: str, tag_text: str) -> bool:
    """根据文件名添加标签"""
    doc = get_document_by_filename(filename)
    if not doc:
        return False
    return add_document_tag(doc['id'], tag_text)


def remove_tag_by_filename(filename: str, tag_text: str) -> bool:
    """根据文件名删除标签"""
    doc = get_document_by_filename(filename)
    if not doc:
        return False
    return remove_document_tag(doc['id'], tag_text)


# ============================================
# 向量化相关操作
# ============================================

def update_chunk_milvus_id(chunk_id: int, milvus_id: str):
    """
    更新 chunk 的 Milvus ID 和状态

    Args:
        chunk_id: chunk 数据库 ID
        milvus_id: Milvus 向量 ID
    """
    with _DB_LOCK:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE document_chunks
                SET milvus_id = ?, status = 2
                WHERE id = ?
                """,
                (milvus_id, chunk_id)
            )


def get_vectorizable_chunks(document_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    获取可向量化的 chunks（status != -1 且 status != 2）

    Args:
        document_id: 可选的文档 ID，用于过滤特定文档

    Returns:
        可向量化的 chunk 列表
    """
    query = """
        SELECT dc.*, d.filename as source_file
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE dc.status != -1 AND dc.status != 2
    """
    params = []

    if document_id:
        query += " AND dc.document_id = ?"
        params.append(document_id)

    query += " ORDER BY dc.document_id, dc.chunk_id"

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

        chunks = []
        for row in rows:
            chunk = dict(row)
            chunk['content_tags'] = _json_load(chunk.get('content_tags'))
            chunks.append(chunk)

        return chunks


def get_vectorization_stats() -> Dict[str, int]:
    """
    获取向量化统计信息

    Returns:
        统计字典: {
            'total': 总chunk数,
            'vectorized': 已向量化数量,
            'pending': 待向量化数量,
            'deprecated': 废弃数量,
            'total_documents': 总文档数,
            'total_tokens': 总token数
        }
    """
    with get_connection() as conn:
        # 获取 chunk 统计
        stats = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 2 THEN 1 ELSE 0 END) as vectorized,
                SUM(CASE WHEN status IN (0, 1) THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = -1 THEN 1 ELSE 0 END) as deprecated,
                SUM(token_count) as total_tokens
            FROM document_chunks
        """).fetchone()
        
        # 获取文档数
        doc_count = conn.execute("""
            SELECT COUNT(DISTINCT document_id) as total_documents
            FROM document_chunks
        """).fetchone()

        return {
            'total': stats['total'] or 0,
            'vectorized': stats['vectorized'] or 0,
            'pending': stats['pending'] or 0,
            'deprecated': stats['deprecated'] or 0,
            'total_documents': doc_count['total_documents'] or 0,
            'total_tokens': stats['total_tokens'] or 0
        }


def get_chunk_by_milvus_id(milvus_id: str) -> Optional[Dict[str, Any]]:
    """
    根据 Milvus ID 获取 chunk

    Args:
        milvus_id: Milvus 向量 ID

    Returns:
        chunk 字典或 None
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM document_chunks WHERE milvus_id = ?",
            (milvus_id,)
        ).fetchone()

        if not row:
            return None

        chunk = dict(row)
        chunk['content_tags'] = _json_load(chunk.get('content_tags'))
        return chunk


# ============================================
# 全局标签管理
# ============================================

def get_all_tags_with_stats() -> List[Dict[str, Any]]:
    """
    获取所有标签及其统计信息

    Returns:
        标签列表，每个标签包含：
        - name: 标签名称
        - type: 标签类型 (user_tag | content_tag | document_tag | multiple)
        - count: 使用次数（chunk 数量）
        - chunk_ids: 包含该标签的 chunk IDs
        - document_count: 文档级标签关联的文档数量
    """
    with get_connection() as conn:
        # 收集所有 user_tag
        user_tag_rows = conn.execute("""
            SELECT user_tag, GROUP_CONCAT(id) as chunk_ids, COUNT(*) as count
            FROM document_chunks
            WHERE user_tag IS NOT NULL AND user_tag != ''
            GROUP BY user_tag
        """).fetchall()

        # 收集所有 content_tags
        content_tag_rows = conn.execute("""
            SELECT id, content_tags
            FROM document_chunks
            WHERE content_tags IS NOT NULL AND content_tags != '[]'
        """).fetchall()

        # 收集所有文档级标签
        document_tag_rows = conn.execute("""
            SELECT tag_text, COUNT(DISTINCT document_id) as doc_count
            FROM document_tags
            GROUP BY tag_text
        """).fetchall()

    # 统计标签
    tag_stats = {}

    # 处理 user_tags
    for row in user_tag_rows:
        tag_name = row['user_tag']
        chunk_ids = [int(x) for x in row['chunk_ids'].split(',')]
        tag_stats[tag_name] = {
            'name': tag_name,
            'type': 'user_tag',
            'count': row['count'],
            'chunk_ids': chunk_ids,
            'document_count': 0
        }

    # 处理 content_tags
    for row in content_tag_rows:
        chunk_id = row['id']
        tags = _json_load(row['content_tags']) or []

        for tag in tags:
            # 移除 @ 前缀（人工标签）
            clean_tag = tag.lstrip('@') if isinstance(tag, str) else tag
            if not clean_tag:
                continue

            if clean_tag in tag_stats:
                # 标签已存在（可能来自 user_tag）
                if tag_stats[clean_tag]['type'] == 'user_tag':
                    tag_stats[clean_tag]['type'] = 'multiple'
                elif tag_stats[clean_tag]['type'] == 'content_tag':
                    pass  # 保持为 content_tag
                tag_stats[clean_tag]['count'] += 1
                if chunk_id not in tag_stats[clean_tag]['chunk_ids']:
                    tag_stats[clean_tag]['chunk_ids'].append(chunk_id)
            else:
                tag_stats[clean_tag] = {
                    'name': clean_tag,
                    'type': 'content_tag',
                    'count': 1,
                    'chunk_ids': [chunk_id],
                    'document_count': 0
                }

    # 处理文档级标签
    for row in document_tag_rows:
        tag_name = row['tag_text'].strip()
        doc_count = row['doc_count']

        if tag_name in tag_stats:
            # 标签已存在于 chunk 标签中
            tag_stats[tag_name]['document_count'] = doc_count
            if tag_stats[tag_name]['type'] in ['user_tag', 'content_tag']:
                tag_stats[tag_name]['type'] = 'multiple'
        else:
            # 纯文档级标签
            tag_stats[tag_name] = {
                'name': tag_name,
                'type': 'document_tag',
                'count': 0,  # chunk 数量为 0
                'chunk_ids': [],
                'document_count': doc_count
            }

    # 返回排序后的列表（按使用次数降序，文档级标签按文档数量排序）
    return sorted(
        tag_stats.values(),
        key=lambda x: (x['count'] + x['document_count'] * 10),  # 文档级标签权重更高
        reverse=True
    )


def delete_tag_from_all_chunks(tag_name: str) -> int:
    """
    从所有 chunks 中删除指定标签

    Args:
        tag_name: 要删除的标签名称

    Returns:
        受影响的 chunk 数量
    """
    affected_count = 0

    with _DB_LOCK:
        with get_connection() as conn:
            # 1. 从 user_tag 中删除
            user_tag_result = conn.execute("""
                UPDATE document_chunks
                SET user_tag = NULL
                WHERE user_tag = ?
            """, (tag_name,))
            affected_count += user_tag_result.rowcount

            # 2. 从 content_tags 中删除
            chunks_with_content_tags = conn.execute("""
                SELECT id, content_tags
                FROM document_chunks
                WHERE content_tags IS NOT NULL AND content_tags != '[]'
            """).fetchall()

            for chunk in chunks_with_content_tags:
                chunk_id = chunk['id']
                tags = _json_load(chunk['content_tags']) or []

                # 清理标签名（移除 @ 前缀）并过滤
                original_len = len(tags)
                new_tags = [
                    tag for tag in tags
                    if (tag.lstrip('@') if isinstance(tag, str) else tag) != tag_name
                ]

                if len(new_tags) < original_len:
                    # 标签被删除，更新数据库
                    conn.execute("""
                        UPDATE document_chunks
                        SET content_tags = ?
                        WHERE id = ?
                    """, (_json_dump(new_tags), chunk_id))
                    affected_count += 1

    return affected_count


def rename_tag_in_all_chunks(old_name: str, new_name: str) -> int:
    """
    在所有 chunks 中重命名标签

    Args:
        old_name: 旧标签名
        new_name: 新标签名

    Returns:
        受影响的 chunk 数量
    """
    affected_count = 0

    with _DB_LOCK:
        with get_connection() as conn:
            # 1. 重命名 user_tag
            user_tag_result = conn.execute("""
                UPDATE document_chunks
                SET user_tag = ?
                WHERE user_tag = ?
            """, (new_name, old_name))
            affected_count += user_tag_result.rowcount

            # 2. 重命名 content_tags 中的标签
            chunks_with_content_tags = conn.execute("""
                SELECT id, content_tags
                FROM document_chunks
                WHERE content_tags IS NOT NULL AND content_tags != '[]'
            """).fetchall()

            for chunk in chunks_with_content_tags:
                chunk_id = chunk['id']
                tags = _json_load(chunk['content_tags']) or []

                # 检查是否包含要重命名的标签
                modified = False
                new_tags = []
                for tag in tags:
                    # 处理带 @ 前缀的标签
                    if isinstance(tag, str):
                        prefix = '@' if tag.startswith('@') else ''
                        clean_tag = tag.lstrip('@')

                        if clean_tag == old_name:
                            new_tags.append(f"{prefix}{new_name}")
                            modified = True
                        else:
                            new_tags.append(tag)
                    else:
                        new_tags.append(tag)

                if modified:
                    conn.execute("""
                        UPDATE document_chunks
                        SET content_tags = ?
                        WHERE id = ?
                    """, (_json_dump(new_tags), chunk_id))
                    affected_count += 1

    return affected_count


def merge_tags_in_all_chunks(source_tags: List[str], target_tag: str) -> Dict[str, int]:
    """
    合并多个标签为一个标签

    Args:
        source_tags: 源标签列表（要被合并的标签）
        target_tag: 目标标签（合并后的标签名）

    Returns:
        {
            'affected_chunks': 受影响的 chunk 数量,
            'merged_count': 被合并的标签数量
        }
    """
    affected_chunks = set()

    with _DB_LOCK:
        with get_connection() as conn:
            # 1. 处理 user_tag
            for source_tag in source_tags:
                chunks = conn.execute("""
                    SELECT id FROM document_chunks
                    WHERE user_tag = ?
                """, (source_tag,)).fetchall()

                if chunks:
                    chunk_ids = [c['id'] for c in chunks]
                    conn.execute(f"""
                        UPDATE document_chunks
                        SET user_tag = ?
                        WHERE id IN ({','.join('?' * len(chunk_ids))})
                    """, [target_tag] + chunk_ids)
                    affected_chunks.update(chunk_ids)

            # 2. 处理 content_tags
            chunks_with_content_tags = conn.execute("""
                SELECT id, content_tags
                FROM document_chunks
                WHERE content_tags IS NOT NULL AND content_tags != '[]'
            """).fetchall()

            for chunk in chunks_with_content_tags:
                chunk_id = chunk['id']
                tags = _json_load(chunk['content_tags']) or []

                # 检查是否包含要合并的标签
                has_source_tags = False
                new_tags = []
                target_added = False

                for tag in tags:
                    if isinstance(tag, str):
                        prefix = '@' if tag.startswith('@') else ''
                        clean_tag = tag.lstrip('@')

                        if clean_tag in source_tags:
                            # 遇到源标签
                            has_source_tags = True
                            if not target_added:
                                # 第一次遇到源标签时，添加目标标签
                                new_tags.append(f"{prefix}{target_tag}")
                                target_added = True
                            # 否则跳过（去重）
                        elif clean_tag == target_tag:
                            # 已经存在目标标签，标记为已添加
                            new_tags.append(tag)
                            target_added = True
                        else:
                            # 保留其他标签
                            new_tags.append(tag)
                    else:
                        new_tags.append(tag)

                if has_source_tags:
                    # 去重
                    final_tags = []
                    seen = set()
                    for tag in new_tags:
                        clean = tag.lstrip('@') if isinstance(tag, str) else tag
                        if clean not in seen:
                            final_tags.append(tag)
                            seen.add(clean)

                    conn.execute("""
                        UPDATE document_chunks
                        SET content_tags = ?
                        WHERE id = ?
                    """, (_json_dump(final_tags), chunk_id))
                    affected_chunks.add(chunk_id)

    return {
        'affected_chunks': len(affected_chunks),
        'merged_count': len(source_tags)
    }


# ============================================
# RAG 配置管理
# ============================================

def get_rag_config(config_key: Optional[str] = None) -> Dict[str, Any]:
    """
    获取 RAG 配置

    Args:
        config_key: 可选的配置键，如果提供则只返回该配置项

    Returns:
        配置字典或配置项
    """
    with get_connection() as conn:
        if config_key:
            row = conn.execute(
                "SELECT * FROM rag_config WHERE config_key = ?",
                (config_key,)
            ).fetchone()
            return dict(row) if row else None
        else:
            rows = conn.execute(
                "SELECT * FROM rag_config ORDER BY category, id"
            ).fetchall()
            return {row['config_key']: dict(row) for row in rows}


def update_rag_config(config_key: str, config_value: float) -> bool:
    """
    更新 RAG 配置项

    Args:
        config_key: 配置键
        config_value: 配置值

    Returns:
        是否更新成功
    """
    with _DB_LOCK:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE rag_config
                SET config_value = ?, updated_at = CURRENT_TIMESTAMP
                WHERE config_key = ?
                """,
                (config_value, config_key)
            )
            return cursor.rowcount > 0


def batch_update_rag_config(configs: Dict[str, float]) -> int:
    """
    批量更新 RAG 配置

    Args:
        configs: 配置字典 {config_key: config_value}

    Returns:
        更新的配置项数量
    """
    updated_count = 0
    with _DB_LOCK:
        with get_connection() as conn:
            for config_key, config_value in configs.items():
                cursor = conn.execute(
                    """
                    UPDATE rag_config
                    SET config_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE config_key = ?
                    """,
                    (config_value, config_key)
                )
                if cursor.rowcount > 0:
                    updated_count += 1
    return updated_count


def init_rag_config_from_env():
    """
    从环境变量初始化 RAG 配置（如果表为空）
    """
    # 检查配置表是否为空
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM rag_config").fetchone()['cnt']
        if count > 0:
            print("ℹ️  RAG 配置已存在，跳过初始化")
            return

    # 定义默认配置（从环境变量读取）
    default_configs = [
        # 对话配置
        {
            'config_key': 'ENABLE_CHAT_MODE',
            'config_value': float(os.getenv('ENABLE_CHAT_MODE', 'true').lower() == 'true'),
            'description': '是否启用闲聊模式',
            'min_value': 0.0,
            'max_value': 1.0,
            'default_value': 1.0,
            'category': 'chat'
        },
        {
            'config_key': 'CHAT_MODE_THRESHOLD',
            'config_value': float(os.getenv('CHAT_MODE_THRESHOLD', '0.7')),
            'description': '闲聊模式阈值（0.0-1.0，超过此值判定为闲聊）',
            'min_value': 0.0,
            'max_value': 1.0,
            'default_value': 0.7,
            'category': 'chat'
        },
        {
            'config_key': 'ENABLE_AUTO_TAG_FILTER',
            'config_value': float(os.getenv('ENABLE_AUTO_TAG_FILTER', 'true').lower() == 'true'),
            'description': '是否启用自动标签识别筛选',
            'min_value': 0.0,
            'max_value': 1.0,
            'default_value': 1.0,
            'category': 'chat'
        },
        {
            'config_key': 'AUTO_TAG_FILTER_THRESHOLD',
            'config_value': float(os.getenv('AUTO_TAG_FILTER_THRESHOLD', '0.5')),
            'description': '自动标签识别置信度阈值（0.0-1.0）',
            'min_value': 0.0,
            'max_value': 1.0,
            'default_value': 0.5,
            'category': 'chat'
        },

        # 置信度阈值
        {
            'config_key': 'RAG_CONFIDENCE_THRESHOLD',
            'config_value': float(os.getenv('RAG_CONFIDENCE_THRESHOLD', '0.47')),
            'description': '置信度阈值：用于判断检索结果是否可信（0.0-1.0）',
            'min_value': 0.0,
            'max_value': 1.0,
            'default_value': 0.5,
            'category': 'threshold'
        },
        {
            'config_key': 'RAG_RERANK_SCORE_THRESHOLD',
            'config_value': float(os.getenv('RAG_RERANK_SCORE_THRESHOLD', '0.16')),
            'description': 'Rerank 分数阈值（0.0-1.0，越大越严格）',
            'min_value': 0.0,
            'max_value': 1.0,
            'default_value': 0.2,
            'category': 'threshold'
        },
        {
            'config_key': 'RAG_L2_DISTANCE_THRESHOLD',
            'config_value': float(os.getenv('RAG_L2_DISTANCE_THRESHOLD', '1.2')),
            'description': 'L2 距离阈值（越小越相关）',
            'min_value': 0.0,
            'max_value': 5.0,
            'default_value': 1.2,
            'category': 'threshold'
        },
        {
            'config_key': 'RAG_RERANK_GOOD_THRESHOLD',
            'config_value': float(os.getenv('RAG_RERANK_GOOD_THRESHOLD', '0.18')),
            'description': 'Rerank 良好分数阈值',
            'min_value': 0.0,
            'max_value': 1.0,
            'default_value': 0.2,
            'category': 'threshold'
        },
        {
            'config_key': 'RAG_RERANK_EXCELLENT_THRESHOLD',
            'config_value': float(os.getenv('RAG_RERANK_EXCELLENT_THRESHOLD', '0.3')),
            'description': 'Rerank 优秀分数阈值',
            'min_value': 0.0,
            'max_value': 1.0,
            'default_value': 0.3,
            'category': 'threshold'
        },
        {
            'config_key': 'RAG_L2_GOOD_THRESHOLD',
            'config_value': float(os.getenv('RAG_L2_GOOD_THRESHOLD', '1.2')),
            'description': 'L2 距离良好阈值',
            'min_value': 0.0,
            'max_value': 5.0,
            'default_value': 1.2,
            'category': 'threshold'
        },
        {
            'config_key': 'RAG_L2_EXCELLENT_THRESHOLD',
            'config_value': float(os.getenv('RAG_L2_EXCELLENT_THRESHOLD', '1.0')),
            'description': 'L2 距离优秀阈值',
            'min_value': 0.0,
            'max_value': 5.0,
            'default_value': 1.0,
            'category': 'threshold'
        },

        # 检索数量配置
        {
            'config_key': 'RAG_ENTITY_TOP_K',
            'config_value': float(os.getenv('RAG_ENTITY_TOP_K', '5')),
            'description': '单实体检索数量',
            'min_value': 1.0,
            'max_value': 50.0,
            'default_value': 3.0,
            'category': 'retrieval'
        },
        {
            'config_key': 'RAG_MULTI_ENTITY_DEDUP_LIMIT',
            'config_value': float(os.getenv('RAG_MULTI_ENTITY_DEDUP_LIMIT', '20')),
            'description': '多实体检索后去重数量',
            'min_value': 1.0,
            'max_value': 100.0,
            'default_value': 15.0,
            'category': 'retrieval'
        },
        {
            'config_key': 'RAG_RERANK_TOP_N',
            'config_value': float(os.getenv('RAG_RERANK_TOP_N', '8')),
            'description': '重排序后保留数量',
            'min_value': 1.0,
            'max_value': 50.0,
            'default_value': 8.0,
            'category': 'retrieval'
        },
        {
            'config_key': 'RAG_SINGLE_QUERY_TOP_K',
            'config_value': float(os.getenv('RAG_SINGLE_QUERY_TOP_K', '8')),
            'description': '单查询检索数量',
            'min_value': 1.0,
            'max_value': 50.0,
            'default_value': 8.0,
            'category': 'retrieval'
        },
        {
            'config_key': 'RAG_FILES_DISPLAY_LIMIT',
            'config_value': float(os.getenv('RAG_FILES_DISPLAY_LIMIT', '5')),
            'description': '文件源显示数量',
            'min_value': 1.0,
            'max_value': 20.0,
            'default_value': 5.0,
            'category': 'retrieval'
        },
    ]

    # 批量插入配置
    with _DB_LOCK:
        with get_connection() as conn:
            for config in default_configs:
                conn.execute(
                    """
                    INSERT INTO rag_config (
                        config_key, config_value, description,
                        min_value, max_value, default_value, category
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        config['config_key'],
                        config['config_value'],
                        config['description'],
                        config['min_value'],
                        config['max_value'],
                        config['default_value'],
                        config['category']
                    )
                )

    print(f"✅ 已初始化 {len(default_configs)} 个 RAG 配置项")


# ============================================
# 提示词配置管理
# ============================================

def get_prompt_config(prompt_key: Optional[str] = None) -> Dict[str, Any]:
    """
    获取提示词配置

    Args:
        prompt_key: 可选的提示词键，如果提供则只返回该配置项

    Returns:
        配置字典或配置项
    """
    with get_connection() as conn:
        if prompt_key:
            row = conn.execute(
                "SELECT * FROM prompt_config WHERE prompt_key = ?",
                (prompt_key,)
            ).fetchone()
            return dict(row) if row else None
        else:
            rows = conn.execute(
                "SELECT * FROM prompt_config ORDER BY category, id"
            ).fetchall()
            return {row['prompt_key']: dict(row) for row in rows}


def update_prompt_config(prompt_key: str, prompt_value: str) -> bool:
    """
    更新提示词配置项

    Args:
        prompt_key: 提示词键
        prompt_value: 提示词内容

    Returns:
        是否更新成功
    """
    with _DB_LOCK:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE prompt_config
                SET prompt_value = ?, updated_at = CURRENT_TIMESTAMP
                WHERE prompt_key = ?
                """,
                (prompt_value, prompt_key)
            )
            return cursor.rowcount > 0


def batch_update_prompt_config(configs: Dict[str, str]) -> int:
    """
    批量更新提示词配置

    Args:
        configs: 配置字典 {prompt_key: prompt_value}

    Returns:
        更新的配置项数量
    """
    updated_count = 0
    with _DB_LOCK:
        with get_connection() as conn:
            for prompt_key, prompt_value in configs.items():
                cursor = conn.execute(
                    """
                    UPDATE prompt_config
                    SET prompt_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE prompt_key = ?
                    """,
                    (prompt_value, prompt_key)
                )
                if cursor.rowcount > 0:
                    updated_count += 1
    return updated_count


def init_prompt_config_from_templates():
    """
    从 prompt_templates.py 初始化提示词配置（如果表为空）
    """
    # 检查配置表是否为空
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM prompt_config").fetchone()['cnt']
        if count > 0:
            print("ℹ️  提示词配置已存在，跳过初始化")
            return

    # 默认提示词配置（使用模板变量，运行时动态替换）
    default_prompts = [
        {
            'prompt_key': 'CLEAN_AND_TAG_SYSTEM',
            'prompt_value': """你是一个专业的文档处理助手。你需要完成两个任务：

## 任务1: 识别并标记版式杂质
识别文档中的版式杂质（非核心内容），使用 <JUNK type="类型">内容</JUNK> 标记。

### 杂质特征参考
{{JUNK_FEATURES}}

## 任务2: 提取文档标签
分析文档内容，提取：
1. 用户标签: 从系统现有标签中选择最匹配的
2. 内容标签: 从系统现有标签中选择 {{CONTENT_TAG_COUNT}} 个最相关的

### 系统现有标签
{{EXISTING_TAGS}}

## 输出格式
请以 JSON 格式返回：
{
    "marked_text": "标记了杂质的完整文档...",
    "user_tag": "用户标签",
    "content_tags": ["标签1", "标签2", "标签3", "标签4", "标签5"]
}

## 重要约束
1. marked_text 必须保留原文的所有格式（换行、空格、Markdown 语法）
2. 只标记明确的杂质，不确定的保留
3. **必须使用系统现有标签**：user_tag 和 content_tags 必须从系统现有标签列表中选择，不允许创建新标签
4. **如果标签数量不足**：如果系统现有标签少于 {{CONTENT_TAG_COUNT}} 个，则尽量选择，不足的部分可以留空或重复使用
5. 标签不要重复
6. 必须返回有效的 JSON 格式

请始终返回有效的 JSON 格式响应。""",
            'description': '文档清洗和标签提取的系统提示词（支持变量：{{JUNK_FEATURES}}, {{EXISTING_TAGS}}, {{CONTENT_TAG_COUNT}}）',
            'category': 'clean_tag',
            'is_system_prompt': 1,
        },
        {
            'prompt_key': 'CHUNKING_SYSTEM',
            'prompt_value': """你是一个专业的文档切分助手，专门为 RAG（检索增强生成）系统准备文档块。

## 任务说明
请将提供的文档片段切分成多个语义连贯的小块，每块用于 RAG 检索。

## 切分原则（语义完整性优先）

**核心原则**: 语义完整性 > Token 数量限制

1. **Token 参考值**:
   - **最小值**: {{FINAL_MIN_TOKENS}} tokens（硬性要求，避免切片过小）
   - **目标值**: {{FINAL_TARGET_TOKENS}} tokens（理想大小，尽量接近）
   - **建议最大值**: {{FINAL_MAX_TOKENS}} tokens（可超出，优先保证语义完整）
   - **硬性上限**: {{FINAL_HARD_LIMIT}} tokens（安全阀，超出此值必须切分）

2. **语义完整性要求**（优先级从高到低）:
   - **句子完整性**: 绝不允许在句子中间切断
   - **段落完整性**: 尽量保持段落完整，不要将段落拆分
   - **小节完整性**: 标题与其内容必须在同一 chunk
   - **语义单元完整性**: 完整的概念、步骤、示例应保持在一起

## 特殊标记（ATOMIC 块）
对于以下内容，**必须**使用 <ATOMIC-TYPE> 标签标记：
- **表格**: <ATOMIC-TABLE>表格内容</ATOMIC-TABLE>
- **代码块**: <ATOMIC-CODE>代码内容</ATOMIC-CODE>
- **步骤序列**: <ATOMIC-STEP>步骤序列内容</ATOMIC-STEP>
- **语义完整的大段落**: <ATOMIC-CONTENT>完整内容</ATOMIC-CONTENT>

## 输出格式
请以 JSON 格式返回切分结果：
{
    "chunks": [
        {
            "content": "第一块的完整内容",
            "is_atomic": false,
            "atomic_type": null
        },
        {
            "content": "<ATOMIC-TABLE>表格内容</ATOMIC-TABLE>",
            "is_atomic": true,
            "atomic_type": "table"
        }
    ]
}

请始终返回有效的 JSON 格式响应。""",
            'description': '文档切分的系统提示词（支持变量：{{FINAL_MIN_TOKENS}}, {{FINAL_TARGET_TOKENS}}, {{FINAL_MAX_TOKENS}}, {{FINAL_HARD_LIMIT}}）',
            'category': 'chunk',
            'is_system_prompt': 1,
        },
    ]

    with _DB_LOCK:
        with get_connection() as conn:
            for prompt in default_prompts:
                # 设置默认值与初始值相同
                prompt['default_value'] = prompt['prompt_value']

                conn.execute(
                    """
                    INSERT INTO prompt_config (
                        prompt_key, prompt_value, description, category,
                        is_system_prompt, default_value
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prompt['prompt_key'],
                        prompt['prompt_value'],
                        prompt['description'],
                        prompt['category'],
                        prompt['is_system_prompt'],
                        prompt['default_value']
                    )
                )

    print(f"✅ 初始化了 {len(default_prompts)} 个提示词配置")


if __name__ == '__main__':
    # 测试：初始化数据库
    init_database()
    print("\n🔍 测试数据库操作:")

    # 测试：查询示例文档
    doc = get_document_by_filename('example.md')
    if doc:
        print(f"  文档: {doc['filename']}, 状态: {doc['status']}, Chunks: {doc['total_chunks']}")

        # 查询chunks
        chunks = get_chunks_by_document(doc['id'])
        print(f"  共有 {len(chunks)} 个切片")
        for chunk in chunks[:2]:
            print(f"    - Chunk #{chunk['chunk_id']}: {chunk['content'][:30]}...")
