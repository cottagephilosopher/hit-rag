"""
文档和切片管理路由
提供文档列表、处理、chunk 更新、向量化、标签管理等功能
"""

import os
import json
import subprocess
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from database import (
    get_chunk,
    update_chunk,
    insert_log,
    get_chunk_logs,
    import_json_to_db,
    get_document_by_filename,
    get_tags_by_filename,
    add_tag_by_filename,
    remove_tag_by_filename,
    update_chunk_milvus_id,
    get_vectorizable_chunks,
    get_vectorization_stats,
    get_connection,
    get_all_tags_with_stats,
    delete_tag_from_all_chunks,
    rename_tag_in_all_chunks,
    merge_tags_in_all_chunks,
    create_document,
    create_chunk,
    get_chunks_by_document,
    # 系统标签管理
    get_system_tags,
    get_system_tags_with_stats,
    add_system_tag,
    remove_system_tag,
    rename_system_tag,
    convert_user_tag_to_system
)

from vector_db.vectorization_manager import VectorizationManager

router = APIRouter()

# ==================== 缓存配置 ====================
# 全局文档列表缓存 - 支持多种排序方式同时缓存
# 结构: {cache_key: (data, timestamp)}
_documents_list_cache = {}
_cache_ttl = timedelta(minutes=360)  # 缓存 5 分钟（文档不常变化）

# 单个文档状态缓存（用于向后兼容）
_document_cache = {}

# ==================== 路径配置 ====================
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).parent))
FILE_DIR = Path(os.getenv("FILE_DIR", BASE_DIR / "files"))
ALL_MD_DIR = Path(os.getenv("ALL_MD_DIR", BASE_DIR / "all-md"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))
CONVERTED_DIR = Path(os.getenv("CONVERTED_DIR", FILE_DIR / "converted"))
IKN_PLUS_DIR = Path(os.getenv("IKN_PLUS_DIR", Path(__file__).parent))

# 存储处理任务状态
processing_tasks = {}

# 初始化向量化管理器（延迟初始化）
vectorization_manager = None

def get_vectorization_manager() -> VectorizationManager:
    """获取向量化管理器（懒加载）"""
    global vectorization_manager
    if vectorization_manager is None:
        try:
            vectorization_manager = VectorizationManager()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"向量化服务初始化失败: {str(e)}")
    return vectorization_manager


# ==================== Pydantic Models ====================

class Document(BaseModel):
    filename: str
    status: str
    output_path: str | None = None
    processed_at: str | None = None
    error: str | None = None
    source_file_type: str | None = None


class ProcessRequest(BaseModel):
    filename: str


class ChunkUpdateRequest(BaseModel):
    edited_content: Optional[str] = None
    status: Optional[int] = None
    content_tags: Optional[List[str]] = None
    user_tag: Optional[str] = None
    editor_id: Optional[str] = "unknown"


class ChunkLogEntry(BaseModel):
    id: int
    action: str
    message: Optional[str]
    created_at: str
    user_id: Optional[str]
    payload: Optional[Dict[str, Any]]


class TagRequest(BaseModel):
    tag_text: str


class VectorizeRequest(BaseModel):
    chunk_ids: List[int]
    document_tags: Optional[List[str]] = None


class VectorizeResponse(BaseModel):
    success_count: int
    failed_count: int
    skipped_count: int
    success_ids: List[int]
    failed_ids: List[int]
    skipped_ids: List[int]


class SingleVectorizeRequest(BaseModel):
    document_tags: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    chunk_id: int
    milvus_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class TagStatsResponse(BaseModel):
    name: str
    type: str
    count: int
    chunk_ids: List[int]
    document_count: int = 0  # 文档级标签关联的文档数量


class TagDeleteRequest(BaseModel):
    tag_name: str


class TagRenameRequest(BaseModel):
    old_name: str
    new_name: str


class TagMergeRequest(BaseModel):
    source_tags: List[str]
    target_tag: str


class TagCreateRequest(BaseModel):
    tag_name: str


class SystemTagResponse(BaseModel):
    id: int
    tag_name: str
    description: Optional[str]
    created_at: str
    created_by: str
    is_active: bool
    usage_count: int


class SystemTagCreateRequest(BaseModel):
    tag_name: str
    description: Optional[str] = None


class SystemTagConvertRequest(BaseModel):
    tag_name: str
    description: Optional[str] = None


# ==================== Helper Functions ====================

def get_output_path(filename: str) -> Path:
    """获取文档的输出路径"""
    stem = Path(filename).stem
    return OUTPUT_DIR / f"{stem}_final_chunks.json"


def get_source_file_type(filename: str) -> Optional[str]:
    """从文件名或数据库推断源文件类型"""
    # 对于 _converted.md 格式，查询数据库获取真实文件类型
    if filename.endswith('_converted.md'):
        try:
            # 使用统一的数据库连接
            with get_connection() as conn:
                row = conn.execute("""
                    SELECT file_type FROM file_uploads
                    WHERE converted_md_filename = ?
                    LIMIT 1
                """, (filename,)).fetchone()

                if row:
                    content_type = row['file_type'].lower()
                    # 根据 MIME 类型映射到文件类型
                    if 'pdf' in content_type:
                        return 'PDF'
                    elif 'word' in content_type or 'msword' in content_type or 'wordprocessing' in content_type:
                        return 'Word'
                    elif 'powerpoint' in content_type or 'presentation' in content_type:
                        return 'PPT'
                    elif 'excel' in content_type or 'spreadsheet' in content_type:
                        return 'Excel'
                    elif 'jpeg' in content_type or 'jpg' in content_type:
                        return 'JPEG'
                    elif 'png' in content_type:
                        return 'PNG'
                    elif 'image' in content_type:
                        return 'Image'
                    elif 'markdown' in content_type:
                        return 'Markdown'
        except Exception as e:
            # 记录错误但不影响主流程
            print(f"查询文件类型失败: {e}")
            pass

    # 对于其他格式，从文件名推断
    import re
    patterns = [
        (r'\.pdf-[a-f0-9-]+\.md$', 'PDF'),
        (r'\.docx?-[a-f0-9-]+\.md$', 'Word'),
        (r'\.pptx?-[a-f0-9-]+\.md$', 'PPT'),
        (r'\.xlsx?-[a-f0-9-]+\.md$', 'Excel'),
        (r'\.jpe?g-[a-f0-9-]+\.md$', 'JPEG'),
        (r'\.png-[a-f0-9-]+\.md$', 'PNG'),
        (r'\.pdf\.md$', 'PDF'),
        (r'\.docx?\.md$', 'Word'),
        (r'\.pptx?\.md$', 'PPT'),
        (r'\.xlsx?\.md$', 'Excel'),
        (r'\.jpe?g\.md$', 'JPEG'),
        (r'\.png\.md$', 'PNG'),
    ]

    for pattern, file_type in patterns:
        if re.search(pattern, filename, re.IGNORECASE):
            return file_type

    return None


def check_document_status(filename: str) -> Dict[str, Any]:
    """检查文档处理状态（同步版本，保留兼容性）"""
    output_path = get_output_path(filename)
    source_file_type = get_source_file_type(filename)

    if filename in processing_tasks:
        task_status = processing_tasks[filename]
        if task_status["status"] == "processing":
            return {
                "filename": filename,
                "status": "processing",
                "output_path": None,
                "source_file_type": source_file_type
            }
        elif task_status["status"] == "error":
            return {
                "filename": filename,
                "status": "error",
                "error": task_status.get("error"),
                "output_path": None,
                "source_file_type": source_file_type
            }

    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed_at = data.get("metadata", {}).get("processed_at")
            return {
                "filename": filename,
                "status": "processed",
                "output_path": f"./output/{output_path.name}",
                "processed_at": processed_at,
                "source_file_type": source_file_type
            }
        except Exception as e:
            return {
                "filename": filename,
                "status": "error",
                "error": f"读取输出文件失败: {str(e)}",
                "output_path": None,
                "source_file_type": source_file_type
            }

    return {
        "filename": filename,
        "status": "not_processed",
        "output_path": None,
        "source_file_type": source_file_type
    }


async def check_document_status_async(filename: str, include_stats: bool = False) -> Dict[str, Any]:
    """
    异步检查文档处理状态（带缓存）
    
    Args:
        filename: 文档文件名
        include_stats: 是否包含统计信息（chunk_count, total_tokens, tags）
    """
    # 检查缓存
    cache_key = f"{filename}:{include_stats}"
    if cache_key in _document_cache:
        cached_data, cached_time = _document_cache[cache_key]
        if datetime.now() - cached_time < _cache_ttl:
            return cached_data
    
    output_path = get_output_path(filename)
    source_file_type = get_source_file_type(filename)
    
    # 检查是否正在处理
    if filename in processing_tasks:
        task_status = processing_tasks[filename]
        if task_status["status"] == "processing":
            result = {
                "filename": filename,
                "status": "processing",
                "output_path": None,
                "source_file_type": source_file_type
            }
            _document_cache[cache_key] = (result, datetime.now())
            return result
        elif task_status["status"] == "error":
            result = {
                "filename": filename,
                "status": "error",
                "error": task_status.get("error"),
                "output_path": None,
                "source_file_type": source_file_type
            }
            _document_cache[cache_key] = (result, datetime.now())
            return result
    
    # 检查输出文件是否存在
    if output_path.exists():
        try:
            # 使用异步文件读取
            async with aiofiles.open(output_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                processed_at = data.get("metadata", {}).get("processed_at")
            
            result = {
                "filename": filename,
                "status": "processed",
                "output_path": f"./output/{output_path.name}",
                "processed_at": processed_at,
                "source_file_type": source_file_type
            }
            
            # 如果需要统计信息，从数据库获取
            if include_stats:
                try:
                    doc = get_document_by_filename(filename)
                    if doc:
                        chunks = get_chunks_by_document(doc['id'])
                        result['chunk_count'] = len(chunks)
                        result['total_tokens'] = sum(c.get('token_count', 0) for c in chunks)
                        result['updated_at'] = max(
                            (c.get('updated_at') for c in chunks if c.get('updated_at')),
                            default=processed_at
                        )
                        # 获取标签
                        tags = get_tags_by_filename(filename)
                        result['tags'] = tags or []
                    else:
                        result['chunk_count'] = 0
                        result['total_tokens'] = 0
                        result['tags'] = []
                except Exception as e:
                    print(f"获取文档 {filename} 统计信息失败: {e}")
                    result['chunk_count'] = 0
                    result['total_tokens'] = 0
                    result['tags'] = []
            
            _document_cache[cache_key] = (result, datetime.now())
            return result
            
        except Exception as e:
            result = {
                "filename": filename,
                "status": "error",
                "error": f"读取输出文件失败: {str(e)}",
                "output_path": None,
                "source_file_type": source_file_type
            }
            _document_cache[cache_key] = (result, datetime.now())
            return result
    
    result = {
        "filename": filename,
        "status": "not_processed",
        "output_path": None,
        "source_file_type": source_file_type
    }
    _document_cache[cache_key] = (result, datetime.now())
    return result


def clear_document_cache(filename: Optional[str] = None):
    """清除文档缓存"""
    global _documents_list_cache
    
    # 清除全局列表缓存（任何文档变更都会影响列表）
    _documents_list_cache.clear()
    
    if filename:
        # 清除特定文档的所有缓存
        keys_to_remove = [k for k in _document_cache.keys() if k.startswith(f"{filename}:")]
        for key in keys_to_remove:
            del _document_cache[key]
    else:
        # 清除所有缓存
        _document_cache.clear()
    
    print(f"🔄 已清除文档缓存{f': {filename}' if filename else ''}")


def get_batch_document_stats(filenames: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    批量获取文档统计信息（一次 SQL 查询）
    
    Returns:
        {filename: {chunk_count, total_tokens, updated_at, tags}}
    """
    if not filenames:
        return {}
    
    stats_map = {}
    
    try:
        with get_connection() as conn:
            # 批量查询文档统计信息
            placeholders = ','.join('?' * len(filenames))
            query = f"""
            SELECT 
                d.filename,
                d.id as document_id,
                COUNT(c.id) as chunk_count,
                COALESCE(SUM(c.token_count), 0) as total_tokens,
                MAX(c.updated_at) as updated_at
            FROM documents d
            LEFT JOIN document_chunks c ON d.id = c.document_id
            WHERE d.filename IN ({placeholders})
            GROUP BY d.id, d.filename
            """
            
            rows = conn.execute(query, filenames).fetchall()
            
            for row in rows:
                stats_map[row['filename']] = {
                    'document_id': row['document_id'],
                    'chunk_count': row['chunk_count'] or 0,
                    'total_tokens': row['total_tokens'] or 0,
                    'updated_at': row['updated_at']
                }
            
            # 批量获取标签（一次查询）
            tags_query = f"""
            SELECT d.filename, dt.tag_text as tag
            FROM documents d
            JOIN document_tags dt ON d.id = dt.document_id
            WHERE d.filename IN ({placeholders})
            ORDER BY d.filename, dt.tag_text
            """
            
            tag_rows = conn.execute(tags_query, filenames).fetchall()
            
            # 组织标签数据
            for row in tag_rows:
                filename = row['filename']
                if filename in stats_map:
                    if 'tags' not in stats_map[filename]:
                        stats_map[filename]['tags'] = []
                    stats_map[filename]['tags'].append(row['tag'])
            
            # 确保所有文档都有 tags 字段
            for filename in stats_map:
                if 'tags' not in stats_map[filename]:
                    stats_map[filename]['tags'] = []
                    
    except Exception as e:
        print(f"批量获取文档统计失败: {e}")
        import traceback
        traceback.print_exc()
    
    return stats_map


async def process_document_task(filename: str):
    """后台任务：处理文档（异步子进程，带超时保护）"""
    md_path = ALL_MD_DIR / filename
    output_path = get_output_path(filename)

    processing_tasks[filename] = {
        "status": "processing",
        "started_at": datetime.now().isoformat()
    }

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 使用异步子进程替代同步subprocess.run
        process = await asyncio.create_subprocess_exec(
            "uv", "run", "main.py",
            str(md_path.resolve()),
            "-o", str(OUTPUT_DIR.resolve()),
            cwd=IKN_PLUS_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # 设置更短的超时时间（5分钟），避免长时间阻塞
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300.0  # 5分钟超时
            )
            
            if process.returncode == 0:
                processing_tasks[filename] = {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "output_path": str(output_path)
                }

                try:
                    import_json_to_db(output_path, filename)
                except Exception as e:
                    print(f"Warning: Failed to import to DB: {e}")
                
                # 清除缓存（文档已处理完成）
                clear_document_cache(filename)
            else:
                error_msg = stderr.decode('utf-8') if stderr else stdout.decode('utf-8') if stdout else "Unknown error"
                processing_tasks[filename] = {
                    "status": "error",
                    "error": error_msg,
                    "completed_at": datetime.now().isoformat()
                }
                # 清除缓存（处理失败也需要更新列表）
                clear_document_cache(filename)
                
        except asyncio.TimeoutError:
            # 超时后强制终止进程
            try:
                process.kill()
                await process.wait()
            except:
                pass
            
            processing_tasks[filename] = {
                "status": "error",
                "error": "处理超时（超过5分钟）",
                "completed_at": datetime.now().isoformat()
            }
            # 清除缓存
            clear_document_cache(filename)

    except Exception as e:
        processing_tasks[filename] = {
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        }
        # 清除缓存
        clear_document_cache(filename)


# ==================== 文档管理 API ====================

@router.get("/api/documents")
async def list_documents(
    limit: Optional[int] = Query(None, description="每页数量，默认返回所有"),
    offset: int = Query(0, description="跳过的文档数量"),
    include_stats: bool = Query(False, description="是否包含统计信息（chunk数、token数、标签等）"),
    sort: str = Query("newest", description="排序方式: newest(从新到旧) | oldest(从旧到新)")
):
    """
    列出文档及其状态
    
    支持分页、排序和可选的统计信息加载，使用批量查询和全局缓存大幅提升性能
    支持同时缓存多种排序方式，切换排序无需重新查询
    """
    global _documents_list_cache
    
    if not ALL_MD_DIR.exists():
        raise HTTPException(status_code=500, detail=f"文档目录不存在: {ALL_MD_DIR}")

    # 检查缓存（包含排序参数）
    cache_key = f"full_list:{include_stats}:{sort}"
    use_global_cache = include_stats and limit is None and offset == 0
    
    if use_global_cache and cache_key in _documents_list_cache:
        cached_data, cached_time = _documents_list_cache[cache_key]
        # 检查缓存是否过期
        if datetime.now() - cached_time < _cache_ttl:
            print(f"✅ 使用缓存的文档列表 (排序: {sort})")
            return cached_data
        else:
            # 缓存过期，删除
            del _documents_list_cache[cache_key]
            print(f"⏱️  缓存已过期，重新查询 (排序: {sort})")

    # 获取所有 md 文件并排序
    # sort="newest": 从新到旧（默认，文件修改时间倒序）
    # sort="oldest": 从旧到新（文件修改时间正序）
    reverse_order = (sort != "oldest")
    all_files = sorted(ALL_MD_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=reverse_order)
    total_count = len(all_files)
    
    # 应用分页
    if limit is not None:
        paginated_files = all_files[offset:offset + limit]
    else:
        paginated_files = all_files[offset:]
    
    # 构建文档列表
    documents = []
    
    if include_stats:
        # 使用批量查询（性能提升关键！）
        print(f"🔍 批量查询 {len(paginated_files)} 个文档的统计信息...")
        
        # 1. 并发读取文档基本状态
        basic_status_tasks = []
        filenames_to_query = []
        
        for file in paginated_files:
            output_path = get_output_path(file.name)
            source_file_type = get_source_file_type(file.name)
            
            # 检查处理状态
            if file.name in processing_tasks:
                task_status = processing_tasks[file.name]
                if task_status["status"] == "processing":
                    documents.append({
                        "filename": file.name,
                        "status": "processing",
                        "output_path": None,
                        "source_file_type": source_file_type,
                        "chunk_count": 0,
                        "total_tokens": 0,
                        "tags": []
                    })
                    continue
                elif task_status["status"] == "error":
                    documents.append({
                        "filename": file.name,
                        "status": "error",
                        "error": task_status.get("error"),
                        "output_path": None,
                        "source_file_type": source_file_type,
                        "chunk_count": 0,
                        "total_tokens": 0,
                        "tags": []
                    })
                    continue
            
            # 检查是否已处理
            if output_path.exists():
                basic_status_tasks.append((file.name, output_path, source_file_type, "processed"))
                filenames_to_query.append(file.name)
            else:
                documents.append({
                    "filename": file.name,
                    "status": "not_processed",
                    "output_path": None,
                    "source_file_type": source_file_type,
                    "chunk_count": 0,
                    "total_tokens": 0,
                    "tags": []
                })
        
        # 2. 批量获取统计信息（一次数据库查询！）
        stats_map = get_batch_document_stats(filenames_to_query)
        
        # 3. 异步读取 JSON 文件获取 processed_at
        async def read_processed_at(filename: str, output_path: Path):
            try:
                async with aiofiles.open(output_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    return filename, data.get("metadata", {}).get("processed_at")
            except:
                return filename, None
        
        processed_at_map = {}
        if basic_status_tasks:
            processed_at_results = await asyncio.gather(
                *[read_processed_at(fn, op) for fn, op, _, _ in basic_status_tasks]
            )
            processed_at_map = dict(processed_at_results)
        
        # 4. 组装结果
        for filename, output_path, source_file_type, status in basic_status_tasks:
            stats = stats_map.get(filename, {})
            documents.append({
                "filename": filename,
                "status": status,
                "output_path": f"./output/{output_path.name}",
                "processed_at": processed_at_map.get(filename),
                "source_file_type": source_file_type,
                "chunk_count": stats.get('chunk_count', 0),
                "total_tokens": stats.get('total_tokens', 0),
                "updated_at": stats.get('updated_at'),
                "tags": stats.get('tags', [])
            })
    else:
        # 不需要统计信息时，只读取基本状态（快速模式）
        for file in paginated_files:
            status_info = check_document_status(file.name)
            documents.append(status_info)
    
    result = {
        "documents": documents,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + len(documents)) < total_count
    }
    
    # 缓存完整列表结果（支持多种排序同时缓存）
    if use_global_cache:
        _documents_list_cache[cache_key] = (result, datetime.now())
        print(f"💾 已缓存文档列表 ({len(documents)} 个文档, 排序: {sort})")
        print(f"📊 当前缓存的排序方式: {list(k.split(':')[-1] for k in _documents_list_cache.keys())}")
    
    return result


@router.get("/api/documents/{filename}/status", response_model=Document)
async def get_document_status(filename: str):
    """获取单个文档的状态"""
    md_path = ALL_MD_DIR / filename
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"文档不存在: {filename}")

    status_info = check_document_status(filename)
    return Document(**status_info)


@router.post("/api/documents/{filename}/process", response_model=Document)
async def process_document(filename: str, background_tasks: BackgroundTasks):
    """触发文档处理"""
    md_path = ALL_MD_DIR / filename
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"文档不存在: {filename}")

    status_info = check_document_status(filename)
    if status_info["status"] == "processing":
        return Document(**status_info)

    # 清除该文档的缓存
    clear_document_cache(filename)
    background_tasks.add_task(process_document_task, filename)

    return Document(
        filename=filename,
        status="processing",
        output_path=None
    )


@router.delete("/api/documents/{filename}/output")
async def delete_output(filename: str):
    """
    删除选项1：删除切片数据
    删除切片文件数据、数据库记录、向量数据（保留.md文件和原始上传文件）
    """
    md_path = ALL_MD_DIR / filename
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"文档不存在: {filename}")

    try:
        # 1. 删除向量库中的数据
        try:
            doc = get_document_by_filename(filename)
            if doc:
                chunks = get_chunks_by_document(doc['id'])
                if chunks:
                    chunk_ids = [chunk['id'] for chunk in chunks]
                    manager = get_vectorization_manager()
                    manager.vector_store.delete_by_chunk_db_ids(chunk_ids)
        except Exception as e:
            print(f"删除向量数据时出错: {e}")

        # 2. 删除数据库记录
        try:
            with get_connection() as conn:
                # 删除文档级标签
                conn.execute("DELETE FROM document_tags WHERE document_id IN (SELECT id FROM documents WHERE filename = ?)", (filename,))
                # 删除 chunk 日志
                conn.execute("DELETE FROM chunk_logs WHERE document_id IN (SELECT id FROM documents WHERE filename = ?)", (filename,))
                # 删除 chunks
                conn.execute("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE filename = ?)", (filename,))
                # 删除文档记录
                conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
                conn.commit()
        except Exception as e:
            print(f"删除数据库记录时出错: {e}")

        # 3. 删除输出文件
        output_path = get_output_path(filename)
        if output_path.exists():
            output_path.unlink()

        # 4. 清理处理任务状态
        if filename in processing_tasks:
            del processing_tasks[filename]

        # 5. 清除缓存
        clear_document_cache(filename)

        return {
            "message": f"已删除切片数据: {filename}",
            "deleted_items": {
                "output_file": str(output_path) if output_path.exists() else None,
                "database_records": True,
                "vectors": True
            }
        }

    except Exception as e:
        import traceback
        error_detail = f"删除切片数据失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"删除切片数据失败: {str(e)}")


@router.delete("/api/documents/{filename}")
async def delete_md_file(filename: str):
    """
    删除选项2：删除.md文件
    删除.md文件及其切片文件数据、数据库记录、向量数据（保留原始上传文件）
    """
    md_path = ALL_MD_DIR / filename

    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"文档不存在: {filename}")

    try:
        # 1. 删除向量库中的数据
        try:
            doc = get_document_by_filename(filename)
            if doc:
                chunks = get_chunks_by_document(doc['id'])
                if chunks:
                    chunk_ids = [chunk['id'] for chunk in chunks]
                    manager = get_vectorization_manager()
                    manager.vector_store.delete_by_chunk_db_ids(chunk_ids)
        except Exception as e:
            print(f"删除向量数据时出错: {e}")

        # 2. 删除数据库记录
        try:
            with get_connection() as conn:
                # 删除文档级标签
                conn.execute("DELETE FROM document_tags WHERE document_id IN (SELECT id FROM documents WHERE filename = ?)", (filename,))
                # 删除 chunk 日志
                conn.execute("DELETE FROM chunk_logs WHERE document_id IN (SELECT id FROM documents WHERE filename = ?)", (filename,))
                # 删除 chunks
                conn.execute("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE filename = ?)", (filename,))
                # 删除文档记录
                conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
                conn.commit()
        except Exception as e:
            print(f"删除数据库记录时出错: {e}")

        # 3. 更新文件上传记录，清除MD文件关联
        upload_record_updated = False
        converted_md_path = None
        try:
            with get_connection() as conn:
                upload_record = conn.execute(
                    """
                    SELECT id, converted_md_path
                    FROM file_uploads
                    WHERE converted_md_filename = ?
                    LIMIT 1
                    """,
                    (filename,)
                ).fetchone()

                if upload_record:
                    converted_md_path = upload_record["converted_md_path"]
                    conn.execute(
                        """
                        UPDATE file_uploads
                        SET converted_md_filename = NULL,
                            converted_md_path = NULL,
                            status = 'pending',
                            mineru_task_id = NULL,
                            conversion_started_at = NULL,
                            conversion_completed_at = NULL,
                            error_message = NULL,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (upload_record["id"],)
                    )
                    conn.commit()
                    upload_record_updated = True
        except Exception as e:
            print(f"更新文件上传记录时出错: {e}")

        # 4. 删除转换文件
        converted_file_deleted_path = None
        try:
            candidate_paths = []
            if converted_md_path:
                candidate_paths.append(Path(converted_md_path))
            candidate_paths.append(CONVERTED_DIR / filename)

            for candidate in candidate_paths:
                if not candidate:
                    continue
                if candidate.exists():
                    candidate.unlink()
                    converted_file_deleted_path = str(candidate)
                    break
        except Exception as e:
            print(f"删除转换文件时出错: {e}")

        # 5. 删除输出文件
        output_path = get_output_path(filename)
        output_file_deleted = False
        if output_path.exists():
            output_path.unlink()
            output_file_deleted = True

        # 6. 清理处理任务状态
        if filename in processing_tasks:
            del processing_tasks[filename]

        # 7. 删除.md文件
        md_path.unlink()

        return {
            "message": f"已删除.md文件及相关数据: {filename}",
            "deleted_items": {
                "md_file": str(md_path),
                "converted_file": converted_file_deleted_path,
                "output_file": str(output_path) if output_file_deleted else None,
                "database_records": True,
                "vectors": True,
                "file_upload_record_updated": upload_record_updated
            }
        }

    except Exception as e:
        import traceback
        error_detail = f"删除.md文件失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"删除.md文件失败: {str(e)}")


@router.delete("/api/documents/{filename}/complete")
async def delete_completely(filename: str):
    """
    删除选项3：彻底删除
    删除所有内容：原始上传文件、.md文件、切片数据、数据库记录、向量数据
    """
    md_path = ALL_MD_DIR / filename

    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"文档不存在: {filename}")

    try:
        deleted_items = {}

        # 1. 删除向量库中的数据
        try:
            doc = get_document_by_filename(filename)
            if doc:
                chunks = get_chunks_by_document(doc['id'])
                if chunks:
                    chunk_ids = [chunk['id'] for chunk in chunks]
                    manager = get_vectorization_manager()
                    manager.vector_store.delete_by_chunk_db_ids(chunk_ids)
            deleted_items["vectors"] = True
        except Exception as e:
            print(f"删除向量数据时出错: {e}")
            deleted_items["vectors"] = False

        # 2. 查找并删除原始上传文件
        try:
            # 使用统一的数据库连接
            with get_connection() as conn:
                row = conn.execute("""
                    SELECT upload_path, converted_md_path
                    FROM file_uploads
                    WHERE converted_md_filename = ?
                    LIMIT 1
                """, (filename,)).fetchone()

                if row:
                    # 删除原始上传文件
                    upload_path = Path(row['upload_path'])
                    if upload_path.exists():
                        upload_path.unlink()
                        deleted_items["original_file"] = str(upload_path)

                    # 删除 CONVERTED_DIR 中的转换文件
                    if row['converted_md_path']:
                        converted_path = Path(row['converted_md_path'])
                        if converted_path.exists():
                            converted_path.unlink()
                            deleted_items["converted_file"] = str(converted_path)

                    # 删除文件上传记录
                    conn.execute("DELETE FROM file_uploads WHERE converted_md_filename = ?", (filename,))
                    conn.commit()

                conn.close()
        except Exception as e:
            print(f"删除原始文件时出错: {e}")

        # 3. 删除数据库记录
        try:
            with get_connection() as conn:
                # 删除文档级标签
                conn.execute("DELETE FROM document_tags WHERE document_id IN (SELECT id FROM documents WHERE filename = ?)", (filename,))
                # 删除 chunk 日志
                conn.execute("DELETE FROM chunk_logs WHERE document_id IN (SELECT id FROM documents WHERE filename = ?)", (filename,))
                # 删除 chunks
                conn.execute("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE filename = ?)", (filename,))
                # 删除文档记录
                conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
                conn.commit()
            deleted_items["database_records"] = True
        except Exception as e:
            print(f"删除数据库记录时出错: {e}")
            deleted_items["database_records"] = False

        # 4. 删除输出文件
        output_path = get_output_path(filename)
        if output_path.exists():
            output_path.unlink()
            deleted_items["output_file"] = str(output_path)

        # 5. 清理处理任务状态
        if filename in processing_tasks:
            del processing_tasks[filename]

        # 6. 删除.md文件
        md_path.unlink()
        deleted_items["md_file"] = str(md_path)

        # 7. 清除缓存
        clear_document_cache(filename)

        return {
            "message": f"已彻底删除所有相关文件和数据: {filename}",
            "deleted_items": deleted_items
        }

    except Exception as e:
        import traceback
        error_detail = f"彻底删除失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"彻底删除失败: {str(e)}")


@router.get("/api/documents/{filename}/chunks")
async def get_document_chunks(filename: str):
    """获取文档的所有chunks"""
    doc = get_document_by_filename(filename)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in database")

    chunks = get_chunks_by_document(doc['id'])

    return {
        "metadata": {
            "source_file": doc['filename'],
            "total_chunks": len(chunks),
            "document_id": doc['id']
        },
        "chunks": chunks
    }


# ==================== Chunk 管理 API ====================

@router.patch("/api/chunks/{chunk_id}")
async def update_chunk_endpoint(chunk_id: int, request: ChunkUpdateRequest):
    """更新chunk内容并记录版本"""
    chunk = get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    old_data = {
        "edited_content": chunk.get("edited_content") or chunk.get("content"),
        "status": chunk.get("status"),
        "content_tags": chunk.get("content_tags", []),
        "user_tag": chunk.get("user_tag")
    }

    # 处理 user_tag：如果传递了 null，表示要清空，转换为空字符串
    user_tag_to_update = request.user_tag
    if 'user_tag' in request.model_dump(exclude_unset=True) and request.user_tag is None:
        user_tag_to_update = ""  # null 转换为空字符串，表示清空

    update_chunk(
        chunk_id=chunk_id,
        edited_content=request.edited_content,
        status=request.status,
        content_tags=request.content_tags,
        user_tag=user_tag_to_update,
        last_editor_id=request.editor_id
    )

    changes = {}
    if request.edited_content and request.edited_content != old_data["edited_content"]:
        changes["edited_content"] = {
            "before": old_data["edited_content"],
            "after": request.edited_content
        }

    if request.status is not None and request.status != old_data["status"]:
        status_names = {-1: "废弃", 0: "初始", 1: "已确认", 2: "已向量化"}
        changes["status"] = {
            "before": old_data["status"],
            "after": request.status,
            "before_name": status_names.get(old_data["status"], "未知"),
            "after_name": status_names.get(request.status, "未知")
        }

    if request.content_tags is not None and request.content_tags != old_data["content_tags"]:
        changes["content_tags"] = {
            "before": old_data["content_tags"],
            "after": request.content_tags
        }

    if request.user_tag and request.user_tag != old_data["user_tag"]:
        changes["user_tag"] = {
            "before": old_data["user_tag"],
            "after": request.user_tag
        }

    if changes:
        action = "status_change" if "status" in changes and len(changes) == 1 else "update"
        message = f"更新了chunk" if action == "update" else f"状态变更"

        insert_log(
            document_id=chunk["document_id"],
            chunk_id=chunk_id,
            action=action,
            message=message,
            user_id=request.editor_id,
            payload={"changes": changes, "timestamp": datetime.utcnow().isoformat()}
        )

    updated_chunk = get_chunk(chunk_id)
    return updated_chunk


@router.get("/api/chunks/{chunk_id}/logs", response_model=List[ChunkLogEntry])
async def get_chunk_logs_endpoint(chunk_id: int, limit: int = 50):
    """获取chunk的版本历史"""
    chunk = get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    logs = get_chunk_logs(chunk_id, limit=limit)

    return [
        ChunkLogEntry(
            id=log["id"],
            action=log["action"],
            message=log["message"],
            created_at=log["created_at"],
            user_id=log["user_id"],
            payload=log["payload"]
        )
        for log in logs
    ]


# ==================== 标签管理 API ====================

@router.get("/api/documents/{filename}/tags")
async def get_document_tags_endpoint(filename: str):
    """获取文档的所有标签"""
    tags = get_tags_by_filename(filename)
    return {"filename": filename, "tags": tags}


@router.post("/api/documents/{filename}/tags")
async def add_document_tag_endpoint(filename: str, request: TagRequest):
    """添加文档标签"""
    success = add_tag_by_filename(filename, request.tag_text)
    if success:
        return {"message": "标签添加成功", "tag": request.tag_text}
    else:
        raise HTTPException(status_code=400, detail="标签已存在或文档不存在")


@router.delete("/api/documents/{filename}/tags/{tag_text}")
async def remove_document_tag_endpoint(filename: str, tag_text: str):
    """删除文档标签"""
    success = remove_tag_by_filename(filename, tag_text)
    if success:
        return {"message": "标签删除成功", "tag": tag_text}
    else:
        raise HTTPException(status_code=404, detail="标签不存在或文档不存在")


@router.get("/api/chunks/tags")
async def get_all_chunk_tags():
    """获取所有标签（包括 chunk 标签和文档级标签）"""
    try:
        with get_connection() as conn:
            # 收集 chunk 的 user_tag
            user_tags = conn.execute("""
                SELECT DISTINCT user_tag
                FROM document_chunks
                WHERE user_tag IS NOT NULL AND user_tag != ''
            """).fetchall()

            # 收集 chunk 的 content_tags
            content_tags_rows = conn.execute("""
                SELECT DISTINCT content_tags
                FROM document_chunks
                WHERE content_tags IS NOT NULL AND content_tags != '[]'
            """).fetchall()

            # 收集文档级标签
            document_tags = conn.execute("""
                SELECT DISTINCT tag_text
                FROM document_tags
            """).fetchall()

        all_tags = set()

        # 处理 user_tags
        for row in user_tags:
            if row['user_tag']:
                all_tags.add(row['user_tag'])

        # 处理 content_tags
        for row in content_tags_rows:
            try:
                tags = json.loads(row['content_tags'])
                if isinstance(tags, list):
                    for tag in tags:
                        clean_tag = tag.lstrip('@') if isinstance(tag, str) else tag
                        if clean_tag:
                            all_tags.add(clean_tag)
            except:
                continue

        # 处理文档级标签
        for row in document_tags:
            if row['tag_text']:
                all_tags.add(row['tag_text'].strip())

        return {"tags": sorted(list(all_tags))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取标签失败: {str(e)}")


@router.get("/api/tags/all", response_model=List[TagStatsResponse])
async def get_all_tags_stats():
    """获取所有标签及统计信息"""
    try:
        tags = get_all_tags_with_stats()
        return tags
    except Exception as e:
        import traceback
        error_detail = f"获取标签统计失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"获取标签统计失败: {str(e)}")


@router.post("/api/tags/delete")
async def delete_tag(request: TagDeleteRequest):
    """删除标签（从所有 chunks 中删除）"""
    try:
        tag_name = request.tag_name.strip()
        if not tag_name:
            raise HTTPException(status_code=400, detail="标签名称不能为空")

        affected_count = delete_tag_from_all_chunks(tag_name)

        return {
            "affected_chunks": affected_count,
            "message": f"已从 {affected_count} 个切片中删除标签 '{tag_name}'"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"删除标签失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"删除标签失败: {str(e)}")


@router.post("/api/tags/rename")
async def rename_tag(request: TagRenameRequest):
    """重命名标签"""
    try:
        old_name = request.old_name.strip()
        new_name = request.new_name.strip()

        if not old_name or not new_name:
            raise HTTPException(status_code=400, detail="标签名称不能为空")

        if old_name == new_name:
            raise HTTPException(status_code=400, detail="新旧标签名称相同")

        affected_count = rename_tag_in_all_chunks(old_name, new_name)

        return {
            "affected_chunks": affected_count,
            "message": f"已将 {affected_count} 个切片中的标签 '{old_name}' 重命名为 '{new_name}'"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"重命名标签失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"重命名标签失败: {str(e)}")


@router.post("/api/tags/merge")
async def merge_tags(request: TagMergeRequest):
    """合并标签"""
    try:
        source_tags = [tag.strip() for tag in request.source_tags if tag.strip()]
        target_tag = request.target_tag.strip()

        if not source_tags:
            raise HTTPException(status_code=400, detail="源标签列表不能为空")

        if not target_tag:
            raise HTTPException(status_code=400, detail="目标标签不能为空")

        if len(source_tags) < 2:
            raise HTTPException(status_code=400, detail="至少需要 2 个源标签才能合并")

        result = merge_tags_in_all_chunks(source_tags, target_tag)

        return {
            "affected_chunks": result['affected_chunks'],
            "merged_count": result['merged_count'],
            "message": f"已将 {result['merged_count']} 个标签合并为 '{target_tag}'，影响 {result['affected_chunks']} 个切片"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"合并标签失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"合并标签失败: {str(e)}")


@router.post("/api/tags/create")
async def create_tag(request: TagCreateRequest):
    """创建新标签"""
    try:
        tag_name = request.tag_name.strip()
        if not tag_name:
            raise HTTPException(status_code=400, detail="标签名称不能为空")

        all_tags = get_all_tags_with_stats()
        if any(tag['name'] == tag_name for tag in all_tags):
            raise HTTPException(status_code=400, detail=f"标签 '{tag_name}' 已存在")

        placeholder_doc = get_document_by_filename("__global_tags__")
        if not placeholder_doc:
            doc_id = create_document(
                filename="__global_tags__",
                source_path="system",
                status="completed"
            )
        else:
            doc_id = placeholder_doc['id']

        existing_chunks = get_chunks_by_document(doc_id)
        next_chunk_id = max([c['chunk_id'] for c in existing_chunks], default=-1) + 1

        create_chunk(
            document_id=doc_id,
            chunk_id=next_chunk_id,
            content=f"全局标签定义: {tag_name}",
            token_start=0,
            token_end=0,
            token_count=0,
            user_tag=None,
            content_tags=[f"@{tag_name}"],
            is_atomic=False,
            atomic_type=None,
            status=-1
        )

        return {
            "tag_name": tag_name,
            "message": f"✅ 已创建新标签 '{tag_name}'"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"创建标签失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"创建标签失败: {str(e)}")


# ==================== 文档和切块导航 API ====================

@router.get("/api/view/document/{filename}/chunk/{chunk_db_id}")
async def view_document_chunk(filename: str, chunk_db_id: int):
    """
    文档切块导航接口
    用于从外部链接跳转到文档并定位到特定切块

    参数：
        filename: 文档文件名
        chunk_db_id: 切块的数据库主键ID（全局唯一）

    返回：HTTP 302 重定向到前端文档界面
    """
    from fastapi.responses import RedirectResponse
    from urllib.parse import quote

    try:
        # 验证文档是否存在
        doc_info = await get_document_status(filename)
        if not doc_info or doc_info.status != 'processed':
            raise HTTPException(
                status_code=404,
                detail=f"文档 {filename} 不存在或未处理"
            )

        # 通过数据库主键ID查询切块
        chunk = get_chunk(chunk_db_id)

        if not chunk:
            raise HTTPException(
                status_code=404,
                detail=f"切块 ID {chunk_db_id} 不存在"
            )

        # 验证切块是否属于该文档
        if chunk.get('source_file') != filename:
            raise HTTPException(
                status_code=400,
                detail=f"切块 {chunk_db_id} 不属于文档 {filename}"
            )

        # 构造前端重定向URL（指向 hit-rag-ui）
        # 使用 chunk_db_id 作为参数，并 URL 编码文件名
        frontend_url = os.getenv("FRONTEND_UI_URL", "http://localhost:5173")
        encoded_filename = quote(filename)
        redirect_url = f"{frontend_url}/?doc={encoded_filename}&chunk={chunk_db_id}"

        # 返回 HTTP 302 重定向
        return RedirectResponse(
            url=redirect_url,
            status_code=302
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理请求失败: {str(e)}"
        )


@router.get("/api/chunk/{chunk_db_id}")
async def get_chunk_info(chunk_db_id: int):
    """
    获取切块详细信息
    用于获取单个切块的完整内容和元数据

    参数：
        chunk_db_id: 切块的数据库主键ID
    """
    try:
        chunk = get_chunk(chunk_db_id)

        if not chunk:
            raise HTTPException(
                status_code=404,
                detail=f"切块 ID {chunk_db_id} 不存在"
            )

        # 获取文档信息
        doc_info = get_document_by_filename(chunk['source_file'])

        return {
            "success": True,
            "chunk": {
                "id": chunk.get('id'),  # 数据库主键ID
                "chunk_sequence": chunk.get('chunk_id'),  # 文档内顺序编号
                "content": chunk.get('content'),
                "edited_content": chunk.get('edited_content'),
                "source_file": chunk.get('source_file'),
                "token_start": chunk.get('token_start'),
                "token_end": chunk.get('token_end'),
                "status": chunk.get('status'),
                "content_tags": chunk.get('content_tags', []),
                "user_tag": chunk.get('user_tag'),
                "token_count": chunk.get('token_count')
            },
            "document": {
                "filename": doc_info.get('filename') if doc_info else chunk.get('source_file'),
                "status": doc_info.get('status') if doc_info else 'unknown'
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取切块信息失败: {str(e)}"
        )


# ==================== 向量化 API ====================

@router.post("/api/chunks/vectorize/batch", response_model=VectorizeResponse)
async def vectorize_chunks_batch(request: VectorizeRequest):
    """批量向量化 chunks"""
    try:
        manager = get_vectorization_manager()

        chunks_to_vectorize = []
        for chunk_id in request.chunk_ids:
            chunk = get_chunk(chunk_id)
            if chunk:
                chunks_to_vectorize.append(chunk)

        if not chunks_to_vectorize:
            raise HTTPException(status_code=404, detail="没有找到有效的 chunks")

        result = manager.vectorize_chunks(chunks_to_vectorize, request.document_tags)

        for success_item in result.get('success', []):
            if success_item and 'chunk_id' in success_item and 'milvus_id' in success_item:
                update_chunk_milvus_id(success_item['chunk_id'], success_item['milvus_id'])

        return VectorizeResponse(
            success_count=len(result.get('success', [])),
            failed_count=len(result.get('failed', [])),
            skipped_count=len(result.get('skipped', [])),
            success_ids=[item.get('chunk_id') for item in result.get('success', []) if item and 'chunk_id' in item],
            failed_ids=[item.get('chunk_id') for item in result.get('failed', []) if item and 'chunk_id' in item],
            skipped_ids=[item.get('chunk_id') for item in result.get('skipped', []) if item and 'chunk_id' in item]
        )

    except Exception as e:
        import traceback
        error_detail = f"批量向量化失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"批量向量化失败: {str(e)}")


@router.post("/api/chunks/{chunk_id}/vectorize")
async def vectorize_single_chunk(chunk_id: int, request: SingleVectorizeRequest = None):
    """单个 chunk 向量化"""
    try:
        chunk = get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")

        if chunk.get('status') == -1:
            raise HTTPException(status_code=400, detail="废弃的 chunk 无法向量化")

        if chunk.get('status') == 2:
            raise HTTPException(status_code=400, detail="该 chunk 已经向量化")

        document_tags = request.document_tags if request else None

        manager = get_vectorization_manager()
        result = manager.vectorize_chunks([chunk], document_tags)

        if result['success']:
            success_item = result['success'][0]
            update_chunk_milvus_id(success_item['chunk_id'], success_item['milvus_id'])

            insert_log(
                document_id=chunk['document_id'],
                chunk_id=chunk_id,
                action="vectorize",
                message="向量化成功",
                user_id="system",
                payload={"milvus_id": success_item['milvus_id']}
            )

            return {"message": "向量化成功", "milvus_id": success_item['milvus_id']}

        elif result['failed']:
            raise HTTPException(status_code=500, detail=result['failed'][0].get('error', '向量化失败'))

        else:
            raise HTTPException(status_code=400, detail="Chunk 被跳过")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量化失败: {str(e)}")


@router.get("/api/vectorization/stats")
async def get_vectorization_stats_endpoint():
    """获取向量化统计信息"""
    try:
        stats = get_vectorization_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/api/chunks/vectorizable")
async def get_vectorizable_chunks_endpoint(limit: Optional[int] = None):
    """获取所有可向量化的 chunks"""
    try:
        chunks = get_vectorizable_chunks(limit=limit)
        return {"count": len(chunks), "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取可向量化 chunks 失败: {str(e)}")


@router.delete("/api/chunks/{chunk_id}/vectorize")
async def delete_chunk_from_vector(chunk_id: int):
    """从向量库删除 chunk"""
    try:
        chunk = get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")

        manager = get_vectorization_manager()
        manager.vector_store.delete_by_chunk_db_ids([chunk_id])

        with get_connection() as conn:
            old_milvus_id = chunk.get('milvus_id')
            conn.execute("""
                UPDATE document_chunks
                SET milvus_id = NULL, status = 0
                WHERE id = ?
            """, (chunk_id,))
            conn.commit()

        insert_log(
            document_id=chunk['document_id'],
            chunk_id=chunk_id,
            action="delete_vector",
            message="从向量库删除（包括所有历史向量）",
            user_id="system",
            payload={"chunk_db_id": chunk_id, "old_milvus_id": old_milvus_id}
        )

        return {"message": "从向量库删除成功", "chunk_db_id": chunk_id}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"删除失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.post("/api/chunks/search", response_model=List[SearchResult])
async def search_chunks(request: SearchRequest):
    """语义搜索 chunks"""
    try:
        manager = get_vectorization_manager()
        results = manager.search_chunks(
            query=request.query,
            k=request.top_k,
            filters=request.filters,
            with_score=True
        )

        search_results = []
        for result in results:
            metadata = result.get('metadata', {})
            chunk_db_id = metadata.get('chunk_db_id')

            if chunk_db_id:
                milvus_id = str(metadata.get('pk', ''))

                if not milvus_id:
                    chunk = get_chunk(chunk_db_id)
                    if chunk and chunk.get('milvus_id'):
                        milvus_id = chunk['milvus_id']
                    else:
                        continue

                search_results.append(SearchResult(
                    chunk_id=chunk_db_id,
                    milvus_id=milvus_id,
                    content=result.get('content', ''),
                    score=result.get('score', 0.0),
                    metadata=metadata
                ))

        return search_results

    except Exception as e:
        import traceback
        error_detail = f"搜索失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


# ============================================
# 系统标签管理 API
# ============================================

@router.get("/api/system-tags", response_model=List[SystemTagResponse])
async def get_all_system_tags():
    """获取所有系统标签及统计信息"""
    try:
        tags = get_system_tags_with_stats()
        return tags
    except Exception as e:
        import traceback
        error_detail = f"获取系统标签失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"获取系统标签失败: {str(e)}")


@router.post("/api/system-tags")
async def create_system_tag(request: SystemTagCreateRequest):
    """创建新的系统标签"""
    try:
        tag_name = request.tag_name.strip()
        if not tag_name:
            raise HTTPException(status_code=400, detail="标签名称不能为空")

        success = add_system_tag(tag_name, request.description, created_by='admin')
        if not success:
            raise HTTPException(status_code=400, detail=f"标签 '{tag_name}' 已存在")

        return {"message": f"系统标签 '{tag_name}' 创建成功"}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"创建系统标签失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"创建系统标签失败: {str(e)}")


@router.delete("/api/system-tags/{tag_name}")
async def delete_system_tag(tag_name: str):
    """删除系统标签（软删除）"""
    try:
        success = remove_system_tag(tag_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"系统标签 '{tag_name}' 不存在")

        return {"message": f"系统标签 '{tag_name}' 已删除"}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"删除系统标签失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"删除系统标签失败: {str(e)}")


@router.post("/api/system-tags/convert")
async def convert_user_tag_to_system_tag(request: SystemTagConvertRequest):
    """将用户标签转换为系统标签"""
    try:
        tag_name = request.tag_name.strip()
        if not tag_name:
            raise HTTPException(status_code=400, detail="标签名称不能为空")

        success = convert_user_tag_to_system(tag_name, request.description)
        if not success:
            raise HTTPException(status_code=400, detail=f"转换失败：标签 '{tag_name}' 不存在或已是系统标签")

        return {"message": f"标签 '{tag_name}' 已转换为系统标签"}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"转换标签失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"转换标签失败: {str(e)}")


@router.put("/api/system-tags/{old_name}")
async def rename_system_tag_endpoint(old_name: str, request: SystemTagCreateRequest):
    """重命名系统标签"""
    try:
        new_name = request.tag_name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="新标签名称不能为空")

        if old_name == new_name:
            raise HTTPException(status_code=400, detail="新旧标签名称相同")

        success = rename_system_tag(old_name, new_name)
        if not success:
            raise HTTPException(status_code=400, detail=f"重命名失败：标签 '{old_name}' 不存在或新标签名 '{new_name}' 已存在")

        return {"message": f"系统标签 '{old_name}' 已重命名为 '{new_name}'"}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"重命名系统标签失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"重命名系统标签失败: {str(e)}")
