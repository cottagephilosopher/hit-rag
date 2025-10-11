"""
文档和切片管理路由
提供文档列表、处理、chunk 更新、向量化、标签管理等功能
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
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
    get_chunks_by_document
)

from vector_db.vectorization_manager import VectorizationManager

router = APIRouter()

# ==================== 路径配置 ====================
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).parent.parent))
ALL_MD_DIR = Path(os.getenv("ALL_MD_DIR", BASE_DIR / "all-md"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "hit-rag-ui" / "public" / "output"))
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


# ==================== Helper Functions ====================

def get_output_path(filename: str) -> Path:
    """获取文档的输出路径"""
    stem = Path(filename).stem
    return OUTPUT_DIR / f"{stem}_final_chunks.json"


def check_document_status(filename: str) -> Dict[str, Any]:
    """检查文档处理状态"""
    output_path = get_output_path(filename)

    if filename in processing_tasks:
        task_status = processing_tasks[filename]
        if task_status["status"] == "processing":
            return {
                "filename": filename,
                "status": "processing",
                "output_path": None
            }
        elif task_status["status"] == "error":
            return {
                "filename": filename,
                "status": "error",
                "error": task_status.get("error"),
                "output_path": None
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
                "processed_at": processed_at
            }
        except Exception as e:
            return {
                "filename": filename,
                "status": "error",
                "error": f"读取输出文件失败: {str(e)}",
                "output_path": None
            }

    return {
        "filename": filename,
        "status": "not_processed",
        "output_path": None
    }


async def process_document_task(filename: str):
    """后台任务：处理文档"""
    md_path = ALL_MD_DIR / filename
    output_path = get_output_path(filename)

    processing_tasks[filename] = {
        "status": "processing",
        "started_at": datetime.now().isoformat()
    }

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv", "run", "main.py",
            str(md_path.resolve()),
            "-o", str(OUTPUT_DIR.resolve())
        ]

        result = subprocess.run(
            cmd,
            cwd=IKN_PLUS_DIR,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            processing_tasks[filename] = {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "output_path": str(output_path)
            }

            try:
                import_json_to_db(output_path, filename)
            except Exception as e:
                print(f"Warning: Failed to import to DB: {e}")
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            processing_tasks[filename] = {
                "status": "error",
                "error": error_msg,
                "completed_at": datetime.now().isoformat()
            }

    except subprocess.TimeoutExpired:
        processing_tasks[filename] = {
            "status": "error",
            "error": "处理超时（超过10分钟）",
            "completed_at": datetime.now().isoformat()
        }

    except Exception as e:
        processing_tasks[filename] = {
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        }


# ==================== 文档管理 API ====================

@router.get("/api/documents", response_model=List[Document])
async def list_documents():
    """列出所有文档及其状态"""
    if not ALL_MD_DIR.exists():
        raise HTTPException(status_code=500, detail=f"文档目录不存在: {ALL_MD_DIR}")

    documents = []
    for file in sorted(ALL_MD_DIR.glob("*.md")):
        status_info = check_document_status(file.name)
        documents.append(Document(**status_info))

    return documents


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

    background_tasks.add_task(process_document_task, filename)

    return Document(
        filename=filename,
        status="processing",
        output_path=None
    )


@router.delete("/api/documents/{filename}/output")
async def delete_output(filename: str):
    """删除文档的输出结果"""
    output_path = get_output_path(filename)

    if output_path.exists():
        output_path.unlink()
        if filename in processing_tasks:
            del processing_tasks[filename]
        return {"message": f"已删除输出文件: {output_path.name}"}
    else:
        raise HTTPException(status_code=404, detail="输出文件不存在")


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

    update_chunk(
        chunk_id=chunk_id,
        edited_content=request.edited_content,
        status=request.status,
        content_tags=request.content_tags,
        user_tag=request.user_tag,
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
    """获取所有 chunk 的标签"""
    try:
        with get_connection() as conn:
            user_tags = conn.execute("""
                SELECT DISTINCT user_tag
                FROM document_chunks
                WHERE user_tag IS NOT NULL AND user_tag != ''
            """).fetchall()

            content_tags_rows = conn.execute("""
                SELECT DISTINCT content_tags
                FROM document_chunks
                WHERE content_tags IS NOT NULL AND content_tags != '[]'
            """).fetchall()

        all_tags = set()

        for row in user_tags:
            if row['user_tag']:
                all_tags.add(row['user_tag'])

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
