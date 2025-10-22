"""操作日志查询路由"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from database import query_document_logs, query_oauth_logs

router = APIRouter()


# ==================== Pydantic Models ====================

class DocumentLogItem(BaseModel):
    id: int
    document_id: int
    filename: str
    chunk_id: Optional[int]
    action: str
    message: Optional[str]
    user_id: Optional[str]
    created_at: str
    payload: Optional[Dict[str, Any]]


class DocumentLogResponse(BaseModel):
    total: int
    items: List[DocumentLogItem]
    available_actions: List[str]


class OAuthLogItem(BaseModel):
    id: int
    user_id: Optional[int]
    provider: str
    action: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    success: bool
    error_message: Optional[str]
    created_at: str


class OAuthLogResponse(BaseModel):
    total: int
    items: List[OAuthLogItem]
    available_actions: List[str]
    available_providers: List[str]


# ==================== Helpers ====================

def _normalize_datetime(value: Optional[str], *, is_end: bool = False) -> Optional[str]:
    """将查询参数标准化为数据库使用的时间字符串"""
    if not value:
        return None

    try:
        if len(value) == 10:
            # 仅日期，例如 2025-01-01
            parsed = datetime.strptime(value, "%Y-%m-%d")
            if is_end:
                parsed = parsed.replace(hour=23, minute=59, second=59)
            else:
                parsed = parsed.replace(hour=0, minute=0, second=0)
        else:
            # ISO 格式，兼容带时区的字符串
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {value}") from exc

    if parsed.tzinfo:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)

    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_list(values: Optional[List[str]]) -> Optional[List[str]]:
    """规范化可能包含逗号分隔的列表查询参数"""
    if not values:
        return None

    results: List[str] = []
    for value in values:
        if not value:
            continue
        parts = [item.strip() for item in value.split(',') if item.strip()]
        results.extend(parts)

    return results or None


# ==================== Routes ====================

@router.get("/api/logs/document", response_model=DocumentLogResponse)
async def list_document_logs(
    document_id: Optional[int] = Query(None, description="文档 ID"),
    filename: Optional[str] = Query(None, description="按照文件名模糊匹配"),
    chunk_id: Optional[int] = Query(None, description="切片 ID"),
    actions: Optional[List[str]] = Query(None, description="操作类型，可多选"),
    user_id: Optional[str] = Query(None, description="操作者 ID"),
    search: Optional[str] = Query(None, description="在消息和 payload 中模糊搜索"),
    start_time: Optional[str] = Query(None, description="起始时间"),
    end_time: Optional[str] = Query(None, description="结束时间"),
    limit: int = Query(50, ge=1, le=200, description="每页数量"),
    offset: int = Query(0, ge=0, description="分页偏移")
) -> DocumentLogResponse:
    """获取文档及切片操作日志"""

    normalized_actions = _normalize_list(actions)
    normalized_start = _normalize_datetime(start_time) if start_time else None
    normalized_end = _normalize_datetime(end_time, is_end=True) if end_time else None

    result = query_document_logs(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        actions=normalized_actions,
        user_id=user_id,
        search=search,
        start_time=normalized_start,
        end_time=normalized_end,
        limit=limit,
        offset=offset
    )

    return DocumentLogResponse(**result)


@router.get("/api/logs/oauth", response_model=OAuthLogResponse)
async def list_oauth_logs(
    provider: Optional[str] = Query(None, description="OAuth 提供商"),
    actions: Optional[List[str]] = Query(None, description="操作类型，可多选"),
    user_id: Optional[int] = Query(None, description="用户 ID"),
    success: Optional[bool] = Query(None, description="是否成功"),
    search: Optional[str] = Query(None, description="错误信息或 User-Agent 搜索"),
    start_time: Optional[str] = Query(None, description="起始时间"),
    end_time: Optional[str] = Query(None, description="结束时间"),
    limit: int = Query(50, ge=1, le=200, description="每页数量"),
    offset: int = Query(0, ge=0, description="分页偏移")
) -> OAuthLogResponse:
    """获取认证与登录操作日志"""

    normalized_actions = _normalize_list(actions)
    normalized_start = _normalize_datetime(start_time) if start_time else None
    normalized_end = _normalize_datetime(end_time, is_end=True) if end_time else None

    result = query_oauth_logs(
        provider=provider,
        action=normalized_actions,
        user_id=user_id,
        success=success,
        search=search,
        start_time=normalized_start,
        end_time=normalized_end,
        limit=limit,
        offset=offset
    )

    return OAuthLogResponse(**result)
