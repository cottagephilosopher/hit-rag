"""
RAG 配置管理 API 路由
提供 RAG 配置的增删改查接口
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import database as db

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    config_key: str
    config_value: float


class BatchConfigUpdateRequest(BaseModel):
    """批量配置更新请求"""
    configs: Dict[str, float]


@router.get("/rag")
async def get_rag_configs(config_key: Optional[str] = None):
    """
    获取 RAG 配置

    参数:
        config_key: 可选的配置键，如果提供则只返回该配置项

    返回:
        - 如果指定了 config_key，返回单个配置项
        - 否则返回所有配置项（按分类分组）
    """
    try:
        configs = db.get_rag_config(config_key)

        if config_key:
            if not configs:
                raise HTTPException(status_code=404, detail=f"配置项 {config_key} 不存在")
            return configs
        else:
            # 按分类分组
            grouped = {}
            for key, config in configs.items():
                category = config.get('category', 'other')
                if category not in grouped:
                    grouped[category] = {}
                grouped[category][key] = config

            return {
                "configs": configs,
                "grouped": grouped
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")


@router.put("/rag")
async def update_rag_config(request: ConfigUpdateRequest):
    """
    更新单个 RAG 配置项

    参数:
        config_key: 配置键
        config_value: 配置值

    返回:
        更新结果
    """
    try:
        # 检查配置项是否存在
        existing_config = db.get_rag_config(request.config_key)
        if not existing_config:
            raise HTTPException(status_code=404, detail=f"配置项 {request.config_key} 不存在")

        # 验证值范围
        min_value = existing_config.get('min_value')
        max_value = existing_config.get('max_value')

        if min_value is not None and request.config_value < min_value:
            raise HTTPException(
                status_code=400,
                detail=f"配置值不能小于最小值 {min_value}"
            )

        if max_value is not None and request.config_value > max_value:
            raise HTTPException(
                status_code=400,
                detail=f"配置值不能大于最大值 {max_value}"
            )

        # 更新配置
        success = db.update_rag_config(request.config_key, request.config_value)

        if not success:
            raise HTTPException(status_code=500, detail="更新配置失败")

        return {
            "success": True,
            "message": f"配置项 {request.config_key} 已更新",
            "config_key": request.config_key,
            "config_value": request.config_value
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


@router.put("/rag/batch")
async def batch_update_rag_configs(request: BatchConfigUpdateRequest):
    """
    批量更新 RAG 配置

    参数:
        configs: 配置字典 {config_key: config_value}

    返回:
        更新结果
    """
    try:
        # 验证所有配置项
        all_configs = db.get_rag_config()
        errors = []

        for config_key, config_value in request.configs.items():
            if config_key not in all_configs:
                errors.append(f"配置项 {config_key} 不存在")
                continue

            existing_config = all_configs[config_key]
            min_value = existing_config.get('min_value')
            max_value = existing_config.get('max_value')

            if min_value is not None and config_value < min_value:
                errors.append(f"{config_key}: 值 {config_value} 小于最小值 {min_value}")

            if max_value is not None and config_value > max_value:
                errors.append(f"{config_key}: 值 {config_value} 大于最大值 {max_value}")

        if errors:
            raise HTTPException(status_code=400, detail="; ".join(errors))

        # 批量更新
        updated_count = db.batch_update_rag_config(request.configs)

        return {
            "success": True,
            "message": f"成功更新 {updated_count} 个配置项",
            "updated_count": updated_count,
            "total_requested": len(request.configs)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量更新配置失败: {str(e)}")


@router.post("/rag/reset")
async def reset_rag_configs():
    """
    重置所有 RAG 配置为默认值
    """
    try:
        all_configs = db.get_rag_config()
        reset_configs = {}

        for config_key, config in all_configs.items():
            default_value = config.get('default_value')
            if default_value is not None:
                reset_configs[config_key] = default_value

        updated_count = db.batch_update_rag_config(reset_configs)

        return {
            "success": True,
            "message": f"成功重置 {updated_count} 个配置项",
            "updated_count": updated_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置配置失败: {str(e)}")
