"""
RAG 配置管理 API 路由
提供 RAG 配置的增删改查接口
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import database as db
import os
from pathlib import Path
from config import VectorConfig

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    config_key: str
    config_value: float


class BatchConfigUpdateRequest(BaseModel):
    """批量配置更新请求"""
    configs: Dict[str, float]


class PromptUpdateRequest(BaseModel):
    """提示词更新请求"""
    prompt_key: str
    prompt_value: str


class BatchPromptUpdateRequest(BaseModel):
    """批量提示词更新请求"""
    prompts: Dict[str, str]


class SystemConfigUpdateRequest(BaseModel):
    """系统配置更新请求"""
    database_file: str
    milvus_collection: str


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


# ============================================
# 提示词配置管理
# ============================================

@router.get("/prompts")
async def get_prompt_configs(prompt_key: Optional[str] = None):
    """
    获取提示词配置

    参数:
        prompt_key: 可选的提示词键，如果提供则只返回该配置项

    返回:
        - 如果指定了 prompt_key，返回单个配置项
        - 否则返回所有配置项（按分类分组）
    """
    try:
        prompts = db.get_prompt_config(prompt_key)

        if prompt_key:
            if not prompts:
                raise HTTPException(status_code=404, detail=f"提示词配置 {prompt_key} 不存在")
            return prompts
        else:
            # 按分类分组
            grouped = {}
            for key, prompt in prompts.items():
                category = prompt.get('category', 'other')
                if category not in grouped:
                    grouped[category] = {}
                grouped[category][key] = prompt

            return {
                "prompts": prompts,
                "grouped": grouped
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取提示词配置失败: {str(e)}")


@router.put("/prompts")
async def update_prompt_config(request: PromptUpdateRequest):
    """
    更新单个提示词配置项

    参数:
        prompt_key: 提示词键
        prompt_value: 提示词内容

    返回:
        更新结果
    """
    try:
        # 检查配置项是否存在
        existing_prompt = db.get_prompt_config(request.prompt_key)
        if not existing_prompt:
            raise HTTPException(status_code=404, detail=f"提示词配置 {request.prompt_key} 不存在")

        # 更新配置
        success = db.update_prompt_config(request.prompt_key, request.prompt_value)

        if not success:
            raise HTTPException(status_code=500, detail="更新提示词配置失败")

        return {
            "success": True,
            "message": f"提示词配置 {request.prompt_key} 已更新",
            "prompt_key": request.prompt_key
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新提示词配置失败: {str(e)}")


@router.put("/prompts/batch")
async def batch_update_prompt_configs(request: BatchPromptUpdateRequest):
    """
    批量更新提示词配置

    参数:
        prompts: 配置字典 {prompt_key: prompt_value}

    返回:
        更新结果
    """
    try:
        # 验证所有配置项
        all_prompts = db.get_prompt_config()
        errors = []

        for prompt_key in request.prompts.keys():
            if prompt_key not in all_prompts:
                errors.append(f"提示词配置 {prompt_key} 不存在")

        if errors:
            raise HTTPException(status_code=400, detail="; ".join(errors))

        # 批量更新
        updated_count = db.batch_update_prompt_config(request.prompts)

        return {
            "success": True,
            "message": f"成功更新 {updated_count} 个提示词配置",
            "updated_count": updated_count,
            "total_requested": len(request.prompts)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量更新提示词配置失败: {str(e)}")


@router.post("/prompts/reset")
async def reset_prompt_configs():
    """
    重置所有提示词配置为默认值
    """
    try:
        all_prompts = db.get_prompt_config()
        reset_prompts = {}

        for prompt_key, prompt in all_prompts.items():
            default_value = prompt.get('default_value')
            if default_value is not None:
                reset_prompts[prompt_key] = default_value

        updated_count = db.batch_update_prompt_config(reset_prompts)

        return {
            "success": True,
            "message": f"成功重置 {updated_count} 个提示词配置",
            "updated_count": updated_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置提示词配置失败: {str(e)}")


# ============================================
# 系统配置管理
# ============================================

@router.get("/system")
async def get_system_config():
    """
    获取系统配置

    返回:
        - database_file: 数据库文件路径
        - database_exists: 数据库文件是否存在
        - milvus_collection: Milvus 集合名称
    """
    try:
        # 使用统一的数据库路径获取方法
        db_path = db.get_db_file()

        # 检查数据库文件是否存在
        database_exists = db_path.exists()

        # 从配置读取 Milvus 集合名称
        milvus_collection = VectorConfig.MILVUS_COLLECTION_NAME

        return {
            "database_file": str(db_path.absolute()),
            "database_exists": database_exists,
            "milvus_collection": milvus_collection
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统配置失败: {str(e)}")


@router.put("/system")
async def update_system_config(request: SystemConfigUpdateRequest):
    """
    更新系统配置

    参数:
        database_file: 数据库文件路径
        milvus_collection: Milvus 集合名称

    返回:
        更新结果
    """
    try:
        from dotenv import load_dotenv, set_key, find_dotenv

        # 查找 .env 文件
        env_file = find_dotenv()
        if not env_file:
            # 如果没有 .env 文件，创建一个
            env_file = Path.cwd() / '.env'
            env_file.touch()

        env_path = Path(env_file)

        # 更新 .env 文件
        set_key(env_path, "DB_FILE", request.database_file)
        set_key(env_path, "MILVUS_COLLECTION_NAME", request.milvus_collection)

        # 检查数据库文件是否存在
        db_path = Path(request.database_file)
        db_exists = db_path.exists()

        # 如果数据库不存在，自动创建
        created_db = False
        if not db_exists:
            try:
                # 确保目录存在
                db_path.parent.mkdir(parents=True, exist_ok=True)

                # 初始化数据库
                db.init_database()
                created_db = True
            except Exception as e:
                print(f"警告：无法自动创建数据库: {e}")

        message = "系统配置已保存到 .env 文件"
        if created_db:
            message += "，数据库已自动创建"
        message += "。请重启服务以使配置生效。"

        return {
            "success": True,
            "message": message,
            "database_file": str(db_path.absolute()),
            "milvus_collection": request.milvus_collection,
            "database_created": created_db
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新系统配置失败: {str(e)}")


@router.post("/system/create-database")
async def create_database():
    """
    创建数据库（如果不存在）

    返回:
        创建结果
    """
    try:
        # 使用统一的数据库路径获取方法
        db_path = db.get_db_file()

        # 如果数据库已存在
        if db_path.exists():
            return {
                "success": True,
                "message": "数据库已存在",
                "database_file": str(db_path.absolute())
            }

        # 确保目录存在
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        db.init_database()

        return {
            "success": True,
            "message": "数据库创建成功",
            "database_file": str(db_path.absolute())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建数据库失败: {str(e)}")


@router.post("/system/initialize")
async def initialize_system():
    """
    初始化系统

    警告：此操作将删除所有数据！

    返回:
        初始化结果
    """
    try:
        import init_system

        # 调用初始化脚本中的函数
        # 注意：这里直接调用初始化函数，跳过交互式确认（因为前端已经有确认）
        success = init_system.init_sqlite_database(force=True)

        if not success:
            raise Exception("数据库初始化失败")

        return {
            "success": True,
            "message": "系统初始化成功，所有数据已清空"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系统初始化失败: {str(e)}")
