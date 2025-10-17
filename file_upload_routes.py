"""
文件上传和MinerU转换路由
支持上传文件（PDF/DOCX/PPTX等）并调用MinerU API转换为Markdown
"""

import os
import aiohttp
import sqlite3
import shutil
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel

# 导入图片处理模块
from image_uploader import create_image_processor

router = APIRouter()
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).parent))
FILE_DIR = Path(os.getenv("FILE_DIR", BASE_DIR / "files"))
ALL_MD_DIR = Path(os.getenv("ALL_MD_DIR", BASE_DIR / "all-md"))
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / ".dbs" / "hit-rag.db"))

# 上传文件保存到 FILE_DIR/uploads 子目录
UPLOAD_DIR = FILE_DIR / "uploads"
# 转换后的 Markdown 文件保存到 FILE_DIR/converted 子目录
CONVERTED_DIR = FILE_DIR / "converted"

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
ALL_MD_DIR.mkdir(parents=True, exist_ok=True)

# MinerU API 配置（从环境变量读取）
MINERU_API_BASE = os.getenv("MINERU_API_BASE", "https://api.mineru.net")
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "")

# 支持的文件格式（根据MinerU官方文档）
SUPPORTED_FILE_TYPES = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/msword': '.doc',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/vnd.ms-powerpoint': '.ppt',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/vnd.ms-excel': '.xls',
    'image/jpeg': '.jpg',
    'image/png': '.png',
}

# ==================== Pydantic Models ====================

class FileUploadResponse(BaseModel):
    id: int
    original_filename: str
    file_size: int
    file_type: str
    status: str
    created_at: str
    mineru_task_id: Optional[str] = None
    converted_md_filename: Optional[str] = None
    error_message: Optional[str] = None


class FileUploadStatusResponse(BaseModel):
    id: int
    original_filename: str
    status: str
    converted_md_filename: Optional[str] = None
    error_message: Optional[str] = None
    conversion_progress: Optional[int] = None


# ==================== Database Functions ====================

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_file_upload_table():
    """初始化文件上传表"""
    schema_path = DB_PATH.parent / "file_upload_schema.sql"
    if not schema_path.exists():
        return

    with get_db_connection() as conn:
        with open(schema_path, 'r', encoding='utf-8') as f:
            conn.executescript(f.read())


def create_file_upload_record(
    original_filename: str,
    file_size: int,
    file_type: str,
    upload_path: str
) -> int:
    """创建文件上传记录"""
    with get_db_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO file_uploads (
                original_filename, file_size, file_type, upload_path, status
            ) VALUES (?, ?, ?, ?, 'pending')
        """, (original_filename, file_size, file_type, upload_path))
        conn.commit()
        return cursor.lastrowid


def update_file_upload_status(
    upload_id: int,
    status: str,
    mineru_task_id: Optional[str] = None,
    converted_md_filename: Optional[str] = None,
    converted_md_path: Optional[str] = None,
    error_message: Optional[str] = None
):
    """更新文件上传状态"""
    with get_db_connection() as conn:
        fields = ["status = ?"]
        params = [status]

        if mineru_task_id:
            fields.append("mineru_task_id = ?")
            params.append(mineru_task_id)

        if converted_md_filename:
            fields.append("converted_md_filename = ?")
            params.append(converted_md_filename)

        if converted_md_path:
            fields.append("converted_md_path = ?")
            params.append(converted_md_path)

        if error_message:
            fields.append("error_message = ?")
            params.append(error_message)

        if status == 'converting':
            fields.append("conversion_started_at = ?")
            params.append(datetime.now().isoformat())
        elif status == 'completed':
            fields.append("conversion_completed_at = ?")
            params.append(datetime.now().isoformat())

        params.append(upload_id)
        query = f"UPDATE file_uploads SET {', '.join(fields)} WHERE id = ?"

        conn.execute(query, params)
        conn.commit()


def get_file_upload_by_id(upload_id: int) -> Optional[dict]:
    """根据ID获取文件上传记录"""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM file_uploads WHERE id = ?",
            (upload_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_file_uploads(limit: int = 100) -> List[dict]:
    """获取所有文件上传记录"""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM file_uploads ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


# ==================== MinerU API Functions ====================

async def upload_file_to_tos(file_path: Path) -> str:
    """
    上传文件到对象存储，返回可访问的URL
    MinerU API需要通过URL访问文件
    """
    try:
        from image_uploader import ImageUploader
        uploader = ImageUploader()

        # 生成对象key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_key = f"mineru-uploads/{timestamp}_{file_path.name}"

        # 上传文件
        logger.info(f"上传文件到TOS: {object_key}")
        file_url = uploader.upload_image(object_key, str(file_path))

        if file_url:
            logger.info(f"文件上传成功: {file_url}")
            return file_url

    except Exception as e:
        logger.error(f"上传文件到TOS失败: {e}")

    raise Exception("需要配置TOS对象存储才能使用MinerU API。请在.env中配置VOLC_ACCESSKEY等参数。")


async def call_mineru_upload_api(file_path: Path, file_type: str) -> dict:
    """
    调用MinerU API创建提取任务

    官方API文档:
    POST https://mineru.net/api/v4/extract/task
    {
        "url": "文件URL",
        "is_ocr": true,
        "enable_formula": false
    }
    """
    if not MINERU_API_KEY:
        raise Exception("MINERU_API_KEY 未配置，请在.env文件中设置")

    # 1. 上传文件到对象存储
    logger.info("上传文件到对象存储...")
    file_url = await upload_file_to_tos(file_path)

    # 2. 调用MinerU API创建任务
    async with aiohttp.ClientSession() as session:
        url = f"{MINERU_API_BASE}/api/v4/extract/task"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {MINERU_API_KEY}'
        }

        data = {
            'url': file_url,
            'is_ocr': True,          # 启用OCR
            'enable_formula': False,  # 可选：启用公式识别
        }

        logger.info(f"调用MinerU API: {url}")
        logger.info(f"请求数据: {data}")

        async with session.post(url, headers=headers, json=data) as resp:
            response_text = await resp.text()

            if resp.status != 200:
                logger.error(f"MinerU API错误: {resp.status} - {response_text}")
                raise Exception(f"MinerU API返回错误: {resp.status} - {response_text}")

            result = await resp.json() if resp.content_type == 'application/json' else {'error': response_text}
            logger.info(f"MinerU API响应: {result}")

            # 检查响应格式
            if 'data' not in result:
                raise Exception(f"MinerU API响应格式错误: {result}")

            task_data = result['data']
            return {
                'task_id': task_data.get('task_id'),
                'status': 'pending'
            }


async def check_mineru_task_status(task_id: str) -> dict:
    """
    查询MinerU任务状态

    官方API: GET https://mineru.net/api/v4/extract/task/{task_id}

    实际返回格式:
    {
        "code": 0,
        "msg": "ok",
        "data": {
            "task_id": "xxx",
            "state": "done" | "processing" | "failed",
            "full_zip_url": "https://...zip",
            "err_msg": ""
        }
    }
    """
    if not MINERU_API_KEY:
        raise Exception("MINERU_API_KEY 未配置")

    async with aiohttp.ClientSession() as session:
        url = f"{MINERU_API_BASE}/api/v4/extract/task/{task_id}"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {MINERU_API_KEY}'
        }

        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"查询任务状态失败: {resp.status} - {error_text}")
                raise Exception(f"查询任务状态失败: {resp.status} - {error_text}")

            result = await resp.json()
            logger.info(f"MinerU任务状态响应: {result}")

            # 检查响应格式
            if 'data' not in result:
                raise Exception(f"响应格式错误: {result}")

            task_data = result['data']

            # ⚠️ 注意：MinerU返回的是 "state" 不是 "status"
            mineru_state = task_data.get('state', 'processing')

            # 映射状态
            # MinerU的state: done | processing | failed
            status_map = {
                'done': 'completed',
                'processing': 'processing',
                'failed': 'failed',
                'error': 'failed'
            }

            mapped_status = status_map.get(mineru_state, 'processing')

            # MinerU不返回实时进度，只能根据状态估算
            progress = 0
            if mineru_state == 'processing':
                progress = 50  # 处理中显示50%
            elif mineru_state == 'done':
                progress = 100

            # ⚠️ 返回的是zip压缩包URL，不是直接的markdown
            result_url = task_data.get('full_zip_url')
            error_msg = task_data.get('err_msg')

            logger.info(f"任务状态: state={mineru_state}, mapped={mapped_status}, progress={progress}, zip_url={result_url}")

            return {
                'status': mapped_status,
                'progress': progress,
                'result_url': result_url,  # 这是zip文件的URL
                'error': error_msg if error_msg else None
            }


async def download_mineru_result(result_url: str, output_path: Path):
    """
    下载MinerU转换结果（zip压缩包）并解压提取markdown

    MinerU返回的是zip压缩包，包含:
    - full.md - 完整的markdown文件
    - images/ - 图片目录（如果有）
    - layout.json - 布局信息
    - *_origin.pdf - 原始PDF
    """
    if not result_url:
        raise Exception("result_url为空")

    logger.info(f"下载MinerU结果zip: {result_url}")

    import zipfile
    import io

    async with aiohttp.ClientSession() as session:
        async with session.get(result_url) as resp:
            if resp.status != 200:
                raise Exception(f"下载zip失败: {resp.status}")

            zip_content = await resp.read()
            logger.info(f"下载完成，zip大小: {len(zip_content)} bytes")

            # 解压zip
            try:
                with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                    # 列出所有文件
                    file_list = zip_file.namelist()
                    logger.info(f"zip包含文件: {file_list}")

                    # 查找markdown文件（优先full.md）
                    md_files = [f for f in file_list if f.endswith('.md')]

                    if not md_files:
                        raise Exception(f"zip中没有找到markdown文件。包含的文件: {file_list}")

                    # 优先使用full.md
                    md_filename = 'full.md' if 'full.md' in md_files else md_files[0]
                    logger.info(f"提取markdown文件: {md_filename}")

                    # 提取markdown内容
                    md_content = zip_file.read(md_filename)

                    # 保存到目标路径
                    with open(output_path, 'wb') as f:
                        f.write(md_content)

                    logger.info(f"Markdown已保存: {output_path} ({len(md_content)} bytes)")

                    # 如果有图片目录，也提取出来
                    image_files = [f for f in file_list if '/images/' in f or f.startswith('images/')]
                    if image_files:
                        # 创建images目录
                        images_dir = output_path.parent / 'images'
                        images_dir.mkdir(exist_ok=True)

                        logger.info(f"提取 {len(image_files)} 个图片文件...")
                        for img_file in image_files:
                            img_content = zip_file.read(img_file)
                            img_filename = Path(img_file).name
                            img_path = images_dir / img_filename
                            with open(img_path, 'wb') as f:
                                f.write(img_content)

                        logger.info(f"图片已保存到: {images_dir}")

            except zipfile.BadZipFile as e:
                raise Exception(f"zip文件损坏: {e}")
            except Exception as e:
                raise Exception(f"解压zip失败: {e}")


# ==================== Background Tasks ====================

async def process_file_conversion(upload_id: int):
    """后台任务：处理文件转换"""
    try:
        # 获取上传记录
        record = get_file_upload_by_id(upload_id)
        if not record:
            raise Exception("上传记录不存在")

        upload_path = Path(record['upload_path'])
        if not upload_path.exists():
            raise Exception("上传文件不存在")

        # 调用MinerU API上传文件
        update_file_upload_status(upload_id, 'converting')

        api_result = await call_mineru_upload_api(upload_path, record['file_type'])
        task_id = api_result.get('task_id')

        if not task_id:
            raise Exception("MinerU API未返回任务ID")

        update_file_upload_status(upload_id, 'converting', mineru_task_id=task_id)

        # 轮询任务状态
        import asyncio
        max_attempts = 60  # 最多等待5分钟（60 * 5秒）

        for _ in range(max_attempts):
            await asyncio.sleep(5)

            status_result = await check_mineru_task_status(task_id)
            status = status_result.get('status')

            if status == 'completed':
                # 下载转换结果
                result_url = status_result.get('result_url')
                if not result_url:
                    raise Exception("MinerU API未返回结果URL")

                # 生成MD文件名和路径
                original_name = Path(record['original_filename']).stem
                md_filename = f"{original_name}_converted.md"

                # 先保存到 CONVERTED_DIR (FILE_DIR/converted)
                converted_path = CONVERTED_DIR / md_filename
                await download_mineru_result(result_url, converted_path)

                logger.info(f"MinerU转换完成，文件已保存: {converted_path}")

                # 处理图片URL（上传图片并替换链接）
                try:
                    logger.info("开始处理Markdown中的图片...")
                    image_processor = create_image_processor()

                    # 将处理后的文件保存到 ALL_MD_DIR
                    md_path = ALL_MD_DIR / md_filename
                    success = image_processor.process_markdown_file(converted_path, md_path)

                    if success:
                        logger.info(f"图片处理完成，文件已保存到: {md_path}")
                        image_processor.print_stats()
                    else:
                        logger.warning("图片处理失败，复制原始文件到 ALL_MD_DIR")
                        shutil.copy2(converted_path, md_path)

                except Exception as e:
                    logger.error(f"图片处理出错: {e}")
                    logger.warning("将使用原始文件")
                    # 如果图片处理失败，直接复制原文件
                    md_path = ALL_MD_DIR / md_filename
                    shutil.copy2(converted_path, md_path)

                update_file_upload_status(
                    upload_id,
                    'completed',
                    converted_md_filename=md_filename,
                    converted_md_path=str(converted_path)  # 记录在 FILE_DIR/converted 中的路径
                )
                return

            elif status == 'failed':
                error_msg = status_result.get('error', '转换失败')
                raise Exception(error_msg)

        raise Exception("转换超时")

    except Exception as e:
        update_file_upload_status(upload_id, 'error', error_message=str(e))
        print(f"文件转换失败 (upload_id={upload_id}): {e}")


# ==================== API Routes ====================

@router.post("/api/upload/file", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    上传文件并触发MinerU转换

    支持的文件类型：PDF, DOCX, PPTX, XLSX, JPG, PNG等
    """
    try:
        # 检查文件类型
        if file.content_type not in SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.content_type}。支持的类型: {', '.join(SUPPORTED_FILE_TYPES.values())}"
            )

        # 保存文件
        file_extension = SUPPORTED_FILE_TYPES[file.content_type]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        upload_path = UPLOAD_DIR / safe_filename

        content = await file.read()
        with open(upload_path, 'wb') as f:
            f.write(content)

        file_size = len(content)

        # 创建数据库记录
        upload_id = create_file_upload_record(
            original_filename=file.filename,
            file_size=file_size,
            file_type=file.content_type,
            upload_path=str(upload_path)
        )

        # 启动后台转换任务
        background_tasks.add_task(process_file_conversion, upload_id)

        record = get_file_upload_by_id(upload_id)

        return FileUploadResponse(
            id=record['id'],
            original_filename=record['original_filename'],
            file_size=record['file_size'],
            file_type=record['file_type'],
            status=record['status'],
            created_at=record['created_at']
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.get("/api/upload/{upload_id}/status", response_model=FileUploadStatusResponse)
async def get_upload_status(upload_id: int):
    """获取文件上传和转换状态"""
    record = get_file_upload_by_id(upload_id)

    if not record:
        raise HTTPException(status_code=404, detail="上传记录不存在")

    # 如果有MinerU任务ID且状态为converting，查询进度
    progress = None
    if record['status'] == 'converting' and record['mineru_task_id']:
        try:
            status_result = await check_mineru_task_status(record['mineru_task_id'])
            progress = status_result.get('progress', 0)
        except:
            pass

    return FileUploadStatusResponse(
        id=record['id'],
        original_filename=record['original_filename'],
        status=record['status'],
        converted_md_filename=record['converted_md_filename'],
        error_message=record['error_message'],
        conversion_progress=progress
    )


@router.get("/api/upload/list", response_model=List[FileUploadResponse])
async def list_uploads(limit: int = 50):
    """获取文件上传列表"""
    records = get_all_file_uploads(limit=limit)

    return [
        FileUploadResponse(
            id=r['id'],
            original_filename=r['original_filename'],
            file_size=r['file_size'],
            file_type=r['file_type'],
            status=r['status'],
            created_at=r['created_at'],
            mineru_task_id=r['mineru_task_id'],
            converted_md_filename=r['converted_md_filename'],
            error_message=r['error_message']
        )
        for r in records
    ]


@router.delete("/api/upload/{upload_id}")
async def delete_upload(upload_id: int):
    """删除上传记录和相关文件"""
    record = get_file_upload_by_id(upload_id)

    if not record:
        raise HTTPException(status_code=404, detail="上传记录不存在")

    try:
        # 删除上传的原始文件
        upload_path = Path(record['upload_path'])
        if upload_path.exists():
            upload_path.unlink()

        # 删除转换后的MD文件
        if record['converted_md_path']:
            md_path = Path(record['converted_md_path'])
            if md_path.exists():
                md_path.unlink()

        # 删除数据库记录
        with get_db_connection() as conn:
            conn.execute("DELETE FROM file_uploads WHERE id = ?", (upload_id,))
            conn.commit()

        return {"message": "删除成功"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# 初始化数据库表
init_file_upload_table()
