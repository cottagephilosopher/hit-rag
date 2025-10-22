"""
图片上传和URL处理模块
用于处理MinerU转换后的Markdown文件中的图片链接
"""

import os
import re
import tos
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ImageUploader:
    """图片上传到火山引擎TOS对象存储"""

    def __init__(self):
        self.ak = os.getenv('VOLC_ACCESSKEY')
        self.sk = os.getenv('VOLC_SECRETKEY')
        self.endpoint = os.getenv('TOS_OBS_ENDPOINT')
        self.region = os.getenv('TOS_OBS_REGION')
        self.bucket_name = os.getenv('TOS_OBS_BUCKET_NAME')
        self.base_url = os.getenv('TOS_OBS_ACCESS_URL')

        # 检查配置
        if not all([self.ak, self.sk, self.endpoint, self.region, self.bucket_name, self.base_url]):
            missing = []
            if not self.ak: missing.append('VOLC_ACCESSKEY')
            if not self.sk: missing.append('VOLC_SECRETKEY')
            if not self.endpoint: missing.append('TOS_OBS_ENDPOINT')
            if not self.region: missing.append('TOS_OBS_REGION')
            if not self.bucket_name: missing.append('TOS_OBS_BUCKET_NAME')
            if not self.base_url: missing.append('TOS_OBS_ACCESS_URL')
            raise ValueError(f"缺少TOS配置环境变量: {', '.join(missing)}")

        # 初始化TOS客户端
        self.client = tos.TosClientV2(self.ak, self.sk, self.endpoint, self.region)
        self.uploaded_cache = set()  # 缓存已上传的图片

    def upload_image(self, object_key: str, file_path: str) -> Optional[str]:
        """
        上传图片到TOS

        Args:
            object_key: 对象存储key，如 "mineru/image.jpg"
            file_path: 本地文件路径

        Returns:
            成功返回图片访问URL，失败返回None
        """
        try:
            # 检查是否已上传
            if object_key in self.uploaded_cache:
                logger.info(f"图片已上传，跳过: {object_key}")
                return f"{self.base_url}/{object_key}"

            # 上传图片
            self.client.put_object_from_file(self.bucket_name, object_key, file_path)
            logger.info(f"图片上传成功: {object_key}")

            # 缓存
            self.uploaded_cache.add(object_key)

            # 返回访问URL
            return f"{self.base_url}/{object_key}"

        except tos.exceptions.TosClientError as e:
            logger.error(f'TOS客户端错误: {e.message}')
        except tos.exceptions.TosServerError as e:
            logger.error(f'TOS服务端错误: {e.code} - {e.message}')
        except Exception as e:
            logger.error(f'上传图片失败: {e}')

        return None


class MarkdownImageProcessor:
    """处理Markdown文件中的图片链接"""

    def __init__(self, uploader: Optional[ImageUploader] = None, max_concurrent: int = 5):
        """
        Args:
            uploader: ImageUploader实例，如果为None则不上传图片
            max_concurrent: 最大并发处理数
        """
        self.uploader = uploader
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.stats = {
            'total_images': 0,
            'uploaded_images': 0,
            'failed_images': 0,
            'skipped_images': 0,
            'invalid_images': 0
        }

    def _is_valid_image_url(self, img_url: str) -> bool:
        """
        验证图片URL是否有效
        
        过滤掉以下无效URL：
        - AWS签名URL片段
        - 超长URL（可能损坏）
        - 不包含常见图片扩展名的URL
        """
        # 跳过AWS签名URL片段
        if 'aws4_request' in img_url or 'X-Amz-' in img_url:
            logger.warning(f"跳过AWS签名URL片段: {img_url[:100]}...")
            return False
        
        # 跳过过长的URL（可能是损坏的）
        if len(img_url) > 500:
            logger.warning(f"跳过过长URL: {len(img_url)} 字符")
            return False
        
        # 检查是否包含常见图片扩展名
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
        url_lower = img_url.lower()
        
        # 对于完整URL，提取文件名部分检查
        if url_lower.startswith(('http://', 'https://')):
            # 从URL中提取文件名（去除查询参数）
            url_path = img_url.split('?')[0]
            if not any(url_path.endswith(ext) for ext in valid_extensions):
                logger.warning(f"跳过不包含图片扩展名的URL: {img_url[:100]}...")
                return False
        else:
            # 相对路径直接检查
            if not any(url_lower.endswith(ext) for ext in valid_extensions):
                logger.warning(f"跳过不包含图片扩展名的路径: {img_url}")
                return False
        
        return True

    def extract_image_refs(self, md_content: str) -> List[Tuple[str, str, str]]:
        """
        从Markdown内容中提取图片引用

        Returns:
            List of (完整匹配文本, 图片URL, alt文本)
        """
        # 匹配 ![alt](url) 格式
        pattern = r'!\[(.*?)\]\(([^)]+)\)'
        matches = re.findall(pattern, md_content)

        results = []
        for match in matches:
            alt_text, img_url = match
            
            # 验证URL有效性
            if not self._is_valid_image_url(img_url):
                self.stats['invalid_images'] += 1
                continue
            
            # 重构完整匹配文本
            full_match = f"![{alt_text}]({img_url})"
            results.append((full_match, img_url, alt_text))

        return results

    def process_image_url(self, img_url: str, md_file_dir: Path) -> Optional[str]:
        """
        处理单个图片URL

        Args:
            img_url: 图片URL（可能是相对路径或绝对URL）
            md_file_dir: Markdown文件所在目录

        Returns:
            处理后的新URL，失败返回None
        """
        self.stats['total_images'] += 1

        # 如果没有配置上传器，跳过处理
        if not self.uploader:
            logger.debug(f"未配置图片上传器，跳过: {img_url}")
            self.stats['skipped_images'] += 1
            return img_url

        # 如果已经是完整的HTTP URL，检查是否需要处理
        if img_url.startswith(('http://', 'https://')):
            # 如果已经是简化的URL格式，不处理
            if '/mineru/' in img_url and img_url.count('/') <= 5:
                logger.debug(f"图片URL已是简化格式，跳过: {img_url}")
                self.stats['skipped_images'] += 1
                return img_url

            # 复杂路径的URL，需要重新上传
            # 提取文件名
            img_filename = Path(img_url).name
        else:
            # 相对路径，构建本地路径
            img_filename = Path(img_url).name

        # 查找本地图片文件
        local_img_path = self._find_local_image(md_file_dir, img_filename)

        if not local_img_path:
            logger.warning(f"找不到本地图片: {img_filename}")
            self.stats['failed_images'] += 1
            return None

        # 生成对象存储key
        object_key = f"mineru/{img_filename}"

        # 上传图片
        new_url = self.uploader.upload_image(object_key, str(local_img_path))

        if new_url:
            self.stats['uploaded_images'] += 1
            return new_url
        else:
            self.stats['failed_images'] += 1
            return None

    def _find_local_image(self, md_file_dir: Path, img_filename: str) -> Optional[Path]:
        """
        查找本地图片文件

        可能的位置：
        - md_file_dir/images/img_filename
        - md_file_dir/img_filename
        - md_file_dir/../images/img_filename
        """
        # 常见的图片目录名
        possible_dirs = [
            md_file_dir / 'images',
            md_file_dir,
            md_file_dir.parent / 'images',
            md_file_dir / 'auto',  # MinerU可能生成的目录
        ]

        for img_dir in possible_dirs:
            img_path = img_dir / img_filename
            if img_path.exists():
                return img_path

        return None

    def process_markdown_file(self, md_file: Path, output_file: Optional[Path] = None) -> bool:
        """
        处理Markdown文件中的所有图片

        Args:
            md_file: 输入Markdown文件路径
            output_file: 输出文件路径（如果为None则覆盖原文件）

        Returns:
            成功返回True，失败返回False
        """
        try:
            # 读取文件
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()

            # 提取图片引用
            image_refs = self.extract_image_refs(md_content)

            if not image_refs:
                logger.info(f"文件中没有图片引用: {md_file.name}")
                # 即使没有图片，也需要复制文件
                if output_file and output_file != md_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                return True

            logger.info(f"找到 {len(image_refs)} 个图片引用")

            # 处理每个图片
            url_mapping = {}
            for full_match, img_url, alt_text in image_refs:
                new_url = self.process_image_url(img_url, md_file.parent)

                if new_url and new_url != img_url:
                    # 构建新的markdown图片引用
                    new_match = f"![{alt_text}]({new_url})"
                    url_mapping[full_match] = new_match

            # 替换URL
            new_content = md_content
            for old_ref, new_ref in url_mapping.items():
                new_content = new_content.replace(old_ref, new_ref)

            # 写入文件
            output_path = output_file or md_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            logger.info(f"成功处理文件: {output_path.name}")
            logger.info(f"  - 替换了 {len(url_mapping)} 个图片链接")

            return True

        except Exception as e:
            logger.error(f"处理Markdown文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def process_markdown_file_async(self, md_file: Path, output_file: Optional[Path] = None) -> bool:
        """
        异步处理Markdown文件中的所有图片（带超时保护）
        
        Args:
            md_file: 输入Markdown文件路径
            output_file: 输出文件路径（如果为None则覆盖原文件）
        
        Returns:
            成功返回True，失败返回False
        """
        try:
            # 异步读取文件
            loop = asyncio.get_event_loop()
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = await loop.run_in_executor(None, f.read)
            
            # 提取图片引用
            image_refs = self.extract_image_refs(md_content)
            
            if not image_refs:
                logger.info(f"文件中没有图片引用: {md_file.name}")
                # 即使没有图片，也需要复制文件
                if output_file and output_file != md_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        await loop.run_in_executor(None, f.write, md_content)
                return True
            
            logger.info(f"找到 {len(image_refs)} 个有效图片引用")
            
            # 限制处理的图片数量（防止过多图片导致系统卡死）
            max_images = 100
            if len(image_refs) > max_images:
                logger.warning(f"图片数量过多({len(image_refs)})，只处理前{max_images}个")
                image_refs = image_refs[:max_images]
            
            # 异步并发处理图片（带超时保护）
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def process_single_image(ref):
                async with semaphore:
                    full_match, img_url, alt_text = ref
                    try:
                        # 每个图片处理设置30秒超时
                        new_url = await asyncio.wait_for(
                            loop.run_in_executor(
                                self.executor,
                                self.process_image_url,
                                img_url,
                                md_file.parent
                            ),
                            timeout=30.0
                        )
                        return (full_match, new_url, alt_text)
                    except asyncio.TimeoutError:
                        logger.warning(f"图片处理超时(30秒): {img_url[:100]}")
                        self.stats['failed_images'] += 1
                        return (full_match, None, alt_text)
                    except Exception as e:
                        logger.error(f"图片处理异常: {e}")
                        self.stats['failed_images'] += 1
                        return (full_match, None, alt_text)
            
            # 并发处理所有图片
            results = await asyncio.gather(*[process_single_image(ref) for ref in image_refs])
            
            # 构建URL映射
            url_mapping = {}
            for full_match, new_url, alt_text in results:
                if new_url:
                    # 只有成功处理的图片才替换
                    img_url = full_match.split('](')[1].rstrip(')')
                    if new_url != img_url:
                        new_match = f"![{alt_text}]({new_url})"
                        url_mapping[full_match] = new_match
            
            # 替换URL
            new_content = md_content
            for old_ref, new_ref in url_mapping.items():
                new_content = new_content.replace(old_ref, new_ref)
            
            # 异步写入文件
            output_path = output_file or md_file
            with open(output_path, 'w', encoding='utf-8') as f:
                await loop.run_in_executor(None, f.write, new_content)
            
            logger.info(f"成功处理文件: {output_path.name}")
            logger.info(f"  - 替换了 {len(url_mapping)} 个图片链接")
            
            return True
            
        except Exception as e:
            logger.error(f"异步处理Markdown文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_stats(self):
        """打印处理统计"""
        logger.info(f"\n图片处理统计:")
        logger.info(f"  总图片数: {self.stats['total_images']}")
        logger.info(f"  有效图片: {self.stats['total_images'] - self.stats['invalid_images']}")
        logger.info(f"  无效图片: {self.stats['invalid_images']}")
        logger.info(f"  成功上传: {self.stats['uploaded_images']}")
        logger.info(f"  上传失败: {self.stats['failed_images']}")
        logger.info(f"  跳过处理: {self.stats['skipped_images']}")


def create_image_processor() -> Optional[MarkdownImageProcessor]:
    """
    创建图片处理器

    如果TOS配置不完整，返回无上传功能的处理器
    """
    try:
        uploader = ImageUploader()
        logger.info("已初始化图片上传器")
        return MarkdownImageProcessor(uploader)
    except ValueError as e:
        logger.warning(f"图片上传配置不完整: {e}")
        logger.warning("将跳过图片上传，直接使用原始URL")
        return MarkdownImageProcessor(uploader=None)
