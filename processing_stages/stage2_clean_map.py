"""
阶段2: 智能清洗与 Token 映射
使用 LLM 标记杂质，清洗文本，并执行反向 Token 映射
"""

import logging
import re
import json
import os
from typing import List, Dict, Any

# 使用绝对导入
try:
    from tokenizer.tokenizer_client import get_tokenizer
    from tokenizer.token_mapper import TokenMapper
    from llm_api.llm_client import get_llm_client
    from llm_api.prompt_templates import get_combined_clean_and_tag_prompts
    from config import PerformanceConfig
except ImportError:
    from ..tokenizer.tokenizer_client import get_tokenizer
    from ..tokenizer.token_mapper import TokenMapper
    from ..llm_api.llm_client import get_llm_client
    from ..llm_api.prompt_templates import get_combined_clean_and_tag_prompts
    from ..config import PerformanceConfig

logger = logging.getLogger(__name__)


class Stage2CleanMap:
    """
    阶段2: 智能清洗与 Token 映射
    """

    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.mapper = TokenMapper(self.tokenizer)
        self.llm_client = get_llm_client()
        self._cached_tags = None  # 缓存系统标签

    def _fetch_system_tags(self) -> List[str]:
        """
        从数据库获取用于LLM标签推理的标签列表

        仅返回内容标签和预置标签，排除文档级标签（文件标签）

        Returns:
            标签列表
        """
        if self._cached_tags is not None:
            return self._cached_tags

        try:
            # 获取内容标签（排除文档级标签）
            try:
                from database import get_content_tags_for_llm
            except ImportError:
                from ..database import get_content_tags_for_llm

            tags = get_content_tags_for_llm()
            self._cached_tags = tags
            logger.info(f"✅ 从数据库获取了 {len(tags)} 个内容标签（已排除文档级标签）: {tags[:10]}...")
            return tags

        except Exception as e:
            logger.warning(f"⚠️ 无法从数据库获取标签: {e}")
            return []

    def process(self, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理阶段2: 清洗并映射 Token

        Args:
            stage1_result: 阶段1的输出结果

        Returns:
            包含 Clean-Chunks 的字典
        """
        logger.info("=" * 60)
        logger.info("阶段2: 开始智能清洗与 Token 映射")
        logger.info("=" * 60)

        base_tokens = stage1_result["base_tokens"]

        # 新增：检查是否有结构树
        if "structure_tree" in stage1_result and stage1_result["structure_tree"]:
            logger.info("🌳 检测到结构树，跳过LLM清洗，直接传递给Stage3")
            logger.info(f"   结构树包含 {len(stage1_result['structure_tree'])} 个章节节点")

            # 直接传递结构树路径，不进行LLM清洗
            result = {
                "base_tokens": base_tokens,
                "structure_tree": stage1_result["structure_tree"],
                "original_text": stage1_result["original_text"],
                "clean_chunks": [],  # Empty for structure path
                "statistics": {
                    "mid_chunk_count": 0,
                    "clean_chunk_count": 0,
                    "total_cleaned_tokens": 0,
                    "avg_chunk_tokens": 0,
                    "structure_nodes": len(stage1_result["structure_tree"])
                }
            }

            logger.info("✅ 结构树传递完成，跳过传统清洗流程")
            return result

        # 原有的LLM清洗路径（向后兼容）
        logger.info("📝 使用传统LLM清洗路径")
        mid_chunks = stage1_result["mid_chunks"]

        # 处理每个 Mid-Chunk
        clean_chunks = []
        for i, mid_chunk in enumerate(mid_chunks, 1):
            logger.info(f"\n处理 Mid-Chunk {i}/{len(mid_chunks)}...")

            try:
                clean_chunk = self._process_mid_chunk(
                    mid_chunk,
                    base_tokens
                )
                clean_chunks.append(clean_chunk)
                logger.info(
                    f"✅ Mid-Chunk {i} 处理完成 "
                    f"(Token: {clean_chunk['token_start']}-{clean_chunk['token_end']})"
                )

            except Exception as e:
                logger.error(f"❌ Mid-Chunk {i} 处理失败: {e}")
                # 创建一个保留原文的 fallback chunk
                fallback_chunk = self._create_fallback_chunk(mid_chunk, base_tokens)
                clean_chunks.append(fallback_chunk)

        # 构建结果（传递结构树和原始文本给Stage3）
        result = {
            "base_tokens": base_tokens,
            "clean_chunks": clean_chunks,
            "structure_tree": stage1_result.get("structure_tree"),  # 新增：传递结构树
            "original_text": stage1_result.get("original_text"),    # 新增：传递原始文本
            "statistics": {
                "mid_chunk_count": len(mid_chunks),
                "clean_chunk_count": len(clean_chunks),
                "total_cleaned_tokens": sum(c["token_count"] for c in clean_chunks),
                "avg_chunk_tokens": sum(c["token_count"] for c in clean_chunks) / len(clean_chunks) if clean_chunks else 0
            }
        }

        logger.info(f"\n阶段2统计:")
        logger.info(f"  处理的 Mid-Chunks: {result['statistics']['mid_chunk_count']}")
        logger.info(f"  生成的 Clean-Chunks: {result['statistics']['clean_chunk_count']}")
        logger.info(f"  清洗后总 Tokens: {result['statistics']['total_cleaned_tokens']}")
        logger.info(f"  平均 Chunk Tokens: {result['statistics']['avg_chunk_tokens']:.1f}")

        return result

    def _process_mid_chunk(
        self,
        mid_chunk: Dict[str, Any],
        base_tokens: List[int]
    ) -> Dict[str, Any]:
        """
        处理单个 Mid-Chunk

        Args:
            mid_chunk: Mid-Chunk 数据
            base_tokens: 基线 Token 序列

        Returns:
            Clean-Chunk 字典
        """
        chunk_text = mid_chunk["text"]
        mid_token_start = mid_chunk["token_start"]

        # 1. 调用 LLM 进行清洗和标签提取
        logger.debug("🔄 调用 LLM 进行清洗和标签提取...")
        llm_result = self._call_llm_for_clean_and_tag(chunk_text)

        marked_text = llm_result.get("marked_text", chunk_text)
        user_tag = llm_result.get("user_tag", "未分类")
        content_tags = llm_result.get("content_tags", [])

        # 2. 移除 JUNK 标签，得到清洗后的文本
        cleaned_text = self._remove_junk_tags(marked_text)
        logger.debug(f"清洗前: {len(chunk_text)} 字符, 清洗后: {len(cleaned_text)} 字符")

        # 3. 执行反向 Token 映射
        logger.debug("🔄 执行反向 Token 映射...")
        clean_token_start, clean_token_end, clean_tokens = self.mapper.reverse_map_cleaned_text(
            original_text=chunk_text,
            cleaned_text=cleaned_text,
            base_token_start=mid_token_start,
            base_tokens=base_tokens
        )

        # 4. 验证 Token 范围
        is_valid = self.mapper.validate_token_range(
            clean_token_start,
            clean_token_end,
            base_tokens,
            cleaned_text
        )

        return {
            "text": cleaned_text,
            "original_text": chunk_text,
            "marked_text": marked_text,
            "token_start": clean_token_start,
            "token_end": clean_token_end,
            "token_count": len(clean_tokens),
            "tokens": clean_tokens,
            "user_tag": user_tag,
            "content_tags": content_tags,
            "validation_passed": is_valid,
            "char_count": len(cleaned_text)
        }

    def _call_llm_for_clean_and_tag(self, chunk_text: str) -> Dict[str, Any]:
        """
        调用 LLM 进行清洗和标签提取

        Args:
            chunk_text: 文档片段

        Returns:
            包含 marked_text, user_tag, content_tags 的字典
        """
        try:
            # 获取系统现有标签
            existing_tags = self._fetch_system_tags()

            # 尝试从数据库获取自定义提示词
            system_prompt = None
            try:
                from database import get_prompt_config
                prompt_config = get_prompt_config('CLEAN_AND_TAG_SYSTEM')
                if prompt_config:
                    system_prompt = prompt_config.get('prompt_value')
                    logger.debug("✅ 使用数据库中的自定义提示词")
            except Exception as e:
                logger.debug(f"从数据库获取提示词失败，使用默认提示词: {e}")

            # 如果数据库中没有，使用默认提示词
            if not system_prompt:
                from llm_api.prompt_templates import PromptTemplates
                system_prompt = PromptTemplates.get_combined_clean_and_tag_system_prompt(existing_tags)
                logger.debug("✅ 使用默认提示词模板")
            else:
                # 替换模板变量
                from config import JunkPatterns, TagConfig

                # 替换杂质特征
                junk_features_str = "\n".join(
                    f"- {jtype}: {desc}"
                    for jtype, desc in JunkPatterns.JUNK_FEATURES.items()
                )
                system_prompt = system_prompt.replace('{{JUNK_FEATURES}}', junk_features_str)

                # 替换标签列表
                tags_str = ", ".join(existing_tags) if existing_tags else "暂无标签"
                system_prompt = system_prompt.replace('{{EXISTING_TAGS}}', tags_str)

                # 替换标签数量
                system_prompt = system_prompt.replace('{{CONTENT_TAG_COUNT}}', str(TagConfig.CONTENT_TAG_COUNT))

                logger.debug("✅ 已替换提示词模板变量")

            # 生成用户提示词
            _, user_prompt = get_combined_clean_and_tag_prompts(
                chunk_text,
                existing_tags=existing_tags
            )

            # 调用 LLM (JSON 模式)
            response = self.llm_client.chat_json_with_system(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # 验证响应
            if not isinstance(response, dict):
                raise ValueError("LLM 响应不是有效的字典")

            marked_text = response.get("marked_text", chunk_text)
            user_tag = response.get("user_tag", "未分类")
            content_tags = response.get("content_tags", [])

            # 验证 content_tags 是列表
            if not isinstance(content_tags, list):
                content_tags = []

            # 【重要】验证并过滤标签：只保留系统中已存在的标签
            # LLM 只能从已有标签中选择，不允许创建新标签
            if existing_tags:
                # 验证 user_tag
                if user_tag not in existing_tags:
                    logger.warning(f"⚠️ LLM 返回的 user_tag '{user_tag}' 不在系统标签列表中，设为默认值")
                    # 优先使用"未分类"（如果存在），否则使用第一个已有标签
                    user_tag = "未分类" if "未分类" in existing_tags else existing_tags[0]

                # 验证 content_tags：只保留已存在的标签
                original_content_tags = content_tags.copy()
                content_tags = [tag for tag in content_tags if tag in existing_tags]

                if len(content_tags) < len(original_content_tags):
                    removed_tags = set(original_content_tags) - set(content_tags)
                    logger.warning(f"⚠️ LLM 返回的部分 content_tags 不在系统标签列表中，已过滤: {removed_tags}")
            else:
                # 如果系统中没有任何标签，设置为 None（不创建新标签）
                logger.warning(f"⚠️ 系统中没有任何标签，user_tag 和 content_tags 设为空")
                user_tag = None
                content_tags = []

            return {
                "marked_text": marked_text,
                "user_tag": user_tag,
                "content_tags": content_tags
            }

        except Exception as e:
            logger.error(f"❌ LLM 调用失败: {e}")
            # 返回默认值 - 如果有已有标签则使用，否则为 None
            default_user_tag = None
            if existing_tags:
                default_user_tag = "未分类" if "未分类" in existing_tags else existing_tags[0]

            return {
                "marked_text": chunk_text,
                "user_tag": default_user_tag,
                "content_tags": []
            }

    def _remove_junk_tags(self, marked_text: str) -> str:
        """
        移除 JUNK 标签及其内容

        Args:
            marked_text: 包含 JUNK 标签的文本

        Returns:
            清洗后的文本
        """
        # 移除 <JUNK type="...">...</JUNK>
        pattern = r'<JUNK\s+type="[^"]*">.*?</JUNK>'
        cleaned = re.sub(pattern, '', marked_text, flags=re.DOTALL)

        # 清理多余的空行（保留最多一个空行）
        cleaned = re.sub(r'\n\n\n+', '\n\n', cleaned)

        # 清理首尾空白
        cleaned = cleaned.strip()

        # 合并标题与表格（确保表格前的标题紧邻表格）
        cleaned = self._merge_title_with_table(cleaned)

        return cleaned

    def _merge_title_with_table(self, text: str) -> str:
        """
        合并标题与表格，删除它们之间的空行

        模式: "标题\n\n<table>" -> "标题\n<table>"

        Args:
            text: 清洗后的文本

        Returns:
            合并后的文本
        """
        # 匹配：标题行 + 一个或多个空行 + 表格开始
        # 标题模式：
        # 1. Markdown 标题（# 开头）
        # 2. 编号标题（如 1.2产品规格）
        # 3. 简单标题（单行，后面紧跟空行和表格）

        patterns = [
            # 模式1: Markdown 标题 + 空行 + 表格
            (r'(^#{1,6}\s+.+?)\n\n+(<table>)', r'\1\n\2'),
            # 模式2: 编号标题（1.2 这种） + 空行 + 表格
            (r'(^\d+\.[\d\.]*\s*.+?)\n\n+(<table>)', r'\1\n\2'),
            # 模式3: 任意单行文字 + 至少2个空行 + 表格（保守匹配，避免误合并段落）
            (r'(^[^\n]{1,30})\n\n\n+(<table>)', r'\1\n\2'),
        ]

        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.MULTILINE)

        logger.debug("✅ 已合并标题与表格之间的空行")
        return result

    def _create_fallback_chunk(
        self,
        mid_chunk: Dict[str, Any],
        base_tokens: List[int]
    ) -> Dict[str, Any]:
        """
        创建 fallback Clean-Chunk（保留原文）

        Args:
            mid_chunk: Mid-Chunk 数据
            base_tokens: 基线 Token 序列

        Returns:
            Clean-Chunk 字典
        """
        chunk_text = mid_chunk["text"]
        tokens = self.tokenizer.encode(chunk_text)

        # 获取系统标签，使用已有标签或 None
        existing_tags = self._fetch_system_tags()
        fallback_user_tag = None
        if existing_tags:
            fallback_user_tag = "未分类" if "未分类" in existing_tags else existing_tags[0]

        return {
            "text": chunk_text,
            "original_text": chunk_text,
            "marked_text": chunk_text,
            "token_start": mid_chunk["token_start"],
            "token_end": mid_chunk["token_end"],
            "token_count": len(tokens),
            "tokens": tokens,
            "user_tag": fallback_user_tag,
            "content_tags": [],
            "validation_passed": True,
            "char_count": len(chunk_text),
            "is_fallback": True
        }

    async def async_process(self, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        异步处理阶段2（并发处理多个 Mid-Chunks）

        Args:
            stage1_result: 阶段1的输出结果

        Returns:
            包含 Clean-Chunks 的字典
        """
        if not PerformanceConfig.ENABLE_ASYNC:
            return self.process(stage1_result)

        logger.info("=" * 60)
        logger.info("阶段2: 开始智能清洗与 Token 映射（异步模式）")
        logger.info("=" * 60)

        base_tokens = stage1_result["base_tokens"]

        # 新增：检查是否有结构树
        if "structure_tree" in stage1_result and stage1_result["structure_tree"]:
            logger.info("🌳 检测到结构树，跳过LLM清洗，直接传递给Stage3")
            logger.info(f"   结构树包含 {len(stage1_result['structure_tree'])} 个章节节点")

            # 直接传递结构树路径，不进行LLM清洗
            result = {
                "base_tokens": base_tokens,
                "structure_tree": stage1_result["structure_tree"],
                "original_text": stage1_result["original_text"],
                "clean_chunks": [],  # Empty for structure path
                "statistics": {
                    "mid_chunk_count": 0,
                    "clean_chunk_count": 0,
                    "total_cleaned_tokens": 0,
                    "avg_chunk_tokens": 0,
                    "structure_nodes": len(stage1_result["structure_tree"])
                }
            }

            logger.info("✅ 结构树传递完成，跳过传统清洗流程")
            return result

        # 原有的LLM清洗路径（向后兼容）
        logger.info("📝 使用传统LLM清洗路径（异步）")
        mid_chunks = stage1_result["mid_chunks"]

        # 批量异步处理
        import asyncio
        from asyncio import Semaphore

        semaphore = Semaphore(PerformanceConfig.MAX_CONCURRENT_REQUESTS)

        async def process_one(mid_chunk, index):
            async with semaphore:
                logger.info(f"处理 Mid-Chunk {index}/{len(mid_chunks)}...")
                try:
                    # 在异步上下文中运行同步代码
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        self._process_mid_chunk,
                        mid_chunk,
                        base_tokens
                    )
                    logger.info(f"✅ Mid-Chunk {index} 处理完成")
                    return result
                except Exception as e:
                    logger.error(f"❌ Mid-Chunk {index} 处理失败: {e}")
                    return self._create_fallback_chunk(mid_chunk, base_tokens)

        tasks = [process_one(chunk, i + 1) for i, chunk in enumerate(mid_chunks)]
        clean_chunks = await asyncio.gather(*tasks)

        # 构建结果
        result = {
            "base_tokens": base_tokens,
            "clean_chunks": clean_chunks,
            "statistics": {
                "mid_chunk_count": len(mid_chunks),
                "clean_chunk_count": len(clean_chunks),
                "total_cleaned_tokens": sum(c["token_count"] for c in clean_chunks),
                "avg_chunk_tokens": sum(c["token_count"] for c in clean_chunks) / len(clean_chunks) if clean_chunks else 0
            }
        }

        logger.info(f"\n阶段2统计:")
        logger.info(f"  处理的 Mid-Chunks: {result['statistics']['mid_chunk_count']}")
        logger.info(f"  生成的 Clean-Chunks: {result['statistics']['clean_chunk_count']}")

        return result


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 模拟阶段1的输出
    try:
        from tokenizer.tokenizer_client import get_tokenizer
    except ImportError:
        from ..tokenizer.tokenizer_client import get_tokenizer

    tokenizer = get_tokenizer()

    test_text = """页眉：公司机密 | 2024年1月

# 产品介绍
这是一个 RAG 系统的文档预处理工具。

⚠️ 警告：此文档仅供内部使用

本工具支持智能清洗和语义切分。
"""

    base_tokens = tokenizer.encode(test_text)

    stage1_result = {
        "base_tokens": base_tokens,
        "original_text": test_text,
        "mid_chunks": [
            {
                "text": test_text,
                "token_start": 0,
                "token_end": len(base_tokens),
                "token_count": len(base_tokens),
                "char_count": len(test_text)
            }
        ]
    }

    # 处理阶段2
    processor = Stage2CleanMap()
    result = processor.process(stage1_result)

    print(f"\n=== 阶段2处理结果 ===")
    print(f"Clean-Chunks: {result['statistics']['clean_chunk_count']}")
    for i, chunk in enumerate(result["clean_chunks"], 1):
        print(f"\nChunk {i}:")
        print(f"  Token 范围: [{chunk['token_start']}:{chunk['token_end']}]")
        print(f"  用户标签: {chunk['user_tag']}")
        print(f"  内容标签: {chunk['content_tags']}")
        print(f"  清洗后内容:\n{chunk['text']}")
