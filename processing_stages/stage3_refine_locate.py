"""
阶段3: 精细切分与最终定位
使用 LLM 对清洗后的文本进行精细切分，并计算最终的 Token 绝对索引
"""

import logging
import re
import json
from typing import List, Dict, Any

# 使用绝对导入
try:
    from tokenizer.tokenizer_client import get_tokenizer
    from tokenizer.token_mapper import TokenMapper
    from llm_api.llm_client import get_llm_client
    from llm_api.prompt_templates import get_chunking_prompts
    from config import ChunkConfig, ValidationConfig
except ImportError:
    from ..tokenizer.tokenizer_client import get_tokenizer
    from ..tokenizer.token_mapper import TokenMapper
    from ..llm_api.llm_client import get_llm_client
    from ..llm_api.prompt_templates import get_chunking_prompts
    from ..config import ChunkConfig, ValidationConfig

logger = logging.getLogger(__name__)


class Stage3RefineLocate:
    """
    阶段3: 精细切分与最终定位
    """

    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.mapper = TokenMapper(self.tokenizer)
        self.llm_client = get_llm_client()

    def process(self, stage2_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理阶段3: 精细切分并定位

        Args:
            stage2_result: 阶段2的输出结果

        Returns:
            包含最终 RAG Chunks 的字典
        """
        logger.info("=" * 60)
        logger.info("阶段3: 开始精细切分与最终定位")
        logger.info("=" * 60)

        base_tokens = stage2_result["base_tokens"]

        # 检查是否有结构树（新策略）
        if "structure_tree" in stage2_result and stage2_result["structure_tree"]:
            logger.info("🌳 使用结构树进行结构化切分")
            structure_tree = stage2_result["structure_tree"]
            original_text = stage2_result["original_text"]
            final_chunks = self._structure_based_chunking(
                structure_tree,
                original_text,
                base_tokens
            )
        else:
            logger.info("📝 使用传统Clean-Chunk切分（向后兼容）")
            clean_chunks = stage2_result["clean_chunks"]

            # 处理每个 Clean-Chunk
            final_chunks = []
            for i, clean_chunk in enumerate(clean_chunks, 1):
                logger.info(f"\n处理 Clean-Chunk {i}/{len(clean_chunks)}...")

                try:
                    chunks = self._process_clean_chunk(
                        clean_chunk,
                        base_tokens
                    )
                    final_chunks.extend(chunks)
                    logger.info(
                        f"✅ Clean-Chunk {i} 处理完成，生成 {len(chunks)} 个最终块"
                    )

                except Exception as e:
                    logger.error(f"❌ Clean-Chunk {i} 处理失败: {e}")
                    # 创建 fallback chunk
                    fallback = self._create_fallback_final_chunk(clean_chunk)
                    final_chunks.append(fallback)

        # Token 溢出校验（第一次：合并前）
        overflow_info = []
        if ValidationConfig.CHECK_TOKEN_OVERFLOW:
            overflow_info = self._validate_token_overflow(final_chunks)

        # 自动修复超限块
        if overflow_info:
            final_chunks = self._auto_fix_overflow_chunks(final_chunks, overflow_info)

        # 合并过小的 chunks
        final_chunks = self._merge_small_chunks(final_chunks, base_tokens)

        # Token 溢出校验（第二次：合并后）
        # 合并操作可能会产生新的超限块，需要再次验证和修复
        if ValidationConfig.CHECK_TOKEN_OVERFLOW:
            overflow_info_after_merge = self._validate_token_overflow(final_chunks)
            if overflow_info_after_merge:
                logger.info("检测到合并后产生的超限块，开始修复...")
                final_chunks = self._auto_fix_overflow_chunks(final_chunks, overflow_info_after_merge)

        # 检测 token gap
        self._validate_token_continuity(final_chunks)

        # 构建结果
        result = {
            "final_chunks": final_chunks,
            "statistics": self._calculate_statistics(final_chunks)
        }

        logger.info(f"\n阶段3统计:")
        stats = result["statistics"]
        logger.info(f"  最终块数量: {stats['total_chunks']}")
        logger.info(f"  平均 Token 数: {stats['avg_tokens']:.1f}")
        logger.info(f"  Token 范围: {stats['min_tokens']}-{stats['max_tokens']}")
        logger.info(f"  ATOMIC 块数量: {stats['atomic_chunks']}")
        logger.info(f"  验证通过率: {stats['validation_pass_rate']:.1%}")

        return result

    def _process_clean_chunk(
        self,
        clean_chunk: Dict[str, Any],
        base_tokens: List[int]
    ) -> List[Dict[str, Any]]:
        """
        处理单个 Clean-Chunk

        Args:
            clean_chunk: Clean-Chunk 数据
            base_tokens: 基线 Token 序列

        Returns:
            Final-Chunk 列表
        """
        chunk_text = clean_chunk["text"]
        clean_token_start = clean_chunk["token_start"]
        clean_tokens = clean_chunk["tokens"]
        user_tag = clean_chunk["user_tag"]
        content_tags = clean_chunk["content_tags"]

        # 估算 Token 数量（使用 token 范围，而不是 tokens 数组长度）
        estimated_tokens = clean_chunk["token_end"] - clean_chunk["token_start"]

        # 检测是否包含特殊结构（表格、代码块等）
        has_special_structure = self._contains_special_structure(chunk_text)

        # 检测是否包含步骤序列
        has_step_sequence = self._contains_step_sequence(chunk_text)

        # 新策略：语义完整性优先
        # 1. 如果在 MIN ~ TARGET 范围内，且无特殊结构，直接返回（理想大小）
        if (ChunkConfig.FINAL_MIN_TOKENS <= estimated_tokens <= ChunkConfig.FINAL_TARGET_TOKENS
            and not has_special_structure
            and not has_step_sequence):
            logger.debug(f"Clean-Chunk 在理想范围内 ({estimated_tokens} tokens)，无需切分")
            return [self._create_final_chunk(
                content=chunk_text,
                token_start=clean_token_start,
                token_end=clean_chunk["token_end"],
                user_tag=user_tag,
                content_tags=content_tags,
                is_atomic=False,
                atomic_type=None,
                base_tokens=base_tokens,
                context_tags=[]
            )]

        # 2. 如果在 TARGET ~ MAX 范围内，且无特殊结构，也直接返回（可接受大小）
        if (ChunkConfig.FINAL_TARGET_TOKENS < estimated_tokens <= ChunkConfig.FINAL_MAX_TOKENS
            and not has_special_structure
            and not has_step_sequence):
            logger.debug(f"Clean-Chunk 略大但可接受 ({estimated_tokens} tokens)，直接保留")
            return [self._create_final_chunk(
                content=chunk_text,
                token_start=clean_token_start,
                token_end=clean_chunk["token_end"],
                user_tag=user_tag,
                content_tags=content_tags,
                is_atomic=False,
                atomic_type=None,
                base_tokens=base_tokens,
                context_tags=[]
            )]

        # 3. 如果在 MAX ~ HARD_LIMIT 范围内，标记为 ATOMIC-CONTENT（优先保证语义完整性）
        if (ChunkConfig.FINAL_MAX_TOKENS < estimated_tokens <= ChunkConfig.FINAL_HARD_LIMIT
            and not has_special_structure
            and not has_step_sequence):
            logger.warning(
                f"⚠️ Clean-Chunk 超过建议最大值但未达硬性上限 ({estimated_tokens} tokens)，"
                f"标记为 ATOMIC-CONTENT 以保持语义完整性"
            )
            return [self._create_final_chunk(
                content=chunk_text,
                token_start=clean_token_start,
                token_end=clean_chunk["token_end"],
                user_tag=user_tag,
                content_tags=content_tags,
                is_atomic=True,
                atomic_type="content",
                base_tokens=base_tokens,
                context_tags=["语义完整", "超大段落"]
            )]

        # 4. 如果超过硬性上限，必须调用 LLM 切分
        if estimated_tokens > ChunkConfig.FINAL_HARD_LIMIT:
            logger.warning(
                f"⚠️ Clean-Chunk 超过硬性上限 ({estimated_tokens} > {ChunkConfig.FINAL_HARD_LIMIT})，"
                f"必须进行切分"
            )

        # 5. 【优先】标题优先切分策略 - 在ATOMIC检测之前执行
        # 如果 chunk 超过 TARGET_TOKENS 且包含章节标题，优先在标题处切分
        if estimated_tokens > ChunkConfig.FINAL_TARGET_TOKENS:
            headers = self._detect_headers_in_chunk(chunk_text)
            if headers:
                # 尝试在标题位置切分
                header_based_chunks = self._split_at_headers(
                    chunk_text,
                    clean_token_start,
                    clean_tokens,
                    headers,
                    user_tag,
                    content_tags,
                    base_tokens
                )
                if header_based_chunks:
                    logger.debug(f"✅ 基于标题切分成功，生成 {len(header_based_chunks)} 个子块，跳过ATOMIC检测")
                    return header_based_chunks

        # 6. 如果检测到步骤序列，且在合理范围内，直接标记为 ATOMIC-STEP
        if has_step_sequence and estimated_tokens <= ChunkConfig.FINAL_HARD_LIMIT:
            logger.debug(
                f"检测到步骤序列 ({estimated_tokens} tokens)，标记为 ATOMIC-STEP 保持完整性"
            )
            return [self._create_final_chunk(
                content=chunk_text,
                token_start=clean_token_start,
                token_end=clean_chunk["token_end"],
                user_tag=user_tag,
                content_tags=content_tags,
                is_atomic=True,
                atomic_type="step",
                base_tokens=base_tokens,
                context_tags=self._extract_step_sequence_context_tags(chunk_text)
            )]

        # 7. 如果包含特殊结构（表格/代码块），需要 LLM 切分以识别 ATOMIC 块
        if has_special_structure:
            logger.debug(f"检测到特殊结构（表格/代码块），需要 LLM 切分以识别 ATOMIC 块")

        # 调用 LLM 进行精细切分
        logger.debug(f"🔄 调用 LLM 进行精细切分 ({estimated_tokens} tokens)...")
        llm_chunks = self._call_llm_for_chunking(chunk_text, estimated_tokens)

        # 处理切分结果并定位
        final_chunks = []
        for llm_chunk in llm_chunks:
            try:
                # 提取 ATOMIC 标签和上下文标签
                content, is_atomic, atomic_type, context_tags = self._extract_atomic_tag(
                    llm_chunk["content"]
                )

                # 定位 Token 索引
                token_start, token_end = self.mapper.locate_final_chunk(
                    clean_chunk_text=chunk_text,
                    clean_chunk_tokens=clean_tokens,
                    clean_chunk_token_start=clean_token_start,
                    final_chunk_text=content
                )

                # 创建最终块
                final_chunk = self._create_final_chunk(
                    content=content,
                    token_start=token_start,
                    token_end=token_end,
                    user_tag=user_tag,
                    content_tags=content_tags,
                    is_atomic=is_atomic,
                    atomic_type=atomic_type,
                    context_tags=context_tags,
                    base_tokens=base_tokens
                )

                final_chunks.append(final_chunk)

            except Exception as e:
                logger.error(f"❌ 处理切分块失败: {e}")
                continue

        # 如果切分失败，返回原始块
        if not final_chunks:
            logger.warning("⚠️ LLM 切分失败，返回原始 Clean-Chunk")
            final_chunks = [self._create_final_chunk(
                content=chunk_text,
                token_start=clean_token_start,
                token_end=clean_chunk["token_end"],
                user_tag=user_tag,
                content_tags=content_tags,
                is_atomic=False,
                atomic_type=None,
                base_tokens=base_tokens,
                context_tags=[]
            )]

        return final_chunks

    def _call_llm_for_chunking(
        self,
        chunk_text: str,
        estimated_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        调用 LLM 进行精细切分

        Args:
            chunk_text: 要切分的文本
            estimated_tokens: 估算的 Token 数

        Returns:
            切分结果列表
        """
        try:
            system_prompt, user_prompt = get_chunking_prompts(
                chunk_text,
                estimated_tokens
            )

            # 调用 LLM (JSON 模式)
            response = self.llm_client.chat_json_with_system(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # 提取 chunks
            chunks = response.get("chunks", [])

            if not chunks:
                raise ValueError("LLM 未返回任何切分结果")

            logger.debug(f"✅ LLM 返回 {len(chunks)} 个切分块")
            return chunks

        except Exception as e:
            logger.error(f"❌ LLM 切分调用失败: {e}")
            # 返回原始文本作为单个块
            return [{
                "content": chunk_text,
                "is_atomic": False,
                "atomic_type": None
            }]

    def _contains_special_structure(self, text: str) -> bool:
        """
        检测文本中是否包含特殊结构（表格、代码块、公式等）

        Args:
            text: 文本内容

        Returns:
            是否包含特殊结构
        """
        import re

        # 检测表格（HTML table 标签）
        if re.search(r'<table[^>]*>.*?</table>', text, re.DOTALL):
            logger.debug("检测到 HTML 表格")
            return True

        # 检测 Markdown 表格（至少2行，包含 | 和 -）
        lines = text.split('\n')
        table_like_lines = [l for l in lines if '|' in l]
        if len(table_like_lines) >= 2:
            # 检查是否有分隔行（如 |-----|-----| 或 | --- | --- |）
            separator_lines = [l for l in table_like_lines
                             if re.search(r'\|[\s\-:]+\|', l) and '-' in l]
            if separator_lines:
                logger.debug("检测到 Markdown 表格")
                return True

        # 检测代码块
        if re.search(r'```[\s\S]*?```', text):
            logger.debug("检测到代码块")
            return True

        # 检测公式（LaTeX 或 $$ 包围）
        if re.search(r'\$\$[\s\S]+?\$\$', text) or re.search(r'\\begin\{equation\}', text):
            logger.debug("检测到数学公式")
            return True

        return False

    def _contains_step_sequence(self, text: str) -> bool:
        """
        检测文本中是否包含编号步骤序列

        识别模式：
        - (1)(2)(3)... 格式
        - 1. 2. 3. ... 格式
        - 1) 2) 3) ... 格式
        - ① ② ③ ... 格式（圆圈数字）

        Args:
            text: 文本内容

        Returns:
            是否包含步骤序列
        """
        import re

        # 模式1: (1)(2)(3) 格式 - 至少3个连续步骤
        pattern1 = r'\((\d+)\)[^\(]*\((\d+)\)[^\(]*\((\d+)\)'
        if re.search(pattern1, text):
            logger.debug("检测到 (1)(2)(3) 格式的步骤序列")
            return True

        # 模式2: 1. 2. 3. 格式（行首） - 至少3个连续步骤
        lines = text.split('\n')
        numbered_lines = []
        for line in lines:
            match = re.match(r'^\s*(\d+)\.\s+', line)
            if match:
                numbered_lines.append(int(match.group(1)))

        # 检查是否有至少3个连续的数字
        if len(numbered_lines) >= 3:
            # 检查是否是连续序列（允许部分连续）
            for i in range(len(numbered_lines) - 2):
                if (numbered_lines[i+1] == numbered_lines[i] + 1 and
                    numbered_lines[i+2] == numbered_lines[i] + 2):
                    logger.debug("检测到 1. 2. 3. 格式的步骤序列")
                    return True

        # 模式3: 1) 2) 3) 格式
        pattern3 = r'(\d+)\)[^\d\)]*(\d+)\)[^\d\)]*(\d+)\)'
        if re.search(pattern3, text):
            logger.debug("检测到 1) 2) 3) 格式的步骤序列")
            return True

        # 模式4: 圆圈数字 ① ② ③
        circle_nums = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']
        found_circle_nums = [cn for cn in circle_nums if cn in text]
        if len(found_circle_nums) >= 3:
            logger.debug("检测到圆圈数字格式的步骤序列")
            return True

        return False

    def _extract_atomic_tag(self, content: str) -> tuple:
        """
        提取 ATOMIC 标签

        Args:
            content: 内容文本

        Returns:
            (clean_content, is_atomic, atomic_type, context_tags) 元组
        """
        import re

        # 匹配 ATOMIC 标签
        atomic_pattern = r'<ATOMIC-(\w+)>(.*?)</ATOMIC-\1>'
        match = re.search(atomic_pattern, content, re.DOTALL)

        if match:
            atomic_type = match.group(1).lower()
            clean_content = match.group(2)

            # 提取上下文标签（用于表格/步骤/内容召回）
            if atomic_type == "step":
                context_tags = self._extract_step_sequence_context_tags(clean_content)
            elif atomic_type == "content":
                # ATOMIC-CONTENT: 提取标题和关键词作为上下文
                context_tags = self._extract_content_context_tags(clean_content)
            else:
                context_tags = self._extract_table_context_tags(clean_content, atomic_type)

            return clean_content, True, atomic_type, context_tags

        return content, False, None, []

    def _extract_table_context_tags(self, content: str, atomic_type: str) -> list:
        """
        提取表格/代码块/步骤序列的上下文标签

        Args:
            content: ATOMIC 块内容
            atomic_type: ATOMIC 类型（table, code, link, formula）

        Returns:
            上下文标签列表
        """
        import re

        tags = []

        # 表格类型：提取标题和加粗文本
        if atomic_type == "table":
            lines = content.split('\n')
            for line in lines:
                line = line.strip()

                # 匹配标题（# 标题）
                title_match = re.match(r'^#+\s+(.+)$', line)
                if title_match:
                    title_text = title_match.group(1).strip()
                    tags.append(title_text)

                # 匹配加粗文本（**文本**）
                bold_matches = re.findall(r'\*\*(.+?)\*\*', line)
                tags.extend(bold_matches)

        # 去重并返回前3个最相关的标签
        unique_tags = []
        for tag in tags:
            if tag and tag not in unique_tags:
                unique_tags.append(tag)

        return unique_tags[:3]  # 最多返回3个标签

    def _detect_headers_in_chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        检测文本中的 Markdown 标题，用于标题优先切分

        Args:
            text: 输入文本

        Returns:
            标题列表，包含位置和级别信息
        """
        import re
        headers = []

        # 匹配 Markdown 标题：# 开头，支持 1-6 级
        header_pattern = r'^(#{1,6})\s+(.+)$'

        lines = text.split('\n')
        current_pos = 0

        for line in lines:
            match = re.match(header_pattern, line.strip())
            if match:
                level = len(match.group(1))  # # 的数量
                header_text = match.group(2).strip()

                # 计算标题在文本中的字符位置
                char_position = text.find(line, current_pos)
                if char_position != -1:
                    char_position_ratio = char_position / len(text) if len(text) > 0 else 0.0

                    # 只记录位置比例 > 0.2 的标题（不在最开头）
                    # 这样可以在合适的地方切分
                    if char_position_ratio > 0.2:
                        headers.append({
                            "level": level,
                            "text": header_text,
                            "char_position": char_position,
                            "char_position_ratio": char_position_ratio,
                            "line": line
                        })

            current_pos += len(line) + 1  # +1 for \n

        return headers

    def _split_at_headers(
        self,
        chunk_text: str,
        clean_token_start: int,
        clean_tokens: List[int],
        headers: List[Dict[str, Any]],
        user_tag: str,
        content_tags: List[str],
        base_tokens: List[int]
    ) -> List[Dict[str, Any]]:
        """
        在标题位置切分文本

        策略：
        1. 找到位置最优的标题作为切分点（30%-70%位置）
        2. 在标题之前切分，确保标题位于新块开头
        3. 递归处理切分后的子块

        Args:
            chunk_text: 要切分的文本
            clean_token_start: Token 起始位置
            clean_tokens: Token 序列
            headers: 检测到的标题列表
            user_tag: 用户标签
            content_tags: 内容标签
            base_tokens: 基线 Token 序列

        Returns:
            Final-Chunk 列表，如果切分失败返回 None
        """
        if not headers:
            return None

        # 找到最优切分点：位置在 30%-70% 之间的标题
        best_header = None
        for header in headers:
            ratio = header["char_position_ratio"]
            if 0.3 <= ratio <= 0.7:
                if not best_header or abs(ratio - 0.5) < abs(best_header["char_position_ratio"] - 0.5):
                    best_header = header

        # 如果没有找到理想位置的标题，使用第一个标题
        if not best_header:
            best_header = headers[0]

        split_pos = best_header["char_position"]

        # 在标题之前切分（确保标题在下一个块的开头）
        before_text = chunk_text[:split_pos].strip()
        after_text = chunk_text[split_pos:].strip()

        if not before_text or not after_text:
            # 切分失败，块太小
            return None

        logger.debug(
            f"📍 在标题处切分: '{best_header['text']}' "
            f"(位置: {best_header['char_position_ratio']:.1%})"
        )

        # 创建两个子块
        final_chunks = []

        try:
            # 第一个块（标题之前的内容）
            before_tokens = self.tokenizer.encode(before_text)
            before_token_start = clean_token_start
            before_token_end = before_token_start + len(before_tokens)

            final_chunks.append(self._create_final_chunk(
                content=before_text,
                token_start=before_token_start,
                token_end=before_token_end,
                user_tag=user_tag,
                content_tags=content_tags,
                is_atomic=False,
                atomic_type=None,
                base_tokens=base_tokens,
                context_tags=[]
            ))

            # 第二个块（标题及之后的内容）
            after_tokens = self.tokenizer.encode(after_text)
            after_token_start = before_token_end
            after_token_end = after_token_start + len(after_tokens)

            final_chunks.append(self._create_final_chunk(
                content=after_text,
                token_start=after_token_start,
                token_end=after_token_end,
                user_tag=user_tag,
                content_tags=content_tags,
                is_atomic=False,
                atomic_type=None,
                base_tokens=base_tokens,
                context_tags=[f"标题: {best_header['text']}"]
            ))

            return final_chunks

        except Exception as e:
            logger.warning(f"⚠️ 标题切分失败: {e}")
            return None

    def _extract_step_sequence_context_tags(self, content: str) -> list:
        """
        提取步骤序列的上下文标签

        提取策略：
        - 提取步骤序列前的标题
        - 提取每个步骤的关键操作动词
        - 提取加粗的关键术语

        Args:
            content: 步骤序列内容

        Returns:
            上下文标签列表
        """
        import re

        tags = []

        lines = content.split('\n')

        # 提取标题
        for line in lines:
            line = line.strip()
            title_match = re.match(r'^#+\s+(.+)$', line)
            if title_match:
                title_text = title_match.group(1).strip()
                tags.append(title_text)

        # 提取步骤中的关键动词（如：安装、配置、启动等）
        action_verbs = ['安装', '配置', '启动', '设置', '连接', '打开', '关闭',
                       '创建', '删除', '修改', '检查', '测试', '运行', '执行']

        for verb in action_verbs:
            if verb in content:
                tags.append(verb)

        # 提取加粗文本
        bold_matches = re.findall(r'\*\*(.+?)\*\*', content)
        tags.extend(bold_matches)

        # 去重并返回前5个最相关的标签
        unique_tags = []
        for tag in tags:
            if tag and tag not in unique_tags and len(tag) <= 10:  # 过滤过长的标签
                unique_tags.append(tag)

        return unique_tags[:5]  # 步骤序列返回更多标签（最多5个）

    def _extract_content_context_tags(self, content: str) -> list:
        """
        提取 ATOMIC-CONTENT 的上下文标签
        
        提取策略：
        - 提取所有标题（#, ##, ### 等）
        - 提取加粗的关键术语
        - 限制标签长度和数量
        
        Args:
            content: ATOMIC-CONTENT 内容
            
        Returns:
            上下文标签列表
        """
        import re
        
        tags = []
        lines = content.split('\n')
        
        # 提取所有标题
        for line in lines:
            line = line.strip()
            title_match = re.match(r'^#+\s+(.+)$', line)
            if title_match:
                title_text = title_match.group(1).strip()
                tags.append(title_text)
        
        # 提取加粗文本
        bold_matches = re.findall(r'\*\*(.+?)\*\*', content)
        tags.extend(bold_matches)
        
        # 去重并返回前3个最相关的标签
        unique_tags = []
        for tag in tags:
            if tag and tag not in unique_tags and len(tag) <= 15:  # 过滤过长的标签
                unique_tags.append(tag)
        
        return unique_tags[:3]  # ATOMIC-CONTENT 返回最多3个标签


    def _create_final_chunk(
        self,
        content: str,
        token_start: int,
        token_end: int,
        user_tag: str,
        content_tags: List[str],
        is_atomic: bool,
        atomic_type: str,
        base_tokens: List[int],
        context_tags: List[str] = None
    ) -> Dict[str, Any]:
        """
        创建最终 RAG Chunk

        Args:
            content: 内容文本
            token_start: 起始 Token 索引
            token_end: 结束 Token 索引
            user_tag: 用户定义标签
            content_tags: 内容推理标签
            is_atomic: 是否为 ATOMIC 块
            atomic_type: ATOMIC 类型
            base_tokens: 基线 Token 序列
            context_tags: 上下文标签（用于表格召回）

        Returns:
            Final-Chunk 字典
        """
        token_count = token_end - token_start

        # 验证 Token 范围
        is_valid = self.mapper.validate_token_range(
            token_start,
            token_end,
            base_tokens,
            content
        )

        chunk = {
            "content": content,
            "token_start": token_start,
            "token_end": token_end,
            "token_count": token_count,
            "user_tag": user_tag,
            "content_tags": content_tags,
            "is_atomic": is_atomic,
            "atomic_type": atomic_type,
            "validation_passed": is_valid,
            "char_count": len(content)
        }

        # 为 ATOMIC 表格添加上下文标签
        if is_atomic and atomic_type == "table" and context_tags:
            chunk["table_context_tags"] = context_tags
            logger.debug(f"提取表格上下文标签: {context_tags}")

        return chunk

    def _create_fallback_final_chunk(
        self,
        clean_chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建 fallback 最终块

        Args:
            clean_chunk: Clean-Chunk 数据

        Returns:
            Final-Chunk 字典
        """
        return {
            "content": clean_chunk["text"],
            "token_start": clean_chunk["token_start"],
            "token_end": clean_chunk["token_end"],
            "token_count": clean_chunk["token_count"],
            "user_tag": clean_chunk.get("user_tag", "未分类"),
            "content_tags": clean_chunk.get("content_tags", []),
            "is_atomic": False,
            "atomic_type": None,
            "validation_passed": True,
            "char_count": clean_chunk["char_count"],
            "is_fallback": True
        }

    def _validate_token_overflow(self, final_chunks: List[Dict[str, Any]]):
        """
        验证 Token 溢出情况

        Args:
            final_chunks: Final-Chunk 列表
        """
        logger.info("\n执行 Token 溢出校验...")

        overflow_chunks = []
        underflow_chunks = []

        for i, chunk in enumerate(final_chunks):
            token_count = chunk["token_count"]
            is_atomic = chunk["is_atomic"]

            # 详细调试日志
            logger.debug(
                f"检查 Chunk {i+1}: token_count={token_count}, is_atomic={is_atomic}, "
                f"FINAL_MAX_TOKENS={ChunkConfig.FINAL_MAX_TOKENS}"
            )

            # 检查超过最大限制
            if token_count > ChunkConfig.FINAL_MAX_TOKENS:
                if is_atomic:
                    # ATOMIC 块允许超过限制
                    if token_count > ChunkConfig.ATOMIC_MAX_TOKENS:
                        logger.warning(
                            f"⚠️ Chunk {i+1}: ATOMIC 块超过限制 "
                            f"({token_count} > {ChunkConfig.ATOMIC_MAX_TOKENS})"
                        )
                        overflow_chunks.append((i+1, token_count, "ATOMIC"))
                    else:
                        logger.debug(
                            f"✅ Chunk {i+1}: ATOMIC 块 ({token_count} tokens)"
                        )
                else:
                    # 非 ATOMIC 块超过限制
                    logger.error(
                        f"❌ Chunk {i+1}: 非 ATOMIC 块超过最大限制 "
                        f"({token_count} > {ChunkConfig.FINAL_MAX_TOKENS})"
                    )
                    overflow_chunks.append((i+1, token_count, "NORMAL"))

                    if ValidationConfig.STRICT_MODE:
                        raise ValueError(
                            f"严格模式：Chunk {i+1} 超过最大 Token 限制"
                        )

            # 检查小于最小限制
            elif token_count < ChunkConfig.FINAL_MIN_TOKENS:
                logger.warning(
                    f"⚠️ Chunk {i+1}: 小于最小限制 "
                    f"({token_count} < {ChunkConfig.FINAL_MIN_TOKENS})"
                )
                underflow_chunks.append((i+1, token_count))

        # 输出摘要
        if overflow_chunks:
            logger.warning(f"发现 {len(overflow_chunks)} 个溢出块")
            # 返回溢出块信息供后续修复使用
            return overflow_chunks
        if underflow_chunks:
            logger.warning(f"发现 {len(underflow_chunks)} 个过小块")

        if not overflow_chunks and not underflow_chunks:
            logger.info("✅ 所有块的 Token 数量均在合理范围内")

        return []

    def _auto_fix_overflow_chunks(
        self,
        chunks: List[Dict[str, Any]],
        overflow_info: List[tuple]
    ) -> List[Dict[str, Any]]:
        """
        自动修复超限的非 ATOMIC 块

        对于超过 FINAL_MAX_TOKENS 但未超过 FINAL_HARD_LIMIT 的非 ATOMIC 块，
        自动标记为 ATOMIC-CONTENT 以保持语义完整性

        Args:
            chunks: Final-Chunk 列表
            overflow_info: 溢出块信息 [(chunk_id, token_count, type), ...]

        Returns:
            修复后的 Final-Chunk 列表
        """
        if not overflow_info:
            return chunks

        logger.info("\n开始自动修复超限块...")

        fixed_count = 0
        fixed_chunks = chunks.copy()

        for chunk_id, token_count, overflow_type in overflow_info:
            if overflow_type == "NORMAL":
                chunk_index = chunk_id - 1

                # 检查是否在硬性上限以内
                if token_count <= ChunkConfig.FINAL_HARD_LIMIT:
                    logger.info(
                        f"🔧 修复 Chunk {chunk_id}: {token_count} tokens "
                        f"→ 标记为 ATOMIC-CONTENT（语义完整性优先）"
                    )

                    # 标记为 ATOMIC-CONTENT
                    fixed_chunks[chunk_index]["is_atomic"] = True
                    fixed_chunks[chunk_index]["atomic_type"] = "content"

                    # 提取上下文标签
                    content = fixed_chunks[chunk_index]["content"]
                    context_tags = self._extract_content_context_tags(content)
                    if context_tags:
                        fixed_chunks[chunk_index]["content_tags"].extend(context_tags)
                        # 去重
                        fixed_chunks[chunk_index]["content_tags"] = list(set(
                            fixed_chunks[chunk_index]["content_tags"]
                        ))

                    fixed_count += 1
                else:
                    logger.error(
                        f"❌ Chunk {chunk_id} 超过硬性上限 ({token_count} > {ChunkConfig.FINAL_HARD_LIMIT})，"
                        f"无法自动修复，建议检查源文档或调整切分策略"
                    )

        if fixed_count > 0:
            logger.info(f"✅ 已自动修复 {fixed_count} 个超限块")

        return fixed_chunks

    def _merge_small_chunks(
        self,
        chunks: List[Dict[str, Any]],
        base_tokens: List[int]
    ) -> List[Dict[str, Any]]:
        """
        合并过小的 chunks（小于 100 tokens 且非 ATOMIC）

        Args:
            chunks: Final-Chunk 列表
            base_tokens: 基线 Token 序列

        Returns:
            合并后的 Final-Chunk 列表
        """
        if not chunks:
            return chunks

        if not ChunkConfig.ENABLE_SMALL_CHUNK_MERGE:
            logger.info("小 chunk 合并功能已禁用")
            return chunks

        MIN_CHUNK_SIZE = ChunkConfig.SMALL_CHUNK_THRESHOLD
        merged = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # 如果当前 chunk 小于阈值、非 ATOMIC，且后面还有 chunk
            if (current["token_count"] < MIN_CHUNK_SIZE and
                not current["is_atomic"] and
                i + 1 < len(chunks)):

                next_chunk = chunks[i + 1]

                # 检查是否可以与下一个 chunk 合并（非 ATOMIC 或合并后不超过最大限制）
                merged_token_count = current["token_count"] + next_chunk["token_count"]

                if (not next_chunk["is_atomic"] and
                    merged_token_count <= ChunkConfig.FINAL_MAX_TOKENS):

                    # 合并内容
                    merged_content = current["content"] + "\n\n" + next_chunk["content"]

                    # 创建合并后的 chunk
                    merged_chunk = self._create_final_chunk(
                        content=merged_content,
                        token_start=current["token_start"],
                        token_end=next_chunk["token_end"],
                        user_tag=current["user_tag"],
                        content_tags=current["content_tags"],
                        is_atomic=False,
                        atomic_type=None,
                        base_tokens=base_tokens,
                        context_tags=[]
                    )

                    logger.info(
                        f"✅ 合并小 chunk: {current['token_count']} + {next_chunk['token_count']} "
                        f"= {merged_chunk['token_count']} tokens"
                    )

                    merged.append(merged_chunk)
                    i += 2  # 跳过已合并的两个 chunk
                    continue

            # 无法合并，保留当前 chunk
            merged.append(current)
            i += 1

        return merged

    def _validate_token_continuity(self, chunks: List[Dict[str, Any]]):
        """
        验证 token 序列的连续性，检测大的 token gap

        Args:
            chunks: Final-Chunk 列表
        """
        if len(chunks) < 2:
            return

        if not ValidationConfig.CHECK_TOKEN_CONTINUITY:
            return

        logger.info("\n执行 Token 连续性检测...")

        GAP_THRESHOLD = ValidationConfig.TOKEN_GAP_THRESHOLD
        gaps_found = []

        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            gap = next_chunk["token_start"] - current["token_end"]

            if gap > GAP_THRESHOLD:
                logger.warning(
                    f"⚠️ 检测到 Token Gap: Chunk {i+1} -> Chunk {i+2}\n"
                    f"   Chunk {i+1} 结束于 token {current['token_end']}\n"
                    f"   Chunk {i+2} 起始于 token {next_chunk['token_start']}\n"
                    f"   Gap 大小: {gap} tokens\n"
                    f"   可能原因: 阶段2清洗时删除了中间内容（图片链接、杂质等）"
                )
                gaps_found.append({
                    "between_chunks": (i+1, i+2),
                    "gap_size": gap,
                    "chunk1_end": current["token_end"],
                    "chunk2_start": next_chunk["token_start"]
                })
            elif gap < 0:
                logger.error(
                    f"❌ Token 重叠错误: Chunk {i+1} -> Chunk {i+2}\n"
                    f"   Chunk {i+1} 结束于 token {current['token_end']}\n"
                    f"   Chunk {i+2} 起始于 token {next_chunk['token_start']}\n"
                    f"   重叠: {-gap} tokens"
                )
            elif gap > 0:
                logger.debug(
                    f"✓ 小 gap: Chunk {i+1} -> Chunk {i+2}, gap={gap} tokens (正常，清洗导致)"
                )

        if gaps_found:
            logger.warning(f"共检测到 {len(gaps_found)} 处较大 Token Gap")
        else:
            logger.info("✅ 未检测到显著的 Token Gap")

    def _structure_based_chunking(
        self,
        structure_tree: List[Dict[str, Any]],
        original_text: str,
        base_tokens: List[int]
    ) -> List[Dict[str, Any]]:
        """
        基于文档结构树进行切分

        策略：
        1. 优先在顶层章节边界切分 (hierarchy_level=1: # 1, # 2, # 3)
        2. 如果顶层章节过大，在二级章节边界切分 (hierarchy_level=2: # 1.1, # 1.2)
        3. 表格必须与其标题保持在一起
        4. 步骤序列保持完整
        5. 每个chunk必须以标题开头

        Args:
            structure_tree: 文档结构树
            original_text: 原始文本
            base_tokens: Token序列

        Returns:
            Final chunks列表
        """
        logger.info(f"开始结构化切分，共 {len(structure_tree)} 个章节节点")

        final_chunks = []
        i = 0

        while i < len(structure_tree):
            node = structure_tree[i]

            # 特殊处理：检测目录区域
            if node.get('is_toc', False):
                # 收集整个目录区域（从 toc_start 到 toc_end）
                toc_start_idx = node.get('toc_start_idx')
                toc_end_idx = node.get('toc_end_idx')

                if toc_start_idx is not None and toc_end_idx is not None:
                    # 收集目录区域的所有节点
                    toc_nodes = []
                    for k in range(len(structure_tree)):
                        if structure_tree[k].get('toc_start_idx') == toc_start_idx:
                            if structure_tree[k].get('is_toc') or structure_tree[k].get('is_toc_part'):
                                toc_nodes.append(structure_tree[k])

                    if toc_nodes:
                        # 计算整个目录区域的范围
                        toc_text_start = toc_nodes[0]['char_start']
                        toc_text_end = toc_nodes[-1]['char_end']
                        toc_full_text = original_text[toc_text_start:toc_text_end]
                        toc_tokens = len(self.tokenizer.encode(toc_full_text))

                        logger.info(f"📑 处理目录区域: 包含 {len(toc_nodes)} 个节点，约 {toc_tokens} tokens")

                        # 检查 ATOMIC 硬性上限 (3000 tokens)
                        if toc_tokens > ChunkConfig.ATOMIC_MAX_TOKENS:
                            logger.warning(f"⚠️ 目录区域超过 ATOMIC_MAX_TOKENS ({toc_tokens} > {ChunkConfig.ATOMIC_MAX_TOKENS})，强制切分")
                            # 强制切分：按节点逐个添加，确保每个chunk不超过3000 tokens
                            current_toc_nodes = []
                            current_toc_tokens = 0

                            for toc_node in toc_nodes:
                                node_text = original_text[toc_node['char_start']:toc_node['char_end']]
                                node_tokens = len(self.tokenizer.encode(node_text))

                                # 如果加上当前节点会超限，先保存当前chunk
                                if current_toc_nodes and current_toc_tokens + node_tokens > ChunkConfig.ATOMIC_MAX_TOKENS:
                                    # 创建当前的TOC chunk
                                    merged_text_start = current_toc_nodes[0]['char_start']
                                    merged_text_end = current_toc_nodes[-1]['char_end']
                                    merged_text = original_text[merged_text_start:merged_text_end]

                                    chunk = self._create_section_chunk(
                                        merged_text,
                                        merged_text_start,
                                        current_toc_nodes[0],  # 使用第一个节点作为标题
                                        base_tokens,
                                        original_text
                                    )
                                    final_chunks.append(chunk)
                                    logger.debug(f"  ✅ 目录部分chunk ({current_toc_tokens} tokens)")

                                    # 重置
                                    current_toc_nodes = []
                                    current_toc_tokens = 0

                                # 添加当前节点
                                current_toc_nodes.append(toc_node)
                                current_toc_tokens += node_tokens

                            # 处理剩余节点
                            if current_toc_nodes:
                                merged_text_start = current_toc_nodes[0]['char_start']
                                merged_text_end = current_toc_nodes[-1]['char_end']
                                merged_text = original_text[merged_text_start:merged_text_end]

                                chunk = self._create_section_chunk(
                                    merged_text,
                                    merged_text_start,
                                    current_toc_nodes[0],
                                    base_tokens,
                                    original_text
                                )
                                final_chunks.append(chunk)
                                logger.debug(f"  ✅ 目录最后部分chunk ({current_toc_tokens} tokens)")
                        else:
                            # 未超限，作为单个ATOMIC chunk
                            chunk = self._create_section_chunk(
                                toc_full_text,
                                toc_text_start,
                                node,  # 使用目录标题节点
                                base_tokens,
                                original_text
                            )
                            final_chunks.append(chunk)
                            logger.debug(f"  ✅ 目录作为ATOMIC chunk ({toc_tokens} tokens)")

                        # 跳过所有目录相关的节点
                        while i < len(structure_tree) and (
                            structure_tree[i].get('is_toc') or structure_tree[i].get('is_toc_part')
                        ):
                            i += 1
                        continue
                    else:
                        # 没有找到目录节点，按单个标题处理
                        section_text = original_text[node['char_start']:node['char_end']]
                        estimated_tokens = len(self.tokenizer.encode(section_text))
                        chunk = self._create_section_chunk(
                            section_text,
                            node['char_start'],
                            node,
                            base_tokens,
                            original_text
                        )
                        final_chunks.append(chunk)
                        logger.debug(f"  ✅ 目录标题作为ATOMIC chunk ({estimated_tokens} tokens)")
                        i += 1
                        continue
                else:
                    # 只有目录标题，没有区域
                    section_text = original_text[node['char_start']:node['char_end']]
                    estimated_tokens = len(self.tokenizer.encode(section_text))
                    chunk = self._create_section_chunk(
                        section_text,
                        node['char_start'],
                        node,
                        base_tokens,
                        original_text
                    )
                    final_chunks.append(chunk)
                    logger.debug(f"  ✅ 目录标题作为ATOMIC chunk ({estimated_tokens} tokens)")
                    i += 1
                    continue

            # 跳过目录内容节点（已在上面处理）
            if node.get('is_toc_part', False):
                i += 1
                continue

            # 获取当前节点及其所有子节点
            section_nodes = [node]
            j = i + 1

            # 收集所有子节点
            # 策略1: 基于编号（如果有编号）
            if node['number']:
                while j < len(structure_tree):
                    next_node = structure_tree[j]
                    if next_node['number'] and next_node['number'].startswith(node['number'] + '.'):
                        section_nodes.append(next_node)
                        j += 1
                    else:
                        break
            else:
                # 策略2: 基于层级关系（无编号的父标题）
                # 收集所有层级更深的子节点，直到遇到同级或更高级的节点
                current_level = node['hierarchy_level']
                while j < len(structure_tree):
                    next_node = structure_tree[j]
                    # 如果下一个节点层级更深，说明是子节点
                    if next_node['hierarchy_level'] > current_level:
                        section_nodes.append(next_node)
                        j += 1
                    else:
                        # 遇到同级或更高级的节点，停止
                        break

            # 计算整个章节的范围
            section_start = node['char_start']
            section_end = section_nodes[-1]['char_end']
            section_text = original_text[section_start:section_end]

            # 估算token数
            estimated_tokens = len(self.tokenizer.encode(section_text))

            # 如果只有标题没有内容（token < 50），跳过这个空节点
            if len(section_nodes) == 1 and estimated_tokens < 50:
                logger.warning(f"⚠️ 跳过空标题节点: {node['title']} (仅 {estimated_tokens} tokens)")
                i = j
                continue

            logger.debug(f"处理章节: {node['number']} {node['title']} "
                        f"(层级:{node['hierarchy_level']}, 包含{len(section_nodes)}个节点, "
                        f"约{estimated_tokens} tokens)")

            # 决策：是否需要切分这个章节
            if estimated_tokens <= ChunkConfig.FINAL_MAX_TOKENS:
                # 整个章节作为一个chunk
                chunk = self._create_section_chunk(
                    section_text,
                    section_start,
                    node,
                    base_tokens,
                    original_text
                )
                final_chunks.append(chunk)
                logger.debug(f"  ✅ 整章节作为一个chunk ({estimated_tokens} tokens)")
                i = j
            else:
                # 章节过大，需要细分
                logger.debug(f"  ⚠️ 章节过大，进行子章节切分...")
                sub_chunks = self._split_large_section(
                    section_nodes,
                    original_text,
                    base_tokens
                )
                final_chunks.extend(sub_chunks)
                i = j

        logger.info(f"✅ 结构化切分完成，生成 {len(final_chunks)} 个chunks")
        return final_chunks

    def _create_section_chunk(
        self,
        section_text: str,
        char_start: int,
        node: Dict[str, Any],
        base_tokens: List[int],
        original_text: str
    ) -> Dict[str, Any]:
        """
        从章节创建chunk

        Args:
            section_text: 章节文本
            char_start: 字符起始位置
            node: 章节节点信息
            base_tokens: Token序列
            original_text: 原始文档全文

        Returns:
            Final chunk
        """
        # 编码获取token信息
        section_tokens = self.tokenizer.encode(section_text)
        token_count = len(section_tokens)

        # 使用字符位置计算token位置（基于原始文本）
        # 计算从文档开始到当前章节开始的token数量
        text_before = original_text[:char_start]
        tokens_before = self.tokenizer.encode(text_before)
        token_start = len(tokens_before)
        token_end = token_start + token_count

        # 确定是否为ATOMIC
        is_atomic = False
        atomic_type = None

        # 优先级1: 目录 (必须保持完整)
        if node.get('is_toc', False):
            is_atomic = True
            atomic_type = "toc"
        # 优先级2: 表格超过限制
        elif node['has_table'] and token_count > ChunkConfig.FINAL_MAX_TOKENS:
            is_atomic = True
            atomic_type = "table"
        # 优先级3: 步骤序列 (保持逻辑连续性)
        elif node['has_steps']:
            is_atomic = True
            atomic_type = "step"
        # 优先级4: 内容超过限制
        elif token_count > ChunkConfig.FINAL_MAX_TOKENS:
            is_atomic = True
            atomic_type = "content"

        # 生成标签
        content_tags = []
        if node['number']:
            content_tags.append(f"章节{node['number']}")
        if node.get('is_toc', False):
            content_tags.append("目录")
        if node['has_table']:
            content_tags.append("表格")
        if node['has_code']:
            content_tags.append("代码")
        if node['has_steps']:
            content_tags.append("步骤")

        return {
            "content": section_text,
            "char_start": char_start,
            "char_end": char_start + len(section_text),
            "token_start": token_start,
            "token_end": token_end,
            "token_count": token_count,
            "is_atomic": is_atomic,
            "atomic_type": atomic_type,
            "user_tag": node.get('title', ''),
            "content_tags": content_tags[:5],
            "context_tags": [f"hierarchy_level_{node['hierarchy_level']}"],
            "section_number": node.get('number'),
            "section_title": node.get('title'),
            "validation_passed": True,  # 新增：用于统计
            "validation_notes": []       # 新增：用于记录验证信息
        }

    def _split_large_section(
        self,
        section_nodes: List[Dict[str, Any]],
        original_text: str,
        base_tokens: List[int]
    ) -> List[Dict[str, Any]]:
        """
        切分过大的章节

        策略：在子章节边界切分

        Args:
            section_nodes: 章节节点列表（父节点+所有子节点）
            original_text: 原始文本
            base_tokens: Token序列

        Returns:
            Final chunks列表
        """
        chunks = []
        parent_node = section_nodes[0]

        # 按二级标题切分
        current_group = []
        current_start_node = None

        for node in section_nodes:
            # 如果是顶层节点或二级节点的开始
            if node == parent_node or (node['hierarchy_level'] == parent_node['hierarchy_level'] + 1):
                # 保存之前的组
                if current_group:
                    group_text = original_text[current_start_node['char_start']:current_group[-1]['char_end']]
                    chunk = self._create_section_chunk(
                        group_text,
                        current_start_node['char_start'],
                        current_start_node,
                        base_tokens,
                        original_text
                    )
                    chunks.append(chunk)

                # 开始新组
                current_group = [node]
                current_start_node = node
            else:
                # 添加到当前组
                current_group.append(node)

        # 处理最后一组
        if current_group and current_start_node:
            group_text = original_text[current_start_node['char_start']:current_group[-1]['char_end']]
            chunk = self._create_section_chunk(
                group_text,
                current_start_node['char_start'],
                current_start_node,
                base_tokens,
                original_text
            )
            chunks.append(chunk)

        logger.debug(f"    子章节切分: {len(section_nodes)} 个节点 → {len(chunks)} 个chunks")
        return chunks

    def _calculate_statistics(self, final_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算统计信息

        Args:
            final_chunks: Final-Chunk 列表

        Returns:
            统计信息字典
        """
        if not final_chunks:
            return {
                "total_chunks": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "atomic_chunks": 0,
                "validation_pass_rate": 0
            }

        token_counts = [c["token_count"] for c in final_chunks]
        atomic_count = sum(1 for c in final_chunks if c["is_atomic"])
        validated_count = sum(1 for c in final_chunks if c["validation_passed"])

        return {
            "total_chunks": len(final_chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "atomic_chunks": atomic_count,
            "validation_pass_rate": validated_count / len(final_chunks)
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        from tokenizer.tokenizer_client import get_tokenizer
    except ImportError:
        from ..tokenizer.tokenizer_client import get_tokenizer

    tokenizer = get_tokenizer()

    # 模拟阶段2的输出
    test_text = """# RAG 系统介绍
这是一个检索增强生成系统。它可以提高 LLM 的准确性。

## 主要特点
1. 智能检索
2. 语义匹配
3. 上下文增强
"""

    base_tokens = tokenizer.encode(test_text)
    clean_tokens = base_tokens

    stage2_result = {
        "base_tokens": base_tokens,
        "clean_chunks": [
            {
                "text": test_text,
                "token_start": 0,
                "token_end": len(base_tokens),
                "token_count": len(clean_tokens),
                "tokens": clean_tokens,
                "user_tag": "技术文档",
                "content_tags": ["RAG", "检索", "生成"],
                "validation_passed": True,
                "char_count": len(test_text)
            }
        ]
    }

    # 处理阶段3
    processor = Stage3RefineLocate()
    result = processor.process(stage2_result)

    print(f"\n=== 阶段3处理结果 ===")
    print(f"最终块数量: {result['statistics']['total_chunks']}")
    for i, chunk in enumerate(result["final_chunks"], 1):
        print(f"\nChunk {i}:")
        print(f"  Token: [{chunk['token_start']}:{chunk['token_end']}] ({chunk['token_count']})")
        print(f"  标签: {chunk['user_tag']} | {chunk['content_tags']}")
        print(f"  ATOMIC: {chunk['is_atomic']}")
        print(f"  内容: {chunk['content'][:100]}...")
