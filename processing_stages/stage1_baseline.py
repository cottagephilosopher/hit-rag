"""
阶段1: 基线建立与粗切
建立 Token 绝对索引基线，并按句子边界粗切成 Mid-Chunks
"""

import logging
import re
from typing import List, Dict, Any, Tuple

# 使用绝对导入
try:
    from tokenizer.tokenizer_client import get_tokenizer
    from tokenizer.token_mapper import TokenMapper
    from config import ChunkConfig
except ImportError:
    from ..tokenizer.tokenizer_client import get_tokenizer
    from ..tokenizer.token_mapper import TokenMapper
    from ..config import ChunkConfig

logger = logging.getLogger(__name__)


class Stage1Baseline:
    """
    阶段1: 基线建立与粗切
    """

    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.mapper = TokenMapper(self.tokenizer)

    def process(self, markdown_text: str) -> Dict[str, Any]:
        """
        处理阶段1: 建立基线并粗切

        Args:
            markdown_text: 原始 Markdown 文本

        Returns:
            包含基线和 Mid-Chunks 的字典
        """
        logger.info("=" * 60)
        logger.info("阶段1: 开始建立基线与粗切")
        logger.info("=" * 60)

        # 1. 建立 Token 绝对索引基线
        base_tokens, original_text = self.mapper.build_baseline(markdown_text)
        logger.info(f"✅ 基线建立完成: {len(base_tokens)} tokens, {len(original_text)} 字符")

        # 2. 构建文档结构树
        structure_tree = self.build_document_structure_tree(original_text)
        logger.info(f"✅ 结构树构建完成: {len(structure_tree)} 个章节")

        # 3. 粗切成 Mid-Chunks (保留用于向后兼容，但后续应使用结构树)
        mid_chunks = self._coarse_split(original_text, base_tokens)
        logger.info(f"✅ 粗切完成: {len(mid_chunks)} 个 Mid-Chunks")

        # 4. 构建结果
        result = {
            "base_tokens": base_tokens,
            "original_text": original_text,
            "structure_tree": structure_tree,  # 新增：文档结构树
            "mid_chunks": mid_chunks,
            "statistics": {
                "total_tokens": len(base_tokens),
                "total_chars": len(original_text),
                "mid_chunk_count": len(mid_chunks),
                "avg_chunk_tokens": sum(c["token_count"] for c in mid_chunks) / len(mid_chunks) if mid_chunks else 0,
                "structure_nodes": len(structure_tree)  # 新增：结构树节点数
            }
        }

        logger.info(f"\n阶段1统计:")
        logger.info(f"  总 Tokens: {result['statistics']['total_tokens']}")
        logger.info(f"  总字符数: {result['statistics']['total_chars']}")
        logger.info(f"  Mid-Chunk 数量: {result['statistics']['mid_chunk_count']}")
        logger.info(f"  平均 Chunk Tokens: {result['statistics']['avg_chunk_tokens']:.1f}")

        return result

    def _coarse_split(
        self,
        text: str,
        base_tokens: List[int]
    ) -> List[Dict[str, Any]]:
        """
        粗切文本成 Mid-Chunks

        Args:
            text: 原始文本
            base_tokens: 基线 Token 序列

        Returns:
            Mid-Chunk 列表
        """
        # 先按段落分割
        paragraphs = self._split_into_paragraphs(text)
        logger.debug(f"文本分割为 {len(paragraphs)} 个段落/块")

        mid_chunks = []
        current_chunk_paras = []
        current_chunk_chars = 0
        current_token_start = 0

        for para in paragraphs:
            para_chars = len(para["text"])
            para_type = para.get("type", "paragraph")

            # 检查单个段落是否过大
            if para_chars > ChunkConfig.MID_CHUNK_MAX_CHARS:
                # 如果当前有累积的段落，先保存
                if current_chunk_paras:
                    chunk = self._create_mid_chunk(
                        current_chunk_paras,
                        current_token_start,
                        base_tokens
                    )
                    mid_chunks.append(chunk)
                    current_token_start = chunk["token_end"]
                    current_chunk_paras = []
                    current_chunk_chars = 0

                # 特殊块（表格、代码块）不切分，作为整体保留
                if para_type in ["html_table", "markdown_table", "code_block"]:
                    logger.debug(f"⚠️ 大型特殊块 ({para_type}, {para_chars} 字符)，作为整体保留")
                    chunk = self._create_mid_chunk_from_text(
                        para["text"],
                        current_token_start,
                        base_tokens
                    )
                    mid_chunks.append(chunk)
                    current_token_start = chunk["token_end"]
                else:
                    # 普通段落：按句子切分
                    large_para_chunks = self._split_large_paragraph(
                        para["text"],
                        current_token_start,
                        base_tokens
                    )
                    mid_chunks.extend(large_para_chunks)
                    if large_para_chunks:
                        current_token_start = large_para_chunks[-1]["token_end"]

            # 检查添加当前段落是否会超过限制
            elif current_chunk_chars + para_chars > ChunkConfig.MID_CHUNK_MAX_CHARS:
                # 保存当前 chunk
                if current_chunk_paras:
                    chunk = self._create_mid_chunk(
                        current_chunk_paras,
                        current_token_start,
                        base_tokens
                    )
                    mid_chunks.append(chunk)
                    current_token_start = chunk["token_end"]

                # 开始新 chunk
                current_chunk_paras = [para]
                current_chunk_chars = para_chars

            else:
                # 添加到当前 chunk
                current_chunk_paras.append(para)
                current_chunk_chars += para_chars

        # 处理剩余的段落
        if current_chunk_paras:
            chunk = self._create_mid_chunk(
                current_chunk_paras,
                current_token_start,
                base_tokens
            )
            mid_chunks.append(chunk)

        return mid_chunks

    def _split_into_paragraphs(self, text: str) -> List[Dict[str, str]]:
        """
        将文本分割成段落和特殊块（保留格式）

        Args:
            text: 输入文本

        Returns:
            段落列表，每个元素包含 text 和 type
        """
        paragraphs = []

        # 识别特殊块（代码块、表格等）
        special_blocks = self._identify_special_blocks(text)

        if not special_blocks:
            # 没有特殊块，按双换行分割
            parts = re.split(r'\n\n+', text)
            for part in parts:
                if part.strip():
                    paragraphs.append({
                        "text": part,
                        "type": "paragraph"
                    })
            return paragraphs

        # 有特殊块，需要按顺序处理
        current_pos = 0
        for block_start, block_end, block_type in special_blocks:
            # 处理特殊块之前的普通文本
            if block_start > current_pos:
                normal_text = text[current_pos:block_start]
                parts = re.split(r'\n\n+', normal_text)
                for part in parts:
                    if part.strip():
                        paragraphs.append({
                            "text": part,
                            "type": "paragraph"
                        })

            # 添加特殊块
            paragraphs.append({
                "text": text[block_start:block_end],
                "type": block_type
            })

            current_pos = block_end

        # 处理最后一个特殊块之后的文本
        if current_pos < len(text):
            normal_text = text[current_pos:]
            parts = re.split(r'\n\n+', normal_text)
            for part in parts:
                if part.strip():
                    paragraphs.append({
                        "text": part,
                        "type": "paragraph"
                    })

        # 后处理：合并表格前的标题段落
        paragraphs = self._merge_title_with_table_paragraphs(paragraphs)

        return paragraphs

    def _merge_title_with_table_paragraphs(self, paragraphs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        合并表格前的标题段落
        
        如果检测到：paragraph + table 的模式，且 paragraph 是标题，则合并
        
        Args:
            paragraphs: 段落列表
            
        Returns:
            合并后的段落列表
        """
        if len(paragraphs) < 2:
            return paragraphs
        
        merged = []
        i = 0
        
        while i < len(paragraphs):
            current = paragraphs[i]
            
            # 检查下一个是否是表格
            if i + 1 < len(paragraphs):
                next_para = paragraphs[i + 1]
                
                # 如果当前是 paragraph，下一个是表格
                if (current["type"] == "paragraph" and 
                    next_para["type"] in ["html_table", "markdown_table"]):
                    
                    # 检查当前段落是否是标题
                    text = current["text"].strip()
                    is_title = (
                        # Markdown 标题
                        text.startswith('#') or
                        # 编号标题（如 "1.2产品规格"）
                        bool(re.match(r'^\d+\.[\d\.]*\s*.+$', text, re.MULTILINE)) or
                        # 短文本（可能是标题）
                        (len(text) < 50 and '\n' not in text)
                    )
                    
                    if is_title:
                        # 合并标题和表格
                        merged_text = current["text"] + "\n\n" + next_para["text"]
                        merged.append({
                            "text": merged_text,
                            "type": next_para["type"]  # 保持表格类型
                        })
                        logger.debug(f"✅ Stage1: 合并标题与表格: {text[:30]}...")
                        i += 2  # 跳过两个段落
                        continue
            
            # 否则保留原段落
            merged.append(current)
            i += 1
        
        return merged

    def _identify_special_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """
        识别文本中的特殊块（代码块、表格等）

        Args:
            text: 输入文本

        Returns:
            (start, end, type) 元组列表
        """
        blocks = []

        # 识别代码块
        code_pattern = r'```[\s\S]*?```'
        for match in re.finditer(code_pattern, text):
            blocks.append((match.start(), match.end(), "code_block"))

        # 识别 HTML 表格（<table>...</table>）
        html_table_pattern = r'<table[^>]*>.*?</table>'
        for match in re.finditer(html_table_pattern, text, re.DOTALL | re.IGNORECASE):
            blocks.append((match.start(), match.end(), "html_table"))

        # 识别 Markdown 表格（连续的 | ... | 行）
        table_pattern = r'(\|.+\|\s*\n)+'
        for match in re.finditer(table_pattern, text):
            blocks.append((match.start(), match.end(), "markdown_table"))

        # 按开始位置排序
        blocks.sort(key=lambda x: x[0])

        return blocks

    def _split_large_paragraph(
        self,
        para_text: str,
        token_start: int,
        base_tokens: List[int]
    ) -> List[Dict[str, Any]]:
        """
        将超大段落按句子切分

        Args:
            para_text: 段落文本
            token_start: 起始 Token 索引
            base_tokens: 基线 Token 序列

        Returns:
            Mid-Chunk 列表
        """
        # 按句子分割
        sentences = self._split_into_sentences(para_text)

        chunks = []
        current_sentences = []
        current_chars = 0

        for sentence in sentences:
            sent_chars = len(sentence)

            if current_chars + sent_chars > ChunkConfig.MID_CHUNK_MAX_CHARS and current_sentences:
                # 保存当前 chunk
                chunk_text = "".join(current_sentences)
                chunk = self._create_mid_chunk_from_text(
                    chunk_text,
                    token_start,
                    base_tokens
                )
                chunks.append(chunk)
                token_start = chunk["token_end"]

                # 开始新 chunk
                current_sentences = [sentence]
                current_chars = sent_chars
            else:
                current_sentences.append(sentence)
                current_chars += sent_chars

        # 处理剩余句子
        if current_sentences:
            chunk_text = "".join(current_sentences)
            chunk = self._create_mid_chunk_from_text(
                chunk_text,
                token_start,
                base_tokens
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 使用正则表达式识别句子结束符
        # 保留换行符作为句子边界
        sentences = []
        current_sentence = []

        # 句子结束符
        sentence_ends = r'[。！？.!?\n]'

        for char in text:
            current_sentence.append(char)
            if re.match(sentence_ends, char):
                sentences.append("".join(current_sentence))
                current_sentence = []

        # 处理剩余字符
        if current_sentence:
            sentences.append("".join(current_sentence))

        return sentences

    def _create_mid_chunk(
        self,
        paragraphs: List[Dict[str, str]],
        token_start: int,
        base_tokens: List[int]
    ) -> Dict[str, Any]:
        """
        从段落列表创建 Mid-Chunk

        Args:
            paragraphs: 段落列表
            token_start: 起始 Token 索引
            base_tokens: 基线 Token 序列

        Returns:
            Mid-Chunk 字典
        """
        chunk_text = "\n\n".join(p["text"] for p in paragraphs)
        return self._create_mid_chunk_from_text(chunk_text, token_start, base_tokens)

    def _create_mid_chunk_from_text(
        self,
        chunk_text: str,
        token_start: int,
        base_tokens: List[int]
    ) -> Dict[str, Any]:
        """
        从文本创建 Mid-Chunk

        Args:
            chunk_text: Chunk 文本
            token_start: 起始 Token 索引
            base_tokens: 基线 Token 序列

        Returns:
            Mid-Chunk 字典
        """
        # 编码 chunk 文本
        chunk_tokens = self.tokenizer.encode(chunk_text)
        token_end = token_start + len(chunk_tokens)

        # 验证 Token 范围
        is_valid = self.mapper.validate_token_range(
            token_start,
            token_end,
            base_tokens,
            chunk_text
        )

        if not is_valid:
            logger.warning(f"⚠️ Mid-Chunk Token 范围验证失败")

        # 检测标题位置
        headers = self._detect_headers(chunk_text)

        return {
            "text": chunk_text,
            "token_start": token_start,
            "token_end": token_end,
            "token_count": len(chunk_tokens),
            "char_count": len(chunk_text),
            "validation_passed": is_valid,
            "headers": headers  # 新增：标题信息
        }

    def _detect_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        检测文本中的 Markdown 标题

        Args:
            text: 输入文本

        Returns:
            标题列表，每个元素包含：
            - level: 标题级别 (1-6)
            - text: 标题文本
            - char_position: 标题在文本中的字符位置
            - char_position_ratio: 标题位置占比 (0.0-1.0)
        """
        headers = []

        # 匹配 Markdown 标题：# 开头，支持 1-6 级
        # 同时匹配编号标题如 "# 1.2 标题" 或 "## 3.1.2 标题"
        header_pattern = r'^(#{1,6})\s+(.+)$'

        lines = text.split('\n')
        current_pos = 0

        for line in lines:
            match = re.match(header_pattern, line.strip())
            if match:
                level = len(match.group(1))  # # 的数量
                header_text = match.group(2).strip()

                # 计算标题在文本中的位置
                char_position = text.find(line, current_pos)
                if char_position != -1:
                    char_position_ratio = char_position / len(text) if len(text) > 0 else 0.0

                    headers.append({
                        "level": level,
                        "text": header_text,
                        "char_position": char_position,
                        "char_position_ratio": char_position_ratio
                    })

            current_pos += len(line) + 1  # +1 for \n

        return headers

    def build_document_structure_tree(self, text: str) -> List[Dict[str, Any]]:
        """
        构建文档结构树，基于标题编号识别层级关系

        Args:
            text: 文档文本

        Returns:
            结构树节点列表，每个节点包含：
            - level: Markdown标题级别 (1-6, 即#的数量)
            - hierarchy_level: 逻辑层级 (基于编号: 1, 1.1, 1.2.1)
            - number: 编号 (如 "1", "1.2", "3.1.2")
            - title: 标题文本
            - char_start: 起始字符位置
            - char_end: 结束字符位置 (下一个同级或上级标题的位置)
            - children: 子节点列表
            - has_table: 是否包含表格
            - has_code: 是否包含代码块
            - has_steps: 是否包含步骤序列
        """
        # 提取所有标题
        header_pattern = r'^(#{1,6})\s+(.+)$'
        headers = []

        for match in re.finditer(header_pattern, text, re.MULTILINE):
            level = len(match.group(1))
            full_title = match.group(2).strip()
            char_start = match.start()

            # 尝试提取编号 (如 "1", "1.2", "3.1.2")
            number_match = re.match(r'^([\d\.]+)\s*(.*)$', full_title)
            if number_match:
                number = number_match.group(1).rstrip('.')
                title = number_match.group(2).strip()
                # 计算逻辑层级：1=顶层, 1.1=二级, 1.2.1=三级
                hierarchy_level = len(number.split('.'))
            else:
                number = None
                title = full_title
                # 无编号的标题，使用Markdown层级
                hierarchy_level = level

            headers.append({
                'level': level,
                'hierarchy_level': hierarchy_level,
                'number': number,
                'title': title,
                'full_title': full_title,
                'char_start': char_start,
                'char_end': None  # 稍后计算
            })

        # 计算每个标题的内容范围
        for i in range(len(headers)):
            if i < len(headers) - 1:
                headers[i]['char_end'] = headers[i + 1]['char_start']
            else:
                headers[i]['char_end'] = len(text)

        # 检测目录区域：从"目录"标题到正文开始的整个区域
        # 策略：识别带页码的目录条目,找到第一个不带页码的正文标题
        toc_start_idx = None
        toc_end_idx = None

        def is_toc_entry(title_text):
            """检测标题是否是目录条目(通常末尾有页码数字)"""
            # 移除标题标记
            clean = title_text.strip().lstrip('#').strip()
            # 检查是否以数字结尾(页码)
            # 例如: "安全信息 2", "产品简介 6", "产品维护 30"
            return bool(re.search(r'\s+\d+\s*$', clean))

        for i, header in enumerate(headers):
            if re.search(r'目录|contents|catalogue|table\s+of\s+contents', header['title'], re.IGNORECASE):
                toc_start_idx = i
                # 从目录标题后查找TOC条目
                for j in range(i + 1, len(headers)):
                    current_title = headers[j]['title']

                    # 检查当前标题是否是TOC条目
                    if not is_toc_entry(current_title):
                        # 不是TOC条目(没有页码),可能是正文开始
                        # 再检查内容:如果有大量正文,确认是正文章节
                        section_content = text[headers[j]['char_start']:headers[j]['char_end']]
                        content_lines = [line for line in section_content.split('\n')
                                       if line.strip() and not line.strip().startswith('#')
                                       and not re.match(r'^\s*[\d\.]+\s+', line)
                                       and not re.match(r'^\s*\d+\.\s*', line)  # 排除编号列表
                                       and len(line.strip()) > 10]  # 排除过短的行

                        if len(content_lines) > 3 or len(section_content) > 200:
                            # 有实质内容,确认TOC结束
                            toc_end_idx = j
                            logger.info(f"📑 检测到目录区域: 从标题 #{i+1} '{header['title']}' 到标题 #{j} '{headers[j]['title']}' (首个不带页码的正文标题)")
                            break
                    # 如果是TOC条目,继续检查下一个标题

                # 如果一直没找到结束位置,取最后一个TOC条目
                if toc_end_idx is None and toc_start_idx is not None:
                    # 找到最后一个带页码的标题
                    for j in range(len(headers) - 1, i, -1):
                        if is_toc_entry(headers[j]['title']):
                            toc_end_idx = j + 1  # TOC结束在最后一个条目的下一个位置
                            logger.info(f"📑 检测到目录区域: 从标题 #{i+1} '{header['title']}' 到标题 #{j+1} (最后一个TOC条目)")
                            break

                break  # 只处理第一个目录

        # 分析每个章节的内容特征
        for i, header in enumerate(headers):
            section_content = text[header['char_start']:header['char_end']]

            # 检测表格
            header['has_table'] = '<table>' in section_content or bool(re.search(r'\|.*\|.*\|', section_content))
            header['table_count'] = section_content.count('<table>') + len(re.findall(r'\n\|.*\|.*\|\n', section_content))

            # 检测代码块
            header['has_code'] = '```' in section_content

            # 检测步骤序列
            header['has_steps'] = bool(re.search(r'\(1\)|\(2\)|\(3\)|^1\.|^2\.|^3\.', section_content, re.MULTILINE))

            # 标记目录区域
            is_toc = False
            is_toc_part = False  # 是否是目录的一部分（但不是开始标题）

            if toc_start_idx is not None and toc_end_idx is not None:
                if i == toc_start_idx:
                    is_toc = True  # 目录开始标题
                elif toc_start_idx < i < toc_end_idx:
                    is_toc_part = True  # 目录内容的一部分
            elif toc_start_idx is not None and i == toc_start_idx:
                # 只有目录标题，没有找到结束位置
                is_toc = True

            header['is_toc'] = is_toc
            header['is_toc_part'] = is_toc_part
            header['toc_start_idx'] = toc_start_idx
            header['toc_end_idx'] = toc_end_idx

            # 计算内容长度
            header['content_length'] = header['char_end'] - header['char_start']

        logger.info(f"✅ 构建文档结构树: {len(headers)} 个章节")
        for h in headers[:10]:  # 只显示前10个
            logger.debug(f"  {'  ' * (h['hierarchy_level']-1)}{h['number'] or ''} {h['title']} "
                        f"({h['content_length']} chars, 表格:{h['table_count']})")

        return headers


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 测试文本
    test_md = """# 文档标题

这是第一段内容。包含多个句子。这是第三句。

## 第一节

这是第二段内容。

```python
def hello():
    print("Hello, World!")
```

这是代码块后的内容。

| 列1 | 列2 |
|-----|-----|
| A   | B   |
| C   | D   |

表格后的内容。
"""

    processor = Stage1Baseline()
    result = processor.process(test_md)

    print(f"\n=== 阶段1处理结果 ===")
    print(f"Total Tokens: {result['statistics']['total_tokens']}")
    print(f"Mid-Chunks: {result['statistics']['mid_chunk_count']}")
    print(f"\nMid-Chunks 详情:")
    for i, chunk in enumerate(result["mid_chunks"], 1):
        print(f"\nChunk {i}:")
        print(f"  Token 范围: [{chunk['token_start']}:{chunk['token_end']}]")
        print(f"  Token 数: {chunk['token_count']}")
        print(f"  字符数: {chunk['char_count']}")
        print(f"  验证: {'✅' if chunk['validation_passed'] else '❌'}")
        print(f"  内容预览: {chunk['text'][:100]}...")
