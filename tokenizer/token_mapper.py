"""
Token 映射器：处理 Token 绝对映射和索引平移逻辑
这是整个系统的核心模块，负责维护 Token 的绝对定位
"""

import logging
from typing import List, Tuple, Optional, Dict

# 使用绝对导入
try:
    from tokenizer.tokenizer_client import TokenizerClient, get_tokenizer
except ImportError:
    from .tokenizer_client import TokenizerClient, get_tokenizer

logger = logging.getLogger(__name__)


class TokenMapper:
    """
    Token 映射器
    处理 Token 的绝对定位、反向映射、子序列查找等核心功能
    """

    def __init__(self, tokenizer: Optional[TokenizerClient] = None):
        """
        初始化 Token 映射器

        Args:
            tokenizer: Tokenizer 客户端，默认使用全局单例
        """
        self.tokenizer = tokenizer or get_tokenizer()

    def build_baseline(self, original_text: str) -> Tuple[List[int], str]:
        """
        建立 Token 绝对索引基线

        Args:
            original_text: 原始文本

        Returns:
            (Token 序列, 原始文本) 元组
        """
        try:
            tokens = self.tokenizer.encode(original_text)
            logger.info(f"✅ 建立基线：共 {len(tokens)} 个 Tokens")
            return tokens, original_text
        except Exception as e:
            logger.error(f"❌ 建立基线失败: {e}")
            raise

    def reverse_map_cleaned_text(
        self,
        original_text: str,
        cleaned_text: str,
        base_token_start: int,
        base_tokens: List[int]
    ) -> Tuple[int, int, List[int]]:
        """
        反向映射：从清洗后的文本映射回原始 Token 序列的绝对索引
        这是阶段2的核心功能

        Args:
            original_text: 原始文本片段（Mid-Chunk）
            cleaned_text: 清洗后的文本
            base_token_start: 该 Mid-Chunk 在基线中的起始 Token 索引
            base_tokens: 完整的基线 Token 序列

        Returns:
            (clean_token_start, clean_token_end, clean_tokens) 元组
            - clean_token_start: 清洗后文本在基线中的起始 Token 索引
            - clean_token_end: 清洗后文本在基线中的结束 Token 索引
            - clean_tokens: 清洗后文本的 Token 序列
        """
        try:
            # 编码清洗后的文本
            clean_tokens = self.tokenizer.encode(cleaned_text)

            if not clean_tokens:
                logger.warning("⚠️ 清洗后文本为空")
                return base_token_start, base_token_start, []

            # 编码原始文本片段
            original_tokens = self.tokenizer.encode(original_text)

            # 在原始 Token 序列中查找清洗后的 Token 子序列
            # 方法1：直接子序列匹配（最精确）
            relative_start = self._find_token_subsequence(
                original_tokens,
                clean_tokens
            )

            if relative_start is not None:
                # 计算绝对索引
                clean_token_start = base_token_start + relative_start
                clean_token_end = clean_token_start + len(clean_tokens)

                logger.debug(
                    f"✅ 反向映射成功 (直接匹配): "
                    f"Token [{clean_token_start}:{clean_token_end}]"
                )
                return clean_token_start, clean_token_end, clean_tokens

            # 方法2：如果直接匹配失败，使用模糊匹配
            logger.warning("⚠️ 直接 Token 匹配失败，尝试模糊匹配")
            relative_start = self._fuzzy_find_tokens(
                original_tokens,
                clean_tokens
            )

            if relative_start is not None:
                clean_token_start = base_token_start + relative_start
                clean_token_end = clean_token_start + len(clean_tokens)

                logger.debug(
                    f"✅ 反向映射成功 (模糊匹配): "
                    f"Token [{clean_token_start}:{clean_token_end}]"
                )
                return clean_token_start, clean_token_end, clean_tokens

            # 方法3：如果仍然失败，使用文本级别的映射
            logger.warning("⚠️ Token 匹配失败，使用文本级别映射")
            return self._text_based_reverse_map(
                original_text,
                cleaned_text,
                base_token_start,
                base_tokens
            )

        except Exception as e:
            logger.error(f"❌ 反向映射失败: {e}")
            raise

    def locate_final_chunk(
        self,
        clean_chunk_text: str,
        clean_chunk_tokens: List[int],
        clean_chunk_token_start: int,
        final_chunk_text: str
    ) -> Tuple[int, int]:
        """
        定位最终切块在基线中的绝对 Token 索引
        这是阶段3的核心功能

        Args:
            clean_chunk_text: 清洗后的 Chunk 文本
            clean_chunk_tokens: 清洗后的 Chunk Token 序列
            clean_chunk_token_start: 清洗后 Chunk 在基线中的起始 Token 索引
            final_chunk_text: 最终切块文本

        Returns:
            (final_token_start, final_token_end) 元组
        """
        try:
            # 编码最终切块
            final_tokens = self.tokenizer.encode(final_chunk_text)

            if not final_tokens:
                logger.warning("⚠️ 最终切块为空")
                return clean_chunk_token_start, clean_chunk_token_start

            # 在 Clean Chunk 的 Token 序列中查找 Final Chunk 的 Token 子序列
            relative_start = self._find_token_subsequence(
                clean_chunk_tokens,
                final_tokens
            )

            if relative_start is not None:
                final_token_start = clean_chunk_token_start + relative_start
                final_token_end = final_token_start + len(final_tokens)

                logger.debug(
                    f"✅ 定位最终切块: "
                    f"Token [{final_token_start}:{final_token_end}], "
                    f"长度: {len(final_tokens)} tokens"
                )
                return final_token_start, final_token_end

            # 如果直接匹配失败，使用文本级别的定位
            logger.warning("⚠️ Token 子序列查找失败，使用文本级别定位")
            return self._text_based_locate(
                clean_chunk_text,
                clean_chunk_token_start,
                clean_chunk_tokens,
                final_chunk_text
            )

        except Exception as e:
            logger.error(f"❌ 定位最终切块失败: {e}")
            raise

    def _find_token_subsequence(
        self,
        base_tokens: List[int],
        search_tokens: List[int]
    ) -> Optional[int]:
        """
        在基础 Token 序列中查找子序列的起始位置

        Args:
            base_tokens: 基础 Token 序列
            search_tokens: 要查找的 Token 子序列

        Returns:
            起始索引，未找到返回 None
        """
        if not search_tokens or not base_tokens:
            return None

        search_len = len(search_tokens)
        base_len = len(base_tokens)

        # KMP 算法优化的子序列查找
        for i in range(base_len - search_len + 1):
            if base_tokens[i:i + search_len] == search_tokens:
                return i

        return None

    def _fuzzy_find_tokens(
        self,
        base_tokens: List[int],
        search_tokens: List[int],
        min_match_ratio: float = 0.8
    ) -> Optional[int]:
        """
        模糊查找 Token 子序列（允许部分不匹配）

        Args:
            base_tokens: 基础 Token 序列
            search_tokens: 要查找的 Token 子序列
            min_match_ratio: 最小匹配比例

        Returns:
            起始索引，未找到返回 None
        """
        if not search_tokens or not base_tokens:
            return None

        search_len = len(search_tokens)
        base_len = len(base_tokens)
        best_match_ratio = 0
        best_match_pos = None

        for i in range(base_len - search_len + 1):
            window = base_tokens[i:i + search_len]
            match_count = sum(
                1 for a, b in zip(window, search_tokens) if a == b
            )
            match_ratio = match_count / search_len

            if match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_match_pos = i

            if match_ratio >= min_match_ratio:
                return i

        # 如果最佳匹配比例达到阈值，返回最佳位置
        if best_match_ratio >= min_match_ratio:
            return best_match_pos

        return None

    def _text_based_reverse_map(
        self,
        original_text: str,
        cleaned_text: str,
        base_token_start: int,
        base_tokens: List[int]
    ) -> Tuple[int, int, List[int]]:
        """
        基于文本的反向映射（当 Token 匹配失败时使用）

        Args:
            original_text: 原始文本
            cleaned_text: 清洗后文本
            base_token_start: 基线起始索引
            base_tokens: 基线 Token 序列

        Returns:
            (start, end, tokens) 元组
        """
        # 查找清洗后文本在原始文本中的起始位置
        char_start = original_text.find(cleaned_text)

        if char_start == -1:
            # 如果找不到，尝试查找部分匹配
            logger.warning("⚠️ 文本完全匹配失败，使用首个非空白字符匹配")
            cleaned_stripped = cleaned_text.strip()
            if cleaned_stripped:
                char_start = original_text.find(cleaned_stripped[:50])

        if char_start == -1:
            logger.error("❌ 文本映射失败，使用原始位置")
            clean_tokens = self.tokenizer.encode(cleaned_text)
            return base_token_start, base_token_start + len(clean_tokens), clean_tokens

        # 计算对应的 Token 位置
        prefix_text = original_text[:char_start]
        prefix_tokens = self.tokenizer.encode(prefix_text)

        clean_tokens = self.tokenizer.encode(cleaned_text)
        clean_token_start = base_token_start + len(prefix_tokens)
        clean_token_end = clean_token_start + len(clean_tokens)

        return clean_token_start, clean_token_end, clean_tokens

    def _text_based_locate(
        self,
        clean_chunk_text: str,
        clean_chunk_token_start: int,
        clean_chunk_tokens: List[int],
        final_chunk_text: str
    ) -> Tuple[int, int]:
        """
        基于文本的定位（当 Token 子序列查找失败时使用）

        Args:
            clean_chunk_text: 清洗后 Chunk 文本
            clean_chunk_token_start: 清洗后 Chunk 起始 Token 索引
            clean_chunk_tokens: 清洗后 Chunk Token 序列
            final_chunk_text: 最终切块文本

        Returns:
            (start, end) 元组
        """
        # 查找最终切块在清洗后文本中的位置
        char_start = clean_chunk_text.find(final_chunk_text)

        if char_start == -1:
            logger.error("❌ 文本定位失败，使用清洗块起始位置")
            final_tokens = self.tokenizer.encode(final_chunk_text)
            return clean_chunk_token_start, clean_chunk_token_start + len(final_tokens)

        # 计算对应的 Token 位置
        prefix_text = clean_chunk_text[:char_start]
        prefix_tokens = self.tokenizer.encode(prefix_text)

        final_tokens = self.tokenizer.encode(final_chunk_text)
        final_token_start = clean_chunk_token_start + len(prefix_tokens)
        final_token_end = final_token_start + len(final_tokens)

        return final_token_start, final_token_end

    def validate_token_range(
        self,
        token_start: int,
        token_end: int,
        base_tokens: List[int],
        text: str
    ) -> bool:
        """
        验证 Token 范围是否正确

        Args:
            token_start: 起始 Token 索引
            token_end: 结束 Token 索引
            base_tokens: 基线 Token 序列
            text: 期望的文本

        Returns:
            是否验证通过
        """
        try:
            if token_start < 0 or token_end > len(base_tokens):
                logger.error(
                    f"❌ Token 范围越界: [{token_start}:{token_end}], "
                    f"基线长度: {len(base_tokens)}"
                )
                return False

            # 从基线中提取 Token 并解码
            extracted_tokens = base_tokens[token_start:token_end]
            extracted_text = self.tokenizer.decode(extracted_tokens)

            # 比较文本（去除首尾空白）
            if extracted_text.strip() == text.strip():
                return True

            logger.warning(
                f"⚠️ Token 范围验证失败:\n"
                f"  期望: {text[:100]}...\n"
                f"  实际: {extracted_text[:100]}..."
            )
            return False

        except Exception as e:
            logger.error(f"❌ Token 范围验证异常: {e}")
            return False


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    mapper = TokenMapper()

    # 测试1：建立基线
    original_text = """# 文档标题
这是一段正文内容。包含多个句子。

<JUNK type="页眉">公司名称 | 日期</JUNK>

这是清洗后保留的内容。
"""
    base_tokens, _ = mapper.build_baseline(original_text)
    print(f"\n基线 Token 数量: {len(base_tokens)}")

    # 测试2：反向映射
    mid_chunk_text = """这是一段正文内容。包含多个句子。

<JUNK type="页眉">公司名称 | 日期</JUNK>

这是清洗后保留的内容。"""

    cleaned_text = """这是一段正文内容。包含多个句子。

这是清洗后保留的内容。"""

    clean_start, clean_end, clean_tokens = mapper.reverse_map_cleaned_text(
        mid_chunk_text,
        cleaned_text,
        base_token_start=10,  # 假设这是第10个token开始
        base_tokens=base_tokens
    )
    print(f"\n反向映射结果: Token [{clean_start}:{clean_end}]")

    # 测试3：定位最终切块
    final_text = "这是一段正文内容。"
    final_start, final_end = mapper.locate_final_chunk(
        cleaned_text,
        clean_tokens,
        clean_start,
        final_text
    )
    print(f"\n最终定位结果: Token [{final_start}:{final_end}]")
