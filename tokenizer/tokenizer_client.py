"""
Tokenizer 客户端：统一封装 tiktoken 或其他分词器
提供 Token 编码、解码、计数等功能
"""

import tiktoken
import logging
from typing import List, Optional

# 使用绝对导入
try:
    from config import TokenizerConfig
except ImportError:
    from ..config import TokenizerConfig

logger = logging.getLogger(__name__)


class TokenizerClient:
    """
    Tokenizer 客户端
    封装 tiktoken，提供统一的 Token 操作接口
    """

    def __init__(self, encoding_name: Optional[str] = None):
        """
        初始化 Tokenizer

        Args:
            encoding_name: tiktoken 编码名称，默认使用配置中的值
        """
        self.encoding_name = encoding_name or TokenizerConfig.ENCODING_NAME
        self._encoder = None
        self._initialize_encoder()

    def _initialize_encoder(self):
        """初始化编码器"""
        try:
            self._encoder = tiktoken.get_encoding(self.encoding_name)
            logger.info(f"✅ Tokenizer 初始化成功: {self.encoding_name}")

            # 尝试获取版本（0.5.2版本没有__version__属性）
            try:
                version = tiktoken.__version__
                logger.info(f"📦 tiktoken 版本: {version}")

                # 验证版本
                if version != TokenizerConfig.TIKTOKEN_VERSION:
                    logger.warning(
                        f"⚠️ tiktoken 版本不匹配！"
                        f"当前: {version}, "
                        f"期望: {TokenizerConfig.TIKTOKEN_VERSION}"
                    )
            except AttributeError:
                # 0.5.2 及更早版本没有 __version__ 属性
                logger.info(f"📦 tiktoken 版本: 0.5.2 (或更早)")

        except Exception as e:
            logger.error(f"❌ Tokenizer 初始化失败: {e}")
            raise

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 Token 列表

        Args:
            text: 输入文本

        Returns:
            Token ID 列表
        """
        if not text:
            return []
        try:
            return self._encoder.encode(text)
        except Exception as e:
            logger.error(f"❌ 编码失败: {e}")
            raise

    def decode(self, tokens: List[int]) -> str:
        """
        将 Token 列表解码为文本

        Args:
            tokens: Token ID 列表

        Returns:
            解码后的文本
        """
        if not tokens:
            return ""
        try:
            return self._encoder.decode(tokens)
        except Exception as e:
            logger.error(f"❌ 解码失败: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        计算文本的 Token 数量

        Args:
            text: 输入文本

        Returns:
            Token 数量
        """
        return len(self.encode(text))

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        批量编码文本

        Args:
            texts: 文本列表

        Returns:
            Token ID 列表的列表
        """
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        批量解码 Token

        Args:
            token_lists: Token ID 列表的列表

        Returns:
            解码后的文本列表
        """
        return [self.decode(tokens) for tokens in token_lists]

    def get_token_at_position(self, text: str, char_position: int) -> Optional[int]:
        """
        获取指定字符位置所在的 Token 索引

        Args:
            text: 文本
            char_position: 字符位置

        Returns:
            Token 索引，如果位置无效则返回 None
        """
        if char_position < 0 or char_position > len(text):
            return None

        # 逐步编码，找到对应位置
        tokens = self.encode(text)
        current_pos = 0

        for token_idx, token_id in enumerate(tokens):
            token_text = self.decode([token_id])
            token_len = len(token_text)

            if current_pos <= char_position < current_pos + token_len:
                return token_idx

            current_pos += token_len

        return len(tokens) - 1 if tokens else None

    def get_char_to_token_mapping(self, text: str) -> List[int]:
        """
        构建字符到 Token 的映射

        Args:
            text: 文本

        Returns:
            长度为 len(text) 的列表，每个位置存储对应的 Token 索引
        """
        tokens = self.encode(text)
        char_to_token = []
        current_token_idx = 0

        for token_id in tokens:
            token_text = self.decode([token_id])
            for _ in token_text:
                char_to_token.append(current_token_idx)
            current_token_idx += 1

        return char_to_token

    def get_token_to_char_mapping(self, text: str) -> List[tuple]:
        """
        构建 Token 到字符位置的映射

        Args:
            text: 文本

        Returns:
            列表，每个元素为 (start_char, end_char) 元组
        """
        tokens = self.encode(text)
        token_to_char = []
        current_pos = 0

        for token_id in tokens:
            token_text = self.decode([token_id])
            token_len = len(token_text)
            token_to_char.append((current_pos, current_pos + token_len))
            current_pos += token_len

        return token_to_char

    def split_text_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0
    ) -> List[str]:
        """
        按照 Token 数量切分文本

        Args:
            text: 输入文本
            max_tokens: 每块最大 Token 数
            overlap_tokens: 重叠 Token 数

        Returns:
            切分后的文本块列表
        """
        tokens = self.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.decode(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break

            start = end - overlap_tokens

        return chunks

    def find_token_sequence(
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
            起始 Token 索引，未找到返回 None
        """
        if not search_tokens:
            return None

        search_len = len(search_tokens)
        base_len = len(base_tokens)

        for i in range(base_len - search_len + 1):
            if base_tokens[i:i + search_len] == search_tokens:
                return i

        return None

    def get_info(self) -> dict:
        """
        获取 Tokenizer 信息

        Returns:
            包含编码名称、版本等信息的字典
        """
        # 尝试获取版本
        try:
            current_version = tiktoken.__version__
        except AttributeError:
            current_version = "0.5.2 (或更早)"

        return {
            "encoding_name": self.encoding_name,
            "tiktoken_version": current_version,
            "expected_version": TokenizerConfig.TIKTOKEN_VERSION,
            "version_match": str(current_version) == TokenizerConfig.TIKTOKEN_VERSION
        }


# 全局单例
_global_tokenizer = None


def get_tokenizer() -> TokenizerClient:
    """
    获取全局 Tokenizer 单例

    Returns:
        TokenizerClient 实例
    """
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = TokenizerClient()
    return _global_tokenizer


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    tokenizer = TokenizerClient()

    # 测试基本功能
    test_text = "这是一个测试文本。This is a test."
    print(f"\n测试文本: {test_text}")

    # 编码
    tokens = tokenizer.encode(test_text)
    print(f"Token 数量: {len(tokens)}")
    print(f"Tokens: {tokens}")

    # 解码
    decoded = tokenizer.decode(tokens)
    print(f"解码文本: {decoded}")
    print(f"解码正确: {decoded == test_text}")

    # Token 计数
    count = tokenizer.count_tokens(test_text)
    print(f"\nToken 计数: {count}")

    # 字符到 Token 映射
    char_to_token = tokenizer.get_char_to_token_mapping(test_text)
    print(f"\n字符到 Token 映射 (前20个字符): {char_to_token[:20]}")

    # Token 到字符映射
    token_to_char = tokenizer.get_token_to_char_mapping(test_text)
    print(f"\nToken 到字符映射: {token_to_char}")

    # 信息
    info = tokenizer.get_info()
    print(f"\nTokenizer 信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
