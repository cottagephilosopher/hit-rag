"""
配置文件：RAG 文档预处理系统
包含 LLM 参数、Tokenizer 版本、杂质特征、标签列表等配置
"""

import os

# ==================== LLM API 配置 ====================
class LLMConfig:
    """LLM API 相关配置"""
    # 使用哪个提供商：'azure' 或 'openai'
    PROVIDER = os.getenv("LLM_PROVIDER", "azure")

    # Azure OpenAI 配置
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

    # OpenAI 配置（可选）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

    # DashScope 配置（阿里云灵积）
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-max")

    # API 调用参数
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
    MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("LLM_RETRY_DELAY", "2"))


# ==================== Tokenizer 配置 ====================
class TokenizerConfig:
    """Tokenizer 版本锁定配置"""
    # 使用的 Tokenizer 类型
    TOKENIZER_TYPE = os.getenv("TOKENIZER_TYPE", "tiktoken")

    # tiktoken 编码名称（必须与 LLM 模型匹配）
    # GPT-4: "cl100k_base", GPT-3.5: "cl100k_base"
    ENCODING_NAME = os.getenv("TOKENIZER_ENCODING_NAME", "cl100k_base")

    # tiktoken 版本锁定（在 requirements.txt 中严格指定）
    TIKTOKEN_VERSION = os.getenv("TIKTOKEN_VERSION", "0.5.2")

    # 缓存设置
    CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


# ==================== Token 切分参数 ====================
class ChunkConfig:
    """文档切分相关配置"""
    # 阶段1：粗切参数
    MID_CHUNK_MAX_CHARS = int(os.getenv("MID_CHUNK_MAX_CHARS", "1536"))
    MID_CHUNK_OVERLAP_CHARS = int(os.getenv("MID_CHUNK_OVERLAP_CHARS", "100"))

    # 阶段3：最终切分 Token 限制（新策略：语义完整性优先）
    FINAL_MIN_TOKENS = int(os.getenv("FINAL_MIN_TOKENS", "300"))
    FINAL_TARGET_TOKENS = int(os.getenv("FINAL_TARGET_TOKENS", "800"))
    FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS", "2000"))
    FINAL_HARD_LIMIT = int(os.getenv("FINAL_HARD_LIMIT", "3000"))

    # ATOMIC 块特殊处理
    ATOMIC_MAX_TOKENS = int(os.getenv("ATOMIC_MAX_TOKENS", "4000"))

    # 语义完整性优先
    SEMANTIC_INTEGRITY_PRIORITY = os.getenv("SEMANTIC_INTEGRITY_PRIORITY", "true").lower() == "true"

    # 小 chunk 合并配置
    ENABLE_SMALL_CHUNK_MERGE = os.getenv("ENABLE_SMALL_CHUNK_MERGE", "true").lower() == "true"
    SMALL_CHUNK_THRESHOLD = int(os.getenv("SMALL_CHUNK_THRESHOLD", "100"))

    # Markdown 结构保持
    PRESERVE_MARKDOWN_STRUCTURE = os.getenv("PRESERVE_MARKDOWN_STRUCTURE", "true").lower() == "true"

    # 句子完整性正则表达式
    SENTENCE_END_PATTERN = r'[。！？\.!?\n][\s]*$'

    # 特殊结构识别模式
    TABLE_PATTERN = r'\|.*\|'  # Markdown 表格
    CODE_BLOCK_PATTERN = r'```.*?```'  # 代码块
    LINK_LIST_PATTERN = r'(\[.*?\]\(.*?\)\s*){3,}'  # 连续链接列表


# ==================== 杂质特征列表 ====================
class JunkPatterns:
    """定义需要识别和清除的文档杂质特征"""

    JUNK_TYPES = [
        "页眉 (Page Header)",
        "页脚 (Page Footer)",
        "页码 (Page Number)",
        "版权声明 (Copyright Notice)",
        "警告信息 (Warning/Alert)",
        "导航链接 (Navigation Links)",
        "目录 (Table of Contents)",
        "免责声明 (Disclaimer)",
        "广告 (Advertisement)",
        "重复内容 (Duplicate Content)"
    ]

    # 杂质识别提示词特征描述
    JUNK_FEATURES = {
        "页眉": "出现在文档顶部，包含公司名称、文档标题、日期等重复信息",
        "页脚": "出现在文档底部，包含版权、联系方式、网址等",
        "页码": "单独的数字或'第X页'类型的内容",
        "版权声明": "包含 ©, Copyright, All Rights Reserved 等",
        "警告信息": "⚠️, Warning, Note, Tip 等提示框内容（注：不包括步骤说明中的'注：'）",
        "导航链接": "首页、返回、下一页等导航元素",
        "目录": "仅包含章节标题和页码的列表",
        "免责声明": "法律声明、使用条款等",
        "广告": "推广、促销、广告相关内容",
        "重复内容": "在文档中多次重复出现的完全相同的段落"
    }


# ==================== 标签配置 ====================
class TagConfig:
    """文档标签相关配置"""

    # 用户定义标签列表（预设类型）
    USER_DEFINED_TAGS = [
        "技术文档",
        "API文档",
        "用户手册",
        "开发指南",
        "市场报告",
        "产品说明",
        "研究论文",
        "教程文档",
        "配置说明",
        "故障排查"
    ]

    # 内容推理标签数量
    CONTENT_TAG_COUNT = int(os.getenv("CONTENT_TAG_COUNT", "5"))

    # 标签推理语言
    TAG_LANGUAGE = os.getenv("TAG_LANGUAGE", "中文")


# ==================== 日志配置 ====================
class LogConfig:
    """日志记录配置"""

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE = os.getenv("LOG_FILE", "logs/rag_preprocessor.log")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 详细日志记录选项
    LOG_TOKEN_DETAILS = os.getenv("LOG_TOKEN_DETAILS", "true").lower() == "true"
    LOG_LLM_REQUESTS = os.getenv("LOG_LLM_REQUESTS", "true").lower() == "true"


# ==================== 输出配置 ====================
class OutputConfig:
    """输出文件配置"""

    OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output"))

    # 输出文件格式
    FINAL_OUTPUT_FILE = "final_chunks.json"

    # 中间结果保存（用于调试）
    SAVE_INTERMEDIATE_RESULTS = os.getenv("SAVE_INTERMEDIATE_RESULTS", "true").lower() == "true"
    STAGE1_OUTPUT = "stage1_mid_chunks.json"
    STAGE2_OUTPUT = "stage2_clean_chunks.json"

    # JSON 格式化
    JSON_INDENT = int(os.getenv("JSON_INDENT", "2"))
    JSON_ENSURE_ASCII = False


# ==================== 性能优化配置 ====================
class PerformanceConfig:
    """性能优化相关配置"""

    # 批量处理
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))

    # 异步处理
    ENABLE_ASYNC = os.getenv("ENABLE_ASYNC", "true").lower() == "true"
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))

    # 缓存
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))


# ==================== 验证配置 ====================
class ValidationConfig:
    """数据验证配置"""

    # Token 溢出检查
    CHECK_TOKEN_OVERFLOW = os.getenv("CHECK_TOKEN_OVERFLOW", "true").lower() == "true"
    STRICT_MODE = os.getenv("STRICT_MODE", "false").lower() == "true"

    # Token 连续性检查
    CHECK_TOKEN_CONTINUITY = os.getenv("CHECK_TOKEN_CONTINUITY", "true").lower() == "true"
    TOKEN_GAP_THRESHOLD = int(os.getenv("TOKEN_GAP_THRESHOLD", "50"))

    # 完整性检查
    CHECK_SENTENCE_INTEGRITY = os.getenv("CHECK_SENTENCE_INTEGRITY", "true").lower() == "true"
    CHECK_TABLE_INTEGRITY = os.getenv("CHECK_TABLE_INTEGRITY", "true").lower() == "true"
    CHECK_CODE_BLOCK_INTEGRITY = os.getenv("CHECK_CODE_BLOCK_INTEGRITY", "true").lower() == "true"


# ==================== 向量化配置 ====================
class VectorConfig:
    """向量化配置"""

    # Milvus 连接配置
    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "knowledges")

    # Embedding 配置
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")  # ollama | azure | openai

    # Ollama 配置
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:latest")

    # Azure OpenAI 配置
    AZURE_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

    # Embedding 维度映射（根据模型自动选择）
    EMBEDDING_DIMENSIONS = {
        # Ollama 模型
        "qwen3-embedding:latest": 4096,
        "qwen2.5:latest": 3584,
        "nomic-embed-text": 768,
        "all-minilm": 384,
        # Azure/OpenAI 模型
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    # 自动获取当前模型的维度
    @classmethod
    def get_embedding_dimension(cls):
        """根据当前配置的 embedding 模型返回对应的维度"""
        if cls.EMBEDDING_PROVIDER == "ollama":
            model = cls.OLLAMA_EMBEDDING_MODEL
        elif cls.EMBEDDING_PROVIDER == "azure":
            model = cls.AZURE_EMBEDDING_MODEL
        else:
            model = "text-embedding-ada-002"  # 默认

        # 从映射表获取维度
        dimension = cls.EMBEDDING_DIMENSIONS.get(model)

        if dimension is None:
            # 如果找不到，返回默认值并警告
            print(f"⚠️  Warning: Unknown embedding model '{model}', using default dimension 1536")
            return 1536

        return dimension

    # 当前 embedding 维度（动态获取）
    EMBEDDING_DIMENSION = property(lambda self: VectorConfig.get_embedding_dimension())

    # 批处理配置
    BATCH_SIZE = int(os.getenv("VECTOR_BATCH_SIZE", "20"))
    MAX_RETRIES = int(os.getenv("VECTOR_MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("VECTOR_RETRY_DELAY", "2"))

    # 索引配置
    INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")  # FLAT | IVF_FLAT | HNSW
    METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "L2")   # L2 | IP | COSINE

    # HNSW 索引参数
    HNSW_M = int(os.getenv("MILVUS_HNSW_M", "16"))
    HNSW_EF_CONSTRUCTION = int(os.getenv("MILVUS_HNSW_EF_CONSTRUCTION", "200"))

    # 搜索配置
    DEFAULT_TOP_K = int(os.getenv("VECTOR_DEFAULT_TOP_K", "5"))
    SEARCH_EF = int(os.getenv("MILVUS_SEARCH_EF", "64"))

    # 向量化策略
    AUTO_VECTORIZE = os.getenv("VECTOR_AUTO_VECTORIZE", "false").lower() == "true"
    SKIP_DEPRECATED = os.getenv("VECTOR_SKIP_DEPRECATED", "true").lower() == "true"
    SKIP_VECTORIZED = os.getenv("VECTOR_SKIP_VECTORIZED", "true").lower() == "true"


# ==================== 工具函数 ====================
def validate_config():
    """验证配置完整性"""
    errors = []

    if LLMConfig.PROVIDER == "azure":
        if not LLMConfig.AZURE_OPENAI_API_KEY:
            errors.append("未设置 AZURE_OPENAI_API_KEY 环境变量")
        if not LLMConfig.AZURE_OPENAI_ENDPOINT:
            errors.append("未设置 AZURE_OPENAI_ENDPOINT 环境变量")
    elif LLMConfig.PROVIDER == "openai":
        if not LLMConfig.OPENAI_API_KEY:
            errors.append("未设置 OPENAI_API_KEY 环境变量")
    elif LLMConfig.PROVIDER == "dashscope":
        if not LLMConfig.DASHSCOPE_API_KEY:
            errors.append("未设置 DASHSCOPE_API_KEY 环境变量")
    else:
        errors.append(f"不支持的 LLM_PROVIDER: {LLMConfig.PROVIDER}")

    if errors:
        raise ValueError(f"配置验证失败:\n" + "\n".join(errors))

    return True


def get_config_summary():
    """获取配置摘要信息"""
    return {
        "llm_provider": LLMConfig.PROVIDER,
        "tokenizer": TokenizerConfig.ENCODING_NAME,
        "chunk_params": {
            "mid_chunk_max_chars": ChunkConfig.MID_CHUNK_MAX_CHARS,
            "final_min_tokens": ChunkConfig.FINAL_MIN_TOKENS,
            "final_max_tokens": ChunkConfig.FINAL_MAX_TOKENS,
        },
        "tag_count": len(TagConfig.USER_DEFINED_TAGS),
        "junk_types": len(JunkPatterns.JUNK_TYPES)
    }


if __name__ == "__main__":
    # 测试配置
    try:
        validate_config()
        print("✅ 配置验证通过")
        print("\n配置摘要:")
        import json
        print(json.dumps(get_config_summary(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ 配置错误: {e}")
