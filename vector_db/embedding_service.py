"""
Embedding 服务
支持 Azure OpenAI Embeddings、Ollama 本地 Embeddings 和 DashScope Embeddings
"""

import os
import logging
import requests
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from openai import OpenAI

logger = logging.getLogger(__name__)


class CustomOllamaEmbeddings(Embeddings):
    """
    自定义 Ollama Embeddings 实现
    使用 requests 直接调用 Ollama API，避免 langchain_ollama 的兼容性问题
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.embed_url = f"{self.base_url}/api/embed"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档 embeddings"""
        if not texts:
            return []

        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """生成单个查询的 embedding（带重试机制）"""
        import time
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.embed_url,
                    json={"model": self.model, "input": text},
                    timeout=60  # 增加超时时间
                )
                response.raise_for_status()
                result = response.json()
                return result["embeddings"][0]
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503 and attempt < max_retries - 1:
                    # 503 Service Unavailable - 重试
                    logger.warning(f"Ollama service unavailable (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                    continue
                logger.error(f"Failed to generate embedding: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                raise

        raise Exception("Failed to generate embedding after multiple retries")


class DashScopeEmbeddings(Embeddings):
    """
    DashScope (阿里云百炼) Embeddings 实现
    使用 OpenAI 兼容 API
    """

    def __init__(self, api_key: str, model: str = "text-embedding-v4", dimensions: int = 1024):
        import httpx

        # 创建不使用代理的 http client（避免 SOCKS 代理问题）
        http_client = httpx.Client(
            proxy=None,  # 禁用代理
            timeout=60.0
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=http_client
        )
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档 embeddings"""
        if not texts:
            return []

        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """生成单个查询的 embedding"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate DashScope embedding: {e}")
            raise


class EmbeddingService:
    """
    Embedding 服务封装
    支持 Azure OpenAI、Ollama 和 DashScope 三种 provider
    """

    def __init__(self):
        """初始化 Embeddings（根据环境变量选择 provider）"""
        # 读取 provider 配置
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

        if embedding_provider == "ollama":
            self._init_ollama()
        elif embedding_provider == "azure":
            self._init_azure()
        elif embedding_provider == "dashscope":
            self._init_dashscope()
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER: {embedding_provider}. "
                "Supported values: 'ollama', 'azure', 'dashscope'"
            )

    def _init_ollama(self):
        """初始化 Ollama Embeddings"""
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:latest")

        logger.info(f"Initializing Ollama Embeddings: {ollama_model} at {ollama_base_url}")

        try:
            self.embeddings = CustomOllamaEmbeddings(
                model=ollama_model,
                base_url=ollama_base_url,
            )

            # 测试连接并获取维度
            test_embedding = self.embeddings.embed_query("测试")
            self.embedding_dimension = len(test_embedding)

            logger.info(f"✅ Ollama Embeddings initialized successfully (dimension: {self.embedding_dimension})")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama Embeddings: {e}")
            raise ValueError(
                f"Failed to connect to Ollama at {ollama_base_url}. "
                "Please ensure Ollama is running and the model is available. "
                f"Error: {e}"
            )

    def _init_azure(self):
        """初始化 Azure OpenAI Embeddings"""
        embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        if not azure_endpoint or not api_key:
            raise ValueError(
                "Missing Azure OpenAI configuration. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env file"
            )

        logger.info(f"Initializing Azure OpenAI Embeddings: {embedding_deployment}")

        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        self.embedding_dimension = 1536  # text-embedding-ada-002 的维度

        logger.info(f"✅ Azure OpenAI Embeddings initialized successfully")

    def _init_dashscope(self):
        """初始化 DashScope Embeddings"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
        dimensions = int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "1024"))

        if not api_key:
            raise ValueError(
                "Missing DashScope configuration. "
                "Please set DASHSCOPE_API_KEY in .env file"
            )

        logger.info(f"Initializing DashScope Embeddings: {model} (dimensions: {dimensions})")

        try:
            self.embeddings = DashScopeEmbeddings(
                api_key=api_key,
                model=model,
                dimensions=dimensions
            )

            # 测试连接并获取维度
            test_embedding = self.embeddings.embed_query("测试")
            self.embedding_dimension = len(test_embedding)

            logger.info(f"✅ DashScope Embeddings initialized successfully (dimension: {self.embedding_dimension})")

        except Exception as e:
            logger.error(f"❌ Failed to initialize DashScope Embeddings: {e}")
            raise ValueError(
                f"Failed to connect to DashScope. "
                "Please ensure DASHSCOPE_API_KEY is valid. "
                f"Error: {e}"
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文档 embeddings

        Args:
            texts: 文本列表

        Returns:
            embedding 向量列表

        Note:
            LangChain 会自动处理批次大小和重试逻辑
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} documents...")

        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"Successfully embedded {len(embeddings)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        生成查询向量

        Args:
            text: 查询文本

        Returns:
            embedding 向量
        """
        if not text:
            raise ValueError("Query text cannot be empty")

        logger.debug(f"Embedding query: {text[:50]}...")

        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug("Successfully embedded query")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise

    def get_dimension(self) -> int:
        """获取 embedding 维度"""
        return self.embedding_dimension


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    try:
        service = EmbeddingService()

        # 测试单个查询
        query_embedding = service.embed_query("测试查询文本")
        print(f"Query embedding dimension: {len(query_embedding)}")
        print(f"First 5 values: {query_embedding[:5]}")

        # 测试批量文档
        docs = ["文档1", "文档2", "文档3"]
        doc_embeddings = service.embed_documents(docs)
        print(f"\nEmbedded {len(doc_embeddings)} documents")
        print(f"Each embedding dimension: {len(doc_embeddings[0])}")

        print("\n✅ Embedding service test passed!")

    except Exception as e:
        print(f"\n❌ Embedding service test failed: {e}")
