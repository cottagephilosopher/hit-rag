# 快速配置指南

## 1. 环境初始化

### 安装依赖
```bash
# 使用 uv 同步依赖
uv sync


## 2. 环境变量配置

### 复制配置模板
```bash
cp env.template .env
```

### 编辑 .env 文件

#### 2.1 路径配置（可选）

如果你的项目结构与默认不同，需要配置以下路径：

```bash
# 基础目录：项目根目录
# 默认：当前项目的父目录的父目录
BASE_DIR=/path/to/your/rag_preprocessor

# Markdown 文档目录：存放所有待处理的 .md 文件
# 默认：${BASE_DIR}/all-md
ALL_MD_DIR=/path/to/your/all-md

# 输出目录：存放处理后的 JSON 文件
# 默认：${BASE_DIR}/rag-visualizer/public/output
OUTPUT_DIR=/path/to/your/output

# 项目工作目录：当前项目目录
# 默认：当前文件所在目录
IKN_PLUS_DIR=/path/to/your/hit-rag
```

**说明**：
- 如果不配置，系统会使用默认路径
- 默认路径适用于标准项目结构
- 建议只在路径不同时才配置

#### 2.2 LLM API 配置（必填）

**使用 Azure OpenAI：**
```bash
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

**或使用 OpenAI：**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4-turbo-preview
```

#### 2.3 向量化配置（可选）

**使用 Ollama（推荐本地开发）：**
```bash
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
```

**或使用 Azure OpenAI：**
```bash
EMBEDDING_PROVIDER=azure
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
```

## 3. 验证配置

运行配置验证：
```bash
uv run python config.py
```

如果配置正确，会显示：
```
✅ 配置验证通过

配置摘要:
{
  "llm_provider": "azure",
  "tokenizer": "cl100k_base",
  "chunk_params": {
    "mid_chunk_max_chars": 1536,
    "final_min_tokens": 300,
    "final_max_tokens": 2000
  },
  "tag_count": 10,
  "junk_types": 10
}
```

## 4. 启动服务

### 启动 API 服务器
```bash
uv run python api_server.py
```

服务器将在 `http://localhost:8000` 启动。

### 运行文档处理
```bash
uv run python main.py input.md
```

## 5. 常见问题

### Q: 路径配置不生效？
**A:** 确保 `.env` 文件在项目根目录，并且已经使用 `load_dotenv()` 加载。

### Q: 找不到 .env 文件？
**A:** 运行 `cp env.template .env` 创建配置文件。

### Q: API 调用失败？
**A:** 检查：
1. API 密钥是否正确
2. 网络连接是否正常
3. LLM_PROVIDER 是否匹配（azure/openai）

### Q: Milvus 连接失败？
**A:** 确保 Milvus 服务已启动：
```bash
# 使用 Docker 启动 Milvus
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
```

### Q: Ollama 模型未找到？
**A:** 先拉取模型：
```bash
ollama pull qwen3-embedding:latest
```

## 6. 目录结构示例

标准项目结构：
```
rag_preprocessor/
├── all-md/              # Markdown 文档目录
│   ├── doc1.md
│   └── doc2.md
├── hit-rag/            # 当前项目
│   ├── .env            # 环境配置（不提交到 git）
│   ├── env.template    # 配置模板
│   ├── config.py
│   ├── api_server.py
│   └── ...
└── rag-visualizer/
    └── public/
        └── output/      # 输出目录
            ├── doc1_final_chunks.json
            └── doc2_final_chunks.json
```

如果你的结构不同，请在 `.env` 中配置相应路径。

