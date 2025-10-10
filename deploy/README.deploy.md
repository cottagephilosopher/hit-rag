# RAG 智能问答系统 - 部署文档

## 📋 目录

- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [手动部署](#手动部署)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [维护指南](#维护指南)

## 🖥 系统要求

### 硬件要求
- CPU: 4核心或以上
- 内存: 8GB 或以上（推荐 16GB）
- 硬盘: 20GB 可用空间

### 软件要求
- Docker 20.10+ 或 Docker Desktop
- Docker Compose 2.0+
- Git 2.30+
- curl（用于健康检查）

### 操作系统
- macOS 10.15+
- Linux（Ubuntu 20.04+, CentOS 8+）
- Windows 10/11 with WSL2

## 🚀 快速开始

### 一键部署

1. **克隆项目**（如果还没有）
   ```bash
   git clone https://github.com/your-org/hit-rag.git
   cd hit-rag
   ```

2. **运行部署脚本**
   ```bash
   chmod +x deploy/deploy.sh
   ./deploy/deploy.sh
   ```

3. **按照提示完成配置**
   - 选择 LLM 提供商（Azure OpenAI 或 OpenAI）
   - 输入 API 密钥
   - 选择 Embedding 提供商
   - 确认文档目录路径

4. **等待部署完成**

   脚本会自动：
   - 配置环境变量
   - 创建必要目录
   - 启动 Docker 服务
   - 验证系统健康

5. **访问系统**
   - 文档管理界面: http://localhost:5173
   - 聊天问答界面: http://localhost:3000
   - 后端 API: http://localhost:8000

## 🔧 手动部署

### 步骤 1: 准备环境

1. **克隆所有项目**
   ```bash
   mkdir -p ~/rags
   cd ~/rags

   # 克隆三个项目
   git clone https://github.com/your-org/hit-rag.git
   git clone https://github.com/your-org/hit-rag-ui.git
   git clone https://github.com/ConcealedGem/versa-chat-view.git
   ```

2. **创建文档目录**
   ```bash
   mkdir -p ~/rags/all-md
   ```

### 步骤 2: 配置环境变量

1. **复制环境配置模板**
   ```bash
   cd ~/rags/hit-rag
   cp env.template .env
   ```

2. **编辑 .env 文件**
   ```bash
   nano .env  # 或使用你喜欢的编辑器
   ```

3. **必须配置的项**
   ```env
   # LLM 配置
   LLM_PROVIDER=azure  # 或 openai
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=gpt-4

   # Embedding 配置
   EMBEDDING_PROVIDER=ollama  # 或 azure/openai

   # 文档目录
   ALL_MD_DIR=/path/to/your/all-md

   # Milvus 配置
   MILVUS_HOST=127.0.0.1
   MILVUS_PORT=19530
   ```

### 步骤 3: 创建软链接

```bash
cd ~/rags/hit-rag
ln -s ~/rags/all-md ./all-md
```

### 步骤 4: 启动服务

```bash
cd ~/rags/hit-rag

# 创建必要目录
mkdir -p volumes/milvus volumes/output volumes/db

# 启动所有服务
docker compose up -d --build

# 查看日志
docker compose logs -f
```

### 步骤 5: 验证部署

```bash
# 检查服务状态
docker compose ps

# 测试 Milvus
curl http://localhost:9091/healthz

# 测试后端 API
curl http://localhost:8000/api/assistants

# 测试前端
curl http://localhost:5173

# 测试聊天界面
curl http://localhost:3000
```

## ⚙️ 配置说明

### 端口配置

默认端口映射：

| 服务 | 容器端口 | 主机端口 | 说明 |
|------|---------|---------|------|
| Milvus | 19530 | 19530 | 向量数据库主端口 |
| Milvus | 9091 | 9091 | 健康检查/监控端口 |
| Backend | 8000 | 8000 | FastAPI 后端服务 |
| Frontend | 5173 | 5173 | Vue.js 前端界面 |
| Chat View | 3000 | 3000 | Next.js 聊天界面 |

如需修改端口，编辑 `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8001:8000"  # 修改主机端口为 8001
```

### 环境变量详解

#### LLM 配置

```env
# 选择 LLM 提供商
LLM_PROVIDER=azure  # 可选: azure, openai

# Azure OpenAI（推荐用于企业）
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# OpenAI（需要国际网络）
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4-turbo-preview

# LLM 参数
LLM_TEMPERATURE=0.1        # 温度参数，0-1
LLM_MAX_TOKENS=4000        # 最大 token 数
LLM_TIMEOUT=120            # 超时时间（秒）
```

#### Embedding 配置

```env
# 选择 Embedding 提供商
EMBEDDING_PROVIDER=ollama  # 可选: ollama, azure, openai

# Ollama（本地部署，免费）
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest

# Azure OpenAI Embedding
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# OpenAI Embedding
# 使用 OPENAI_API_KEY
```

#### Milvus 配置

```env
MILVUS_HOST=127.0.0.1      # Docker 内部使用 'milvus'
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=knowledge

# 索引配置
MILVUS_INDEX_TYPE=HNSW
MILVUS_METRIC_TYPE=L2      # L2距离，越小越相似

# 搜索配置
VECTOR_DEFAULT_TOP_K=5
```

### 文档目录结构

```
all-md/
├── document1.md
├── document2.md
├── subfolder/
│   ├── document3.md
│   └── document4.md
└── ...
```

- 支持任意深度的子目录
- 自动递归扫描所有 `.md` 文件
- 文件名将作为文档名显示

### 数据持久化

Docker 卷映射：

```yaml
volumes:
  - ./volumes/milvus:/var/lib/milvus      # Milvus 数据
  - ./volumes/output:/app/output           # 处理后的文档
  - ./volumes/db:/app/db                   # SQLite 数据库
  - ./all-md:/app/all-md                   # 源文档目录
```

数据保存在 `volumes/` 目录下，即使删除容器也不会丢失。

## 🐛 常见问题

### 问题 1: Docker 启动失败

**症状**: `docker compose up` 失败

**解决方案**:
```bash
# 检查 Docker 是否运行
docker info

# 检查端口是否被占用
lsof -i :8000
lsof -i :19530

# 清理旧容器
docker compose down
docker system prune -a
```

### 问题 2: Milvus 连接失败

**症状**: 后端日志显示 "Failed to connect to Milvus"

**解决方案**:
```bash
# 检查 Milvus 状态
docker compose ps milvus
docker compose logs milvus

# 重启 Milvus
docker compose restart milvus

# 等待 Milvus 完全启动（约60秒）
curl http://localhost:9091/healthz
```

### 问题 3: 前端无法连接后端

**症状**: 前端页面空白或显示网络错误

**解决方案**:
```bash
# 检查后端是否正常
curl http://localhost:8000/api/assistants

# 检查环境变量
docker compose exec frontend env | grep VITE_API_BASE_URL

# 重启前端
docker compose restart frontend
```

### 问题 4: Ollama 模型下载失败

**症状**: Embedding 服务报错

**解决方案**:
```bash
# 安装 Ollama（如果还没有）
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# 启动 Ollama
ollama serve &

# 拉取 Embedding 模型
ollama pull qwen3-embedding:latest

# 验证模型
ollama list
```

### 问题 5: 文档上传后无法搜索

**症状**: 文档已上传但搜索无结果

**解决方案**:
```bash
# 1. 检查文档是否已处理
curl http://localhost:8000/api/documents

# 2. 检查 chunks 是否生成
curl http://localhost:8000/api/documents/{document_name}/chunks

# 3. 检查是否已向量化
curl http://localhost:8000/api/vectorization/stats

# 4. 手动触发向量化
curl -X POST http://localhost:8000/api/chunks/vectorize/batch \
  -H "Content-Type: application/json" \
  -d '{"chunk_ids": [1,2,3...]}'
```

### 问题 6: API 响应慢

**症状**: 查询响应时间超过10秒

**解决方案**:
```bash
# 1. 检查资源使用
docker stats

# 2. 调整 Docker 资源限制
# Docker Desktop -> Settings -> Resources
# CPU: 4核心
# Memory: 8GB

# 3. 优化 Milvus 索引参数（.env）
MILVUS_HNSW_M=16           # 降低以减少内存
MILVUS_SEARCH_EF=64        # 降低以加快搜索

# 4. 使用更快的 LLM（如 gpt-3.5-turbo）
AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo
```

## 🔄 维护指南

### 查看日志

```bash
# 所有服务
docker compose logs -f

# 特定服务
docker compose logs -f backend
docker compose logs -f milvus
docker compose logs -f frontend

# 最近100行
docker compose logs --tail=100 backend
```

### 重启服务

```bash
# 重启所有服务
docker compose restart

# 重启特定服务
docker compose restart backend
docker compose restart milvus
```

### 更新代码

```bash
# 拉取最新代码
cd ~/rags/hit-rag
git pull

cd ~/rags/hit-rag-ui
git pull

cd ~/rags/versa-chat-view
git pull

# 重新构建并启动
cd ~/rags/hit-rag
docker compose down
docker compose up -d --build
```

### 备份数据

```bash
# 备份脚本
#!/bin/bash
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 备份 Milvus 数据
cp -r volumes/milvus "$BACKUP_DIR/"

# 备份数据库
cp -r volumes/db "$BACKUP_DIR/"

# 备份处理后的文档
cp -r volumes/output "$BACKUP_DIR/"

echo "备份完成: $BACKUP_DIR"
```

### 清理数据

```bash
# 停止服务
docker compose down

# 清理所有数据（谨慎！）
rm -rf volumes/milvus/*
rm -rf volumes/db/*
rm -rf volumes/output/*

# 重新启动
docker compose up -d
```

### 监控系统

```bash
# 查看容器状态
docker compose ps

# 查看资源使用
docker stats

# 查看磁盘使用
du -sh volumes/*

# 查看网络连接
docker compose exec backend netstat -tlnp
```

### 性能调优

1. **Milvus 索引优化**
   ```env
   # 适合小数据集（<10万向量）
   MILVUS_HNSW_M=8
   MILVUS_HNSW_EF_CONSTRUCTION=100

   # 适合大数据集（>100万向量）
   MILVUS_HNSW_M=32
   MILVUS_HNSW_EF_CONSTRUCTION=400
   ```

2. **批处理优化**
   ```env
   # 提高向量化批处理大小
   VECTOR_BATCH_SIZE=50

   # 提高文档处理并发
   MAX_CONCURRENT_REQUESTS=5
   ```

3. **缓存配置**
   ```env
   ENABLE_CACHE=true
   CACHE_TTL=7200  # 2小时
   ```

## 📞 获取帮助

- 项目文档: [https://github.com/your-org/hit-rag](https://github.com/your-org/hit-rag)
- 问题反馈: [https://github.com/your-org/hit-rag/issues](https://github.com/your-org/hit-rag/issues)
- 讨论区: [https://github.com/your-org/hit-rag/discussions](https://github.com/your-org/hit-rag/discussions)

## 📄 许可证

[MIT License](LICENSE)
