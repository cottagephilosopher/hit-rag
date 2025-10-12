# Hit-RAG 后端部署指南

## 概述

本目录包含 Hit-RAG 后端服务的 Docker 部署配置，采用简化的容器化方案。

## 架构

- **Milvus**: 向量数据库服务（端口 19530, 9091）
- **Backend**: FastAPI 后端服务（端口 8000）

## 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- 已配置好的 `.env` 文件（参考项目根目录的 `env.template`）

## 快速开始

### 1. 准备配置文件

在项目根目录创建 `.env` 文件：

```bash
cd /Users/idw/rags/hit-rag
cp env.template .env
```

编辑 `.env` 文件，配置必要参数：
- `ALL_MD_DIR`: 文档目录路径（存放 .md 文件）
- `LLM_PROVIDER`: LLM 提供商（azure/openai）
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API Key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI Endpoint
- 其他配置项...

### 2. 准备目录

确保以下目录存在：

```bash
# 创建日志目录
mkdir -p logs

# 创建输出目录
mkdir -p output

# 确保文档目录存在（根据 .env 中的 ALL_MD_DIR 配置）
# 例如：mkdir -p /path/to/all-md
```

### 3. 启动服务

```bash
cd deploy
docker compose up -d --build
```

### 4. 验证服务

检查服务状态：

```bash
docker compose ps
```

访问后端 API：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/assistants

查看日志：

```bash
docker compose logs -f backend
```

## 目录映射说明

容器会自动映射以下目录：

| 宿主机路径 | 容器内路径 | 说明 |
|-----------|-----------|------|
| `.env` | `/app/.env` | 配置文件（只读） |
| `$ALL_MD_DIR` | `/app/all-md` | 文档目录（只读） |
| `logs/` | `/app/logs` | 日志文件目录 |
| `output/` | `/app/output` | 输出文件目录 |
| Docker Volume | `/app/db` | 数据库文件 |

## 常用命令

```bash
# 启动服务
docker compose up -d

# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看日志
docker compose logs -f

# 查看特定服务日志
docker compose logs -f backend

# 重新构建并启动
docker compose up -d --build

# 清理所有数据（包括数据库）
docker compose down -v
```

## 环境变量

关键环境变量（在 `.env` 文件中配置）：

```bash
# 路径配置
ALL_MD_DIR=/path/to/your/markdown/files
OUTPUT_DIR=/Users/idw/rags/hit-rag/output

# LLM 配置
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4

# Milvus 配置（容器内自动配置）
MILVUS_HOST=milvus
MILVUS_PORT=19530
```

## 故障排查

### 服务无法启动

1. 检查 Docker 是否运行：`docker ps`
2. 检查端口占用：`lsof -i :8000` 或 `lsof -i :19530`
3. 查看容器日志：`docker compose logs`

### Milvus 连接失败

- 确保 Milvus 容器健康：`docker compose ps`
- 检查 Milvus 健康检查：`curl http://localhost:9091/healthz`

### 文档目录找不到

- 检查 `.env` 中的 `ALL_MD_DIR` 配置是否正确
- 确保该目录存在且有读取权限
- 可以使用绝对路径

## 生产环境建议

1. **数据备份**: 定期备份数据库和文档
2. **日志管理**: 配置日志轮转，避免日志文件过大
3. **监控**: 添加服务监控和告警
4. **资源限制**: 在 `docker-compose.yml` 中添加资源限制
5. **网络安全**: 限制端口暴露，使用防火墙

## 技术栈

- Python 3.11
- FastAPI
- Milvus 2.3.3
- Docker & Docker Compose
