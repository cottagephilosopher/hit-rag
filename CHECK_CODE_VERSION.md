# 检查Docker容器中的代码版本

## 方法1：检查vector_store.py的关键行

在服务器上执行：
```bash
docker exec -it <backend-container-name> grep -n "max_tokens=max_embedding_tokens - 800" /app/vector_db/vector_store.py
```

**预期输出**：应该显示行号（如197行）

**如果没有输出**：说明代码未更新，需要重新部署


## 方法2：检查embedding_service.py的截断逻辑

```bash
docker exec -it <backend-container-name> grep -n "max_tokens = 8000" /app/vector_db/embedding_service.py
```

**预期输出**：应该显示行号（如117行）

**如果没有输出**：说明代码未更新


## 方法3：查看容器启动时间

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

查看backend容器的运行时间是否与您重启的时间一致。


## 重新部署的正确方法

### 如果使用docker-compose：

```bash
# 停止服务
docker compose -f deploy/docker-compose.yml down backend

# 重新构建镜像（重要！）
docker compose -f deploy/docker-compose.yml build --no-cache backend

# 启动服务
docker compose -f deploy/docker-compose.yml up -d backend

# 查看日志
docker compose -f deploy/docker-compose.yml logs -f backend
```

### 如果直接使用docker：

```bash
# 找到容器ID
docker ps -a | grep backend

# 停止并删除容器
docker stop <container-id>
docker rm <container-id>

# 删除旧镜像
docker rmi <image-name>

# 重新构建
docker build -t <image-name> .

# 启动新容器
docker run -d --name backend <image-name>
```

## 验证修复是否生效

重新上传表格文档后，应该看到以下日志：

```
⚠️  Chunk 3476 exceeds embedding token limit (8163 >= 7692), splitting into segments...
🔄 Starting split for chunk 3476, max_tokens=7392
✂️  Split chunk 3476 into 2 segments
    📏 Segment 0: 7392 tokens
    📏 Segment 1: 871 tokens
✅ Successfully added 2 chunks to Milvus
```

如果仍然没有看到这些日志，说明代码更新未生效。
