#!/bin/bash
set -e

# 部署配置
SERVER="versex"
REMOTE_DIR="/root/hit-rag"
LOCAL_DIR="/Users/idw/rags/hit-rag"

echo "=== 开始部署 hit-rag 到 versex 服务器 ==="

# 1. 同步项目文件（排除不需要的文件）
echo "1. 同步项目文件..."
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='logs/' \
  --exclude='.dbs/' \
  --exclude='output/' \
  --exclude='all-md/' \
  --exclude='files/' \
  --exclude='.DS_Store' \
  --exclude='*.md' \
  --exclude='.claude/' \
  --exclude='.lsp/' \
  --exclude='.clj-kondo/' \
  --exclude='test_*.py' \
  --exclude='verify_fixes.sh' \
  "${LOCAL_DIR}/" "${SERVER}:${REMOTE_DIR}/"

# 2. 复制并修改 .env 文件
echo "2. 配置环境变量..."
scp "${LOCAL_DIR}/.env" "${SERVER}:${REMOTE_DIR}/.env.tmp"

# 3. 在服务器上修改配置
echo "3. 修改服务器配置..."
ssh ${SERVER} << 'EOF'
cd /root/hit-rag

# 修改 .env 配置
sed -i 's/^MILVUS_HOST=.*/MILVUS_HOST=milvus-standalone/' .env.tmp
sed -i 's/^API_PORT=.*/API_PORT=8086/' .env.tmp
sed -i 's/^API_RELOAD=.*/API_RELOAD=false/' .env.tmp
mv .env.tmp .env

# 创建必要的目录
mkdir -p logs .dbs output all-md files

echo "配置修改完成"
EOF

# 4. 创建 docker-compose 配置
echo "4. 创建 docker-compose 配置..."
cat > /tmp/docker-compose.versex.yml << 'COMPOSE_EOF'
services:
  backend:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    container_name: hit-rag-backend
    environment:
      - MILVUS_HOST=milvus-standalone
      - MILVUS_PORT=19530
      - API_PORT=8086
      - ALL_MD_DIR=/app/all-md
      - OUTPUT_DIR=/app/output
      - LOG_FILE=/app/logs/rag_preprocessor.log
      - DB_FILE=/app/.dbs/rag_preprocessor.db
      - FILE_DIR=/app/files
    env_file:
      - .env
    volumes:
      - ./.env:/app/.env:rw
      - ./logs:/app/logs:rw
      - ./.dbs:/app/.dbs:rw
      - ./output:/app/output:rw
      - ./all-md:/app/all-md:rw
      - ./files:/app/files:rw
    ports:
      - "8086:8086"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8086/api/assistants"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    command: ["uv", "run", "python", "api_server.py"]
    networks:
      - versex-net

networks:
  versex-net:
    external: true
COMPOSE_EOF

scp /tmp/docker-compose.versex.yml "${SERVER}:${REMOTE_DIR}/docker-compose.yml"

# 5. 构建并启动服务
echo "5. 构建并启动服务..."
ssh ${SERVER} << 'EOF'
cd /root/hit-rag

# 停止旧容器（如果存在）
docker compose down 2>/dev/null || true

# 构建镜像
echo "构建 Docker 镜像..."
docker compose build --no-cache

# 启动服务
echo "启动服务..."
docker compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查服务状态
docker compose ps
docker compose logs --tail=50

EOF

echo ""
echo "=== 部署完成 ==="
echo "服务地址: http://114.55.136.229:8086"
echo "查看日志: ssh versex 'cd /root/hit-rag && docker compose logs -f'"
echo "检查状态: ssh versex 'cd /root/hit-rag && docker compose ps'"
