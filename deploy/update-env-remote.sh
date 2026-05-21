#!/bin/bash

# 更新远程 .env 配置并重启服务
# 使用方法: ./deploy/update-env-remote.sh

set -e

SERVER="versex"
REMOTE_DIR="/root/hit-rag"

echo "🔄 更新远程配置..."

# 1. 同步 .env 文件
echo "📤 同步 .env 配置文件..."
rsync -avz --progress .env ${SERVER}:${REMOTE_DIR}/

# 2. 重新创建容器使配置生效（restart 不会重新加载 env_file）
echo "🔄 重新创建 Docker 容器..."
ssh ${SERVER} "cd ${REMOTE_DIR} && docker compose up -d --force-recreate"

# 3. 等待服务启动
echo "⏳ 等待服务启动..."
sleep 5

# 4. 检查服务状态
echo "✅ 检查服务状态..."
ssh ${SERVER} "cd ${REMOTE_DIR} && docker compose ps && docker logs hit-rag-backend --tail 20"

# 5. 测试 API
echo ""
echo "🧪 测试 API 接口..."
ssh ${SERVER} "curl -s http://localhost:8086/api/assistants | head -100"

echo ""
echo "✅ 配置更新完成！"
echo "📖 API 文档: http://114.55.136.229:8086/docs"
