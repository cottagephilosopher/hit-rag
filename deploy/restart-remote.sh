#!/bin/bash

# 远程重启脚本 - 用于更新配置后重启服务
# 使用方法: ./deploy/restart-remote.sh

set -e

SERVER="versex"
REMOTE_DIR="/root/hit-rag"

echo "🔄 开始重启远程服务..."

# 1. 同步 .env 文件和密码重置脚本
echo "📤 同步配置文件..."
rsync -avz --progress .env ${SERVER}:${REMOTE_DIR}/
rsync -avz --progress reset_admin_password.py ${SERVER}:${REMOTE_DIR}/

# 2. 在容器内重置管理员密码
echo "🔑 重置管理员密码..."
ssh ${SERVER} "cd ${REMOTE_DIR} && docker compose exec backend python reset_admin_password.py"

# 3. 重启容器使配置生效
echo "🔄 重启 Docker 容器..."
ssh ${SERVER} "cd ${REMOTE_DIR} && docker compose restart"

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
echo "✅ 重启完成！"
echo "📖 API 文档: http://114.55.136.229:8086/docs"
