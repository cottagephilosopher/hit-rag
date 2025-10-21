#!/bin/bash

# 快速构建脚本 - 基于您提供的构建命令
# 简化版本，专门用于常用的构建场景

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 RAG 系统快速构建${NC}"
echo ""

# 默认代理配置
PROXY_HOST="host.docker.internal"
PROXY_PORT="7897"
INDEX_URL="https://pypi.org/simple"

# 检测系统类型并选择合适的 compose 文件
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux 系统，检查是否是 Ubuntu
    if command -v lsb_release >/dev/null 2>&1; then
        if lsb_release -d | grep -q "Ubuntu"; then
            COMPOSE_FILE="docker-compose.ubuntu.yml"
            echo "检测到 Ubuntu 系统，使用 Ubuntu 专用配置"
        else
            COMPOSE_FILE="docker-compose.yml"
        fi
    else
        COMPOSE_FILE="docker-compose.yml"
    fi
else
    # macOS 或其他系统
    COMPOSE_FILE="docker-compose.yml"
fi

# 解析参数
USE_PROXY=false
USE_TSINGHUA=false
START_AFTER_BUILD=false
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --proxy)
            USE_PROXY=true
            shift
            ;;
        --tsinghua)
            USE_TSINGHUA=true
            INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
            shift
            ;;
        --start)
            START_AFTER_BUILD=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --proxy     使用代理构建（默认: host.docker.internal:7897）"
            echo "  --tsinghua  使用清华镜像源"
            echo "  --start     构建后启动服务"
            echo "  --clean     清除缓存重新构建"
            echo "  -h, --help  显示帮助"
            echo ""
            echo "示例:"
            echo "  $0 --proxy --start    # 代理构建并启动"
            echo "  $0 --tsinghua --clean  # 清华源清除缓存构建"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 显示构建配置
echo "构建配置:"
echo "  代理: $USE_PROXY"
echo "  清华源: $USE_TSINGHUA"
echo "  镜像源: $INDEX_URL"
echo "  清除缓存: $CLEAN_BUILD"
echo "  构建后启动: $START_AFTER_BUILD"
echo ""

# 清除缓存（如果需要）
if [ "$CLEAN_BUILD" = "true" ]; then
    echo "🧹 清除缓存..."
    docker compose down --remove-orphans 2>/dev/null || true
    docker builder prune -f 2>/dev/null || true
    echo "✅ 缓存清除完成"
    echo ""
fi

# 构建命令
BUILD_CMD="docker compose -f $COMPOSE_FILE build"

# 添加代理参数
if [ "$USE_PROXY" = "true" ]; then
    BUILD_CMD="$BUILD_CMD --build-arg HTTP_PROXY=http://$PROXY_HOST:$PROXY_PORT"
    BUILD_CMD="$BUILD_CMD --build-arg HTTPS_PROXY=http://$PROXY_HOST:$PROXY_PORT"
    BUILD_CMD="$BUILD_CMD --build-arg NO_PROXY=localhost,127.0.0.1,host.docker.internal"
fi

# 添加镜像源参数
BUILD_CMD="$BUILD_CMD --build-arg UV_INDEX_URL=$INDEX_URL"

# 添加无缓存参数（如果需要）
if [ "$CLEAN_BUILD" = "true" ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

echo "🔨 执行构建命令:"
echo "$BUILD_CMD"
echo ""

# 执行构建
if eval $BUILD_CMD; then
    echo -e "${GREEN}✅ 构建成功！${NC}"
else
    echo "❌ 构建失败"
    exit 1
fi

# 启动服务（如果需要）
if [ "$START_AFTER_BUILD" = "true" ]; then
    echo ""
    echo "🚀 启动服务..."
    docker compose -f $COMPOSE_FILE up -d
    echo -e "${GREEN}✅ 服务已启动！${NC}"
    echo ""
    echo "查看状态: docker compose -f $COMPOSE_FILE ps"
    echo "查看日志: docker compose -f $COMPOSE_FILE logs -f"
    echo "停止服务: docker compose -f $COMPOSE_FILE down"
fi
