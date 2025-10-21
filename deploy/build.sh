#!/bin/bash

# RAG 系统构建脚本
# 支持多种构建选项：普通编译、清除缓存重新编译、代理编译等

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_PROXY_HOST="host.docker.internal"
DEFAULT_PROXY_PORT="7897"
DEFAULT_INDEX_URL="https://pypi.org/simple"
DEFAULT_TSINGHUA_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# 显示帮助信息
show_help() {
    echo -e "${BLUE}RAG 系统构建脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -n, --normal            普通编译（默认）"
    echo "  -c, --clean             清除缓存重新编译"
    echo "  -p, --proxy             使用代理编译"
    echo "  -t, --tsinghua          使用清华镜像源编译"
    echo "  -s, --start             编译后启动服务"
    echo "  -d, --daemon            编译后后台启动服务"
    echo "  --proxy-host HOST       代理主机地址（默认: host.docker.internal）"
    echo "  --proxy-port PORT       代理端口（默认: 7897）"
    echo "  --no-cache              不使用 Docker 缓存"
    echo "  --pull                  拉取最新基础镜像"
    echo ""
    echo "示例:"
    echo "  $0 -n                   普通编译"
    echo "  $0 -c -s               清除缓存编译并启动"
    echo "  $0 -p -d               代理编译并后台启动"
    echo "  $0 -t --no-cache       使用清华源无缓存编译"
    echo "  $0 --proxy-host 192.168.1.100 --proxy-port 8080 -p"
    echo ""
}

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Docker 是否运行
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker 未运行或无法访问，请检查 Docker 服务状态"
        exit 1
    fi
}

# 检查 docker-compose 文件
check_compose_file() {
    # 检测系统类型并选择合适的 compose 文件
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux 系统，检查是否是 Ubuntu
        if command -v lsb_release >/dev/null 2>&1; then
            if lsb_release -d | grep -q "Ubuntu"; then
                COMPOSE_FILE="docker-compose.ubuntu.yml"
                log_info "检测到 Ubuntu 系统，使用 Ubuntu 专用配置"
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
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "未找到 $COMPOSE_FILE 文件，请在 deploy 目录下运行此脚本"
        exit 1
    fi
    
    log_info "使用 compose 文件: $COMPOSE_FILE"
}

# 构建 Docker 镜像
build_image() {
    local build_args=""
    local extra_args=""
    
    # 构建参数
    if [ "$USE_PROXY" = "true" ]; then
        build_args="$build_args --build-arg HTTP_PROXY=http://$PROXY_HOST:$PROXY_PORT"
        build_args="$build_args --build-arg HTTPS_PROXY=http://$PROXY_HOST:$PROXY_PORT"
        build_args="$build_args --build-arg NO_PROXY=localhost,127.0.0.1,host.docker.internal"
        log_info "使用代理编译: $PROXY_HOST:$PROXY_PORT"
    fi
    
    if [ "$USE_TSINGHUA" = "true" ]; then
        build_args="$build_args --build-arg UV_INDEX_URL=$DEFAULT_TSINGHUA_INDEX_URL"
        log_info "使用清华镜像源: $DEFAULT_TSINGHUA_INDEX_URL"
    else
        build_args="$build_args --build-arg UV_INDEX_URL=$INDEX_URL"
        log_info "使用镜像源: $INDEX_URL"
    fi
    
    # 额外参数
    if [ "$NO_CACHE" = "true" ]; then
        extra_args="$extra_args --no-cache"
        log_info "不使用 Docker 缓存"
    fi
    
    if [ "$PULL_LATEST" = "true" ]; then
        extra_args="$extra_args --pull"
        log_info "拉取最新基础镜像"
    fi
    
    # 执行构建
    log_info "开始构建 Docker 镜像..."
    echo "构建命令: docker compose -f $COMPOSE_FILE build $build_args $extra_args"
    
    if docker compose -f $COMPOSE_FILE build $build_args $extra_args; then
        log_success "Docker 镜像构建完成"
    else
        log_error "Docker 镜像构建失败"
        exit 1
    fi
}

# 启动服务
start_services() {
    if [ "$DAEMON_MODE" = "true" ]; then
        log_info "后台启动服务..."
        docker compose -f $COMPOSE_FILE up -d
        log_success "服务已在后台启动"
        echo ""
        echo "查看服务状态: docker compose -f $COMPOSE_FILE ps"
        echo "查看日志: docker compose -f $COMPOSE_FILE logs -f"
        echo "停止服务: docker compose -f $COMPOSE_FILE down"
    else
        log_info "启动服务..."
        docker compose -f $COMPOSE_FILE up
    fi
}

# 清理缓存
clean_cache() {
    log_info "清理 Docker 缓存..."
    
    # 停止并删除容器
    docker compose -f $COMPOSE_FILE down --remove-orphans 2>/dev/null || true
    
    # 删除构建缓存
    docker builder prune -f 2>/dev/null || true
    
    # 删除未使用的镜像
    docker image prune -f 2>/dev/null || true
    
    log_success "缓存清理完成"
}

# 显示构建信息
show_build_info() {
    echo ""
    echo "==================== 构建配置 ===================="
    echo "构建模式: $BUILD_MODE"
    echo "代理设置: $USE_PROXY"
    if [ "$USE_PROXY" = "true" ]; then
        echo "代理地址: $PROXY_HOST:$PROXY_PORT"
    fi
    echo "镜像源: $INDEX_URL"
    echo "无缓存: $NO_CACHE"
    echo "拉取最新: $PULL_LATEST"
    echo "启动服务: $START_SERVICES"
    echo "后台模式: $DAEMON_MODE"
    echo "=================================================="
    echo ""
}

# 主函数
main() {
    # 默认值
    BUILD_MODE="normal"
    USE_PROXY="false"
    USE_TSINGHUA="false"
    PROXY_HOST="$DEFAULT_PROXY_HOST"
    PROXY_PORT="$DEFAULT_PROXY_PORT"
    INDEX_URL="$DEFAULT_INDEX_URL"
    NO_CACHE="false"
    PULL_LATEST="false"
    START_SERVICES="false"
    DAEMON_MODE="false"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -n|--normal)
                BUILD_MODE="normal"
                shift
                ;;
            -c|--clean)
                BUILD_MODE="clean"
                shift
                ;;
            -p|--proxy)
                USE_PROXY="true"
                shift
                ;;
            -t|--tsinghua)
                USE_TSINGHUA="true"
                INDEX_URL="$DEFAULT_TSINGHUA_INDEX_URL"
                shift
                ;;
            -s|--start)
                START_SERVICES="true"
                shift
                ;;
            -d|--daemon)
                START_SERVICES="true"
                DAEMON_MODE="true"
                shift
                ;;
            --proxy-host)
                PROXY_HOST="$2"
                shift 2
                ;;
            --proxy-port)
                PROXY_PORT="$2"
                shift 2
                ;;
            --no-cache)
                NO_CACHE="true"
                shift
                ;;
            --pull)
                PULL_LATEST="true"
                shift
                ;;
            *)
                log_error "未知选项: $1"
                echo "使用 -h 或 --help 查看帮助信息"
                exit 1
                ;;
        esac
    done
    
    # 检查环境
    check_docker
    check_compose_file
    
    # 显示构建信息
    show_build_info
    
    # 执行构建流程
    if [ "$BUILD_MODE" = "clean" ]; then
        clean_cache
    fi
    
    build_image
    
    if [ "$START_SERVICES" = "true" ]; then
        start_services
    else
        log_success "构建完成！"
        echo ""
        echo "启动服务: docker compose up"
        echo "后台启动: docker compose up -d"
    fi
}

# 执行主函数
main "$@"
