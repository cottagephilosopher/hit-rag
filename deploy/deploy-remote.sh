#!/bin/bash

# 远程服务器部署脚本
# 服务器信息：versex (114.55.136.229) - Ubuntu 20.04

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 服务器配置
SERVER_HOST="114.55.136.229"
SERVER_USER="root"
PROJECT_NAME="hit-rag"
PROJECT_DIR="/root/${PROJECT_NAME}"
BACKUP_DIR="/root/${PROJECT_NAME}_backup_$(date +%Y%m%d_%H%M%S)"

# 本地项目路径
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

# 显示帮助信息
show_help() {
    echo -e "${BLUE}Hit-RAG 远程部署脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -p, --prepare           准备服务器环境"
    echo "  -d, --deploy            部署项目"
    echo "  -u, --upload-only       仅上传文件（不构建）"
    echo "  -b, --backup            备份远程服务器项目"
    echo "  -t, --test-connect      测试服务器连接"
    echo "  -s, --start             启动服务"
    echo "  -k, --stop              停止服务"
    echo "  -r, --restart           重启服务"
    echo "  --no-backup             部署时不备份原项目"
    echo ""
    echo "示例:"
    echo "  $0 -t                   测试服务器连接"
    echo "  $0 -p                   准备服务器环境"
    echo "  $0 -d -s                部署并启动服务"
    echo "  $0 -u -s                仅上传文件并启动服务"
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

# 测试服务器连接
test_connection() {
    log_info "测试服务器连接: ${SERVER_USER}@${SERVER_HOST}"

    if ssh -q ${SERVER_USER}@${SERVER_HOST} "echo 'Connection successful'"; then
        log_success "服务器连接成功"
    else
        log_error "服务器连接失败"
        exit 1
    fi
}

# 准备服务器环境
prepare_server() {
    log_info "准备服务器环境..."

    # 检查 Docker 是否运行
    log_info "检查 Docker 服务状态..."
    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        if ! systemctl is-active --quiet docker; then
            echo "Docker 服务未运行，正在启动..."
            systemctl start docker
            systemctl enable docker
        fi
        docker --version
EOF

    log_info "检查 Docker Compose 是否安装..."
    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        if ! command -v docker &> /dev/null || ! command -v docker compose &> /dev/null; then
            echo "Docker 或 Docker Compose 未安装"
            exit 1
        fi
        docker compose version
EOF

    # 创建项目目录
    log_info "创建项目目录..."
    ssh ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${PROJECT_DIR}"

    log_success "服务器环境准备完成"
}

# 备份远程项目
backup_project() {
    log_info "备份远程项目..."

    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        if [ -d "${PROJECT_DIR}" ]; then
            echo "项目目录存在，创建备份..."
            mkdir -p "$(dirname ${BACKUP_DIR})"
            cp -r ${PROJECT_DIR} ${BACKUP_DIR}
            echo "备份完成: ${BACKUP_DIR}"
        else
            echo "项目目录不存在，无需备份"
        fi
EOF

    log_success "备份完成"
}

# 上传项目文件
upload_files() {
    log_info "上传项目文件到远程服务器..."

    # 压缩项目文件（排除不需要的文件）
    local temp_tar="hit-rag-$(date +%Y%m%d_%H%M%S).tar.gz"

    log_info "创建项目压缩包..."
    cd "${LOCAL_PROJECT_DIR}"
    tar -zcvf "${temp_tar}" \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.venv' \
        --exclude='.git' \
        --exclude='.dbs' \
        --exclude='uploads' \
        --exclude='files' \
        --exclude='logs' \
        --exclude='output' \
        --exclude='all-md' \
        .

    log_info "上传项目压缩包..."
    scp "${temp_tar}" ${SERVER_USER}@${SERVER_HOST}:/tmp/

    log_info "解压项目文件..."
    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        mkdir -p ${PROJECT_DIR}
        cd ${PROJECT_DIR}
        tar -zxvf /tmp/${temp_tar} -C ${PROJECT_DIR}
        rm -f /tmp/${temp_tar}
        echo "项目文件解压完成"
EOF

    # 清理本地临时文件
    rm -f "${temp_tar}"

    log_info "创建数据目录..."
    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        cd ${PROJECT_DIR}
        mkdir -p .dbs
        mkdir -p logs
        mkdir -p output
        mkdir -p all-md
        mkdir -p uploads
        mkdir -p files
        chmod +x deploy/build.sh
        chmod +x deploy/quick-build.sh
EOF

    log_success "项目文件上传完成"
}

# 配置环境变量
configure_env() {
    log_info "配置环境变量..."

    # 上传 .env 文件（如果不存在则从模板创建）
    if [ -f "${LOCAL_PROJECT_DIR}/.env" ]; then
        log_info "上传 .env 配置文件..."
        scp "${LOCAL_PROJECT_DIR}/.env" ${SERVER_USER}@${SERVER_HOST}:${PROJECT_DIR}/.env
    else
        log_warning ".env 文件不存在，使用模板创建"
        scp "${LOCAL_PROJECT_DIR}/env.template" ${SERVER_USER}@${SERVER_HOST}:${PROJECT_DIR}/.env
    fi

    # 更新环境变量
    log_info "更新环境配置..."
    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        cd ${PROJECT_DIR}

        # 确保 API 端口配置正确
        if ! grep -q "API_PORT=8086" .env; then
            echo "API_PORT=8086" >> .env
        fi

        # 确保 Milvus 配置正确（Docker 内部使用服务名）
        if ! grep -q "MILVUS_HOST=milvus" .env; then
            echo "MILVUS_HOST=milvus" >> .env
        fi
        if ! grep -q "MILVUS_PORT=19530" .env; then
            echo "MILVUS_PORT=19530" >> .env
        fi
EOF

    log_success "环境配置完成"
}

# 创建 Docker Compose 配置
create_compose_config() {
    log_info "创建 Docker Compose 配置..."

    cat > "${LOCAL_PROJECT_DIR}/deploy/docker-compose.remote.yml" <<'EOF'
version: '3.8'

services:
  # Milvus 向量数据库
  milvus:
    image: docker.m.daocloud.io/milvusdb/milvus:v2.6.2
    container_name: hit-rag-milvus
    command: milvus run standalone
    environment:
      ETCD_USE_EMBED: "true"
      ETCD_DATA_DIR: /var/lib/milvus/etcd
      COMMON_STORAGETYPE: local
      DEPLOY_MODE: STANDALONE
    volumes:
      - ~/docker_file/milvus:/var/lib/milvus
    ports:
      - "19531:19530"
      - "9092:9091"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 3
      start_period: 90s
    restart: unless-stopped
    networks:
      - rag-network

  # 后端 API 服务
  backend:
    build:
      context: ..
      dockerfile: deploy/Dockerfile
      args:
        HTTP_PROXY: ${HTTP_PROXY:-}
        HTTPS_PROXY: ${HTTPS_PROXY:-}
      extra_hosts:
        - "host.docker.internal:host-gateway"
    container_name: hit-rag-backend
    environment:
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
      - ALL_MD_DIR=/app/all-md
      - OUTPUT_DIR=/app/output
      - LOG_FILE=/app/logs/rag_preprocessor.log
      - DB_FILE=/app/.dbs/rag_preprocessor.db
      - FILE_DIR=/files
    env_file:
      - ../.env
    volumes:
      - ../.env:/app/.env:rw
      - ../.dbs:/app/.dbs
      - ../logs:/app/logs:rw
      - ../output:/app/output:rw
      - ../all-md:/app/all-md:rw
      - ../files:/files:rw
      - ../uploads:/app/uploads:rw
    ports:
      - "8086:8086"
    depends_on:
      milvus:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8086/api/assistants"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    command: ["uv", "run", "python", "api_server.py"]
    extra_hosts:
      - "host.docker.internal:host-gateway"
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge
EOF

    log_info "上传 Docker Compose 配置..."
    scp "${LOCAL_PROJECT_DIR}/deploy/docker-compose.remote.yml" ${SERVER_USER}@${SERVER_HOST}:${PROJECT_DIR}/deploy/

    log_success "Docker Compose 配置创建完成"
}

# 构建并启动服务
build_and_start() {
    log_info "构建并启动服务..."

    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        cd ${PROJECT_DIR}/deploy

        # 检查是否已在运行
        if docker ps --filter "name=hit-rag-" --format "{{.Names}}" | grep -q .; then
            echo "服务正在运行，先停止..."
            docker compose -f docker-compose.remote.yml down
        fi

        # 构建服务
        echo "开始构建 Docker 镜像..."
        ./build.sh -p --proxy-host 127.0.0.1 --proxy-port 7890 --tsinghua --daemon

        echo "服务启动成功！"
EOF

    log_success "服务构建并启动完成"
}

# 启动服务
start_services() {
    log_info "启动服务..."

    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        cd ${PROJECT_DIR}/deploy
        docker compose -f docker-compose.remote.yml up -d
EOF

    log_success "服务启动完成"
}

# 停止服务
stop_services() {
    log_info "停止服务..."

    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        cd ${PROJECT_DIR}/deploy
        docker compose -f docker-compose.remote.yml down
EOF

    log_success "服务停止完成"
}

# 重启服务
restart_services() {
    log_info "重启服务..."
    stop_services
    start_services
}

# 显示远程服务器状态
show_server_status() {
    log_info "显示远程服务器状态..."

    ssh ${SERVER_USER}@${SERVER_HOST} <<EOF
        echo "=================== 服务器信息 ==================="
        hostnamectl
        echo ""
        echo "=================== 内存使用 ==================="
        free -h
        echo ""
        echo "=================== 磁盘使用 ==================="
        df -h
        echo ""
        echo "=================== Docker 状态 ==================="
        docker info | head -20
        echo ""
        echo "=================== 运行中的容器 ==================="
        docker ps
        echo ""
        echo "=================== 项目目录 ==================="
        ls -la ${PROJECT_DIR}
EOF
}

# 测试 API 接口
test_api() {
    log_info "测试后端 API 接口..."

    local retry_count=0
    local max_retries=10
    local api_url="http://${SERVER_HOST}:8086/api/assistants"

    while [ $retry_count -lt $max_retries ]; do
        log_info "第 $((retry_count + 1)) 次测试 API 接口: $api_url"

        if curl -s "$api_url" > /dev/null; then
            log_success "API 接口测试成功!"
            echo "API 响应:"
            curl -s "$api_url"
            return 0
        fi

        retry_count=$((retry_count + 1))
        log_warning "API 接口未响应，等待 10 秒后重试..."
        sleep 10
    done

    log_error "API 接口测试失败，已重试 $max_retries 次"
    return 1
}

# 主函数
main() {
    local prepare_env="false"
    local deploy="false"
    local upload_only="false"
    local backup="false"
    local test_connect="false"
    local start="false"
    local stop="false"
    local restart="false"
    local no_backup="false"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -p|--prepare)
                prepare_env="true"
                shift
                ;;
            -d|--deploy)
                deploy="true"
                shift
                ;;
            -u|--upload-only)
                upload_only="true"
                shift
                ;;
            -b|--backup)
                backup="true"
                shift
                ;;
            -t|--test-connect)
                test_connect="true"
                shift
                ;;
            -s|--start)
                start="true"
                shift
                ;;
            -k|--stop)
                stop="true"
                shift
                ;;
            -r|--restart)
                restart="true"
                shift
                ;;
            --no-backup)
                no_backup="true"
                shift
                ;;
            *)
                log_error "未知选项: $1"
                echo "使用 -h 或 --help 查看帮助信息"
                exit 1
                ;;
        esac
    done

    # 执行操作
    if [ "$test_connect" = "true" ]; then
        test_connection
        exit 0
    fi

    if [ "$prepare_env" = "true" ]; then
        prepare_server
    fi

    if [ "$backup" = "true" ] && [ "$no_backup" = "false" ]; then
        backup_project
    fi

    if [ "$deploy" = "true" ] || [ "$upload_only" = "true" ]; then
        if [ "$no_backup" = "false" ]; then
            backup_project
        fi
        upload_files
        configure_env
        create_compose_config
    fi

    if [ "$deploy" = "true" ]; then
        build_and_start
    fi

    if [ "$start" = "true" ]; then
        start_services
    fi

    if [ "$stop" = "true" ]; then
        stop_services
    fi

    if [ "$restart" = "true" ]; then
        restart_services
    fi

    # 如果执行了部署或启动操作，测试 API
    if [ "$deploy" = "true" ] || [ "$start" = "true" ]; then
        log_info "等待服务启动..."
        sleep 30
        test_api
    fi

    log_success "部署脚本执行完成!"
}

# 执行主函数
main "$@"