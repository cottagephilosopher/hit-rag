#!/bin/bash

# ==================== RAG 系统一键部署脚本 ====================
# 功能：
# 1. 引导用户配置 .env 文件
# 2. 设置文档目录（all-md）
# 3. 克隆三个 GitHub 项目
# 4. 启动 Docker Compose（包含 Milvus）
# 5. 验证系统健康状态

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_compose() {
    if docker compose version &> /dev/null; then
        docker compose -f "${SCRIPT_DIR}/docker-compose.yml" "$@"
    else
        docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" "$@"
    fi
}

# 显示欢迎信息
show_welcome() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║            RAG 智能问答系统 - 一键部署脚本                      ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    log_info "本脚本将帮助您完成以下任务："
    echo "  1. 配置环境变量（LLM API 密钥等）"
    echo "  2. 设置文档目录"
    echo "  3. 克隆必要的 GitHub 项目"
    echo "  4. 启动 Docker Compose 服务"
    echo "  5. 验证系统健康状态"
    echo ""
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."

    local missing_deps=()

    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    # 检查 Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_deps+=("docker-compose")
    fi

    # 检查 Git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi

    # 检查 curl
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "缺少必要的依赖: ${missing_deps[*]}"
        echo ""
        echo "请先安装以下依赖："
        echo "  - Docker: https://docs.docker.com/get-docker/"
        echo "  - Docker Compose: https://docs.docker.com/compose/install/"
        echo "  - Git: https://git-scm.com/downloads"
        exit 1
    fi

    log_success "所有依赖检查通过"
}

# 配置 .env 文件
configure_env() {
    log_info "配置环境变量文件..."

    if [ -f ".env" ]; then
        read -p "$(echo -e ${YELLOW}检测到已存在 .env 文件，是否覆盖？[y/N]: ${NC})" overwrite
        if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
            log_info "跳过 .env 配置"
            return
        fi
    fi

    # 复制模板
    cp env.template .env
    log_success ".env 文件已创建"

    echo ""
    echo -e "${YELLOW}请配置以下必要参数：${NC}"
    echo ""

    # LLM 提供商选择
    echo "1. 选择 LLM 提供商："
    echo "   [1] Azure OpenAI"
    echo "   [2] OpenAI"
    read -p "请选择 [1-2]: " llm_choice

    if [ "$llm_choice" = "1" ]; then
        # Azure OpenAI 配置
        read -p "Azure OpenAI API Key: " azure_key
        read -p "Azure OpenAI Endpoint (例如: https://your-resource.openai.azure.com/): " azure_endpoint
        read -p "Azure OpenAI Deployment (默认: gpt-4): " azure_deployment
        azure_deployment=${azure_deployment:-gpt-4}

        sed -i.bak "s|LLM_PROVIDER=azure|LLM_PROVIDER=azure|g" .env
        sed -i.bak "s|AZURE_OPENAI_API_KEY=your_azure_api_key_here|AZURE_OPENAI_API_KEY=$azure_key|g" .env
        sed -i.bak "s|AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/|AZURE_OPENAI_ENDPOINT=$azure_endpoint|g" .env
        sed -i.bak "s|AZURE_OPENAI_DEPLOYMENT=gpt-4|AZURE_OPENAI_DEPLOYMENT=$azure_deployment|g" .env
    else
        # OpenAI 配置
        read -p "OpenAI API Key: " openai_key

        sed -i.bak "s|LLM_PROVIDER=azure|LLM_PROVIDER=openai|g" .env
        sed -i.bak "s|# OPENAI_API_KEY=your_openai_api_key_here|OPENAI_API_KEY=$openai_key|g" .env
    fi

    echo ""
    echo "2. 选择 Embedding 提供商："
    echo "   [1] Ollama (本地，需要先安装 Ollama)"
    echo "   [2] Azure OpenAI"
    echo "   [3] OpenAI"
    read -p "请选择 [1-3]: " embedding_choice

    case $embedding_choice in
        1)
            sed -i.bak "s|EMBEDDING_PROVIDER=ollama|EMBEDDING_PROVIDER=ollama|g" .env
            log_warn "请确保已安装 Ollama 并拉取了 qwen3-embedding:latest 模型"
            ;;
        2)
            sed -i.bak "s|EMBEDDING_PROVIDER=ollama|EMBEDDING_PROVIDER=azure|g" .env
            read -p "Azure Embedding Deployment (默认: text-embedding-ada-002): " embedding_deployment
            embedding_deployment=${embedding_deployment:-text-embedding-ada-002}
            sed -i.bak "s|# AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002|AZURE_OPENAI_EMBEDDING_DEPLOYMENT=$embedding_deployment|g" .env
            ;;
        3)
            sed -i.bak "s|EMBEDDING_PROVIDER=ollama|EMBEDDING_PROVIDER=openai|g" .env
            ;;
    esac

    # 清理备份文件
    rm -f .env.bak

    log_success "环境变量配置完成"
}

# 设置文档目录
setup_document_directory() {
    log_info "设置文档目录..."

    # 获取工作空间根目录（hit-rag 的父目录）
    DEFAULT_MD_DIR="${WORKSPACE_ROOT}/all-md"

    echo ""
    echo "文档目录用于存放所有待处理的 Markdown 文件"
    echo "默认路径: $DEFAULT_MD_DIR"
    echo ""
    read -p "$(echo -e ${YELLOW}使用默认路径？[Y/n]: ${NC})" use_default

    if [[ "$use_default" =~ ^[Nn]$ ]]; then
        read -p "请输入文档目录的完整路径: " MD_DIR
    else
        MD_DIR=$DEFAULT_MD_DIR
    fi

    # 创建目录（如果不存在）
    if [ ! -d "$MD_DIR" ]; then
        mkdir -p "$MD_DIR"
        log_success "已创建文档目录: $MD_DIR"
    else
        log_info "文档目录已存在: $MD_DIR"
    fi

    # 创建软链接到项目根目录
    if [ ! -L "./all-md" ]; then
        ln -s "$MD_DIR" "./all-md"
        log_success "已创建软链接: ./all-md -> $MD_DIR"
    fi

    # 更新 .env 文件
    if grep -q "^ALL_MD_DIR=" .env; then
        sed -i.bak "s|^ALL_MD_DIR=.*|ALL_MD_DIR=$MD_DIR|g" .env
    else
        echo "ALL_MD_DIR=$MD_DIR" >> .env
    fi
    rm -f .env.bak

    echo ""
    log_info "请将您的 Markdown 文档放入: $MD_DIR"
}

# 克隆 GitHub 项目
clone_projects() {
    log_info "检查并克隆 GitHub 项目..."

    cd "${WORKSPACE_ROOT}"

    # 定义项目列表
    declare -A projects=(
        ["hit-rag"]="https://github.com/your-org/hit-rag.git"
        ["hit-rag-ui"]="https://github.com/your-org/hit-rag-ui.git"
        ["versa-chat-view"]="https://github.com/ConcealedGem/versa-chat-view.git"
    )

    for project in "${!projects[@]}"; do
        if [ -d "$project" ]; then
            log_info "$project 已存在，跳过克隆"
        else
            log_info "克隆 $project..."
            git clone "${projects[$project]}" "$project"
            log_success "$project 克隆完成"
        fi
    done

    cd "${PROJECT_ROOT}"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."

    mkdir -p volumes/milvus
    mkdir -p volumes/output
    mkdir -p volumes/db

    log_success "目录创建完成"
}

# 启动 Docker Compose
start_services() {
    log_info "启动 Docker Compose 服务..."

    echo ""
    log_info "这可能需要几分钟时间，请耐心等待..."
    echo ""

    # 构建并启动服务
    run_compose up -d --build

    log_success "Docker 服务已启动"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务启动..."

    local max_attempts=60
    local attempt=0

    echo ""
    echo "正在检查服务状态："

    # 等待 Milvus
    echo -n "  - Milvus (19530): "
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
        echo -n "."
    done

    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}✗${NC}"
        log_error "Milvus 启动超时"
        return 1
    fi

    # 等待后端 API
    echo -n "  - Backend API (8000): "
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/api/assistants > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
        echo -n "."
    done

    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}✗${NC}"
        log_error "Backend API 启动超时"
        return 1
    fi

    # 等待前端 UI
    echo -n "  - Frontend UI (5173): "
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:5173 > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
        echo -n "."
    done

    # 等待 Chat View
    echo -n "  - Chat View (3000): "
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
        echo -n "."
    done

    echo ""
    log_success "所有服务已就绪"
}

# 验证系统健康状态
verify_system() {
    log_info "验证系统健康状态..."

    echo ""

    # 测试 Milvus 连接
    echo -n "1. 测试 Milvus 连接: "
    if curl -s http://localhost:9091/healthz | grep -q "OK"; then
        echo -e "${GREEN}✓ 正常${NC}"
    else
        echo -e "${RED}✗ 异常${NC}"
    fi

    # 测试后端 API
    echo -n "2. 测试后端 API: "
    response=$(curl -s http://localhost:8000/api/assistants)
    if echo "$response" | grep -q "rag-assistant"; then
        echo -e "${GREEN}✓ 正常${NC}"
    else
        echo -e "${RED}✗ 异常${NC}"
    fi

    # 测试文档列表
    echo -n "3. 测试文档列表 API: "
    if curl -s http://localhost:8000/api/documents | grep -q "\["; then
        echo -e "${GREEN}✓ 正常${NC}"
    else
        echo -e "${RED}✗ 异常${NC}"
    fi

    # 测试标签系统
    echo -n "4. 测试标签系统 API: "
    if curl -s http://localhost:8000/api/tags/all | grep -q "\["; then
        echo -e "${GREEN}✓ 正常${NC}"
    else
        echo -e "${RED}✗ 异常${NC}"
    fi

    # 测试前端 UI
    echo -n "5. 测试前端 UI: "
    if curl -s http://localhost:5173 | grep -q "html"; then
        echo -e "${GREEN}✓ 正常${NC}"
    else
        echo -e "${RED}✗ 异常${NC}"
    fi

    # 测试 Chat View
    echo -n "6. 测试 Chat View: "
    if curl -s http://localhost:3000 | grep -q "html"; then
        echo -e "${GREEN}✓ 正常${NC}"
    else
        echo -e "${RED}✗ 异常${NC}"
    fi

    echo ""
    log_success "系统验证完成"
}

# 显示访问信息
show_access_info() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║                    部署成功！                                   ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}访问地址：${NC}"
    echo ""
    echo -e "  📄 文档管理界面:    ${YELLOW}http://localhost:5173${NC}"
    echo -e "  💬 聊天问答界面:    ${YELLOW}http://localhost:3000${NC}"
    echo -e "  🔌 后端 API:        ${YELLOW}http://localhost:8000${NC}"
    echo -e "  🗄️  Milvus 管理:    ${YELLOW}http://localhost:9091${NC}"
    echo ""
    echo -e "${BLUE}常用命令：${NC}"
    echo ""
    echo -e "  查看日志:           ${YELLOW}docker compose -f deploy/docker-compose.yml logs -f${NC}"
    echo -e "  查看特定服务日志:   ${YELLOW}docker compose -f deploy/docker-compose.yml logs -f backend${NC}"
    echo -e "  停止服务:           ${YELLOW}docker compose -f deploy/docker-compose.yml down${NC}"
    echo -e "  重启服务:           ${YELLOW}docker compose -f deploy/docker-compose.yml restart${NC}"
    echo -e "  重新构建:           ${YELLOW}docker compose -f deploy/docker-compose.yml up -d --build${NC}"
    echo ""
    echo -e "${BLUE}下一步操作：${NC}"
    echo ""
    echo "  1. 将 Markdown 文档放入文档目录"
    echo "  2. 访问文档管理界面，上传并处理文档"
    echo "  3. 向量化文档内容"
    echo "  4. 在聊天界面进行智能问答"
    echo ""
}

# 主函数
main() {
    show_welcome

    # 检查依赖
    check_dependencies

    # 配置环境
    configure_env

    # 设置文档目录
    setup_document_directory

    # 克隆项目（如果需要）
    read -p "$(echo -e ${YELLOW}是否需要克隆 GitHub 项目？[y/N]: ${NC})" clone_choice
    if [[ "$clone_choice" =~ ^[Yy]$ ]]; then
        clone_projects
    fi

    # 创建目录
    create_directories

    # 启动服务
    echo ""
    read -p "$(echo -e ${YELLOW}准备启动 Docker 服务，是否继续？[Y/n]: ${NC})" start_choice
    if [[ ! "$start_choice" =~ ^[Nn]$ ]]; then
        start_services
        wait_for_services
        verify_system
        show_access_info
    else
        log_info "已取消启动服务"
        echo ""
        echo "您可以稍后手动启动服务："
        echo "  cd $(pwd)"
        echo "  docker compose -f deploy/docker-compose.yml up -d"
        echo ""
    fi
}

# 运行主函数
main
