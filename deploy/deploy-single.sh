#!/bin/bash

# ==================== RAG 系统单容器部署脚本 ====================
# 特点：
# 1. 只启动 3 个容器：Milvus + Backend + Frontend(Unified)
# 2. Frontend 容器整合了 Vue.js 和 Next.js 两个前端
# 3. 通过 Nginx 统一入口，只占用一个端口（80）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
        docker compose -f "${SCRIPT_DIR}/docker-compose.single.yml" "$@"
    else
        docker-compose -f "${SCRIPT_DIR}/docker-compose.single.yml" "$@"
    fi
}

# 显示欢迎信息
show_welcome() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║         RAG 智能问答系统 - 单容器部署方案                        ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    log_info "单容器方案特点："
    echo "  ✓ 只需 3 个容器（Milvus + Backend + Frontend）"
    echo "  ✓ 统一访问入口（端口 80）"
    echo "  ✓ 更简单的网络配置"
    echo "  ✓ 更少的资源占用"
    echo ""
    echo "访问地址："
    echo "  - http://localhost      → 文档管理界面"
    echo "  - http://localhost/chat → 聊天问答界面"
    echo "  - http://localhost/api  → 后端 API"
    echo ""
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."

    local missing_deps=()

    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_deps+=("docker-compose")
    fi

    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "缺少必要的依赖: ${missing_deps[*]}"
        exit 1
    fi

    log_success "所有依赖检查通过"
}

# 检查项目结构
check_project_structure() {
    log_info "检查项目结构..."

    local workspace_dir="${WORKSPACE_ROOT}"

    # 检查必要的项目目录
    local missing_dirs=()

    if [ ! -d "${workspace_dir}/hit-rag" ]; then
        missing_dirs+=("hit-rag")
    fi

    if [ ! -d "${workspace_dir}/hit-rag-ui" ]; then
        missing_dirs+=("hit-rag-ui")
    fi

    if [ ! -d "${workspace_dir}/versa-chat-view" ]; then
        missing_dirs+=("versa-chat-view")
    fi

    if [ ${#missing_dirs[@]} -ne 0 ]; then
        log_error "缺少必要的项目目录: ${missing_dirs[*]}"
        echo ""
        echo "请确保以下项目在同一父目录下："
        echo "  rags/"
        echo "  ├── hit-rag/"
        echo "  ├── hit-rag-ui/"
        echo "  └── versa-chat-view/"
        echo ""
        read -p "$(echo -e ${YELLOW}是否需要自动克隆缺失的项目？[y/N]: ${NC})" clone_choice
        if [[ "$clone_choice" =~ ^[Yy]$ ]]; then
            cd "${workspace_dir}"
            for dir in "${missing_dirs[@]}"; do
                case $dir in
                    "hit-rag")
                        git clone https://github.com/your-org/hit-rag.git
                        ;;
                    "hit-rag-ui")
                        git clone https://github.com/your-org/hit-rag-ui.git
                        ;;
                    "versa-chat-view")
                        git clone https://github.com/ConcealedGem/versa-chat-view.git
                        ;;
                esac
            done
            log_success "项目克隆完成"
        else
            exit 1
        fi
    else
        log_success "项目结构检查通过"
    fi
}

# 配置环境变量
configure_env() {
    log_info "配置环境变量..."

    if [ -f ".env" ]; then
        read -p "$(echo -e ${YELLOW}检测到已存在 .env 文件，是否覆盖？[y/N]: ${NC})" overwrite
        if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
            log_info "跳过 .env 配置"
            return
        fi
    fi

    cp env.template .env
    log_success ".env 文件已创建"

    echo ""
    echo -e "${YELLOW}请配置以下必要参数：${NC}"
    echo ""

    # LLM 配置
    echo "1. 选择 LLM 提供商："
    echo "   [1] Azure OpenAI (推荐)"
    echo "   [2] OpenAI"
    read -p "请选择 [1-2]: " llm_choice

    if [ "$llm_choice" = "1" ]; then
        read -p "Azure OpenAI API Key: " azure_key
        read -p "Azure OpenAI Endpoint: " azure_endpoint
        read -p "Azure OpenAI Deployment (默认: gpt-4): " azure_deployment
        azure_deployment=${azure_deployment:-gpt-4}

        sed -i.bak "s|AZURE_OPENAI_API_KEY=your_azure_api_key_here|AZURE_OPENAI_API_KEY=$azure_key|g" .env
        sed -i.bak "s|AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/|AZURE_OPENAI_ENDPOINT=$azure_endpoint|g" .env
        sed -i.bak "s|AZURE_OPENAI_DEPLOYMENT=gpt-4|AZURE_OPENAI_DEPLOYMENT=$azure_deployment|g" .env
    else
        read -p "OpenAI API Key: " openai_key
        sed -i.bak "s|LLM_PROVIDER=azure|LLM_PROVIDER=openai|g" .env
        sed -i.bak "s|# OPENAI_API_KEY=your_openai_api_key_here|OPENAI_API_KEY=$openai_key|g" .env
    fi

    # Embedding 配置
    echo ""
    echo "2. 选择 Embedding 提供商："
    echo "   [1] Ollama (本地免费)"
    echo "   [2] Azure OpenAI"
    read -p "请选择 [1-2]: " embedding_choice

    if [ "$embedding_choice" = "2" ]; then
        sed -i.bak "s|EMBEDDING_PROVIDER=ollama|EMBEDDING_PROVIDER=azure|g" .env
    fi

    rm -f .env.bak
    log_success "环境变量配置完成"
}

# 设置文档目录
setup_document_directory() {
    log_info "设置文档目录..."

    local default_md_dir="${WORKSPACE_ROOT}/all-md"

    echo ""
    read -p "$(echo -e ${YELLOW}文档目录 (默认: $default_md_dir): ${NC})" MD_DIR
    MD_DIR=${MD_DIR:-$default_md_dir}

    if [ ! -d "$MD_DIR" ]; then
        mkdir -p "$MD_DIR"
        log_success "已创建文档目录: $MD_DIR"
    fi

    if [ ! -L "./all-md" ]; then
        ln -s "$MD_DIR" "./all-md"
        log_success "已创建软链接"
    fi
}

# 启动服务
start_services() {
    log_info "启动单容器服务..."

    echo ""
    log_info "正在构建并启动容器..."
    echo ""

    # 切换到项目根目录（docker-compose.single.yml 位于 deploy/ 中）
    cd "${PROJECT_ROOT}" || {
        log_error "无法进入项目根目录"
        exit 1
    }

    # 使用单容器配置
    run_compose up -d --build

    log_success "容器已启动"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务启动..."

    local max_attempts=60
    local attempt=0

    echo ""
    echo "正在检查服务状态："

    # 等待 Milvus
    echo -n "  - Milvus: "
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
        echo -n "."
    done

    # 等待后端
    echo -n "  - Backend API: "
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

    # 等待前端（统一入口）
    echo -n "  - Frontend: "
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost > /dev/null 2>&1; then
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

# 显示访问信息
show_access_info() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║                    部署成功！                                   ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}访问地址（统一入口）：${NC}"
    echo ""
    echo -e "  🏠 主页:              ${YELLOW}http://localhost${NC}"
    echo -e "  📄 文档管理:          ${YELLOW}http://localhost/ui${NC}"
    echo -e "  💬 聊天问答:          ${YELLOW}http://localhost/chat${NC}"
    echo -e "  🔌 后端 API:          ${YELLOW}http://localhost/api${NC}"
    echo -e "  📖 API 文档:          ${YELLOW}http://localhost/docs${NC}"
    echo ""
    echo -e "${BLUE}容器状态：${NC}"
    echo ""
    # 确保在项目根目录下
    cd "${PROJECT_ROOT}" || true
    run_compose ps
    echo ""
    echo -e "${BLUE}常用命令（在项目根目录下执行）：${NC}"
    echo ""
    echo -e "  查看日志:    ${YELLOW}cd ${PROJECT_ROOT} && docker compose -f deploy/docker-compose.single.yml logs -f${NC}"
    echo -e "  停止服务:    ${YELLOW}cd ${PROJECT_ROOT} && docker compose -f deploy/docker-compose.single.yml down${NC}"
    echo -e "  重启服务:    ${YELLOW}cd ${PROJECT_ROOT} && docker compose -f deploy/docker-compose.single.yml restart${NC}"
    echo ""
}

# 主函数
main() {
    show_welcome

    check_dependencies
    check_project_structure
    configure_env
    setup_document_directory

    echo ""
    read -p "$(echo -e ${YELLOW}准备启动服务，是否继续？[Y/n]: ${NC})" start_choice
    if [[ ! "$start_choice" =~ ^[Nn]$ ]]; then
        # 创建必要目录
        mkdir -p volumes/{milvus,db,output}

        start_services
        wait_for_services
        show_access_info
    else
        log_info "已取消启动"
    fi
}

main
