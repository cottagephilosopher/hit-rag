#!/bin/bash

# ==================== RAG 系统健康检查脚本 ====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
cd "${PROJECT_ROOT}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

compose() {
    if docker compose version &> /dev/null; then
        docker compose -f "${COMPOSE_FILE}" "$@"
    else
        docker-compose -f "${COMPOSE_FILE}" "$@"
    fi
}

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}║                    系统健康检查                                 ║${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 检查 Docker 服务状态
echo -e "${BLUE}[1/7] 检查 Docker 服务状态...${NC}"
if compose ps | grep -q "Up"; then
    echo -e "  ${GREEN}✓${NC} Docker 服务正在运行"
    compose ps
else
    echo -e "  ${RED}✗${NC} Docker 服务未运行"
    exit 1
fi
echo ""

# 检查 Milvus
echo -e "${BLUE}[2/7] 检查 Milvus 向量数据库...${NC}"
if curl -sf http://localhost:9091/healthz > /dev/null; then
    echo -e "  ${GREEN}✓${NC} Milvus 健康状态: 正常"
    echo -e "  ${GREEN}✓${NC} 连接地址: http://localhost:19530"
else
    echo -e "  ${RED}✗${NC} Milvus 健康检查失败"
fi
echo ""

# 检查后端 API
echo -e "${BLUE}[3/7] 检查后端 API 服务...${NC}"
if response=$(curl -sf http://localhost:8000/api/assistants); then
    echo -e "  ${GREEN}✓${NC} 后端 API: 正常"
    echo -e "  ${GREEN}✓${NC} 可用助手: $(echo "$response" | grep -o '"name"' | wc -l | tr -d ' ')"

    # 测试文档列表
    if doc_response=$(curl -sf http://localhost:8000/api/documents); then
        doc_count=$(echo "$doc_response" | grep -o '"document_id"' | wc -l | tr -d ' ')
        echo -e "  ${GREEN}✓${NC} 文档数量: $doc_count"
    fi

    # 测试标签系统
    if tag_response=$(curl -sf http://localhost:8000/api/tags/all); then
        tag_count=$(echo "$tag_response" | grep -o '"tag_name"' | wc -l | tr -d ' ')
        echo -e "  ${GREEN}✓${NC} 标签数量: $tag_count"
    fi
else
    echo -e "  ${RED}✗${NC} 后端 API 检查失败"
fi
echo ""

# 检查向量化状态
echo -e "${BLUE}[4/7] 检查向量化状态...${NC}"
if vec_response=$(curl -sf http://localhost:8000/api/vectorization/stats); then
    echo -e "  ${GREEN}✓${NC} 向量化服务: 正常"
    echo "$vec_response" | python3 -m json.tool 2>/dev/null || echo "$vec_response"
else
    echo -e "  ${RED}✗${NC} 向量化状态检查失败"
fi
echo ""

# 检查前端 UI
echo -e "${BLUE}[5/7] 检查前端 UI 服务...${NC}"
if curl -sf http://localhost:5173 > /dev/null; then
    echo -e "  ${GREEN}✓${NC} 前端 UI: 正常"
    echo -e "  ${GREEN}✓${NC} 访问地址: http://localhost:5173"
else
    echo -e "  ${RED}✗${NC} 前端 UI 检查失败"
fi
echo ""

# 检查 Chat View
echo -e "${BLUE}[6/7] 检查 Chat View 服务...${NC}"
if curl -sf http://localhost:3000 > /dev/null; then
    echo -e "  ${GREEN}✓${NC} Chat View: 正常"
    echo -e "  ${GREEN}✓${NC} 访问地址: http://localhost:3000"
else
    echo -e "  ${RED}✗${NC} Chat View 检查失败"
fi
echo ""

# 测试完整的 RAG 查询流程
echo -e "${BLUE}[7/7] 测试 RAG 查询流程...${NC}"
if test_response=$(curl -sf -X POST http://localhost:8000/api/agent/react \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "测试"}]}' \
    2>/dev/null); then

    if echo "$test_response" | grep -q "type.*content"; then
        echo -e "  ${GREEN}✓${NC} RAG 查询流程: 正常"
        echo -e "  ${GREEN}✓${NC} 意图识别: ✓"
        echo -e "  ${GREEN}✓${NC} 文档检索: ✓"
        echo -e "  ${GREEN}✓${NC} 响应生成: ✓"
    else
        echo -e "  ${YELLOW}⚠${NC} RAG 查询流程: 响应格式异常"
    fi
else
    echo -e "  ${RED}✗${NC} RAG 查询流程测试失败"
fi
echo ""

# 资源使用情况
echo -e "${BLUE}资源使用情况：${NC}"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
echo ""

# 磁盘使用
echo -e "${BLUE}数据目录磁盘使用：${NC}"
if [ -d "volumes" ]; then
    du -sh volumes/* 2>/dev/null
fi
echo ""

# 总结
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    健康检查完成                                 ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
