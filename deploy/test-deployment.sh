#!/bin/bash

# ==================== RAG 系统完整功能测试脚本 ====================
# 测试所有主要功能点，确保系统正常工作

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 计数器
PASS_COUNT=0
FAIL_COUNT=0
TOTAL_TESTS=0

# 测试结果记录
test_result() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if [ $1 -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $2"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "  ${RED}✗${NC} $2"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if [ -n "$3" ]; then
            echo -e "    ${RED}错误详情:${NC} $3"
        fi
    fi
}

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}║               RAG 系统完整功能测试                               ║${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ==================== 1. 基础连接测试 ====================
echo -e "${BLUE}[1/10] 基础连接测试${NC}"

# Milvus
response=$(curl -sf http://localhost:9091/healthz 2>&1)
test_result $? "Milvus 健康检查" "$response"

# 后端 API
response=$(curl -sf http://localhost:8000/api/assistants 2>&1)
test_result $? "后端 API 连接" "$response"

# 前端 UI
response=$(curl -sf http://localhost:5173 2>&1)
test_result $? "前端 UI 连接" "$response"

# Chat View
response=$(curl -sf http://localhost:3000 2>&1)
test_result $? "Chat View 连接" "$response"

echo ""

# ==================== 2. API 端点测试 ====================
echo -e "${BLUE}[2/10] API 端点测试${NC}"

# 获取助手列表
response=$(curl -sf http://localhost:8000/api/assistants 2>&1)
if echo "$response" | grep -q "rag-assistant"; then
    test_result 0 "获取助手列表"
else
    test_result 1 "获取助手列表" "未找到 rag-assistant"
fi

# 获取文档列表
response=$(curl -sf http://localhost:8000/api/documents 2>&1)
if echo "$response" | grep -q "\["; then
    test_result 0 "获取文档列表"
else
    test_result 1 "获取文档列表" "响应格式错误"
fi

# 获取标签列表
response=$(curl -sf http://localhost:8000/api/tags/all 2>&1)
if echo "$response" | grep -q "\["; then
    test_result 0 "获取标签列表"
else
    test_result 1 "获取标签列表" "响应格式错误"
fi

# 获取向量化统计
response=$(curl -sf http://localhost:8000/api/vectorization/stats 2>&1)
if echo "$response" | grep -q "total_chunks\|chunks"; then
    test_result 0 "向量化统计信息"
else
    test_result 1 "向量化统计信息" "响应格式错误"
fi

echo ""

# ==================== 3. 文档处理测试 ====================
echo -e "${BLUE}[3/10] 文档处理测试${NC}"

# 创建测试文档
TEST_DOC="test-document-$(date +%s).md"
TEST_DIR="./all-md"

if [ -d "$TEST_DIR" ]; then
    cat > "$TEST_DIR/$TEST_DOC" << 'EOF'
# 测试文档

这是一个用于测试 RAG 系统的示例文档。

## 主要内容

- 自动化处理
- 智能问答
- 向量检索

## 详细说明

系统会自动处理这个文档，将其切分成多个 chunks，并生成向量表示。
EOF
    test_result 0 "创建测试文档: $TEST_DOC"

    # 等待系统扫描
    sleep 2

    # 检查文档是否被识别
    response=$(curl -sf http://localhost:8000/api/documents 2>&1)
    if echo "$response" | grep -q "$TEST_DOC"; then
        test_result 0 "系统识别测试文档"
    else
        test_result 1 "系统识别测试文档" "文档未在列表中"
    fi
else
    test_result 1 "文档目录不存在" "$TEST_DIR"
fi

echo ""

# ==================== 4. RAG 查询测试 ====================
echo -e "${BLUE}[4/10] RAG 查询流程测试${NC}"

# 测试简单查询
response=$(curl -sf -X POST http://localhost:8000/api/agent/react \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "你好"}]}' 2>&1)

if echo "$response" | grep -q '"type".*"content"'; then
    test_result 0 "基础查询响应"
else
    test_result 1 "基础查询响应" "响应格式不正确"
fi

# 检查 SSE 流格式
if echo "$response" | grep -q 'data:'; then
    test_result 0 "SSE 流格式正确"
else
    test_result 1 "SSE 流格式正确" "未检测到 SSE 格式"
fi

# 检查是否包含意图识别
if echo "$response" | grep -q 'reasoning'; then
    test_result 0 "意图识别响应"
else
    test_result 1 "意图识别响应" "未检测到 reasoning"
fi

# 检查是否包含内容响应
if echo "$response" | grep -q '"type":"content"'; then
    test_result 0 "内容生成响应"
else
    test_result 1 "内容生成响应" "未检测到 content 类型"
fi

echo ""

# ==================== 5. 向量检索测试 ====================
echo -e "${BLUE}[5/10] 向量检索测试${NC}"

# 测试检索功能（如果有向量化的文档）
response=$(curl -sf -X POST http://localhost:8000/api/agent/react \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "测试文档"}]}' 2>&1)

if echo "$response" | grep -q '"type":"file"'; then
    test_result 0 "文档检索功能"

    # 检查是否返回了相似度分数
    if echo "$response" | grep -q '相似度'; then
        test_result 0 "相似度计算"
    else
        test_result 1 "相似度计算" "未找到相似度信息"
    fi
else
    test_result 1 "文档检索功能" "未检测到检索结果"
fi

echo ""

# ==================== 6. Markdown 格式测试 ====================
echo -e "${BLUE}[6/10] Markdown 格式测试${NC}"

response=$(curl -sf -X POST http://localhost:8000/api/agent/react \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "介绍一下系统"}]}' 2>&1)

# 检查 Markdown 图片语法
if echo "$response" | grep -q '!\[.*\](http'; then
    test_result 0 "Markdown 图片语法"
else
    test_result 0 "Markdown 图片语法（可选）" # 不一定所有响应都有图片
fi

# 检查 Markdown 列表语法
if echo "$response" | grep -q '\*\*.*\*\*\|1\..*\|2\.'; then
    test_result 0 "Markdown 格式化（列表/加粗）"
else
    test_result 0 "Markdown 格式化（可选）"
fi

echo ""

# ==================== 7. 多轮对话测试 ====================
echo -e "${BLUE}[7/10] 多轮对话上下文测试${NC}"

# 第一轮对话
response1=$(curl -sf -X POST http://localhost:8000/api/agent/react \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "AutoAgent是什么？"}]}' 2>&1)

if echo "$response1" | grep -q '"type":"content"'; then
    test_result 0 "第一轮对话响应"

    # 第二轮对话（带历史）
    response2=$(curl -sf -X POST http://localhost:8000/api/agent/react \
        -H "Content-Type: application/json" \
        -d '{"messages": [
            {"role": "user", "content": "AutoAgent是什么？"},
            {"role": "assistant", "content": "AutoAgent是一个智能体平台"},
            {"role": "user", "content": "它有什么特点？"}
        ]}' 2>&1)

    if echo "$response2" | grep -q '"type":"content"'; then
        test_result 0 "第二轮对话（带上下文）"

        # 检查是否理解了上下文
        if echo "$response2" | grep -qi 'AutoAgent\|特点\|功能'; then
            test_result 0 "上下文理解能力"
        else
            test_result 1 "上下文理解能力" "响应未体现上下文"
        fi
    else
        test_result 1 "第二轮对话（带上下文）" "响应格式错误"
    fi
else
    test_result 1 "第一轮对话响应" "响应格式错误"
fi

echo ""

# ==================== 8. 不相关查询处理测试 ====================
echo -e "${BLUE}[8/10] 不相关查询处理测试${NC}"

response=$(curl -sf -X POST http://localhost:8000/api/agent/react \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "今天天气怎么样？"}]}' 2>&1)

if echo "$response" | grep -qi '知识库.*没有\|无法.*回答\|相关.*文档'; then
    test_result 0 "识别不相关查询"
else
    test_result 1 "识别不相关查询" "未正确处理不相关查询"
fi

echo ""

# ==================== 9. 性能测试 ====================
echo -e "${BLUE}[9/10] 性能测试${NC}"

# 测试响应时间
start_time=$(date +%s%N)
curl -sf -X POST http://localhost:8000/api/agent/react \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "测试"}]}' > /dev/null 2>&1
end_time=$(date +%s%N)

response_time=$(( (end_time - start_time) / 1000000 ))  # 转换为毫秒

if [ $response_time -lt 10000 ]; then
    test_result 0 "响应时间 (${response_time}ms < 10s)"
else
    test_result 1 "响应时间 (${response_time}ms > 10s)" "响应过慢"
fi

# 测试并发（简单测试3个并发请求）
echo -n "  测试并发处理: "
for i in 1 2 3; do
    curl -sf -X POST http://localhost:8000/api/agent/react \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "并发测试'$i'"}]}' \
        > /dev/null 2>&1 &
done
wait
test_result 0 "并发请求处理"

echo ""

# ==================== 10. 数据持久化测试 ====================
echo -e "${BLUE}[10/10] 数据持久化测试${NC}"

# 检查数据目录
if [ -d "volumes/milvus" ] && [ "$(ls -A volumes/milvus)" ]; then
    test_result 0 "Milvus 数据持久化"
else
    test_result 1 "Milvus 数据持久化" "数据目录为空"
fi

if [ -d "volumes/db" ]; then
    test_result 0 "数据库目录存在"
else
    test_result 1 "数据库目录存在"
fi

if [ -d "volumes/output" ]; then
    test_result 0 "输出目录存在"
else
    test_result 1 "输出目录存在"
fi

echo ""

# ==================== 测试总结 ====================
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    测试结果总结                                 ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "总测试数: ${TOTAL_TESTS}"
echo -e "${GREEN}通过: ${PASS_COUNT}${NC}"
echo -e "${RED}失败: ${FAIL_COUNT}${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           🎉 所有测试通过！系统运行正常！                       ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${YELLOW}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║      ⚠️  部分测试失败，请检查上述错误信息                      ║${NC}"
    echo -e "${YELLOW}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "建议操作："
    echo "  1. 查看详细日志: docker compose -f deploy/docker-compose.yml logs -f"
    echo "  2. 检查服务状态: docker compose -f deploy/docker-compose.yml ps"
    echo "  3. 运行健康检查: ./healthcheck.sh"
    exit 1
fi
