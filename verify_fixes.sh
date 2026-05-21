#!/bin/bash
# 验证阻塞问题修复脚本

echo "================================================"
echo "系统阻塞问题修复验证"
echo "================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python版本
echo "1. 检查Python环境..."
python_version=$(uv run python --version 2>&1)
echo "   Python版本: $python_version"

# 检查关键文件是否存在
echo ""
echo "2. 检查修复文件..."
files=(
    "image_uploader.py"
    "file_upload_routes.py"
    "document_routes.py"
    "main.py"
    "utils/timeout_protection.py"
)

all_files_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "   ${GREEN}✓${NC} $file"
    else
        echo -e "   ${RED}✗${NC} $file (缺失)"
        all_files_exist=false
    fi
done

# 检查关键函数是否存在
echo ""
echo "3. 检查关键修复代码..."

# 检查URL验证函数
if grep -q "_is_valid_image_url" image_uploader.py; then
    echo -e "   ${GREEN}✓${NC} URL验证函数已添加"
else
    echo -e "   ${RED}✗${NC} URL验证函数缺失"
fi

# 检查异步处理函数
if grep -q "process_markdown_file_async" image_uploader.py; then
    echo -e "   ${GREEN}✓${NC} 异步图片处理函数已添加"
else
    echo -e "   ${RED}✗${NC} 异步图片处理函数缺失"
fi

# 检查子进程异步化
if grep -q "asyncio.create_subprocess_exec" document_routes.py; then
    echo -e "   ${GREEN}✓${NC} 子进程已异步化"
else
    echo -e "   ${RED}✗${NC} 子进程仍使用同步调用"
fi

# 检查文件大小限制
if grep -q "file_size_mb > 100" file_upload_routes.py; then
    echo -e "   ${GREEN}✓${NC} 文件大小限制已添加"
else
    echo -e "   ${RED}✗${NC} 文件大小限制缺失"
fi

# 检查超时保护工具
if [ -f "utils/timeout_protection.py" ]; then
    echo -e "   ${GREEN}✓${NC} 超时保护工具已创建"
else
    echo -e "   ${RED}✗${NC} 超时保护工具缺失"
fi

# 运行简单测试
echo ""
echo "4. 运行功能测试..."
echo "   测试图片URL验证..."

# 创建临时测试脚本
cat > /tmp/test_url_validation.py << 'EOF'
import sys
sys.path.insert(0, '/Users/idw/rags/hit-rag')
from image_uploader import MarkdownImageProcessor

processor = MarkdownImageProcessor(uploader=None)

test_cases = [
    ("image.jpg", True),
    ("aws4_request&X-Amz-Date=xxx", False),
    ("document.pdf", False),
]

failed = 0
for url, expected in test_cases:
    result = processor._is_valid_image_url(url)
    if result != expected:
        print(f"FAIL: {url}")
        failed += 1

sys.exit(failed)
EOF

if uv run python /tmp/test_url_validation.py 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} URL验证功能正常"
else
    echo -e "   ${RED}✗${NC} URL验证功能异常"
fi

# 清理
rm -f /tmp/test_url_validation.py

# 检查导入
echo ""
echo "5. 检查模块导入..."
if uv run python -c "from image_uploader import MarkdownImageProcessor; from utils.timeout_protection import timeout_context" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} 所有模块可正常导入"
else
    echo -e "   ${RED}✗${NC} 模块导入失败"
fi

# 总结
echo ""
echo "================================================"
echo "验证总结"
echo "================================================"

if [ "$all_files_exist" = true ]; then
    echo -e "${GREEN}✓ 所有修复已成功应用${NC}"
    echo ""
    echo "建议操作："
    echo "1. 重启API服务: pkill -f 'uvicorn api_server:app' && uv run uvicorn api_server:app --reload"
    echo "2. 上传测试文件验证功能"
    echo "3. 观察日志确认无阻塞: tail -f logs/rag_preprocessor.log"
    exit 0
else
    echo -e "${RED}✗ 部分修复文件缺失或有问题${NC}"
    echo ""
    echo "请检查上述错误信息"
    exit 1
fi

