#!/bin/bash
# 下载 Linux 版本的 uv（用于 Docker 构建）
# 自动检测宿主机操作系统和架构，下载对应的 Linux 版本

set -e

UV_VERSION="0.5.2"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检测宿主机操作系统
HOST_OS=$(uname -s)
HOST_ARCH=$(uname -m)

echo "检测到宿主机: $HOST_OS $HOST_ARCH"

# 根据宿主机架构确定 Linux 容器架构
# macOS arm64 (M1/M2/M3) -> Linux aarch64
# macOS x86_64 (Intel)   -> Linux x86_64
# Linux arm64/aarch64    -> Linux aarch64
# Linux x86_64           -> Linux x86_64

case "$HOST_ARCH" in
    arm64|aarch64)
        TARGET_ARCH="aarch64"
        DOWNLOAD_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-aarch64-unknown-linux-gnu.tar.gz"
        DIR_NAME="uv-aarch64-unknown-linux-gnu"
        ;;
    x86_64|amd64)
        TARGET_ARCH="x86_64"
        DOWNLOAD_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
        DIR_NAME="uv-x86_64-unknown-linux-gnu"
        ;;
    *)
        echo "❌ 错误：不支持的架构 $HOST_ARCH"
        echo "支持的架构: arm64/aarch64, x86_64/amd64"
        exit 1
        ;;
esac

echo "目标平台: Linux $TARGET_ARCH"
echo "下载 uv ${UV_VERSION}..."

cd "$DEPLOY_DIR"

# 下载并解压
echo "正在下载: $DOWNLOAD_URL"
if ! curl -LsSf "$DOWNLOAD_URL" -o uv-linux.tar.gz; then
    echo "❌ 下载失败！请检查网络连接或使用代理"
    echo "提示：如果在国内，可以设置代理："
    echo "  export http_proxy=http://127.0.0.1:7890"
    echo "  export https_proxy=http://127.0.0.1:7890"
    exit 1
fi

tar -xzf uv-linux.tar.gz
mv "$DIR_NAME/uv" ./uv
chmod +x ./uv

# 清理
rm -rf uv-linux.tar.gz "$DIR_NAME"

FILE_SIZE=$(ls -lh uv | awk '{print $5}')
echo ""
echo "✅ uv 下载完成！"
echo "   文件: deploy/uv"
echo "   大小: $FILE_SIZE"
echo "   架构: Linux $TARGET_ARCH"
echo ""
echo "下一步: 运行 'docker compose build' 构建镜像"
