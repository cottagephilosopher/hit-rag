# 使用 Python 3.11 官方镜像作为基础镜像（使用国内镜像加速）
FROM docker.m.daocloud.io/library/python:3.11-slim

# 设置工作目录
WORKDIR /app

# 接收代理参数（仅用于 uv sync）
ARG UV_HTTP_PROXY
ARG UV_HTTPS_PROXY

# 配置 apt 使用国内镜像源加速
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true

# 安装系统依赖（apt 使用国内镜像，不需要代理）
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 从构建上下文复制 uv 二进制文件（避免网络下载）
# 注意：构建上下文是项目根目录，所以路径是 deploy/uv
COPY deploy/uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv && uv --version

# 先复制依赖文件（利用 Docker 缓存）
COPY pyproject.toml uv.lock ./

# 安装 Python 依赖（使用 BuildKit 缓存挂载大幅加速）
# 缓存位置：/root/.cache/uv（uv 默认缓存目录）
# 使用代理（如果提供）+ 清华镜像源
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export http_proxy="${UV_HTTP_PROXY:-}" https_proxy="${UV_HTTPS_PROXY:-}" && \
    uv sync --frozen --no-install-project --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 再复制项目文件（代码改动不会触发依赖重装）
COPY . .

# 暴露端口
EXPOSE 8086

# 默认命令
CMD ["uv", "run", "python", "api_server.py"]
