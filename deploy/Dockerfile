FROM docker.m.daocloud.io/library/python:3.11-slim

# 代理和索引配置（可通过 --build-arg 传入）
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 允许构建阶段使用代理和备用镜像源
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}
ENV UV_INDEX_URL=${UV_INDEX_URL}

# 设置工作目录
WORKDIR /app

# 配置 apt 使用国内镜像源加速
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true

# 安装系统依赖（apt 使用国内镜像），禁用代理避免宿主 DNS 差异
RUN set -ex \ 
    && export HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= \ 
    && apt-get update \ 
    && apt-get install -y \ 
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
# 使用指定镜像源同步依赖，命中不到的包会走代理
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-install-project --index-url ${UV_INDEX_URL}

# 再复制项目文件（代码改动不会触发依赖重装）
COPY . .

# 清理仅用于构建阶段的代理变量，避免污染运行时
ENV HTTP_PROXY= \
    HTTPS_PROXY= \
    NO_PROXY= \
    http_proxy= \
    https_proxy= \
    no_proxy=

# 暴露端口（通过环境变量配置）
EXPOSE ${API_PORT:-8086}

# 默认命令
CMD ["uv", "run", "python", "api_server.py"]
