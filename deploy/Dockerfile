# 使用 Python 3.11 官方镜像作为基础镜像（使用国内镜像加速）
FROM docker.m.daocloud.io/library/python:3.11-slim

# 设置工作目录
WORKDIR /app

# 配置 apt 使用国内镜像源加速
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 配置 pip 使用国内镜像源加速（清华源）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 uv（Python 包管理器）- 使用 pip 安装更可靠
RUN pip install --no-cache-dir uv

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY . .

# 安装 Python 依赖（使用国内镜像源加速）
RUN uv sync --frozen --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露端口
EXPOSE 8000

# 默认命令
CMD ["uv", "run", "python", "api_server.py"]
