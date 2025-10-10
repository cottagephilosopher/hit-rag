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

# 安装 uv（Python 包管理器）并设置环境变量
ENV CARGO_HOME="/root/.cargo"
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY . .

# 安装 Python 依赖（在同一个 shell 中设置 PATH 并运行）
RUN . $CARGO_HOME/env && uv sync --frozen

# 暴露端口
EXPOSE 8000

# 默认命令（使用 shell 形式确保环境变量生效）
CMD ["/bin/sh", "-c", ". $CARGO_HOME/env && uv run python api_server.py"]
