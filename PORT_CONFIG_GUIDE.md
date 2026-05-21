# 端口配置指南

## 概述

系统现在支持通过环境变量配置 API 服务器端口，默认端口为 8086。

## 配置方式

### 1. 环境变量配置

在 `.env` 文件中设置以下变量：

```bash
# API 服务器端口（默认：8086）
API_PORT=8086
# API 服务器主机地址（默认：0.0.0.0，允许外部访问）
API_HOST=0.0.0.0
# 是否启用热重载（开发模式，默认：true）
API_RELOAD=true
# 工作进程数量（生产环境建议设置为 CPU 核心数）
API_WORKERS=1
```

### 2. 不同环境的端口配置示例

#### 开发环境
```bash
API_PORT=8086
API_HOST=0.0.0.0
API_RELOAD=true
API_WORKERS=1
```

#### 生产环境
```bash
API_PORT=80
API_HOST=0.0.0.0
API_RELOAD=false
API_WORKERS=4
```

#### 测试环境
```bash
API_PORT=8087
API_HOST=127.0.0.1
API_RELOAD=false
API_WORKERS=1
```

## 使用方法

### 1. 本地开发

```bash
# 使用默认端口 8086
uv run python api_server.py

# 使用自定义端口
API_PORT=8087 uv run python api_server.py
```

### 2. Docker 部署

```bash
# 使用默认端口
docker-compose up

# 使用自定义端口
API_PORT=8087 docker-compose up
```

### 3. 环境变量文件

创建 `.env` 文件：
```bash
cp env.template .env
# 编辑 .env 文件，设置 API_PORT=8087
```

## 配置验证

运行配置验证命令：
```bash
uv run python config.py
```

## 相关文件

- `config.py` - 服务器配置类
- `api_server.py` - 主服务器文件
- `env.template` - 环境变量模板
- `deploy/docker-compose.yml` - Docker 编排文件
- `deploy/Dockerfile` - Docker 构建文件

## 注意事项

1. **端口冲突**：确保选择的端口没有被其他服务占用
2. **防火墙**：确保防火墙允许访问配置的端口
3. **Docker 映射**：Docker 容器内的端口映射会自动使用配置的端口
4. **健康检查**：Docker 健康检查会自动使用配置的端口
5. **文档更新**：相关文档中的端口引用已更新为使用环境变量

## 故障排除

### 端口被占用
```bash
# 检查端口占用
lsof -i :8086
# 或
netstat -tulpn | grep 8086
```

### 权限问题
```bash
# 如果使用 80 端口，需要 sudo
sudo API_PORT=80 uv run python api_server.py
```

### Docker 端口映射
确保 Docker 端口映射正确：
```yaml
ports:
  - "${API_PORT:-8086}:${API_PORT:-8086}"
```
