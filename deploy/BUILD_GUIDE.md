# 构建脚本使用指南

## 概述

提供了两个构建脚本来简化 Docker 构建和部署流程：

- `build.sh` - 功能完整的构建脚本
- `quick-build.sh` - 简化的快速构建脚本

## 快速开始

### 1. 使用您当前的构建命令

```bash
# 进入 deploy 目录
cd deploy

# 使用快速构建脚本（等同于您的命令）
./quick-build.sh --proxy --start
```

### 2. 常用构建场景

```bash
# 普通构建
./quick-build.sh

# 代理构建并启动
./quick-build.sh --proxy --start

# 使用清华源构建
./quick-build.sh --tsinghua --start

# 清除缓存重新构建
./quick-build.sh --proxy --clean --start
```

## 详细使用说明

### quick-build.sh（推荐）

简化的构建脚本，专门用于常用场景：

```bash
# 基本用法
./quick-build.sh [选项]

# 选项说明
--proxy     使用代理构建（默认: host.docker.internal:7897）
--tsinghua  使用清华镜像源
--start     构建后启动服务
--clean     清除缓存重新构建
-h, --help  显示帮助
```

**使用示例：**

```bash
# 1. 代理构建并启动（等同于您的命令）
./quick-build.sh --proxy --start

# 2. 使用清华源构建
./quick-build.sh --tsinghua --start

# 3. 清除缓存重新构建
./quick-build.sh --proxy --clean --start

# 4. 仅构建，不启动
./quick-build.sh --proxy
```

### build.sh（完整功能）

功能完整的构建脚本，支持更多选项：

```bash
# 基本用法
./build.sh [选项]

# 常用选项
-n, --normal            普通编译（默认）
-c, --clean             清除缓存重新编译
-p, --proxy             使用代理编译
-t, --tsinghua          使用清华镜像源编译
-s, --start             编译后启动服务
-d, --daemon            编译后后台启动服务
--no-cache              不使用 Docker 缓存
--pull                  拉取最新基础镜像
```

**使用示例：**

```bash
# 1. 普通构建
./build.sh -n

# 2. 代理构建并启动
./build.sh -p -s

# 3. 清除缓存代理构建并后台启动
./build.sh -c -p -d

# 4. 使用清华源无缓存构建
./build.sh -t --no-cache

# 5. 拉取最新镜像并构建
./build.sh --pull -p -s
```

## 构建命令对比

### 原始命令
```bash
docker compose build --build-arg HTTP_PROXY=http://host.docker.internal:7897 --build-arg HTTPS_PROXY=http://host.docker.internal:7897 --build-arg NO_PROXY=localhost,127.0.0.1,host.docker.internal --build-arg UV_INDEX_URL=https://pypi.org/simple
docker compose up -d
```

### 使用脚本
```bash
# 等同于上述命令
./quick-build.sh --proxy --start
```

## 环境配置

### 代理设置

如果您的代理地址不是默认的 `host.docker.internal:7897`，可以修改脚本中的配置：

```bash
# 编辑 quick-build.sh
PROXY_HOST="your-proxy-host"
PROXY_PORT="your-proxy-port"
```

### 镜像源配置

支持多种镜像源：

- 默认：`https://pypi.org/simple`
- 清华源：`https://pypi.tuna.tsinghua.edu.cn/simple`
- 阿里源：`https://mirrors.aliyun.com/pypi/simple/`

## Linux 系统特殊说明

### 自动系统检测

构建脚本会自动检测 Linux 发行版并使用专用配置：

```bash
# 脚本会自动检测系统类型并使用对应的 compose 文件
./build.sh -p -s
```

### 支持的 Linux 发行版

- **Ubuntu**：使用 `docker-compose.ubuntu.yml`
- **CentOS/RHEL/Rocky/AlmaLinux**：使用 `docker-compose.centos.yml`
- **其他 Linux**：使用 `docker-compose.yml`

### 常见问题

1. **host-gateway 错误**：脚本会自动使用兼容配置
2. **代理问题**：建议使用清华源 `./quick-build.sh --tsinghua --start`
3. **权限问题**：确保用户在 docker 组中
4. **SELinux 问题**：CentOS 系统可能需要调整 SELinux 设置

详细故障排除请参考：
- [UBUNTU_TROUBLESHOOTING.md](./UBUNTU_TROUBLESHOOTING.md) - Ubuntu 系统
- [CENTOS_TROUBLESHOOTING.md](./CENTOS_TROUBLESHOOTING.md) - CentOS 系统

## 故障排除

### 1. 构建失败

```bash
# 清除所有缓存重新构建
./quick-build.sh --clean --proxy --start

# 或者使用完整脚本
./build.sh -c -p --no-cache -s
```

### 2. 代理问题

```bash
# 检查代理是否可用
curl -x http://host.docker.internal:7897 https://pypi.org/simple

# 使用清华源（通常更稳定）
./quick-build.sh --tsinghua --start
```

### 3. 端口冲突

```bash
# 检查端口占用
lsof -i :8086

# 停止现有服务
docker compose down
```

### 4. 权限问题

```bash
# 确保脚本有执行权限
chmod +x build.sh quick-build.sh
```

## 服务管理

### 启动服务
```bash
docker compose up -d
```

### 查看状态
```bash
docker compose ps
```

### 查看日志
```bash
# 查看所有服务日志
docker compose logs -f

# 查看特定服务日志
docker compose logs -f backend
```

### 停止服务
```bash
docker compose down
```

### 重启服务
```bash
docker compose restart
```

## 开发建议

1. **开发环境**：使用 `./quick-build.sh --proxy --start`
2. **生产环境**：使用 `./build.sh -c -p -d` 清除缓存并后台运行
3. **网络问题**：使用 `./quick-build.sh --tsinghua --start` 清华源
4. **调试构建**：使用 `./build.sh -p --no-cache -s` 无缓存构建

## 注意事项

1. 确保在 `deploy` 目录下运行脚本
2. 确保 Docker 和 docker-compose 已安装
3. 代理设置需要根据实际网络环境调整
4. 生产环境建议使用 `--clean` 选项确保构建一致性
