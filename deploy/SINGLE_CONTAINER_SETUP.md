# 单容器部署方案说明

## 📦 架构概述

单容器方案将 Vue.js 和 Next.js 两个前端应用整合到一个 Nginx 容器中，通过统一入口（端口 80）提供服务。

### 容器架构

```
┌─────────────────────────────────────────────────────────────┐
│                      rags/ (父目录)                           │
│                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │   hit-rag/     │  │  hit-rag-ui/   │  │ versa-chat-    │ │
│  │                │  │                │  │    -view/      │ │
│  │  - Backend     │  │  - Vue.js 前端 │  │  - Next.js 前端│ │
│  │  - 部署脚本     │  │                │  │                │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
│                                                               │
│  Docker Compose 构建上下文: rags/                             │
└─────────────────────────────────────────────────────────────┘

运行时容器:

┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐
│   Milvus    │  │   Backend   │  │  Frontend (Unified)       │
│  :19530     │  │   :8000     │  │  :80                     │
└─────────────┘  └─────────────┘  │  ┌───────────────────┐   │
                                   │  │ Nginx             │   │
                                   │  │ ├─ /ui -> Vue.js  │   │
                                   │  │ ├─ /chat -> Next  │   │
                                   │  │ └─ /api -> Backend│   │
                                   │  └───────────────────┘   │
                                   │  ┌───────────────────┐   │
                                   │  │ Next.js Server    │   │
                                   │  │ (localhost:3000)  │   │
                                   │  └───────────────────┘   │
                                   └──────────────────────────┘
```

## 🏗️ 构建流程

### Dockerfile.frontend-unified 多阶段构建

```dockerfile
# Stage 1: 构建 Vue.js (hit-rag-ui)
FROM node:20-slim AS build-vue-ui
WORKDIR /build-vue
COPY hit-rag-ui/package*.json ./
RUN npm install
COPY hit-rag-ui/ ./
RUN npm run build
# 输出: /build-vue/dist

# Stage 2: 构建 Next.js (versa-chat-view)
FROM node:20-slim AS build-chat-view
WORKDIR /build-chat
COPY versa-chat-view/package*.json ./
RUN npm install
COPY versa-chat-view/ ./
RUN npm run build
# 输出: /build-chat/.next

# Stage 3: 组装最终镜像
FROM nginx:alpine
RUN apk add --no-cache nodejs npm

# 复制 Nginx 配置
COPY hit-rag/nginx-unified.conf /etc/nginx/conf.d/default.conf

# 复制 Vue.js 构建产物
COPY --from=build-vue-ui /build-vue/dist /usr/share/nginx/html/ui

# 复制 Next.js 应用
WORKDIR /app/chat
COPY --from=build-chat-view /build-chat/.next ./.next
COPY --from=build-chat-view /build-chat/node_modules ./node_modules
COPY --from=build-chat-view /build-chat/package*.json ./

# 启动脚本：后台启动 Next.js，前台启动 Nginx
CMD ["/start.sh"]
```

### 构建上下文

**关键点**：Dockerfile 需要从父目录（`rags/`）构建，才能访问三个项目目录：

```bash
# 正确的构建命令
cd /Users/idw/rags
docker build -f hit-rag/Dockerfile.frontend-unified -t hit-rag-frontend .

# 错误的构建命令（会失败）
cd /Users/idw/rags/hit-rag
docker build -f Dockerfile.frontend-unified -t hit-rag-frontend .
# ❌ 错误：无法找到 hit-rag-ui/ 和 versa-chat-view/ 目录
```

## 🔧 Docker Compose 配置

[docker-compose.single.yml](docker-compose.single.yml:65):

```yaml
frontend:
  build:
    context: ../  # 父目录 (rags/)
    dockerfile: hit-rag/Dockerfile.frontend-unified
  container_name: hit-rag-frontend-unified
  ports:
    - "80:80"
```

**说明**：
- `context: ../` 将构建上下文设置为父目录
- `dockerfile: hit-rag/Dockerfile.frontend-unified` 指定 Dockerfile 相对于上下文的路径

## 🌐 Nginx 路由配置

[nginx-unified.conf](nginx-unified.conf) 提供统一入口：

```nginx
upstream backend {
    server backend:8000;
}

upstream chat_app {
    server localhost:3000;  # Next.js 在容器内运行
}

server {
    listen 80;

    # 后端 API 代理
    location /api/ {
        proxy_pass http://backend;
        # SSE 支持
        proxy_buffering off;
        proxy_read_timeout 300s;
    }

    # Next.js 聊天界面
    location /chat {
        proxy_pass http://chat_app;
    }

    # Next.js 静态资源
    location /_next/ {
        proxy_pass http://chat_app;
    }

    # Vue.js 文档管理界面
    location /ui {
        alias /usr/share/nginx/html/ui;
        try_files $uri $uri/ /ui/index.html;
    }

    # 根路径重定向
    location = / {
        return 301 /ui/;
    }
}
```

## 🚀 部署步骤

### 方式一：使用部署脚本（推荐）

```bash
# 1. 确保在父目录
cd /Users/idw/rags

# 2. 运行部署脚本
./hit-rag/deploy/deploy-single.sh
```

脚本会自动：
- 检查项目结构
- 配置 `.env` 文件
- 设置文档目录
- 构建并启动容器
- 验证服务健康

### 方式二：手动部署

```bash
# 1. 进入 hit-rag 目录
cd /Users/idw/rags/hit-rag

# 2. 配置环境变量
cp env.template .env
# 编辑 .env 文件，配置 LLM API 密钥

# 3. 准备文档目录
mkdir -p all-md
# 将 Markdown 文档放入 all-md/ 目录

# 4. 启动服务
docker compose -f deploy/docker-compose.single.yml up -d --build

# 5. 查看日志
docker compose -f deploy/docker-compose.single.yml logs -f
```

## 📝 访问地址

部署成功后，所有服务通过统一端口（80）访问：

| 服务 | URL | 说明 |
|------|-----|------|
| 主页 | http://localhost | 自动重定向到 `/ui/` |
| 文档管理 | http://localhost/ui | Vue.js 界面 |
| 聊天问答 | http://localhost/chat | Next.js 界面 |
| 后端 API | http://localhost/api | FastAPI 后端 |
| API 文档 | http://localhost/docs | Swagger UI |

## 🔍 故障排查

### 问题 1：构建失败，提示找不到 hit-rag-ui 目录

**原因**：构建上下文不正确

**解决**：
```bash
# 确保在父目录执行
cd /Users/idw/rags
docker compose -f hit-rag/deploy/docker-compose.single.yml up -d --build
```

### 问题 2：前端容器启动失败

**诊断**：
```bash
# 查看容器日志
cd /Users/idw/rags/hit-rag
docker compose -f deploy/docker-compose.single.yml logs frontend

# 进入容器检查
docker compose -f deploy/docker-compose.single.yml exec frontend sh
ps aux | grep -E 'nginx|node'
```

**常见原因**：
- Next.js 服务未启动：检查 `/start.sh` 脚本
- Nginx 配置错误：`nginx -t` 测试配置

### 问题 3：/chat 路由 404

**原因**：Next.js 服务未运行或 Nginx 代理配置错误

**解决**：
```bash
# 进入容器
docker compose -f deploy/docker-compose.single.yml exec frontend sh

# 检查 Next.js 是否运行
ps aux | grep node

# 手动启动 Next.js（测试）
cd /app/chat
NODE_ENV=production npm start

# 检查端口 3000 是否监听
netstat -tuln | grep 3000
```

### 问题 4：API 代理不工作

**诊断**：
```bash
# 从容器内测试后端连接
docker compose -f deploy/docker-compose.single.yml exec frontend sh
wget -O- http://backend:8000/api/assistants
```

**解决**：
- 确保 backend 服务健康：`docker compose -f deploy/docker-compose.single.yml ps`
- 检查网络：`docker network inspect hit-rag_rag-network`

## 🆚 对比多容器方案

| 特性 | 多容器方案 | 单容器方案 |
|------|-----------|-----------|
| 容器数量 | 4 个 | 3 个 |
| 端口占用 | 19530, 9091, 8000, 5173, 3000 | 19530, 9091, 80 |
| 访问入口 | 多个独立端口 | 统一端口 80 |
| 热重载 | ✅ 支持 | ❌ 需重新构建 |
| 开发便利 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 生产部署 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 资源占用 | ~1.2GB | ~900MB |
| 构建时间 | ~180秒 | ~240秒 |
| 启动时间 | ~90秒 | ~60秒 |

## 📚 相关文档

- [完整部署指南](README.deploy.md)
- [部署方案对比](DEPLOYMENT_COMPARISON.md)
- [快速开始](QUICKSTART.md)
- [测试脚本](test-deployment.sh)

## 💡 最佳实践

### 生产环境

1. **启用 HTTPS**
   ```nginx
   # 修改 nginx-unified.conf
   listen 443 ssl http2;
   ssl_certificate /path/to/cert.pem;
   ssl_certificate_key /path/to/key.pem;
   ```

2. **配置日志轮转**
   ```bash
   # logrotate 配置
   /var/log/nginx/*.log {
       daily
       missingok
       rotate 14
       compress
       delaycompress
       notifempty
   }
   ```

3. **资源限制**
   ```yaml
   # docker-compose.single.yml
   frontend:
     deploy:
       resources:
         limits:
           cpus: '1.0'
           memory: 512M
   ```

### 开发环境

对于开发，建议使用多容器方案（`deploy/deploy.sh`）以获得热重载支持。

## 🎯 总结

单容器方案适合：
- ✅ 生产环境部署
- ✅ 简化的端口管理
- ✅ 统一的访问入口
- ✅ 较低的资源占用
- ✅ 配置 SSL/反向代理

关键要点：
1. 必须从父目录（`rags/`）构建
2. Nginx 作为统一入口和反向代理
3. Next.js 在容器内运行，通过 Nginx 代理访问
4. Vue.js 静态文件直接由 Nginx 服务
