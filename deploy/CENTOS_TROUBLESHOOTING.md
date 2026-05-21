# CentOS 系统故障排除指南

## 常见问题及解决方案

### 1. host-gateway 错误

**错误信息：**
```
unable to derive the IP value for host-gateway: host-gateway is not supported by the docker-container driver
```

**原因：** CentOS 系统的 Docker 容器驱动不支持 `host-gateway` 特性。

**解决方案：**

1. **使用 CentOS 专用配置（推荐）：**
   ```bash
   # 脚本会自动检测 CentOS 系统并使用 docker-compose.centos.yml
   ./build.sh -p -s
   ```

2. **手动指定 compose 文件：**
   ```bash
   docker compose -f docker-compose.centos.yml build
   docker compose -f docker-compose.centos.yml up -d
   ```

3. **检查 Docker 驱动：**
   ```bash
   docker info | grep "Driver"
   ```

### 2. 代理连接问题

**问题：** 在 CentOS 系统中代理无法正常工作。

**解决方案：**

1. **检查代理服务：**
   ```bash
   # 检查代理是否运行
   curl -x http://host.docker.internal:7897 https://pypi.org/simple
   ```

2. **使用主机 IP 替代 host.docker.internal：**
   ```bash
   # 获取主机 IP
   ip route get 1 | awk '{print $7; exit}'
   
   # 修改 docker-compose.centos.yml 中的代理地址
   # 将 host.docker.internal 替换为实际的主机 IP
   ```

3. **使用清华镜像源（推荐）：**
   ```bash
   ./quick-build.sh --tsinghua --start
   ```

### 3. 权限问题

**问题：** Docker 权限不足或文件权限问题。

**解决方案：**

1. **添加用户到 docker 组：**
   ```bash
   sudo usermod -aG docker $USER
   # 重新登录或执行
   newgrp docker
   ```

2. **检查文件权限：**
   ```bash
   # 确保脚本有执行权限
   chmod +x build.sh quick-build.sh
   
   # 检查目录权限
   ls -la ../logs ../output ../files
   ```

3. **SELinux 问题（如果启用）：**
   ```bash
   # 检查 SELinux 状态
   sestatus
   
   # 临时禁用 SELinux（重启后恢复）
   sudo setenforce 0
   
   # 永久禁用 SELinux（需要重启）
   sudo sed -i 's/SELINUX=enforcing/SELINUX=disabled/' /etc/selinux/config
   ```

### 4. 端口冲突

**问题：** 端口被占用或无法绑定。

**解决方案：**

1. **检查端口占用：**
   ```bash
   # 检查 8086 端口
   sudo netstat -tulpn | grep 8086
   
   # 检查 19530 端口（Milvus）
   sudo netstat -tulpn | grep 19530
   ```

2. **使用不同端口：**
   ```bash
   # 设置环境变量
   export API_PORT=8087
   ./quick-build.sh --proxy --start
   ```

3. **停止冲突服务：**
   ```bash
   # 停止所有相关容器
   docker compose -f docker-compose.centos.yml down
   
   # 强制停止占用端口的进程
   sudo kill -9 $(sudo lsof -t -i:8086)
   ```

### 5. 网络连接问题

**问题：** 容器间无法通信或外部网络访问失败。

**解决方案：**

1. **检查 Docker 网络：**
   ```bash
   docker network ls
   docker network inspect rag-network
   ```

2. **重建网络：**
   ```bash
   docker compose -f docker-compose.centos.yml down
   docker network prune -f
   docker compose -f docker-compose.centos.yml up -d
   ```

3. **检查防火墙：**
   ```bash
   # CentOS 防火墙
   sudo firewall-cmd --state
   sudo firewall-cmd --permanent --add-port=8086/tcp
   sudo firewall-cmd --permanent --add-port=19530/tcp
   sudo firewall-cmd --reload
   ```

### 6. 构建缓存问题

**问题：** 构建失败或使用过期的缓存。

**解决方案：**

1. **清除所有缓存：**
   ```bash
   ./build.sh -c -p -s
   ```

2. **手动清理：**
   ```bash
   docker system prune -a -f
   docker builder prune -a -f
   ```

3. **无缓存构建：**
   ```bash
   ./build.sh -p --no-cache -s
   ```

## CentOS 系统优化建议

### 1. Docker 配置优化

创建或编辑 `/etc/docker/daemon.json`：

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "live-restore": true
}
```

重启 Docker 服务：
```bash
sudo systemctl restart docker
```

### 2. 系统资源优化

1. **增加文件描述符限制：**
   ```bash
   echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
   echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
   ```

2. **优化内存使用：**
   ```bash
   # 在 docker-compose.centos.yml 中添加资源限制
   deploy:
     resources:
       limits:
         memory: 2G
       reservations:
         memory: 1G
   ```

### 3. 网络优化

1. **配置 DNS：**
   ```bash
   # 编辑 /etc/docker/daemon.json
   {
     "dns": ["8.8.8.8", "8.8.4.4"]
   }
   ```

2. **使用国内镜像源：**
   ```bash
   # 配置 Docker 镜像加速
   sudo mkdir -p /etc/docker
   sudo tee /etc/docker/daemon.json <<-'EOF'
   {
     "registry-mirrors": [
       "https://docker.m.daocloud.io",
       "https://hub-mirror.c.163.com"
     ]
   }
   EOF
   ```

### 4. 系统服务管理

1. **启用 Docker 服务：**
   ```bash
   sudo systemctl enable docker
   sudo systemctl start docker
   ```

2. **检查服务状态：**
   ```bash
   sudo systemctl status docker
   ```

## 调试命令

### 查看详细日志
```bash
# 查看构建日志
docker compose -f docker-compose.centos.yml build --progress=plain

# 查看服务日志
docker compose -f docker-compose.centos.yml logs -f backend

# 查看 Milvus 日志
docker compose -f docker-compose.centos.yml logs -f milvus
```

### 进入容器调试
```bash
# 进入后端容器
docker exec -it hit-rag-backend bash

# 进入 Milvus 容器
docker exec -it hit-rag-milvus bash
```

### 检查服务状态
```bash
# 检查容器状态
docker compose -f docker-compose.centos.yml ps

# 检查健康状态
curl http://localhost:8086/health
curl http://localhost:9091/healthz
```

### 系统信息检查
```bash
# 检查系统版本
cat /etc/os-release

# 检查内核版本
uname -r

# 检查 Docker 版本
docker --version
docker-compose --version
```

## 联系支持

如果问题仍然存在，请提供以下信息：

1. CentOS 版本：`cat /etc/os-release`
2. Docker 版本：`docker --version`
3. 错误日志：`docker compose -f docker-compose.centos.yml logs`
4. 系统信息：`uname -a`
5. SELinux 状态：`sestatus`








