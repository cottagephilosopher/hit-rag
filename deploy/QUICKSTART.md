# 🚀 快速启动指南

## 30秒快速开始

```bash
# 1. 克隆项目
git clone https://github.com/your-org/hit-rag.git
cd hit-rag

# 2. 运行部署脚本
chmod +x deploy/deploy.sh
./deploy/deploy.sh

# 3. 等待完成后访问
open http://localhost:5173  # 文档管理
open http://localhost:3000  # 聊天问答
```

## 📝 部署前准备

### 必需项
- ✅ Docker Desktop 已安装并运行
- ✅ Git 已安装
- ✅ LLM API 密钥（Azure OpenAI 或 OpenAI）

### 可选项
- 📄 准备一些 Markdown 文档
- 🔧 Ollama（如果想使用本地 Embedding）

## 🎯 部署步骤

### 步骤 1: 运行部署脚本

```bash
./deploy/deploy.sh
```

脚本会交互式地引导你：

1. **选择 LLM 提供商**
   ```
   1. Azure OpenAI (推荐)
   2. OpenAI
   ```

2. **输入 API 密钥**
   ```
   Azure OpenAI API Key: sk-xxxxx
   Azure OpenAI Endpoint: https://your-resource.openai.azure.com/
   ```

3. **选择 Embedding 提供商**
   ```
   1. Ollama (本地免费)
   2. Azure OpenAI
   3. OpenAI
   ```

4. **确认文档目录**
   ```
   默认路径: /Users/you/rags/all-md
   使用默认路径？[Y/n]:
   ```

### 步骤 2: 等待服务启动

脚本会自动：
- ✅ 配置环境变量
- ✅ 创建必要目录
- ✅ 启动 Docker 服务（Milvus, Backend, Frontend, Chat View）
- ✅ 验证系统健康

⏱️ 预计等待时间：3-5分钟

### 步骤 3: 访问系统

部署成功后，你会看到：

```
╔═══════════════════════════════════════════════════════════════╗
║                    部署成功！                                   ║
╚═══════════════════════════════════════════════════════════════╝

访问地址：

  📄 文档管理界面:    http://localhost:5173
  💬 聊天问答界面:    http://localhost:3000
  🔌 后端 API:        http://localhost:8000
  🗄️  Milvus 管理:    http://localhost:9091
```

## 📂 使用系统

### 1. 上传文档

1. 打开文档管理界面：http://localhost:5173
2. 将 Markdown 文件拖放到 `all-md` 目录
3. 刷新页面，文档会自动显示

### 2. 处理文档

1. 点击文档名称
2. 等待系统自动分析和切分文档
3. 查看生成的 chunks（文档片段）

### 3. 向量化文档

1. 选择要向量化的 chunks
2. 点击"批量向量化"按钮
3. 等待向量化完成

### 4. 开始问答

1. 打开聊天界面：http://localhost:3000
2. 输入问题，例如："这份文档讲了什么？"
3. 系统会基于文档内容生成回答

## 🔍 验证部署

运行健康检查脚本：

```bash
./deploy/healthcheck.sh
```

你应该看到所有服务状态为 ✓（正常）：

```
[1/7] 检查 Docker 服务状态...
  ✓ Docker 服务正在运行

[2/7] 检查 Milvus 向量数据库...
  ✓ Milvus 健康状态: 正常

[3/7] 检查后端 API 服务...
  ✓ 后端 API: 正常
  ✓ 可用助手: 1
  ✓ 文档数量: 0

[4/7] 检查向量化状态...
  ✓ 向量化服务: 正常

[5/7] 检查前端 UI 服务...
  ✓ 前端 UI: 正常

[6/7] 检查 Chat View 服务...
  ✓ Chat View: 正常

[7/7] 测试 RAG 查询流程...
  ✓ RAG 查询流程: 正常
```

## 🎬 完整示例

### 示例：添加并查询文档

1. **准备文档**
   ```bash
   # 创建示例文档
   cat > all-md/example.md << 'EOF'
   # AutoAgent 介绍

   AutoAgent 是一个企业智能体平台，具有以下特点：

   ## 主要功能
   - 自动化任务执行
   - 低代码开发
   - 高并发可扩展
   - 开放生态系统

   ## 应用场景
   - 数据分析自动化
   - 智能客服
   - 业务流程优化
   EOF
   ```

2. **访问文档管理界面**
   ```
   http://localhost:5173
   ```

3. **处理文档**
   - 点击 "example.md"
   - 等待自动处理完成
   - 查看生成的 chunks

4. **向量化**
   - 选择所有 chunks
   - 点击"批量向量化"
   - 等待完成（约10-30秒）

5. **开始问答**
   ```
   打开: http://localhost:3000

   问: "AutoAgent有哪些主要功能？"

   答: AutoAgent的主要功能包括：
   1. **自动化任务执行**: 平台能够自动拆解任务...
   2. **低代码开发**: 提供低代码化、可视化的开发环境...
   3. **高并发可扩展**: 通过消息队列的异步设计...
   4. **开放生态系统**: 提供丰富的工具和插件生态...
   ```

## 🛠️ 常用命令

```bash
# 查看所有服务状态
docker compose ps

# 查看服务日志
docker compose logs -f

# 查看特定服务日志
docker compose logs -f backend
docker compose logs -f milvus

# 重启服务
docker compose restart

# 停止服务
docker compose down

# 重新构建并启动
docker compose up -d --build

# 健康检查
./deploy/healthcheck.sh
```

## ⚠️ 常见问题

### Q1: 部署脚本卡住不动？

**A**: 可能是在下载 Docker 镜像，耐心等待。可以开启另一个终端查看进度：

```bash
docker compose logs -f
```

### Q2: 端口被占用？

**A**: 修改 `docker-compose.yml` 中的端口映射：

```yaml
services:
  backend:
    ports:
      - "8001:8000"  # 改为其他端口
```

### Q3: Milvus 启动失败？

**A**: 检查磁盘空间和内存：

```bash
# 检查磁盘
df -h

# 检查内存
docker stats

# 清理 Docker
docker system prune -a
```

### Q4: 无法连接 OpenAI API？

**A**: 检查网络和 API 密钥：

```bash
# 测试连接
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Q5: 向量化失败？

**A**: 检查 Embedding 服务：

```bash
# 如果使用 Ollama
ollama list
ollama pull qwen3-embedding:latest

# 测试 Ollama
curl http://localhost:11434/api/embeddings \
  -d '{"model": "qwen3-embedding:latest", "prompt": "test"}'
```

## 📚 下一步

部署成功后，建议阅读：

- [完整部署文档](README.deploy.md) - 详细的配置和维护指南
- [用户使用手册](USAGE.md) - 如何使用系统的各项功能
- [API 文档](http://localhost:8000/docs) - 后端 API 接口文档

## 💡 小贴士

1. **性能优化**
   - 推荐配置：4核心 CPU + 8GB 内存
   - 大量文档时，考虑增加 `VECTOR_BATCH_SIZE`

2. **数据安全**
   - 定期备份 `volumes/` 目录
   - 不要将 `.env` 文件提交到 Git

3. **成本控制**
   - 优先使用 Ollama 本地 Embedding（免费）
   - 使用 Azure OpenAI 的 gpt-3.5-turbo（更便宜）

4. **最佳实践**
   - 文档以主题分类存放在子目录
   - 定期清理不需要的文档
   - 使用有意义的文档命名

## 🆘 获取帮助

遇到问题？

1. 查看日志：`docker compose logs -f`
2. 运行健康检查：`./deploy/healthcheck.sh`
3. 查看完整文档：`README.deploy.md`
4. 提交问题：[GitHub Issues](https://github.com/your-org/hit-rag/issues)

---

**祝你使用愉快！** 🎉
