# 阻塞问题修复 - 快速参考

## ✅ 修复完成状态

所有6项修复任务已完成：

1. ✅ 修复图片处理模块：添加URL验证、超时保护和异步处理
2. ✅ 优化子进程调用：将subprocess.run改为异步子进程
3. ✅ 改进文件读取：添加大文件检测和异步读取
4. ✅ 添加全局超时保护机制
5. ✅ 更新file_upload_routes调用图片处理的方式
6. ✅ 测试修复后的系统稳定性

## 🚀 立即部署

### 方法1: 重启服务（推荐）

```bash
cd /Users/idw/rags/hit-rag

# 停止旧服务
pkill -f "uvicorn api_server:app"

# 启动新服务
uv run uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### 方法2: 使用deploy脚本

```bash
cd /Users/idw/rags/hit-rag/deploy
./deploy.sh
```

## 🔍 验证修复

### 快速验证脚本

```bash
cd /Users/idw/rags/hit-rag
./verify_fixes.sh
```

### 手动验证步骤

1. **上传包含大量图片的Markdown文件**
   - 访问前端上传页面
   - 选择包含多个图片引用的.md文件
   - 观察是否卡死（应该不会）

2. **查看日志确认修复生效**
   ```bash
   tail -f logs/rag_preprocessor.log | grep "跳过"
   ```
   应该看到类似输出：
   ```
   跳过AWS签名URL片段: aws4_request&X-Amz-Date=...
   跳过过长URL: 604 字符
   ```

3. **测试超时保护**
   - 上传大文件（50MB+）
   - 应该在5分钟内完成或返回超时错误
   - 不会出现无限等待

## 🐛 问题排查

### 如果仍然卡死

1. **检查Python版本**
   ```bash
   uv run python --version  # 应该是3.10+
   ```

2. **检查修复是否应用**
   ```bash
   grep "_is_valid_image_url" image_uploader.py
   ```

3. **查看详细错误**
   ```bash
   tail -100 logs/rag_preprocessor.log
   ```

### 如果导入错误

```bash
# 重新安装依赖
cd /Users/idw/rags/hit-rag
uv sync
```

### 如果性能问题

```bash
# 检查CPU/内存使用
top -pid $(pgrep -f "uvicorn api_server")
```

## 📊 关键指标

修复后的性能指标：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| **无效URL处理** | 卡死 | 自动跳过 |
| **图片处理超时** | 无限制 | 5分钟 |
| **单图片超时** | 无限制 | 30秒 |
| **文档处理超时** | 10分钟 | 5分钟 |
| **最大文件大小** | 无限制 | 100MB |
| **并发处理** | 串行 | 5并发 |
| **图片数量限制** | 无限制 | 100张/文档 |

## 🔧 配置调整

如需调整超时时间或并发数，编辑以下位置：

### 图片处理超时

`file_upload_routes.py`:
```python
success = await asyncio.wait_for(
    image_processor.process_markdown_file_async(...),
    timeout=300.0  # 修改这里（秒）
)
```

### 并发数

`image_uploader.py`:
```python
def __init__(self, uploader=None, max_concurrent: int = 5):  # 修改这里
    self.max_concurrent = max_concurrent
```

### 文件大小限制

`file_upload_routes.py`:
```python
if file_size_mb > 100:  # 修改这里（MB）
    raise HTTPException(...)
```

## 📝 监控建议

### 关键日志

```bash
# 实时监控
tail -f logs/rag_preprocessor.log | grep -E "(超时|跳过|失败)"

# 统计无效URL数量
grep -c "跳过AWS签名URL" logs/rag_preprocessor.log

# 查看超时事件
grep "处理超时" logs/rag_preprocessor.log
```

### 系统资源

```bash
# CPU使用率
top -pid $(pgrep -f "api_server")

# 内存使用
ps aux | grep "api_server" | awk '{print $4"%"}'
```

## 🆘 紧急回滚

如果出现严重问题，立即回滚：

```bash
cd /Users/idw/rags/hit-rag

# 停止服务
pkill -f "uvicorn api_server:app"

# 回滚代码（假设之前的commit是 abc123）
git stash
git checkout abc123

# 重启服务
uv run uvicorn api_server:app --reload
```

## 📞 联系支持

如遇到问题：

1. 收集日志：`logs/rag_preprocessor.log`
2. 运行验证脚本：`./verify_fixes.sh`
3. 记录错误信息和复现步骤

## 📚 相关文档

- 详细修复说明：`BLOCKING_FIXES_SUMMARY.md`
- 测试脚本：`test_blocking_fixes.py`
- 验证脚本：`verify_fixes.sh`
- 超时保护工具：`utils/timeout_protection.py`

---

**最后更新**: 2025-10-22  
**版本**: 1.0  
**状态**: ✅ 已验证可用

