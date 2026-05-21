# 文件上传和转换功能使用指南

## 功能概述

该功能允许用户上传各种格式的文档（PDF、Word、PPT等），通过 MinerU API 自动转换为 Markdown 格式，并进行智能解析和切片处理。

## 完整流程

```
用户上传文件
    ↓
保存到 FILE_DIR/uploads/ 目录
    ↓
调用 MinerU API 转换
    ↓
轮询转换进度 (0-100%)
    ↓
下载转换结果到 FILE_DIR/converted/ 目录
    ↓
处理Markdown中的图片（可选）
  - 提取图片引用
  - 上传图片到对象存储（如火山引擎TOS）
  - 替换图片URL为简化路径
    ↓
将处理后的Markdown保存到 ALL_MD_DIR
    ↓
自动触发文档处理 (解析+切片+标签)
    ↓
轮询处理状态
    ↓
完成！文档就绪
```

## 进度显示详解

弹窗会实时显示以下阶段：

1. **准备上传** (0-10%)
   - 📦 准备上传文件

2. **文件上传** (10-30%)
   - ⬆️ 正在上传文件到服务器

3. **MinerU 转换** (30-80%)
   - ⏳ MinerU 正在转换文档 (显示实时进度)
   - 支持的格式：PDF, DOCX, PPTX, XLSX, JPG, PNG

4. **转换完成 & 图片处理** (80-85%)
   - ✅ 转换完成！文件已保存到 FILE_DIR/converted
   - 🖼️ 处理Markdown中的图片（如果配置了TOS）
     - 上传图片到对象存储
     - 替换为简化的URL路径

5. **文档处理** (85-95%)
   - 🔄 开始解析文档
   - ⏳ 正在进行智能解析和切片处理
   - 包括：Markdown 解析、Token 切分、标签推理

6. **处理完成** (95-100%)
   - 🎉 处理完成！文档已就绪

## 配置要求

### 1. 环境变量配置

在 `.env` 文件中添加：

```bash
# 文件存储目录（必需）
FILE_DIR=/path/to/files

# MinerU API 配置（必需）
# 注意：是 mineru.net 不是 api.mineru.net
MINERU_API_BASE=https://mineru.net
MINERU_API_KEY=your_mineru_api_key_here

# 火山引擎TOS配置（必需！MinerU需要通过URL访问文件）
VOLC_ACCESSKEY=your_volc_access_key
VOLC_SECRETKEY=your_volc_secret_key
TOS_OBS_ENDPOINT=tos-cn-beijing.volces.com
TOS_OBS_REGION=cn-beijing
TOS_OBS_BUCKET_NAME=your_bucket_name
TOS_OBS_ACCESS_URL=https://your-bucket.tos-cn-beijing.volces.com

# 前端UI地址（可选）
FRONTEND_UI_URL=http://localhost:5173
```

**⚠️ 重要说明：**

MinerU API 不支持直接上传文件，必须通过URL访问。因此：
- **必须配置TOS对象存储** - 系统会先上传文件到TOS，再把URL传给MinerU
- **Bucket需要公开访问** - 确保MinerU能访问到上传的文件
- **双重作用**：
  1. 上传待转换文件（存到 `mineru-uploads/` 目录）
  2. 优化转换后的图片URL（存到 `mineru/` 目录）

### 2. 获取 MinerU API 密钥

1. 访问 [MinerU 官网](https://mineru.net)
2. 注册账号
3. 在 API 管理页面创建 API 密钥
4. 将密钥配置到 `.env` 文件

### 3. 配置火山引擎TOS（必需）

**为什么必须配置TOS？**

MinerU API 要求通过URL访问文件，不支持直接上传。因此需要：
1. 先上传文件到TOS对象存储
2. 获取文件的公开访问URL
3. 将URL传递给MinerU进行转换

**配置步骤：**

1. 注册[火山引擎](https://console.volcengine.com/)账号
2. 开通TOS对象存储服务
3. 创建Bucket（建议区域：cn-beijing）
4. **重要：设置Bucket为公开读取**
   - 在Bucket权限设置中，允许公开读取
   - 或配置访问策略，允许MinerU访问
5. 获取AccessKey和SecretKey
   - 在"访问控制" → "密钥管理"中创建
6. 将配置信息填入 `.env` 文件

**TOS的双重作用：**
- ✅ **上传待转换文件** - 提供给MinerU访问（`mineru-uploads/` 目录）
- ✅ **优化图片URL** - 将转换后的图片重新上传并简化路径（`mineru/` 目录）

### 4. 数据库初始化

```bash
cd /Users/idw/rags/hit-rag
sqlite3 .dbs/hit-rag.db < .dbs/file_upload_schema.sql
```

### 4. 安装依赖

```bash
uv add aiohttp
```

## 使用方法

### 前端操作

1. 在文档列表页点击 📤 **上传文件** 按钮
2. 在弹窗中：
   - **拖拽文件**到上传区域，或
   - **点击选择**文件
3. 选择文件后，点击 **上传并转换** 按钮
4. 等待处理完成（弹窗会显示完整进度）
5. 处理完成后弹窗自动关闭，文档列表自动刷新

### API 接口

#### 上传文件

```bash
POST /api/upload/file
Content-Type: multipart/form-data

file: <file binary>
```

**响应示例：**
```json
{
  "id": 1,
  "original_filename": "document.pdf",
  "file_size": 1048576,
  "file_type": "application/pdf",
  "status": "pending",
  "created_at": "2024-01-01T00:00:00"
}
```

#### 查询状态

```bash
GET /api/upload/{upload_id}/status
```

**响应示例：**
```json
{
  "id": 1,
  "original_filename": "document.pdf",
  "status": "converting",
  "conversion_progress": 65,
  "converted_md_filename": null,
  "error_message": null
}
```

**状态说明：**
- `pending` - 待上传
- `uploading` - 上传中
- `converting` - MinerU 转换中
- `completed` - 转换完成
- `error` - 转换失败

#### 获取上传列表

```bash
GET /api/upload/list?limit=50
```

#### 删除上传记录

```bash
DELETE /api/upload/{upload_id}
```

## 支持的文件格式

| 格式 | 扩展名 | MIME Type |
|-----|--------|-----------|
| PDF | .pdf | application/pdf |
| Word | .doc, .docx | application/msword, application/vnd.openxmlformats-officedocument.wordprocessingml.document |
| PowerPoint | .ppt, .pptx | application/vnd.ms-powerpoint, application/vnd.openxmlformats-officedocument.presentationml.presentation |
| Excel | .xls, .xlsx | application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet |
| 图片 | .jpg, .png | image/jpeg, image/png |

## 文件存储结构

所有上传和转换的文件都保存在 `FILE_DIR` 配置的目录中：

```
FILE_DIR/                       # 由 .env 中的 FILE_DIR 配置
├── uploads/                    # 原始上传文件
│   └── 20241017_143052_document.pdf
└── converted/                  # MinerU 转换后的 Markdown
    └── document_converted.md

all-md/                         # 文档处理目录（从 converted 复制）
└── document_converted.md       # 供文档解析和切片使用

.dbs/
└── hit-rag.db                 # 包含 file_uploads 表
```

**目录说明：**
- `FILE_DIR/uploads/` - 保存原始上传的文件（PDF、DOCX、PPTX等）
- `FILE_DIR/converted/` - 保存 MinerU 转换后的 Markdown 文件
- `ALL_MD_DIR/` - 文档处理目录，从 converted 复制一份供后续处理使用

## 数据库表结构

```sql
CREATE TABLE file_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_filename TEXT NOT NULL,
    file_size INTEGER,
    file_type TEXT,
    upload_path TEXT NOT NULL,
    mineru_task_id TEXT,
    converted_md_filename TEXT,
    converted_md_path TEXT,
    status TEXT DEFAULT 'pending',
    conversion_started_at TIMESTAMP,
    conversion_completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 错误处理

### 常见错误及解决方法

1. **MINERU_API_KEY 未配置**
   - 错误：`MINERU_API_KEY 未配置，请在.env文件中设置`
   - 解决：在 `.env` 文件中添加 `MINERU_API_KEY`

2. **不支持的文件格式**
   - 错误：`不支持的文件类型: application/zip`
   - 解决：只上传支持的文件格式（PDF、DOCX、PPTX、XLSX、JPG、PNG）

3. **MinerU API 返回错误**
   - 错误：`MinerU API返回错误: 401 - Unauthorized`
   - 解决：检查 API 密钥是否正确、是否过期

4. **转换超时**
   - 错误：`转换超时（超过10分钟）`
   - 解决：文件可能太大或太复杂，可以稍后手动查看转换状态

5. **文档处理失败**
   - 错误：`文件已转换，但处理失败`
   - 解决：转换成功但处理失败，可以在文档列表中手动触发处理

## 性能优化建议

1. **文件大小限制**
   - 建议单个文件不超过 50MB
   - 大文件转换时间较长，请耐心等待

2. **批量上传**
   - 暂不支持批量上传
   - 建议一个一个上传，确保每个文件都正确处理

3. **网络要求**
   - 需要稳定的网络连接到 MinerU API
   - 企业环境可能需要配置代理

## 安全注意事项

1. **API 密钥保护**
   - 不要将 `.env` 文件提交到版本控制
   - 定期更换 API 密钥

2. **文件上传限制**
   - 后端应设置文件大小限制
   - 建议配置文件类型白名单

3. **存储清理**
   - 定期清理 `uploads/` 目录中的原始文件
   - 已处理的文件可以删除原始上传文件

## 故障排查

### 查看上传记录

```bash
sqlite3 .dbs/hit-rag.db
SELECT * FROM file_uploads ORDER BY created_at DESC LIMIT 10;
```

### 查看转换状态

```bash
curl http://localhost:${API_PORT:-8086}/api/upload/1/status
```

### 查看文档处理状态

```bash
curl http://localhost:${API_PORT:-8086}/api/documents/document_converted.md/status
```

### 重新处理文档

如果文档转换成功但处理失败，可以在文档列表中点击"处理"按钮手动触发处理。

## 开发调试

### 本地测试 MinerU API

如果 MinerU API 不可用，可以暂时注释掉实际调用，模拟返回：

```python
# 在 file_upload_routes.py 中
async def call_mineru_upload_api(file_path: Path, file_type: str) -> dict:
    # 模拟 MinerU API 响应
    import uuid
    return {
        'task_id': str(uuid.uuid4()),
        'status': 'pending'
    }
```

### 查看日志

```bash
# 查看 FastAPI 服务日志
tail -f logs/rag_preprocessor.log

# 查看前端控制台
# 在浏览器开发者工具中查看
```

## 未来改进

- [ ] 支持批量上传
- [ ] 上传进度显示（而不是完成后才显示）
- [ ] 支持暂停/取消转换
- [ ] 转换历史记录查看
- [ ] 失败任务自动重试
- [ ] 支持更多文件格式
- [ ] 转换质量评估
- [ ] 自定义转换参数
