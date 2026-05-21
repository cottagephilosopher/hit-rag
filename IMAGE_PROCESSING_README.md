# 图片处理功能说明

## 功能概述

在文件上传和MinerU转换流程中，增加了图片处理功能，用于优化Markdown文件中的图片URL。

## 处理流程

```
MinerU转换完成
    ↓
保存到 FILE_DIR/converted/
    ↓
读取Markdown文件
    ↓
提取图片引用 ![alt](url)
    ↓
查找本地图片文件
    ↓
上传到TOS对象存储（如果配置）
    ↓
替换为简化URL
    ↓
保存到 ALL_MD_DIR/
    ↓
供后续文档处理使用
```

## 为什么需要图片处理？

### 问题

MinerU转换后的Markdown文件中，图片URL可能是复杂的路径：

```markdown
![image](https://iss.tos-cn-beijing.volces.com/mineru/abc123def456/xyz789/image.jpg)
```

**问题：**
1. URL路径复杂，不便管理
2. 图片可能存储在临时路径
3. 依赖外部服务的URL稳定性

### 解决方案

将图片上传到自己的对象存储，使用简化路径：

```markdown
![image](https://your-bucket.tos-cn-beijing.volces.com/mineru/image.jpg)
```

**优势：**
1. URL简洁清晰
2. 图片存储在自己的Bucket，可控
3. 便于统一管理和备份

## 实现细节

### 核心模块：image_uploader.py

#### ImageUploader 类

负责图片上传到火山引擎TOS：

```python
uploader = ImageUploader()
new_url = uploader.upload_image("mineru/image.jpg", "/path/to/local/image.jpg")
```

**功能：**
- 初始化TOS客户端
- 上传图片到指定Bucket
- 返回访问URL
- 缓存已上传图片，避免重复

#### MarkdownImageProcessor 类

负责处理Markdown文件中的图片：

```python
processor = MarkdownImageProcessor(uploader)
success = processor.process_markdown_file(input_file, output_file)
```

**功能：**
- 提取图片引用：`![alt](url)`
- 查找本地图片文件（支持多种目录结构）
- 调用uploader上传图片
- 替换URL为新地址
- 保存处理后的文件
- 统计处理结果

### 集成到文件上传流程

在 `file_upload_routes.py` 中，MinerU转换完成后：

```python
# 1. 下载转换结果
converted_path = CONVERTED_DIR / md_filename
await download_mineru_result(result_url, converted_path)

# 2. 处理图片
image_processor = create_image_processor()
md_path = ALL_MD_DIR / md_filename
success = image_processor.process_markdown_file(converted_path, md_path)

# 3. 如果处理失败，使用原始文件
if not success:
    shutil.copy2(converted_path, md_path)
```

## 配置说明

### 必需配置

```bash
# FILE_DIR - 文件存储根目录
FILE_DIR=../files
```

### 可选配置（TOS图片上传）

如果不配置，将跳过图片上传，直接使用MinerU返回的URL：

```bash
# 火山引擎TOS配置
VOLC_ACCESSKEY=your_access_key
VOLC_SECRETKEY=your_secret_key
TOS_OBS_ENDPOINT=tos-cn-beijing.volces.com
TOS_OBS_REGION=cn-beijing
TOS_OBS_BUCKET_NAME=your_bucket
TOS_OBS_ACCESS_URL=https://your-bucket.tos-cn-beijing.volces.com
```

### 配置获取方式

1. 访问 [火山引擎控制台](https://console.volcengine.com/)
2. 开通TOS对象存储服务
3. 创建Bucket
4. 在"访问控制"中创建AccessKey和SecretKey
5. 记录Bucket的访问域名

## 目录结构

```
FILE_DIR/
├── uploads/                    # 原始上传文件
│   └── 20241017_143052_doc.pdf
└── converted/                  # 转换后的原始MD（含复杂URL）
    ├── doc_converted.md        # 原始转换结果
    └── images/                 # 本地图片（如果有）
        └── image.jpg

all-md/                         # 处理后的MD（已优化URL）
└── doc_converted.md            # 图片URL已替换
```

## 错误处理

### 配置缺失

如果TOS配置不完整，系统会：
1. 打印警告日志
2. 跳过图片上传
3. 直接使用原始URL
4. 不影响文档转换流程

```
WARNING: 图片上传配置不完整: 缺少TOS配置环境变量: VOLC_ACCESSKEY, VOLC_SECRETKEY
WARNING: 将跳过图片上传，直接使用原始URL
```

### 图片文件缺失

如果找不到本地图片文件：
1. 记录警告日志
2. 保留原始URL
3. 继续处理其他图片

```
WARNING: 找不到本地图片: image.jpg
```

### 上传失败

如果图片上传到TOS失败：
1. 记录错误日志
2. 保留原始URL
3. 继续处理其他图片

```
ERROR: 上传图片失败: TOS服务端错误: 403 - Access Denied
```

### 完全失败

如果整个图片处理流程失败：
1. 记录错误日志
2. 将原始转换结果复制到ALL_MD_DIR
3. 不影响后续文档处理

```
ERROR: 图片处理出错: ...
WARNING: 将使用原始文件
```

## 日志示例

### 成功处理

```
INFO: MinerU转换完成，文件已保存: /path/to/FILE_DIR/converted/doc_converted.md
INFO: 开始处理Markdown中的图片...
INFO: 已初始化图片上传器
INFO: 找到 3 个图片引用
INFO: 图片上传成功: mineru/image1.jpg
INFO: 图片上传成功: mineru/image2.jpg
INFO: 图片已上传，跳过: mineru/image3.jpg
INFO: 成功处理文件: doc_converted.md
INFO:   - 替换了 3 个图片链接
INFO: 图片处理完成，文件已保存到: /path/to/all-md/doc_converted.md

图片处理统计:
  总图片数: 3
  成功上传: 2
  上传失败: 0
  跳过处理: 1
```

### 跳过图片上传

```
INFO: MinerU转换完成，文件已保存: /path/to/FILE_DIR/converted/doc_converted.md
INFO: 开始处理Markdown中的图片...
WARNING: 图片上传配置不完整: 缺少TOS配置环境变量: VOLC_ACCESSKEY
WARNING: 将跳过图片上传，直接使用原始URL
INFO: 文件中没有图片引用: doc_converted.md
```

## 性能优化

### 上传缓存

已上传的图片会被缓存，避免重复上传：

```python
if object_key in self.uploaded_cache:
    logger.info(f"图片已上传，跳过: {object_key}")
    return f"{self.base_url}/{object_key}"
```

### 批量处理

未来可以考虑批量上传图片以提升性能。

## 扩展性

### 支持其他对象存储

可以轻松扩展支持其他对象存储服务：

```python
class S3ImageUploader:
    """AWS S3 图片上传器"""
    def upload_image(self, object_key, file_path):
        # 实现S3上传逻辑
        pass

class AliOSSImageUploader:
    """阿里云OSS 图片上传器"""
    def upload_image(self, object_key, file_path):
        # 实现OSS上传逻辑
        pass
```

### 图片处理

可以在上传前对图片进行处理：

```python
def process_image_before_upload(self, file_path):
    """压缩、格式转换等"""
    # 使用PIL或其他库处理图片
    pass
```

## 测试

### 手动测试

1. 配置TOS环境变量
2. 上传一个包含图片的PDF
3. 查看日志，确认图片是否上传成功
4. 检查生成的MD文件，确认URL是否正确替换

### 不配置TOS测试

1. 不配置TOS环境变量
2. 上传文件
3. 确认系统能正常工作，使用原始URL

## 常见问题

### Q: 为什么需要同时保存到 converted 和 all-md？

A:
- `FILE_DIR/converted/` - 保存原始转换结果，便于调试和备份
- `ALL_MD_DIR/` - 保存处理后的结果，供文档解析使用

### Q: 如果图片上传失败会怎样？

A: 系统会自动降级使用原始URL，不影响文档转换和处理。

### Q: 支持哪些图片格式？

A: 支持所有Markdown中引用的图片格式（jpg, png, gif, svg等）。

### Q: 可以自定义图片存储路径吗？

A: 目前固定为 `mineru/{filename}`，可以修改代码自定义路径。

### Q: 图片会被压缩吗？

A: 目前直接上传原图，未来可以增加压缩功能。

## 安全注意事项

1. **AccessKey保护** - 不要将密钥提交到版本控制
2. **Bucket权限** - 设置合理的访问控制策略
3. **文件大小限制** - 建议在上传前检查图片大小
4. **文件类型校验** - 验证文件确实是图片格式

## 未来改进

- [ ] 支持图片压缩
- [ ] 支持批量上传
- [ ] 支持更多对象存储服务
- [ ] 图片去重（相同MD5的图片只上传一次）
- [ ] 上传进度回调
- [ ] 异步并发上传
- [ ] CDN加速配置
