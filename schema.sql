-- RAG 文档预处理系统数据库 Schema
-- SQLite 数据库设计

-- ============================================
-- 1. 文档表 (documents)
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,              -- 文档文件名（唯一）
    source_path TEXT NOT NULL,                  -- 源文件路径
    status TEXT DEFAULT 'pending',              -- pending | processing | completed | error
    total_chunks INTEGER DEFAULT 0,             -- 总切片数
    total_tokens INTEGER DEFAULT 0,             -- 总token数
    processed_at TIMESTAMP,                     -- 处理完成时间
    error_message TEXT,                         -- 错误信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 2. 文档切片表 (document_chunks)
-- ============================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,               -- 所属文档ID
    chunk_id INTEGER NOT NULL,                  -- chunk编号（从1开始）

    -- 内容字段
    content TEXT NOT NULL,                      -- 切片内容（原始）
    edited_content TEXT,                        -- 编辑后的内容

    -- Token 位置信息
    token_start INTEGER NOT NULL,               -- 起始token位置
    token_end INTEGER NOT NULL,                 -- 结束token位置
    token_count INTEGER NOT NULL,               -- token数量
    char_start INTEGER,                         -- 起始字符位置
    char_end INTEGER,                           -- 结束字符位置

    -- 标签与元数据
    user_tag TEXT,                              -- 用户标签
    content_tags TEXT,                          -- 内容标签（JSON数组）
    is_atomic BOOLEAN DEFAULT FALSE,            -- 是否为原子块
    atomic_type TEXT,                           -- 原子类型（table/code等）

    -- 状态管理
    status INTEGER DEFAULT 0,                   -- -1=废弃 0=初始 1=已确认 2=已向量化
    version INTEGER DEFAULT 1,                  -- 📌 版本号（每次修改+1）
    last_editor_id TEXT,                        -- 📌 最后编辑者ID

    -- 向量化相关
    milvus_id TEXT,                             -- Milvus 向量ID（向量化后填充）

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 📌 最后更新时间

    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, chunk_id)               -- 同一文档内chunk_id唯一
);

-- ============================================
-- 3. 文档操作日志表 (document_logs)
-- ============================================
CREATE TABLE IF NOT EXISTS document_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,               -- 所属文档ID
    chunk_id INTEGER,                           -- 所属chunk ID（可为空）

    -- 操作信息
    action TEXT NOT NULL,                       -- 操作类型：create | update | status_change | delete
    message TEXT,                               -- 操作描述

    -- 用户信息
    user_id TEXT,                               -- 📌 操作者ID

    -- 详细变更数据
    payload TEXT,                               -- 📌 JSON格式的详细变更信息

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY(chunk_id) REFERENCES document_chunks(id) ON DELETE SET NULL
);

-- ============================================
-- 4. 文档标签表 (document_tags)
-- ============================================
CREATE TABLE IF NOT EXISTS document_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,               -- 所属文档ID
    tag_text TEXT NOT NULL,                     -- 标签文本
    tag_order INTEGER DEFAULT 0,                -- 标签显示顺序
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, tag_text)               -- 同一文档内标签不重复
);

-- ============================================
-- 5. 索引优化
-- ============================================

-- 文档表索引
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- 切片表索引
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON document_chunks(status);
CREATE INDEX IF NOT EXISTS idx_chunks_version ON document_chunks(version);
CREATE INDEX IF NOT EXISTS idx_chunks_updated ON document_chunks(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_milvus_id ON document_chunks(milvus_id);
CREATE INDEX IF NOT EXISTS idx_chunks_vectorized ON document_chunks(status) WHERE status = 2;

-- 日志表索引
CREATE INDEX IF NOT EXISTS idx_logs_document ON document_logs(document_id);
CREATE INDEX IF NOT EXISTS idx_logs_chunk ON document_logs(chunk_id);
CREATE INDEX IF NOT EXISTS idx_logs_action ON document_logs(action);
CREATE INDEX IF NOT EXISTS idx_logs_created ON document_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_chunk_time ON document_logs(chunk_id, created_at DESC);

-- 标签表索引
CREATE INDEX IF NOT EXISTS idx_tags_document ON document_tags(document_id);
CREATE INDEX IF NOT EXISTS idx_tags_order ON document_tags(document_id, tag_order);

-- ============================================
-- 5. 触发器：自动更新 updated_at
-- ============================================

-- 文档表更新触发器
CREATE TRIGGER IF NOT EXISTS update_documents_timestamp
AFTER UPDATE ON documents
FOR EACH ROW
BEGIN
    UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- 切片表更新触发器
CREATE TRIGGER IF NOT EXISTS update_chunks_timestamp
AFTER UPDATE ON document_chunks
FOR EACH ROW
BEGIN
    UPDATE document_chunks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================
-- 6. 示例数据（用于测试）
-- ============================================

-- 插入示例文档
INSERT OR IGNORE INTO documents (id, filename, source_path, status, total_chunks, total_tokens, processed_at)
VALUES (1, 'example.md', '/all-md/example.md', 'completed', 3, 1234, CURRENT_TIMESTAMP);

-- 插入示例切片
INSERT OR IGNORE INTO document_chunks (
    id, document_id, chunk_id, content, token_start, token_end, token_count,
    user_tag, content_tags, status, version
) VALUES
(1, 1, 1, '# 产品说明\n\n这是产品的介绍内容...', 0, 100, 100, '产品说明', '["产品", "介绍"]', 0, 1),
(2, 1, 2, '## 技术参数\n\n分辨率: 1920x1080', 100, 200, 100, '技术参数', '["参数", "规格"]', 0, 1),
(3, 1, 3, '## 使用说明\n\n详细的使用步骤...', 200, 300, 100, '使用说明', '["说明", "教程"]', 0, 1);

-- 插入示例日志
INSERT OR IGNORE INTO document_logs (
    document_id, chunk_id, action, message, user_id, payload
) VALUES
(1, 1, 'create', '创建了chunk', 'system', '{"initial_version": 1}'),
(1, 2, 'create', '创建了chunk', 'system', '{"initial_version": 1}'),
(1, 3, 'create', '创建了chunk', 'system', '{"initial_version": 1}');
