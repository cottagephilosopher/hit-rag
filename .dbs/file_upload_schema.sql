-- 文件上传和转换记录表
-- 用于管理通过MinerU API上传和转换的文件

CREATE TABLE IF NOT EXISTS file_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- 原始文件信息
    original_filename TEXT NOT NULL,            -- 原始文件名
    file_size INTEGER,                          -- 文件大小（字节）
    file_type TEXT,                             -- 文件类型（pdf/docx/pptx等）
    upload_path TEXT NOT NULL,                  -- 上传文件存储路径

    -- MinerU 转换信息
    mineru_task_id TEXT,                        -- MinerU任务ID
    converted_md_filename TEXT,                 -- 转换后的MD文件名
    converted_md_path TEXT,                     -- 转换后的MD文件路径

    -- 状态管理
    status TEXT DEFAULT 'pending',              -- pending | uploading | converting | completed | error
    conversion_started_at TIMESTAMP,            -- 转换开始时间
    conversion_completed_at TIMESTAMP,          -- 转换完成时间
    error_message TEXT,                         -- 错误信息

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_file_uploads_status ON file_uploads(status);
CREATE INDEX IF NOT EXISTS idx_file_uploads_original_filename ON file_uploads(original_filename);
CREATE INDEX IF NOT EXISTS idx_file_uploads_created_at ON file_uploads(created_at DESC);

-- 触发器：自动更新 updated_at
CREATE TRIGGER IF NOT EXISTS update_file_uploads_timestamp
AFTER UPDATE ON file_uploads
FOR EACH ROW
BEGIN
    UPDATE file_uploads SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
