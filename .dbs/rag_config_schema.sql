-- RAG 配置表
-- 存储 RAG Pipeline 的运行时配置参数

CREATE TABLE IF NOT EXISTS rag_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key TEXT NOT NULL UNIQUE,            -- 配置键名
    config_value REAL NOT NULL,                 -- 配置值（数字）
    description TEXT,                           -- 配置说明
    min_value REAL,                             -- 最小值
    max_value REAL,                             -- 最大值
    default_value REAL NOT NULL,                -- 默认值
    category TEXT NOT NULL,                     -- 配置分类：chat | threshold | retrieval
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_rag_config_category ON rag_config(category);
CREATE INDEX IF NOT EXISTS idx_rag_config_key ON rag_config(config_key);