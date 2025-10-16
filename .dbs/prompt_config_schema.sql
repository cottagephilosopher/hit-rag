-- 提示词配置表
-- 存储 LLM 提示词模板的配置

CREATE TABLE IF NOT EXISTS prompt_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_key TEXT NOT NULL UNIQUE,            -- 提示词键名
    prompt_value TEXT NOT NULL,                 -- 提示词内容（文本）
    description TEXT,                           -- 提示词说明
    category TEXT NOT NULL,                     -- 提示词分类：clean | tag | chunk
    is_system_prompt BOOLEAN DEFAULT 1,         -- 是否为system prompt（1=system, 0=user）
    default_value TEXT NOT NULL,                -- 默认提示词
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_prompt_config_category ON prompt_config(category);
CREATE INDEX IF NOT EXISTS idx_prompt_config_key ON prompt_config(prompt_key);
