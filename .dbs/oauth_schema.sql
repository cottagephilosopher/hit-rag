-- OAuth 相关表结构

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,                    -- 用户名（可选）
    email TEXT UNIQUE,                       -- 邮箱
    password_hash TEXT,                      -- 密码哈希（用于用户名密码登录）
    avatar_url TEXT,                         -- 头像 URL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP                  -- 最后登录时间
);

-- OAuth 绑定表（支持多平台）
CREATE TABLE IF NOT EXISTS oauth_bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,                -- 关联用户 ID
    provider TEXT NOT NULL,                  -- OAuth 提供商（github, qq, wechat 等）
    provider_user_id TEXT NOT NULL,          -- 第三方平台的用户 ID
    provider_username TEXT,                  -- 第三方平台的用户名
    access_token TEXT,                       -- OAuth access token（可选存储）
    refresh_token TEXT,                      -- OAuth refresh token（可选）
    token_expires_at TIMESTAMP,              -- Token 过期时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(provider, provider_user_id)       -- 同一平台同一用户只能绑定一次
);

-- OAuth 登录日志表
CREATE TABLE IF NOT EXISTS oauth_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,                         -- 用户 ID（如果登录成功）
    provider TEXT NOT NULL,                  -- OAuth 提供商
    action TEXT NOT NULL,                    -- 操作类型（login, bind, unbind）
    ip_address TEXT,                         -- 用户 IP 地址
    user_agent TEXT,                         -- 用户 User-Agent
    success INTEGER DEFAULT 1,               -- 是否成功（1: 成功，0: 失败）
    error_message TEXT,                      -- 错误信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_oauth_bindings_user_id ON oauth_bindings(user_id);
CREATE INDEX IF NOT EXISTS idx_oauth_bindings_provider ON oauth_bindings(provider, provider_user_id);
CREATE INDEX IF NOT EXISTS idx_oauth_logs_user_id ON oauth_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_oauth_logs_created_at ON oauth_logs(created_at);
