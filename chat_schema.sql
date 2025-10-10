-- 对话系统数据库表结构

-- 对话会话表
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    metadata TEXT,  -- JSON: 用户信息、配置等
    status TEXT DEFAULT 'active'  -- active | archived
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status);

-- 聊天消息表
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' | 'assistant' | 'system'
    content TEXT NOT NULL,
    intent TEXT,  -- 'question' | 'clarification' | 'chitchat' | NULL
    sources TEXT,  -- JSON: 引用的文档片段 [{"chunk_id": 123, "score": 0.95, ...}]
    metadata TEXT,  -- JSON: 额外信息（改写后的查询、实体等）
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp);
