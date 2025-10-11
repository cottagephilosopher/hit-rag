"""
RAG 文档预处理 API 服务 - 主入口
提供文档列表、处理状态查询和处理触发接口
同时提供 chunk 更新和版本历史接口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入数据库操作模块
from database import init_database

# 导入路由模块
from chat_routes import router as chat_router
from document_routes import router as document_router
from agent_routes import router as agent_router

# 创建 FastAPI 应用
app = FastAPI(title="RAG Preprocessor API", version="2.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router, tags=["Chat"])
app.include_router(document_router, tags=["Documents & Chunks"])
app.include_router(agent_router, tags=["Agent"])


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "RAG Preprocessor API",
        "version": "2.0.0",
        "description": "文档预处理和智能对话 API",
        "endpoints": {
            # 文档管理
            "documents": {
                "list": "GET /api/documents",
                "status": "GET /api/documents/{filename}/status",
                "chunks": "GET /api/documents/{filename}/chunks",
                "process": "POST /api/documents/{filename}/process",
                "delete_output": "DELETE /api/documents/{filename}/output"
            },
            # 文档标签
            "document_tags": {
                "get": "GET /api/documents/{filename}/tags",
                "add": "POST /api/documents/{filename}/tags",
                "remove": "DELETE /api/documents/{filename}/tags/{tag_text}"
            },
            # Chunk 管理
            "chunks": {
                "update": "PATCH /api/chunks/{chunk_id}",
                "logs": "GET /api/chunks/{chunk_id}/logs",
                "tags": "GET /api/chunks/tags",
                "vectorizable": "GET /api/chunks/vectorizable",
                "search": "POST /api/chunks/search"
            },
            # 向量化
            "vectorization": {
                "batch": "POST /api/chunks/vectorize/batch",
                "single": "POST /api/chunks/{chunk_id}/vectorize",
                "delete": "DELETE /api/chunks/{chunk_id}/vectorize",
                "stats": "GET /api/vectorization/stats"
            },
            # 全局标签管理
            "tags": {
                "all": "GET /api/tags/all",
                "create": "POST /api/tags/create",
                "delete": "POST /api/tags/delete",
                "rename": "POST /api/tags/rename",
                "merge": "POST /api/tags/merge"
            },
            # 对话
            "chat": {
                "message": "POST /api/chat/message",
                "assistants": "GET /api/assistants",
                "tools": "GET /api/agent/tools",
                "react": "POST /api/agent/react"
            }
        },
        "modules": {
            "document_routes": "文档和切片管理",
            "agent_routes": "Agent 对话",
            "chat_routes": "普通对话"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    # 初始化数据库
    try:
        init_database()
        print("✅ 数据库初始化成功")
    except Exception as e:
        print(f"⚠️  Warning: Database initialization failed: {e}")

    print("🚀 启动 RAG API 服务...")
    print("📚 文档管理路由: /api/documents/*")
    print("💬 对话路由: /api/chat/*")
    print("🤖 Agent 路由: /api/agent/*")
    print("📖 API 文档: http://localhost:8000/docs")

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
