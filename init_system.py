#!/usr/bin/env python3
"""
RAG 系统统一初始化脚本
清空并初始化所有数据库和向量库
"""

import os
import sys
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_step(step: str, status: str = ""):
    """打印步骤信息"""
    if status:
        print(f"  {step}... {status}")
    else:
        print(f"\n📋 {step}")


def init_sqlite_database(force: bool = False):
    """初始化 SQLite 数据库"""
    print_step("初始化 SQLite 数据库")

    # 数据库路径（从 .env 读取）
    db_file = Path(os.getenv("DB_FILE", ".dbs/rag_preprocessor.db"))
    db_dir = db_file.parent

    # 创建目录
    db_dir.mkdir(parents=True, exist_ok=True)

    # 检查数据库是否存在
    if db_file.exists():
        if not force:
            print(f"  ⚠️  数据库已存在: {db_file}")
            confirm = input("  是否删除并重新创建？(输入 'yes' 确认): ")
            if confirm.lower() != 'yes':
                print("  ⏭️  跳过数据库初始化")
                return False

        # 删除旧数据库
        db_file.unlink()
        print(f"  🗑️  已删除旧数据库")

    # 读取 schema 文件
    schema_file = db_dir / "schema.sql"
    chat_schema_file = db_dir / "chat_schema.sql"
    rag_config_schema_file = db_dir / "rag_config_schema.sql"

    if not schema_file.exists():
        print(f"  ❌ Schema 文件不存在: {schema_file}")
        return False

    # 执行 SQL 脚本
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # 执行文档 schema
        print_step("创建文档相关表", "")
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        cursor.executescript(schema_sql)

        # 执行对话 schema（如果存在）
        if chat_schema_file.exists():
            print_step("创建对话相关表", "")
            with open(chat_schema_file, 'r', encoding='utf-8') as f:
                chat_schema_sql = f.read()
            cursor.executescript(chat_schema_sql)

        # 执行 RAG 配置 schema（如果存在）
        if rag_config_schema_file.exists():
            print_step("创建 RAG 配置表", "")
            with open(rag_config_schema_file, 'r', encoding='utf-8') as f:
                rag_config_schema_sql = f.read()
            cursor.executescript(rag_config_schema_sql)

        conn.commit()
        conn.close()

        # 初始化 RAG 配置数据
        print_step("初始化 RAG 配置", "")
        from database import init_rag_config_from_env
        init_rag_config_from_env()

        # 重新连接以验证
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # 验证表是否创建成功
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        print(f"\n  ✅ 数据库初始化成功！")
        print(f"  📊 已创建 {len(tables)} 个表:")
        for table in tables:
            print(f"     - {table}")

        conn.close()
        return True

    except Exception as e:
        print(f"  ❌ 数据库初始化失败: {e}")
        return False


def init_milvus_collection(force: bool = False):
    """初始化 Milvus 集合"""
    print_step("初始化 Milvus 向量库")

    try:
        from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

        # 读取配置
        milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "autoagent")
        embedding_dim = int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "2048"))

        print(f"  📡 连接到 Milvus: {milvus_host}:{milvus_port}")
        print(f"  📦 集合名称: {collection_name}")
        print(f"  📐 向量维度: {embedding_dim}")

        # 连接 Milvus
        connections.connect(
            alias="default",
            host=milvus_host,
            port=milvus_port
        )
        print_step("连接 Milvus", "✅")

        # 检查集合是否存在
        if utility.has_collection(collection_name):
            if not force:
                print(f"  ⚠️  集合已存在: {collection_name}")
                confirm = input("  是否删除并重新创建？(输入 'yes' 确认): ")
                if confirm.lower() != 'yes':
                    print("  ⏭️  跳过 Milvus 初始化")
                    connections.disconnect("default")
                    return False

            # 删除旧集合
            utility.drop_collection(collection_name)
            print_step("删除旧集合", "✅")

        # 定义字段 schema
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_db_id", dtype=DataType.INT64),  # 数据库主键 ID
            FieldSchema(name="document_id", dtype=DataType.INT64, default_value=0),  # 文档 ID
            FieldSchema(name="chunk_sequence", dtype=DataType.INT64, default_value=0),  # 文档内顺序编号
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500, default_value="unknown"),
            # 标签字段（重要：用于标签筛选）
            FieldSchema(name="user_tag", dtype=DataType.VARCHAR, max_length=200, default_value="none"),
            FieldSchema(name="content_tags", dtype=DataType.VARCHAR, max_length=2000, default_value="[]"),  # JSON 数组
            # ATOMIC 相关字段
            FieldSchema(name="is_atomic", dtype=DataType.BOOL, default_value=False),
            FieldSchema(name="atomic_type", dtype=DataType.VARCHAR, max_length=50, default_value="none"),
            # 其他元数据
            FieldSchema(name="token_count", dtype=DataType.INT64, default_value=0),
            FieldSchema(name="vectorized_at", dtype=DataType.INT64, default_value=0),
            FieldSchema(name="original_content", dtype=DataType.VARCHAR, max_length=65535, default_value=""),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64, default_value=""),
        ]

        # 创建集合 schema
        schema = CollectionSchema(
            fields=fields,
            description="RAG Document Chunks",
            enable_dynamic_field=True  # 允许动态字段
        )

        # 创建集合
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default'
        )
        print_step("创建集合", "✅")

        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }

        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        print_step("创建索引", "✅")

        # 加载集合到内存
        collection.load()
        print_step("加载集合", "✅")

        print(f"\n  ✅ Milvus 集合初始化成功！")
        print(f"  📊 集合信息:")
        print(f"     - 名称: {collection_name}")
        print(f"     - 向量维度: {embedding_dim}")
        print(f"     - 索引类型: HNSW")
        print(f"     - 距离度量: L2")

        connections.disconnect("default")
        return True

    except ImportError:
        print("  ⚠️  pymilvus 未安装，跳过 Milvus 初始化")
        print("  提示: pip install pymilvus")
        return False
    except Exception as e:
        print(f"  ❌ Milvus 初始化失败: {e}")
        try:
            connections.disconnect("default")
        except:
            pass
        return False


def verify_environment():
    """验证环境配置"""
    print_step("验证环境配置")

    issues = []

    # 检查必需的环境变量
    required_vars = [
        "API_BASE_URL",
        "FRONTEND_UI_URL",
        "MILVUS_HOST",
        "MILVUS_PORT",
        "MILVUS_COLLECTION_NAME",
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_EMBEDDING_MODEL",
        "DASHSCOPE_EMBEDDING_DIMENSIONS"
    ]

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            issues.append(f"环境变量 {var} 未配置")
        else:
            print(f"  ✅ {var}: {value[:50]}...")

    if issues:
        print("\n  ⚠️  发现配置问题:")
        for issue in issues:
            print(f"     - {issue}")
        return False

    print(f"\n  ✅ 环境配置验证通过！")
    return True


def main():
    """主函数"""
    print_section("RAG 系统初始化")

    print("""
    此脚本将：
    1. 清空并初始化 SQLite 数据库
    2. 清空并初始化 Milvus 向量库
    3. 验证所有配置

    ⚠️  警告: 这将删除所有现有数据！
    """)

    # 检查是否强制模式
    force = '--force' in sys.argv or '-f' in sys.argv

    if not force:
        confirm = input("是否继续？(输入 'yes' 确认): ")
        if confirm.lower() != 'yes':
            print("\n❌ 操作已取消")
            return 1

    # 验证环境
    print_section("步骤 1: 验证环境配置")
    if not verify_environment():
        print("\n❌ 环境配置验证失败，请检查 .env 文件")
        return 1

    # 初始化 SQLite
    print_section("步骤 2: 初始化 SQLite 数据库")
    if not init_sqlite_database(force):
        print("\n❌ SQLite 初始化失败")
        return 1

    # 初始化 Milvus
    print_section("步骤 3: 初始化 Milvus 向量库")
    milvus_success = init_milvus_collection(force)
    if not milvus_success:
        print("\n⚠️  Milvus 初始化未完成（可能未安装或连接失败）")
        print("提示: 如果需要使用向量检索，请确保 Milvus 正在运行")

    # 完成
    print_section("初始化完成")
    print("""
    ✅ 系统初始化成功！

    下一步：
    1. 将文档放入 all-md 目录
    2. 启动 API 服务: python api_server.py
    3. 通过 API 或前端处理文档

    常用命令：
    - 查看数据库: sqlite3 .dbs/rag_preprocessor.db
    - 启动服务: python api_server.py
    - 查看帮助: python api_server.py --help
    """)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
