"""
数据库迁移脚本：添加对话系统表
"""

import sqlite3
import os
from pathlib import Path

def migrate_chat_tables():
    """添加对话系统所需的表"""

    # 数据库路径
    db_path = Path(__file__).parent / ".dbs/rag_preprocessor.db"

    print(f"📊 连接数据库: {db_path}")

    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 读取 SQL 文件
        schema_path = Path(__file__).parent / "chat_schema.sql"
        with open(schema_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        # 执行 SQL 脚本
        print("🔨 创建对话系统表...")
        cursor.executescript(sql_script)
        conn.commit()

        # 验证表是否创建成功
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND (name='chat_sessions' OR name='chat_messages')
        """)
        tables = cursor.fetchall()

        if len(tables) == 2:
            print("✅ 对话系统表创建成功:")
            for table in tables:
                print(f"   - {table[0]}")

                # 显示表结构
                cursor.execute(f"PRAGMA table_info({table[0]})")
                columns = cursor.fetchall()
                print(f"     字段: {', '.join([col[1] for col in columns])}")
        else:
            print(f"⚠️  只创建了 {len(tables)} 个表，预期为 2 个")

        print("\n✅ 数据库迁移完成！")

    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_chat_tables()
