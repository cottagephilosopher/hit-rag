#!/usr/bin/env python3
"""
重置管理员密码脚本
使用方法: python reset_admin_password.py [新密码]
如果不提供密码参数，将从环境变量 ADMIN_PASSWORD 读取，默认为 1qaz
"""

import os
import sys
import sqlite3
from pathlib import Path
from auth_utils import hash_password
from datetime import datetime

def reset_admin_password(new_password: str = None):
    """重置管理员密码"""

    # 获取数据库路径
    db_file = Path(".dbs/rag_preprocessor.db")

    if not db_file.exists():
        print(f"❌ 数据库文件不存在: {db_file}")
        return False

    # 获取新密码
    if new_password is None:
        new_password = os.getenv("ADMIN_PASSWORD", "1qaz")

    admin_username = os.getenv("ADMIN_USERNAME", "admin")

    try:
        # 连接数据库
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # 检查管理员是否存在
        user = cursor.execute(
            "SELECT id, username FROM users WHERE username = ?",
            (admin_username,)
        ).fetchone()

        if not user:
            print(f"❌ 管理员账号不存在: {admin_username}")
            conn.close()
            return False

        user_id, username = user

        # 生成新密码哈希
        password_hash = hash_password(new_password)

        # 更新密码
        cursor.execute("""
            UPDATE users
            SET password_hash = ?, updated_at = ?
            WHERE username = ?
        """, (password_hash, datetime.utcnow().isoformat(), admin_username))

        conn.commit()
        conn.close()

        print(f"✅ 管理员密码重置成功")
        print(f"   用户名: {username}")
        print(f"   新密码: {new_password}")
        print(f"   ⚠️  请妥善保管密码！")

        return True

    except Exception as e:
        print(f"❌ 重置密码失败: {e}")
        return False

if __name__ == "__main__":
    # 从命令行参数获取新密码
    new_pwd = sys.argv[1] if len(sys.argv) > 1 else None

    if new_pwd:
        print(f"🔄 使用命令行参数设置新密码...")
    else:
        print(f"🔄 从环境变量读取密码配置...")

    reset_admin_password(new_pwd)
