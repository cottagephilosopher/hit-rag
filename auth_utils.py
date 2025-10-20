"""
认证工具模块
提供 JWT token 生成、验证和密码加密功能
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# JWT 配置
JWT_SECRET = os.getenv("UP_JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("UP_JWT_EXPIRE_MINUTES",180))  # 3 小时


# ==================== 密码相关 ====================

def hash_password(password: str) -> str:
    """
    加密密码（使用 bcrypt）

    Args:
        password: 明文密码

    Returns:
        加密后的密码（字符串格式）
    """
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码

    Args:
        plain_password: 明文密码
        hashed_password: 加密的密码

    Returns:
        密码是否匹配
    """
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


# ==================== JWT Token ====================

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    创建 JWT access token

    Args:
        data: 要编码的数据（通常包含 user_id, username 等）
        expires_delta: 过期时间（可选，默认使用配置的 JWT_EXPIRE_MINUTES）

    Returns:
        JWT token 字符串
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow()
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    解码并验证 JWT token

    Args:
        token: JWT token 字符串

    Returns:
        解码后的数据，如果验证失败返回 None
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        # Token 已过期
        return None
    except jwt.InvalidTokenError:
        # Token 无效
        return None


def create_token_response(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建包含 token 的登录响应

    Args:
        user: 用户信息字典

    Returns:
        包含 token 和用户信息的响应字典
    """
    access_token = create_access_token(
        data={
            "user_id": user["id"],
            "username": user.get("username"),
            "email": user.get("email")
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRE_MINUTES * 60,  # 转换为秒
        "user": {
            "id": user["id"],
            "username": user.get("username"),
            "email": user.get("email"),
            "avatar_url": user.get("avatar_url")
        }
    }


# ==================== 用户验证 ====================

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    验证用户名和密码

    Args:
        username: 用户名或邮箱
        password: 明文密码

    Returns:
        用户信息字典（验证成功）或 None（验证失败）
    """
    from database import get_connection

    with get_connection() as conn:
        # 查找用户（支持用户名或邮箱登录）
        user_row = conn.execute(
            """
            SELECT * FROM users
            WHERE username = ? OR email = ?
            """,
            (username, username)
        ).fetchone()

        if not user_row:
            return None

        # 转换为字典以便使用 .get() 方法
        user = dict(user_row)

        # 验证密码
        if not user.get("password_hash"):
            # 用户没有设置密码（可能是第三方登录用户）
            return None

        if not verify_password(password, user["password_hash"]):
            return None

        # 更新最后登录时间
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (datetime.utcnow(), user["id"])
        )

        return user


if __name__ == "__main__":
    # 测试密码加密
    password = "test123"
    hashed = hash_password(password)
    print(f"原密码: {password}")
    print(f"加密后: {hashed}")
    print(f"验证成功: {verify_password(password, hashed)}")
    print(f"验证失败: {verify_password('wrong', hashed)}")

    # 测试 JWT token
    user_data = {
        "id": 1,
        "username": "admin",
        "email": "admin@example.com"
    }

    token = create_access_token(user_data)
    print(f"\nJWT Token: {token}")

    decoded = decode_access_token(token)
    print(f"解码后: {decoded}")
