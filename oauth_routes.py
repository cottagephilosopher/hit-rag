"""
统一认证路由
支持用户名密码登录、GitHub OAuth、QQ OAuth 等多种登录方式
"""

import os
import secrets
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx

from database import get_connection
from auth_utils import (
    create_token_response,
    authenticate_user,
    decode_access_token,
    hash_password
)

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# OAuth 配置
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
OAUTH_CALLBACK_BASE_URL = os.getenv("OAUTH_CALLBACK_BASE_URL", f"http://localhost:{os.getenv('API_PORT', '8086')}")
FRONTEND_URL = os.getenv("FRONTEND_UI_URL", "http://localhost:3001")


# 安全配置
security = HTTPBearer()


# ==================== Pydantic Models ====================

class LoginRequest(BaseModel):
    """用户名密码登录请求"""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """用户注册请求"""
    username: str
    email: str
    password: str
    avatar_url: Optional[str] = None


class OAuthCallbackRequest(BaseModel):
    """OAuth 回调请求"""
    code: str
    state: Optional[str] = None


class UserResponse(BaseModel):
    """用户信息响应"""
    id: int
    username: Optional[str]
    email: Optional[str]
    avatar_url: Optional[str]
    created_at: str


class TokenResponse(BaseModel):
    """Token 响应"""
    access_token: str
    token_type: str
    expires_in: int
    user: dict


# ==================== 数据库操作 ====================

def get_or_create_user(
    provider: str,
    provider_user_id: str,
    provider_username: str,
    email: Optional[str] = None,
    avatar_url: Optional[str] = None
) -> dict:
    """
    获取或创建用户

    Args:
        provider: OAuth 提供商（github, qq 等）
        provider_user_id: 第三方平台用户 ID
        provider_username: 第三方平台用户名
        email: 邮箱
        avatar_url: 头像 URL

    Returns:
        用户信息字典
    """
    with get_connection() as conn:
        # 查找是否已存在 OAuth 绑定
        oauth_binding = conn.execute(
            """
            SELECT user_id FROM oauth_bindings
            WHERE provider = ? AND provider_user_id = ?
            """,
            (provider, provider_user_id)
        ).fetchone()

        if oauth_binding:
            # 已存在，获取用户信息
            user_id = oauth_binding['user_id']
            user = conn.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()

            if user:
                # 更新最后登录时间
                conn.execute(
                    "UPDATE users SET last_login_at = ? WHERE id = ?",
                    (datetime.utcnow(), user_id)
                )
                return dict(user)

        # 创建新用户
        cursor = conn.execute(
            """
            INSERT INTO users (username, email, avatar_url, last_login_at)
            VALUES (?, ?, ?, ?)
            """,
            (provider_username, email, avatar_url, datetime.utcnow())
        )
        user_id = cursor.lastrowid

        # 创建 OAuth 绑定
        conn.execute(
            """
            INSERT INTO oauth_bindings (
                user_id, provider, provider_user_id, provider_username
            ) VALUES (?, ?, ?, ?)
            """,
            (user_id, provider, provider_user_id, provider_username)
        )

        # 返回新创建的用户
        user = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

        return dict(user)


def log_oauth_action(
    provider: str,
    action: str,
    user_id: Optional[int] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None
):
    """
    记录 OAuth 操作日志

    Args:
        provider: OAuth 提供商
        action: 操作类型（login, bind, unbind）
        user_id: 用户 ID
        ip_address: 用户 IP
        user_agent: User-Agent
        success: 是否成功
        error_message: 错误信息
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO oauth_logs (
                user_id, provider, action, ip_address,
                user_agent, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                provider,
                action,
                ip_address,
                user_agent,
                1 if success else 0,
                error_message
            )
        )


# ==================== Token 验证中间件 ====================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    验证 JWT token 并返回当前用户信息

    Args:
        credentials: HTTP Authorization Bearer token

    Returns:
        用户信息字典

    Raises:
        HTTPException: Token 无效或已过期
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token 格式错误")

    # 从数据库获取用户信息
    with get_connection() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

        if not user:
            raise HTTPException(status_code=401, detail="用户不存在")

        return dict(user)


# ==================== 用户名密码登录 ====================

@router.post("/api/oauth/login", response_model=TokenResponse)
async def login(request: LoginRequest, req: Request):
    """
    用户名密码登录

    Args:
        request: 登录请求（用户名/邮箱 + 密码）
        req: HTTP 请求对象

    Returns:
        JWT token 和用户信息
    """
    try:
        # 获取客户端 IP 和 User-Agent
        ip_address = req.client.host if req.client else None
        user_agent = req.headers.get("user-agent", "")

        logger.info(f"用户登录尝试: username={request.username}, ip={ip_address}")

        # 验证用户名和密码
        user = authenticate_user(request.username, request.password)

        if not user:
            logger.warning(f"登录失败: 用户名或密码错误 - {request.username}")
            log_oauth_action("password", "login", None, ip_address, user_agent, False, "用户名或密码错误")
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        # 记录登录日志
        log_oauth_action("password", "login", user["id"], ip_address, user_agent, True)

        logger.info(f"✅ 用户登录成功: user_id={user['id']}, username={user['username']}")

        # 生成 JWT token
        return create_token_response(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"登录失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")


@router.post("/api/oauth/register", response_model=TokenResponse)
async def register(request: RegisterRequest, req: Request):
    """
    用户注册

    Args:
        request: 注册请求
        req: HTTP 请求对象

    Returns:
        JWT token 和用户信息
    """
    try:
        # 获取客户端 IP 和 User-Agent
        ip_address = req.client.host if req.client else None
        user_agent = req.headers.get("user-agent", "")

        logger.info(f"用户注册尝试: username={request.username}, email={request.email}")

        with get_connection() as conn:
            # 检查用户名是否已存在
            existing_user = conn.execute(
                "SELECT id FROM users WHERE username = ? OR email = ?",
                (request.username, request.email)
            ).fetchone()

            if existing_user:
                raise HTTPException(status_code=400, detail="用户名或邮箱已存在")

            # 加密密码
            password_hash = hash_password(request.password)

            # 创建用户
            cursor = conn.execute(
                """
                INSERT INTO users (username, email, password_hash, avatar_url, last_login_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (request.username, request.email, password_hash, request.avatar_url, datetime.utcnow())
            )
            user_id = cursor.lastrowid

            # 获取新创建的用户
            user = conn.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()

            user = dict(user)

        # 记录注册日志
        log_oauth_action("password", "register", user["id"], ip_address, user_agent, True)

        logger.info(f"✅ 用户注册成功: user_id={user['id']}, username={user['username']}")

        # 生成 JWT token
        return create_token_response(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")


# ==================== GitHub OAuth ====================

@router.get("/api/oauth/github/authorize")
async def github_authorize():
    """
    GitHub OAuth 授权
    生成授权 URL 并重定向
    """
    if not GITHUB_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GitHub OAuth 未配置")

    # 生成 state 用于防止 CSRF 攻击
    state = secrets.token_urlsafe(32)

    # 使用后端 API 地址作为回调地址
    callback_uri = f"{OAUTH_CALLBACK_BASE_URL}/api/oauth/github"
    authorize_url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri={callback_uri}"
        f"&scope=user:email"
        f"&state={state}"
    )

    logger.info(f"GitHub OAuth 授权: state={state}, callback={callback_uri}")

    return {
        "authorize_url": authorize_url,
        "state": state
    }


@router.get("/api/oauth/github")
async def github_callback(code: str, state: Optional[str] = None, request: Request = None):
    """
    GitHub OAuth 回调处理
    GitHub 授权后会跳转到这里，处理完成后重定向到前端

    Args:
        code: GitHub 返回的授权码
        state: CSRF 防护状态码
    """
    try:
        # 获取客户端 IP 和 User-Agent
        ip_address = request.client.host if request else None
        user_agent = request.headers.get("user-agent") if request else None

        logger.info(f"GitHub OAuth 回调: code={code[:10]}..., state={state}")

        if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
            raise HTTPException(status_code=500, detail="GitHub OAuth 未配置")

        # 1. 用 code 换取 access_token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": code
                },
                timeout=60.0  # 增加超时时间到 60 秒
            )

        if token_response.status_code != 200:
            error_msg = f"获取 access_token 失败: {token_response.text}"
            logger.error(error_msg)
            log_oauth_action("github", "login", None, ip_address, user_agent, False, error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        token_data = token_response.json()
        access_token = token_data.get("access_token")

        if not access_token:
            error_msg = f"未获取到 access_token: {token_data}"
            logger.error(error_msg)
            log_oauth_action("github", "login", None, ip_address, user_agent, False, error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        logger.info(f"✅ 获取 GitHub access_token 成功")

        # 2. 用 access_token 获取用户信息
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json"
                },
                timeout=60.0  # 增加超时时间到 60 秒
            )

        if user_response.status_code != 200:
            error_msg = f"获取用户信息失败: {user_response.text}"
            logger.error(error_msg)
            log_oauth_action("github", "login", None, ip_address, user_agent, False, error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        github_user = user_response.json()

        logger.info(f"✅ 获取 GitHub 用户信息成功: {github_user.get('login')}")

        # 3. 创建或获取本地用户
        user = get_or_create_user(
            provider="github",
            provider_user_id=str(github_user["id"]),
            provider_username=github_user.get("login"),
            email=github_user.get("email"),
            avatar_url=github_user.get("avatar_url")
        )

        logger.info(f"✅ 用户登录成功: user_id={user['id']}, username={user['username']}")

        # 4. 记录登录日志
        log_oauth_action(
            provider="github",
            action="login",
            user_id=user["id"],
            ip_address=ip_address,
            user_agent=user_agent,
            success=True
        )

        # 5. 生成 JWT token
        token_data = create_token_response(user)

        # 6. 重定向到前端，并携带 token 和用户信息
        import urllib.parse
        import json

        # 将用户信息编码为 JSON 字符串
        user_json = json.dumps(token_data['user'])

        redirect_url = (
            f"{FRONTEND_URL}/oauth/callback"
            f"?success=true"
            f"&token={token_data['access_token']}"
            f"&user={urllib.parse.quote(user_json)}"
            f"&provider=github"
        )

        logger.info(f"✅ GitHub 登录成功，重定向到: {redirect_url}")

        return RedirectResponse(url=redirect_url)

    except HTTPException as http_exc:
        # HTTP 异常，重定向到前端并携带错误信息
        error_msg = str(http_exc.detail)
        redirect_url = f"{FRONTEND_URL}/oauth/callback?success=false&error={urllib.parse.quote(error_msg)}&provider=github"
        logger.error(f"❌ GitHub OAuth 失败: {error_msg}")
        return RedirectResponse(url=redirect_url)
    except Exception as e:
        error_msg = f"GitHub OAuth 登录失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        log_oauth_action("github", "login", None, ip_address, user_agent, False, error_msg)

        # 重定向到前端并携带错误信息
        import urllib.parse
        redirect_url = f"{FRONTEND_URL}/oauth/callback?success=false&error={urllib.parse.quote(error_msg)}&provider=github"
        return RedirectResponse(url=redirect_url)


# ==================== 用户信息 ====================

@router.get("/api/oauth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    获取当前登录用户的信息（需要 JWT token）

    Args:
        current_user: 当前用户（通过 JWT token 验证）

    Returns:
        用户信息
    """
    return {
        "id": current_user["id"],
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "avatar_url": current_user.get("avatar_url"),
        "created_at": current_user.get("created_at"),
        "last_login_at": current_user.get("last_login_at")
    }


@router.get("/api/oauth/user/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """
    获取用户信息

    Args:
        user_id: 用户 ID
    """
    with get_connection() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        return UserResponse(**dict(user))


@router.get("/api/oauth/user/{user_id}/bindings")
async def get_user_bindings(user_id: int):
    """
    获取用户的 OAuth 绑定列表

    Args:
        user_id: 用户 ID
    """
    with get_connection() as conn:
        bindings = conn.execute(
            """
            SELECT provider, provider_username, created_at
            FROM oauth_bindings
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,)
        ).fetchall()

        return {
            "user_id": user_id,
            "bindings": [dict(b) for b in bindings]
        }


# ==================== QQ OAuth（预留接口）====================

@router.get("/api/oauth/qq/authorize")
async def qq_authorize():
    """QQ OAuth 授权（待实现）"""
    raise HTTPException(status_code=501, detail="QQ OAuth 暂未实现")


@router.get("/api/oauth/qq")
async def qq_callback():
    """QQ OAuth 回调（待实现）"""
    raise HTTPException(status_code=501, detail="QQ OAuth 暂未实现")
