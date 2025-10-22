"""
全局超时保护工具
为关键操作提供超时保护，防止系统卡死
"""

import asyncio
import functools
import logging
from typing import Optional, Callable, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class _TimeoutContext:
    """超时上下文实现类（兼容Python 3.10）"""
    def __init__(self, timeout_seconds: int, operation_name: str):
        self.timeout_seconds = timeout_seconds
        self.operation_name = operation_name
    
    async def __aenter__(self):
        logger.debug(f"开始{self.operation_name}（超时: {self.timeout_seconds}秒）")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logger.debug(f"{self.operation_name}完成")
        return False


def timeout_context(timeout_seconds: int, operation_name: str = "操作"):
    """
    超时保护上下文管理器（兼容Python 3.10）
    注意：这个版本主要用于日志记录，实际超时保护需要配合asyncio.wait_for使用
    
    Args:
        timeout_seconds: 超时秒数
        operation_name: 操作名称（用于日志）
    
    Usage:
        async with timeout_context(30, "图片处理"):
            await asyncio.wait_for(some_operation(), timeout=30)
    """
    return _TimeoutContext(timeout_seconds, operation_name)


def timeout_decorator(timeout_seconds: int, operation_name: Optional[str] = None):
    """
    超时保护装饰器（用于异步函数，兼容Python 3.10）
    
    Args:
        timeout_seconds: 超时秒数
        operation_name: 操作名称（可选，默认使用函数名）
    
    Usage:
        @timeout_decorator(30, "处理文档")
        async def process_document():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            try:
                logger.debug(f"开始执行 {op_name}（超时: {timeout_seconds}秒）")
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
                logger.debug(f"{op_name} 执行完成")
                return result
            except asyncio.TimeoutError:
                logger.error(f"❌ {op_name} 超时（{timeout_seconds}秒）")
                raise TimeoutError(f"{op_name}超时")
        return wrapper
    return decorator


async def run_with_timeout(
    coro,
    timeout_seconds: int,
    operation_name: str = "操作",
    default_on_timeout: Any = None
):
    """
    运行协程并设置超时，超时后返回默认值（兼容Python 3.10）
    
    Args:
        coro: 要运行的协程
        timeout_seconds: 超时秒数
        operation_name: 操作名称
        default_on_timeout: 超时时返回的默认值
    
    Returns:
        协程结果或默认值
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"⚠️ {operation_name}超时（{timeout_seconds}秒），返回默认值")
        return default_on_timeout


class TimeoutMonitor:
    """超时监控器，用于批量操作的超时管理"""
    
    def __init__(self, total_timeout: int, operation_name: str = "批量操作"):
        """
        Args:
            total_timeout: 总超时时间（秒）
            operation_name: 操作名称
        """
        self.total_timeout = total_timeout
        self.operation_name = operation_name
        self.start_time = None
        self.warnings = []
    
    def start(self):
        """开始监控"""
        import time
        self.start_time = time.time()
        logger.info(f"🕐 {self.operation_name}开始，总超时: {self.total_timeout}秒")
    
    def check_timeout(self) -> bool:
        """
        检查是否超时
        
        Returns:
            True表示已超时，False表示未超时
        """
        if self.start_time is None:
            return False
        
        import time
        elapsed = time.time() - self.start_time
        if elapsed > self.total_timeout:
            logger.error(f"❌ {self.operation_name}总超时（{elapsed:.1f}秒 > {self.total_timeout}秒）")
            return True
        
        # 警告：超过80%时间
        if elapsed > self.total_timeout * 0.8 and len(self.warnings) == 0:
            logger.warning(f"⚠️ {self.operation_name}已用时 {elapsed:.1f}秒，接近超时")
            self.warnings.append("80%")
        
        return False
    
    def get_remaining_time(self) -> float:
        """
        获取剩余时间（秒）
        
        Returns:
            剩余秒数，如果已超时返回0
        """
        if self.start_time is None:
            return self.total_timeout
        
        import time
        elapsed = time.time() - self.start_time
        remaining = max(0, self.total_timeout - elapsed)
        return remaining
    
    def finish(self):
        """完成监控"""
        if self.start_time is None:
            return
        
        import time
        elapsed = time.time() - self.start_time
        logger.info(f"✅ {self.operation_name}完成，用时: {elapsed:.1f}秒")

