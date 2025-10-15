"""
LLM API 客户端：统一封装 LLM API 调用
支持 Azure OpenAI 和 OpenAI，提供重试、超时等功能
"""

import logging
import time
import json
from typing import Optional, Dict, Any, List
from openai import AzureOpenAI, OpenAI

# 使用绝对导入
try:
    from config import LLMConfig
except ImportError:
    from ..config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM API 统一客户端
    封装 Azure OpenAI 和 OpenAI API 调用
    """

    def __init__(self, provider: Optional[str] = None):
        """
        初始化 LLM 客户端

        Args:
            provider: 'azure'、'openai' 或 'dashscope'，默认使用配置中的值
        """
        self.provider = provider or LLMConfig.PROVIDER
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化 LLM API 客户端"""
        try:
            if self.provider == "azure":
                self.client = AzureOpenAI(
                    api_key=LLMConfig.AZURE_OPENAI_API_KEY,
                    api_version=LLMConfig.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=LLMConfig.AZURE_OPENAI_ENDPOINT
                )
                self.model_name = LLMConfig.DEPLOYMENT_NAME
                logger.info(
                    f"✅ Azure OpenAI 客户端初始化成功: {self.model_name}"
                )

            elif self.provider == "openai":
                self.client = OpenAI(
                    api_key=LLMConfig.OPENAI_API_KEY
                )
                self.model_name = LLMConfig.OPENAI_MODEL
                logger.info(
                    f"✅ OpenAI 客户端初始化成功: {self.model_name}"
                )

            elif self.provider == "dashscope":
                self.client = OpenAI(
                    api_key=LLMConfig.DASHSCOPE_API_KEY,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                self.model_name = LLMConfig.DASHSCOPE_MODEL
                logger.info(
                    f"✅ DashScope 客户端初始化成功: {self.model_name}"
                )

            else:
                raise ValueError(f"不支持的 provider: {self.provider}")

        except Exception as e:
            logger.error(f"❌ LLM 客户端初始化失败: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        调用 Chat Completion API

        Args:
            messages: 消息列表，格式: [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 Token 数
            response_format: 响应格式，例如 {"type": "json_object"}

        Returns:
            LLM 响应文本
        """
        temperature = temperature or LLMConfig.TEMPERATURE
        max_tokens = max_tokens or LLMConfig.MAX_TOKENS

        for attempt in range(LLMConfig.MAX_RETRIES):
            try:
                logger.debug(
                    f"🔄 LLM API 调用 (尝试 {attempt + 1}/{LLMConfig.MAX_RETRIES})"
                )

                # 构建请求参数
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                # 添加响应格式（如果指定）
                if response_format:
                    kwargs["response_format"] = response_format

                # 调用 API
                start_time = time.time()
                response = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - start_time

                # 提取响应内容
                content = response.choices[0].message.content
                usage = response.usage

                logger.info(
                    f"✅ LLM API 调用成功 "
                    f"(耗时: {elapsed:.2f}s, "
                    f"输入: {usage.prompt_tokens}, "
                    f"输出: {usage.completion_tokens})"
                )

                return content

            except Exception as e:
                logger.warning(
                    f"⚠️ LLM API 调用失败 (尝试 {attempt + 1}): {e}"
                )

                if attempt < LLMConfig.MAX_RETRIES - 1:
                    wait_time = LLMConfig.RETRY_DELAY * (2 ** attempt)
                    logger.info(f"⏳ 等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ LLM API 调用失败，已达最大重试次数")
                    raise

    def chat_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """
        使用系统提示词和用户提示词调用 API

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 其他参数传递给 chat 方法

        Returns:
            LLM 响应文本
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.chat(messages, **kwargs)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用 API 并要求返回 JSON 格式

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            解析后的 JSON 对象
        """
        # 使用 JSON 模式
        response = self.chat(
            messages,
            response_format={"type": "json_object"},
            **kwargs
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 解析失败: {e}")
            logger.error(f"原始响应: {response}")
            raise

    def chat_json_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用系统提示词调用 API 并返回 JSON

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 其他参数

        Returns:
            解析后的 JSON 对象
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.chat_json(messages, **kwargs)

    async def async_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        异步调用 Chat API（用于批量并发处理）

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            LLM 响应文本
        """
        # 注意：这是一个简单的异步封装
        # 实际生产环境中可能需要使用 httpx 等异步库
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chat(messages, **kwargs)
        )

    def batch_chat(
        self,
        message_list: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        批量调用 Chat API（串行）

        Args:
            message_list: 消息列表的列表
            **kwargs: 其他参数

        Returns:
            响应列表
        """
        results = []
        total = len(message_list)

        for i, messages in enumerate(message_list, 1):
            logger.info(f"📋 批量处理进度: {i}/{total}")
            response = self.chat(messages, **kwargs)
            results.append(response)

        return results

    async def async_batch_chat(
        self,
        message_list: List[List[Dict[str, str]]],
        max_concurrent: int = 3,
        **kwargs
    ) -> List[str]:
        """
        批量异步调用 Chat API（并发）

        Args:
            message_list: 消息列表的列表
            max_concurrent: 最大并发数
            **kwargs: 其他参数

        Returns:
            响应列表
        """
        import asyncio
        from asyncio import Semaphore

        semaphore = Semaphore(max_concurrent)

        async def limited_chat(messages):
            async with semaphore:
                return await self.async_chat(messages, **kwargs)

        tasks = [limited_chat(messages) for messages in message_list]
        return await asyncio.gather(*tasks)

    def get_info(self) -> Dict[str, Any]:
        """
        获取客户端信息

        Returns:
            包含 provider、model 等信息的字典
        """
        return {
            "provider": self.provider,
            "model": self.model_name,
            "temperature": LLMConfig.TEMPERATURE,
            "max_tokens": LLMConfig.MAX_TOKENS,
            "max_retries": LLMConfig.MAX_RETRIES
        }


# 全局单例
_global_llm_client = None


def get_llm_client() -> LLMClient:
    """
    获取全局 LLM 客户端单例

    Returns:
        LLMClient 实例
    """
    global _global_llm_client
    if _global_llm_client is None:
        _global_llm_client = LLMClient()
    return _global_llm_client


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    try:
        client = LLMClient()

        # 测试基本调用
        response = client.chat_with_system(
            system_prompt="你是一个有帮助的助手。",
            user_prompt="请用一句话介绍什么是 RAG。"
        )
        print(f"\n响应:\n{response}")

        # 测试 JSON 模式
        json_response = client.chat_json_with_system(
            system_prompt="你是一个 JSON 生成器。请始终返回有效的 JSON。",
            user_prompt='请生成一个包含 "name" 和 "description" 字段的 JSON 对象，描述 RAG 技术。'
        )
        print(f"\nJSON 响应:\n{json.dumps(json_response, indent=2, ensure_ascii=False)}")

        # 客户端信息
        info = client.get_info()
        print(f"\n客户端信息:\n{json.dumps(info, indent=2)}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
