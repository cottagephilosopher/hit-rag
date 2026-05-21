"""
Tests for the system .env configuration manager.
"""

import tempfile
import unittest
from pathlib import Path

from system_env_config import MASKED_SECRET, get_system_env_config, update_system_env_config


class SystemEnvConfigTests(unittest.TestCase):
    def test_template_configs_are_grouped_and_secret_values_are_masked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "env.template"
            env_file = root / ".env"
            template.write_text(
                """# ==================== LLM 配置 ====================
# LLM 服务提供商
LLM_PROVIDER=dashscope
# OpenAI Key
OPENAI_API_KEY=template-key
# 最大 Token
LLM_MAX_TOKENS=4000
# JWT 过期时间
UP_JWT_EXPIRE_MINUTES=180
# ==================== RAG 配置 ====================
RAG_CONFIDENCE_THRESHOLD=0.5
""",
                encoding="utf-8",
            )
            env_file.write_text(
                "LLM_PROVIDER=openai\nOPENAI_API_KEY=real-key\nLLM_MAX_TOKENS=8000\nEXTRA_FLAG=true\n",
                encoding="utf-8",
            )

            result = get_system_env_config(env_path=env_file, template_path=template)

            self.assertIn("llm", result["grouped"])
            self.assertNotIn("rag", result["grouped"])
            self.assertEqual(result["configs"]["LLM_PROVIDER"]["value"], "openai")
            self.assertEqual(result["configs"]["LLM_MAX_TOKENS"]["type"], "number")
            self.assertEqual(result["configs"]["UP_JWT_EXPIRE_MINUTES"]["type"], "number")
            self.assertEqual(result["configs"]["OPENAI_API_KEY"]["value"], MASKED_SECRET)
            self.assertTrue(result["configs"]["OPENAI_API_KEY"]["has_value"])
            self.assertEqual(result["configs"]["EXTRA_FLAG"]["type"], "boolean")

    def test_update_skips_unchanged_masked_secret_and_writes_new_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "env.template"
            env_file = root / ".env"
            template.write_text(
                """# ==================== LLM 配置 ====================
OPENAI_API_KEY=template-key
LLM_PROVIDER=dashscope
ENABLE_CACHE=true
""",
                encoding="utf-8",
            )
            env_file.write_text(
                "OPENAI_API_KEY=real-key\nLLM_PROVIDER=dashscope\nENABLE_CACHE=true\n",
                encoding="utf-8",
            )

            result = update_system_env_config(
                {
                    "OPENAI_API_KEY": MASKED_SECRET,
                    "LLM_PROVIDER": "openai",
                    "ENABLE_CACHE": False,
                },
                env_path=env_file,
                template_path=template,
            )
            text = env_file.read_text(encoding="utf-8")

            self.assertEqual(result["updated_count"], 2)
            self.assertIn("OPENAI_API_KEY=real-key", text)
            self.assertIn("LLM_PROVIDER=openai", text)
            self.assertIn("ENABLE_CACHE=false", text)


if __name__ == "__main__":
    unittest.main()
