"""
System .env configuration discovery and update helpers.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import dotenv_values, set_key

MASKED_SECRET = "********"

DB_MANAGED_KEYS = {
    "ENABLE_CHAT_MODE",
    "CHAT_MODE_THRESHOLD",
    "ENABLE_AUTO_TAG_FILTER",
    "AUTO_TAG_FILTER_THRESHOLD",
    "RAG_CONFIDENCE_THRESHOLD",
    "RAG_RERANK_SCORE_THRESHOLD",
    "RAG_L2_DISTANCE_THRESHOLD",
    "RAG_RERANK_GOOD_THRESHOLD",
    "RAG_RERANK_EXCELLENT_THRESHOLD",
    "RAG_L2_GOOD_THRESHOLD",
    "RAG_L2_EXCELLENT_THRESHOLD",
    "RAG_ENTITY_TOP_K",
    "RAG_MULTI_ENTITY_DEDUP_LIMIT",
    "RAG_RERANK_TOP_N",
    "RAG_SINGLE_QUERY_TOP_K",
    "RAG_FILES_DISPLAY_LIMIT",
    "ENABLE_WEB_SEARCH_FALLBACK",
    "WEB_SEARCH_MAX_RESULTS",
    "WEB_SEARCH_TIMEOUT_SECONDS",
}

SECTION_SLUGS = {
    "路径配置": "paths",
    "登录鉴权配置": "auth",
    "文档解析配置": "document_parsing",
    "LLM 配置": "llm",
    "Embedding 配置": "embedding",
    "向量库 Milvus 配置": "vector_db",
    "Tokenizer 配置": "tokenizer",
    "Chunk 切分配置": "chunk",
    "标签配置": "tagging",
    "日志配置": "logging",
    "输出配置": "output",
    "性能优化配置": "performance",
    "验证配置": "validation",
    "服务器配置": "server",
    "服务 URL 配置": "service_urls",
    "互联网检索兜底配置": "web_search",
}

SECRET_PATTERNS = (
    "API_KEY",
    "SECRET",
    "ACCESS_TOKEN",
    "REFRESH_TOKEN",
    "PASSWORD",
    "ACCESSKEY",
    "ACCESS_KEY",
    "CLIENT_SECRET",
)

KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
SECTION_RE = re.compile(r"^#\s*=+\s*(?P<title>.*?)\s*=+\s*$")


def default_env_path() -> Path:
    return Path(__file__).parent / ".env"


def default_template_path() -> Path:
    return Path(__file__).parent / "env.template"


def get_system_env_config(
    env_path: Optional[Path] = None,
    template_path: Optional[Path] = None,
) -> Dict[str, Any]:
    env_path = Path(env_path or default_env_path())
    template_path = Path(template_path or default_template_path())

    template_items = _parse_template(template_path)
    env_values = dict(dotenv_values(env_path)) if env_path.exists() else {}

    configs: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    group_labels: Dict[str, str] = {}

    for item in template_items:
        key = item["key"]
        if key in DB_MANAGED_KEYS:
            continue

        value = env_values.get(key)
        source = ".env" if value is not None else "default"
        if value is None:
            value = item["default_value"]

        config = _build_config_item(
            key=key,
            value=value,
            default_value=item["default_value"],
            description=item["description"],
            group=item["group"],
            group_label=item["group_label"],
            source=source,
        )
        configs[key] = config
        grouped.setdefault(item["group"], {})[key] = config
        group_labels[item["group"]] = item["group_label"]

    for key, value in env_values.items():
        if not key or key in configs or key in DB_MANAGED_KEYS:
            continue
        if not KEY_RE.match(key):
            continue

        group = "other"
        config = _build_config_item(
            key=key,
            value=value,
            default_value="",
            description="未在 env.template 中声明的环境变量",
            group=group,
            group_label="其他配置",
            source=".env",
        )
        configs[key] = config
        grouped.setdefault(group, {})[key] = config
        group_labels[group] = "其他配置"

    db_file_value = _raw_value(configs.get("DB_FILE", {}).get("value", env_values.get("DB_FILE", ".dbs/rag_preprocessor.db")))
    database_exists = Path(db_file_value).exists()

    return {
        "env_file": str(env_path.absolute()),
        "template_file": str(template_path.absolute()),
        "configs": configs,
        "grouped": grouped,
        "group_labels": group_labels,
        "database_file": str(Path(db_file_value).absolute()),
        "database_exists": database_exists,
        "milvus_collection": _raw_value(configs.get("MILVUS_COLLECTION_NAME", {}).get("value", "")),
    }


def update_system_env_config(
    updates: Dict[str, Any],
    env_path: Optional[Path] = None,
    template_path: Optional[Path] = None,
) -> Dict[str, Any]:
    env_path = Path(env_path or default_env_path())
    template_path = Path(template_path or default_template_path())
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if not env_path.exists():
        env_path.touch()

    current = get_system_env_config(env_path=env_path, template_path=template_path)
    known_configs = current["configs"]

    updated_keys: List[str] = []
    skipped_keys: List[str] = []

    for key, value in updates.items():
        if not KEY_RE.match(key):
            raise ValueError(f"非法配置键: {key}")
        if key in DB_MANAGED_KEYS:
            raise ValueError(f"{key} 由 RAG 配置页管理，不能在系统配置中更新")

        existing = known_configs.get(key)
        is_secret = existing.get("is_secret", _is_secret_key(key)) if existing else _is_secret_key(key)
        if is_secret and value == MASKED_SECRET:
            skipped_keys.append(key)
            continue

        formatted_value = _format_value(value)
        set_key(env_path, key, formatted_value, quote_mode="never")
        os.environ[key] = formatted_value
        updated_keys.append(key)

    return {
        "success": True,
        "updated_count": len(updated_keys),
        "updated_keys": updated_keys,
        "skipped_keys": skipped_keys,
        "config": get_system_env_config(env_path=env_path, template_path=template_path),
    }


def _parse_template(template_path: Path) -> List[Dict[str, str]]:
    if not template_path.exists():
        return []

    items: List[Dict[str, str]] = []
    current_group_label = "其他配置"
    current_group = "other"
    pending_comments: List[str] = []

    for raw_line in template_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        section_match = SECTION_RE.match(line)
        if section_match:
            current_group_label = section_match.group("title").strip() or "其他配置"
            current_group = SECTION_SLUGS.get(current_group_label, _slugify(current_group_label))
            pending_comments = []
            continue

        if not line:
            pending_comments = []
            continue

        if line.startswith("#"):
            comment = line.lstrip("#").strip()
            if comment:
                pending_comments.append(comment)
            continue

        if "=" not in line:
            pending_comments = []
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not KEY_RE.match(key):
            pending_comments = []
            continue

        items.append({
            "key": key,
            "default_value": _strip_quotes(value.strip()),
            "description": " ".join(pending_comments).strip() or key,
            "group": current_group,
            "group_label": current_group_label,
        })
        pending_comments = []

    return items


def _build_config_item(
    key: str,
    value: Any,
    default_value: Any,
    description: str,
    group: str,
    group_label: str,
    source: str,
) -> Dict[str, Any]:
    value = "" if value is None else str(value)
    default_value = "" if default_value is None else str(default_value)
    is_secret = _is_secret_key(key)
    has_value = value != ""

    return {
        "key": key,
        "value": MASKED_SECRET if is_secret and has_value else _coerce_display_value(value),
        "raw_type": "secret" if is_secret else _infer_type(value, default_value),
        "type": "secret" if is_secret else _infer_type(value, default_value),
        "is_secret": is_secret,
        "has_value": has_value,
        "description": description,
        "default_value": MASKED_SECRET if is_secret and default_value else _coerce_display_value(default_value),
        "group": group,
        "group_label": group_label,
        "source": source,
        "requires_restart": True,
    }


def _is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(pattern in upper for pattern in SECRET_PATTERNS)


def _infer_type(value: str, default_value: str) -> str:
    sample = (value if value != "" else default_value).strip().lower()
    if sample in {"true", "false"}:
        return "boolean"
    if _is_number(sample):
        return "number"
    return "text"


def _coerce_display_value(value: str) -> Any:
    lower = value.strip().lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if _is_number(value):
        if "." in value:
            return float(value)
        return int(value)
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _raw_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _is_number(value: str) -> bool:
    try:
        float(value)
        return value != ""
    except (TypeError, ValueError):
        return False


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "other"
