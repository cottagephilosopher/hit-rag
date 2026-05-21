"""
Internet search helpers for the RAG fallback path.
"""

import os
import re
from html import unescape
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests
from loguru import logger


class WebSearchClient:
    """Small search client with API-key providers and a no-key DuckDuckGo fallback."""

    def __init__(
        self,
        provider: Optional[str] = None,
        session: Optional[requests.Session] = None,
        user_agent: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")).lower()
        self.session = session or requests.Session()
        self.user_agent = user_agent or os.getenv(
            "WEB_SEARCH_USER_AGENT",
            "Mozilla/5.0 (compatible; HIT-RAG/2.0; +https://example.local)",
        )

    def search(self, query: str, max_results: int = 5, timeout: float = 5.0) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query or max_results <= 0:
            return []

        try:
            if self.provider == "brave":
                return self._search_brave(query, max_results, timeout)
            if self.provider == "serper":
                return self._search_serper(query, max_results, timeout)
            return self._search_duckduckgo(query, max_results, timeout)
        except Exception as exc:
            logger.warning(f"Web search failed via {self.provider}: {exc}")
            return []

    def _search_duckduckgo(self, query: str, max_results: int, timeout: float) -> List[Dict[str, Any]]:
        response = self.session.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": self.user_agent},
            timeout=timeout,
        )
        response.raise_for_status()
        return self._parse_duckduckgo_html(response.text, max_results)

    def _search_brave(self, query: str, max_results: int, timeout: float) -> List[Dict[str, Any]]:
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            logger.warning("BRAVE_SEARCH_API_KEY is not configured")
            return []

        response = self.session.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
                "User-Agent": self.user_agent,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": item.get("title", "").strip(),
                "url": item.get("url", "").strip(),
                "snippet": item.get("description", "").strip(),
                "provider": "brave",
            }
            for item in data.get("web", {}).get("results", [])[:max_results]
            if item.get("url")
        ]

    def _search_serper(self, query: str, max_results: int, timeout: float) -> List[Dict[str, Any]]:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            logger.warning("SERPER_API_KEY is not configured")
            return []

        response = self.session.post(
            "https://google.serper.dev/search",
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": api_key,
                "User-Agent": self.user_agent,
            },
            json={"q": query, "num": max_results},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": item.get("title", "").strip(),
                "url": item.get("link", "").strip(),
                "snippet": item.get("snippet", "").strip(),
                "provider": "serper",
            }
            for item in data.get("organic", [])[:max_results]
            if item.get("link")
        ]

    def _parse_duckduckgo_html(self, html: str, max_results: int) -> List[Dict[str, Any]]:
        title_pattern = re.compile(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>'
            r'(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        matches = list(title_pattern.finditer(html or ""))
        results: List[Dict[str, Any]] = []
        seen_urls = set()

        for index, match in enumerate(matches):
            if len(results) >= max_results:
                break

            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(html)
            fragment = html[match.end():next_start]
            snippet = self._extract_duckduckgo_snippet(fragment)
            url = self._decode_duckduckgo_url(match.group("href"))

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            results.append({
                "title": self._strip_html(match.group("title")),
                "url": url,
                "snippet": snippet,
                "provider": "duckduckgo",
            })

        return results

    def _extract_duckduckgo_snippet(self, fragment: str) -> str:
        snippet_match = re.search(
            r'<(?:a|div|span)[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
            fragment or "",
            re.IGNORECASE | re.DOTALL,
        )
        if not snippet_match:
            return ""
        return self._strip_html(snippet_match.group(1))

    def _decode_duckduckgo_url(self, href: str) -> str:
        url = unescape(href or "").strip()
        if url.startswith("//"):
            url = f"https:{url}"

        parsed = urlparse(url)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            uddg = parse_qs(parsed.query).get("uddg")
            if uddg:
                return unquote(uddg[0])

        return url

    def _strip_html(self, value: str) -> str:
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", value or "", flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()
