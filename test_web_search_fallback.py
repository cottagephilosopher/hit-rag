"""
Unit tests for the internet-search fallback used by the RAG chat pipeline.
"""

import unittest

from chat.dspy_pipeline import DSPyRAGPipeline
from chat.web_search import WebSearchClient


class FakeSearchClient:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, query, max_results=5, timeout=5.0):
        self.calls.append({
            "query": query,
            "max_results": max_results,
            "timeout": timeout,
        })
        return self.results


def make_pipeline(search_client, enabled=True):
    pipeline = DSPyRAGPipeline.__new__(DSPyRAGPipeline)
    pipeline.web_search_client = search_client
    pipeline.web_search_provider = "duckduckgo"

    def load_config():
        return {
            "enabled": enabled,
            "max_results": 3,
            "timeout": 4.0,
        }

    def generate_response(user_query, retrieved_chunks, conversation_history="", clarification_hint="", intent_note=""):
        return {
            "response": f"answer for {user_query} from {len(retrieved_chunks)} sources",
            "source_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
            "confidence": 0.72,
            "sources": retrieved_chunks,
        }

    pipeline._load_web_search_config = load_config
    pipeline.generate_response = generate_response
    return pipeline


class WebSearchClientTests(unittest.TestCase):
    def test_parse_duckduckgo_html_results(self):
        html = """
        <div class="result">
          <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc">
            Example &amp; Result
          </a>
          <a class="result__snippet">A useful <b>snippet</b> for the page.</a>
        </div>
        """

        results = WebSearchClient()._parse_duckduckgo_html(html, max_results=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Example & Result")
        self.assertEqual(results[0]["url"], "https://example.com/a?x=1")
        self.assertEqual(results[0]["snippet"], "A useful snippet for the page.")

    def test_fallback_respects_disabled_config(self):
        search_client = FakeSearchClient(results=[{
            "title": "Result",
            "url": "https://example.com",
            "snippet": "Snippet",
        }])
        pipeline = make_pipeline(search_client, enabled=False)

        result = pipeline._answer_with_web_search(
            user_query="missing answer",
            search_query="optimized query",
            conversation_history="",
            reason="no_results",
        )

        self.assertIsNone(result)
        self.assertEqual(search_client.calls, [])

    def test_fallback_builds_web_answer_and_sources(self):
        search_client = FakeSearchClient(results=[{
            "title": "Install Guide",
            "url": "https://example.com/install",
            "snippet": "Installation steps from the web.",
        }])
        pipeline = make_pipeline(search_client, enabled=True)

        result = pipeline._answer_with_web_search(
            user_query="how to install",
            search_query="install query",
            conversation_history="history",
            reason="low_relevance",
        )

        self.assertEqual(result["type"], "web_search_answer")
        self.assertEqual(search_client.calls[0]["query"], "install query")
        self.assertIn("互联网搜索结果", result["response"])
        self.assertEqual(result["sources"][0]["metadata"]["type"], "web_search")
        self.assertEqual(result["sources"][0]["metadata"]["url"], "https://example.com/install")
        self.assertEqual(result["source_ids"], ["web-1"])


if __name__ == "__main__":
    unittest.main()
