"""Web search tool for troubleshooting — optional, requires API key."""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for troubleshooting information. Use when you encounter
    errors that you cannot resolve with local knowledge alone.

    Args:
        query: Search query string describing the problem.
    """
    # Use DuckDuckGo instant answer API (no key required)
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
    try:
        req = Request(url, headers={"User-Agent": "auto-research-agent/0.1"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = []
        if data.get("AbstractText"):
            results.append(f"📖 {data['AbstractText']}")
        for item in data.get("RelatedTopics", [])[:5]:
            if isinstance(item, dict) and item.get("Text"):
                results.append(f"• {item['Text']}")
        return "\n\n".join(results) if results else f"No results for: {query}"
    except Exception as e:
        return f"Web search failed: {e}. Try a different query or use local knowledge."


def create_web_tools() -> list:
    """Create web tools (currently just search)."""
    return [web_search]
