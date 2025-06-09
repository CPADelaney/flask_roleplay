from __future__ import annotations

import aiohttp
from agents import function_tool

@function_tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return result snippets."""
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
    }
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.duckduckgo.com/", params=params, timeout=10) as resp:
            data = await resp.json()
    results: list[str] = []
    for item in data.get("RelatedTopics", [])[:max_results]:
        if isinstance(item, dict) and "Text" in item and "FirstURL" in item:
            results.append(f"{item['Text']} - {item['FirstURL']}")
    return "\n".join(results)
