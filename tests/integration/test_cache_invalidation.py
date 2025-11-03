import asyncio

from context.unified_cache import CacheOperationRequest, context_cache, invalidate_prefixes


def test_invalidate_prefixes_clears_all_levels():
    async def runner():
        await context_cache.set_request(
            CacheOperationRequest(key="context:1:2:alpha", cache_level=1, importance=0.4),
            {"foo": "bar"},
        )
        await context_cache.set_request(
            CacheOperationRequest(key="context:lore:1:2:beta", cache_level=2, importance=0.4),
            {"lore": "bundle"},
        )
        await context_cache.set_request(
            CacheOperationRequest(key="other:keep", cache_level=3, importance=0.1),
            {"should": "stay"},
        )

        removed = await invalidate_prefixes(["context:1:2", "context:lore:1:2"])

        assert removed == 2
        assert "context:1:2:alpha" not in context_cache.l1_cache
        assert "context:lore:1:2:beta" not in context_cache.l2_cache
        assert "other:keep" in context_cache.l3_cache

    asyncio.run(runner())
