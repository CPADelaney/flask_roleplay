# utils/caching.py
"""
Caching utilities for Nyx.

This module provides:
- EnhancedCache: async, multi-level cache (L1 in-memory + optional Redis)
- MemoryCache: simple thread-safe in-memory cache with TTL
- ComputationCache: function-call result caching
- Global cache instances and thin async wrappers: get_cache/set_cache/delete_cache
- Prometheus metrics (graceful no-op fallback if prometheus_client is unavailable)

Public API kept stable as a drop-in:
- Globals: enhanced_main_cache, NPC_CACHE, LOCATION_CACHE, AGGREGATOR_CACHE, TIME_CACHE,
           COMPUTATION_CACHE, USER_MODEL_CACHE, MEMORY_CACHE
- Wrappers: get_cache, set_cache, delete_cache (and get/set/delete aliases)
- Decorator: cache.cached(timeout=...)
- Constants: CACHE_TTL, NPC_DIRECTIVE_CACHE, AGENT_DIRECTIVE_CACHE
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import hashlib
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict
from os import environ

# --- Optional deps with graceful fallback ------------------------------------

try:
    from prometheus_client import Counter, Histogram, Gauge  # type: ignore
    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover
    _METRICS_AVAILABLE = False

    class _NoopMetric:
        def labels(self, *_, **__): return self
        def inc(self, *_ , **__): pass
        def set(self, *_ , **__): pass
        def observe(self, *_ , **__): pass

    Counter = Histogram = Gauge = _NoopMetric  # type: ignore

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None

logger = logging.getLogger(__name__)

# --- Prometheus metrics -------------------------------------------------------

CACHE_HITS = Counter('cache_hits_total', 'Number of cache hits', ['cache_level'])
CACHE_MISSES = Counter('cache_misses_total', 'Number of cache misses', ['cache_level'])
CACHE_SIZE = Gauge('cache_size_bytes', 'Current cache size in bytes', ['cache_level'])
CACHE_EVICTIONS = Counter('cache_evictions_total', 'Number of cache evictions', ['cache_level', 'reason'])
CACHE_LATENCY = Histogram('cache_operation_latency_seconds', 'Cache operation latency in seconds', ['operation'])

# --- General constants / small maps ------------------------------------------

# General-purpose directive caches (tiny dicts; used by other modules occasionally)
NPC_DIRECTIVE_CACHE: Dict[str, Any] = {}
AGENT_DIRECTIVE_CACHE: Dict[str, Any] = {}

class CACHE_TTL:
    """Time-to-live constants (seconds)."""
    DIRECTIVES = 300     # 5 minutes for directives
    AGENT_STATE = 60     # 1 minute
    NPC_STATE = 120      # 2 minutes
    MEMORY = 600         # 10 minutes
    DEFAULT = 3600       # 1 hour default

# --- Cache data structures ----------------------------------------------------

@dataclass
class CacheItem:
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    tags: Set[str]

# --- Eviction policies --------------------------------------------------------

class EvictionPolicy:
    def select_items_to_evict(self, items: Dict[str, CacheItem], required_space: int) -> List[str]:
        raise NotImplementedError

class LRUEvictionPolicy(EvictionPolicy):
    def select_items_to_evict(self, items: Dict[str, CacheItem], required_space: int) -> List[str]:
        sorted_items = sorted(items.items(), key=lambda kv: kv[1].last_accessed)
        freed = 0
        to_evict: List[str] = []
        for k, v in sorted_items:
            if freed >= required_space:
                break
            freed += v.size_bytes
            to_evict.append(k)
        return to_evict

class LFUEvictionPolicy(EvictionPolicy):
    def select_items_to_evict(self, items: Dict[str, CacheItem], required_space: int) -> List[str]:
        sorted_items = sorted(items.items(), key=lambda kv: kv[1].access_count)
        freed = 0
        to_evict: List[str] = []
        for k, v in sorted_items:
            if freed >= required_space:
                break
            freed += v.size_bytes
            to_evict.append(k)
        return to_evict

class TTLEvictionPolicy(EvictionPolicy):
    def __init__(self, ttl: float):
        self.ttl = ttl

    def select_items_to_evict(self, items: Dict[str, CacheItem], required_space: int) -> List[str]:
        now = time.time()
        expired: List[str] = []
        freed = 0
        for k, v in items.items():
            if now - v.created_at > self.ttl:
                expired.append(k)
                freed += v.size_bytes
        if freed >= required_space:
            return expired
        # If still not enough, fall back to LRU on remaining
        remaining = {k: v for k, v in items.items() if k not in expired}
        if not remaining:
            return expired
        more = LRUEvictionPolicy().select_items_to_evict(remaining, required_space - freed)
        return expired + more

# --- Enhanced async cache (L1 + optional Redis) ------------------------------

class EnhancedCache:
    """
    Multi-level cache:
    - L1: in-memory, OrderedDict with TTL & eviction
    - L2: optional Redis (JSON-serialized values)
    """

    def __init__(
        self,
        max_size_mb: float = 100.0,
        redis_url: Optional[str] = None,
        eviction_policy: Optional[EvictionPolicy] = None,
        default_ttl: int = 300
    ):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl = int(default_ttl)

        # L1 structures
        self.l1_cache: "OrderedDict[str, CacheItem]" = OrderedDict()
        self.current_size_bytes = 0

        # Optional Redis
        self.redis_client: Optional["redis.Redis"] = None
        if redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("EnhancedCache: connected to Redis.")
            except Exception as e:
                logger.warning(f"EnhancedCache: Redis unavailable ({e!s}). Falling back to L1 only.")
                self.redis_client = None

        # Eviction policy
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()

        # Access pattern tracking (lightweight)
        self.access_patterns: Dict[Tuple[str, str], int] = {}
        self._last_access: Tuple[Optional[str], float] = (None, 0.0)

        # Stats
        self.stats = {
            'hits_l1': 0,
            'hits_redis': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0,
        }

        # Concurrency
        self.lock = asyncio.Lock()
        self._maintenance_task: Optional[asyncio.Task] = None

        logger.info(
            f"EnhancedCache initialized (L1={max_size_mb}MB, TTL={self.default_ttl}s, "
            f"Redis={'on' if self.redis_client else 'off'})"
        )

    async def start(self):
        if self._maintenance_task is None or self._maintenance_task.done():
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            logger.info("EnhancedCache maintenance task started.")

    async def stop(self):
        if self._maintenance_task and not self._maintenance_task.done():
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"EnhancedCache.stop: task join error: {e!s}")
        self._maintenance_task = None
        logger.info("EnhancedCache maintenance task stopped.")

    async def get(self, key: str) -> Optional[Any]:
        t0 = time.time()
        cache_level = "none"
        try:
            # L1
            async with self.lock:
                item = self.l1_cache.get(key)
                if item:
                    if (time.time() - item.created_at) <= self.default_ttl:
                        item.last_accessed = time.time()
                        item.access_count += 1
                        self.l1_cache.move_to_end(key)
                        CACHE_HITS.labels(cache_level='l1').inc()
                        self.stats['hits_l1'] += 1
                        cache_level = "l1"
                        self._record_access_pattern(key)
                        return item.value
                    else:
                        # expired
                        await self._evict_items([key], reason='ttl')

            # Redis
            if self.redis_client:
                try:
                    value_str = await asyncio.to_thread(self.redis_client.get, key)
                    if value_str is not None:
                        try:
                            value = json.loads(value_str)
                        except Exception:
                            # Fallback: raw string if not JSON
                            value = value_str
                        await self.set(key, value, ttl=self.default_ttl)
                        CACHE_HITS.labels(cache_level='redis').inc()
                        self.stats['hits_redis'] += 1
                        cache_level = "redis"
                        return value
                except Exception as e:
                    logger.debug(f"EnhancedCache.get: Redis error for {key}: {e!s}")

            # Miss
            CACHE_MISSES.labels(cache_level='all').inc()
            self.stats['misses'] += 1
            return None
        finally:
            CACHE_LATENCY.labels(operation='get').observe(time.time() - t0)
            logger.debug(f"EnhancedCache.get('{key}') -> {cache_level}")

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        t0 = time.time()
        ttl = int(ttl or self.default_ttl)
        try:
            # Serialize to JSON for size estimation and optional Redis
            try:
                value_str = json.dumps(value)
                size_bytes = len(value_str.encode("utf-8"))
            except Exception:
                # Store un-serialized in L1 (size estimate best-effort)
                value_str = None
                size_bytes = sys.getsizeof(value)

            async with self.lock:
                existing = self.l1_cache.get(key)
                size_diff = size_bytes - (existing.size_bytes if existing else 0)

                if self.current_size_bytes + size_diff > self.max_size_bytes:
                    required = (self.current_size_bytes + size_diff) - self.max_size_bytes
                    await self._evict_by_policy(required_space=required)

                now = time.time()
                item = CacheItem(
                    key=key,
                    value=value,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    size_bytes=size_bytes,
                    tags=set(),
                )

                if existing:
                    self.current_size_bytes -= existing.size_bytes
                self.l1_cache[key] = item
                self.l1_cache.move_to_end(key)
                self.current_size_bytes += size_bytes
                self.stats['size_bytes'] = self.current_size_bytes
                CACHE_SIZE.labels(cache_level='l1').set(self.current_size_bytes)

            if self.redis_client and value_str is not None:
                try:
                    await asyncio.to_thread(self.redis_client.setex, key, ttl, value_str)
                except Exception as e:
                    logger.debug(f"EnhancedCache.set: Redis setex({key}) failed: {e!s}")

            return True
        finally:
            CACHE_LATENCY.labels(operation='set').observe(time.time() - t0)

    async def invalidate(self, key: str):
        t0 = time.time()
        try:
            async with self.lock:
                if key in self.l1_cache:
                    item = self.l1_cache.pop(key)
                    self.current_size_bytes -= item.size_bytes
                    self.stats['size_bytes'] = self.current_size_bytes
                    CACHE_SIZE.labels(cache_level='l1').set(self.current_size_bytes)

            if self.redis_client:
                try:
                    await asyncio.to_thread(self.redis_client.delete, key)
                except Exception as e:
                    logger.debug(f"EnhancedCache.invalidate: Redis delete({key}) failed: {e!s}")
        finally:
            CACHE_LATENCY.labels(operation='invalidate').observe(time.time() - t0)

    # --- internals -----------------------------------------------------------

    async def _evict_items(self, keys: List[str], reason: str = 'unknown'):
        for k in keys:
            if k in self.l1_cache:
                item = self.l1_cache.pop(k)
                self.current_size_bytes -= item.size_bytes
                self.stats['evictions'] += 1
                CACHE_EVICTIONS.labels(cache_level='l1', reason=reason).inc()
        CACHE_SIZE.labels(cache_level='l1').set(self.current_size_bytes)

    async def _evict_by_policy(self, required_space: int):
        if required_space <= 0 or not self.l1_cache:
            return
        keys = self.eviction_policy.select_items_to_evict(dict(self.l1_cache), required_space)
        await self._evict_items(keys, reason='size')
        logger.info(f"EnhancedCache: evicted {len(keys)} item(s) to free space (~{required_space}B).")

    def _record_access_pattern(self, key: str):
        now = time.time()
        last_key, last_t = self._last_access
        if last_key and (now - last_t) < 5.0:
            pattern = (last_key, key)
            self.access_patterns[pattern] = self.access_patterns.get(pattern, 0) + 1
        self._last_access = (key, now)

    async def _maintenance_loop(self):
        logger.debug("EnhancedCache maintenance loop started.")
        try:
            while True:
                await asyncio.sleep(60)
                now = time.time()
                expired: List[str] = []
                async with self.lock:
                    for k, v in list(self.l1_cache.items()):
                        if now - v.created_at > self.default_ttl:
                            expired.append(k)
                    if expired:
                        await self._evict_items(expired, reason='ttl')
                        logger.info(f"EnhancedCache: evicted {len(expired)} expired item(s).")
        except asyncio.CancelledError:
            logger.debug("EnhancedCache maintenance loop cancelled.")
        except Exception as e:
            logger.warning(f"EnhancedCache maintenance loop error: {e!s}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'size_l1_mb': self.current_size_bytes / (1024 * 1024),
            'items_l1': len(self.l1_cache),
            'tracked_patterns': len(self.access_patterns),
        }

# --- Simple in-memory cache with TTL (thread-safe) ---------------------------

class MemoryCache:
    """
    Thread-safe in-memory cache with per-key TTL and light stats.
    """

    def __init__(self, name: str = "default", max_size: int = 100, default_ttl: int = 60):
        self.name = name
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttls: Dict[str, int] = {}
        self.max_size = int(max_size)
        self.default_ttl = int(default_ttl)

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lock = threading.RLock()

        self.estimated_memory_usage = 0
        self.last_cleanup = time.time()
        logger.info(f"MemoryCache[{self.name}] init (max_size={self.max_size}, ttl={self.default_ttl}s)")

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            if self._is_expired(key):
                self._remove(key)
                self.misses += 1
                return None
            self.hits += 1
            self.timestamps[key] = time.time()
            return self.cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self.lock:
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict_oldest()

            try:
                value_size = sys.getsizeof(value)
                key_size = sys.getsizeof(key)
                if key in self.cache:
                    old_size = sys.getsizeof(self.cache[key])
                    self.estimated_memory_usage += max(0, value_size - old_size)
                else:
                    self.estimated_memory_usage += (key_size + value_size)
            except Exception:
                pass

            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.ttls[key] = int(ttl) if ttl is not None else self.default_ttl
            self._maybe_cleanup()

    def delete(self, key: str) -> None:
        with self.lock:
            self._remove(key)

    # ---- internals ----

    def _is_expired(self, key: str) -> bool:
        ts = self.timestamps.get(key)
        ttl = self.ttls.get(key)
        if ts is None or ttl is None:
            return True
        return time.time() > (ts + ttl)

    def _remove(self, key: str) -> None:
        if key in self.cache:
            try:
                self.estimated_memory_usage -= (sys.getsizeof(key) + sys.getsizeof(self.cache[key]))
            except Exception:
                pass
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
            self.ttls.pop(key, None)

    def _evict_oldest(self) -> None:
        if not self.timestamps:
            return
        oldest_key = min(self.timestamps, key=self.timestamps.get)
        self._remove(oldest_key)
        self.evictions += 1
        if self.evictions % 50 == 0:
            logger.info(f"MemoryCache[{self.name}]: evictions={self.evictions}")

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self.last_cleanup < 60:
            return
        self.last_cleanup = now
        expired = [k for k in list(self.cache.keys()) if self._is_expired(k)]
        for k in expired:
            self._remove(k)
        if expired:
            logger.debug(f"MemoryCache[{self.name}]: cleaned {len(expired)} expired")

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.ttls.clear()
            self.estimated_memory_usage = 0
            self.hits = self.misses = self.evictions = 0

    def remove_pattern(self, pattern: str) -> int:
        with self.lock:
            keys = [k for k in self.cache if pattern in k]
            for k in keys:
                self._remove(k)
            return len(keys)

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_ratio = (self.hits / total) if total else 0.0
            return {
                "name": self.name,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": f"{hit_ratio:.2%}",
                "evictions": self.evictions,
                "memory_usage_bytes": max(0, self.estimated_memory_usage),
                "memory_usage_mb": f"{max(0, self.estimated_memory_usage) / (1024 * 1024):.3f}",
            }

# --- Computation cache (function call results) -------------------------------

class ComputationCache:
    """Cache results of (sync/async) functions keyed by function+args."""

    def __init__(self, name: str = "computation", max_size: int = 50, default_ttl: int = 300):
        self.memory_cache = MemoryCache(name=name, max_size=max_size, default_ttl=default_ttl)

    async def cached_call(self, func: Callable, *args, ttl: Optional[int] = None, **kwargs) -> Any:
        # Robust key
        try:
            parts = [func.__name__]
            parts.extend([repr(a) for a in args])
            parts.extend([f"{k}={repr(v)}" for k, v in sorted(kwargs.items())])
            raw_key = ":".join(parts)
            cache_key = f"comp:{func.__name__}:{hashlib.sha256(raw_key.encode()).hexdigest()[:16]}"
        except Exception:
            cache_key = f"comp:{func.__name__}:{hash(hashlib.sha256.__name__)}"  # fallback

        cached = self.memory_cache.get(cache_key)
        if cached is not None:
            return cached

        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = await asyncio.to_thread(func, *args, **kwargs)

        self.memory_cache.set(cache_key, result, ttl=ttl)
        return result

# --- Global instances & env-config ------------------------------------------

NPC_CACHE_SIZE = int(environ.get("NPC_CACHE_SIZE", "500"))
NPC_CACHE_TTL = int(environ.get("NPC_CACHE_TTL", "300"))
LOCATION_CACHE_SIZE = int(environ.get("LOCATION_CACHE_SIZE", "100"))
LOCATION_CACHE_TTL = int(environ.get("LOCATION_CACHE_TTL", "600"))
AGGREGATOR_CACHE_SIZE = int(environ.get("AGGREGATOR_CACHE_SIZE", "50"))
AGGREGATOR_CACHE_TTL = int(environ.get("AGGREGATOR_CACHE_TTL", "60"))
TIME_CACHE_SIZE = int(environ.get("TIME_CACHE_SIZE", "5"))
TIME_CACHE_TTL = int(environ.get("TIME_CACHE_TTL", "10"))
COMPUTATION_CACHE_SIZE = int(environ.get("COMPUTATION_CACHE_SIZE", "100"))
COMPUTATION_CACHE_TTL = int(environ.get("COMPUTATION_CACHE_TTL", "600"))

# Enhanced cache (L1 + optional Redis)
redis_url_env = environ.get("REDIS_URL")

# Initialize without making import fatal if Redis/Prometheus are missing
enhanced_main_cache = EnhancedCache(max_size_mb=100, redis_url=redis_url_env, default_ttl=300)
# Note: Call enhanced_main_cache.start()/stop() from your app lifecycle if desired.

# Per-domain in-memory caches
NPC_CACHE = MemoryCache(name="npc", max_size=NPC_CACHE_SIZE, default_ttl=NPC_CACHE_TTL)
LOCATION_CACHE = MemoryCache(name="location", max_size=LOCATION_CACHE_SIZE, default_ttl=LOCATION_CACHE_TTL)
AGGREGATOR_CACHE = MemoryCache(name="aggregator", max_size=AGGREGATOR_CACHE_SIZE, default_ttl=AGGREGATOR_CACHE_TTL)
TIME_CACHE = MemoryCache(name="time", max_size=TIME_CACHE_SIZE, default_ttl=TIME_CACHE_TTL)

# Global computation cache
COMPUTATION_CACHE = ComputationCache(
    name="global_computation",
    max_size=COMPUTATION_CACHE_SIZE,
    default_ttl=COMPUTATION_CACHE_TTL
)

# Additional small caches exposed in original module
USER_MODEL_CACHE = MemoryCache(name="user_model", max_size=100, default_ttl=300)
MEMORY_CACHE = MemoryCache(name="memory", max_size=100, default_ttl=300)

# --- Thin async wrappers (stable names) --------------------------------------

async def get(key: str):
    return await enhanced_main_cache.get(key)

async def set(key: str, value: Any, ttl: int = 300):
    return await enhanced_main_cache.set(key, value, ttl)

async def delete(key: str):
    return await enhanced_main_cache.invalidate(key)

# Backwards-compat aliases
async def get_cache(key: str):
    return await get(key)

async def set_cache(key: str, value: Any, ttl: int = 300):
    return await set(key, value, ttl)

async def delete_cache(key: str):
    return await delete(key)

# --- Simple decorator using NPC_CACHE (kept for compatibility) ---------------

class CacheDecorator:
    def cached(self, timeout=300):
        def decorator(f):
            async def decorated(*args, **kwargs):
                key = f"{f.__name__}:{args!r}:{kwargs!r}"
                cached = NPC_CACHE.get(key)
                if cached is not None:
                    return cached
                if asyncio.iscoroutinefunction(f):
                    result = await f(*args, **kwargs)
                else:
                    # non-async function invoked from async context
                    result = await asyncio.to_thread(f, *args, **kwargs)
                NPC_CACHE.set(key, result, timeout)
                return result
            return decorated
        return decorator

# Singleton decorator instance
cache = CacheDecorator()

logger.info("utils.caching loaded successfully.")
