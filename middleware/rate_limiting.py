# /middleware/rate_limiting.py

import os
import asyncio
import time
import logging
from functools import wraps
from quart import request, jsonify, current_app, g, abort
from typing import Dict, Optional, Tuple, Set, Any
import aioredis # Changed from redis to aioredis
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class TokenBucketConfig:
    capacity: int
    refill_rate: float # tokens per second
    # refill_time is implicitly handled by how often _refill is called (on each consume)

class TokenBucket: # This class remains synchronous as it's CPU-bound logic
    def __init__(self, config: TokenBucketConfig):
        self.capacity = config.capacity
        self.tokens = float(config.capacity)
        self.refill_rate = config.refill_rate
        self.last_refill = time.time()
        self.lock = asyncio.Lock() # Use asyncio.Lock if methods become async, but consume is sync for now

    def _refill(self): # Stays synchronous
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed > 0: # Ensure some time has passed
            refill_amount = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + refill_amount)
            self.last_refill = now

    async def consume(self, tokens_to_consume: int = 1) -> bool: # Changed to async due to lock
        async with self.lock: # Use asyncio.Lock if this method is awaited
            self._refill() # _refill itself is sync
            if self.tokens >= tokens_to_consume:
                self.tokens -= tokens_to_consume
                return True
            return False

class AsyncRateLimiter: # Renamed to reflect async nature
    def __init__(self):
        self.local_buckets: Dict[str, TokenBucket] = {}
        self.local_buckets_lock = asyncio.Lock() # For managing local_buckets dict

    async def _get_redis(self) -> Optional[aioredis.Redis]: # For AsyncRateLimiter
        try:
            # Use the pool from the app context
            pool = getattr(current_app, 'aioredis_rate_limit_pool', None)
            if not pool:
                logger.warning("aioredis_rate_limit_pool not found on app context.")
                return None
            
            # Get a connection from the pool for this operation/request
            # Storing the connection on 'g' per request is still a valid pattern
            if not hasattr(g, 'rate_limit_aioredis_conn') or g.rate_limit_aioredis_conn.closed:
                g.rate_limit_aioredis_conn = await pool.client() # Get a new client connection
                # Optional: await g.rate_limit_aioredis_conn.ping() # Ping if you want to be sure
            return g.rate_limit_aioredis_conn
            
        except (aioredis.RedisError, ConnectionRefusedError, asyncio.TimeoutError) as e:
            logger.error(f"Failed to connect to aioredis for AsyncRateLimiter: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting aioredis connection: {e}")
            return None

    async def _get_local_bucket(self, key: str, config: TokenBucketConfig) -> TokenBucket:
        """Get or create a local token bucket asynchronously."""
        async with self.local_buckets_lock:
            if key not in self.local_buckets:
                self.local_buckets[key] = TokenBucket(config)
            return self.local_buckets[key]

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        period: int, # seconds
        cost: int = 1,
        distributed: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if the rate limit is exceeded (asynchronous)."""
        try:
            redis_failed = False
            if distributed:
                redis_client = await self._get_redis()
                if redis_client:
                    bucket_key = f"bucket:{key}"
                    now = time.time()

                    pipe = redis_client.pipeline()
                    pipe.hmget(bucket_key, ['tokens', 'last_refill'])
                    # await pipe.hmget(...) is not how pipeline works, add commands then execute
                    results = await pipe.execute()
                    current_vals = results[0] # Result of hmget

                    tokens_str, last_refill_str = current_vals if current_vals else (None, None)
                    current_tokens = float(tokens_str) if tokens_str is not None else float(limit)
                    last_refill_ts = float(last_refill_str) if last_refill_str is not None else now

                    elapsed = now - last_refill_ts
                    refill_rate_per_second = limit / period
                    refill_amount = elapsed * refill_rate_per_second
                    tokens_after_refill = min(float(limit), current_tokens + refill_amount)

                    allowed: bool
                    if tokens_after_refill >= cost:
                        new_tokens_val = tokens_after_refill - cost
                        pipe.multi() # Start transaction
                        pipe.hset(bucket_key, mapping={'tokens': new_tokens_val, 'last_refill': now})
                        pipe.expire(bucket_key, period + 5) # Add a small buffer to expiry
                        allowed = True
                    else:
                        allowed = False
                        # No need to update bucket if not allowed, just report current state

                    # Analytics
                    analytics_key = f"analytics:{key}"
                    pipe.hincrby(analytics_key, 'total_requests', 1)
                    if not allowed:
                        pipe.hincrby(analytics_key, 'blocked_requests', 1)
                    pipe.expire(analytics_key, period * 2)
                    await pipe.execute() # Execute HSET/EXPIRE and analytics

                    remaining = max(0, int(tokens_after_refill - cost if allowed else tokens_after_refill))

                else: # Redis unavailable
                    logger.warning("aioredis unavailable for distributed rate limit check, falling back to local.")
                    redis_failed = True
                    allowed, remaining = await self._check_local_limit_async(key, limit, period, cost)
            else: # Not distributed
                 allowed, remaining = await self._check_local_limit_async(key, limit, period, cost)

            reset_time = int(now + period * ((limit - remaining + cost -1) // limit if remaining < cost else 0 )) # More accurate reset
            # A simpler reset time approximation is just now + period if not allowed.
            # If allowed, reset is when the bucket *would* be full again if it were empty.
            # For token bucket, reset is less about a fixed window and more about when tokens regenerate.
            # The provided headers (X-RateLimit-Reset) are more common for fixed window.
            # For token bucket, perhaps "Retry-After" is more meaningful if blocked.
            # Let's keep a simple reset for X-RateLimit-Reset for now.
            simple_reset_time = int(now + period)


            return allowed, {
                'limit': limit,
                'remaining': remaining,
                'reset': simple_reset_time, # Standard header, typically unix timestamp
                'cost': cost,
                'retry_after': int(period * (cost / limit)) if not allowed and cost > remaining else 0 # Rough estimate
            }

        except (aioredis.RedisError, ConnectionRefusedError, asyncio.TimeoutError) as e:
            logger.error(f"aioredis error during rate limiting check for key {key}: {e}")
            allowed, remaining = await self._check_local_limit_async(key, limit, period, cost)
            return allowed, {
                'limit': limit, 'remaining': remaining, 'reset': int(time.time() + period),
                'cost': cost, 'retry_after': period if not allowed else 0
            }
        except Exception as e:
             logger.exception(f"Unexpected error during async rate limit check for key {key}: {e}")
             return True, {'limit': limit, 'remaining': limit, 'reset': int(time.time()), 'cost': cost, 'retry_after': 0} # Fail open

    async def _check_local_limit_async(self, key: str, limit: int, period: int, cost: int) -> Tuple[bool, int]:
        config = TokenBucketConfig(capacity=limit, refill_rate=float(limit) / period)
        bucket = await self._get_local_bucket(key, config)
        allowed = await bucket.consume(cost) # consume is now async
        remaining = int(bucket.tokens) # tokens is updated synchronously within consume's lock
        return allowed, remaining

# --- Global Instance ---
global_async_rate_limiter = AsyncRateLimiter() # Use the async version

# --- Decorator ---
# This decorator structure was already good for async, just ensure it calls the async rate limiter
def rate_limit(
    limit: int = 100,
    period: int = 60,
    key_func: Optional[callable] = None,
    cost_func: Optional[callable] = None,
    distributed: bool = True
):
    def decorator(f):
        @wraps(f)
        async def wrapped_async_handler(*args, **kwargs):
            key = _get_request_key(key_func) # This helper can remain sync
            current_cost = cost_func(request) if cost_func else 1 # This can remain sync

            # Call the ASYNC check_rate_limit method
            allowed, limit_info = await global_async_rate_limiter.check_rate_limit(
                key=key, limit=limit, period=period, cost=current_cost, distributed=distributed
            )

            if not allowed:
                # Create the 429 response
                data_for_429, status_code_429 = _rate_limit_exceeded_response_data(limit_info)
                response_429 = await current_app.make_response(jsonify(data_for_429))
                response_429.status_code = status_code_429
                if limit_info.get('retry_after', 0) > 0:
                    response_429.headers['Retry-After'] = str(limit_info['retry_after'])
                # Add other X-RateLimit headers
                return await _add_rate_limit_headers_to_response_obj(response_429, limit_info)

            # Call the original function (f)
            if asyncio.iscoroutinefunction(f):
                result_from_f = await f(*args, **kwargs)
            else:
                # If f is sync, run it in an executor to not block the event loop
                # This is crucial if 'f' itself might do blocking work.
                # For simple sync Quart routes returning data, Quart handles ensure_async.
                # But if 'f' is a truly blocking operation, this is safer.
                # For now, let's assume Quart handles it or 'f' is quick.
                # loop = asyncio.get_running_loop()
                # result_from_f = await loop.run_in_executor(None, f, *args, **kwargs)
                result_from_f = f(*args, **kwargs)


            if not isinstance(result_from_f, current_app.response_class):
                final_response = await current_app.make_response(result_from_f)
            else:
                final_response = result_from_f
            
            return await _add_rate_limit_headers_to_response_obj(final_response, limit_info)

        return wrapped_async_handler
    return decorator

# --- Helper functions for decorator (can remain mostly synchronous, except header adding) ---
def _get_request_key(key_func: Optional[callable]) -> str:
    if key_func:
        return key_func(request)
    else:
        user_id_str = str(session.get("user_id", "anonymous")) if session else "anonymous_no_session"
        key_parts = [
            request.access_route[0] if request.access_route else (request.remote_addr or 'unknown_ip'),
            request.endpoint or 'unknown_endpoint',
            user_id_str
        ]
        return ':'.join(filter(None, key_parts))

def _rate_limit_exceeded_response_data(limit_info: Dict) -> Tuple[Dict[str, Any], int]:
    """Helper to create the 429 response data and status."""
    return {
        "error": "Rate limit exceeded",
        "message": f"Too many requests. Please try again later.",
        "limit": limit_info['limit'],
        "remaining": limit_info['remaining'],
        "reset_timestamp": limit_info['reset'], # Unix timestamp
        "retry_after_seconds": limit_info.get('retry_after', 0)
    }, 429

async def _add_rate_limit_headers_to_response_obj(response: QuartResponse, limit_info: Dict): # หรือ 'quart.Response'
    """Actually sets headers on a given Response object."""
    try:
        response.headers.set("X-RateLimit-Limit", str(limit_info['limit']))
        response.headers.set("X-RateLimit-Remaining", str(limit_info['remaining']))
        response.headers.set("X-RateLimit-Reset", str(limit_info['reset']))
    except Exception as e:
        logger.error(f"Failed to add rate limit headers: {e}", exc_info=True)
    return response


# --- Async IPBlockList ---
class AsyncIPBlockList:
    def __init__(self):
        self.local_blocked_ips: Set[str] = set()
        self.redis_key_prefix = "ip_blocklist:"
        self.local_lock = asyncio.Lock()

    async def _get_redis(self) -> Optional[aioredis.Redis]: # Same as RateLimiter's
        try:
            if not hasattr(g, 'ip_block_aioredis_pool'): # Use a different pool attribute on g
                redis_url = current_app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
                g.ip_block_aioredis_pool = aioredis.from_url(redis_url, decode_responses=True)
            if not hasattr(g, 'ip_block_aioredis_conn') or g.ip_block_aioredis_conn.closed:
                 g.ip_block_aioredis_conn = await g.ip_block_aioredis_pool.client()
                 await g.ip_block_aioredis_conn.ping()
            return g.ip_block_aioredis_conn
        except (aioredis.RedisError, ConnectionRefusedError, asyncio.TimeoutError) as e:
            logger.error(f"Failed to connect to aioredis for AsyncIPBlockList: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting aioredis for AsyncIPBlockList: {e}")
            return None

    async def is_blocked(self, ip_address: str, distributed: bool = True) -> bool:
        async with self.local_lock:
            if ip_address in self.local_blocked_ips:
                logger.debug(f"IP {ip_address} found in local blocklist (async)")
                return True

        if distributed:
            redis_client = await self._get_redis()
            if redis_client:
                try:
                    key = f"{self.redis_key_prefix}{ip_address}"
                    if await redis_client.exists(key):
                        logger.debug(f"IP {ip_address} found in aioredis blocklist")
                        return True
                except aioredis.RedisError as e:
                    logger.error(f"aioredis error checking blocked IP {ip_address}: {e}")
        return False

    async def block_ip(self, ip_address: str, reason: str = "Manual block",
                      duration: int = 86400, distributed: bool = True) -> bool:
        async with self.local_lock:
            self.local_blocked_ips.add(ip_address)

        if distributed:
            redis_client = await self._get_redis()
            if redis_client:
                try:
                    key = f"{self.redis_key_prefix}{ip_address}"
                    block_info = json.dumps({"reason": reason, "blocked_at": time.time()})
                    await redis_client.setex(key, duration, block_info)
                    logger.info(f"IP {ip_address} blocked in aioredis for {duration}s: {reason}")
                    return True
                except aioredis.RedisError as e:
                    logger.error(f"aioredis error blocking IP {ip_address}: {e}")
                    return False
            else:
                logger.warning(f"aioredis unavailable, IP {ip_address} blocked locally only.")
                return False
        logger.info(f"IP {ip_address} blocked locally only: {reason}")
        return True # Local block succeeded

    async def unblock_ip(self, ip_address: str, distributed: bool = True) -> bool:
        async with self.local_lock:
            self.local_blocked_ips.discard(ip_address)

        if distributed:
            redis_client = await self._get_redis()
            if redis_client:
                try:
                    key = f"{self.redis_key_prefix}{ip_address}"
                    deleted_count = await redis_client.delete(key)
                    logger.info(f"IP {ip_address} unblocked in aioredis (deleted: {deleted_count})")
                    return True
                except aioredis.RedisError as e:
                    logger.error(f"aioredis error unblocking IP {ip_address}: {e}")
                    return False
            else:
                logger.warning(f"aioredis unavailable, IP {ip_address} unblocked locally only.")
                return False
        logger.info(f"IP {ip_address} unblocked locally")
        return True

    async def get_blocked_ips(self, distributed: bool = True) -> Dict[str, Dict]:
        async with self.local_lock:
            result = {ip: {"reason": "Local block", "blocked_at": 0} for ip in self.local_blocked_ips}

        if distributed:
            redis_client = await self._get_redis()
            if redis_client:
                try:
                    keys_to_fetch = []
                    async for key in redis_client.scan_iter(match=f"{self.redis_key_prefix}*"):
                        keys_to_fetch.append(key)
                    
                    if keys_to_fetch:
                        values = await redis_client.mget(keys_to_fetch)
                        for i, key_full_name in enumerate(keys_to_fetch):
                            ip = key_full_name.replace(self.redis_key_prefix, "")
                            data_str = values[i]
                            if data_str:
                                try:
                                    info = json.loads(data_str)
                                    result[ip] = info
                                except json.JSONDecodeError:
                                    result[ip] = {"reason": "Invalid data in Redis", "blocked_at": 0}
                            else: # Key expired between SCAN and MGET
                                result.pop(ip, None)
                except aioredis.RedisError as e:
                    logger.error(f"aioredis error getting blocked IPs: {e}")
            else:
                logger.warning("aioredis unavailable for get_blocked_ips, returning local list only.")
        return result

# Global instance for IP blocking
global_async_ip_block_list = AsyncIPBlockList()

# Async Middleware for IP Blocking
async def async_ip_block_middleware():
    """Async middleware function for IP blocking."""
    # Try to get client IP from common headers first, then remote_addr
    # This handles proxies better. Ensure your proxy (e.g., Nginx, Render's LB) sets X-Forwarded-For.
    # You might need to configure Quart's PROXY_FIX settings if behind multiple proxies.
    x_forwarded_for = request.headers.get('X-Forwarded-For')
    if x_forwarded_for:
        client_ip = x_forwarded_for.split(',')[0].strip()
    else:
        client_ip = request.remote_addr # remote_addr might be proxy IP if not configured

    if client_ip:
        if await global_async_ip_block_list.is_blocked(client_ip):
            logger.warning(f"Blocked request from IP: {client_ip} (async)")
            # abort() is synchronous. For async, we return a response.
            # Or raise a specific HTTPException that Quart can handle.
            # For simplicity, let's construct the response directly.
            return jsonify({"error": "Forbidden", "message": "Your IP address has been blocked."}), 403
    else:
        logger.warning("Could not determine client IP address for blocking check (async).")
    # If not blocked, allow request to proceed (by returning None from before_request)
    return None


# In your main.py or app factory:
# from middleware.rate_limiting import rate_limit, async_ip_block_middleware
#
# app = Quart(__name__)
# app.before_request(async_ip_block_middleware) # Register async middleware
#
# @app.route('/limited_async')
# @rate_limit(limit=5, period=60)
# async def limited_route_async():
#     return {"message": "This is an async rate-limited route."}
