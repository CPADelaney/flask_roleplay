# /middleware/rate_limiting.py

"""
Enhanced rate limiting middleware with token bucket algorithm and user-based limits.
"""
import os
import asyncio
import time
import threading
import logging
# Removed asyncio import as it's not needed for the core logic here anymore
from functools import wraps
from quart import request, jsonify, current_app, g, abort # Added abort
from typing import Dict, Optional, Tuple, Set
import redis
from dataclasses import dataclass
import json # Keep json import

# Import database connections
# from db.connection import get_db_connection_context # Keep if needed elsewhere

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TokenBucketConfig:
    """Configuration for token bucket rate limiter."""
    capacity: int
    refill_rate: float
    refill_time: float

class TokenBucket:
    """Token bucket implementation for rate limiting."""
    def __init__(self, config: TokenBucketConfig):
        self.capacity = config.capacity
        self.tokens = float(config.capacity) # Use float for accuracy
        self.refill_rate = config.refill_rate
        self.refill_time = config.refill_time # Time between minimum refills
        self.last_refill = time.time()
        self.lock = threading.Lock() # Keep threading lock for local bucket

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        # Calculate tokens to add since last refill check
        refill_amount = elapsed * self.refill_rate
        if refill_amount > 0:
            self.tokens = min(self.capacity, self.tokens + refill_amount)
            self.last_refill = now

    def consume(self, tokens: int = 1) -> bool: # Changed to synchronous
        """Try to consume tokens from the bucket."""
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class RateLimiter:
    """Enhanced rate limiter with Redis backend and analytics."""
    def __init__(self):
        # self.redis_client = None # No need to store instance variable if using 'g'
        self.local_buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def _get_redis(self) -> Optional[redis.Redis]:
        """Get or create Redis connection (synchronous)."""
        try:
            if not hasattr(g, 'rate_limit_redis'):
                # Use os.environ directly to get REDIS_URL as a fallback
                redis_url = current_app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
                g.rate_limit_redis = redis.from_url(redis_url, decode_responses=True)
                g.rate_limit_redis.ping() # Test connection
                logger.debug(f"Redis connection established for RateLimiter using {redis_url}")
            return g.rate_limit_redis
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis for RateLimiter: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting Redis connection for RateLimiter: {e}")
            return None


    # --- Made Synchronous ---
    def _get_bucket(self, key: str, config: TokenBucketConfig) -> TokenBucket:
        """Get or create a local token bucket (synchronous)."""
        with self.lock:
            if key not in self.local_buckets:
                self.local_buckets[key] = TokenBucket(config)
            return self.local_buckets[key]

    # --- Made Synchronous ---
    def check_rate_limit(
        self,
        key: str,
        limit: int,
        period: int,
        cost: int = 1,
        distributed: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Check if the rate limit is exceeded (synchronous).
        """
        try:
            redis_failed = False
            if distributed:
                redis_client = self._get_redis()
                if redis_client:
                    # Use token bucket algorithm with Redis
                    bucket_key = f"bucket:{key}"
                    now = time.time()

                    # Use Redis pipeline for atomic operations
                    pipe = redis_client.pipeline()
                    pipe.hmget(bucket_key, ['tokens', 'last_refill'])
                    current_vals = pipe.execute()[0]

                    tokens_str, last_refill_str = current_vals
                    # Default values if key doesn't exist or fields are missing
                    current_tokens = float(tokens_str) if tokens_str is not None else float(limit)
                    last_refill = float(last_refill_str) if last_refill_str is not None else now

                    # Calculate refill
                    elapsed = now - last_refill
                    refill_amount = elapsed * (limit / period)
                    tokens = min(limit, current_tokens + refill_amount)

                    # Try to consume tokens
                    if tokens >= cost:
                        new_tokens = tokens - cost
                        # Update bucket atomically
                        pipe.hset(bucket_key, mapping={
                            'tokens': new_tokens,
                            'last_refill': now
                        })
                        # Set expiry only if the key is new or state changed significantly,
                        # or more simply, always set it. TTL will be updated.
                        pipe.expire(bucket_key, period)
                        pipe.execute()
                        allowed = True
                    else:
                        allowed = False

                    # Get remaining tokens accurately
                    remaining = max(0, int(tokens - cost if allowed else tokens))

                    # Update analytics (consider if this needs full atomicity)
                    analytics_key = f"analytics:{key}"
                    pipe.hincrby(analytics_key, 'total_requests', 1)
                    if not allowed:
                        pipe.hincrby(analytics_key, 'blocked_requests', 1)
                    pipe.expire(analytics_key, period * 2) # Keep analytics longer
                    pipe.execute()

                else:
                    # Redis connection failed
                    logger.warning("Redis unavailable for distributed rate limit check, falling back to local.")
                    redis_failed = True
                    allowed, remaining = self._check_local_limit(key, limit, period, cost)

            else: # Not distributed
                 allowed, remaining = self._check_local_limit(key, limit, period, cost)

            reset_time = int(time.time() + period) # Simple approximation

            return allowed, {
                'limit': limit,
                'remaining': remaining,
                'reset': reset_time,
                'cost': cost
            }

        except redis.RedisError as e:
            logger.error(f"Redis error during rate limiting check for key {key}: {e}")
            # Fallback to local bucket on Redis error
            allowed, remaining = self._check_local_limit(key, limit, period, cost)
            reset_time = int(time.time() + period)
            return allowed, {
                'limit': limit,
                'remaining': remaining,
                'reset': reset_time,
                'cost': cost
            }
        except Exception as e:
             logger.exception(f"Unexpected error during rate limit check for key {key}: {e}")
             # Fail open (allow) or closed (block)? Let's fail open for now.
             return True, {
                 'limit': limit,
                 'remaining': limit, # Assume full
                 'reset': int(time.time()),
                 'cost': cost
            }

    # --- Added helper for local check ---
    def _check_local_limit(self, key: str, limit: int, period: int, cost: int) -> Tuple[bool, int]:
        """Checks rate limit using local token bucket (synchronous)."""
        config = TokenBucketConfig(
            capacity=limit,
            refill_rate=float(limit) / period,
            refill_time=1.0  # Refill check happens on consume
        )
        bucket = self._get_bucket(key, config) # Uses synchronous _get_bucket
        allowed = bucket.consume(cost) # Uses synchronous consume
        remaining = int(bucket.tokens)
        return allowed, remaining

global_rate_limiter = RateLimiter()
# --- Decorator ---
def rate_limit(
    limit: int = 100,
    period: int = 60,
    key_func: Optional[callable] = None,
    cost_func: Optional[callable] = None,
    distributed: bool = True
):
    def decorator(f):
        is_async = asyncio.iscoroutinefunction(f)

        if is_async:
            @wraps(f)
            async def wrapped(*args, **kwargs):
                key = _get_request_key(key_func)
                cost = cost_func(request) if cost_func else 1

                # Use the global instance
                allowed, limit_info = global_rate_limiter.check_rate_limit( # <--- CHANGED
                    key=key, limit=limit, period=period, cost=cost, distributed=distributed
                )

                if not allowed:
                    return _rate_limit_exceeded_response(limit_info)
                response = await f(*args, **kwargs)
                return _add_rate_limit_headers(response, limit_info)
        else:
             @wraps(f)
             def wrapped(*args, **kwargs):
                key = _get_request_key(key_func)
                cost = cost_func(request) if cost_func else 1

                # Use the global instance
                allowed, limit_info = global_rate_limiter.check_rate_limit( # <--- CHANGED
                    key=key, limit=limit, period=period, cost=cost, distributed=distributed
                )

                if not allowed:
                    return _rate_limit_exceeded_response(limit_info)
                response = f(*args, **kwargs)
                return _add_rate_limit_headers(response, limit_info)

        return wrapped
    return decorator

# --- Helper functions for decorator ---
def _get_request_key(key_func: Optional[callable]) -> str:
    """Helper to get the rate limit key."""
    if key_func:
        return key_func(request)
    else:
        # Default to IP + endpoint + user_id if available
        key_parts = [
            request.remote_addr or 'unknown_ip', # Handle missing remote_addr
            request.endpoint or 'unknown_endpoint', # Handle missing endpoint
            str(getattr(g, 'user_id', 'anonymous')) # Handle missing user_id
        ]
        return ':'.join(filter(None, key_parts))

def _rate_limit_exceeded_response(limit_info: Dict) -> Tuple[str, int]:
    """Helper to create the 429 response."""
    retry_after = max(1, limit_info['reset'] - int(time.time()))
    response_data = jsonify({
        "error": "Rate limit exceeded",
        "limit": limit_info['limit'],
        "remaining": limit_info['remaining'],
        "reset": limit_info['reset'],
        "retry_after": retry_after
    })
    response = current_app.make_response(response_data)
    response.status_code = 429
    response.headers['Retry-After'] = str(retry_after)
    # Also add standard rate limit headers to the 429 response
    response.headers['X-RateLimit-Limit'] = str(limit_info['limit'])
    response.headers['X-RateLimit-Remaining'] = str(limit_info['remaining'])
    response.headers['X-RateLimit-Reset'] = str(limit_info['reset'])
    return response


def _add_rate_limit_headers(response, limit_info: Dict):
    """Helper to add rate limit headers to a response."""
    # Ensure response is a Flask Response object
    if not isinstance(response, current_app.response_class):
         # Handle cases where view returns data directly, or tuple (data, status), etc.
         response = current_app.make_response(response)

    try:
        response.headers.set("X-RateLimit-Limit", str(limit_info['limit']))
        response.headers.set("X-RateLimit-Remaining", str(limit_info['remaining']))
        response.headers.set("X-RateLimit-Reset", str(limit_info['reset']))
    except Exception as e:
        # Handle cases where response might not be mutable or headers dict missing
        logger.error(f"Failed to add rate limit headers: {e}", exc_info=True)

    return response


# --- Made Synchronous ---
def get_rate_limit_stats(key_pattern: str = "*") -> Dict:
    """Get rate limiting statistics (synchronous)."""
    try:
        redis_client = rate_limiter._get_redis() # Uses synchronous _get_redis
        if not redis_client:
            return {"error": "Redis unavailable"}

        stats = {}
        # Use decode_responses=True in client for automatic decoding
        for key in redis_client.scan_iter(f"analytics:{key_pattern}"):
            key_stats = redis_client.hgetall(key)
            # Keys and values are already strings if decode_responses=True
            stats[key] = {k: int(v) for k, v in key_stats.items()}

        return stats
    except redis.RedisError as e:
        logger.error(f"Error getting rate limit stats: {e}")
        return {}
    except Exception as e:
        logger.exception(f"Unexpected error getting rate limit stats: {e}")
        return {}

# --- IPBlockList ---
class IPBlockList:
    """IP address blocklist (synchronous)."""
    def __init__(self):
        self.blocked_ips: Set[str] = set() # Use Set for efficiency
        # self.redis_client = None # Use _get_redis
        self.redis_key_prefix = "ip_blocklist:"
        self.lock = threading.Lock() # Lock for modifying local set

    # --- Made Synchronous ---
    def _get_redis(self) -> Optional[redis.Redis]:
        """Get or create Redis connection (synchronous)."""
        try:
            # Use a different key for g to avoid conflicts if needed
            if not hasattr(g, 'ip_block_redis_conn'):
                # Use os.environ directly to get REDIS_URL as a fallback
                redis_url = current_app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
                g.ip_block_redis_conn = redis.from_url(redis_url, decode_responses=True)
                g.ip_block_redis_conn.ping() # Test connection
                logger.debug(f"Redis connection established for IPBlockList using {redis_url}")
            return g.ip_block_redis_conn
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis for IPBlockList: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting Redis connection for IPBlockList: {e}")
            return None

    # --- Made Synchronous ---
    def is_blocked(self, ip_address: str, distributed: bool = True) -> bool:
        """Check if an IP address is blocked (synchronous)."""
        # Check local blocklist first (thread-safe access not strictly needed for read, but good practice)
        with self.lock:
            if ip_address in self.blocked_ips:
                logger.debug(f"IP {ip_address} found in local blocklist")
                return True

        # Check Redis if distributed mode is enabled
        if distributed:
            redis_client = self._get_redis()
            if redis_client:
                try:
                    key = f"{self.redis_key_prefix}{ip_address}"
                    exists = redis_client.exists(key)
                    if exists:
                         logger.debug(f"IP {ip_address} found in Redis blocklist")
                         return True
                except redis.RedisError as e:
                    logger.error(f"Redis error checking blocked IP {ip_address}: {e}")
                    # Fall back to local check only (already done)

        return False

    # --- Made Synchronous ---
    def block_ip(self, ip_address: str, reason: str = "Manual block",
                      duration: int = 86400, distributed: bool = True) -> bool:
        """Block an IP address (synchronous)."""
        # Add to local blocklist thread-safely
        with self.lock:
            self.blocked_ips.add(ip_address)

        # Add to Redis if distributed mode is enabled
        if distributed:
            redis_client = self._get_redis()
            if redis_client:
                try:
                    key = f"{self.redis_key_prefix}{ip_address}"
                    block_info = json.dumps({
                        "reason": reason,
                        "blocked_at": time.time()
                    })
                    redis_client.setex(key, duration, block_info)
                    logger.info(f"IP {ip_address} blocked in Redis for {duration}s: {reason}")
                    return True # Successfully blocked in Redis
                except redis.RedisError as e:
                    logger.error(f"Redis error blocking IP {ip_address}: {e}")
                    # Blocked locally anyway, but log Redis failure
                    return False # Indicate Redis block failed
            else:
                 logger.warning(f"Redis unavailable, IP {ip_address} blocked locally only.")
                 return False # Indicate Redis block failed

        logger.info(f"IP {ip_address} blocked locally only: {reason}")
        return True # Indicate local block succeeded

    # --- Made Synchronous ---
    def unblock_ip(self, ip_address: str, distributed: bool = True) -> bool:
        """Unblock an IP address (synchronous)."""
        # Remove from local blocklist thread-safely
        with self.lock:
            self.blocked_ips.discard(ip_address) # Use discard to avoid KeyError

        # Remove from Redis if distributed mode is enabled
        if distributed:
            redis_client = self._get_redis()
            if redis_client:
                try:
                    key = f"{self.redis_key_prefix}{ip_address}"
                    deleted_count = redis_client.delete(key)
                    if deleted_count > 0:
                        logger.info(f"IP {ip_address} unblocked in Redis")
                    else:
                        logger.debug(f"IP {ip_address} not found in Redis blocklist for unblocking.")
                    return True # Attempted Redis unblock
                except redis.RedisError as e:
                    logger.error(f"Redis error unblocking IP {ip_address}: {e}")
                    return False # Indicate Redis unblock failed
            else:
                logger.warning(f"Redis unavailable, IP {ip_address} unblocked locally only.")
                return False # Indicate Redis unblock failed

        logger.info(f"IP {ip_address} unblocked locally")
        return True # Indicate local unblock succeeded

    # --- Made Synchronous ---
    def get_blocked_ips(self, distributed: bool = True) -> Dict[str, Dict]:
        """Get all blocked IPs with metadata (synchronous)."""
        # Get local IPs first
        with self.lock:
            result = {ip: {"reason": "Local block", "blocked_at": 0} for ip in self.blocked_ips}

        # Get from Redis if distributed mode is enabled
        if distributed:
            redis_client = self._get_redis()
            if redis_client:
                try:
                    # Use a cursor to avoid blocking for large lists
                    cursor = '0'
                    while cursor != 0:
                        cursor, keys = redis_client.scan(cursor=cursor, match=f"{self.redis_key_prefix}*", count=100)
                        if keys:
                            # Get values for the scanned keys
                            values = redis_client.mget(keys)
                            for i, key in enumerate(keys):
                                ip = key.replace(self.redis_key_prefix, "")
                                data = values[i]
                                if data:
                                    try:
                                        info = json.loads(data) # Already decoded if decode_responses=True
                                        result[ip] = info
                                    except json.JSONDecodeError:
                                        result[ip] = {"reason": "Invalid data in Redis", "blocked_at": 0}
                                else:
                                     # Key might have expired between scan and mget
                                     result.pop(ip, None) # Remove if only locally blocked previously

                except redis.RedisError as e:
                    logger.error(f"Redis error getting blocked IPs: {e}")
                    # Return local blocklist only in case of error during scan
            else:
                 logger.warning("Redis unavailable for get_blocked_ips, returning local list only.")

        return result

# Create global instance
ip_block_list = IPBlockList()

# --- Middleware ---
def ip_block_middleware():
    """
    Middleware function for IP blocking (now synchronous).

    Usage:
        # In app setup:
        app.before_request(ip_block_middleware)
    """
    # This function runs synchronously before each request
    client_ip = request.remote_addr
    if client_ip:  # Ensure we have an IP address
        # Call the synchronous is_blocked method directly - no asyncio needed
        if ip_block_list.is_blocked(client_ip):
            logger.warning(f"Blocked request from IP: {client_ip}")
            # Use Flask's abort to stop request processing
            abort(403, description="Your IP address has been blocked.")
    else:
        logger.warning("Could not determine client IP address for blocking check.")

# Example usage in Flask app setup:
# from flask import Flask
# app = Flask(__name__)
# app.config['REDIS_URL'] = 'redis://localhost:6379/1' # Example config
#
# # IMPORTANT: Apply middleware *before* routes are defined or processed
# app.before_request(ip_block_middleware)
#
# @app.route('/')
# def index():
#     # If execution reaches here, the IP was not blocked
#     return "Welcome!"
#
# # Example route using rate limiting
# @app.route('/limited')
# @rate_limit(limit=5, period=60) # 5 requests per minute per IP/endpoint/user
# def limited_route():
#      user_id = getattr(g, 'user_id', 'test_user') # Example: Get user ID if set by auth middleware
#      return f"This is a rate-limited route. User: {user_id}"
#
# if __name__ == '__main__':
#     # Run with eventlet (or gevent)
#     import eventlet
#     eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
