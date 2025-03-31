"""
Enhanced rate limiting middleware with token bucket algorithm and user-based limits.
"""

import time
import threading
import logging
from functools import wraps
from flask import request, jsonify, current_app, g
from typing import Dict, Optional, Tuple
import redis
from dataclasses import dataclass

# Import database connections
from db.connection import get_db_connection_context

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TokenBucketConfig:
    """Configuration for token bucket rate limiter."""
    capacity: int  # Maximum number of tokens
    refill_rate: float  # Tokens per second
    refill_time: float  # Time between refills in seconds

class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, config: TokenBucketConfig):
        self.capacity = config.capacity
        self.tokens = config.capacity
        self.refill_rate = config.refill_rate
        self.refill_time = config.refill_time
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed >= self.refill_time:
            refill_amount = int(elapsed * self.refill_rate)
            self.tokens = min(self.capacity, self.tokens + refill_amount)
            self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
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
        self.redis_client = None
        self.local_buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()
    
    def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if not hasattr(g, 'rate_limit_redis'):
            redis_url = current_app.config.get('REDIS_URL', 'redis://localhost:6379/0')
            g.rate_limit_redis = redis.from_url(redis_url)
        return g.rate_limit_redis
    
    def _get_bucket(self, key: str, config: TokenBucketConfig) -> TokenBucket:
        """Get or create a token bucket."""
        with self.lock:
            if key not in self.local_buckets:
                self.local_buckets[key] = TokenBucket(config)
            return self.local_buckets[key]
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        period: int,
        cost: int = 1,
        distributed: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Check if the rate limit is exceeded.
        
        Args:
            key: Rate limit key
            limit: Maximum number of requests
            period: Time period in seconds
            cost: Cost of the current request in tokens
            distributed: Whether to use distributed Redis-based limiting
            
        Returns:
            Tuple of (is_allowed, limit_info)
        """
        try:
            if distributed:
                # Use Redis-based distributed rate limiting
                redis_client = self._get_redis()
                
                # Use token bucket algorithm with Redis
                bucket_key = f"bucket:{key}"
                
                # Use Redis pipeline for atomic operations
                pipe = redis_client.pipeline()
                
                # Get current tokens and last refill time
                pipe.hgetall(bucket_key)
                current = pipe.execute()[0]
                
                now = time.time()
                tokens = float(current.get(b'tokens', limit))
                last_refill = float(current.get(b'last_refill', now))
                
                # Calculate refill
                elapsed = now - last_refill
                tokens = min(limit, tokens + (elapsed * limit / period))
                
                # Try to consume tokens
                if tokens >= cost:
                    # Update bucket
                    pipe.hmset(bucket_key, {
                        'tokens': tokens - cost,
                        'last_refill': now
                    })
                    pipe.expire(bucket_key, period)
                    pipe.execute()
                    
                    allowed = True
                else:
                    allowed = False
                
                # Get remaining tokens
                remaining = max(0, int(tokens - cost if allowed else tokens))
                
                # Update analytics
                analytics_key = f"analytics:{key}"
                pipe.hincrby(analytics_key, 'total_requests', 1)
                if not allowed:
                    pipe.hincrby(analytics_key, 'blocked_requests', 1)
                pipe.execute()
                
            else:
                # Use local token bucket
                config = TokenBucketConfig(
                    capacity=limit,
                    refill_rate=limit / period,
                    refill_time=1.0  # Refill every second
                )
                bucket = self._get_bucket(key, config)
                allowed = bucket.consume(cost)
                remaining = bucket.tokens
            
            reset_time = int(time.time() + period)
            
            return allowed, {
                'limit': limit,
                'remaining': remaining,
                'reset': reset_time,
                'cost': cost
            }
            
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiting: {e}")
            # Fallback to local bucket on Redis error
            return await self.check_rate_limit(key, limit, period, cost, distributed=False)

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(
    limit: int = 100,
    period: int = 60,
    key_func: Optional[callable] = None,
    cost_func: Optional[callable] = None,
    distributed: bool = True
):
    """
    Enhanced rate limiting decorator.
    
    Args:
        limit: Maximum number of requests per period
        period: Time period in seconds
        key_func: Function to generate rate limit key
        cost_func: Function to calculate request cost
        distributed: Whether to use distributed Redis-based limiting
    """
    def decorator(f):
        @wraps(f)
        async def wrapped(*args, **kwargs):
            # Get rate limit key
            if key_func:
                key = key_func(request)
            else:
                # Default to IP + endpoint + user_id if available
                key_parts = [
                    request.remote_addr,
                    request.endpoint,
                    str(getattr(g, 'user_id', ''))
                ]
                key = ':'.join(filter(None, key_parts))
            
            # Get request cost
            cost = cost_func(request) if cost_func else 1
            
            # Check rate limit
            allowed, limit_info = await rate_limiter.check_rate_limit(
                key=key,
                limit=limit,
                period=period,
                cost=cost,
                distributed=distributed
            )
            
            if not allowed:
                response = jsonify({
                    "error": "Rate limit exceeded",
                    "limit": limit_info['limit'],
                    "remaining": limit_info['remaining'],
                    "reset": limit_info['reset'],
                    "retry_after": limit_info['reset'] - int(time.time())
                })
                response.status_code = 429
                return response
            
            # Add rate limit headers
            response = await f(*args, **kwargs)
            
            # Check if response is a tuple (response, status_code)
            if isinstance(response, tuple):
                resp, code = response
                resp.headers.extend({
                    "X-RateLimit-Limit": str(limit_info['limit']),
                    "X-RateLimit-Remaining": str(limit_info['remaining']),
                    "X-RateLimit-Reset": str(limit_info['reset'])
                })
                return resp, code
            else:
                response.headers.extend({
                    "X-RateLimit-Limit": str(limit_info['limit']),
                    "X-RateLimit-Remaining": str(limit_info['remaining']),
                    "X-RateLimit-Reset": str(limit_info['reset'])
                })
                return response
                
        return wrapped
    return decorator

async def get_rate_limit_stats(key_pattern: str = "*") -> Dict:
    """Get rate limiting statistics."""
    try:
        redis_client = rate_limiter._get_redis()
        stats = {}
        
        # Get all analytics keys
        for key in redis_client.scan_iter(f"analytics:{key_pattern}"):
            key_stats = redis_client.hgetall(key)
            stats[key.decode('utf-8')] = {
                k.decode('utf-8'): int(v)
                for k, v in key_stats.items()
            }
        
        return stats
    except redis.RedisError as e:
        logger.error(f"Error getting rate limit stats: {e}")
        return {}
