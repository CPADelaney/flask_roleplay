"""
Enhanced rate limiting middleware with token bucket algorithm and user-based limits.
"""

import time
import threading
import logging
import asyncio
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
    
    async def consume(self, tokens: int = 1) -> bool:
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
    
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if not hasattr(g, 'rate_limit_redis'):
            redis_url = current_app.config.get('REDIS_URL', 'redis://localhost:6379/0')
            g.rate_limit_redis = redis.from_url(redis_url)
        return g.rate_limit_redis
    
    async def _get_bucket(self, key: str, config: TokenBucketConfig) -> TokenBucket:
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
                redis_client = await self._get_redis()
                
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
                bucket = await self._get_bucket(key, config)
                allowed = await bucket.consume(cost)
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
        redis_client = await rate_limiter._get_redis()
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
class IPBlockList:
    """IP address blocklist for explicitly blocking certain IP addresses."""
    
    def __init__(self):
        self.blocked_ips = set()  # Local in-memory blocklist
        self.redis_client = None
        self.redis_key_prefix = "ip_blocklist:"
    
    async def _get_redis(self) -> Optional[redis.Redis]:
        """Get or create Redis connection."""
        try:
            if not hasattr(g, 'ip_block_redis'):
                redis_url = current_app.config.get('REDIS_URL', 'redis://localhost:6379/0')
                g.ip_block_redis = redis.from_url(redis_url)
            return g.ip_block_redis
        except Exception as e:
            logger.error(f"Redis connection error in IPBlockList: {e}")
            return None
    
    async def is_blocked(self, ip_address: str, distributed: bool = True) -> bool:
        """
        Check if an IP address is blocked.
        
        Args:
            ip_address: The IP address to check
            distributed: Whether to use Redis for distributed blocking
            
        Returns:
            bool: True if IP is blocked, False otherwise
        """
        # Check local blocklist first
        if ip_address in self.blocked_ips:
            logger.debug(f"IP {ip_address} found in local blocklist")
            return True
        
        # Check Redis if distributed mode is enabled
        if distributed:
            try:
                redis_client = await self._get_redis()
                if redis_client:
                    key = f"{self.redis_key_prefix}{ip_address}"
                    return bool(redis_client.exists(key))
            except redis.RedisError as e:
                logger.error(f"Redis error checking blocked IP: {e}")
                # Fall back to local check only
        
        return False
    
    async def block_ip(self, ip_address: str, reason: str = "Manual block", 
                      duration: int = 86400, distributed: bool = True) -> bool:
        """
        Block an IP address.
        
        Args:
            ip_address: The IP address to block
            reason: Reason for blocking
            duration: Duration of block in seconds (default: 24 hours)
            distributed: Whether to use Redis for distributed blocking
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Always add to local blocklist
        self.blocked_ips.add(ip_address)
        
        # Add to Redis if distributed mode is enabled
        if distributed:
            try:
                redis_client = await self._get_redis()
                if redis_client:
                    key = f"{self.redis_key_prefix}{ip_address}"
                    redis_client.setex(
                        key,
                        duration,
                        json.dumps({
                            "reason": reason,
                            "blocked_at": time.time()
                        })
                    )
                    logger.info(f"IP {ip_address} blocked in Redis for {duration}s: {reason}")
                    return True
            except redis.RedisError as e:
                logger.error(f"Redis error blocking IP: {e}")
                # Continue with local block only
        
        logger.info(f"IP {ip_address} blocked locally: {reason}")
        return True
    
    async def unblock_ip(self, ip_address: str, distributed: bool = True) -> bool:
        """
        Unblock an IP address.
        
        Args:
            ip_address: The IP address to unblock
            distributed: Whether to use Redis for distributed blocking
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Remove from local blocklist
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
        
        # Remove from Redis if distributed mode is enabled
        if distributed:
            try:
                redis_client = await self._get_redis()
                if redis_client:
                    key = f"{self.redis_key_prefix}{ip_address}"
                    redis_client.delete(key)
                    logger.info(f"IP {ip_address} unblocked in Redis")
                    return True
            except redis.RedisError as e:
                logger.error(f"Redis error unblocking IP: {e}")
                # Continue with local unblock only
        
        logger.info(f"IP {ip_address} unblocked locally")
        return True
    
    async def get_blocked_ips(self, distributed: bool = True) -> Dict[str, Dict]:
        """
        Get all blocked IPs with metadata.
        
        Args:
            distributed: Whether to use Redis for distributed blocking
            
        Returns:
            Dict: Dictionary of blocked IPs with metadata
        """
        result = {ip: {"reason": "Local block", "blocked_at": 0} for ip in self.blocked_ips}
        
        # Get from Redis if distributed mode is enabled
        if distributed:
            try:
                redis_client = await self._get_redis()
                if redis_client:
                    for key in redis_client.scan_iter(f"{self.redis_key_prefix}*"):
                        ip = key.decode('utf-8').replace(self.redis_key_prefix, "")
                        data = redis_client.get(key)
                        if data:
                            try:
                                info = json.loads(data.decode('utf-8'))
                                result[ip] = info
                            except json.JSONDecodeError:
                                result[ip] = {"reason": "Unknown", "blocked_at": 0}
            except redis.RedisError as e:
                logger.error(f"Redis error getting blocked IPs: {e}")
                # Return local blocklist only
        
        return result

def ip_block_middleware():
    """
    Middleware function for IP blocking.
    
    Usage:
        # In app setup:
        app.before_request(ip_block_middleware)
    """
    async def check_ip():
        client_ip = request.remote_addr
        if await ip_block_list.is_blocked(client_ip):
            logger.warning(f"Blocked request from IP: {client_ip}")
            abort(403, "Your IP address has been blocked")
    
    # Run async function in current event loop
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(check_ip())
    else:
        asyncio.run(check_ip())

# Create global instance
ip_block_list = IPBlockList()
