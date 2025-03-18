"""
Enhanced error handling utilities with circuit breaker pattern and error aggregation.
"""

import time
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening circuit
    reset_timeout: int = 60  # Seconds to wait before attempting reset
    half_open_timeout: int = 30  # Seconds to wait in half-open state

class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, or half-open
        self.lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        with self.lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if (time.time() - self.last_failure_time) > self.config.reset_timeout:
                    self.state = "half-open"
                    return True
                return False
            else:  # half-open
                return True
    
    def record_success(self):
        """Record a successful execution."""
        with self.lock:
            self.failures = 0
            self.state = "closed"
    
    def record_failure(self):
        """Record a failed execution."""
        with self.lock:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.config.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker {self.name} opened after {self.failures} failures")

class ErrorAggregator:
    """Aggregates and tracks error patterns."""
    
    def __init__(self):
        self.error_counts = defaultdict(lambda: defaultdict(int))
        self.error_samples = defaultdict(list)
        self.last_reset = time.time()
        self.reset_interval = 3600  # 1 hour
        self.lock = threading.Lock()
    
    def record_error(self, error_type: str, error_message: str, context: Dict = None):
        """Record an error occurrence."""
        with self.lock:
            current_time = time.time()
            
            # Reset counters if interval has passed
            if current_time - self.last_reset > self.reset_interval:
                self.error_counts.clear()
                self.error_samples.clear()
                self.last_reset = current_time
            
            # Update counts
            self.error_counts[error_type]["total"] += 1
            
            # Store error sample with timestamp
            self.error_samples[error_type].append({
                "timestamp": datetime.now().isoformat(),
                "message": error_message,
                "context": context
            })
            
            # Keep only last 10 samples per error type
            if len(self.error_samples[error_type]) > 10:
                self.error_samples[error_type].pop(0)
    
    def get_error_stats(self) -> Dict:
        """Get error statistics."""
        with self.lock:
            return {
                "counts": dict(self.error_counts),
                "samples": dict(self.error_samples),
                "last_reset": datetime.fromtimestamp(self.last_reset).isoformat()
            }

# Global instances
circuit_breakers: Dict[str, CircuitBreaker] = {}
error_aggregator = ErrorAggregator()

def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker pattern to a function."""
    def decorator(func: Callable):
        if name not in circuit_breakers:
            circuit_breakers[name] = CircuitBreaker(name, config)
        breaker = circuit_breakers[name]
        
        async def wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise Exception(f"Circuit breaker {name} is open")
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                error_aggregator.record_error(
                    type(e).__name__,
                    str(e),
                    {"service": name, "args": args, "kwargs": kwargs}
                )
                raise
        
        return wrapper
    return decorator

def track_error(error_type: str, context: Dict = None):
    """Decorator to track errors in the aggregator."""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_aggregator.record_error(
                    error_type,
                    str(e),
                    context or {"args": args, "kwargs": kwargs}
                )
                raise
        return wrapper
    return decorator

def get_error_stats() -> Dict:
    """Get current error statistics."""
    return error_aggregator.get_error_stats()

def get_circuit_breaker_states() -> Dict:
    """Get current states of all circuit breakers."""
    return {
        name: {
            "state": breaker.state,
            "failures": breaker.failures,
            "last_failure": datetime.fromtimestamp(breaker.last_failure_time).isoformat() if breaker.last_failure_time else None
        }
        for name, breaker in circuit_breakers.items()
    } 