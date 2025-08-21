# nyx/governance/core.py
"""
Core governance class that combines all mixins.
"""
import logging
import asyncio
import uuid
import yaml
import os
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING, Protocol, Tuple
from datetime import datetime, timedelta
from contextvars import ContextVar
from functools import wraps
from enum import Enum
import importlib
import importlib.resources
from collections import defaultdict

# Remove direct OpenAI import - use chatgpt_integration instead
# from openai import AsyncOpenAI
# from openai.types.beta.assistant import Assistant

try:
    from openai import BadRequestError
    OPENAI_ERRORS_AVAILABLE = True
except ImportError:
    OPENAI_ERRORS_AVAILABLE = False
    # Create a fallback BadRequestError class
    class BadRequestError(Exception):
        """Fallback BadRequestError when OpenAI is not available."""
        pass

# Helper functions for OpenAI integration
RESPONSES_ALLOWED_KEYS = {
    "model", "name", "instructions", "temperature", "top_p", 
    "max_tokens", "tools", "custom_capabilities"
}

def _is_unsupported_model(error: Exception) -> bool:
    """Check if the error indicates an unsupported model."""
    error_msg = str(error).lower()
    unsupported_phrases = [
        "model not supported",
        "unsupported model", 
        "invalid model",
        "does not exist",
        "not available for assistants",
        "gpt-5"  # gpt-5 models might not be supported yet
    ]
    return any(phrase in error_msg for phrase in unsupported_phrases)

def _filter_responses_kwargs(kwargs: dict) -> dict:
    """Filter kwargs to only include valid Responses API parameters."""
    filtered = {}
    for key in RESPONSES_ALLOWED_KEYS:
        if key in kwargs:
            filtered[key] = kwargs[key]
    return filtered

class ResponsesAssistant:
    """Shim class to provide Assistant-like interface for Responses API fallback."""
    def __init__(self, id: str, name: str, model: str, 
                 instructions: str = "", tools: list = None, 
                 _custom_params: dict = None):
        self.id = id
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools or []
        self._custom_params = _custom_params or {}
        self.created_at = datetime.now().isoformat()

# Import metrics (assuming prometheus_client is available)
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    # Dummy implementations
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self

# Import all the mixins
from .story import StoryGovernanceMixin
from .npc import NPCGovernanceMixin
from .conflict import ConflictGovernanceMixin
from .world import WorldGovernanceMixin
from .agents import AgentGovernanceMixin
from .player import PlayerGovernanceMixin
from .constants import DirectiveType, DirectivePriority, AgentType

# Other imports
from utils.caching import CACHE_TTL
from db.connection import get_db_connection_context

# Import LLM access from chatgpt_integration

# Try to import Assistant type for type hints
try:
    from openai.types.beta.assistant import Assistant
    ASSISTANT_TYPE_AVAILABLE = True
except ImportError:
    ASSISTANT_TYPE_AVAILABLE = False
    Assistant = Any  # Fallback type

# Pydantic for validation
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

if TYPE_CHECKING:
    from lore.core.lore_system import LoreSystem

# Context variable for request tracing
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

# Metrics
governor_init_duration = Histogram(
    'nyx_governor_init_duration_seconds',
    'Time spent initializing governor'
)
governor_agents_registered = Counter(
    'nyx_governor_agents_registered_total',
    'Total agents registered',
    ['agent_type']
)
governor_resource_violations = Counter(
    'nyx_governor_resource_limit_violations_total',
    'Resource limit violations',
    ['resource_type', 'violation_type']
)
governor_active_agents = Gauge(
    'nyx_governor_active_agents',
    'Currently active agents',
    ['agent_type']
)

# Valid OpenAI Assistant parameters
VALID_OPENAI_PARAMS = {
    "model", "name", "description", "instructions", "tools",
    "file_ids", "metadata", "temperature", "top_p",
    "response_format", "tool_resources"
}

# Protocol for type safety
class SupportsInitialize(Protocol):
    async def initialize(self) -> None: ...

class SupportsExecute(Protocol):
    async def execute(self, directive: Dict[str, Any]) -> Any: ...


# ===== EXCEPTIONS =====
class GovernanceError(Exception):
    """Base exception for all governance-related errors."""
    pass


class PlayerNotFoundError(GovernanceError):
    """Raised when player information is required but not available."""
    pass


class ResourceLimitError(GovernanceError):
    """Raised when resource limits are exceeded."""
    def __init__(self, message: str, resource_type: str, current_usage: float, limit: float):
        super().__init__(message)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class AgentRegistrationError(GovernanceError):
    """Raised when agent registration fails."""
    pass


class ConfigurationError(GovernanceError):
    """Raised when configuration is invalid."""
    pass


# ===== CONFIGURATION MODELS =====
if PYDANTIC_AVAILABLE:
    class ResourceLimit(BaseModel):
        limit: float = Field(..., gt=0)
        soft_limit: Optional[float] = None
        warning_threshold: float = Field(0.8, ge=0, le=1)
        description: Optional[str] = None
        agent_type: Optional[str] = None
        
        @validator('soft_limit')
        def validate_soft_limit(cls, v, values):
            if v is not None and v > values.get('limit', 0):
                raise ValueError('Soft limit cannot exceed hard limit')
            return v
        
        @validator('warning_threshold')
        def validate_warning_threshold(cls, v, values):
            soft_limit = values.get('soft_limit')
            limit = values.get('limit')
            if soft_limit is not None and limit is not None:
                min_threshold = soft_limit / limit
                if v < min_threshold:
                    raise ValueError(
                        f'Warning threshold ({v}) must be >= soft_limit/limit ({min_threshold:.2f})'
                    )
            return v

    class ResourceConfig(BaseModel):
        resource_limits: Dict[str, Union[ResourceLimit, Dict[str, ResourceLimit]]] = Field(default_factory=dict)
        cache_ttl_minutes: float = Field(5.0, gt=0)
        
    def validate_config(config_dict: Dict[str, Any]) -> ResourceConfig:
        """Validate configuration dictionary."""
        return ResourceConfig(**config_dict)
else:
    def validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback validation without pydantic."""
        logger.warning(
            "Pydantic not available - skipping schema validation. "
            "Install pydantic for configuration validation."
        )
        return config_dict


# ===== LOGGING HELPERS =====
class ContextualLogger:
    """Logger that includes request ID in all messages for tracing."""
    def __init__(self, logger):
        self.logger = logger
    
    def _log(self, level, msg, *args, exc_info=None, **kwargs):
        request_id = request_id_var.get()
        if request_id:
            msg = f"[{request_id}] {msg}"
        
        log_method = getattr(self.logger, level)
        if exc_info and level == 'error':
            # Use exception logging for better stack traces
            self.logger.exception(msg, *args, **kwargs)
        else:
            log_method(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self._log('debug', msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self._log('info', msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._log('warning', msg, *args, **kwargs)
    
    def error(self, msg, *args, exc_info=True, **kwargs):
        self._log('error', msg, *args, exc_info=exc_info, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self._log('critical', msg, *args, **kwargs)

logger = ContextualLogger(logging.getLogger(__name__))


def with_request_id(func):
    """Decorator to ensure request ID is set for tracing."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        current_id = request_id_var.get()
        new_id = None
        
        try:
            if current_id is None:
                new_id = str(uuid.uuid4())[:8]
                request_id_var.set(new_id)
            
            return await func(*args, **kwargs)
        finally:
            # Clean up if we set a new ID
            if new_id is not None:
                request_id_var.set(None)
    
    return wrapper


# ===== RESOURCE LIMIT MANAGER =====
class ResourceLimitManager:
    """
    Manages resource limits for the governance system.
    Supports both YAML configuration and database-backed limits with caching.
    """
    
    # Single-flight lock for initialization
    _init_lock = asyncio.Lock()
    _initialized_tables = False
    
    def __init__(self, config_path: Optional[str] = None):
        # Use importlib.resources for package-safe path
        if config_path:
            self.config_path = config_path
        else:
            # Try environment variable first
            self.config_path = os.environ.get('NYX_RESOURCE_LIMITS_PATH')
            if not self.config_path:
                # Fall back to /etc/nyx if exists
                etc_path = '/etc/nyx/resource_limits.yaml'
                if os.path.exists(etc_path):
                    self.config_path = etc_path
                else:
                    # Use package default as template
                    try:
                        if hasattr(importlib.resources, 'files'):
                            # Python 3.9+
                            self.config_path = str(
                                importlib.resources.files('nyx.governance').joinpath('resource_limits.yaml')
                            )
                        else:
                            # Python 3.7-3.8
                            with importlib.resources.path('nyx.governance', 'resource_limits.yaml') as p:
                                self.config_path = str(p)
                    except Exception:
                        self.config_path = None
        
        # Cache with composite keys: (resource_type, agent_type)
        self._cache: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
        self._cache_timestamps: Dict[Tuple[str, Optional[str]], datetime] = {}
        self._cache_ttl = timedelta(minutes=5)  # Default, overridden by config
        self._lock = asyncio.Lock()
        self._initialized = False
        self._registered_agent_keys: Set[Tuple[str, str]] = set()
        
        # Configuration
        self._config: Optional[ResourceConfig] = None
        
        # Default limits if no config exists
        self._defaults = {
            "compute": {"limit": 1000, "soft_limit": 800, "warning_threshold": 0.8},
            "memory": {"limit": 500, "soft_limit": 400, "warning_threshold": 0.75},
            "api_calls": {"limit": 100, "soft_limit": 80, "warning_threshold": 0.7},
            "concurrent_agents": {"limit": 50, "soft_limit": 40, "warning_threshold": 0.8},
        }
        
        # Check if config path is writable
        if self.config_path and os.path.exists(self.config_path):
            try:
                # Test write access
                test_path = f"{self.config_path}.tmp"
                with open(test_path, 'w') as f:
                    f.write("")
                os.remove(test_path)
            except (IOError, OSError):
                logger.warning(
                    f"Config path '{self.config_path}' is not writable. "
                    f"Consider using /etc/nyx/resource_limits.yaml or setting NYX_RESOURCE_LIMITS_PATH"
                )
        
    async def initialize(self):
        """Initialize the resource manager and ensure database tables exist."""
        if self._initialized:
            return
            
        logger.info("Initializing resource limit manager")
        
        # Single-flight protection for table creation
        async with self._init_lock:
            if not self._initialized_tables:
                await self._ensure_tables()
                self._initialized_tables = True
        
        # Load initial configuration
        await self._load_configuration()
        
        self._initialized = True
        logger.info("Resource limit manager initialized")
    
    async def _ensure_tables(self):
        """Ensure resource limit tables exist in the database."""
        # Skip DDL for unit tests
        if os.environ.get('NYX_SKIP_DDL', '').lower() in ('1', 'true', 'yes'):
            logger.info("Skipping DDL creation (NYX_SKIP_DDL is set)")
            return
            
        try:
            async with get_db_connection_context() as conn:
                # Use transaction for atomicity
                async with conn.transaction():
                    # Create resource limits table if it doesn't exist
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS ResourceLimits (
                            id SERIAL PRIMARY KEY,
                            resource_type VARCHAR(100) NOT NULL,
                            agent_type VARCHAR(100),
                            limit_name VARCHAR(100) NOT NULL,
                            limit_value FLOAT NOT NULL,
                            soft_limit FLOAT,
                            warning_threshold FLOAT DEFAULT 0.8,
                            description TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(resource_type, agent_type, limit_name)
                        )
                    """)
                    
                    # Create resource usage tracking table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS ResourceUsage (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER NOT NULL,
                            conversation_id INTEGER NOT NULL,
                            resource_type VARCHAR(100) NOT NULL,
                            agent_type VARCHAR(100),
                            usage_value FLOAT NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB
                        )
                    """)
                    
                    # Create index separately
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_resource_usage_lookup 
                        ON ResourceUsage(user_id, conversation_id, resource_type, timestamp DESC)
                    """)
                    
                # Seed with defaults if empty
                count = await conn.fetchval("SELECT COUNT(*) FROM ResourceLimits")
                if count == 0:
                    for resource_type, config in self._defaults.items():
                        await conn.execute("""
                            INSERT INTO ResourceLimits 
                            (resource_type, limit_name, limit_value, soft_limit, warning_threshold)
                            VALUES ($1, $2, $3, $4, $5)
                        """, resource_type, 'default', config['limit'], 
                            config['soft_limit'], config['warning_threshold'])
                        
        except Exception as e:
            logger.error(f"Error ensuring resource tables", exc_info=True)
            # Continue anyway - use in-memory defaults
    
    async def _load_configuration(self):
        """Load and validate configuration from YAML file or database."""
        # Try to load from YAML first if it exists
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    raw_config = yaml.safe_load(f)
                    
                if raw_config:
                    # Validate configuration
                    try:
                        self._config = validate_config(raw_config)
                        # Update cache TTL from config
                        self._cache_ttl = timedelta(minutes=self._config.cache_ttl_minutes)
                        
                        # Process validated limits
                        for resource_type, limits in self._config.resource_limits.items():
                            if isinstance(limits, dict):
                                for agent_type, limit_config in limits.items():
                                    cache_key = (resource_type, agent_type)
                                    self._cache[cache_key] = limit_config.dict() if hasattr(limit_config, 'dict') else limit_config
                                    self._cache_timestamps[cache_key] = datetime.now()
                            else:
                                cache_key = (resource_type, None)
                                self._cache[cache_key] = limits.dict() if hasattr(limits, 'dict') else limits
                                self._cache_timestamps[cache_key] = datetime.now()
                                
                    except Exception as e:
                        logger.error(f"Invalid configuration schema", exc_info=True)
                        raise ConfigurationError(f"Invalid resource limits configuration: {str(e)}")
                        
            except Exception as e:
                logger.warning(f"Could not load YAML config: {e}")
        
        # Load from database and override (this also populates cache)
        await self._load_db_config()
    
    async def _load_db_config(self):
        """Load configuration from database."""
        # Skip DB loading if DDL is skipped (for tests)
        if os.environ.get('NYX_SKIP_DDL', '').lower() in ('1', 'true', 'yes'):
            logger.info("Skipping DB config load (NYX_SKIP_DDL is set)")
            # Use defaults
            for resource_type, config in self._defaults.items():
                cache_key = (resource_type, None)
                self._cache[cache_key] = config
                self._cache_timestamps[cache_key] = datetime.now()
            return
            
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("SELECT * FROM ResourceLimits")
                for row in rows:
                    resource_type = row['resource_type']
                    agent_type = row.get('agent_type')
                    cache_key = (resource_type, agent_type)
                    
                    self._cache[cache_key] = {
                        'limit': row['limit_value'],
                        'soft_limit': row.get('soft_limit', row['limit_value'] * 0.8),
                        'warning_threshold': row.get('warning_threshold', 0.8),
                        'description': row.get('description')
                    }
                    self._cache_timestamps[cache_key] = datetime.now()
                    
        except Exception as e:
            logger.warning(f"Could not load DB config", exc_info=True)
            # Use defaults if DB fails
            for resource_type, config in self._defaults.items():
                cache_key = (resource_type, None)
                self._cache[cache_key] = config
                self._cache_timestamps[cache_key] = datetime.now()
            
            # Also set default cache TTL if no config was loaded
            if not self._config:
                self._cache_ttl = timedelta(minutes=5)
    
    async def get_limit(self, resource_type: str, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Get resource limits with caching, checking agent-specific then global."""
        async with self._lock:
            # Check agent-specific limit first
            if agent_type:
                specific_key = (resource_type, agent_type)
                if specific_key in self._cache_timestamps:
                    age = datetime.now() - self._cache_timestamps[specific_key]
                    if age < self._cache_ttl:
                        return self._cache.get(specific_key, {})
            
            # Check global limit
            global_key = (resource_type, None)
            if global_key in self._cache_timestamps:
                age = datetime.now() - self._cache_timestamps[global_key]
                if age < self._cache_ttl:
                    return self._cache.get(global_key, self._defaults.get(resource_type, {}))
            
            # Reload from database if cache expired
            await self._load_db_config()
            
            # Try agent-specific first, then global, then default
            if agent_type and (resource_type, agent_type) in self._cache:
                return self._cache[(resource_type, agent_type)]
            elif (resource_type, None) in self._cache:
                return self._cache[(resource_type, None)]
            else:
                return self._defaults.get(resource_type, {})
    
    async def check_usage(self, resource_type: str, usage: float, 
                         agent_type: Optional[str] = None,
                         user_id: Optional[int] = None,
                         conversation_id: Optional[int] = None) -> bool:
        """Check if resource usage is within limits and track usage."""
        limits = await self.get_limit(resource_type, agent_type)
        
        if not limits:
            logger.warning(f"No limits defined for resource type: {resource_type}")
            return True
        
        hard_limit = limits.get('limit', float('inf'))
        soft_limit = limits.get('soft_limit', hard_limit * 0.8)
        warning_threshold = limits.get('warning_threshold', 0.8)
        
        # Track usage in database if user/conversation provided
        if user_id is not None and conversation_id is not None:
            try:
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        INSERT INTO ResourceUsage 
                        (user_id, conversation_id, resource_type, agent_type, usage_value)
                        VALUES ($1, $2, $3, $4, $5)
                    """, user_id, conversation_id, resource_type, agent_type, usage)
            except Exception as e:
                logger.error(f"Failed to track resource usage", exc_info=True)
        
        # Check limits and emit metrics
        if usage > hard_limit:
            governor_resource_violations.labels(
                resource_type=resource_type,
                violation_type='hard_limit'
            ).inc()
            raise ResourceLimitError(
                f"Hard limit exceeded for {resource_type}",
                resource_type, usage, hard_limit
            )
        
        if usage > soft_limit:
            governor_resource_violations.labels(
                resource_type=resource_type,
                violation_type='soft_limit'
            ).inc()
            logger.warning(
                f"Soft limit exceeded for {resource_type}: {usage}/{soft_limit}"
            )
        
        if usage > hard_limit * warning_threshold:
            logger.warning(
                f"Approaching limit for {resource_type}: {usage}/{hard_limit}"
            )
        
        return True


# ===== MAIN GOVERNOR CLASS =====
class NyxUnifiedGovernor(
    StoryGovernanceMixin,
    NPCGovernanceMixin,
    ConflictGovernanceMixin,
    WorldGovernanceMixin,
    AgentGovernanceMixin,
    PlayerGovernanceMixin
):
    """
    Enhanced unified governance system for Nyx to control all agents with agentic capabilities.
    
    This class combines all governance functionality through mixins while maintaining
    the same public API for backward compatibility.
    
    Usage:
        # Recommended: Use as async context manager
        async with NyxUnifiedGovernor(user_id, conversation_id, player_name="Alice") as governor:
            # Use governor...
            pass  # Automatically cleaned up
        
        # Alternative: Manual lifecycle
        governor = NyxUnifiedGovernor(user_id, conversation_id)
        await governor.initialize()  # Will auto-detect player name from DB or default to 'Chase'
        try:
            # Use governor...
        finally:
            await governor.shutdown()  # REQUIRED to prevent resource leaks
    
    This class provides:
      1. Central authority over all agents with enhanced coordination
      2. Goal-oriented agent coordination and planning
      3. Adaptive decision making with learning capabilities
      4. Performance monitoring and feedback loops
      5. Cross-agent communication and collaboration
      6. Dynamic goal prioritization and resource allocation
      7. Agent learning and adaptation tracking
      8. Enhanced conflict resolution with context awareness
      9. Temporal consistency enforcement
      10. User preference integration
    """
    _assistants: dict[str, Assistant] = {}
    _openai_client = None  # Cached client instance
    
    def __init__(self, user_id: int, conversation_id: int, player_name: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.player_name = player_name  # Can be set later via set_player_name()

        self._discovery_completed: bool = False 
        
        # Will be initialized in _initialize_systems() to avoid circular dependency
        self.lore_system: Optional[Any] = None

        # Core systems and state
        self.memory_system = None
        self.game_state = None
        self.registered_agents: Dict[str, Dict[str, Any]] = {}     # {agent_type: {agent_id: instance}}

        # Use async queue for agent registration to avoid lock starvation
        self._agent_registration_queue = asyncio.Queue()
        self._registration_task: Optional[asyncio.Task] = None
        
        # Concurrency protection
        self._state_lock = asyncio.Lock()
        self._resource_lock = asyncio.Lock()

        # Multi-agent analytics
        self.active_goals: List[Dict[str, Any]] = []
        self.agent_goals: Dict[str, Dict[str, Any]] = {}           # {agent_type: {agent_id: ...}}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}     # {agent_type: {agent_id: ...}}
        self.agent_learning: Dict[str, Dict[str, Any]] = {}        # {agent_type: {agent_id: ...}}
        self.coordination_history: List[Dict[str, Any]] = []
        self.memory_integration = None
        self.memory_graph = None
    
        # Learning state
        self.strategy_effectiveness: Dict[str, Any] = {}
        self.adaptation_patterns: Dict[str, Any] = {}
        self.collaboration_success: Dict[str, Dict[str, Any]] = {}

        # Disagreement history
        self.disagreement_history: Dict[str, List[Dict[str, Any]]] = {}
        self.disagreement_thresholds: Dict[str, float] = {
            "narrative_impact": 0.7,
            "character_consistency": 0.8,
            "world_integrity": 0.9,
            "player_experience": 0.6
        }

        # Directive/action reports
        self.directives: Dict[str, Dict[str, Any]] = {}
        self.action_reports: Dict[str, Dict[str, Any]] = {}
        
        # Resource limit manager
        self.resource_manager = ResourceLimitManager()
        
        # Agent counters for efficient metric updates
        self._agent_counts: Dict[str, int] = defaultdict(int)
        
        # Flag to track initialization
        self._initialized = False

    async def set_player_name(self, player_name: str):
        """
        Set the player name after initialization if not provided in constructor.
        This is now async to allow database persistence.
        """
        if not player_name:
            raise ValueError("Player name cannot be empty")
        self.player_name = player_name
        logger.info(f"Player name set to: {player_name}")
        
        # Store in database for persistence
        try:
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'PlayerName', $3)
                    ON CONFLICT (user_id, conversation_id, key) 
                    DO UPDATE SET value = EXCLUDED.value
                """, self.user_id, self.conversation_id, player_name)
        except Exception as e:
            logger.error(f"Failed to store player name in database", exc_info=True)
        
        # Store in database for persistence
        async def store_player_name():
            try:
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                        VALUES ($1, $2, 'PlayerName', $3)
                        ON CONFLICT (user_id, conversation_id, key) 
                        DO UPDATE SET value = EXCLUDED.value
                    """, self.user_id, self.conversation_id, player_name)
            except Exception as e:
                logger.error(f"Failed to store player name in database", exc_info=True)
        
        # Run the async storage in the background if we're in an event loop
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(store_player_name())
        except RuntimeError:
            # No event loop, skip database storage
            logger.warning("No event loop available to store player name in database")

    @with_request_id
    async def initialize(self) -> "NyxUnifiedGovernor":
        """
        Initialize the governance system asynchronously.
        This must be called after creating a new NyxUnifiedGovernor instance.
        """
        if hasattr(self, '_initialized') and self._initialized:
            logger.info("Governor already initialized, skipping")
            return self

        with governor_init_duration.time():
            logger.info(f"Initializing governor for user_id={self.user_id}, conversation_id={self.conversation_id}")
            
            # Initialize resource manager
            await self.resource_manager.initialize()
            
            # Start agent registration processor
            self._registration_task = asyncio.create_task(self._process_agent_registrations())
            
            # Initialize other systems
            await self._initialize_systems()
            self._initialized = True
            
            logger.info("Governor initialization complete")
        
        return self

    async def _process_agent_registrations(self):
        """Process agent registrations from queue to avoid lock contention."""
        while True:
            try:
                registration = await self._agent_registration_queue.get()
                if registration is None:  # Shutdown signal
                    break
                    
                agent_type, agent_id, agent_instance = registration
                
                # Fast check using set
                agent_key = (agent_type, agent_id)
                if agent_key in self._registered_agent_keys:
                    logger.debug(f"Agent {agent_type}/{agent_id} already registered, skipping")
                    continue
                
                # Perform the actual registration
                if agent_type not in self.registered_agents:
                    self.registered_agents[agent_type] = {}
                
                self.registered_agents[agent_type][agent_id] = agent_instance
                self._registered_agent_keys.add(agent_key)  # Add to set
                
                if agent_id in self.registered_agents[agent_type]:
                    logger.warning(f"Agent {agent_id} already registered, updating")
                
                self.registered_agents[agent_type][agent_id] = agent_instance
                
                # Initialize tracking structures
                for tracking_dict in [self.agent_goals, self.agent_performance, self.agent_learning]:
                    if agent_type not in tracking_dict:
                        tracking_dict[agent_type] = {}
                
                # Update metrics
                governor_agents_registered.labels(agent_type=agent_type).inc()
                
                # Update active agents gauge efficiently
                self._agent_counts[agent_type] = len(self.registered_agents[agent_type])
                governor_active_agents.labels(agent_type=agent_type).set(self._agent_counts[agent_type])
                
                logger.info(f"Successfully registered agent {agent_id} of type {agent_type}")
                
            except Exception as e:
                logger.error(f"Error processing agent registration", exc_info=True)

    async def _initialize_systems(self):
        """Initialize memory system, game state, and discover agents."""
        # Prevent re-entry
        if hasattr(self, '_systems_initializing') and self._systems_initializing:
            logger.warning("Already initializing systems, skipping to prevent infinite loop")
            return
        
        self._systems_initializing = True
        try:
            logger.info("Initializing core systems")
            
            # Import LoreSystem locally to avoid circular import
            from lore.core.lore_system import LoreSystem
            
            # Get an instance of the LoreSystem
            self.lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
            
            # Set the governor on the lore system (dependency injection)
            self.lore_system.set_governor(self)
            
            # Initialize other systems
            from memory.memory_nyx_integration import get_memory_nyx_bridge
            self.memory_system = await get_memory_nyx_bridge(self.user_id, self.conversation_id, governor=self)
            logger.info("Memory bridge initialized")
            
            # Initialize memory integration
            from memory.memory_integration import MemoryIntegration
            # Since MemoryIntegration extends MemoryNyxBridge, we need to pass the governor
            self.memory_integration = MemoryIntegration(self.user_id, self.conversation_id, governor=self)
            logger.info("About to initialize MemoryIntegration")
            await self.memory_integration.initialize()
            logger.info("MemoryIntegration initialized")
            
            from nyx.integrate import JointMemoryGraph
            self.memory_graph = JointMemoryGraph(self.user_id, self.conversation_id)
            logger.info("JointMemoryGraph created")
            
            logger.info("About to initialize game state")
            self.game_state = await self.initialize_game_state()
            logger.info("Game state initialized")
            
            # Call the mixin version, not our deleted placeholder
            logger.info("About to discover and register agents")
            await super().discover_and_register_agents()
            logger.info("Agents discovered and registered")
            
            logger.info("About to load initial state")
            await self._load_initial_state()
            logger.info("Initial state loaded")
            
            logger.info("Core systems initialized successfully")
        finally:
            self._systems_initializing = False

    async def _load_initial_state(self):
        """Load goals and agent state from memory."""
        logger.info("Loading initial state from memory")
        
        goal_memories = await self.memory_system.recall(
            entity_type="nyx",
            entity_id=self.conversation_id,
            query="active goals",
            context="system goals",
            limit=10
        )
        
        # Call the mixin version
        self.active_goals = await super()._extract_goals_from_memories(goal_memories.get("memories", []))

        # Load all agents' goals
        for agent_type, agents_dict in self.registered_agents.items():
            for agent_id, agent in agents_dict.items():
                if hasattr(agent, "get_active_goals"):
                    self.agent_goals.setdefault(agent_type, {})[agent_id] = await agent.get_active_goals()
                else:
                    self.agent_goals.setdefault(agent_type, {})[agent_id] = []

        # Call mixin versions
        await super()._update_performance_metrics()
        await super()._load_learning_state()
        
        logger.info(f"Loaded {len(self.active_goals)} active goals")

    async def initialize_game_state(self, *, force: bool = False, player_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch and return the game-state snapshot for this user/conversation.
    
        Args:
            force: Force reload even if cached
            player_name: Override the instance player_name for this call
            
        Raises:
            PlayerNotFoundError: If no player name is available
            GovernanceError: For other initialization errors
        """
        if getattr(self, "game_state", None) and not force:
            logger.info("Using cached game state")
            return self.game_state
    
        # Determine which player name to use
        effective_player_name = player_name or self.player_name
        
        # If still no player name, try to find one in the database
        if not effective_player_name:
            try:
                async with get_db_connection_context() as conn:
                    # Check PlayerStats first
                    effective_player_name = await conn.fetchval("""
                        SELECT player_name FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2
                        LIMIT 1
                    """, self.user_id, self.conversation_id)
                    
                    # Check CurrentRoleplay if not found
                    if not effective_player_name:
                        effective_player_name = await conn.fetchval("""
                            SELECT value FROM CurrentRoleplay
                            WHERE user_id = $1 AND conversation_id = $2 AND key = 'PlayerName'
                        """, self.user_id, self.conversation_id)
                    
                    # Default to 'Chase' for new games
                    if not effective_player_name:
                        effective_player_name = 'Chase'
                        logger.info(f"No player name found, using default: {effective_player_name}")
                        # Store it for future use
                        await conn.execute("""
                            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                            VALUES ($1, $2, 'PlayerName', $3)
                            ON CONFLICT (user_id, conversation_id, key) 
                            DO UPDATE SET value = EXCLUDED.value
                        """, self.user_id, self.conversation_id, effective_player_name)
                        
            except Exception as e:
                logger.warning(f"Error looking up player name: {e}")
                effective_player_name = 'Chase'  # Fallback default
    
        logger.info(f"Initializing game state for player '{effective_player_name}'")
    
        game_state = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "player_name": effective_player_name,
            "current_location": None,
            "current_npcs": [],
            "current_time": None,
            "active_quests": [],
            "player_stats": {},
            "narrative_state": {},
            "world_state": {},
        }
    
        try:
            async with get_db_connection_context() as conn:
                # Get current location
                row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                if row: 
                    game_state["current_location"] = row["value"]
                else:
                    logger.warning("No current location found in database")
                
                # Get current time
                row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentTime'
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                if row: 
                    game_state["current_time"] = row["value"]
                else:
                    logger.warning("No current time found in database")

                # Get player stats using the effective player name
                row = await conn.fetchrow("""
                    SELECT * FROM PlayerStats
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    LIMIT 1
                """, self.user_id, self.conversation_id, effective_player_name)
                if row: 
                    game_state["player_stats"] = dict(row)
                else:
                    logger.warning(f"No player stats found for player '{effective_player_name}'")

                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                game_state["current_npcs"] = [dict(r) for r in rows]

                rows = await conn.fetch("""
                    SELECT * FROM Quests
                    WHERE user_id = $1 AND conversation_id = $2 AND status = 'In Progress'
                """, self.user_id, self.conversation_id)
                game_state["active_quests"] = [dict(r) for r in rows]

                logger.info(
                    f"Game state initialized with "
                    f"{len(game_state['current_npcs'])} NPCs and "
                    f"{len(game_state['active_quests'])} quests"
                )
        except Exception as e:
            logger.error(f"Error initializing game state", exc_info=True)
            raise GovernanceError(f"Failed to initialize game state: {str(e)}") from e
    
        # Cache it for future calls
        async with self._state_lock:
            self.game_state = game_state
        
        return game_state

    async def _get_openai_client(self):
        """Get or create the OpenAI client using chatgpt_integration."""
        from logic.chatgpt_integration import get_async_openai_client
        if self._openai_client is None:
            self._openai_client = get_async_openai_client()
        return self._openai_client

    def _migrate_legacy_kwargs(self, kwargs: dict) -> None:
        """
        Mutate kwargs in-place:
          • convert deprecated 'capabilities' to new 'tools' format
          • remove unknown keys but log them once
        """
        caps = kwargs.pop("capabilities", None)
        if caps:
            tools = kwargs.setdefault("tools", [])
            
            # Handle different capability formats
            if isinstance(caps, list):
                # If capabilities is a list of strings, treat them as custom tools
                # For now, just log them - they won't translate to OpenAI tools
                logger.info(
                    "create_agent: capabilities list %s cannot be converted to OpenAI tools",
                    caps
                )
                # Store them as custom params for potential future use
                kwargs["custom_capabilities"] = caps
            elif isinstance(caps, dict):
                # Original dict-based handling
                if caps.get("code_interpreter") is not None:
                    tools.append({"type": "code_interpreter"})
                    
                if caps.get("retrieval") is not None:
                    tools.append({"type": "file_search"})
                    
                # v1 sometimes used "function": {...} (single function)
                if caps.get("function"):
                    tools.append({"type": "function", "function": caps["function"]})
                    
                # or "functions": [...] (array of functions)
                for f in caps.get("functions", []):
                    tools.append({"type": "function", "function": f})
                    
                logger.warning(
                    "create_agent: converted legacy 'capabilities' kwarg to %d tool(s)",
                    len(tools)
                )
            else:
                logger.warning(
                    "create_agent: unexpected capabilities type %s, ignoring",
                    type(caps).__name__
                )
    
        # Log & drop any keys the OpenAI SDK won't recognise
        # Need to define VALID_OPENAI_PARAMS if not already defined
        if not hasattr(self, 'VALID_OPENAI_PARAMS'):
            # These are the common OpenAI Assistant parameters
            VALID_OPENAI_PARAMS = {
                "model", "name", "description", "instructions", "tools",
                "file_ids", "metadata", "temperature", "top_p",
                "response_format", "tool_resources"
            }
        else:
            VALID_OPENAI_PARAMS = self.VALID_OPENAI_PARAMS
            
        unknown = set(kwargs) - VALID_OPENAI_PARAMS
        if unknown:
            logger.warning(
                "create_agent: ignoring unsupported kwargs: %s",
                ", ".join(sorted(unknown))
            )
            for k in unknown:
                kwargs.pop(k, None)
        
    async def create_agent(
        self,
        agent_type: str,
        agent_id: str,
        *,
        use_openai_sdk: bool = True,
        **kwargs,
    ) -> Any:
        """
        Create / register an agent.
    
        If use_openai_sdk=True (default), we try Assistants first.
        If the model is not supported there, we transparently fall back to Responses
        (keeping the same model and tools).
        """
        logger.info("Creating agent via SDK=%s, type=%s, id=%s",
                    use_openai_sdk, agent_type, agent_id)
    
        # Return cached if exists
        if agent_id in getattr(self, "_assistants", {}):
            return self._assistants[agent_id]
    
        sdk_defaults = {
            "name": f"{agent_type}:{agent_id}",
            "model": kwargs.get("model", "gpt-5-nano"),  # keep your preferred default
            "instructions": kwargs.get("instructions", f"You are the {agent_type} agent."),
            "tools": kwargs.get("tools", []),
        }
    
        # Migrate legacy kwargs and filter for Assistants
        sdk_defaults.update(kwargs)
        if hasattr(self, "_migrate_legacy_kwargs"):
            self._migrate_legacy_kwargs(sdk_defaults)
    
        # Try Assistants first (if requested)
        if use_openai_sdk:
            try:
                client = await self._get_openai_client()
            
                # --- FIX: ensure we keep 'model' and other valid fields ---
                allowed = getattr(self, "VALID_OPENAI_PARAMS", None)
                if not allowed:
                    # sensible default whitelist for Assistants.create
                    allowed = {
                        "model", "name", "instructions", "tools", "metadata",
                        "temperature", "top_p", "response_format", "tool_resources",
                        "tool_choice", "timeout_ms",  # include any you support
                    }
            
                assistant_kwargs = {k: v for k, v in sdk_defaults.items()
                                    if k in allowed and v is not None}
            
                # Guarantee 'model' survives even if external config forgot to whitelist it
                assistant_kwargs.setdefault("model", sdk_defaults["model"])
            
                # (Optional) debug what you're actually sending
                logger.debug("assistant.create kwargs: %s", {k: type(v).__name__ for k,v in assistant_kwargs.items()})
            
                assistant = await client.beta.assistants.create(**assistant_kwargs)
            
                # keep custom params
                custom = {k: v for k, v in sdk_defaults.items() if k not in allowed}
                setattr(assistant, "_custom_params", custom)
            
                self._assistants[agent_id] = assistant
                logger.info("Assistant %s created (id=%s)", assistant.name, assistant.id)
                return assistant
            
            except BadRequestError as e:
                # Fall back to Responses only when the model is unsupported for Assistants
                if _is_unsupported_model(e):
                    logger.warning(
                        "Model %s not supported on Assistants; falling back to Responses for %s/%s",
                        sdk_defaults["model"], agent_type, agent_id
                    )
                else:
                    emsg = str(e).lower()
                    if "invalid" in emsg or "bad request" in emsg or "400" in emsg:
                        logger.error("Assistant creation failed (bad request): %s", e, exc_info=True)
                        raise AgentRegistrationError(
                            f"Assistant creation failed (bad request - check parameters): {e}"
                        ) from e
                    logger.error("OpenAI Assistant creation failed: %s", e, exc_info=True)
                    raise AgentRegistrationError(
                        f"Failed to create Assistant {agent_type}/{agent_id}: {e}"
                    ) from e
            except Exception as e:
                s = str(e).lower()
                if "unauthorized" in s or "api key" in s or "401" in s:
                    logger.error("Assistant creation failed (auth): %s", e, exc_info=True)
                    raise AgentRegistrationError(
                        f"Assistant creation failed (authentication issue): {e}"
                    ) from e
                if "rate limit" in s or "429" in s:
                    logger.error("Assistant creation failed (rate limit): %s", e, exc_info=True)
                    raise AgentRegistrationError(
                        f"Assistant creation failed (rate limit exceeded): {e}"
                    ) from e
                logger.error("OpenAI Assistant creation failed: %s", e, exc_info=True)
                raise AgentRegistrationError(
                    f"Failed to create Assistant {agent_type}/{agent_id}: {e}"
                ) from e
    
        # ────────────────────────────────────────────────────────────────────
        # Responses fallback (or direct path if use_openai_sdk=False)
        # ────────────────────────────────────────────────────────────────────
        resp_kwargs = _filter_responses_kwargs(sdk_defaults)
        # We “register” a shim with the same info so the caller can use it later.
        shim = ResponsesAssistant(
            id=f"resp_{agent_id}",
            name=resp_kwargs.get("name") or f"{agent_type}:{agent_id}",
            model=resp_kwargs["model"],
            instructions=resp_kwargs.get("instructions", ""),
            tools=resp_kwargs.get("tools", []),
            _custom_params={k: v for k, v in sdk_defaults.items() if k not in RESPONSES_ALLOWED_KEYS},
        )
        # Cache under the same dictionary so existing call-sites keep working
        self._assistants[agent_id] = shim
        logger.info("Responses-backed agent created for %s (id=%s, model=%s)",
                    agent_id, shim.id, shim.model)
        return shim

    async def register_agent(self, *args, **kwargs):
        """
        Register an agent via queue to avoid lock contention.
        
        Supports both signatures for backward compatibility:
        - New (preferred): register_agent(agent_type, agent_id, agent_instance)
        - Old (deprecated): register_agent(agent_type, agent_instance, agent_id)
        
        Examples:
            # Preferred new signature
            await governor.register_agent("npc", "npc_123", npc_instance)
            await governor.register_agent(
                agent_type="story",
                agent_id="story_main", 
                agent_instance=story_agent
            )
            
            # Deprecated old signature (will log warning)
            await governor.register_agent("npc", npc_instance, "npc_123")
        
        Args:
            agent_type: Type of the agent (e.g., 'npc', 'story', 'world')
            agent_id: Unique identifier for the agent
            agent_instance: The agent instance to register (can be None for lazy creation)
            
        Raises:
            GovernanceError: If governor not initialized
            AgentRegistrationError: If registration fails
            ValueError: If required arguments are missing
        """
        # Handle positional arguments for backward compatibility
        if len(args) == 3 and not isinstance(args[1], str):
            # Old signature: (agent_type, agent_instance, agent_id)
            agent_type, agent_instance, agent_id = args
            logger.warning(
                "Using deprecated register_agent signature (type, instance, id). "
                "Please update to (type, id, instance)"
            )
        elif len(args) == 3:
            # New signature: (agent_type, agent_id, agent_instance)
            agent_type, agent_id, agent_instance = args
        else:
            # Keyword arguments
            agent_type = kwargs.get('agent_type', args[0] if args else None)
            agent_id = kwargs.get('agent_id', args[1] if len(args) > 1 else None)
            agent_instance = kwargs.get('agent_instance', args[2] if len(args) > 2 else None)
        
        # Only require agent_type and agent_id; agent_instance can be None for lazy creation
        if not agent_type or not agent_id:
            raise ValueError("agent_type and agent_id are required")
        
        # Log if registering with None instance (lazy creation)
        if agent_instance is None:
            logger.info(f"Registering agent {agent_type}/{agent_id} for lazy creation")
        else:
            logger.info(f"Queueing agent registration: type={agent_type}, id={agent_id}")
        
        # Check if initialized
        if not hasattr(self, '_registration_task') or self._registration_task is None:
            raise GovernanceError(
                "Governor not initialized. Call await governor.initialize() before registering agents."
            )
        
        try:
            await self._agent_registration_queue.put((agent_type, agent_id, agent_instance))
        except Exception as e:
            logger.error(f"Failed to queue agent registration", exc_info=True)
            raise AgentRegistrationError(f"Failed to register agent {agent_id}: {str(e)}") from e
        
    async def get_resource_limit(self, resource_type: str, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get resource limits from the resource manager.
        
        Args:
            resource_type: Type of resource (e.g., 'compute', 'memory', 'api_calls')
            agent_type: Optional agent type for agent-specific limits
            
        Returns:
            Dictionary with limit configuration
        """
        async with self._resource_lock:
            return await self.resource_manager.get_limit(resource_type, agent_type)

    async def check_resource_usage(self, resource_type: str, usage: float, agent_type: Optional[str] = None) -> bool:
        """
        Check if resource usage is within limits.
        
        Args:
            resource_type: Type of resource
            usage: Current usage amount
            agent_type: Optional agent type for specific limits
            
        Returns:
            True if within limits, False otherwise
            
        Raises:
            ResourceLimitError: If usage exceeds hard limits
        """
        async with self._resource_lock:
            return await self.resource_manager.check_usage(
                resource_type, usage, agent_type,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )

    @with_request_id
    async def get_current_state(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the current state of the game world including narrative, character, and world state.
        
        Args:
            user_id: Must match self.user_id if provided (for backward compatibility)
            conversation_id: Must match self.conversation_id if provided (for backward compatibility)
            
        Raises:
            ValueError: If provided IDs don't match this governor's IDs
        """
        # Default to instance values
        effective_user_id = user_id if user_id is not None else self.user_id
        effective_conversation_id = conversation_id if conversation_id is not None else self.conversation_id
        
        # Security check - prevent lateral access
        if effective_user_id != self.user_id or effective_conversation_id != self.conversation_id:
            raise ValueError(
                f"Access denied: This governor is for user {self.user_id}, "
                f"conversation {self.conversation_id}. Cannot access state for "
                f"user {effective_user_id}, conversation {effective_conversation_id}."
            )
        
        logger.info(f"Getting current state for user={effective_user_id}, conversation={effective_conversation_id}")
        
        # Use the existing game state if available
        if hasattr(self, 'game_state') and self.game_state:
            base_state = self.game_state
        else:
            base_state = await self.initialize_game_state(force=True)
        
        # Get additional narrative context
        narrative_context = {}
        character_state = {}
        world_state = {}
        
        player_name = base_state.get('player_name') or self.player_name
        if not player_name:
            raise PlayerNotFoundError("Cannot get current state without player name")
        
        try:
            async with get_db_connection_context() as conn:
                # Get current narrative stage/arc
                narrative_stage = await conn.fetchval("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'NarrativeStage'
                """, effective_user_id, effective_conversation_id)
                
                if narrative_stage:
                    narrative_context["current_arc"] = narrative_stage
                else:
                    narrative_context["current_arc"] = None
                    logger.warning("No narrative stage found")
                    
                # Get active plot points
                active_quests = await conn.fetch("""
                    SELECT quest_name, progress_detail FROM Quests
                    WHERE user_id = $1 AND conversation_id = $2 AND status = 'In Progress'
                """, effective_user_id, effective_conversation_id)
                
                narrative_context["plot_points"] = [
                    {"name": q["quest_name"], "details": q["progress_detail"]} 
                    for q in active_quests
                ]
                
                # Get player character state using the player name
                player_stats = await conn.fetchrow("""
                    SELECT * FROM PlayerStats
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                """, effective_user_id, effective_conversation_id, player_name)
                
                if player_stats:
                    character_state = dict(player_stats)
                else:
                    logger.warning(f"No player stats found for '{player_name}'")
                    
                # Get player relationships
                relationships = await conn.fetch("""
                    SELECT sl.*, ns.npc_name 
                    FROM SocialLinks sl
                    JOIN NPCStats ns ON sl.entity2_id = ns.npc_id
                    WHERE sl.user_id = $1 AND sl.conversation_id = $2 
                    AND sl.entity1_type = 'player' AND sl.entity2_type = 'npc'
                """, effective_user_id, effective_conversation_id)
                
                character_state["relationships"] = {
                    r["npc_name"]: {
                        "type": r["link_type"],
                        "level": r["link_level"],
                        "stage": r["relationship_stage"]
                    } for r in relationships
                }
                
                # Get world rules and systems
                world_rules = await conn.fetch("""
                    SELECT rule_name, condition, effect FROM GameRules
                """)
                
                world_state["rules"] = {
                    r["rule_name"]: {
                        "condition": r["condition"],
                        "effect": r["effect"]
                    } for r in world_rules
                }
                
                # Get current setting info
                setting_name = await conn.fetchval("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSetting'
                """, effective_user_id, effective_conversation_id)
                
                if setting_name:
                    world_state["setting"] = setting_name
                else:
                    logger.warning("No current setting found")
                    world_state["setting"] = None
                    
        except Exception as e:
            logger.error(f"Error getting current state", exc_info=True)
            raise GovernanceError(f"Failed to get current state: {str(e)}") from e
        
        return {
            "game_state": base_state,
            "narrative_context": narrative_context,
            "character_state": character_state,
            "world_state": world_state
        }

    async def shutdown(self):
        """Clean shutdown of the governor."""
        logger.info("Shutting down governor")
        
        # Signal registration processor to stop
        if self._registration_task:
            await self._agent_registration_queue.put(None)
            await self._registration_task
            
        # Clear gauge metrics to prevent stale data
        for agent_type in self._agent_counts:
            governor_active_agents.labels(agent_type=agent_type).set(0)
        self._agent_counts.clear()
        
        # Any other cleanup...
        logger.info("Governor shutdown complete")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures cleanup."""
        await self.shutdown()
        return False  # Don't suppress exceptions
