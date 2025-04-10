# render_config.py

"""
Centralized configuration for the application.

This module loads configuration from environment variables with sensible defaults.
It centralizes all configuration to avoid hardcoding values throughout the codebase.
"""

import os
import secrets
import logging
from datetime import timedelta

"""
Render-specific configuration settings.
Handles environment setup and vector database configuration for Render deployment.
"""

import os
from typing import Dict, Any, Optional

def get_render_config() -> Dict[str, Any]:
    """
    Get Render-specific configuration settings.
    Automatically detects Render environment and configures accordingly.
    """
    is_render = os.environ.get('RENDER', '').lower() == 'true'
    
    # Base configuration
    config = {
        "is_render": is_render,
        "environment": os.environ.get('RENDER_ENV', 'production'),
        
        # Vector database configuration
        "vector_db": {
            "enabled": True,
            "db_type": "qdrant",  # Using Qdrant as the vector database
            "url": os.environ.get('QDRANT_URL', 'http://localhost:6333'),
            "api_key": os.environ.get('QDRANT_API_KEY'),
            "dimension": 384,  # Consistent with existing configuration
            
            # Render-specific settings
            "connection_timeout": 30,  # seconds
            "operation_timeout": 15,  # seconds
            "max_retries": 3,
            "retry_delay": 1,  # seconds
            
            # Connection pool settings
            "pool_size": int(os.environ.get('QDRANT_POOL_SIZE', '10')),
            "max_connections": int(os.environ.get('QDRANT_MAX_CONNECTIONS', '20')),
            
            # Performance optimization for Render
            "prefer_grpc": True,  # Use gRPC when available
            "optimize_for_render": True
        },
        
        # Cache settings optimized for Render
        "cache": {
            "enabled": True,
            "ttl_seconds": {
                "l1": 300,    # 5 minutes
                "l2": 1800,   # 30 minutes
                "l3": 7200    # 2 hours
            },
            "max_size": {
                "l1": 1000,   # Increased for better performance
                "l2": 5000,
                "l3": 10000
            }
        }
    }
    
    # Render-specific optimizations
    if is_render:
        # Adjust vector database settings based on Render environment
        if os.environ.get('RENDER_ENV') == 'production':
            config['vector_db'].update({
                "pool_size": int(os.environ.get('QDRANT_POOL_SIZE', '20')),
                "max_connections": int(os.environ.get('QDRANT_MAX_CONNECTIONS', '40')),
                "connection_timeout": 45,  # Increased for production
                "operation_timeout": 30
            })
        
        # Configure automatic cleanup
        config['vector_db']['cleanup'] = {
            "enabled": True,
            "interval": 3600,  # 1 hour
            "max_age": 86400 * 30  # 30 days
        }
    
    return config

def configure_render_vector_db():
    """
    Configure vector database specifically for Render deployment.
    Sets up necessary environment variables and optimizations.
    """
    config = get_render_config()
    
    if not config['is_render']:
        return config
    
    # Set up environment variables if not already set
    if 'QDRANT_URL' not in os.environ:
        os.environ['QDRANT_URL'] = config['vector_db']['url']
    
    if 'QDRANT_API_KEY' not in os.environ:
        os.environ['QDRANT_API_KEY'] = config['vector_db'].get('api_key', '')
    
    return config 

# Configure logging
logger = logging.getLogger(__name__)

# Environment detection
ENVIRONMENT = os.getenv("FLASK_ENV", "development")
IS_PRODUCTION = ENVIRONMENT == "production"
IS_DEVELOPMENT = ENVIRONMENT == "development"
IS_TESTING = ENVIRONMENT == "testing"

# Database configuration
DB_CONFIG = {
    "dsn": os.getenv("DB_DSN", "postgresql://postgres:postgres@localhost:5432/roleplay"),
    "min_connections": int(os.getenv("DB_MIN_CONNECTIONS", 5)),
    "max_connections": int(os.getenv("DB_MAX_CONNECTIONS", 20)),
    "connection_timeout": int(os.getenv("DB_CONNECTION_TIMEOUT", 30)),
    "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", 300)),  # 5 minutes
    "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", 30)),
    "max_retries": int(os.getenv("DB_MAX_RETRIES", 3)),
    "retry_delay": float(os.getenv("DB_RETRY_DELAY", 0.5)),
}

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if IS_PRODUCTION:
        logger.warning("SECRET_KEY not set in production environment. Generating a temporary one.")
    SECRET_KEY = secrets.token_hex(32)

SECURITY_CONFIG = {
    "secret_key": SECRET_KEY,
    "session_cookie_secure": IS_PRODUCTION,
    "session_cookie_httponly": True,
    "session_cookie_samesite": "Lax",
    "permanent_session_lifetime": int(os.getenv("SESSION_LIFETIME", 86400)),  # 24 hours
    "bcrypt_log_rounds": 12 if IS_PRODUCTION else 4,
    "cors_allowed_origins": os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if os.getenv("CORS_ALLOWED_ORIGINS") else ["*"],
    
    # Enhanced security settings
    "password_history_size": int(os.getenv("PASSWORD_HISTORY_SIZE", 5)),
    "max_login_attempts": int(os.getenv("MAX_LOGIN_ATTEMPTS", 5)),
    "lockout_duration": int(os.getenv("LOCKOUT_DURATION", 300)),  # 5 minutes
    "password_min_length": int(os.getenv("PASSWORD_MIN_LENGTH", 12)),
    "password_complexity": {
        "min_uppercase": 1,
        "min_lowercase": 1,
        "min_numbers": 1,
        "min_special": 1
    },
    "session_max_concurrent": int(os.getenv("SESSION_MAX_CONCURRENT", 3)),
    "require_mfa": IS_PRODUCTION,
    "mfa_issuer": "NPC Roleplay",
    "jwt_expiration": int(os.getenv("JWT_EXPIRATION", 3600)),  # 1 hour
    "jwt_refresh_expiration": int(os.getenv("JWT_REFRESH_EXPIRATION", 2592000)),  # 30 days
    "api_rate_limits": {
        "default": "100/hour",
        "login": "5/minute",
        "register": "3/hour"
    },
    "content_security_policy": {
        "default-src": ["'self'"],
        "script-src": ["'self'", "'unsafe-inline'"],
        "style-src": ["'self'", "'unsafe-inline'"],
        "img-src": ["'self'", "data:", "https:"],
        "connect-src": ["'self'"],
        "frame-ancestors": ["'none'"],
        "form-action": ["'self'"]
    },
    "security_headers": {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
}

# Rate limiting settings
RATE_LIMIT_CONFIG = {
    "default_limit": int(os.getenv("RATE_LIMIT_DEFAULT", 100)),
    "default_period": int(os.getenv("RATE_LIMIT_PERIOD", 60)),
    "login_limit": int(os.getenv("RATE_LIMIT_LOGIN", 5)),
    "login_period": int(os.getenv("RATE_LIMIT_LOGIN_PERIOD", 60)),
    "register_limit": int(os.getenv("RATE_LIMIT_REGISTER", 3)),
    "register_period": int(os.getenv("RATE_LIMIT_REGISTER_PERIOD", 300)),
    "api_limit": int(os.getenv("RATE_LIMIT_API", 20)),
    "api_period": int(os.getenv("RATE_LIMIT_API_PERIOD", 60)),
}

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": LOG_FORMAT
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "filename": os.getenv("LOG_FILE", "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "encoding": "utf8"
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"] if IS_PRODUCTION else ["console"],
            "level": LOG_LEVEL,
            "propagate": True
        },
        "werkzeug": {
            "level": "WARNING" if IS_PRODUCTION else "INFO",
            "propagate": True
        },
    }
}

# NPC learning configuration
NPC_LEARNING_CONFIG = {
    "learning_cycle_interval": int(os.getenv("NPC_LEARNING_CYCLE_INTERVAL", 3600)),  # 1 hour
    "max_npcs_per_batch": int(os.getenv("NPC_MAX_BATCH_SIZE", 50)),
    "max_npcs_per_event": int(os.getenv("NPC_MAX_EVENT_SIZE", 10)),
    "memory_limit": int(os.getenv("NPC_MEMORY_LIMIT", 100)),
}

# Nyx agent configuration
NYX_CONFIG = {
    "memory_limit": int(os.getenv("NYX_MEMORY_LIMIT", 1000)),
    "maintain_interval": int(os.getenv("NYX_MAINTAIN_INTERVAL", 86400)),  # 24 hours
    "memory_retention_days": int(os.getenv("NYX_MEMORY_RETENTION_DAYS", 30)),
}

# Helper function to get configuration
def get_config(section=None):
    """
    Get configuration values.
    
    Args:
        section (str, optional): Configuration section to return
        
    Returns:
        dict: Configuration values
    """
    if section == "db":
        return DB_CONFIG
    elif section == "security":
        return SECURITY_CONFIG
    elif section == "rate_limit":
        return RATE_LIMIT_CONFIG
    elif section == "logging":
        return LOGGING_CONFIG
    elif section == "npc_learning":
        return NPC_LEARNING_CONFIG
    elif section == "nyx":
        return NYX_CONFIG
    else:
        # Return all config if no section specified
        return {
            "environment": ENVIRONMENT,
            "is_production": IS_PRODUCTION,
            "db": DB_CONFIG,
            "security": SECURITY_CONFIG,
            "rate_limit": RATE_LIMIT_CONFIG,
            "logging": LOGGING_CONFIG,
            "npc_learning": NPC_LEARNING_CONFIG,
            "nyx": NYX_CONFIG,
        }
