# render_config.py

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
            "dimension": 1536,  # Consistent with existing configuration
            
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
