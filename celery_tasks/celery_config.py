# celery_tasks/celery_config.py

"""
Forwarding module for Celery configuration.
This module imports from the main celery_config.py to maintain backward compatibility.
"""

import sys
import os

# Add parent directory to path to import from root celery_config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import everything from the main celery_config
from celery_config import *

# Log that we're using the forwarding module
import logging
logger = logging.getLogger(__name__)
logger.info("Using forwarding celery_tasks/celery_config.py -> importing from root celery_config.py")
