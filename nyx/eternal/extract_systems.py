# nyx/eternal/extract_systems.py

import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_system_classes(input_file: str, output_dir: str):
    """Extract system classes from the input file and save to separate files"""
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Extract system classes
    meta_learning_match = re.search(r'class MetaLearningSystem:.*?(?=class )', content, re.DOTALL)
    dynamic_adaptation_match = re.search(r'class DynamicAdaptationSystem:.*?(?=class )', content, re.DOTALL)
    internal_feedback_match = re.search(r'class InternalFeedbackSystem:.*?(?=class )', content, re.DOTALL)
    
    if meta_learning_match:
        meta_learning_content = meta_learning_match.group(0)
        write_system_file(output_dir, 'meta_learning_system.py', meta_learning_content)
    
    if dynamic_adaptation_match:
        dynamic_adaptation_content = dynamic_adaptation_match.group(0)
        write_system_file(output_dir, 'dynamic_adaptation_system.py', dynamic_adaptation_content)
    
    if internal_feedback_match:
        internal_feedback_content = internal_feedback_match.group(0)
        write_system_file(output_dir, 'internal_feedback_system.py', internal_feedback_content)
    
    # Create __init__.py
    init_content = """
# Core systems from OpenAI Agents SDK implementation
from .meta_learning_system import MetaLearningSystem
from .dynamic_adaptation_system import DynamicAdaptationSystem
from .internal_feedback_system import InternalFeedbackSystem
"""
    with open(os.path.join(output_dir, '__init__.py'), 'w') as f:
        f.write(init_content.strip())
    
    logger.info(f"Extracted system classes to {output_dir}")

def write_system_file(output_dir: str, filename: str, content: str):
    """Write a system file with proper imports"""
    imports = """
import asyncio
import json
import logging
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import math
import time
import random

# For standalone functionality
try:
    from scipy import stats as scipy_stats
except ImportError:
    # Define minimal scipy functionality
    class scipy_stats:
        @staticmethod
        def sem(data):
            # Standard error of the mean
            if not data:
                return 0
            return np.std(data, ddof=1) / np.sqrt(len(data))
            
        class t:
            @staticmethod
            def ppf(q, df):
                # Simple approximation
                return 2.0  # Approximation for 95% confidence

logger = logging.getLogger(__name__)
"""
    
    # Add function_tool replacement for standalone functionality
    function_tool_replacement = """
# Function tool decorator for standalone use
def function_tool(func):
    \"\"\"Decorator for standalone function tools\"\"\"
    return func
"""
    
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(imports)
        f.write(function_tool_replacement)
        f.write(content)
    
    logger.info(f"Wrote {filename}")

# Example usage
if __name__ == "__main__":
    extract_system_classes('paste.txt', './systems')
