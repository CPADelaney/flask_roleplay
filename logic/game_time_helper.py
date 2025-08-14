# logic/game_time_helper.py
"""
Central helper module for converting real-time operations to in-game time.
Replace all datetime.now() and similar calls with functions from this module.
"""

import asyncio
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Import your existing time system functions
from logic.time_cycle import (
    get_current_time,
    set_current_time,
    get_current_time_model,
    TIME_PHASES,
    DAYS_PER_MONTH,
    MONTHS_PER_YEAR
)
from logic.calendar import load_calendar_names

# Cache for calendar names to avoid repeated DB calls
_calendar_cache = {}

async def get_game_datetime(user_id: int, conversation_id: int) -> datetime:
    """
    Convert in-game time to a datetime object for compatibility.
    This is useful when you need a datetime object but want to use game time.
    """
    year, month, day, time_of_day = await get_current_time(user_id, conversation_id)
    
    # Map time phases to hours
    time_mapping = {
        "Morning": 6,
        "Afternoon": 12,
        "Evening": 18,
        "Night": 0
    }
    hour = time_mapping.get(time_of_day, 6)
    
    # Create a datetime object using game time
    # Note: We use year 2000 as base and add game years as offset
    base_year = 2000
    actual_year = base_year + (year - 1)  # Game year 1 = 2000
    
    # Ensure valid month/day for datetime
    month = max(1, min(12, month))
    day = max(1, min(31, day))
    
    try:
        return datetime(actual_year, month, day, hour, 0, 0)
    except ValueError:
        # Handle invalid dates (e.g., Feb 31)
        return datetime(actual_year, month, 1, hour, 0, 0)

async def get_game_timestamp(user_id: int, conversation_id: int) -> float:
    """
    Get a timestamp based on game time instead of real time.
    Returns seconds since a game epoch.
    """
    dt = await get_game_datetime(user_id, conversation_id)
    return dt.timestamp()

async def get_game_time_string(user_id: int, conversation_id: int, 
                              include_date: bool = True,
                              include_time: bool = True) -> str:
    """
    Get a formatted string representation of current game time.
    """
    year, month, day, time_of_day = await get_current_time(user_id, conversation_id)
    
    # Try to get calendar names
    calendar_key = f"{user_id}_{conversation_id}"
    if calendar_key not in _calendar_cache:
        _calendar_cache[calendar_key] = await load_calendar_names(user_id, conversation_id)
    
    calendar = _calendar_cache[calendar_key]
    
    result = []
    
    if include_date:
        month_name = calendar["months"][month-1] if month <= len(calendar["months"]) else f"Month {month}"
        
        # Calculate day of week
        days_since_start = (year - 1) * MONTHS_PER_YEAR * DAYS_PER_MONTH + (month - 1) * DAYS_PER_MONTH + (day - 1)
        day_of_week_idx = days_since_start % 7
        day_name = calendar["days"][day_of_week_idx] if day_of_week_idx < len(calendar["days"]) else f"Day {day_of_week_idx}"
        
        result.append(f"{day_name}, {month_name} {day}, Year {year}")
    
    if include_time:
        result.append(time_of_day)
    
    return " - ".join(result) if result else "Unknown Time"

async def get_game_iso_string(user_id: int, conversation_id: int) -> str:
    """
    Get an ISO-format string based on game time.
    Useful for replacing datetime.now().isoformat() calls.
    """
    dt = await get_game_datetime(user_id, conversation_id)
    return dt.isoformat()

def get_game_time_sync(user_id: int, conversation_id: int) -> Tuple[int, int, int, str]:
    """
    Synchronous wrapper for get_current_time.
    Use sparingly - async version is preferred.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context, can't use run_until_complete
            logger.warning("Sync game time called from async context - returning defaults")
            return (1, 1, 1, "Morning")
        return loop.run_until_complete(get_current_time(user_id, conversation_id))
    except Exception as e:
        logger.error(f"Error getting sync game time: {e}")
        return (1, 1, 1, "Morning")

class GameTimeContext:
    """
    Context manager for operations that need consistent game time.
    Caches the time for the duration of the context.
    """
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.cached_time = None
        
    async def __aenter__(self):
        self.cached_time = await get_current_time(self.user_id, self.conversation_id)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cached_time = None
    
    @property
    def year(self):
        return self.cached_time[0] if self.cached_time else 1
    
    @property
    def month(self):
        return self.cached_time[1] if self.cached_time else 1
    
    @property
    def day(self):
        return self.cached_time[2] if self.cached_time else 1
    
    @property
    def time_of_day(self):
        return self.cached_time[3] if self.cached_time else "Morning"
    
    async def to_datetime(self):
        """Convert to datetime object within context."""
        return await get_game_datetime(self.user_id, self.conversation_id)
    
    async def to_string(self, include_date=True, include_time=True):
        """Get formatted string within context."""
        return await get_game_time_string(
            self.user_id, self.conversation_id, 
            include_date, include_time
        )
