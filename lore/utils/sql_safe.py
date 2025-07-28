# lore/utils/sql_safe.py
import asyncpg
import re
from typing import Optional

_VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Whitelist of allowed table names for extra safety
ALLOWED_TABLES = {
    'worldlore',
    'factions', 
    'culturalelements',
    'historicalevents',
    'geographicregions',
    'locationlore',
    'urbanmyths',
    'localhistories',
    'landmarks',
    'notablefigures',
    'lorechangehistory',
    'characters',
    'worldstate',
    'nations',
    'locations'
}

# Whitelist of allowed column names
ALLOWED_COLUMNS = {
    'id',
    'location_id',
    'name',
    'description',
    'embedding',
    'user_id',
    'conversation_id',
    'category',
    'significance',
    'tags',
    'type',
    'element_type',
    'practiced_by',
    'believability',
    'spread_rate',
    'territory',
    'rivals',
    'allies',
    'governing_faction',
    'reputation',
    'values',
    'goals',
    'founding_story',
    'date_description',
    'consequences',
    'event_type',
    'region_type',
    'climate',
    'strategic_value',
    'government_type',
    'location_name',
    'event_name',  # Add this for LocalHistories
}

def safe_ident(raw: str, allowed_set: Optional[set] = None) -> str:
    """
    Whitelist and quote a SQL identifier for asyncpg queries.
    Raises ValueError if it doesn't look like a plain identifier.
    
    Args:
        raw: The raw identifier string
        allowed_set: Optional whitelist of allowed values
        
    Returns:
        Quoted identifier safe for SQL interpolation
        
    Raises:
        ValueError: If the identifier is unsafe or not in whitelist
    """
    if not _VALID_IDENTIFIER_RE.match(raw):
        raise ValueError(f"Unsafe SQL identifier: {raw!r}")
    
    # Check whitelist if provided
    if allowed_set and raw.lower() not in allowed_set:
        raise ValueError(f"Identifier not in whitelist: {raw!r}")
    
    # asyncpg's builtin quoting
    return asyncpg.utils._quote_ident(raw)

def safe_table_name(table: str) -> str:
    """
    Safely quote a table name with whitelist validation.
    
    Args:
        table: The table name
        
    Returns:
        Quoted table name safe for SQL interpolation
        
    Raises:
        ValueError: If the table name is unsafe or not whitelisted
    """
    return safe_ident(table.lower(), ALLOWED_TABLES)

def safe_column_name(column: str) -> str:
    """
    Safely quote a column name with whitelist validation.
    
    Args:
        column: The column name
        
    Returns:
        Quoted column name safe for SQL interpolation
        
    Raises:
        ValueError: If the column name is unsafe or not whitelisted
    """
    return safe_ident(column.lower(), ALLOWED_COLUMNS)

def unquote_ident(quoted: str) -> str:
    """
    Remove quotes from a quoted identifier for use as a parameter value.
    
    Args:
        quoted: A quoted identifier
        
    Returns:
        The unquoted identifier
    """
    return quoted.strip('"')
