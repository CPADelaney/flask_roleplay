# logic/calendar.py

"""
REFACTORED: All database writes now go through canon
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncpg
import asyncio

from db.connection import get_db_connection_context
from agents import RunContextWrapper
from lore.core import canon

# Configure logging as needed.
logging.basicConfig(level=logging.INFO)

class EventType(Enum):
    """Types of calendar events"""
    SINGLE = "single"           # One-time events
    DAILY = "daily"             # Repeats every day
    WEEKLY = "weekly"           # Repeats weekly on specific days
    MONTHLY = "monthly"         # Repeats monthly on specific date
    STORYLINE = "storyline"     # Story-critical events
    DEADLINE = "deadline"       # Time-limited events with consequences
    SOCIAL = "social"           # Social events with NPCs
    EXAM = "exam"              # Academic events
    WORK = "work"              # Work shifts
    SPECIAL = "special"        # Special/holiday events

class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    MANDATORY = 5  # Cannot be skipped

class EventVisibility(Enum):
    """When events become visible to player"""
    ALWAYS = "always"           # Always visible
    WEEK_BEFORE = "week_before" # Visible 1 week before
    DAY_BEFORE = "day_before"   # Visible 1 day before
    DAY_OF = "day_of"          # Only visible on the day
    DISCOVERED = "discovered"   # Must be discovered through gameplay

# ===============================================================================
# Database Schema for Calendar Events
# ===============================================================================

async def ensure_calendar_tables(user_id: int, conversation_id: int):
    """Ensure calendar tables exist"""
    async with get_db_connection_context() as conn:
        # Create CalendarEvents table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS CalendarEvents (
                event_id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                event_name VARCHAR(255) NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                event_priority INTEGER DEFAULT 2,
                event_visibility VARCHAR(50) DEFAULT 'always',
                description TEXT,
                location VARCHAR(255),
                
                -- Timing fields
                year INTEGER,
                month INTEGER,
                day INTEGER,
                time_of_day VARCHAR(50),  -- Morning/Afternoon/Evening/Night
                duration INTEGER DEFAULT 1, -- How many time periods it takes
                
                -- Recurrence fields
                is_recurring BOOLEAN DEFAULT FALSE,
                recurrence_pattern JSONB,  -- {"days": ["Monday", "Wednesday"], "until": "2025-12-31"}
                parent_event_id INTEGER REFERENCES CalendarEvents(event_id),
                
                -- NPC involvement
                involved_npcs JSONB DEFAULT '[]',  -- [{"npc_id": 1, "role": "host"}]
                
                -- Requirements and rewards
                requirements JSONB DEFAULT '{}',   -- {"stats": {"intelligence": 50}, "items": ["textbook"]}
                rewards JSONB DEFAULT '{}',        -- {"stats": {"intelligence": 2}, "money": 100}
                consequences JSONB DEFAULT '{}',   -- {"miss": {"trust": -10, "narrative": "disappointed"}}
                
                -- Status tracking
                is_completed BOOLEAN DEFAULT FALSE,
                is_missed BOOLEAN DEFAULT FALSE,
                is_cancelled BOOLEAN DEFAULT FALSE,
                completion_data JSONB DEFAULT '{}',
                
                -- Metadata
                tags TEXT[] DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Unique constraint to prevent duplicates
                UNIQUE(user_id, conversation_id, event_name, year, month, day, time_of_day)
            )
        """)
        
        # Create index for efficient queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calendar_events_lookup 
            ON CalendarEvents(user_id, conversation_id, year, month, day)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calendar_events_recurring 
            ON CalendarEvents(user_id, conversation_id, is_recurring, parent_event_id)
        """)


# ---------------------------------------------------------------------------
async def get_chatgpt_response_no_function(
    conversation_id: int,
    aggregator_text: str,
    user_input: str,
) -> dict:
    """
    Fire a call to the Responses endpoint and return the plain-text output.
    """
    from logic.chatgpt_integration import get_async_openai_client, build_message_history, safe_json_loads
    client = get_async_openai_client()   # Use the existing function

    # prepare the request payload ------------------------------------------
    messages = await build_message_history(
        conversation_id, aggregator_text, user_input, limit=15
    )

    # call the Responses API -----------------------------------------------
    response = await client.responses.create(
        model="gpt-4.1-nano",   # or "gpt-4.1-nano" if you really need that tier
        input=messages         # Responses API uses the *single* `input` field
    )                          # Streaming? add stream=True and iterate.

    # extract what we need --------------------------------------------------
    response_text = response.output_text
    tokens_used   = response.usage.total_tokens        # guaranteed to exist

    return {
        "type": "text",
        "response": response_text,
        "tokens_used": tokens_used,
    }

    
async def generate_calendar_names(environment_desc, conversation_id):
    """
    Use GPT to generate immersive calendar names for the in-game time system.
    
    The GPT prompt asks for:
      - "year_name": a creative name for the overall year,
      - "months": an array of 12 unique month names,
      - "days": an array of 7 unique day names.
    
    Returns a dictionary with these keys.
    """
    prompt = (
        "Based on the following environment description, generate an immersive and thematic naming scheme for the in-game calendar. "
        "Keep in mind the context is 'femdom daily-life sim roleplaying game' and the names should reflect this. "
        "Ensure names are creative and unique, and are rooted in the universe and history of the setting. "
        "Your response should be in JSON format with exactly the following keys:\n"
        "  - \"year_name\": a creative name for the overall year (e.g., 'The Age of Ember', 'The Silver Cycle'),\n"
        "  - \"months\": an array of exactly 12 creative and unique month names,\n"
        "  - \"days\": an array of exactly 7 creative and unique day names for the week.\n\n"
        "IMPORTANT: Ensure your JSON is valid - all array elements must be separated by commas.\n"
        "Environment description: " + environment_desc + "\n\n"
        "Return only the JSON object with no additional explanation or markdown formatting."
    )
    
    logging.info("Calling GPT for calendar names with prompt:\n%s", prompt)
    # Use the no-function variant for calendar names.
    gpt_response = await get_chatgpt_response_no_function(conversation_id, environment_desc, prompt)
    logging.info("GPT calendar naming response: %s", gpt_response)
    
    calendar_names = {}
    try:
        response_text = gpt_response.get("response", "").strip()
        
        # Remove markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()
        
        # Fix common JSON issues
        response_text = fix_common_json_issues(response_text)
        
        calendar_names = json.loads(response_text)
        
        # Validate the structure
        if not isinstance(calendar_names.get("months"), list) or len(calendar_names["months"]) != 12:
            raise ValueError("Invalid months array - must have exactly 12 months")
        if not isinstance(calendar_names.get("days"), list) or len(calendar_names["days"]) != 7:
            raise ValueError("Invalid days array - must have exactly 7 days")
        if not calendar_names.get("year_name"):
            raise ValueError("Missing year_name")
            
    except Exception as e:
        logging.error("Failed to parse calendar names JSON: %s", e, exc_info=True)
        logging.error("Raw response text: %s", response_text if 'response_text' in locals() else 'N/A')
        
        # Fallback to a default naming scheme if GPT fails
        calendar_names = {
            "year_name": "The Eternal Cycle",
            "months": ["Aurora", "Blaze", "Crimson", "Dusk", "Ember", "Frost", 
                      "Gleam", "Haze", "Iris", "Jade", "Knell", "Lumen"],
            "days": ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]
        }
    
    return calendar_names


def fix_common_json_issues(json_str: str) -> str:
    """
    Fix common JSON formatting issues from LLM outputs.
    """
    import re
    
    # Fix missing commas between array elements
    # Pattern: "element1"\n"element2" -> "element1",\n"element2"
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    
    # Fix missing commas between array elements on same line
    # Pattern: "element1" "element2" -> "element1", "element2"
    json_str = re.sub(r'"\s+"', '", "', json_str)
    
    # Fix trailing commas before closing brackets
    json_str = re.sub(r',\s*\]', ']', json_str)
    json_str = re.sub(r',\s*\}', '}', json_str)
    
    # Fix missing quotes around keys (simple cases)
    # Pattern: key: "value" -> "key": "value"
    json_str = re.sub(r'(\w+):\s*"', r'"\1": "', json_str)
    
    # Ensure arrays have proper formatting
    # Fix cases like ["item1""item2"] -> ["item1","item2"]
    json_str = re.sub(r'"\s*"', '", "', json_str)
    
    return json_str

async def store_calendar_names(user_id: int, conversation_id: int, calendar_names: dict, conn: asyncpg.Connection):
    """
    REFACTORED: Uses canon to store calendar names in CurrentRoleplay
    Connection is passed from calling function.
    """
    try:
        # Convert data to JSON string
        value_json = json.dumps(calendar_names)
        
        # Create a context object for canon - UPDATE THIS
        ctx = RunContextWrapper(context={
            'user_id': user_id,
            'conversation_id': conversation_id
        })
        
        # Use canon to update CurrentRoleplay
        await canon.update_current_roleplay(
            ctx, conn, 
            'CalendarNames', value_json
        )
    except Exception as e:
        logging.error(f"Error storing calendar names: {e}")
        raise  # Re-raise the exception after logging

async def update_calendar_names(user_id, conversation_id, environment_desc) -> dict:
    """
    REFACTORED: Generates immersive calendar names based on the provided environment description,
    stores them using canon, and returns the resulting dictionary.
    """
    calendar_names = await generate_calendar_names(environment_desc, conversation_id)
    
    # Get connection from context manager and pass it to store function
    async with get_db_connection_context() as conn:
        await store_calendar_names(user_id, conversation_id, calendar_names, conn)
    
    return calendar_names

async def load_calendar_names(user_id, conversation_id):
    """
    Retrieves the calendar names (year_name, months, days) 
    from CurrentRoleplay where key='CalendarNames'.
    Returns a dict with keys 'year_name', 'months', and 'days'.
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT value 
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
                LIMIT 1
            """, user_id, conversation_id)
            
            if row:
                try:
                    return json.loads(row['value'])
                except json.JSONDecodeError as e:
                    logging.warning("Calendar JSON invalid, returning fallback.")
    except Exception as e:
        logging.error(f"Error loading calendar names: {e}")
    
    # Fallback if not found or invalid
    return {
        "year_name": "The Eternal Cycle",
        "months": [
            "Aurora", "Blaze", "Crimson", "Dusk",
            "Ember", "Frost", "Gleam", "Haze",
            "Iris", "Jade", "Knell", "Lumen"
        ],
        "days": ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]
    }

async def add_calendar_event(
    user_id: int,
    conversation_id: int,
    event_name: str,
    event_type: Union[str, EventType],
    year: int,
    month: int,
    day: int,
    time_of_day: str = "Afternoon",
    description: Optional[str] = None,
    location: Optional[str] = None,
    duration: int = 1,
    priority: Union[int, EventPriority] = EventPriority.MEDIUM,
    visibility: Union[str, EventVisibility] = EventVisibility.ALWAYS,
    involved_npcs: Optional[List[Dict[str, Any]]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    rewards: Optional[Dict[str, Any]] = None,
    consequences: Optional[Dict[str, Any]] = None,
    is_recurring: bool = False,
    recurrence_pattern: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Add a calendar event (one-time or recurring).
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        event_name: Name of the event
        event_type: Type of event (EventType enum or string)
        year, month, day: Date of the event
        time_of_day: Morning/Afternoon/Evening/Night
        description: Event description
        location: Where the event takes place
        duration: How many time periods it takes
        priority: Event priority level
        visibility: When the event becomes visible
        involved_npcs: NPCs involved in the event
        requirements: Requirements to participate
        rewards: Rewards for completing
        consequences: Consequences for missing
        is_recurring: Whether this is a recurring event
        recurrence_pattern: Pattern for recurring events
        tags: Tags for categorization
    
    Returns:
        Dict with event details and ID
    """
    try:
        # Ensure tables exist
        await ensure_calendar_tables(user_id, conversation_id)
        
        # Convert enums to strings
        if isinstance(event_type, EventType):
            event_type = event_type.value
        if isinstance(priority, EventPriority):
            priority = priority.value
        if isinstance(visibility, EventVisibility):
            visibility = visibility.value
        
        async with get_db_connection_context() as conn:
            # Check if event already exists
            existing = await conn.fetchrow("""
                SELECT event_id FROM CalendarEvents
                WHERE user_id = $1 AND conversation_id = $2 
                AND event_name = $3 AND year = $4 AND month = $5 
                AND day = $6 AND time_of_day = $7
            """, user_id, conversation_id, event_name, year, month, day, time_of_day)
            
            if existing:
                return {
                    "success": False,
                    "message": "Event already exists",
                    "event_id": existing['event_id']
                }
            
            # Insert the event
            event_id = await conn.fetchval("""
                INSERT INTO CalendarEvents (
                    user_id, conversation_id, event_name, event_type,
                    event_priority, event_visibility, description, location,
                    year, month, day, time_of_day, duration,
                    is_recurring, recurrence_pattern,
                    involved_npcs, requirements, rewards, consequences,
                    tags
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18, $19, $20
                ) RETURNING event_id
            """, 
                user_id, conversation_id, event_name, event_type,
                priority, visibility, description, location,
                year, month, day, time_of_day, duration,
                is_recurring, json.dumps(recurrence_pattern) if recurrence_pattern else None,
                json.dumps(involved_npcs or []),
                json.dumps(requirements or {}),
                json.dumps(rewards or {}),
                json.dumps(consequences or {}),
                tags or []
            )
            
            # If it's a recurring event, generate instances
            if is_recurring and recurrence_pattern:
                await generate_recurring_instances(
                    conn, event_id, user_id, conversation_id,
                    event_name, event_type, year, month, day,
                    time_of_day, recurrence_pattern, duration,
                    priority, visibility, location, involved_npcs,
                    requirements, rewards, consequences, tags
                )
            
            # Log the event creation
            logger.info(f"Added calendar event '{event_name}' for user {user_id}")
            
            return {
                "success": True,
                "event_id": event_id,
                "event_name": event_name,
                "date": f"{year}-{month:02d}-{day:02d}",
                "time": time_of_day,
                "is_recurring": is_recurring
            }
            
    except Exception as e:
        logger.error(f"Error adding calendar event: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def generate_recurring_instances(
    conn: asyncpg.Connection,
    parent_event_id: int,
    user_id: int,
    conversation_id: int,
    event_name: str,
    event_type: str,
    start_year: int,
    start_month: int,
    start_day: int,
    time_of_day: str,
    recurrence_pattern: Dict[str, Any],
    duration: int,
    priority: int,
    visibility: str,
    location: Optional[str],
    involved_npcs: Optional[List[Dict]],
    requirements: Optional[Dict],
    rewards: Optional[Dict],
    consequences: Optional[Dict],
    tags: Optional[List[str]]
):
    """Generate recurring event instances based on pattern"""
    
    pattern_type = recurrence_pattern.get("type", "weekly")
    until_date = recurrence_pattern.get("until")
    max_occurrences = recurrence_pattern.get("max_occurrences", 52)  # Default 1 year
    
    if pattern_type == "daily":
        # Generate daily events
        current_date = datetime(start_year, start_month, start_day)
        end_date = datetime.strptime(until_date, "%Y-%m-%d") if until_date else current_date + timedelta(days=365)
        occurrences = 0
        
        while current_date <= end_date and occurrences < max_occurrences:
            if current_date.day != start_day or current_date.month != start_month:  # Skip the original
                await conn.execute("""
                    INSERT INTO CalendarEvents (
                        user_id, conversation_id, event_name, event_type,
                        event_priority, event_visibility, description, location,
                        year, month, day, time_of_day, duration,
                        is_recurring, parent_event_id,
                        involved_npcs, requirements, rewards, consequences, tags
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19, $20
                    ) ON CONFLICT DO NOTHING
                """,
                    user_id, conversation_id, event_name, event_type,
                    priority, visibility, f"[Recurring] {event_name}", location,
                    current_date.year, current_date.month, current_date.day,
                    time_of_day, duration, True, parent_event_id,
                    json.dumps(involved_npcs or []),
                    json.dumps(requirements or {}),
                    json.dumps(rewards or {}),
                    json.dumps(consequences or {}),
                    tags or []
                )
                occurrences += 1
            current_date += timedelta(days=1)
    
    elif pattern_type == "weekly":
        # Generate weekly events on specific days
        days_of_week = recurrence_pattern.get("days", [])
        if not days_of_week:
            return
        
        # Map day names to weekday numbers
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        
        current_date = datetime(start_year, start_month, start_day)
        end_date = datetime.strptime(until_date, "%Y-%m-%d") if until_date else current_date + timedelta(days=365)
        occurrences = 0
        
        while current_date <= end_date and occurrences < max_occurrences:
            # Get calendar day name (you'd need to map this to your calendar system)
            weekday = current_date.weekday()
            
            # Check if this day is in our recurrence pattern
            for day_name in days_of_week:
                if day_map.get(day_name) == weekday:
                    if current_date.day != start_day or current_date.month != start_month:
                        await conn.execute("""
                            INSERT INTO CalendarEvents (
                                user_id, conversation_id, event_name, event_type,
                                event_priority, event_visibility, description, location,
                                year, month, day, time_of_day, duration,
                                is_recurring, parent_event_id,
                                involved_npcs, requirements, rewards, consequences, tags
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                                $14, $15, $16, $17, $18, $19, $20
                            ) ON CONFLICT DO NOTHING
                        """,
                            user_id, conversation_id, event_name, event_type,
                            priority, visibility, f"[Recurring] {event_name}", location,
                            current_date.year, current_date.month, current_date.day,
                            time_of_day, duration, True, parent_event_id,
                            json.dumps(involved_npcs or []),
                            json.dumps(requirements or {}),
                            json.dumps(rewards or {}),
                            json.dumps(consequences or {}),
                            tags or []
                        )
                        occurrences += 1
                    break
            
            current_date += timedelta(days=1)

async def get_calendar_events(
    user_id: int,
    conversation_id: int,
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    time_of_day: Optional[str] = None,
    event_type: Optional[str] = None,
    include_completed: bool = False,
    include_missed: bool = True,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get calendar events with various filters.
    
    Returns events sorted by date and time.
    """
    try:
        await ensure_calendar_tables(user_id, conversation_id)
        
        async with get_db_connection_context() as conn:
            # Build query dynamically
            query = """
                SELECT * FROM CalendarEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND is_cancelled = FALSE
            """
            params = [user_id, conversation_id]
            param_count = 2
            
            if not include_completed:
                query += " AND is_completed = FALSE"
            
            if not include_missed:
                query += " AND is_missed = FALSE"
            
            if year is not None:
                param_count += 1
                query += f" AND year = ${param_count}"
                params.append(year)
            
            if month is not None:
                param_count += 1
                query += f" AND month = ${param_count}"
                params.append(month)
            
            if day is not None:
                param_count += 1
                query += f" AND day = ${param_count}"
                params.append(day)
            
            if time_of_day:
                param_count += 1
                query += f" AND time_of_day = ${param_count}"
                params.append(time_of_day)
            
            if event_type:
                param_count += 1
                query += f" AND event_type = ${param_count}"
                params.append(event_type)
            
            # Order by date and time
            time_order = """
                ORDER BY year, month, day,
                CASE time_of_day
                    WHEN 'Morning' THEN 1
                    WHEN 'Afternoon' THEN 2
                    WHEN 'Evening' THEN 3
                    WHEN 'Night' THEN 4
                    ELSE 5
                END
            """
            query += time_order
            
            param_count += 1
            query += f" LIMIT ${param_count}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            events = []
            for row in rows:
                event = dict(row)
                # Parse JSON fields
                for json_field in ['recurrence_pattern', 'involved_npcs', 'requirements', 'rewards', 'consequences', 'completion_data']:
                    if event.get(json_field):
                        try:
                            event[json_field] = json.loads(event[json_field]) if isinstance(event[json_field], str) else event[json_field]
                        except json.JSONDecodeError:
                            event[json_field] = {}
                
                events.append(event)
            
            return events
            
    except Exception as e:
        logger.error(f"Error getting calendar events: {e}", exc_info=True)
        return []

async def get_events_for_display(
    user_id: int,
    conversation_id: int,
    current_year: int,
    current_month: int,
    current_day: int,
    lookahead_days: int = 7
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get events formatted for calendar display (Persona 5 style).
    
    Returns a dict with dates as keys and event lists as values.
    """
    try:
        events_by_date = {}
        
        # Get events for the next N days
        current_date = datetime(current_year, current_month, current_day)
        
        for days_ahead in range(lookahead_days):
            check_date = current_date + timedelta(days=days_ahead)
            
            # Get events for this date
            day_events = await get_calendar_events(
                user_id, conversation_id,
                year=check_date.year,
                month=check_date.month,
                day=check_date.day,
                include_completed=False,
                include_missed=False
            )
            
            if day_events:
                date_key = f"{check_date.year}-{check_date.month:02d}-{check_date.day:02d}"
                
                # Format events for display
                formatted_events = []
                for event in day_events:
                    # Check visibility
                    visibility = event.get('event_visibility', 'always')
                    
                    if visibility == 'always':
                        show = True
                    elif visibility == 'week_before':
                        show = days_ahead <= 7
                    elif visibility == 'day_before':
                        show = days_ahead <= 1
                    elif visibility == 'day_of':
                        show = days_ahead == 0
                    else:
                        show = event.get('is_discovered', False)
                    
                    if show:
                        formatted_events.append({
                            'name': event['event_name'],
                            'time': event['time_of_day'],
                            'type': event['event_type'],
                            'priority': event['event_priority'],
                            'location': event.get('location', 'Unknown'),
                            'description': event.get('description', ''),
                            'is_recurring': event.get('is_recurring', False),
                            'has_requirements': bool(event.get('requirements')),
                            'has_rewards': bool(event.get('rewards')),
                            'involved_npcs': event.get('involved_npcs', [])
                        })
                
                if formatted_events:
                    events_by_date[date_key] = formatted_events
        
        return events_by_date
        
    except Exception as e:
        logger.error(f"Error getting events for display: {e}", exc_info=True)
        return {}

async def mark_event_completed(
    user_id: int,
    conversation_id: int,
    event_id: int,
    completion_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Mark an event as completed and process rewards"""
    try:
        async with get_db_connection_context() as conn:
            # Get event details
            event = await conn.fetchrow("""
                SELECT * FROM CalendarEvents
                WHERE event_id = $1 AND user_id = $2 AND conversation_id = $3
            """, event_id, user_id, conversation_id)
            
            if not event:
                return {"success": False, "error": "Event not found"}
            
            # Mark as completed
            await conn.execute("""
                UPDATE CalendarEvents
                SET is_completed = TRUE,
                    completion_data = $1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE event_id = $2
            """, json.dumps(completion_data or {}), event_id)
            
            # Process rewards if any
            rewards = json.loads(event['rewards']) if event['rewards'] else {}
            applied_rewards = []
            
            if rewards:
                # Apply stat rewards
                if 'stats' in rewards:
                    from logic.stats_logic import apply_stat_changes
                    await apply_stat_changes(
                        user_id, conversation_id, "Chase",
                        rewards['stats'],
                        f"Event reward: {event['event_name']}"
                    )
                    applied_rewards.append({"type": "stats", "value": rewards['stats']})
                
                # Apply money rewards
                if 'money' in rewards:
                    # You'd implement money addition here
                    applied_rewards.append({"type": "money", "value": rewards['money']})
                
                # Apply item rewards
                if 'items' in rewards:
                    from logic.inventory_system_sdk import add_item
                    for item in rewards['items']:
                        await add_item(
                            user_id, conversation_id, "Chase",
                            item['name'], item.get('description'),
                            item.get('effect')
                        )
                    applied_rewards.append({"type": "items", "value": rewards['items']})
            
            return {
                "success": True,
                "event_name": event['event_name'],
                "rewards_applied": applied_rewards
            }
            
    except Exception as e:
        logger.error(f"Error marking event completed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def mark_event_missed(
    user_id: int,
    conversation_id: int,
    event_id: int
) -> Dict[str, Any]:
    """Mark an event as missed and process consequences"""
    try:
        async with get_db_connection_context() as conn:
            # Get event details
            event = await conn.fetchrow("""
                SELECT * FROM CalendarEvents
                WHERE event_id = $1 AND user_id = $2 AND conversation_id = $3
            """, event_id, user_id, conversation_id)
            
            if not event:
                return {"success": False, "error": "Event not found"}
            
            # Mark as missed
            await conn.execute("""
                UPDATE CalendarEvents
                SET is_missed = TRUE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE event_id = $1
            """, event_id)
            
            # Process consequences if any
            consequences = json.loads(event['consequences']) if event['consequences'] else {}
            applied_consequences = []
            
            if consequences:
                # Apply stat penalties
                if 'stats' in consequences:
                    from logic.stats_logic import apply_stat_changes
                    await apply_stat_changes(
                        user_id, conversation_id, "Chase",
                        consequences['stats'],
                        f"Missed event: {event['event_name']}"
                    )
                    applied_consequences.append({"type": "stats", "value": consequences['stats']})
                
                # Apply relationship penalties
                if 'relationships' in consequences:
                    from logic.dynamic_relationships import OptimizedRelationshipManager
                    manager = OptimizedRelationshipManager(user_id, conversation_id)
                    
                    for npc_id, impacts in consequences['relationships'].items():
                        state = await manager.get_relationship_state(
                            'player', 1, 'npc', int(npc_id)
                        )
                        
                        for dimension, change in impacts.items():
                            if hasattr(state.dimensions, dimension):
                                current = getattr(state.dimensions, dimension)
                                setattr(state.dimensions, dimension, current + change)
                        
                        state.dimensions.clamp()
                        await manager._queue_update(state)
                    
                    await manager._flush_updates()
                    applied_consequences.append({"type": "relationships", "value": consequences['relationships']})
                
                # Narrative consequences
                if 'narrative' in consequences:
                    # Store as a memory
                    from logic.memory_logic import MemoryManager, MemoryType, MemorySignificance
                    await MemoryManager.add_memory(
                        user_id, conversation_id,
                        entity_id=1, entity_type="player",
                        memory_text=f"Missed {event['event_name']}: {consequences['narrative']}",
                        memory_type=MemoryType.INTERACTION,
                        significance=MemorySignificance.HIGH,
                        emotional_valence=-0.5,
                        tags=["missed_event", event['event_type']]
                    )
                    applied_consequences.append({"type": "narrative", "value": consequences['narrative']})
            
            return {
                "success": True,
                "event_name": event['event_name'],
                "consequences_applied": applied_consequences
            }
            
    except Exception as e:
        logger.error(f"Error marking event missed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def check_event_requirements(
    user_id: int,
    conversation_id: int,
    event_id: int
) -> Tuple[bool, List[str]]:
    """
    Check if player meets requirements for an event.
    
    Returns (can_participate, missing_requirements)
    """
    try:
        async with get_db_connection_context() as conn:
            event = await conn.fetchrow("""
                SELECT requirements FROM CalendarEvents
                WHERE event_id = $1 AND user_id = $2 AND conversation_id = $3
            """, event_id, user_id, conversation_id)
            
            if not event:
                return False, ["Event not found"]
            
            requirements = json.loads(event['requirements']) if event['requirements'] else {}
            
            if not requirements:
                return True, []
            
            missing = []
            
            # Check stat requirements
            if 'stats' in requirements:
                from logic.stats_logic import get_all_player_stats
                player_stats = await get_all_player_stats(user_id, conversation_id, "Chase")
                
                for stat, required_value in requirements['stats'].items():
                    if player_stats.get(stat, 0) < required_value:
                        missing.append(f"{stat} >= {required_value}")
            
            # Check item requirements
            if 'items' in requirements:
                from logic.inventory_system_sdk import get_inventory
                inventory = await get_inventory(user_id, conversation_id, "Chase")
                player_items = [item['item_name'] for item in inventory.get('items', [])]
                
                for required_item in requirements['items']:
                    if required_item not in player_items:
                        missing.append(f"Item: {required_item}")
            
            # Check relationship requirements
            if 'relationships' in requirements:
                from logic.dynamic_relationships import OptimizedRelationshipManager
                manager = OptimizedRelationshipManager(user_id, conversation_id)
                
                for npc_id, min_values in requirements['relationships'].items():
                    state = await manager.get_relationship_state(
                        'player', 1, 'npc', int(npc_id)
                    )
                    
                    for dimension, required in min_values.items():
                        current = getattr(state.dimensions, dimension, 0)
                        if current < required:
                            missing.append(f"NPC {npc_id} {dimension} >= {required}")
            
            return len(missing) == 0, missing
            
    except Exception as e:
        logger.error(f"Error checking event requirements: {e}", exc_info=True)
        return False, [f"Error: {str(e)}"]

# ===============================================================================
# Specialized Event Creation Functions
# ===============================================================================

async def add_class_schedule(
    user_id: int,
    conversation_id: int,
    class_name: str,
    days: List[str],
    time_of_day: str,
    location: str,
    start_date: Tuple[int, int, int],
    end_date: Tuple[int, int, int],
    professor_npc_id: Optional[int] = None
) -> Dict[str, Any]:
    """Add a recurring class to the calendar"""
    
    return await add_calendar_event(
        user_id, conversation_id,
        event_name=class_name,
        event_type=EventType.WEEKLY,
        year=start_date[0],
        month=start_date[1],
        day=start_date[2],
        time_of_day=time_of_day,
        description=f"Attend {class_name}",
        location=location,
        duration=1,
        priority=EventPriority.HIGH,
        visibility=EventVisibility.ALWAYS,
        involved_npcs=[{"npc_id": professor_npc_id, "role": "professor"}] if professor_npc_id else None,
        requirements={"stats": {"energy": 20}},
        rewards={"stats": {"intelligence": 1}},
        consequences={"stats": {"intelligence": -1}, "narrative": "You missed an important lecture"},
        is_recurring=True,
        recurrence_pattern={
            "type": "weekly",
            "days": days,
            "until": f"{end_date[0]}-{end_date[1]:02d}-{end_date[2]:02d}"
        },
        tags=["academic", "class", "mandatory"]
    )

async def add_exam(
    user_id: int,
    conversation_id: int,
    exam_name: str,
    subject: str,
    year: int,
    month: int,
    day: int,
    time_of_day: str = "Morning",
    location: str = "Exam Hall",
    difficulty: int = 50
) -> Dict[str, Any]:
    """Add an exam to the calendar"""
    
    return await add_calendar_event(
        user_id, conversation_id,
        event_name=exam_name,
        event_type=EventType.EXAM,
        year=year,
        month=month,
        day=day,
        time_of_day=time_of_day,
        description=f"{subject} examination",
        location=location,
        duration=2,
        priority=EventPriority.CRITICAL,
        visibility=EventVisibility.WEEK_BEFORE,
        requirements={"stats": {"intelligence": difficulty, "energy": 30}},
        rewards={"stats": {"confidence": 5, "intelligence": 2}},
        consequences={"stats": {"confidence": -10, "mental_resilience": -5}},
        tags=["academic", "exam", "critical", subject.lower()]
    )

async def add_social_event(
    user_id: int,
    conversation_id: int,
    event_name: str,
    host_npc_id: int,
    year: int,
    month: int,
    day: int,
    time_of_day: str,
    location: str,
    other_attendees: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Add a social event with NPCs"""
    
    involved_npcs = [{"npc_id": host_npc_id, "role": "host"}]
    if other_attendees:
        for npc_id in other_attendees:
            involved_npcs.append({"npc_id": npc_id, "role": "attendee"})
    
    return await add_calendar_event(
        user_id, conversation_id,
        event_name=event_name,
        event_type=EventType.SOCIAL,
        year=year,
        month=month,
        day=day,
        time_of_day=time_of_day,
        description=f"Social event hosted by NPC {host_npc_id}",
        location=location,
        duration=1,
        priority=EventPriority.MEDIUM,
        visibility=EventVisibility.DAY_BEFORE,
        involved_npcs=involved_npcs,
        rewards={"relationships": {str(host_npc_id): {"trust": 5, "affection": 3}}},
        consequences={"relationships": {str(host_npc_id): {"trust": -5, "affection": -5}}},
        tags=["social", "npc_event"]
    )

async def add_work_shift(
    user_id: int,
    conversation_id: int,
    job_name: str,
    days: List[str],
    time_of_day: str,
    location: str,
    pay: int,
    start_date: Tuple[int, int, int],
    duration_weeks: int = 52
) -> Dict[str, Any]:
    """Add recurring work shifts"""
    
    end_date = datetime(*start_date) + timedelta(weeks=duration_weeks)
    
    return await add_calendar_event(
        user_id, conversation_id,
        event_name=f"{job_name} Shift",
        event_type=EventType.WORK,
        year=start_date[0],
        month=start_date[1],
        day=start_date[2],
        time_of_day=time_of_day,
        description=f"Work at {job_name}",
        location=location,
        duration=2,
        priority=EventPriority.HIGH,
        visibility=EventVisibility.ALWAYS,
        requirements={"stats": {"energy": 30}},
        rewards={"money": pay, "stats": {"endurance": 1}},
        consequences={"money": -pay, "narrative": "You lose pay for missing work"},
        is_recurring=True,
        recurrence_pattern={
            "type": "weekly",
            "days": days,
            "until": end_date.strftime("%Y-%m-%d")
        },
        tags=["work", "income", "recurring"]
    )

# ===============================================================================
# Integration with Time System
# ===============================================================================

async def check_current_events(
    user_id: int,
    conversation_id: int,
    year: int,
    month: int,
    day: int,
    time_of_day: str
) -> List[Dict[str, Any]]:
    """Check what events are happening right now"""
    
    events = await get_calendar_events(
        user_id, conversation_id,
        year=year,
        month=month,
        day=day,
        time_of_day=time_of_day,
        include_completed=False,
        include_missed=False
    )
    
    # Check requirements for each event
    available_events = []
    for event in events:
        can_participate, missing = await check_event_requirements(
            user_id, conversation_id, event['event_id']
        )
        
        event['can_participate'] = can_participate
        event['missing_requirements'] = missing
        available_events.append(event)
    
    return available_events

async def auto_process_missed_events(
    user_id: int,
    conversation_id: int,
    year: int,
    month: int,
    day: int,
    time_of_day: str
):
    """Automatically mark past events as missed if not completed"""
    
    try:
        async with get_db_connection_context() as conn:
            # Find events that should have happened but weren't completed
            await conn.execute("""
                UPDATE CalendarEvents
                SET is_missed = TRUE, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = $1 AND conversation_id = $2
                AND is_completed = FALSE AND is_missed = FALSE
                AND (
                    year < $3 OR
                    (year = $3 AND month < $4) OR
                    (year = $3 AND month = $4 AND day < $5) OR
                    (year = $3 AND month = $4 AND day = $5 AND 
                     CASE time_of_day
                        WHEN 'Morning' THEN 1
                        WHEN 'Afternoon' THEN 2
                        WHEN 'Evening' THEN 3
                        WHEN 'Night' THEN 4
                     END < 
                     CASE $6
                        WHEN 'Morning' THEN 1
                        WHEN 'Afternoon' THEN 2
                        WHEN 'Evening' THEN 3
                        WHEN 'Night' THEN 4
                     END)
                )
            """, user_id, conversation_id, year, month, day, time_of_day)
            
    except Exception as e:
        logger.error(f"Error auto-processing missed events: {e}")
