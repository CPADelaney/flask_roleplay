# logic/time_cycle.py
from db.connection import get_db_connection

TIME_PHASES = ["Morning", "Afternoon", "Evening", "Night"]

def get_current_daytime():
    """
    Returns (current_day, time_of_day) from CurrentRoleplay as (int, str).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='CurrentDay'")
    row_day = cursor.fetchone()
    current_day = int(row_day[0]) if row_day else 1

    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='TimeOfDay'")
    row_time = cursor.fetchone()
    time_of_day = row_time[0] if row_time else "Morning"

    conn.close()
    return current_day, time_of_day

def set_current_daytime(new_day, new_phase):
    """
    Overwrites the day/time in CurrentRoleplay.
    new_day: int
    new_phase: str, one of TIME_PHASES
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO CurrentRoleplay (key, value)
        VALUES ('CurrentDay', %s)
        ON CONFLICT (key) DO UPDATE SET value=excluded.value
    """, (str(new_day),))

    cursor.execute("""
        INSERT INTO CurrentRoleplay (key, value)
        VALUES ('TimeOfDay', %s)
        ON CONFLICT (key) DO UPDATE SET value=excluded.value
    """, (new_phase,))

    conn.commit()
    conn.close()

def advance_time(increment=1):
    """
    Advances time by 'increment' phases. If we wrap past 'Night', we move to next day.
    Returns (updated_day, updated_phase).
    """
    current_day, current_phase = get_current_daytime()

    phase_index = TIME_PHASES.index(current_phase) if current_phase in TIME_PHASES else 0
    new_index = phase_index + increment

    # How many times do we cross 'Night' boundary?
    day_increment = new_index // len(TIME_PHASES)
    new_index = new_index % len(TIME_PHASES)

    new_phase = TIME_PHASES[new_index]
    new_day = current_day + day_increment

    set_current_daytime(new_day, new_phase)
    return new_day, new_phase

def update_npc_schedules_for_time(day, time_of_day):
    """
    Each time we move to a new time_of_day, update NPCs' location/status accordingly.
    Example: Read NPCStats.schedule, find the location string for the current time_of_day,
             store it in some 'current_location' column, or 'is_available', etc.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch all NPCs who have schedules
    cursor.execute("""
        SELECT npc_id, schedule
        FROM NPCStats
        WHERE schedule IS NOT NULL
    """)
    rows = cursor.fetchall()

    for (npc_id, schedule_json) in rows:
        if schedule_json is None:
            continue
        # schedule_json is a dict, e.g. {"Morning":"...", "Afternoon":"...", ...}
        # Attempt to get location
        location = schedule_json.get(time_of_day, "Unknown/Off-duty")

        # For demonstration, let's store it in a column 'current_location' (TEXT).
        # If you don't have that column, add it:
        # ALTER TABLE NPCStats ADD COLUMN current_location TEXT;
        cursor.execute("""
            UPDATE NPCStats
            SET current_location = %s
            WHERE npc_id = %s
        """, (location, npc_id))

    conn.commit()
    conn.close()

def advance_time_and_update(increment=1):
    """
    Combines advance_time() + update_npc_schedules_for_time(), for a single call.
    """
    new_day, new_phase = advance_time(increment)
    update_npc_schedules_for_time(new_day, new_phase)
    return new_day, new_phase
