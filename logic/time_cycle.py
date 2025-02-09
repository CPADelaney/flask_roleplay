from db.connection import get_db_connection

# Define constants
DAYS_PER_MONTH = 30
MONTHS_PER_YEAR = 12  # Adjust if needed

# Define your phases (this remains the same)
TIME_PHASES = ["Morning", "Afternoon", "Evening", "Night"]

def get_current_time(user_id, conversation_id):
    """
    Returns a tuple (current_year, current_month, current_day, time_of_day) for the given user_id and conversation_id.
    Falls back to defaults: year=1, month=1, day=1, phase="Morning".
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get CurrentYear
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key='CurrentYear'
    """, (user_id, conversation_id))
    row_year = cursor.fetchone()
    current_year = int(row_year[0]) if row_year else 1

    # Get CurrentMonth
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key='CurrentMonth'
    """, (user_id, conversation_id))
    row_month = cursor.fetchone()
    current_month = int(row_month[0]) if row_month else 1

    # Get CurrentDay
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key='CurrentDay'
    """, (user_id, conversation_id))
    row_day = cursor.fetchone()
    current_day = int(row_day[0]) if row_day else 1

    # Get TimeOfDay
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key='TimeOfDay'
    """, (user_id, conversation_id))
    row_time = cursor.fetchone()
    time_of_day = row_time[0] if row_time else "Morning"

    conn.close()
    return current_year, current_month, current_day, time_of_day

def set_current_time(user_id, conversation_id, new_year, new_month, new_day, new_phase):
    """
    Upserts CurrentYear, CurrentMonth, CurrentDay, and TimeOfDay in the CurrentRoleplay table for the given user_id and conversation_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Upsert CurrentYear
    cursor.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'CurrentYear', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, str(new_year)))

    # Upsert CurrentMonth
    cursor.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'CurrentMonth', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, str(new_month)))

    # Upsert CurrentDay
    cursor.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'CurrentDay', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, str(new_day)))

    # Upsert TimeOfDay
    cursor.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'TimeOfDay', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, new_phase))

    conn.commit()
    conn.close()

def advance_time(user_id, conversation_id, increment=1):
    """
    Advances the current time by a given number of phases.
    
    1. Reads the current year, month, day, and time_of_day.
    2. Advances the time_of_day by 'increment' steps (wrapping around if necessary).
       Every full cycle of phases increases the day by 1.
    3. If the new day exceeds DAYS_PER_MONTH, resets day to 1 and increments the month.
       If the new month exceeds MONTHS_PER_YEAR, resets month to 1 and increments the year.
    4. Saves the new year, month, day, and time_of_day back to the database.
    5. Returns (new_year, new_month, new_day, new_phase).
    """
    current_year, current_month, current_day, current_phase = get_current_time(user_id, conversation_id)

    try:
        phase_index = TIME_PHASES.index(current_phase)
    except ValueError:
        phase_index = 0

    new_index = phase_index + increment
    day_increment = new_index // len(TIME_PHASES)
    new_index = new_index % len(TIME_PHASES)

    new_phase = TIME_PHASES[new_index]
    new_day = current_day + day_increment

    new_month = current_month
    new_year = current_year

    # Check if day exceeds the maximum for the month.
    if new_day > DAYS_PER_MONTH:
        new_day = 1  # Reset day to 1
        new_month += 1  # Increment the month
        # Check if month exceeds the maximum for the year.
        if new_month > MONTHS_PER_YEAR:
            new_month = 1  # Reset month to 1
            new_year += 1  # Increment the year

    # Save the updated time values.
    set_current_time(user_id, conversation_id, new_year, new_month, new_day, new_phase)
    return new_year, new_month, new_day, new_phase

def advance_time_and_update(user_id, conversation_id, increment=1):
    """
    Combines advance_time(...) with any updates you need (such as updating NPC schedules).
    Returns (new_year, new_month, new_day, new_phase).
    """
    new_year, new_month, new_day, new_phase = advance_time(user_id, conversation_id, increment)
    update_npc_schedules_for_time(user_id, conversation_id, new_day, new_phase)
    return new_year, new_month, new_day, new_phase

def update_npc_schedules_for_time(user_id, conversation_id, day, time_of_day):
    """
    For each NPC, checks if there is an override (in PlannedEvents) for the given day and time_of_day.
    If so, uses that; otherwise, it uses the NPC’s default schedule from their schedule JSON.
    Updates each NPC's current_location accordingly.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch planned events for overrides.
    cursor.execute("""
        SELECT npc_id, override_location
        FROM PlannedEvents
        WHERE user_id=%s AND conversation_id=%s
          AND day=%s AND time_of_day=%s
    """, (user_id, conversation_id, day, time_of_day))
    override_rows = cursor.fetchall()
    override_dict = {r[0]: r[1] for r in override_rows}

    # Fetch all NPC schedules.
    cursor.execute("""
        SELECT npc_id, schedule
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    npc_rows = cursor.fetchall()

    # For each NPC, update the current_location.
    for (npc_id, schedule_json) in npc_rows:
        if npc_id in override_dict:
            new_location = override_dict[npc_id]
        else:
            if schedule_json:
                # If the schedule is stored as a JSON object, e.g.
                # {"Morning": "Cafe", "Afternoon": "Park", ...}
                # then get the location for the current time_of_day.
                new_location = schedule_json.get(time_of_day, "Unknown")
            else:
                new_location = "No schedule"

        cursor.execute("""
            UPDATE NPCStats
            SET current_location=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (new_location, user_id, conversation_id, npc_id))

    conn.commit()
    conn.close()
