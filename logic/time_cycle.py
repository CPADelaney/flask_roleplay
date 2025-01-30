from db.connection import get_db_connection

# If you prefer a fixed list of phases:
TIME_PHASES = ["Morning", "Afternoon", "Evening", "Night"]

def get_current_daytime(user_id, conversation_id):
    """
    Returns (current_day, time_of_day) for this user_id + conversation_id,
    from CurrentRoleplay table where:
        user_id=? AND conversation_id=? AND key in ('CurrentDay','TimeOfDay')
    If no rows, defaults to day=1, phase="Morning".
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Grab 'CurrentDay' for this user/convo
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s
          AND key='CurrentDay'
    """, (user_id, conversation_id))
    row_day = cursor.fetchone()
    current_day = int(row_day[0]) if row_day else 1

    # Grab 'TimeOfDay'
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s
          AND key='TimeOfDay'
    """, (user_id, conversation_id))
    row_time = cursor.fetchone()
    time_of_day = row_time[0] if row_time else "Morning"

    conn.close()
    return current_day, time_of_day

def set_current_daytime(user_id, conversation_id, new_day, new_phase):
    """
    Upserts 'CurrentDay' and 'TimeOfDay' in CurrentRoleplay
    for the given user/conversation.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

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
    1) Reads current day/time from (user_id, conversation_id).
    2) Advances 'time_of_day' by 'increment' phases (wrapping around).
       Each full cycle of phases increments the day by 1.
    3) Saves the updated day/time back to CurrentRoleplay.
    4) Returns (new_day, new_phase).
    """
    current_day, current_phase = get_current_daytime(user_id, conversation_id)

    try:
        phase_index = TIME_PHASES.index(current_phase)
    except ValueError:
        # If unknown phase, fallback to 0
        phase_index = 0

    new_index = phase_index + increment
    day_increment = new_index // len(TIME_PHASES)
    new_index = new_index % len(TIME_PHASES)

    new_phase = TIME_PHASES[new_index]
    new_day = current_day + day_increment

    # Save them
    set_current_daytime(user_id, conversation_id, new_day, new_phase)
    return new_day, new_phase

def advance_time_and_update(user_id, conversation_id, increment=1):
    """
    Combines advance_time(...) + update_npc_schedules_for_time(...).
    i.e. increments day/phase, then updates NPCs' current_location
    based on new day/time for that user & conversation.
    """
    new_day, new_phase = advance_time(user_id, conversation_id, increment)
    update_npc_schedules_for_time(user_id, conversation_id, new_day, new_phase)
    return new_day, new_phase

def update_npc_schedules_for_time(user_id, conversation_id, day, time_of_day):
    """
    1) Checks PlannedEvents for overrides (where user_id=? AND conversation_id=? AND day=? AND time_of_day=?).
    2) Otherwise uses the npc's default schedule from NPCStats(schedule).
    3) Updates NPCStats.current_location for each relevant NPC.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Fetch planned events for overrides
    cursor.execute("""
        SELECT npc_id, override_location
        FROM PlannedEvents
        WHERE user_id=%s AND conversation_id=%s
          AND day=%s AND time_of_day=%s
    """, (user_id, conversation_id, day, time_of_day))
    override_rows = cursor.fetchall()

    # Build a dict { npc_id: override_location }
    override_dict = {r[0]: r[1] for r in override_rows}

    # 2) Fetch all NPC schedules for this user+conversation
    cursor.execute("""
        SELECT npc_id, schedule
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    npc_rows = cursor.fetchall()

    # 3) For each NPC, decide new_location
    for (npc_id, schedule_json) in npc_rows:
        if npc_id in override_dict:
            # Use override
            new_location = override_dict[npc_id]
        else:
            # fallback to default schedule in npc's schedule JSON
            if schedule_json:
                # schedule_json is e.g. { "Morning": "Cafe", "Afternoon": "Park", ... }
                new_location = schedule_json.get(time_of_day, "Unknown")
            else:
                new_location = "No schedule"

        # 4) Update NPCStats.current_location
        cursor.execute("""
            UPDATE NPCStats
            SET current_location=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (new_location, user_id, conversation_id, npc_id))

    conn.commit()
    conn.close()
