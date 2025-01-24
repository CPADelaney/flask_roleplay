# logic/time_cycle.py

from db.connection import get_db_connection

TIME_PHASES = ["Morning", "Afternoon", "Evening", "Night"]

def get_current_daytime():
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
    current_day, current_phase = get_current_daytime()

    try:
        phase_index = TIME_PHASES.index(current_phase)
    except ValueError:
        phase_index = 0  # fallback if unknown

    new_index = phase_index + increment

    day_increment = new_index // len(TIME_PHASES)
    new_index = new_index % len(TIME_PHASES)

    new_phase = TIME_PHASES[new_index]
    new_day = current_day + day_increment

    set_current_daytime(new_day, new_phase)
    return new_day, new_phase

def advance_time_and_update(increment=1):
    new_day, new_phase = advance_time(increment)
    update_npc_schedules_for_time(new_day, new_phase)
    return new_day, new_phase

def update_npc_schedules_for_time(day, time_of_day):
    """
    SINGLE unified function that:
    - Looks for any 'PlannedEvents' overrides for this day/time
    - Otherwise uses the npc's default schedule
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # fetch planned events for overrides
    cursor.execute("""
        SELECT npc_id, override_location
        FROM PlannedEvents
        WHERE day = %s AND time_of_day = %s
    """, (day, time_of_day))
    override_rows = cursor.fetchall()
    override_dict = {r[0]: r[1] for r in override_rows}

    # fetch all npc schedules
    cursor.execute("""
        SELECT npc_id, schedule
        FROM NPCStats
    """)
    npc_rows = cursor.fetchall()

    for (npc_id, schedule_json) in npc_rows:
        if npc_id in override_dict:
            # use override
            new_location = override_dict[npc_id]
        else:
            # fallback to default schedule
            if schedule_json:
                # schedule_json is e.g. { "Morning": "Cafe", "Afternoon": "Park", ... }
                new_location = schedule_json.get(time_of_day, "Unknown")
            else:
                new_location = "No schedule"

        cursor.execute("""
            UPDATE NPCStats
            SET current_location = %s
            WHERE npc_id = %s
        """, (new_location, npc_id))

    conn.commit()
    conn.close()
