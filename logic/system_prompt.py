# logic/system_prompt.py 

from db.connection import get_db_connection
import json

def build_system_prompt(player_name):
    """
    Gathers all relevant context from the DB: advanced flags, stats, current setting,
    setting history, NPC relationships, recent events, etc.
    Then builds a comprehensive system prompt so GPT can keep the narrative consistent.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Advanced flags
    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='defianceLocked'")
    row = cursor.fetchone()
    defiance_locked = (row and row[0] == "True")

    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='cravingHumiliation'")
    row = cursor.fetchone()
    craving = (row and row[0] == "True")

    # 2) Player stats
    cursor.execute("""
        SELECT corruption, confidence, willpower, obedience, dependency,
               lust, mental_resilience, physical_endurance
        FROM PlayerStats
        WHERE player_name=%s
    """, (player_name,))
    row = cursor.fetchone()
    if row:
        (corr, conf, willp, obed, dep, lust, mres, pend) = row
    else:
        (corr, conf, willp, obed, dep, lust, mres, pend) = (0,0,0,0,0,0,0,0)

    # 3) Current Setting (if you store it in CurrentRoleplay, for example)
    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='CurrentSetting'")
    row = cursor.fetchone()
    current_setting = row[0] if row else "Unknown"

    # 4) Setting history (list of used settings or events)
    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='UsedSettings'")
    row = cursor.fetchone()
    if row:
        # might be a JSON array
        try:
            used_settings = json.loads(row[0])
        except:
            used_settings = []
    else:
        used_settings = []

    # 5) NPC relationships/stats
    # For example, fetch all NPCStats for major NPCs
    cursor.execute("""
        SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity
        FROM NPCStats
        ORDER BY npc_id ASC
    """)
    npc_rows = cursor.fetchall()

    npc_info_list = []
    for r in npc_rows:
        (n_name, dom, cru, clos, tru, resp, inten) = r
        npc_info_list.append({
            "name": n_name,
            "dominance": dom,
            "cruelty": cru,
            "closeness": clos,
            "trust": tru,
            "respect": resp,
            "intensity": inten
        })

    # 6) Recent events from memory (e.g., last 5 roleplay segments or so)
    # We'll pick those with keys like "Effect_..."
    cursor.execute("""
        SELECT key, value
        FROM CurrentRoleplay
        WHERE key LIKE 'Effect_%'
        ORDER BY key DESC
        LIMIT 5
    """)
    recent_rows = cursor.fetchall()

    recent_events = []
    for (k, v) in recent_rows:
        recent_events.append(v)

    conn.close()

    # Build a textual summary
    advanced_flags_summary = f"""
    - defianceLocked={defiance_locked}
    - cravingHumiliation={craving}
    """

    stats_summary = f"""
    [Player Stats: {player_name}]
      Corruption={corr}, Confidence={conf}, Willpower={willp},
      Obedience={obed}, Dependency={dep}, Lust={lust},
      MentalResilience={mres}, PhysicalEndurance={pend}
    """

    setting_history_summary = f"Current Setting: {current_setting}\nUsed Settings: {used_settings}"

    # Summarize NPCs
    npc_lines = []
    for info in npc_info_list:
        line = (f"- {info['name']}: Dom={info['dominance']}, Cru={info['cruelty']}, "
                f"Close={info['closeness']}, Trust={info['trust']}, "
                f"Respect={info['respect']}, Int={info['intensity']}")
        npc_lines.append(line)
    npc_summary = "\n".join(npc_lines)

    recent_event_text = "\n".join([f"- {evt}" for evt in recent_events]) or "No recent events"

    # Final prompt
    prompt = f"""
    You are the DM and creative engine for a stat-driven roleplaying scenario.
    You must maintain consistency with the following data from the database:

    === ADVANCED FLAGS ===
    {advanced_flags_summary}

    === PLAYER STATS ===
    {stats_summary}

    === SETTING & HISTORY ===
    {setting_history_summary}

    === NPCS & RELATIONSHIPS ===
    {npc_summary}

    === RECENT EVENTS ===
    {recent_event_text}

    Use these details to guide your next response. 
    If defianceLocked=True, the player cannot resist or defy commands. 
    If cravingHumiliation=True, the player openly seeks humiliating tasks. 
    If Corruption>90 or Obedience>80, emphasize near-total compliance, etc.

    Feel free to create new tasks, punishments, or narrative twists, 
    referencing the relevant intensity tiers or advanced rules as needed. 
    Maintain continuity with the stats and events provided.
    """

    return prompt
