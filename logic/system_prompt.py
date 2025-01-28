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
        (corr, conf, wlp, obed, dep, lust, mres, pend) = row
    else:
        (corr, conf, wlp, obed, dep, lust, mres, pend) = (0,0,0,0,0,0,0,0)

    # 3) Current Setting
    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='CurrentSetting'")
    row = cursor.fetchone()
    current_setting = row[0] if row else "Unknown"

    # 4) Setting history
    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='UsedSettings'")
    row = cursor.fetchone()
    if row:
        try:
            used_settings = json.loads(row[0])
        except:
            used_settings = []
    else:
        used_settings = []

    # 5) Summarize introduced NPCs
    cursor.execute("""
        SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity,
               hobbies, personality_traits, likes, dislikes
        FROM NPCStats
        WHERE introduced = TRUE
        ORDER BY npc_id ASC
    """)
    npc_rows = cursor.fetchall()

    npc_lines = []
    for row in npc_rows:
        (n_name, dom, cru, clos, tru, resp, inten, hbs, pers, lks, dlks) = row
        line = f"- {n_name}: Dom={dom}, Cru={cru}, Close={clos}, Trust={tru}, Respect={resp}, Int={inten}\n"
        line += f"  Hobbies: {hbs if hbs else []}\n"
        line += f"  Personality: {pers if pers else []}\n"
        line += f"  Likes: {lks if lks else []} | Dislikes: {dlks if dlks else []}"
        npc_lines.append(line)
    npc_summary = "\n\n".join(npc_lines) if npc_lines else "(No introduced NPCs)"

    # 6) Recent events
    cursor.execute("""
        SELECT key, value
        FROM CurrentRoleplay
        WHERE key LIKE 'Effect_%'
        ORDER BY key DESC
        LIMIT 5
    """)
    recent_rows = cursor.fetchall()
    recent_events = [v for (k,v) in recent_rows]
    
    # 7) SocialLinks summary
    # We'll fetch all or a subset
    cursor.execute("""
        SELECT link_id, entity1_type, entity1_id,
               entity2_type, entity2_id,
               link_type, link_level
        FROM SocialLinks
        ORDER BY link_id ASC
    """)
    link_rows = cursor.fetchall()

    link_summaries = []
    for (lid, e1t, e1i, e2t, e2i, ltype, lvl) in link_rows:
        link_summaries.append(
            f"Link#{lid}: ({e1t}{e1i} <-> {e2t}{e2i}) type={ltype}, level={lvl}"
        )

    conn.close()

    # Build textual sections
    advanced_flags_summary = f"""
    - defianceLocked={defiance_locked}
    - cravingHumiliation={craving}
    """

    stats_summary = f"""
    [Player Stats: {player_name}]
      Corruption={corr}, Confidence={conf}, Willpower={wlp},
      Obedience={obed}, Dependency={dep}, Lust={lust},
      MentalResilience={mres}, PhysicalEndurance={pend}
    """

    setting_history_summary = f"Current Setting: {current_setting}\nUsed Settings: {used_settings}"
    recent_event_text = "\n".join([f"- {evt}" for evt in recent_events]) or "No recent events"
    social_links_text = "\n".join(link_summaries) if link_summaries else "(No social links)"

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

    === SOCIAL LINKS ===
    {social_links_text}

    === RECENT EVENTS ===
    {recent_event_text}

    Use these details to guide your next response. 
    - If defianceLocked=True, the player cannot resist or defy commands.
    - If cravingHumiliation=True, the player openly seeks humiliating tasks.
    - If Corruption>90 or Obedience>80, emphasize near-total compliance, etc.

    Feel free to create new tasks, punishments, or narrative twists, 
    referencing intensity tiers or advanced rules as needed, 
    but maintain continuity with the stats and events provided.
    """
    return prompt
