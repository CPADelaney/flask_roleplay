# logic/aggregator.py 
from db.connection import get_db_connection
import json

def get_aggregated_roleplay_context(player_name="Chase"):
    """
    Gathers everything from multiple tables: PlayerStats, NPCStats, meltdown states,
    environment from CurrentRoleplay, etc. Returns a single Python dict
    representing the entire roleplay state.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Player stats
    cursor.execute("""
        SELECT corruption, confidence, willpower, obedience, dependency,
               lust, mental_resilience, physical_endurance
        FROM PlayerStats
        WHERE player_name=%s
    """, (player_name,))
    row = cursor.fetchone()
    if row:
        (corr, conf, wlp, obed, dep, lust, mres, pend) = row
        player_stats = {
            "name": player_name,
            "corruption": corr,
            "confidence": conf,
            "willpower": wlp,
            "obedience": obed,
            "dependency": dep,
            "lust": lust,
            "mental_resilience": mres,
            "physical_endurance": pend
        }
    else:
        player_stats = {}

    # 2) NPC stats
    cursor.execute("""
        SELECT npc_id, npc_name, dominance, cruelty, closeness, trust, respect, intensity
        FROM NPCStats
        ORDER BY npc_id
    """)
    npc_rows = cursor.fetchall()

    npc_list = []
    for (nid, nname, dom, cru, clos, tru, resp, inten) in npc_rows:
        npc_list.append({
            "npc_id": nid,
            "npc_name": nname,
            "dominance": dom,
            "cruelty": cru,
            "closeness": clos,
            "trust": tru,
            "respect": resp,
            "intensity": inten
        })

    # 3) Current environment or meltdown states from CurrentRoleplay
    #    e.g. environment might be stored as ("CurrentSetting", "UsedSettings", etc.)
    cursor.execute("""
        SELECT key, value
        FROM CurrentRoleplay
    """)
    rows = cursor.fetchall()

    currentroleplay_data = {}
    for (k, v) in rows:
        # If you store JSON in 'value', you might parse it. Otherwise keep as string.
        # We'll do a naive approach:
        currentroleplay_data[k] = v

    conn.close()

    # 4) Construct a single aggregated dict
    #    You can structure it however you want. Example:
    aggregated = {
        "playerStats": player_stats,
        "npcStats": npc_list,
        "currentRoleplay": currentroleplay_data
    }

    # Optionally add meltdown if you store meltdown level or meltdown events in a separate table
    # e.g. meltdown table:
    # meltdown_data = ...
    # aggregated["meltdownStates"] = meltdown_data

    return aggregated
