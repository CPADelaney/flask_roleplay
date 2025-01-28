# logic/aggregator.py

from db.connection import get_db_connection
import json

def get_aggregated_roleplay_context(player_name="Chase"):
    """
    Gathers everything from multiple tables: PlayerStats, NPCStats, meltdown states,
    environment from CurrentRoleplay, etc., plus SocialLinks. Returns a single Python dict
    representing the entire roleplay state.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) get day/time
    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='CurrentDay'")
    row = cursor.fetchone()
    current_day = row[0] if row else "1"

    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='TimeOfDay'")
    row = cursor.fetchone()
    time_of_day = row[0] if row else "Morning"

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

    # 3a) NPC stats
    cursor.execute("""
        SELECT npc_id, npc_name,
               dominance, cruelty, closeness, trust, respect, intensity,
               hobbies, personality_traits, likes, dislikes
        FROM NPCStats
        WHERE introduced = TRUE
        ORDER BY npc_id
    """)
    introduced_rows = cursor.fetchall()

    npc_list = []
    for row in introduced_rows:
        (nid, nname,
         dom, cru, clos, tru, resp, inten,
         hbs, pers, lks, dlks) = row

        npc_list.append({
            "npc_id": nid,
            "npc_name": nname,
            "dominance": dom,
            "cruelty": cru,
            "closeness": clos,
            "trust": tru,
            "respect": resp,
            "intensity": inten,
            "hobbies": hbs if hbs else [],
            "personality_traits": pers if pers else [],
            "likes": lks if lks else [],
            "dislikes": dlks if dlks else [],
        })

    # 3b) Unintroduced NPCs
    cursor.execute("""
        SELECT npc_id, npc_name,
               dominance, cruelty, closeness, trust, respect, intensity,
               hobbies, personality_traits, likes, dislikes
        FROM NPCStats
        WHERE introduced = FALSE
        ORDER BY npc_id
    """)
    unintroduced_rows = cursor.fetchall()
    unintroduced_list = []
    for row in npc_rows:
        (nid, nname,
         dom, cru, clos, tru, resp, inten,
         hbs, pers, lks, dlks) = row

        npc_list.append({
            "npc_id": nid,
            "npc_name": nname,
            "dominance": dom,
            "cruelty": cru,
            "closeness": clos,
            "trust": tru,
            "respect": resp,
            "intensity": inten,
            "hobbies": hbs if hbs else [],
            "personality_traits": pers if pers else [],
            "likes": lks if lks else [],
            "dislikes": dlks if dlks else [],

    # 4) Current environment / meltdown states from CurrentRoleplay
    cursor.execute("""
        SELECT key, value
        FROM CurrentRoleplay
    """)
    rows = cursor.fetchall()

    currentroleplay_data = {}
    for (k, v) in rows:
        currentroleplay_data[k] = v

    # 5) Social Links
    # We'll fetch all links. If you only want NPC↔NPC or only the ones involving "Chase," you can filter.
    cursor.execute("""
        SELECT link_id,
               entity1_type, entity1_id,
               entity2_type, entity2_id,
               link_type, link_level, link_history
        FROM SocialLinks
        ORDER BY link_id
    """)
    link_rows = cursor.fetchall()

    social_links = []
    for row in link_rows:
        (lid, e1_type, e1_id,
         e2_type, e2_id,
         link_type, link_level, link_hist) = row
        social_links.append({
            "link_id": lid,
            "entity1_type": e1_type,
            "entity1_id": e1_id,
            "entity2_type": e2_type,
            "entity2_id": e2_id,
            "link_type": link_type,
            "link_level": link_level,
            "link_history": link_hist if link_hist else []
        })

    conn.close()

    # 6) Construct the aggregator dict
    aggregated = {
        "playerStats": player_stats,
        "npcStats": npc_list,
        "currentRoleplay": currentroleplay_data,
        "day": current_day,
        "timeOfDay": time_of_day,
        "socialLinks": social_links   # <--- new key
    }

    # 7) Player social link perks
    cursor.execute("""
        SELECT perk_name, perk_description, perk_effect
        FROM PlayerPerks
        WHERE player_name=%s
    """, (player_name,))
    perk_rows = cursor.fetchall()
    
    player_perks = []
    for (p_name, p_desc, p_fx) in perk_rows:
        player_perks.append({
            "perk_name": p_name,
            "perk_description": p_desc,
            "perk_effect": p_fx
        })
    
    aggregated["playerPerks"] = player_perks
    

    return aggregated
