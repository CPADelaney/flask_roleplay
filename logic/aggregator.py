# logic/aggregator.py

from db.connection import get_db_connection
import json

def get_aggregated_roleplay_context(player_name="Chase"):
    """
    Gathers everything from multiple tables: PlayerStats, NPCStats, meltdown states,
    environment from CurrentRoleplay, plus SocialLinks, PlayerPerks,
    PlayerInventory (all?), Events, PlannedEvents, Locations, GameRules, Quests, and StatDefinitions.
    Returns a single Python dict representing the entire roleplay state.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Current day/time
    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='CurrentDay'")
    row = cursor.fetchone()
    current_day = row[0] if row else "1"

    cursor.execute("SELECT value FROM CurrentRoleplay WHERE key='TimeOfDay'")
    row = cursor.fetchone()
    time_of_day = row[0] if row else "Morning"

    # 2) Player Stats
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

    # 3a) Introduced NPCs
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
        (nid, nname, dom, cru, clos, tru, resp, inten, hbs, pers, lks, dlks) = row
        npc_list.append({
            "npc_id": nid,
            "npc_name": nname,
            "dominance": dom,
            "cruelty": cru,
            "closeness": clos,
            "trust": tru,
            "respect": resp,
            "intensity": inten,
            "hobbies": hbs or [],
            "personality_traits": pers or [],
            "likes": lks or [],
            "dislikes": dlks or []
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
    for row in unintroduced_rows:
        (nid, nname, dom, cru, clos, tru, resp, inten, hbs, pers, lks, dlks) = row
        npc_list.append({
            "npc_id": nid,
            "npc_name": nname,
            "dominance": dom,
            "cruelty": cru,
            "closeness": clos,
            "trust": tru,
            "respect": resp,
            "intensity": inten,
            "hobbies": hbs or [],
            "personality_traits": pers or [],
            "likes": lks or [],
            "dislikes": dlks or []
        })

    # 4) CurrentRoleplay data
    cursor.execute("""
        SELECT key, value
        FROM CurrentRoleplay
    """)
    rows = cursor.fetchall()
    currentroleplay_data = {}
    for (k, v) in rows:
        currentroleplay_data[k] = v

    # 5) SocialLinks
    cursor.execute("""
        SELECT link_id, entity1_type, entity1_id,
               entity2_type, entity2_id, link_type, link_level, link_history
        FROM SocialLinks
        ORDER BY link_id
    """)
    link_rows = cursor.fetchall()
    social_links = []
    for (lid, e1t, e1i, e2t, e2i, ltype, lvl, hist) in link_rows:
        social_links.append({
            "link_id": lid,
            "entity1_type": e1t,
            "entity1_id": e1i,
            "entity2_type": e2t,
            "entity2_id": e2i,
            "link_type": ltype,
            "link_level": lvl,
            "link_history": hist or []
        })

    # 6) PlayerPerks
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

    # 7) PlayerInventory (All or just for 'Chase'?)
    # If you want to see *all* items for all players, remove WHERE. If only for Chase:
    cursor.execute("""
        SELECT player_name, item_name, item_description, item_effect, quantity, category
        FROM PlayerInventory
        WHERE player_name = %s
    """, (player_name,))
    inv_rows = cursor.fetchall()
    inventory_list = []
    for (p_n, iname, idesc, ieffect, qty, cat) in inv_rows:
        inventory_list.append({
            "player_name": p_n,
            "item_name": iname,
            "item_description": idesc,
            "item_effect": ieffect,
            "quantity": qty,
            "category": cat
        })

    # 8) Events
    cursor.execute("""
        SELECT id, event_name, description, start_time, end_time, location
        FROM Events
        ORDER BY id
    """)
    events_list = []
    for row in cursor.fetchall():
        (eid, ename, edesc, stime, etime, loc) = row
        events_list.append({
            "event_id": eid,
            "event_name": ename,
            "description": edesc,
            "start_time": stime,
            "end_time": etime,
            "location": loc
        })

    # 9) PlannedEvents
    cursor.execute("""
        SELECT event_id, npc_id, day, time_of_day, override_location
        FROM PlannedEvents
        ORDER BY event_id
    """)
    planned_events_list = []
    for row in cursor.fetchall():
        (eid, npc_id, day, tod, ov_loc) = row
        planned_events_list.append({
            "event_id": eid,
            "npc_id": npc_id,
            "day": day,
            "time_of_day": tod,
            "override_location": ov_loc
        })

    # 10) GameRules
    cursor.execute("""
        SELECT rule_name, condition, effect
        FROM GameRules
        ORDER BY rule_name
    """)
    game_rules_list = []
    for row in cursor.fetchall():
        (rname, cond, eff) = row
        game_rules_list.append({
            "rule_name": rname,
            "condition": cond,
            "effect": eff
        })

    # 11) Quests
    cursor.execute("""
        SELECT quest_id, quest_name, status, progress_detail, quest_giver, reward
        FROM Quests
        ORDER BY quest_id
    """)
    quest_list = []
    for row in cursor.fetchall():
        (qid, qname, qstatus, qdetail, qgiver, qreward) = row
        quest_list.append({
            "quest_id": qid,
            "quest_name": qname,
            "status": qstatus,
            "progress_detail": qdetail,
            "quest_giver": qgiver,
            "reward": qreward
        })

    # 12) StatDefinitions
    cursor.execute("""
        SELECT id, scope, stat_name, range_min, range_max, definition, effects, progression_triggers
        FROM StatDefinitions
        ORDER BY id
    """)
    stat_def_list = []
    for row in cursor.fetchall():
        (sid, scp, sname, rmin, rmax, sdef, seff, sprg) = row
        stat_def_list.append({
            "id": sid,
            "scope": scp,
            "stat_name": sname,
            "range_min": rmin,
            "range_max": rmax,
            "definition": sdef,
            "effects": seff,
            "progression_triggers": sprg
        })

    conn.close()

    # Final aggregator_data
    aggregated = {
        # Existing entries
        "playerStats": player_stats,
        "npcStats": npc_list,
        "currentRoleplay": currentroleplay_data,
        "day": current_day,
        "timeOfDay": time_of_day,
        "socialLinks": social_links,
        "playerPerks": player_perks,

        # Newly added
        "inventory": inventory_list,
        "events": events_list,
        "plannedEvents": planned_events_list,
        "gameRules": game_rules_list,
        "quests": quest_list,
        "statDefinitions": stat_def_list
    }

    return aggregated
