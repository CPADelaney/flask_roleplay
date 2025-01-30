import json
from db.connection import get_db_connection

def get_aggregated_roleplay_context(user_id, conversation_id, player_name="Chase"):
    """
    Gathers data from multiple tables, all scoped by (user_id, conversation_id) plus player_name where needed.
    Returns a single dict representing the entire roleplay state.

    This presumes:
    - CurrentRoleplay, NPCStats, SocialLinks, etc. each have columns user_id, conversation_id.
    - PlayerStats, PlayerPerks, PlayerInventory also have user_id, conversation_id, plus a 'player_name'.
    - Some “global” tables like GameRules, StatDefinitions remain unscoped or partially scoped.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # ----------------------------------------------------------------
    # 1) Day/time from CurrentRoleplay
    # ----------------------------------------------------------------
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s
          AND conversation_id=%s
          AND key='CurrentDay'
    """, (user_id, conversation_id))
    row = cursor.fetchone()
    current_day = row[0] if row else "1"

    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s
          AND conversation_id=%s
          AND key='TimeOfDay'
    """, (user_id, conversation_id))
    row = cursor.fetchone()
    time_of_day = row[0] if row else "Morning"

    # ----------------------------------------------------------------
    # 2) Player Stats (scoped by user_id, conversation_id, player_name)
    # ----------------------------------------------------------------
    cursor.execute("""
        SELECT corruption, confidence, willpower,
               obedience, dependency, lust,
               mental_resilience, physical_endurance
        FROM PlayerStats
        WHERE user_id=%s
          AND conversation_id=%s
          AND player_name=%s
    """, (user_id, conversation_id, player_name))
    row = cursor.fetchone()
    if row:
        (corr, conf, wlp, obed, dep, lst, mres, pend) = row
        player_stats = {
            "name": player_name,
            "corruption": corr,
            "confidence": conf,
            "willpower": wlp,
            "obedience": obed,
            "dependency": dep,
            "lust": lst,
            "mental_resilience": mres,
            "physical_endurance": pend
        }
    else:
        player_stats = {}

    # ----------------------------------------------------------------
    # 3) NPC Stats (Introduced & Unintroduced) 
    #    (scoped by user_id, conversation_id)
    # ----------------------------------------------------------------
    npc_list = []

    # 3a) Introduced NPCs
    cursor.execute("""
        SELECT npc_id, npc_name,
               dominance, cruelty, closeness,
               trust, respect, intensity,
               hobbies, personality_traits, likes, dislikes
        FROM NPCStats
        WHERE user_id=%s
          AND conversation_id=%s
          AND introduced=TRUE
        ORDER BY npc_id
    """, (user_id, conversation_id))
    introduced_rows = cursor.fetchall()
    for (nid, nname, dom, cru, clos, tru, resp, inten, hbs, pers, lks, dlks) in introduced_rows:
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
               dominance, cruelty, closeness,
               trust, respect, intensity,
               hobbies, personality_traits, likes, dislikes
        FROM NPCStats
        WHERE user_id=%s
          AND conversation_id=%s
          AND introduced=FALSE
        ORDER BY npc_id
    """, (user_id, conversation_id))
    unintroduced_rows = cursor.fetchall()
    for (nid, nname, dom, cru, clos, tru, resp, inten, hbs, pers, lks, dlks) in unintroduced_rows:
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

    # ----------------------------------------------------------------
    # 4) SocialLinks (scoped by user_id, conversation_id)
    # ----------------------------------------------------------------
    cursor.execute("""
        SELECT link_id, entity1_type, entity1_id,
               entity2_type, entity2_id,
               link_type, link_level, link_history
        FROM SocialLinks
        WHERE user_id=%s
          AND conversation_id=%s
        ORDER BY link_id
    """, (user_id, conversation_id))
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

    # ----------------------------------------------------------------
    # 5) PlayerPerks (scoped by user_id, conversation_id, player_name)
    # ----------------------------------------------------------------
    cursor.execute("""
        SELECT perk_name, perk_description, perk_effect
        FROM PlayerPerks
        WHERE user_id=%s
          AND conversation_id=%s
          AND player_name=%s
    """, (user_id, conversation_id, player_name))
    perk_rows = cursor.fetchall()
    player_perks = []
    for (p_name, p_desc, p_fx) in perk_rows:
        player_perks.append({
            "perk_name": p_name,
            "perk_description": p_desc,
            "perk_effect": p_fx
        })

    # ----------------------------------------------------------------
    # 6) PlayerInventory (scoped by user_id, conversation_id, player_name)
    # ----------------------------------------------------------------
    cursor.execute("""
        SELECT player_name, item_name, item_description, item_effect,
               quantity, category
        FROM PlayerInventory
        WHERE user_id=%s
          AND conversation_id=%s
          AND player_name=%s
    """, (user_id, conversation_id, player_name))
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

    # ----------------------------------------------------------------
    # 7) Events (scoped by user_id, conversation_id) if you want them unique
    # ----------------------------------------------------------------
    cursor.execute("""
        SELECT id, event_name, description, start_time, end_time, location
        FROM Events
        WHERE user_id=%s
          AND conversation_id=%s
        ORDER BY id
    """, (user_id, conversation_id))
    events_list = []
    for (eid, ename, edesc, stime, etime, loc) in cursor.fetchall():
        events_list.append({
            "event_id": eid,
            "event_name": ename,
            "description": edesc,
            "start_time": stime,
            "end_time": etime,
            "location": loc
        })

    # ----------------------------------------------------------------
    # 8) PlannedEvents (scoped by user_id, conversation_id)
    # ----------------------------------------------------------------
    cursor.execute("""
        SELECT event_id, npc_id, day, time_of_day, override_location
        FROM PlannedEvents
        WHERE user_id=%s
          AND conversation_id=%s
        ORDER BY event_id
    """, (user_id, conversation_id))
    planned_events_list = []
    for (eid, npc_id, day, tod, ov_loc) in cursor.fetchall():
        planned_events_list.append({
            "event_id": eid,
            "npc_id": npc_id,
            "day": day,
            "time_of_day": tod,
            "override_location": ov_loc
        })

    # ----------------------------------------------------------------
    # 9) Quests (scoped by user_id, conversation_id) if you want them separate
    # ----------------------------------------------------------------
    cursor.execute("""
        SELECT quest_id, quest_name, status, progress_detail,
               quest_giver, reward
        FROM Quests
        WHERE user_id=%s
          AND conversation_id=%s
        ORDER BY quest_id
    """, (user_id, conversation_id))
    quest_list = []
    for (qid, qname, qstatus, qdetail, qgiver, qreward) in cursor.fetchall():
        quest_list.append({
            "quest_id": qid,
            "quest_name": qname,
            "status": qstatus,
            "progress_detail": qdetail,
            "quest_giver": qgiver,
            "reward": qreward
        })

    # ----------------------------------------------------------------
    # 10) Some tables might be "global" (GameRules, StatDefinitions) 
    #     no user_id scoping
    # ----------------------------------------------------------------

    # 10a) GameRules
    cursor.execute("""
        SELECT rule_name, condition, effect
        FROM GameRules
        ORDER BY rule_name
    """)
    game_rules_list = []
    for (rname, cond, eff) in cursor.fetchall():
        game_rules_list.append({
            "rule_name": rname,
            "condition": cond,
            "effect": eff
        })

    # 10b) StatDefinitions
    cursor.execute("""
        SELECT id, scope, stat_name, range_min, range_max,
               definition, effects, progression_triggers
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

    # ----------------------------------------------------------------
    # 11) CurrentRoleplay details
    # ----------------------------------------------------------------
    cursor.execute("""
        SELECT key, value
        FROM CurrentRoleplay
        WHERE user_id=%s
          AND conversation_id=%s
    """, (user_id, conversation_id))
    all_rows = cursor.fetchall()

    currentroleplay_data = {}
    for (k, v) in all_rows:
        # If you store JSON in v, you might try to parse it:
        if k == "ChaseSchedule":
            try:
                currentroleplay_data[k] = json.loads(v)
            except:
                currentroleplay_data[k] = v
        else:
            currentroleplay_data[k] = v

    conn.close()

    # ----------------------------------------------------------------
    # 12) Build final aggregator response
    # ----------------------------------------------------------------
    aggregated = {
        "playerStats": player_stats,
        "npcStats": npc_list,
        "currentRoleplay": currentroleplay_data,
        "day": current_day,
        "timeOfDay": time_of_day,
        "socialLinks": social_links,
        "playerPerks": player_perks,
        "inventory": inventory_list,
        "events": events_list,
        "plannedEvents": planned_events_list,
        "gameRules": game_rules_list,
        "quests": quest_list,
        "statDefinitions": stat_def_list
    }

    return aggregated
