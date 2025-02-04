import json
from db.connection import get_db_connection

def get_aggregated_roleplay_context(user_id, conversation_id, player_name):
    """
    Gathers data from multiple tables, merges them into a single aggregator dict,
    and also manages an incremental 'GlobalSummary' to reduce context size.

    For unintroduced NPCs, we only store minimal info:
      - npc_id, npc_name, location, day-slice schedule
    That ensures the player can potentially encounter them in the world,
    but we don't load the rest of their stats yet.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    #----------------------------------------------------------------
    # 0) Retrieve existing summary from CurrentRoleplay (if any)
    #----------------------------------------------------------------
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s
          AND conversation_id=%s
          AND key='GlobalSummary'
    """, (user_id, conversation_id))
    row = cursor.fetchone()
    existing_summary = row[0] if row else ""

    #----------------------------------------------------------------
    # 1) Day/time from CurrentRoleplay
    #----------------------------------------------------------------
    current_day = "1"
    time_of_day = "Morning"

    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s
          AND conversation_id=%s
          AND key='CurrentDay'
    """, (user_id, conversation_id))
    row = cursor.fetchone()
    if row:
        current_day = row[0]

    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s
          AND conversation_id=%s
          AND key='TimeOfDay'
    """, (user_id, conversation_id))
    row = cursor.fetchone()
    if row:
        time_of_day = row[0]

    #----------------------------------------------------------------
    # 2) Player Stats
    #----------------------------------------------------------------
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
    player_stats = {}
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

    #----------------------------------------------------------------
    # 3) NPC Stats (Introduced)
    #----------------------------------------------------------------
    introduced_npcs = []

    cursor.execute("""
        SELECT npc_id, npc_name,
               dominance, cruelty, closeness,
               trust, respect, intensity,
               hobbies, personality_traits, likes, dislikes,
               schedule, current_location
        FROM NPCStats
        WHERE user_id=%s
          AND conversation_id=%s
          AND introduced=TRUE
        ORDER BY npc_id
    """, (user_id, conversation_id))
    introduced_rows = cursor.fetchall()
    for (nid, nname, dom, cru, clos, tru, resp, inten,
         hbs, pers, lks, dlks, sched, curr_loc) in introduced_rows:

        # Trim schedule to the current day
        trimmed_schedule = {}
        if sched:
            try:
                full_sched = json.loads(sched)
                if current_day in full_sched:
                    trimmed_schedule[current_day] = full_sched[current_day]
            except:
                pass

        introduced_npcs.append({
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
            "dislikes": dlks or [],
            "schedule": trimmed_schedule,
            "current_location": curr_loc or "Unknown"
        })

    #----------------------------------------------------------------
    # 4) NPC Minimal Info (Unintroduced)
    #----------------------------------------------------------------
    unintroduced_npcs = []
    cursor.execute("""
        SELECT npc_id, npc_name,
               schedule, current_location
        FROM NPCStats
        WHERE user_id=%s
          AND conversation_id=%s
          AND introduced=FALSE
        ORDER BY npc_id
    """, (user_id, conversation_id))
    unintroduced_rows = cursor.fetchall()
    for (nid, nname, sched, curr_loc) in unintroduced_rows:
        trimmed_schedule = {}
        if sched:
            try:
                full_sched = json.loads(sched)
                if current_day in full_sched:
                    trimmed_schedule[current_day] = full_sched[current_day]
            except:
                pass
        unintroduced_npcs.append({
            "npc_id": nid,
            "npc_name": nname,
            "current_location": curr_loc or "Unknown",
            "schedule": trimmed_schedule  # minimal day-by-day
        })

    #----------------------------------------------------------------
    # 5) Social Links
    #----------------------------------------------------------------
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

    #----------------------------------------------------------------
    # 6) PlayerPerks
    #----------------------------------------------------------------
    cursor.execute("""
        SELECT perk_name, perk_description, perk_effect
        FROM PlayerPerks
        WHERE user_id=%s
          AND conversation_id=%s
    """, (user_id, conversation_id))
    perk_rows = cursor.fetchall()
    player_perks = []
    for (p_name, p_desc, p_fx) in perk_rows:
        player_perks.append({
            "perk_name": p_name,
            "perk_description": p_desc,
            "perk_effect": p_fx
        })

    #----------------------------------------------------------------
    # 7) Inventory
    #----------------------------------------------------------------
    cursor.execute("""
        SELECT player_name, item_name, item_description, item_effect,
               quantity, category
        FROM PlayerInventory
        WHERE user_id=%s
          AND conversation_id=%s
    """, (user_id, conversation_id))
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

    #----------------------------------------------------------------
    # 8) Events
    #----------------------------------------------------------------
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

    #----------------------------------------------------------------
    # 9) PlannedEvents
    #----------------------------------------------------------------
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

    #----------------------------------------------------------------
    # 10) Quests
    #----------------------------------------------------------------
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

    #----------------------------------------------------------------
    # 11) Global tables: GameRules, StatDefinitions
    #----------------------------------------------------------------
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

    #----------------------------------------------------------------
    # 12) CurrentRoleplay details
    #----------------------------------------------------------------
    cursor.execute("""
        SELECT key, value
        FROM CurrentRoleplay
        WHERE user_id=%s
          AND conversation_id=%s
    """, (user_id, conversation_id))
    all_rows = cursor.fetchall()

    currentroleplay_data = {}
    for (k, v) in all_rows:
        if k == "ChaseSchedule":
            try:
                currentroleplay_data[k] = json.loads(v)
            except:
                currentroleplay_data[k] = v
        else:
            currentroleplay_data[k] = v

    conn.close()

    #----------------------------------------------------------------
    # 13) Build a final aggregator dictionary
    #----------------------------------------------------------------
    aggregated = {
        "playerStats": player_stats,
        "introducedNPCs": introduced_npcs,   # Full stats for introduced
        "unintroducedNPCs": unintroduced_npcs,  # Minimal info for unintroduced
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

    #----------------------------------------------------------------
    # 14) Summarize new changes & update 'GlobalSummary' in DB
    #----------------------------------------------------------------
    new_changes_summary = build_changes_summary(aggregated)
    if new_changes_summary.strip():
        updated_summary = update_global_summary(existing_summary, new_changes_summary)
        conn2 = get_db_connection()
        c2 = conn2.cursor()
        c2.execute("""
            INSERT INTO CurrentRoleplay(user_id, conversation_id, key, value)
            VALUES(%s, %s, 'GlobalSummary', %s)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, (user_id, conversation_id, updated_summary))
        conn2.commit()
        conn2.close()
        existing_summary = updated_summary

    aggregator_text = (
        f"{existing_summary}\n\n"
        f"Day {current_day}, {time_of_day}.\n"
        "Scene Snapshot:\n"
        f"{make_minimal_scene_info(aggregated)}"
    )

    aggregated["short_summary"] = existing_summary
    aggregated["aggregator_text"] = aggregator_text

    return aggregated


#--------------------- HELPER FUNCTIONS ---------------------#

def build_changes_summary(aggregated):
    """
    Quick example: mention how many introduced/unintroduced NPCs, 
    or any quest changes, etc.
    """
    lines = []

    introduced_count = len(aggregated["introducedNPCs"])
    unintroduced_count = len(aggregated["unintroducedNPCs"])

    lines.append(f"Introduced NPCs: {introduced_count}, Unintroduced: {unintroduced_count}")

    # Possibly mention major quest changes or new items, etc.
    # For brevity, we'll skip that.
    
    text = "\n".join(lines)
    if introduced_count == 0 and unintroduced_count == 0:
        text = ""  # if no real changes
    return text


def update_global_summary(old_summary, new_stuff, max_len=3000):
    combined = old_summary.strip() + "\n\n" + new_stuff.strip()
    if len(combined) > max_len:
        combined = combined[-max_len:]
    return combined


def make_minimal_scene_info(aggregated):
    """
    Provide a small snippet: day/time, maybe top 2 introduced NPCs + 
    top 2 unintroduced with location, so GPT can handle chance encounters, etc.
    """
    lines = []
    day = aggregated["day"]
    tod = aggregated["timeOfDay"]
    lines.append(f"- It is Day {day}, {tod}.\n")

    # Introduced NPCs snippet
    lines.append("Introduced NPCs in the area:")
    for npc in aggregated["introducedNPCs"][:2]:
        loc = npc.get("current_location", "Unknown")
        lines.append(f"  - {npc['npc_name']} is at {loc}")

    # Unintroduced snippet
    lines.append("Unintroduced NPCs (possible random encounters):")
    for npc in aggregated["unintroducedNPCs"][:2]:
        loc = npc.get("current_location", "Unknown")
        lines.append(f"  - ???: '{npc['npc_name']}' lurking around {loc}")

    return "\n".join(lines)
