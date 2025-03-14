# logic/aggregator.py

import json
from db.connection import get_db_connection
from logic.calendar import update_calendar_names, load_calendar_names
import logging

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
    # 1) Retrieve time info from CurrentRoleplay (or fallback to immersive defaults)
    #----------------------------------------------------------------
    current_year = "1040"   # Default to a more immersive starting year
    current_month = "6"     # Default to mid-year (e.g., 6th month)
    current_day = "15"      # Default to mid-month
    time_of_day = "Morning"

    for key in ["CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"]:
        cursor.execute("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=%s AND conversation_id=%s AND key=%s
        """, (user_id, conversation_id, key))
        row = cursor.fetchone()
        if row:
            if key == "CurrentYear":
                current_year = row[0]
            elif key == "CurrentMonth":
                current_month = row[0]
            elif key == "CurrentDay":
                current_day = row[0]
            elif key == "TimeOfDay":
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
               schedule, current_location, physical_description, archetype_extras_summary
        FROM NPCStats
        WHERE user_id=%s
          AND conversation_id=%s
          AND introduced=TRUE
        ORDER BY npc_id
    """, (user_id, conversation_id))
    introduced_rows = cursor.fetchall()
    for (nid, nname, dom, cru, clos, tru, resp, inten,
         hbs, pers, lks, dlks, sched, curr_loc, phys_desc, extras) in introduced_rows:
        try:
            trimmed_schedule = json.loads(sched) if sched else {}
        except Exception:
            trimmed_schedule = {}
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
            "current_location": curr_loc or "Unknown",
            "physical_description": phys_desc or "",
            "archetype_extras_summary": extras or ""
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
        try:
            trimmed_schedule = json.loads(sched) if sched else {}
        except Exception:
            trimmed_schedule = {}
        unintroduced_npcs.append({
            "npc_id": nid,
            "npc_name": nname,
            "current_location": curr_loc or "Unknown",
            "schedule": trimmed_schedule
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
    # 8) Events (now including full date info)
    #----------------------------------------------------------------
    cursor.execute("""
        SELECT id, event_name, description, start_time, end_time, location,
               year, month, day, time_of_day
        FROM Events
        WHERE user_id=%s
          AND conversation_id=%s
        ORDER BY id
    """, (user_id, conversation_id))
    events_list = []
    for (eid, ename, edesc, stime, etime, loc, eyear, emonth, eday, etod) in cursor.fetchall():
        events_list.append({
            "event_id": eid,
            "event_name": ename,
            "description": edesc,
            "start_time": stime,
            "end_time": etime,
            "location": loc,
            "year": eyear,
            "month": emonth,
            "day": eday,
            "time_of_day": etod
        })

    #----------------------------------------------------------------
    # 9) PlannedEvents (now including full date info)
    #----------------------------------------------------------------
    cursor.execute("""
        SELECT event_id, npc_id, year, month, day, time_of_day, override_location
        FROM PlannedEvents
        WHERE user_id=%s
          AND conversation_id=%s
        ORDER BY event_id
    """, (user_id, conversation_id))
    planned_events_list = []
    for (eid, npc_id, pyear, pmonth, pday, ptod, ov_loc) in cursor.fetchall():
        planned_events_list.append({
            "event_id": eid,
            "npc_id": npc_id,
            "year": pyear,
            "month": pmonth,
            "day": pday,
            "time_of_day": ptod,
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
    # 12) Locations (New)
    #----------------------------------------------------------------
    cursor.execute("""
        SELECT id, location_name, description
        FROM Locations
        WHERE user_id=%s
          AND conversation_id=%s
        ORDER BY id
    """, (user_id, conversation_id))
    locations_list = []
    for (lid, lname, ldesc) in cursor.fetchall():
        locations_list.append({
            "location_id": lid,
            "location_name": lname,
            "description": ldesc
        })

    #----------------------------------------------------------------
    # 13) CurrentRoleplay details
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
            except Exception:
                currentroleplay_data[k] = v
        else:
            currentroleplay_data[k] = v

    conn.close()

    #----------------------------------------------------------------
    # 14) Build a final aggregator dictionary
    #----------------------------------------------------------------

    calendar_raw = currentroleplay_data.get("CalendarNames", None)
    if isinstance(calendar_raw, str):
        try:
            calendar = json.loads(calendar_raw)
        except Exception as e:
            logging.warning(f"Failed to parse CalendarNames: {e}")
            calendar = {"year_name": "Year 1", "months": [], "days": []}
    else:
        calendar = calendar_raw or {"year_name": "Year 1", "months": [], "days": []}
        
    aggregated = {
        "playerStats": player_stats,
        "introducedNPCs": introduced_npcs,
        "unintroducedNPCs": unintroduced_npcs,
        "currentRoleplay": currentroleplay_data,
        "calendar": calendar,
        "year": current_year,
        "month": current_month,
        "day": current_day,
        "timeOfDay": time_of_day,
        "socialLinks": social_links,
        "playerPerks": player_perks,
        "inventory": inventory_list,
        "events": events_list,
        "plannedEvents": planned_events_list,
        "quests": quest_list,
        "gameRules": game_rules_list,
        "statDefinitions": stat_def_list,
        "locations": locations_list
    }

    #----------------------------------------------------------------
    # 15) Summarize new changes & update 'GlobalSummary' in DB
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

    #----------------------------------------------------------------
    # 16) Build aggregator text for GPT using immersive calendar info
    #----------------------------------------------------------------
    calendar_info = load_calendar_names(user_id, conversation_id)
    # Ensure calendar_info is a dictionary
    if isinstance(calendar_info, str):
        try:
            calendar_info = json.loads(calendar_info)
        except Exception as e:
            logging.warning(f"Failed to parse calendar_info: {e}")
            calendar_info = {}
    
    immersive_date = f"Year: {calendar_info.get('year_name', current_year)}"
    months = calendar_info.get("months", [])
    if months and current_month.isdigit():
        month_index = int(current_month) - 1
        if 0 <= month_index < len(months):
            immersive_date += f", Month: {months[month_index]}"
    immersive_date += f", Day: {current_day}, {time_of_day}."

    aggregator_text = (
        f"{existing_summary}\n\n"
        f"{immersive_date}\n"
        "Scene Snapshot:\n"
        f"{make_minimal_scene_info(aggregated)}"
    )
    
    # Append additional context from CurrentRoleplay if available.
    if "EnvironmentDesc" in aggregated["currentRoleplay"]:
        aggregator_text += "\n\nEnvironment Description:\n" + aggregated["currentRoleplay"]["EnvironmentDesc"]
    if "PlayerRole" in aggregated["currentRoleplay"]:
        aggregator_text += "\n\nPlayer Role:\n" + aggregated["currentRoleplay"]["PlayerRole"]
    if "MainQuest" in aggregated["currentRoleplay"]:
        aggregator_text += "\n\nMain Quest (hint):\n" + aggregated["currentRoleplay"]["MainQuest"]
    if "ChaseSchedule" in aggregated["currentRoleplay"]:
        aggregator_text += "\n\nChase Schedule:\n" + json.dumps(aggregated["currentRoleplay"]["ChaseSchedule"], indent=2)
    
    # Optionally append Notable Events and Locations
    if aggregated.get("events"):
        aggregator_text += "\n\nNotable Events:\n"
        for ev in aggregated["events"][:3]:
            aggregator_text += f"- {ev['event_name']}: {ev['description']} (at {ev['location']})\n"
    if aggregated.get("locations"):
        aggregator_text += "\n\nNotable Locations:\n"
        for loc in aggregated["locations"][:3]:
            aggregator_text += f"- {loc['location_name']}: {loc['description']}\n"
    
    # Incorporate MegaSettingModifiers if available
    modifiers_str = aggregated["currentRoleplay"].get("MegaSettingModifiers", "")
    if modifiers_str:
        aggregator_text += "\n\n=== MEGA SETTING MODIFIERS ===\n"
        try:
            mod_dict = json.loads(modifiers_str)
            for k, v in mod_dict.items():
                aggregator_text += f"- {k}: {v}\n"
        except Exception:
            aggregator_text += "(Could not parse MegaSettingModifiers)\n"

    cursor.execute("""
        SELECT npc_id, current_state, last_decision
        FROM NPCAgentState
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    
    npc_agent_states = {}
    for row in cursor.fetchall():
        npc_id, current_state, last_decision = row
        npc_agent_states[npc_id] = {
            "current_state": current_state,
            "last_decision": last_decision
        }
    
    # Add agent state to aggregated data
    aggregated["npcAgentStates"] = npc_agent_states
    
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
    text = "\n".join(lines)
    if introduced_count == 0 and unintroduced_count == 0:
        text = ""
    return text

def update_global_summary(old_summary, new_stuff, max_len=3000):
    combined = old_summary.strip() + "\n\n" + new_stuff.strip()
    if len(combined) > max_len:
        combined = combined[-max_len:]
    return combined

def make_minimal_scene_info(aggregated):
    """
    Provide a small snippet: full date info plus a brief list of introduced
    and unintroduced NPCs for chance encounters.
    """
    lines = []
    # Retrieve calendar information and ensure it's a dictionary.
    calendar_data = aggregated.get("calendar_names", {})
    if isinstance(calendar_data, str):
        try:
            calendar_data = json.loads(calendar_data)
        except Exception:
            calendar_data = {}

    # Now use the dictionary to build the date string.
    year_str = calendar_data.get("year_name", aggregated.get("year", "Unknown Year"))
    month_str = calendar_data.get("month_name", aggregated.get("month", "Unknown Month"))
    day_str = calendar_data.get("day", aggregated.get("day", "Unknown Day"))
    tod = aggregated.get("timeOfDay", "Unknown Time")
    
    lines.append(f"- It is {year_str}, {month_str} {day_str}, {tod}.\n")
    
    # Introduced NPCs snippet
    lines.append("Introduced NPCs in the area:")
    for npc in aggregated["introducedNPCs"][:4]:
        loc = npc.get("current_location", "Unknown")
        lines.append(f"  - {npc['npc_name']} is at {loc}")
    
    # Unintroduced NPCs snippet
    lines.append("Unintroduced NPCs (possible random encounters):")
    for npc in aggregated["unintroducedNPCs"][:2]:
        loc = npc.get("current_location", "Unknown")
        lines.append(f"  - ???: '{npc['npc_name']}' lurking around {loc}")
    
    return "\n".join(lines)
