import json
import logging
import asyncpg
from datetime import datetime, date

from logic.npc_creation import (
    create_npc_partial,
    insert_npc_stub_into_db,
    recalc_npc_stats_with_new_archetypes,
    refine_npc_final_data
)
from logic.social_links import (
    get_social_link, create_social_link,
    update_link_type_and_level, add_link_event
)

#############################
# 1) Helper to normalize smart quotes
#############################
def normalize_smart_quotes_inplace(obj):
    """
    Recursively replaces curly quotes/apostrophes with straight ASCII quotes
    in all strings within a nested dict/list structure.
    """
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = normalize_smart_quotes_inplace(obj[k])
        return obj
    elif isinstance(obj, list):
        return [normalize_smart_quotes_inplace(x) for x in obj]
    elif isinstance(obj, str):
        return (
            obj.replace("’", "'")
               .replace("‘", "'")
               .replace("“", '"')
               .replace("”", '"')
        )
    else:
        return obj


async def apply_universal_updates_async(user_id, conversation_id, data, conn) -> dict:
    """
    Applies universal data updates from a GPT or user-provided payload, 
    with partial affiliation merges, curly-quote cleanup, and an optional
    "refine NPC data" pass at the end.
    """
    try:
        # -----------------------------------------------------------
        # (5) Normalize curly quotes in the entire incoming data dict
        # -----------------------------------------------------------
        data = normalize_smart_quotes_inplace(data)

        logging.info("=== [apply_universal_updates_async] Incoming data (after normalization) ===")
        logging.info(json.dumps(data, indent=2))

        data_user_id = data.get("user_id", user_id)
        data_conv_id = data.get("conversation_id", conversation_id)
        if not data_user_id or not data_conv_id:
            logging.error("Missing user_id or conversation_id in universal_update data.")
            return {"error": "Missing user_id or conversation_id in universal_update"}

        # Pull out day_names & environment_desc if you need them for refining
        # (some flows store them in roleplay_updates; adjust as needed)
        day_names = data.get("day_names", ["Commandday","Bindday","Chastiseday","Overlorday","Submissday","Whipday","Obeyday"])
        environment_desc = data.get("environment_desc", "A corporate gothic environment")

        # Keep track of NPCs we might re-refine after updates
        npcs_to_refine = set()

        # 1) Process roleplay_updates
        rp_updates = data.get("roleplay_updates", {})
        logging.info(f"[apply_universal_updates_async] roleplay_updates: {rp_updates}")
        for key, val in rp_updates.items():
            stored_val = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay(user_id, conversation_id, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, key) DO UPDATE 
                  SET value = EXCLUDED.value
                """,
                user_id, conversation_id, key, stored_val
            )
            logging.info(f"  Insert/Update CurrentRoleplay => key={key}, value={val}")

        # 2) Process npc_creations
        npc_creations = data.get("npc_creations", [])
        logging.info(f"[universal_updater] npc_creations => count={len(npc_creations)}")
        for npc_data in npc_creations:
            name = npc_data.get("npc_name", "Unnamed NPC")
            introduced = npc_data.get("introduced", False)

            row = await conn.fetchrow(
                """
                SELECT npc_id FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND LOWER(npc_name)=$3
                LIMIT 1
                """,
                user_id, conversation_id, name.lower()
            )
            if row:
                logging.info(
                    f"[universal_updater] Skipping creation. NPC '{name}' already exists (npc_id={row['npc_id']})."
                )
                continue

            environment_override = npc_data.get("environment_desc", environment_desc)
            partial_npc = create_npc_partial(
                user_id=user_id,
                conversation_id=conversation_id,
                sex=npc_data.get("sex", "female"),
                environment_desc=environment_override
            )

            override_keys = [
                "npc_name", "introduced", "sex", "dominance", "cruelty", "closeness",
                "trust", "respect", "intensity", "archetypes", "archetype_summary",
                "archetype_extras_summary", "likes", "dislikes", "hobbies",
                "personality_traits", "age", "birthdate"
            ]
            for key in override_keys:
                if key in npc_data:
                    partial_npc[key] = npc_data[key]

            logging.info(
                f"[universal_updater] Inserting new NPC '{partial_npc['npc_name']}' with birthdate={partial_npc.get('birthdate')}"
            )
            new_npc_id = await insert_npc_stub_into_db(partial_npc, user_id, conversation_id)
            # Mark for refining if the user wants
            # (You could auto-refine newly created NPCs if desired)
            npcs_to_refine.add(new_npc_id)

        # 3) Process npc_updates
        npc_updates = data.get("npc_updates", [])
        logging.info(f"[apply_universal_updates_async] npc_updates: {npc_updates}")
        for up in npc_updates:
            npc_id = up.get("npc_id")
            if not npc_id:
                logging.warning("Skipping npc_update: missing npc_id.")
                continue

            fields_map = {
                "npc_name": "npc_name",
                "introduced": "introduced",
                "dominance": "dominance",
                "cruelty": "cruelty",
                "closeness": "closeness",
                "trust": "trust",
                "respect": "respect",
                "intensity": "intensity",
                "monica_level": "monica_level",
                "sex": "sex"
            }

            set_clauses = []
            set_vals = []
            for field_key, db_col in fields_map.items():
                if field_key in up:
                    set_clauses.append(f"{db_col} = ${len(set_vals) + 1}")
                    set_vals.append(up[field_key])

            # If user changes stats or name, we update them
            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_vals += [npc_id, user_id, conversation_id]
                query = f"""
                    UPDATE NPCStats
                    SET {set_str}
                    WHERE npc_id=${len(set_vals)-2}
                      AND user_id=${len(set_vals)-1}
                      AND conversation_id=${len(set_vals)}
                """
                logging.info(f"  Updating NPC {npc_id}: {set_clauses}")
                await conn.execute(query, *set_vals)

            # (2) If "archetypes" changed, store it & recalc stats
            if "archetypes" in up:
                new_arcs = up["archetypes"]
                logging.info(f"  Overwriting archetypes for NPC {npc_id} => {new_arcs}")
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET archetypes=$1
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                    """,
                    json.dumps(new_arcs), npc_id, user_id, conversation_id
                )
                # Recalculate NPC stats based on new archetypes
                await recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id)
                # Mark for refining if we want updated schedule/description
                npcs_to_refine.add(npc_id)

            # Memory (append)
            if "memory" in up:
                new_mem_entries = up["memory"]
                if isinstance(new_mem_entries, str):
                    new_mem_entries = [new_mem_entries]
                logging.info(f"  Appending memory to NPC {npc_id}: {new_mem_entries}")
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb($1::json)
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                    """,
                    json.dumps(new_mem_entries), npc_id, user_id, conversation_id
                )
                # Possibly refine if memory changes should appear in the GPT narrative
                npcs_to_refine.add(npc_id)

            # Full schedule overwrite
            if "schedule" in up:
                new_schedule = up["schedule"]
                logging.info(f"  Overwriting schedule for NPC {npc_id}: {new_schedule}")
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET schedule=$1
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                    """,
                    json.dumps(new_schedule), npc_id, user_id, conversation_id
                )
                npcs_to_refine.add(npc_id)

            # Partial schedule update
            if "schedule_updates" in up:
                partial_sched = up["schedule_updates"]
                logging.info(f"  Merging schedule_updates for NPC {npc_id}: {partial_sched}")
                row = await conn.fetchrow(
                    """
                    SELECT schedule
                    FROM NPCStats
                    WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                    """,
                    npc_id, user_id, conversation_id
                )
                if row:
                    existing_schedule = row["schedule"] or {}
                    for day_key, times_map in partial_sched.items():
                        if day_key not in existing_schedule:
                            existing_schedule[day_key] = {}
                        existing_schedule[day_key].update(times_map)
                    await conn.execute(
                        """
                        UPDATE NPCStats
                        SET schedule=$1
                        WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                        """,
                        json.dumps(existing_schedule), npc_id, user_id, conversation_id
                    )
                npcs_to_refine.add(npc_id)

        # 3.5) Player schedule updates
        chase_sched = data.get("ChaseSchedule")
        if chase_sched:
            logging.info(f"Storing ChaseSchedule into CurrentRoleplay: {chase_sched}")
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'ChaseSchedule', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
                """,
                user_id, conversation_id, json.dumps(chase_sched)
            )

        # 4) Process character_stat_updates
        char_update = data.get("character_stat_updates", {})
        logging.info(f"[apply_universal_updates_async] character_stat_updates: {char_update}")
        if char_update:
            p_name = char_update.get("player_name", "Chase")
            stats = char_update.get("stats", {})
            stat_map = {
                "corruption": "corruption",
                "confidence": "confidence",
                "willpower": "willpower",
                "obedience": "obedience",
                "dependency": "dependency",
                "lust": "lust",
                "mental_resilience": "mental_resilience",
                "physical_endurance": "physical_endurance"
            }
            set_clauses = []
            set_vals = []
            for k, col in stat_map.items():
                if k in stats:
                    set_clauses.append(f"{col} = ${len(set_vals) + 1}")
                    set_vals.append(stats[k])
            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_vals += [p_name, user_id, conversation_id]
                logging.info(f"  Updating player stats for {p_name}: {stats}")
                await conn.execute(
                    f"""
                    UPDATE PlayerStats
                    SET {set_str}
                    WHERE player_name=$%d AND user_id=$%d AND conversation_id=$%d
                    """ % (len(set_vals)-2, len(set_vals)-1, len(set_vals)),
                    *set_vals
                )

        # 5) Process relationship_updates
        rel_updates = data.get("relationship_updates", [])
        logging.info(f"[apply_universal_updates_async] relationship_updates: {rel_updates}")
        for r in rel_updates:
            npc_id = r.get("npc_id")
            if not npc_id:
                logging.warning("Skipping relationship update: no npc_id.")
                continue

            # (2) Instead of overwriting affiliations, we MERGE them
            # so we treat r["affiliations"] as items to add/append
            aff_list = r.get("affiliations", None)
            if aff_list is not None:
                logging.info(f"  Appending affiliations for NPC {npc_id}: {aff_list}")
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET affiliations = COALESCE(affiliations, '[]'::jsonb) || to_jsonb($1::json)
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                    """,
                    json.dumps(aff_list), npc_id, user_id, conversation_id
                )
                npcs_to_refine.add(npc_id)

        # 5.5) Process shared_memory_updates
        shared_memory_updates = data.get("shared_memory_updates", [])
        logging.info(f"[apply_universal_updates_async] shared_memory_updates: {shared_memory_updates}")
        for sm_update in shared_memory_updates:
            npc_id = sm_update.get("npc_id")
            relationship = sm_update.get("relationship")
            if not npc_id or not relationship:
                logging.warning("Skipping shared memory update: missing npc_id or relationship.")
                continue
            row = await conn.fetchrow(
                """
                SELECT npc_name FROM NPCStats
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                """,
                npc_id, user_id, conversation_id
            )
            if not row:
                logging.warning(f"Shared memory update: NPC with id {npc_id} not found.")
                continue
            npc_name = row["npc_name"]

            from logic.memory import get_shared_memory
            shared_memory_text = get_shared_memory(user_id, conversation_id, relationship, npc_name)
            logging.info(f"Generated shared memory for NPC {npc_id}: {shared_memory_text}")
            await conn.execute(
                """
                UPDATE NPCStats
                SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb($1::text)
                WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                """,
                shared_memory_text, npc_id, user_id, conversation_id
            )
            npcs_to_refine.add(npc_id)

        # 6) Process npc_introductions
        npc_intros = data.get("npc_introductions", [])
        logging.info(f"[apply_universal_updates_async] npc_introductions: {npc_intros}")
        for intro in npc_intros:
            nid = intro.get("npc_id")
            if nid:
                logging.info(f"  Marking NPC {nid} as introduced")
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET introduced=TRUE
                    WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                    """,
                    nid, user_id, conversation_id
                )

        # 7) Process location_creations
        loc_creations = data.get("location_creations", [])
        logging.info(f"[apply_universal_updates_async] location_creations: {loc_creations}")
        for loc in loc_creations:
            loc_name = loc.get("location_name", "Unnamed")
            desc = loc.get("description", "")
            open_hours = loc.get("open_hours", [])
            row = await conn.fetchrow(
                """
                SELECT id FROM Locations
                WHERE user_id=$1 AND conversation_id=$2 AND LOWER(location_name)=$3
                LIMIT 1
                """,
                user_id, conversation_id, loc_name.lower()
            )
            if row:
                logging.info(f"  Skipping location creation, '{loc_name}' already exists.")
                continue
            logging.info(f"  Inserting location => location_name={loc_name}, description={desc}, open_hours={open_hours}")
            await conn.execute(
                """
                INSERT INTO Locations (user_id, conversation_id, location_name, description, open_hours)
                VALUES ($1, $2, $3, $4, $5)
                """,
                user_id, conversation_id, loc_name, desc, json.dumps(open_hours)
            )

        # 8) Process event_list_updates
        event_updates = data.get("event_list_updates", [])
        logging.info(f"[apply_universal_updates_async] event_list_updates: {event_updates}")
        for ev in event_updates:
            if "npc_id" in ev and "day" in ev and "time_of_day" in ev:
                npc_id = ev["npc_id"]
                year = ev.get("year", 1)
                month = ev.get("month", 1)
                day = ev["day"]
                tod = ev["time_of_day"]
                ov_loc = ev.get("override_location", "Unknown")
                row = await conn.fetchrow(
                    """
                    SELECT event_id FROM PlannedEvents
                    WHERE user_id=$1 AND conversation_id=$2
                      AND npc_id=$3 AND year=$4 AND month=$5 AND day=$6 AND time_of_day=$7
                    LIMIT 1
                    """,
                    user_id, conversation_id, npc_id, year, month, day, tod
                )
                if row:
                    logging.info("  Skipping planned event creation; already exists.")
                    continue
                logging.info(f"  Inserting PlannedEvent for npc_id={npc_id}")
                await conn.execute(
                    """
                    INSERT INTO PlannedEvents (
                        user_id, conversation_id, npc_id, year, month, day, 
                        time_of_day, override_location
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    user_id, conversation_id, npc_id, year, month, day, tod, ov_loc
                )

        # 9) Process inventory_updates
        inv_updates = data.get("inventory_updates", {})
        logging.info(f"[apply_universal_updates_async] inventory_updates: {inv_updates}")
        if inv_updates:
            p_n = inv_updates.get("player_name", "Chase")
            added = inv_updates.get("added_items", [])
            removed = inv_updates.get("removed_items", [])
            for item in added:
                if isinstance(item, dict):
                    item_name = item.get("item_name", "Unnamed")
                    item_desc = item.get("item_description", "")
                    item_fx   = item.get("item_effect", "")
                    category  = item.get("category", "")
                    logging.info(f"  Adding item for {p_n}: {item_name}")
                    await conn.execute(
                        """
                        INSERT INTO PlayerInventory (
                            user_id, conversation_id,
                            player_name, item_name, item_description, 
                            item_effect, category, quantity
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 1)
                        ON CONFLICT (user_id, conversation_id, player_name, item_name)
                        DO UPDATE SET quantity = PlayerInventory.quantity + 1
                        """,
                        user_id, conversation_id, p_n, item_name, item_desc, item_fx, category
                    )
                elif isinstance(item, str):
                    logging.info(f"  Adding item (string) for {p_n}: {item}")
                    await conn.execute(
                        """
                        INSERT INTO PlayerInventory (
                            user_id, conversation_id,
                            player_name, item_name, quantity
                        )
                        VALUES ($1, $2, $3, $4, 1)
                        ON CONFLICT (user_id, conversation_id, player_name, item_name)
                        DO UPDATE SET quantity = PlayerInventory.quantity + 1
                        """,
                        user_id, conversation_id, p_n, item
                    )
    
            for item in removed:
                if isinstance(item, dict):
                    i_name = item.get("item_name")
                    if i_name:
                        logging.info(f"  Removing item for {p_n}: {i_name}")
                        await conn.execute(
                            """
                            DELETE FROM PlayerInventory
                            WHERE user_id=$1 AND conversation_id=$2
                              AND player_name=$3 AND item_name=$4
                            """,
                            user_id, conversation_id, p_n, i_name
                        )
                elif isinstance(item, str):
                    logging.info(f"  Removing item for {p_n}: {item}")
                    await conn.execute(
                        """
                        DELETE FROM PlayerInventory
                        WHERE user_id=$1 AND conversation_id=$2
                          AND player_name=$3 AND item_name=$4
                        """,
                        user_id, conversation_id, p_n, item
                    )

        # 10) Process quest_updates
        quest_updates = data.get("quest_updates", [])
        logging.info(f"[apply_universal_updates_async] quest_updates: {quest_updates}")
        for qu in quest_updates:
            qid = qu.get("quest_id")
            status = qu.get("status", "In Progress")
            detail = qu.get("progress_detail", "")
            qgiver = qu.get("quest_giver", "")
            reward = qu.get("reward", "")
            qname  = qu.get("quest_name", None)
            if qid:
                logging.info(f"  Updating Quest {qid}: status={status}, detail={detail}")
                await conn.execute(
                    """
                    UPDATE Quests
                    SET status=$1, progress_detail=$2, quest_giver=$3, reward=$4
                    WHERE quest_id=$5 AND user_id=$6 AND conversation_id=$7
                    """,
                    status, detail, qgiver, reward, qid, user_id, conversation_id
                )
            else:
                logging.info(f"  Inserting new quest: {qname}, status={status}")
                await conn.execute(
                    """
                    INSERT INTO Quests (
                        user_id, conversation_id,
                        quest_name, status, progress_detail,
                        quest_giver, reward
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    user_id, conversation_id, qname or "Unnamed Quest",
                    status, detail, qgiver, reward
                )

        # 11) Process social_links
        rel_links = data.get("social_links", [])
        logging.info(f"[apply_universal_updates_async] social_links: {rel_links}")
        for link_data in rel_links:
            e1_type = link_data.get("entity1_type")
            e1_id   = link_data.get("entity1_id")
            e2_type = link_data.get("entity2_type")
            e2_id   = link_data.get("entity2_id")
            if not (e1_type and e1_id and e2_type and e2_id):
                logging.warning(f"Skipping social link creation, missing entity info: {link_data}")
                continue
            existing_link = get_social_link(e1_type, e1_id, e2_type, e2_id, user_id, conversation_id)
            if not existing_link:
                link_type = link_data.get("link_type", "neutral")
                lvl_change = link_data.get("level_change", 0)
                logging.info(f"  Creating new social link: {e1_type}({e1_id}) <-> {e2_type}({e2_id}), type={link_type}")
                new_link_id = create_social_link(
                    e1_type, e1_id, e2_type, e2_id,
                    link_type=link_type, link_level=lvl_change,
                    user_id=user_id, conversation_id=conversation_id
                )
                if "new_event" in link_data:
                    add_link_event(new_link_id, link_data["new_event"])
            else:
                ltype = link_data.get("link_type")
                lvl_change = link_data.get("level_change", 0)
                logging.info(f"  Updating social link {existing_link['link_id']}: new_type={ltype}, level_change={lvl_change}")
                update_link_type_and_level(
                    existing_link["link_id"],
                    new_type=ltype, level_change=lvl_change,
                    user_id=user_id, conversation_id=conversation_id
                )
                if "new_event" in link_data:
                    add_link_event(existing_link["link_id"], link_data["new_event"])

        # 12) Process perk_unlocks
        perk_unlocks = data.get("perk_unlocks", [])
        logging.info(f"[apply_universal_updates_async] perk_unlocks: {perk_unlocks}")
        for perk in perk_unlocks:
            perk_name = perk.get("perk_name")
            perk_desc = perk.get("perk_description", "")
            perk_fx   = perk.get("perk_effect", "")
            player_name = perk.get("player_name", "Chase")
            if perk_name:
                logging.info(f"  Inserting perk {perk_name} for player {player_name}")
                await conn.execute(
                    """
                    INSERT INTO PlayerPerks (
                        user_id, conversation_id,
                        player_name, perk_name,
                        perk_description, perk_effect
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT DO NOTHING
                    """,
                    user_id, conversation_id, player_name, perk_name, perk_desc, perk_fx
                )

        # ------------------------------------------------
        # (7) After collecting all updated NPC IDs,
        #     optionally refine them in one pass.
        # ------------------------------------------------
        logging.info(f"NPCs to refine: {npcs_to_refine}")
        for nid in npcs_to_refine:
            logging.info(f"[universal_updater] Running refine_npc_final_data on NPC {nid}")
            await refine_npc_final_data(user_id, conversation_id, nid, day_names, environment_desc)

        logging.info("=== [apply_universal_updates_async] Success (transaction committed) ===")
        return {"message": "Universal update successful"}

    except Exception as e:
        logging.exception("[apply_universal_updates_async] Error encountered (transaction rolled back)")
        return {"error": str(e)}

apply_universal_updates = apply_universal_updates_async
