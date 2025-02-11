import os
import json
import random
import logging

from logic.chatgpt_integration import get_openai_client
from logic.gpt_utils import spaced_gpt_call
from db.connection import get_db_connection
from logic.memory_logic import get_shared_memory, record_npc_event
from logic.social_links import create_social_link
from logic.universal_updater import apply_universal_updates_async

###################
# 1) File Paths & Data Loading
###################

current_dir = os.path.dirname(os.path.abspath(__file__))

DATA_FILES = {
    "hobbies": os.path.join(current_dir, "..", "data", "npc_hobbies.json"),
    "likes": os.path.join(current_dir, "..", "data", "npc_likes.json"),
    "dislikes": os.path.join(current_dir, "..", "data", "npc_dislikes.json"),
    "personalities": os.path.join(current_dir, "..", "data", "npc_personalities.json"),
    "archetypes": os.path.join(current_dir, "..", "data", "archetypes_data.json") 
}

def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        return {}

DATA = {
    "hobbies_pool": [],
    "likes_pool": [],
    "dislikes_pool": [],
    "personality_pool": [],
    "archetypes_table": []
}

def init_data():
    # Hobbies
    hobbies_json = load_json_file(DATA_FILES["hobbies"])
    DATA["hobbies_pool"] = hobbies_json.get("hobbies_pool", [])
    
    # Likes
    likes_json = load_json_file(DATA_FILES["likes"])
    DATA["likes_pool"] = likes_json.get("npc_likes", [])
    
    # Dislikes
    dislikes_json = load_json_file(DATA_FILES["dislikes"])
    DATA["dislikes_pool"] = dislikes_json.get("dislikes_pool", [])
    
    # Personalities
    personalities_json = load_json_file(DATA_FILES["personalities"])
    DATA["personality_pool"] = personalities_json.get("personality_pool", [])
    
    # Archetypes
    arcs_json = load_json_file(DATA_FILES["archetypes"])
    table = arcs_json.get("archetypes", [])
    arcs_list = []
    for item in table:
        arcs_list.append((item["id"], item["name"], item.get("baseline_stats", {})))
    DATA["archetypes_table"] = arcs_list

init_data()

###################
# 2) Combine Archetype Stats
###################

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def combine_archetype_stats(archetype_rows):
    sums = {k: 0 for k in ["dominance","cruelty","closeness","trust","respect","intensity"]}
    count = len(archetype_rows)
    if count == 0:
        for k in sums.keys():
            sums[k] = random.randint(0, 30)
        return sums

    for (arc_id, arc_name, bs_json) in archetype_rows:
        if isinstance(bs_json, str):
            try:
                bs = json.loads(bs_json)
            except:
                bs = {}
        else:
            bs = bs_json or {}
        
        for stat_key in sums.keys():
            rng_key = f"{stat_key}_range"
            mod_key = f"{stat_key}_modifier"
            if rng_key in bs and mod_key in bs:
                low, high = bs[rng_key]
                mod = bs[mod_key]
                val = random.randint(low, high) + mod
            else:
                val = random.randint(0, 30)
            sums[stat_key] += val

    for sk in sums.keys():
        sums[sk] = sums[sk] / count
        if sk in ["trust","respect"]:
            sums[sk] = clamp(int(sums[sk]), -100, 100)
        else:
            sums[sk] = clamp(int(sums[sk]), 0, 100)
    return sums

###################
# 3) GPT synergy (name + summary)
###################

def get_archetype_synergy_description(archetypes_list, provided_npc_name=None):
    if not archetypes_list:
        default_name = provided_npc_name if provided_npc_name else f"NPC_{random.randint(1000,9999)}"
        return json.dumps({
            "npc_name": default_name,
            "archetype_summary": "No special archetype synergy."
        })

    archetype_names = [a["name"] for a in archetypes_list]
    if provided_npc_name:
        name_instruction = f"Use the provided NPC name: '{provided_npc_name}'."
    else:
        name_instruction = "Generate a creative, unique name for the NPC."

    system_msg = (
        f"You are an expert creative writer, merging these archetypes: {', '.join(archetype_names)}.\n"
        f"{name_instruction}\n"
        "Return a single JSON with exactly two keys:\n"
        "\"npc_name\", \"archetype_summary\".\n"
        "Do not include any extra text or markdown. Output only the JSON."
    )

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_msg}],
            temperature=0.7,
            max_tokens=300
        )
        raw = resp.choices[0].message.content.strip()
        return raw
    except Exception as e:
        logging.error(f"[get_archetype_synergy_description] error: {e}")
        default_name = provided_npc_name or f"NPC_{random.randint(1000,9999)}"
        return json.dumps({
            "npc_name": default_name,
            "archetype_summary": "GPT call failed. Summaries are unavailable."
        })

def get_archetype_extras_summary(archetypes_list, npc_name):
    combined = []
    for arc in archetypes_list:
        combined.append(f"{arc['name']}: synergy or extra traits.")
    joined = "\n".join(combined)
    return f"Additional synergy details for {npc_name}:\n{joined}"

###################
# 4) Creating Partial NPC
###################

REROLL_IDS = list(range(62, 73))

def create_npc_partial(sex="female", total_archetypes=4, reroll_extra=False) -> dict:
    chosen_arcs_rows = []
    final_stats = {}

    if sex.lower() == "male":
        final_stats = {
            "dominance": random.randint(0, 30),
            "cruelty": random.randint(0, 30),
            "closeness": random.randint(0, 30),
            "trust": random.randint(-30, 30),
            "respect": random.randint(-30, 30),
            "intensity": random.randint(0, 30)
        }
        chosen_arcs_rows = []
    else:
        all_arcs = DATA["archetypes_table"]
        reroll_pool = [row for row in all_arcs if row[0] in REROLL_IDS]
        normal_pool = [row for row in all_arcs if row[0] not in REROLL_IDS]
        
        if len(normal_pool) < total_archetypes:
            chosen_arcs_rows = normal_pool
        else:
            chosen_arcs_rows = random.sample(normal_pool, total_archetypes)
        if reroll_extra and reroll_pool:
            chosen_arcs_rows.append(random.choice(reroll_pool))

        final_stats = combine_archetype_stats(chosen_arcs_rows)

    arcs_list_for_json = []
    for (aid, aname, _bs) in chosen_arcs_rows:
        arcs_list_for_json.append({"id": aid, "name": aname})

    synergy_json = get_archetype_synergy_description(arcs_list_for_json, provided_npc_name=None)
    try:
        synergy_parsed = json.loads(synergy_json)
        synergy_npc_name = synergy_parsed.get("npc_name", f"NPC_{random.randint(1000,9999)}")
        synergy_text = synergy_parsed.get("archetype_summary", "")
    except:
        synergy_npc_name = f"NPC_{random.randint(1000,9999)}"
        synergy_text = "No synergy text"

    extras_summary = get_archetype_extras_summary(arcs_list_for_json, synergy_npc_name)

    hobby_pool = DATA.get("hobbies_pool", [])
    personality_pool = DATA.get("personality_pool", [])
    likes_pool = DATA.get("likes_pool", [])
    dislikes_pool = DATA.get("dislikes_pool", [])

    hbs = random.sample(hobby_pool, min(3, len(hobby_pool))) if hobby_pool else []
    pers = random.sample(personality_pool, min(3, len(personality_pool))) if personality_pool else []
    lks = random.sample(likes_pool, min(3, len(likes_pool))) if likes_pool else []
    dlks = random.sample(dislikes_pool, min(3, len(dislikes_pool))) if dislikes_pool else []

    npc_dict = {
        "npc_name": synergy_npc_name,
        "introduced": False,
        "sex": sex.lower(),

        "dominance": final_stats["dominance"],
        "cruelty": final_stats["cruelty"],
        "closeness": final_stats["closeness"],
        "trust": final_stats["trust"],
        "respect": final_stats["respect"],
        "intensity": final_stats["intensity"],

        "archetypes": arcs_list_for_json,
        "archetype_summary": synergy_text,
        "archetype_extras_summary": extras_summary,

        "hobbies": hbs,
        "personality_traits": pers,
        "likes": lks,
        "dislikes": dlks,

        "age": random.randint(20, 45),
        "birthdate": "1000-02-10"
    }

    return npc_dict

###################
# 5) GPT Refinement
###################

async def refine_npc_with_gpt(npc_partial: dict, environment_desc: str, day_names: list, conversation_id: int) -> dict:
    prompt = f"""
We have a partially created NPC in a femdom environment. Partial data:
{json.dumps(npc_partial, indent=2)}

Environment description:
{environment_desc}

We want to fill in or adapt these fields:
  - npc_name (confirm or revise),
  - physical_description (a paragraph),
  - schedule: use these day names => {day_names}, each day has Morning,Afternoon,Evening,Night,
  - affiliations: any relevant groups/factions,
  - memory: a short array of meaningful past events,
  - current_location: pick from the schedule (where are they right now?)

Return only a JSON object with these keys:
  "npc_name", "physical_description", "schedule", "affiliations", "memory", "current_location"

No extra text, no function calls.
"""

    result = await spaced_gpt_call_with_retry(
        conversation_id=conversation_id,
        context=environment_desc,
        prompt=prompt,
        delay=1.0
    )

    if result.get("type") == "function_call":
        return result.get("function_args", {})
    else:
        raw_text = result.get("response", "").strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()
        try:
            parsed = json.loads(raw_text)
            return parsed
        except Exception as e:
            logging.warning(f"[refine_npc_with_gpt] parse error: {e}")
            return {}

###################
# 6) Extended Reciprocal Archetypes (Examples)
###################
"""
Below is a bigger dictionary mapping *Dominant Archetype* -> *Inverse Archetype*.
Feel free to rename or prune these. The 'id' is arbitrary; you can define them 
to avoid collisions with existing IDs in your archetypes table.
"""
EXTENDED_RECIPROCAL_ARCHETYPES = {
    # Family / Household
    "Mother":        {"id": 9501, "name": "Child"},
    "Stepmother":    {"id": 9502, "name": "Step-Child"},
    "Aunt":          {"id": 9503, "name": "Niece/Nephew"},
    "Older Sister":  {"id": 9504, "name": "Younger Sibling"},
    "Stepsister":    {"id": 9505, "name": "Step-Sibling"},
    "Babysitter":    {"id": 9506, "name": "Child"},

    # Workplace / Power
    "CEO":                {"id": 9510, "name": "Employee"},
    "Boss/Supervisor":    {"id": 9511, "name": "Employee"},
    "Corporate Dominator":{"id": 9512, "name": "Underling"},
    "Teacher/Principal":  {"id": 9513, "name": "Student"},
    "Landlord":           {"id": 9514, "name": "Tenant"},
    "Warden":             {"id": 9515, "name": "Prisoner"},
    "Loan Shark":         {"id": 9516, "name": "Debtor"},
    "Slave Overseer":     {"id": 9517, "name": "Slave"},
    "Therapist":          {"id": 9518, "name": "Patient"},
    "Doctor":             {"id": 9519, "name": "Patient"},
    "Social Media Influencer": {"id": 9520, "name": "Follower"},
    "Bartender":          {"id": 9521, "name": "Patron"},
    "Fitness Trainer":    {"id": 9522, "name": "Client"},
    "Cheerleader/Team Captain": {"id": 9523, "name": "Junior Team Member"},
    "Martial Artist":     {"id": 9524, "name": "Sparring Dummy"},
    "Professional Wrestler": {"id": 9525, "name": "Defeated Opponent"},

    # Supernatural / Hunting
    "Demon":              {"id": 9530, "name": "Thrall"},
    "Demoness":           {"id": 9531, "name": "Bound Mortal"},
    "Devil":              {"id": 9532, "name": "Damned Soul"},
    "Villain (RPG-Esque)": {"id": 9533, "name": "Captured Hero"},
    "Haunted Entity":     {"id": 9534, "name": "Haunted Mortal"},
    "Sorceress":          {"id": 9535, "name": "Cursed Subject"},
    "Witch":              {"id": 9536, "name": "Hexed Victim"},
    "Eldritch Abomination":{"id": 9537, "name": "Insane Acolyte"},
    "Primal Huntress":    {"id": 9538, "name": "Prey"},
    "Primal Predator":    {"id": 9539, "name": "Prey"},
    "Serial Killer":      {"id": 9540, "name": "Victim"},

    # Others
    "Rockstar":           {"id": 9541, "name": "Fan"},
    "Celebrity":          {"id": 9542, "name": "Fan"},
    "Ex-Girlfriend/Ex-Wife": {"id": 9543, "name": "Ex-Partner"},
    "Politician":         {"id": 9544, "name": "Constituent"},
    "Queen":              {"id": 9545, "name": "Subject"},
    "Empress":            {"id": 9546, "name": "Subject"},
    "Royal Knight":       {"id": 9547, "name": "Challenged Rival"},
    "Gladiator":          {"id": 9548, "name": "Arena Opponent"},
    "Pirate":             {"id": 9549, "name": "Captive"},
    "Bank Robber":        {"id": 9550, "name": "Hostage"},
    "Cybercriminal":      {"id": 9551, "name": "Hacked Victim"},
    "Huntress":           {"id": 9552, "name": "Prey"}, 
    "Arsonist":           {"id": 9553, "name": "Burned Victim"},
    "Drug Dealer":        {"id": 9554, "name": "Addict"},
    "Artificial Intelligence": {"id": 9555, "name": "User/Victim"},
    "Fey":                {"id": 9556, "name": "Ensorcelled Mortal"},
    "Nun":                {"id": 9557, "name": "Sinner"},
    "Priestess":          {"id": 9558, "name": "Acolyte"},
    "A True Goddess":     {"id": 9559, "name": "Worshipper"},
    "Haruhi Suzumiya-Type Goddess": {"id": 9560, "name": "Reality Pawn"},
    "Bowsette Personality": {"id": 9561, "name": "Castle Captive"},
    "Juri Han Personality": {"id": 9562, "name": "Beaten Opponent"},
    "Neighbor":           {"id": 9563, "name": "Targeted Neighbor"},
    "Hero (RPG-Esque)":   {"id": 9564, "name": "Sidekick / Rescued Target"},
    # etc. (You can continue expanding)
}

###################
# 7) Relationship Archetype Map (legacy example)
###################
"""
In your example, you had a smaller RELATIONSHIP_ARCHETYPE_MAP that 
adds an archetype to the new NPC based on the relationship chosen. 
We'll keep it, but you can combine it with EXTENDED_RECIPROCAL_ARCHETYPES
or keep them separate if you want multi-step logic.
"""

RELATIONSHIP_ARCHETYPE_MAP = {
    "mother":   {"id": 9001, "name": "Maternal Overlord"},
    "underling": {"id": 9002, "name": "Servile Underling"},
    "CEO": {"id": 9003, "name": "Corporate Dominator"}
    # etc.
}

###################
# 8) Adding / Recalculating Archetypes
###################

async def await_prompted_synergy_after_add_archetype(arcs_list, user_id, conversation_id, npc_id):
    """
    (Same as your code snippet) - an async function to call GPT for synergy update 
    after we add a new archetype to an existing NPC.
    """
    archetype_names = [arc["name"] for arc in arcs_list]
    system_instructions = f"""
We just appended a new archetype to this NPC. Now they have: {', '.join(archetype_names)}.
Please provide an updated synergy summary, in JSON with a single key "archetype_summary".
No extra text or function calls.
"""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_instructions}],
            temperature=0.7,
            max_tokens=200
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        data = json.loads(raw)
        return data.get("archetype_summary","")
    except Exception as e:
        logging.warning(f"Error in await_prompted_synergy_after_add_archetype: {e}")
        return "Could not update synergy"

def recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT archetypes 
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (user_id, conversation_id, npc_id))
    row = cursor.fetchone()
    if not row:
        logging.warning(f"No NPC found for npc_id={npc_id}, cannot recalc stats.")
        conn.close()
        return

    arcs_json = row[0] or "[]"
    try:
        arcs_list = json.loads(arcs_json)
    except:
        arcs_list = []

    chosen_rows = []
    for arc_obj in arcs_list:
        arc_id = arc_obj.get("id")
        arc_name = arc_obj.get("name", "Unknown")
        
        # Attempt to find in "archetypes_table"
        match = None
        for r in DATA["archetypes_table"]:
            if r[0] == arc_id:
                match = r
                break
        if match:
            chosen_rows.append(match)
        else:
            # fallback for custom arcs
            dummy_stats = {"dominance_range":[20,30],"dominance_modifier":5}
            chosen_rows.append((arc_id, arc_name, dummy_stats))

    final_stats = combine_archetype_stats(chosen_rows)

    cursor.execute("""
        UPDATE NPCStats
        SET dominance=%s, cruelty=%s, closeness=%s, trust=%s, respect=%s, intensity=%s
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (final_stats["dominance"], final_stats["cruelty"], final_stats["closeness"], 
          final_stats["trust"], final_stats["respect"], final_stats["intensity"],
          user_id, conversation_id, npc_id))
    conn.commit()
    conn.close()
    logging.info(f"[recalc_npc_stats_with_new_archetypes] Updated stats => {final_stats} for npc_id={npc_id}.")


async def add_archetype_to_npc(user_id, conversation_id, npc_id, new_arc):
    """
    Marked as async if we want to do GPT calls for synergy update. 
    If you prefer sync, remove 'async' and do a direct client call or skip GPT synergy.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT archetypes
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        LIMIT 1
    """, (user_id, conversation_id, npc_id))
    row = cursor.fetchone()
    if not row:
        logging.warning(f"No NPCStats record found for npc_id={npc_id}; can't add archetype.")
        conn.close()
        return

    existing_arcs_json = row[0] or '[]'
    try:
        existing_arcs = json.loads(existing_arcs_json)
    except:
        existing_arcs = []

    # 2) Append if not already present
    if not any(a.get("id") == new_arc["id"] for a in existing_arcs):
        existing_arcs.append(new_arc)
    else:
        logging.info(f"NPC {npc_id} already has archetype {new_arc['name']}; skipping add.")
        conn.close()
        return

    updated_arcs_json = json.dumps(existing_arcs)

    # (A) Optionally re-run synergy via GPT
    updated_synergy = await_prompted_synergy_after_add_archetype(existing_arcs, user_id, conversation_id, npc_id)
    if not updated_synergy:
        updated_synergy = "No updated synergy available."

    # (B) Update DB with new arcs + synergy
    cursor.execute("""
        UPDATE NPCStats
        SET archetypes=%s,
            archetype_summary=%s
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (updated_arcs_json, updated_synergy, user_id, conversation_id, npc_id))
    conn.commit()

    # (C) Recalc stats
    recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id)

    conn.close()
    logging.info(f"Added new archetype '{new_arc['name']}' to NPC {npc_id}, synergy & stats re-updated.")

###################
# 9) The Relationship function
###################

def assign_random_relationships(user_id, conversation_id, new_npc_id, new_npc_name):
    familial = ["mother", "sister", "aunt"]
    non_familial = ["enemy", "friend", "lover", "neighbor", 
                    "colleague", "classmate", "teammate", 
                    "underling", "CEO"]

    relationships = []

    # Chance for relationship with player
    if random.random() < 0.5:
        if random.random() < 0.2:
            rel_type = random.choice(familial)
        else:
            rel_type = random.choice(non_familial)
        relationships.append({"target": "player", "target_name": "the player", "type": rel_type})

    # Gather existing NPCs
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT npc_id, npc_name FROM NPCStats WHERE user_id = %s AND conversation_id = %s AND npc_id != %s",
        (user_id, conversation_id, new_npc_id)
    )
    existing_npcs = cursor.fetchall()
    conn.close()

    for npc_row in existing_npcs:
        candidate_id, candidate_name = npc_row
        if random.random() < 0.3:
            if random.random() < 0.2:
                rel_type = random.choice(familial)
            else:
                rel_type = random.choice(non_familial)
            if not any(rel.get("target") == candidate_id for rel in relationships):
                relationships.append({
                    "target": candidate_id,
                    "target_name": candidate_name,
                    "type": rel_type
                })

    # For each chosen relationship, record memory, create SocialLinks, possibly add archetypes
    for rel in relationships:
        memory_text = get_shared_memory(user_id, conversation_id, rel, new_npc_name)
        record_npc_event(user_id, conversation_id, new_npc_id, memory_text)

        if rel["target"] == "player":
            create_social_link(
                user_id, conversation_id,
                "player", 0,
                "npc", new_npc_id,
                link_type=rel["type"], link_level=0
            )
        else:
            create_social_link(
                user_id, conversation_id,
                "npc", rel["target"],
                "npc", new_npc_id,
                link_type=rel["type"], link_level=0
            )

        # If relationship type implies a special archetype for the new NPC
        special_arc = RELATIONSHIP_ARCHETYPE_MAP.get(rel["type"])
        if special_arc:
            # adding an archetype to new_npc
            logging.info(f"Relationship '{rel['type']}' => adding archetype {special_arc} to NPC {new_npc_id}")
            # This is an async call, so we might do:
            import asyncio
            asyncio.run(add_archetype_to_npc(user_id, conversation_id, new_npc_id, special_arc))

        # (Optional) If the new NPC has an archetype from EXTENDED_RECIPROCAL_ARCHETYPES, 
        # we might want to add the reciprocal to the other side.
        # But you'd typically do that after we already know what archetype new_npc has.

    logging.info(f"[assign_random_relationships] Completed for NPC {new_npc_name} (id={new_npc_id}).")


###################
# 10) Main Function: Spawn, Refine, Then Relationship
###################

async def spawn_and_refine_npcs_with_relationships(
    user_id: int,
    conversation_id: int,
    environment_desc: str,
    day_names: list,
    conn,
    count=3
):
    """
    Step A: partial creation + GPT refine => we get final NPC data.
    Step B: apply_universal_updates_async => store them in DB.
    Step C: assign relationships => each new NPC forms random ties to existing NPCs or the player.

    This code uses the extended approach but doesn't fully demonstrate 
    the "inverse archetype" logic. You can adapt further if you want the 
    'other side' (like if the new NPC is 'Mother') to forcibly get 'Child' 
    archetype. That logic typically belongs in a post-creation step, 
    scanning the new NPC's archetypes to see if there's an entry in 
    EXTENDED_RECIPROCAL_ARCHETYPES, etc.
    """
    final_npc_list = []

    # -- PHASE A: partial creation + refine
    for _ in range(count):
        partial_npc = create_npc_partial(sex="female", total_archetypes=3)
        refine_data = await refine_npc_with_gpt(partial_npc, environment_desc, day_names, conversation_id)

        for k, v in refine_data.items():
            partial_npc[k] = v

        final_npc_list.append(partial_npc)

    # -- PHASE B: Insert them with universal updater
    update_payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "npc_creations": final_npc_list
    }
    result = await apply_universal_updates_async(user_id, conversation_id, update_payload, conn)
    logging.info("Spawned NPCs via universal update: %s", result)

    # -- PHASE C: Relationship assignment
    # For each newly created NPC by name, find npc_id, assign relationships
    for npc_data in final_npc_list:
        created_name = npc_data["npc_name"]
        row = await conn.fetchrow("""
            SELECT npc_id 
            FROM NPCStats 
            WHERE user_id=$1 AND conversation_id=$2
              AND LOWER(npc_name)=LOWER($3)
            LIMIT 1
        """, user_id, conversation_id, created_name)
        if row:
            new_npc_id = row["npc_id"]
            assign_random_relationships(user_id, conversation_id, new_npc_id, created_name)

            # Additional: If the new NPC's archetypes (like "Mother", "Demoness", etc.)
            # are in EXTENDED_RECIPROCAL_ARCHETYPES, 
            # you might parse them and add the corresponding inverse to the "target" side, 
            # but that requires you to see which relationship was chosen, or do a final pass 
            # linking each discovered relationship target to their new role. 
            #
            # e.g. if new NPC has "Mother", we find the social_link with link_type='mother' 
            # and add "Child" to the other side. 
            #
            # That logic can be done in a "post-relationships" pass.

        else:
            logging.warning(f"Could not find newly created NPC in DB by name={created_name}. Skipping relationships.")

    logging.info(f"Assigned random relationships for the {count} newly spawned NPCs.")
    return {"message": f"Spawned {count} NPCs and assigned relationships."}
