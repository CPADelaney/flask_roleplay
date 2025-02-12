import os
import json
import random
import logging
import asyncio

from logic.chatgpt_integration import get_openai_client, get_chatgpt_response
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
    """Load all local JSON data into the DATA dictionary for reuse."""
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
        # Minimal fields needed
        arcs_list.append({
            "name": item["name"],
            "baseline_stats": item.get("baseline_stats", {}),
            "progression_rules": item.get("progression_rules", []),
            "unique_traits": item.get("unique_traits", []),
            "preferred_kinks": item.get("preferred_kinks", [])
        })
    DATA["archetypes_table"] = arcs_list

init_data()

###################
# 2) pick_with_reroll_replacement
###################

def pick_with_reroll_replacement(n=3):
    """
    Pick n archetypes from the entire table. If any is a 'placeholder' (i.e. name includes "Add an extra modifier"),
    replace it with a real pick plus add an extra real archetype.
    """
    all_arcs = DATA["archetypes_table"]
    placeholders = [a for a in all_arcs if "Add an extra modifier" in a["name"]]
    reals = [a for a in all_arcs if "Add an extra modifier" not in a["name"]]

    chosen = random.sample(all_arcs, n)  # from entire set

    final_list = []
    for arc in chosen:
        if arc in placeholders:
            # Replace the placeholder with a real pick
            real_pick = random.choice(reals)
            final_list.append(real_pick)
            # plus an extra real archetype
            extra_pick = random.choice(reals)
            final_list.append(extra_pick)
        else:
            final_list.append(arc)

    return final_list

###################
# 3) Stat Combiner
###################

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def combine_archetype_stats(archetype_list):
    """
    For each archetype in archetype_list, we pick random stats from
    baseline_stats ranges + modifiers, sum them, then average + clamp.
    """
    sums = {
        "dominance": 0,
        "cruelty": 0,
        "closeness": 0,
        "trust": 0,
        "respect": 0,
        "intensity": 0
    }
    count = len(archetype_list)
    if count == 0:
        for k in sums:
            sums[k] = random.randint(0, 30)
        return sums

    for arc in archetype_list:
        bs = arc.get("baseline_stats", {})
        for stat_key in sums:
            rng_key = f"{stat_key}_range"
            mod_key = f"{stat_key}_modifier"
            if rng_key in bs and mod_key in bs:
                low, high = bs[rng_key]
                mod = bs[mod_key]
                val = random.randint(low, high) + mod
            else:
                val = random.randint(0, 30)
            sums[stat_key] += val

    for sk in sums:
        sums[sk] = sums[sk] / count
        # clamp
        if sk in ["trust", "respect"]:
            sums[sk] = clamp(int(sums[sk]), -100, 100)
        else:
            sums[sk] = clamp(int(sums[sk]), 0, 100)
    return sums

###################
# 4) GPT synergy calls
###################

def get_archetype_synergy_description(archetypes_list, provided_npc_name=None):
    """
    Summon GPT for synergy text. Return JSON with "npc_name" and "archetype_summary".
    """
    if not archetypes_list:
        default_name = provided_npc_name or f"NPC_{random.randint(1000,9999)}"
        return json.dumps({
            "npc_name": default_name,
            "archetype_summary": "No special archetype synergy."
        })

    archetype_names = [a["name"] for a in archetypes_list]
    if provided_npc_name:
        name_instruction = f"Use the provided NPC name: '{provided_npc_name}'"
    else:
        name_instruction = "Generate a creative, unique name for the NPC."

    system_msg = (
        f"You are an expert creative writer merging these archetypes: {', '.join(archetype_names)}.\n"
        f"{name_instruction}\n"
        "Output a JSON with exactly two keys: \"npc_name\" and \"archetype_summary\".\n"
        "No extra text, no markdown."
    )

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_msg}],
            temperature=0.7,
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"[get_archetype_synergy_description] error: {e}")
        fallback_name = provided_npc_name or f"NPC_{random.randint(1000,9999)}"
        return json.dumps({
            "npc_name": fallback_name,
            "archetype_summary": "GPT call failed or synergy unavailable."
        })

def get_archetype_extras_summary(archetypes_list, npc_name):
    lines = []
    for arc in archetypes_list:
        lines.append(f"{arc['name']}: synergy or extra traits.")
    joined = "\n".join(lines)
    return f"Additional synergy details for {npc_name}:\n{joined}"

###################
# 5) Create partial NPC
###################

def create_npc_partial(sex="female", total_archetypes=3) -> dict:
    """
    If female => pick `total_archetypes` with pick_with_reroll_replacement,
    combine stats => synergy => final partial NPC.
    If male => skip archetypes and random stats.
    """
    if sex.lower() == "male":
        # random stats only
        final_stats = {
            "dominance": random.randint(0, 30),
            "cruelty": random.randint(0, 30),
            "closeness": random.randint(0, 30),
            "trust": random.randint(-30, 30),
            "respect": random.randint(-30, 30),
            "intensity": random.randint(0, 30)
        }
        chosen_arcs = []
    else:
        chosen_arcs = pick_with_reroll_replacement(total_archetypes)
        final_stats = combine_archetype_stats(chosen_arcs)

    synergy_str = get_archetype_synergy_description(chosen_arcs, None)
    try:
        synergy_data = json.loads(synergy_str)
        synergy_name = synergy_data.get("npc_name", f"NPC_{random.randint(1000,9999)}")
        synergy_text = synergy_data.get("archetype_summary", "")
    except:
        synergy_name = f"NPC_{random.randint(1000,9999)}"
        synergy_text = "No synergy text"

    extras_text = get_archetype_extras_summary(chosen_arcs, synergy_name)

    # minimal JSON representation
    arcs_for_json = [{"name": arc["name"]} for arc in chosen_arcs]

    # random flavor
    hpool = DATA["hobbies_pool"]
    ppool = DATA["personality_pool"]
    lpool = DATA["likes_pool"]
    dpool = DATA["dislikes_pool"]

    hobbies = random.sample(hpool, min(3, len(hpool)))
    personalities = random.sample(ppool, min(3, len(ppool)))
    likes = random.sample(lpool, min(3, len(lpool)))
    dislikes = random.sample(dpool, min(3, len(dpool)))

    # build partial NPC
    npc_dict = {
        "npc_name": synergy_name,
        "introduced": False,
        "sex": sex.lower(),
        "dominance": final_stats["dominance"],
        "cruelty": final_stats["cruelty"],
        "closeness": final_stats["closeness"],
        "trust": final_stats["trust"],
        "respect": final_stats["respect"],
        "intensity": final_stats["intensity"],
        "archetypes": arcs_for_json,
        "archetype_summary": synergy_text,
        "archetype_extras_summary": extras_text,
        "hobbies": hobbies,
        "personality_traits": personalities,
        "likes": likes,
        "dislikes": dislikes,
        "age": random.randint(20, 45),
        "birthdate": "1000-02-10"  # originally just a string
    }

    # --- FIX: Convert birthdate string to a real date ---
    birth_str = npc_dict["birthdate"]  # e.g. "1000-02-10"
    npc_dict["birthdate"] = datetime.strptime(birth_str, "%Y-%m-%d").date()

    return npc_dict

###################
# 6) refine_npc_with_gpt
###################

async def refine_npc_with_gpt(
    npc_partial: dict,
    environment_desc: str,
    day_names: list,
    conversation_id: int
) -> dict:
    """
    Asks GPT to fill in or adapt:
      - npc_name,
      - physical_description,
      - schedule (with day_names),
      - affiliations,
      - memory (past events),
      - current_location
    Returns that as a JSON object.
    """
    prompt = f"""
We have a partially created NPC in a femdom environment. Partial data:
{json.dumps(npc_partial, indent=2)}

Environment description:
{environment_desc}

We want to fill or adapt these fields:
  - npc_name (confirm or revise),
  - physical_description (a paragraph),
  - schedule: use these day names => {day_names} (Morning,Afternoon,Evening,Night),
  - affiliations,
  - memory (short array of meaningful past events),
  - current_location (one from the schedule)

Return only JSON with keys:
  "npc_name", "physical_description", "schedule", "affiliations", "memory", "current_location"

No extra text or function calls.
"""

    # For demonstration, we just do a direct GPT call
    await asyncio.sleep(1.0)
    raw_gpt = await asyncio.to_thread(
        get_chatgpt_response, conversation_id, environment_desc, prompt
    )

    if raw_gpt.get("type") == "function_call":
        return raw_gpt.get("function_args", {})
    else:
        text = raw_gpt.get("response", "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            return parsed
        except Exception as e:
            logging.warning(f"[refine_npc_with_gpt] parse error: {e}")
            return {}

###################
# 7) Relationship + Archetype expansions
###################

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


def recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id):
    """
    Re-fetch the NPC's archetypes from the DB and re-run combine_archetype_stats
    to update the final stats.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT archetypes FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
        (user_id, conversation_id, npc_id)
    )
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

    # match by name
    chosen_arcs = []
    for arc_obj in arcs_list:
        a_name = arc_obj.get("name")
        found = None
        for cand in DATA["archetypes_table"]:
            if cand["name"] == a_name:
                found = cand
                break
        if found:
            chosen_arcs.append(found)
        else:
            chosen_arcs.append({"baseline_stats": {}})

    final_stats = combine_archetype_stats(chosen_arcs)

    cursor.execute("""
        UPDATE NPCStats
        SET dominance=%s, cruelty=%s, closeness=%s, trust=%s, respect=%s, intensity=%s
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (
        final_stats["dominance"],
        final_stats["cruelty"],
        final_stats["closeness"],
        final_stats["trust"],
        final_stats["respect"],
        final_stats["intensity"],
        user_id, conversation_id, npc_id
    ))
    conn.commit()
    conn.close()

    logging.info(f"[recalc_npc_stats_with_new_archetypes] updated => {final_stats} for npc_id={npc_id}.")

async def await_prompted_synergy_after_add_archetype(arcs_list, user_id, conversation_id, npc_id):
    """
    If we just added a new archetype, re-run synergy to incorporate it.
    arcs_list ~ [{"name":"Mother"}, ...]
    """
    archetype_names = [arc["name"] for arc in arcs_list]
    system_instructions = f"""
We just appended a new archetype to this NPC. Now they have: {', '.join(archetype_names)}.
Please provide an updated synergy summary, in JSON with key "archetype_summary".
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
        return data.get("archetype_summary", "")
    except Exception as e:
        logging.warning(f"Error synergy after new arc: {e}")
        return "Could not update synergy"

async def add_archetype_to_npc(user_id, conversation_id, npc_id, new_arc):
    """
    Insert new_arc into the NPC's archetypes array, re-run synergy, recalc stats.
    new_arc e.g. {"id": 9001, "name": "Maternal Overlord"}
    We'll store only 'name' in the DB, ignoring 'id' for consistency.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT archetypes FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s LIMIT 1",
        (user_id, conversation_id, npc_id)
    )
    row = cursor.fetchone()
    if not row:
        logging.warning(f"No NPCStats found for npc_id={npc_id}, can't add archetype.")
        conn.close()
        return

    arcs_str = row[0] or "[]"
    try:
        existing_arcs = json.loads(arcs_str)
    except:
        existing_arcs = []

    # Only add if not present
    if any(a.get("name") == new_arc["name"] for a in existing_arcs):
        logging.info(f"NPC {npc_id} already has archetype '{new_arc['name']}'; skipping.")
        conn.close()
        return

    existing_arcs.append({"name": new_arc["name"]})
    new_arcs_json = json.dumps(existing_arcs)

    updated_synergy = await await_prompted_synergy_after_add_archetype(existing_arcs, user_id, conversation_id, npc_id)
    if not updated_synergy:
        updated_synergy = "No updated synergy"

    cursor.execute("""
        UPDATE NPCStats
        SET archetypes=%s,
            archetype_summary=%s
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (new_arcs_json, updated_synergy, user_id, conversation_id, npc_id))
    conn.commit()
    conn.close()

    # recalc
    recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id)
    logging.info(f"[add_archetype_to_npc] added '{new_arc['name']}' to npc_id={npc_id}.")

###################
# 8) assign_random_relationships
###################

def assign_random_relationships(user_id, conversation_id, new_npc_id, new_npc_name):
    """
    Potentially create a relationship with the player, or existing NPCs.
    Then record memory, create social link, optionally add an archetype.
    """
    familial = ["mother", "sister", "aunt"]
    non_familial = ["enemy", "friend", "lover", "neighbor",
                    "colleague", "classmate", "teammate",
                    "underling", "CEO"]

    relationships = []

    # 50% chance to link with the player
    if random.random() < 0.5:
        if random.random() < 0.2:
            rel_type = random.choice(familial)
        else:
            rel_type = random.choice(non_familial)
        relationships.append({"target": "player", "target_name": "the player", "type": rel_type})

    # gather existing NPCs
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT npc_id, npc_name FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id!=%s",
        (user_id, conversation_id, new_npc_id)
    )
    rows = cursor.fetchall()
    conn.close()

    for (candidate_id, candidate_name) in rows:
        # 30% chance
        if random.random() < 0.3:
            if random.random() < 0.2:
                rel_type = random.choice(familial)
            else:
                rel_type = random.choice(non_familial)
            # avoid duplicates
            if not any(r.get("target") == candidate_id for r in relationships):
                relationships.append({
                    "target": candidate_id,
                    "target_name": candidate_name,
                    "type": rel_type
                })

    # process
    for rel in relationships:
        memory_text = get_shared_memory(user_id, conversation_id, rel, new_npc_name)
        record_npc_event(user_id, conversation_id, new_npc_id, memory_text)

        # create the link
        if rel["target"] == "player":
            create_social_link(user_id, conversation_id, "player", 0, "npc", new_npc_id, link_type=rel["type"], link_level=0)
        else:
            create_social_link(user_id, conversation_id, "npc", rel["target"], "npc", new_npc_id, link_type=rel["type"], link_level=0)

        # special arc if in RELATIONSHIP_ARCHETYPE_MAP
        special_arc = RELATIONSHIP_ARCHETYPE_MAP.get(rel["type"])
        if special_arc:
            logging.info(f"[assign_random_relationships] Relationship '{rel['type']}' => add {special_arc}")
            import asyncio
            asyncio.run(add_archetype_to_npc(user_id, conversation_id, new_npc_id, special_arc))

    logging.info(f"[assign_random_relationships] done for NPC '{new_npc_name}' (id={new_npc_id}).")

async def refine_multiple_npcs_individually(
    count: int,
    environment_desc: str,
    day_names: list,
    conversation_id: int
) -> list:
    """
    1) Creates `count` partial NPCs (female by default).
    2) For each partial, calls refine_npc_with_gpt(...) => environment-appropriate.
    3) Calls adjust_npc_complete(...) to fill missing keys if GPT didn't provide them.
    4) Returns a list of fully refined NPC dictionaries (still missing schedule).
    """
    from logic.gpt_helpers import adjust_npc_complete  # assuming your code structure

    refined_npcs = []

    for _ in range(count):
        # Step A: partial creation
        partial_npc = create_npc_partial(sex="female", total_archetypes=3)

        # If your DB column is a date type, parse:
        birth_str = partial_npc["birthdate"]  # e.g. "1000-02-10"
        partial_npc["birthdate"] = datetime.strptime(birth_str, "%Y-%m-%d").date()

        # Step B: refine with GPT to remove anachronisms, etc.
        refined_result = await refine_npc_with_gpt(
            npc_partial=partial_npc,
            environment_desc=environment_desc,
            day_names=day_names,
            conversation_id=conversation_id
        )

        # Merge
        for k, v in refined_result.items():
            partial_npc[k] = v

        # Step C: ensure all required keys are present
        # e.g. "likes","dislikes","hobbies","affiliations","schedule" 
        # (We do a final check, even though we haven't done the big schedule pass yet.)
        adjusted_npc = await adjust_npc_complete(
            npc_data=partial_npc,
            environment_desc=environment_desc,
            conversation_id=conversation_id,
            immersive_days=day_names,  # pass dayNames if you want schedule placeholders
            max_retries=2  # or however many times you want
        )

        refined_npcs.append(adjusted_npc)

    return refined_npcs

###################
# 9) spawn_and_refine_npcs_with_relationships
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
    1) create partial NPCs => refine individually => fill missing keys
    2) single GPT call => final schedules for all NPCs + chase
    3) store in DB => assign relationships
    """

    # A) Create & refine each NPC
    refined_npcs_no_schedules = await refine_multiple_npcs_individually(
        count=count,
        environment_desc=environment_desc,
        day_names=day_names,
        conversation_id=conversation_id
    )

    # B) Single GPT call => final schedules for them + "Chase"
    from logic.npc_creation import call_gpt_for_final_schedules  # or wherever you define it
    schedule_data = await call_gpt_for_final_schedules(
        conversation_id=conversation_id,
        refined_npcs=refined_npcs_no_schedules,
        environment_desc=environment_desc,
        day_names=day_names
    )
    # schedule_data might be: { "npc_creations": [...], "ChaseSchedule": {...} }

    if not schedule_data:
        logging.warning("No schedule data returned by GPT.")
        return {"error": "No final schedules generated."}

    # C) Merge final schedules into each refined NPC
    # GPT might have adjusted the same NPC_name, or kept them identical.
    name_map = {}
    for npc_obj in schedule_data.get("npc_creations", []):
        key_nm = npc_obj.get("npc_name", "").lower()
        name_map[key_nm] = npc_obj

    final_npcs = []
    for npc_ref in refined_npcs_no_schedules:
        nm = npc_ref["npc_name"].lower()
        matched_sched_npc = name_map.get(nm)
        if matched_sched_npc:
            # Overwrite schedule and maybe affiliations, likes, etc. 
            # (since GPT might have updated them in the final pass)
            npc_ref["schedule"]     = matched_sched_npc.get("schedule", {})
            npc_ref["affiliations"] = matched_sched_npc.get("affiliations", npc_ref["affiliations"])
            npc_ref["likes"]        = matched_sched_npc.get("likes", npc_ref["likes"])
            npc_ref["dislikes"]     = matched_sched_npc.get("dislikes", npc_ref["dislikes"])
            npc_ref["hobbies"]      = matched_sched_npc.get("hobbies", npc_ref["hobbies"])
        else:
            logging.warning(f"No schedule found for NPC '{npc_ref['npc_name']}'")
        final_npcs.append(npc_ref)

    chase_schedule = schedule_data.get("ChaseSchedule", {})

    # D) Insert them using universal_updater
    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "npc_creations": final_npcs,
        "ChaseSchedule": chase_schedule
    }
    res = await apply_universal_updates_async(user_id, conversation_id, payload, conn)
    logging.info(f"[spawn_and_refine_npcs_with_relationships] universal update => {res}")

    # E) Assign relationships
    for npc_data in final_npcs:
        npc_name = npc_data["npc_name"]
        row = await conn.fetchrow("""
            SELECT npc_id 
            FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2
              AND LOWER(npc_name)=LOWER($3)
            LIMIT 1
        """, user_id, conversation_id, npc_name)
        if row:
            new_npc_id = row["npc_id"]
            assign_random_relationships(user_id, conversation_id, new_npc_id, npc_name)
        else:
            logging.warning(f"[spawn_and_refine_npcs_with_relationships] No DB row found for newly created NPC '{npc_name}'")

    logging.info(f"[spawn_and_refine_npcs_with_relationships] assigned relationships for {count} new NPCs.")
    return {"message": f"Spawned {count} NPCs and assigned relationships."}
