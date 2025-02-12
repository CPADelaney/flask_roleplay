# logic/npc_creation.py

import os
import json
import random
import logging
import asyncio
from datetime import datetime

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

NPC_PROMPT = """
You are finalizing schedules for multiple NPCs in this environment:
{environment_desc}

Here is each NPC’s **fully refined** data (but schedule is missing):
{refined_npc_data}

We also need a schedule for the player "Chase."

Ensure NPC and player schedules are immersive and make sense within the current setting, as well as with the character's role.

Return exactly one JSON with keys:
  "npc_creations": [ {{ ... }}, ... ],
  "ChaseSchedule": {{}}

Where each NPC in "npc_creations" has:
  - "npc_name" (same as in refined_npc_data)
  - "likes", "dislikes", "hobbies", "affiliations"
  - "schedule": with day-based subkeys (Morning, Afternoon, Evening, Night) 
     for these days: {day_names}

Constraints:
- The schedules must reflect each NPC’s existing archetypes/likes/dislikes/hobbies/etc.
- Example: an NPC with a 'student' archetype is likely to have class most of the week.
- Output only JSON, no extra commentary.
"""

print("DEBUG NPC_PROMPT ->", repr(NPC_PROMPT))

def init_data():
    """Load all local JSON data into the DATA dictionary for reuse."""
    hobbies_json = load_json_file(DATA_FILES["hobbies"])
    DATA["hobbies_pool"] = hobbies_json.get("hobbies_pool", [])

    likes_json = load_json_file(DATA_FILES["likes"])
    DATA["likes_pool"] = likes_json.get("npc_likes", [])

    dislikes_json = load_json_file(DATA_FILES["dislikes"])
    DATA["dislikes_pool"] = dislikes_json.get("dislikes_pool", [])

    personalities_json = load_json_file(DATA_FILES["personalities"])
    DATA["personality_pool"] = personalities_json.get("personality_pool", [])

    arcs_json = load_json_file(DATA_FILES["archetypes"])
    table = arcs_json.get("archetypes", [])

    arcs_list = []
    for item in table:
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
            real_pick = random.choice(reals)
            final_list.append(real_pick)
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
    Combine baseline stats from each archetype, then average + clamp.
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
        if sk in ["trust", "respect"]:
            sums[sk] = clamp(int(sums[sk]), -100, 100)
        else:
            sums[sk] = clamp(int(sums[sk]), 0, 100)
    return sums

###################
# 4) GPT synergy calls
###################

def get_archetype_synergy_description(archetypes_list, provided_npc_name=None):
    if not archetypes_list:
        default_name = provided_npc_name or f"NPC_{random.randint(1000,9999)}"
        logging.info("[get_archetype_synergy_description] No archetypes => using placeholder name.")
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

    logging.info(f"[get_archetype_synergy_description] GPT prompt => {system_msg}")

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_msg}],
            temperature=0.7,
            max_tokens=300
        )
        synergy_str = resp.choices[0].message.content.strip()

        logging.info(f"[get_archetype_synergy_description] Raw synergy GPT output => {synergy_str}")

        return synergy_str
    except Exception as e:
        logging.error(f"[get_archetype_synergy_description] error: {e}")
        fallback_name = provided_npc_name or f"NPC_{random.randint(1000,9999)}"
        return json.dumps({
            "npc_name": fallback_name,
            "archetype_summary": "GPT call failed or synergy unavailable."
        })

def get_archetype_extras_summary_gpt(archetypes_list, npc_name):
    """
    Calls GPT to merge each archetype's progression_rules, unique_traits, and preferred_kinks
    into one cohesive textual summary. Returns a string stored as "archetype_extras_summary".

    The final JSON must have exactly 1 key: "archetype_extras_summary".
    If GPT can't provide it, fallback to a minimal text.
    """
    import json

    if not archetypes_list:
        return "No extras summary available."

    # Gather the data from each archetype
    lines = []
    for arc in archetypes_list:
        name = arc["name"]
        # These fields might be missing or empty, so we guard with .get(...)
        progression = arc.get("progression_rules", [])
        traits = arc.get("unique_traits", [])
        kinks = arc.get("preferred_kinks", [])
        lines.append(
            f"Archetype: {name}\n"
            f"  progression_rules: {progression}\n"
            f"  unique_traits: {traits}\n"
            f"  preferred_kinks: {kinks}\n"
        )

    combined_text = "\n".join(lines)

    # We'll build a system prompt that clarifies the format we want:
    system_prompt = f"""
You are merging multiple archetype 'extras' for an NPC named '{npc_name}'.
Below are the extras from each archetype:

{combined_text}

We want to unify these details (progression_rules, unique_traits, preferred_kinks)
into one cohesive textual summary that references how these merges shape the NPC’s special powers,
quirks, or flavor.

Return **strictly valid JSON** with exactly 1 top-level key:
  "archetype_extras_summary"

The value should be a single string that concisely describes how these rules/traits/kinks
blend into one cohesive set of extras for this NPC.

No extra commentary, no additional keys, no code fences.
If you cannot comply, return an empty JSON object {{}}.
"""

    # Next, we do a GPT call with system_prompt
    client = get_openai_client()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            max_tokens=400
        )
        raw_text = resp.choices[0].message.content.strip()
        logging.info(f"[get_archetype_extras_summary_gpt] raw GPT extras => {raw_text!r}")

        # Remove fences if present
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            # remove first triple-backticks
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            # remove last triple-backticks
            if lines and lines[-1].startswith("```"):
                lines.pop()
            raw_text = "\n".join(lines).strip()

        parsed = json.loads(raw_text)
        if (
            isinstance(parsed, dict)
            and "archetype_extras_summary" in parsed
        ):
            # Valid
            return parsed["archetype_extras_summary"]
        else:
            logging.warning("[get_archetype_extras_summary_gpt] Missing 'archetype_extras_summary' key, falling back.")
            return "No extras summary available."

    except Exception as e:
        logging.warning(f"[get_archetype_extras_summary_gpt] error => {e}")
        return "No extras summary available."

def get_archetype_synergy_description(archetypes_list, provided_npc_name=None):
    """
    Generate synergy text for the given archetypes via GPT, returning a JSON string
    with exactly two top-level keys: "npc_name" and "archetype_summary".

    This version is stricter about the output, logs the raw GPT response, 
    strips code fences, and falls back gracefully if it can't parse valid JSON.
    """
    if not archetypes_list:
        default_name = provided_npc_name or f"NPC_{random.randint(1000,9999)}"
        return json.dumps({
            "npc_name": default_name,
            "archetype_summary": "No special archetype synergy."
        })

    archetype_names = [a["name"] for a in archetypes_list]

    if provided_npc_name:
        name_instruction = f'Use the provided NPC name: "{provided_npc_name}".'
    else:
        name_instruction = "Invent a creative, unique name for the NPC."

    system_prompt = (
        "You are an expert at merging multiple archetypes into a single cohesive persona. "
        "Output **strictly valid JSON** with exactly these two keys:\n"
        '  "npc_name" (string)\n'
        '  "archetype_summary" (string)\n'
        "No additional keys, no extra commentary, no markdown fences.\n\n"
        "If you cannot comply, output an empty JSON object {}.\n\n"
        f"Archetypes to merge: {', '.join(archetype_names)}\n"
        f"{name_instruction}\n"
    )

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        synergy_raw = resp.choices[0].message.content.strip()
        logging.info(f"[get_archetype_synergy_description] Raw synergy GPT output => {synergy_raw!r}")

        # Strip code fences if present
        if synergy_raw.startswith("```"):
            lines = synergy_raw.splitlines()
            # remove the first ``` line
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            # remove the last ``` line if present
            if lines and lines[-1].startswith("```"):
                lines.pop()
            synergy_raw = "\n".join(lines).strip()

        # Attempt to parse JSON
        synergy_data = json.loads(synergy_raw)

        # Validate the essential keys are present
        if not isinstance(synergy_data, dict):
            logging.warning("[get_archetype_synergy_description] synergy_data is not a dict—falling back.")
            return "{}"
        if "npc_name" not in synergy_data or "archetype_summary" not in synergy_data:
            logging.warning("[get_archetype_synergy_description] synergy_data missing required keys—falling back.")
            return "{}"

        # If we got here, synergy_data should have the right shape
        return json.dumps(synergy_data, ensure_ascii=False)

    except Exception as e:
        logging.warning(f"[get_archetype_synergy_description] parse or GPT error: {e}")
        # Return an empty JSON, or if you prefer a minimal fallback
        return "{}"


def create_npc_partial(sex="female", total_archetypes=3) -> dict:
    """
    Creates a partial NPC. Leaves 'birthdate' as a simple string.
    Logs the synergy string prior to parse, logs the partial NPC afterwards.
    """
    if sex.lower() == "male":
        final_stats = {
            "dominance":  random.randint(0, 30),
            "cruelty":    random.randint(0, 30),
            "closeness":  random.randint(0, 30),
            "trust":      random.randint(-30, 30),
            "respect":    random.randint(-30, 30),
            "intensity":  random.randint(0, 30)
        }
        chosen_arcs = []
    else:
        chosen_arcs = pick_with_reroll_replacement(total_archetypes)
        final_stats = combine_archetype_stats(chosen_arcs)

    synergy_str = get_archetype_synergy_description(chosen_arcs, None)  # returns valid JSON or "{}"
    logging.info(f"[create_npc_partial] synergy_str (raw) => {synergy_str!r}")

    # Attempt to parse synergy JSON:
    try:
        synergy_data = json.loads(synergy_str)
        synergy_name = synergy_data.get("npc_name") or f"NPC_{random.randint(1000,9999)}"
        synergy_text = synergy_data.get("archetype_summary") or "No synergy text"
    except json.JSONDecodeError as e:
        logging.warning(f"[create_npc_partial] synergy parse error => {e}")
        synergy_name = f"NPC_{random.randint(1000,9999)}"
        synergy_text = "No synergy text"

    # 2) extras approach (call GPT):
    extras_text = get_archetype_extras_summary_gpt(chosen_arcs, synergy_name)
    arcs_for_json = [{"name": arc["name"]} for arc in chosen_arcs]

    # Random flavor
    hpool = DATA["hobbies_pool"]
    ppool = DATA["personality_pool"]
    lpool = DATA["likes_pool"]
    dpool = DATA["dislikes_pool"]

    # Random birthdate in some approximate medieval range
    year  = random.randint(990, 1040)
    month = random.randint(1, 12)
    day   = random.randint(1, 28)
    birth_str = f"{year:04d}-{month:02d}-{day:02d}"

    npc_dict = {
        "npc_name":    synergy_name,
        "introduced":  False,
        "sex":         sex.lower(),
        "dominance":   final_stats["dominance"],
        "cruelty":     final_stats["cruelty"],
        "closeness":   final_stats["closeness"],
        "trust":       final_stats["trust"],
        "respect":     final_stats["respect"],
        "intensity":   final_stats["intensity"],
        "archetypes":  arcs_for_json,
        "archetype_summary": synergy_text,
        "archetype_extras_summary": extras_text,
        "hobbies":     random.sample(hpool, min(3, len(hpool))),
        "personality_traits": random.sample(ppool, min(3, len(ppool))),
        "likes":       random.sample(lpool, min(3, len(lpool))),
        "dislikes":    random.sample(dpool, min(3, len(dpool))),
        "age":         random.randint(20, 45),
        "birthdate":   birth_str
    }

    logging.info(
        "[create_npc_partial] Created partial NPC => "
        f"name='{npc_dict['npc_name']}', arcs={[arc['name'] for arc in chosen_arcs]}, "
        f"archetype_summary='{npc_dict['archetype_summary']}', birthdate={npc_dict['birthdate']}"
    )
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
    prompt = f"""
We have a partially created NPC in a femdom environment. Partial data:
{json.dumps(npc_partial, indent=2)}

Environment description:
{environment_desc}

We want to fill or adapt these fields:
  - npc_name (confirm or revise),
  - physical_description,
  - schedule (using day names => {day_names}),
  - affiliations,
  - memory (past events),
  - current_location

Return only JSON with keys:
  "npc_name", "physical_description", "schedule", "affiliations", "memory", "current_location"

No extra text or function calls.
"""

    logging.info(f"[refine_npc_with_gpt] Requesting GPT refine => partial_npc_name='{npc_partial['npc_name']}'")

    raw_gpt = await asyncio.to_thread(
        get_chatgpt_response,
        conversation_id,
        environment_desc,
        prompt
    )
    # ^ If you do spaced_gpt_call, then you'd do that. But let's assume get_chatgpt_response is your function.

    if raw_gpt.get("type") == "function_call":
        result_dict = raw_gpt.get("function_args", {})
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
            result_dict = json.loads(text)
        except Exception as e:
            logging.warning(f"[refine_npc_with_gpt] parse error: {e}")
            result_dict = {}

    logging.info(
        f"[refine_npc_with_gpt] GPT refine result => npc_name={result_dict.get('npc_name')}, "
        f"affiliations={result_dict.get('affiliations')}, schedule_len={len(result_dict.get('schedule', {}))}"
    )

    return result_dict



###################
# 7) Relationship + Archetype expansions
###################

RELATIONSHIP_ARCHETYPE_MAP = {
    # Family / Household
    "Mother":        {"name": "Child"},
    "Stepmother":    {"name": "Step-Child"},
    "Aunt":          {"name": "Niece/Nephew"},
    "Older Sister":  {"name": "Younger Sibling"},
    "Stepsister":    {"name": "Step-Sibling"},
    "Babysitter":    {"name": "Child"},

    # Workplace / Power
    "CEO":                {"name": "Employee"},
    "Boss/Supervisor":    {"name": "Employee"},
    "Corporate Dominator":{"name": "Underling"},
    "Teacher/Principal":  {"name": "Student"},
    "Landlord":           {"name": "Tenant"},
    "Warden":             {"name": "Prisoner"},
    "Loan Shark":         {"name": "Debtor"},
    "Slave Overseer":     {"name": "Slave"},
    "Therapist":          {"name": "Patient"},
    "Doctor":             {"name": "Patient"},
    "Social Media Influencer": {"name": "Follower"},
    "Bartender":          {"name": "Patron"},
    "Fitness Trainer":    {"name": "Client"},
    "Cheerleader/Team Captain": {"name": "Junior Team Member"},
    "Martial Artist":     {"name": "Sparring Dummy"},
    "Professional Wrestler": {"name": "Defeated Opponent"},

    # Supernatural / Hunting
    "Demon":              {"name": "Thrall"},
    "Demoness":           {"name": "Bound Mortal"},
    "Devil":              {"name": "Damned Soul"},
    "Villain (RPG-Esque)": {"name": "Captured Hero"},
    "Haunted Entity":     {"name": "Haunted Mortal"},
    "Sorceress":          {"name": "Cursed Subject"},
    "Witch":              {"name": "Hexed Victim"},
    "Eldritch Abomination":{"name": "Insane Acolyte"},
    "Primal Huntress":    {"name": "Prey"},
    "Primal Predator":    {"name": "Prey"},
    "Serial Killer":      {"name": "Victim"},

    # Others
    "Rockstar":           {"name": "Fan"},
    "Celebrity":          {"name": "Fan"},
    "Ex-Girlfriend/Ex-Wife": {"name": "Ex-Partner"},
    "Politician":         {"name": "Constituent"},
    "Queen":              {"name": "Subject"},
    "Empress":            {"name": "Subject"},
    "Royal Knight":       {"name": "Challenged Rival"},
    "Gladiator":          {"name": "Arena Opponent"},
    "Pirate":             {"name": "Captive"},
    "Bank Robber":        {"name": "Hostage"},
    "Cybercriminal":      {"name": "Hacked Victim"},
    "Huntress":           {"name": "Prey"}, 
    "Arsonist":           {"name": "Burned Victim"},
    "Drug Dealer":        {"name": "Addict"},
    "Artificial Intelligence": {"name": "User/Victim"},
    "Fey":                {"name": "Ensorcelled Mortal"},
    "Nun":                {"name": "Sinner"},
    "Priestess":          {"name": "Acolyte"},
    "A True Goddess":     {"name": "Worshipper"},
    "Haruhi Suzumiya-Type Goddess": {"name": "Reality Pawn"},
    "Bowsette Personality": {"name": "Castle Captive"},
    "Juri Han Personality": {"name": "Beaten Opponent"},
    "Neighbor":           {"name": "Targeted Neighbor"},
    "Hero (RPG-Esque)":   {"name": "Sidekick / Rescued Target"},
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

async def append_relationship_to_npc(
    user_id: int,
    conversation_id: int,
    npc_id: int,
    rel_label: str,
    target_npc_id: int
):
    """
    Appends a new relationship object into the 'relationships' column
    for the given NPC.
    e.g. { "relationship_label":"Thrall", "with_npc_id":123 }
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # 1) Fetch existing relationships
    cursor.execute(
        "SELECT relationships FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
        (user_id, conversation_id, npc_id)
    )
    row = cursor.fetchone()
    if not row:
        logging.warning(f"[append_relationship_to_npc] NPC {npc_id} not found. Cannot store relationship.")
        conn.close()
        return

    relationships_data = row[0] or []
    try:
        if isinstance(relationships_data, str):
            relationships_data = json.loads(relationships_data)
    except:
        relationships_data = []

    # 2) Append
    new_rel_obj = {
        "relationship_label": rel_label,
        "with_npc_id": target_npc_id
    }
    relationships_data.append(new_rel_obj)

    # 3) Update
    cursor.execute(
        """
        UPDATE NPCStats
        SET relationships = %s
        WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
        """,
        (json.dumps(relationships_data), npc_id, user_id, conversation_id)
    )
    conn.commit()
    conn.close()
    logging.info(f"[append_relationship_to_npc] Added relationship '{rel_label}' -> npc_id={target_npc_id} for npc_id={npc_id}.")

def recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id):
    """
    Re-fetch the NPC's archetypes from the DB and re-run combine_archetype_stats to update final stats.
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

async def assign_random_relationships(user_id, conversation_id, new_npc_id, new_npc_name):
    """
    Potentially create a relationship with the player, or existing NPCs.
    Then record memory, create social link, optionally add an archetype.
    """
    familial = ["mother", "sister", "aunt"]
    non_familial = ["enemy", "friend", "lover", "neighbor",
                    "colleague", "classmate", "teammate",
                    "underling", "CEO", "rival"]

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

        # if there's a special archetype for this relationship, we do:
        special_arc = RELATIONSHIP_ARCHETYPE_MAP.get(rel["type"])
        if special_arc:
            logging.info(f"[assign_random_relationships] Relationship '{rel['type']}' => add {special_arc}")
            # *** FIX: We await the add_archetype_to_npc(...) call ***
            await add_archetype_to_npc(user_id, conversation_id, new_npc_id, special_arc)

    logging.info(f"[assign_random_relationships] done for NPC '{new_npc_name}' (id={new_npc_id}).")

async def refine_multiple_npcs_individually(
    count: int,
    environment_desc: str,
    day_names: list,
    conversation_id: int
) -> list:
    from logic.gpt_helpers import adjust_npc_complete

    refined_npcs = []

    for i in range(count):
        partial_npc = create_npc_partial(sex="female")
        logging.info(f"[refine_multiple_npcs] Step1 partial => index={i}, name='{partial_npc['npc_name']}'")

        refined_result = await refine_npc_with_gpt(
            npc_partial=partial_npc,
            environment_desc=environment_desc,
            day_names=day_names,
            conversation_id=conversation_id
        )

        for k, v in refined_result.items():
            partial_npc[k] = v
        logging.info(f"[refine_multiple_npcs] After refine => name='{partial_npc['npc_name']}'")

        # Step C: ensure required keys
        adjusted_npc = await adjust_npc_complete(
            npc_data=partial_npc,
            environment_desc=environment_desc,
            conversation_id=conversation_id,
            immersive_days=day_names,
            max_retries=2
        )
        logging.info(f"[refine_multiple_npcs] After adjust_npc_complete => name='{adjusted_npc['npc_name']}'")

        refined_npcs.append(adjusted_npc)

    return refined_npcs


async def call_gpt_for_final_schedules(
    conversation_id: int,
    refined_npcs: list,
    environment_desc: str,
    day_names: list
) -> dict:
    refined_json_str = json.dumps(refined_npcs, indent=2)

    logging.info(
        f"[call_gpt_for_final_schedules] about to request schedule for {len(refined_npcs)} NPC(s). "
        f"Example name[0] = {refined_npcs[0].get('npc_name') if refined_npcs else 'NONE'}"
    )

    prompt = NPC_PROMPT.format(
        environment_desc=environment_desc,
        refined_npc_data=refined_json_str,
        day_names=", ".join(day_names)
    )

    result = await spaced_gpt_call(
        conversation_id,
        environment_desc,
        prompt,
        delay=1.0
    )

    if result.get("type") == "function_call":
        schedule_data = result.get("function_args", {})
    else:
        raw_text = result.get("response", "").strip()
        # remove triple backticks if present
        if raw_text.startswith("```"):
            ...
        try:
            schedule_data = json.loads(raw_text)
        except Exception as e:
            logging.error(f"[call_gpt_for_final_schedules] parse error: {e}")
            schedule_data = {}

    logging.info(
        f"[call_gpt_for_final_schedules] GPT returned => npc_creations keys: "
        f"{[npc.get('npc_name') for npc in schedule_data.get('npc_creations', [])]}, "
        f"HasChaseSchedule={bool(schedule_data.get('ChaseSchedule'))}"
    )

    return schedule_data


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
    refined_npcs_no_schedules = await refine_multiple_npcs_individually(
        count=count,
        environment_desc=environment_desc,
        day_names=day_names,
        conversation_id=conversation_id
    )
    logging.info("[spawn_and_refine_npcs] All partial + refine done. Checking final name(s):")
    for npc_ref in refined_npcs_no_schedules:
        logging.info(f"   => name='{npc_ref['npc_name']}', likes={npc_ref.get('likes')}, schedule={npc_ref.get('schedule')}")

    schedule_data = await call_gpt_for_final_schedules(
        conversation_id=conversation_id,
        refined_npcs=refined_npcs_no_schedules,
        environment_desc=environment_desc,
        day_names=day_names
    )
    if not schedule_data:
        logging.warning("No schedule data returned by GPT.")
        return {"error": "No final schedules generated."}

    # 3) Merge schedules
    name_map = {}
    for npc_obj in schedule_data.get("npc_creations", []):
        nm = npc_obj.get("npc_name","").lower()
        name_map[nm] = npc_obj

    final_npcs = []
    for npc_ref in refined_npcs_no_schedules:
        nm_lower = npc_ref["npc_name"].lower()
        matched_sched_npc = name_map.get(nm_lower)
        if matched_sched_npc:
            npc_ref["schedule"]     = matched_sched_npc.get("schedule", {})
            npc_ref["affiliations"] = matched_sched_npc.get("affiliations", npc_ref["affiliations"])
            npc_ref["likes"]        = matched_sched_npc.get("likes", npc_ref["likes"])
            npc_ref["dislikes"]     = matched_sched_npc.get("dislikes", npc_ref["dislikes"])
            npc_ref["hobbies"]      = matched_sched_npc.get("hobbies", npc_ref["hobbies"])
            logging.info(f"[spawn_and_refine_npcs] Merged schedule for '{npc_ref['npc_name']}'.")
        else:
            logging.warning(f"[spawn_and_refine_npcs] No schedule for NPC '{npc_ref['npc_name']}'")
        final_npcs.append(npc_ref)

    chase_schedule = schedule_data.get("ChaseSchedule",{})

    # 4) Insert in DB
    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "npc_creations": final_npcs,
        "ChaseSchedule": chase_schedule
    }

    logging.info("[spawn_and_refine_npcs] Final NPC data before universal_updater =>")
    for npc_obj in final_npcs:
        logging.info(
            f"  => name='{npc_obj['npc_name']}', scheduleDays={list(npc_obj.get('schedule', {}).keys())}, "
            f"likesLen={len(npc_obj.get('likes', []))}"
        )

    res = await apply_universal_updates_async(user_id, conversation_id, payload, conn)
    logging.info(f"[spawn_and_refine_npcs] universal update => {res}")

    # 5) assign relationships
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
            # *** NOTICE we "await" now! ***
            await assign_random_relationships(
                user_id, conversation_id, new_npc_id, npc_name
            )
        else:
            logging.warning(f"No DB row found for newly created NPC '{npc_name}'")
