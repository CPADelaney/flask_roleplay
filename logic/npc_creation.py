# logic/npc_creation.py

import os
import json
import random
import logging
import asyncio
from datetime import datetime

from logic.chatgpt_integration import get_openai_client, get_chatgpt_response
from logic.gpt_utils import spaced_gpt_call
from logic.gpt_helpers import fetch_npc_name
from db.connection import get_db_connection
from logic.memory_logic import get_shared_memory, record_npc_event, propagate_shared_memories
from logic.social_links import create_social_link
from logic.calendar import load_calendar_names


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

existing_npc_names = set()

def get_unique_npc_name(proposed_name: str) -> str:
    # If the name is "Seraphina" or already exists, choose an alternative from a predefined list.
    unwanted_names = {"seraphina"}
    if proposed_name.strip().lower() in unwanted_names or proposed_name in existing_npc_names:
        alternatives = ["Aurora", "Celeste", "Luna", "Nova", "Ivy", "Evelyn", "Isolde", "Marina"]
        # Filter out any alternatives already in use
        available = [name for name in alternatives if name not in existing_npc_names and name.lower() not in unwanted_names]
        if available:
            new_name = random.choice(available)
        else:
            # If none available, simply append a random number
            new_name = f"{proposed_name}{random.randint(2, 99)}"
        return new_name
    return proposed_name

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
        name_instruction = f'Use the provided NPC name: "{provided_npc_name}".'
    else:
        name_instruction = (
            "Generate a creative, unique, and fitting feminine name for the NPC. "
            "The name must be unmistakably feminine—do not include any masculine honorifics or traditionally male names (e.g. 'Prince', 'Lord', 'Sir', 'Eduard'). "
            "Do not use 'Seraphina', and ensure the name is not the same as any previously generated name."
        )

    system_prompt = (
        "You are an expert at merging multiple archetypes into a single cohesive persona for a female NPC. "
        "Output strictly valid JSON with exactly these two keys:\n"
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
            if lines and lines[0].startswith("```"):
                lines.pop(0)
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

        # Post-processing: Check for masculine markers in the name.
        npc_name = synergy_data["npc_name"]
        masculine_markers = ["Prince", "Lord", "Sir", "Eduard", "William", "John"]
        if any(marker in npc_name for marker in masculine_markers):
            logging.info("Masculine markers detected in NPC name; replacing with fallback feminine name.")
            synergy_data["npc_name"] = "Lady Celestine"

        # Post-processing: Ensure the name is unique and not overused.
        original_name = synergy_data["npc_name"].strip()
        unique_name = get_unique_npc_name(original_name)
        if unique_name != original_name:
            logging.info(f"Name '{original_name}' replaced with unique name '{unique_name}'.")
        synergy_data["npc_name"] = unique_name
        existing_npc_names.add(unique_name)

        return json.dumps(synergy_data, ensure_ascii=False)

    except Exception as e:
        logging.warning(f"[get_archetype_synergy_description] parse or GPT error: {e}")
        return "{}"

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
You are merging multiple archetype 'extras' for a female NPC named '{npc_name}'.
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
        "You are an expert at merging multiple archetypes into a single cohesive persona for a female NPC. "
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

###################
# 4.5) Adaptation
###################

def adapt_list_for_environment(environment_desc, archetype_summary, original_list, list_type="likes"):
    """
    Calls GPT to adapt each item in 'original_list' so it fits better with the environment
    and the NPC's archetype_summary. 'list_type' can be "likes", "dislikes", or "hobbies"
    to let GPT know how to adapt them.

    Returns a *new* list (strings).
    """
    import json
    if not original_list:
        return original_list

    system_prompt = f"""
Environment:
{environment_desc}

NPC's archetype summary:
{archetype_summary}

Original {list_type} list:
{original_list}

Please transform each item so it fits more cohesively with the environment's theme and the NPC's archetype,
retaining a similar 'topic' or 'concept' but making it more in-universe.

Output strictly valid JSON: a single array of strings, with no extra commentary or keys.
"""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        raw_text = resp.choices[0].message.content.strip()

        # remove triple-backticks if present
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        new_list = json.loads(raw_text)
        if isinstance(new_list, list) and all(isinstance(x, str) for x in new_list):
            return new_list
        else:
            logging.warning(f"[adapt_list_for_environment] GPT returned something not a list of strings, fallback to original.")
            return original_list
    except Exception as e:
        logging.warning(f"[adapt_list_for_environment] GPT error => {e}")
        return original_list

###################
# 5) create_npc_partial
###################

def create_npc_partial(user_id: int, conversation_id: int, sex: str = "female",
                       total_archetypes: int = 4, environment_desc: str = "A default environment") -> dict:
    import random
    calendar_data = load_calendar_names(user_id, conversation_id)
    months_list = calendar_data.get("months", [])
    if len(months_list) < 12:
        months_list = [
            "Frostmoon", "Windspeak", "Bloomrise", "Dawnsveil",
            "Emberlight", "Goldencrest", "Shadowleaf", "Harvesttide",
            "Stormcall", "Nightwhisper", "Snowbound", "Yearsend"
        ]
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
    
    synergy_str = get_archetype_synergy_description(chosen_arcs, None)
    logging.info(f"[create_npc_partial] synergy_str (raw) => {synergy_str!r}")
    try:
        synergy_data = json.loads(synergy_str)
        synergy_name = synergy_data.get("npc_name") or f"NPC_{random.randint(1000,9999)}"
        synergy_text = synergy_data.get("archetype_summary") or "No synergy text"
    except json.JSONDecodeError as e:
        logging.warning(f"[create_npc_partial] synergy parse error => {e}")
        synergy_name = f"NPC_{random.randint(1000,9999)}"
        synergy_text = "No synergy text"

    # 4) extras
    extras_text = get_archetype_extras_summary_gpt(chosen_arcs, synergy_name)
    arcs_for_json = [{"name": arc["name"]} for arc in chosen_arcs]
    hpool = DATA["hobbies_pool"]
    lpool = DATA["likes_pool"]
    dpool = DATA["dislikes_pool"]
    tmp_hobbies  = random.sample(hpool, min(3, len(hpool)))
    tmp_likes    = random.sample(lpool, min(3, len(lpool)))
    tmp_dislikes = random.sample(dpool, min(3, len(dpool)))
    adapted_hobbies  = adapt_list_for_environment(environment_desc, synergy_text, tmp_hobbies, "hobbies")
    adapted_likes    = adapt_list_for_environment(environment_desc, synergy_text, tmp_likes, "likes")
    adapted_dislikes = adapt_list_for_environment(environment_desc, synergy_text, tmp_dislikes, "dislikes")

    # 7) Age + birthdate
    # Define age adjustments (role: (modifier_min, modifier_max))
    role_base_age_ranges = {
        "mother": (30, 55),
        "stepmother": (18, 55),
        "aunt": (25, 60),
        "older sister": (19, 40),
        "stepsister": (18, 45),
        "babysitter": (20, 50),
        "teacher": (30, 50),
        "principal": (30, 50),
        "milf": (30, 60),
        "dowager": (55, 65),
        "domestic authority": (30, 50),
        "foreign royalty": (20, 45),
        "college student": (18, 24),
        "intern": (18, 24),
        "student": (18, 24),
        "manic pixie dream girl": (18, 30)
    }
    
    # Check chosen archetypes for any familial roles
    familial_roles_found = [
        arc["name"].strip().lower() 
        for arc in chosen_arcs 
        if arc["name"].strip().lower() in role_base_age_ranges
    ]
    
    if familial_roles_found:
        # If more than one family role is found, choose the one with the highest minimum age
        selected_role = max(familial_roles_found, key=lambda role: role_base_age_ranges[role][0])
        base_age = random.randint(*role_base_age_ranges[selected_role])
    else:
        base_age = random.randint(20, 50)
    
    # Optionally, you could add a minor random offset if desired:
    # extra_years = random.randint(-2, 2)
    # npc_age = base_age + extra_years
    
    npc_age = base_age  # Now, npc_age is drawn from a role-appropriate range
    
    birth_month = random.choice(months_list)
    birth_day   = random.randint(1, 28)
    birth_str   = f"{birth_month} {birth_day}"

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
        "hobbies": adapted_hobbies,
        "personality_traits": random.sample(DATA["personality_pool"], min(3, len(DATA["personality_pool"]))),
        "likes": adapted_likes,
        "dislikes": adapted_dislikes,
        "age": npc_age,
        "birthdate": birth_str
    }
    logging.info(
        "[create_npc_partial] Created partial NPC => "
        f"name='{npc_dict['npc_name']}', arcs={[arc['name'] for arc in chosen_arcs]}, "
        f"archetype_summary='{npc_dict['archetype_summary']}', "
        f"birthdate={npc_dict['birthdate']}, age={npc_age}"
    )
    return npc_dict

    logging.info(
        "[create_npc_partial] Created partial NPC => "
        f"name='{npc_dict['npc_name']}', arcs={[arc['name'] for arc in chosen_arcs]}, "
        f"archetype_summary='{npc_dict['archetype_summary']}', "
        f"birthdate={npc_dict['birthdate']}, age={npc_age}"
    )

    return npc_dict



###################
# 6) DB Insert Stub
###################
async def insert_npc_stub_into_db(partial_npc: dict, user_id: int, conversation_id: int) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO NPCStats (
          user_id, conversation_id,
          npc_name, introduced, sex,
          dominance, cruelty, closeness, trust, respect, intensity,
          archetypes, archetype_summary, archetype_extras_summary,
          likes, dislikes, hobbies, personality_traits,
          age, birthdate,          -- 'birthdate' now TEXT in DB
          relationships, memory, schedule,
          physical_description
        )
        VALUES (%s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,   -- pass partial_npc["birthdate"] as text
                '[]'::jsonb, '[]'::jsonb, '{}'::jsonb,
                ''
        )
        RETURNING npc_id
        """,
        (
            user_id, conversation_id,
            partial_npc["npc_name"],
            partial_npc.get("introduced", False),
            partial_npc["sex"],

            partial_npc["dominance"],
            partial_npc["cruelty"],
            partial_npc["closeness"],
            partial_npc["trust"],
            partial_npc["respect"],
            partial_npc["intensity"],

            json.dumps(partial_npc["archetypes"]),
            partial_npc.get("archetype_summary", ""),
            partial_npc.get("archetype_extras_summary", ""),

            json.dumps(partial_npc.get("likes", [])),
            json.dumps(partial_npc.get("dislikes", [])),
            json.dumps(partial_npc.get("hobbies", [])),
            json.dumps(partial_npc.get("personality_traits", [])),

            partial_npc.get("age", 25),
            partial_npc.get("birthdate", ""),  # TEXT field now
        )
    )
    row = cur.fetchone()
    npc_id = row[0]
    conn.commit()
    conn.close()
    return npc_id

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

def dynamic_reciprocal_relationship(rel_type: str, archetype_summary: str = "") -> str:
    """
    Given a relationship label (e.g., "thrall", "underling", "friend"),
    return a reciprocal label in a dynamic and context‐sensitive way.
    Fixed family relationships use fixed mappings; for "friend" or "best friend" the reciprocal is identical.
    For other types, we pick from a list, possibly influenced by keywords in the archetype summary.
    """
    fixed = {
        "mother": "child",
        "sister": "younger sibling",
        "aunt": "nephew/niece"
    }
    rel_lower = rel_type.lower()
    if rel_lower in fixed:
        return fixed[rel_lower]
    if rel_lower in ["friend", "best friend"]:
        return rel_type  # mutual relationship
    dynamic_options = {
        "underling": ["boss", "leader", "overseer"],
        "thrall": ["master", "controller", "dominator"],
        "enemy": ["rival", "adversary"],
        "lover": ["lover", "beloved"],
        "colleague": ["colleague"],
        "neighbor": ["neighbor"],
        "classmate": ["classmate"],
        "teammate": ["teammate"],
        "rival": ["rival", "competitor"],
    }
    if rel_lower in dynamic_options:
        if "dominant" in archetype_summary.lower() or "domina" in archetype_summary.lower():
            if rel_lower in ["underling", "thrall"]:
                return "boss"
        return random.choice(dynamic_options[rel_lower])
    return "associate"

    # For other relationships, use a dynamic approach.
    # For example, if the relationship is "thrall" or "underling", maybe the reciprocal is "master" or "boss".
    dynamic_options = {
        "underling": ["boss", "leader", "overseer"],
        "thrall": ["master", "controller", "dominator"],
        "enemy": ["rival", "adversary"],
        "lover": ["lover", "beloved"],
        "colleague": ["colleague"],
        "neighbor": ["neighbor"],
        "classmate": ["classmate"],
        "teammate": ["teammate"],
        "rival": ["rival", "competitor"],
    }
    if rel_lower in dynamic_options:
        # As a simple context-sensitive tweak, if the archetype summary contains keywords like "dominant" (case insensitive),
        # we might lean toward a more authoritative reciprocal.
        if "dominant" in archetype_summary.lower() or "domina" in archetype_summary.lower():
            # For example, if the relationship is "underling", force "boss"
            if rel_lower in ["underling", "thrall"]:
                return "boss"
        return random.choice(dynamic_options[rel_lower])
    # Fallback default:
    return "associate"

###################
# 7) Relationship Archetype Map (legacy example)
###################
"""
In your example, you had a smaller RELATIONSHIP_ARCHETYPE_MAP that 
adds an archetype to the new NPC based on the relationship chosen. 
We'll keep it, but you can combine it with EXTENDED_RECIPROCAL_ARCHETYPES
or keep them separate if you want multi-step logic.
"""

def append_relationship_to_npc(
    user_id, conversation_id,
    npc_id,                     # Which NPC in NPCStats is being updated
    rel_label,                  # "thrall", "lover", etc.
    target_entity_type,         # "npc" or "player"
    target_entity_id            # The actual ID for the other side
):
    """
    Synchronously appends a relationship record (as JSON) into the 'relationships' column in NPCStats.
    Example record: {"relationship_label": "thrall", "with_npc_id": 1234}
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT relationships FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
        (user_id, conversation_id, npc_id)
    )
    row = cur.fetchone()
    if row:
        try:
            rel_data = row[0] or "[]"
            if isinstance(rel_data, str):
                rel_list = json.loads(rel_data)
            else:
                rel_list = rel_data
        except Exception as e:
            logging.warning(f"[append_relationship_to_npc] JSON parse error: {e}")
            rel_list = []
    if not row:
        logging.warning(f"[append_relationship_to_npc] NPC {npc_id} not found.")
        conn.close()
        return
    new_record = {
        "relationship_label": rel_label,
        "entity_type": target_entity_type,
        "entity_id": target_entity_id
    }
    rel_list.append(new_record)
    updated = json.dumps(rel_list)
    cur.execute(
        "UPDATE NPCStats SET relationships = %s WHERE npc_id=%s AND user_id=%s AND conversation_id=%s",
        (updated, npc_id, user_id, conversation_id)
    )
    conn.commit()
    conn.close()
    logging.info(f"[append_relationship_to_npc] Added relationship '{rel_label}' -> {target_entity_id} for NPC {npc_id}.")



def recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id):
    """
    Re-fetch the NPC's archetypes from the DB and re-run combine_archetype_stats to update final stats.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT archetypes FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
        (user_id, conversation_id, npc_id)
    )
    row = cur.fetchone()
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

    cur.execute("""
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
    cur = conn.cursor()
    cur.execute(
        "SELECT archetypes FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s LIMIT 1",
        (user_id, conversation_id, npc_id)
    )
    row = cur.fetchone()
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

    cur.execute("""
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
# 8) assign_random_relationships (RE-ADDED)
###################
async def assign_random_relationships(user_id, conversation_id, new_npc_id, new_npc_name):
    import random

    # Possibly some relationship labels...
    familial = ["mother", "sister", "aunt"]
    non_familial = ["enemy", "friend", "best friend", "lover", "neighbor",
                    "colleague", "classmate", "teammate", "underling", "thrall", "rival"]

    relationships = []

    # 1) Maybe 50% chance to relate with the player
    if random.random() < 0.5:
        rel_type = random.choice(familial) if random.random() < 0.2 else random.choice(non_familial)
        relationships.append({
            "target_entity_type": "player",
            "target_entity_id": user_id,  # store your user_id 
            "relationship_label": rel_type
        })

    # 2) Gather other NPCs
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_id, npc_name, archetype_summary
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
          AND npc_id!=%s
    """, (user_id, conversation_id, new_npc_id))
    rows = cursor.fetchall()
    conn.close()

    # 3) For each other NPC, 30% chance
    for (old_npc_id, old_npc_name, old_arche_summary) in rows:
        if random.random() < 0.3:
            rel_type = (random.choice(familial)
                        if random.random() < 0.2 else
                        random.choice(non_familial))
            relationships.append({
                "target_entity_type": "npc",
                "target_entity_id": old_npc_id,
                "relationship_label": rel_type,
                "target_archetype_summary": old_arche_summary or ""
            })

    # 4) Actually create them
    for rel in relationships:
        # Possibly build some memory text:
        memory_text = get_shared_memory(user_id, conversation_id, rel, new_npc_name)
        record_npc_event(user_id, conversation_id, new_npc_id, memory_text)

        from logic.social_links import create_social_link
        from logic.npc_creation import dynamic_reciprocal_relationship

        # If the target is a player:
        if rel["target_entity_type"] == "player":
            create_social_link(
                user_id, conversation_id,
                entity1_type="npc",  entity1_id=new_npc_id,
                entity2_type="player", entity2_id=rel["target_entity_id"],
                link_type=rel["relationship_label"]
            )
            # Save it to NPCStats relationships JSON
            await asyncio.to_thread(
                append_relationship_to_npc,
                user_id, conversation_id,
                new_npc_id, 
                rel["relationship_label"],
                "player", rel["target_entity_id"]
            )

        else:
            # It's another NPC
            old_npc_id = rel["target_entity_id"]
            # new npc -> old npc
            create_social_link(
                user_id, conversation_id,
                entity1_type="npc", entity1_id=new_npc_id,
                entity2_type="npc", entity2_id=old_npc_id,
                link_type=rel["relationship_label"]
            )
            await asyncio.to_thread(
                append_relationship_to_npc,
                user_id, conversation_id,
                new_npc_id,
                rel["relationship_label"],
                "npc", old_npc_id
            )

            # reciprocal
            rec_type = dynamic_reciprocal_relationship(
                rel["relationship_label"],
                rel.get("target_archetype_summary","")
            )
            create_social_link(
                user_id, conversation_id,
                entity1_type="npc", entity1_id=old_npc_id,
                entity2_type="npc", entity2_id=new_npc_id,
                link_type=rec_type
            )
            await asyncio.to_thread(
                append_relationship_to_npc,
                user_id, conversation_id,
                old_npc_id,
                rec_type,
                "npc", new_npc_id
            )

    logging.info(f"[assign_random_relationships] Done building relationships for NPC {new_npc_id}.")

async def refine_npc_final_data(user_id: int, conversation_id: int, npc_id: int, day_names: list, environment_desc: str):
    """
    1) Fetch the now-updated NPC from DB (with relationships).
    2) GPT to generate:
       - physical_description
       - schedule
       - memory
       - affiliations
       - current_location
    3) Update DB with them.
    """
    # 1) Fetch current NPC record
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
       SELECT npc_name, introduced, sex,
              dominance, cruelty, closeness, trust, respect, intensity,
              archetypes, archetype_summary, archetype_extras_summary,
              likes, dislikes, hobbies, personality_traits,
              age, birthdate, relationships, memory, schedule,
              physical_description
       FROM NPCStats
       WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
       LIMIT 1
    """, (user_id, conversation_id, npc_id))
    row = cur.fetchone()
    conn.close()

    if not row:
        logging.warning(f"[refine_npc_final_data] NPC {npc_id} not found.")
        return

    columns = [
        "npc_name", "introduced", "sex", "dominance", "cruelty", "closeness", "trust", "respect", "intensity",
        "archetypes", "archetype_summary", "archetype_extras_summary",
        "likes", "dislikes", "hobbies", "personality_traits",
        "age", "birthdate", "relationships", "memory", "schedule", "affiliations",
        "physical_description"
    ]
    npc_data = dict(zip(columns, row))

    # Parse JSON fields
    json_fields = ["archetypes", "likes", "dislikes", "hobbies", "personality_traits", "relationships", "affiliations", "memory", "schedule"]
    for fld in json_fields:
        val = npc_data.get(fld)
        if isinstance(val, str):
            try:
                npc_data[fld] = json.loads(val or "[]" if fld != "schedule" else val or "{}")
            except Exception as ex:
                logging.warning(f"[refine_npc_final_data] Failed to parse JSON field '{fld}': {ex}")
                npc_data[fld] = [] if fld != "schedule" else {}

    logging.info(f"[refine_npc_final_data] Fetched NPC {npc_id} data: {npc_data.get('npc_name', 'Unknown')}")

    # 2) Build GPT prompt
    prompt = f"""
We have an NPC in a femdom environment. Current data:
{json.dumps(npc_data, indent=2, default=str)}

Environment description:
{environment_desc}

We want to fill or adapt these fields:
  - physical_description,
  - schedule (using day names => {day_names}),
  - affiliations (e.g. clubs, secret societies, workplaces, or other groups the NPC is a member of),
  - memory (past events),
  - current_location

**SCHEDULE REQUIREMENTS**:
1. Use EXACTLY these day names in this order: {', '.join(day_names)}.
2. For each day, provide "morning", "afternoon", "evening", and "night".
3. Avoid giving the same location/activity in all four time-slots.
4. Reflect the NPC’s likes, dislikes, hobbies, archetypes, and relationships.
5. Keep it realistic or fitting to the setting.

**MEMORY REQUIREMENTS**:
- For each relationship, provide at least three distinct memory entries referencing the relationships in 'npc_data["relationships"]' if relevant.
- Each memory is either: a short reference to a past event that reveals how these relationships formed, any relevant backstory, or a major/impactful event involving both parties.
- Memories can be shared between 3 or more characters ONLY if they all have a relationship, and they must all have the memory.

**PHYSICAL DESCRIPTION**:
- Over-the-top curvaceous (inspired by M-size games). Almost comical level of boobs/ass, but keep it in-lore.

Return only JSON with keys:
  "physical_description",
  "schedule",
  "memory",
  "affiliations",
  "current_location"

No extra text or function calls.
"""
    logging.info(f"[refine_npc_final_data] GPT Prompt for NPC {npc_id}: {prompt}")

    # 3) Call GPT and process response
    raw_gpt = await asyncio.to_thread(
        get_chatgpt_response,
        conversation_id,
        environment_desc,
        prompt
    )
    logging.info(f"[refine_npc_final_data] Raw GPT response for NPC {npc_id}: {raw_gpt}")

    if raw_gpt.get("type") == "function_call":
        result_dict = raw_gpt.get("function_args", {})
        logging.info(f"[refine_npc_final_data] Function call result for NPC {npc_id}: {result_dict}")
        # Check if the response includes npc_updates and extract from the first update.
        if "npc_updates" in result_dict and isinstance(result_dict["npc_updates"], list) and result_dict["npc_updates"]:
            npc_update = result_dict["npc_updates"][0]
            physical_desc = npc_update.get("physical_description", "")
            schedule = npc_update.get("schedule", {})
            memories = npc_update.get("memory", [])
            affiliations = npc_update.get("affiliations", [])
            current_location = npc_update.get("current_location", "")
        else:
            physical_desc = result_dict.get("physical_description", "")
            schedule = result_dict.get("schedule", {})
            memories = result_dict.get("memory", [])
            affiliations = result_dict.get("affiliations", [])
            current_location = result_dict.get("current_location", "")
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
            logging.info(f"[refine_npc_final_data] Parsed GPT JSON for NPC {npc_id}: {result_dict}")
        except Exception as e:
            logging.warning(f"[refine_npc_final_data] JSON parse error for NPC {npc_id}: {e}. Raw text: {text}")
            result_dict = {}
        physical_desc = result_dict.get("physical_description", "")
        schedule = result_dict.get("schedule", {})
        memories = result_dict.get("memory", [])
        affiliations = result_dict.get("affiliations", [])
        current_location = result_dict.get("current_location", "")

    logging.info(
        f"[refine_npc_final_data] GPT parsed data for NPC {npc_id}: physical_desc: {physical_desc}, "
        f"schedule: {schedule}, memory: {memories}, affiliations: {affiliations}, current_location: {current_location}"
    )


    # 4) Update DB with refined data
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
       UPDATE NPCStats
       SET physical_description=%s,
           schedule=%s,
           memory=%s,
           affiliations=%s,
           current_location=%s
       WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (
       physical_desc,
       json.dumps(schedule),
       json.dumps(memories),
       json.dumps(affiliations),
       current_location,
       user_id, conversation_id, npc_id
    ))
    rows_updated = cur.rowcount
    conn.commit()
    conn.close()
    logging.info(f"[refine_npc_final_data] DB update for NPC {npc_id} completed. Rows updated: {rows_updated}")

    # 5) Propagate shared memories
    npc_name = fetch_npc_name(user_id, conversation_id, npc_id) or "Unknown"
    logging.info(f"[refine_npc_final_data] Propagating memories for NPC {npc_id} ({npc_name}). Memories: {memories}")
    propagate_shared_memories(
        user_id=user_id,
        conversation_id=conversation_id,
        source_npc_id=npc_id,
        source_npc_name=npc_name,
        memories=memories
    )

    logging.info(f"[refine_npc_final_data] Updated NPC {npc_id} => physical_desc, schedule, memory, affiliations, and current_location.")
    return {
        "physical_description": physical_desc,
        "schedule": schedule,
        "memory": memories
    }
async def spawn_single_npc(
    user_id: int,
    conversation_id: int,
    environment_desc: str,
    day_names: list
) -> int:
    """
    1) Create partial NPC (archetypes, stats, likes).
    2) Insert stub in DB => get npc_id.
    3) Assign random relationships => references npc_id.
    4) Final GPT call => produce physical_description, schedule, memory referencing relationships.
    5) Return npc_id
    """
    # IMPORTANT: pass environment_desc to create_npc_partial!
    partial_npc = create_npc_partial(
        user_id=user_id,
        conversation_id=conversation_id,
        sex="female",
        total_archetypes=4,
        environment_desc=environment_desc
    )

    # 2) Insert => npc_id
    npc_id = await insert_npc_stub_into_db(partial_npc, user_id, conversation_id)

    # 3) Assign relationships
    await assign_random_relationships(user_id, conversation_id, npc_id, partial_npc["npc_name"])

    # 4) Final call => physical_description, schedule, memory
    await refine_npc_final_data(user_id, conversation_id, npc_id, day_names, environment_desc)

    logging.info(f"[spawn_single_npc] NPC {partial_npc['npc_name']} (ID={npc_id}) done.")
    return npc_id

###################
# 11) Spawn multiple NPCs
###################

async def spawn_multiple_npcs(
    user_id: int,
    conversation_id: int,
    environment_desc: str,
    day_names: list,
    count=3
):
    """
    Loop spawn_single_npc for 'count' times. 
    Return list of new NPC IDs.
    """
    npc_ids = []
    for i in range(count):
        new_id = await spawn_single_npc(
            user_id,
            conversation_id,
            environment_desc,
            day_names
        )
        npc_ids.append(new_id)
    return npc_ids

async def generate_chase_schedule(
    user_id: int,
    conversation_id: int,
    environment_desc: str,
    day_names: list
) -> dict:
    """
    1) Gather 'Chase' stats from PlayerStats (for flavor).
    2) GPT call to produce Chase's schedule as a dict keyed by each day => subkeys (morning,afternoon, etc.)
    3) Store the schedule in CurrentRoleplay or wherever you want.
    4) Return the schedule.
    """
    import json
    import asyncio
    from logic.gpt_utils import spaced_gpt_call
    from db.connection import get_db_connection

    # Step A: load 'Chase' stats from PlayerStats
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT corruption, confidence, willpower, obedience, dependency, lust,
               mental_resilience, physical_endurance
        FROM PlayerStats
        WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        LIMIT 1
    """, (user_id, conversation_id))
    row = cur.fetchone()
    conn.close()

    if not row:
        logging.warning("[generate_chase_schedule] Player 'Chase' not found in PlayerStats.")
        return {}

    (corrupt, confid, willp, obed, dep, lust, ment_res, phys_end) = row

    # Build a short "Chase partial data" to feed GPT
    chase_data = {
        "player_name": "Chase",
        "corruption": corrupt,
        "confidence": confid,
        "willpower": willp,
        "obedience": obed,
        "dependency": dep,
        "lust": lust,
        "mental_resilience": ment_res,
        "physical_endurance": phys_end
    }

    # Step B: Build the prompt
    # We'll re-use the idea from your old NPC_PROMPT.
    chase_prompt = f"""
We have a player character 'Chase' in this femdom environment.
Environment:
{environment_desc}

Chase partial data:
{json.dumps(chase_data, indent=2)}

The list of days is: {day_names}

Please generate a realistic daily schedule for "Chase" for each of the days listed. Your output must be a valid JSON object with exactly one top-level key, "ChaseSchedule". The value of "ChaseSchedule" must be an object whose keys exactly match the days in the list (e.g., if the list is ["Monday", "Tuesday", ...], then these must be the keys). For each day, the value must be an object with exactly the following keys:
- "Morning"
- "Afternoon"
- "Evening"
- "Night"

Each of these keys should map to a short string describing Chase's activity during that time slot.

Do not include any extra keys, text, or commentary. Do not wrap your output in code fences.
"""

    # Step C: Do the GPT call
    response_dict = await spaced_gpt_call(conversation_id, environment_desc, chase_prompt, delay=1.0)
    if response_dict.get("type") == "function_call":
        # If GPT returned function_call, parse result
        chase_sched_data = response_dict.get("function_args", {})
    else:
        raw_text = response_dict.get("response", "").strip()
        # remove triple-backticks if needed
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            if lines and lines[-1].startswith("```"):
                lines.pop()
            raw_text = "\n".join(lines).strip()

        try:
            chase_sched_data = json.loads(raw_text)
        except Exception as e:
            logging.error("[generate_chase_schedule] parse error: %s", e)
            chase_sched_data = {}

    chase_schedule = chase_sched_data.get("ChaseSchedule", {})
    # If GPT doesn't include 'ChaseSchedule', fallback empty
    if not chase_schedule:
        logging.warning("[generate_chase_schedule] GPT gave no 'ChaseSchedule'.")
        chase_schedule = {}

    # Step D: Optionally store in DB (CurrentRoleplay or PlayerStats).
    # For example, store in CurrentRoleplay as 'ChaseSchedule':
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'ChaseSchedule', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, json.dumps(chase_schedule)))
    conn.commit()
    conn.close()

    logging.info("[generate_chase_schedule] Stored chase schedule => %s", chase_schedule)
    return chase_schedule

# --- NEW: Define relationship groups for propagation ---
RELATIONSHIP_GROUPS = {
    # Family relationships:
    "family": {
        "mother": {
            "related": ["sister", "stepsister", "cousin"],  # Cousins can be related too
            "base": "mother",
            "reciprocal": "child"
        },
        "stepmother": {
            "related": ["sister", "stepsister", "cousin"],
            "base": "stepmother",
            "reciprocal": "child"
        },
        "aunt": {
            "related": ["cousin"],  # Optionally, cousins of an aunt could also be linked
            "base": "aunt",
            "reciprocal": "niece/nephew"
        },
        "cousin": {
            "related": ["cousin"],  # Cousins are symmetric
            "base": "cousin",
            "reciprocal": "cousin"
        }
    },
    # Work/Professional relationships:
    "work": {
        "boss": {
            "related": ["colleague"],
            "base": "boss",
            "reciprocal": "employee"
        }
    },
    # Team/Group relationships:
    "team": {
        "captain": {
            "related": ["teammate"],
            "base": "captain",
            "reciprocal": "team member"
        },
        "classmate": {
            "related": ["classmate"],
            "base": "classmate",
            "reciprocal": "classmate"  # symmetric
        }
    },
    # Neighborhood:
    "neighbors": {
        "neighbor": {
            "related": ["neighbor"],
            "base": "neighbor",
            "reciprocal": "neighbor"
        }
    }
}

# --- NEW: A general relationship propagation function ---
def propagate_relationships(user_id, conversation_id):
    """
    Scan all NPCs for relationships and, based on RELATIONSHIP_GROUPS,
    propagate additional links. For example, if NPC A is a mother to target X,
    and NPC B is a sister (or stepsister) of someone who also relates to X,
    then NPC B should also gain the mother relationship to X,
    and X should gain the reciprocal 'child' link if not already present.
    
    This logic is applied for each defined group (family, work, team, neighbors).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT npc_id, npc_name, relationships
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    rows = cur.fetchall()
    # Build a dictionary of npc data
    npc_dict = {}
    for npc_id, npc_name, rel_str in rows:
        try:
            rels = json.loads(rel_str) if rel_str else []
        except Exception:
            rels = []
        npc_dict[npc_id] = {"npc_name": npc_name, "relationships": rels}
    
    # For each propagation group:
    for group in RELATIONSHIP_GROUPS.values():
        for base_label, settings in group.items():
            # For every NPC that has a base relationship in this group...
            for npc_id, data in npc_dict.items():
                for rel in data["relationships"]:
                    # Compare case-insensitively
                    if rel.get("relationship_label", "").lower() == base_label.lower():
                        target = rel.get("entity_id")
                        # Now, for every other NPC in the same group that has one of the related labels
                        for other_id, other_data in npc_dict.items():
                            if other_id == npc_id:
                                continue
                            for other_rel in other_data["relationships"]:
                                if other_rel.get("relationship_label", "").lower() in [r.lower() for r in settings["related"]]:
                                    if other_rel.get("entity_id") == target:
                                        # If this other NPC does not already have the base relationship,
                                        # add it.
                                        if not any(r.get("relationship_label", "").lower() == base_label.lower() and r.get("entity_id") == target
                                                   for r in other_data["relationships"]):
                                            other_data["relationships"].append({
                                                "relationship_label": base_label,
                                                "entity_type": "npc",
                                                "entity_id": target
                                            })
                                            # Also add the reciprocal relationship to the target NPC,
                                            # if target exists in our npc_dict.
                                            if target in npc_dict:
                                                target_rels = npc_dict[target]["relationships"]
                                                if not any(r.get("relationship_label", "").lower() == settings["reciprocal"].lower() and r.get("entity_id") == other_id
                                                           for r in target_rels):
                                                    target_rels.append({
                                                        "relationship_label": settings["reciprocal"],
                                                        "entity_type": "npc",
                                                        "entity_id": other_id
                                                    })
    # Write the updated relationships back to the DB.
    for npc_id, data in npc_dict.items():
        new_rel_str = json.dumps(data["relationships"])
        cur.execute("""
            UPDATE NPCStats
            SET relationships=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (new_rel_str, user_id, conversation_id, npc_id))
    conn.commit()
    conn.close()

def adjust_family_ages(user_id, conversation_id):
    """
    Adjust NPC ages based on familial relationships.
    For example, if an NPC has a "mother" relationship with another,
    ensure that the mother's age is at least a specified number of years greater
    than the child's.
    """
    # Define minimum age differences for roles (all keys in lowercase)
    min_age_diff = {
        "mother": 16,
        "stepmother": 0,
        "aunt": 5,
        "older sister": 1,
        "stepsister": 1,
        "babysitter": 2,
        "teacher": 10,
        "principal": 10,
        "milf": 10,
        "dowager": 15,
        "domestic authority": 5,
        "foreign royalty": 0,
        "cousin": 0  # could be same age or slight difference
    }
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT npc_id, age, relationships
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    rows = cur.fetchall()
    npc_info = {}
    for row in rows:
        npc_id, age, rel_str = row
        try:
            relationships = json.loads(rel_str) if rel_str else []
        except Exception:
            relationships = []
        npc_info[npc_id] = {"age": age, "relationships": relationships}
    
    # For each familial relationship, adjust ages accordingly.
    # For example, if NPC A has a "mother" relationship to NPC B,
    # ensure A.age >= B.age + min_age_diff["mother"]
    for npc_id, info in npc_info.items():
        for rel in info["relationships"]:
            label = rel.get("relationship_label", "").lower()
            target = rel.get("entity_id")
            if target not in npc_info:
                continue
            if label in ["mother", "stepmother", "aunt"]:
                required = min_age_diff.get(label, 0)
                target_age = npc_info[target]["age"]
                if info["age"] < target_age + required:
                    info["age"] = target_age + required
            # Similarly, if the relationship indicates that this NPC is the child
            # (or younger sibling), adjust so they are at least a few years younger.
            if label in ["child", "younger sibling"]:
                # For simplicity, require at least 1 year difference
                if info["age"] >= npc_info[target]["age"]:
                    info["age"] = max(18, npc_info[target]["age"] - 1)
    
    # Write updated ages back to the database.
    for npc_id, info in npc_info.items():
        cur.execute("""
            UPDATE NPCStats
            SET age=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (info["age"], user_id, conversation_id, npc_id))
    conn.commit()
    conn.close()

# 2) Now do a separate GPT call for Chase
async def init_chase_schedule():
    chase_sched = await generate_chase_schedule(
        user_id=user_id,
        conversation_id=conversation_id,
        environment_desc=combined_env,
        day_names=day_names
    )
    return chase_sched

# And then call it from an async context:
if __name__ == '__main__':
    import asyncio
    chase_schedule = asyncio.run(init_chase_schedule())
    print("Chase Schedule:", chase_schedule)
