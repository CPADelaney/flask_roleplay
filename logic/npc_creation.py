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
                "Generate a creative, extremely unique, and varied feminine name for the NPC. "
                "The name must be unmistakably feminine and should not be a common or overused name. "
                "Do not use any names that are frequently repeated in this game (for example, do not use 'Seraphina', 'Veronica', or similar names). "
                "Instead, invent a name that is entirely original, unexpected, and richly evocative of a fantastical, mythological, or diverse cultural background that makes sense for the setting and the character (Eg., Isis, Artemis, Megan, Thoth, Cassandra, Mizuki, etc.). "
                "Ensure that the name is unlike any generated previously in this playthrough. "
                "Output strictly valid JSON with exactly one key: 'npc_name', whose value is the generated name as a string. "
                "Do not include any extra commentary, formatting, or additional keys."
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
async def assign_random_relationships(user_id, conversation_id, new_npc_id, new_npc_name, npc_archetypes=None):
    logging.info(f"[assign_random_relationships] Assigning relationships for NPC {new_npc_id} ({new_npc_name})")
    import random

    relationships = []

    # Automatically add familial relationships based on NPC archetypes
    familial_set = {"mother", "stepmother", "aunt", "older sister", "stepsister", "babysitter"}
    if npc_archetypes:
        for arc in npc_archetypes:
            arc_name = arc.get("name", "").strip().lower()
            if arc_name in familial_set:
                # Automatically add a relationship from this NPC to the player
                relationships.append({
                    "target_entity_type": "player",
                    "target_entity_id": user_id,  # player ID
                    "relationship_label": arc_name  # e.g., "aunt"
                })
                logging.info(f"[assign_random_relationships] Automatically added familial relationship '{arc_name}' for NPC {new_npc_id} to player.")

    # Existing random relationship assignment:
    familial = ["mother", "sister", "aunt"]
    non_familial = ["enemy", "friend", "best friend", "lover", "neighbor",
                    "colleague", "classmate", "teammate", "underling", "thrall", "rival"]

    # 1) Maybe 50% chance to relate with the player (in addition to familial)
    if random.random() < 0.5:
        rel_type = random.choice(familial) if random.random() < 0.2 else random.choice(non_familial)
        relationships.append({
            "target_entity_type": "player",
            "target_entity_id": user_id,
            "relationship_label": rel_type
        })

    # 2) Gather other NPCs
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_id, npc_name, archetype_summary
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id!=%s
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

    # 4) Actually create them (existing logic remains unchanged)
    for rel in relationships:
        memory_text = get_shared_memory(user_id, conversation_id, rel, new_npc_name)
        record_npc_event(user_id, conversation_id, new_npc_id, memory_text)

        from logic.social_links import create_social_link
        from logic.npc_creation import dynamic_reciprocal_relationship

        if rel["target_entity_type"] == "player":
            create_social_link(
                user_id, conversation_id,
                entity1_type="npc",  entity1_id=new_npc_id,
                entity2_type="player", entity2_id=rel["target_entity_id"],
                link_type=rel["relationship_label"]
            )
            await asyncio.to_thread(
                append_relationship_to_npc,
                user_id, conversation_id,
                new_npc_id, 
                rel["relationship_label"],
                "player", rel["target_entity_id"]
            )
        else:
            old_npc_id = rel["target_entity_id"]
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
            rec_type = dynamic_reciprocal_relationship(
                rel["relationship_label"],
                rel.get("target_archetype_summary", "")
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

    logging.info(f"[assign_random_relationships] Finished relationships for NPC {new_npc_id}.")

import re

def parse_bullet_schedule_from_narrative(text: str, day_names: list) -> dict:
    """
    1) Finds bullet-list schedule blocks in 'text' that look like:
         - **SOMEDAY**:
           - Morning: ...
           - Afternoon: ...
           - Evening: ...
           - Night: ...
    2) Maps each block (in order of appearance) onto the given day_names array.
    3) If fewer blocks than day_names, fill leftover days with "Free time".
    4) Returns a dict:
       {
         "Commandday": {"morning": "...", "afternoon": "...", ...},
         "Bindday": {...},
         ...
       }
    """

    # Regex for a day heading:   - **SOMEDAY**:
    day_heading_pattern = re.compile(r"^- \*\*(.+?)\*\*:\s*$")
    # Regex for timeslots:      - Morning: some text
    timeslot_pattern = re.compile(r"^- (Morning|Afternoon|Evening|Night):\s*(.*)$", re.IGNORECASE)

    lines = text.splitlines()
    blocks = []
    current_block = None

    for line in lines:
        line = line.strip()

        # 1) Check if this starts a new day block
        day_match = day_heading_pattern.match(line)
        if day_match:
            # If we were building a block, store it
            if current_block is not None:
                blocks.append(current_block)
            # Start a new block
            current_block = {
                "title": day_match.group(1),  # e.g. "Obedience" or "Deference"
                "slots": {}
            }
            continue

        # 2) Timeslot lines inside a block
        if current_block is not None:
            slot_match = timeslot_pattern.match(line)
            if slot_match:
                slot_time = slot_match.group(1).lower()  # morning, afternoon, etc.
                slot_desc = slot_match.group(2).strip()
                current_block["slots"][slot_time] = slot_desc

    # End of file: store the last block if it exists
    if current_block is not None:
        blocks.append(current_block)

    schedule = {}
    # Map discovered blocks to our day_names in order
    for i, block in enumerate(blocks):
        if i >= len(day_names):
            break
        day_label = day_names[i]
        sub_dict = {}
        for t in ["morning", "afternoon", "evening", "night"]:
            sub_dict[t] = block["slots"].get(t, "Free time")
        schedule[day_label] = sub_dict

    # If GPT gave fewer blocks, fill the rest with “Free time”
    if len(blocks) < len(day_names):
        for i in range(len(blocks), len(day_names)):
            day_label = day_names[i]
            schedule[day_label] = {
                "morning": "Free time",
                "afternoon": "Free time",
                "evening": "Free time",
                "night": "Free time",
            }

    return schedule

import re

def parse_bullet_physical_description(text: str) -> str:
    """
    Extracts physical description from various text formats including:
    - Bullet points with labels like "Physical Description:" 
    - JSON-like sections with physical description labels
    - Multiple lines or paragraphs of description
    Returns the concatenated description or empty string if not found.
    """
    # Multiple patterns to match different formats
    bullet_pattern = re.compile(r"(?:^|\n)[-*•]?\s*(Physical\s*Description|PhysicalDesc|Body|Appearance|Looks|Features)\s*:?\s*(.+?)(?=\n[-*•]|\n\n|$)", re.IGNORECASE | re.DOTALL)
    json_pattern = re.compile(r'"(physical_?description|appearance|physique)":\s*"(.+?)"(?=,|\})', re.IGNORECASE | re.DOTALL)
    
    # Try bullet pattern first (more common in GPT outputs)
    matches = bullet_pattern.findall(text)
    if matches:
        # Combine all matches into paragraphs
        descriptions = [match[1].strip() for match in matches]
        return "\n\n".join(descriptions)
    
    # Try JSON-like pattern
    matches = json_pattern.findall(text)
    if matches:
        descriptions = [match[1].strip() for match in matches]
        return "\n\n".join(descriptions)
    
    # Look for standalone paragraphs that might be descriptions
    description_keywords = ["tall", "height", "build", "hair", "eyes", "face", "skin", "wearing", "dressed", "outfit", "figure", "physique"]
    paragraphs = re.split(r'\n\s*\n', text)
    
    for para in paragraphs:
        # If paragraph contains multiple description keywords and no obvious headers, it might be a description
        if sum(1 for kw in description_keywords if kw.lower() in para.lower()) >= 3 and ":" not in para[:20]:
            return para.strip()
    
    return ""
    
def parse_bullet_current_location(text: str) -> str:
    """
    If GPT lumps current location in lines like:
      - Location: The Obsidian Tower
    or
      - CurrentLocation: The Gilded Hall
    Return the portion after the colon.
    """
    pattern = re.compile(r"^- (Location|Current\s*Location)\s*:\s*(.+)$", re.IGNORECASE)
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        match = pattern.match(line)
        if match:
            return match.group(2).strip()
    return ""

def parse_bullet_memory_entries(text: str) -> list:
    """
    If GPT lumps memory lines like:
      - Memory: She taught me the labyrinth trick...
      - Memory: We shared a midnight feast...
    Return them as a list of strings.
    """
    pattern = re.compile(r"^- Memory\s*:\s*(.+)$", re.IGNORECASE)
    lines = text.splitlines()
    found = []
    for line in lines:
        line = line.strip()
        match = pattern.match(line)
        if match:
            found.append(match.group(1).strip())
    return found



def parse_bullet_relationships_from_narrative(text: str) -> list:
    """
    A quick example that finds lines like:
        - Relationship with (ID 123): Rival
        - Relationship with (ID 999): Friend
      Then returns an array of relationship objects:
        [
          {
            "entity_id": 123,
            "entity_type": "npc",   # or "player" if you want to detect that
            "relationship_label": "Rival"
          },
          ...
        ]
    You can adapt the regex to match your exact bullet style.
    """

    # e.g.:  - Relationship with (ID 42): Rival
    rel_pattern = re.compile(r"^- Relationship with \(ID\s+(\d+)\):\s*(.+)$", re.IGNORECASE)
    relationships = []

    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        match = rel_pattern.match(line)
        if match:
            # The ID number
            ent_id_str = match.group(1)
            relationship_label = match.group(2).strip()
            try:
                ent_id = int(ent_id_str)
            except ValueError:
                ent_id = 0  # fallback if can't parse

            # For simplicity, assume entity_type = "npc"
            # You can parse the line if you want to detect "player"
            relationships.append({
                "entity_id": ent_id,
                "entity_type": "npc",
                "relationship_label": relationship_label
            })

    return relationships


def remap_day_blocks(schedule_data: dict, day_names: list) -> dict:
    """
    If GPT returned custom day keys, we forcibly map them onto your official 'day_names'
    in order. If GPT returned fewer blocks, fill leftover with 'Free time';
    if more, ignore extras.
    """
    items = list(schedule_data.items())
    final_schedule = {}
    for i, (gpt_day, slots) in enumerate(items):
        if i >= len(day_names):
            break
        official_day = day_names[i]
        final_schedule[official_day] = {
            "morning":   slots.get("morning", "Free time"),
            "afternoon": slots.get("afternoon", "Free time"),
            "evening":   slots.get("evening", "Free time"),
            "night":     slots.get("night", "Free time")
        }
    # Fill leftover if GPT had fewer blocks
    used = len(final_schedule)
    while used < len(day_names):
        final_schedule[day_names[used]] = {
            "morning": "Free time",
            "afternoon": "Free time",
            "evening": "Free time",
            "night": "Free time"
        }
        used += 1
    return final_schedule

async def refine_npc_final_data(
    user_id: int,
    conversation_id: int,
    npc_id: int,
    day_names: list,
    environment_desc: str,
    max_retries=3  # Increased retries
):
    """
    1) Fetch NPC from DB
    2) GPT prompt: let it go rogue. We'll parse and map to fields:
       physical_description, schedule, memory, affiliations, current_location, relationships
    3) fallback parse from bullet lines if missing
    4) forcibly re-map schedule day blocks
    5) store in DB
    6) propagate memories
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
       SELECT npc_name, introduced, sex,
              dominance, cruelty, closeness, trust, respect, intensity,
              archetypes, archetype_summary, archetype_extras_summary,
              likes, dislikes, hobbies, personality_traits,
              age, birthdate, relationships, memory, schedule,
              affiliations, physical_description
       FROM NPCStats
       WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
       LIMIT 1
    """, (user_id, conversation_id, npc_id))
    row = cur.fetchone()
    if not row:
        logging.warning(f"[refine_npc_final_data] NPC {npc_id} not found.")
        conn.close()
        return

    columns = [
        "npc_name","introduced","sex","dominance","cruelty","closeness","trust","respect","intensity",
        "archetypes","archetype_summary","archetype_extras_summary",
        "likes","dislikes","hobbies","personality_traits",
        "age","birthdate","relationships","memory","schedule",
        "affiliations","physical_description"
    ]
    npc_data = dict(zip(columns, row))
    conn.close()

    # parse JSON
    json_fields = ["archetypes","likes","dislikes","hobbies","personality_traits","relationships","affiliations","memory","schedule"]
    for f in json_fields:
        val = npc_data.get(f)
        if isinstance(val, str):
            try:
                if f == "schedule":
                    npc_data[f] = json.loads(val or "{}")
                else:
                    npc_data[f] = json.loads(val or "[]")
            except json.JSONDecodeError:
                npc_data[f] = {} if f == "schedule" else []

    logging.info(f"[refine_npc_final_data] Loaded NPC {npc_data.get('npc_name')} => ID={npc_id}")

    # We'll do up to 'max_retries' attempts
    attempt = 0
    physical_desc = npc_data.get('physical_description', '')
    schedule_obj = npc_data.get('schedule', {})
    memories = npc_data.get('memory', [])
    affiliations = npc_data.get('affiliations', [])
    current_loc = ""
    relationships = npc_data.get('relationships', [])

    # Construct day template for prompt
    days_template = ""
    for day in day_names:
        days_template += f'    "{day}": {{\n'
        days_template += '      "morning": "Activity description",\n'
        days_template += '      "afternoon": "Activity description",\n'
        days_template += '      "evening": "Activity description",\n'
        days_template += '      "night": "Activity description"\n'
        days_template += '    },\n'
    days_template = days_template.rstrip(",\n")  # Remove trailing comma

    while attempt < max_retries:
        attempt += 1

        # Build the GPT prompt with cleaner JSON example structure
        system_prompt = f"""
You have an NPC in a femdom environment. Current data:
{json.dumps(npc_data, indent=2)}

Environment:
{environment_desc}

Return a JSON object with EXACTLY these keys:
{{
  "physical_description": "Detailed physical appearance, at least 2 paragraphs",
  "current_location": "Where the character is currently located",
  "schedule": {{
{days_template}
  }},
  "memory": [
    "First memory about a relationship from the point of view of the NPC",
    "Second memory about a relationship from the point of view of the NPC",
    "Third memory about a relationship from the point of view of the NPC"
  ],
  "affiliations": [
    "Group or faction affiliation 1",
    "Group or faction affiliation 2",
    "Group or faction affiliation 3",
    "Group or faction affiliation 4",
    "Group or faction affiliation 5",
  ],
  "relationships": [
    {{
      "entity_id": 123,
      "entity_type": "npc",
      "relationship_label": "The relationship description"
    }}
  ]
}}

Return ONLY valid JSON with no explanation, comments, or code blocks.
"""

        # Call GPT
        raw_gpt = await asyncio.to_thread(
            get_chatgpt_response,
            conversation_id,
            environment_desc,
            system_prompt
        )
        logging.info(f"[refine_npc_final_data] Attempt={attempt}, got response type={raw_gpt.get('type')}")

        # full text from GPT - extract it regardless of response type
        text_response = raw_gpt.get("response", "")

        # Try multiple extraction approaches
        extracted_data = {}
        
        # 1. First try function call args if present
        if raw_gpt.get("type") == "function_call":
            try:
                fn_args = raw_gpt.get("function_args", {})
                extracted_data = fn_args
                logging.info(f"[refine_npc_final_data] Extracted from function_args: {list(fn_args.keys())}")
            except Exception as e:
                logging.warning(f"[refine_npc_final_data] Error parsing function args: {e}")
                
        # 2. Try to extract JSON from the text response
        if not extracted_data:
            try:
                # Look for JSON-like structure with regex to handle cases where text might surround JSON
                import re
                json_match = re.search(r'({[\s\S]*})', text_response)
                if json_match:
                    json_str = json_match.group(1)
                    extracted_data = json.loads(json_str)
                    logging.info(f"[refine_npc_final_data] Extracted JSON via regex: {list(extracted_data.keys())}")
            except Exception as e:
                logging.warning(f"[refine_npc_final_data] Error parsing JSON from text: {e}")
                
        # 3. Try parsing the entire response as JSON
        if not extracted_data:
            try:
                extracted_data = json.loads(text_response)
                logging.info(f"[refine_npc_final_data] Parsed entire response as JSON: {list(extracted_data.keys())}")
            except Exception as e:
                logging.warning(f"[refine_npc_final_data] Error parsing entire response as JSON: {e}")

        # Enhanced synonym mapping with more variations
        synonym_map = {
            # Physical description variations
            "physical_description": ["physical_description", "physicaldescription", "physique", "appearance", "body", "desc", "looks", "features"],
            "schedule": ["schedule", "itinerary", "timetable", "routine", "daily_activities", "activities"],
            "memory": ["memory", "memories", "recollections", "past_events", "history"],
            "affiliations": ["affiliations", "factions", "groups", "alliances", "organizations", "loyalties"],
            "current_location": ["current_location", "location", "currentlocation", "whereabouts", "place", "current_place"],
            "relationships": ["relationships", "relations", "connections", "bonds", "associates"]
        }

        # More robust flattening function
        def flatten_nested_data(data, target_fields=None):
            """Recursively search through nested structures for target fields or their synonyms"""
            if target_fields is None:
                target_fields = list(synonym_map.keys())
                
            results = {field: None for field in target_fields}
            
            def search_item(item, path=""):
                if isinstance(item, dict):
                    for k, v in item.items():
                        # Check if this key matches any of our target fields or synonyms
                        for field, synonyms in synonym_map.items():
                            if any(syn in k.lower().replace("_", "") for syn in [s.lower().replace("_", "") for s in synonyms]):
                                results[field] = v
                                
                        # Continue recursion
                        search_item(v, f"{path}.{k}" if path else k)
                elif isinstance(item, list):
                    for i, v in enumerate(item):
                        search_item(v, f"{path}[{i}]")
            
            search_item(data)
            return results
            
        # Extract fields using our robust flattening
        field_values = flatten_nested_data(extracted_data)
        
        # Update our variables with any found values
        # Enhanced field extraction with better physical description handling
        if field_values["physical_description"] is not None:
            # Handle various formats physical_description might come in
            if isinstance(field_values["physical_description"], str):
                physical_desc = field_values["physical_description"].strip()
            elif isinstance(field_values["physical_description"], list):
                # Join paragraphs if it's a list
                physical_desc = "\n\n".join([str(p) for p in field_values["physical_description"]])
            elif isinstance(field_values["physical_description"], dict):
                # Sometimes may come as nested object with paragraph keys
                physical_desc = "\n\n".join([str(v) for v in field_values["physical_description"].values()])
            else:
                physical_desc = str(field_values["physical_description"])
        
        # Improved fallback extraction for physical description
        if not physical_desc or len(physical_desc.strip()) < 30:  # Require substantial description
            extracted_desc = parse_bullet_physical_description(text_response)
            if extracted_desc and len(extracted_desc) > len(physical_desc):
                physical_desc = extracted_desc
                
            # If still inadequate, try to find any substantial text block that might be a description
            if not physical_desc or len(physical_desc.strip()) < 30:
                # Look for the longest paragraph that might be a description
                paragraphs = re.split(r'\n\s*\n', text_response)
                desc_candidates = []
                
                for para in paragraphs:
                    para = para.strip()
                    # Skip obvious non-descriptions (JSON blocks, schedule entries, etc.)
                    if (para.startswith('{') or para.startswith('[') or 
                        re.match(r'^[-*•]', para) or 
                        len(para) < 50 or
                        any(label in para.lower() for label in ['schedule:', 'memory:', 'relationship:'])):
                        continue
                    
                    desc_candidates.append(para)
                
                # Take the longest paragraph as likely description if substantial
                if desc_candidates:
                    longest = max(desc_candidates, key=len)
                    if len(longest) > 100:  # Only use if reasonably long
                        physical_desc = longest
        
        # Final validation before database update
        if not physical_desc or len(physical_desc.strip()) < 30:
            # Create a reasonable default description based on other NPC attributes
            sex = npc_data.get('sex', '').lower() or 'person'
            age = npc_data.get('age', 'adult')
            traits = npc_data.get('personality_traits', ['mysterious'])
            trait_text = traits[0] if isinstance(traits, list) and traits else 'distinctive'
            
            physical_desc = (f"A {age} {sex} with a {trait_text} demeanor. "
                            f"Their appearance reflects their personality, showing subtle signs of "
                            f"their {npc_data.get('dominance', 'balanced')} nature and "
                            f"{npc_data.get('cruelty', 'neutral')} tendencies.")
        
        if field_values["schedule"] and isinstance(field_values["schedule"], dict):
            schedule_obj = field_values["schedule"]
            
        if field_values["memory"] and isinstance(field_values["memory"], list):
            memories = field_values["memory"]
            
        if field_values["affiliations"] and isinstance(field_values["affiliations"], list):
            affiliations = field_values["affiliations"]
            
        if field_values["current_location"] and isinstance(field_values["current_location"], str):
            current_loc = field_values["current_location"]
            
        if field_values["relationships"] and isinstance(field_values["relationships"], list):
            relationships = field_values["relationships"]

        # Fallback to bullet parsing if we're still missing fields
        if not physical_desc:
            physical_desc = parse_bullet_physical_description(text_response)
            
        if not current_loc:
            current_loc = parse_bullet_current_location(text_response)
            
        # For the schedule specifically, implement a fallback parser 
        # (since parse_bullet_schedule_from_narrative was referenced but not defined)
        if not schedule_obj or not isinstance(schedule_obj, dict) or not schedule_obj:
            # Define a simple fallback schedule parser
            def parse_bullet_schedule_from_narrative(text, day_names):
                """Extract schedule information from bullet points"""
                result = {}
                
                # Look for patterns like: "- Monday: Morning - Training, Afternoon - Meetings"
                # or "- Monday Schedule: Morning: Training, Afternoon: Meetings"
                for day in day_names:
                    day_pattern = re.compile(rf"- {day}(?:\s*Schedule)?\s*:(.+?)(?=- \w+:|$)", re.IGNORECASE | re.DOTALL)
                    day_match = day_pattern.search(text)
                    
                    if day_match:
                        day_text = day_match.group(1).strip()
                        times = {}
                        
                        # Extract time blocks
                        for block in ["morning", "afternoon", "evening", "night"]:
                            block_pattern = re.compile(rf"{block}\s*[:-]\s*([^,;]+)", re.IGNORECASE)
                            block_match = block_pattern.search(day_text)
                            
                            if block_match:
                                times[block.lower()] = block_match.group(1).strip()
                            else:
                                times[block.lower()] = "Free time"
                                
                        result[day] = times
                
                return result
            
            parsed_schedule = parse_bullet_schedule_from_narrative(text_response, day_names)
            if parsed_schedule:
                schedule_obj = parsed_schedule
                
        if not memories:
            memories = parse_bullet_memory_entries(text_response)
            
        if not relationships:
            relationships = parse_bullet_relationships_from_narrative(text_response)

        # Ensure schedule is properly formatted
        schedule_obj = remap_day_blocks(schedule_obj, day_names)
        
        # Check if we have enough data to continue
        if physical_desc and (current_loc or attempt >= max_retries):
            break
        
        logging.warning(f"[refine_npc_final_data] Missing critical fields after attempt {attempt}, retrying...")

    # Use defaults for any remaining empty fields
    if not current_loc:
        # Try to infer from schedule or default to generic location
        schedule_values = []
        for day_data in schedule_obj.values():
            schedule_values.extend(day_data.values())
        
        location_keywords = ['in', 'at', 'near', 'visiting', 'inside', 'outside']
        for activity in schedule_values:
            for keyword in location_keywords:
                if f" {keyword} " in f" {activity} ":
                    parts = activity.split(f" {keyword} ")
                    if len(parts) > 1:
                        current_loc = parts[1].split(".")[0].strip()
                        break
            if current_loc:
                break
                
        if not current_loc:
            current_loc = "Unknown location within the environment"

    # Update the database
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
           UPDATE NPCStats
           SET physical_description = %s,
               schedule = %s,
               memory = %s,
               affiliations = %s,
               current_location = %s,
               relationships = %s
           WHERE user_id = %s AND conversation_id = %s AND npc_id = %s
        """, (
            physical_desc.strip(),
            json.dumps(schedule_obj),
            json.dumps(memories),
            json.dumps(affiliations),
            current_loc.strip(),
            json.dumps(relationships),
            user_id, conversation_id, npc_id
        ))
        conn.commit()
        
        logging.info(f"[refine_npc_final_data] Updated NPC data successfully: PD={len(physical_desc)}, Loc={current_loc}, Schedule days={len(schedule_obj)}")
        
        # propagate memories
        npc_name = fetch_npc_name(user_id, conversation_id, npc_id) or "Unknown"
        propagate_shared_memories(user_id, conversation_id, npc_id, npc_name, memories)
        
    except Exception as e:
        conn.rollback()
        logging.error(f"[refine_npc_final_data] Database update failed: {e}")
    finally:
        cur.close()
        conn.close()

    return {
        "physical_description": physical_desc,
        "schedule": schedule_obj,
        "memory": memories,
        "affiliations": affiliations,
        "relationships": relationships,
        "current_location": current_loc
    }

async def spawn_single_npc(user_id: int, conversation_id: int, environment_desc: str, day_names: list) -> int:
    logging.info("[spawn_single_npc] Starting spawn for a new NPC.")
    partial_npc = create_npc_partial(user_id, conversation_id, sex="female", total_archetypes=4, environment_desc=environment_desc)
    logging.info(f"[spawn_single_npc] Partial NPC created: {partial_npc}")
    npc_id = await insert_npc_stub_into_db(partial_npc, user_id, conversation_id)
    logging.info(f"[spawn_single_npc] NPC stub inserted with ID: {npc_id}")
    # Pass partial_npc["archetypes"] to assign_random_relationships
    await assign_random_relationships(user_id, conversation_id, npc_id, partial_npc["npc_name"], partial_npc.get("archetypes", []))
    logging.info(f"[spawn_single_npc] Relationships assigned for NPC ID: {npc_id}")
    await refine_npc_final_data(user_id, conversation_id, npc_id, day_names, environment_desc)
    logging.info(f"[spawn_single_npc] Final refinement completed for NPC ID: {npc_id}")
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
