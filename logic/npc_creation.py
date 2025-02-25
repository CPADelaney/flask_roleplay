# logic/npc_creation.py

import os
import json
import re
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

def enforce_correct_npc_id(gpt_id: int, correct_id: int, context_str: str) -> int:
    """
    If GPT provided 'gpt_id' is not the same as the 'correct_id',
    log a warning and override with correct_id.
    'context_str' is e.g. 'npc_updates' or 'relationship_updates' so we know where it happened.
    """
    if gpt_id is not None and gpt_id != correct_id:
        logging.warning(
            f"[{context_str}] GPT provided npc_id={gpt_id}, but we are using npc_id={correct_id} => overriding."
        )
        return correct_id
    return correct_id  # or gpt_id if it matches

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
    """
    Insert the partial_npc data into NPCStats, returning the actual npc_id from the DB.
    """
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
          age, birthdate,
          relationships, memory, schedule,
          physical_description
        )
        VALUES (%s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
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
            partial_npc.get("birthdate", ""),
        )
    )
    row = cur.fetchone()
    npc_id = row[0]
    conn.commit()
    conn.close()

    logging.info(f"[insert_npc_stub_into_db] Inserted NPC => assigned npc_id={npc_id}")
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
    We'll store only 'name' in the DB, ignoring GPT-provided numeric ID if any.
    """
    # If GPT gave us 'new_arc' that includes "npc_id" we might do:

    # new_arc["npc_id"] = enforce_correct_npc_id(
    #    new_arc.get("npc_id"), npc_id, "add_archetype_to_npc"
    # )

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT archetypes FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s LIMIT 1",
        (user_id, conversation_id, npc_id)
    )
    row = cur.fetchone()
    if not row:
        logging.warning(f"[add_archetype_to_npc] No NPCStats found for npc_id={npc_id}.")
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

    # synergy
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

    # recalc final stats
    recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id)
    logging.info(f"[add_archetype_to_npc] added '{new_arc['name']}' to npc_id={npc_id}.")

###################
# 8) assign_random_relationships (RE-ADDED)
###################
async def assign_random_relationships(user_id, conversation_id, new_npc_id, new_npc_name, npc_archetypes=None):
    logging.info(f"[assign_random_relationships] Assigning relationships for NPC {new_npc_id} ({new_npc_name})")
    import random

    relationships = []

    # Define explicit mapping for archetypes to relationship labels.
    # This mapping covers both familial and non-familial roles.
    explicit_role_map = {
        "mother": "mother",
        "stepmother": "stepmother",
        "aunt": "aunt",
        "older sister": "older sister",
        "stepsister": "stepsister",
        "babysitter": "babysitter",
        "friend from online interactions": "online friend",
        "neighbor": "neighbor",
        "rival": "rival",
        "classmate": "classmate",
        "lover": "lover",
        "colleague": "colleague",
        "teammate": "teammate",
        "boss/supervisor": "boss/supervisor",
        "teacher/principal": "teacher/principal",
        "landlord": "landlord",
        "roommate/housemate": "roommate",
        "ex-girlfriend/ex-wife": "ex-partner",
        "therapist": "therapist",
        "domestic authority": "head of household",
        "the one who got away": "the one who got away",
        "childhood friend": "childhood friend",
        "friend's wife": "friend",
        "friend's girlfriend": "friend",
        "best friend's sister": "friend's sister"
    }
    
    # First, add relationships based on explicit archetype mapping.
    if npc_archetypes:
        for arc in npc_archetypes:
            arc_name = arc.get("name", "").strip().lower()
            if arc_name in explicit_role_map:
                rel_label = explicit_role_map[arc_name]
                # Add relationship from NPC to player using the explicit role.
                relationships.append({
                    "target_entity_type": "player",
                    "target_entity_id": user_id,  # player ID
                    "relationship_label": rel_label
                })
                logging.info(f"[assign_random_relationships] Added explicit relationship '{rel_label}' for NPC {new_npc_id} to player.")
    
    # Next, determine which explicit roles (if any) were already added.
    explicit_roles_added = {rel["relationship_label"] for rel in relationships}
    
    # Define default lists for random selection.
    default_familial = ["mother", "sister", "aunt"]
    default_non_familial = ["enemy", "friend", "best friend", "lover", "neighbor",
                              "colleague", "classmate", "teammate", "underling", "rival", "ex-girlfriend", "ex-wife", "boss", "roommate", "childhood friend"]
    
    # If no explicit familial role was added, consider assigning a random non-familial relationship with the player.
    if not (explicit_roles_added & set(default_familial)):
        if random.random() < 0.5:
            rel_type = random.choice(default_non_familial)
            relationships.append({
                "target_entity_type": "player",
                "target_entity_id": user_id,
                "relationship_label": rel_type
            })
            logging.info(f"[assign_random_relationships] Randomly added non-familial relationship '{rel_type}' for NPC {new_npc_id} to player.")
    
    # Now add relationships with other NPCs.
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_id, npc_name, archetype_summary
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id!=%s
    """, (user_id, conversation_id, new_npc_id))
    rows = cursor.fetchall()
    conn.close()
    
    # For each other NPC, use explicit mapping if possible; otherwise, fall back to random choice.
    for (old_npc_id, old_npc_name, old_arche_summary) in rows:
        if random.random() < 0.3:
            # Check if the current NPC's explicit roles should be used.
            if explicit_roles_added:
                # Prefer one of the explicit roles if available.
                rel_type = random.choice(list(explicit_roles_added))
            else:
                rel_type = random.choice(default_non_familial)
            relationships.append({
                "target_entity_type": "npc",
                "target_entity_id": old_npc_id,
                "relationship_label": rel_type,
                "target_archetype_summary": old_arche_summary or ""
            })
            logging.info(f"[assign_random_relationships] Added relationship '{rel_type}' between NPC {new_npc_id} and NPC {old_npc_id}.")
    
    # Finally, create these relationships in the database and generate associated memories.
    for rel in relationships:
        memory_text = get_shared_memory(user_id, conversation_id, rel, new_npc_name)
        record_npc_event(user_id, conversation_id, new_npc_id, memory_text)

        from logic.social_links import create_social_link
        from logic.npc_creation import dynamic_reciprocal_relationship

        if rel["target_entity_type"] == "player":
            create_social_link(
                user_id, conversation_id,
                entity1_type="npc", entity1_id=new_npc_id,
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


def extract_field_from_function_call(
    raw_gpt: dict,
    field_name: str,
    target_npc_id: int = None
) -> str | dict | list:
    """
    Looks for `field_name` in either npc_creations or npc_updates within a GPT function_call.
    If `target_npc_id` is provided, tries to match that in npc_updates/npc_creations;
    otherwise, if there is only one item, uses that one.
    Returns the extracted field (string/list/dict) or "" / {} / [] if not found.
    """
    fn_args = raw_gpt.get("function_args", {})
    if not isinstance(fn_args, dict):
        return ""

    # 1) Possibly check top-level first
    if field_name in fn_args and fn_args[field_name]:
        return fn_args[field_name]

    # 2) Check npc_creations
    creations = fn_args.get("npc_creations", [])
    if isinstance(creations, list) and creations:
        # If we have a target_npc_id, try to match
        if target_npc_id is not None:
            match = next(
                (c for c in creations if c.get("npc_id") == target_npc_id),
                None
            )
            if match and field_name in match:
                return match[field_name]
        # If no target_npc_id or none matched, but there's only 1 item, fallback
        if len(creations) == 1:
            candidate = creations[0].get(field_name)
            if candidate not in [None, ""]:
                return candidate

    # 3) Check npc_updates
    updates = fn_args.get("npc_updates", [])
    if isinstance(updates, list) and updates:
        if target_npc_id is not None:
            match = next(
                (u for u in updates if u.get("npc_id") == target_npc_id),
                None
            )
            if match and field_name in match:
                return match[field_name]
        # fallback if single update
        if len(updates) == 1:
            candidate = updates[0].get(field_name)
            if candidate not in [None, ""]:
                return candidate

    # Nothing found
    return ""

async def refine_physical_description(
    user_id: int,
    conversation_id: int,
    npc_id: int, 
    npc_data: dict,
    environment_desc: str,
    max_retries: int = 2
) -> str:
    """
    1) GPT request focuses ONLY on 'physical_description'.
    2) We ask for strictly valid JSON with top-level key "physical_description".
    3) The value must be at least 2 paragraphs describing the NPC's (over-the-top) appearance.
    4) Also handle function_call => npc_creations / npc_updates with "physical_description" inside.
    """
    attempt = 0
    final_description = npc_data.get("physical_description", "") or ""
    npc_name = npc_data.get("npc_name", "Unknown NPC")

    logging.info(f"[refine_physical_description] Starting refinement for NPC {npc_name}")

    while attempt < max_retries:
        attempt += 1
        logging.info(f"[refine_physical_description] Attempt {attempt} of {max_retries}")

        # Build the prompt
        system_prompt = f"""
We have an NPC in a femdom environment.
NPC partial data (no schedule or memory, just background):
{json.dumps(npc_data, indent=2)}
Environment:
{environment_desc}

Return strictly valid JSON with exactly 1 key:
  "physical_description"

It must be at least 2 paragraphs describing the NPC's body/appearance, especially over-the-top curves. 
If you cannot comply, return {{}}.
"""
        # 1) Call GPT
        raw_gpt = await asyncio.to_thread(
            get_chatgpt_response,
            conversation_id,
            environment_desc,
            system_prompt
        )
        text_response = raw_gpt.get("response", "")
        logging.debug(f"[refine_physical_description] GPT raw => {raw_gpt}")

        # 2) Attempt to parse from function_call
        desc_candidate = ""
        if raw_gpt.get("type") == "function_call":
            candidate = extract_field_from_function_call(raw_gpt, "physical_description", npc_id)
            if candidate:
                desc_candidate = candidate
            else:
                # Maybe there's a 'narrative' we can parse as a fallback
                narrative_text = raw_gpt.get("function_args", {}).get("narrative", "")
                if narrative_text:
                    desc_candidate = parse_physical_desc_from_text(narrative_text)
        else:
            # 3) If normal JSON, try top-level
            try:
                data = json.loads(text_response)
                if "physical_description" in data:
                    desc_candidate = data["physical_description"]
            except:
                # 4) If we can’t parse entire text, look for curly braces
                match_j = re.search(r'(\{[\s\S]*\})', text_response)
                if match_j:
                    jstr = match_j.group(1)
                    try:
                        data = json.loads(jstr)
                        if "physical_description" in data:
                            desc_candidate = data["physical_description"]
                    except:
                        pass

        # 5) If still empty, fallback to bullet parse in text_response
        if not desc_candidate:
            desc_candidate = parse_physical_desc_from_text(text_response)

        desc_candidate = (desc_candidate or "").strip()
        logging.info(f"[refine_physical_description] Candidate desc length={len(desc_candidate)}")

        # Validate length
        if len(desc_candidate) >= 30:
            final_description = desc_candidate
            logging.info("[refine_physical_description] Successfully got valid description.")
            break
        else:
            logging.warning("[refine_physical_description] insufficient => retrying.")

    # 6) If still missing, fallback
    if not final_description or len(final_description) < 30:
        sex = npc_data.get("sex","female")
        final_description = f"A generic {sex} NPC with no distinguishing features."

    logging.info(f"[refine_physical_description] Final desc length={len(final_description)}")
    return final_description


def parse_physical_desc_from_text(text: str) -> str:
    """
    Minimal bullet/paragraph fallback if GPT didn't give direct JSON.
    """
    # A simple example: look for lines with "Physical Description:"
    bullet_pattern = re.compile(r"(?i)(?:^|\n)-?\s*Physical\s*Description\s*:\s*(.+?)(?:\n|$)")
    match = bullet_pattern.search(text)
    if match:
        return match.group(1).strip()

    # fallback paragraph guess
    paragraphs = re.split(r"\n\s*\n", text)
    best = ""
    for para in paragraphs:
        if len(para) > len(best) and "hair" in para.lower() and "eyes" in para.lower():
            best = para
    return best.strip()



async def refine_schedule(
    user_id: int,
    conversation_id: int,
    npc_id: int, 
    npc_data: dict,
    environment_desc: str,
    day_names: list,
    max_retries: int = 2
) -> dict:
    """
    1) GPT request focusing ONLY on "schedule".
    2) We pass official day_names, and require JSON with key "schedule" => object.
    3) Each day must have morning/afternoon/evening/night short strings.
    """   
    attempt = 0
    final_schedule = {}

    # If NPC already has a partial schedule, we keep it if GPT fails
    existing_schedule = npc_data.get("schedule", {})

    while attempt < max_retries:
        attempt += 1

        # Build example structure
        days_template = ""
        for day in day_names:
            days_template += f'    "{day}": {{\n'
            days_template += '      "morning": "Activity description",\n'
            days_template += '      "afternoon": "Activity description",\n'
            days_template += '      "evening": "Activity description",\n'
            days_template += '      "night": "Activity description"\n'
            days_template += '    },\n'
        days_template = days_template.rstrip(",\n")  # Remove trailing comma

        system_prompt = f"""
NPC Info:
{json.dumps(npc_data, indent=2)}
Setting:
{environment_desc}

For this NPC, create a detailed daily schedule that follows these requirements:
1. Use EXACTLY these days in order: {', '.join(day_names)}
2. Each day MUST include all four time periods: morning, afternoon, evening, and night
3. Activities should vary throughout the day - avoid repetition in time slots
4. Schedule must reflect the NPC's:
   - Personal interests and hobbies
   - Social relationships
   - Role and archetype
   - Likes and dislikes
5. All activities must be realistic for the setting and maintain appropriate themes

Return ONLY valid JSON matching this structure:
{{
  "schedule": {{
{days_template}
  }}
}}

Return {{}} if these requirements cannot be met. No additional text or formatting.
"""

        raw_gpt = await asyncio.to_thread(
            get_chatgpt_response,
            conversation_id,
            environment_desc,
            system_prompt
        )
        text_response = raw_gpt.get("response", "")

        # 1) Function call check
        new_sched = {}
        if raw_gpt.get("type") == "function_call":
            candidate = extract_field_from_function_call(raw_gpt, "schedule", npc_id)
            if candidate and isinstance(candidate, dict):
                new_sched = candidate
        else:
            # 2) Parse top-level
            try:
                data = json.loads(text_response)
                if "schedule" in data and isinstance(data["schedule"], dict):
                    new_sched = data["schedule"]
            except:
                match_j = re.search(r'(\{[\s\S]*\})', text_response)
                if match_j:
                    jstr = match_j.group(1)
                    try:
                        data = json.loads(jstr)
                        if "schedule" in data and isinstance(data["schedule"], dict):
                            new_sched = data["schedule"]
                    except:
                        pass

        # 3) Validate new_sched structure (day_names, each day => 4 timeslots)
        if new_sched and all(day in new_sched for day in day_names):
            final_schedule = new_sched
            break
        else:
            logging.warning("[refine_schedule] attempt=%d, invalid => retry.", attempt)

    # fallback
    if not final_schedule:
        logging.warning("[refine_schedule] using fallback => free time")
        final_schedule = {
            d: {
                "Morning": "Free time",
                "Afternoon": "Free time",
                "Evening": "Free time",
                "Night": "Free time"
            } for d in day_names
        }

    return final_schedule


async def refine_memories(
    user_id: int,
    conversation_id: int,
    npc_id: int, 
    npc_data: dict,
    environment_desc: str,
    max_retries: int = 2
) -> list:
    """
    Phase 2: Refine existing memories to ensure they are consistent with the NPC's archetypes,
    relationships, and the environment. This phase takes the initial memories (generated in Phase 1)
    and enriches them with additional details and internal consistency.
    
    Returns a JSON array under the "memory" key with refined memory strings.
    """
    import json, re, asyncio, logging

    logger = logging.getLogger(f"refine_memories_{npc_id}")
    logger.setLevel(logging.DEBUG)
    
    # Grab the initial memories (from get_shared_memory, already stored in npc_data)
    existing_memories = npc_data.get("memory", [])
    logger.debug(f"Initial memories: {json.dumps(existing_memories, indent=2)}")
    
    # Build additional context from NPC data:
    archetypes_list = [arc.get("name", "").strip() for arc in npc_data.get("archetypes", [])]
    archetypes_str = ", ".join(archetypes_list)
    personality = json.dumps(npc_data.get("personality_traits", []))
    likes = json.dumps(npc_data.get("likes", []))
    dislikes = json.dumps(npc_data.get("dislikes", []))
    hobbies = json.dumps(npc_data.get("hobbies", []))
    schedule = json.dumps(npc_data.get("schedule", {}), indent=2)
    
    extra_context = (
        f"NPC Archetypes: {archetypes_str}\n"
        f"Personality Traits: {personality}\n"
        f"Likes: {likes}\n"
        f"Dislikes: {dislikes}\n"
        f"Hobbies: {hobbies}\n"
        f"Schedule: {schedule}\n"
    )
    
    # (Optional) Build a summary of current relationships.
    relationships = npc_data.get("relationships", [])
    rel_info = "\n".join(
        f"- ID={r.get('entity_id')}, Type={r.get('entity_type')}, Label='{r.get('relationship_label')}'"
        for r in relationships
    )
    
    # Define needed_count based on unique relationship targets:
    unique_targets = set(r.get("entity_id") for r in relationships if r.get("entity_id") is not None)
    needed_count = 3 * len(unique_targets)
    logger.info(f"Need {needed_count} memories (3 per unique target, {len(unique_targets)} unique targets)")
    
    # Construct the system prompt
    system_prompt = f"""
We have generated the following initial memories for NPC {npc_data.get("npc_name", "Unknown NPC")}:
{json.dumps(existing_memories, indent=2)}

Environment:
{environment_desc}

Additional Context:
{extra_context}

Relationships:
{rel_info}

Refine the above memories to ensure they are fully consistent with the NPC's archetypes and relationship dynamics. 
In your refinement, please:
- Ensure that all of the NPC's archetypal traits are naturally integrated.
- Adjust any details that seem inconsistent or strange.
- Add any extra sensory or descriptive details that enhance the internal consistency of the narrative.
- Ensure at least one memory has conflict between the characters.

Return strictly valid JSON with exactly one key, "memory", whose value is an array of refined memory strings. Follow exactly this format:

{{
  "memory": [
    "<refined memory 1>",
    "<refined memory 2>",
    "<refined memory 3>"
  ]
}}

No extra commentary, code fences, or additional keys. If you cannot comply, return {{}}
"""
    logger.debug(f"System prompt length: {len(system_prompt)} characters")
    
    attempt = 0
    final_mem = existing_memories[:]  # fallback if needed
    while attempt < max_retries:
        attempt += 1
        logger.info(f"Starting attempt {attempt}/{max_retries}")

        try:
            logger.info("Calling GPT with system prompt")
            raw_gpt = await asyncio.to_thread(
                get_chatgpt_response,
                conversation_id,
                environment_desc,
                system_prompt
            )
            logger.debug(f"GPT response type: {raw_gpt.get('type', 'unknown')}")
            text_response = raw_gpt.get("response", "")
            logger.debug(f"Text response length: {len(text_response)} characters")
            if len(text_response) < 500:
                logger.debug(f"Text response content: {text_response}")
            else:
                logger.debug(f"Text response content (truncated): {text_response[:500]}...")
        except Exception as e:
            logger.error(f"GPT call failed: {str(e)}", exc_info=True)
            attempt += 1
            continue

        if raw_gpt.get("type") == "function_call":
            logger.info("Processing function call response")
            try:
                logger.debug("Attempting to extract memory from function call")
                candidate_mem = extract_field_from_function_call(raw_gpt, "memory", npc_id)
                logger.debug(f"Extracted candidate memory: {type(candidate_mem)}, " +
                             (f"length: {len(candidate_mem)}" if candidate_mem else "None"))
                
                if candidate_mem:
                    if isinstance(candidate_mem, list) and len(candidate_mem) >= needed_count:
                        logger.info(f"Successfully extracted {len(candidate_mem)} memories from function call")
                        final_mem = candidate_mem
                        break
                    else:
                        logger.info("Candidate memory insufficient, falling back to text parsing")
                        new_mem = parse_memory_from_text(text_response)
                        logger.debug(f"Parsed {len(new_mem)} memories from text")
                else:
                    logger.info("No memory field found in function call, parsing from text")
                    new_mem = parse_memory_from_text(text_response)
                    logger.debug(f"Parsed {len(new_mem)} memories from text")
            except Exception as e:
                logger.error(f"Error extracting memory from function call: {str(e)}", exc_info=True)
                new_mem = parse_memory_from_text(text_response)
                logger.debug(f"Fallback: parsed {len(new_mem)} memories from text")
        else:
            logger.info("No function call detected, parsing memory from text")
            try:
                new_mem = parse_memory_from_text(text_response)
                logger.debug(f"Parsed {len(new_mem)} memories from text")
            except Exception as e:
                logger.error(f"Error parsing memory from text: {str(e)}", exc_info=True)
                new_mem = []

        if len(new_mem) >= needed_count:
            logger.info(f"Sufficient memories found: {len(new_mem)} >= {needed_count}")
            final_mem = new_mem
            break
        else:
            logger.warning(f"Attempt {attempt}: insufficient memory entries ({len(new_mem)}/{needed_count}) => retry.")

    if len(final_mem) < needed_count:
        logger.warning(f"All attempts failed. Returning {len(existing_memories)} existing memories.")
        return existing_memories

    logger.info(f"Successfully generated {len(final_mem)} memories")
    logger.debug(f"Final memories: {json.dumps(final_mem, indent=2)}")
    return final_mem


def parse_memory_from_text(text: str) -> list:
    """
    Try to parse a "memory" array from plain text JSON. Returns [] if not found or invalid.
    """
    import json, re
    import logging
    
    logger = logging.getLogger("parse_memory_from_text")
    logger.setLevel(logging.DEBUG)
    
    logger.debug(f"Attempting to parse memory from text of length {len(text)}")
    
    # 1) Direct parse
    try:
        logger.debug("Trying direct JSON parse")
        data = json.loads(text)
        mem = data.get("memory")
        if isinstance(mem, list):
            logger.info(f"Successfully parsed memory directly: {len(mem)} entries")
            return mem
        else:
            logger.debug(f"Found 'memory' key but it's not a list: {type(mem)}")
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {str(e)}")

    # 2) Fallback: look for a curly-brace substring
    logger.debug("Attempting regex extraction of JSON")
    match_j = re.search(r'(\{[\s\S]*\})', text)
    if match_j:
        jstr = match_j.group(1)
        logger.debug(f"Found JSON-like substring of length {len(jstr)}")
        try:
            data = json.loads(jstr)
            mem = data.get("memory")
            if isinstance(mem, list):
                logger.info(f"Successfully parsed memory from substring: {len(mem)} entries")
                return mem
            else:
                logger.debug(f"Found 'memory' key in substring but it's not a list: {type(mem)}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse of substring failed: {str(e)}")
    else:
        logger.debug("No JSON-like pattern found with regex")

    # 3) Additional robust attempts
    logger.debug("Attempting to find array pattern directly")
    try:
        array_match = re.search(r'"memory"\s*:\s*(\[[\s\S]*?\])', text)
        if array_match:
            array_str = array_match.group(1)
            logger.debug(f"Found array-like pattern of length {len(array_str)}")
            try:
                mem_array = json.loads(array_str)
                if isinstance(mem_array, list):
                    logger.info(f"Successfully parsed memory array directly: {len(mem_array)} entries")
                    return mem_array
            except json.JSONDecodeError as e:
                logger.debug(f"Array parse failed: {str(e)}")
    except Exception as e:
        logger.debug(f"Array extraction attempt failed: {str(e)}")

    logger.warning("All parsing attempts failed, returning empty list")
    return []



async def refine_affiliations(
    user_id: int,
    conversation_id: int,
    npc_id: int,
    npc_data: dict,
    environment_desc: str,
    max_retries: int = 2
) -> list:
    """
    Ask GPT solely for an 'affiliations' array of 3-5 distinct groups or factions.
    If GPT fails, we fallback to npc_data["affiliations"].
    """
    existing_affils = npc_data.get("affiliations", [])
    attempt = 0
    final_affils = []

    schedule_txt = json.dumps(npc_data.get("schedule", {}), indent=2)
    hobbies_txt = npc_data.get("hobbies", [])
    likes_txt = npc_data.get("likes", [])
    npc_name = npc_data.get("npc_name", "UnknownNPC")

    while attempt < max_retries:
        attempt += 1
        logging.info(f"[refine_affiliations] Attempt {attempt} of {max_retries}")

        # Build the prompt
        system_prompt = f"""
NPC name: {npc_name}
Environment:
{environment_desc}

Return strictly valid JSON with exactly 1 top-level key:
  "affiliations"

The value is an array of 3-5 distinct groups or factions the NPC belongs to.
They must make sense given the NPC's schedule, hobbies, likes, and environment.

Schedule:
{schedule_txt}
Hobbies: {hobbies_txt}
Likes: {likes_txt}

No duplicates or near-duplicates. If you cannot comply, return {{}}.
"""

        # Call GPT
        raw_gpt = await asyncio.to_thread(
            get_chatgpt_response,
            conversation_id,
            environment_desc,
            system_prompt
        )

        text_response = raw_gpt.get("response", "")
        logging.debug(f"[refine_affiliations] GPT raw => {raw_gpt}")

        extracted_affils = []

        # 1) If GPT used function_call
        if raw_gpt.get("type") == "function_call":
            # Attempt to extract 'affiliations' from function call
            extracted_affils = extract_field_from_function_call(raw_gpt, "affiliations", npc_id)
            if not extracted_affils:
                # Optionally parse 'narrative' or other text if you want a fallback
                narrative_text = raw_gpt.get("function_args", {}).get("narrative", "")
                if narrative_text:
                    # If you want, parse bullet lines from narrative
                    pass  # not implemented here

        else:
            # 2) Normal text response => parse top-level JSON
            try:
                data = json.loads(text_response)
                maybe = data.get("affiliations")
                if isinstance(maybe, list):
                    extracted_affils = maybe
            except:
                # 3) curly brace fallback
                match_j = re.search(r'(\{[\s\S]*\})', text_response)
                if match_j:
                    jstr = match_j.group(1)
                    try:
                        data = json.loads(jstr)
                        maybe = data.get("affiliations")
                        if isinstance(maybe, list):
                            extracted_affils = maybe
                    except:
                        pass

        # Now check if we have 3-5 distinct items
        if extracted_affils and 3 <= len(extracted_affils) <= 5:
            final_affils = extracted_affils
            logging.info("[refine_affiliations] Found valid affiliations array.")
            break
        else:
            logging.warning(f"[refine_affiliations] attempt={attempt} => invalid array => retrying...")

    # If still empty, fallback
    if not final_affils:
        logging.warning("[refine_affiliations] Fallback => using existing affiliations or empty.")
        return existing_affils

    return final_affils

async def refine_location_and_relationships(
    user_id: int,
    conversation_id: int,
    npc_id: int, 
    npc_data: dict,
    environment_desc: str,
    max_retries: int = 2
) -> tuple[str, list]:
    """
    1) GPT request focusing on ONLY "current_location" and "relationships".
    2) We show existing relationships as context, and GPT can revise them or
       add new ones if it makes sense. We also want a short location string.
    3) If GPT fails, fallback to old or blank.
    """
    attempt = 0
    final_loc = ""
    final_rels = npc_data.get("relationships", [])

    while attempt < max_retries:
        attempt += 1
        existing_rels_text = "\n".join(
            f"- entity_id={r.get('entity_id')}, type={r.get('entity_type')}, label='{r.get('relationship_label')}'"
            for r in final_rels
        )
        system_prompt = f"""
We have an NPC in a femdom environment.

Return strictly valid JSON with EXACTLY these 2 keys:
{{
  "current_location": "Short string describing where the NPC is now",
  "relationships": [
    {{
      "entity_id": 123,
      "entity_type": "npc" or "player",
      "relationship_label": "ally or mother or friend, etc."
    }},
    ...
  ]
}}

We already have these relationships:
{existing_rels_text}

You may revise or add new ones if needed. 
No extra commentary, code fences, or function calls. If you cannot comply, return {{}}.

NPC data (partial):
{json.dumps(npc_data, indent=2)}
Environment: {environment_desc}
"""
        raw_gpt = await asyncio.to_thread(
            get_chatgpt_response,
            conversation_id,
            environment_desc,
            system_prompt
        )
        text_response = raw_gpt.get("response","")

        new_loc = ""
        new_rels = []

        if raw_gpt.get("type") == "function_call":
            loc_candidate = extract_field_from_function_call(raw_gpt, "current_location", npc_id)
            rel_candidate = extract_field_from_function_call(raw_gpt, "relationships", npc_id)

            if isinstance(loc_candidate, str):
                new_loc = loc_candidate
            if isinstance(rel_candidate, list):
                new_rels = rel_candidate

        else:
            try:
                data = json.loads(text_response)
                loc_val = data.get("current_location","")
                rel_val = data.get("relationships", [])
                if isinstance(loc_val, str):
                    new_loc = loc_val
                if isinstance(rel_val, list):
                    new_rels = rel_val
            except:
                match_j = re.search(r'(\{[\s\S]*\})', text_response)
                if match_j:
                    jstr = match_j.group(1)
                    try:
                        data = json.loads(jstr)
                        loc_val = data.get("current_location", "")
                        rel_val = data.get("relationships", [])
                        if isinstance(loc_val, str):
                            new_loc = loc_val
                        if isinstance(rel_val, list):
                            new_rels = rel_val
                    except:
                        pass

        if new_loc and isinstance(new_rels, list):
            final_loc = new_loc
            final_rels = new_rels
            break
        else:
            logging.warning("[refine_location_and_relationships] attempt=%d => invalid, retry", attempt)

    return (final_loc, final_rels)

async def refine_npc_final_data(
    user_id: int,
    conversation_id: int,
    npc_id: int,
    day_names: list,
    environment_desc: str
):
    """
    Orchestrates multiple smaller GPT calls instead of 1 big one:
      1) refine_physical_description
      2) refine_schedule
      3) refine_location_and_relationships  (so final relationships are set)
      4) refine_memories_and_affiliations   (memories can reference updated relationships)

    Then merges & updates DB.
    """

    ### 1) Fetch from DB
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
    conn.close()

    if not row:
        logging.warning(f"[refine_npc_final_data] NPC {npc_id} not found.")
        return

    columns = [
        "npc_name","introduced","sex","dominance","cruelty","closeness","trust","respect","intensity",
        "archetypes","archetype_summary","archetype_extras_summary",
        "likes","dislikes","hobbies","personality_traits",
        "age","birthdate","relationships","memory","schedule",
        "affiliations","physical_description"
    ]
    npc_data = dict(zip(columns, row))

    # parse JSON fields
    to_jsonify = [
        "archetypes","likes","dislikes","hobbies","personality_traits",
        "relationships","affiliations","memory","schedule"
    ]
    for f in to_jsonify:
        val = npc_data.get(f)
        if isinstance(val, str):
            try:
                if f == "schedule":
                    npc_data[f] = json.loads(val or "{}")
                else:
                    npc_data[f] = json.loads(val or "[]")
            except:
                if f == "schedule":
                    npc_data[f] = {}
                else:
                    npc_data[f] = []

    logging.info(f"[refine_npc_final_data] Loaded NPC {npc_data.get('npc_name')} => ID={npc_id}")

    # a) physical description
    new_desc = await refine_physical_description(
        user_id, conversation_id, npc_id, npc_data, environment_desc
    )

    if isinstance(new_desc, dict):
        new_desc = new_desc.get("physical_description", "")

    # b) schedule
    new_sched = await refine_schedule(
        user_id, conversation_id, npc_id, npc_data, environment_desc, day_names
    )

    # c) location + relationships => we finalize relationships before memories
    new_loc, new_rels = await refine_location_and_relationships(
        user_id, conversation_id, npc_id, npc_data, environment_desc
    )

    if isinstance(new_loc, dict):
        new_loc = new_loc.get("current_location", "")

    # d) memories + affiliations (now that relationships are final)
    new_mem = await refine_memories(
        user_id, conversation_id, npc_id, npc_data, environment_desc
    )

    new_affil = await refine_affiliations(
        user_id, conversation_id, npc_id, npc_data, environment_desc
    )

    #
    # 3) Merge results with existing data
    #

    # Physical description
    if len(new_desc) >= 30:  # or whatever threshold you want
        npc_data["physical_description"] = new_desc

    # Schedule
    if new_sched:
        npc_data["schedule"] = new_sched

    # Current location
    if new_loc:
        npc_data["current_location"] = new_loc

    # Relationships
    if new_rels:
        npc_data["relationships"] = new_rels

    # Memories
    if new_mem:
        npc_data["memory"] = new_mem

    # Affiliations
    if new_affil:
        npc_data["affiliations"] = new_affil

    #
    # 4) Write final data to DB
    #
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE NPCStats
            SET 
                physical_description=%s,
                schedule=%s,
                memory=%s,
                affiliations=%s,
                current_location=%s,
                relationships=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (
            npc_data["physical_description"],  # This will now be text, not JSON
            json.dumps(npc_data["schedule"]),
            json.dumps(npc_data["memory"]),
            json.dumps(npc_data["affiliations"]),
            npc_data.get("current_location", ""),
            json.dumps(npc_data["relationships"]),
            user_id, conversation_id, npc_id
        ))
        conn.commit()
        logging.info(f"[refine_npc_final_data] Updated NPC {npc_id} after multi-step GPT calls.")
    except Exception as e:
        conn.rollback()
        logging.error(f"[refine_npc_final_data] DB update failed: {e}")
    finally:
        cur.close()
        conn.close()

    #
    # 5) Propagate memories
    #
    npc_name = fetch_npc_name(user_id, conversation_id, npc_id) or "Unknown"
    propagate_shared_memories(user_id, conversation_id, npc_id, npc_name, npc_data["memory"])

    # Return final data
    return {
        "physical_description": npc_data["physical_description"],
        "schedule": npc_data["schedule"],
        "memory": npc_data["memory"],
        "affiliations": npc_data["affiliations"],
        "relationships": npc_data["relationships"],
        "current_location": npc_data.get("current_location", "")
    }

    
async def spawn_single_npc(
    user_id: int,
    conversation_id: int,
    environment_desc: str,
    day_names: list
) -> int:
    """
    Create a partial NPC stub, insert into DB => get real NPC ID,
    assign relationships, refine data. 
    Ensures we keep the same npc_id throughout so we don't mismatch GPT IDs.
    """
    logging.info("[spawn_single_npc] Starting spawn for a new NPC.")
    partial_npc = create_npc_partial(
        user_id, conversation_id, sex="female", total_archetypes=4, environment_desc=environment_desc
    )
    logging.info(f"[spawn_single_npc] Partial NPC created: {partial_npc}")

    # Insert => get the real ID from DB
    npc_id = await insert_npc_stub_into_db(partial_npc, user_id, conversation_id)
    logging.info(f"[spawn_single_npc] NPC stub inserted with ID: {npc_id}")

    # Assign relationships using that official ID
    await assign_random_relationships(
        user_id, conversation_id, npc_id, partial_npc["npc_name"], partial_npc.get("archetypes", [])
    )
    logging.info(f"[spawn_single_npc] Relationships assigned for NPC ID: {npc_id}")

    # Final refinement
    await refine_npc_final_data(
        user_id, conversation_id, npc_id, day_names, environment_desc
    )
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

def extract_chase_schedule(data):
    """Helper function to extract the ChaseSchedule from various possible structures"""
    if isinstance(data, dict):
        # Direct match at top level
        if "ChaseSchedule" in data:
            return data["ChaseSchedule"]
        
        # Look one level deeper in case it's nested
        for key, value in data.items():
            if isinstance(value, dict) and "ChaseSchedule" in value:
                return value["ChaseSchedule"]
            
            # If we have a key that matches a day name, this might be the schedule itself
            # without the "ChaseSchedule" wrapper
            if key in day_names and all(period in value for period in ["Morning", "Afternoon", "Evening", "Night"]):
                # We found what looks like a day's schedule, so the parent object might be the schedule
                if all(day in data for day in day_names):
                    return data
        
        # As a last resort, check if the structure matches our expected format directly
        if all(day in data for day in day_names) and all(
            all(period in data[day] for period in ["Morning", "Afternoon", "Evening", "Night"])
            for day in data if day in day_names
        ):
            return data
    
    return {}

async def generate_chase_schedule(
    user_id: int,
    conversation_id: int,
    environment_desc: str,
    day_names: list
) -> dict:
    """
    1) Gather 'Chase' stats from PlayerStats (for flavor).
    2) GPT call to produce Chase's schedule as a dict keyed by each day => subkeys (Morning, Afternoon, Evening, Night)
    3) Store the schedule in CurrentRoleplay.
    4) Return the schedule.
    """
    import json
    import logging

    # Step A: Load 'Chase' stats from PlayerStats
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

    # Build Chase partial data for GPT
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
    chase_prompt = f"""
We have a player character 'Chase' in this femdom environment.
Environment:
{environment_desc}

Chase partial data:
{json.dumps(chase_data, indent=2)}

The list of days is: {day_names}

Using details from the setting, please generate a realistic daily schedule for "Chase" for each of the days listed. Your output must be a valid JSON object with exactly one top-level key, "ChaseSchedule". The value of "ChaseSchedule" must be an object whose keys exactly match the days in the list (e.g., if the list is ["Monday", "Tuesday", ...], then these must be the keys). For each day, the value must be an object with exactly the following keys:
- "Morning"
- "Afternoon"
- "Evening"
- "Night"

Each of these keys should map to a short string describing Chase's activity during that time slot.

Do not include any extra keys, text, or commentary. Do not wrap your output in code fences.
"""

    # Step C: Call GPT and capture the response
    try:
        response = get_openai_client().chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": chase_prompt}],
            temperature=0.7,
        )
        # Convert the response to a dictionary
        response_dict = response.dict()
    except Exception as e:
        logging.error("[generate_chase_schedule] GPT call error: %s", e)
        return {}

    # Initialize an empty dict for extracted data
    chase_sched_data = {}

    # Process the response depending on its type
    if response_dict.get("type") == "function_call":
        chase_sched_data = response_dict.get("function_args", {})
        # Extra extraction: if "ChaseSchedule" is not found, check for a nested "response"
        if not chase_sched_data.get("ChaseSchedule"):
            nested = chase_sched_data.get("response")
            if isinstance(nested, dict) and nested.get("ChaseSchedule"):
                chase_sched_data = nested
        logging.debug(f"Function call extraction yields: {chase_sched_data}")
        chase_schedule = extract_chase_schedule(chase_sched_data)
    else:
        raw_text = response_dict.get("response", "").strip()
        # Remove triple-backticks if present
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            if lines and lines[-1].startswith("```"):
                lines.pop()
            raw_text = "\n".join(lines).strip()
        try:
            chase_sched_data = json.loads(raw_text)
            chase_schedule = extract_chase_schedule(chase_sched_data)
        except Exception as e:
            logging.error("[generate_chase_schedule] parse error: %s", e)
            chase_sched_data = {}
            chase_schedule = {}

    # Ensure we have a valid ChaseSchedule
    chase_schedule = chase_sched_data.get("ChaseSchedule", {})
    if not chase_schedule:
        logging.warning("[generate_chase_schedule] GPT gave no 'ChaseSchedule'.")
        chase_schedule = {}

    # Step D: Store the schedule in the database (CurrentRoleplay)
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
async def init_chase_schedule(user_id, conversation_id, combined_env, day_names):
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
