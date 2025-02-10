# logic/gpt_helpers.py
import json
import logging
from logic.gpt_utils import spaced_gpt_call  # Import from the separate GPT utils module

async def adjust_npc_complete(npc_data, environment_desc, conversation_id, immersive_days=None):
    """
    Adapt NPC details by combining preference adjustments and schedule/affiliation generation
    into one GPT call. Tailor all details to an environment dominated by powerful females.
    
    We ask GPT to produce a JSON with exactly these keys:
      - "likes": (array of strings)
      - "dislikes": (array of strings)
      - "hobbies": (array of strings)
      - "affiliations": (array of strings)
      - "schedule": (object with custom day names, each having "Morning","Afternoon","Evening","Night")

    The schedule must use the immersive day names (e.g., "Alpha", "Beta", etc.) you provide. If none
    are provided, a fallback array is used.

    Parameters:
      npc_data: dict of the NPC's current details (preferences, archetype summary, etc.).
      environment_desc: str describing the environment.
      conversation_id: conversation context id for the GPT call.
      immersive_days: a list of immersive day names (e.g. ["Alpha","Beta","Gamma"]) for the schedule.
                     Defaults to ["Sol","Luna","Terra","Vesta","Mercury","Venus","Mars"] if None.

    Returns:
      A dict with keys "likes", "dislikes", "hobbies", "affiliations", "schedule".
      If GPT fails after retries, returns fallback from npc_data.
    """
    # Use provided day names or fallback
    if immersive_days is None:
        immersive_days = ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]

    # Retrieve NPC details
    archetype_summary = npc_data.get("archetype_summary", "")
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    hobbies = npc_data.get("hobbies", [])

    # Build the combined prompt
    prompt = (
        "Given the following NPC information:\n"
        f"Archetype Summary: {archetype_summary}\n"
        f"Environment: {environment_desc}\n\n"
        "NPC Preferences:\n"
        f"Likes: {likes}\n"
        f"Dislikes: {dislikes}\n"
        f"Hobbies: {hobbies}\n\n"
        "Instructions:\n"
        "1. Adapt the above preferences so that they are specifically tailored "
        "   to an environment dominated by powerful females.\n"
        "2. Determine the NPC's affiliations (teams, clubs, partnerships, associations, etc.) "
        "   that fit within this environment.\n"
        "3. Create a detailed weekly schedule using these immersive day names: "
        f"{', '.join(immersive_days)}. For each day, include 'Morning', 'Afternoon', "
        "'Evening', 'Night'.\n\n"
        "Return only a JSON object with exactly the following five keys (and no extra keys):\n"
        "  - \"likes\"\n"
        "  - \"dislikes\"\n"
        "  - \"hobbies\"\n"
        "  - \"affiliations\"\n"
        "  - \"schedule\"\n"
    )

    logging.info("Adjusting complete NPC details with prompt: %s", prompt)

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        reply = await spaced_gpt_call(conversation_id, environment_desc, prompt)

        # 1) If GPT used a function call
        if reply.get("type") == "function_call":
            args = reply.get("function_args", {})

            # Case 1: Top-level keys are exactly present
            if all(key in args for key in ["likes", "dislikes", "hobbies", "affiliations", "schedule"]):
                return {
                    "likes": args["likes"],
                    "dislikes": args["dislikes"],
                    "hobbies": args["hobbies"],
                    "affiliations": args["affiliations"],
                    "schedule": args["schedule"]
                }

            # Case 2: GPT puts them inside "npc_updates"
            elif "npc_updates" in args:
                updates = args["npc_updates"]
                if isinstance(updates, list) and len(updates) > 0:
                    update_obj = updates[0]
                    if all(key in update_obj for key in ["likes", "dislikes", "hobbies", "affiliations", "schedule"]):
                        return update_obj

            # (Optional) Case 3: GPT might nest them under "npc_creations" (if you ever do that)
            elif "npc_creations" in args:
                creations = args["npc_creations"]
                if isinstance(creations, list) and len(creations) > 0:
                    creation_obj = creations[0]
                    if all(key in creation_obj for key in ["likes", "dislikes", "hobbies", "affiliations", "schedule"]):
                        return creation_obj

            # If it reached here, it didn't have the required 5 keys
            logging.warning("Incomplete keys in function_args; retrying.")
            retry_count += 1
            continue

        # 2) If GPT returned a plain text "response" (non-function-call)
        elif reply.get("response"):
            try:
                response_text = reply["response"].strip()

                # Strip code fences if present
                if response_text.startswith("```"):
                    lines = response_text.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    response_text = "\n".join(lines).strip()

                data_parsed = json.loads(response_text)

                if all(key in data_parsed for key in ["likes", "dislikes", "hobbies", "affiliations", "schedule"]):
                    return data_parsed
                else:
                    logging.warning("Response JSON missing expected keys; retrying.")
                    retry_count += 1
                    continue
            except Exception as e:
                logging.error("Error parsing JSON from response: %s", e)
                retry_count += 1
                continue

        # 3) If neither function_call nor text is valid
        else:
            logging.warning("GPT did not return a valid response; retrying.")
            retry_count += 1

    # If all retries failed, fall back
    logging.error("Max retries reached; returning original values.")
    return {
        "likes": likes,
        "dislikes": dislikes,
        "hobbies": hobbies,
        "affiliations": npc_data.get("affiliations", []),
        "schedule": npc_data.get("schedule", {})
    }
