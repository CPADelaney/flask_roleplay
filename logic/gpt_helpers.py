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

    The schedule must use the immersive day names you provide. If none
    are provided, we fallback to ["Sol","Luna","Terra","Vesta","Mercury","Venus","Mars"].

    Returns:
      dict with keys "likes", "dislikes", "hobbies", "affiliations", "schedule".
      If GPT fails after 3 tries, returns fallback from npc_data.
    """
    if immersive_days is None:
        immersive_days = ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]

    # Retrieve existing preferences
    archetype_summary = npc_data.get("archetype_summary", "")
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    hobbies = npc_data.get("hobbies", [])

    # Build the prompt
    prompt = (
        "Given the following NPC information:\n"
        f"Archetype Summary: {archetype_summary}\n"
        f"Environment: {environment_desc}\n\n"
        "NPC Preferences:\n"
        f"Likes: {likes}\n"
        f"Dislikes: {dislikes}\n"
        f"Hobbies: {hobbies}\n\n"
        "Instructions:\n"
        "1. Adapt these preferences so that they are specifically tailored "
        "   to an environment dominated by powerful females.\n"
        "2. Determine the NPC's affiliations (teams, clubs, partnerships, associations, etc.)\n"
        "   that fit within this environment.\n"
        "3. Create a detailed weekly schedule using these immersive day names: "
        f"{', '.join(immersive_days)}. For each day, include 'Morning', 'Afternoon', "
        "'Evening', 'Night'.\n\n"
        "Return only a JSON object with exactly the following five keys:\n"
        "\"likes\", \"dislikes\", \"hobbies\", \"affiliations\", \"schedule\".\n"
        "No extra keys or text."
    )

    logging.info("Adjusting complete NPC details with prompt: %s", prompt)

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        # Call GPT with your new spaced_gpt_call, including a short initial delay
        reply = await spaced_gpt_call(
            conversation_id,
            environment_desc,
            prompt,
            delay=1.0,         # you can adjust
            max_retries=5      # how many times to try if we see 429
        )

        # If GPT responded with a function call
        if reply.get("type") == "function_call":
            args = reply.get("function_args", {})
            # If top-level keys are exactly present
            if all(k in args for k in ["likes","dislikes","hobbies","affiliations","schedule"]):
                return {
                    "likes": args["likes"],
                    "dislikes": args["dislikes"],
                    "hobbies": args["hobbies"],
                    "affiliations": args["affiliations"],
                    "schedule": args["schedule"]
                }
            # Or possibly in npc_updates
            elif "npc_updates" in args:
                updates = args["npc_updates"]
                if isinstance(updates, list) and len(updates) > 0:
                    first_update = updates[0]
                    if all(k in first_update for k in ["likes","dislikes","hobbies","affiliations","schedule"]):
                        return first_update
            # Or possibly in npc_creations
            elif "npc_creations" in args:
                creations = args["npc_creations"]
                if isinstance(creations, list) and len(creations) > 0:
                    creation_obj = creations[0]
                    if all(k in creation_obj for k in ["likes","dislikes","hobbies","affiliations","schedule"]):
                        return creation_obj

            logging.warning("Incomplete keys in function_args; retrying. Attempt=%d", retry_count+1)
            retry_count += 1
            continue

        # If GPT returned plain text "response"
        elif reply.get("response"):
            response_text = reply["response"].strip()
            # Remove fences if present
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            try:
                data_parsed = json.loads(response_text)
                if all(k in data_parsed for k in ["likes","dislikes","hobbies","affiliations","schedule"]):
                    return data_parsed
                else:
                    logging.warning("Response JSON missing keys; retrying. Attempt=%d", retry_count+1)
                    retry_count += 1
            except Exception as e:
                logging.error("Error parsing JSON from response: %s. Attempt=%d", e, retry_count+1)
                retry_count += 1

        else:
            logging.warning("GPT did not return function_call or response. Attempt=%d", retry_count+1)
            retry_count += 1

    # If we exhausted all attempts
    logging.error("Max attempts reached in adjust_npc_complete; returning fallback from npc_data.")
    return {
        "likes": likes,
        "dislikes": dislikes,
        "hobbies": hobbies,
        "affiliations": npc_data.get("affiliations", []),
        "schedule": npc_data.get("schedule", {})
    }
