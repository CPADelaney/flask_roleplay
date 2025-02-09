# logic/gpt_helpers.py
import json
import logging
from logic.gpt_utils import spaced_gpt_call  # Import from the separate GPT utils module

async def adjust_npc_preferences(npc_data, environment_desc, conversation_id):
    """
    Query GPT to generate updated NPC preferences (likes, dislikes, and hobbies)
    tailored to an environment dominated by powerful females.
    If GPT returns its output in a function call payload, this function checks both:
      - A top-level JSON object with keys "likes", "dislikes", and "hobbies"
      - OR a JSON object with a key "npc_creations" (a list) whose first element has those keys.
    Retries up to max_retries times and falls back to the original values if unsuccessful.
    """
    original_values = {
        "likes": npc_data.get("likes", []),
        "dislikes": npc_data.get("dislikes", []),
        "hobbies": npc_data.get("hobbies", [])
    }
    likes = original_values["likes"]
    dislikes = original_values["dislikes"]
    hobbies = original_values["hobbies"]
    
    prompt = (
        "Given the following environment description and NPC preferences:\n"
        "Environment: {environment_desc}\n\n"
        "NPC Preferences:\n"
        "Likes: {likes}\n"
        "Dislikes: {dislikes}\n"
        "Hobbies: {hobbies}\n\n"
        "Adapt these preferences so that they are specifically tailored to an environment dominated by powerful females. "
        "Return only a JSON object with one key: 'npc_creations'. This key should map to an array containing exactly one object "
        "that has the following keys:\n"
        "  - 'likes': an array of strings representing the adjusted likes,\n"
        "  - 'dislikes': an array of strings representing the adjusted dislikes, and\n"
        "  - 'hobbies': an array of strings representing the adjusted hobbies.\n"
        "Do not include any extra text, markdown formatting, or anything else."
    ).format(
        environment_desc=environment_desc,
        likes=likes,
        dislikes=dislikes,
        hobbies=hobbies
    )
    
    logging.info("Adjusting NPC preferences with prompt: %s", prompt)
    
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        reply = await spaced_gpt_call(conversation_id, environment_desc, prompt)
        logging.debug("Raw GPT reply: %s", reply)
        
        # Check for function call responses first.
        if reply.get("type") == "function_call":
            func_args = reply.get("function_args", {})
            # First try to see if keys are present at the top level.
            if all(k in func_args for k in ["likes", "dislikes", "hobbies"]):
                logging.info("Successfully extracted updated preferences (direct) on retry %d", retry_count)
                return {
                    "likes": func_args["likes"],
                    "dislikes": func_args["dislikes"],
                    "hobbies": func_args["hobbies"]
                }
            # Otherwise, check if a key "npc_creations" exists.
            elif "npc_creations" in func_args:
                updates = func_args["npc_creations"]
                if isinstance(updates, list) and len(updates) > 0:
                    first_update = updates[0]
                    if all(k in first_update for k in ["likes", "dislikes", "hobbies"]):
                        logging.info("Extracted preferences from 'npc_creations' on retry %d", retry_count)
                        return {
                            "likes": first_update["likes"],
                            "dislikes": first_update["dislikes"],
                            "hobbies": first_update["hobbies"]
                        }
                    else:
                        missing_keys = [k for k in ["likes", "dislikes", "hobbies"] if k not in first_update]
                        logging.warning("Missing keys in 'npc_creations': %s", missing_keys)
                else:
                    logging.warning("'npc_creations' update is empty or not a list; retrying.")
            else:
                logging.warning("GPT function call did not include expected keys. Received: %s", func_args)
        
        # Otherwise, check for a plain text response.
        elif reply.get("response"):
            try:
                response_text = reply["response"].strip()
                if response_text.startswith("```"):
                    lines = response_text.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    response_text = "\n".join(lines).strip()
                data = json.loads(response_text)
                if all(k in data for k in ["likes", "dislikes", "hobbies"]):
                    logging.info("Successfully extracted updated preferences from response on retry %d", retry_count)
                    return data
                elif "npc_creations" in data:
                    updates = data["npc_creations"]
                    if isinstance(updates, list) and len(updates) > 0:
                        first_update = updates[0]
                        if all(k in first_update for k in ["likes", "dislikes", "hobbies"]):
                            logging.info("Extracted preferences from 'npc_creations' in response on retry %d", retry_count)
                            return {
                                "likes": first_update["likes"],
                                "dislikes": first_update["dislikes"],
                                "hobbies": first_update["hobbies"]
                            }
                        else:
                            missing_keys = [k for k in ["likes", "dislikes", "hobbies"] if k not in first_update]
                            logging.warning("Missing keys in 'npc_creations' from response: %s", missing_keys)
                    else:
                        logging.warning("Response JSON 'npc_creations' is empty or not a list; retrying.")
                else:
                    logging.warning("Response JSON does not contain expected keys; full response: %s", data)
            except Exception as e:
                logging.error("Error parsing JSON from response: %s", e, exc_info=True)
        else:
            logging.warning("GPT did not return a valid response; retrying.")
        
        retry_count += 1

    logging.error("Max retries reached; using original preferences instead.")
    return original_values



async def generate_npc_affiliations_and_schedule(npc_data, environment_desc, conversation_id):
    """
    Using an NPC's archetype summary along with their likes, dislikes, and hobbies,
    query GPT to generate a JSON object containing both the NPC's affiliations (e.g., teams, clubs, partnerships, associations, etc.)
    and a detailed weekly schedule. The schedule should use your immersive day names and have nested keys for 'Morning', 'Afternoon', 'Evening', and 'Night'.
    """
    # Fallback immersive day names
    immersive_days = ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]

    archetype_summary = npc_data.get("archetype_summary", "")
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    hobbies = npc_data.get("hobbies", [])

    prompt = (
        "Given the following NPC information:\n"
        "Archetype Summary: {archetype_summary}\n"
        "Environment: {environment_desc}\n\n"
        "Likes: {likes}\n"
        "Dislikes: {dislikes}\n"
        "Hobbies: {hobbies}\n\n"
        "Determine the NPC's affiliations (teams, clubs, partnerships, associations, etc.) and create a detailed weekly schedule. "
        "The schedule must use the following immersive day names for the week: {immersive_days}. "
        "For each day, include nested keys 'Morning', 'Afternoon', 'Evening', and 'Night' representing the NPC's activities. "
        "Return only a JSON object with two keys: 'affiliations' (an array) and 'schedule' (the weekly schedule). "
        "Ensure the affiliations and schedule make sense within the environment."
    ).format(
        archetype_summary=archetype_summary,
        environment_desc=environment_desc,
        likes=likes,
        dislikes=dislikes,
        hobbies=hobbies,
        immersive_days=", ".join(immersive_days)
    )

    logging.info("Generating NPC affiliations and schedule with prompt: %s", prompt)
    
    reply = await spaced_gpt_call(conversation_id, environment_desc, prompt)
    
    # Log the full raw reply for debugging purposes.
    logging.debug("Raw GPT reply in generate_npc_affiliations_and_schedule: %s", json.dumps(reply, indent=2))

    # Determine if we got a plain response or a function call response.
    if reply.get("response"):
        affiliations_schedule_text = reply["response"].strip()
    elif reply.get("type") == "function_call":
        args = reply.get("function_args", {})
        if "affiliations_schedule" in args:
            affiliations_schedule_text = args["affiliations_schedule"].strip()
        elif "npc_updates" in args:
            updates = args.get("npc_updates")
            if isinstance(updates, list) and updates:
                first_update = updates[0]
                result_dict = {
                    "affiliations": first_update.get("affiliations", []),
                    "schedule": first_update.get("schedule", {})
                }
                logging.info("Using 'npc_updates' payload to derive affiliations and schedule: %s", result_dict)
                return result_dict
            else:
                logging.warning("'npc_updates' is empty or not a list; returning default values.")
                return {"affiliations": [], "schedule": {}}
        else:
            logging.warning("GPT function call did not include expected keys. Received: %s", args)
            return {"affiliations": [], "schedule": {}}
    else:
        logging.warning("GPT did not return a valid response; using default empty values.")
        return {"affiliations": [], "schedule": {}}

    # Remove markdown code fences if present.
    if affiliations_schedule_text.startswith("```"):
        lines = affiliations_schedule_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        affiliations_schedule_text = "\n".join(lines).strip()

    logging.debug("Processed GPT output for affiliations and schedule: %s", affiliations_schedule_text)

    # Try parsing the JSON and log detailed errors if it fails.
    try:
        result = json.loads(affiliations_schedule_text)
    except Exception as e:
        logging.error("Error parsing affiliations and schedule JSON: %s. Raw text: %s", e, affiliations_schedule_text, exc_info=True)
        result = {"affiliations": [], "schedule": {}}

    # Validate the schema of the parsed JSON.
    if not (isinstance(result, dict) and "affiliations" in result and "schedule" in result):
        logging.error("Parsed result does not match expected schema. Got: %s", result)
        result = {"affiliations": [], "schedule": {}}
    
    return result
