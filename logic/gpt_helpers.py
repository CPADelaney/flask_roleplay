# logic/gpt_helpers.py
import json
import logging
from logic.gpt_utils import spaced_gpt_call  # Import from the separate GPT utils module

async def adjust_npc_preferences(npc_data, environment_desc, conversation_id):
    """
    Query GPT to generate updated NPC preferences (likes, dislikes, and hobbies)
    tailored to an environment dominated by powerful females.
    This function expects GPT to return a function call payload with a key "npc_creations"
    that is a list of update objects. It then extracts the likes, dislikes, and hobbies
    from the first object in that list.

    If GPT returns a placeholder value (e.g. just "npc_creations"), the function will
    retry up to max_retries times, then fall back to the original values.
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
        "Return only a JSON object with one key: 'npc_creations'. This key should map to an array containing exactly one object that has the following keys:\n"
        "  - 'likes': an array of strings representing the adjusted likes,\n"
        "  - 'dislikes': an array of strings representing the adjusted dislikes, and\n"
        "  - 'hobbies': an array of strings representing the adjusted hobbies.\n"
        "Do not include any additional text or markdown formatting. "
        "Do not include anything within parenthesis."
        "Ensure the likes, dislikes, and hobbies make sense within the environment."
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
        # Process the GPT response:
        if reply.get("type") == "function_call":
            args = reply.get("function_args", {})
            if "npc_creations" in args:
                updates = args["npc_creations"]

                # If updates is a string, process it:
                if isinstance(updates, str):
                    stripped_updates = updates.strip()
                    logging.debug("Stripped npc_creations value: %r", stripped_updates)
                    # Remove any surrounding quotes before comparing
                    cleaned = stripped_updates.strip('"').lower()
                    if cleaned == "npc_creations":
                        logging.error("Received placeholder output for npc_creations: %r", stripped_updates)
                        retry_count += 1
                        continue
                    try:
                        updates = json.loads(stripped_updates)
                    except Exception as e:
                        logging.error("Failed to parse npc_creations string as JSON: %s", e)
                        retry_count += 1
                        continue

                # Now expect updates to be a list.
                if isinstance(updates, list) and updates:
                    update_obj = updates[0]
                    if isinstance(update_obj, str):
                        stripped_obj = update_obj.strip()
                        logging.debug("Stripped update object: %r", stripped_obj)
                        cleaned_obj = stripped_obj.strip('"').lower()
                        if cleaned_obj == "npc_creations":
                            logging.error("Received placeholder text in update object: %r", stripped_obj)
                            retry_count += 1
                            continue
                        if not stripped_obj.startswith("{"):
                            logging.error("Update object string does not look like a JSON object: %r", stripped_obj)
                            retry_count += 1
                            continue
                        try:
                            update_obj = json.loads(stripped_obj)
                        except Exception as e:
                            logging.error("JSON parsing failed for update_obj: %s", e)
                            retry_count += 1
                            continue
                    # Verify that all expected keys exist.
                    if all(k in update_obj for k in ["likes", "dislikes", "hobbies"]):
                        logging.info("Successfully extracted updated preferences on retry %d", retry_count)
                        return {
                            "likes": update_obj["likes"],
                            "dislikes": update_obj["dislikes"],
                            "hobbies": update_obj["hobbies"]
                        }
                    else:
                        logging.warning("Expected keys not found in npc_creations update; retrying.")
                        retry_count += 1
                        continue
                else:
                    logging.warning("npc_creations update is empty or not a list; retrying.")
                    retry_count += 1
                    continue
            else:
                logging.warning("GPT function call did not include 'npc_creations'; retrying.")
                retry_count += 1
                continue
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
                logging.debug("Raw GPT response text: %r", response_text)
                data = json.loads(response_text)
                if all(k in data for k in ["likes", "dislikes", "hobbies"]):
                    logging.info("Successfully extracted updated preferences from response on retry %d", retry_count)
                    return data
                else:
                    logging.warning("Response JSON does not contain expected keys; retrying.")
                    retry_count += 1
                    continue
            except Exception as e:
                logging.error("Error parsing JSON from response: %s", e)
                retry_count += 1
                continue
        else:
            logging.warning("GPT did not return a valid response; retrying.")
            retry_count += 1

    logging.error("Max retries reached; using original preferences instead.")
    return original_values



async def generate_npc_affiliations_and_schedule(npc_data, environment_desc, conversation_id):
    """
    Using an NPC's archetype summary along with their likes, dislikes, and hobbies,
    query GPT to generate a JSON object containing both the NPC's affiliations (e.g., teams, clubs, partnerships, associations, etc.)
    and a detailed weekly schedule. The schedule should have keys for each day of the week (using your immersive day names),
    with nested keys for 'Morning', 'Afternoon', 'Evening', and 'Night'.
    """
    # For this example, we use a fallback array for immersive day names.
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

    # Log the raw GPT reply for troubleshooting
    logging.debug("Raw GPT reply: %s", reply)

    # Determine if we got a plain response or a function call response.
    if reply.get("response"):
        affiliations_schedule_text = reply["response"].strip()
    elif reply.get("type") == "function_call":
        args = reply.get("function_args", {})
        # Try first the "affiliations_schedule" key…
        if "affiliations_schedule" in args:
            affiliations_schedule_text = args["affiliations_schedule"].strip()
        # …or else try "npc_updates"
        elif "npc_updates" in args:
            updates = args.get("npc_updates")
            if isinstance(updates, list) and len(updates) > 0:
                first_update = updates[0]
                result_dict = {
                    "affiliations": first_update.get("affiliations", []),
                    "schedule": first_update.get("schedule", {})
                }
                logging.info("Using npc_updates payload to derive affiliations and schedule: %s", result_dict)
                return result_dict
            else:
                logging.warning("npc_updates is empty or not a list; returning default values.")
                return {"affiliations": [], "schedule": {}}
        else:
            logging.warning("GPT function call did not include expected keys; returning defaults.")
            return {"affiliations": [], "schedule": {}}
    else:
        logging.warning("GPT did not return affiliations and schedule; using default empty values.")
        return {"affiliations": [], "schedule": {}}

    # Strip markdown code fences if present.
    if affiliations_schedule_text.startswith("```"):
        lines = affiliations_schedule_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        affiliations_schedule_text = "\n".join(lines).strip()

    # Log the processed raw text before parsing JSON.
    logging.debug("Processed GPT output for affiliations and schedule: %s", affiliations_schedule_text)

    # Try parsing the JSON and log a detailed error if it fails.
    try:
        result = json.loads(affiliations_schedule_text)
    except Exception as e:
        logging.error("Error parsing affiliations and schedule JSON: %s. Raw text: %s", e, affiliations_schedule_text, exc_info=True)
        result = {"affiliations": [], "schedule": {}}

    # Validate the result schema.
    if not (isinstance(result, dict) and "affiliations" in result and "schedule" in result):
        logging.error("Parsed result does not match expected schema: %s", result)
        result = {"affiliations": [], "schedule": {}}

    return result
