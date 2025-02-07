# logic/gpt_helpers.py
import json
import logging
from logic.gpt_utils import spaced_gpt_call  # Import from the separate GPT utils module

import json
import logging
from logic.gpt_utils import spaced_gpt_call  # Import the GPT call helper

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
        "Given the following NPC preferences:\n"
        "Likes: {likes}\n"
        "Dislikes: {dislikes}\n"
        "Hobbies: {hobbies}\n\n"
        "Adapt these preferences so that they are specifically tailored to an environment dominated by powerful females. "
        "For each entry, produce one cohesive version that fits the current setting. "
        "Output your answer as a function call payload conforming exactly to the following JSON schema with no additional text or markdown:\n\n"
        "{\n"
        "  \"npc_creations\": [\n"
        "    {\n"
        "      \"likes\": [\"string\", ...],\n"
        "      \"dislikes\": [\"string\", ...],\n"
        "      \"hobbies\": [\"string\", ...]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    ).format(likes=likes, dislikes=dislikes, hobbies=hobbies)

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
    and a detailed weekly schedule. The schedule should have keys for each day (Monday through Sunday),
    with nested keys for 'Morning', 'Afternoon', 'Evening', and 'Night'.
    """
    archetype_summary = npc_data.get("archetype_summary", "")
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    hobbies = npc_data.get("hobbies", [])
    
    prompt = (
        "Given the following NPC information:\n"
        "Archetype Summary: {archetype_summary}\n"
        "Likes: {likes}\n"
        "Dislikes: {dislikes}\n"
        "Hobbies: {hobbies}\n\n"
        "Determine the NPC's affiliations (teams, clubs, partnerships, associations, etc.) and create a detailed weekly schedule. "
        "The schedule should be formatted as a JSON object with keys for each day of the week (Monday to Sunday), "
        "and for each day include nested keys 'Morning', 'Afternoon', 'Evening', and 'Night'. "
        "Return only a JSON object with two keys: 'affiliations' (an array) and 'schedule' (the weekly schedule)."
    ).format(
        archetype_summary=archetype_summary,
        likes=likes,
        dislikes=dislikes,
        hobbies=hobbies
    )
    
    logging.info("Generating NPC affiliations and schedule with prompt: %s", prompt)
    reply = await spaced_gpt_call(conversation_id, environment_desc, prompt)
    
    # First check for a direct response.
    if reply.get("response"):
        affiliations_schedule_text = reply["response"].strip()
    # Then check if it's a function call.
    elif reply.get("type") == "function_call":
        args = reply.get("function_args", {})
        if "affiliations_schedule" in args:
            affiliations_schedule_text = args["affiliations_schedule"].strip()
        elif "npc_updates" in args:
            # Assume GPT returned a list of npc_updates; use the first one.
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
    
    # Remove markdown code fences if present.
    if affiliations_schedule_text.startswith("```"):
        lines = affiliations_schedule_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        affiliations_schedule_text = "\n".join(lines).strip()
    
    try:
        result = json.loads(affiliations_schedule_text)
    except Exception as e:
        logging.error("Error parsing affiliations and schedule: %s", e)
        result = {"affiliations": [], "schedule": {}}
    
    return result
