# logic/gpt_helpers.py
import json
import logging
from logic.gpt_utils import spaced_gpt_call  # Import from the separate GPT utils module

async def adjust_npc_preferences(npc_data, environment_desc, conversation_id):
    """
    Query GPT to generate updated NPC preferences (likes, dislikes, and hobbies)
    that are tailored to an environment dominated by powerful females.
    This function is designed to be used as part of a function call payload for apply_universal_update.
    It expects GPT to return a function call payload with a key "npc_creations" that is a list of update objects.
    If found, it extracts the likes, dislikes, and hobbies from the first object.
    """
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    hobbies = npc_data.get("hobbies", [])
    
    prompt = (
        "Given the following NPC preferences:\n"
        "Likes: {likes}\n"
        "Dislikes: {dislikes}\n"
        "Hobbies: {hobbies}\n\n"
        "Adapt these preferences so that they are specifically tailored to an environment dominated by powerful females. "
        "For each entry, produce one cohesive version that fits the current setting. "
        "Output your answer as a function call payload conforming to the following JSON schema:\n\n"
        "{\n"
        "  \"npc_creations\": [\n"
        "    {\n"
        "      \"likes\": [string, ...],\n"
        "      \"dislikes\": [string, ...],\n"
        "      \"hobbies\": [string, ...]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Do not output any additional text."
    ).format(likes=likes, dislikes=dislikes, hobbies=hobbies)
    
    logging.info("Adjusting NPC preferences with prompt: %s", prompt)
    reply = await spaced_gpt_call(conversation_id, environment_desc, prompt)
    
    # Check if GPT returned a function call.
    if reply.get("type") == "function_call":
        args = reply.get("function_args", {})
        if "npc_creations" in args:
            updates = args["npc_creations"]
            if isinstance(updates, list) and updates:
                update_obj = updates[0]
                # If update_obj is a string, try parsing it as JSON.
                if isinstance(update_obj, str):
                    try:
                        update_obj = json.loads(update_obj)
                    except Exception as e:
                        logging.error("Could not parse update_obj as JSON: %s", e)
                        return {"likes": likes, "dislikes": dislikes, "hobbies": hobbies}
                # Now check for the expected keys.
                if all(k in update_obj for k in ["likes", "dislikes", "hobbies"]):
                    return {
                        "likes": update_obj["likes"],
                        "dislikes": update_obj["dislikes"],
                        "hobbies": update_obj["hobbies"]
                    }
                else:
                    logging.warning("Expected keys not found in npc_creations update; falling back.")
                    return {"likes": likes, "dislikes": dislikes, "hobbies": hobbies}
            else:
                logging.warning("npc_creations update is empty or not a list; falling back.")
                return {"likes": likes, "dislikes": dislikes, "hobbies": hobbies}
        else:
            logging.warning("GPT function call did not include 'npc_creations'; falling back.")
            return {"likes": likes, "dislikes": dislikes, "hobbies": hobbies}
    elif reply.get("response"):
        try:
            data = json.loads(reply["response"].strip())
            return data
        except Exception as e:
            logging.error("Error parsing adjusted preferences JSON: %s", e)
            return {"likes": likes, "dislikes": dislikes, "hobbies": hobbies}
    else:
        logging.warning("GPT did not return any valid preferences; falling back to original values.")
        return {"likes": likes, "dislikes": dislikes, "hobbies": hobbies}


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
