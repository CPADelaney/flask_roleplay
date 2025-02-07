# logic/gpt_helpers.py
import json
import logging
from logic.gpt_utils import spaced_gpt_call  # Import from the separate GPT utils module

async def adjust_npc_preferences(npc_data, environment_desc, conversation_id):
    """
    Given an NPC's current likes, dislikes, and hobbies, query GPT to generate updated preferences
    that fit the environment. Returns a dictionary with keys: 'likes', 'dislikes', and 'hobbies'.
    """
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    hobbies = npc_data.get("hobbies", [])
    
    prompt = (
        "Given the following NPC preferences:\n"
        "Likes: {likes}\n"
        "Dislikes: {dislikes}\n"
        "Hobbies: {hobbies}\n\n"
        "Update these preferences so that they are more fitting for an environment dominated by powerful females. "
        "Return only a JSON object with the keys 'likes', 'dislikes', and 'hobbies'."
    ).format(likes=likes, dislikes=dislikes, hobbies=hobbies)
    
    logging.info("Adjusting NPC preferences with prompt: %s", prompt)
    reply = await spaced_gpt_call(conversation_id, environment_desc, prompt)
    
    # Extract the text from the reply.
    if reply.get("response"):
        preferences_text = reply["response"].strip()
    elif (reply.get("type") == "function_call" and 
          reply.get("function_args", {}).get("preferences")):
        preferences_text = reply["function_args"]["preferences"].strip()
    else:
        logging.warning("GPT did not return adjusted preferences; falling back to original values.")
        return {"likes": likes, "dislikes": dislikes, "hobbies": hobbies}
    
    # Remove markdown code fences if present.
    if preferences_text.startswith("```"):
        lines = preferences_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        preferences_text = "\n".join(lines).strip()
    
    try:
        updated_preferences = json.loads(preferences_text)
    except Exception as e:
        logging.error("Error parsing updated preferences: %s", e)
        updated_preferences = {"likes": likes, "dislikes": dislikes, "hobbies": hobbies}
    
    return updated_preferences

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
