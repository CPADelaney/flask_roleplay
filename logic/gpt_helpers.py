import json
import logging
from logic.gpt_utils import spaced_gpt_call  # Import the GPT call helper from our new module

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
    # Then check if it's a function call with a specific key.
    elif reply.get("type") == "function_call":
        args = reply.get("function_args", {})
        if "affiliations_schedule" in args:
            affiliations_schedule_text = args["affiliations_schedule"].strip()
        elif "npc_updates" in args:
            # Assume that GPT returned a list of npc_updates.
            updates = args.get("npc_updates")
            if isinstance(updates, list) and len(updates) > 0:
                # Optionally, you could try to filter by npc_id if your npc_data has an id.
                first_update = updates[0]
                # Construct a JSON object from the update.
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
