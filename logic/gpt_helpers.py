# logic/gpt_helpers.py
import json
import logging
from logic.gpt_utils import spaced_gpt_call  # Import from the separate GPT utils module

async def adjust_npc_complete(npc_data, environment_desc, conversation_id, immersive_days=None):
    """
    Adapt NPC details by combining preferences adjustments and schedule/affiliation generation
    into one GPT call. Tailor all details to an environment dominated by powerful females.

    Expects the following keys to be returned in a JSON object:
      - "likes": an array of strings (adjusted likes)
      - "dislikes": an array of strings (adjusted dislikes)
      - "hobbies": an array of strings (adjusted hobbies)
      - "affiliations": an array of strings (adjusted affiliations)
      - "schedule": an object representing a weekly schedule using the immersive day names,
                    where each day (e.g., "Alpha", "Beta", etc.) has nested keys for "Morning",
                    "Afternoon", "Evening", and "Night".

    Parameters:
      npc_data: A dict containing the NPC's current details (preferences, archetype summary, etc.)
      environment_desc: A string describing the environment.
      conversation_id: The conversation context id for the GPT call.
      immersive_days: A list of immersive day names to use in the schedule. If None, a fallback list is used.
    """
    # Use the actual immersive day names if provided; otherwise, fall back.
    if immersive_days is None:
        immersive_days = ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]

    # Retrieve NPC details.
    archetype_summary = npc_data.get("archetype_summary", "")
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    hobbies = npc_data.get("hobbies", [])

    # Build the combined prompt.
    prompt = (
        "Given the following NPC information:\n"
        "Archetype Summary: {archetype_summary}\n"
        "Environment: {environment_desc}\n\n"
        "NPC Preferences:\n"
        "Likes: {likes}\n"
        "Dislikes: {dislikes}\n"
        "Hobbies: {hobbies}\n\n"
        "Instructions:\n"
        "1. Adapt the above preferences so that they are specifically tailored to an environment dominated by powerful females.\n"
        "2. Determine the NPC's affiliations (teams, clubs, partnerships, associations, etc.) that fit within this environment.\n"
        "3. Create a detailed weekly schedule. The schedule must use the following immersive day names: {immersive_days}. "
        "For each day, include nested keys 'Morning', 'Afternoon', 'Evening', and 'Night' representing the NPC's activities.\n\n"
        "Return only a JSON object with exactly the following five keys (and no additional keys or text):\n"
        "  - \"likes\": an array of strings representing the adjusted likes,\n"
        "  - \"dislikes\": an array of strings representing the adjusted dislikes,\n"
        "  - \"hobbies\": an array of strings representing the adjusted hobbies,\n"
        "  - \"affiliations\": an array of strings representing the adjusted affiliations,\n"
        "  - \"schedule\": an object representing the adjusted weekly schedule using the immersive day names.\n"
    ).format(
        archetype_summary=archetype_summary,
        environment_desc=environment_desc,
        likes=likes,
        dislikes=dislikes,
        hobbies=hobbies,
        immersive_days=", ".join(immersive_days)
    )

    logging.info("Adjusting complete NPC details with prompt: %s", prompt)

    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        reply = await spaced_gpt_call(conversation_id, environment_desc, prompt)
        
        # If GPT returns a function call style response.
        if reply.get("type") == "function_call":
            args = reply.get("function_args", {})
            if all(key in args for key in ["likes", "dislikes", "hobbies", "affiliations", "schedule"]):
                return {
                    "likes": args["likes"],
                    "dislikes": args["dislikes"],
                    "hobbies": args["hobbies"],
                    "affiliations": args["affiliations"],
                    "schedule": args["schedule"]
                }
            else:
                logging.warning("Incomplete keys in function_args; retrying.")
                retry_count += 1
                continue
        
        # Otherwise, try parsing a plain text response.
        elif reply.get("response"):
            try:
                response_text = reply["response"].strip()
                # Remove markdown code fences if present.
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
        else:
            logging.warning("GPT did not return a valid response; retrying.")
            retry_count += 1

    logging.error("Max retries reached; returning original values.")
    return {
        "likes": likes,
        "dislikes": dislikes,
        "hobbies": hobbies,
        "affiliations": npc_data.get("affiliations", []),
        "schedule": npc_data.get("schedule", {})
    }
