# logic/story_flow.py

import openai
from logic.rule_enforcement import enforce_all_rules_on_player
from logic.system_prompt import build_system_prompt
from logic.memory_logic import store_roleplay_segment
from routes.settings_routes import generate_mega_setting  # if you want to call it here

def next_storybeat(player_name, user_input):
    """
    1) Enforce rules => triggers DB updates
    2) Possibly generate a new setting if needed
    3) Build system prompt with all relevant context
    4) Single GPT call
    5) Store GPT output if desired
    6) Return the GPT text
    """

    # 1) Enforce rules
    triggered = enforce_all_rules_on_player(player_name)
    # (Optional) store the triggered rules for reference
    if triggered:
        store_roleplay_segment({
            "key": f"TriggeredRules_{player_name}",
            "value": str(triggered)
        })

    # 2) Possibly generate a new setting if needed
    # For example, if we detect we've used X settings, or we want a new environment each time:
    # new_env = generate_mega_setting() # if we want to pick a new environment now
    # store_roleplay_segment({"key": "CurrentSetting", "value": new_env["mega_name"]})

    # 3) Build the system prompt from the DB
    system_prompt = build_system_prompt(player_name)

    # 4) GPT call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=1.0
    )
    gpt_output = response["choices"][0]["message"]["content"]

    # 5) Log GPT output if we want
    store_roleplay_segment({
        "key": f"GPTOutput_{player_name}",
        "value": gpt_output
    })

    # 6) Return final text
    return gpt_output
