# logic/story_flow.py

import openai
from logic.rule_enforcement import enforce_all_rules_on_player
from logic.system_prompt import build_system_prompt
from logic.memory_logic import store_roleplay_segment
from logic.settings_routes import generate_mega_setting  # If you want to pick new settings
# ... import anything else you need ...

def next_storybeat(player_name, user_input):
    """
    Orchestrates the entire flow for generating the next chunk of narrative.
    1) Possibly enforce rules => triggers DB updates
    2) Possibly generate or pick a new environment if needed
    3) Build a 'system prompt' that includes all relevant context from DB
    4) Call GPT with that system prompt + user input
    5) Log the GPT response back into the DB or memory
    6) Return GPT's final text
    """

    # 1) Enforce rules
    triggered = enforce_all_rules_on_player(player_name)
    if triggered:
        # If you like, you can store or log them:
        store_roleplay_segment({
            "key": f"RulesTriggered_{player_name}",
            "value": str(triggered)
        })

    # 2) Possibly generate a new environment if needed
    # e.g. if you track "UsedSettings" and they've exhausted them, do:
    #   generate_mega_setting() or pick from existing
    # For a simple example, we won't always do it, let's show a pseudo-logic:
    # used_settings_count = ...
    # if used_settings_count >= 3:
    #     generate_mega_setting(...) # store in CurrentRoleplay "CurrentSetting"

    # 3) Build the system prompt
    system_prompt = build_system_prompt(player_name)

    # 4) Call GPT (the single GPT call we do each time)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=1.0
    )

    gpt_output = response["choices"][0]["message"]["content"]

    # 5) Optionally store the GPT's text in memory or DB so we can reference later
    store_roleplay_segment({
        "key": f"GPTOutput_{player_name}",
        "value": gpt_output
    })

    # If the GPT text includes instructions that might further update the DB, 
    # you'd parse them here. For now, let's assume we just store the raw text.

    # 6) Return the GPT's final text
    return gpt_output
