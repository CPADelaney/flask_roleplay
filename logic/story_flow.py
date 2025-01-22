from logic.rule_enforcement import enforce_all_rules_on_player
from logic.system_prompt import build_system_prompt
from logic.memory_logic import store_roleplay_segment
import openai

def next_storybeat(player_name, user_input):
    # 1) enforce rules => DB updates
    triggered = enforce_all_rules_on_player(player_name)
    # (optional) store them in memory, or do any new_game logic

    # 2) build system prompt
    system_prompt = build_system_prompt(player_name)

    # 3) call GPT
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": user_input}
        ],
        temperature=1.0
    )
    gpt_output = response["choices"][0]["message"]["content"]

    # 4) log GPT's text if desired
    store_roleplay_segment({
        "key": f"GPTOutput_{player_name}",
        "value": gpt_output
    })

    return gpt_output
