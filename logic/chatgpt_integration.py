# logic/chatgpt_integration.py
import os
from openai import OpenAI
client = OpenAI()

from logic.prompts import SYSTEM_PROMPT, DB_SCHEMA_PROMPT

client.api_key = os.getenv("OPENAI_API_KEY")

def get_chatgpt_response(user_input: str, model="gpt-4o") -> str:
    """
    Calls the ChatCompletion endpoint with 'gpt-4o' and the roles
    'developer'/'user' from the doc snippet.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            # If your environment supports store=, keep it; else remove it:
            store=True,
            messages=[
                # 1) The 'developer' role for your system-level instructions
                {"role": "developer", "content": SYSTEM_PROMPT},

                # 2) Another 'developer' message for the DB schema, if you like
                {"role": "developer", "content": DB_SCHEMA_PROMPT},

                # 3) The user's actual prompt
                {"role": "user", "content": user_input}
            ]
        )
        # The doc indicates the text is in completion.choices[0].message["content"]
        return completion.choices[0].message["content"]

    except Exception as e:
        print(f"Error calling GPT-4o: {e}")
        return f"Error: Unable to fetch response from GPT-4o. Reason: {e}"
