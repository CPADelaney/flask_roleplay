# logic/chatgpt_integration.py
import os
from openai import OpenAI

from logic.prompts import SYSTEM_PROMPT, DB_SCHEMA_PROMPT

openai.api_key = os.getenv("GPTAPI")

def get_chatgpt_response(user_input: str) -> str:
    """
    Example of calling ChatCompletion endpoint with 'gpt-4o',
    a 'developer' role, and a 'user' role as described in the doc snippet.
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            # Notice the doc said something like store: true. 
            # In the Python library, that might appear as a top-level param if your version supports it:
            store=True,  # <--- if your new environment actually supports store=...
            messages=[
                
                # 1) Primary system prompt
                {"role": "developer", "content": SYSTEM_PROMPT},
                
                # 2) Extra system prompt or 'developer' context: the DB schema
                {"role": "developer", "content": DB_SCHEMA_PROMPT},
        
                # 3) The user message
                {"role": "user", "content": user_input}
                }
            ]
        )
        # The doc suggests the result is in `.choices[0].message`
        return completion.choices[0].message["content"]

    except Exception as e:
        print(f"Error calling GPT-4o: {e}")
        return f"Error: Unable to fetch response from GPT-4o. Reason: {e}"
