# logic/chatgpt_integration.py
import os
import openai

from logic.prompts import SYSTEM_PROMPT, DB_SCHEMA_PROMPT

openai.api_key = os.getenv("GPTAPI")

def get_chatgpt_response(user_input: str, model="chatgpt-4o-latest") -> str:
    """
    Sends a list of messages to the OpenAI ChatCompletion endpoint
    and returns the content of the first choice's message.
    """

    messages = [
        # 1) Primary system prompt
        {"role": "system", "content": SYSTEM_PROMPT},
        
        # 2) Extra system prompt or 'developer' context: the DB schema
      #  {"role": "system", "content": DB_SCHEMA_PROMPT},

        # 3) The user message
        {"role": "user", "content": user_input}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error: Unable to fetch response from ChatGPT."
