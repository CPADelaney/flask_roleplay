# logic/chatgpt_integration.py
import os
from openai import OpenAI
client = OpenAI()

from logic.prompts import SYSTEM_PROMPT, DB_SCHEMA_PROMPT

client.api_key = os.getenv("OPENAI_API_KEY")

def get_chatgpt_response(user_input: str) -> str:
    """
    Calls the ChatCompletion endpoint with 'gpt-4o' and the roles
    'developer'/'user' from the doc snippet.
    """
    messages=[
        # 1) The 'developer' role for your system-level instructions
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",  # Ensure correct model name is used
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
        frequency_penalty=0.0
    )
    response_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens
    
    return {"response": response_text, "tokens_used": tokens_used}
