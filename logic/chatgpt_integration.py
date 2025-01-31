# logic/chatgpt_integration.py
import os
from openai import OpenAI
from logic.prompts import SYSTEM_PROMPT, DB_SCHEMA_PROMPT

def get_openai_client():
    """
    Create and return a new OpenAI client, using the OPENAI_API_KEY from env.
    Raises an exception if the key isn't set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")
    
    # You can pass the key directly into the constructor:
    client = OpenAI(api_key=api_key)
    return client

def get_chatgpt_response(user_input: str) -> dict:
    """
    Calls the ChatCompletion endpoint with 'gpt-4o' and the roles
    'developer'/'user' from the doc snippet.
    Returns a dictionary with {"response": "...", "tokens_used": X}
    """
    # Acquire a fresh client each time (or cache it if you prefer)
    client = get_openai_client()

    messages = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",  # or whichever model name you need
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
        frequency_penalty=0.0
    )
    response_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens
    
    return {"response": response_text, "tokens_used": tokens_used}
