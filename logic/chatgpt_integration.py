# logic/chatgpt_integration.py

import os
import openai

# Set your API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chatgpt_response(messages, model="gpt-3.5-turbo"):
    """
    Sends a list of messages to the OpenAI ChatCompletion endpoint
    and returns the content of the first choice's message.
    
    messages: A list of dicts, e.g.:
        [
          {"role": "system", "content": "System prompt/instructions"},
          {"role": "user",   "content": "User's message"}
        ]
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,  # or any parameter you want
        )
        # Return the assistant's message content
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Oops! Something went wrong with the AI. (Debug info logged.)"
