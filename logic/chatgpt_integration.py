# logic/chatgpt_integration.py
import os
import logging
from openai import OpenAI
from db.connection import get_db_connection
from logic.prompts import SYSTEM_PROMPT, DB_SCHEMA_PROMPT

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")
    return OpenAI(api_key=api_key)

def build_message_history(conversation_id: int, aggregator_text: str, user_input: str, limit: int = 15):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT sender, content
        FROM messages
        WHERE conversation_id=%s
        ORDER BY id DESC
        LIMIT %s
    """, (conversation_id, limit))
    rows = cur.fetchall()
    conn.close()

    rows.reverse()  # Oldest first

    # Convert DB rows => ChatCompletion roles
    chat_history = []
    for sender, content in rows:
        role = "user" if sender.lower() == "user" else "assistant"
        chat_history.append({"role": role, "content": content})

    # Now build the final messages array
    messages = []

    # 1) Your "always enforced" system-level instructions
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "system", "content": DB_SCHEMA_PROMPT})

    # 2) The aggregator text (some dev/system-level context)
    messages.append({"role": "developer", "content": aggregator_text})

    # 3) The prior conversation
    messages.extend(chat_history)

    # 4) The new user message
    messages.append({"role": "user", "content": user_input})

    return messages

def get_chatgpt_response(conversation_id: int, aggregator_text: str, user_input: str) -> dict:
    client = get_openai_client()

    messages = build_message_history(conversation_id, aggregator_text, user_input, limit=15)

    response = client.chat.completions.create(
        model="o3-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
        frequency_penalty=0.0
    )
    response_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    return {"response": response_text, "tokens_used": tokens_used}
