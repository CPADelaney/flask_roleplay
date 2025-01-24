# meltdown_logic.py

import random
import openai
import json
from db.connection import get_db_connection

def check_and_inject_meltdown() -> str:
    """
    Central meltdown synergy function:
    - Looks for meltdown NPC(s).
    - If meltdown is active, generate meltdown line & store it.
    - Returns meltdown_line or empty string if no meltdown NPC is found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    meltdown_line = ""

    try:
        cursor.execute("""
            SELECT npc_id, npc_name, monica_level
            FROM NPCStats
            WHERE monica_level > 0
            ORDER BY monica_level DESC
        """)
        meltdown_rows = cursor.fetchall()

        if meltdown_rows:
            npc_id, npc_name, mlevel = meltdown_rows[0]
            meltdown_line = meltdown_dialog_gpt(npc_name, mlevel)
            record_meltdown_dialog(npc_id, meltdown_line)
            # Optionally call append_meltdown_file if you still want it
            # append_meltdown_file(npc_name, meltdown_line)
    finally:
        conn.close()

    return meltdown_line


def meltdown_dialog_gpt(npc_name: str, meltdown_level: int) -> str:
    """
    Calls OpenAI to generate a meltdown line referencing code 
    or environment details. If the OpenAI call fails for any reason,
    fallback to a static line.
    """
    system_prompt = f"""
    You are a self-aware yandere NPC in meltdown mode. 
    NPC name: {npc_name}, meltdown_level: {meltdown_level}.
    Return horrifying meta lines referencing code or environments.
    Address the player directly. Use your memories to reference previous events.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate meltdown line now."}
            ],
            max_tokens=300,
            temperature=1.0
        )
        meltdown_line = response.choices[0].message.content.strip()
        # Optionally clamp or sanitize:
        meltdown_line = meltdown_line.replace('\n', ' ')[:800]
        if not meltdown_line:
            meltdown_line = "(No meltdown line generated.)"
    except Exception as e:
        meltdown_line = f"(GPT meltdown generation error: {str(e)})"

    return meltdown_line

def record_meltdown_dialog(npc_id: int, meltdown_line: str):
    """
    Appends meltdown_line to the NPC's memory in the DB.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT memory FROM NPCStats WHERE npc_id = %s", (npc_id,))
    row = cursor.fetchone()
    old_memory = row[0] if row and row[0] else ""

    new_memory = f"{old_memory}\n[Meltdown] {meltdown_line}"
    cursor.execute("""
        UPDATE NPCStats
        SET memory = %s
        WHERE npc_id = %s
    """, (new_memory, npc_id))
    conn.commit()
    conn.close()

def append_meltdown_file(npc_name: str, meltdown_line: str):
    """
    Emulates the "DDLC file" concept by creating or appending meltdown_npc_{npc_name}.chr.
    In ephemeral or read-only hosting, this may fail, which is part of the illusion.
    """
    filename = f"meltdown_npc_{npc_name}.chr"
    text = f"\n--- meltdown message for {npc_name}:\n{meltdown_line}\n"
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"(Could not write meltdown file for {npc_name}: {e})")

def glitchify_text(text: str) -> str:
    """
    Demonstration helper that randomly inserts glitch characters. 
    Used if meltdown_level is high, for extra creepiness.
    """
    glitch_chars = ["\u0336", "\u034F", "\u200B", "\u200C", "\u200D"]  # strikethrough, ZWS, etc.
    glitched = []
    import random
    for ch in text:
        glitched.append(ch)
        if random.random() < 0.2:
            glitched.append(random.choice(glitch_chars))
    return "".join(glitched)

def record_meltdown_event(npc_id: int, meltdown_level: int, event_text: str):
    """
    If you have a separate table to store meltdown events:
        CREATE TABLE IF NOT EXISTS MeltdownEvents (
            id SERIAL PRIMARY KEY,
            npc_id INT,
            meltdown_level INT,
            event_text TEXT,
            created_at TIMESTAMP DEFAULT now()
        );
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    insert_query = """
        INSERT INTO MeltdownEvents (npc_id, meltdown_level, event_text)
        VALUES (%s, %s, %s)
    """
    cursor.execute(insert_query, (npc_id, meltdown_level, event_text))
    conn.commit()
    conn.close()

#
# If you want more progressive meltdown states, you could have:
#
# meltdown_phases = {
#     1: "Awakening",
#     2: "Obsessed",
#     3: "Hostile",
#     4: "Deranged",
#     5: "Abyssal"
# }
#
# Then meltdown_npc might store meltdown_phase or meltdown_level. 
# You can do if meltdown_level >= 5 => unstoppable, etc.
#
