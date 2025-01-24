# logic/meltdown_logic.py

import openai
from db.connection import get_db_connection

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
    (Optional) Attempts to write meltdown lines to a .chr file. 
    """
    filename = f"meltdown_npc_{npc_name}.chr"
    text = f"\n--- meltdown message for {npc_name}:\n{meltdown_line}\n"
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"(Could not write meltdown file for {npc_name}: {e})")

def check_and_inject_meltdown() -> str:
    """
    Central meltdown synergy function:
      1) Looks for meltdown NPC(s), highest meltdown_level first.
      2) If meltdown is active, generate meltdown line, store it, return the line.
      3) If no meltdown NPC found, return empty string.
    """
    meltdown_line = ""
    conn = get_db_connection()
    cursor = conn.cursor()

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
            # optionally also call append_meltdown_file(npc_name, meltdown_line)
    finally:
        conn.close()

    return meltdown_line
