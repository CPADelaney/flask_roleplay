# logic/meltdown_logic.py

import random
import os
from db.connection import get_db_connection

def meltdown_dialog(npc_name, monica_level):
    # Return random meltdown line
    ...

def record_meltdown_dialog(npc_id, meltdown_line):
    # Append meltdown text to NPC memory
    ...

def append_meltdown_file(npc_name, meltdown_line):
    # Write meltdown_npc_{npc_name}.chr
    ...
