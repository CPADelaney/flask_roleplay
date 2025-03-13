# setting_analyzer.py

import logging
import json
from typing import Dict, Any
from db.connection import get_db_connection

class SettingAnalyzer:
    """
    Analyzes the setting's NPC data to feed into an agentic tool 
    that generates organizations, etc.
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    def aggregate_npc_data(self) -> Dict[str, Any]:
        """
        Collect all NPC data (likes, hobbies, archetypes, affiliations) into a unified format,
        but do NOT call GPT here. Just gather data for the sub-agent.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT npc_id, npc_name, archetypes, likes, dislikes, 
                       hobbies, affiliations, personality_traits, 
                       current_location, archetype_summary
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s
            """, (self.user_id, self.conversation_id))
            
            rows = cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
        
        # Process rows into a structured dict
        all_npcs = []
        all_archetypes, all_likes, all_hobbies, all_affiliations, all_locations = (
            set(), set(), set(), set(), set()
        )

        for row in rows:
            npc_id, npc_name, archetypes_json, likes_json, dislikes_json, \
            hobbies_json, affiliations_json, personality_json, \
            current_location, archetype_summary = row

            # Safely load JSON fields
            def safe_load(s):
                try:
                    return json.loads(s) if s else []
                except:
                    return []

            archetypes = safe_load(archetypes_json)
            likes = safe_load(likes_json)
            dislikes = safe_load(dislikes_json)
            hobbies = safe_load(hobbies_json)
            affiliations = safe_load(affiliations_json)
            personality_traits = safe_load(personality_json)

            # Update sets
            all_archetypes.update(archetypes)
            all_likes.update(likes)
            all_hobbies.update(hobbies)
            all_affiliations.update(affiliations)
            if current_location:
                all_locations.add(current_location)
            
            all_npcs.append({
                "npc_id": npc_id,
                "npc_name": npc_name,
                "archetypes": archetypes,
                "likes": likes,
                "dislikes": dislikes,
                "hobbies": hobbies,
                "affiliations": affiliations,
                "personality_traits": personality_traits,
                "current_location": current_location,
                "archetype_summary": archetype_summary
            })

        # Optionally get the environment desc from your DB
        setting_desc, setting_name = self._get_current_setting_info()

        return {
            "setting_name": setting_name,
            "setting_description": setting_desc,
            "npcs": all_npcs,
            "aggregated": {
                "archetypes": list(all_archetypes),
                "likes": list(all_likes),
                "hobbies": list(all_hobbies),
                "affiliations": list(all_affiliations),
                "locations": list(all_locations),
            }
        }

    def _get_current_setting_info(self):
        """
        Helper to fetch the current setting name and environment desc from the DB.
        """
        setting_desc = "A setting with no description."
        setting_name = "The Setting"
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='EnvironmentDesc'
            """, (self.user_id, self.conversation_id))
            row = cursor.fetchone()
            if row:
                setting_desc = row[0] or setting_desc

            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='CurrentSetting'
            """, (self.user_id, self.conversation_id))
            row = cursor.fetchone()
            if row:
                setting_name = row[0] or setting_name
        finally:
            cursor.close()
            conn.close()

        return setting_desc, setting_name
