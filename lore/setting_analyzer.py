# lore/setting_analyzer.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Set

from db.connection import get_db_connection
from logic.chatgpt_integration import get_chatgpt_response

class SettingAnalyzer:
    """
    Analyzes the setting and NPCs to determine what organizations, factions, 
    and cultural elements would naturally exist in this world.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    async def aggregate_npc_data(self) -> Dict[str, Any]:
        """
        Collect all NPC data (likes, hobbies, archetypes, affiliations) into a unified format.
        
        Returns:
            Dict containing aggregated NPC data
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all NPCs for this conversation
            cursor.execute("""
                SELECT npc_id, npc_name, archetypes, likes, dislikes, 
                       hobbies, affiliations, personality_traits, 
                       current_location, archetype_summary
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s
            """, (self.user_id, self.conversation_id))
            
            all_npcs = []
            
            # Collect lists for aggregation
            all_archetypes = set()
            all_likes = set()
            all_hobbies = set()
            all_affiliations = set()
            all_locations = set()
            
            for row in cursor.fetchall():
                # Process JSON fields
                try:
                    archetypes = json.loads(row[2]) if row[2] else []
                except:
                    archetypes = []
                
                try:
                    likes = json.loads(row[3]) if row[3] else []
                except:
                    likes = []
                
                try:
                    dislikes = json.loads(row[4]) if row[4] else []
                except:
                    dislikes = []
                
                try:
                    hobbies = json.loads(row[5]) if row[5] else []
                except:
                    hobbies = []
                
                try:
                    affiliations = json.loads(row[6]) if row[6] else []
                except:
                    affiliations = []
                
                try:
                    traits = json.loads(row[7]) if row[7] else []
                except:
                    traits = []
                
                # Add to aggregated sets
                all_archetypes.update(archetypes)
                all_likes.update(likes)
                all_hobbies.update(hobbies)
                all_affiliations.update(affiliations)
                
                location = row[8]
                if location:
                    all_locations.add(location)
                
                # Store NPC data
                all_npcs.append({
                    "npc_id": row[0],
                    "npc_name": row[1],
                    "archetypes": archetypes,
                    "likes": likes,
                    "dislikes": dislikes,
                    "hobbies": hobbies,
                    "affiliations": affiliations,
                    "personality_traits": traits,
                    "current_location": location,
                    "archetype_summary": row[9]
                })
            
            # Get current setting description
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='EnvironmentDesc'
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            setting_desc = row[0] if row else "A setting with no description."
            
            # Current setting name
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='CurrentSetting'
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            setting_name = row[0] if row else "The Setting"
            
            return {
                "setting_name": setting_name,
                "setting_description": setting_desc,
                "npcs": all_npcs,
                "aggregated": {
                    "archetypes": list(all_archetypes),
                    "likes": list(all_likes),
                    "hobbies": list(all_hobbies),
                    "affiliations": list(all_affiliations),
                    "locations": list(all_locations)
                }
            }
            
        finally:
            cursor.close()
            conn.close()
    
    async def analyze_setting_for_organizations(self) -> Dict[str, Any]:
        """
        Analyze the setting and NPC data to determine what organizations/factions would exist.
        
        Returns:
            Dict containing organization categories and their members
        """
        # Get aggregated NPC data
        data = await self.aggregate_npc_data()
        
        # Format a prompt for GPT
        archetypes_text = ", ".join(data["aggregated"]["archetypes"])
        hobbies_text = ", ".join(data["aggregated"]["hobbies"])
        likes_text = ", ".join(data["aggregated"]["likes"])
        existing_affiliations = ", ".join(data["aggregated"]["affiliations"]) if data["aggregated"]["affiliations"] else "None explicitly mentioned"
        locations_text = ", ".join(data["aggregated"]["locations"])
        
        prompt = f"""
        Analyze this setting and the NPCs living within it to determine what organizations, factions, and groups would naturally exist.
        
        Setting: {data["setting_name"]}
        
        Setting Description: {data["setting_description"]}
        
        NPC Information:
        - Archetypes: {archetypes_text}
        - Hobbies: {hobbies_text}
        - Likes/Interests: {likes_text}
        - Existing Affiliations: {existing_affiliations}
        - Locations: {locations_text}
        
        Based on this information, create a comprehensive and consistent set of organizations that would exist in this setting, organized by category.
        Include organizations that are directly implied by NPC archetypes and hobbies, as well as logical extensions based on the setting.
        
        For example, if NPCs include "students" or "teachers", there should be an educational institution. 
        If hobbies include "swimming" or "track", there should be relevant athletic teams/departments.
        
        Return a JSON object with these keys:
        
        1. "academic": Array of academic institutions, departments, and educational groups
        2. "athletic": Array of sports teams, fitness groups, and physical activity organizations
        3. "social": Array of social clubs, friend groups, and informal gatherings
        4. "professional": Array of businesses, workplaces, and professional organizations
        5. "cultural": Array of cultural, artistic, and heritage-focused groups
        6. "political": Array of political, government, and power structures
        7. "other": Array of any other relevant organizations
        
        For each organization, include:
        - "name": Official name of the organization
        - "type": Specific type (e.g., "sports team", "student club")
        - "description": Brief description of its purpose and activities
        - "membership_basis": What qualifies someone to be a member
        - "hierarchy": Brief description of any leadership structure
        - "gathering_location": Where members typically meet (should relate to existing locations if possible)
        
        Ensure organizations are named consistently and avoid duplicates or closely overlapping purposes.
        """
        
        # Make the API call
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are an expert social anthropologist who specializes in analyzing communities and their organizational structures.",
            user_prompt=prompt
        )
        
        # Extract the JSON
        if isinstance(response, dict) and "function_args" in response:
            return response["function_args"]
        elif isinstance(response, dict) and "response" in response:
            # Try to extract JSON from the text response
            try:
                import re
                text = response["response"]
                
                # Look for JSON block
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                if match:
                    return json.loads(match.group(1))
                
                # Try finding the first { and last }
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
                
                logging.warning("Could not extract JSON from response")
                return {}
            except Exception as e:
                logging.error(f"Error extracting JSON from response: {e}")
                return {}
        else:
            logging.error("Unexpected response format")
            return {}
