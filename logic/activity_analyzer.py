# logic/activity_analyzer.py

import logging
import json
import re
from db.connection import get_db_connection
from logic.chatgpt_integration import get_chatgpt_response
from logic.resource_management import ResourceManager

class ActivityAnalyzer:
    """
    Analyzes player activities using GPT to determine dynamic resource effects.
    Stores and retrieves activity effects from the ActivityEffects table.
    """
    
    def __init__(self, user_id, conversation_id):
        """Initialize the activity analyzer."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.resource_manager = ResourceManager(user_id, conversation_id)
    
    async def analyze_activity(self, activity_text, setting_context=None, apply_effects=True):
        """
        Analyze an activity to determine its resource effects.
        
        Args:
            activity_text: The text description of the activity
            setting_context: Optional context about the current setting
            apply_effects: Whether to immediately apply the determined effects
            
        Returns:
            Dict with activity analysis and effects
        """
        # Extract activity type and details
        activity_type, activity_details = self._extract_activity_info(activity_text)
        
        # Check if we already have effects for this activity
        existing_effects = await self._get_existing_effects(activity_type, activity_details)
        if existing_effects:
            logging.info(f"Found existing effects for {activity_type}: {activity_details}")
            if apply_effects:
                await self._apply_resource_effects(existing_effects["effects"])
            return existing_effects
        
        # If not found, get context for better GPT analysis
        if not setting_context:
            setting_context = await self._get_setting_context()
        
        # Use GPT to analyze the activity
        effects = await self._generate_activity_effects(activity_text, activity_type, activity_details, setting_context)
        
        # Store the effects in the database
        await self._store_activity_effects(activity_type, activity_details, setting_context, effects)
        
        # Apply the effects if requested
        if apply_effects:
            await self._apply_resource_effects(effects["resource_changes"])
        
        return {
            "activity_type": activity_type,
            "activity_details": activity_details,
            "effects": effects["resource_changes"],
            "description": effects["description"],
            "flags": effects.get("flags", {})
        }
    
    def _extract_activity_info(self, activity_text):
        """
        Extract the activity type and details from the activity text.
        
        Example:
        "eating a hamburger" -> ("eating", "hamburger")
        "drinking coffee at the cafe" -> ("drinking", "coffee at the cafe")
        """
        # Common activity types to look for
        activity_types = [
            "eating", "drinking", "working", "studying", "relaxing", "exercising",
            "shopping", "talking", "walking", "reading", "writing", "sleeping",
            "cleaning", "cooking", "meeting", "playing", "watching", "listening"
        ]
        
        activity_text = activity_text.lower()
        
        # Try to match common activity types
        for activity_type in activity_types:
            if activity_text.startswith(activity_type):
                # Extract the details after the activity type
                details = activity_text[len(activity_type):].strip()
                # Remove common prepositions at the beginning
                details = re.sub(r"^(a|an|the|some|my|to|at|in|on|with)\s+", "", details)
                return activity_type, details
        
        # If no match, make a best guess
        words = activity_text.split()
        if words:
            # Use the first word as the activity type
            activity_type = words[0]
            # And the rest as details
            details = " ".join(words[1:]).strip()
            # Remove common prepositions at the beginning
            details = re.sub(r"^(a|an|the|some|my|to|at|in|on|with)\s+", "", details)
            return activity_type, details
        
        # Fallback
        return "unknown", activity_text
    
    async def _get_existing_effects(self, activity_type, activity_details):
        """Get existing effects for this activity from the database."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Try to find an exact match first
            cursor.execute("""
                SELECT effects, description, flags
                FROM ActivityEffects
                WHERE user_id=%s AND conversation_id=%s
                  AND activity_name=%s AND activity_details=%s
                LIMIT 1
            """, (self.user_id, self.conversation_id, activity_type, activity_details))
            
            row = cursor.fetchone()
            
            if row:
                effects_json, description, flags_json = row
                
                try:
                    effects = json.loads(effects_json) if isinstance(effects_json, str) else effects_json
                    flags = json.loads(flags_json) if isinstance(flags_json, str) and flags_json else {}
                except json.JSONDecodeError:
                    effects = {}
                    flags = {}
                
                return {
                    "activity_type": activity_type,
                    "activity_details": activity_details,
                    "effects": effects,
                    "description": description,
                    "flags": flags
                }
            
            # If no exact match, try to find a partial match
            # For example, if we're looking for "hamburger with fries",
            # we might find "hamburger" as a partial match
            cursor.execute("""
                SELECT activity_details, effects, description, flags
                FROM ActivityEffects
                WHERE user_id=%s AND conversation_id=%s
                  AND activity_name=%s 
                  AND %s LIKE CONCAT('%%', activity_details, '%%')
                ORDER BY LENGTH(activity_details) DESC
                LIMIT 1
            """, (self.user_id, self.conversation_id, activity_type, activity_details))
            
            row = cursor.fetchone()
            
            if row:
                details, effects_json, description, flags_json = row
                
                try:
                    effects = json.loads(effects_json) if isinstance(effects_json, str) else effects_json
                    flags = json.loads(flags_json) if isinstance(flags_json, str) and flags_json else {}
                except json.JSONDecodeError:
                    effects = {}
                    flags = {}
                
                return {
                    "activity_type": activity_type,
                    "activity_details": details,  # Use the matching details
                    "effects": effects,
                    "description": description,
                    "flags": flags
                }
            
            return None
            
        finally:
            cursor.close()
            conn.close()
    
    async def _get_setting_context(self):
        """Get the current setting context for better activity analysis."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT value 
                FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='EnvironmentDesc'
                LIMIT 1
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if row:
                return row[0]
            
            # Fallback
            return "A modern setting with typical resources and activities."
            
        finally:
            cursor.close()
            conn.close()
    
    async def _generate_activity_effects(self, activity_text, activity_type, activity_details, setting_context):
        """
        Use GPT to analyze the activity and determine its resource effects.
        """
        prompt = f"""
        As a game system, analyze this player activity: "{activity_text}"
        
        Setting Context: {setting_context}
        
        Determine the realistic resource effects of this activity. Consider the following:
        1. Effects on hunger (0-100 scale, higher = less hungry)
        2. Effects on energy (0-100 scale, higher = more energy)
        3. Money costs or gains
        4. Supplies used or gained
        5. Influence changes
        
        For example:
        - Eating a burger might cost 7 money, increase hunger by 25, and change energy by -5 (food coma) or +5 (energizing)
        - Drinking coffee might cost 3 money, increase hunger by only 5, and increase energy by 15
        - Working might earn 15 money, decrease energy by 20, and decrease hunger by 10
        
        Provide your analysis in JSON format with these exact keys:
        - resource_changes: object with numeric values for "hunger", "energy", "money", "supplies", "influence" (only include if relevant)
        - description: brief description of the effects
        - flags: object with any special conditions (like "temporary", "stacking", "diminishing_returns")
        
        Include only realistic, reasonable values relative to the setting.
        """
        
        try:
            gpt_response = await get_chatgpt_response(
                self.conversation_id,
                setting_context,
                prompt
            )
            
            # Extract JSON data
            if gpt_response.get("type") == "function_call":
                effects = gpt_response.get("function_args", {})
                # Make sure we have the expected keys
                if "resource_changes" not in effects:
                    effects["resource_changes"] = {}
                if "description" not in effects:
                    effects["description"] = f"Effects of {activity_text}"
                return effects
            else:
                response_text = gpt_response.get("response", "")
                # Try to extract JSON from the response
                json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
                if json_match:
                    try:
                        effects = json.loads(json_match.group(1))
                        if "resource_changes" not in effects:
                            effects["resource_changes"] = {}
                        if "description" not in effects:
                            effects["description"] = f"Effects of {activity_text}"
                        return effects
                    except json.JSONDecodeError:
                        pass
                
                # If no JSON found, extract key information using regex
                resource_changes = {}
                
                # Look for hunger effects
                hunger_match = re.search(r'hunger\s*(?:by|:)\s*([+-]?\d+)', response_text)
                if hunger_match:
                    resource_changes["hunger"] = int(hunger_match.group(1))
                
                # Look for energy effects
                energy_match = re.search(r'energy\s*(?:by|:)\s*([+-]?\d+)', response_text)
                if energy_match:
                    resource_changes["energy"] = int(energy_match.group(1))
                
                # Look for money effects
                money_match = re.search(r'money\s*(?:by|:)\s*([+-]?\d+)', response_text)
                if money_match:
                    resource_changes["money"] = int(money_match.group(1))
                
                # Look for supplies effects
                supplies_match = re.search(r'supplies\s*(?:by|:)\s*([+-]?\d+)', response_text)
                if supplies_match:
                    resource_changes["supplies"] = int(supplies_match.group(1))
                
                # Look for influence effects
                influence_match = re.search(r'influence\s*(?:by|:)\s*([+-]?\d+)', response_text)
                if influence_match:
                    resource_changes["influence"] = int(influence_match.group(1))
                
                # Default fallback effects if we couldn't extract anything
                if not resource_changes:
                    # Default effects based on activity type
                    if activity_type == "eating":
                        resource_changes = {"hunger": 20, "energy": 5, "money": -5}
                    elif activity_type == "drinking":
                        resource_changes = {"hunger": 5, "energy": 10, "money": -3}
                    elif activity_type == "working":
                        resource_changes = {"hunger": -10, "energy": -20, "money": 15}
                    elif activity_type == "studying":
                        resource_changes = {"hunger": -5, "energy": -15, "influence": 2}
                    elif activity_type == "relaxing":
                        resource_changes = {"energy": 15, "hunger": -5}
                    elif activity_type == "exercising":
                        resource_changes = {"energy": -25, "hunger": -15, "supplies": -1}
                    else:
                        resource_changes = {"energy": -5, "hunger": -3}
                
                # Create final effects object
                return {
                    "resource_changes": resource_changes,
                    "description": f"Effects of {activity_text}",
                    "flags": {}
                }
                
        except Exception as e:
            logging.error(f"Error generating activity effects: {e}")
            # Default fallback effects
            return {
                "resource_changes": {"energy": -5, "hunger": -3},
                "description": f"Default effects for {activity_text}",
                "flags": {}
            }
    
    async def _store_activity_effects(self, activity_type, activity_details, setting_context, effects):
        """Store the activity effects in the database for future reference."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Extract the components we need
            resource_changes = effects.get("resource_changes", {})
            description = effects.get("description", f"Effects of {activity_type} {activity_details}")
            flags = effects.get("flags", {})
            
            cursor.execute("""
                INSERT INTO ActivityEffects 
                (user_id, conversation_id, activity_name, activity_details, 
                 setting_context, effects, description, flags)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, conversation_id, activity_name, activity_details)
                DO UPDATE SET 
                    effects = EXCLUDED.effects,
                    description = EXCLUDED.description,
                    flags = EXCLUDED.flags,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                self.user_id, self.conversation_id, activity_type, activity_details,
                setting_context, json.dumps(resource_changes), description, json.dumps(flags)
            ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error storing activity effects: {e}")
        finally:
            cursor.close()
            conn.close()
    
    async def _apply_resource_effects(self, effects):
        """
        Apply the resource effects to the player.
        
        Args:
            effects: Dict with resource changes
        
        Returns:
            Dict with results for each resource change
        """
        results = {}
        
        # Apply hunger changes
        if "hunger" in effects:
            results["hunger"] = await self.resource_manager.modify_hunger(
                effects["hunger"], "activity", "Activity effect"
            )
        
        # Apply energy changes
        if "energy" in effects:
            results["energy"] = await self.resource_manager.modify_energy(
                effects["energy"], "activity", "Activity effect"
            )
        
        # Apply money changes
        if "money" in effects:
            results["money"] = await self.resource_manager.modify_money(
                effects["money"], "activity", "Activity expense/income"
            )
        
        # Apply supplies changes
        if "supplies" in effects:
            results["supplies"] = await self.resource_manager.modify_supplies(
                effects["supplies"], "activity", "Activity supplies"
            )
        
        # Apply influence changes
        if "influence" in effects:
            results["influence"] = await self.resource_manager.modify_influence(
                effects["influence"], "activity", "Activity influence"
            )
        
        return results
