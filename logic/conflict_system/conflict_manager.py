# logic/conflict_system/conflict_manager.py

import logging
import json
import random
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from db.connection import get_db_connection
from logic.chatgpt_integration import get_chatgpt_response
from logic.resource_management import ResourceManager

logger = logging.getLogger(__name__)

class ConflictManager:
    """
    Manages the creation, progression, and resolution of conflicts.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the conflict manager with user and conversation IDs."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.resource_manager = ResourceManager(user_id, conversation_id)
    
    async def get_active_conflicts(self) -> List[Dict[str, Any]]:
        """
        Get all active conflicts for the current user and conversation.
        
        Returns:
            List of active conflict dictionaries
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT conflict_id, conflict_name, conflict_type, description, 
                       brewing_description, active_description, climax_description,
                       resolution_description, progress, phase, start_day,
                       estimated_duration, faction_a_name, faction_b_name,
                       resources_required, success_rate, outcome
                FROM Conflicts
                WHERE user_id = %s AND conversation_id = %s AND is_active = TRUE
                ORDER BY conflict_id DESC
            """, (self.user_id, self.conversation_id))
            
            conflicts = []
            for row in cursor.fetchall():
                conflict_id, conflict_name, conflict_type, description, brewing_desc, active_desc, climax_desc, \
                resolution_desc, progress, phase, start_day, estimated_duration, faction_a_name, faction_b_name, \
                resources_required, success_rate, outcome = row
                
                # Parse resources_required JSON
                try:
                    if isinstance(resources_required, str):
                        resources_required = json.loads(resources_required)
                except (json.JSONDecodeError, TypeError):
                    resources_required = {"money": 0, "supplies": 0, "influence": 0}
                
                # Create conflict dictionary
                conflict = {
                    "conflict_id": conflict_id,
                    "conflict_name": conflict_name,
                    "conflict_type": conflict_type,
                    "description": description,
                    "brewing_description": brewing_desc,
                    "active_description": active_desc,
                    "climax_description": climax_desc,
                    "resolution_description": resolution_desc,
                    "progress": progress,
                    "phase": phase,
                    "start_day": start_day,
                    "estimated_duration": estimated_duration,
                    "faction_a_name": faction_a_name,
                    "faction_b_name": faction_b_name,
                    "resources_required": resources_required,
                    "success_rate": success_rate,
                    "outcome": outcome
                }
                
                conflicts.append(conflict)
            
            return conflicts
        except Exception as e:
            logger.error(f"Error getting active conflicts: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def get_conflict(self, conflict_id: int) -> Dict[str, Any]:
        """
        Get a specific conflict by ID.
        
        Args:
            conflict_id: ID of the conflict to retrieve
            
        Returns:
            Conflict dictionary if found, empty dict otherwise
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT conflict_id, conflict_name, conflict_type, description, 
                       brewing_description, active_description, climax_description,
                       resolution_description, progress, phase, start_day,
                       estimated_duration, faction_a_name, faction_b_name,
                       resources_required, success_rate, outcome
                FROM Conflicts
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
            """, (conflict_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if not row:
                return {}
            
            conflict_id, conflict_name, conflict_type, description, brewing_desc, active_desc, climax_desc, \
            resolution_desc, progress, phase, start_day, estimated_duration, faction_a_name, faction_b_name, \
            resources_required, success_rate, outcome = row
            
            # Parse resources_required JSON
            try:
                if isinstance(resources_required, str):
                    resources_required = json.loads(resources_required)
            except (json.JSONDecodeError, TypeError):
                resources_required = {"money": 0, "supplies": 0, "influence": 0}
            
            # Create conflict dictionary
            conflict = {
                "conflict_id": conflict_id,
                "conflict_name": conflict_name,
                "conflict_type": conflict_type,
                "description": description,
                "brewing_description": brewing_desc,
                "active_description": active_desc,
                "climax_description": climax_desc,
                "resolution_description": resolution_desc,
                "progress": progress,
                "phase": phase,
                "start_day": start_day,
                "estimated_duration": estimated_duration,
                "faction_a_name": faction_a_name,
                "faction_b_name": faction_b_name,
                "resources_required": resources_required,
                "success_rate": success_rate,
                "outcome": outcome
            }
            
            return conflict
        except Exception as e:
            logger.error(f"Error getting conflict {conflict_id}: {e}", exc_info=True)
            return {}
        finally:
            cursor.close()
            conn.close()
    
    async def get_conflict_npcs(self, conflict_id: int) -> List[Dict[str, Any]]:
        """
        Get NPCs involved in a specific conflict.
        
        Args:
            conflict_id: ID of the conflict
            
        Returns:
            List of NPCs involved in the conflict
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT c.npc_id, n.npc_name, c.faction, c.role, c.influence_level
                FROM ConflictNPCs c
                JOIN NPCStats n ON c.npc_id = n.npc_id
                WHERE c.conflict_id = %s
                ORDER BY c.influence_level DESC
            """, (conflict_id,))
            
            npcs = []
            for row in cursor.fetchall():
                npc_id, npc_name, faction, role, influence_level = row
                
                npcs.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "faction": faction,
                    "role": role,
                    "influence_level": influence_level
                })
            
            return npcs
        except Exception as e:
            logger.error(f"Error getting conflict NPCs for conflict {conflict_id}: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def get_player_involvement(self, conflict_id: int) -> Dict[str, Any]:
        """
        Get the player's involvement in a specific conflict.
        
        Args:
            conflict_id: ID of the conflict
            
        Returns:
            Dictionary with player involvement details
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT involvement_level, faction, money_committed, supplies_committed, 
                       influence_committed, actions_taken
                FROM PlayerConflictInvolvement
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
            """, (conflict_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if not row:
                return {
                    "involvement_level": "none",
                    "faction": "neutral",
                    "resources_committed": {
                        "money": 0,
                        "supplies": 0,
                        "influence": 0
                    },
                    "actions_taken": []
                }
            
            involvement_level, faction, money_committed, supplies_committed, influence_committed, actions_taken = row
            
            # Parse actions_taken JSON
            try:
                if isinstance(actions_taken, str):
                    actions_taken = json.loads(actions_taken)
            except (json.JSONDecodeError, TypeError):
                actions_taken = []
            
            return {
                "involvement_level": involvement_level,
                "faction": faction,
                "resources_committed": {
                    "money": money_committed,
                    "supplies": supplies_committed,
                    "influence": influence_committed
                },
                "actions_taken": actions_taken
            }
        except Exception as e:
            logger.error(f"Error getting player involvement for conflict {conflict_id}: {e}", exc_info=True)
            return {
                "involvement_level": "none",
                "faction": "neutral",
                "resources_committed": {
                    "money": 0,
                    "supplies": 0,
                    "influence": 0
                },
                "actions_taken": []
            }
        finally:
            cursor.close()
            conn.close()
    
    async def set_player_involvement(
        self, 
        conflict_id: int,
        involvement_level: str,
        faction: str = "neutral",
        money_committed: int = 0,
        supplies_committed: int = 0,
        influence_committed: int = 0,
        action: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set or update the player's involvement in a conflict.
        
        Args:
            conflict_id: ID of the conflict
            involvement_level: Level of involvement (none, observing, participating, leading)
            faction: Which faction to support (a, b, neutral)
            money_committed: Money committed to the conflict
            supplies_committed: Supplies committed to the conflict
            influence_committed: Influence committed to the conflict
            action: Optional specific action taken
            
        Returns:
            Updated involvement information
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # First, check if the player has an existing involvement
            cursor.execute("""
                SELECT involvement_level, faction, money_committed, supplies_committed, 
                       influence_committed, actions_taken
                FROM PlayerConflictInvolvement
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
            """, (conflict_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if row:
                existing_level, existing_faction, existing_money, existing_supplies, existing_influence, existing_actions = row
                
                # Parse existing actions
                try:
                    if isinstance(existing_actions, str):
                        existing_actions = json.loads(existing_actions)
                except (json.JSONDecodeError, TypeError):
                    existing_actions = []
                
                # Append the new action if provided
                if action:
                    existing_actions.append({
                        "action": action,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Update the existing involvement
                cursor.execute("""
                    UPDATE PlayerConflictInvolvement
                    SET involvement_level = %s, faction = %s, 
                        money_committed = %s, supplies_committed = %s, influence_committed = %s,
                        actions_taken = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
                    RETURNING id
                """, (
                    involvement_level, faction, 
                    existing_money + money_committed, 
                    existing_supplies + supplies_committed,
                    existing_influence + influence_committed,
                    json.dumps(existing_actions),
                    conflict_id, self.user_id, self.conversation_id
                ))
                
                involvement_id = cursor.fetchone()[0]
            else:
                # Create a new involvement entry
                actions = []
                if action:
                    actions.append({
                        "action": action,
                        "timestamp": datetime.now().isoformat()
                    })
                
                cursor.execute("""
                    INSERT INTO PlayerConflictInvolvement
                    (conflict_id, user_id, conversation_id, player_name, involvement_level,
                     faction, money_committed, supplies_committed, influence_committed, actions_taken)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    conflict_id, self.user_id, self.conversation_id, "Chase",
                    involvement_level, faction, money_committed, supplies_committed, influence_committed,
                    json.dumps(actions)
                ))
                
                involvement_id = cursor.fetchone()[0]
            
            # Commit resources if needed
            if money_committed > 0 or supplies_committed > 0 or influence_committed > 0:
                await self.resource_manager.commit_resources_to_conflict(
                    conflict_id, money_committed, supplies_committed, influence_committed
                )
            
            # Create a conflict memory event
            if action:
                await self._create_conflict_memory(
                    conflict_id,
                    f"Player {action} in the conflict, supporting {faction} faction.",
                    significance=5
                )
            
            # Update conflict progress based on involvement
            if involvement_level in ["participating", "leading"]:
                progress_increment = 5 if involvement_level == "participating" else 10
                await self.update_conflict_progress(conflict_id, progress_increment)
            
            conn.commit()
            
            # Return updated involvement
            return await self.get_player_involvement(conflict_id)
        except Exception as e:
            conn.rollback()
            logger.error(f"Error setting player involvement: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def generate_conflict(self, conflict_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a new conflict of the specified type.
        
        Args:
            conflict_type: Type of conflict to generate (major, minor, standard, catastrophic).
                          If None, a type will be selected based on the current game state.
                          
        Returns:
            The generated conflict
        """
        # Get current time (day and time of day)
        current_day = await self._get_current_day()
        
        # Get active conflicts to ensure we don't generate too many
        active_conflicts = await self.get_active_conflicts()
        
        # If there are already too many active conflicts (3+), make this a minor one
        if len(active_conflicts) >= 3 and not conflict_type:
            conflict_type = "minor"
        
        # If there are no conflicts at all and no type specified, make this a standard one
        if len(active_conflicts) == 0 and not conflict_type:
            conflict_type = "standard"
        
        # If still no type specified, choose randomly with weighted probabilities
        if not conflict_type:
            weights = {
                "minor": 0.4,
                "standard": 0.4,
                "major": 0.15,
                "catastrophic": 0.05
            }
            
            conflict_type = random.choices(
                list(weights.keys()),
                weights=list(weights.values()),
                k=1
            )[0]
        
        # Get existing NPCs to use as potential faction leaders
        npcs = await self._get_available_npcs()
        
        if len(npcs) < 2:
            # Not enough NPCs for a conflict, generate generic faction names
            faction_a_name = "Local Collective"
            faction_b_name = "Unity Group"
        else:
            # Randomly select two different NPCs for faction leaders
            faction_leaders = random.sample(npcs, 2)
            faction_a_name = f"{faction_leaders[0]['npc_name']}'s Circle"
            faction_b_name = f"{faction_leaders[1]['npc_name']}'s Faction"
        
        # Generate the conflict details using the AI
        conflict_data = await self._generate_conflict_details(
            conflict_type, faction_a_name, faction_b_name, current_day, npcs
        )
        
        # Create the conflict in the database
        conflict_id = await self._create_conflict_record(conflict_data, current_day)
        
        # Assign NPCs to the conflict
        await self._assign_npcs_to_conflict(conflict_id, conflict_data, npcs)
        
        # Create initial memory event for the conflict
        await self._create_conflict_memory(
            conflict_id,
            f"A new conflict has emerged: {conflict_data['conflict_name']}. "
            f"It involves {faction_a_name} and {faction_b_name}.",
            significance=6
        )
        
        # Return the created conflict
        return await self.get_conflict(conflict_id)
    
    async def update_conflict_progress(self, conflict_id: int, progress_increment: float) -> Dict[str, Any]:
        """
        Update the progress of a conflict, potentially changing its phase.
        
        Args:
            conflict_id: ID of the conflict to update
            progress_increment: Amount to increment progress (0-100)
            
        Returns:
            Updated conflict information
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current conflict info
            cursor.execute("""
                SELECT progress, phase, conflict_name, conflict_type, 
                       faction_a_name, faction_b_name, resources_required
                FROM Conflicts
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
                FOR UPDATE
            """, (conflict_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Conflict {conflict_id} not found")
            
            current_progress, current_phase, conflict_name, conflict_type, faction_a_name, faction_b_name, resources_required = row
            
            # Calculate new progress
            new_progress = min(100, current_progress + progress_increment)
            
            # Determine if the phase should change
            new_phase = current_phase
            
            # Phase transition thresholds
            phase_thresholds = {
                "brewing": 30,    # brewing -> active
                "active": 60,     # active -> climax
                "climax": 90      # climax -> resolution
            }
            
            # Check if we should transition to a new phase
            if current_phase in phase_thresholds and new_progress >= phase_thresholds[current_phase]:
                if current_phase == "brewing":
                    new_phase = "active"
                elif current_phase == "active":
                    new_phase = "climax"
                elif current_phase == "climax":
                    new_phase = "resolution"
            
            # Update the conflict
            cursor.execute("""
                UPDATE Conflicts
                SET progress = %s, phase = %s, updated_at = CURRENT_TIMESTAMP
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
            """, (new_progress, new_phase, conflict_id, self.user_id, self.conversation_id))
            
            # If the phase changed, create a memory event
            if new_phase != current_phase:
                phase_transition_descriptions = {
                    "active": f"The {conflict_name} conflict has moved from brewing to active phase. Both {faction_a_name} and {faction_b_name} are now openly engaged.",
                    "climax": f"The {conflict_name} conflict has reached its climax. Tensions between {faction_a_name} and {faction_b_name} are at their peak.",
                    "resolution": f"The {conflict_name} conflict is approaching resolution. The outcome between {faction_a_name} and {faction_b_name} will soon be determined."
                }
                
                if new_phase in phase_transition_descriptions:
                    await self._create_conflict_memory(
                        conflict_id,
                        phase_transition_descriptions[new_phase],
                        significance=7
                    )
            
            conn.commit()
            
            # Return the updated conflict
            return await self.get_conflict(conflict_id)
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating conflict progress: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def resolve_conflict(self, conflict_id: int) -> Dict[str, Any]:
        """
        Resolve a conflict and apply consequences.
        
        Args:
            conflict_id: ID of the conflict to resolve
            
        Returns:
            Resolution outcome and consequences
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get conflict details
            conflict = await self.get_conflict(conflict_id)
            
            if not conflict:
                raise ValueError(f"Conflict {conflict_id} not found")
            
            # Get player involvement
            player_involvement = await self.get_player_involvement(conflict_id)
            
            # Generate resolution outcome
            outcome, consequences = await self._generate_resolution_outcome(conflict, player_involvement)
            
            # Update the conflict
            cursor.execute("""
                UPDATE Conflicts
                SET progress = 100, phase = 'resolved', outcome = %s, is_active = FALSE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
            """, (outcome, conflict_id, self.user_id, self.conversation_id))
            
            # Create consequences records
            for consequence in consequences:
                entity_type = consequence.get("entity_type", "npc")
                entity_id = consequence.get("entity_id")
                
                cursor.execute("""
                    INSERT INTO ConflictConsequences
                    (conflict_id, consequence_type, entity_type, entity_id, description, applied)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    conflict_id, consequence.get("consequence_type", "relationship"),
                    entity_type, entity_id, consequence.get("description", ""),
                    False  # Not applied yet
                ))
            
            # Create memory event for resolution
            await self._create_conflict_memory(
                conflict_id,
                f"The {conflict['conflict_name']} conflict has been resolved with outcome: {outcome}",
                significance=8
            )
            
            conn.commit()
            
            return {
                "conflict_id": conflict_id,
                "outcome": outcome,
                "consequences": consequences,
                "success": True
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"Error resolving conflict: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def add_conflict_to_narrative(self, narrative_text: str) -> Dict[str, Any]:
        """
        Analyze a narrative text and potentially generate a conflict from it.
        
        Args:
            narrative_text: The narrative text to analyze
            
        Returns:
            Analysis results and possibly a new conflict
        """
        # Check if narrative text is substantial enough
        if len(narrative_text) < 50:
            return {
                "analysis": {
                    "conflict_intensity": 0,
                    "matched_keywords": []
                },
                "conflict_generated": False,
                "message": "Narrative text too short for analysis"
            }
        
        # Analyze the narrative text for conflict keywords
        conflict_keywords = [
            "argument", "disagreement", "tension", "rivalry", "competition",
            "dispute", "feud", "clash", "confrontation", "battle", "fight",
            "war", "conflict", "power struggle", "contest", "strife"
        ]
        
        matched_keywords = []
        for keyword in conflict_keywords:
            if keyword in narrative_text.lower():
                matched_keywords.append(keyword)
        
        # Calculate conflict intensity based on keyword matches
        conflict_intensity = min(10, len(matched_keywords) * 2)
        
        analysis = {
            "conflict_intensity": conflict_intensity,
            "matched_keywords": matched_keywords
        }
        
        # If conflict intensity is high enough, generate a conflict
        if conflict_intensity >= 4:
            try:
                # Use AI to determine conflict details from narrative
                conflict_data = await self._extract_conflict_from_narrative(narrative_text)
                
                # Generate the conflict
                conflict_type = conflict_data.get("conflict_type", "standard")
                conflict = await self.generate_conflict(conflict_type)
                
                return {
                    "analysis": analysis,
                    "conflict_generated": True,
                    "conflict": conflict,
                    "message": f"Generated conflict from narrative: {conflict['conflict_name']}"
                }
            except Exception as e:
                logger.error(f"Error generating conflict from narrative: {e}", exc_info=True)
                return {
                    "analysis": analysis,
                    "conflict_generated": False,
                    "message": f"Error generating conflict: {str(e)}"
                }
        else:
            return {
                "analysis": analysis,
                "conflict_generated": False,
                "message": "Narrative does not contain sufficient conflict elements"
            }

    async def process_activity_for_conflict_impact(
        self, 
        activity_type: str, 
        description: str
    ) -> Dict[str, Any]:
        """
        Process a player activity to determine its impact on active conflicts.
        
        Args:
            activity_type: Type of activity the player performed
            description: Description of the activity
            
        Returns:
            Dictionary with impacts on conflicts
        """
        active_conflicts = await self.get_active_conflicts()
        
        if not active_conflicts:
            return {
                "conflicts_affected": 0,
                "impacts": []
            }
        
        impacts = []
        
        # Default progress increments by activity type
        activity_impacts = {
            "sleep": 2,  # Sleeping advances conflicts slightly
            "work_shift": 5,  # Working advances conflicts moderately
            "class_attendance": 3,  # Classes advance conflicts somewhat
            "social_event": 4,  # Social events can advance conflicts
            "training": 3,  # Training advances conflicts somewhat
            "extended_conversation": 4,  # Conversations can advance conflicts
            "personal_time": 2  # Personal time advances conflicts slightly
        }
        
        # Default progress increment if activity type not recognized
        default_increment = 1
        
        for conflict in active_conflicts:
            conflict_id = conflict["conflict_id"]
            conflict_name = conflict["conflict_name"]
            faction_a = conflict["faction_a_name"]
            faction_b = conflict["faction_b_name"]
            
            # Check if the activity is directly relevant to this conflict
            is_relevant = False
            relevance_score = 0
            
            # Look for conflict name or faction names in the description
            if conflict_name.lower() in description.lower():
                is_relevant = True
                relevance_score += 3
            
            if faction_a.lower() in description.lower():
                is_relevant = True
                relevance_score += 2
            
            if faction_b.lower() in description.lower():
                is_relevant = True
                relevance_score += 2
            
            # Determine progress increment
            progress_increment = activity_impacts.get(activity_type, default_increment)
            
            # If activity is directly relevant, increase the impact
            if is_relevant:
                progress_increment = progress_increment * (1 + relevance_score * 0.5)
            
            # Scale based on conflict type
            if conflict["conflict_type"] == "major":
                progress_increment *= 0.5  # Major conflicts progress slower
            elif conflict["conflict_type"] == "minor":
                progress_increment *= 1.5  # Minor conflicts progress faster
            
            # Apply a small randomization factor
            progress_increment *= random.uniform(0.8, 1.2)
            
            # Round to 1 decimal place
            progress_increment = round(progress_increment, 1)
            
            # Update conflict progress
            if progress_increment > 0:
                try:
                    updated_conflict = await self.update_conflict_progress(conflict_id, progress_increment)
                    
                    impacts.append({
                        "conflict_id": conflict_id,
                        "conflict_name": conflict_name,
                        "progress_increment": progress_increment,
                        "new_progress": updated_conflict["progress"],
                        "new_phase": updated_conflict["phase"],
                        "is_relevant": is_relevant,
                        "relevance_score": relevance_score
                    })
                except Exception as e:
                    logger.error(f"Error updating conflict {conflict_id} during activity processing: {e}", exc_info=True)
        
        return {
            "conflicts_affected": len(impacts),
            "impacts": impacts
        }
    
    # ----- Helper methods -----
    
    async def _get_current_day(self) -> int:
        """Get the current in-game day."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='CurrentDay'
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            return int(row[0]) if row else 1
        except Exception as e:
            logger.error(f"Error getting current day: {e}", exc_info=True)
            return 1
        finally:
            cursor.close()
            conn.close()
    
    async def _get_available_npcs(self) -> List[Dict[str, Any]]:
        """Get available NPCs that could be involved in conflicts."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT npc_id, npc_name, dominance, cruelty, closeness, trust
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
                ORDER BY dominance DESC
            """, (self.user_id, self.conversation_id))
            
            npcs = []
            for row in cursor.fetchall():
                npc_id, npc_name, dominance, cruelty, closeness, trust = row
                
                npcs.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "closeness": closeness,
                    "trust": trust
                })
            
            return npcs
        except Exception as e:
            logger.error(f"Error getting available NPCs: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def _generate_conflict_details(
        self, 
        conflict_type: str, 
        faction_a_name: str, 
        faction_b_name: str,
        current_day: int,
        npcs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate conflict details using the AI.
        
        Args:
            conflict_type: Type of conflict
            faction_a_name: Name of the first faction
            faction_b_name: Name of the second faction
            current_day: Current in-game day
            npcs: List of available NPCs
            
        Returns:
            Dictionary with generated conflict details
        """
        # Prepare NPC information for the prompt
        npc_info = ""
        for i, npc in enumerate(npcs[:5]):  # Limit to 5 NPCs for prompt clarity
            npc_info += f"{i+1}. {npc['npc_name']} (Dominance: {npc['dominance']}, Cruelty: {npc['cruelty']})\n"
        
        prompt = f"""
        As an AI game system, generate a femdom-themed conflict between two factions.

        Conflict Type: {conflict_type.capitalize()}
        Faction A: {faction_a_name}
        Faction B: {faction_b_name}
        Current Day: {current_day}
        
        Available NPCs:
        {npc_info}
        
        Generate the following details:
        1. A compelling conflict name
        2. A general description of the conflict
        3. Phase-specific descriptions (brewing, active, climax, resolution)
        4. How long the conflict should last (in days)
        5. Resources required to meaningfully participate
        
        Return your response in the following exact JSON format:
        {{
            "conflict_name": "Name of the conflict",
            "conflict_type": "{conflict_type}",
            "description": "General description of the conflict",
            "brewing_description": "Description of the brewing phase",
            "active_description": "Description of the active phase",
            "climax_description": "Description of the climax phase",
            "resolution_description": "Description of the resolution phase",
            "estimated_duration": 7,
            "resources_required": {{
                "money": 20,
                "supplies": 5,
                "influence": 10
            }}
        }}
        
        Ensure all descriptions maintain the femdom theme but keep them appropriate.
        """
        
        response = await get_chatgpt_response(
            self.conversation_id,
            conflict_type,
            prompt
        )
        
        if response and "function_args" in response:
            function_args = response["function_args"]
            
            # Make sure the required fields are present
            required_fields = [
                "conflict_name", "conflict_type", "description", 
                "brewing_description", "active_description", 
                "climax_description", "resolution_description",
                "estimated_duration", "resources_required"
            ]
            
            for field in required_fields:
                if field not in function_args:
                    if field == "resources_required":
                        function_args[field] = {"money": 10, "supplies": 5, "influence": 5}
                    elif field == "estimated_duration":
                        function_args[field] = 7
                    else:
                        function_args[field] = f"Default {field.replace('_', ' ')}"
            
            return function_args
        else:
            # If function call format didn't work, try to extract JSON from text response
            try:
                response_text = response.get("response", "{}")
                
                # Find JSON in the response
                json_match = re.search(r'{.*}', response_text, re.DOTALL)
                if json_match:
                    conflict_data = json.loads(json_match.group(0))
                    
                    # Ensure required fields are present
                    if "conflict_name" not in conflict_data:
                        conflict_data["conflict_name"] = f"{faction_a_name} vs {faction_b_name}"
                    
                    if "conflict_type" not in conflict_data:
                        conflict_data["conflict_type"] = conflict_type
                    
                    if "description" not in conflict_data:
                        conflict_data["description"] = f"A {conflict_type} conflict between {faction_a_name} and {faction_b_name}"
                    
                    for phase in ["brewing", "active", "climax", "resolution"]:
                        phase_desc = f"{phase}_description"
                        if phase_desc not in conflict_data:
                            conflict_data[phase_desc] = f"Default {phase} phase description"
                    
                    if "estimated_duration" not in conflict_data:
                        conflict_data["estimated_duration"] = 7
                    
                    if "resources_required" not in conflict_data:
                        conflict_data["resources_required"] = {"money": 10, "supplies": 5, "influence": 5}
                    
                    return conflict_data
            except Exception as e:
                logger.error(f"Error parsing conflict data from response: {e}", exc_info=True)
            
            # Fallback if no valid JSON could be extracted
            return {
                "conflict_name": f"{faction_a_name} vs {faction_b_name}",
                "conflict_type": conflict_type,
                "description": f"A {conflict_type} conflict between {faction_a_name} and {faction_b_name}",
                "brewing_description": f"{faction_a_name} and {faction_b_name} are building tension.",
                "active_description": f"{faction_a_name} and {faction_b_name} are in active opposition.",
                "climax_description": f"The conflict between {faction_a_name} and {faction_b_name} has reached its peak.",
                "resolution_description": f"The conflict between {faction_a_name} and {faction_b_name} is being resolved.",
                "estimated_duration": 7,
                "resources_required": {"money": 10, "supplies": 5, "influence": 5}
            }
    
    async def _create_conflict_record(self, conflict_data: Dict[str, Any], current_day: int) -> int:
        """
        Create a conflict record in the database.
        
        Args:
            conflict_data: Dictionary with conflict details
            current_day: Current in-game day
            
        Returns:
            ID of the created conflict
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Base success rate on conflict type
            success_rate = {
                "minor": 0.75,
                "standard": 0.5,
                "major": 0.25,
                "catastrophic": 0.1
            }.get(conflict_data.get("conflict_type", "standard"), 0.5)
            
            # If resources_required is a string, parse it
            resources_required = conflict_data.get("resources_required", {})
            if isinstance(resources_required, str):
                try:
                    resources_required = json.loads(resources_required)
                except json.JSONDecodeError:
                    resources_required = {"money": 10, "supplies": 5, "influence": 5}
            
            cursor.execute("""
                INSERT INTO Conflicts 
                (user_id, conversation_id, conflict_name, conflict_type,
                 description, brewing_description, active_description, climax_description,
                 resolution_description, progress, phase, start_day, estimated_duration,
                 faction_a_name, faction_b_name, resources_required, success_rate, outcome, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING conflict_id
            """, (
                self.user_id, self.conversation_id,
                conflict_data.get("conflict_name", "Unnamed Conflict"),
                conflict_data.get("conflict_type", "standard"),
                conflict_data.get("description", "Default description"),
                conflict_data.get("brewing_description", "Default brewing description"),
                conflict_data.get("active_description", "Default active description"),
                conflict_data.get("climax_description", "Default climax description"),
                conflict_data.get("resolution_description", "Default resolution description"),
                0.0,  # Initial progress
                "brewing",  # Initial phase
                current_day,
                conflict_data.get("estimated_duration", 7),
                conflict_data.get("faction_a_name", "Faction A"),
                conflict_data.get("faction_b_name", "Faction B"),
                json.dumps(resources_required),
                success_rate,
                "pending",  # Initial outcome
                True  # Is active
            ))
            
            conflict_id = cursor.fetchone()[0]
            conn.commit()
            
            return conflict_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating conflict record: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def _assign_npcs_to_conflict(
        self, 
        conflict_id: int, 
        conflict_data: Dict[str, Any],
        available_npcs: List[Dict[str, Any]]
    ) -> None:
        """
        Assign NPCs to the conflict factions.
        
        Args:
            conflict_id: ID of the conflict
            conflict_data: Dictionary with conflict details
            available_npcs: List of available NPCs
        """
        if not available_npcs:
            return
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Determine how many NPCs to involve based on conflict type
            npc_count = {
                "minor": min(2, len(available_npcs)),
                "standard": min(3, len(available_npcs)),
                "major": min(4, len(available_npcs)),
                "catastrophic": min(5, len(available_npcs))
            }.get(conflict_data.get("conflict_type", "standard"), min(3, len(available_npcs)))
            
            # Select NPCs to involve
            involved_npcs = random.sample(available_npcs, npc_count)
            
            # Assign roles and factions
            for i, npc in enumerate(involved_npcs):
                # Alternate factions
                faction = "a" if i % 2 == 0 else "b"
                
                # Assign role based on position
                if i == 0:
                    role = "leader"
                elif i == 1:
                    role = "leader" if faction == "b" else "supporter"
                else:
                    role = "supporter"
                
                # Influence level based on dominance and randomization
                influence_level = min(100, max(0, npc["dominance"] + random.randint(-10, 10)))
                
                cursor.execute("""
                    INSERT INTO ConflictNPCs 
                    (conflict_id, npc_id, faction, role, influence_level)
                    VALUES (%s, %s, %s, %s, %s)
                """, (conflict_id, npc["npc_id"], faction, role, influence_level))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error assigning NPCs to conflict: {e}", exc_info=True)
        finally:
            cursor.close()
            conn.close()
    
    async def _create_conflict_memory(
        self, 
        conflict_id: int, 
        memory_text: str,
        significance: int = 5
    ) -> int:
        """
        Create a memory event for a conflict.
        
        Args:
            conflict_id: ID of the conflict
            memory_text: Text of the memory
            significance: Significance level (1-10)
            
        Returns:
            ID of the created memory
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO ConflictMemoryEvents 
                (conflict_id, memory_text, significance, entity_type, entity_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (conflict_id, memory_text, significance, "conflict", conflict_id))
            
            memory_id = cursor.fetchone()[0]
            conn.commit()
            
            return memory_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating conflict memory: {e}", exc_info=True)
            return 0
        finally:
            cursor.close()
            conn.close()
    
    async def _extract_conflict_from_narrative(self, narrative_text: str) -> Dict[str, Any]:
        """
        Use AI to extract conflict details from a narrative text.
        
        Args:
            narrative_text: The narrative text to analyze
            
        Returns:
            Dictionary with extracted conflict details
        """
        prompt = f"""
        As an AI game system, analyze this narrative text for conflict elements.
        
        Narrative Text:
        {narrative_text}
        
        Extract potential conflict details including:
        1. What type of conflict this represents (minor, standard, major, catastrophic)
        2. The main parties/factions involved
        3. The nature of their disagreement or opposition
        4. How advanced the conflict appears to be (brewing, active, climax, resolution)
        
        Return your analysis in the following exact JSON format:
        {{
            "conflict_type": "standard",
            "faction_a": "Name or description of first party",
            "faction_b": "Name or description of second party",
            "conflict_nature": "Description of what they're in conflict about",
            "current_phase": "brewing"
        }}
        """
        
        response = await get_chatgpt_response(
            self.conversation_id,
            "conflict_analysis",
            prompt
        )
        
        if response and "function_args" in response:
            return response["function_args"]
        else:
            # Try to extract JSON from text response
            try:
                response_text = response.get("response", "{}")
                
                # Find JSON in the response
                json_match = re.search(r'{.*}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
            except Exception as e:
                logger.error(f"Error parsing conflict analysis from response: {e}", exc_info=True)
            
            # Fallback
            return {
                "conflict_type": "standard",
                "faction_a": "Faction A",
                "faction_b": "Faction B",
                "conflict_nature": "Unspecified disagreement",
                "current_phase": "brewing"
            }
    
    async def _generate_resolution_outcome(
        self, 
        conflict: Dict[str, Any],
        player_involvement: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate resolution outcome and consequences for a conflict.
        
        Args:
            conflict: The conflict being resolved
            player_involvement: Player's involvement in the conflict
            
        Returns:
            Tuple of (outcome, consequences)
        """
        # Get NPCs involved in the conflict
        involved_npcs = await self.get_conflict_npcs(conflict["conflict_id"])
        
        # Determine outcome based on player involvement and random factors
        player_faction = player_involvement.get("faction", "neutral")
        player_level = player_involvement.get("involvement_level", "none")
        
        # Base outcome probabilities
        if player_faction == "neutral":
            # No clear winner if player is neutral
            base_outcomes = {
                "faction_a_victory": 0.3,
                "faction_b_victory": 0.3,
                "compromise": 0.4
            }
        elif player_faction == "a":
            # Player supports faction A
            if player_level == "leading":
                base_outcomes = {
                    "faction_a_victory": 0.7,
                    "faction_b_victory": 0.1,
                    "compromise": 0.2
                }
            elif player_level == "participating":
                base_outcomes = {
                    "faction_a_victory": 0.5,
                    "faction_b_victory": 0.2,
                    "compromise": 0.3
                }
            else:
                base_outcomes = {
                    "faction_a_victory": 0.4,
                    "faction_b_victory": 0.3,
                    "compromise": 0.3
                }
        else:  # player_faction == "b"
            # Player supports faction B
            if player_level == "leading":
                base_outcomes = {
                    "faction_a_victory": 0.1,
                    "faction_b_victory": 0.7,
                    "compromise": 0.2
                }
            elif player_level == "participating":
                base_outcomes = {
                    "faction_a_victory": 0.2,
                    "faction_b_victory": 0.5,
                    "compromise": 0.3
                }
            else:
                base_outcomes = {
                    "faction_a_victory": 0.3,
                    "faction_b_victory": 0.4,
                    "compromise": 0.3
                }
        
        # Adjust based on resource commitment
        resources_committed = player_involvement.get("resources_committed", {})
        total_committed = sum(resources_committed.values())
        resources_required = conflict.get("resources_required", {})
        total_required = sum(resources_required.values()) if isinstance(resources_required, dict) else 0
        
        resource_ratio = total_committed / max(1, total_required)
        
        # Adjust outcome probabilities based on resource commitment
        if player_faction != "neutral" and resource_ratio > 0:
            adjustment = min(0.2, resource_ratio * 0.1)  # Cap adjustment at 0.2
            
            if player_faction == "a":
                base_outcomes["faction_a_victory"] += adjustment
                base_outcomes["faction_b_victory"] -= adjustment / 2
                base_outcomes["compromise"] -= adjustment / 2
            else:
                base_outcomes["faction_b_victory"] += adjustment
                base_outcomes["faction_a_victory"] -= adjustment / 2
                base_outcomes["compromise"] -= adjustment / 2
            
            # Normalize probabilities
            total_prob = sum(base_outcomes.values())
            for key in base_outcomes:
                base_outcomes[key] /= total_prob
        
        # Determine the outcome
        outcomes = list(base_outcomes.keys())
        probabilities = list(base_outcomes.values())
        
        outcome = random.choices(outcomes, weights=probabilities, k=1)[0]
        
        # Generate consequences based on outcome
        consequences = []
        
        # Common consequences regardless of outcome
        for npc in involved_npcs:
            npc_id = npc["npc_id"]
            npc_name = npc["npc_name"]
            npc_faction = npc["faction"]
            
            # Determine relationship impact based on faction and outcome
            if (npc_faction == "a" and outcome == "faction_a_victory") or \
               (npc_faction == "b" and outcome == "faction_b_victory"):
                # NPC's faction won - positive for allies, negative for opponents
                if player_faction == npc_faction:
                    consequences.append({
                        "consequence_type": "relationship_gain",
                        "entity_type": "npc",
                        "entity_id": npc_id,
                        "description": f"{npc_name}'s respect for you has increased due to your support in their victory."
                    })
                elif player_faction != "neutral":
                    consequences.append({
                        "consequence_type": "relationship_loss",
                        "entity_type": "npc",
                        "entity_id": npc_id,
                        "description": f"{npc_name} views you with some disdain for supporting the losing side."
                    })
            elif (npc_faction == "a" and outcome == "faction_b_victory") or \
                 (npc_faction == "b" and outcome == "faction_a_victory"):
                # NPC's faction lost - negative for allies, mixed for opponents
                if player_faction == npc_faction:
                    consequences.append({
                        "consequence_type": "relationship_mixed",
                        "entity_type": "npc",
                        "entity_id": npc_id,
                        "description": f"{npc_name} appreciates your support despite their defeat, but harbors some resentment."
                    })
                elif player_faction != "neutral":
                    consequences.append({
                        "consequence_type": "relationship_mild_gain",
                        "entity_type": "npc",
                        "entity_id": npc_id,
                        "description": f"{npc_name} reluctantly acknowledges your faction's victory, with mixed feelings toward you."
                    })
            else:  # compromise
                # Compromise - generally positive but mild
                if player_faction != "neutral":
                    consequences.append({
                        "consequence_type": "relationship_mild_gain",
                        "entity_type": "npc",
                        "entity_id": npc_id,
                        "description": f"{npc_name} appreciates the reasonable resolution, though some tension remains."
                    })
        
        # Resource consequences
        if player_level in ["participating", "leading"]:
            if outcome == f"faction_{player_faction}_victory" and player_faction != "neutral":
                # Player's faction won - resource gain
                consequences.append({
                    "consequence_type": "resource_gain",
                    "entity_type": "player",
                    "entity_id": None,
                    "description": "Your successful involvement has earned you additional resources and influence."
                })
            elif outcome == "compromise":
                # Compromise - small resource gain
                consequences.append({
                    "consequence_type": "resource_mild_gain",
                    "entity_type": "player",
                    "entity_id": None,
                    "description": "The compromise resolution has resulted in a modest return on your invested resources."
                })
        
        # Format outcome text based on the result
        outcome_text = ""
        if outcome == "faction_a_victory":
            outcome_text = f"{conflict['faction_a_name']} Victory"
        elif outcome == "faction_b_victory":
            outcome_text = f"{conflict['faction_b_name']} Victory"
        else:
            outcome_text = "Compromise Resolution"
        
        return outcome_text, consequences
