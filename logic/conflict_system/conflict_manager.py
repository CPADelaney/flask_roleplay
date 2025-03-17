# logic/conflict_system/conflict_manager.py

import logging
import json
import random
import asyncio
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from db.connection import get_db_connection
from logic.chatgpt_integration import get_chatgpt_response
from logic.resource_management import ResourceManager
from logic.npc_relationship_manager import get_relationship_status, get_manipulation_leverage

logger = logging.getLogger(__name__)

class ConflictManager:
    """
    Manages character-driven conflicts with multiple stakeholders, resolution paths,
    internal faction dynamics, and player manipulation mechanics.
    """
    
    # Involvement levels for player
    INVOLVEMENT_LEVELS = ["none", "observing", "participating", "leading"]
    
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
                SELECT c.conflict_id, c.conflict_name, c.conflict_type, 
                       c.description, c.progress, c.phase, c.start_day,
                       c.estimated_duration, c.success_rate, c.outcome, c.is_active
                FROM Conflicts c
                WHERE c.user_id = %s AND c.conversation_id = %s AND c.is_active = TRUE
                ORDER BY c.conflict_id DESC
            """, (self.user_id, self.conversation_id))
            
            conflicts = []
            for row in cursor.fetchall():
                conflict_id, conflict_name, conflict_type, description, progress, phase, \
                start_day, estimated_duration, success_rate, outcome, is_active = row
                
                # Build conflict dictionary
                conflict = {
                    "conflict_id": conflict_id,
                    "conflict_name": conflict_name,
                    "conflict_type": conflict_type,
                    "description": description,
                    "progress": progress,
                    "phase": phase,
                    "start_day": start_day,
                    "estimated_duration": estimated_duration,
                    "success_rate": success_rate,
                    "outcome": outcome,
                    "is_active": is_active
                }
                
                # Get stakeholders
                stakeholders = await self.get_conflict_stakeholders(conflict_id)
                conflict["stakeholders"] = stakeholders
                
                # Get resolution paths
                resolution_paths = await self.get_resolution_paths(conflict_id)
                conflict["resolution_paths"] = resolution_paths
                
                # Get internal faction conflicts
                internal_conflicts = await self.get_internal_faction_conflicts(conflict_id)
                if internal_conflicts:
                    conflict["internal_faction_conflicts"] = internal_conflicts
                
                # Get player involvement
                player_involvement = await self.get_player_involvement(conflict_id)
                conflict["player_involvement"] = player_involvement
                
                # Get manipulation attempts targeted at player
                manipulation_attempts = await self.get_player_manipulation_attempts(conflict_id)
                if manipulation_attempts:
                    conflict["manipulation_attempts"] = manipulation_attempts
                
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
        Get a specific conflict by ID with all related data.
        
        Args:
            conflict_id: ID of the conflict to retrieve
            
        Returns:
            Conflict dictionary if found, empty dict otherwise
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT c.conflict_id, c.conflict_name, c.conflict_type, 
                       c.description, c.progress, c.phase, c.start_day,
                       c.estimated_duration, c.success_rate, c.outcome, c.is_active
                FROM Conflicts c
                WHERE c.conflict_id = %s AND c.user_id = %s AND c.conversation_id = %s
            """, (conflict_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if not row:
                return {}
            
            conflict_id, conflict_name, conflict_type, description, progress, phase, \
            start_day, estimated_duration, success_rate, outcome, is_active = row
            
            # Build conflict dictionary
            conflict = {
                "conflict_id": conflict_id,
                "conflict_name": conflict_name,
                "conflict_type": conflict_type,
                "description": description,
                "progress": progress,
                "phase": phase,
                "start_day": start_day,
                "estimated_duration": estimated_duration,
                "success_rate": success_rate,
                "outcome": outcome,
                "is_active": is_active
            }
            
            # Get stakeholders
            stakeholders = await self.get_conflict_stakeholders(conflict_id)
            conflict["stakeholders"] = stakeholders
            
            # Get resolution paths
            resolution_paths = await self.get_resolution_paths(conflict_id)
            conflict["resolution_paths"] = resolution_paths
            
            # Get internal faction conflicts
            internal_conflicts = await self.get_internal_faction_conflicts(conflict_id)
            if internal_conflicts:
                conflict["internal_faction_conflicts"] = internal_conflicts
            
            # Get player involvement
            player_involvement = await self.get_player_involvement(conflict_id)
            conflict["player_involvement"] = player_involvement
            
            # Get manipulation attempts targeted at player
            manipulation_attempts = await self.get_player_manipulation_attempts(conflict_id)
            if manipulation_attempts:
                conflict["manipulation_attempts"] = manipulation_attempts
            
            return conflict
        except Exception as e:
            logger.error(f"Error getting conflict {conflict_id}: {e}", exc_info=True)
            return {}
        finally:
            cursor.close()
            conn.close()
    
    async def get_conflict_stakeholders(self, conflict_id: int) -> List[Dict[str, Any]]:
        """
        Get all stakeholders involved in a conflict.
        
        Args:
            conflict_id: ID of the conflict
            
        Returns:
            List of stakeholder dictionaries
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT s.npc_id, n.npc_name, s.faction_id, s.faction_name,
                       s.faction_position, s.public_motivation, s.private_motivation,
                       s.desired_outcome, s.involvement_level, s.alliances, s.rivalries,
                       s.leadership_ambition, s.faction_standing, s.willing_to_betray_faction
                FROM ConflictStakeholders s
                JOIN NPCStats n ON s.npc_id = n.npc_id
                WHERE s.conflict_id = %s
                ORDER BY s.involvement_level DESC
            """, (conflict_id,))
            
            stakeholders = []
            for row in cursor.fetchall():
                npc_id, npc_name, faction_id, faction_name, faction_position, public_motivation, \
                private_motivation, desired_outcome, involvement_level, alliances, rivalries, \
                leadership_ambition, faction_standing, willing_to_betray = row
                
                # Parse JSON fields
                try:
                    alliances_dict = json.loads(alliances) if isinstance(alliances, str) else alliances or {}
                except (json.JSONDecodeError, TypeError):
                    alliances_dict = {}
                
                try:
                    rivalries_dict = json.loads(rivalries) if isinstance(rivalries, str) else rivalries or {}
                except (json.JSONDecodeError, TypeError):
                    rivalries_dict = {}
                
                # Get stakeholder secrets
                secrets = await self._get_stakeholder_secrets(conflict_id, npc_id)
                
                # Check if stakeholder manipulates player
                manipulates_player = await self._check_stakeholder_manipulates_player(conflict_id, npc_id)
                
                # Get relationship with player
                relationship_with_player = await self._get_npc_relationship_with_player(npc_id)
                
                stakeholder = {
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "faction_id": faction_id,
                    "faction_name": faction_name,
                    "faction_position": faction_position,
                    "public_motivation": public_motivation,
                    "private_motivation": private_motivation,
                    "desired_outcome": desired_outcome,
                    "involvement_level": involvement_level,
                    "alliances": alliances_dict,
                    "rivalries": rivalries_dict,
                    "leadership_ambition": leadership_ambition,
                    "faction_standing": faction_standing,
                    "willing_to_betray_faction": willing_to_betray,
                    "secrets": secrets,
                    "manipulates_player": manipulates_player,
                    "relationship_with_player": relationship_with_player
                }
                
                stakeholders.append(stakeholder)
            
            return stakeholders
        except Exception as e:
            logger.error(f"Error getting stakeholders for conflict {conflict_id}: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def get_resolution_paths(self, conflict_id: int) -> List[Dict[str, Any]]:
        """
        Get all resolution paths for a conflict.
        
        Args:
            conflict_id: ID of the conflict
            
        Returns:
            List of resolution path dictionaries
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT path_id, name, description, approach_type, difficulty,
                       requirements, stakeholders_involved, key_challenges,
                       progress, is_completed, completion_date
                FROM ResolutionPaths
                WHERE conflict_id = %s
                ORDER BY difficulty ASC
            """, (conflict_id,))
            
            paths = []
            for row in cursor.fetchall():
                path_id, name, description, approach_type, difficulty, requirements, \
                stakeholders_involved, key_challenges, progress, is_completed, completion_date = row
                
                # Parse JSON fields
                try:
                    requirements_dict = json.loads(requirements) if isinstance(requirements, str) else requirements or {}
                except (json.JSONDecodeError, TypeError):
                    requirements_dict = {}
                
                try:
                    stakeholders_list = json.loads(stakeholders_involved) if isinstance(stakeholders_involved, str) else stakeholders_involved or []
                except (json.JSONDecodeError, TypeError):
                    stakeholders_list = []
                
                try:
                    challenges_list = json.loads(key_challenges) if isinstance(key_challenges, str) else key_challenges or []
                except (json.JSONDecodeError, TypeError):
                    challenges_list = []
                
                # Get story beats for this path
                story_beats = await self._get_path_story_beats(conflict_id, path_id)
                
                path = {
                    "path_id": path_id,
                    "name": name,
                    "description": description,
                    "approach_type": approach_type,
                    "difficulty": difficulty,
                    "requirements": requirements_dict,
                    "stakeholders_involved": stakeholders_list,
                    "key_challenges": challenges_list,
                    "progress": progress,
                    "is_completed": is_completed,
                    "completion_date": completion_date.isoformat() if completion_date else None,
                    "story_beats": story_beats
                }
                
                paths.append(path)
            
            return paths
        except Exception as e:
            logger.error(f"Error getting resolution paths for conflict {conflict_id}: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def get_internal_faction_conflicts(self, conflict_id: int) -> List[Dict[str, Any]]:
        """
        Get all internal faction conflicts related to a main conflict.
        
        Args:
            conflict_id: ID of the main conflict
            
        Returns:
            List of internal faction conflict dictionaries
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT struggle_id, faction_id, conflict_name, description,
                       primary_npc_id, target_npc_id, prize, approach, 
                       public_knowledge, current_phase, progress
                FROM InternalFactionConflicts
                WHERE parent_conflict_id = %s
                ORDER BY progress DESC
            """, (conflict_id,))
            
            internal_conflicts = []
            for row in cursor.fetchall():
                struggle_id, faction_id, conflict_name, description, primary_npc_id, \
                target_npc_id, prize, approach, public_knowledge, current_phase, progress = row
                
                # Get faction details
                faction_name = await self._get_faction_name(faction_id)
                
                # Get NPC names
                primary_npc_name = await self._get_npc_name(primary_npc_id)
                target_npc_name = await self._get_npc_name(target_npc_id)
                
                # Get faction members' positions
                faction_members = await self._get_faction_struggle_members(struggle_id)
                
                struggle = {
                    "struggle_id": struggle_id,
                    "faction_id": faction_id,
                    "faction_name": faction_name,
                    "conflict_name": conflict_name,
                    "description": description,
                    "primary_npc_id": primary_npc_id,
                    "primary_npc_name": primary_npc_name,
                    "target_npc_id": target_npc_id,
                    "target_npc_name": target_npc_name,
                    "prize": prize,
                    "approach": approach,
                    "public_knowledge": public_knowledge,
                    "current_phase": current_phase,
                    "progress": progress,
                    "faction_members": faction_members
                }
                
                internal_conflicts.append(struggle)
            
            return internal_conflicts
        except Exception as e:
            logger.error(f"Error getting internal faction conflicts for conflict {conflict_id}: {e}", exc_info=True)
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
                       influence_committed, actions_taken, manipulated_by
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
                    "actions_taken": [],
                    "is_manipulated": False,
                    "manipulated_by": None
                }
            
            involvement_level, faction, money_committed, supplies_committed, \
            influence_committed, actions_taken, manipulated_by = row
            
            # Parse actions_taken JSON
            try:
                actions_list = json.loads(actions_taken) if isinstance(actions_taken, str) else actions_taken or []
            except (json.JSONDecodeError, TypeError):
                actions_list = []
            
            # Parse manipulated_by JSON
            try:
                manipulated_by_dict = json.loads(manipulated_by) if isinstance(manipulated_by, str) else manipulated_by or None
            except (json.JSONDecodeError, TypeError):
                manipulated_by_dict = None
            
            # Get manipulator name if applicable
            manipulator_name = None
            if manipulated_by_dict and "npc_id" in manipulated_by_dict:
                manipulator_name = await self._get_npc_name(manipulated_by_dict["npc_id"])
            
            return {
                "involvement_level": involvement_level,
                "faction": faction,
                "resources_committed": {
                    "money": money_committed,
                    "supplies": supplies_committed,
                    "influence": influence_committed
                },
                "actions_taken": actions_list,
                "is_manipulated": manipulated_by_dict is not None,
                "manipulated_by": manipulated_by_dict,
                "manipulator_name": manipulator_name
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
                "actions_taken": [],
                "is_manipulated": False,
                "manipulated_by": None
            }
        finally:
            cursor.close()
            conn.close()
    
    async def get_player_manipulation_attempts(self, conflict_id: int) -> List[Dict[str, Any]]:
        """
        Get all manipulation attempts targeted at the player for a specific conflict.
        
        Args:
            conflict_id: ID of the conflict
            
        Returns:
            List of manipulation attempt dictionaries
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT attempt_id, npc_id, manipulation_type, content, goal,
                       success, player_response, leverage_used, intimacy_level,
                       created_at, resolved_at
                FROM PlayerManipulationAttempts
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
                ORDER BY created_at DESC
            """, (conflict_id, self.user_id, self.conversation_id))
            
            attempts = []
            for row in cursor.fetchall():
                attempt_id, npc_id, manipulation_type, content, goal, success, \
                player_response, leverage_used, intimacy_level, created_at, resolved_at = row
                
                # Get NPC name
                npc_name = await self._get_npc_name(npc_id)
                
                # Parse JSON fields
                try:
                    goal_dict = json.loads(goal) if isinstance(goal, str) else goal or {}
                except (json.JSONDecodeError, TypeError):
                    goal_dict = {}
                
                try:
                    leverage_dict = json.loads(leverage_used) if isinstance(leverage_used, str) else leverage_used or {}
                except (json.JSONDecodeError, TypeError):
                    leverage_dict = {}
                
                attempt = {
                    "attempt_id": attempt_id,
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "manipulation_type": manipulation_type,
                    "content": content,
                    "goal": goal_dict,
                    "success": success,
                    "player_response": player_response,
                    "leverage_used": leverage_dict,
                    "intimacy_level": intimacy_level,
                    "created_at": created_at.isoformat() if created_at else None,
                    "resolved_at": resolved_at.isoformat() if resolved_at else None,
                    "is_resolved": resolved_at is not None
                }
                
                attempts.append(attempt)
            
            return attempts
        except Exception as e:
            logger.error(f"Error getting player manipulation attempts for conflict {conflict_id}: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def generate_conflict(self, conflict_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a new complex conflict with multiple stakeholders and resolution paths.
        
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
        
        # Get available NPCs to use as potential stakeholders
        npcs = await self._get_available_npcs()
        
        if len(npcs) < 3:
            return {"error": "Not enough NPCs available to create a complex conflict"}
        
        # Determine how many stakeholders to involve
        stakeholder_count = {
            "minor": min(3, len(npcs)),
            "standard": min(4, len(npcs)),
            "major": min(5, len(npcs)),
            "catastrophic": min(6, len(npcs))
        }.get(conflict_type, min(4, len(npcs)))
        
        # Select NPCs to involve as stakeholders
        stakeholder_npcs = random.sample(npcs, stakeholder_count)
        
        # Generate the conflict details using the AI
        conflict_data = await self._generate_conflict_details(
            conflict_type, stakeholder_npcs, current_day
        )
        
        # Create the conflict in the database
        conflict_id = await self._create_conflict_record(conflict_data, current_day)
        
        # Create stakeholders
        await self._create_stakeholders(conflict_id, conflict_data, stakeholder_npcs)
        
        # Create resolution paths
        await self._create_resolution_paths(conflict_id, conflict_data)
        
        # Create internal faction conflicts if applicable
        if "internal_faction_conflicts" in conflict_data:
            await self._create_internal_faction_conflicts(conflict_id, conflict_data)
        
        # Generate player manipulation attempts from stakeholders
        await self._generate_player_manipulation_attempts(conflict_id, stakeholder_npcs)
        
        # Create initial memory event for the conflict
        await self._create_conflict_memory(
            conflict_id,
            f"A new conflict has emerged: {conflict_data['conflict_name']}. It involves multiple stakeholders with their own agendas.",
            significance=6
        )
        
        # Return the created conflict
        return await self.get_conflict(conflict_id)
    
    async def create_player_manipulation_attempt(
        self,
        conflict_id: int,
        npc_id: int,
        manipulation_type: str,
        content: str,
        goal: Dict[str, Any],
        leverage_used: Dict[str, Any],
        intimacy_level: int = 0
    ) -> Dict[str, Any]:
        """
        Create a manipulation attempt by an NPC targeted at the player.
        
        Args:
            conflict_id: ID of the conflict
            npc_id: ID of the NPC doing the manipulation
            manipulation_type: Type of manipulation (domination, blackmail, seduction, etc.)
            content: Content of the manipulation attempt
            goal: What the NPC wants the player to do
            leverage_used: What leverage the NPC is using
            intimacy_level: Level of intimacy in the manipulation (0-10)
            
        Returns:
            The created manipulation attempt
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check if NPC has relationship with player
            relationship = await self._get_npc_relationship_with_player(npc_id)
            
            # Get NPC name
            npc_name = await self._get_npc_name(npc_id)
            
            # Insert the manipulation attempt
            cursor.execute("""
                INSERT INTO PlayerManipulationAttempts
                (conflict_id, user_id, conversation_id, npc_id, manipulation_type, 
                 content, goal, success, leverage_used, intimacy_level, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING attempt_id
            """, (
                conflict_id, self.user_id, self.conversation_id, npc_id,
                manipulation_type, content, json.dumps(goal), False,
                json.dumps(leverage_used), intimacy_level
            ))
            
            attempt_id = cursor.fetchone()[0]
            
            # Create a memory for this manipulation attempt
            await self._create_conflict_memory(
                conflict_id,
                f"{npc_name} attempted to {manipulation_type} the player regarding the conflict.",
                significance=7
            )
            
            conn.commit()
            
            return {
                "attempt_id": attempt_id,
                "npc_id": npc_id,
                "npc_name": npc_name,
                "manipulation_type": manipulation_type,
                "content": content,
                "goal": goal,
                "leverage_used": leverage_used,
                "intimacy_level": intimacy_level,
                "success": False,
                "is_resolved": False
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating player manipulation attempt: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def resolve_manipulation_attempt(
        self,
        attempt_id: int,
        success: bool,
        player_response: str
    ) -> Dict[str, Any]:
        """
        Resolve a manipulation attempt by the player.
        
        Args:
            attempt_id: ID of the manipulation attempt
            success: Whether the manipulation was successful
            player_response: The player's response to the manipulation
            
        Returns:
            Updated manipulation attempt
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get the manipulation attempt
            cursor.execute("""
                SELECT conflict_id, npc_id, manipulation_type, goal
                FROM PlayerManipulationAttempts
                WHERE attempt_id = %s AND user_id = %s AND conversation_id = %s
            """, (attempt_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if not row:
                return {"error": "Manipulation attempt not found"}
            
            conflict_id, npc_id, manipulation_type, goal = row
            
            # Update the manipulation attempt
            cursor.execute("""
                UPDATE PlayerManipulationAttempts
                SET success = %s, player_response = %s, resolved_at = CURRENT_TIMESTAMP
                WHERE attempt_id = %s
                RETURNING attempt_id
            """, (success, player_response, attempt_id))
            
            # If successful, update player involvement
            if success:
                # Parse goal
                try:
                    goal_dict = json.loads(goal) if isinstance(goal, str) else goal or {}
                except (json.JSONDecodeError, TypeError):
                    goal_dict = {}
                
                # Get current involvement
                involvement = await self.get_player_involvement(conflict_id)
                
                # Update involvement based on goal
                if "faction" in goal_dict:
                    faction = goal_dict.get("faction", "neutral")
                else:
                    faction = involvement["faction"]
                
                if "involvement_level" in goal_dict:
                    involvement_level = goal_dict.get("involvement_level", "observing")
                else:
                    involvement_level = involvement["involvement_level"]
                    if involvement_level == "none":
                        involvement_level = "observing"
                
                # Record that player was manipulated
                cursor.execute("""
                    UPDATE PlayerConflictInvolvement
                    SET involvement_level = %s, faction = %s, manipulated_by = %s
                    WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
                """, (
                    involvement_level, faction, 
                    json.dumps({"npc_id": npc_id, "manipulation_type": manipulation_type, "attempt_id": attempt_id}),
                    conflict_id, self.user_id, self.conversation_id
                ))
                
                # If no rows updated, insert new involvement
                if cursor.rowcount == 0:
                    cursor.execute("""
                        INSERT INTO PlayerConflictInvolvement
                        (conflict_id, user_id, conversation_id, player_name, involvement_level,
                        faction, money_committed, supplies_committed, influence_committed, 
                        actions_taken, manipulated_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        conflict_id, self.user_id, self.conversation_id, "Chase",
                        involvement_level, faction, 0, 0, 0, "[]",
                        json.dumps({"npc_id": npc_id, "manipulation_type": manipulation_type, "attempt_id": attempt_id})
                    ))
            
            # Get NPC name
            npc_name = await self._get_npc_name(npc_id)
            
            # Create a memory for the resolution
            if success:
                await self._create_conflict_memory(
                    conflict_id,
                    f"Player succumbed to {npc_name}'s {manipulation_type} attempt in the conflict.",
                    significance=8
                )
            else:
                await self._create_conflict_memory(
                    conflict_id,
                    f"Player resisted {npc_name}'s {manipulation_type} attempt in the conflict.",
                    significance=7
                )
            
            conn.commit()
            
            return {
                "attempt_id": attempt_id,
                "success": success,
                "player_response": player_response,
                "is_resolved": True
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"Error resolving manipulation attempt: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def track_story_beat(
        self,
        conflict_id: int,
        path_id: str,
        beat_description: str,
        involved_npcs: List[int],
        progress_value: float
    ) -> Dict[str, Any]:
        """
        Track a story beat for a resolution path, advancing progress.
        
        Args:
            conflict_id: ID of the conflict
            path_id: ID of the resolution path
            beat_description: Description of what happened
            involved_npcs: List of NPC IDs involved
            progress_value: Progress value (0-100)
            
        Returns:
            Updated path information
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Create the story beat
            cursor.execute("""
                INSERT INTO PathStoryBeats
                (conflict_id, path_id, description, involved_npcs, progress_value, created_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING beat_id
            """, (
                conflict_id, path_id, beat_description, 
                json.dumps(involved_npcs), progress_value
            ))
            
            beat_id = cursor.fetchone()[0]
            
            # Get current path progress
            cursor.execute("""
                SELECT progress, is_completed
                FROM ResolutionPaths
                WHERE conflict_id = %s AND path_id = %s
            """, (conflict_id, path_id))
            
            row = cursor.fetchone()
            if not row:
                return {"error": "Resolution path not found"}
            
            current_progress, is_completed = row
            
            # Calculate new progress
            new_progress = min(100, current_progress + progress_value)
            is_now_completed = new_progress >= 100
            
            # Update the path progress
            cursor.execute("""
                UPDATE ResolutionPaths
                SET progress = %s, is_completed = %s,
                    completion_date = %s
                WHERE conflict_id = %s AND path_id = %s
            """, (
                new_progress, is_now_completed,
                "CURRENT_TIMESTAMP" if is_now_completed else None,
                conflict_id, path_id
            ))
            
            # If path completed, check if conflict should advance
            if is_now_completed:
                await self._check_conflict_advancement(conflict_id)
            
            # Create a memory for this story beat
            await self._create_conflict_memory(
                conflict_id,
                f"Progress made on path '{path_id}': {beat_description}",
                significance=6
            )
            
            conn.commit()
            
            return {
                "beat_id": beat_id,
                "conflict_id": conflict_id,
                "path_id": path_id,
                "description": beat_description,
                "progress_value": progress_value,
                "new_progress": new_progress,
                "is_completed": is_now_completed
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"Error tracking story beat: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def initiate_faction_power_struggle(
        self,
        conflict_id: int,
        faction_id: int,
        challenger_npc_id: int,
        target_npc_id: int,
        prize: str,
        approach: str,
        is_public: bool = False
    ) -> Dict[str, Any]:
        """
        Initiate a power struggle within a faction.
        
        Args:
            conflict_id: The main conflict ID
            faction_id: The faction where struggle occurs
            challenger_npc_id: NPC initiating the challenge
            target_npc_id: NPC being challenged (usually leader)
            prize: What's at stake (position, policy, etc.)
            approach: How the challenge is made (direct, subtle, etc.)
            is_public: Whether other stakeholders are aware
            
        Returns:
            The created internal faction conflict
        """
        # Get faction name
        faction_name = await self._get_faction_name(faction_id)
        
        # Get NPC names
        challenger_name = await self._get_npc_name(challenger_npc_id)
        target_name = await self._get_npc_name(target_npc_id)
        
        # Generate struggle details
        struggle_details = await self._generate_struggle_details(
            faction_id, challenger_npc_id, target_npc_id, prize, approach
        )
        
        # Create in database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO InternalFactionConflicts
                (faction_id, conflict_name, description, primary_npc_id, target_npc_id,
                 prize, approach, public_knowledge, current_phase, progress, parent_conflict_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING struggle_id
            """, (
                faction_id, struggle_details["conflict_name"], struggle_details["description"],
                challenger_npc_id, target_npc_id, prize, approach, is_public,
                "brewing", 10, conflict_id
            ))
            
            struggle_id = cursor.fetchone()[0]
            
            # Insert faction members positions
            for member in struggle_details.get("faction_members", []):
                cursor.execute("""
                    INSERT INTO FactionStruggleMembers
                    (struggle_id, npc_id, position, side, standing, 
                     loyalty_strength, reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    struggle_id, member["npc_id"], member.get("position", "Member"),
                    member.get("side", "neutral"), member.get("standing", 50),
                    member.get("loyalty_strength", 50), member.get("reason", "")
                ))
            
            # Insert ideological differences
            for diff in struggle_details.get("ideological_differences", []):
                cursor.execute("""
                    INSERT INTO FactionIdeologicalDifferences
                    (struggle_id, issue, incumbent_position, challenger_position)
                    VALUES (%s, %s, %s, %s)
                """, (
                    struggle_id, diff.get("issue", ""), 
                    diff.get("incumbent_position", ""),
                    diff.get("challenger_position", "")
                ))
            
            # Create a memory for this power struggle
            await self._create_conflict_memory(
                conflict_id,
                f"Internal power struggle has emerged in {faction_name} between {challenger_name} and {target_name}.",
                significance=7
            )
            
            conn.commit()
            
            # Return the created struggle
            return {
                "struggle_id": struggle_id,
                "faction_id": faction_id,
                "faction_name": faction_name,
                "conflict_name": struggle_details["conflict_name"],
                "description": struggle_details["description"],
                "primary_npc_id": challenger_npc_id,
                "primary_npc_name": challenger_name,
                "target_npc_id": target_npc_id,
                "target_npc_name": target_name,
                "prize": prize,
                "approach": approach,
                "public_knowledge": is_public,
                "current_phase": "brewing",
                "progress": 10
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"Error initiating faction power struggle: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def attempt_faction_coup(
        self,
        struggle_id: int,
        approach: str,
        supporting_npcs: List[int],
        resources_committed: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Attempt a coup within a faction to forcefully resolve a power struggle.
        
        Args:
            struggle_id: ID of the internal faction struggle
            approach: The approach used (direct, subtle, force, blackmail)
            supporting_npcs: List of NPCs supporting the coup
            resources_committed: Resources committed to the coup
            
        Returns:
            Coup attempt results
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get struggle details
            cursor.execute("""
                SELECT faction_id, primary_npc_id, target_npc_id, parent_conflict_id
                FROM InternalFactionConflicts
                WHERE struggle_id = %s
            """, (struggle_id,))
            
            row = cursor.fetchone()
            if not row:
                return {"error": "Struggle not found"}
            
            faction_id, primary_npc_id, target_npc_id, parent_conflict_id = row
            
            # Calculate coup success chance
            success_chance = await self._calculate_coup_success_chance(
                struggle_id, approach, supporting_npcs, resources_committed
            )
            
            # Determine outcome
            success = random.random() * 100 <= success_chance
            
            # Record coup attempt
            cursor.execute("""
                INSERT INTO FactionCoupAttempts
                (struggle_id, approach, supporting_npcs, resources_committed,
                 success, success_chance, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (
                struggle_id, approach, json.dumps(supporting_npcs),
                json.dumps(resources_committed), success, success_chance
            ))
            
            coup_id = cursor.fetchone()[0]
            
            # Get primary and target names
            primary_name = await self._get_npc_name(primary_npc_id)
            target_name = await self._get_npc_name(target_npc_id)
            faction_name = await self._get_faction_name(faction_id)
            
            # Generate result based on success
            if success:
                # Update the struggle
                cursor.execute("""
                    UPDATE InternalFactionConflicts
                    SET current_phase = %s, progress = 100,
                        resolved_at = CURRENT_TIMESTAMP
                    WHERE struggle_id = %s
                """, ("resolved", struggle_id))
                
                # Create memory for successful coup
                await self._create_conflict_memory(
                    parent_conflict_id,
                    f"{primary_name}'s coup against {target_name} in {faction_name} has succeeded.",
                    significance=8
                )
                
                result = {
                    "outcome": "success",
                    "description": f"{primary_name} has successfully overthrown {target_name} and taken control of {faction_name}.",
                    "consequences": [
                        f"{primary_name} now controls {faction_name}",
                        f"{target_name} has been removed from power",
                        "The balance of power in the conflict has shifted"
                    ]
                }
            else:
                # Update the struggle
                cursor.execute("""
                    UPDATE InternalFactionConflicts
                    SET current_phase = %s, primary_npc_id = %s, target_npc_id = %s,
                        description = %s
                    WHERE struggle_id = %s
                """, (
                    "aftermath",
                    target_npc_id,  # Roles reversed now
                    primary_npc_id, 
                    f"After a failed coup attempt, {target_name} has consolidated power and {primary_name} is now at their mercy.",
                    struggle_id
                ))
                
                # Create memory for failed coup
                await self._create_conflict_memory(
                    parent_conflict_id,
                    f"{primary_name}'s coup against {target_name} in {faction_name} has failed.",
                    significance=8
                )
                
                result = {
                    "outcome": "failure",
                    "description": f"{primary_name}'s attempt to overthrow {target_name} has failed, leaving them vulnerable to retaliation.",
                    "consequences": [
                        f"{target_name} has strengthened their position in {faction_name}",
                        f"{primary_name} is now in a dangerous position",
                        "Supporting NPCs may face punishment"
                    ]
                }
            
            # Record result in database
            cursor.execute("""
                UPDATE FactionCoupAttempts
                SET result = %s
                WHERE id = %s
            """, (json.dumps(result), coup_id))
            
            conn.commit()
            
            return {
                "coup_id": coup_id,
                "struggle_id": struggle_id,
                "approach": approach,
                "success": success,
                "success_chance": success_chance,
                "result": result
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"Error attempting faction coup: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
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
                SELECT npc_id, npc_name, dominance, cruelty, closeness, trust,
                       respect, intensity, sex, current_location, faction_affiliations
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
                ORDER BY dominance DESC
            """, (self.user_id, self.conversation_id))
            
            npcs = []
            for row in cursor.fetchall():
                npc_id, npc_name, dominance, cruelty, closeness, trust, \
                respect, intensity, sex, current_location, faction_affiliations = row
                
                # Parse faction affiliations
                try:
                    affiliations = json.loads(faction_affiliations) if isinstance(faction_affiliations, str) else faction_affiliations or []
                except (json.JSONDecodeError, TypeError):
                    affiliations = []
                
                # Get relationships with player
                relationship = await self._get_npc_relationship_with_player(npc_id)
                
                npc = {
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "closeness": closeness,
                    "trust": trust,
                    "respect": respect,
                    "intensity": intensity,
                    "sex": sex,
                    "current_location": current_location,
                    "faction_affiliations": affiliations,
                    "relationship_with_player": relationship
                }
                
                npcs.append(npc)
            
            return npcs
        except Exception as e:
            logger.error(f"Error getting available NPCs: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def _get_npc_relationship_with_player(self, npc_id: int) -> Dict[str, Any]:
        """Get an NPC's relationship with the player."""
        try:
            # Use relationship manager to get status
            relationship = await get_relationship_status(
                self.user_id, self.conversation_id, npc_id
            )
            
            # Get potential leverage
            leverage = await get_manipulation_leverage(
                self.user_id, self.conversation_id, npc_id
            )
            
            return {
                "closeness": relationship.get("closeness", 0),
                "trust": relationship.get("trust", 0),
                "respect": relationship.get("respect", 0),
                "intimidation": relationship.get("intimidation", 0),
                "dominance": relationship.get("dominance", 0),
                "has_leverage": len(leverage) > 0,
                "leverage_types": [l.get("type") for l in leverage],
                "manipulation_potential": relationship.get("dominance", 0) > 70 or relationship.get("closeness", 0) > 80
            }
        except Exception as e:
            logger.error(f"Error getting NPC relationship with player: {e}", exc_info=True)
            return {
                "closeness": 0,
                "trust": 0,
                "respect": 0,
                "intimidation": 0,
                "dominance": 0,
                "has_leverage": False,
                "leverage_types": [],
                "manipulation_potential": False
            }
    
    async def _get_stakeholder_secrets(self, conflict_id: int, npc_id: int) -> List[Dict[str, Any]]:
        """Get secrets for a stakeholder in a conflict."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT secret_id, secret_type, content, target_npc_id,
                       is_revealed, revealed_to, is_public
                FROM StakeholderSecrets
                WHERE conflict_id = %s AND npc_id = %s
            """, (conflict_id, npc_id))
            
            secrets = []
            for row in cursor.fetchall():
                secret_id, secret_type, content, target_npc_id, \
                is_revealed, revealed_to, is_public = row
                
                # Only return details if the secret is revealed
                if is_revealed:
                    secrets.append({
                        "secret_id": secret_id,
                        "secret_type": secret_type,
                        "content": content,
                        "target_npc_id": target_npc_id,
                        "is_revealed": is_revealed,
                        "revealed_to": revealed_to,
                        "is_public": is_public
                    })
                else:
                    # Otherwise just return that a secret exists
                    secrets.append({
                        "secret_id": secret_id,
                        "secret_type": secret_type,
                        "is_revealed": False
                    })
            
            return secrets
        except Exception as e:
            logger.error(f"Error getting stakeholder secrets: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def _check_stakeholder_manipulates_player(self, conflict_id: int, npc_id: int) -> bool:
        """Check if a stakeholder has manipulation attempts against the player."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT COUNT(*)
                FROM PlayerManipulationAttempts
                WHERE conflict_id = %s AND npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (conflict_id, npc_id, self.user_id, self.conversation_id))
            
            count = cursor.fetchone()[0]
            
            return count > 0
        except Exception as e:
            logger.error(f"Error checking if stakeholder manipulates player: {e}", exc_info=True)
            return False
        finally:
            cursor.close()
            conn.close()
    
    async def _get_path_story_beats(self, conflict_id: int, path_id: str) -> List[Dict[str, Any]]:
        """Get story beats for a resolution path."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT beat_id, description, involved_npcs, progress_value, created_at
                FROM PathStoryBeats
                WHERE conflict_id = %s AND path_id = %s
                ORDER BY created_at ASC
            """, (conflict_id, path_id))
            
            beats = []
            for row in cursor.fetchall():
                beat_id, description, involved_npcs, progress_value, created_at = row
                
                # Parse involved_npcs JSON
                try:
                    npcs_list = json.loads(involved_npcs) if isinstance(involved_npcs, str) else involved_npcs or []
                except (json.JSONDecodeError, TypeError):
                    npcs_list = []
                
                # Get NPC names
                npc_names = []
                for npc_id in npcs_list:
                    name = await self._get_npc_name(npc_id)
                    if name:
                        npc_names.append(name)
                
                beats.append({
                    "beat_id": beat_id,
                    "description": description,
                    "involved_npcs": npcs_list,
                    "npc_names": npc_names,
                    "progress_value": progress_value,
                    "created_at": created_at.isoformat() if created_at else None
                })
            
            return beats
        except Exception as e:
            logger.error(f"Error getting path story beats: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def _get_npc_name(self, npc_id: int) -> str:
        """Get an NPC's name by ID."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT npc_name
                FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (npc_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            return row[0] if row else f"NPC {npc_id}"
        except Exception as e:
            logger.error(f"Error getting NPC name for ID {npc_id}: {e}", exc_info=True)
            return f"NPC {npc_id}"
        finally:
            cursor.close()
            conn.close()
    
    async def _get_faction_name(self, faction_id: int) -> str:
        """Get a faction's name by ID."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT faction_name
                FROM Factions
                WHERE faction_id = %s
            """, (faction_id,))
            
            row = cursor.fetchone()
            
            return row[0] if row else f"Faction {faction_id}"
        except Exception as e:
            logger.error(f"Error getting faction name for ID {faction_id}: {e}", exc_info=True)
            return f"Faction {faction_id}"
        finally:
            cursor.close()
            conn.close()
    
    async def _get_faction_struggle_members(self, struggle_id: int) -> List[Dict[str, Any]]:
        """Get faction members involved in a power struggle."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT m.npc_id, n.npc_name, m.position, m.side,
                       m.standing, m.loyalty_strength, m.reason
                FROM FactionStruggleMembers m
                JOIN NPCStats n ON m.npc_id = n.npc_id
                WHERE m.struggle_id = %s
            """, (struggle_id,))
            
            members = []
            for row in cursor.fetchall():
                npc_id, npc_name, position, side, standing, loyalty_strength, reason = row
                
                members.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "position": position,
                    "side": side,
                    "standing": standing,
                    "loyalty_strength": loyalty_strength,
                    "reason": reason
                })
            
            return members
        except Exception as e:
            logger.error(f"Error getting faction struggle members: {e}", exc_info=True)
            return []
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
                (conflict_id, memory_text, significance, entity_type, entity_id, user_id, conversation_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (conflict_id, memory_text, significance, "conflict", conflict_id, self.user_id, self.conversation_id))
            
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
    
    async def _generate_conflict_details(
        self, 
        conflict_type: str, 
        stakeholder_npcs: List[Dict[str, Any]],
        current_day: int
    ) -> Dict[str, Any]:
        """
        Generate conflict details using the AI.
        
        Args:
            conflict_type: Type of conflict
            stakeholder_npcs: List of NPC dictionaries to use as stakeholders
            current_day: Current in-game day
            
        Returns:
            Dictionary with generated conflict details
        """
        # Prepare NPC information for the prompt
        npc_info = ""
        for i, npc in enumerate(stakeholder_npcs):
            npc_info += f"{i+1}. {npc['npc_name']} (Dominance: {npc['dominance']}, Cruelty: {npc['cruelty']}, Closeness: {npc['closeness']})\n"
        
        # Get player stats for context
        player_stats = await self._get_player_stats()
        
        prompt = f"""
        As an AI game system, generate a femdom-themed conflict with multiple stakeholders and complex motivations.

        Conflict Type: {conflict_type.capitalize()}
        Current Day: {current_day}
        
        Available NPCs to use as stakeholders:
        {npc_info}
        
        Player Stats:
        {json.dumps(player_stats, indent=2)}
        
        Generate the following details:
        1. A compelling conflict name and description
        2. 3-5 stakeholders with their own motivations and goals
        3. At least 3 distinct resolution paths with different approaches
        4. Potential internal faction conflicts that might emerge
        5. Opportunities for NPCs to manipulate the player using femdom themes
        
        Create stakeholders with:
        - Public and private motivations (what they claim vs. what they really want)
        - Relationships with other stakeholders (alliances and rivalries)
        - Faction affiliations and positions where applicable
        - Secrets that could be revealed during the conflict
        - Potential to manipulate the player based on dominance, corruption, etc.
        
        Create resolution paths that:
        - Allow different play styles (social, investigative, direct, etc.)
        - Require engaging with specific stakeholders
        - Have interesting narrative implications
        - Include key challenges to overcome
        
        Include opportunities for femdom-themed manipulation where:
        - Dominant female NPCs could try to control or influence the player
        - NPCs might use blackmail, seduction, or direct commands
        - Different paths could affect player corruption, obedience, etc.
        
        Return your response in JSON format including all these elements.
        """
        
        response = await get_chatgpt_response(
            self.conversation_id,
            conflict_type,
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
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Fallback to basic structure
            return {
                "conflict_name": f"{conflict_type.capitalize()} Conflict",
                "conflict_type": conflict_type,
                "description": f"A {conflict_type} conflict involving multiple stakeholders with their own agendas.",
                "stakeholders": [
                    {
                        "npc_id": npc["npc_id"],
                        "public_motivation": f"{npc['npc_name']} wants to resolve the conflict peacefully.",
                        "private_motivation": f"{npc['npc_name']} actually wants to gain power through the conflict.",
                        "desired_outcome": "Control the outcome to their advantage",
                        "faction_id": npc.get("faction_affiliations", [{}])[0].get("faction_id") if npc.get("faction_affiliations") else None,
                        "faction_name": npc.get("faction_affiliations", [{}])[0].get("faction_name") if npc.get("faction_affiliations") else None
                    }
                    for npc in stakeholder_npcs
                ],
                "resolution_paths": [
                    {
                        "path_id": "diplomatic",
                        "name": "Diplomatic Resolution",
                        "description": "Resolve the conflict through negotiation and compromise.",
                        "approach_type": "social",
                        "difficulty": 5,
                        "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs[:2]],
                        "key_challenges": ["Building trust", "Finding common ground", "Managing expectations"]
                    },
                    {
                        "path_id": "force",
                        "name": "Forceful Resolution",
                        "description": "Resolve the conflict through direct action and confrontation.",
                        "approach_type": "direct",
                        "difficulty": 7,
                        "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs[1:3]],
                        "key_challenges": ["Overcoming resistance", "Managing collateral damage", "Securing victory"]
                    },
                    {
                        "path_id": "manipulation",
                        "name": "Manipulative Resolution",
                        "description": "Resolve the conflict by playing stakeholders against each other.",
                        "approach_type": "deception",
                        "difficulty": 8,
                        "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs[:3]],
                        "key_challenges": ["Maintaining deception", "Balancing interests", "Avoiding exposure"]
                    }
                ],
                "internal_faction_conflicts": [],
                "player_manipulation_opportunities": [
                    {
                        "npc_id": stakeholder_npcs[0]["npc_id"],
                        "manipulation_type": "domination",
                        "content": f"{stakeholder_npcs[0]['npc_name']} demands your help in the conflict, using her position of power over you.",
                        "goal": {
                            "faction": "a",
                            "involvement_level": "participating"
                        }
                    }
                ]
            }
    
    async def _get_player_stats(self) -> Dict[str, Any]:
        """Get player stats for context generation."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT corruption, confidence, willpower, obedience,
                       dependency, lust, mental_resilience, physical_endurance
                FROM PlayerStats
                WHERE user_id = %s AND conversation_id = %s
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if not row:
                return {
                    "corruption": 0,
                    "confidence": 50,
                    "willpower": 50,
                    "obedience": 0,
                    "dependency": 0,
                    "lust": 20,
                    "mental_resilience": 50,
                    "physical_endurance": 50
                }
            
            corruption, confidence, willpower, obedience, dependency, lust, mental_resilience, physical_endurance = row
            
            return {
                "corruption": corruption,
                "confidence": confidence,
                "willpower": willpower,
                "obedience": obedience,
                "dependency": dependency,
                "lust": lust,
                "mental_resilience": mental_resilience,
                "physical_endurance": physical_endurance
            }
        except Exception as e:
            logger.error(f"Error getting player stats: {e}", exc_info=True)
            return {
                "corruption": 0,
                "confidence": 50,
                "willpower": 50,
                "obedience": 0,
                "dependency": 0,
                "lust": 20,
                "mental_resilience": 50,
                "physical_endurance": 50
            }
        finally:
            cursor.close()
            conn.close()
    
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
            
            cursor.execute("""
                INSERT INTO Conflicts 
                (user_id, conversation_id, conflict_name, conflict_type,
                 description, progress, phase, start_day, estimated_duration,
                 success_rate, outcome, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING conflict_id
            """, (
                self.user_id, self.conversation_id,
                conflict_data.get("conflict_name", "Unnamed Conflict"),
                conflict_data.get("conflict_type", "standard"),
                conflict_data.get("description", "Default description"),
                0.0,  # Initial progress
                "brewing",  # Initial phase
                current_day,
                conflict_data.get("estimated_duration", 7),
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
    
    async def _create_stakeholders(
        self,
        conflict_id: int,
        conflict_data: Dict[str, Any],
        stakeholder_npcs: List[Dict[str, Any]]
    ) -> None:
        """
        Create stakeholders for a conflict.
        
        Args:
            conflict_id: ID of the conflict
            conflict_data: Dictionary with conflict details
            stakeholder_npcs: List of NPC dictionaries to use as stakeholders
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get stakeholders from conflict data
            stakeholders = conflict_data.get("stakeholders", [])
            
            # If no stakeholders in data, create from NPCs
            if not stakeholders:
                stakeholders = []
                for npc in stakeholder_npcs:
                    stakeholder = {
                        "npc_id": npc["npc_id"],
                        "public_motivation": f"{npc['npc_name']} wants to resolve the conflict peacefully.",
                        "private_motivation": f"{npc['npc_name']} actually wants to gain power through the conflict.",
                        "desired_outcome": "Control the outcome to their advantage",
                        "faction_id": npc.get("faction_affiliations", [{}])[0].get("faction_id") if npc.get("faction_affiliations") else None,
                        "faction_name": npc.get("faction_affiliations", [{}])[0].get("faction_name") if npc.get("faction_affiliations") else None,
                        "involvement_level": 7 - stakeholder_npcs.index(npc)  # Decreasing involvement
                    }
                    stakeholders.append(stakeholder)
            
            # Create stakeholders in database
            for stakeholder in stakeholders:
                npc_id = stakeholder.get("npc_id")
                
                # Get NPC from list
                npc = next((n for n in stakeholder_npcs if n["npc_id"] == npc_id), None)
                if not npc:
                    continue
                
                # Default faction info
                faction_id = stakeholder.get("faction_id")
                faction_name = stakeholder.get("faction_name")
                
                # If not specified, try to get from NPC
                if not faction_id and npc.get("faction_affiliations"):
                    faction_id = npc.get("faction_affiliations", [{}])[0].get("faction_id")
                    faction_name = npc.get("faction_affiliations", [{}])[0].get("faction_name")
                
                # Default alliances and rivalries
                alliances = stakeholder.get("alliances", {})
                rivalries = stakeholder.get("rivalries", {})
                
                # Insert stakeholder
                cursor.execute("""
                    INSERT INTO ConflictStakeholders
                    (conflict_id, npc_id, faction_id, faction_name, faction_position,
                     public_motivation, private_motivation, desired_outcome,
                     involvement_level, alliances, rivalries, leadership_ambition,
                     faction_standing, willing_to_betray_faction)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    conflict_id, npc_id, faction_id, faction_name, stakeholder.get("faction_position", "Member"),
                    stakeholder.get("public_motivation", "Resolve the conflict favorably"),
                    stakeholder.get("private_motivation", "Gain advantage from the conflict"),
                    stakeholder.get("desired_outcome", "Success for their side"),
                    stakeholder.get("involvement_level", 5),
                    json.dumps(alliances),
                    json.dumps(rivalries),
                    stakeholder.get("leadership_ambition", npc.get("dominance", 50) // 10),
                    stakeholder.get("faction_standing", 50),
                    stakeholder.get("willing_to_betray_faction", npc.get("cruelty", 20) > 60)
                ))
                
                # Create secrets if specified
                if "secrets" in stakeholder:
                    for secret in stakeholder["secrets"]:
                        cursor.execute("""
                            INSERT INTO StakeholderSecrets
                            (conflict_id, npc_id, secret_id, secret_type, content,
                             target_npc_id, is_revealed, revealed_to, is_public)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            conflict_id, npc_id, 
                            secret.get("secret_id", f"secret_{npc_id}_{random.randint(1000, 9999)}"),
                            secret.get("secret_type", "personal"),
                            secret.get("content", "A hidden secret"),
                            secret.get("target_npc_id"),
                            False, None, False
                        ))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating stakeholders: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def _create_resolution_paths(
        self,
        conflict_id: int,
        conflict_data: Dict[str, Any]
    ) -> None:
        """
        Create resolution paths for a conflict.
        
        Args:
            conflict_id: ID of the conflict
            conflict_data: Dictionary with conflict details
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get resolution paths from conflict data
            paths = conflict_data.get("resolution_paths", [])
            
            # Create paths in database
            for path in paths:
                cursor.execute("""
                    INSERT INTO ResolutionPaths
                    (conflict_id, path_id, name, description, approach_type,
                     difficulty, requirements, stakeholders_involved, key_challenges,
                     progress, is_completed)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    conflict_id, 
                    path.get("path_id", f"path_{random.randint(1000, 9999)}"),
                    path.get("name", "Unnamed Path"),
                    path.get("description", "A path to resolve the conflict"),
                    path.get("approach_type", "standard"),
                    path.get("difficulty", 5),
                    json.dumps(path.get("requirements", {})),
                    json.dumps(path.get("stakeholders_involved", [])),
                    json.dumps(path.get("key_challenges", [])),
                    0.0,  # Initial progress
                    False  # Not completed
                ))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating resolution paths: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def _create_internal_faction_conflicts(
        self,
        conflict_id: int,
        conflict_data: Dict[str, Any]
    ) -> None:
        """
        Create internal faction conflicts for a main conflict.
        
        Args:
            conflict_id: ID of the main conflict
            conflict_data: Dictionary with conflict details
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get internal faction conflicts from conflict data
            internal_conflicts = conflict_data.get("internal_faction_conflicts", [])
            
            # Create internal conflicts in database
            for internal in internal_conflicts:
                cursor.execute("""
                    INSERT INTO InternalFactionConflicts
                    (faction_id, conflict_name, description, primary_npc_id, target_npc_id,
                     prize, approach, public_knowledge, current_phase, progress, parent_conflict_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING struggle_id
                """, (
                    internal.get("faction_id", 0),
                    internal.get("conflict_name", "Internal Faction Struggle"),
                    internal.get("description", "A power struggle within the faction"),
                    internal.get("primary_npc_id", 0),
                    internal.get("target_npc_id", 0),
                    internal.get("prize", "Leadership"),
                    internal.get("approach", "subtle"),
                    internal.get("public_knowledge", False),
                    "brewing",  # Initial phase
                    10,  # Initial progress
                    conflict_id
                ))
                
                struggle_id = cursor.fetchone()[0]
                
                # Create faction members if specified
                if "faction_members" in internal:
                    for member in internal["faction_members"]:
                        cursor.execute("""
                            INSERT INTO FactionStruggleMembers
                            (struggle_id, npc_id, position, side, standing, 
                             loyalty_strength, reason)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            struggle_id,
                            member.get("npc_id", 0),
                            member.get("position", "Member"),
                            member.get("side", "neutral"),
                            member.get("standing", 50),
                            member.get("loyalty_strength", 50),
                            member.get("reason", "")
                        ))
                
                # Create ideological differences if specified
                if "ideological_differences" in internal:
                    for diff in internal["ideological_differences"]:
                        cursor.execute("""
                            INSERT INTO FactionIdeologicalDifferences
                            (struggle_id, issue, incumbent_position, challenger_position)
                            VALUES (%s, %s, %s, %s)
                        """, (
                            struggle_id,
                            diff.get("issue", ""),
                            diff.get("incumbent_position", ""),
                            diff.get("challenger_position", "")
                        ))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating internal faction conflicts: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
            conn.close()
    
    async def _generate_player_manipulation_attempts(
        self,
        conflict_id: int,
        stakeholder_npcs: List[Dict[str, Any]]
    ) -> None:
        """
        Generate manipulation attempts targeted at the player.
        
        Args:
            conflict_id: ID of the conflict
            stakeholder_npcs: List of NPC dictionaries
        """
        # Find eligible NPCs for manipulation attempts
        eligible_npcs = []
        for npc in stakeholder_npcs:
            # Female NPCs with high dominance or close relationship with player
            if (npc.get("sex", "female") == "female" and 
                (npc.get("dominance", 0) > 70 or 
                 npc.get("relationship_with_player", {}).get("closeness", 0) > 70)):
                eligible_npcs.append(npc)
        
        if not eligible_npcs:
            return
        
        # Generate manipulation attempts
        manipulation_types = ["domination", "blackmail", "seduction", "coercion", "bribery"]
        involvement_levels = ["observing", "participating", "leading"]
        factions = ["a", "b", "neutral"]
        
        for npc in eligible_npcs[:2]:  # Limit to top 2 eligible NPCs
            # Skip if random check fails (not all NPCs will attempt manipulation)
            if random.random() > 0.7:
                continue
            
            # Select manipulation type based on NPC traits
            if npc.get("dominance", 0) > 80:
                manipulation_type = "domination"
            elif npc.get("cruelty", 0) > 70:
                manipulation_type = "blackmail"
            elif npc.get("relationship_with_player", {}).get("closeness", 0) > 80:
                manipulation_type = "seduction"
            else:
                manipulation_type = random.choice(manipulation_types)
            
            # Select involvement level and faction based on NPC relationship
            involvement_level = random.choice(involvement_levels)
            faction = random.choice(factions)
            
            # Generate content based on manipulation type
            content = self._generate_manipulation_content(
                npc, manipulation_type, involvement_level, faction
            )
            
            # Generate goal
            goal = {
                "faction": faction,
                "involvement_level": involvement_level,
                "specific_actions": random.choice([
                    "Spy on rival faction",
                    "Convince another NPC to join their side",
                    "Sabotage rival faction's plans",
                    "Gather information about a specific stakeholder"
                ])
            }
            
            # Generate leverage
            leverage = {
                "type": random.choice([
                    "dominance", "blackmail", "relationship", "seduction", "debt"
                ]),
                "strength": min(100, npc.get("dominance", 60) + random.randint(-10, 10))
            }
            
            # Determine intimacy level based on relationship and manipulation type
            intimacy_level = 0
            if manipulation_type == "seduction":
                intimacy_level = min(10, (npc.get("relationship_with_player", {}).get("closeness", 0) // 10) + 3)
            elif manipulation_type == "domination":
                intimacy_level = min(10, (npc.get("dominance", 0) // 10) - 2)
            
            # Create the manipulation attempt
            await self.create_player_manipulation_attempt(
                conflict_id, npc["npc_id"], manipulation_type,
                content, goal, leverage, intimacy_level
            )
    
    def _generate_manipulation_content(
        self,
        npc: Dict[str, Any],
        manipulation_type: str,
        involvement_level: str,
        faction: str
    ) -> str:
        """
        Generate content for a manipulation attempt.
        
        Args:
            npc: NPC dictionary
            manipulation_type: Type of manipulation
            involvement_level: Desired involvement level
            faction: Desired faction
            
        Returns:
            Manipulation content text
        """
        npc_name = npc.get("npc_name", "Unknown")
        
        if manipulation_type == "domination":
            return random.choice([
                f"{npc_name} commands you to help with the conflict. 'You will support {faction} faction at the {involvement_level} level. That's an order.'",
                f"'I expect your complete obedience in this matter,' {npc_name} states firmly. 'You will {involvement_level} for {faction} faction, or there will be consequences.'",
                f"{npc_name} fixes you with a stern gaze. 'This isn't a request. You will help {faction} faction by {involvement_level}. I won't tolerate disobedience.'"
            ])
        elif manipulation_type == "blackmail":
            return random.choice([
                f"{npc_name} smiles coldly. 'I know about your secrets. Help {faction} faction by {involvement_level}, or everyone else will know too.'",
                f"'It would be a shame if certain information about you became public,' {npc_name} says casually. 'Support {faction} faction by {involvement_level}, and your secret stays safe.'",
                f"{npc_name} shows you evidence of your indiscretions. 'This remains between us if you help {faction} faction by {involvement_level}. Otherwise...'"
            ])
        elif manipulation_type == "seduction":
            return random.choice([
                f"{npc_name} traces a finger along your cheek. 'Help {faction} faction by {involvement_level}, and I'll make it worth your while,' she purrs.",
                f"'I can offer special rewards for your help,' {npc_name} whispers, pressing close to you. 'Just support {faction} faction by {involvement_level}.'",
                f"{npc_name} gives you a smoldering look. 'Imagine what pleasures await if you help {faction} faction by {involvement_level}. Don't you want to please me?'"
            ])
        elif manipulation_type == "coercion":
            return random.choice([
                f"{npc_name} makes it clear that refusing isn't an option. 'You will help {faction} faction by {involvement_level}, or face the consequences.'",
                f"'Let me be perfectly clear,' {npc_name} says coldly. 'Support {faction} faction by {involvement_level}, or find yourself with powerful enemies.'",
                f"{npc_name} explains exactly how your life could become difficult if you don't help {faction} faction by {involvement_level}."
            ])
        else:  # bribery or other
            return random.choice([
                f"{npc_name} offers various incentives if you help {faction} faction by {involvement_level}. 'The rewards will be substantial.'",
                f"'Let's talk about what you stand to gain,' {npc_name} says with a knowing smile. 'Help {faction} faction by {involvement_level}, and these benefits are yours.'",
                f"{npc_name} outlines the advantages you'll receive for supporting {faction} faction by {involvement_level}. 'A mutually beneficial arrangement.'"
            ])
    
    async def _generate_struggle_details(
        self,
        faction_id: int,
        challenger_npc_id: int,
        target_npc_id: int,
        prize: str,
        approach: str
    ) -> Dict[str, Any]:
        """
        Generate details for a faction power struggle.
        
        Args:
            faction_id: ID of the faction
            challenger_npc_id: ID of the challenging NPC
            target_npc_id: ID of the target NPC
            prize: What's at stake
            approach: How the challenge is made
            
        Returns:
            Dictionary with struggle details
        """
        # Get faction name
        faction_name = await self._get_faction_name(faction_id)
        
        # Get NPC names
        challenger_name = await self._get_npc_name(challenger_npc_id)
        target_name = await self._get_npc_name(target_npc_id)
        
        # Get faction members
        members = await self._get_faction_members(faction_id)
        
        # Generate a conflict name and description
        conflict_name = f"Power struggle in {faction_name}"
        description = f"{challenger_name} challenges {target_name} for {prize} within {faction_name}."
        
        # Divide members between challenger, target, and neutral
        from collections import defaultdict
        sides = defaultdict(list)
        
        for member in members:
            # Skip challenger and target
            if member["npc_id"] == challenger_npc_id or member["npc_id"] == target_npc_id:
                continue
            
            # Assign based on relationships and random chance
            affinity_to_challenger = random.randint(0, 100)
            affinity_to_target = random.randint(0, 100)
            
            if abs(affinity_to_challenger - affinity_to_target) < 20:
                side = "neutral"
            elif affinity_to_challenger > affinity_to_target:
                side = "challenger"
            else:
                side = "incumbent"
            
            sides[side].append(member)
        
        # Create faction members list with positions
        faction_members = [
            {
                "npc_id": challenger_npc_id,
                "position": "Challenger",
                "side": "challenger",
                "standing": 70,
                "loyalty_strength": 100,
                "reason": "Leading the challenge"
            },
            {
                "npc_id": target_npc_id,
                "position": "Incumbent",
                "side": "incumbent",
                "standing": 80,
                "loyalty_strength": 100,
                "reason": "Defending position"
            }
        ]
        
        # Add supporters
        for side, members_list in sides.items():
            for i, member in enumerate(members_list):
                faction_members.append({
                    "npc_id": member["npc_id"],
                    "position": member.get("position", "Member"),
                    "side": side,
                    "standing": random.randint(30, 70),
                    "loyalty_strength": random.randint(40, 90),
                    "reason": f"Supports {side}"
                })
        
        # Generate ideological differences
        ideological_differences = [
            {
                "issue": f"Approach to {prize}",
                "incumbent_position": f"{target_name}'s traditional approach",
                "challenger_position": f"{challenger_name}'s new vision"
            },
            {
                "issue": "Faction methodology",
                "incumbent_position": "Maintain current methods",
                "challenger_position": "Implement reforms"
            }
        ]
        
        # Create the full struggle details
        struggle_details = {
            "conflict_name": conflict_name,
            "description": description,
            "faction_members": faction_members,
            "ideological_differences": ideological_differences
        }
        
        return struggle_details
    
    async def _get_faction_members(self, faction_id: int) -> List[Dict[str, Any]]:
        """Get members of a faction."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # This assumes NPCs store faction affiliations in a JSON array field
            cursor.execute("""
                SELECT npc_id, npc_name, dominance, cruelty, faction_affiliations
                FROM NPCStats
                WHERE user_id = %s AND conversation_id = %s
            """, (self.user_id, self.conversation_id))
            
            members = []
            for row in cursor.fetchall():
                npc_id, npc_name, dominance, cruelty, faction_affiliations = row
                
                # Parse faction affiliations
                try:
                    affiliations = json.loads(faction_affiliations) if isinstance(faction_affiliations, str) else faction_affiliations or []
                except (json.JSONDecodeError, TypeError):
                    affiliations = []
                
                # Check if NPC is affiliated with this faction
                is_member = False
                position = "Member"
                
                for affiliation in affiliations:
                    if affiliation.get("faction_id") == faction_id:
                        is_member = True
                        position = affiliation.get("position", "Member")
                        break
                
                if is_member:
                    members.append({
                        "npc_id": npc_id,
                        "npc_name": npc_name,
                        "dominance": dominance,
                        "cruelty": cruelty,
                        "position": position
                    })
            
            return members
        except Exception as e:
            logger.error(f"Error getting faction members: {e}", exc_info=True)
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def _calculate_coup_success_chance(
            self,
            struggle_id: int,
            approach: str,
            supporting_npcs: List[int],
            resources_committed: Dict[str, int]
        ) -> float:
            """
            Calculate the success chance of a coup attempt.
            
            Args:
                struggle_id: ID of the internal faction struggle
                approach: The approach used
                supporting_npcs: List of supporting NPC IDs
                resources_committed: Resources committed
                
            Returns:
                Success chance (0-100)
            """
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                # Get struggle details
                cursor.execute("""
                    SELECT primary_npc_id, target_npc_id
                    FROM InternalFactionConflicts
                    WHERE struggle_id = %s
                """, (struggle_id,))
                
                row = cursor.fetchone()
                if not row:
                    return 0
                
                primary_npc_id, target_npc_id = row
                
                # Get challenger and target stats
                challenger = await self._get_npc_stats(primary_npc_id)
                target = await self._get_npc_stats(target_npc_id)
                
                # Base success chance based on challenger vs target
                base_chance = 50 + (challenger.get("dominance", 50) - target.get("dominance", 50)) / 5
                
                # Adjust based on approach
                approach_modifiers = {
                    "direct": 0,       # Neutral modifier
                    "subtle": 10,      # Subtle approaches have advantage
                    "force": -5,       # Force is risky
                    "blackmail": 15    # Blackmail has high success chance
                }
                base_chance += approach_modifiers.get(approach, 0)
                
                # Adjust for supporting NPCs
                support_power = 0
                for npc_id in supporting_npcs:
                    npc = await self._get_npc_stats(npc_id)
                    support_power += npc.get("dominance", 50) / 10
                
                base_chance += min(25, support_power)  # Cap at +25
                
                # Adjust for resources committed
                resource_total = sum(resources_committed.values())
                resource_modifier = min(15, resource_total / 10)  # Cap at +15
                base_chance += resource_modifier
                
                # Get faction members and their loyalty to incumbent
                cursor.execute("""
                    SELECT npc_id, loyalty_strength
                    FROM FactionStruggleMembers
                    WHERE struggle_id = %s AND side = 'incumbent'
                """, (struggle_id,))
                
                total_loyalty = 0
                for row in cursor.fetchall():
                    _, loyalty = row
                    total_loyalty += loyalty
                
                # Loyalty to incumbent reduces success chance
                loyalty_modifier = min(30, total_loyalty / 20)  # Cap at -30
                base_chance -= loyalty_modifier
                
                # Ensure chance is between 5 and 95
                return max(5, min(95, base_chance))
            except Exception as e:
                logger.error(f"Error calculating coup success chance: {e}", exc_info=True)
                return 30  # Default moderate chance on error
            finally:
                cursor.close()
                conn.close()
        
        async def _get_npc_stats(self, npc_id: int) -> Dict[str, Any]:
            """Get basic stats for an NPC."""
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT npc_name, dominance, cruelty, closeness, trust, respect
                    FROM NPCStats
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                """, (npc_id, self.user_id, self.conversation_id))
                
                row = cursor.fetchone()
                
                if not row:
                    return {"dominance": 50, "cruelty": 50}
                
                npc_name, dominance, cruelty, closeness, trust, respect = row
                
                return {
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "closeness": closeness,
                    "trust": trust,
                    "respect": respect
                }
            except Exception as e:
                logger.error(f"Error getting NPC stats for ID {npc_id}: {e}", exc_info=True)
                return {"dominance": 50, "cruelty": 50}
            finally:
                cursor.close()
                conn.close()
        
        async def _check_conflict_advancement(self, conflict_id: int) -> None:
            """
            Check if a conflict should advance to the next phase.
            
            Args:
                conflict_id: ID of the conflict to check
            """
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                # Get conflict details
                cursor.execute("""
                    SELECT progress, phase
                    FROM Conflicts
                    WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
                """, (conflict_id, self.user_id, self.conversation_id))
                
                row = cursor.fetchone()
                if not row:
                    return
                
                progress, phase = row
                
                # Phase transition thresholds
                phase_thresholds = {
                    "brewing": 30,    # brewing -> active
                    "active": 60,     # active -> climax
                    "climax": 90      # climax -> resolution
                }
                
                # Check if we should transition to a new phase
                new_phase = phase
                if phase in phase_thresholds and progress >= phase_thresholds[phase]:
                    if phase == "brewing":
                        new_phase = "active"
                    elif phase == "active":
                        new_phase = "climax"
                    elif phase == "climax":
                        new_phase = "resolution"
                
                # If phase changed, update the conflict
                if new_phase != phase:
                    cursor.execute("""
                        UPDATE Conflicts
                        SET phase = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
                    """, (new_phase, conflict_id, self.user_id, self.conversation_id))
                    
                    # Create a memory for the phase transition
                    await self._create_conflict_memory(
                        conflict_id,
                        f"The conflict has progressed from {phase} to {new_phase} phase.",
                        significance=7
                    )
                    
                    conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Error checking conflict advancement: {e}", exc_info=True)
            finally:
                cursor.close()
                conn.close()
