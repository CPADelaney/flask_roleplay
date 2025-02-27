# logic.integrated_npc_management.py

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

# Import existing modules to utilize their functionality
from db.connection import get_db_connection
from logic.npc_creation import (
    create_npc_partial, 
    insert_npc_stub_into_db, 
    assign_random_relationships,
    gpt_generate_physical_description,
    gpt_generate_schedule,
    gpt_generate_memories,
    gpt_generate_affiliations,
    integrate_femdom_elements,
    propagate_shared_memories
)
from logic.social_links import (
    create_social_link,
    update_link_type_and_level,
    add_link_event,
    get_relationship_dynamic_level,
    update_relationship_dynamic,
    check_for_relationship_crossroads,
    check_for_relationship_ritual,
    apply_crossroads_choice,
    EnhancedRelationshipManager
)
from logic.time_cycle import (
    advance_time_with_events,
    get_current_time,
    set_current_time,
    update_npc_schedules_for_time,
    TIME_PHASES,
    ActivityManager
)
from logic.memory_logic import (
    record_npc_event,
    get_shared_memory,
    MemoryManager,
    EnhancedMemory,
    MemoryType,
    MemorySignificance,
    ProgressiveRevealManager
)
from logic.stats_logic import (
    apply_stat_change,
    apply_activity_effects,
    get_player_current_tier,
    check_for_combination_triggers,
    record_stat_change_event,
    STAT_THRESHOLDS,
    STAT_COMBINATIONS
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedNPCSystem:
    """
    Central system that integrates NPC creation, social dynamics, time management,
    memory systems, and stat progression.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.activity_manager = ActivityManager()
    
    #=================================================================
    # NPC CREATION AND MANAGEMENT
    #=================================================================
    
    async def create_new_npc(self, environment_desc: str, day_names: List[str], sex: str = "female") -> int:
        """
        Create a new NPC using the enhanced creation process.
        
        Args:
            environment_desc: Description of the environment
            day_names: List of day names used in the calendar
            sex: Sex of the NPC ("female" by default)
            
        Returns:
            The newly created NPC ID
        """
        # Step 1: Create the partial NPC (base data)
        partial_npc = create_npc_partial(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            sex=sex,
            total_archetypes=4,
            environment_desc=environment_desc
        )
        
        # Step 1.5: Integrate subtle femdom elements based on dominance
        partial_npc = await integrate_femdom_elements(partial_npc)
        
        # Step 2: Insert the partial NPC into the database
        npc_id = await insert_npc_stub_into_db(
            partial_npc, self.user_id, self.conversation_id
        )
        logger.info(f"Created NPC stub with ID {npc_id} and name {partial_npc['npc_name']}")
        
        # Step 3: Assign relationships
        await assign_random_relationships(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            new_npc_id=npc_id,
            new_npc_name=partial_npc["npc_name"],
            npc_archetypes=partial_npc.get("archetypes", [])
        )
        
        # Step 4: Get relationships for memory generation
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT relationships FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
            (self.user_id, self.conversation_id, npc_id)
        )
        row = cursor.fetchone()
        if row and row[0]:
            if isinstance(row[0], str):
                relationships = json.loads(row[0])
            else:
                relationships = row[0]
        else:
            relationships = []
        conn.close()

        # Step 5: Generate enhanced fields using GPT
        physical_description = await gpt_generate_physical_description(
            self.user_id, self.conversation_id, partial_npc, environment_desc
        )
        schedule = await gpt_generate_schedule(
            self.user_id, self.conversation_id, partial_npc, environment_desc, day_names
        )
        memories = await gpt_generate_memories(
            self.user_id, self.conversation_id, partial_npc, environment_desc, relationships
        )
        affiliations = await gpt_generate_affiliations(
            self.user_id, self.conversation_id, partial_npc, environment_desc
        )
        
        # Step 6: Determine current location based on time of day and schedule
        current_year, current_month, current_day, time_of_day = get_current_time(
            self.user_id, self.conversation_id
        )
        
        # Calculate day index
        day_index = (current_day - 1) % len(day_names)
        current_day_name = day_names[day_index]
        
        # Extract current location from schedule
        current_location = "Unknown"
        if schedule and current_day_name in schedule and time_of_day in schedule[current_day_name]:
            activity = schedule[current_day_name][time_of_day]
            # Extract location from activity description
            location_keywords = ["at the", "in the", "at", "in"]
            for keyword in location_keywords:
                if keyword in activity:
                    parts = activity.split(keyword, 1)
                    if len(parts) > 1:
                        potential_location = parts[1].split(".")[0].split(",")[0].strip()
                        if len(potential_location) > 3:  # Avoid very short fragments
                            current_location = potential_location
                            break
        
        # If we couldn't extract a location, use a random location from the database
        if current_location == "Unknown":
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT location_name FROM Locations WHERE user_id=%s AND conversation_id=%s ORDER BY RANDOM() LIMIT 1",
                (self.user_id, self.conversation_id)
            )
            random_location = cursor.fetchone()
            if random_location:
                current_location = random_location[0]
            conn.close()
        
        # Step 7: Update the NPC with all refined data
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE NPCStats 
            SET physical_description=%s,
                schedule=%s,
                memory=%s,
                current_location=%s,
                affiliations=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (
            physical_description,
            json.dumps(schedule),
            json.dumps(memories),
            current_location,
            json.dumps(affiliations),
            self.user_id, self.conversation_id, npc_id
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully refined NPC {npc_id} ({partial_npc['npc_name']})")
        
        # Step 8: Propagate memories to other connected NPCs
        propagate_shared_memories(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            source_npc_id=npc_id,
            source_npc_name=partial_npc["npc_name"],
            memories=memories
        )
        
        # Step 9: Initialize mask for the NPC
        await ProgressiveRevealManager.initialize_npc_mask(
            self.user_id, self.conversation_id, npc_id
        )
        
        return npc_id
    
    async def create_multiple_npcs(self, environment_desc: str, day_names: List[str], count: int = 3) -> List[int]:
        """
        Create multiple NPCs in the system.
        
        Args:
            environment_desc: Description of the environment
            day_names: List of day names used in the calendar
            count: Number of NPCs to create
            
        Returns:
            List of created NPC IDs
        """
        npc_ids = []
        for i in range(count):
            npc_id = await self.create_new_npc(environment_desc, day_names)
            npc_ids.append(npc_id)
            # Add a small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        return npc_ids
    
    async def get_npc_details(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an NPC.
        
        Args:
            npc_id: The ID of the NPC to retrieve
            
        Returns:
            Dictionary with NPC details or None if not found
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT npc_id, npc_name, introduced, sex, dominance, cruelty, 
                       closeness, trust, respect, intensity, archetype_summary,
                       physical_description, current_location, memory
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (self.user_id, self.conversation_id, npc_id))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            npc_id, npc_name, introduced, sex, dominance, cruelty, closeness, trust, respect, intensity, archetype_summary, physical_description, current_location, memory_json = row
            
            # Parse memory
            memories = []
            if memory_json:
                if isinstance(memory_json, str):
                    try:
                        memories = json.loads(memory_json)
                    except json.JSONDecodeError:
                        memories = []
                else:
                    memories = memory_json
            
            # Get mask information
            mask_info = await ProgressiveRevealManager.get_npc_mask(
                self.user_id, self.conversation_id, npc_id
            )
            
            # Get social links
            cursor.execute("""
                SELECT link_id, entity2_type, entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s 
                AND entity1_type='npc' AND entity1_id=%s
            """, (self.user_id, self.conversation_id, npc_id))
            
            links = []
            for link_row in cursor.fetchall():
                link_id, entity2_type, entity2_id, link_type, link_level = link_row
                
                # Get target name
                target_name = "Unknown"
                if entity2_type == "player":
                    target_name = "Chase"
                elif entity2_type == "npc":
                    cursor.execute("""
                        SELECT npc_name FROM NPCStats
                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                    """, (self.user_id, self.conversation_id, entity2_id))
                    name_row = cursor.fetchone()
                    if name_row:
                        target_name = name_row[0]
                
                links.append({
                    "link_id": link_id,
                    "target_type": entity2_type,
                    "target_id": entity2_id,
                    "target_name": target_name,
                    "link_type": link_type,
                    "link_level": link_level
                })
            
            # Build response
            npc_details = {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "introduced": introduced,
                "sex": sex,
                "stats": {
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "closeness": closeness,
                    "trust": trust,
                    "respect": respect,
                    "intensity": intensity
                },
                "archetype_summary": archetype_summary,
                "physical_description": physical_description,
                "current_location": current_location,
                "memories": memories[:5],  # Latest 5 memories
                "memory_count": len(memories),
                "mask": mask_info if mask_info and "error" not in mask_info else {"integrity": 100},
                "relationships": links
            }
            
            return npc_details
            
        except Exception as e:
            logger.error(f"Error getting NPC details: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    
    async def introduce_npc(self, npc_id: int) -> bool:
        """
        Mark an NPC as introduced.
        
        Args:
            npc_id: The ID of the NPC to introduce
            
        Returns:
            True if successful, False otherwise
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE NPCStats
                SET introduced=TRUE
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                RETURNING npc_name
            """, (self.user_id, self.conversation_id, npc_id))
            
            row = cursor.fetchone()
            if not row:
                return False
                
            npc_name = row[0]
            
            # Add to player journal
            cursor.execute("""
                INSERT INTO PlayerJournal 
                (user_id, conversation_id, entry_type, entry_text, timestamp)
                VALUES (%s, %s, 'npc_introduction', %s, CURRENT_TIMESTAMP)
            """, (
                self.user_id, self.conversation_id,
                f"Met {npc_name} for the first time."
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error introducing NPC: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    #=================================================================
    # SOCIAL LINKS AND RELATIONSHIPS
    #=================================================================
    
    async def create_relationship(self, 
                                  entity1_type: str, entity1_id: int,
                                  entity2_type: str, entity2_id: int,
                                  relationship_type: str = None, 
                                  initial_level: int = 0) -> Dict[str, Any]:
        """
        Create a new relationship between two entities.
        
        Args:
            entity1_type: Type of first entity ("npc" or "player")
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            relationship_type: Type of relationship
            initial_level: Initial relationship level
            
        Returns:
            Dictionary with result information
        """
        # Use the EnhancedRelationshipManager to create the relationship
        result = await EnhancedRelationshipManager.create_relationship(
            self.user_id, self.conversation_id,
            entity1_type, entity1_id,
            entity2_type, entity2_id,
            relationship_type, initial_level
        )
        
        return result
    
    async def update_relationship_dimensions(self, 
                                           link_id: int, 
                                           dimension_changes: Dict[str, int],
                                           reason: str = None) -> Dict[str, Any]:
        """
        Update dimensions of a relationship.
        
        Args:
            link_id: ID of the relationship link
            dimension_changes: Dictionary of dimension changes (e.g., {"trust": +5, "fear": -2})
            reason: Reason for the changes
            
        Returns:
            Dictionary with update results
        """
        # Use EnhancedRelationshipManager to update dimensions
        result = await EnhancedRelationshipManager.update_relationship_dimensions(
            self.user_id, self.conversation_id, link_id, dimension_changes,
            add_history_event=reason
        )
        
        return result
    
    async def increase_relationship_tension(self, link_id: int, amount: int, reason: str = None) -> Dict[str, Any]:
        """
        Increase tension in a relationship.
        
        Args:
            link_id: ID of the relationship link
            amount: Amount to increase tension by
            reason: Reason for the tension increase
            
        Returns:
            Dictionary with update results
        """
        result = await EnhancedRelationshipManager.increase_relationship_tension(
            self.user_id, self.conversation_id, link_id, amount, reason
        )
        
        return result
    
    async def release_relationship_tension(self, 
                                         link_id: int, 
                                         amount: int, 
                                         resolution_type: str = "positive",
                                         reason: str = None) -> Dict[str, Any]:
        """
        Release tension in a relationship.
        
        Args:
            link_id: ID of the relationship link
            amount: Amount to decrease tension by
            resolution_type: Type of resolution ("positive", "negative", "dominance", "submission")
            reason: Reason for the tension release
            
        Returns:
            Dictionary with update results
        """
        result = await EnhancedRelationshipManager.release_relationship_tension(
            self.user_id, self.conversation_id, link_id, amount, resolution_type, reason
        )
        
        return result
    
    async def check_for_relationship_events(self) -> List[Dict[str, Any]]:
        """
        Check for significant relationship events that might occur.
        
        Returns:
            List of event dictionaries
        """
        events = []
        
        # Check for relationship crossroads
        crossroads = await check_for_relationship_crossroads(self.user_id, self.conversation_id)
        if crossroads:
            events.append({
                "type": "relationship_crossroads",
                "data": crossroads
            })
        
        # Check for relationship rituals
        ritual = await check_for_relationship_ritual(self.user_id, self.conversation_id)
        if ritual:
            events.append({
                "type": "relationship_ritual",
                "data": ritual
            })
        
        # Check for relationship stage changes
        await EnhancedRelationshipManager.detect_relationship_stage_changes(
            self.user_id, self.conversation_id
        )
        
        return events
    
    async def apply_crossroads_choice(self, link_id: int, crossroads_name: str, choice_index: int) -> Dict[str, Any]:
        """
        Apply a choice in a relationship crossroads.
        
        Args:
            link_id: ID of the relationship link
            crossroads_name: Name of the crossroads
            choice_index: Index of the selected choice
            
        Returns:
            Dictionary with the results
        """
        result = await apply_crossroads_choice(
            self.user_id, self.conversation_id, link_id, crossroads_name, choice_index
        )
        
        return result
    
    #=================================================================
    # MEMORY MANAGEMENT
    #=================================================================
    
    async def add_memory_to_npc(self, 
                               npc_id: int, 
                               memory_text: str,
                               memory_type: str = "interaction",
                               significance: int = 3,
                               emotional_valence: int = 0,
                               tags: List[str] = None) -> bool:
        """
        Add a memory to an NPC.
        
        Args:
            npc_id: ID of the NPC
            memory_text: Text of the memory
            memory_type: Type of memory
            significance: Significance level (1-10)
            emotional_valence: Emotional impact (-10 to +10)
            tags: List of tags for the memory
            
        Returns:
            True if successful, False otherwise
        """
        # Use the MemoryManager to add the memory
        result = await MemoryManager.add_memory(
            self.user_id, self.conversation_id,
            npc_id, "npc",
            memory_text, memory_type,
            significance, emotional_valence,
            tags
        )
        
        return result
    
    async def retrieve_relevant_memories(self, 
                                       npc_id: int, 
                                       context: str = None,
                                       tags: List[str] = None,
                                       limit: int = 5) -> List[EnhancedMemory]:
        """
        Retrieve memories relevant to a context.
        
        Args:
            npc_id: ID of the NPC
            context: Context to retrieve memories for
            tags: List of tags to filter by
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory objects
        """
        memories = await MemoryManager.retrieve_relevant_memories(
            self.user_id, self.conversation_id,
            npc_id, "npc",
            context, tags, limit
        )
        
        return memories
    
    async def generate_flashback(self, npc_id: int, current_context: str) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback for an NPC.
        
        Args:
            npc_id: ID of the NPC
            current_context: Current context that may trigger a flashback
            
        Returns:
            Flashback data or None if no flashback was generated
        """
        flashback = await MemoryManager.generate_flashback(
            self.user_id, self.conversation_id, npc_id, current_context
        )
        
        return flashback
    
    async def propagate_significant_memory(self, 
                                         source_npc_id: int,
                                         memory_text: str,
                                         memory_type: str = "emotional",
                                         significance: int = 5,
                                         emotional_valence: int = 0) -> bool:
        """
        Propagate a significant memory to related NPCs.
        
        Args:
            source_npc_id: ID of the source NPC
            memory_text: Text of the memory
            memory_type: Type of memory
            significance: Significance level (1-10)
            emotional_valence: Emotional impact (-10 to +10)
            
        Returns:
            True if successful, False otherwise
        """
        # First, add the memory to the source NPC
        await self.add_memory_to_npc(
            source_npc_id, memory_text, memory_type, significance, emotional_valence
        )
        
        # Then propagate to related NPCs
        result = await MemoryManager.propagate_significant_memory(
            self.user_id, self.conversation_id,
            source_npc_id, "npc",
            EnhancedMemory(
                memory_text, memory_type, significance
            )
        )
        
        return result
    
    #=================================================================
    # MASK REVELATIONS AND PROGRESSIVE EVOLUTION
    #=================================================================
    
    async def initialize_npc_mask(self, npc_id: int, overwrite: bool = False) -> Dict[str, Any]:
        """
        Initialize a mask for an NPC.
        
        Args:
            npc_id: ID of the NPC
            overwrite: Whether to overwrite an existing mask
            
        Returns:
            Dictionary with result information
        """
        result = await ProgressiveRevealManager.initialize_npc_mask(
            self.user_id, self.conversation_id, npc_id, overwrite
        )
        
        return result
    
    async def get_npc_mask(self, npc_id: int) -> Dict[str, Any]:
        """
        Get mask information for an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with mask information
        """
        result = await ProgressiveRevealManager.get_npc_mask(
            self.user_id, self.conversation_id, npc_id
        )
        
        return result
    
    async def generate_mask_slippage(self, 
                                   npc_id: int, 
                                   trigger: str = None,
                                   severity: int = None,
                                   reveal_type: str = None) -> Dict[str, Any]:
        """
        Generate a mask slippage event for an NPC.
        
        Args:
            npc_id: ID of the NPC
            trigger: What triggered the slippage
            severity: Severity level of the slippage
            reveal_type: Type of revelation
            
        Returns:
            Dictionary with slippage information
        """
        result = await ProgressiveRevealManager.generate_mask_slippage(
            self.user_id, self.conversation_id, npc_id, trigger, severity, reveal_type
        )
        
        return result
    
    async def check_for_mask_slippage(self, npc_id: int) -> Optional[List[Dict[str, Any]]]:
        """
        Check if an NPC has reached thresholds where their true nature begins to show.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            List of slippage events or None
        """
        result = await ProgressiveRevealManager.check_for_mask_slippage(
            self.user_id, self.conversation_id, npc_id
        )
        
        return result
    
    #=================================================================
    # TIME MANAGEMENT AND ACTIVITIES
    #=================================================================
    
    async def get_current_game_time(self) -> Tuple[int, int, int, str]:
        """
        Get the current game time.
        
        Returns:
            Tuple of (year, month, day, time_of_day)
        """
        return get_current_time(self.user_id, self.conversation_id)
    
    async def set_game_time(self, year: int, month: int, day: int, time_of_day: str) -> bool:
        """
        Set the game time.
        
        Args:
            year: Year to set
            month: Month to set
            day: Day to set
            time_of_day: Time of day to set
            
        Returns:
            True if successful
        """
        set_current_time(self.user_id, self.conversation_id, year, month, day, time_of_day)
        update_npc_schedules_for_time(self.user_id, self.conversation_id, day, time_of_day)
        return True
    
    async def advance_time_with_activity(self, activity_type: str) -> Dict[str, Any]:
        """
        Advance time based on an activity type.
        
        Args:
            activity_type: Type of activity
            
        Returns:
            Dictionary with results including any events that occurred
        """
        result = await advance_time_with_events(
            self.user_id, self.conversation_id, activity_type
        )
        
        return result
    
    async def process_player_activity(self, player_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a player's activity, determining if time should advance and handling events.
        
        Args:
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Dictionary with processing results
        """
        result = await self.activity_manager.process_activity(
            self.user_id, self.conversation_id, player_input, context
        )
        
        return result
    
    async def perform_npc_daily_activity(self, npc_id: int, time_of_day: str) -> bool:
        """
        Have an NPC perform activities during their daily schedule.
        
        Args:
            npc_id: ID of the NPC
            time_of_day: Current time of day
            
        Returns:
            True if successful, False otherwise
        """
        result = await ProgressiveRevealManager.perform_npc_daily_activity(
            self.user_id, self.conversation_id, npc_id, time_of_day
        )
        
        return result
    
    async def update_npc_schedules(self, day: int, time_of_day: str) -> bool:
        """
        Update all NPC schedules for the current time.
        
        Args:
            day: Current day
            time_of_day: Current time of day
            
        Returns:
            True if successful
        """
        update_npc_schedules_for_time(self.user_id, self.conversation_id, day, time_of_day)
        return True
    
    #=================================================================
    # STATS AND PROGRESSION
    #=================================================================
    
    async def apply_stat_changes(self, changes: Dict[str, int], cause: str = "") -> bool:
        """
        Apply multiple stat changes to the player.
        
        Args:
            changes: Dictionary of stat changes
            cause: Reason for the changes
            
        Returns:
            True if successful, False otherwise
        """
        result = apply_stat_change(self.user_id, self.conversation_id, changes, cause)
        return result
    
    async def apply_activity_effects(self, activity_name: str, intensity: float = 1.0) -> bool:
        """
        Apply stat changes based on a specific activity.
        
        Args:
            activity_name: Name of the activity
            intensity: Intensity multiplier
            
        Returns:
            True if successful, False otherwise
        """
        result = apply_activity_effects(
            self.user_id, self.conversation_id, activity_name, intensity
        )
        return result
    
    async def get_player_current_tier(self, stat_name: str) -> Optional[Dict[str, Any]]:
        """
        Determine which tier a player is in for a given stat.
        
        Args:
            stat_name: Name of the stat
            
        Returns:
            Threshold dictionary or None
        """
        result = get_player_current_tier(
            self.user_id, self.conversation_id, stat_name
        )
        return result
    
    async def check_for_combination_triggers(self) -> List[Dict[str, Any]]:
        """
        Check if player stats trigger any special combination states.
        
        Returns:
            List of triggered combinations
        """
        result = check_for_combination_triggers(
            self.user_id, self.conversation_id
        )
        return result
    
    #=================================================================
    # HIGH-LEVEL INTERACTION HANDLERS
    #=================================================================
    
    async def handle_npc_interaction(self, 
                                   npc_id: int, 
                                   interaction_type: str,
                                   player_input: str,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a complete interaction between player and NPC.
        
        This high-level method coordinates multiple subsystems:
        1. Processes the player activity
        2. Updates relationships based on interaction
        3. Generates appropriate memories
        4. Checks for mask slippage
        5. Applies stat effects
        6. Advances time if needed
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction (conversation, command, etc.)
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Comprehensive result dictionary
        """
        results = {
            "npc_id": npc_id,
            "interaction_type": interaction_type,
            "events": [],
            "memories_created": [],
            "stat_changes": {},
            "time_advanced": False
        }
        
        # Step 1: Get NPC details
        npc_details = await self.get_npc_details(npc_id)
        if not npc_details:
            return {"error": f"NPC with ID {npc_id} not found"}
        
        # Step 2: Process the activity and potentially advance time
        activity_result = await self.process_player_activity(player_input, context)
        results["activity_processed"] = activity_result
        
        if activity_result.get("time_advanced", False):
            results["time_advanced"] = True
            results["new_time"] = activity_result.get("new_time")
            
            # If time advanced, add any events that occurred
            for event in activity_result.get("events", []):
                results["events"].append(event)
        
        # Step 3: Update relationships based on interaction
        # Find link between player and this NPC
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT link_id FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
            AND ((entity1_type='player' AND entity1_id=%s AND entity2_type='npc' AND entity2_id=%s)
            OR (entity1_type='npc' AND entity1_id=%s AND entity2_type='player' AND entity2_id=%s))
        """, (
            self.user_id, self.conversation_id,
            self.user_id, npc_id,
            npc_id, self.user_id
        ))
        
        link_row = cursor.fetchone()
        conn.close()
        
        if link_row:
            link_id = link_row[0]
            
            # Update relationship dimensions based on interaction type
            dimension_changes = {}
            
            if interaction_type == "friendly_conversation":
                dimension_changes = {
                    "trust": +3,
                    "respect": +2
                }
            elif interaction_type == "defiant_response":
                dimension_changes = {
                    "tension": +5,
                    "respect": -2
                }
            elif interaction_type == "submissive_response":
                dimension_changes = {
                    "control": +5,
                    "dependency": +3
                }
            elif interaction_type == "flirtatious_remark":
                dimension_changes = {
                    "intimacy": +4,
                    "tension": +2
                }
            
            if dimension_changes:
                relationship_result = await self.update_relationship_dimensions(
                    link_id, dimension_changes, 
                    reason=f"Player interaction: {interaction_type}"
                )
                results["relationship_updated"] = relationship_result
        
        # Step 4: Generate appropriate memories
        memory_text = f"Interaction with player: {interaction_type}. The player said: '{player_input[:50]}...'"
        memory_result = await self.add_memory_to_npc(
            npc_id, memory_text, 
            memory_type=MemoryType.INTERACTION,
            significance=MemorySignificance.MEDIUM
        )
        results["memory_created"] = memory_result
        results["memories_created"].append(memory_text)
        
        # Step 5: Check for mask slippage
        if interaction_type in ["defiant_response", "probing_question"]:
            # Higher chance of slippage during confrontational interactions
            if random.random() < 0.4:  # 40% chance
                slippage_result = await self.generate_mask_slippage(
                    npc_id, trigger=f"Player {interaction_type}"
                )
                if slippage_result and "error" not in slippage_result:
                    results["mask_slippage"] = slippage_result
                    results["events"].append({
                        "type": "mask_slippage",
                        "data": slippage_result
                    })
        
        # Step 6: Apply stat effects to player
        stat_changes = {}
        
        # Base the stat changes on NPC's attributes and interaction type
        dominance = npc_details["stats"]["dominance"]
        cruelty = npc_details["stats"]["cruelty"]
        
        if interaction_type == "submissive_response":
            # Submitting to a dominant NPC increases corruption and obedience
            dominance_factor = dominance / 100  # 0.0 to 1.0
            stat_changes = {
                "corruption": int(2 + (dominance_factor * 3)),
                "obedience": int(3 + (dominance_factor * 4)),
                "willpower": -2,
                "confidence": -1
            }
        elif interaction_type == "defiant_response":
            # Defying increases willpower and confidence but may decrease other stats
            # More cruel NPCs cause more mental damage when defied
            cruelty_factor = cruelty / 100  # 0.0 to 1.0
            stat_changes = {
                "willpower": +3,
                "confidence": +2,
                "mental_resilience": int(-1 - (cruelty_factor * 3))
            }
        
        if stat_changes:
            stat_result = await self.apply_stat_changes(
                stat_changes, 
                cause=f"Interaction with {npc_details['npc_name']}: {interaction_type}"
            )
            results["stat_changes"] = stat_changes
        
        # Step 7: Check for relationship events
        relationship_events = await self.check_for_relationship_events()
        for event in relationship_events:
            results["events"].append(event)
        
        return results
    
    async def generate_multi_npc_scene(self, 
                                     npc_ids: List[int], 
                                     location: str = None,
                                     include_player: bool = True) -> Dict[str, Any]:
        """
        Generate a scene with multiple NPCs.
        
        Args:
            npc_ids: List of NPC IDs to include
            location: Location for the scene
            include_player: Whether to include the player
            
        Returns:
            Scene information
        """
        # Import MultiNPCInteractionManager for this specific functionality
        from logic.social_links import MultiNPCInteractionManager
        
        result = await MultiNPCInteractionManager.generate_multi_npc_scene(
            self.user_id, self.conversation_id,
            npc_ids, location, include_player
        )
        
        return result
    
    async def generate_overheard_conversation(self, 
                                           npc_ids: List[int],
                                           topic: str = None,
                                           about_player: bool = False) -> Dict[str, Any]:
        """
        Generate a conversation between NPCs that the player can overhear.
        
        Args:
            npc_ids: List of NPC IDs to include
            topic: Topic of conversation
            about_player: Whether the conversation is about the player
            
        Returns:
            Conversation information
        """
        # Import MultiNPCInteractionManager for this specific functionality
        from logic.social_links import MultiNPCInteractionManager
        
        result = await MultiNPCInteractionManager.generate_overheard_conversation(
            self.user_id, self.conversation_id,
            npc_ids, topic, about_player
        )
        
        return result
    
    async def create_npc_group(self, 
                             name: str, 
                             description: str, 
                             member_ids: List[int]) -> Dict[str, Any]:
        """
        Create a group of NPCs.
        
        Args:
            name: Name of the group
            description: Description of the group
            member_ids: List of NPC IDs to include
            
        Returns:
            Group information
        """
        # Import MultiNPCInteractionManager for this specific functionality
        from logic.social_links import MultiNPCInteractionManager
        
        result = await MultiNPCInteractionManager.create_npc_group(
            self.user_id, self.conversation_id,
            name, description, member_ids
        )
        
        return result

#=================================================================
# UTILITY FUNCTIONS
#=================================================================

def safe_json_loads(text, default=None):
    """Safely parse JSON with multiple fallback methods."""
    if not text:
        return default if default is not None else {}
    
    # Method 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Look for JSON object within text
    try:
        import re
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            return json.loads(json_match.group(1))
    except json.JSONDecodeError:
        pass
    
    # Method 3: Try to fix common JSON syntax errors
    try:
        # Replace single quotes with double quotes
        fixed_text = text.replace("'", '"')
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass
    
    # Return default if all parsing attempts fail
    return default if default is not None else {}

def extract_field_from_text(text, field_name):
    """
    Extract a specific field from text that might contain JSON or key-value patterns.
    Returns the field value or empty string if not found.
    """
    # Try parsing as JSON first
    data = safe_json_loads(text)
    if data and field_name in data:
        return data[field_name]
    
    # Try regex patterns for field extraction
    import re
    patterns = [
        rf'"{field_name}"\s*:\s*"([^"]*)"',      # For string values: "field": "value"
        rf'"{field_name}"\s*:\s*(\[[^\]]*\])',     # For array values: "field": [...]
        rf'"{field_name}"\s*:\s*(\{{[^}}]*\}})',  # For object values: "field": {...}
        rf'{field_name}:\s*(.*?)(?:\n|$)',          # For plain text: field: value
    ]

    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return ""

#=================================================================
# USAGE EXAMPLES
#=================================================================

async def example_usage():
    """Example usage of the IntegratedNPCSystem."""
    user_id = 1
    conversation_id = 123
    
    # Initialize the system
    npc_system = IntegratedNPCSystem(user_id, conversation_id)
    
    # Create a new NPC
    environment_desc = "A elegant mansion with sprawling gardens and opulent interior."
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    npc_id = await npc_system.create_new_npc(environment_desc, day_names)
    
    # Get NPC details
    npc_details = await npc_system.get_npc_details(npc_id)
    print(f"Created NPC: {npc_details['npc_name']}")
    
    # Introduce the NPC
    await npc_system.introduce_npc(npc_id)
    
    # Create a relationship between player and NPC
    relationship = await npc_system.create_relationship(
        "player", user_id, "npc", npc_id, "dominant"
    )
    
    # Handle a player interaction with the NPC
    interaction_result = await npc_system.handle_npc_interaction(
        npc_id, "submissive_response", "Yes, I'll do whatever you ask."
    )
    
    # Advance time
    time_result = await npc_system.advance_time_with_activity("extended_conversation")
    
    # Generate a multi-NPC scene
    scene = await npc_system.generate_multi_npc_scene([npc_id])
    
    print("NPC system demo completed successfully!")

if __name__ == "__main__":
    # Run the example usage
    import asyncio
    asyncio.run(example_usage())
