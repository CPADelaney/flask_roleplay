# logic/fully_integrated_npc_system.py

import os
import json
import logging
import asyncio
import random
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
        logger.info(f"Initialized IntegratedNPCSystem for user={user_id}, conversation={conversation_id}")
        logger.info(f"Available time phases: {TIME_PHASES}")
        
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
        logger.info(f"Creating new NPC in environment: {environment_desc[:30]}...")
        
        # Step 1: Create the partial NPC (base data) - Using create_npc_partial
        partial_npc = create_npc_partial(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            sex=sex,
            total_archetypes=4,
            environment_desc=environment_desc
        )
        
        # Step 1.5: Integrate subtle femdom elements - Using integrate_femdom_elements
        partial_npc = await integrate_femdom_elements(partial_npc)
        
        # Step 2: Insert the partial NPC into the database - Using insert_npc_stub_into_db
        npc_id = await insert_npc_stub_into_db(
            partial_npc, self.user_id, self.conversation_id
        )
        logger.info(f"Created NPC stub with ID {npc_id} and name {partial_npc['npc_name']}")
        
        # Step 3: Assign relationships - Using assign_random_relationships
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
        # Using gpt_generate_physical_description
        physical_description = await gpt_generate_physical_description(
            self.user_id, self.conversation_id, partial_npc, environment_desc
        )
        
        # Using gpt_generate_schedule
        schedule = await gpt_generate_schedule(
            self.user_id, self.conversation_id, partial_npc, environment_desc, day_names
        )
        
        # Using gpt_generate_memories
        memories = await gpt_generate_memories(
            self.user_id, self.conversation_id, partial_npc, environment_desc, relationships
        )
        
        # Using gpt_generate_affiliations
        affiliations = await gpt_generate_affiliations(
            self.user_id, self.conversation_id, partial_npc, environment_desc
        )
        
        # Step 6: Determine current location based on time of day and schedule
        # Using get_current_time
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
        
        # Step 8: Propagate memories to other connected NPCs - Using propagate_shared_memories
        propagate_shared_memories(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            source_npc_id=npc_id,
            source_npc_name=partial_npc["npc_name"],
            memories=memories
        )
        
        # Step 9: Initialize mask for the NPC - Using ProgressiveRevealManager
        await ProgressiveRevealManager.initialize_npc_mask(
            self.user_id, self.conversation_id, npc_id
        )
        
        # Step 10: Create a direct memory event - Using record_npc_event
        creation_memory = f"I was created on {current_year}-{current_month}-{current_day} during {time_of_day}."
        record_npc_event(
            self.user_id, self.conversation_id, npc_id, creation_memory
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
            
            # Create an introduction memory for the NPC - Using record_npc_event
            introduction_memory = f"I was formally introduced to the player today."
            record_npc_event(
                self.user_id, self.conversation_id, npc_id, introduction_memory
            )
            
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
    
    async def create_direct_social_link(self, 
                                      entity1_type: str, entity1_id: int,
                                      entity2_type: str, entity2_id: int,
                                      link_type: str = "neutral", 
                                      link_level: int = 0) -> int:
        """
        Create a direct social link between two entities.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            link_type: Type of link
            link_level: Level of the link
            
        Returns:
            The ID of the created link
        """
        # Using create_social_link
        link_id = create_social_link(
            self.user_id, self.conversation_id,
            entity1_type, entity1_id,
            entity2_type, entity2_id,
            link_type, link_level
        )
        
        logger.info(f"Created social link (ID: {link_id}) between {entity1_type}:{entity1_id} and {entity2_type}:{entity2_id}")
        return link_id
    
    async def update_link_details(self, link_id: int, new_type: str = None, level_change: int = 0) -> Dict[str, Any]:
        """
        Update the type and level of a social link.
        
        Args:
            link_id: ID of the link
            new_type: New type for the link (or None to keep current type)
            level_change: Amount to change the level by
            
        Returns:
            Dictionary with update results
        """
        # Using update_link_type_and_level
        result = update_link_type_and_level(
            self.user_id, self.conversation_id,
            link_id, new_type, level_change
        )
        
        if result:
            logger.info(f"Updated link {link_id}: type={result['new_type']}, level={result['new_level']}")
        
        return result
    
    async def add_event_to_link(self, link_id: int, event_text: str) -> bool:
        """
        Add an event to a social link's history.
        
        Args:
            link_id: ID of the link
            event_text: Text describing the event
            
        Returns:
            True if successful
        """
        # Using add_link_event
        add_link_event(
            self.user_id, self.conversation_id,
            link_id, event_text
        )
        
        logger.info(f"Added event to link {link_id}: {event_text[:50]}...")
        return True
    
    async def get_dynamic_level(self, 
                              entity1_type: str, entity1_id: int,
                              entity2_type: str, entity2_id: int,
                              dynamic_name: str) -> int:
        """
        Get the level of a specific relationship dynamic.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            dynamic_name: Name of the dynamic
            
        Returns:
            Current level of the dynamic
        """
        # Using get_relationship_dynamic_level
        level = get_relationship_dynamic_level(
            self.user_id, self.conversation_id,
            entity1_type, entity1_id,
            entity2_type, entity2_id,
            dynamic_name
        )
        
        logger.info(f"Dynamic '{dynamic_name}' level between {entity1_type}:{entity1_id} and {entity2_type}:{entity2_id} is {level}")
        return level
    
    async def update_dynamic(self, 
                           entity1_type: str, entity1_id: int,
                           entity2_type: str, entity2_id: int,
                           dynamic_name: str, change: int) -> int:
        """
        Update a specific relationship dynamic.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            dynamic_name: Name of the dynamic
            change: Amount to change the dynamic by
            
        Returns:
            New level of the dynamic
        """
        # Using update_relationship_dynamic
        new_level = update_relationship_dynamic(
            self.user_id, self.conversation_id,
            entity1_type, entity1_id,
            entity2_type, entity2_id,
            dynamic_name, change
        )
        
        logger.info(f"Updated dynamic '{dynamic_name}' to level {new_level}")
        return new_level
    
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
        
        # Using check_for_relationship_crossroads
        crossroads = await check_for_relationship_crossroads(self.user_id, self.conversation_id)
        if crossroads:
            events.append({
                "type": "relationship_crossroads",
                "data": crossroads
            })
        
        # Using check_for_relationship_ritual
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
        # Using apply_crossroads_choice
        result = await apply_crossroads_choice(
            self.user_id, self.conversation_id, link_id, crossroads_name, choice_index
        )
        
        return result
    
    #=================================================================
    # MEMORY MANAGEMENT
    #=================================================================
    
    async def record_memory_event(self, npc_id: int, event_description: str) -> bool:
        """
        Record a memory event for an NPC.
        
        Args:
            npc_id: ID of the NPC
            event_description: Description of the event
            
        Returns:
            True if successful
        """
        # Using record_npc_event
        record_npc_event(
            self.user_id, self.conversation_id, npc_id, event_description
        )
        logger.info(f"Recorded memory event for NPC {npc_id}: {event_description[:50]}...")
        return True
    
    async def generate_shared_memory(self, 
                                   npc_id: int, 
                                   target: str = "player", 
                                   target_name: str = "Chase", 
                                   rel_type: str = "related") -> str:
        """
        Generate a shared memory between an NPC and another entity.
        
        Args:
            npc_id: ID of the NPC
            target: Target type
            target_name: Target name
            rel_type: Relationship type
            
        Returns:
            Generated memory text
        """
        # Using get_shared_memory
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT npc_name, archetype_summary, archetype_extras_summary 
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (self.user_id, self.conversation_id, npc_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return "Memory generation failed: NPC not found."
            
        npc_name, archetype_summary, archetype_extras_summary = row
        
        # Create relationship object
        relationship = {
            "target": target,
            "target_name": target_name,
            "type": rel_type
        }
        
        memory_text = get_shared_memory(
            self.user_id, self.conversation_id,
            relationship, npc_name,
            archetype_summary, archetype_extras_summary
        )
        
        # Record the generated memory
        record_npc_event(
            self.user_id, self.conversation_id, npc_id, memory_text
        )
        
        return memory_text
    
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
        # Using MemoryManager.add_memory
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
        # Using MemoryManager.retrieve_relevant_memories
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
        # Using MemoryManager.generate_flashback
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
        
        # Using MemoryManager.propagate_significant_memory
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
        # Using ProgressiveRevealManager.initialize_npc_mask
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
        # Using ProgressiveRevealManager.get_npc_mask
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
        # Using ProgressiveRevealManager.generate_mask_slippage
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
        # Using ProgressiveRevealManager.check_for_mask_slippage
        result = await ProgressiveRevealManager.check_for_mask_slippage(
            self.user_id, self.conversation_id, npc_id
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
        # Using ProgressiveRevealManager.perform_npc_daily_activity
        result = await ProgressiveRevealManager.perform_npc_daily_activity(
            self.user_id, self.conversation_id, npc_id, time_of_day
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
        # Using get_current_time
        time_info = get_current_time(self.user_id, self.conversation_id)
        
        # Log using TIME_PHASES
        current_phase_idx = TIME_PHASES.index(time_info[3]) if time_info[3] in TIME_PHASES else 0
        next_phase_idx = (current_phase_idx + 1) % len(TIME_PHASES)
        logger.info(f"Current time phase: {time_info[3]}, Next phase: {TIME_PHASES[next_phase_idx]}")
        
        return time_info
    
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
        # Validate time_of_day using TIME_PHASES
        if time_of_day not in TIME_PHASES:
            logger.warning(f"Invalid time phase '{time_of_day}'. Using {TIME_PHASES[0]} instead.")
            time_of_day = TIME_PHASES[0]
        
        # Using set_current_time
        set_current_time(self.user_id, self.conversation_id, year, month, day, time_of_day)
        
        # Using update_npc_schedules_for_time
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
        # Using advance_time_with_events
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
        # Using ActivityManager.process_activity
        result = await self.activity_manager.process_activity(
            self.user_id, self.conversation_id, player_input, context
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
        # Using update_npc_schedules_for_time
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
        # Using apply_stat_change
        result = apply_stat_change(self.user_id, self.conversation_id, changes, cause)
        
        # Record each stat change separately - Using record_stat_change_event
        for stat_name, change_value in changes.items():
            # Get current value first
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT {stat_name} FROM PlayerStats 
                WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                old_value = row[0]
                new_value = old_value + change_value
                # Using record_stat_change_event
                record_stat_change_event(
                    self.user_id, self.conversation_id, 
                    stat_name, old_value, new_value, cause
                )
        
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
        # Using apply_activity_effects
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
        # Using get_player_current_tier
        result = get_player_current_tier(
            self.user_id, self.conversation_id, stat_name
        )
        
        # Using STAT_THRESHOLDS
        if result and stat_name in STAT_THRESHOLDS:
            logger.info(f"Player tier for {stat_name}: {result['name']} (level {result['level']})")
            logger.info(f"Possible tiers for {stat_name}: {[tier['name'] for tier in STAT_THRESHOLDS[stat_name]]}")
        
        return result
    
    async def check_for_combination_triggers(self) -> List[Dict[str, Any]]:
        """
        Check if player stats trigger any special combination states.
        
        Returns:
            List of triggered combinations
        """
        # Using check_for_combination_triggers
        result = check_for_combination_triggers(
            self.user_id, self.conversation_id
        )
        
        # Using STAT_COMBINATIONS
        if result:
            logger.info(f"Triggered combinations: {[combo['name'] for combo in result]}")
            logger.info(f"Available combinations: {[combo['name'] for combo in STAT_COMBINATIONS]}")
        
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
                # Using update_relationship_dimensions
                relationship_result = await self.update_relationship_dimensions(
                    link_id, dimension_changes, 
                    reason=f"Player interaction: {interaction_type}"
                )
                results["relationship_updated"] = relationship_result
                
                # Using add_link_event
                await self.add_event_to_link(
                    link_id,
                    f"Player interaction: {interaction_type} - '{player_input[:30]}...'"
                )
        
        # Step 4: Generate appropriate memories
        memory_text = f"Interaction with player: {interaction_type}. The player said: '{player_input[:50]}...'"
        
        # Using record_npc_event
        record_npc_event(
            self.user_id, self.conversation_id, npc_id, memory_text
        )
        
        # Also add using MemoryManager for richer memory tracking
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
                # Using generate_mask_slippage
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
            # Using apply_stat_changes
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
# USAGE EXAMPLES
#=================================================================

async def comprehensive_example():
    """Comprehensive usage example demonstrating all functions."""
    user_id = 1
    conversation_id = 123
    
    # Initialize the system
    npc_system = IntegratedNPCSystem(user_id, conversation_id)
    
    # 1. Create a new NPC
    environment_desc = "A elegant mansion with sprawling gardens and opulent interior."
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    npc_id = await npc_system.create_new_npc(environment_desc, day_names)
    
    # 2. Get NPC details
    npc_details = await npc_system.get_npc_details(npc_id)
    print(f"Created NPC: {npc_details['npc_name']}")
    
    # 3. Introduce the NPC
    await npc_system.introduce_npc(npc_id)
    
    # 4. Create a direct social link between player and NPC
    link_id = await npc_system.create_direct_social_link(
        "player", user_id, "npc", npc_id, "neutral", 10
    )
    
    # 5. Update link details
    await npc_system.update_link_details(link_id, "dominant", +20)
    
    # 6. Add event to link
    await npc_system.add_event_to_link(link_id, "Initial meeting and power assessment")
    
    # 7. Create relationship with dimensions
    relationship = await npc_system.create_relationship(
        "player", user_id, "npc", npc_id, "dominant"
    )
    
    # 8. Get dynamic level
    control_level = await npc_system.get_dynamic_level(
        "player", user_id, "npc", npc_id, "control"
    )
    print(f"Control level: {control_level}")
    
    # 9. Update dynamic
    new_control = await npc_system.update_dynamic(
        "player", user_id, "npc", npc_id, "control", +15
    )
    print(f"New control level: {new_control}")
    
    # 10. Update relationship dimensions
    await npc_system.update_relationship_dimensions(
        link_id, {"trust": +10, "fear": +5}, "Getting to know each other"
    )
    
    # 11. Add tension
    await npc_system.increase_relationship_tension(
        link_id, 25, "Uncomfortable power imbalance"
    )
    
    # 12. Release tension
    await npc_system.release_relationship_tension(
        link_id, 10, "submission", "Player acquiesces to demands"
    )
    
    # 13. Record memory event
    await npc_system.record_memory_event(
        npc_id, "The player was hesitant at first but eventually followed instructions."
    )
    
    # 14. Generate shared memory
    shared_memory = await npc_system.generate_shared_memory(
        npc_id, "player", "Chase", "dominant"
    )
    print(f"Generated shared memory: {shared_memory[:50]}...")
    
    # 15. Add enhanced memory
    await npc_system.add_memory_to_npc(
        npc_id, 
        "I noticed the player watching me carefully, analyzing my behaviors.",
        MemoryType.OBSERVATION,
        MemorySignificance.MEDIUM,
        +3,
        ["observation", "assessment"]
    )
    
    # 16. Retrieve relevant memories
    memories = await npc_system.retrieve_relevant_memories(
        npc_id, "watching behavior", ["observation"]
    )
    print(f"Retrieved {len(memories)} relevant memories")
    
    # 17. Generate flashback
    flashback = await npc_system.generate_flashback(
        npc_id, "The player is watching me again, just like that day..."
    )
    if flashback:
        print(f"Flashback generated: {flashback['text'][:50]}...")
    
    # 18. Propagate significant memory
    await npc_system.propagate_significant_memory(
        npc_id,
        "There was a significant incident where several NPCs witnessed the player show unexpected strength.",
        MemoryType.EMOTIONAL,
        MemorySignificance.HIGH,
        -5
    )
    
    # 19. Initialize mask
    mask_result = await npc_system.initialize_npc_mask(npc_id)
    print(f"Mask initialized with integrity: {mask_result.get('mask_created', False)}")
    
    # 20. Get mask info
    mask_info = await npc_system.get_npc_mask(npc_id)
    if mask_info:
        print(f"Mask integrity: {mask_info.get('integrity', 100)}")
    
    # 21. Generate mask slippage
    slippage = await npc_system.generate_mask_slippage(
        npc_id, "Player challenged authority", None, "verbal_slip"
    )
    if slippage:
        print(f"Mask slippage: {slippage.get('description', '')[:50]}...")
    
    # 22. Check for mask slippage
    slippage_events = await npc_system.check_for_mask_slippage(npc_id)
    print(f"Found {len(slippage_events) if slippage_events else 0} slippage events")
    
    # 23. Perform NPC daily activity
    await npc_system.perform_npc_daily_activity(npc_id, "Morning")
    
    # 24. Get current game time
    year, month, day, time_of_day = await npc_system.get_current_game_time()
    print(f"Current game time: Year {year}, Month {month}, Day {day}, {time_of_day}")
    
    # 25. Set game time
    await npc_system.set_game_time(year, month, day + 1, "Afternoon")
    
    # 26. Advance time with activity
    time_result = await npc_system.advance_time_with_activity("extended_conversation")
    print(f"Time advanced: {time_result.get('time_advanced', False)}")
    
    # 27. Process player activity
    activity_result = await npc_system.process_player_activity(
        "I want to spend some time getting to know the other residents."
    )
    print(f"Activity processed: {activity_result.get('activity_type', '')}")
    
    # 28. Update NPC schedules
    await npc_system.update_npc_schedules(day, time_of_day)
    
    # 29. Apply stat changes
    await npc_system.apply_stat_changes(
        {"corruption": +5, "confidence": -3},
        "Prolonged exposure to dominant NPC"
    )
    
    # 30. Apply activity effects
    await npc_system.apply_activity_effects("public_humiliation", 1.2)
    
    # 31. Get player current tier
    corruption_tier = await npc_system.get_player_current_tier("corruption")
    if corruption_tier:
        print(f"Corruption tier: {corruption_tier['name']} (level {corruption_tier['level']})")
    
    # 32. Check for combination triggers
    combinations = await npc_system.check_for_combination_triggers()
    print(f"Found {len(combinations)} stat combinations")
    
    # 33. Handle a complete NPC interaction
    interaction_result = await npc_system.handle_npc_interaction(
        npc_id, "submissive_response", "Yes, I'll do whatever you ask."
    )
    print(f"Interaction complete with {len(interaction_result.get('events', []))} events")
    
    # 34. Generate a multi-NPC scene
    scene = await npc_system.generate_multi_npc_scene([npc_id])
    if scene:
        print(f"Generated scene: {scene.get('opening_description', '')[:50]}...")
    
    # 35. Generate overheard conversation
    second_npc_id = await npc_system.create_new_npc(environment_desc, day_names)
    conversation = await npc_system.generate_overheard_conversation(
        [npc_id, second_npc_id], None, True
    )
    if conversation:
        print(f"Generated conversation: {conversation.get('conversation', [''])[0][:50]}...")
    
    # 36. Create NPC group
    group = await npc_system.create_npc_group(
        "Mansion Elite", "The dominant figures in the mansion hierarchy", 
        [npc_id, second_npc_id]
    )
    print(f"Created group with ID: {group.get('group_id')}")
    
    print("Comprehensive NPC system demo completed successfully!")

if __name__ == "__main__":
    # Run the comprehensive example
    import asyncio
    asyncio.run(comprehensive_example())
