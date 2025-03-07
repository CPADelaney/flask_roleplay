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
    add_link_event
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
    MemoryType,
    MemorySignificance
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

# Import agent-based architecture components
from logic.npc_agents.npc_agent import NPCAgent
from logic.npc_agents.agent_system import NPCAgentSystem
from logic.npc_agents.agent_coordinator import NPCAgentCoordinator
from logic.npc_agents.decision_engine import NPCDecisionEngine
from logic.npc_agents.relationship_manager import NPCRelationshipManager
from logic.npc_agents.memory_manager import EnhancedMemoryManager

# Import for memory system
try:
    from memory.wrapper import MemorySystem
    from memory.core import Memory
    from memory.emotional import EmotionalMemoryManager
    from memory.schemas import MemorySchemaManager
    from memory.masks import ProgressiveRevealManager
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    MEMORY_SYSTEM_AVAILABLE = False
    logging.warning("Advanced memory system not available. Some features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedNPCSystem:
    """
    Central system that integrates NPC creation, social dynamics, time management,
    memory systems, and stat progression using an agent-based architecture.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize the activity manager
        self.activity_manager = ActivityManager()
        
        # Initialize the agent system - core component for NPC agentic behavior
        self.agent_system = NPCAgentSystem(user_id, conversation_id)
        
        logger.info(f"Initialized IntegratedNPCSystem with NPCAgentSystem for user={user_id}, conversation={conversation_id}")
        logger.info(f"Available time phases: {TIME_PHASES}")
    
    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        return await MemorySystem.get_instance(self.user_id, self.conversation_id)
    
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
        
        # Step 1: Create the partial NPC (base data)
        partial_npc = create_npc_partial(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            sex=sex,
            total_archetypes=4,
            environment_desc=environment_desc
        )
        
        # Step 1.5: Integrate subtle femdom elements
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
        await propagate_shared_memories(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            source_npc_id=npc_id,
            source_npc_name=partial_npc["npc_name"],
            memories=memories
        )
        
        # Step 9: Create NPC Agent and initialize mask
        # This utilizes the agent framework by explicitly creating the agent
        agent = NPCAgent(npc_id, self.user_id, self.conversation_id)
        self.agent_system.npc_agents[npc_id] = agent
        
        # Initialize mask using the agent's capabilities
        mask_manager = await agent._get_mask_manager()
        await mask_manager.initialize_npc_mask(npc_id)
        
        # Step 10: Create a direct memory event using the agent's memory system
        memory_system = await agent._get_memory_system()
        creation_memory = f"I was created on {current_year}-{current_month}-{current_day} during {time_of_day}."
        
        await memory_system.remember(
            entity_type="npc",
            entity_id=npc_id,
            memory_text=creation_memory,
            importance="medium",
            tags=["creation", "origin"]
        )
        
        # Step 11: Initialize agent's perception of environment
        initial_context = {
            "location": current_location,
            "time_of_day": time_of_day,
            "description": f"Initial perception upon creation at {current_location}"
        }
        await agent.perceive_environment(initial_context)
        
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
        Get detailed information about an NPC, enhanced with agent-based data.
        
        Args:
            npc_id: The ID of the NPC to retrieve
            
        Returns:
            Dictionary with NPC details or None if not found
        """
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        
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
            
            # Get enhanced memory from the agent's memory system
            memory_system = await agent._get_memory_system()
            memory_result = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                limit=5
            )
            
            agent_memories = memory_result.get("memories", [])
            
            # Get mask information using agent
            mask_info = await agent._get_mask_manager().get_npc_mask(npc_id)
            
            # Get emotional state using agent
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            
            # Get beliefs using agent
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=npc_id,
                topic="player"
            )
            
            # Get current perception through agent
            current_perception = None
            if agent.last_perception:
                current_perception = {
                    "location": agent.last_perception.get("environment", {}).get("location"),
                    "time_of_day": agent.last_perception.get("environment", {}).get("time_of_day"),
                    "entities_present": agent.last_perception.get("environment", {}).get("entities_present", [])
                }
            
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
            
            # Build enhanced response with agent-based data
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
                "memories": agent_memories or memories[:5],  # Prefer agent memories
                "memory_count": len(memories),
                "mask": mask_info if mask_info and "error" not in mask_info else {"integrity": 100},
                "emotional_state": emotional_state,
                "beliefs": beliefs,
                "current_perception": current_perception,
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
        Mark an NPC as introduced, updating agent memory.
        
        Args:
            npc_id: The ID of the NPC to introduce
            
        Returns:
            True if successful, False otherwise
        """
        # Get or create the NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        
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
            
            # Create an introduction memory using the agent's memory system
            memory_system = await agent._get_memory_system()
            introduction_memory = f"I was formally introduced to the player today."
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=introduction_memory,
                importance="medium",
                tags=["introduction", "player_interaction", "first_meeting"]
            )
            
            # Update the agent's emotional state based on introduction
            await memory_system.update_npc_emotion(
                npc_id=npc_id,
                emotion="curiosity", 
                intensity=0.7
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
        
        # If this involves an NPC, update their relationship manager
        if entity1_type == "npc":
            # Get or create NPC agent
            if entity1_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[entity1_id] = NPCAgent(entity1_id, self.user_id, self.conversation_id)
            
            # Create memory of this link
            agent = self.agent_system.npc_agents[entity1_id]
            memory_system = await agent._get_memory_system()
            
            target_name = "Unknown"
            if entity2_type == "player":
                target_name = "Chase"
            elif entity2_type == "npc":
                # Get NPC name
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT npc_name FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (self.user_id, self.conversation_id, entity2_id))
                row = cursor.fetchone()
                if row:
                    target_name = row[0]
                conn.close()
            
            memory_text = f"I formed a {link_type} relationship with {target_name}."
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=entity1_id,
                memory_text=memory_text,
                importance="medium",
                tags=["relationship", entity2_type, link_type]
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
        # Get the relationship details before update
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
            FROM SocialLinks
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
        """, (link_id, self.user_id, self.conversation_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {"error": "Link not found"}
            
        entity1_type, entity1_id, entity2_type, entity2_id, old_type, old_level = row
        
        # Using update_link_type_and_level
        result = update_link_type_and_level(
            self.user_id, self.conversation_id,
            link_id, new_type, level_change
        )
        
        # Update agent memory if an NPC is involved
        if result and entity1_type == "npc":
            # Get or create NPC agent
            if entity1_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[entity1_id] = NPCAgent(entity1_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[entity1_id]
            memory_system = await agent._get_memory_system()
            
            target_name = "Unknown"
            if entity2_type == "player":
                target_name = "Chase"
            elif entity2_type == "npc":
                # Get NPC name
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT npc_name FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (self.user_id, self.conversation_id, entity2_id))
                row = cursor.fetchone()
                if row:
                    target_name = row[0]
                conn.close()
            
            if new_type and new_type != old_type:
                memory_text = f"My relationship with {target_name} changed from {old_type} to {new_type}."
            else:
                direction = "improved" if level_change > 0 else "worsened"
                memory_text = f"My relationship with {target_name} {direction} from level {old_level} to {result['new_level']}."
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=entity1_id,
                memory_text=memory_text,
                importance="medium",
                tags=["relationship_change", entity2_type]
            )
            
            # Update emotional state based on relationship change
            if abs(level_change) >= 10:
                if level_change > 0:
                    await memory_system.update_npc_emotion(
                        npc_id=entity1_id,
                        emotion="joy",
                        intensity=0.6
                    )
                else:
                    await memory_system.update_npc_emotion(
                        npc_id=entity1_id,
                        emotion="sadness",
                        intensity=0.6
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
        # Get the relationship details
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity1_type, entity1_id, entity2_type, entity2_id
            FROM SocialLinks
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
        """, (link_id, self.user_id, self.conversation_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return False
            
        entity1_type, entity1_id, entity2_type, entity2_id = row
        
        # Using add_link_event
        add_link_event(
            self.user_id, self.conversation_id,
            link_id, event_text
        )
        
        # Create memory record for NPC agents involved
        if entity1_type == "npc":
            # Get or create NPC agent
            if entity1_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[entity1_id] = NPCAgent(entity1_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[entity1_id]
            memory_system = await agent._get_memory_system()
            
            target_name = "Unknown"
            if entity2_type == "player":
                target_name = "Chase"
            elif entity2_type == "npc":
                # Get NPC name
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT npc_name FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (self.user_id, self.conversation_id, entity2_id))
                row = cursor.fetchone()
                if row:
                    target_name = row[0]
                conn.close()
            
            memory_text = f"With {target_name}: {event_text}"
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=entity1_id,
                memory_text=memory_text,
                importance="medium",
                tags=["relationship_event", entity2_type]
            )
        
        logger.info(f"Added event to link {link_id}: {event_text[:50]}...")
        return True
    
    async def update_relationship_from_interaction(self, 
                                                npc_id: int, 
                                                player_action: Dict[str, Any],
                                                npc_action: Dict[str, Any]) -> bool:
        """
        Update relationship between NPC and player based on an interaction.
        
        Args:
            npc_id: ID of the NPC
            player_action: Description of the player's action
            npc_action: Description of the NPC's action
            
        Returns:
            True if successful
        """
        # Get or create NPCRelationshipManager for this NPC
        relationship_manager = NPCRelationshipManager(npc_id, self.user_id, self.conversation_id)
        
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        # Update relationship through the manager
        await relationship_manager.update_relationship_from_interaction(
            "player", self.user_id, player_action, npc_action
        )
        
        return True
    
    #=================================================================
    # MEMORY MANAGEMENT
    #=================================================================
    
    async def add_memory_to_npc(self, 
                              npc_id: int, 
                              memory_text: str,
                              importance: str = "medium",
                              emotional: bool = False,
                              tags: List[str] = None) -> bool:
        """
        Add a memory to an NPC using the agent architecture.
        
        Args:
            npc_id: ID of the NPC
            memory_text: Text of the memory
            importance: Importance of the memory ("low", "medium", "high")
            emotional: Whether the memory has emotional content
            tags: List of tags for the memory
            
        Returns:
            True if successful, False otherwise
        """
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        memory_system = await agent._get_memory_system()
        
        # Create memory using the agent's memory system
        memory_id = await memory_system.remember(
            entity_type="npc",
            entity_id=npc_id,
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags or []
        )
        
        return memory_id is not None
    
    async def retrieve_relevant_memories(self, 
                                       npc_id: int, 
                                       query: str = None,
                                       context: Dict[str, Any] = None,
                                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to a context.
        
        Args:
            npc_id: ID of the NPC
            query: Search query
            context: Context dictionary
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory objects
        """
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        memory_system = await agent._get_memory_system()
        
        # Retrieve memories using the agent's memory system
        context_obj = context or {}
        if query:
            context_obj["query"] = query
        
        result = await memory_system.recall(
            entity_type="npc",
            entity_id=npc_id,
            query=query,
            context=context_obj,
            limit=limit
        )
        
        return result.get("memories", [])
    
    async def generate_flashback(self, npc_id: int, current_context: str) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback for an NPC using the agent's capabilities.
        
        Args:
            npc_id: ID of the NPC
            current_context: Current context that may trigger a flashback
            
        Returns:
            Flashback data or None if no flashback was generated
        """
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        memory_system = await agent._get_memory_system()
        
        # Generate flashback using the agent's memory system
        flashback = await memory_system.npc_flashback(
            npc_id=npc_id,
            context=current_context
        )
        
        return flashback
    
    async def propagate_memory_to_related_npcs(self, 
                                           source_npc_id: int,
                                           memory_text: str,
                                           importance: str = "medium") -> bool:
        """
        Propagate a memory to NPCs related to the source NPC.
        
        Args:
            source_npc_id: ID of the source NPC
            memory_text: Text of the memory to propagate
            importance: Importance of the memory
            
        Returns:
            True if successful, False otherwise
        """
        # First, add the memory to the source NPC
        await self.add_memory_to_npc(
            source_npc_id, memory_text, importance
        )
        
        # Get or create source NPC agent
        if source_npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[source_npc_id] = NPCAgent(source_npc_id, self.user_id, self.conversation_id)
        
        # Find related NPCs
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity2_id, link_level 
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s 
            AND entity1_type='npc' AND entity1_id=%s
            AND entity2_type='npc'
            AND link_level > 30
        """, (self.user_id, self.conversation_id, source_npc_id))
        
        related_npcs = []
        for row in cursor.fetchall():
            related_npcs.append((row[0], row[1]))
        
        # Get source NPC name
        source_name = "Unknown"
        cursor.execute("""
            SELECT npc_name FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (self.user_id, self.conversation_id, source_npc_id))
        
        name_row = cursor.fetchone()
        if name_row:
            source_name = name_row[0]
        
        conn.close()
        
        # Propagate memory to each related NPC
        for npc_id, link_level in related_npcs:
            # Modify the memory text based on relationship
            relationship_factor = link_level / 100.0  # 0.0 to 1.0
            
            # Higher relationship means more accurate propagation
            if relationship_factor > 0.7:
                propagated_text = f"I heard from {source_name} that {memory_text}"
            else:
                # Add potential distortion
                words = memory_text.split()
                if len(words) > 5:
                    # Replace 1-2 words to create slight distortion
                    for _ in range(random.randint(1, 2)):
                        if len(words) > 3:
                            idx = random.randint(0, len(words) - 1)
                            
                            # Replace with similar word or opposite
                            replacements = {
                                "good": "nice", "bad": "terrible", "happy": "pleased",
                                "sad": "unhappy", "angry": "upset", "large": "big",
                                "small": "tiny", "important": "critical", "interesting": "fascinating"
                            }
                            
                            if words[idx].lower() in replacements:
                                words[idx] = replacements[words[idx].lower()]
                
                distorted_text = " ".join(words)
                propagated_text = f"I heard from {source_name} that {distorted_text}"
            
            # Create the propagated memory
            await self.add_memory_to_npc(
                npc_id, 
                propagated_text, 
                "low" if importance == "medium" else "medium" if importance == "high" else "low",
                tags=["hearsay", "secondhand", "rumor"]
            )
        
        return True
    
    #=================================================================
    # MASK AND EMOTIONAL STATE MANAGEMENT
    #=================================================================
    
    async def get_npc_mask(self, npc_id: int) -> Dict[str, Any]:
        """
        Get mask information for an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with mask information
        """
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        mask_manager = await agent._get_mask_manager()
        
        # Get mask using the agent's mask manager
        result = await mask_manager.get_npc_mask(npc_id)
        
        return result
    
    async def generate_mask_slippage(self, 
                                   npc_id: int, 
                                   trigger: str = None,
                                   severity: int = None) -> Dict[str, Any]:
        """
        Generate a mask slippage event for an NPC.
        
        Args:
            npc_id: ID of the NPC
            trigger: What triggered the slippage
            severity: Severity level of the slippage
            
        Returns:
            Dictionary with slippage information
        """
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        memory_system = await agent._get_memory_system()
        
        # Generate mask slippage using the memory system
        result = await memory_system.reveal_npc_trait(
            npc_id=npc_id,
            trigger=trigger,
            severity=severity
        )
        
        return result
    
    async def update_npc_emotional_state(self, 
                                      npc_id: int, 
                                      emotion: str,
                                      intensity: float) -> Dict[str, Any]:
        """
        Update an NPC's emotional state.
        
        Args:
            npc_id: ID of the NPC
            emotion: Primary emotion
            intensity: Intensity of the emotion (0.0-1.0)
            
        Returns:
            Updated emotional state
        """
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        memory_system = await agent._get_memory_system()
        
        # Update emotional state using the memory system
        result = await memory_system.update_npc_emotion(
            npc_id=npc_id,
            emotion=emotion,
            intensity=intensity
        )
        
        return result
    
    async def get_npc_emotional_state(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's current emotional state.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Current emotional state
        """
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        memory_system = await agent._get_memory_system()
        
        # Get emotional state using the memory system
        result = await memory_system.get_npc_emotion(npc_id)
        
        return result
    
    #=================================================================
    # TIME MANAGEMENT AND NPC ACTIVITIES
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
        
        # After time advances, process scheduled activities for all NPCs
        if result.get("time_advanced", False):
            await self.process_npc_scheduled_activities()
        
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
    
    async def process_npc_scheduled_activities(self) -> Dict[str, Any]:
        """
        Process scheduled activities for all NPCs using the agent system.
        
        Returns:
            Dictionary with results of NPC activities
        """
        # Using the NPCAgentSystem to process scheduled activities
        result = await self.agent_system.process_npc_scheduled_activities()
        
        return result
    
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
        
        # Record each stat change separately
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
        Handle a complete interaction between player and NPC using the agent architecture.
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction (conversation, command, etc.)
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Comprehensive result dictionary
        """
        # Create player action object
        player_action = {
            "type": interaction_type,
            "description": player_input,
            "target_npc_id": npc_id
        }
        
        # Prepare context
        context_obj = context or {}
        context_obj["interaction_type"] = interaction_type
        
        # Process through the agent system - this is the key change utilizing the agent architecture
        result = await self.agent_system.handle_player_action(player_action, context_obj)
        
        # Process the activity and potentially advance time
        activity_result = await self.process_player_activity(player_input, context_obj)
        
        # Combine results
        combined_result = {
            "npc_id": npc_id,
            "interaction_type": interaction_type,
            "npc_responses": result.get("npc_responses", []),
            "events": [],
            "memories_created": [],
            "stat_changes": {},
            "time_advanced": activity_result.get("time_advanced", False)
        }
        
        # Add time advancement info if applicable
        if activity_result.get("time_advanced", False):
            combined_result["new_time"] = activity_result.get("new_time")
            
            # If time advanced, add any events that occurred
            for event in activity_result.get("events", []):
                combined_result["events"].append(event)
        
        # Apply stat effects to player
        stat_changes = {}
        
        # Get NPC details
        npc_details = await self.get_npc_details(npc_id)
        
        if npc_details:
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
            # Apply stat changes
            await self.apply_stat_changes(
                stat_changes, 
                cause=f"Interaction with {npc_details['npc_name'] if npc_details else 'NPC'}: {interaction_type}"
            )
            combined_result["stat_changes"] = stat_changes
        
        return combined_result
    
    async def handle_group_interaction(self,
                                    npc_ids: List[int],
                                    interaction_type: str,
                                    player_input: str,
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle an interaction between player and multiple NPCs using the agent architecture.
        
        Args:
            npc_ids: List of NPC IDs to interact with
            interaction_type: Type of interaction
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Comprehensive result dictionary
        """
        # Create player action object
        player_action = {
            "type": interaction_type,
            "description": player_input,
            "group_interaction": True
        }
        
        # Prepare context
        context_obj = context or {}
        context_obj["interaction_type"] = interaction_type
        context_obj["group_interaction"] = True
        context_obj["affected_npcs"] = npc_ids
        
        # Process through the agent system's coordinator
        result = await self.agent_system.handle_group_npc_interaction(npc_ids, player_action, context_obj)
        
        # Process the activity and potentially advance time
        activity_result = await self.process_player_activity(player_input, context_obj)
        
        # Combine results
        combined_result = {
            "npc_ids": npc_ids,
            "interaction_type": interaction_type,
            "npc_responses": result.get("npc_responses", []),
            "events": [],
            "stat_changes": {},
            "time_advanced": activity_result.get("time_advanced", False)
        }
        
        # Add time advancement info if applicable
        if activity_result.get("time_advanced", False):
            combined_result["new_time"] = activity_result.get("new_time")
            
            # If time advanced, add any events that occurred
            for event in activity_result.get("events", []):
                combined_result["events"].append(event)
        
        # Apply stat effects to player based on the group interaction
        stat_changes = {}
        
        # Calculate group dominance average
        total_dominance = 0
        total_cruelty = 0
        npc_count = 0
        
        for npc_id in npc_ids:
            npc_details = await self.get_npc_details(npc_id)
            if npc_details:
                total_dominance += npc_details["stats"]["dominance"]
                total_cruelty += npc_details["stats"]["cruelty"]
                npc_count += 1
        
        if npc_count > 0:
            avg_dominance = total_dominance / npc_count
            avg_cruelty = total_cruelty / npc_count
            
            if interaction_type == "submissive_response":
                # Submitting to a group increases effects
                dominance_factor = avg_dominance / 100  # 0.0 to 1.0
                stat_changes = {
                    "corruption": int(3 + (dominance_factor * 4)),
                    "obedience": int(4 + (dominance_factor * 5)),
                    "willpower": -3,
                    "confidence": -2
                }
            elif interaction_type == "defiant_response":
                # Defying a group is more impactful
                cruelty_factor = avg_cruelty / 100  # 0.0 to 1.0
                stat_changes = {
                    "willpower": +4,
                    "confidence": +3,
                    "mental_resilience": int(-2 - (cruelty_factor * 4))
                }
        
        if stat_changes:
            # Apply stat changes
            await self.apply_stat_changes(
                stat_changes, 
                cause=f"Group interaction with {npc_count} NPCs: {interaction_type}"
            )
            combined_result["stat_changes"] = stat_changes
        
        return combined_result
    
    async def generate_npc_scene(self, 
                               npc_ids: List[int], 
                               location: str = None) -> Dict[str, Any]:
        """
        Generate a scene with NPCs using the agent framework.
        
        Args:
            npc_ids: List of NPC IDs to include
            location: Location for the scene
            
        Returns:
            Scene information
        """
        # Prepare context
        context = {
            "location": location,
            "description": f"NPCs interacting at {location}"
        }
        
        # Use coordinator to generate group decisions
        result = await self.agent_system.coordinator.make_group_decisions(
            npc_ids, 
            shared_context=context
        )
        
        # Format scene from result
        scene = {
            "location": location,
            "npcs": npc_ids,
            "actions": result.get("group_actions", []),
            "individual_actions": result.get("individual_actions", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        return scene
    
    #=================================================================
    # AGENT SYSTEM MAINTENANCE AND MANAGEMENT
    #=================================================================
    
    async def run_agent_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run memory maintenance for all agents.
        
        Returns:
            Results of maintenance operations
        """
        return await self.agent_system.run_maintenance()
    
    async def get_all_npc_beliefs_about_player(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get all NPC's beliefs about the player.
        
        Returns:
            Dictionary mapping NPC IDs to lists of beliefs
        """
        return await self.agent_system.get_all_npc_beliefs_about_player()
    
    async def get_player_beliefs_about_npcs(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get player's beliefs about all NPCs.
        
        Returns:
            Dictionary mapping NPC IDs to lists of beliefs
        """
        return await self.agent_system.get_player_beliefs_about_npcs()

#=================================================================
# USAGE EXAMPLES
#=================================================================

async def example_usage():
    """Example demonstrating key agent-based functionality."""
    user_id = 1
    conversation_id = 123
    
    # Initialize the system
    npc_system = IntegratedNPCSystem(user_id, conversation_id)
    
    # Create a new NPC
    environment_desc = "A mansion with sprawling gardens and opulent interior."
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    npc_id = await npc_system.create_new_npc(environment_desc, day_names)
    
    # Get NPC details
    npc_details = await npc_system.get_npc_details(npc_id)
    print(f"Created NPC: {npc_details['npc_name']}")
    
    # Introduce the NPC
    await npc_system.introduce_npc(npc_id)
    
    # Handle an interaction with the NPC using the agent architecture
    interaction_result = await npc_system.handle_npc_interaction(
        npc_id, "conversation", "Hello, nice to meet you."
    )
    print(f"Interaction result: {interaction_result}")
    
    # Update NPC's emotional state
    await npc_system.update_npc_emotional_state(npc_id, "joy", 0.7)
    
    # Create another NPC
    second_npc_id = await npc_system.create_new_npc(environment_desc, day_names)
    
    # Handle a group interaction
    group_result = await npc_system.handle_group_interaction(
        [npc_id, second_npc_id], "conversation", "Hello everyone!"
    )
    print(f"Group interaction result: {group_result}")
    
    # Generate a scene with both NPCs
    scene = await npc_system.generate_npc_scene([npc_id, second_npc_id], "Garden")
    print(f"Generated scene: {scene}")
    
    # Process scheduled activities
    await npc_system.process_npc_scheduled_activities()
    
    # Run memory maintenance
    await npc_system.run_agent_memory_maintenance()
    
    print("Agent-based NPC system demo completed successfully!")

if __name__ == "__main__":
    # Run the agent-based example
    import asyncio
    asyncio.run(example_usage())
