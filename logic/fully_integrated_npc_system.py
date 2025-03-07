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
        Enhanced to use agent's relationship manager and memory system for more nuanced updates.
        
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
        
        agent = self.agent_system.npc_agents[npc_id]
        
        # Get agent's current emotional state and perception for context
        memory_system = await agent._get_memory_system()
        emotional_state = await memory_system.get_npc_emotion(npc_id)
        
        # Enhance relationship update with emotional context
        context = {
            "emotional_state": emotional_state,
            "recent_interactions": [],  # Could populate from memory
            "interaction_type": player_action.get("type", "unknown")
        }
        
        # Get recent memories to inform relationship change
        memory_result = await memory_system.recall(
            entity_type="npc",
            entity_id=npc_id,
            query="player interaction",
            limit=3
        )
        
        context["recent_interactions"] = memory_result.get("memories", [])
        
        # Update relationship with enhanced context
        await relationship_manager.update_relationship_from_interaction(
            "player", self.user_id, player_action, npc_action, context
        )
        
        # Create a memory of this relationship change
        # Format memory text based on action types
        if player_action.get("type") in ["help", "assist", "support"]:
            memory_text = "The player helped me, improving our relationship."
        elif player_action.get("type") in ["insult", "mock", "threaten"]:
            memory_text = "The player was hostile to me, damaging our relationship."
        else:
            memory_text = f"My relationship with the player changed after they {player_action.get('description', 'interacted with me')}."
        
        await memory_system.remember(
            entity_type="npc",
            entity_id=npc_id,
            memory_text=memory_text,
            importance="medium",
            tags=["relationship_change", "player_interaction"]
        )
        
        return True

    async def record_memory_event(self, npc_id: int, memory_text: str, tags: List[str] = None) -> bool:
    """
    Record a memory event for an NPC using the agent's memory system.
    
    Args:
        npc_id: ID of the NPC
        memory_text: The memory text to record
        tags: Optional tags for the memory
        
    Returns:
        True if successful, False otherwise
    """
    # Get or create the NPC agent
    if npc_id not in self.agent_system.npc_agents:
        self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
    
    agent = self.agent_system.npc_agents[npc_id]
    memory_system = await agent._get_memory_system()
    
    # Create the memory
    try:
        await memory_system.remember(
            entity_type="npc",
            entity_id=npc_id,
            memory_text=memory_text,
            importance="medium",  # Default importance
            tags=tags or ["player_interaction"]
        )
        return True
    except Exception as e:
        logger.error(f"Error recording memory for NPC {npc_id}: {e}")
        return False

    async def check_for_relationship_events(self) -> List[Dict[str, Any]]:
        """
        Check for special relationship events like crossroads or rituals.
        
        Returns:
            List of relationship events
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        events = []
        
        try:
            # Get social links with sufficiently high levels that might trigger events
            cursor.execute("""
                SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id, 
                       link_type, link_level
                FROM SocialLinks
                WHERE user_id = %s AND conversation_id = %s
                  AND (entity1_type = 'player' OR entity2_type = 'player')
                  AND link_level >= 50
            """, (self.user_id, self.conversation_id))
            
            links = []
            for row in cursor.fetchall():
                link_id, e1_type, e1_id, e2_type, e2_id, link_type, link_level = row
                
                # Get NPC name if applicable
                npc_id = e1_id if e1_type == 'npc' else e2_id if e2_type == 'npc' else None
                npc_name = None
                
                if npc_id:
                    cursor.execute("""
                        SELECT npc_name FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """, (npc_id, self.user_id, self.conversation_id))
                    name_row = cursor.fetchone()
                    if name_row:
                        npc_name = name_row[0]
                
                links.append({
                    "link_id": link_id,
                    "entity1_type": e1_type,
                    "entity1_id": e1_id,
                    "entity2_type": e2_type,
                    "entity2_id": e2_id,
                    "link_type": link_type,
                    "link_level": link_level,
                    "npc_id": npc_id,
                    "npc_name": npc_name
                })
            
            # Check each link for potential events
            for link in links:
                # Check for crossroads event - significant decision point
                if link["link_level"] >= 70 and random.random() < 0.2:  # 20% chance for high level links
                    # Get or create NPC agent for better decision making
                    npc_id = link["npc_id"]
                    if npc_id and npc_id not in self.agent_system.npc_agents:
                        self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                    
                    # Get NPC agent if available
                    agent = self.agent_system.npc_agents.get(npc_id) if npc_id else None
                    
                    # Generate crossroads event (potentially using agent for better decision modeling)
                    crossroads_data = await self._generate_relationship_crossroads(link, agent)
                    
                    if crossroads_data:
                        events.append({
                            "type": "relationship_crossroads",
                            "data": crossroads_data
                        })
        
        except Exception as e:
            logger.error(f"Error checking for relationship events: {e}")
        
        finally:
            cursor.close()
            conn.close()
        
        return events
    
    async def _generate_relationship_crossroads(self, link: Dict[str, Any], agent: Optional[NPCAgent] = None) -> Dict[str, Any]:
        """
        Generate a relationship crossroads event based on link details and NPC agent.
        
        Args:
            link: The social link data
            agent: Optional NPC agent for better decision modeling
            
        Returns:
            Crossroads event data
        """
        # Default crossroads types based on relationship level
        crossroads_types = [
            "trust_test",
            "commitment_decision",
            "loyalty_challenge",
            "boundary_setting",
            "power_dynamic_shift"
        ]
        
        # Use agent to refine crossroads type if available
        selected_type = random.choice(crossroads_types)
        if agent:
            # Get agent's current emotional state and perception for better context
            memory_system = await agent._get_memory_system()
            emotional_state = await memory_system.get_npc_emotion(link["npc_id"])
            
            # Use emotional state to influence crossroads type
            if emotional_state and "current_emotion" in emotional_state:
                emotion = emotional_state["current_emotion"]
                primary = emotion.get("primary", {})
                emotion_name = primary.get("name", "neutral")
                
                # Adjust crossroads type based on emotional state
                if emotion_name == "anger":
                    selected_type = "boundary_setting" if random.random() < 0.7 else "power_dynamic_shift"
                elif emotion_name == "joy":
                    selected_type = "commitment_decision" if random.random() < 0.7 else "trust_test"
                elif emotion_name == "fear":
                    selected_type = "trust_test" if random.random() < 0.7 else "loyalty_challenge"
        
        # Generate crossroads options based on type
        options = self._generate_crossroads_options(selected_type, link)
        
        # Create crossroads data
        crossroads_data = {
            "link_id": link["link_id"],
            "npc_id": link["npc_id"],
            "npc_name": link["npc_name"],
            "type": selected_type,
            "description": self._get_crossroads_description(selected_type, link),
            "options": options,
            "expires_in": 3  # Number of interactions before expiring
        }
        
        return crossroads_data
    
    def _generate_crossroads_options(self, crossroads_type: str, link: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate options for a relationship crossroads based on type."""
        if crossroads_type == "trust_test":
            return [
                {
                    "text": "Trust completely",
                    "stat_effects": {"trust": 10, "respect": 5, "willpower": -5},
                    "outcome": f"Your trust in {link['npc_name']} deepens significantly."
                },
                {
                    "text": "Remain cautious",
                    "stat_effects": {"trust": 0, "willpower": 5, "respect": 0},
                    "outcome": f"You maintain your guard with {link['npc_name']}."
                },
                {
                    "text": "Express distrust",
                    "stat_effects": {"trust": -10, "respect": 0, "willpower": 10},
                    "outcome": f"Your relationship with {link['npc_name']} becomes more distant."
                }
            ]
        elif crossroads_type == "commitment_decision":
            # Generate options based on commitment decision
            return [
                {
                    "text": "Commit fully",
                    "stat_effects": {"closeness": 15, "willpower": -10, "obedience": 10},
                    "outcome": f"Your relationship with {link['npc_name']} becomes much closer."
                },
                {
                    "text": "Partial commitment",
                    "stat_effects": {"closeness": 5, "willpower": 0, "obedience": 0},
                    "outcome": f"You become somewhat closer to {link['npc_name']}."
                },
                {
                    "text": "Maintain independence",
                    "stat_effects": {"willpower": 10, "closeness": -5, "obedience": -5},
                    "outcome": f"You maintain your independence from {link['npc_name']}."
                }
            ]
        # Add options for other crossroads types
        # ...
        
        # Default options if type not recognized
        return [
            {
                "text": "Strengthen relationship",
                "stat_effects": {"closeness": 10, "respect": 5},
                "outcome": f"Your bond with {link['npc_name']} strengthens."
            },
            {
                "text": "Maintain status quo",
                "stat_effects": {"closeness": 0, "respect": 0},
                "outcome": f"Your relationship with {link['npc_name']} continues unchanged."
            },
            {
                "text": "Create distance",
                "stat_effects": {"closeness": -10, "respect": -5, "willpower": 5},
                "outcome": f"You create some distance between yourself and {link['npc_name']}."
            }
        ]
    
    def _get_crossroads_description(self, crossroads_type: str, link: Dict[str, Any]) -> str:
        """Get description text for a relationship crossroads."""
        npc_name = link.get("npc_name", "The NPC")
        
        descriptions = {
            "trust_test": f"{npc_name} has shared something important with you. How much will you trust them?",
            "commitment_decision": f"Your relationship with {npc_name} has reached a critical point. How committed will you be?",
            "loyalty_challenge": f"{npc_name} is testing your loyalty. How will you respond?",
            "boundary_setting": f"{npc_name} is pushing boundaries in your relationship. How will you establish limits?",
            "power_dynamic_shift": f"The power dynamic with {npc_name} is shifting. How will you position yourself?"
        }
        
        return descriptions.get(crossroads_type, f"You've reached a crossroads in your relationship with {npc_name}.")
    
    async def apply_crossroads_choice(self, link_id: int, crossroads_name: str, choice_index: int) -> Dict[str, Any]:
        """
        Apply a choice in a relationship crossroads.
        
        Args:
            link_id: ID of the social link
            crossroads_name: Type/name of the crossroads
            choice_index: Index of the chosen option
            
        Returns:
            Result of the choice
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get link details
            cursor.execute("""
                SELECT entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE link_id = %s AND user_id = %s AND conversation_id = %s
            """, (link_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if not row:
                return {"error": "Link not found"}
            
            e1_type, e1_id, e2_type, e2_id, link_type, link_level = row
            
            # Get NPC details if applicable
            npc_id = e1_id if e1_type == 'npc' else e2_id if e2_type == 'npc' else None
            npc_name = None
            
            if npc_id:
                cursor.execute("""
                    SELECT npc_name FROM NPCStats
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                """, (npc_id, self.user_id, self.conversation_id))
                name_row = cursor.fetchone()
                if name_row:
                    npc_name = name_row[0]
            
            # Reconstruct link data
            link = {
                "link_id": link_id,
                "entity1_type": e1_type,
                "entity1_id": e1_id,
                "entity2_type": e2_type,
                "entity2_id": e2_id,
                "link_type": link_type,
                "link_level": link_level,
                "npc_id": npc_id,
                "npc_name": npc_name
            }
            
            # Generate options to find the chosen one
            options = self._generate_crossroads_options(crossroads_name, link)
            
            if choice_index < 0 or choice_index >= len(options):
                return {"error": "Invalid choice index"}
            
            chosen_option = options[choice_index]
            
            # Apply stat effects
            if "stat_effects" in chosen_option:
                # Convert dict to a list of separate changes for apply_stat_change
                await self.apply_stat_changes(
                    chosen_option["stat_effects"],
                    f"Crossroads choice in relationship with {npc_name}"
                )
            
            # Apply relationship changes
            # Determine change based on choice index
            level_change = 10 if choice_index == 0 else 0 if choice_index == 1 else -10
            
            # Update relationship
            cursor.execute("""
                UPDATE SocialLinks
                SET link_level = GREATEST(0, LEAST(100, link_level + %s))
                WHERE link_id = %s
            """, (level_change, link_id))
            
            # Add event to link history
            cursor.execute("""
                UPDATE SocialLinks
                SET link_history = COALESCE(link_history, '[]'::jsonb) || %s::jsonb
                WHERE link_id = %s
            """, (json.dumps([f"Crossroads choice: {chosen_option['text']}"]), link_id))
            
            conn.commit()
            
            # Create memory for NPC
            if npc_id:
                await self.add_memory_to_npc(
                    npc_id,
                    f"The player made a choice about our relationship: {chosen_option['text']}",
                    importance="high",
                    tags=["crossroads", "relationship_choice"]
                )
            
            return {
                "success": True,
                "outcome_text": chosen_option.get("outcome", "Your choice has been recorded."),
                "stat_effects": chosen_option.get("stat_effects", {})
            }
        
        except Exception as e:
            conn.rollback()
            logger.error(f"Error applying crossroads choice: {e}")
            return {"error": str(e)}
        
        finally:
            cursor.close()
            conn.close()
    
    async def get_relationship(self, entity1_type: str, entity1_id: int, entity2_type: str, entity2_id: int) -> Dict[str, Any]:
        """
        Get the relationship between two entities.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            
        Returns:
            Dictionary with relationship details
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check for direct relationship
            cursor.execute("""
                SELECT link_id, link_type, link_level, link_history
                FROM SocialLinks
                WHERE user_id = %s AND conversation_id = %s
                  AND ((entity1_type = %s AND entity1_id = %s AND entity2_type = %s AND entity2_id = %s)
                    OR (entity1_type = %s AND entity1_id = %s AND entity2_type = %s AND entity2_id = %s))
            """, (
                self.user_id, self.conversation_id,
                entity1_type, entity1_id, entity2_type, entity2_id,
                entity2_type, entity2_id, entity1_type, entity1_id
            ))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            link_id, link_type, link_level, link_history = row
            
            # Convert link_history to Python list if it's not None
            if link_history:
                if isinstance(link_history, str):
                    try:
                        history = json.loads(link_history)
                    except json.JSONDecodeError:
                        history = []
                else:
                    history = link_history
            else:
                history = []
            
            # Get entity names
            entity1_name = await self._get_entity_name(entity1_type, entity1_id, cursor)
            entity2_name = await self._get_entity_name(entity2_type, entity2_id, cursor)
            
            # Use agent system for memory-enriched relationship data if one of the entities is an NPC
            relationship_memories = []
            if entity1_type == "npc" and entity1_id in self.agent_system.npc_agents:
                agent = self.agent_system.npc_agents[entity1_id]
                memory_system = await agent._get_memory_system()
                
                # Get memories related to the relationship
                query = f"{entity2_name}"
                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=entity1_id,
                    query=query,
                    limit=5
                )
                
                relationship_memories.extend(memory_result.get("memories", []))
            
            elif entity2_type == "npc" and entity2_id in self.agent_system.npc_agents:
                agent = self.agent_system.npc_agents[entity2_id]
                memory_system = await agent._get_memory_system()
                
                # Get memories related to the relationship
                query = f"{entity1_name}"
                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=entity2_id,
                    query=query,
                    limit=5
                )
                
                relationship_memories.extend(memory_result.get("memories", []))
            
            relationship = {
                "link_id": link_id,
                "entity1_type": entity1_type,
                "entity1_id": entity1_id,
                "entity1_name": entity1_name,
                "entity2_type": entity2_type,
                "entity2_id": entity2_id,
                "entity2_name": entity2_name,
                "link_type": link_type,
                "link_level": link_level,
                "link_history": history[-5:],  # Get last 5 events
                "relationship_memories": relationship_memories
            }
            
            return relationship
        
        except Exception as e:
            logger.error(f"Error getting relationship: {e}")
            return None
        
        finally:
            cursor.close()
            conn.close()
    
    async def _get_entity_name(self, entity_type: str, entity_id: int, cursor) -> str:
        """Get the name of an entity."""
        if entity_type == "player":
            return "Player"
        elif entity_type == "npc":
            cursor.execute("""
                SELECT npc_name FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (entity_id, self.user_id, self.conversation_id))
            row = cursor.fetchone()
            return row[0] if row else f"NPC-{entity_id}"
        else:
            return f"{entity_type}-{entity_id}"
    
    async def generate_multi_npc_scene(self, npc_ids: List[int], location: str = None, include_player: bool = True) -> Dict[str, Any]:
        """
        Generate a scene with multiple NPCs interacting.
        
        Args:
            npc_ids: List of NPC IDs to include in the scene
            location: Optional location for the scene
            include_player: Whether to include the player in the scene
            
        Returns:
            Scene information
        """
        # Get current time information
        year, month, day, time_of_day = await self.get_current_game_time()
        
        # If location not provided, get a common location for the NPCs
        if not location:
            location = await self._find_common_location(npc_ids)
        
        # Initialize the scene data
        scene = {
            "location": location,
            "time_of_day": time_of_day,
            "day": day,
            "npc_ids": npc_ids,
            "include_player": include_player,
            "interactions": [],
            "description": f"Scene at {location} during {time_of_day}"
        }
        
        # Use the agent coordinator for group behavior
        coordinator = self.agent_system.coordinator
        
        # Prepare context for the scene
        context = {
            "location": location,
            "time_of_day": time_of_day,
            "day": day,
            "include_player": include_player,
            "description": f"NPCs interacting at {location} during {time_of_day}"
        }
        
        # Generate group actions using the coordinator
        action_plan = await coordinator.make_group_decisions(npc_ids, context)
        
        # Add actions to the scene
        scene["group_actions"] = action_plan.get("group_actions", [])
        scene["individual_actions"] = action_plan.get("individual_actions", {})
        
        # Create interactions
        interactions = []
        
        # Process group actions
        for group_action in action_plan.get("group_actions", []):
            npc_id = group_action.get("npc_id")
            if npc_id is None:
                continue
                
            action_data = group_action.get("action", {})
            
            # Get NPC name
            npc_name = await self.get_npc_name(npc_id)
            
            interaction = {
                "type": "group_action",
                "npc_id": npc_id,
                "npc_name": npc_name,
                "action": action_data.get("type", "interact"),
                "description": action_data.get("description", "does something"),
                "target": action_data.get("target", "group")
            }
            
            interactions.append(interaction)
        
        # Process individual actions
        for npc_id, actions in action_plan.get("individual_actions", {}).items():
            npc_name = await self.get_npc_name(npc_id)
            
            for action_data in actions:
                target = action_data.get("target")
                target_name = action_data.get("target_name")
                
                interaction = {
                    "type": "individual_action",
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "action": action_data.get("type", "interact"),
                    "description": action_data.get("description", "does something"),
                    "target": target,
                    "target_name": target_name
                }
                
                interactions.append(interaction)
        
        # Add interactions to the scene
        scene["interactions"] = interactions
        
        return scene
    
    async def _find_common_location(self, npc_ids: List[int]) -> str:
        """Find a common location for a group of NPCs."""
        if not npc_ids:
            return "Unknown"
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get all locations for the NPCs
            cursor.execute("""
                SELECT npc_id, current_location
                FROM NPCStats
                WHERE npc_id = ANY(%s) AND user_id = %s AND conversation_id = %s
            """, (npc_ids, self.user_id, self.conversation_id))
            
            locations = {}
            for row in cursor.fetchall():
                _, loc = row
                locations[loc] = locations.get(loc, 0) + 1
            
            # Find the most common location
            common_location = max(locations.items(), key=lambda x: x[1])[0] if locations else "Unknown"
            
            return common_location
        
        except Exception as e:
            logger.error(f"Error finding common location: {e}")
            return "Unknown"
        
        finally:
            cursor.close()
            conn.close()
    
    async def generate_overheard_conversation(self, npc_ids: List[int], topic: str = None, about_player: bool = False) -> Dict[str, Any]:
        """
        Generate a conversation between NPCs that the player can overhear.
        
        Args:
            npc_ids: List of NPCs involved in the conversation
            topic: Optional topic of conversation
            about_player: Whether the conversation is about the player
            
        Returns:
            Conversation details
        """
        if len(npc_ids) < 2:
            return {"error": "Need at least 2 NPCs for a conversation"}
        
        # Get current location and time
        year, month, day, time_of_day = await self.get_current_game_time()
        location = await self._find_common_location(npc_ids)
        
        # Get NPC details
        npc_details = {}
        for npc_id in npc_ids:
            details = await self.get_npc_details(npc_id)
            if details:
                npc_details[npc_id] = details
        
        # Prepare context for conversation
        topic_text = topic or ("the player" if about_player else "general matters")
        context = {
            "location": location,
            "time_of_day": time_of_day,
            "topic": topic_text,
            "about_player": about_player,
            "description": f"NPCs conversing about {topic_text} at {location}"
        }
        
        # Use agent system for generating conversation
        conversation_lines = []
        
        # Generate initial statement from first NPC
        first_npc = npc_ids[0]
        agent1 = self.agent_system.npc_agents.get(first_npc)
        if not agent1:
            self.agent_system.npc_agents[first_npc] = NPCAgent(first_npc, self.user_id, self.conversation_id)
            agent1 = self.agent_system.npc_agents[first_npc]
        
        first_perception = await agent1.perceive_environment(context)
        first_action = await agent1.make_decision(first_perception)
        
        npc1_name = npc_details.get(first_npc, {}).get("npc_name", f"NPC-{first_npc}")
        first_line = {
            "npc_id": first_npc,
            "npc_name": npc1_name,
            "text": first_action.get("description", f"starts talking about {topic_text}")
        }
        conversation_lines.append(first_line)
        
        # Generate responses from other NPCs
        for npc_id in npc_ids[1:]:
            agent = self.agent_system.npc_agents.get(npc_id)
            if not agent:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                agent = self.agent_system.npc_agents[npc_id]
            
            # Add the previous statement to context
            response_context = context.copy()
            response_context["previous_statement"] = first_line.get("text")
            response_context["previous_speaker"] = first_line.get("npc_name")
            
            # Generate response using agent
            perception = await agent.perceive_environment(response_context)
            response_action = await agent.make_decision(perception)
            
            npc_name = npc_details.get(npc_id, {}).get("npc_name", f"NPC-{npc_id}")
            response_line = {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "text": response_action.get("description", f"responds about {topic_text}")
            }
            conversation_lines.append(response_line)
        
        # Create memories of this conversation for each NPC
        for npc_id in npc_ids:
            other_npcs = [n for n in npc_ids if n != npc_id]
            other_names = [npc_details.get(n, {}).get("npc_name", f"NPC-{n}") for n in other_npcs]
            
            memory_text = f"I had a conversation with {', '.join(other_names)} about {topic_text} at {location}."
            
            await self.add_memory_to_npc(
                npc_id,
                memory_text,
                importance="medium" if about_player else "low",
                tags=["conversation", "overheard"] + (["player_related"] if about_player else [])
            )
        
        # Format the conversation
        conversation = {
            "location": location,
            "time_of_day": time_of_day,
            "topic": topic_text,
            "about_player": about_player,
            "lines": conversation_lines
        }
        
        return conversation
    
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
        Enhanced with agent perception and memory formation.
        
        Args:
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Dictionary with processing results
        """
        # Create base context if not provided
        context_obj = context or {}
        
        # Create standardized player action
        player_action = {
            "type": "activity",
            "description": player_input,
            "text": player_input,
            "context": context_obj
        }
        
        # Determine activity type using activity manager
        activity_result = await self.activity_manager.process_activity(
            self.user_id, self.conversation_id, player_input, context_obj
        )
        
        # Update player action with determined activity type
        player_action["type"] = activity_result.get("activity_type", "generic_activity")
        
        # Add activity perception to nearby NPCs via agent system
        # This ensures NPCs are aware of what the player is doing
        current_location = context_obj.get("location")
        
        if current_location:
            # Get NPCs at current location
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npc_id 
                FROM NPCStats 
                WHERE user_id=%s AND conversation_id=%s AND current_location=%s
            """, (self.user_id, self.conversation_id, current_location))
            
            nearby_npc_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Update perception for each nearby NPC agent
            for npc_id in nearby_npc_ids:
                if npc_id not in self.agent_system.npc_agents:
                    self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                
                agent = self.agent_system.npc_agents[npc_id]
                
                # Create perception context
                perception_context = {
                    "location": current_location,
                    "player_action": player_action,
                    "description": f"Player {player_input}"
                }
                
                # Update agent's perception
                await agent.perceive_environment(perception_context)
                
                # Create a memory of observing the player's action
                memory_system = await agent._get_memory_system()
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=f"I observed the player {player_input} at {current_location}",
                    importance="low",  # Low importance for routine observations
                    tags=["player_observation", player_action["type"]]
                )
        
        # Return the original activity result
        return activity_result
    
    async def process_npc_scheduled_activities(self) -> Dict[str, Any]:
        """
        Process scheduled activities for all NPCs using the agent system.
        Enhanced with agent-based decision making and memory formation.
        
        Returns:
            Dictionary with results of NPC activities
        """
        # Get current time information for context
        year, month, day, time_of_day = await self.get_current_game_time()
        
        # Create base context for all NPCs
        base_context = {
            "year": year,
            "month": month,
            "day": day,
            "time_of_day": time_of_day,
            "activity_type": "scheduled"
        }
        
        # Process each NPC's scheduled activities
        results = []
        conn = None
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all NPCs with their current locations
            cursor.execute("""
                SELECT npc_id, npc_name, current_location 
                FROM NPCStats 
                WHERE user_id=%s AND conversation_id=%s
            """, (self.user_id, self.conversation_id))
            
            npc_data = {}
            for row in cursor.fetchall():
                npc_id, npc_name, location = row
                npc_data[npc_id] = {
                    "name": npc_name,
                    "location": location
                }
                
            # For each NPC, process their scheduled activity
            for npc_id, data in npc_data.items():
                try:
                    # Get or create NPC agent
                    if npc_id not in self.agent_system.npc_agents:
                        self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                    
                    agent = self.agent_system.npc_agents[npc_id]
                    
                    # Create context with location
                    context = base_context.copy()
                    context["location"] = data["location"]
                    context["description"] = f"Scheduled activity at {data['location']} during {time_of_day}"
                    
                    # Use agent to perceive environment and make decision
                    perception = await agent.perceive_environment(context)
                    
                    # Perform scheduled activity using the agent
                    activity_result = await agent.perform_scheduled_activity()
                    
                    if activity_result:
                        # Format result for output
                        formatted_result = {
                            "npc_id": npc_id,
                            "npc_name": data["name"],
                            "location": data["location"],
                            "action": activity_result.get("action", {}),
                            "result": activity_result.get("result", {})
                        }
                        
                        # Add emotional impact data from agent if available
                        memory_system = await agent._get_memory_system()
                        emotional_state = await memory_system.get_npc_emotion(npc_id)
                        
                        if emotional_state:
                            formatted_result["emotional_state"] = emotional_state
                        
                        results.append(formatted_result)
                except Exception as e:
                    logger.error(f"Error processing scheduled activity for NPC {npc_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error in process_npc_scheduled_activities: {e}")
        
        finally:
            if conn:
                conn.close()
        
        # Let agent system process all scheduled activities
        # This handles coordination between NPCs, shared observations, etc.
        agent_system_result = await self.agent_system.process_npc_scheduled_activities()
        
        # Combine our results with agent system results
        combined_results = {
            "npc_responses": results,
            "agent_system_responses": agent_system_result.get("npc_responses", []),
            "count": len(results)
        }
        
        return combined_results

    async def run_memory_maintenance(self) -> Dict[str, Any]:
    """
    Run memory maintenance for all agents with enhanced processing.
    
    Returns:
        Results of maintenance operations
    """
    try:
        # Run system-wide maintenance first
        system_results = await self.agent_system.run_maintenance()
        
        # Run individual agent maintenance for any that need special processing
        individual_results = {}
        
        for npc_id, agent in self.agent_system.npc_agents.items():
            try:
                # Run specific maintenance on NPC agents that might need it
                # These include highly active NPCs or those with special relationships
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Check if this NPC has been frequently active or has important relationships
                cursor.execute("""
                    SELECT COUNT(*) FROM SocialLinks
                    WHERE user_id = %s AND conversation_id = %s
                    AND entity1_type = 'npc' AND entity1_id = %s
                    AND link_level > 50
                """, (self.user_id, self.conversation_id, npc_id))
                
                important_relationships = cursor.fetchone()[0] > 0
                
                cursor.execute("""
                    SELECT COUNT(*) FROM NPCAgentState
                    WHERE user_id = %s AND conversation_id = %s
                    AND npc_id = %s
                    AND last_updated > NOW() - INTERVAL '1 day'
                """, (self.user_id, self.conversation_id, npc_id))
                
                recently_active = cursor.fetchone()[0] > 0
                
                conn.close()
                
                # Only run individual maintenance for NPCs that need it
                if important_relationships or recently_active:
                    agent_result = await agent.run_memory_maintenance()
                    individual_results[npc_id] = agent_result
            
            except Exception as e:
                logger.error(f"Error in agent-specific maintenance for NPC {npc_id}: {e}")
                individual_results[npc_id] = {"error": str(e)}
        
        # Combine results
        combined_results = {
            "system_maintenance": system_results,
            "individual_maintenance": individual_results
        }
        
        return combined_results
    
    except Exception as e:
        logger.error(f"Error in run_memory_maintenance: {e}")
        return {"error": str(e)}
    
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
