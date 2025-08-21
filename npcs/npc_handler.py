# npcs/npc_handler.py

"""
NPC interaction handler for managing interactions between NPCs and players.
Refactored to use the new dynamic relationships system.
"""

import logging
import json
import asyncio
import random
from typing import List, Dict, Any, Optional
import os
import openai

from pydantic import BaseModel, Field
from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem
from logic.activities_logic import get_all_activities, filter_activities_for_npc
from npcs.npc_learning_adaptation import NPCLearningAdaptation, NPCLearningManager

# Import centralized LLM functions
from logic.chatgpt_integration import get_chatgpt_response, get_async_openai_client, TEMPERATURE_SETTINGS

# Import new dynamic relationships system
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    process_relationship_interaction_tool,
    get_relationship_summary_tool,
    poll_relationship_events_tool,
    event_generator
)

# Configuration
DB_DSN = os.getenv("DB_DSN")
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Utility function for OpenAI calls with retry
# -------------------------------------------------------

async def call_openai_with_retry(client, **kwargs):
    """
    Call OpenAI API with retry logic for rate limits.
    """
    max_retries = 5
    initial_delay = 1
    backoff_factor = 2
    
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (backoff_factor ** attempt)
                logger.warning(f"Rate limit hit on attempt {attempt+1}/{max_retries}: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                raise
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    raise Exception("Max retries exceeded")

# -------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------

class NPCInteractionInput(BaseModel):
    npc_id: int
    player_input: str
    context: Dict[str, Any] = Field(default_factory=dict)
    interaction_type: str = "standard_interaction"

class NPCInteractionOutput(BaseModel):
    npc_id: int
    npc_name: str
    response: str
    stat_changes: Dict[str, int] = Field(default_factory=dict)
    memory_created: bool = False
    relationship_event: Optional[Dict[str, Any]] = None
    relationship_changes: Dict[str, float] = Field(default_factory=dict)

class NPCHandler:
    """Handles NPC interactions with players and other NPCs"""
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the NPC handler.
        
        Args:
            user_id: User/player ID
            conversation_id: Conversation/scene ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize the learning manager for handling multiple NPCs
        self.learning_manager = NPCLearningManager(user_id, conversation_id)
        
        # Initialize relationship manager
        self.relationship_manager = OptimizedRelationshipManager(user_id, conversation_id)

    async def handle_interaction(
        self,
        npc_id: int,
        interaction_type: str,
        player_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an interaction between the player and an NPC using centralized ChatGPT.
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction
            player_input: Player's input text
            context: Additional context for the interaction
            
        Returns:
            Dictionary with the NPC's response
        """
        # Get NPC details
        npc_details = await self.get_npc_details(npc_id)
        
        # Get NPC memories
        memories = await self.get_npc_memory(npc_id)
        
        # Get player relationship using new system
        relationship = await self.get_relationship_details("npc", npc_id, "player", 0)
        
        # Build the response using centralized ChatGPT with function calling
        result = await self._generate_npc_response_with_gpt(
            npc_id=npc_id,
            npc_details=npc_details,
            player_input=player_input,
            interaction_type=interaction_type,
            memories=memories,
            relationship=relationship,
            context=context or {}
        )
        
        # Process the result
        if result.get("type") == "function_call" and result.get("function_name") == "apply_universal_update":
            # Extract response from function args
            function_args = result.get("function_args", {})
            narrative = function_args.get("narrative", "")
            
            # Check for stat changes in character_stat_updates
            stat_updates = function_args.get("character_stat_updates", {})
            stat_changes = {}
            if stat_updates and "stats" in stat_updates:
                # Convert to simple stat changes (deltas would need to be calculated)
                # For now, we'll just note which stats were mentioned
                for stat, value in stat_updates["stats"].items():
                    if stat in ["dominance", "cruelty", "closeness", "trust", "respect", "intensity"]:
                        # This is a simplified approach - in reality you'd calculate deltas
                        stat_changes[stat] = 0
            
            response_data = {
                "npc_id": npc_id,
                "npc_name": npc_details.get("npc_name", f"NPC_{npc_id}"),
                "response": narrative,
                "stat_changes": stat_changes,
                "memory_created": bool(function_args.get("npc_updates", [])) or bool(function_args.get("journal_updates", []))
            }
        else:
            # Fallback to text response
            response_text = result.get("response", "I don't know how to respond to that.")
            response_data = {
                "npc_id": npc_id,
                "npc_name": npc_details.get("npc_name", f"NPC_{npc_id}"),
                "response": response_text,
                "stat_changes": {},
                "memory_created": False
            }
        
        # Process the interaction with the new relationship system
        interaction_mapping = {
            "friendly": "helpful_action",
            "hostile": "criticism_harsh",
            "intimate": "vulnerability_shared",
            "suspicious": "boundary_violated",
            "supportive": "support_provided"
        }
        
        # Determine interaction type for relationship system
        rel_interaction_type = interaction_mapping.get(interaction_type, "helpful_action")
        
        # Use context from RunContextWrapper
        from agents import RunContextWrapper
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        
        # Process the relationship interaction
        rel_result = await process_relationship_interaction_tool(
            ctx=ctx,
            entity1_type="npc",
            entity1_id=npc_id,
            entity2_type="player",
            entity2_id=0,
            interaction_type=rel_interaction_type,
            context="conversation",
            check_for_event=True
        )
        
        # Add relationship changes to response
        if rel_result.get("dimensions_diff"):
            response_data["relationship_changes"] = rel_result["dimensions_diff"]
        
        # Check for relationship event
        if rel_result.get("event"):
            response_data["relationship_event"] = rel_result["event"]
        
        # Store the interaction in memory if it's significant
        if response_data["memory_created"] or len(player_input) > 50:
            await self._store_interaction_memory(
                npc_id, 
                player_input, 
                response_data["response"]
            )
        
        # Apply stat changes if any
        if response_data["stat_changes"]:
            await self._apply_stat_changes(npc_id, response_data["stat_changes"])
        
        # Record interaction with the learning adaptation system
        await self._record_interaction_for_learning(
            npc_id,
            interaction_type,
            player_input,
            response_data["response"],
            context
        )
        
        return response_data

    async def _generate_npc_response_with_gpt(
        self,
        npc_id: int,
        npc_details: Dict[str, Any],
        player_input: str,
        interaction_type: str,
        memories: List[Dict[str, Any]],
        relationship: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate NPC response using centralized ChatGPT integration.
        """
        # Build aggregator text with NPC context
        aggregator_text = f"""Current NPC Context:

NPC: {npc_details['npc_name']} (ID: {npc_id})
Personality Stats:
- Dominance: {npc_details.get('dominance', 50)}
- Cruelty: {npc_details.get('cruelty', 50)}
- Closeness: {npc_details.get('closeness', 50)}
- Trust: {npc_details.get('trust', 50)}
- Respect: {npc_details.get('respect', 50)}
- Intensity: {npc_details.get('intensity', 50)}

Personality Traits: {', '.join(npc_details.get('personality_traits', []))}
Current Location: {npc_details.get('current_location', 'unknown')}

Relationship with Player:
- Trust: {relationship.get('trust', 0)}
- Respect: {relationship.get('respect', 0)}
- Affection: {relationship.get('affection', 0)}
- Patterns: {', '.join(relationship.get('patterns', []))}
- Archetypes: {', '.join(relationship.get('archetypes', []))}

Recent Memories:
"""
        
        # Add top 3 memories
        for i, memory in enumerate(memories[:3]):
            aggregator_text += f"- {memory.get('memory_text', 'Unknown memory')}\n"
        
        aggregator_text += f"\nInteraction Type: {interaction_type}"
        
        if context:
            aggregator_text += f"\nAdditional Context: {json.dumps(context, indent=2)}"
        
        aggregator_text += """

Instructions:
Generate a response that:
- Is consistent with the NPC's personality and stats
- Considers their memories and relationship with the player
- Fits the interaction type and context
- Includes subtle elements of control when appropriate (based on dominance)
- Maintains psychological realism

The response should maintain a balance between mundane interaction and subtle power dynamics.
"""
        
        # Call centralized ChatGPT with the universal update function
        result = await get_chatgpt_response(
            conversation_id=self.conversation_id,
            aggregator_text=aggregator_text,
            user_input=f"{npc_details['npc_name']} responds to: \"{player_input}\"",
            reflection_enabled=False,
            use_nyx_integration=False
        )
        
        return result

    async def generate_npc_npc_interaction(
        self,
        npc1_id: int,
        npc2_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate an interaction between two NPCs using async OpenAI client.
        
        Args:
            npc1_id: ID of the first NPC
            npc2_id: ID of the second NPC
            context: Additional context
            
        Returns:
            Dictionary with interaction details
        """
        # Get details for both NPCs
        npc1 = await self.get_npc_details(npc1_id)
        npc2 = await self.get_npc_details(npc2_id)
        
        # Get their relationship using new system
        relationship = await self.get_relationship_details("npc", npc1_id, "npc", npc2_id)
        
        # Get async OpenAI client
        client = get_async_openai_client()
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": f"""You are simulating a conversation between two NPCs in a roleplay game.

NPC 1: {npc1['npc_name']}
- Dominance: {npc1.get('dominance', 50)}
- Personality: {', '.join(npc1.get('personality_traits', []))}

NPC 2: {npc2['npc_name']}
- Dominance: {npc2.get('dominance', 50)}
- Personality: {', '.join(npc2.get('personality_traits', []))}

Their Relationship:
- Trust: {relationship.get('trust', 0)}
- Respect: {relationship.get('respect', 0)}
- Affection: {relationship.get('affection', 0)}
- Patterns: {', '.join(relationship.get('patterns', []))}

Generate a brief, natural interaction between them that reflects their personalities and relationship.
The interaction should be 2-4 exchanges (back and forth).
Include subtle power dynamics if appropriate based on their dominance levels."""
            },
            {
                "role": "user",
                "content": f"Generate a short conversation between {npc1['npc_name']} and {npc2['npc_name']}"
                          + (f" in the context of: {json.dumps(context)}" if context else "")
            }
        ]
        
        try:
            # Use appropriate temperature for dialogue
            temperature = TEMPERATURE_SETTINGS.get("decision", 0.7) + 0.1
            
            response = await call_openai_with_retry(
                client,
                model="gpt-5-nano",
                messages=messages,
            )
            
            interaction_text = response.choices[0].message.content
            
            # Process the NPC-NPC interaction with the relationship system
            from agents import RunContextWrapper
            ctx = RunContextWrapper(context={
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            })
            
            # Process as a shared interaction
            rel_result = await process_relationship_interaction_tool(
                ctx=ctx,
                entity1_type="npc",
                entity1_id=npc1_id,
                entity2_type="npc",
                entity2_id=npc2_id,
                interaction_type="helpful_action",  # Default positive interaction
                context="casual",
                check_for_event=False
            )
            
            return {
                "npc1_id": npc1_id,
                "npc1_name": npc1['npc_name'],
                "npc2_id": npc2_id,
                "npc2_name": npc2['npc_name'],
                "interaction": interaction_text,
                "relationship_changes": rel_result.get("dimensions_diff", {}),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating NPC-NPC interaction: {e}")
            return {
                "npc1_id": npc1_id,
                "npc2_id": npc2_id,
                "interaction": f"{npc1['npc_name']} and {npc2['npc_name']} exchange brief greetings.",
                "success": False,
                "error": str(e)
            }

    async def generate_npc_activity_response(
        self,
        npc_id: int,
        activity_type: str,
        activity_context: Dict[str, Any]
    ) -> str:
        """
        Generate an NPC's response to an activity using GPT.
        
        Args:
            npc_id: ID of the NPC
            activity_type: Type of activity
            activity_context: Context about the activity
            
        Returns:
            Generated response text
        """
        # Get NPC details
        npc_details = await self.get_npc_details(npc_id)
        
        # Get async OpenAI client
        client = get_async_openai_client()
        
        # Build prompt based on activity type
        activity_descriptions = {
            "training": "is undergoing training or conditioning",
            "punishment": "is experiencing consequences for their actions",
            "reward": "is being rewarded",
            "task": "is performing a task",
            "social": "is in a social situation"
        }
        
        activity_desc = activity_descriptions.get(activity_type, "is engaged in an activity")
        
        messages = [
            {
                "role": "system",
                "content": f"""You are generating an NPC's internal thoughts and reactions.

NPC: {npc_details['npc_name']}
- Dominance: {npc_details.get('dominance', 50)}
- Trust: {npc_details.get('trust', 50)}
- Personality: {', '.join(npc_details.get('personality_traits', []))}

The NPC {activity_desc}.

Generate their internal thoughts or reactions (1-2 sentences).
Consider their personality and how they would respond to this situation.
Include subtle psychological elements if appropriate."""
            },
            {
                "role": "user",
                "content": f"Context: {json.dumps(activity_context)}\n\nGenerate {npc_details['npc_name']}'s internal response."
            }
        ]
        
        try:
            temperature = TEMPERATURE_SETTINGS.get("reflection", 0.5)
            
            response = await call_openai_with_retry(
                client,
                model="gpt-5-nano",
                messages=messages,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating activity response: {e}")
            return f"{npc_details['npc_name']} continues with the {activity_type}."

    async def _store_interaction_memory(self, npc_id: int, player_input: str, response: str) -> None:
        """
        Store an interaction in the NPC's memory using canon system.
        
        Args:
            npc_id: ID of the NPC
            player_input: Player's input
            response: NPC's response
        """
        try:
            # Create unified memory entry (allowed)
            memory_text = f"Interaction with player: {player_input} - Response: {response}"
            memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance="medium",
                tags=["interaction", "player_interaction"],
                emotional=True
            )
            
            # Also store in unified_memories table for compatibility
            async with get_db_connection_context() as conn:
                # Create context for canon
                from agents import RunContextWrapper
                from lore.core import canon
                
                ctx = RunContextWrapper(context={
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                    'npc_id': npc_id
                })
                
                # Use canon to create memory entry
                await canon.create_journal_entry(
                    ctx, conn,
                    entry_type="npc_interaction",
                    entry_text=memory_text,
                    tags=["interaction", "npc", f"npc_{npc_id}"],
                    importance=0.5
                )
                
        except Exception as e:
            logger.error(f"Error storing interaction memory: {e}")

    async def _apply_stat_changes(self, npc_id: int, stat_changes: Dict[str, int]) -> None:
        """
        Apply stat changes to an NPC using LoreSystem.
        
        Args:
            npc_id: ID of the NPC
            stat_changes: Dictionary of stat changes
        """
        try:
            # Get LoreSystem instance
            from lore.core.lore_system import LoreSystem
            lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
            
            # Create context for governance
            from agents import RunContextWrapper
            
            ctx = RunContextWrapper(context={
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
                'npc_id': npc_id
            })
            
            # Get current stats first
            async with get_db_connection_context() as conn:
                current_stats = {}
                valid_stats = ["dominance", "cruelty", "closeness", "trust", "respect", "intensity"]
                
                for stat in valid_stats:
                    if stat in stat_changes:
                        row = await conn.fetchrow(
                            f"SELECT {stat} FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3",
                            self.user_id, self.conversation_id, npc_id
                        )
                        if row:
                            current_stats[stat] = row[stat]
            
            # Calculate new values and prepare updates
            updates = {}
            for stat, change in stat_changes.items():
                if stat in current_stats:
                    new_value = max(0, min(100, current_stats[stat] + change))
                    if new_value != current_stats[stat]:
                        updates[stat] = new_value
            
            # Apply updates if any
            if updates:
                result = await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NPCStats",
                    entity_identifier={"npc_id": npc_id},
                    updates=updates,
                    reason=f"Interaction caused stat changes: {stat_changes}"
                )
                
                if result.get("status") != "committed":
                    logger.error(f"Failed to apply stat changes: {result}")
                    
        except Exception as e:
            logger.error(f"Error applying stat changes: {e}")

    async def get_npc_details(self, npc_id: int) -> Dict[str, Any]:
        """
        Get details about a specific NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with NPC details
        """
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT npc_id, npc_name, introduced, archetypes, archetype_summary, 
                       archetype_extras_summary, physical_description, relationships,
                       dominance, cruelty, closeness, trust, respect, intensity,
                       hobbies, personality_traits, likes, dislikes, affiliations,
                       schedule, current_location, sex, age
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                LIMIT 1
            """, self.user_id, self.conversation_id, npc_id)
            
            if not row:
                return {"error": "NPC not found"}
            
            # Process JSON fields
            archetypes = row["archetypes"]
            if archetypes:
                if isinstance(archetypes, str):
                    try:
                        archetypes = json.loads(archetypes)
                    except (json.JSONDecodeError, TypeError):
                        archetypes = []
            else:
                archetypes = []
            
            relationships = row["relationships"]
            if relationships:
                if isinstance(relationships, str):
                    try:
                        relationships = json.loads(relationships)
                    except (json.JSONDecodeError, TypeError):
                        relationships = {}
            else:
                relationships = {}
            
            hobbies = row["hobbies"]
            if hobbies:
                if isinstance(hobbies, str):
                    try:
                        hobbies = json.loads(hobbies)
                    except (json.JSONDecodeError, TypeError):
                        hobbies = []
            else:
                hobbies = []
            
            personality_traits = row["personality_traits"]
            if personality_traits:
                if isinstance(personality_traits, str):
                    try:
                        personality_traits = json.loads(personality_traits)
                    except (json.JSONDecodeError, TypeError):
                        personality_traits = []
            else:
                personality_traits = []
            
            likes = row["likes"]
            if likes:
                if isinstance(likes, str):
                    try:
                        likes = json.loads(likes)
                    except (json.JSONDecodeError, TypeError):
                        likes = []
            else:
                likes = []
            
            dislikes = row["dislikes"]
            if dislikes:
                if isinstance(dislikes, str):
                    try:
                        dislikes = json.loads(dislikes)
                    except (json.JSONDecodeError, TypeError):
                        dislikes = []
            else:
                dislikes = []
            
            affiliations = row["affiliations"]
            if affiliations:
                if isinstance(affiliations, str):
                    try:
                        affiliations = json.loads(affiliations)
                    except (json.JSONDecodeError, TypeError):
                        affiliations = []
            else:
                affiliations = []
            
            schedule = row["schedule"]
            if schedule:
                if isinstance(schedule, str):
                    try:
                        schedule = json.loads(schedule)
                    except (json.JSONDecodeError, TypeError):
                        schedule = {}
            else:
                schedule = {}
            
            return {
                "npc_id": row["npc_id"],
                "npc_name": row["npc_name"],
                "introduced": row["introduced"],
                "archetypes": archetypes,
                "archetype_summary": row["archetype_summary"],
                "archetype_extras_summary": row["archetype_extras_summary"],
                "physical_description": row["physical_description"],
                "relationships": relationships,
                "dominance": row["dominance"],
                "cruelty": row["cruelty"],
                "closeness": row["closeness"],
                "trust": row["trust"],
                "respect": row["respect"],
                "intensity": row["intensity"],
                "hobbies": hobbies,
                "personality_traits": personality_traits,
                "likes": likes,
                "dislikes": dislikes,
                "affiliations": affiliations,
                "schedule": schedule,
                "current_location": row["current_location"],
                "sex": row["sex"],
                "age": row["age"]
            }

    async def get_npc_memory(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get memories associated with an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            List of memory objects
        """
        async with get_db_connection_context() as conn:
            # First check if the NPC has memories in the NPCStats table
            row = await conn.fetchrow("""
                SELECT memory
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                LIMIT 1
            """, self.user_id, self.conversation_id, npc_id)
            
            if row and row["memory"]:
                try:
                    if isinstance(row["memory"], str):
                        memories = json.loads(row["memory"])
                    else:
                        memories = row["memory"]
                    
                    return memories
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Try the legacy NPCMemories table
            rows = await conn.fetch("""
                SELECT id, memory_text, emotional_intensity, significance, memory_type
                FROM NPCMemories
                WHERE npc_id=$1 AND status='active'
                ORDER BY timestamp DESC
                LIMIT 10
            """, npc_id)
            
            if rows:
                memories = []
                for row in rows:
                    memories.append({
                        "id": row["id"],
                        "memory_text": row["memory_text"],
                        "emotional_intensity": row["emotional_intensity"],
                        "significance": row["significance"],
                        "memory_type": row["memory_type"]
                    })
                
                return memories
            
            # Finally, try the unified_memories table
            rows = await conn.fetch("""
                SELECT id, memory_text, emotional_intensity, significance, memory_type
                FROM unified_memories
                WHERE entity_type='npc' AND entity_id=$1 AND user_id=$2 AND conversation_id=$3 AND status='active'
                ORDER BY timestamp DESC
                LIMIT 10
            """, npc_id, self.user_id, self.conversation_id)
            
            memories = []
            for row in rows:
                memories.append({
                    "id": row["id"],
                    "memory_text": row["memory_text"],
                    "emotional_intensity": row["emotional_intensity"],
                    "significance": row["significance"],
                    "memory_type": row["memory_type"]
                })
            
            return memories

    async def get_relationship_details(
        self,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int
    ) -> Dict[str, Any]:
        """
        Get relationship details between two entities using new system.
        
        Args:
            entity1_type: Type of the first entity (e.g., "npc", "player")
            entity1_id: ID of the first entity
            entity2_type: Type of the second entity
            entity2_id: ID of the second entity
            
        Returns:
            Dictionary with relationship details
        """
        from agents import RunContextWrapper
        
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        
        # Get relationship summary using new tool
        relationship = await get_relationship_summary_tool(
            ctx=ctx,
            entity1_type=entity1_type,
            entity1_id=entity1_id,
            entity2_type=entity2_type,
            entity2_id=entity2_id
        )
        
        # Convert to the format expected by the rest of the code
        dimensions = relationship.get('dimensions', {})
        
        return {
            "entity1_type": entity1_type,
            "entity1_id": entity1_id,
            "entity2_type": entity2_type,
            "entity2_id": entity2_id,
            "trust": dimensions.get('trust', 0),
            "respect": dimensions.get('respect', 0),
            "affection": dimensions.get('affection', 0),
            "intimacy": dimensions.get('intimacy', 0),
            "influence": dimensions.get('influence', 0),
            "volatility": dimensions.get('volatility', 0),
            "patterns": relationship.get('patterns', []),
            "archetypes": relationship.get('archetypes', []),
            "momentum_magnitude": relationship.get('momentum_magnitude', 0),
            "duration_days": relationship.get('duration_days', 0)
        }

    async def get_nearby_npcs(self, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get NPCs that are at a specific location.
        
        Args:
            location: Location to filter by (optional)
            
        Returns:
            List of nearby NPCs
        """
        async with get_db_connection_context() as conn:
            if location:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 
                    AND current_location=$3
                    ORDER BY introduced DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id, location)
            else:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                    ORDER BY introduced DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
            
            nearby_npcs = []
            for row in rows:
                nearby_npcs.append({
                    "npc_id": row["npc_id"],
                    "npc_name": row["npc_name"],
                    "current_location": row["current_location"],
                    "dominance": row["dominance"],
                    "cruelty": row["cruelty"]
                })
            
            return nearby_npcs

    async def process_daily_npc_activities(self) -> Dict[str, Any]:
        """
        Process daily activities for all NPCs using canon system.
        Update locations, create memories, and handle NPC interactions.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Get LoreSystem instance
            from lore.core import canon
            from lore.core.lore_system import LoreSystem
            lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
            
            # Create context for governance
            from agents import RunContextWrapper
            
            ctx = RunContextWrapper(context={
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            })
            
            async with get_db_connection_context() as conn:
                # Get current time data
                year, month, day, time_of_day = 1, 1, 1, "Morning"
                
                for key in ["CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"]:
                    row = await conn.fetchrow(
                        "SELECT value FROM CurrentRoleplay WHERE user_id=$1 AND conversation_id=$2 AND key=$3",
                        self.user_id, self.conversation_id, key
                    )
                    
                    if row:
                        if key == "CurrentYear":
                            year = int(row["value"]) if row["value"].isdigit() else 1
                        elif key == "CurrentMonth":
                            month = int(row["value"]) if row["value"].isdigit() else 1
                        elif key == "CurrentDay":
                            day = int(row["value"]) if row["value"].isdigit() else 1
                        elif key == "TimeOfDay":
                            time_of_day = row["value"]
                
                # Get day name
                row = await conn.fetchrow(
                    "SELECT value FROM CurrentRoleplay WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'",
                    self.user_id, self.conversation_id
                )
                
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                if row:
                    try:
                        calendar_data = json.loads(row["value"])
                        if "days" in calendar_data:
                            day_names = calendar_data["days"]
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                day_of_week = day_names[day % len(day_names)]
                
                # Get all NPCs
                rows = await conn.fetch(
                    "SELECT npc_id, npc_name, schedule, current_location FROM NPCStats WHERE user_id=$1 AND conversation_id=$2",
                    self.user_id, self.conversation_id
                )
                
                results = []
                
                for row in rows:
                    npc_id = row["npc_id"]
                    npc_name = row["npc_name"]
                    schedule = row["schedule"]
                    current_location = row["current_location"]
                    
                    # Parse schedule
                    if schedule:
                        try:
                            if isinstance(schedule, str):
                                schedule_data = json.loads(schedule)
                            else:
                                schedule_data = schedule
                            
                            # Check if this day is in the schedule
                            if day_of_week in schedule_data:
                                day_schedule = schedule_data[day_of_week]
                                
                                # Check if the current time period is in the schedule
                                if time_of_day in day_schedule:
                                    new_location = day_schedule[time_of_day]
                                    
                                    # Update location if different using LoreSystem
                                    if new_location and new_location != current_location:
                                        # Update context for this NPC
                                        ctx.npc_id = npc_id
                                        
                                        result = await lore_system.propose_and_enact_change(
                                            ctx=ctx,
                                            entity_type="NPCStats",
                                            entity_identifier={"npc_id": npc_id},
                                            updates={"current_location": new_location},
                                            reason=f"Scheduled movement to {new_location} during {time_of_day} on {day_of_week}"
                                        )
                                        
                                        if result.get("status") == "committed":
                                            # Create memory of location change using canon
                                            memory_text = f"Moved to {new_location} during {time_of_day} on {day_of_week}."
                                            
                                            await canon.create_journal_entry(
                                                ctx, conn,
                                                entry_type="movement",
                                                entry_text=memory_text,
                                                tags=["movement", "location_change", f"npc_{npc_id}"],
                                                importance=0.2
                                            )
                                            
                                            results.append({
                                                "npc_id": npc_id,
                                                "npc_name": npc_name,
                                                "action": "moved",
                                                "old_location": current_location,
                                                "new_location": new_location
                                            })
                        except (json.JSONDecodeError, TypeError):
                            pass
            
            # Handle NPC interactions
            # Get NPCs at the same location
            location_npcs = {}
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, current_location
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
            """, self.user_id, self.conversation_id)
            
            for row in rows:
                location = row["current_location"]
                if location not in location_npcs:
                    location_npcs[location] = []
                
                location_npcs[location].append({
                    "npc_id": row["npc_id"],
                    "npc_name": row["npc_name"]
                })
            
            # Process interactions for NPCs at the same location
            for location, npcs in location_npcs.items():
                if len(npcs) >= 2:
                    # Use GPT to generate more dynamic interactions
                    for _ in range(min(3, len(npcs))):
                        npc1, npc2 = random.sample(npcs, 2)
                        
                        # Generate interaction using GPT
                        interaction_result = await self.generate_npc_npc_interaction(
                            npc1['npc_id'],
                            npc2['npc_id'],
                            {"location": location, "time_of_day": time_of_day}
                        )
                        
                        if interaction_result.get('success'):
                            # Store the generated interaction as memories
                            interaction_text = interaction_result['interaction']
                            
                            # Create memories based on the generated interaction
                            await self._store_interaction_memory(
                                npc1['npc_id'],
                                f"Interaction with {npc2['npc_name']}",
                                interaction_text
                            )
                            
                            await self._store_interaction_memory(
                                npc2['npc_id'],
                                f"Interaction with {npc1['npc_name']}",
                                interaction_text
                            )
                            
                            results.append({
                                "type": "interaction",
                                "npc1": npc1["npc_name"],
                                "npc2": npc2["npc_name"],
                                "location": location,
                                "interaction": interaction_text,
                                "relationship_changes": interaction_result.get('relationship_changes', {})
                            })
            
            return {
                "year": year,
                "month": month,
                "day": day,
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error processing daily activities: {e}")
            return {"error": str(e), "results": []}

    async def _record_interaction_for_learning(
        self, 
        npc_id: int, 
        interaction_type: str, 
        player_input: str, 
        npc_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an interaction with the NPC learning adaptation system.
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction
            player_input: Player's input
            npc_response: NPC's response
            context: Optional context information
        """
        try:
            # Get or create a learning system for this NPC
            learning_system = self.learning_manager.get_learning_system_for_npc(npc_id)
            await learning_system.initialize()
            
            # Determine player response characteristics based on the input
            # Enhanced version with more comprehensive analysis
            player_input_lower = player_input.lower()
            
            # Initialize variables
            compliance_level = 0
            emotional_response = "neutral"
            intensity_level = 0
            respect_level = 0
            fear_level = 0
            
            # ENHANCED: Structured emotion and response detection
            emotion_patterns = {
                # Compliance patterns
                "strong_compliance": [
                    "absolutely", "of course", "my pleasure", "happy to", "gladly",
                    "at your service", "as you command", "without question", "immediately",
                    "right away", "certainly", "with pleasure", "i'd be delighted"
                ],
                "compliance": [
                    "yes", "okay", "sure", "fine", "will do", "as you wish", "alright",
                    "i'll do it", "agreed", "very well", "i understand", "if you say so"
                ],
                "reluctant_compliance": [
                    "i guess", "if i must", "if i have to", "suppose so", "whatever",
                    "fine then", "sigh", "hesitantly", "reluctantly", "not like i have a choice"
                ],
                
                # Defiance patterns
                "strong_defiance": [
                    "absolutely not", "never", "no way", "forget it", "i refuse",
                    "not a chance", "hell no", "over my dead body", "i won't", "not happening"
                ],
                "defiance": [
                    "no", "won't", "refuse", "not going to", "can't make me", "i don't think so",
                    "nope", "not interested", "i decline", "not now", "i don't want to"
                ],
                "mild_defiance": [
                    "maybe later", "not right now", "i'd rather not", "do i have to",
                    "is this necessary", "can we do something else", "i don't feel like it"
                ],
                
                # Emotional responses
                "fear": [
                    "afraid", "scared", "terrified", "fear", "worried", "nervous", "anxious",
                    "frightened", "petrified", "dread", "panic", "horror", "trembling"
                ],
                "anger": [
                    "angry", "mad", "furious", "outraged", "annoyed", "irritated", "enraged",
                    "hostile", "resent", "hate", "despise", "contempt", "disgusted"
                ],
                "sadness": [
                    "sad", "upset", "unhappy", "miserable", "depressed", "heartbroken",
                    "devastated", "disappointed", "dejected", "hopeless", "grief", "sorrow"
                ],
                "joy": [
                    "happy", "excited", "delighted", "pleased", "glad", "joyful", "thrilled",
                    "ecstatic", "content", "satisfied", "elated", "jubilant", "overjoyed"
                ],
                "surprise": [
                    "surprised", "shocked", "astonished", "amazed", "stunned", "startled",
                    "bewildered", "confused", "taken aback", "unexpected", "wow", "oh"
                ],
                
                # Respect and submission
                "respect": [
                    "respect", "admire", "look up to", "honor", "esteem", "regard highly",
                    "value your", "appreciate your", "trust your", "your wisdom", "your guidance"
                ],
                "submission": [
                    "submit", "yield", "surrender", "obey", "comply", "follow", "serve",
                    "bow to", "defer to", "at your mercy", "under your control", "your command"
                ]
            }
            
            # Analyze for each emotional pattern
            detected_patterns = {}
            for category, patterns in emotion_patterns.items():
                detected_patterns[category] = 0
                for pattern in patterns:
                    if pattern in player_input_lower:
                        detected_patterns[category] += 1
            
            # Calculate strongest emotional response
            strongest_emotion = "neutral"
            strongest_value = 0
            emotional_categories = ["fear", "anger", "sadness", "joy", "surprise"]
            for emotion in emotional_categories:
                if detected_patterns[emotion] > strongest_value:
                    strongest_value = detected_patterns[emotion]
                    strongest_emotion = emotion
            
            # Only set emotional response if we detected something
            if strongest_value > 0:
                emotional_response = strongest_emotion
            
            # Calculate compliance level
            if detected_patterns["strong_compliance"] > 0:
                compliance_level = 8
            elif detected_patterns["compliance"] > 0:
                compliance_level = 5
            elif detected_patterns["reluctant_compliance"] > 0:
                compliance_level = 2
            elif detected_patterns["mild_defiance"] > 0:
                compliance_level = -3
            elif detected_patterns["defiance"] > 0:
                compliance_level = -5
            elif detected_patterns["strong_defiance"] > 0:
                compliance_level = -8
            
            # Calculate respect level
            respect_level = detected_patterns["respect"] * 2
            
            # Calculate submission level
            submission_level = detected_patterns["submission"] * 2
            
            # Calculate fear level
            fear_level = detected_patterns["fear"] * 2
            
            # Adjust compliance level based on fear (scared compliance is different)
            if compliance_level > 0 and fear_level > 0:
                # If complying out of fear, mark it differently
                compliance_type = "fearful_compliance"
            elif compliance_level > 0 and respect_level > 0:
                # If complying out of respect, mark it differently
                compliance_type = "respectful_compliance"
            elif compliance_level > 0 and submission_level > 0:
                # If complying out of submission, mark it differently
                compliance_type = "submissive_compliance"
            elif compliance_level > 0:
                compliance_type = "willing_compliance"
            elif compliance_level < 0:
                compliance_type = "defiance"
            else:
                compliance_type = "neutral"
            
            # Calculate intensity based on the strength of the emotional response
            intensity_level = max(
                detected_patterns["strong_compliance"] * 2,
                detected_patterns["strong_defiance"] * 2,
                detected_patterns["fear"] * 2,
                detected_patterns["anger"] * 2,
                detected_patterns["submission"] * 2
            )
            
            # Create a detailed analysis result
            analysis_result = {
                "compliance_level": compliance_level,
                "compliance_type": compliance_type,
                "emotional_response": emotional_response,
                "intensity_level": intensity_level,
                "respect_level": respect_level,
                "fear_level": fear_level,
                "submission_level": submission_level,
                "pattern_matches": detected_patterns
            }
            
            # Log detailed analysis for debugging
            logging.debug(f"Player input analysis for NPC {npc_id}: {analysis_result}")
            
            # Record the learning interaction with enhanced analysis
            await learning_system.record_player_interaction(
                interaction_type=interaction_type,
                player_input=player_input,
                npc_response=npc_response,
                compliance_level=compliance_level,
                emotional_response=emotional_response,
                context=context,
                analysis_result=analysis_result
            )
            
            # Log the learning interaction
            logging.info(f"Recorded learning interaction for NPC {npc_id}: {interaction_type}")
            
        except Exception as e:
            logging.error(f"Error recording interaction for learning: {e}", exc_info=True)

    # Add a method to explicitly trigger memory-based learning
    async def process_npc_learning_cycle(self, npc_id: int) -> Dict[str, Any]:
        """
        Trigger a learning cycle for an NPC to process memories and adapt behavior.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with learning results
        """
        try:
            # Get learning system for this NPC
            learning_system = self.learning_manager.get_learning_system_for_npc(npc_id)
            await learning_system.initialize()
            
            # Process memories for learning
            memory_result = await learning_system.process_recent_memories_for_learning()
            
            # Process relationship changes
            relationship_result = await learning_system.adapt_to_relationship_changes()
            
            return {
                "npc_id": npc_id,
                "memory_learning": memory_result,
                "relationship_adaptation": relationship_result,
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Error processing NPC learning cycle: {e}", exc_info=True)
            return {
                "npc_id": npc_id,
                "success": False,
                "error": str(e)
            }
    
    async def check_relationship_events(self) -> List[Dict[str, Any]]:
        """
        Check for any pending relationship events
        
        Returns:
            List of pending events
        """
        from agents import RunContextWrapper
        
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        
        # Poll for events
        events = []
        for _ in range(10):  # Check up to 10 events
            event_result = await poll_relationship_events_tool(ctx=ctx, timeout=0.01)
            if event_result.get("has_event"):
                events.append(event_result["event"])
            else:
                break
        
        return events
