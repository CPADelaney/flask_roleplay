# npcs/npc_behavior.py

"""
BehaviorEvolution system that evolves NPC behaviors over time.
Refactored from the original behavior_evolution.py.
"""

import random
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Database connection
from db.connection import get_db_connection_context

from memory.wrapper import MemorySystem
from agents import RunContextWrapper



logger = logging.getLogger(__name__)

# npcs/npc_behavior.py

"""
BehaviorEvolution system that evolves NPC behaviors over time.
Combined version with all features using asyncpg/PostgreSQL.
"""

import random
import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncpg

# Database connection
from db.connection import get_db_connection_context

from memory.wrapper import MemorySystem
from agents import RunContextWrapper

logger = logging.getLogger(__name__)

class BehaviorEvolution:
    """
    Evolves NPC behavior over time, modifying their tactics based on past events.
    NPCs will develop hidden agendas, adjust their manipulation strategies, 
    and attempt to control the world around them.
    """

    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the behavior evolution system.
        
        Args:
            user_id: The user/player ID
            conversation_id: The current conversation/scene ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None
        
        # Cache for NPC data to reduce DB calls
        self.npc_data_cache = {}
        self.cache_expiry = {}
        self.cache_ttl = timedelta(minutes=5)  # Cache TTL for NPC data

    async def get_memory_system(self) -> MemorySystem:
        """Lazy-load the memory system."""
        if self.memory_system is None:
            self.memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self.memory_system

    async def evaluate_npc_scheming_with_user_model(self, npc_id: int) -> Dict[str, Any]:
        """
        Evolve NPC behavior considering user preferences.
        
        Args:
            npc_id: ID of the NPC to evolve
            
        Returns:
            Dictionary with evolution adjustments
        """
        # Get base evolution
        base_adjustments = await self.evaluate_npc_scheming(npc_id)
        
        try:
            # Get user model from Nyx (placeholder - to be implemented)
            from npcs.user_model import UserModelManager
            user_model = UserModelManager(self.user_id, self.conversation_id)
            user_guidance = await user_model.get_response_guidance()
            
            # Adjust scheming based on user preferences
            if user_guidance and "suggested_dominance" in user_guidance:
                suggestion = user_guidance.get("suggested_dominance", 0.5)
                if suggestion > 0.7:
                    # User prefers dominant characters, enhance scheming
                    base_adjustments["scheme_level"] += 2
                elif suggestion < 0.3:
                    # User prefers less dominant characters, reduce scheming
                    base_adjustments["scheme_level"] = max(0, base_adjustments["scheme_level"] - 1)
            
            return base_adjustments
        except ImportError:
            # If user model not available, return base adjustments
            logger.info("UserModelManager not available, returning base adjustments")
            return base_adjustments
        except Exception as e:
            logger.error(f"Error in evaluate_npc_scheming_with_user_model: {e}")
            return base_adjustments

    async def evaluate_npc_scheming(self, npc_id: int) -> Dict[str, Any]:
        """
        Evaluate if an NPC should adjust their behavior, escalate plans, or set new secret goals.
        This is called periodically to evolve NPC behavior.

        Args:
            npc_id: ID of the NPC to evaluate
        
        Returns:
            Dictionary containing the NPC's updated behavior adjustments
        """
        try:
            memory_system = await self.get_memory_system()
            
            # Retrieve NPC history
            npc_data = await self._get_npc_data(npc_id)
            if not npc_data:
                logger.warning(f"NPC data not found for {npc_id} during scheming eval.")
                return {"error": "NPC data not found"}

            name = npc_data["npc_name"]
            dominance = npc_data["dominance"]
            cruelty = npc_data["cruelty"]
            paranoia = "paranoid" in npc_data.get("personality_traits", [])
            deceptive = "manipulative" in npc_data.get("personality_traits", [])

            # Retrieve past manipulations & betrayals
            betrayals = await memory_system.recall(
                entity_type="npc", 
                entity_id=npc_id,
                query="betrayal", 
                limit=5
            )
            successful_lies = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="deception success", 
                limit=5
            )
            failed_lies = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="deception failure", 
                limit=3
            )
            loyalty_tests = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="tested loyalty", 
                limit=3
            )

            # Define NPC evolution behavior
            adjustments = {
                "scheme_level": 0,  # Determines how aggressive their plotting is
                "trust_modifiers": {},
                "loyalty_tests": 0,
                "betrayal_planning": False,
                "targeting_player": False,
                "npc_recruits": [],
                "paranoia_level": 5 if paranoia else 2
            }

            # Adjust scheming level based on success rate
            if "memories" in successful_lies:
                adjustments["scheme_level"] += len(successful_lies["memories"])
            if "memories" in failed_lies:
                adjustments["scheme_level"] -= len(failed_lies["memories"])  # Punishes failures
            if "memories" in betrayals:
                adjustments["scheme_level"] += len(betrayals["memories"]) * 2  # Increases scheming if they've been betrayed

            # If their deception is failing often, they become either cautious or reckless
            if "memories" in failed_lies and paranoia:
                adjustments["scheme_level"] += 3  # Paranoia increases scheming
                adjustments["paranoia_level"] += 2  # Increase paranoia level too

            # If an NPC has tested loyalty and found weak targets, they begin manipulating more
            if "memories" in loyalty_tests:
                adjustments["loyalty_tests"] += len(loyalty_tests["memories"])
                
                # Extract NPCs who failed loyalty tests
                weak_targets = []
                for memory in loyalty_tests.get("memories", []):
                    memory_text = memory.get("text", "").lower()
                    # Extract NPC IDs from text (requires consistent memory format)
                    if "npc_" in memory_text and "failed" in memory_text:
                        for word in memory_text.split():
                            if word.startswith("npc_") and word[4:].isdigit():
                                weak_targets.append(int(word[4:]))

                adjustments["npc_recruits"].extend(weak_targets)

            # Dominant NPCs escalate manipulation if they see success
            if dominance > 70 and successful_lies.get("memories", []):
                adjustments["scheme_level"] += 2

            # Cruel NPCs escalate based on betrayals
            if cruelty > 70 and betrayals.get("memories", []):
                adjustments["betrayal_planning"] = True

            # Paranoid NPCs will target anyone they suspect of deception
            if paranoia and failed_lies.get("memories", []):
                adjustments["targeting_player"] = True

            # Final checks: If the NPC is in full scheming mode, they begin long-term plans
            if adjustments["scheme_level"] >= 5:
                logger.info(f"{name} is entering full scheming mode.")

                # Set a secret goal
                secret_goal = f"{name} is planning to manipulate the world around them."
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=secret_goal,
                    importance="high",
                    emotional=True
                )

                # If deceptive, they will now actively deceive the player
                if deceptive:
                    adjustments["targeting_player"] = True

                # NPC starts actively recruiting allies if they aren't already doing so
                if not adjustments["npc_recruits"]:
                    all_npcs = await self._get_all_npcs()
                    potential_recruits = [n["npc_id"] for n in all_npcs if n.get("dominance", 50) < 50]
                    adjustments["npc_recruits"].extend(potential_recruits[:2])

            return adjustments

        except Exception as e:
            logger.error(f"Error evaluating NPC scheming: {e}")
            return {"error": str(e)}

    async def _get_npc_data(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve NPC data from database with caching.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with NPC data or None if not found
        """
        # Check cache first
        cache_key = f"npc_{npc_id}"
        
        now = datetime.now()
        if (cache_key in self.npc_data_cache and 
            cache_key in self.cache_expiry and 
            self.cache_expiry[cache_key] > now):
            return self.npc_data_cache[cache_key]
        
        # Not in cache or expired, fetch from DB
        query = """
            SELECT npc_id, npc_name, dominance, cruelty, personality_traits,
                   scheming_level, betrayal_planning
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(query, npc_id, self.user_id, self.conversation_id)

            if row:
                # Parse personality_traits safely
                traits = []
                raw_traits = row['personality_traits']
                if raw_traits:
                    try:
                        # asyncpg might already parse JSONB/JSON, check type
                        if isinstance(raw_traits, list):
                            traits = raw_traits
                        elif isinstance(raw_traits, str):
                            traits = json.loads(raw_traits)
                        # Add handling for dict if needed, though list seems expected
                    except (json.JSONDecodeError, TypeError) as parse_err:
                        logger.warning(f"Failed to parse personality_traits for NPC {npc_id}: {parse_err}. Data: {raw_traits}")
                        traits = []

                npc_data = {
                    "npc_id": row['npc_id'],
                    "npc_name": row['npc_name'],
                    "dominance": row['dominance'],
                    "cruelty": row['cruelty'],
                    "personality_traits": traits,
                    "scheming_level": row['scheming_level'] if row['scheming_level'] is not None else 0,
                    "betrayal_planning": bool(row['betrayal_planning']) if row['betrayal_planning'] is not None else False
                }
                
                # Update cache
                self.npc_data_cache[cache_key] = npc_data
                self.cache_expiry[cache_key] = now + self.cache_ttl
                
                return npc_data
            else:
                return None


class NPCBehavior:
    """Manages NPC behavior and decision-making."""
    def __init__(self, npc_id: int):
        self.npc_id = npc_id
        self.nyx_client = None
        self.memory_system = MemorySystem()
        self._user_model = None
    
    def get_nyx_client(self):
        """Lazy-load the Nyx client to avoid circular imports."""
        if self.nyx_client is None:
            from nyx.integrate import get_nyx_client
            self.nyx_client = get_nyx_client()
        return self.nyx_client
    
    async def get_user_model(self) -> Dict[str, Any]:
        """Get or create the Nyx user model for this NPC."""
        if self._user_model is None:
            try:
                # Get NPC data using asyncpg
                query = """
                    SELECT name, personality, traits, background
                    FROM NPCs
                    WHERE id = $1
                """
                async with get_db_connection_context() as conn:
                    row = await conn.fetchrow(query, self.npc_id)
                    
                    if not row:
                        raise ValueError(f"NPC {self.npc_id} not found")
                    
                    # Create user model from NPC data
                    self._user_model = {
                        'id': f"npc_{self.npc_id}",
                        'name': row['name'],
                        'personality': row['personality'],
                        'traits': row['traits'],
                        'background': row['background'],
                        'type': 'npc',
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Register with Nyx
                    nyx_client = self.get_nyx_client()
                    await nyx_client.register_user(self._user_model)
                    
                    # Initialize memory system
                    await self.memory_system.initialize(self._user_model['id'])
            except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
                logger.error(f"Database error creating user model for NPC {self.npc_id}: {db_err}")
                raise
            except Exception as e:
                logger.error(f"Error creating user model for NPC {self.npc_id}: {e}")
                raise
        
        return self._user_model
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on context and NPC's personality."""
        try:
            # Ensure user model exists
            user_model = await self.get_user_model()
            
            # Get relevant memories
            memories = await self.memory_system.get_relevant_memories(
                context,
                limit=5
            )
            
            # Prepare decision context
            decision_context = {
                'npc': user_model,
                'context': context,
                'memories': memories,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get decision from Nyx
            nyx_client = self.get_nyx_client()
            decision = await nyx_client.get_decision(decision_context)
            
            # Store decision in memory
            await self.memory_system.add_memory(
                memory_text=f"Made decision: {decision['action']}",
                memory_type="decision",
                importance="medium",
                metadata={
                    'action': decision['action'],
                    'reasoning': decision['reasoning'],
                    'context': context
                }
            )
            
            return decision
        except Exception as e:
            logger.error(f"Error making decision for NPC {self.npc_id}: {e}")
            return {
                'action': 'wait',
                'reasoning': 'Error in decision making process',
                'error': str(e)
            }
    
    async def process_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process an interaction with the NPC."""
        try:
            # Ensure user model exists
            user_model = await self.get_user_model()
            
            # Get relevant memories
            memories = await self.memory_system.get_relevant_memories(
                interaction,
                limit=5
            )
            
            # Prepare interaction context
            interaction_context = {
                'npc': user_model,
                'interaction': interaction,
                'memories': memories,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get response from Nyx
            nyx_client = self.get_nyx_client()
            response = await nyx_client.get_response(interaction_context)
            
            # Store interaction in memory
            await self.memory_system.add_memory(
                memory_text=f"Interaction: {interaction.get('type', 'unknown')}",
                memory_type="interaction",
                importance="medium",
                metadata={
                    'interaction': interaction,
                    'response': response,
                    'context': interaction_context
                }
            )
            
            return response
        except Exception as e:
            logger.error(f"Error processing interaction for NPC {self.npc_id}: {e}")
            return {
                'response': "I'm not sure how to respond to that.",
                'error': str(e)
            }
    
    async def update_state(self, new_state: Dict[str, Any]) -> bool:
        """Update the NPC's state."""
        try:
            # Ensure user model exists
            user_model = await self.get_user_model()
            
            # Update state in database using asyncpg
            query = """
                UPDATE NPCs
                SET state = $1, updated_at = NOW()
                WHERE id = $2
            """
            async with get_db_connection_context() as conn:
                await conn.execute(query, json.dumps(new_state), self.npc_id)
            
            # Store state change in memory
            await self.memory_system.add_memory(
                memory_text="State updated",
                memory_type="state_change",
                importance="low",
                metadata={
                    'old_state': user_model.get('state', {}),
                    'new_state': new_state
                }
            )
            
            # Update user model
            user_model['state'] = new_state
            return True
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error updating state for NPC {self.npc_id}: {db_err}")
            return False
        except Exception as e:
            logger.error(f"Error updating state for NPC {self.npc_id}: {e}")
            return False  # NPC not found
                
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error fetching NPC data for {npc_id}: {db_err}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching NPC data for {npc_id}: {e}", exc_info=True)
            return None

    async def _get_all_npcs(self) -> List[Dict[str, Any]]:
        """
        Get all NPCs for the current user/conversation.
        
        Returns:
            List of NPC data dictionaries
        """
        npcs = []
        query = """
            SELECT npc_id, npc_name, dominance, cruelty
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2
        """
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(query, self.user_id, self.conversation_id)

            for row in rows:
                npcs.append({
                    "npc_id": row['npc_id'],
                    "npc_name": row['npc_name'],
                    "dominance": row['dominance'],
                    "cruelty": row['cruelty']
                })
            return npcs
            
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error fetching all NPCs for user {self.user_id}, convo {self.conversation_id}: {db_err}", exc_info=True)
            return []  # Return empty list on error
        except Exception as e:
            logger.error(f"Unexpected error fetching all NPCs: {e}", exc_info=True)
            return []

    async def apply_scheming_adjustments(self, npc_id: int, adjustments: Dict[str, Any]) -> bool:
        """
        Apply scheming behavior adjustments to the NPC using LoreSystem.
        
        Args:
            npc_id: ID of the NPC
            adjustments: The adjustments to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get LoreSystem instance
            from lore.core.lore_system import LoreSystem
            lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
            
            # Create context for governance
            ctx = RunContextWrapper(context={
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
                'npc_id': npc_id
            })
            
            # Prepare updates
            updates = {}
            
            new_level = adjustments.get("scheme_level", 0)
            updates["scheming_level"] = new_level
            
            betrayal_planning = adjustments.get("betrayal_planning", False)
            updates["betrayal_planning"] = betrayal_planning
            
            # Use LoreSystem to update
            result = await lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="NPCStats",
                entity_identifier={"npc_id": npc_id},
                updates=updates,
                reason=f"Scheming adjustments applied: level={new_level}, betrayal={betrayal_planning}"
            )
            
            if result.get("status") == "committed":
                # Optional: Log successful memory update after DB commit
                if adjustments.get("targeting_player"):
                    memory_system = await self.get_memory_system()
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text="I decided to target the player, suspecting them of deception.",
                        importance="high",
                        tags=["scheming", "targeting_player"]
                    )
                
                # Invalidate cache for this NPC
                cache_key = f"npc_{npc_id}"
                if cache_key in self.npc_data_cache:
                    del self.npc_data_cache[cache_key]
                    if cache_key in self.cache_expiry:
                        del self.cache_expiry[cache_key]
                
                logger.info(f"Applied scheming adjustments for NPC {npc_id}: level={new_level}, betrayal={betrayal_planning}")
                return True
            else:
                logger.error(f"Failed to apply scheming adjustments via LoreSystem: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying scheming adjustments for NPC {npc_id}: {e}", exc_info=True)
            return False

    async def evaluate_npc_scheming_for_all(self, npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate and update scheming behavior for multiple NPCs.
        
        Args:
            npc_ids: List of NPC IDs to evaluate
            
        Returns:
            Dictionary mapping NPC IDs to their scheming adjustments
        """
        results = {}
        
        for npc_id in npc_ids:
            try:
                # Evaluate scheming for this NPC
                adjustments = await self.evaluate_npc_scheming(npc_id)
                
                # Apply the adjustments
                if "error" not in adjustments:
                    success = await self.apply_scheming_adjustments(npc_id, adjustments)
                    adjustments["applied"] = success
                
                results[npc_id] = adjustments
            except Exception as e:
                logger.error(f"Error evaluating scheming for NPC {npc_id}: {e}")
                results[npc_id] = {"error": str(e)}
        
        return results

    async def generate_scheming_opportunity(self, npc_id: int, trigger_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a scheming opportunity based on a trigger event.
        
        Args:
            npc_id: ID of the NPC
            trigger_event: Event that might trigger scheming
            
        Returns:
            Opportunity details if one is generated, None otherwise
        """
        # Check if the NPC is prone to scheming
        npc_data = await self._get_npc_data(npc_id)
        if not npc_data:
            return None
        
        scheming_level = npc_data.get("scheming_level", 0)
        if scheming_level < 3:
            # Not scheming enough to generate opportunities
            return None
        
        # Get emotional state
        memory_system = await self.get_memory_system()
        emotional_state = await memory_system.get_npc_emotion(npc_id)
        
        # Extract emotion data
        current_emotion = None
        emotional_intensity = 0.0
        if emotional_state and "current_emotion" in emotional_state:
            emotion_data = emotional_state["current_emotion"]
            if isinstance(emotion_data.get("primary"), dict):
                current_emotion = emotion_data["primary"].get("name")
                emotional_intensity = emotion_data["primary"].get("intensity", 0.0)
            else:
                current_emotion = emotion_data.get("primary")
                emotional_intensity = emotion_data.get("intensity", 0.0)
        
        # Determine if this is a good opportunity
        trigger_type = trigger_event.get("type", "unknown")
        is_opportunity = False
        opportunity_type = "none"
        
        # Vulnerability opportunities
        if trigger_type == "vulnerability" or "vulnerable" in trigger_event.get("description", "").lower():
            is_opportunity = True
            opportunity_type = "exploit_vulnerability"
        
        # Information opportunities
        elif trigger_type == "information" or "secret" in trigger_event.get("description", "").lower():
            is_opportunity = True
            opportunity_type = "leverage_information"
        
        # Trust opportunities
        elif trigger_type == "trust" or "trusted" in trigger_event.get("description", "").lower():
            is_opportunity = True
            opportunity_type = "betray_trust"
        
        # Emotional opportunities
        elif current_emotion in ["anger", "fear", "sadness"] and emotional_intensity > 0.6:
            is_opportunity = True
            opportunity_type = "exploit_emotion"
        
        # If this is an opportunity, generate details
        if is_opportunity:
            # Create a memory of spotting this opportunity
            memory_text = f"I noticed an opportunity to {opportunity_type.replace('_', ' ')} when {trigger_event.get('description', 'something happened')}."
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance="medium",
                tags=["scheming", "opportunity", opportunity_type]
            )
            
            # Return opportunity details
            return {
                "npc_id": npc_id,
                "type": opportunity_type,
                "trigger": trigger_event,
                "description": memory_text,
                "scheming_level": scheming_level
            }
        
        return None


class NPCBehavior:
    """Manages NPC behavior and decision-making."""
    def __init__(self, npc_id: int):
        self.npc_id = npc_id
        self.nyx_client = None
        self.memory_system = MemorySystem()
        self._user_model = None
    
    def get_nyx_client(self):
        """Lazy-load the Nyx client to avoid circular imports."""
        if self.nyx_client is None:
            from nyx.integrate import get_nyx_client
            self.nyx_client = get_nyx_client()
        return self.nyx_client
    
    async def get_user_model(self) -> Dict[str, Any]:
        """Get or create the Nyx user model for this NPC."""
        if self._user_model is None:
            try:
                # Get NPC data using asyncpg
                query = """
                    SELECT name, personality, traits, background
                    FROM NPCs
                    WHERE id = $1
                """
                async with get_db_connection_context() as conn:
                    row = await conn.fetchrow(query, self.npc_id)
                    
                    if not row:
                        raise ValueError(f"NPC {self.npc_id} not found")
                    
                    # Create user model from NPC data
                    self._user_model = {
                        'id': f"npc_{self.npc_id}",
                        'name': row['name'],
                        'personality': row['personality'],
                        'traits': row['traits'],
                        'background': row['background'],
                        'type': 'npc',
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Register with Nyx
                    nyx_client = self.get_nyx_client()
                    await nyx_client.register_user(self._user_model)
                    
                    # Initialize memory system
                    await self.memory_system.initialize(self._user_model['id'])
            except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
                logger.error(f"Database error creating user model for NPC {self.npc_id}: {db_err}")
                raise
            except Exception as e:
                logger.error(f"Error creating user model for NPC {self.npc_id}: {e}")
                raise
        
        return self._user_model
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on context and NPC's personality."""
        try:
            # Ensure user model exists
            user_model = await self.get_user_model()
            
            # Get relevant memories
            memories = await self.memory_system.get_relevant_memories(
                context,
                limit=5
            )
            
            # Prepare decision context
            decision_context = {
                'npc': user_model,
                'context': context,
                'memories': memories,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get decision from Nyx
            nyx_client = self.get_nyx_client()
            decision = await nyx_client.get_decision(decision_context)
            
            # Store decision in memory
            await self.memory_system.add_memory(
                memory_text=f"Made decision: {decision['action']}",
                memory_type="decision",
                importance="medium",
                metadata={
                    'action': decision['action'],
                    'reasoning': decision['reasoning'],
                    'context': context
                }
            )
            
            return decision
        except Exception as e:
            logger.error(f"Error making decision for NPC {self.npc_id}: {e}")
            return {
                'action': 'wait',
                'reasoning': 'Error in decision making process',
                'error': str(e)
            }
    
    async def process_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process an interaction with the NPC."""
        try:
            # Ensure user model exists
            user_model = await self.get_user_model()
            
            # Get relevant memories
            memories = await self.memory_system.get_relevant_memories(
                interaction,
                limit=5
            )
            
            # Prepare interaction context
            interaction_context = {
                'npc': user_model,
                'interaction': interaction,
                'memories': memories,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get response from Nyx
            nyx_client = self.get_nyx_client()
            response = await nyx_client.get_response(interaction_context)
            
            # Store interaction in memory
            await self.memory_system.add_memory(
                memory_text=f"Interaction: {interaction.get('type', 'unknown')}",
                memory_type="interaction",
                importance="medium",
                metadata={
                    'interaction': interaction,
                    'response': response,
                    'context': interaction_context
                }
            )
            
            return response
        except Exception as e:
            logger.error(f"Error processing interaction for NPC {self.npc_id}: {e}")
            return {
                'response': "I'm not sure how to respond to that.",
                'error': str(e)
            }
    
    async def update_state(self, new_state: Dict[str, Any]) -> bool:
        """Update the NPC's state."""
        try:
            # Ensure user model exists
            user_model = await self.get_user_model()
            
            # Update state in database using asyncpg
            query = """
                UPDATE NPCs
                SET state = $1, updated_at = NOW()
                WHERE id = $2
            """
            async with get_db_connection_context() as conn:
                await conn.execute(query, json.dumps(new_state), self.npc_id)
            
            # Store state change in memory
            await self.memory_system.add_memory(
                memory_text="State updated",
                memory_type="state_change",
                importance="low",
                metadata={
                    'old_state': user_model.get('state', {}),
                    'new_state': new_state
                }
            )
            
            # Update user model
            user_model['state'] = new_state
            return True
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error updating state for NPC {self.npc_id}: {db_err}")
            return False
        except Exception as e:
            logger.error(f"Error updating state for NPC {self.npc_id}: {e}")
            return False
