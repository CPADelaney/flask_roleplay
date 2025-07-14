# npcs/npc_learning_adaptation.py

"""
NPC Learning and Adaptation System

This module provides functionality for NPCs to continuously learn and adapt based on
their experiences with the player. A key feature is the evolution of NPC intensity,
which represents how aggressively each NPC attempts to dominate the player.

REFACTORED: Now uses LoreSystem for all database updates.
"""

import logging
import asyncio
import random
from typing import Dict, List, Any, Optional
import json

# Update the import to use the new async connection context
from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem
from nyx.nyx_governance import AgentType, DirectiveType, NyxUnifiedGovernor
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from npcs.npc_relationship import NPCRelationshipManager
from lore.lore_system import LoreSystem
from lore.core import canon

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NPCLearningAdaptation:
    """
    System for NPCs to continuously learn and adapt based on their experiences.
    REFACTORED: Uses LoreSystem for all stat updates.
    
    Key features:
    1. Intensity Adaptation: NPCs adjust their domination intensity based on player responses
    2. Learning Triggers: NPCs learn from specific player actions and behaviors
    3. Memory-Based Evolution: NPCs evolve based on their accumulated memories
    4. Belief-Guided Adaptation: NPC belief system influences their learning
    5. Relationship-Driven Changes: NPC relationships affect learning patterns
    """
    
    def __init__(self, user_id: int, conversation_id: int, npc_id: int):
        """
        Initialize the learning and adaptation system.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID to track learning for
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_id = npc_id
        self.memory_system = None
        self.nyx_governance = None
        self.relationship_manager = None
        self._lore_system = None
        
    async def initialize(self):
        """Initialize the system and set up components."""
        self.memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        self.nyx_governance = NyxGovernanceManager(self.user_id, self.conversation_id)
        self.relationship_manager = NPCRelationshipManager(
            self.npc_id, self.user_id, self.conversation_id
        )
        self._lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
        
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="record_interaction_for_learning",
        action_description="NPC {npc_id} recording interaction for learning",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def record_player_interaction(
        self,
        ctx,
        interaction_type: str,
        interaction_details: Dict[str, Any],
        player_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record a player interaction to drive NPC learning and adaptation.
        REFACTORED: Uses LoreSystem for any stat updates.
        
        Args:
            interaction_type: Type of interaction (e.g., 'command', 'conversation', 'dominance_display')
            interaction_details: Details about the interaction
            player_response: Optional information about how the player responded
            
        Returns:
            Learning outcomes and adaptation results
        """
        # Get current NPC stats - READ ONLY
        npc_data = await NPCDataAccess.get_npc_details(self.npc_id, self.user_id, self.conversation_id)
        if not npc_data:
            return {"error": "NPC not found"}
        
        # Extract key NPC metrics
        current_intensity = npc_data.get("intensity", 50)
        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 50)
        intelligence = npc_data.get("intelligence", 50) if "intelligence" in npc_data else 50
        
        # Create learning record in memory (allowed)
        await self.memory_system.remember(
            entity_type="npc",
            entity_id=self.npc_id,
            memory_text=f"Player interaction: {interaction_type}. {interaction_details.get('summary', '')}",
            importance="medium",
            tags=["learning", interaction_type],
            emotional=True,
            metadata={
                "interaction_type": interaction_type,
                "details": interaction_details,
                "player_response": player_response
            }
        )
        
        # Calculate adaptation effects
        adaptation_results = await self._calculate_adaptations(
            interaction_type, 
            interaction_details, 
            player_response,
            current_intensity,
            dominance,
            cruelty,
            intelligence
        )
        
        # Apply adaptations if any
        stats_updated = False
        if adaptation_results.get("intensity_change", 0) != 0:
            await self._update_npc_intensity(
                current_intensity, 
                adaptation_results["intensity_change"],
                adaptation_results.get("reason", "")
            )
            stats_updated = True
            
        # Make other personality adaptations if appropriate
        if adaptation_results.get("other_adaptations"):
            for stat, change in adaptation_results["other_adaptations"].items():
                if change != 0:
                    await self._update_npc_stat(stat, npc_data.get(stat, 50), change)
                    stats_updated = True
        
        # Record this learning event
        if stats_updated:
            await self._record_learning_event(adaptation_results)
            
        return {
            "interaction_recorded": True,
            "adaptation_results": adaptation_results,
            "stats_updated": stats_updated
        }
        
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="process_memory_for_learning",
        action_description="NPC {npc_id} processing memories for learning",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def process_recent_memories_for_learning(self, ctx, days: int = 7) -> Dict[str, Any]:
        """
        Process recent memories to drive NPC learning, especially for intensity adaptation.
        REFACTORED: Uses LoreSystem for updates.
        
        Args:
            days: Number of days of memories to process
            
        Returns:
            Learning outcomes and adaptation results
        """
        # Get recent memories
        memories = await self.memory_system.recall(
            entity_type="npc",
            entity_id=self.npc_id,
            limit=50,  # Reasonable limit to prevent processing too many
            recency_days=days
        )
        
        if not memories or "memories" not in memories or not memories["memories"]:
            return {"processed": False, "reason": "No recent memories found"}
        
        # Get current NPC stats - READ ONLY
        npc_data = await NPCDataAccess.get_npc_details(self.npc_id, self.user_id, self.conversation_id)
        
        # Initialize counters for memory-based triggers
        submission_signals = 0
        defiance_signals = 0
        positive_reinforcement = 0
        negative_reinforcement = 0
        
        # Analyze memories for learning signals
        for memory in memories["memories"]:
            memory_text = memory.get("memory_text", "").lower()
            tags = memory.get("tags", [])
            
            # Look for submission signals
            if any(term in memory_text for term in ["obey", "submit", "comply", "follow", "yield"]):
                submission_signals += 1
                
            # Look for defiance signals  
            if any(term in memory_text for term in ["defy", "resist", "refuse", "disobey", "challenge"]):
                defiance_signals += 1
                
            # Check for positive reinforcement
            if any(term in memory_text for term in ["enjoyed", "pleased", "satisfied", "successful"]):
                positive_reinforcement += 1
                
            # Check for negative reinforcement
            if any(term in memory_text for term in ["frustrated", "annoyed", "failed", "unsuccessful"]):
                negative_reinforcement += 1
        
        # Calculate intensity adaptation based on memory analysis
        intensity_change = 0
        adaptation_reason = []
        
        # More submission leads to increased intensity
        if submission_signals > defiance_signals and submission_signals >= 3:
            intensity_change += min(5, submission_signals)
            adaptation_reason.append(f"Player showed submission {submission_signals} times")
            
        # More defiance can lead to increased or decreased intensity depending on personality
        if defiance_signals > submission_signals and defiance_signals >= 3:
            # High cruelty NPCs increase intensity when defied
            if npc_data.get("cruelty", 50) > 70:
                intensity_change += min(7, defiance_signals)
                adaptation_reason.append(f"Player showed defiance {defiance_signals} times, triggering stronger domination")
            # Lower cruelty NPCs may back off
            else:
                intensity_change -= min(3, defiance_signals)
                adaptation_reason.append(f"Player showed defiance {defiance_signals} times, causing recalibration")
        
        # Positive reinforcement encourages current strategy
        if positive_reinforcement > negative_reinforcement:
            current_intensity = npc_data.get("intensity", 50)
            # If already high intensity, increase further
            if current_intensity > 70:
                intensity_change += 2
                adaptation_reason.append("High intensity approach was successful")
            # If low intensity, maintain that approach
            elif current_intensity < 30:
                intensity_change -= 1
                adaptation_reason.append("Low intensity approach was successful")
        
        # Apply the intensity change if significant
        if abs(intensity_change) >= 2:
            await self._update_npc_intensity(
                npc_data.get("intensity", 50),
                intensity_change,
                ", ".join(adaptation_reason)
            )
            
            return {
                "processed": True,
                "memories_analyzed": len(memories["memories"]),
                "intensity_change": intensity_change,
                "reason": ", ".join(adaptation_reason),
                "submission_signals": submission_signals,
                "defiance_signals": defiance_signals
            }
        
        return {
            "processed": True,
            "memories_analyzed": len(memories["memories"]),
            "intensity_change": 0,
            "reason": "No significant changes needed based on memory analysis"
        }
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="adapt_to_relationship_changes",
        action_description="NPC {npc_id} adapting to relationship changes",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def adapt_to_relationship_changes(self, ctx) -> Dict[str, Any]:
        """
        Adapt NPC intensity based on relationship changes with the player.
        REFACTORED: Uses LoreSystem for updates.
        
        Returns:
            Adaptation results
        """
        # Get current relationship - READ ONLY
        relationship = await self.relationship_manager.get_relationship_with_player()
        if not relationship:
            return {"adapted": False, "reason": "No relationship data found"}
        
        # Get current NPC stats - READ ONLY
        npc_data = await NPCDataAccess.get_npc_details(self.npc_id, self.user_id, self.conversation_id)
        
        # Initialize adaptation variables
        intensity_change = 0
        adaptation_reason = []
        
        # Closeness affects intensity
        closeness = relationship.get("closeness", 0)
        # Higher closeness initially decreases intensity as NPC gets comfortable
        if closeness > 70 and npc_data.get("intensity", 50) > 60:
            intensity_change -= 3
            adaptation_reason.append("High closeness is creating comfortable relationship")
        # Very high closeness can eventually increase intensity as NPC feels secure
        elif closeness > 90 and npc_data.get("cruelty", 50) > 60:
            intensity_change += 5
            adaptation_reason.append("Extremely close relationship allows for more intense domination")
            
        # Trust affects intensity
        trust = relationship.get("trust", 0)
        # High trust with high dominance increases intensity
        if trust > 80 and npc_data.get("dominance", 50) > 70:
            intensity_change += 4
            adaptation_reason.append("High trust enables more intense control")
        # Low trust decreases intensity as NPC is more cautious
        elif trust < 30 and trust > 0:
            intensity_change -= 3
            adaptation_reason.append("Low trust requires more careful approach")
            
        # Apply the intensity change if significant
        if abs(intensity_change) >= 2:
            await self._update_npc_intensity(
                npc_data.get("intensity", 50),
                intensity_change,
                ", ".join(adaptation_reason)
            )
            
            return {
                "adapted": True,
                "intensity_change": intensity_change,
                "reason": ", ".join(adaptation_reason),
                "relationship_factors": {
                    "closeness": closeness,
                    "trust": trust
                }
            }
        
        return {
            "adapted": False,
            "reason": "Relationship doesn't warrant intensity changes",
            "relationship_factors": {
                "closeness": closeness,
                "trust": trust
            }
        }
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="respond_to_direct_trigger",
        action_description="NPC {npc_id} responding to learning trigger",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def respond_to_trigger(
        self, 
        ctx,
        trigger_type: str,
        trigger_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Respond to a specific learning trigger that should cause immediate adaptation.
        REFACTORED: Uses LoreSystem for updates.
        
        Args:
            trigger_type: Type of trigger (e.g., 'extreme_submission', 'direct_challenge')
            trigger_details: Details about the trigger
            
        Returns:
            Response and adaptation results
        """
        # Get current NPC stats - READ ONLY
        npc_data = await NPCDataAccess.get_npc_details(self.npc_id, self.user_id, self.conversation_id)
        
        # Initialize response variables
        intensity_change = 0
        other_stat_changes = {}
        response_type = "neutral"
        message = ""
        
        # Handle various trigger types
        if trigger_type == "extreme_submission":
            # Player showed extreme submission - increase intensity substantially
            intensity_change = random.randint(5, 10)
            message = "Your complete submission will only intensify my control over you."
            response_type = "intensify_dominance"
            
        elif trigger_type == "direct_challenge":
            # Player directly challenged NPC
            if npc_data.get("cruelty", 50) > 70:
                # Cruel NPCs intensify when challenged
                intensity_change = random.randint(7, 12)
                other_stat_changes["cruelty"] = 3
                message = "Your defiance only makes me more determined to break you completely."
                response_type = "punishing"
            else:
                # Less cruel NPCs might back off slightly but will remember
                intensity_change = random.randint(-3, -1)
                message = "I see you need a different approach. We'll try something else... for now."
                response_type = "tactical_retreat"
                
        elif trigger_type == "positive_feedback":
            # Player responded very positively to NPC's approach
            current_intensity = npc_data.get("intensity", 50)
            # Reinforce current approach
            if current_intensity > 70:
                intensity_change = random.randint(2, 5)
                message = "I see you respond well to my firm control. Good."
                response_type = "reinforcing"
            elif current_intensity < 40:
                intensity_change = random.randint(-3, -1)
                message = "Your response confirms my approach is right for you."
                response_type = "affirming"
                
        elif trigger_type == "safeword_use":
            # Player used safeword - immediate decrease in intensity
            intensity_change = random.randint(-15, -10)
            message = "I understand. We'll take a different approach."
            response_type = "respectful_retreat"
            
        # Apply any changes
        if intensity_change != 0:
            await self._update_npc_intensity(
                npc_data.get("intensity", 50),
                intensity_change,
                f"Responding to {trigger_type} trigger"
            )
            
        for stat, change in other_stat_changes.items():
            await self._update_npc_stat(stat, npc_data.get(stat, 50), change)
            
        # Create a memory of this significant event (allowed)
        await self.memory_system.remember(
            entity_type="npc",
            entity_id=self.npc_id,
            memory_text=f"Responded to {trigger_type}: {trigger_details.get('summary', '')}",
            importance="high",
            tags=["adaptation", trigger_type],
            emotional=True,
            metadata={
                "trigger_type": trigger_type,
                "response_type": response_type,
                "intensity_change": intensity_change
            }
        )
            
        return {
            "responded": True,
            "response_type": response_type,
            "message": message,
            "intensity_change": intensity_change,
            "other_changes": other_stat_changes
        }
    
    # ----- Internal helper methods -----
    
    async def _calculate_adaptations(
        self,
        interaction_type: str,
        interaction_details: Dict[str, Any],
        player_response: Optional[Dict[str, Any]],
        current_intensity: int,
        dominance: int,
        cruelty: int,
        intelligence: int
    ) -> Dict[str, Any]:
        """Calculate how the NPC should adapt based on an interaction."""
        adaptation_results = {
            "intensity_change": 0,
            "other_adaptations": {},
            "reason": []
        }
        
        # Extract player's reaction if available
        player_compliance = 0  # -10 to 10 scale
        player_emotional_response = "neutral"
        if player_response:
            player_compliance = player_response.get("compliance_level", 0)
            player_emotional_response = player_response.get("emotional_response", "neutral")
        
        # Adaptation factors based on interaction type
        if interaction_type == "command":
            # Command interactions affect intensity based on compliance
            if player_compliance > 5:  # High compliance
                # Higher compliance leads to higher intensity for dominant NPCs
                if dominance > 60:
                    intensity_change = random.randint(1, 3)
                    adaptation_results["reason"].append("Player was highly compliant to commands")
                    adaptation_results["intensity_change"] += intensity_change
            elif player_compliance < -5:  # Resistance
                # Resistance can trigger different responses based on personality
                if cruelty > 70:  # Cruel NPCs increase intensity when resisted
                    intensity_change = random.randint(2, 4)
                    adaptation_results["reason"].append("Player resistance triggered escalation")
                    adaptation_results["intensity_change"] += intensity_change
                else:  # Less cruel NPCs may back off slightly
                    intensity_change = random.randint(-2, -1)
                    adaptation_results["reason"].append("Recalibrating approach due to resistance")
                    adaptation_results["intensity_change"] += intensity_change
                    
        elif interaction_type == "dominance_display":
            # Dominance displays affect intensity based on player response
            if player_emotional_response in ["fear", "submission", "awe"]:
                # Positive response to dominance increases intensity
                intensity_change = random.randint(2, 4)
                adaptation_results["reason"].append(f"Player showed {player_emotional_response} to dominance display")
                adaptation_results["intensity_change"] += intensity_change
            elif player_emotional_response in ["defiance", "anger", "mockery"]:
                # Negative response to dominance
                if cruelty > 60:
                    # Cruel NPCs double down
                    intensity_change = random.randint(3, 5)
                    adaptation_results["reason"].append("Intensifying to counter defiance")
                    adaptation_results["intensity_change"] += intensity_change
                else:
                    # Less cruel NPCs reconsider approach
                    intensity_change = random.randint(-3, -1)
                    adaptation_results["reason"].append("Adjusting approach due to negative response")
                    adaptation_results["intensity_change"] += intensity_change
        
        elif interaction_type == "conversation":
            # Conversations have more subtle effects on intensity
            topic = interaction_details.get("topic", "general")
            
            if topic in ["submission", "power", "control"]:
                # Topics related to power dynamics can affect intensity
                if player_compliance > 3:
                    intensity_change = random.randint(1, 2)
                    adaptation_results["reason"].append(f"Player was receptive to {topic} discussion")
                    adaptation_results["intensity_change"] += intensity_change
            
            # Long, positive conversations can increase closeness
            if interaction_details.get("duration", 0) > 5 and player_emotional_response in ["happy", "interested"]:
                adaptation_results["other_adaptations"]["closeness"] = random.randint(1, 3)
                adaptation_results["reason"].append("Positive conversation increased closeness")
        
        # Intelligence affects learning rate
        learning_factor = 0.5 + (intelligence / 100.0)
        adaptation_results["intensity_change"] = int(adaptation_results["intensity_change"] * learning_factor)
        
        # Ensure the reason is a string
        adaptation_results["reason"] = ", ".join(adaptation_results["reason"])
        
        return adaptation_results
    
    async def _update_npc_intensity(self, current_intensity: int, change: int, reason: str) -> bool:
        """Update the NPC's intensity level using LoreSystem."""
        try:
            # Calculate new intensity, keeping within bounds
            new_intensity = max(10, min(100, current_intensity + change))
            
            # Don't bother updating if change is minimal
            if new_intensity == current_intensity:
                return False
            
            # Use LoreSystem to update the intensity value
            result = await self._lore_system.propose_and_enact_change(
                ctx=RunContextWrapper(context={
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                    'npc_id': self.npc_id
                }),
                entity_type="NPCStats",
                entity_identifier={"npc_id": self.npc_id},
                updates={"intensity": new_intensity},
                reason=reason
            )
            
            if result.get("status") == "committed":
                # Log the change
                logger.info(f"NPC {self.npc_id} intensity updated: {current_intensity} -> {new_intensity} ({reason})")
                
                # Create a memory of this change (allowed)
                await self.memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=f"I adjusted my dominance intensity from {current_intensity} to {new_intensity}. {reason}",
                    importance="medium",
                    tags=["adaptation", "intensity_change"],
                    emotional=True
                )
                
                return True
            else:
                logger.warning(f"Failed to update NPC intensity: {result}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating NPC intensity: {e}")
            return False
    
    async def _update_npc_stat(self, stat_name: str, current_value: int, change: int) -> bool:
        """Update any NPC stat using LoreSystem."""
        try:
            # Calculate new value, keeping within bounds
            new_value = max(0, min(100, current_value + change))
            
            # Don't bother updating if change is minimal
            if new_value == current_value:
                return False
            
            # Use LoreSystem to update the stat value
            result = await self._lore_system.propose_and_enact_change(
                ctx=RunContextWrapper(context={
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                    'npc_id': self.npc_id
                }),
                entity_type="NPCStats",
                entity_identifier={"npc_id": self.npc_id},
                updates={stat_name: new_value},
                reason=f"Learning adaptation: {stat_name} changed from {current_value} to {new_value}"
            )
            
            if result.get("status") == "committed":
                # Log the change
                logger.info(f"NPC {self.npc_id} {stat_name} updated: {current_value} -> {new_value}")
                return True
            else:
                logger.warning(f"Failed to update NPC stat {stat_name}: {result}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating NPC stat {stat_name}: {e}")
            return False
    
    async def _record_learning_event(self, adaptation_results: Dict[str, Any]) -> None:
        """Record a significant learning/adaptation event in memory (allowed)."""
        # Create a memory about the adaptation
        intensity_change = adaptation_results.get("intensity_change", 0)
        direction = "increased" if intensity_change > 0 else "decreased"
        
        if abs(intensity_change) > 0:
            memory_text = f"I have {direction} my dominance intensity because: {adaptation_results.get('reason', 'of recent interactions')}."
            
            await self.memory_system.remember(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=memory_text,
                importance="medium",
                tags=["learning", "adaptation"],
                emotional=True
            )


# Define the NPCDataAccess class with async DB methods - READ ONLY
class NPCDataAccess:
    """Data access methods for NPC data - READ ONLY operations"""
    
    @staticmethod
    async def get_npc_details(npc_id: int, user_id: int, conversation_id: int) -> Dict[str, Any]:
        """Get NPC details from the database - READ ONLY."""
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT npc_id, npc_name, dominance, cruelty, personality_traits,
                           intensity, scheming_level, betrayal_planning
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """,
                    npc_id, user_id, conversation_id
                )
                
                if row:
                    # Process personality traits
                    personality_traits = []
                    raw_traits = row['personality_traits']
                    if raw_traits:
                        try:
                            if isinstance(raw_traits, list):
                                personality_traits = raw_traits
                            elif isinstance(raw_traits, str):
                                personality_traits = json.loads(raw_traits)
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse personality_traits for NPC {npc_id}: {e}")
                    
                    return {
                        "npc_id": row["npc_id"],
                        "npc_name": row["npc_name"],
                        "dominance": row["dominance"],
                        "cruelty": row["cruelty"],
                        "intensity": row["intensity"],
                        "scheming_level": row.get("scheming_level", 0),
                        "betrayal_planning": row.get("betrayal_planning", False),
                        "personality_traits": personality_traits
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting NPC details: {e}")
            return None


class NPCLearningManager:
    """
    Manages the learning and adaptation for multiple NPCs in the game.
    
    This class provides methods to:
    1. Initialize learning systems for NPCs
    2. Process global events for multiple NPCs
    3. Schedule regular adaptation cycles
    4. Track learning progress across NPCs
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the learning manager.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.learning_systems = {}  # Map of NPC ID to learning systems
        
    async def initialize(self):
        """Initialize the manager."""
        pass
    
    def get_learning_system_for_npc(self, npc_id: int) -> NPCLearningAdaptation:
        """
        Get or create a learning system for an NPC.
        
        Args:
            npc_id: NPC ID
            
        Returns:
            NPCLearningAdaptation instance for this NPC
        """
        if npc_id not in self.learning_systems:
            self.learning_systems[npc_id] = NPCLearningAdaptation(
                self.user_id, self.conversation_id, npc_id
            )
        return self.learning_systems[npc_id]
    
    async def process_event_for_learning(
        self,
        event_text: str,
        event_type: str,
        npc_ids: List[int],
        player_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a game event for learning by multiple NPCs.
        
        Args:
            event_text: Description of the event
            event_type: Type of event
            npc_ids: List of NPC IDs who witnessed the event
            player_response: Optional information about player's response
            
        Returns:
            Learning results per NPC
        """
        results = {
            "event_processed": True,
            "event_text": event_text,
            "event_type": event_type,
            "npc_learning": {}
        }
        
        # Create context for governance
        ctx = type('obj', (object,), {
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        
        # Process the event for each NPC
        for npc_id in npc_ids:
            learning_system = self.get_learning_system_for_npc(npc_id)
            await learning_system.initialize()
            
            # Update context for this NPC
            ctx.npc_id = npc_id
            
            try:
                interaction_details = {
                    "summary": event_text,
                    "event_type": event_type
                }
                
                learning_result = await learning_system.record_player_interaction(
                    ctx=ctx,
                    interaction_type=event_type,
                    interaction_details=interaction_details,
                    player_response=player_response
                )
                
                results["npc_learning"][npc_id] = learning_result
            except Exception as e:
                logger.error(f"Error processing learning for NPC {npc_id}: {e}")
                results["npc_learning"][npc_id] = {"error": str(e)}
        
        return results
    
    async def run_regular_adaptation_cycle(self, npc_ids: List[int]) -> Dict[str, Any]:
        """
        Run a regular adaptation cycle for multiple NPCs.
        This should be called periodically (e.g., daily in game time).
        
        Args:
            npc_ids: List of NPC IDs to process
            
        Returns:
            Adaptation results per NPC
        """
        results = {
            "cycle_completed": True,
            "npc_adaptations": {}
        }
        
        # Create context for governance
        ctx = type('obj', (object,), {
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        
        for npc_id in npc_ids:
            learning_system = self.get_learning_system_for_npc(npc_id)
            await learning_system.initialize()
            
            # Update context for this NPC
            ctx.npc_id = npc_id
            
            try:
                # Process recent memories for learning
                memory_learning = await learning_system.process_recent_memories_for_learning(ctx, days=7)
                
                # Adapt to relationship changes
                relationship_adaptation = await learning_system.adapt_to_relationship_changes(ctx)
                
                results["npc_adaptations"][npc_id] = {
                    "memory_learning": memory_learning,
                    "relationship_adaptation": relationship_adaptation
                }
            except Exception as e:
                logger.error(f"Error in adaptation cycle for NPC {npc_id}: {e}")
                results["npc_adaptations"][npc_id] = {"error": str(e)}
        
        return results
