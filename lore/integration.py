# lore/integration.py

"""
Lore Integration Components

This module provides specialized components for integrating lore with other systems.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import random

# Import data access layer
from .data_access import (
    NPCDataAccess,
    LocationDataAccess,
    FactionDataAccess,
    LoreKnowledgeAccess
)

# Import Nyx governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting

logger = logging.getLogger(__name__)

class BaseIntegration:
    """Base class for all integration components."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """
        Initialize the base integration component.
        
        Args:
            user_id: Optional user ID for filtering
            conversation_id: Optional conversation ID for filtering
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None  # Will be set externally
        self.initialized = False
        
        # Initialize data access components
        self.npc_data = NPCDataAccess(user_id, conversation_id)
        self.location_data = LocationDataAccess(user_id, conversation_id)
        self.faction_data = FactionDataAccess(user_id, conversation_id)
        self.lore_knowledge = LoreKnowledgeAccess(user_id, conversation_id)
    
    def set_governor(self, governor):
        """Set the governor externally to avoid circular dependencies."""
        self.governor = governor
        
    async def initialize(self) -> bool:
        """
        Initialize the integration component.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.initialized:
            return True
            
        try:
            # Don't try to get governance here - it should be set externally
            # Remove this line:
            # from nyx.integrate import get_central_governance
            # self.governor = await get_central_governance(self.user_id, self.conversation_id)
            
            # Initialize data access components
            await self.npc_data.initialize()
            await self.location_data.initialize()
            await self.faction_data.initialize()
            await self.lore_knowledge.initialize()
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing {self.__class__.__name__}: {e}")
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Cleanup data access components
            await self.npc_data.cleanup()
            await self.location_data.cleanup()
            await self.faction_data.cleanup()
            await self.lore_knowledge.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class NPCLoreIntegration(BaseIntegration):
    """Integrates lore into NPC agent behavior and memory systems."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None, npc_id: Optional[int] = None):
        """
        Initialize the NPC lore integration.
        
        Args:
            user_id: Optional user ID for filtering
            conversation_id: Optional conversation ID for filtering
            npc_id: Optional NPC ID for specific NPC integration
        """
        super().__init__(user_id, conversation_id)
        self.npc_id = npc_id
        
        # Initialize belief systems
        from npcs.npc_belief_formation import NPCBeliefFormation
        from npcs.belief_system_integration import NPCBeliefSystemIntegration
        self.belief_integration = NPCBeliefSystemIntegration(user_id, conversation_id)
        self.belief_system = None
        if npc_id:
            self.belief_system = self.belief_integration.get_belief_system_for_npc(npc_id)
    
    async def initialize(self) -> bool:
        """Initialize the NPC lore integration."""
        if not await super().initialize():
            return False
            
        try:
            # Initialize belief system if needed
            from npcs.npc_belief_formation import NPCBeliefFormation
            from npcs.belief_system_integration import NPCBeliefSystemIntegration
            if self.npc_id and self.belief_system:
                await self.belief_system.initialize()
                
            await self.belief_integration.initialize()
            
            # Register with governance
            await self.register_with_nyx_governance()
            
            return True
        except Exception as e:
            logger.error(f"Error initializing NPCLoreIntegration: {e}")
            return False
    
    async def register_with_nyx_governance(self):
        """Register with Nyx governance system."""
        # Only register if governor is set
        if not self.governor:
            logger.warning(f"Cannot register NPCLoreIntegration - no governor set")
            return
            
        # Register this integration with governance
        await self.governor.register_agent(
            agent_type=AgentType.NPC,
            agent_id=self.npc_id or "npc_lore_integration",
            agent_instance=self
        )
        
        logger.info(f"NPCLoreIntegration registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="initialize_npc_lore_knowledge",
        action_description="Initializing lore knowledge for NPC {npc_id}",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def initialize_npc_lore_knowledge(self, ctx, npc_id: int, 
                                         cultural_background: str,
                                         faction_affiliations: List[str]) -> Dict[str, Any]:
        """
        Initialize an NPC's knowledge of lore based on their background.
        
        Args:
            ctx: Governance context
            npc_id: ID of the NPC
            cultural_background: Cultural background of the NPC
            faction_affiliations: List of faction names the NPC is affiliated with
            
        Returns:
            Dictionary of knowledge granted
        """
        # Track knowledge granted
        knowledge_granted = {
            "world_lore": [],
            "cultural_elements": [],
            "factions": [],
            "historical_events": []
        }
        
        # Get IDs of the factions the NPC is affiliated with
        faction_ids = []
        for faction_name in faction_affiliations:
            faction = await self.faction_data.get_faction_by_name(faction_name)
            if faction:
                faction_ids.append(faction["id"])
        
        # 1. Give knowledge of the NPC's own factions
        for faction_id in faction_ids:
            # Knowledge level 7-9 for own factions (high knowledge)
            knowledge_level = 7 + min(2, len(faction_affiliations))
            
            await self.lore_knowledge.add_lore_knowledge(
                "npc", npc_id,
                "Factions", faction_id,
                knowledge_level=knowledge_level
            )
            
            # Add to tracking
            faction_data = await self.lore_knowledge.get_lore_by_id("Factions", faction_id)
            if faction_data:
                knowledge_granted["factions"].append({
                    "id": faction_id,
                    "name": faction_data.get("name"),
                    "knowledge_level": knowledge_level
                })
        
        # 2. Give cultural knowledge based on background
        cultural_elements = await self.lore_knowledge.get_relevant_lore(
            cultural_background,
            lore_types=["CulturalElements"],
            min_relevance=0.6,
            limit=10
        )
        
        for element in cultural_elements:
            # Knowledge level 5-8 for cultural elements
            practiced_by = element.get("practiced_by", [])
            # Higher knowledge if it's practiced by their culture or factions
            knowledge_level = 5
            
            if any(affiliation in practiced_by for affiliation in faction_affiliations):
                knowledge_level += 2
            
            if cultural_background in practiced_by:
                knowledge_level += 1
            
            await self.lore_knowledge.add_lore_knowledge(
                "npc", npc_id,
                "CulturalElements", element["id"],
                knowledge_level=knowledge_level
            )
            
            # Add to tracking
            knowledge_granted["cultural_elements"].append({
                "id": element["id"],
                "name": element.get("name"),
                "knowledge_level": knowledge_level
            })
        
        # 3. Give knowledge of world lore relevant to their background
        background_query = f"{cultural_background} {' '.join(faction_affiliations)}"
        world_lore = await self.lore_knowledge.get_relevant_lore(
            background_query,
            lore_types=["WorldLore"],
            min_relevance=0.5,
            limit=5
        )
        
        for lore in world_lore:
            # Knowledge level 3-6 for general world lore
            # Base knowledge level depends on lore significance
            significance = lore.get("significance", 5)
            knowledge_level = min(3 + (significance // 3), 6)
            
            await self.lore_knowledge.add_lore_knowledge(
                "npc", npc_id,
                "WorldLore", lore["id"],
                knowledge_level=knowledge_level
            )
            
            # Add to tracking
            knowledge_granted["world_lore"].append({
                "id": lore["id"],
                "name": lore.get("name"),
                "category": lore.get("category"),
                "knowledge_level": knowledge_level
            })
        
        # 4. Give knowledge of historical events related to their factions
        for faction_id in faction_ids:
            # Get historical events involving this faction
            events = await self.lore_knowledge.get_relevant_lore(
                f"faction:{faction_id}",
                lore_types=["HistoricalEvents"],
                min_relevance=0.5,
                limit=5
            )
            
            for event in events:
                # Knowledge level based on significance
                significance = event.get("significance", 5)
                knowledge_level = min(4 + (significance // 2), 8)
                
                await self.lore_knowledge.add_lore_knowledge(
                    "npc", npc_id,
                    "HistoricalEvents", event["id"],
                    knowledge_level=knowledge_level
                )
                
                # Add to tracking
                knowledge_granted["historical_events"].append({
                    "id": event["id"],
                    "name": event["name"],
                    "knowledge_level": knowledge_level
                })
        
        # Create memories for the most significant knowledge
        await self._create_lore_memories(npc_id, knowledge_granted)
        
        return knowledge_granted
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="process_npc_lore_interaction",
        action_description="Processing lore interaction for NPC {npc_id}",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def process_npc_lore_interaction(self, ctx, npc_id: int, player_input: str) -> Dict[str, Any]:
        """
        Process a potential lore interaction between the player and an NPC.
        
        Args:
            ctx: Governance context
            npc_id: ID of the NPC
            player_input: The player's input text
            
        Returns:
            Lore response information if relevant
        """
        # Check if the player is asking about lore/knowledge
        lore_keywords = [
            "tell me about", "what do you know about", "what's the history of",
            "who are", "what are", "history", "tell me the story of", "legend",
            "myth", "how was", "founded", "created", "origin", "where did", "why do"
        ]
        
        is_lore_question = any(keyword in player_input.lower() for keyword in lore_keywords)
        
        if not is_lore_question:
            # Not a lore-focused interaction
            return {"is_lore_interaction": False}
        
        # Determine what the player is asking about
        relevant_lore = await self.lore_knowledge.get_relevant_lore(
            player_input,
            min_relevance=0.6,
            limit=3
        )
        
        if not relevant_lore:
            # No matching lore found
            return {
                "is_lore_interaction": True,
                "has_knowledge": False,
                "response_type": "no_knowledge",
                "message": "I don't know much about that, I'm afraid."
            }
        
        # Check if the NPC knows about this lore
        top_lore = relevant_lore[0]
        lore_type = top_lore["lore_type"]
        lore_id = top_lore["id"]
        
        # Get all knowledge for this NPC
        npc_knowledge = await self.lore_knowledge.get_entity_knowledge("npc", npc_id)
        
        # Find if NPC knows about this specific lore
        knowledge_item = next(
            (k for k in npc_knowledge if k["lore_type"] == lore_type and k["lore_id"] == lore_id),
            None
        )
        
        if not knowledge_item:
            # NPC doesn't know about this
            return {
                "is_lore_interaction": True,
                "has_knowledge": False,
                "response_type": "no_knowledge",
                "lore_type": lore_type,
                "lore_name": top_lore.get("name", "that topic"),
                "message": f"I don't know anything about {top_lore.get('name', 'that topic')}."
            }
        
        # NPC knows about this - formulate a response based on knowledge level
        knowledge_level = knowledge_item["knowledge_level"]
        is_secret = knowledge_item["is_secret"]
        
        # Get NPC's personality to influence response style
        npc_personality = await self.npc_data.get_npc_personality(npc_id)
        
        # Generate response based on knowledge level and personality
        response = await self._generate_lore_response(
            top_lore, knowledge_level, is_secret, npc_personality
        )
        
        # Check if sharing this lore should create a discovery for the player
        should_grant_player_knowledge = not is_secret and knowledge_level >= 3
        
        result = {
            "is_lore_interaction": True,
            "has_knowledge": True,
            "knowledge_level": knowledge_level,
            "is_secret": is_secret,
            "lore_type": lore_type,
            "lore_id": lore_id,
            "lore_name": top_lore.get("name", "that topic"),
            "response": response,
            "should_grant_player_knowledge": should_grant_player_knowledge
        }
        
        if should_grant_player_knowledge:
            # Calculate player knowledge level (slightly less than NPC's)
            player_knowledge_level = max(2, knowledge_level - 2)
            
            result["player_knowledge_level"] = player_knowledge_level
        
        return result
    
    async def apply_dialect_to_text(self, text: str, dialect_id: int, 
                                   intensity: str = 'medium',
                                   npc_id: Optional[int] = None) -> str:
        """
        Apply dialect features to a text.
        
        Args:
            text: Original text
            dialect_id: ID of the dialect to apply
            intensity: Intensity of dialect application ('light', 'medium', 'strong')
            npc_id: Optional NPC ID for personalized dialect features
            
        Returns:
            Modified text with dialect features applied
        """
        if not text or not text.strip():
            return text
        
        try:
            import re
            import random
            
            # Get dialect details from language system
            from lore.core.lore_system import LoreSystem
            
            # Get dialect details
            dialect = await self._get_dialect_details(dialect_id)
            if not dialect:
                logger.warning(f"Dialect with ID {dialect_id} not found")
                return text
            
            # Determine modification probability based on intensity
            probabilities = {
                'light': 0.3,
                'medium': 0.6,
                'strong': 0.9
            }
            probability = probabilities.get(intensity.lower(), 0.5)
            
            # Get NPC-specific speech patterns if an NPC ID is provided
            npc_speech_patterns = None
            if npc_id:
                npc_speech_patterns = await self._get_npc_speech_patterns(npc_id)
            
            # Extract dialect features
            accent_features = dialect.get('accent', {})
            vocabulary = dialect.get('vocabulary', {})
            grammar_rules = dialect.get('grammar', {})
            
            # Precompile regex patterns for efficiency
            accent_patterns = {}
            for original, replacement in accent_features.items():
                if original and replacement:  # Skip empty entries
                    try:
                        accent_patterns[original] = (
                            re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE),
                            replacement
                        )
                    except re.error:
                        logger.warning(f"Invalid regex pattern for accent feature: {original}")
            
            vocab_patterns = {}
            for standard_word, dialect_word in vocabulary.items():
                if standard_word and dialect_word:  # Skip empty entries
                    try:
                        vocab_patterns[standard_word] = (
                            re.compile(r'\b' + re.escape(standard_word) + r'\b', re.IGNORECASE),
                            dialect_word
                        )
                    except re.error:
                        logger.warning(f"Invalid regex pattern for vocabulary: {standard_word}")
            
            grammar_patterns = {}
            for grammar_rule, replacement in grammar_rules.items():
                if grammar_rule and replacement and " " in grammar_rule:  # Skip empty entries and single words
                    try:
                        grammar_patterns[grammar_rule] = (
                            re.compile(r'\b' + re.escape(grammar_rule) + r'\b', re.IGNORECASE),
                            replacement
                        )
                    except re.error:
                        logger.warning(f"Invalid regex pattern for grammar rule: {grammar_rule}")
            
            # Better sentence boundary detection
            abbreviations = r'(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|etc|e\.g|i\.e)\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s'
            sentence_boundary = re.compile(abbreviations)
            
            # Split text into sentences
            sentences = sentence_boundary.split(text)
            if not sentences:
                return text  # Nothing to process
            
            # Process each sentence
            modified_sentences = []
            
            for sentence in sentences:
                modified_sentence = sentence
                
                # Apply accent modifications based on intensity/probability
                if intensity == 'strong':
                    # Apply more accent features for strong intensity
                    for _, (pattern, replacement) in accent_patterns.items():
                        modified_sentence = pattern.sub(replacement, modified_sentence)
                else:
                    # For medium/light intensity, apply fewer changes
                    for _, (pattern, replacement) in accent_patterns.items():
                        if random.random() < probability:
                            modified_sentence = pattern.sub(replacement, modified_sentence)
                
                # Apply vocabulary substitutions based on intensity/probability
                for _, (pattern, replacement) in vocab_patterns.items():
                    if random.random() < probability:
                        modified_sentence = pattern.sub(replacement, modified_sentence)
                
                # Apply grammar modifications for medium/strong intensity only
                if intensity != 'light':
                    for _, (pattern, replacement) in grammar_patterns.items():
                        if random.random() < probability:
                            modified_sentence = pattern.sub(replacement, modified_sentence)
                
                # Add to modified sentences
                modified_sentences.append(modified_sentence)
            
            # Join sentences back together
            modified_text = ' '.join(modified_sentences)
            
            # Add cultural expressions if available (for strong intensity only)
            if intensity == 'strong' and npc_speech_patterns and npc_speech_patterns.get('cultural_expressions'):
                # Get a random cultural expression
                if npc_speech_patterns['cultural_expressions']:
                    expression = random.choice(npc_speech_patterns['cultural_expressions'])
                    
                    # Add the expression at the beginning or end of the text
                    if random.random() > 0.5:
                        modified_text = f"{expression['phrase']} {modified_text}"
                    else:
                        modified_text = f"{modified_text} {expression['phrase']}"
            
            return modified_text
            
        except Exception as e:
            logger.error(f"Error applying dialect to text: {e}", exc_info=True)
            return text  # Return original text on error
    
    async def _create_lore_memories(self, npc_id: int, knowledge_granted: Dict[str, List[Dict[str, Any]]]):
        """
        Create memories for the most significant lore knowledge.
        
        Args:
            npc_id: ID of the NPC
            knowledge_granted: Dictionary of granted knowledge
        """
        # Import memory system
        from memory.wrapper import MemorySystem
        # Get memory system for this NPC
        memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        
        # Get belief system for this NPC
        from npcs.npc_belief_formation import NPCBeliefFormation
        from npcs.belief_system_integration import NPCBeliefSystemIntegration
        belief_system = self.belief_integration.get_belief_system_for_npc(npc_id)
        await belief_system.initialize()
        
        # Create memories for faction knowledge
        for faction in knowledge_granted["factions"]:
            if faction["knowledge_level"] >= 7:
                # Fetch complete faction data
                faction_data = await self.lore_knowledge.get_lore_by_id("Factions", faction["id"])
                
                if faction_data:
                    # Create memory for faction affiliation
                    memory_text = f"I am affiliated with {faction_data['name']}, a {faction_data['type']} faction. "
                    
                    # Add factional values and goals if knowledge is high
                    if faction["knowledge_level"] >= 8:
                        values = faction_data.get("values", [])
                        goals = faction_data.get("goals", [])
                        
                        if values:
                            memory_text += f"We value {', '.join(values[:3])}. "
                        
                        if goals:
                            memory_text += f"Our goals include {', '.join(goals[:2])}."
                    
                    # Store the memory
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=memory_text,
                        importance="high",
                        tags=["lore", "faction", "identity"]
                    )
                    
                    # Form a belief about the faction (high factuality since it's their own faction)
                    from npcs.npc_belief_formation import NPCBeliefFormation
                    from npcs.belief_system_integration import NPCBeliefSystemIntegration
                    await belief_system.form_subjective_belief_from_observation(
                        observation=f"{faction_data['name']} is a faction that represents my interests and values.",
                        factuality=0.9
                    )
        
        # Create memories for cultural knowledge
        significant_culture = [c for c in knowledge_granted["cultural_elements"] if c["knowledge_level"] >= 6]
        if significant_culture:
            # Only create memories for the most significant cultural elements
            for culture in significant_culture[:3]:
                culture_data = await self.lore_knowledge.get_lore_by_id("CulturalElements", culture["id"])
                
                if culture_data:
                    from npcs.npc_belief_formation import NPCBeliefFormation
                    from npcs.belief_system_integration import NPCBeliefSystemIntegration
                    memory_text = f"I observe the {culture_data['name']}, which is a {culture_data['type']}. "
                    memory_text += culture_data.get("description", "")[:100]
                    
                    # Store the memory
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=memory_text,
                        importance="medium",
                        tags=["lore", "culture", "beliefs"]
                    )
                    
                    # Form a cultural belief (medium-high factuality)
                    if culture_data.get("type") in ["religion", "faith", "belief system"]:
                        await belief_system.form_culturally_influenced_belief(
                            subject="values"
                        )
                    else:
                        await belief_system.form_culturally_influenced_belief(
                            subject="society"
                        )
        
        # Create memories for significant historical events
        significant_events = [e for e in knowledge_granted["historical_events"] if e["knowledge_level"] >= 7]
        if significant_events:
            # Only create memories for the most significant events
            for event in significant_events[:2]:
                event_data = await self.lore_knowledge.get_lore_by_id("HistoricalEvents", event["id"])
                
                if event_data:
                    memory_text = f"I remember the {event_data['name']} which happened {event_data.get('date_description', 'in the past')}. "
                    memory_text += event_data.get("description", "")[:100]
                    
                    # Store the memory
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=memory_text,
                        importance="medium",
                        tags=["lore", "history", "event"]
                    )
                    
                    # Form a belief about the historical event (with medium factuality)
                    # Historical interpretations often have bias
                    from npcs.npc_belief_formation import NPCBeliefFormation
                    from npcs.belief_system_integration import NPCBeliefSystemIntegration
                    await belief_system.form_subjective_belief_from_observation(
                        observation=f"The {event_data['name']} had a significant impact on our history.",
                        factuality=0.7  # Allow for some personal interpretation
                    )
    
    async def _get_npc_speech_patterns(self, npc_id: int) -> Dict[str, Any]:
        """
        Get speech patterns for an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with speech pattern information
        """
        try:
            # Get NPC's cultural attributes
            cultural_attrs = await self.npc_data.get_npc_cultural_attributes(npc_id)
            
            if not cultural_attrs:
                return {
                    "primary_language": None,
                    "dialect": None,
                    "dialect_features": {},
                    "formality_level": "neutral",
                    "cultural_expressions": []
                }
            
            # Get NPC details for personality traits
            npc_personality = await self.npc_data.get_npc_personality(npc_id)
            
            # Initialize speech pattern data
            speech_patterns = {
                "primary_language": None,
                "dialect": None,
                "dialect_features": {},
                "formality_level": "neutral",  # Default formality
                "cultural_expressions": []
            }
            
            # Set primary language (if any languages are known)
            languages = cultural_attrs.get("languages", [])
            if languages:
                # Sort by fluency (highest first)
                languages.sort(key=lambda x: x.get("fluency", 0), reverse=True)
                primary_language = languages[0]
                
                speech_patterns["primary_language"] = {
                    "id": primary_language.get("id"),
                    "name": primary_language.get("name"),
                    "fluency": primary_language.get("fluency", 7),
                    "writing_system": primary_language.get("writing_system")
                }
                
                # Set dialect information if available
                if cultural_attrs.get("primary_dialect"):
                    speech_patterns["dialect"] = cultural_attrs["primary_dialect"]
                    speech_patterns["dialect_features"] = cultural_attrs.get("dialect_features", {})
                
                # Extract cultural expressions if available
                speech_patterns["cultural_expressions"] = cultural_attrs.get("cultural_expressions", [])
            
            # Determine formality level based on personality
            personality_traits = npc_personality.get("traits", {})
            
            # Default to neutral formality
            formality = "neutral"
            
            # Check personality traits for formality indicators
            formal_traits = ["proper", "pompous", "dignified", "aristocratic", "strict", "refined"]
            casual_traits = ["casual", "relaxed", "easygoing", "informal", "blunt", "direct"]
            
            # Check if personality traits indicate formality
            for trait, value in personality_traits.items():
                if any(formal in trait.lower() for formal in formal_traits) and value > 5:
                    formality = "formal"
                    break
                elif any(casual in trait.lower() for casual in casual_traits) and value > 5:
                    formality = "casual"
                    break
            
            # Cultural norms can override personality
            cultural_norms = cultural_attrs.get("cultural_norms_followed", [])
            for norm in cultural_norms:
                if norm.get("category", "").lower() in ["speech", "communication", "greetings"]:
                    if "formal" in norm.get("description", "").lower():
                        formality = "formal"
                        break
                    elif "casual" in norm.get("description", "").lower() or "informal" in norm.get("description", "").lower():
                        formality = "casual"
                        break
            
            speech_patterns["formality_level"] = formality
            
            return speech_patterns
            
        except Exception as e:
            logger.error(f"Error getting NPC speech patterns: {e}")
            return {
                "primary_language": None,
                "dialect": None,
                "dialect_features": {},
                "formality_level": "neutral",
                "cultural_expressions": []
            }
    
    async def _get_dialect_details(self, dialect_id: int) -> Dict[str, Any]:
        """
        Get dialect details.
        
        Args:
            dialect_id: ID of the dialect
            
        Returns:
            Dialect details
        """
        # This would typically call a language/dialect service
        # For now, return a simple mock
        return {
            "id": dialect_id,
            "name": "Sample Dialect",
            "accent": {
                "th": "d",
                "ing": "in'",
                "er": "ah"
            },
            "vocabulary": {
                "hello": "howdy",
                "goodbye": "see ya",
                "friend": "partner"
            },
            "grammar": {
                "I am": "I'm",
                "You are": "You're",
                "They are": "They're"
            }
        }
    
    async def _generate_lore_response(self, lore: Dict[str, Any], knowledge_level: int,
                                     is_secret: bool, personality: Dict[str, Any]) -> str:
        """
        Generate an appropriate lore response based on knowledge level and personality.
        
        Args:
            lore: Lore data
            knowledge_level: NPC's knowledge level (1-10)
            is_secret: Whether this is secret knowledge
            personality: NPC's personality data
            
        Returns:
            Formatted response text
        """
        # Base response components
        name = lore.get("name", "that")
        description = lore.get("description", "")
        lore_type = lore["lore_type"]
        
        # Truncate description based on knowledge level
        if knowledge_level <= 3:
            if len(description) > 100:
                description = description[:100] + "... that's about all I know."
        elif knowledge_level <= 6:
            if len(description) > 200:
                description = description[:200] + "... that's what I know about it."
        
        # Add personality styling
        traits = personality.get("traits", {})
        dominance = personality.get("dominance", 50)
        
        # Determine tone based on personality
        tone = "neutral"
        
        if any("arrogant" in trait or "boastful" in trait for trait in traits.keys()) or dominance > 70:
            tone = "confident"
        elif any("shy" in trait or "timid" in trait for trait in traits.keys()) or dominance < 30:
            tone = "hesitant"
        elif any("scholarly" in trait or "intellectual" in trait for trait in traits.keys()):
            tone = "scholarly"
        elif any("secretive" in trait for trait in traits.keys()) or is_secret:
            tone = "secretive"
        
        # Format intro based on tone and knowledge level
        intro = ""
        if tone == "confident":
            intro = f"Of course I know about {name}. "
            if knowledge_level > 7:
                intro = f"I know everything worth knowing about {name}. "
        elif tone == "hesitant":
            intro = f"I think I know a bit about {name}... "
            if knowledge_level < 5:
                intro = f"I'm not sure, but I believe {name} is... "
        elif tone == "scholarly":
            intro = f"According to what I've studied, {name} "
            if knowledge_level > 7:
                intro = f"Having extensively researched {name}, I can tell you that "
        elif tone == "secretive":
            if is_secret:
                intro = f"I shouldn't really talk about {name}, but... "
            else:
                intro = f"Not many know this about {name}, but "
        else:  # neutral
            intro = f"About {name}? "
            if knowledge_level > 6:
                intro = f"I know quite a bit about {name}. "
        
        # Add specific details based on lore type
        if lore_type == "Factions":
            faction_type = lore.get("type", "group")
            values = lore.get("values", [])
            goals = lore.get("goals", [])
            
            faction_details = f"They're a {faction_type}. "
            
            if values and knowledge_level >= 5:
                faction_details += f"They value {', '.join(values[:2])}. "
            
            if goals and knowledge_level >= 6:
                faction_details += f"Their goals include {', '.join(goals[:1])}. "
            
            return intro + faction_details + description
        
        elif lore_type == "CulturalElements":
            element_type = lore.get("type", "tradition")
            practiced_by = lore.get("practiced_by", [])
            
            cultural_details = f"It's a {element_type}. "
            
            if practiced_by and knowledge_level >= 4:
                if len(practiced_by) > 1:
                    cultural_details += f"It's practiced by {', '.join(practiced_by[:2])}. "
                else:
                    cultural_details += f"It's practiced by {practiced_by[0]}. "
            
            return intro + cultural_details + description
        
        elif lore_type == "HistoricalEvents":
            date = lore.get("date_description", "sometime in the past")
            consequences = lore.get("consequences", [])
            
            event_details = f"It happened {date}. "
            
            if consequences and knowledge_level >= 5:
                event_details += f"It led to {consequences[0]}. "
            
            return intro + event_details + description
        
        # Default for other lore types
        return intro + description


class ConflictIntegration(BaseIntegration):
    """Integrates lore into conflict system behavior."""
    
    async def get_conflict_lore(self, conflict_id: int) -> List[Dict[str, Any]]:
        """
        Get lore relevant to a specific conflict.
        
        Args:
            conflict_id: ID of the conflict
            
        Returns:
            List of relevant lore items
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get conflict details
            conflict = await self._get_conflict_details(conflict_id)
            if not conflict:
                return []
                
            # Build search query from conflict details
            faction_a_name = conflict.get("faction_a_name", "")
            faction_b_name = conflict.get("faction_b_name", "")
            description = conflict.get("description", "")
            
            search_query = f"{faction_a_name} {faction_b_name} {description}"
            
            # Get relevant lore
            relevant_lore = await self.lore_knowledge.get_relevant_lore(
                search_query,
                min_relevance=0.5,
                limit=10
            )
            
            return relevant_lore
                
        except Exception as e:
            logger.error(f"Error getting conflict lore: {e}")
            return []
    
    async def get_faction_conflicts(self, faction_id: int) -> List[Dict[str, Any]]:
        """
        Get conflicts involving a specific faction.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            List of conflicts
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get faction details
            faction = await self.faction_data.get_faction_details(faction_id)
            if not faction:
                return []
                
            # Get conflicts involving this faction
            conflicts = await self._get_faction_conflicts(faction_id)
            
            return conflicts
                
        except Exception as e:
            logger.error(f"Error getting faction conflicts: {e}")
            return []
    
    async def generate_faction_conflict(self, faction_a_id: int, faction_b_id: int) -> Dict[str, Any]:
        """
        Generate a conflict between two factions based on lore.
        
        Args:
            faction_a_id: ID of the first faction
            faction_b_id: ID of the second faction
            
        Returns:
            Generated conflict
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get faction details
            faction_a = await self.faction_data.get_faction_details(faction_a_id)
            faction_b = await self.faction_data.get_faction_details(faction_b_id)
            
            if not faction_a or not faction_b:
                return {}
                
            # Get relationship between factions
            relationships = await self.faction_data.get_faction_relationships(faction_a_id)
            
            relationship = next(
                (r for r in relationships if r["target_id"] == faction_b_id or r["source_id"] == faction_b_id),
                None
            )
            
            # Determine conflict intensity based on relationship
            conflict_strength = 5  # Default medium intensity
            
            if relationship:
                # Adjust based on relationship strength and type
                connection_type = relationship.get("connection_type", "")
                strength = relationship.get("strength", 5)
                
                if connection_type == "conflicts_with":
                    conflict_strength = strength
                elif connection_type == "allied_with":
                    conflict_strength = 10 - strength  # Invert for allies
                
            # Generate conflict data
            conflict_type = "standard"
            if conflict_strength >= 8:
                conflict_type = "major"
            elif conflict_strength >= 5:
                conflict_type = "minor"
            
            # Create conflict data
            conflict_data = {
                "conflict_type": conflict_type,
                "faction_a_name": faction_a["name"],
                "faction_b_name": faction_b["name"],
                "description": f"Conflict between {faction_a['name']} and {faction_b['name']}.",
                "brewing_description": f"Tensions are brewing between {faction_a['name']} and {faction_b['name']}.",
                "active_description": f"Open conflict has erupted between {faction_a['name']} and {faction_b['name']}.",
                "climax_description": f"The conflict between {faction_a['name']} and {faction_b['name']} has reached a critical point.",
                "resolution_description": f"The conflict between {faction_a['name']} and {faction_b['name']} has been resolved.",
                "estimated_duration": self._get_duration_for_conflict_type(conflict_type),
                "resources_required": self._get_resources_for_conflict_type(conflict_type),
                "lore_based": True
            }
            
            # Add specific reasons for conflict based on faction values and goals
            faction_a_values = faction_a.get("values", [])
            faction_b_values = faction_b.get("values", [])
            
            faction_a_goals = faction_a.get("goals", [])
            faction_b_goals = faction_b.get("goals", [])
            
            # Find conflicting values/goals
            conflicting_values = []
            for value_a in faction_a_values:
                for value_b in faction_b_values:
                    if self._are_conflicting_values(value_a, value_b):
                        conflicting_values.append((value_a, value_b))
            
            conflicting_goals = []
            for goal_a in faction_a_goals:
                for goal_b in faction_b_goals:
                    if self._are_conflicting_goals(goal_a, goal_b):
                        conflicting_goals.append((goal_a, goal_b))
            
            # Add conflict reasons based on values and goals
            if conflicting_values:
                value_a, value_b = conflicting_values[0]
                conflict_data["description"] += f" The {faction_a['name']} value {value_a}, while the {faction_b['name']} prioritize {value_b}."
            
            if conflicting_goals:
                goal_a, goal_b = conflicting_goals[0]
                conflict_data["description"] += f" The {faction_a['name']} seek to {goal_a}, which directly opposes the {faction_b['name']}'s goal to {goal_b}."
            
            return conflict_data
                
        except Exception as e:
            logger.error(f"Error generating faction conflict: {e}")
            return {}
    
    def _are_conflicting_values(self, value_a: str, value_b: str) -> bool:
        """
        Determine if two values are conflicting.
        
        Args:
            value_a: First value
            value_b: Second value
            
        Returns:
            True if values conflict, False otherwise
        """
        # Define pairs of opposing values
        opposing_values = [
            ("freedom", "control"),
            ("tradition", "progress"),
            ("individualism", "collectivism"),
            ("order", "chaos"),
            ("strength", "compassion"),
            ("honor", "pragmatism"),
            ("spirituality", "materialism"),
            ("loyalty", "independence")
        ]
        
        # Check if the values are in any opposing pair
        for v1, v2 in opposing_values:
            if (v1 in value_a.lower() and v2 in value_b.lower()) or (v2 in value_a.lower() and v1 in value_b.lower()):
                return True
        
        return False
    
    def _are_conflicting_goals(self, goal_a: str, goal_b: str) -> bool:
        """
        Determine if two goals are conflicting.
        
        Args:
            goal_a: First goal
            goal_b: Second goal
            
        Returns:
            True if goals conflict, False otherwise
        """
        # Define conflicting goal keywords
        conflicting_keywords = [
            ("control", "liberate"),
            ("expand", "defend"),
            ("conserve", "exploit"),
            ("destroy", "protect"),
            ("conquer", "resist"),
            ("overthrow", "maintain")
        ]
        
        # Check if the goals contain conflicting keywords
        for k1, k2 in conflicting_keywords:
            if (k1 in goal_a.lower() and k2 in goal_b.lower()) or (k2 in goal_a.lower() and k1 in goal_b.lower()):
                return True
        
        return False
    
    async def _get_conflict_details(self, conflict_id: int) -> Dict[str, Any]:
        try:
            query = "SELECT * FROM Conflicts WHERE id = $1"
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(query, conflict_id)
                return dict(row) if row else {}
        except Exception as e:
            logger.error(f"Error getting conflict details: {e}")
            return {}

    
    async def _get_faction_conflicts(self, faction_id: int) -> List[Dict[str, Any]]:
        try:
            query = "SELECT * FROM Conflicts WHERE faction_a_id = $1 OR faction_b_id = $1"
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(query, faction_id)
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error getting faction conflicts: {e}")
            return []
    
    def _get_duration_for_conflict_type(self, conflict_type: str) -> int:
        """Get estimated duration based on conflict type."""
        import random
        
        if conflict_type == "major":
            return random.randint(14, 21)  # 2-3 weeks
        elif conflict_type == "minor":
            return random.randint(4, 10)   # 4-10 days
        else:  # standard
            return random.randint(2, 5)    # 2-5 days
    
    def _get_resources_for_conflict_type(self, conflict_type: str) -> Dict[str, int]:
        """Get required resources based on conflict type."""
        if conflict_type == "major":
            return {"money": 500, "supplies": 15, "influence": 50}
        elif conflict_type == "minor":
            return {"money": 200, "supplies": 8, "influence": 25}
        else:  # standard
            return {"money": 100, "supplies": 5, "influence": 10}


class ContextEnhancer(BaseIntegration):
    """Enhances contexts with relevant lore."""
    
    async def enhance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a context with relevant lore.
        
        Args:
            context: Original context
            
        Returns:
            Enhanced context
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Create a copy of the context
            enhanced_context = context.copy()
            
            # Extract relevant information from context
            location = context.get("location", "")
            player_input = context.get("player_input", "")
            current_npcs = context.get("current_npcs", [])
            
            # Build search query
            npc_names = [npc.get("npc_name", "") for npc in current_npcs]
            search_query = f"{location} {player_input} {' '.join(npc_names)}"
            
            # Get relevant lore
            relevant_lore = await self.lore_knowledge.get_relevant_lore(
                search_query,
                min_relevance=0.7,
                limit=5
            )
            
            # Add lore to context
            enhanced_context["relevant_lore"] = relevant_lore
            
            # Get location lore if in a known location
            if location:
                location_data = await self.location_data.get_location_by_name(location)
                
                if location_data:
                    location_id = location_data.get("id")
                    
                    # Get full location context
                    location_context = await self.location_data.get_comprehensive_location_context(location_id)
                    
                    enhanced_context["location_context"] = location_context
            
            # Get active conflicts for additional context
            from logic.conflict_system.conflict_integration import ConflictSystemIntegration
            conflict_integration = ConflictSystemIntegration(self.user_id, self.conversation_id)
            active_conflicts = await conflict_integration.get_active_conflicts()
            
            # Add conflict data to context
            enhanced_context["active_conflicts"] = active_conflicts
            
            return enhanced_context
                
        except Exception as e:
            logger.error(f"Error enhancing context: {e}")
            return context  # Return original context on error
    
    async def generate_scene_description(self, location: str) -> Dict[str, Any]:
        """
        Generate a scene description enhanced with lore.
        
        Args:
            location: Location name
            
        Returns:
            Scene description
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get location details
            location_data = await self.location_data.get_location_by_name(location)
            
            if not location_data:
                return {
                    "base_description": f"You are at {location}",
                    "enhanced_description": f"You are at {location}."
                }
                
            location_id = location_data.get("id")
            base_description = location_data.get("description", f"You are at {location}")
            
            # Get comprehensive location context
            location_context = await self.location_data.get_comprehensive_location_context(location_id)
            
            # Extract elements for description
            controlling_factions = location_context.get("political_context", {}).get("ruling_factions", [])
            cultural_elements = location_context.get("cultural_context", {}).get("elements", [])
            location_lore = location_context.get("lore", {})
            
            # Compile enhanced description
            result = {
                "base_description": base_description,
                "lore_elements": {
                    "hidden_secrets": location_lore.get("hidden_secrets", []),
                    "local_legends": location_lore.get("local_legends", []),
                    "controlling_factions": controlling_factions,
                    "cultural_elements": cultural_elements
                }
            }
            
            # Generate a complete description
            from logic.chatgpt_integration import get_chatgpt_response
            
            prompt = f"""
            Generate an atmospheric scene description for this location that subtly incorporates lore:
            
            Location: {location}
            Base Description: {base_description}
            
            Controlling Factions: {", ".join([f["name"] for f in controlling_factions]) if controlling_factions else "None"}
            
            Cultural Elements: {", ".join([e["name"] for e in cultural_elements]) if cultural_elements else "None"}
            
            Local Legends: {", ".join(location_lore.get("local_legends", ["None"]))[:150]}
            
            Write a rich, sensory description (200-300 words) that:
            1. Establishes the physical space and atmosphere
            2. Subtly hints at faction influence and cultural elements
            3. Potentially alludes to hidden history or secrets
            4. Feels immersive and authentic to the setting
            
            The description should feel natural, not like an exposition dump.
            """
            
            response = await get_chatgpt_response(
                self.conversation_id,
                system_prompt="You are an AI that specializes in atmospheric scene descriptions that subtly incorporate worldbuilding.",
                user_prompt=prompt
            )
            
            if isinstance(response, dict) and "response" in response:
                result["enhanced_description"] = response["response"]
            else:
                result["enhanced_description"] = str(response)
            
            return result
                
        except Exception as e:
            logger.error(f"Error generating scene description: {e}")
            return {
                "base_description": f"You are at {location}",
                "enhanced_description": f"You are at {location}."
            }
