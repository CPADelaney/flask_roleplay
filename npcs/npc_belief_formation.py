# npcs/npc_belief_formation.py

import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple
import json

from memory.wrapper import MemorySystem
from npcs.npc_agent import NPCAgent
from data.npc_dal import NPCDataAccess
from lore.lore_system import LoreSystem
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from db.connection import get_db_connection_context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NPCBeliefFormation:
    """
    System for NPCs to form subjective beliefs based on their experiences,
    personality traits, and knowledge. This ensures NPCs have their own
    unique perspectives that may differ from objective reality.
    """
    
    def __init__(self, user_id: int, conversation_id: int, npc_id: int):
        """
        Initialize the belief formation system.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: ID of the NPC
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_id = npc_id
        self.lore_manager = LoreSystem(user_id, conversation_id)
        self.memory_system = None
        
    async def initialize(self):
        """Initialize the system."""
        self.memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="form_subjective_belief",
        action_description="NPC {npc_id} forming a subjective belief",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def form_subjective_belief_from_observation(
        self, 
        ctx,
        observation: str, 
        factuality: float = 1.0
    ) -> Dict[str, Any]:
        """
        Form a subjective belief based on an observation.
        
        Args:
            observation: The observed event or information
            factuality: How factual the belief should be (0.0-1.0)
                        Lower values create more biased/incorrect beliefs
                        
        Returns:
            Information about the formed belief
        """
        # Get NPC personality traits to influence belief formation
        npc_data = await NPCDataAccess.get_npc_details(self.npc_id, self.user_id, self.conversation_id)
        
        # Get key personality metrics
        dominance = npc_data.get('dominance', 50) / 100.0
        cruelty = npc_data.get('cruelty', 50) / 100.0
        intelligence = npc_data.get('intelligence', 50) / 100.0
        
        # Higher intelligence = more factual beliefs (unless intentionally biased)
        factuality = factuality * (0.5 + (intelligence * 0.5))
        
        # Initialize belief components
        belief_text = observation
        belief_biases = []
        confidence = 0.7  # Default confidence
        
        # Apply personality-based biases
        if dominance > 0.6 and random.random() < dominance:
            belief_biases.append("power_dynamics")
            if "conflict" in observation.lower() or "disagree" in observation.lower():
                belief_text = self._apply_dominance_bias(observation, dominance)
        
        if cruelty > 0.6 and random.random() < cruelty:
            belief_biases.append("negative_intent")
            if "intention" in observation.lower() or "motive" in observation.lower():
                belief_text = self._apply_cruelty_bias(observation, cruelty)
        
        # Adjust confidence based on personality
        confidence_modifier = 0.0
        if dominance > 0.7:
            confidence_modifier += 0.2  # More dominant NPCs are more confident
        if intelligence < 0.4 and random.random() < 0.7:
            confidence_modifier += 0.1  # Less intelligent NPCs can be overconfident
        
        # Apply factuality modifier - chance to misinterpret information
        if random.random() > factuality:
            belief_text = self._introduce_misinterpretation(observation)
            confidence_modifier -= 0.1  # Slightly less confident in misinterpreted info
        
        final_confidence = min(0.95, max(0.3, confidence + confidence_modifier))
        
        # Determine topic based on observation content
        topic = self._determine_topic(observation)
        
        # Create the belief
        if self.memory_system:
            belief_result = await self.memory_system.create_belief(
                entity_type="npc",
                entity_id=self.npc_id,
                belief_text=belief_text,
                confidence=final_confidence,
                topic=topic
            )
            
            return {
                "belief_id": belief_result.get("id"),
                "belief_text": belief_text,
                "confidence": final_confidence,
                "topic": topic,
                "biases_applied": belief_biases,
                "original_observation": observation
            }
        
        return {"error": "Memory system not initialized"}
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="form_narrative_belief",
        action_description="NPC {npc_id} forming a narrative about a series of events",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def form_narrative_from_memories(
        self, 
        ctx,
        related_memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Form a narrative belief by connecting multiple memories.
        This creates a subjective story/explanation that makes sense to the NPC.
        
        Args:
            related_memories: List of related memories to form a narrative from
                            
        Returns:
            Information about the formed narrative belief
        """
        if not related_memories or len(related_memories) < 2:
            return {"error": "Need at least two memories to form a narrative"}
        
        # Get NPC personality traits
        npc_data = await NPCDataAccess.get_npc_details(self.npc_id, self.user_id, self.conversation_id)
        
        # Get key personality metrics
        dominance = npc_data.get('dominance', 50) / 100.0
        cruelty = npc_data.get('cruelty', 50) / 100.0  
        intelligence = npc_data.get('intelligence', 50) / 100.0
        
        # Higher intelligence = more complex narratives
        complexity = 0.3 + (intelligence * 0.7)
        
        # Construct narrative based on memories
        memory_texts = [m.get("memory_text", "") for m in related_memories if "memory_text" in m]
        
        # Base narrative just connects the events
        base_narrative = "Based on what I've experienced, " + " Then ".join(memory_texts)
        
        # Apply personality-based interpretation
        if dominance > 0.6:
            # Dominant characters tend to see events in terms of power dynamics
            narrative = f"I believe these events show how {self._apply_dominance_narrative(memory_texts, dominance)}"
        elif cruelty > 0.6:
            # Cruel characters tend to assume negative intentions
            narrative = f"I've concluded that {self._apply_cruelty_narrative(memory_texts, cruelty)}"
        else:
            # Neutral interpretation
            narrative = f"I understand that {self._apply_neutral_narrative(memory_texts)}"
        
        # Determine confidence based on complexity and memory count
        confidence = min(0.9, 0.5 + (len(memory_texts) * 0.05) + (complexity * 0.2))
        
        # Create the narrative belief
        if self.memory_system:
            # Topic is based on the most common tags in the memories
            all_tags = []
            for memory in related_memories:
                all_tags.extend(memory.get("tags", []))
            
            topic = "narrative"
            if all_tags:
                # Find most common tag
                from collections import Counter
                most_common = Counter(all_tags).most_common(1)
                if most_common:
                    topic = most_common[0][0]
            
            belief_result = await self.memory_system.create_belief(
                entity_type="npc",
                entity_id=self.npc_id,
                belief_text=narrative,
                confidence=confidence,
                topic=topic
            )
            
            return {
                "belief_id": belief_result.get("id"),
                "narrative": narrative,
                "confidence": confidence,
                "topic": topic,
                "memories_used": len(memory_texts),
                "memory_ids": [m.get("id") for m in related_memories if "id" in m]
            }
        
        return {"error": "Memory system not initialized"}
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="form_cultural_belief",
        action_description="NPC {npc_id} forming a culturally-influenced belief",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def form_culturally_influenced_belief(
        self, 
        ctx,
        subject: str
    ) -> Dict[str, Any]:
        """
        Form a belief heavily influenced by the NPC's cultural background.
        This ensures NPCs have culturally distinctive worldviews.
        
        Args:
            subject: The subject to form a belief about
                            
        Returns:
            Information about the formed belief
        """
        # Get NPC's cultural attributes
        cultural_attributes = await NPCDataAccess.get_npc_cultural_attributes(
            self.npc_id, self.user_id, self.conversation_id
        )
        
        if not cultural_attributes:
            return {"error": "No cultural attributes found for NPC"}
        
        # Extract cultural elements
        background = cultural_attributes.get("cultural_background", "")
        nationality = cultural_attributes.get("nationality", {})
        faith = cultural_attributes.get("faith", {})
        
        # Base belief confidence on strength of cultural identity
        identity_strength = cultural_attributes.get("identity_strength", 5) / 10.0
        faith_strength = 0
        if faith and "strength" in faith:
            faith_strength = faith.get("strength", 0) / 10.0
        
        # Stronger cultural identity = stronger cultural beliefs
        belief_strength = max(0.3, min(0.9, identity_strength))
        
        # Generate belief based on cultural elements
        belief_text = ""
        belief_topic = "culture"
        
        # Faith-based belief
        if faith and subject.lower() in ["morality", "ethics", "behavior", "tradition", "values"]:
            belief_text = self._generate_faith_belief(subject, faith, faith_strength)
            belief_topic = "faith"
        
        # Nationality-based belief  
        elif nationality and subject.lower() in ["society", "governance", "community", "leadership", "loyalty"]:
            belief_text = self._generate_nationality_belief(subject, nationality)
            belief_topic = "nationality"
        
        # General cultural belief
        else:
            belief_text = self._generate_general_cultural_belief(subject, background)
        
        if not belief_text:
            return {"error": "Could not generate a cultural belief"}
            
        # Create the belief
        if self.memory_system:
            belief_result = await self.memory_system.create_belief(
                entity_type="npc",
                entity_id=self.npc_id,
                belief_text=belief_text,
                confidence=belief_strength,
                topic=belief_topic
            )
            
            return {
                "belief_id": belief_result.get("id"),
                "belief_text": belief_text,
                "confidence": belief_strength,
                "topic": belief_topic,
                "cultural_influence": "high",
                "subject": subject
            }
        
        return {"error": "Memory system not initialized"}
    
    async def reevaluate_beliefs_based_on_new_knowledge(
        self, 
        knowledge_type: str, 
        knowledge_id: int
    ) -> Dict[str, Any]:
        """
        Reevaluate existing beliefs based on newly acquired knowledge.
        This allows NPCs to change their beliefs as they learn new information.
        
        Args:
            knowledge_type: Type of knowledge (lore_type)
            knowledge_id: ID of new knowledge
                            
        Returns:
            Information about updated beliefs
        """
        if not self.memory_system:
            await self.initialize()
            
        # Get the new knowledge
        knowledge = await self.lore_manager.get_lore_by_id(knowledge_type, knowledge_id)
        if not knowledge:
            return {"error": "Knowledge not found"}
            
        # Get related beliefs that might need updating
        knowledge_name = knowledge.get("name", "")
        
        # First get all beliefs
        beliefs = await self.memory_system.get_beliefs(
            entity_type="npc",
            entity_id=self.npc_id
        )
        
        # Filter for potentially related beliefs
        related_beliefs = []
        for belief in beliefs:
            belief_text = belief.get("belief", "")
            if any(term in belief_text.lower() for term in knowledge_name.lower().split()):
                related_beliefs.append(belief)
                
        if not related_beliefs:
            return {"updated": 0, "message": "No related beliefs found"}
            
        # Get knowledge level to determine impact
        knowledge_level = 5  # Default
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                knowledge_row = await conn.fetchrow("""
                    SELECT knowledge_level FROM LoreKnowledge
                    WHERE entity_type = 'npc' AND entity_id = $1
                    AND lore_type = $2 AND lore_id = $3
                """, self.npc_id, knowledge_type, knowledge_id)
                
                if knowledge_row:
                    knowledge_level = knowledge_row["knowledge_level"]
        
        # Knowledge impact factor
        impact = knowledge_level / 10.0
        
        # Update beliefs
        updated_beliefs = 0
        for belief in related_beliefs:
            belief_id = belief.get("id")
            if not belief_id:
                continue
                
            original_confidence = belief.get("confidence", 0.5)
            
            # Determine if knowledge contradicts or supports belief
            contradicts = await self._check_knowledge_contradicts_belief(knowledge, belief.get("belief", ""))
            
            if contradicts:
                # Knowledge contradicts belief - reduce confidence
                new_confidence = max(0.1, original_confidence - (impact * 0.3))
                reason = f"Contradicted by new knowledge about {knowledge_name}"
            else:
                # Knowledge supports belief - increase confidence
                new_confidence = min(0.95, original_confidence + (impact * 0.2))
                reason = f"Supported by new knowledge about {knowledge_name}"
                
            # Only update if confidence changes significantly
            if abs(new_confidence - original_confidence) > 0.05:
                await self.memory_system.update_belief_confidence(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    belief_id=belief_id,
                    new_confidence=new_confidence,
                    reason=reason
                )
                updated_beliefs += 1
        
        return {
            "updated": updated_beliefs,
            "message": f"Updated {updated_beliefs} beliefs based on new knowledge",
            "knowledge_name": knowledge_name,
            "knowledge_level": knowledge_level
        }
    
    async def create_opposing_belief(self, belief_id: str) -> Dict[str, Any]:
        """
        Create a contrasting/opposing belief to an existing one.
        This adds nuance to the NPC's belief system. NPCs often have 
        contradictory beliefs, just like real people.
        
        Args:
            belief_id: ID of the belief to create an opposing view for
                            
        Returns:
            Information about the new opposing belief
        """
        if not self.memory_system:
            await self.initialize()
            
        # Get the original belief
        beliefs = await self.memory_system.get_beliefs(
            entity_type="npc",
            entity_id=self.npc_id
        )
        
        original_belief = None
        for belief in beliefs:
            if belief.get("id") == belief_id:
                original_belief = belief
                break
                
        if not original_belief:
            return {"error": "Original belief not found"}
            
        original_text = original_belief.get("belief", "")
        original_confidence = original_belief.get("confidence", 0.5)
        topic = original_belief.get("topic", "general")
        
        # Generate opposing viewpoint
        opposing_text = self._generate_opposing_viewpoint(original_text)
        
        # Opposing beliefs have lower confidence
        opposing_confidence = max(0.2, original_confidence - 0.3)
        
        # Create the opposing belief
        belief_result = await self.memory_system.create_belief(
            entity_type="npc",
            entity_id=self.npc_id,
            belief_text=opposing_text,
            confidence=opposing_confidence,
            topic=topic
        )
        
        return {
            "belief_id": belief_result.get("id"),
            "belief_text": opposing_text,
            "confidence": opposing_confidence,
            "topic": topic,
            "original_belief_id": belief_id,
            "original_belief": original_text
        }
    
    # -------------------------------------------------------------------------
    # Helper methods for belief formation
    # -------------------------------------------------------------------------
    
    def _apply_dominance_bias(self, observation: str, dominance_level: float) -> str:
        """Apply a dominant personality bias to an observation."""
        if "disagree" in observation.lower():
            return observation.replace(
                "disagree", "challenge my authority"
            )
        elif "suggestion" in observation.lower():
            return observation.replace(
                "suggestion", "attempt to influence my decision"
            )
        elif "help" in observation.lower() and dominance_level > 0.8:
            return observation.replace(
                "help", "gain favor with"
            )
        return observation
    
    def _apply_cruelty_bias(self, observation: str, cruelty_level: float) -> str:
        """Apply a cruel personality bias to an observation."""
        if "mistake" in observation.lower():
            return observation.replace(
                "mistake", "deliberate failure"
            )
        elif "accident" in observation.lower():
            return observation.replace(
                "accident", "carelessness"
            )
        elif "intent" in observation.lower() or "intention" in observation.lower():
            return observation.replace(
                "intent", "ulterior motive"
            ).replace("intention", "hidden agenda")
        return observation
    
    def _introduce_misinterpretation(self, observation: str) -> str:
        """Introduce a misinterpretation into an observation."""
        # Simple word replacements that change meaning
        replacements = [
            ("happy", "merely polite"),
            ("angry", "defensive"),
            ("help", "manipulate"),
            ("friend", "ally of convenience"),
            ("gift", "bribe"),
            ("suggestion", "demand"),
            ("request", "order"),
            ("thanks", "obligation"),
            ("coincidence", "calculated move")
        ]
        
        modified = observation
        for original, replacement in replacements:
            if original in modified.lower():
                modified = modified.replace(original, replacement)
                break
                
        return modified
    
    def _determine_topic(self, observation: str) -> str:
        """Determine the most likely topic for a belief."""
        topics = {
            "player": ["player", "Chase"],
            "social": ["community", "people", "society", "social"],
            "power": ["leadership", "authority", "control", "influence"],
            "ethics": ["right", "wrong", "moral", "ethics", "good", "evil"],
            "relationship": ["friend", "enemy", "ally", "relationship", "trust"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in observation.lower() for keyword in keywords):
                return topic
                
        return "general"
    
    def _apply_dominance_narrative(self, memory_texts: List[str], dominance_level: float) -> str:
        """Create a dominance-focused narrative from memories."""
        if dominance_level > 0.8:
            return f"these events clearly demonstrate a power struggle that I need to control. {' '.join(memory_texts)}"
        else:
            return f"these events reflect the natural hierarchy at work. {' '.join(memory_texts)}"
    
    def _apply_cruelty_narrative(self, memory_texts: List[str], cruelty_level: float) -> str:
        """Create a negative-focused narrative from memories."""
        if cruelty_level > 0.8:
            return f"people are acting out of selfish interests, as evidenced by {' and '.join(memory_texts)}"
        else:
            return f"there are concerning motives behind these events: {' and '.join(memory_texts)}"
    
    def _apply_neutral_narrative(self, memory_texts: List[str]) -> str:
        """Create a neutral narrative from memories."""
        return f"these events are connected in the following way: {' then '.join(memory_texts)}"
    
    def _generate_faith_belief(self, subject: str, faith: Dict[str, Any], faith_strength: float) -> str:
        """Generate a belief based on the NPC's faith."""
        faith_name = faith.get("name", "my beliefs")
        
        if subject.lower() == "morality" or subject.lower() == "ethics":
            if faith_strength > 0.7:
                return f"The teachings of {faith_name} are the true foundation of all morality and ethical behavior."
            else:
                return f"Many moral principles can be found in the teachings of {faith_name}, which guides my understanding of right and wrong."
        
        elif subject.lower() == "behavior":
            if faith_strength > 0.7:
                return f"One should always act in accordance with {faith_name}'s teachings, even when it's difficult."
            else:
                return f"I try to follow the principles of {faith_name} in how I behave, though I'm not always perfect."
        
        elif subject.lower() == "tradition":
            if faith_strength > 0.7:
                return f"Traditional practices from {faith_name} should be preserved exactly as they've been passed down."
            else:
                return f"The traditions of {faith_name} offer valuable guidance, though they can be adapted to modern circumstances."
        
        elif subject.lower() == "values":
            if faith_strength > 0.7:
                return f"The values taught by {faith_name} are absolute and should never be compromised."
            else:
                return f"I value many of the teachings from {faith_name}, which helps me make sense of the world."
        
        return f"{faith_name} offers important wisdom about {subject} that I try to apply in my life."
    
    def _generate_nationality_belief(self, subject: str, nationality: Dict[str, Any]) -> str:
        """Generate a belief based on the NPC's nationality."""
        nation_name = nationality.get("name", "my homeland")
        
        if subject.lower() == "society":
            return f"The society of {nation_name} represents the best balance of freedom and order, unlike other places."
        
        elif subject.lower() == "governance":
            return f"The way we govern in {nation_name} may not be perfect, but it's better than the alternatives I've seen."
        
        elif subject.lower() == "community":
            return f"In {nation_name}, we understand that the community must sometimes come before the individual."
        
        elif subject.lower() == "leadership":
            return f"Leaders from {nation_name} tend to be more honorable and effective than those from other regions."
        
        elif subject.lower() == "loyalty":
            return f"Loyalty to {nation_name} and its people is one of the highest virtues one can embody."
        
        return f"The ways of {nation_name} have shaped my views on {subject} in profound ways."
    
    def _generate_general_cultural_belief(self, subject: str, cultural_background: str) -> str:
        """Generate a general culturally-influenced belief."""
        if not cultural_background:
            cultural_background = "my upbringing and experiences"
            
        if subject.lower() in ["family", "kinship", "relations"]:
            return f"Having grown up in a {cultural_background}, I believe that family bonds are sacred and take precedence over most other relationships."
        
        elif subject.lower() in ["honor", "respect", "reputation"]:
            return f"In {cultural_background}, a person's honor and reputation are their most valuable possessions, worth protecting at all costs."
        
        elif subject.lower() in ["wealth", "prosperity", "success"]:
            return f"Coming from {cultural_background}, I view true wealth not just in material possessions but in the strength of one's community standing."
        
        elif subject.lower() in ["knowledge", "wisdom", "learning"]:
            return f"In {cultural_background}, we believe that knowledge is meant to be earned through experience, not just study."
        
        return f"My background in {cultural_background} has taught me to view {subject} differently than others might."
    
    def _generate_opposing_viewpoint(self, belief_text: str) -> str:
        """Generate an opposing viewpoint to an existing belief."""
        # Simple opposites for common belief components
        opposites = {
            "always": "sometimes not",
            "never": "occasionally",
            "must": "don't always have to",
            "should": "might not need to",
            "all": "not all",
            "none": "some",
            "everyone": "not everyone",
            "no one": "some people",
            "best": "not always ideal",
            "worst": "not always terrible",
            "only": "not the only",
            "absolute": "relative",
            "true": "not entirely accurate",
            "false": "partially true"
        }
        
        # Try to find and replace opposites
        modified = belief_text
        for term, opposite in opposites.items():
            if term in modified.lower():
                # Replace only whole words
                import re
                modified = re.sub(rf'\b{term}\b', opposite, modified, flags=re.IGNORECASE)
                
        # If no replacements were made, add a qualifying phrase
        if modified == belief_text:
            qualifying_phrases = [
                "While I generally believe that ",
                "Although I often think that ",
                "Despite my usual view that ",
                "Even though I typically feel that "
            ]
            
            contradicting_endings = [
                ", there are exceptions worth considering.",
                ", this isn't always the case.",
                ", sometimes the opposite proves true.",
                ", I recognize there are other valid perspectives."
            ]
            
            return random.choice(qualifying_phrases) + belief_text + random.choice(contradicting_endings)
            
        return modified
    
    async def _check_knowledge_contradicts_belief(self, knowledge: Dict[str, Any], belief_text: str) -> bool:
        """Check if new knowledge contradicts an existing belief."""
        # Simple keyword-based contradiction check
        knowledge_desc = knowledge.get("description", "")
        
        # Extract key statements from knowledge
        knowledge_statements = knowledge_desc.split(". ")
        
        # Look for direct contradictions using simple negation patterns
        contradiction_pairs = [
            ("always", "never"),
            ("all", "none"),
            ("must", "must not"),
            ("is", "is not"),
            ("can", "cannot"),
            ("will", "will not")
        ]
        
        for statement in knowledge_statements:
            statement_lower = statement.lower()
            for pos, neg in contradiction_pairs:
                if pos in belief_text.lower() and neg in statement_lower:
                    return True
                if neg in belief_text.lower() and pos in statement_lower:
                    return True
        
        # If no direct contradiction, assume it supports the belief
        return False
