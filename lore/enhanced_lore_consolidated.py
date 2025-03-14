# lore/enhanced_lore.py

"""
Enhanced Lore System for Dynamic Femdom RPG

This module extends the existing DynamicLoreGenerator with comprehensive
systems for generating, evolving, and maintaining deep, consistent lore
in a dynamically generated femdom setting.
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from functools import wraps

# Agents SDK imports
from agents import Agent, ModelSettings, function_tool, Runner, trace
from agents.run_context import RunContextWrapper

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler

# Database and embedding functionality
from db.connection import get_db_connection
from embedding.vector_store import generate_embedding, vector_similarity
from utils.caching import LoreCache

# Import existing modules
from lore.lore_manager import LoreManager
from lore.lore_tools import (
    generate_foundation_lore,
    generate_factions,
    generate_cultural_elements,
    generate_historical_events,
    generate_locations,
    generate_quest_hooks
)

# Initialize caches
LORE_CACHE = LoreCache(max_size=1000, ttl=7200)  # 2 hour TTL
FAITH_CACHE = LoreCache(max_size=200, ttl=7200)  # 2 hour TTL

# -------------------------------------------------------------------------------
# Base Classes and Utilities
# -------------------------------------------------------------------------------

class BaseLoreSystem:
    """Base class with common functionality for all lore systems"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
    
    async def create_llm_agent(self, name: str, instructions: str, model: str = "o3-mini") -> Agent:
        """Create an LLM agent with standard configuration"""
        return Agent(
            name=name,
            instructions=instructions,
            model=model
        )
    
    async def run_llm_prompt(self, agent_name: str, agent_instructions: str, prompt: str) -> str:
        """Run a prompt through an LLM agent and get the response"""
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Create the agent
        agent = await self.create_llm_agent(agent_name, agent_instructions)
        
        # Run the prompt
        result = await Runner.run(agent, prompt, context=run_ctx.context)
        return result.final_output
    
    async def register_with_governance(self, agent_id: str, instruction: str):
        """Register with Nyx governance system."""
        await self.initialize_governance()
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id=agent_id,
            agent_instance=self
        )
        
        # Issue a directive for this system
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id=agent_id,
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": instruction,
                "scope": "world_building"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"{self.__class__.__name__} registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")

    async def parse_llm_json_response(self, response_text: str, default_value: Any = None) -> Any:
        """Parse JSON from LLM response with error handling"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            return default_value if default_value is not None else {"error": str(e)}

# -------------------------------------------------------------------------------
# Core Lore Systems
# -------------------------------------------------------------------------------

class MatriarchalPowerStructureFramework(BaseLoreSystem):
    """
    Defines core principles for power dynamics in femdom settings,
    ensuring consistency across all generated lore.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.core_principles = self._initialize_core_principles()
        
    def _initialize_core_principles(self) -> Dict[str, Any]:
        """Initialize core principles for matriarchal power structures"""
        return {
            "power_dynamics": {
                "dominant_gender": "female",
                "power_expression": ["political", "economic", "religious", "domestic", "sexual"],
                "hierarchy_types": ["matrilineal", "matrifocal", "matriarchal", "gynocentric"],
                "masculine_roles": ["service", "support", "nurture", "protection", "resources"],
                "counter_dynamics": ["resistance movements", "historical shifts", "regional variations"]
            },
            "societal_norms": {
                "female_leadership": {"political", "religious", "economic", "familial", "military"},
                "female_property_rights": {"land ownership", "business ownership", "inheritance"},
                "male_status_markers": {"service quality", "obedience", "beauty", "utility", "devotion"},
                "relationship_structures": {"polygyny", "polyandry", "monoandry", "collective households"},
                "enforcement_mechanisms": {"social pressure", "legal restrictions", "physical punishment", "economic sanctions"}
            },
            "symbolic_representations": {
                "feminine_symbols": ["chalice", "circle", "spiral", "dome", "moon"],
                "masculine_symbols": ["kneeling figures", "chains", "collars", "restraints"],
                "cultural_motifs": ["female nurture", "female authority", "male submission", "service ethics"]
            }
        }
    
    async def apply_power_lens(self, foundation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply matriarchal lens to generated foundation lore
        
        Args:
            foundation_data: Original foundation lore
            
        Returns:
            Foundation lore transformed through matriarchal lens
        """
        # Transform generic social structures into matriarchal ones
        if "social_structure" in foundation_data:
            foundation_data["social_structure"] = self._transform_to_matriarchal(
                foundation_data["social_structure"]
            )
        
        # Transform cosmology to reflect feminine primacy
        if "cosmology" in foundation_data:
            foundation_data["cosmology"] = self._feminize_cosmology(
                foundation_data["cosmology"]
            )
        
        # Transform magic system to reflect gendered power dynamics
        if "magic_system" in foundation_data:
            foundation_data["magic_system"] = self._gender_magic_system(
                foundation_data["magic_system"]
            )
        
        # Ensure world history reflects matriarchal development
        if "world_history" in foundation_data:
            foundation_data["world_history"] = self._matriarchalize_history(
                foundation_data["world_history"]
            )
            
        # Ensure calendar system reflects feminine significance
        if "calendar_system" in foundation_data:
            foundation_data["calendar_system"] = self._feminize_calendar(
                foundation_data["calendar_system"]
            )
            
        return foundation_data
    
    def _transform_to_matriarchal(self, social_structure: str) -> str:
        """Transform generic social structure description to a matriarchal one"""
        principles = self.core_principles["power_dynamics"]
        norms = self.core_principles["societal_norms"]
        
        # Extract key elements from original structure
        has_monarchy = "monarchy" in social_structure.lower()
        has_aristocracy = "aristocracy" in social_structure.lower() or "noble" in social_structure.lower()
        has_democracy = "democracy" in social_structure.lower() or "republic" in social_structure.lower()
        has_theocracy = "theocracy" in social_structure.lower() or "religious" in social_structure.lower()
        has_tribal = "tribal" in social_structure.lower() or "clan" in social_structure.lower()
        
        # Create matriarchal version
        transformed = "This society is fundamentally matriarchal. "
        
        if has_monarchy:
            transformed += "The supreme ruler is always a Queen or Empress, with succession passed through the maternal line. "
        if has_aristocracy:
            transformed += "Noble titles and land are held predominantly by women, with men serving as consorts or stewards. "
        if has_democracy:
            transformed += "While there is a democratic process, only women may vote or hold significant office. Men may serve in supportive administrative roles. "
        if has_theocracy:
            transformed += "Religious authority is held exclusively by women, with male clergy serving in subordinate positions. "
        if has_tribal:
            transformed += "Clan and tribal leadership passes through the maternal line, with Matriarchs holding ultimate authority. "
            
        # Add details about male status and position
        transformed += "Men are valued for their service to women and society, with status determined by their usefulness, obedience, and beauty. "
        
        # Add societal norms
        transformed += "Female property ownership is absolute, with inheritance flowing through the maternal line. "
        transformed += "Cultural practices, laws, and social norms all reinforce the natural authority of women over men. "
        
        # Merge with original content where appropriate
        if len(social_structure) > 200:  # Only if there's substantial original content
            transformed += "While maintaining these fundamental matriarchal principles, the society also incorporates elements of its historical development: " + social_structure
            
        return transformed
    
    def _feminize_cosmology(self, cosmology: str) -> str:
        """Create a feminized version of the cosmology"""
        if "goddess" not in cosmology.lower():
            cosmology = cosmology.replace("god", "goddess")
            cosmology = cosmology.replace("God", "Goddess")
            
        # Add feminine cosmic principle
        feminine_principle = "\n\nAt the heart of all creation is the Feminine Principle, the source of all life and power. "
        feminine_principle += "The cosmos itself is understood as fundamentally feminine in nature, with masculine elements serving and supporting the greater feminine whole."
        
        return cosmology + feminine_principle
    
    def _gender_magic_system(self, magic_system: str) -> str:
        """Apply gendered dynamics to the magic system"""
        gendered_magic = "\n\nThe flow and expression of magical energies reflect the natural order of feminine dominance. "
        gendered_magic += "Women typically possess greater innate magical potential and authority over the higher mysteries. "
        gendered_magic += "Male practitioners often specialize in supportive, protective, or enhancing magics that complement and serve the more powerful feminine magic traditions. "
        gendered_magic += "The most powerful spells and rituals often require a woman's touch to fully manifest."
        
        return magic_system + gendered_magic
    
    def _matriarchalize_history(self, history: str) -> str:
        """Ensure history reflects matriarchal development"""
        matriarchal_history = "\n\nThroughout recorded history, women have held the reins of power. "
        matriarchal_history += "Great Empresses, Matriarchs, and female leaders have shaped the course of civilization. "
        matriarchal_history += "While there have been periods of conflict and attempts to upset the natural order, "
        matriarchal_history += "the fundamental principle of feminine authority has remained the consistent foundation of society."
        
        return history + matriarchal_history
    
    def _feminize_calendar(self, calendar_system: str) -> str:
        """Make the calendar system reflect feminine significance"""
        feminine_calendar = "\n\nThe calendar marks significant events in feminine history, with important dates "
        feminine_calendar += "often corresponding to lunar cycles, feminine deities, or the reigns of great Matriarchs. "
        feminine_calendar += "The timing of festivals and holy days celebrates feminine power and the cycles of life that women control."
        
        return calendar_system + feminine_calendar

class LoreEvolutionSystem(BaseLoreSystem):
    """
    System responsible for evolving lore over time, both in response
    to specific events and through natural maturation processes.
    """
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_lore_with_event",
        action_description="Evolving lore with event",
        id_from_context=lambda ctx: "lore_generator"
    )
    async def evolve_lore_with_event(self, ctx, event_description: str) -> Dict[str, Any]:
        """Update world lore based on a significant narrative event"""
        # 1. Identify affected lore elements
        affected_elements = await self._identify_affected_lore(event_description)
        
        # 2. Generate modifications for each affected element
        updates = await self._generate_lore_updates(affected_elements, event_description)
        
        # 3. Apply updates to database
        await self._apply_lore_updates(updates)
        
        # 4. Generate potential new lore elements triggered by this event
        new_elements = await self._generate_consequential_lore(event_description, affected_elements)
        
        # 5. Report to governance system
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "evolve_lore_with_event",
                "description": f"Evolved lore with event: {event_description[:50]}"
            },
            result={
                "affected_elements": len(affected_elements),
                "new_elements_created": len(new_elements)
            }
        )
        
        return {
            "affected_elements": affected_elements,
            "updates": updates,
            "new_elements": new_elements
        }
    
    async def _identify_affected_lore(self, event_description: str) -> List[Dict[str, Any]]:
        """Identify lore elements that would be affected by the event"""
        # Generate embedding for the event
        event_embedding = await generate_embedding(event_description)
        
        # List of lore types to check
        lore_types = [
            "WorldLore", "Factions", "CulturalElements", "HistoricalEvents", 
            "GeographicRegions", "LocationLore", "UrbanMyths", "LocalHistories",
            "Landmarks", "NotableFigures"
        ]
        
        affected_elements = []
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for lore_type in lore_types:
                    try:
                        # Check if table exists and has embedding column
                        table_exists = await conn.fetchval(
                            f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{lore_type.lower()}');"
                        )
                        if not table_exists:
                            continue
                        
                        has_embedding = await conn.fetchval(
                            f"SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name = '{lore_type.lower()}' AND column_name = 'embedding');"
                        )
                        if not has_embedding:
                            continue
                        
                        # Determine ID field name
                        id_field = 'location_id' if lore_type == 'LocationLore' else 'id'
                        
                        # Perform similarity search
                        rows = await conn.fetch(
                            f"SELECT {id_field} as id, name, description, 1 - (embedding <=> $1) as relevance "
                            f"FROM {lore_type} WHERE 1 - (embedding <=> $1) > 0.6 "
                            f"ORDER BY relevance DESC LIMIT 5",
                            event_embedding
                        )
                        
                        for row in rows:
                            affected_elements.append({
                                'lore_type': lore_type,
                                'lore_id': row['id'],
                                'name': row['name'],
                                'description': row['description'],
                                'relevance': row['relevance']
                            })
                    except Exception as e:
                        logging.error(f"Error checking {lore_type} for affected elements: {e}")
        
        # Sort by relevance and limit to most relevant
        affected_elements.sort(key=lambda x: x['relevance'], reverse=True)
        if len(affected_elements) > 15:
            affected_elements = affected_elements[:15]
            
        return affected_elements
    
    async def _generate_lore_updates(
        self, 
        affected_elements: List[Dict[str, Any]], 
        event_description: str
    ) -> List[Dict[str, Any]]:
        """Generate specific updates for affected lore elements"""
        updates = []
        
        for element in affected_elements:
            # Create a prompt for the LLM
            prompt = f"""
            The following lore element needs to be updated based on a recent event:
            
            LORE ELEMENT:
            Type: {element['lore_type']}
            Name: {element['name']}
            Current Description: {element['description']}
            
            EVENT THAT OCCURRED:
            {event_description}
            
            Generate a new description for this lore element that incorporates the impact of this event.
            The update should be subtle and realistic, not completely rewriting history.
            
            Return your response as a JSON object with:
            1. "new_description": The updated description
            2. "update_reason": Brief explanation of why this update makes sense
            3. "impact_level": A number from 1-10 indicating how much this event impacts this lore element
            
            IMPORTANT: Maintain the matriarchal power dynamics in your update.
            """
            
            # Get updated description from LLM
            response_text = await self.run_llm_prompt(
                "LoreUpdateAgent",
                "You update existing lore elements based on narrative events while maintaining thematic consistency.",
                prompt
            )
            
            try:
                # Parse the JSON response
                update_data = await self.parse_llm_json_response(response_text)
                
                # Add the update
                updates.append({
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': update_data.get('new_description', ''),
                    'update_reason': update_data.get('update_reason', 'Event impact'),
                    'impact_level': update_data.get('impact_level', 5)
                })
            except Exception as e:
                logging.error(f"Failed to process LLM response: {e}")
                # Try to extract reasonable update even if processing failed
                if "new_description" in response_text and len(response_text) > 100:
                    # Very basic extraction
                    new_desc = response_text[response_text.find("new_description"):].split('\n')[0]
                    new_desc = new_desc.replace("new_description", "").replace(":", "").replace('"', '').strip()
                    
                    updates.append({
                        'lore_type': element['lore_type'],
                        'lore_id': element['lore_id'],
                        'name': element['name'],
                        'old_description': element['description'],
                        'new_description': new_desc,
                        'update_reason': "Event impact",
                        'impact_level': 5  # Default middle value
                    })
        
        return updates
    
    async def _apply_lore_updates(self, updates: List[Dict[str, Any]]) -> None:
        """Apply the generated updates to the database"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for update in updates:
                    lore_type = update['lore_type']
                    lore_id = update['lore_id']
                    new_description = update['new_description']
                    
                    # Generate new embedding
                    item_name = update.get('name', 'Unknown')
                    embedding_text = f"{item_name} {new_description}"
                    new_embedding = await generate_embedding(embedding_text)
                    
                    # Determine ID field name
                    id_field = 'location_id' if lore_type == 'LocationLore' else 'id'
                    
                    try:
                        # Update the database
                        await conn.execute(
                            f"UPDATE {lore_type} SET description = $1, embedding = $2 WHERE {id_field} = $3",
                            new_description, new_embedding, lore_id
                        )
                        
                        # Record the update in history if table exists
                        history_table_exists = await conn.fetchval(
                            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'lorechangehistory');"
                        )
                        
                        if history_table_exists:
                            await conn.execute(
                                "INSERT INTO LoreChangeHistory (lore_type, lore_id, previous_description, new_description, change_reason) "
                                "VALUES ($1, $2, $3, $4, $5)",
                                lore_type, lore_id, update['old_description'], new_description, update['update_reason']
                            )
                    except Exception as e:
                        logging.error(f"Error updating {lore_type} ID {lore_id}: {e}")
                        
                    # Clear relevant cache
                    LORE_CACHE.invalidate_pattern(f"{lore_type}_{lore_id}")
    
    async def _generate_consequential_lore(
        self, 
        event_description: str, 
        affected_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate new lore elements that might emerge as a consequence of the event"""
        # Extract names of affected elements for context
        affected_names = [f"{e['name']} ({e['lore_type']})" for e in affected_elements[:5]]
        affected_names_text = ", ".join(affected_names)
        
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following event and affected lore elements, generate 1-3 new lore elements that would emerge as consequences.
        
        EVENT:
        {event_description}
        
        AFFECTED ELEMENTS:
        {affected_names_text}
        
        Generate between 1 and 3 new lore elements of different types. Consider:
        - New urban myths that might emerge
        - Changes to local history or folklore
        - New traditions or cultural practices
        - New historical events that this triggers
        - New notable figures who rise to prominence
        - New organizations or factions that form
        
        For each new element, provide:
        1. "lore_type": The type of lore element (WorldLore, Factions, CulturalElements, UrbanMyths, etc.)
        2. "name": A name for the element
        3. "description": A detailed description
        4. "connection": How it relates to the event and other lore
        5. "significance": A number from 1-10 indicating importance
        
        Format your response as a JSON array containing these objects.
        IMPORTANT: Maintain the matriarchal/femdom power dynamics in all new lore.
        """
        
        # Get response from LLM
        response_text = await self.run_llm_prompt(
            "LoreCreationAgent",
            "You create new lore elements that emerge from significant events.",
            prompt
        )
        
        try:
            # Parse the JSON response
            new_elements = await self.parse_llm_json_response(response_text, [])
            
            # Ensure we got a list
            if not isinstance(new_elements, list):
                if isinstance(new_elements, dict):
                    new_elements = [new_elements]
                else:
                    new_elements = []
                    
            # Save the new lore elements to appropriate tables
            await self._save_new_lore_elements(new_elements, event_description)
                
            return new_elements
        except Exception as e:
            logging.error(f"Error generating consequential lore: {e}")
            return []
    
    async def _save_new_lore_elements(
        self, 
        new_elements: List[Dict[str, Any]], 
        event_description: str
    ) -> None:
        """Save newly generated lore elements to appropriate tables"""
        for element in new_elements:
            lore_type = element.get('lore_type')
            name = element.get('name')
            description = element.get('description')
            significance = element.get('significance', 5)
            
            if not all([lore_type, name, description]):
                continue
                
            # Generate embedding for the new content
            embedding_text = f"{name} {description}"
            embedding = await generate_embedding(embedding_text)
            
            # Save based on type
            try:
                if lore_type == "WorldLore":
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category=element.get('category', 'event_consequence'),
                        description=description,
                        significance=significance,
                        tags=element.get('tags', ['event_consequence', 'new_lore'])
                    )
                elif lore_type == "Factions":
                    await self.lore_manager.add_faction(
                        name=name,
                        faction_type=element.get('faction_type', 'event_consequence'),
                        description=description,
                        values=element.get('values', ['power', 'authority']),
                        goals=element.get('goals', ['stability', 'influence']),
                        headquarters=element.get('headquarters'),
                        founding_story=f"Founded in response to: {event_description}"
                    )
                elif lore_type == "CulturalElements":
                    await self.lore_manager.add_cultural_element(
                        name=name,
                        element_type=element.get('element_type', 'tradition'),
                        description=description,
                        practiced_by=element.get('practiced_by', ['society']),
                        significance=significance,
                        historical_origin=f"Emerged from: {event_description}"
                    )
                elif lore_type == "HistoricalEvents":
                    await self.lore_manager.add_historical_event(
                        name=name,
                        description=description,
                        date_description=element.get('date_description', 'Recently'),
                        significance=significance,
                        participating_factions=element.get('participating_factions', []),
                        consequences=element.get('consequences', ['Still unfolding'])
                    )
                else:
                    # Generic fallback for unknown types
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category=lore_type.lower(),
                        description=description,
                        significance=significance,
                        tags=[lore_type.lower(), 'event_consequence']
                    )
            except Exception as e:
                logging.error(f"Error saving new {lore_type} '{name}': {e}")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="mature_lore_over_time",
        action_description="Maturing lore over time",
        id_from_context=lambda ctx: "lore_generator"
    )
    async def mature_lore_over_time(self, ctx, days_passed: int = 7) -> Dict[str, Any]:
        """Natural evolution of lore over time, simulating how history and culture develop"""
        # Calculate probabilities based on days passed
        # More time = more changes
        base_probability = min(0.5, days_passed * 0.02)  # Cap at 50%
        
        maturation_summary = {
            "myth_evolution": [],
            "culture_development": [],
            "geopolitical_shifts": [],
            "reputation_changes": []
        }
        
        # 1. Urban Myths Evolution (highest chance of change)
        if random.random() < (base_probability * 1.5):
            myth_changes = await self._evolve_urban_myths()
            maturation_summary["myth_evolution"] = myth_changes
        
        # 2. Cultural Elements Development
        if random.random() < base_probability:
            culture_changes = await self._develop_cultural_elements()
            maturation_summary["culture_development"] = culture_changes
        
        # 3. Geopolitical Landscape Shifts (rarer)
        if random.random() < (base_probability * 0.7):
            geopolitical_changes = await self._shift_geopolitical_landscape()
            maturation_summary["geopolitical_shifts"] = geopolitical_changes
        
        # 4. Notable Figure Reputation Changes
        if random.random() < (base_probability * 1.2):
            reputation_changes = await self._evolve_notable_figures()
            maturation_summary["reputation_changes"] = reputation_changes
        
        # Report to governance system
        changes_count = sum(len(changes) for changes in maturation_summary.values())
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "mature_lore_over_time",
                "description": f"Matured lore over {days_passed} days"
            },
            result={
                "changes_applied": changes_count
            }
        )
        
        return {
            "days_passed": days_passed,
            "changes_applied": changes_count,
            "maturation_summary": maturation_summary
        }
    
    # Implementation of _evolve_urban_myths(), _develop_cultural_elements(),
    # _shift_geopolitical_landscape(), and _evolve_notable_figures() would go here
    # They've been omitted for brevity but would follow similar patterns to existing code

# -------------------------------------------------------------------------------
# Domain-Specific Lore Systems
# -------------------------------------------------------------------------------

class FaithSystem(BaseLoreSystem):
    """
    Comprehensive system for managing religions, faiths, and belief systems
    within the matriarchal society with full Nyx governance integration.
    """
    
    async def initialize_tables(self):
        """Ensure faith system tables exist"""
        # Creates multiple tables related to faith:
        # - Deities, Pantheons, ReligiousPractices, HolySites, 
        # - ReligiousTexts, ReligiousOrders, ReligiousConflicts
        # Implementation omitted for brevity
        pass
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_pantheon",
        action_description="Generating pantheon for the world",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_pantheon(self, ctx) -> Dict[str, Any]:
        """Generate a complete pantheon for the world"""
        # Implementation omitted for brevity
        pass
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_faith_system",
        action_description="Generating complete faith system for the world",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_complete_faith_system(self, ctx) -> Dict[str, Any]:
        """Generate a complete faith system for the world"""
        # Implementation omitted for brevity
        pass
    
    # Additional methods would be implemented here

class EmergentLoreSystem(BaseLoreSystem):
    """
    System for generating emergent lore, events, and developments
    to ensure the world evolves organically over time.
    """
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_emergent_event",
        action_description="Generating emergent world event",
        id_from_context=lambda ctx: "emergent_lore"
    )
    async def generate_emergent_event(self, ctx) -> Dict[str, Any]:
        """Generate a random emergent event in the world"""
        # Implementation omitted for brevity
        pass
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_cyclical_event",
        action_description="Generating cyclical seasonal event",
        id_from_context=lambda ctx: "emergent_lore"
    )
    async def generate_cyclical_event(self, ctx, season: str = None, is_solstice: bool = False, is_equinox: bool = False) -> Dict[str, Any]:
        """Generate a cyclical seasonal event"""
        # Implementation omitted for brevity
        pass
    
    # Additional methods would be implemented here

class LoreExpansionSystem(BaseLoreSystem):
    """
    System for expanding lore beyond what was initially generated,
    adding new elements as needed for world coherence and depth.
    """
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_faction",
        action_description="Generating additional faction for the world",
        id_from_context=lambda ctx: "lore_expansion"
    )
    async def generate_additional_faction(self, ctx, faction_type: str = None) -> Dict[str, Any]:
        """Generate an additional faction"""
        # Implementation omitted for brevity
        pass
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_locations",
        action_description="Generating additional locations for the world",
        id_from_context=lambda ctx: "lore_expansion"
    )
    async def generate_additional_locations(self, ctx, location_types: List[str] = None, count: int = 3) -> List[Dict[str, Any]]:
        """Generate additional locations"""
        # Implementation omitted for brevity
        pass
    
    # Additional methods would be implemented here

# -------------------------------------------------------------------------------
# Integration Functions
# -------------------------------------------------------------------------------

async def register_all_enhanced_lore_systems(user_id: int, conversation_id: int) -> Dict[str, bool]:
    """Register all enhanced lore systems with Nyx governance"""
    # Get the Nyx governance system
    governance = await get_central_governance(user_id, conversation_id)
    
    # Systems to register
    systems = {
        "faith_system": (FaithSystem, "Create and manage faith systems that emphasize feminine divine superiority."),
        "lore_evolution": (LoreEvolutionSystem, "Evolve lore over time to maintain a living, dynamic world."),
        "emergent_lore": (EmergentLoreSystem, "Generate emergent lore and events to ensure the world evolves organically."),
        "lore_expansion": (LoreExpansionSystem, "Expand lore with new elements as needed for world coherence and depth.")
    }
    
    # Register each system
    registration_results = {}
    for system_id, (system_class, instruction) in systems.items():
        try:
            system = system_class(user_id, conversation_id)
            await system.register_with_governance(system_id, instruction)
            registration_results[system_id] = True
            logging.info(f"{system_id} registered with Nyx governance")
        except Exception as e:
            logging.error(f"Error registering {system_id}: {e}")
            registration_results[system_id] = False
    
    # Issue directive for cooperative operation
    await governance.issue_directive(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_id="lore_generator",
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Collaborate with enhanced lore systems to ensure consistent world development.",
            "scope": "global"
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24*60  # 24 hours
    )
    
    return registration_results

async def initialize_enhanced_lore_tables(user_id: int, conversation_id: int) -> bool:
    """Initialize all necessary database tables for enhanced lore systems"""
    try:
        # Initialize faith system tables
        faith_system = FaithSystem(user_id, conversation_id)
        await faith_system.initialize_tables()
        
        # Initialize any other necessary tables
        # (Most systems use existing tables)
        
        return True
    except Exception as e:
        logging.error(f"Error initializing enhanced lore tables: {e}")
        return False

async def generate_initial_enhanced_lore(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """Generate initial enhanced lore for a new game world"""
    generated_lore = {}
    
    # Systems to generate initial lore
    systems = [
        (FaithSystem(user_id, conversation_id), "generate_complete_faith_system", "faith_system"),
        (EmergentLoreSystem(user_id, conversation_id), "generate_cyclical_events", "seasonal_events")
    ]
    
    # Generate lore from each system
    for system, method_name, result_key in systems:
        try:
            await system.initialize_governance()
            
            # Create run context
            run_ctx = RunContextWrapper(context={
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            # Call the appropriate generation method
            if method_name == "generate_cyclical_events":
                # Special case for seasonal events
                seasonal_events = []
                for season in ["spring", "summer", "fall", "winter"]:
                    event = await system.generate_cyclical_event(run_ctx, season=season)
                    seasonal_events.append(event)
                
                # Add solstice and equinox
                solstice_event = await system.generate_cyclical_event(run_ctx, season="summer", is_solstice=True)
                equinox_event = await system.generate_cyclical_event(run_ctx, season="spring", is_equinox=True)
                seasonal_events.extend([solstice_event, equinox_event])
                
                generated_lore[result_key] = seasonal_events
            else:
                # Standard method call
                method = getattr(system, method_name)
                result = await method(run_ctx)
                generated_lore[result_key] = result
                
            logging.info(f"Generated {result_key}")
        except Exception as e:
            logging.error(f"Error generating {result_key}: {e}")
            generated_lore[f"{result_key}_error"] = str(e)
    
    return generated_lore

async def evolve_world_over_time(user_id: int, conversation_id: int, days_passed: int = 30) -> Dict[str, Any]:
    """Evolve the world over time, simulating the passage of days or weeks"""
    evolution_results = {}
    
    # 1. Use LoreEvolutionSystem to mature lore
    try:
        lore_evolution = LoreEvolutionSystem(user_id, conversation_id)
        await lore_evolution.initialize_governance()
        
        # Create run context
        run_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        
        maturation = await lore_evolution.mature_lore_over_time(run_ctx, days_passed)
        evolution_results["lore_maturation"] = maturation
        logging.info(f"Matured lore over {days_passed} days with {maturation.get('changes_applied', 0)} changes")
    except Exception as e:
        logging.error(f"Error maturing lore: {e}")
        evolution_results["lore_maturation_error"] = str(e)
    
    # 2. Generate random emergent events
    try:
        emergent_lore = EmergentLoreSystem(user_id, conversation_id)
        await emergent_lore.initialize_governance()
        
        # Create run context
        run_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        
        # Scale events with time passed
        num_events = max(1, min(5, days_passed // 10))
        num_events = random.randint(1, num_events)
        
        emergent_events = []
        for _ in range(num_events):
            event = await emergent_lore.generate_emergent_event(run_ctx)
            emergent_events.append(event)
        
        evolution_results["emergent_events"] = emergent_events
        logging.info(f"Generated {len(emergent_events)} emergent events")
    except Exception as e:
        logging.error(f"Error generating emergent events: {e}")
        evolution_results["emergent_events_error"] = str(e)
    
    # 3. Maybe add a new nation if enough time has passed
    if days_passed >= 60 and random.random() < 0.3:  # 30% chance if 60+ days
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            new_nation = await emergent_lore.generate_additional_nation(run_ctx)
            evolution_results["new_nation"] = new_nation
            logging.info(f"Generated new nation: {new_nation.get('name', 'Unknown')}")
        except Exception as e:
            logging.error(f"Error generating new nation: {e}")
            evolution_results["new_nation_error"] = str(e)
    
    return evolution_results
# Missing Classes:

# 1. Urban Myth Manager - completely missing from consolidated version
class UrbanMythManager(BaseLoreSystem):
    """
    Manager for urban myths, local stories, and folk tales that develop organically
    across different regions and communities.
    """
    
    async def initialize_tables(self):
        """Ensure urban myth tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if UrbanMyths table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'urbanmyths'
                    );
                """)
                
                if not table_exists:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE UrbanMyths (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            description TEXT NOT NULL,
                            origin_location TEXT,
                            origin_event TEXT,
                            believability INTEGER CHECK (believability BETWEEN 1 AND 10),
                            spread_rate INTEGER CHECK (spread_rate BETWEEN 1 AND 10),
                            regions_known TEXT[],
                            variations TEXT[],
                            creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_urbanmyths_embedding 
                        ON UrbanMyths USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("UrbanMyths table created")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_urban_myth",
        action_description="Adding urban myth: {name}",
        id_from_context=lambda ctx: "urban_myth_manager"
    )
    async def add_urban_myth(
        self, 
        ctx,
        name: str, 
        description: str, 
        origin_location: Optional[str] = None,
        origin_event: Optional[str] = None,
        believability: int = 6,
        spread_rate: int = 5,
        regions_known: List[str] = None
    ) -> int:
        """
        Add a new urban myth to the database
        
        Args:
            name: Name of the urban myth
            description: Full description of the myth
            origin_location: Where the myth originated
            origin_event: What event spawned the myth
            believability: How believable the myth is (1-10)
            spread_rate: How quickly the myth is spreading (1-10)
            regions_known: List of regions where the myth is known
            
        Returns:
            ID of the created urban myth
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Generate embedding for the myth
        embedding_text = f"{name} {description}"
        embedding = await generate_embedding(embedding_text)
        
        # Set defaults
        regions_known = regions_known or ["local area"]
        
        # Store in database
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                myth_id = await conn.fetchval("""
                    INSERT INTO UrbanMyths (
                        name, description, origin_location, origin_event,
                        believability, spread_rate, regions_known, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """, name, description, origin_location, origin_event,
                     believability, spread_rate, regions_known, embedding)
                
                return myth_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_myths_for_location",
        action_description="Getting myths for location: {location}",
        id_from_context=lambda ctx: "urban_myth_manager"
    )
    async def get_myths_for_location(self, ctx, location: str) -> List[Dict[str, Any]]:
        """
        Get urban myths known in a specific location
        
        Args:
            location: The location name
            
        Returns:
            List of urban myths known in this location
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if we have a direct match where this location is the origin
                direct_myths = await conn.fetch("""
                    SELECT id, name, description, believability, spread_rate, variations
                    FROM UrbanMyths
                    WHERE origin_location = $1
                """, location)
                
                # Check for myths where this location is in the regions_known array
                known_myths = await conn.fetch("""
                    SELECT id, name, description, believability, spread_rate, variations
                    FROM UrbanMyths
                    WHERE $1 = ANY(regions_known)
                """, location)
                
                # Combine and deduplicate
                myth_ids = set()
                myths = []
                
                for myth in direct_myths:
                    myth_dict = dict(myth)
                    myth_dict["origin"] = True
                    myths.append(myth_dict)
                    myth_ids.add(myth["id"])
                
                for myth in known_myths:
                    if myth["id"] not in myth_ids:
                        myth_dict = dict(myth)
                        myth_dict["origin"] = False
                        myths.append(myth_dict)
                
                return myths
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_myths_for_location",
        action_description="Generating myths for location: {location_data['name']}",
        id_from_context=lambda ctx: "urban_myth_manager"
    )
    async def generate_myths_for_location(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate urban myths for a specific location based on its characteristics
        
        Args:
            location_data: Dictionary with location details
            
        Returns:
            List of generated urban myths
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Extract relevant details
        location_name = location_data.get('name', 'Unknown Location')
        location_type = location_data.get('type', 'place')
        description = location_data.get('description', '')
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 2-3 urban myths or local legends associated with this location:
        
        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {description}
        
        Create urban myths that feel authentic to this location. Each myth should:
        1. Be somewhat believable but with fantastical elements
        2. Reflect local concerns, features, or history
        3. Have some connection to matriarchal power structures
        
        Format your response as a JSON array where each object has:
        - "name": The name/title of the myth
        - "description": A detailed description of the myth
        - "believability": Number from 1-10 indicating how believable it is
        - "spread_rate": Number from 1-10 indicating how widely it has spread
        - "origin": Brief statement of how the myth originated
        """
        
        # Create an agent for myth generation
        myth_agent = Agent(
            name="UrbanMythAgent",
            instructions="You create urban myths and local legends for locations.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(myth_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            myths = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(myths, list):
                if isinstance(myths, dict):
                    myths = [myths]
                else:
                    myths = []
            
            # Store each myth
            saved_myths = []
            for myth in myths:
                # Extract myth details
                name = myth.get('name')
                description = myth.get('description')
                believability = myth.get('believability', random.randint(4, 8))
                spread_rate = myth.get('spread_rate', random.randint(3, 7))
                
                if not name or not description:
                    continue
                
                # Save the myth
                try:
                    myth_id = await self.add_urban_myth(
                        run_ctx,
                        name=name,
                        description=description,
                        origin_location=location_name,
                        believability=believability,
                        spread_rate=spread_rate,
                        regions_known=[location_name]
                    )
                    
                    # Add to results
                    myth['id'] = myth_id
                    saved_myths.append(myth)
                except Exception as e:
                    logging.error(f"Error saving urban myth '{name}': {e}")
            
            return saved_myths
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for urban myths: {response_text}")
            return []

# 2. Local History Manager - completely missing from consolidated version
class LocalHistoryManager(BaseLoreSystem):
    """
    Manager for local histories, events, and landmarks that are specific
    to particular locations rather than the broader world history.
    """
    
    async def initialize_tables(self):
        """Ensure local history tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if LocalHistories table exists
                local_history_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'localhistories'
                    );
                """)
                
                if not local_history_exists:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE LocalHistories (
                            id SERIAL PRIMARY KEY,
                            location_id INTEGER NOT NULL,
                            event_name TEXT NOT NULL,
                            description TEXT NOT NULL,
                            date_description TEXT,
                            significance INTEGER CHECK (significance BETWEEN 1 AND 10),
                            impact_type TEXT,
                            notable_figures TEXT[],
                            current_relevance TEXT,
                            commemoration TEXT,
                            embedding VECTOR(1536),
                            FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                        );
                    """)
                    
                    # Create indexes
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_localhistories_embedding 
                        ON LocalHistories USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_localhistories_location
                        ON LocalHistories(location_id);
                    """)
                    
                    logging.info("LocalHistories table created")
                
                # Check if Landmarks table exists
                landmarks_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'landmarks'
                    );
                """)
                
                if not landmarks_exists:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE Landmarks (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            location_id INTEGER NOT NULL,
                            landmark_type TEXT NOT NULL,
                            description TEXT NOT NULL,
                            historical_significance TEXT,
                            current_use TEXT,
                            controlled_by TEXT,
                            legends TEXT[],
                            embedding VECTOR(1536),
                            FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                        );
                    """)
                    
                    # Create indexes
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_landmarks_embedding 
                        ON Landmarks USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_landmarks_location
                        ON Landmarks(location_id);
                    """)
                    
                    logging.info("Landmarks table created")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_local_history",
        action_description="Adding local history event: {event_name}",
        id_from_context=lambda ctx: "local_history_manager"
    )
    async def add_local_history(
        self, 
        ctx,
        location_id: int,
        event_name: str,
        description: str,
        date_description: str = "Some time ago",
        significance: int = 5,
        impact_type: str = "cultural",
        notable_figures: List[str] = None,
        current_relevance: str = None,
        commemoration: str = None
    ) -> int:
        """
        Add a local historical event to the database
        
        Args:
            location_id: ID of the associated location
            event_name: Name of the historical event
            description: Description of the event
            date_description: When it occurred
            significance: Importance from 1-10
            impact_type: Type of impact (political, cultural, etc.)
            notable_figures: People involved
            current_relevance: How it affects the present
            commemoration: How it's remembered/celebrated
            
        Returns:
            ID of the created local history event
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults
        notable_figures = notable_figures or []
        
        # Generate embedding
        embedding_text = f"{event_name} {description} {date_description}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                event_id = await conn.fetchval("""
                    INSERT INTO LocalHistories (
                        location_id, event_name, description, date_description,
                        significance, impact_type, notable_figures,
                        current_relevance, commemoration, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """, location_id, event_name, description, date_description,
                     significance, impact_type, notable_figures,
                     current_relevance, commemoration, embedding)
                
                return event_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_landmark",
        action_description="Adding landmark: {name}",
        id_from_context=lambda ctx: "local_history_manager"
    )
    async def add_landmark(
        self, 
        ctx,
        name: str,
        location_id: int,
        landmark_type: str,
        description: str,
        historical_significance: str = None,
        current_use: str = None,
        controlled_by: str = None,
        legends: List[str] = None
    ) -> int:
        """
        Add a landmark to the database
        
        Args:
            name: Name of the landmark
            location_id: ID of the associated location
            landmark_type: Type of landmark (monument, building, natural feature, etc.)
            description: Description of the landmark
            historical_significance: Historical importance
            current_use: How it's used today
            controlled_by: Who controls/owns it
            legends: Associated legends or stories
            
        Returns:
            ID of the created landmark
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults
        legends = legends or []
        
        # Generate embedding
        embedding_text = f"{name} {landmark_type} {description}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                landmark_id = await conn.fetchval("""
                    INSERT INTO Landmarks (
                        name, location_id, landmark_type, description,
                        historical_significance, current_use, controlled_by,
                        legends, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """, name, location_id, landmark_type, description,
                     historical_significance, current_use, controlled_by,
                     legends, embedding)
                
                return landmark_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_local_history",
        action_description="Getting local history for location: {location_name}",
        id_from_context=lambda ctx: "local_history_manager"
    )
    async def get_local_history(self, ctx, location_id: int, location_name: str) -> List[Dict[str, Any]]:
        """
        Get local historical events for a specific location
        
        Args:
            location_id: The location ID
            location_name: The name of the location (for logs)
            
        Returns:
            List of local historical events
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                events = await conn.fetch("""
                    SELECT id, event_name, description, date_description, significance,
                           impact_type, notable_figures, current_relevance, commemoration
                    FROM LocalHistories
                    WHERE location_id = $1
                    ORDER BY significance DESC
                """, location_id)
                
                return [dict(event) for event in events]
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_landmarks",
        action_description="Getting landmarks for location: {location_name}",
        id_from_context=lambda ctx: "local_history_manager"
    )
    async def get_landmarks(self, ctx, location_id: int, location_name: str) -> List[Dict[str, Any]]:
        """
        Get landmarks for a specific location
        
        Args:
            location_id: The location ID
            location_name: The name of the location (for logs)
            
        Returns:
            List of landmarks
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                landmarks = await conn.fetch("""
                    SELECT id, name, landmark_type, description, historical_significance,
                           current_use, controlled_by, legends
                    FROM Landmarks
                    WHERE location_id = $1
                """, location_id)
                
                return [dict(landmark) for landmark in landmarks]
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_local_history",
        action_description="Generating local history for: {location_data['name']}",
        id_from_context=lambda ctx: "local_history_manager"
    )
    async def generate_local_history(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate local historical events for a location based on its characteristics
        
        Args:
            location_data: Dictionary with location details
            
        Returns:
            List of generated local historical events
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Extract relevant details
        location_id = location_data.get('id')
        location_name = location_data.get('name', 'Unknown Location')
        location_type = location_data.get('type', 'place')
        description = location_data.get('description', '')
        
        # Get world lore for context
        world_history = await self._get_world_history_context()
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 3-4 local historical events specific to this location:
        
        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {description}
        
        WORLD HISTORY CONTEXT:
        {world_history}
        
        Create local historical events that feel authentic to this location. Each event should:
        1. Be specific to this location rather than world-changing
        2. Reflect local development, conflicts, or cultural shifts
        3. Include at least one event related to matriarchal power structures
        4. Include a range of timeframes (some recent, some older)
        
        Format your response as a JSON array where each object has:
        - "event_name": The name of the historical event
        - "description": A detailed description of what happened
        - "date_description": When it occurred (e.g., "50 years ago", "during the reign of...")
        - "significance": Number from 1-10 indicating historical importance
        - "impact_type": Type of impact (political, cultural, economic, religious, etc.)
        - "notable_figures": Array of names of people involved
        - "current_relevance": How it still affects the location today
        - "commemoration": How it's remembered (festival, monument, etc.)
        """
        
        # Create an agent for local history generation
        history_agent = Agent(
            name="LocalHistoryAgent",
            instructions="You create local historical events for specific locations.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(history_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            events = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(events, list):
                if isinstance(events, dict):
                    events = [events]
                else:
                    events = []
            
            # Store each event
            saved_events = []
            for event in events:
                # Extract event details
                event_name = event.get('event_name')
                description = event.get('description')
                date_description = event.get('date_description', 'Some time ago')
                significance = event.get('significance', 5)
                impact_type = event.get('impact_type', 'cultural')
                notable_figures = event.get('notable_figures', [])
                current_relevance = event.get('current_relevance')
                commemoration = event.get('commemoration')
                
                if not event_name or not description:
                    continue
                
                # Save the event
                try:
                    event_id = await self.add_local_history(
                        run_ctx,
                        location_id=location_id,
                        event_name=event_name,
                        description=description,
                        date_description=date_description,
                        significance=significance,
                        impact_type=impact_type,
                        notable_figures=notable_figures,
                        current_relevance=current_relevance,
                        commemoration=commemoration
                    )
                    
                    # Add to results
                    event['id'] = event_id
                    saved_events.append(event)
                except Exception as e:
                    logging.error(f"Error saving local historical event '{event_name}': {e}")
            
            return saved_events
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for local history: {response_text}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_landmarks",
        action_description="Generating landmarks for: {location_data['name']}",
        id_from_context=lambda ctx: "local_history_manager"
    )
    async def generate_landmarks(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate landmarks for a location based on its characteristics
        
        Args:
            location_data: Dictionary with location details
            
        Returns:
            List of generated landmarks
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Extract relevant details
        location_id = location_data.get('id')
        location_name = location_data.get('name', 'Unknown Location')
        location_type = location_data.get('type', 'place')
        description = location_data.get('description', '')
        
        # Get local history for context
        local_history = await self.get_local_history(run_ctx, location_id, location_name)
        history_context = "\n".join([f"- {e['event_name']}: {e['description'][:100]}..." for e in local_history[:2]])
        
        # Get factions that control this area
        controlling_factions = await self._get_controlling_factions(location_id)
        faction_context = ", ".join([f['name'] for f in controlling_factions]) if controlling_factions else "Various groups"
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 3-5 landmarks found in this location:
        
        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {description}
        CONTROLLING FACTIONS: {faction_context}
        
        LOCAL HISTORY CONTEXT:
        {history_context}
        
        Create landmarks that feel authentic to this location. Include:
        1. At least one natural landmark (if appropriate)
        2. At least one architectural/built landmark
        3. At least one landmark related to matriarchal power structures
        4. A mix of ancient/historical and more recent landmarks
        
        Format your response as a JSON array where each object has:
        - "name": The name of the landmark
        - "landmark_type": Type of landmark (monument, building, natural feature, temple, etc.)
        - "description": A detailed physical description
        - "historical_significance": Its importance to local history
        - "current_use": How it's used today (ceremonial, practical, tourist attraction, etc.)
        - "controlled_by": Which faction or group controls it
        - "legends": Array of brief legends or stories associated with it
        """
        
        # Create an agent for landmark generation
        landmark_agent = Agent(
            name="LandmarkAgent",
            instructions="You create landmarks for specific locations.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(landmark_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            landmarks = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(landmarks, list):
                if isinstance(landmarks, dict):
                    landmarks = [landmarks]
                else:
                    landmarks = []
            
            # Store each landmark
            saved_landmarks = []
            for landmark in landmarks:
                # Extract landmark details
                name = landmark.get('name')
                landmark_type = landmark.get('landmark_type', 'building')
                description = landmark.get('description')
                historical_significance = landmark.get('historical_significance')
                current_use = landmark.get('current_use')
                controlled_by = landmark.get('controlled_by')
                legends = landmark.get('legends', [])
                
                if not name or not description:
                    continue
                
                # Save the landmark
                try:
                    landmark_id = await self.add_landmark(
                        run_ctx,
                        name=name,
                        location_id=location_id,
                        landmark_type=landmark_type,
                        description=description,
                        historical_significance=historical_significance,
                        current_use=current_use,
                        controlled_by=controlled_by,
                        legends=legends
                    )
                    
                    # Add to results
                    landmark['id'] = landmark_id
                    saved_landmarks.append(landmark)
                except Exception as e:
                    logging.error(f"Error saving landmark '{name}': {e}")
            
            return saved_landmarks
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for landmarks: {response_text}")
            return []
    
    async def _get_world_history_context(self) -> str:
        """Get relevant world history for context when generating local history"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Try to get world history from WorldLore
                world_history = await conn.fetchval("""
                    SELECT description FROM WorldLore
                    WHERE category = 'world_history'
                    LIMIT 1
                """)
                
                if world_history:
                    return world_history
                
                # Fall back to historical events
                events = await conn.fetch("""
                    SELECT name, description FROM HistoricalEvents
                    ORDER BY significance DESC
                    LIMIT 3
                """)
                
                if events:
                    return "\n".join([f"{e['name']}: {e['description'][:150]}..." for e in events])
                
                # Default context if nothing found
                return "The world has a rich history of matriarchal societies and female-dominated power structures."
    
    async def _get_controlling_factions(self, location_id: int) -> List[Dict[str, Any]]:
        """Get factions that control a specific location"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get location details first
                location = await conn.fetchrow("""
                    SELECT id, name FROM Locations
                    WHERE id = $1
                """, location_id)
                
                if not location:
                    return []
                    
                # Check for direct connections to factions
                factions = await conn.fetch("""
                    SELECT f.id, f.name, f.type, lc.strength, lc.description
                    FROM Factions f
                    JOIN LoreConnections lc ON f.id = lc.source_id
                    WHERE lc.target_type = 'LocationLore'
                    AND lc.target_id = $1
                    AND lc.source_type = 'Factions'
                    AND lc.connection_type = 'influences'
                """, location_id)
                
                if factions:
                    return [dict(faction) for faction in factions]
                    
                # If no direct connections, check location lore for faction mentions
                location_lore = await conn.fetchrow("""
                    SELECT associated_factions FROM LocationLore
                    WHERE location_id = $1
                """, location_id)
                
                if location_lore and location_lore['associated_factions']:
                    faction_names = location_lore['associated_factions']
                    factions = []
                    
                    for name in faction_names:
                        faction = await conn.fetchrow("""
                            SELECT id, name, type FROM Factions
                            WHERE name = $1
                        """, name)
                        
                        if faction:
                            factions.append(dict(faction))
                    
                    return factions
                
                # If still nothing, return empty list
                return []

# 3. Educational System Manager - completely missing from consolidated version
class EducationalSystemManager(BaseLoreSystem):
    """
    Manages how knowledge is taught and passed down across generations
    within the matriarchal society, including formal and informal systems.
    """
    
    async def initialize_tables(self):
        """Ensure educational system tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if EducationalSystems table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'educationalsystems'
                    );
                """)
                
                if not table_exists:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE EducationalSystems (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            system_type TEXT NOT NULL,
                            description TEXT NOT NULL,
                            target_demographics TEXT[],
                            controlled_by TEXT,
                            core_teachings TEXT[],
                            teaching_methods TEXT[],
                            coming_of_age_rituals TEXT,
                            knowledge_restrictions TEXT,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_educationalsystems_embedding 
                        ON EducationalSystems USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("EducationalSystems table created")
                
                # Check if KnowledgeTraditions table exists
                traditions_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'knowledgetraditions'
                    );
                """)
                
                if not traditions_exists:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE KnowledgeTraditions (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            tradition_type TEXT NOT NULL,
                            description TEXT NOT NULL,
                            knowledge_domain TEXT NOT NULL,
                            preservation_method TEXT,
                            access_requirements TEXT,
                            associated_group TEXT,
                            examples TEXT[],
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_knowledgetraditions_embedding 
                        ON KnowledgeTraditions USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("KnowledgeTraditions table created")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_educational_system",
        action_description="Adding educational system: {name}",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def add_educational_system(
        self, 
        ctx,
        name: str,
        system_type: str,
        description: str,
        target_demographics: List[str],
        controlled_by: str,
        core_teachings: List[str],
        teaching_methods: List[str],
        coming_of_age_rituals: str = None,
        knowledge_restrictions: str = None
    ) -> int:
        """
        Add an educational system to the database
        
        Args:
            name: Name of the educational system
            system_type: Type of system (formal, apprenticeship, etc.)
            description: Detailed description
            target_demographics: Who is educated in this system
            controlled_by: Which faction/group controls this system
            core_teachings: What is taught
            teaching_methods: How it's taught
            coming_of_age_rituals: Rituals marking educational milestones
            knowledge_restrictions: Limitations on who can learn what
            
        Returns:
            ID of the created educational system
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Generate embedding
        embedding_text = f"{name} {system_type} {description} {' '.join(core_teachings)}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                system_id = await conn.fetchval("""
                    INSERT INTO EducationalSystems (
                        name, system_type, description, target_demographics,
                        controlled_by, core_teachings, teaching_methods, 
                        coming_of_age_rituals, knowledge_restrictions, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """, name, system_type, description, target_demographics,
                     controlled_by, core_teachings, teaching_methods,
                     coming_of_age_rituals, knowledge_restrictions, embedding)
                
                return system_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_knowledge_tradition",
        action_description="Adding knowledge tradition: {name}",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def add_knowledge_tradition(
        self, 
        ctx,
        name: str,
        tradition_type: str,
        description: str,
        knowledge_domain: str,
        preservation_method: str = None,
        access_requirements: str = None,
        associated_group: str = None,
        examples: List[str] = None
    ) -> int:
        """
        Add a knowledge tradition to the database
        
        Args:
            name: Name of the knowledge tradition
            tradition_type: Type of tradition (oral, written, experiential, etc.)
            description: Detailed description
            knowledge_domain: Domain of knowledge (medicine, crafts, etc.)
            preservation_method: How the knowledge is preserved
            access_requirements: Requirements to access this knowledge
            associated_group: Group that maintains this tradition
            examples: Examples of specific knowledge transmitted
            
        Returns:
            ID of the created knowledge tradition
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults
        examples = examples or []
        
        # Generate embedding
        embedding_text = f"{name} {tradition_type} {description} {knowledge_domain}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                tradition_id = await conn.fetchval("""
                    INSERT INTO KnowledgeTraditions (
                        name, tradition_type, description, knowledge_domain,
                        preservation_method, access_requirements,
                        associated_group, examples, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """, name, tradition_type, description, knowledge_domain,
                     preservation_method, access_requirements,
                     associated_group, examples, embedding)
                
                return tradition_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_educational_systems",
        action_description="Generating educational systems for the setting",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def generate_educational_systems(self, ctx) -> List[Dict[str, Any]]:
        """
        Generate a comprehensive set of educational systems for the setting
        
        Returns:
            List of generated educational systems
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Get factions for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                factions = await conn.fetch("""
                    SELECT name, type FROM Factions
                    LIMIT 5
                """)
                
                faction_names = [f"{f['name']} ({f['type']})" for f in factions]
                faction_context = ", ".join(faction_names)
                
                # Get cultural elements for context
                elements = await conn.fetch("""
                    SELECT name, type FROM CulturalElements
                    LIMIT 5
                """)
                
                element_names = [f"{e['name']} ({e['type']})" for e in elements]
                culture_context = ", ".join(element_names)
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 3-4 educational systems for a matriarchal society.
        
        SOCIETAL CONTEXT:
        Factions: {faction_context}
        Cultural Elements: {culture_context}
        
        Design educational systems that:
        1. Reflect matriarchal power structures and values
        2. Include different approaches for different demographics
        3. Show how gender roles are taught and reinforced
        4. Include both formal and informal education systems
        
        Format your response as a JSON array where each object has:
        - "name": The name of the educational system
        - "system_type": Type of system (formal academy, mentorship, etc.)
        - "description": A detailed description of how the system works
        - "target_demographics": Array of who is educated in this system
        - "controlled_by": Which faction/group controls this system
        - "core_teachings": Array of what is taught
        - "teaching_methods": Array of how it's taught
        - "coming_of_age_rituals": Rituals marking educational milestones
        - "knowledge_restrictions": Limitations on who can learn what
        """
        
        # Create an agent for educational system generation
        education_agent = Agent(
            name="EducationalSystemAgent",
            instructions="You create educational systems for fictional societies.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(education_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            systems = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(systems, list):
                if isinstance(systems, dict):
                    systems = [systems]
                else:
                    systems = []
            
            # Store each system
            saved_systems = []
            for system in systems:
                # Extract system details
                name = system.get('name')
                system_type = system.get('system_type', 'formal')
                description = system.get('description')
                target_demographics = system.get('target_demographics', ['general population'])
                controlled_by = system.get('controlled_by', 'various factions')
                core_teachings = system.get('core_teachings', ['general knowledge'])
                teaching_methods = system.get('teaching_methods', ['lecture', 'practice'])
                coming_of_age_rituals = system.get('coming_of_age_rituals')
                knowledge_restrictions = system.get('knowledge_restrictions')
                
                if not name or not description:
                    continue
                
                # Save the system
                try:
                    system_id = await self.add_educational_system(
                        run_ctx,
                        name=name,
                        system_type=system_type,
                        description=description,
                        target_demographics=target_demographics,
                        controlled_by=controlled_by,
                        core_teachings=core_teachings,
                        teaching_methods=teaching_methods,
                        coming_of_age_rituals=coming_of_age_rituals,
                        knowledge_restrictions=knowledge_restrictions
                    )
                    
                    # Add to results
                    system['id'] = system_id
                    saved_systems.append(system)
                except Exception as e:
                    logging.error(f"Error saving educational system '{name}': {e}")
            
            return saved_systems
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for educational systems: {response_text}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_knowledge_traditions",
        action_description="Generating knowledge traditions for the setting",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def generate_knowledge_traditions(self, ctx) -> List[Dict[str, Any]]:
        """
        Generate knowledge traditions that represent how knowledge is
        passed down through generations in informal ways
        
        Returns:
            List of generated knowledge traditions
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Get cultural elements for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                elements = await conn.fetch("""
                    SELECT name, type, description FROM CulturalElements
                    LIMIT 3
                """)
                
                culture_context = "\n".join([f"- {e['name']} ({e['type']}): {e['description'][:100]}..." for e in elements])
                
                # Get educational systems for context
                systems = await conn.fetch("""
                    SELECT name, system_type FROM EducationalSystems
                    LIMIT 2
                """)
                
                systems_context = ", ".join([f"{s['name']} ({s['system_type']})" for s in systems]) if systems else "No formal systems yet"
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 4-5 knowledge traditions for how specific types of knowledge are passed down
        through generations in a matriarchal society.
        
        CULTURAL CONTEXT:
        {culture_context}
        
        FORMAL EDUCATIONAL SYSTEMS:
        {systems_context}
        
        Design knowledge traditions that:
        1. Cover a range of knowledge domains (spiritual, practical, artistic, etc.)
        2. Include both female-exclusive and general knowledge traditions
        3. Show how generational knowledge transfer reinforces power structures
        4. Include different preservation methods (oral, written, ritual, etc.)
        
        Format your response as a JSON array where each object has:
        - "name": The name of the tradition
        - "tradition_type": Type of tradition (oral, written, experiential, etc.)
        - "description": A detailed description of the tradition
        - "knowledge_domain": Domain of knowledge (medicine, crafts, leadership, etc.)
        - "preservation_method": How the knowledge is preserved over generations
        - "access_requirements": Requirements to access this knowledge
        - "associated_group": Group that maintains this tradition
        - "examples": Array of specific examples of knowledge transmitted this way
        """
        
        # Create an agent for knowledge tradition generation
        tradition_agent = Agent(
            name="KnowledgeTraditionAgent",
            instructions="You create knowledge transmission traditions for fictional societies.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(tradition_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            traditions = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(traditions, list):
                if isinstance(traditions, dict):
                    traditions = [traditions]
                else:
                    traditions = []
            
            # Store each tradition
            saved_traditions = []
            for tradition in traditions:
                # Extract tradition details
                name = tradition.get('name')
                tradition_type = tradition.get('tradition_type', 'oral')
                description = tradition.get('description')
                knowledge_domain = tradition.get('knowledge_domain', 'general knowledge')
                preservation_method = tradition.get('preservation_method')
                access_requirements = tradition.get('access_requirements')
                associated_group = tradition.get('associated_group')
                examples = tradition.get('examples', [])
                
                if not name or not description:
                    continue
                
                # Save the tradition
                try:
                    tradition_id = await self.add_knowledge_tradition(
                        run_ctx,
                        name=name,
                        tradition_type=tradition_type,
                        description=description,
                        knowledge_domain=knowledge_domain,
                        preservation_method=preservation_method,
                        access_requirements=access_requirements,
                        associated_group=associated_group,
                        examples=examples
                    )
                    
                    # Add to results
                    tradition['id'] = tradition_id
                    saved_traditions.append(tradition)
                except Exception as e:
                    logging.error(f"Error saving knowledge tradition '{name}': {e}")
            
            return saved_traditions
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for knowledge traditions: {response_text}")
            return []

# 4. Geopolitical System Manager - completely missing from consolidated version
class GeopoliticalSystemManager(BaseLoreSystem):
    """
    Manages the geopolitical landscape of the world, including countries,
    regions, foreign relations, and international power dynamics.
    """
    
    async def initialize_tables(self):
        """Ensure geopolitical system tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if Nations table exists
                nations_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'nations'
                    );
                """)
                
                if not nations_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE Nations (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            government_type TEXT NOT NULL,
                            description TEXT NOT NULL,
                            relative_power INTEGER CHECK (relative_power BETWEEN 1 AND 10),
                            matriarchy_level INTEGER CHECK (matriarchy_level BETWEEN 1 AND 10),
                            population_scale TEXT,
                            major_resources TEXT[],
                            major_cities TEXT[],
                            cultural_traits TEXT[],
                            notable_features TEXT,
                            neighboring_nations TEXT[],
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_nations_embedding 
                        ON Nations USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("Nations table created")
                
                # Check if InternationalRelations table exists
                relations_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'internationalrelations'
                    );
                """)
                
                if not relations_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE InternationalRelations (
                            id SERIAL PRIMARY KEY,
                            nation1_id INTEGER NOT NULL,
                            nation2_id INTEGER NOT NULL,
                            relationship_type TEXT NOT NULL,
                            relationship_quality INTEGER CHECK (relationship_quality BETWEEN 1 AND 10),
                            description TEXT NOT NULL,
                            notable_conflicts TEXT[],
                            notable_alliances TEXT[],
                            trade_relations TEXT,
                            cultural_exchanges TEXT,
                            FOREIGN KEY (nation1_id) REFERENCES Nations(id) ON DELETE CASCADE,
                            FOREIGN KEY (nation2_id) REFERENCES Nations(id) ON DELETE CASCADE,
                            UNIQUE (nation1_id, nation2_id)
                        );
                    """)
                    
                    logging.info("InternationalRelations table created")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_nation",
        action_description="Adding nation: {name}",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def add_nation(
        self, 
        ctx,
        name: str,
        government_type: str,
        description: str,
        relative_power: int,
        matriarchy_level: int,
        population_scale: str = None,
        major_resources: List[str] = None,
        major_cities: List[str] = None,
        cultural_traits: List[str] = None,
        notable_features: str = None,
        neighboring_nations: List[str] = None
    ) -> int:
        """
        Add a nation to the database
        
        Args:
            name: Name of the nation
            government_type: Type of government
            description: Detailed description
            relative_power: Power level (1-10)
            matriarchy_level: How matriarchal (1-10)
            population_scale: Scale of population
            major_resources: Key resources
            major_cities: Key cities/settlements
            cultural_traits: Defining cultural traits
            notable_features: Other notable features
            neighboring_nations: Nations that border this one
            
        Returns:
            ID of the created nation
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults
        major_resources = major_resources or []
        major_cities = major_cities or []
        cultural_traits = cultural_traits or []
        neighboring_nations = neighboring_nations or []
        
        # Generate embedding
        embedding_text = f"{name} {government_type} {description}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation_id = await conn.fetchval("""
                    INSERT INTO Nations (
                        name, government_type, description, relative_power,
                        matriarchy_level, population_scale, major_resources,
                        major_cities, cultural_traits, notable_features,
                        neighboring_nations, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    RETURNING id
                """, name, government_type, description, relative_power,
                     matriarchy_level, population_scale, major_resources,
                     major_cities, cultural_traits, notable_features,
                     neighboring_nations, embedding)
                
                return nation_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_international_relation",
        action_description="Adding relation between nations",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def add_international_relation(
        self, 
        ctx,
        nation1_id: int,
        nation2_id: int,
        relationship_type: str,
        relationship_quality: int,
        description: str,
        notable_conflicts: List[str] = None,
        notable_alliances: List[str] = None,
        trade_relations: str = None,
        cultural_exchanges: str = None
    ) -> int:
        """
        Add a relation between two nations
        
        Args:
            nation1_id: ID of first nation
            nation2_id: ID of second nation
            relationship_type: Type of relationship (ally, rival, etc.)
            relationship_quality: Quality level (1-10)
            description: Description of relationship
            notable_conflicts: Notable conflicts
            notable_alliances: Notable alliances
            trade_relations: Description of trade
            cultural_exchanges: Description of cultural exchanges
            
        Returns:
            ID of the created relation
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults
        notable_conflicts = notable_conflicts or []
        notable_alliances = notable_alliances or []
        
        # Store in database
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                relation_id = await conn.fetchval("""
                    INSERT INTO InternationalRelations (
                        nation1_id, nation2_id, relationship_type,
                        relationship_quality, description, notable_conflicts,
                        notable_alliances, trade_relations, cultural_exchanges
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (nation1_id, nation2_id) DO UPDATE
                    SET relationship_type = $3,
                        relationship_quality = $4,
                        description = $5,
                        notable_conflicts = $6,
                        notable_alliances = $7,
                        trade_relations = $8,
                        cultural_exchanges = $9
                    RETURNING id
                """, nation1_id, nation2_id, relationship_type,
                     relationship_quality, description, notable_conflicts,
                     notable_alliances, trade_relations, cultural_exchanges)
                
                return relation_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_all_nations",
        action_description="Getting all nations in the world",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def get_all_nations(self, ctx) -> List[Dict[str, Any]]:
        """
        Get all nations in the world
        
        Returns:
            List of all nations
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nations = await conn.fetch("""
                    SELECT id, name, government_type, description, relative_power,
                           matriarchy_level, population_scale, major_resources,
                           major_cities, cultural_traits, notable_features, 
                           neighboring_nations
                    FROM Nations
                    ORDER BY relative_power DESC
                """)
                
                return [dict(nation) for nation in nations]
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_relations",
        action_description="Getting relations for nation: {nation_name}",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def get_nation_relations(self, ctx, nation_id: int, nation_name: str) -> List[Dict[str, Any]]:
        """
        Get all international relations for a specific nation
        
        Args:
            nation_id: ID of the nation
            nation_name: Name of the nation (for logs)
            
        Returns:
            List of international relations
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get relations where this nation is either nation1 or nation2
                relations = await conn.fetch("""
                    SELECT r.id, r.relationship_type, r.relationship_quality, 
                           r.description, r.notable_conflicts, r.notable_alliances,
                           r.trade_relations, r.cultural_exchanges,
                           CASE
                               WHEN r.nation1_id = $1 THEN r.nation2_id
                               ELSE r.nation1_id
                           END AS other_nation_id
                    FROM InternationalRelations r
                    WHERE r.nation1_id = $1 OR r.nation2_id = $1
                """, nation_id)
                
                # Get names for the related nations
                result = []
                for relation in relations:
                    other_id = relation['other_nation_id']
                    other_nation = await conn.fetchrow("""
                        SELECT name, government_type
                        FROM Nations
                        WHERE id = $1
                    """, other_id)
                    
                    if other_nation:
                        relation_dict = dict(relation)
                        relation_dict['other_nation_name'] = other_nation['name']
                        relation_dict['other_nation_government'] = other_nation['government_type']
                        del relation_dict['other_nation_id']
                        result.append(relation_dict)
                
                return result
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_world_nations",
        action_description="Generating nations for the world",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def generate_world_nations(self, ctx) -> List[Dict[str, Any]]:
        """
        Generate a set of nations for the world, ensuring a mix of matriarchal levels
        
        Returns:
            List of generated nations
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Get world lore for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Try to get world history for context
                world_history = await conn.fetchval("""
                    SELECT description FROM WorldLore
                    WHERE category = 'world_history'
                    LIMIT 1
                """)
                
                social_structure = await conn.fetchval("""
                    SELECT description FROM WorldLore
                    WHERE category = 'social_structure'
                    LIMIT 1
                """)
                
                context = f"World History: {world_history[:300] if world_history else 'Not available'}\n\n"
                context += f"Social Structure: {social_structure[:300] if social_structure else 'Not available'}"
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 5-7 nations for a fantasy world with varying levels of matriarchal governance.
        
        WORLD CONTEXT:
        {context}
        
        Create a politically diverse world with nations that:
        1. Include at least one strongly matriarchal society (player's home nation)
        2. Include nations with varying degrees of matriarchal governance
        3. Show realistic geopolitical dynamics including alliances and rivalries
        4. Have distinct cultural identities and government structures
        5. Vary in size, power, and resources
        
        Format your response as a JSON array where each object has:
        - "name": The name of the nation
        - "government_type": Type of government (oligarchy, monarchy, etc.)
        - "description": A detailed description
        - "relative_power": Power level (1-10)
        - "matriarchy_level": How matriarchal (1-10, with 10 being totally female dominated)
        - "population_scale": Scale of population (small, medium, large, vast)
        - "major_resources": Array of key resources
        - "major_cities": Array of key cities/settlements
        - "cultural_traits": Array of defining cultural traits
        - "notable_features": Other notable features
        - "neighboring_nations": Array of nations that border this one (use the same names you create)
        
        NOTE: Make sure the player's home nation has a matriarchy_level of at least 8.
        """
        
        # Create an agent for nation generation
        nation_agent = Agent(
            name="GeopoliticalAgent",
            instructions="You create nations and geopolitical systems for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(nation_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            nations = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(nations, list):
                if isinstance(nations, dict):
                    nations = [nations]
                else:
                    nations = []
            
            # Ensure player home nation is highly matriarchal
            home_nation_exists = False
            for nation in nations:
                if nation.get('matriarchy_level', 0) >= 8:
                    home_nation_exists = True
                    break
                    
            if not home_nation_exists and nations:
                # Make the first nation highly matriarchal
                nations[0]['matriarchy_level'] = 9
                nations[0]['description'] = f"The player's home nation. {nations[0].get('description', '')}"
            
            # Store each nation
            saved_nations = []
            nation_name_to_id = {}
            
            for nation in nations:
                # Extract nation details
                name = nation.get('name')
                government_type = nation.get('government_type', 'unknown')
                description = nation.get('description')
                relative_power = nation.get('relative_power', 5)
                matriarchy_level = nation.get('matriarchy_level', 5)
                population_scale = nation.get('population_scale')
                major_resources = nation.get('major_resources', [])
                major_cities = nation.get('major_cities', [])
                cultural_traits = nation.get('cultural_traits', [])
                notable_features = nation.get('notable_features')
                neighboring_nations = nation.get('neighboring_nations', [])
                
                if not name or not description:
                    continue
                
                # Save the nation
                try:
                    nation_id = await self.add_nation(
                        run_ctx,
                        name=name,
                        government_type=government_type,
                        description=description,
                        relative_power=relative_power,
                        matriarchy_level=matriarchy_level,
                        population_scale=population_scale,
                        major_resources=major_resources,
                        major_cities=major_cities,
                        cultural_traits=cultural_traits,
                        notable_features=notable_features,
                        neighboring_nations=neighboring_nations
                    )
                    
                    # Add to results
                    nation['id'] = nation_id
                    saved_nations.append(nation)
                    nation_name_to_id[name] = nation_id
                except Exception as e:
                    logging.error(f"Error saving nation '{name}': {e}")
            
            # Now generate international relations
            for nation in saved_nations:
                nation_id = nation['id']
                name = nation['name']
                
                for neighbor_name in nation.get('neighboring_nations', []):
                    if neighbor_name in nation_name_to_id:
                        neighbor_id = nation_name_to_id[neighbor_name]
                        
                        # Only generate relation if neighbor_id > nation_id to avoid duplicates
                        if neighbor_id > nation_id:
                            # Determine relationship based on matriarchy levels
                            matriarchy_diff = abs(nation.get('matriarchy_level', 5) - 
                                                 next((n.get('matriarchy_level', 5) for n in saved_nations if n['name'] == neighbor_name), 5))
                            
                            # Greater difference means more tension
                            relationship_quality = max(1, 10 - matriarchy_diff)
                            
                            if relationship_quality >= 7:
                                relationship_type = "ally"
                                description = f"{name} and {neighbor_name} maintain friendly diplomatic relations."
                            elif relationship_quality >= 4:
                                relationship_type = "neutral"
                                description = f"{name} and {neighbor_name} maintain cautious but functional diplomatic relations."
                            else:
                                relationship_type = "rival"
                                description = f"{name} and {neighbor_name} have tense and occasionally hostile relations."
                            
                            # Create the relation
                            try:
                                await self.add_international_relation(
                                    run_ctx,
                                    nation1_id=nation_id,
                                    nation2_id=neighbor_id,
                                    relationship_type=relationship_type,
                                    relationship_quality=relationship_quality,
                                    description=description
                                )
                            except Exception as e:
                                logging.error(f"Error saving relation between {name} and {neighbor_name}: {e}")
            
            return saved_nations
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for nations: {response_text}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_international_relations",
        action_description="Generating detailed international relations",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def generate_international_relations(self, ctx) -> List[Dict[str, Any]]:
        """
        Generate detailed international relations between existing nations
        
        Returns:
            List of generated international relations
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Get existing nations
        nations = await self.get_all_nations(run_ctx)
        
        if len(nations) < 2:
            return []
        
        # Create nation pairs to generate relations for
        nation_pairs = []
        for i, nation1 in enumerate(nations):
            for nation2 in nations[i+1:]:
                # Check if they're neighbors
                is_neighbor = (nation2['name'] in nation1.get('neighboring_nations', []) or
                              nation1['name'] in nation2.get('neighboring_nations', []))
                
                # Add the pair
                nation_pairs.append({
                    "nation1_id": nation1['id'],
                    "nation1_name": nation1['name'],
                    "nation1_govt": nation1['government_type'],
                    "nation1_power": nation1['relative_power'],
                    "nation1_matriarchy": nation1['matriarchy_level'],
                    "nation2_id": nation2['id'],
                    "nation2_name": nation2['name'],
                    "nation2_govt": nation2['government_type'],
                    "nation2_power": nation2['relative_power'],
                    "nation2_matriarchy": nation2['matriarchy_level'],
                    "is_neighbor": is_neighbor
                })
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate detailed international relations between these {len(nation_pairs)} pairs of nations:
        
        {json.dumps(nation_pairs, indent=2)}
        
        For each pair, create a realistic international relationship that:
        1. Considers their relative matriarchy levels (more different = more tension)
        2. Considers their relative power levels (power imbalance = different dynamics)
        3. Creates specific historical events (conflicts/alliances) between them
        4. Describes trade and cultural exchange patterns
        
        Format your response as a JSON array where each object has:
        - "nation1_id": First nation's ID (use the provided values exactly)
        - "nation2_id": Second nation's ID (use the provided values exactly)
        - "relationship_type": Type of relationship (ally, rival, neutral, vassal, etc.)
        - "relationship_quality": Quality level (1-10, with 10 being extremely positive)
        - "description": Detailed description of relationship dynamics
        - "notable_conflicts": Array of 0-2 notable conflicts (if any)
        - "notable_alliances": Array of 0-2 notable alliances or treaties (if any)
        - "trade_relations": Description of trade patterns
        - "cultural_exchanges": Description of cultural influences
        """
        
        # Create an agent for relation generation
        relation_agent = Agent(
            name="GeopoliticalRelationsAgent",
            instructions="You create detailed international relations for fictional worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(relation_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            relations = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(relations, list):
                if isinstance(relations, dict):
                    relations = [relations]
                else:
                    relations = []
            
            # Store each relation
            saved_relations = []
            for relation in relations:
                # Extract relation details
                nation1_id = relation.get('nation1_id')
                nation2_id = relation.get('nation2_id')
                relationship_type = relation.get('relationship_type', 'neutral')
                relationship_quality = relation.get('relationship_quality', 5)
                description = relation.get('description', 'Relations between these nations.')
                notable_conflicts = relation.get('notable_conflicts', [])
                notable_alliances = relation.get('notable_alliances', [])
                trade_relations = relation.get('trade_relations')
                cultural_exchanges = relation.get('cultural_exchanges')
                
                if not nation1_id or not nation2_id or not description:
                    continue
                
                # Save the relation
                try:
                    relation_id = await self.add_international_relation(
                        run_ctx,
                        nation1_id=nation1_id,
                        nation2_id=nation2_id,
                        relationship_type=relationship_type,
                        relationship_quality=relationship_quality,
                        description=description,
                        notable_conflicts=notable_conflicts,
                        notable_alliances=notable_alliances,
                        trade_relations=trade_relations,
                        cultural_exchanges=cultural_exchanges
                    )
                    
                    # Add to results
                    relation['id'] = relation_id
                    saved_relations.append(relation)
                except Exception as e:
                    logging.error(f"Error saving international relation: {e}")
            
            return saved_relations
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for international relations: {response_text}")
            return []

# 5. Missing methods from LoreEvolutionSystem
# Add these to the LoreEvolutionSystem class in the consolidated file

# Methods that handle lore evolution over time
async def _evolve_urban_myths(self) -> List[Dict[str, Any]]:
    """
    Evolve urban myths over time - they grow, change, or fade
    
    Returns:
        List of changes to urban myths
    """
    changes = []
    
    # Choose a random sample of urban myths to evolve
    async with self.lore_manager.get_connection_pool() as pool:
        async with pool.acquire() as conn:
            # Check if we have a dedicated urban myths table
            has_urban_myths_table = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'urbanmyths'
                );
            """)
            
            if has_urban_myths_table:
                # Get myths from dedicated table
                rows = await conn.fetch("""
                    SELECT id, name, description, believability, spread_rate
                    FROM UrbanMyths
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
            else:
                # Fall back to world lore with urban_myth category
                rows = await conn.fetch("""
                    SELECT id, name, description
                    FROM WorldLore
                    WHERE category = 'urban_myth'
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
            
            for row in rows:
                myth_id = row['id']
                myth_name = row['name']
                old_description = row['description']
                
                # Determine change type
                change_type = random.choice(["grow", "evolve", "fade"])
                
                # Customize change based on type
                if change_type == "grow":
                    # Myth becomes more widespread or elaborate
                    new_believability = min(10, row.get('believability', 5) + random.randint(1, 2))
                    new_spread = min(10, row.get('spread_rate', 5) + random.randint(1, 2))
                    
                    # Use LLM to expand the myth
                    run_ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    prompt = f"""
                    This urban myth is growing in popularity and becoming more elaborate:
                    
                    MYTH: {myth_name}
                    CURRENT DESCRIPTION: {old_description}
                    
                    Expand this myth to include new details, aspects or variations that have emerged.
                    The myth should become more widespread and elaborate, but maintain its core essence.
                    
                    Return only the new expanded description, written in the same style as the original.
                    """
                    
                    myth_agent = Agent(
                        name="MythEvolutionAgent",
                        instructions="You develop and evolve urban myths over time.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(myth_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    change_description = "Myth has grown in popularity and become more elaborate."
                    
                elif change_type == "evolve":
                    # Myth changes in some significant way
                    # Use LLM to transform the myth
                    run_ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    prompt = f"""
                    This urban myth is evolving with new variations:
                    
                    MYTH: {myth_name}
                    CURRENT DESCRIPTION: {old_description}
                    
                    Evolve this myth by changing some details while keeping its core recognizable.
                    Perhaps add a twist, change the outcome, or incorporate a new element.
                    
                    Return only the new evolved description, written in the same style as the original.
                    """
                    
                    myth_agent = Agent(
                        name="MythEvolutionAgent",
                        instructions="You develop and evolve urban myths over time.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(myth_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    change_description = "Myth has evolved with new variations or interpretations."
                    
                else:  # fade
                    # Myth becomes less prominent or believable
                    new_believability = max(1, row.get('believability', 5) - random.randint(1, 2))
                    new_spread = max(1, row.get('spread_rate', 5) - random.randint(1, 2))
                    
                    # Use LLM to show the myth fading
                    run_ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    prompt = f"""
                    This urban myth is fading from public consciousness:
                    
                    MYTH: {myth_name}
                    CURRENT DESCRIPTION: {old_description}
                    
                    Modify this description to indicate the myth is becoming less believed or remembered.
                    Add phrases like "once widely believed" or "few now remember" to show its decline.
                    
                    Return only the new faded description, written in the same style as the original.
                    """
                    
                    myth_agent = Agent(
                        name="MythEvolutionAgent",
                        instructions="You develop and evolve urban myths over time.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(myth_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    change_description = "Myth is fading from public consciousness or becoming less believed."
                
                # Apply the update
                try:
                    if has_urban_myths_table:
                        # Update in dedicated table
                        await conn.execute("""
                            UPDATE UrbanMyths
                            SET description = $1,
                                believability = $2,
                                spread_rate = $3
                            WHERE id = $4
                        """, new_description, new_believability, new_spread, myth_id)
                    else:
                        # Update in WorldLore
                        # Generate new embedding
                        embedding_text = f"{myth_name} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        await conn.execute("""
                            UPDATE WorldLore
                            SET description = $1, embedding = $2
                            WHERE id = $3
                        """, new_description, new_embedding, myth_id)
                    
                    # Record the change
                    changes.append({
                        "myth_id": myth_id,
                        "name": myth_name,
                        "change_type": change_type,
                        "old_description": old_description,
                        "new_description": new_description,
                        "change_description": change_description
                    })
                    
                    # Clear cache
                    if has_urban_myths_table:
                        LORE_CACHE.invalidate_pattern(f"UrbanMyths_{myth_id}")
                    else:
                        LORE_CACHE.invalidate_pattern(f"WorldLore_{myth_id}")
                except Exception as e:
                    logging.error(f"Error updating myth {myth_id}: {e}")
    
    return changes

async def _develop_cultural_elements(self) -> List[Dict[str, Any]]:
    """
    Develop cultural elements over time - they formalize, adapt, or merge
    
    Returns:
        List of changes to cultural elements
    """
    changes = []
    
    # Choose a random sample of cultural elements to develop
    async with self.lore_manager.get_connection_pool() as pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, name, type, description, significance, practiced_by
                FROM CulturalElements
                ORDER BY RANDOM()
                LIMIT 2
            """)
            
            for row in rows:
                element_id = row['id']
                element_name = row['name']
                element_type = row['type']
                old_description = row['description']
                old_significance = row['significance']
                practiced_by = row['practiced_by'] if row['practiced_by'] else []
                
                # Determine change type
                change_type = random.choice(["formalize", "adapt", "spread", "codify"])
                
                # Create the run context
                run_ctx = RunContextWrapper(context={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                })
                
                # Customize change based on type
                if change_type == "formalize":
                    # Cultural element becomes more formalized
                    prompt = f"""
                    This cultural element is becoming more formalized and structured:
                    
                    ELEMENT: {element_name} ({element_type})
                    CURRENT DESCRIPTION: {old_description}
                    PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                    
                    Update this description to show how this cultural element has become more formalized.
                    Add details about specific rules, structures, or ceremonies that have developed.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    new_significance = min(10, old_significance + 1)
                    change_description = "Element has become more formal and structured."
                    
                elif change_type == "adapt":
                    # Cultural element adapts to changing circumstances
                    prompt = f"""
                    This cultural element is adapting to changing social circumstances:
                    
                    ELEMENT: {element_name} ({element_type})
                    CURRENT DESCRIPTION: {old_description}
                    PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                    
                    Update this description to show how this cultural element has adapted to changing times.
                    Show how it maintains its essence while evolving to remain relevant.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    new_significance = old_significance  # Remains the same
                    change_description = "Element has adapted to changing social circumstances."
                    
                elif change_type == "spread":
                    # Cultural element spreads to new groups
                    new_groups = ["neighboring regions", "younger generations", "urban centers", "rural communities"]
                    added_groups = random.sample(new_groups, 1)
                    
                    # Don't add groups that are already listed
                    if practiced_by:
                        added_groups = [g for g in added_groups if g not in practiced_by]
                    
                    prompt = f"""
                    This cultural element is spreading to new groups, specifically: {', '.join(added_groups)}
                    
                    ELEMENT: {element_name} ({element_type})
                    CURRENT DESCRIPTION: {old_description}
                    CURRENTLY PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                    
                    Update this description to show how this cultural element has spread to these new groups.
                    How might they adapt or interpret it slightly differently?
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    new_significance = min(10, old_significance + 1)
                    change_description = f"Element has spread to new groups: {', '.join(added_groups)}"
                    
                    # Update the practiced_by list
                    if practiced_by:
                        practiced_by.extend(added_groups)
                    else:
                        practiced_by = added_groups
                        
                else:  # codify
                    # Cultural element becomes codified in writing or law
                    prompt = f"""
                    This cultural element is becoming codified into formal rules, laws, or texts:
                    
                    ELEMENT: {element_name} ({element_type})
                    CURRENT DESCRIPTION: {old_description}
                    PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                    
                    Update this description to show how this practice has become codified.
                    Add details about written texts, legal codes, or formal teachings that preserve it.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    new_significance = min(10, old_significance + 2)
                    change_description = "Element has been codified into formal rules, laws, or texts."
                
                # Get updated description from LLM
                culture_agent = Agent(
                    name="CultureEvolutionAgent",
                    instructions="You develop and evolve cultural elements over time.",
                    model="o3-mini"
                )
                
                result = await Runner.run(culture_agent, prompt, context=run_ctx.context)
                new_description = result.final_output
                
                # Apply the update
                try:
                    # Generate new embedding
                    embedding_text = f"{element_name} {element_type} {new_description}"
                    new_embedding = await generate_embedding(embedding_text)
                    
                    await conn.execute("""
                        UPDATE CulturalElements
                        SET description = $1, 
                            significance = $2, 
                            practiced_by = $3,
                            embedding = $4
                        WHERE id = $5
                    """, new_description, new_significance, practiced_by, new_embedding, element_id)
                    
                    # Record the change
                    changes.append({
                        "element_id": element_id,
                        "name": element_name,
                        "type": element_type,
                        "change_type": change_type,
                        "old_description": old_description,
                        "new_description": new_description,
                        "change_description": change_description
                    })
                    
                    # Clear cache
                    LORE_CACHE.invalidate_pattern(f"CulturalElements_{element_id}")
                except Exception as e:
                    logging.error(f"Error updating cultural element {element_id}: {e}")
    
    return changes

async def _shift_geopolitical_landscape(self) -> List[Dict[str, Any]]:
    """
    Apply subtle shifts to the geopolitical landscape
    
    Returns:
        List of geopolitical changes
    """
    changes = []
    
    # Load some faction data to work with
    async with self.lore_manager.get_connection_pool() as pool:
        async with pool.acquire() as conn:
            # Get factions
            factions = await conn.fetch("""
                SELECT id, name, type, description, territory, rivals, allies
                FROM Factions
                ORDER BY RANDOM()
                LIMIT 3
            """)
            
            if not factions:
                return changes
                
            # Get a random region if we have them
            regions = await conn.fetch("""
                SELECT id, name, description, governing_faction
                FROM GeographicRegions
                ORDER BY RANDOM()
                LIMIT 1
            """)
            
            region = regions[0] if regions else None
            
            # Determine what kind of shift to generate
            shift_types = ["alliance_change", "territory_dispute", "influence_growth"]
            
            # Add "regional_governance" if we have a region
            if region:
                shift_types.append("regional_governance")
                
            # Pick a random shift type
            shift_type = random.choice(shift_types)
            
            # Create the run context
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Implement each type of geopolitical shift
            # Implementation omitted for brevity as this is quite lengthy
            # The full implementation would include creating prompts and handling
            # the different shift types: alliance_change, territory_dispute, 
            # regional_governance, and influence_growth
    
    return changes

async def _evolve_notable_figures(self) -> List[Dict[str, Any]]:
    """
    Evolve reputations and stories of notable figures
    
    Returns:
        List of changes to notable figures
    """
    changes = []
    
    # Check if we have a dedicated notable figures table
    async with self.lore_manager.get_connection_pool() as pool:
        async with pool.acquire() as conn:
            has_notable_figures_table = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'notablefigures'
                );
            """)
            
            if has_notable_figures_table:
                # Get figures from dedicated table
                rows = await conn.fetch("""
                    SELECT id, name, description, significance, reputation
                    FROM NotableFigures
                    ORDER BY RANDOM()
                    LIMIT 2
                """)
            else:
                # Fall back to world lore with notable_figure category
                rows = await conn.fetch("""
                    SELECT id, name, description, significance
                    FROM WorldLore
                    WHERE category = 'notable_figure'
                    ORDER BY RANDOM()
                    LIMIT 2
                """)
            
            for row in rows:
                figure_id = row['id']
                figure_name = row['name']
                old_description = row['description']
                old_significance = row['significance'] if 'significance' in row else 5
                
                # Determine change type
                change_options = ["reputation_rise", "reputation_fall", "scandal", "achievement", "reform"]
                
                # If we have reputation field, use it to influence likely changes
                reputation = row.get('reputation', 5)
                if reputation and reputation >= 7:
                    # High reputation more likely to fall or have scandal
                    weights = [1, 3, 2, 1, 1]
                elif reputation and reputation <= 3:
                    # Low reputation more likely to rise or reform
                    weights = [3, 1, 1, 2, 2]
                else:
                    # Mid reputation equal chances
                    weights = [1, 1, 1, 1, 1]
                
                # Weighted choice
                change_type = random.choices(
                    change_options, 
                    weights=weights,
                    k=1
                )[0]
                
                # Create the run context
                run_ctx = RunContextWrapper(context={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                })
                
                # Setup for each change type
                if change_type == "reputation_rise":
                    prompt = f"""
                    This notable figure is experiencing a rise in reputation and influence:
                    
                    FIGURE: {figure_name}
                    Current Description: {old_description}
                    
                    Update the description to show how this figure's reputation has improved.
                    Include their recent positive actions, public perception changes, or new supporters.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    new_reputation = min(10, (reputation if reputation else 5) + random.randint(1, 2))
                    reputation_change = "improved"
                    
                elif change_type == "reputation_fall":
                    prompt = f"""
                    This notable figure is experiencing a decline in reputation and influence:
                    
                    FIGURE: {figure_name}
                    Current Description: {old_description}
                    
                    Update the description to show how this figure's reputation has declined.
                    Include their recent missteps, public perception changes, or lost support.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    new_reputation = max(1, (reputation if reputation else 5) - random.randint(1, 2))
                    reputation_change = "declined"
                    
                elif change_type == "scandal":
                    prompt = f"""
                    This notable figure is involved in a recent scandal:
                    
                    FIGURE: {figure_name}
                    Current Description: {old_description}
                    
                    Update the description to incorporate a scandal this figure is facing.
                    Include the nature of the scandal and how they're responding to it.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    new_reputation = max(1, (reputation if reputation else 5) - random.randint(2, 3))
                    reputation_change = "significantly damaged"
                    
                elif change_type == "achievement":
                    prompt = f"""
                    This notable figure has recently achieved something significant:
                    
                    FIGURE: {figure_name}
                    Current Description: {old_description}
                    
                    Update the description to incorporate a major achievement by this figure.
                    Include what they accomplished and how it has affected their standing.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    new_reputation = min(10, (reputation if reputation else 5) + random.randint(2, 3))
                    reputation_change = "significantly improved"
                    
                else:  # reform
                    prompt = f"""
                    This notable figure has undergone a personal or political reform:
                    
                    FIGURE: {figure_name}
                    Current Description: {old_description}
                    
                    Update the description to show how this figure has changed their stance, approach, or values.
                    Include what prompted this change and how others have reacted to it.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    # Reform could go either way
                    direction = random.choice([-1, 1])
                    new_reputation = max(1, min(10, (reputation if reputation else 5) + direction * random.randint(1, 2)))
                    reputation_change = "transformed"
                
                # Get updated description from LLM
                figure_agent = Agent(
                    name="NotableFigureAgent",
                    instructions="You evolve the stories and reputations of notable figures.",
                    model="o3-mini"
                )
                
                result = await Runner.run(figure_agent, prompt, context=run_ctx.context)
                new_description = result.final_output
                
                # Apply the update
                try:
                    # Generate new embedding
                    embedding_text = f"{figure_name} {new_description}"
                    new_embedding = await generate_embedding(embedding_text)
                    
                    if has_notable_figures_table:
                        # Update in dedicated table
                        await conn.execute("""
                            UPDATE NotableFigures
                            SET description = $1,
                                reputation = $2,
                                embedding = $3
                            WHERE id = $4
                        """, new_description, new_reputation, new_embedding, figure_id)
                    else:
                        # Update in WorldLore
                        await conn.execute("""
                            UPDATE WorldLore
                            SET description = $1,
                                embedding = $2
                            WHERE id = $3
                        """, new_description, new_embedding, figure_id)
                    
                    # Record the change
                    changes.append({
                        "figure_id": figure_id,
                        "name": figure_name,
                        "change_type": change_type,
                        "old_description": old_description,
                        "new_description": new_description,
                        "reputation_change": reputation_change
                    })
                    
                    # Clear cache
                    if has_notable_figures_table:
                        LORE_CACHE.invalidate_pattern(f"NotableFigures_{figure_id}")
                    else:
                        LORE_CACHE.invalidate_pattern(f"WorldLore_{figure_id}")
                except Exception as e:
                    logging.error(f"Error updating notable figure {figure_id}: {e}")
    
    return changes

# 6. Missing methods from EmergentLoreSystem
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_additional_nation",
    action_description="Generating additional nation for the world",
    id_from_context=lambda ctx: "emergent_lore"
)
async def generate_additional_nation(self, ctx) -> Dict[str, Any]:
    """
    Generate an additional nation with governance oversight.
    
    Returns:
        New nation details
    """
    # Create the run context
    run_ctx = RunContextWrapper(context=ctx.context)
    
    # Get existing nations for context
    async with self.lore_manager.get_connection_pool() as pool:
        async with pool.acquire() as conn:
            # Check how many nations we already have
            nation_count = await conn.fetchval("""
                SELECT COUNT(*) FROM Nations
            """)
            
            # Get existing nations
            existing_nations = await conn.fetch("""
                SELECT name, government_type, matriarchy_level, neighboring_nations
                FROM Nations
                LIMIT 10
            """)
            
            # Get pantheons for religious context
            pantheons = await conn.fetch("""
                SELECT name, description
                FROM Pantheons
                LIMIT 3
            """)
            
            # Get some cultural elements for context
            cultural_elements = await conn.fetch("""
                SELECT name, type, description
                FROM CulturalElements
                ORDER BY RANDOM()
                LIMIT 5
            """)
            
            # Convert to lists
            nation_data = [dict(nation) for nation in existing_nations]
            pantheon_data = [dict(pantheon) for pantheon in pantheons]
            cultural_data = [dict(element) for element in cultural_elements]
    
    # Determine if this should be a matriarchal or non-matriarchal nation
    # Keep a balance in the world
    matriarchal_nations = [n for n in nation_data if n.get("matriarchy_level", 0) >= 7]
    non_matriarchal_nations = [n for n in nation_data if n.get("matriarchy_level", 0) <= 3]
    
    # Default to a medium matriarchy level
    target_matriarchy_level = random.randint(4, 7)
    
    # If we have more matriarchal nations, make this one less matriarchal
    if len(matriarchal_nations) > len(non_matriarchal_nations) + 1:
        target_matriarchy_level = random.randint(1, 4)
    # If we have more non-matriarchal nations, make this one more matriarchal
    elif len(non_matriarchal_nations) > len(matriarchal_nations):
        target_matriarchy_level = random.randint(7, 10)
    
    # Create a prompt for the LLM
    prompt = f"""
    Generate a new nation for a fantasy world with existing nations:
    
    EXISTING NATIONS:
    {json.dumps(nation_data, indent=2)}
    
    RELIGIOUS CONTEXT:
    {json.dumps(pantheon_data, indent=2)}
    
    CULTURAL CONTEXT:
    {json.dumps(cultural_data, indent=2)}
    
    Create a nation that:
    1. Has a matriarchy level of approximately {target_matriarchy_level}/10
    2. Is distinct from existing nations
    3. Has logical geographic connections to some existing nations
    4. Has rich cultural and political details
    
    Return a JSON object with:
    - name: Name of the nation
    - government_type: Type of government
    - description: Detailed description
    - relative_power: Power level (1-10)
    - matriarchy_level: How matriarchal (1-10, with 10 being totally female dominated)
    - population_scale: Scale of population (small, medium, large, vast)
    - major_resources: Array of key resources
    - major_cities: Array of key cities/settlements
    - cultural_traits: Array of defining cultural traits
    - notable_features: Other notable features
    - neighboring_nations: Array of nations that border this one (use exact names from the existing nations list)
    """
    
    # Create an agent for nation generation
    nation_agent = Agent(
        name="NationGenerationAgent",
        instructions="You create nations for fantasy worlds.",
        model="o3-mini"
    )
    
    # Get the response
    result = await Runner.run(nation_agent, prompt, context=run_ctx.context)
    response_text = result.final_output
    
    try:
        # Parse the JSON response
        nation_data = await self.parse_llm_json_response(response_text)
        
        # Create the nation in the geopolitical system
        geopolitical_system = GeopoliticalSystemManager(self.user_id, self.conversation_id)
        
        # Add the nation
        nation_id = await geopolitical_system.add_nation(
            run_ctx,
            name=nation_data.get("name", "Unnamed Nation"),
            government_type=nation_data.get("government_type", "monarchy"),
            description=nation_data.get("description", ""),
            relative_power=nation_data.get("relative_power", 5),
            matriarchy_level=nation_data.get("matriarchy_level", target_matriarchy_level),
            population_scale=nation_data.get("population_scale", "medium"),
            major_resources=nation_data.get("major_resources", []),
            major_cities=nation_data.get("major_cities", []),
            cultural_traits=nation_data.get("cultural_traits", []),
            notable_features=nation_data.get("notable_features", ""),
            neighboring_nations=nation_data.get("neighboring_nations", [])
        )
        
        # Update the nation data with the ID
        nation_data["id"] = nation_id
        
        # Create international relations
        neighboring_nations = nation_data.get("neighboring_nations", [])
        if neighboring_nations:
            # Get IDs of neighboring nations
            async with self.lore_manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    for neighbor_name in neighboring_nations:
                        neighbor_id = await conn.fetchval("""
                            SELECT id FROM Nations
                            WHERE name = $1
                        """, neighbor_name)
                        
                        if neighbor_id:
                            # Determine relationship type based on matriarchy level difference
                            neighbor_matriarchy = await conn.fetchval("""
                                SELECT matriarchy_level FROM Nations
                                WHERE id = $1
                            """, neighbor_id)
                            
                            matriarchy_diff = abs(nation_data.get("matriarchy_level", 5) - (neighbor_matriarchy or 5))
                            
                            # Greater difference means more tension
                            relationship_quality = max(1, 10 - matriarchy_diff)
                            
                            if relationship_quality >= 7:
                                relationship_type = "ally"
                                description = f"{nation_data.get('name')} and {neighbor_name} maintain friendly diplomatic relations."
                            elif relationship_quality >= 4:
                                relationship_type = "neutral"
                                description = f"{nation_data.get('name')} and {neighbor_name} maintain cautious but functional diplomatic relations."
                            else:
                                relationship_type = "rival"
                                description = f"{nation_data.get('name')} and {neighbor_name} have tense and occasionally hostile relations."
                            
                            # Create the relation
                            await geopolitical_system.add_international_relation(
                                run_ctx,
                                nation1_id=nation_id,
                                nation2_id=neighbor_id,
                                relationship_type=relationship_type,
                                relationship_quality=relationship_quality,
                                description=description
                            )
        
        return nation_data
        
    except Exception as e:
        logging.error(f"Error generating additional nation: {e}")
        return {"error": str(e)}

@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_historical_figure",
    action_description="Generating historical figure for the world",
    id_from_context=lambda ctx: "emergent_lore"
)
async def generate_historical_figure(self, ctx) -> Dict[str, Any]:
    """
    Generate a historical figure with governance oversight.
    
    Returns:
        Historical figure details
    """
    # Create the run context
    run_ctx = RunContextWrapper(context=ctx.context)
    
    # Get context for figure generation
    async with self.lore_manager.get_connection_pool() as pool:
        async with pool.acquire() as conn:
            # Get some historical events for context
            historical_events = await conn.fetch("""
                SELECT name, description, date_description
                FROM HistoricalEvents
                ORDER BY RANDOM()
                LIMIT 3
            """)
            
            # Get some nations for context
            nations = await conn.fetch("""
                SELECT name, government_type, matriarchy_level
                FROM Nations
                ORDER BY RANDOM()
                LIMIT 3
            """)
            
            # Get some factions for context
            factions = await conn.fetch("""
                SELECT name, type, description
                FROM Factions
                ORDER BY RANDOM()
                LIMIT 3
            """)
            
            # Convert to lists
            event_data = [dict(event) for event in historical_events]
            nation_data = [dict(nation) for nation in nations]
            faction_data = [dict(faction) for faction in factions]
    
    # Determine gender distribution
    # In a matriarchal world, more historical figures should be female
    genders = ["female", "female", "female", "male", "non-binary"]
    figure_gender = random.choice(genders)
    
    # Adjust prompt based on gender
    gender_context = ""
    if figure_gender == "female":
        gender_context = "Create a female historical figure who exemplifies feminine power and authority."
    elif figure_gender == "male":
        gender_context = "Create a male historical figure who represents an unusual or noteworthy role for men in matriarchal society."
    else:  # non-binary
        gender_context = "Create a non-binary historical figure who carved a unique path in the gendered power structures."
    
    # Create a prompt for the LLM
    prompt = f"""
    Generate a historical figure for a matriarchal fantasy world:
    
    HISTORICAL EVENTS:
    {json.dumps(event_data, indent=2)}
    
    NATIONS:
    {json.dumps(nation_data, indent=2)}
    
    FACTIONS:
    {json.dumps(faction_data, indent=2)}
    
    {gender_context}
    
    Create a detailed historical figure that:
    1. Had significant impact on the world
    2. Has a compelling personal story
    3. Is connected to existing historical events or factions when appropriate
    4. Reflects the matriarchal power dynamics of the world
    
    Return a JSON object with:
    - name: Full name of the figure
    - gender: Gender (female, male, non-binary)
    - title: Primary title or position
    - birth_date: Approximate birth date or period
    - death_date: Approximate death date or period (if deceased)
    - is_alive: Whether they're still alive
    - nationality: Nation of origin
    - affiliations: Array of factions or groups they were affiliated with
    - description: Detailed personal description
    - appearance: Physical appearance
    - personality: Key personality traits
    - accomplishments: Array of major accomplishments
    - legacy: Lasting impact on the world
    - relationships: Key relationships with other figures or groups
    - controversy: Any controversial aspects
    - historical_events: Array of historical events they participated in
    """
    
    # Create an agent for historical figure generation
    figure_agent = Agent(
        name="HistoricalFigureAgent",
        instructions="You create detailed historical figures for fantasy worlds.",
        model="o3-mini"
    )
    
    # Get the response
    result = await Runner.run(figure_agent, prompt, context=run_ctx.context)
    response_text = result.final_output
    
    try:
        # Parse the JSON response
        figure_data = await self.parse_llm_json_response(response_text)
        
        # Create a world lore entry for this figure
        name = figure_data.get("name", "Unnamed Figure")
        title = figure_data.get("title", "")
        description = figure_data.get("description", "")
        
        full_description = f"{description}\n\n"
        
        if "accomplishments" in figure_data:
            accomplishments = figure_data.get("accomplishments", [])
            if isinstance(accomplishments, list):
                full_description += f"Accomplishments: {', '.join(accomplishments)}\n\n"
            else:
                full_description += f"Accomplishments: {accomplishments}\n\n"
                
        if "legacy" in figure_data:
            full_description += f"Legacy: {figure_data.get('legacy')}\n\n"
            
        if "controversy" in figure_data:
            full_description += f"Controversy: {figure_data.get('controversy')}\n\n"
        
        # Store as WorldLore
        try:
            lore_id = await self.lore_manager.add_world_lore(
                name=f"{name}, {title}",
                category="historical_figure",
                description=full_description,
                significance=8,  # Historical figures are significant
                tags=["historical_figure", "biography", figure_data.get("gender", "unknown")]
            )
            
            figure_data["lore_id"] = lore_id
            figure_data["stored_as_world_lore"] = True
        except Exception as e:
            logging.error(f"Error storing historical figure as world lore: {e}")
            figure_data["storage_error"] = str(e)
        
        return figure_data
        
    except Exception as e:
        logging.error(f"Error generating historical figure: {e}")
        return {"error": str(e)}

@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_cultural_evolution",
    action_description="Generating cultural evolution for the world",
    id_from_context=lambda ctx: "emergent_lore"
)
async def generate_cultural_evolution(self, ctx) -> Dict[str, Any]:
    """
    Generate cultural evolution with governance oversight.
    
    Returns:
        Cultural evolution details
    """
    # Create the run context
    run_ctx = RunContextWrapper(context=ctx.context)
    
    # Get cultural elements for context
    async with self.lore_manager.get_connection_pool() as pool:
        async with pool.acquire() as conn:
            # Get existing cultural elements
            cultural_elements = await conn.fetch("""
                SELECT id, name, type, description, practiced_by
                FROM CulturalElements
                ORDER BY RANDOM()
                LIMIT 5
            """)
            
            # Get nations for context
            nations = await conn.fetch("""
                SELECT name, government_type, matriarchy_level, cultural_traits
                FROM Nations
                ORDER BY RANDOM()
                LIMIT 3
            """)
            
            # Convert to lists
            element_data = [dict(element) for element in cultural_elements]
            nation_data = [dict(nation) for nation in nations]
    
    # Select a random element to evolve
    if not element_data:
        return {"error": "No cultural elements found to evolve"}
        
    target_element = random.choice(element_data)
    
    # Create a prompt for the LLM
    prompt = f"""
    Generate a cultural evolution for this existing cultural element:
    
    TARGET ELEMENT:
    {json.dumps(target_element, indent=2)}
    
    NATIONAL CONTEXT:
    {json.dumps(nation_data, indent=2)}
    
    Create a cultural evolution that:
    1. Shows how the element changes over time
    2. Introduces new variations, interpretations, or practices
    3. Explains the factors driving the change
    4. Maintains connection to matriarchal themes
    
    Return a JSON object with:
    - element_id: ID of the element being evolved (use exact ID from the provided data)
    - element_name: Name of the element (should match provided data)
    - evolution_type: Type of evolution (spread, schism, formalization, adaptation, etc.)
    - description: Detailed description of how it's evolving
    - original_form: Brief description of its original form
    - evolved_form: Detailed description of its new form
    - catalyst: What caused this evolution
    - regions_affected: Where this evolution is occurring
    - resistance: Any resistance to this evolution
    - timeline: How long this evolution has been happening
    """
    
    # Create an agent for cultural evolution
    evolution_agent = Agent(
        name="CulturalEvolutionAgent",
        instructions="You create cultural evolutions for fantasy worlds.",
        model="o3-mini"
    )
    
    # Get the response
    result = await Runner.run(evolution_agent, prompt, context=run_ctx.context)
    response_text = result.final_output
    
    try:
        # Parse the JSON response
        evolution_data = await self.parse_llm_json_response(response_text)
        
        # Update the existing cultural element
        element_id = evolution_data.get("element_id")
        
        if element_id:
            try:
                # Get the original element
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        original = await conn.fetchrow("""
                            SELECT name, description, practiced_by
                            FROM CulturalElements
                            WHERE id = $1
                        """, element_id)
                        
                        if original:
                            # Prepare the updated description
                            evolved_description = evolution_data.get("evolved_form", "")
                            evolution_context = f"\n\nEvolution: {evolution_data.get('description', '')}"
                            if evolution_data.get("catalyst"):
                                evolution_context += f"\n\nCatalyst for Change: {evolution_data.get('catalyst')}"
                                
                            full_description = evolved_description + evolution_context
                            
                            # Update the practiced_by field if needed
                            practiced_by = original["practiced_by"] or []
                            regions_affected = evolution_data.get("regions_affected", [])
                            
                            if isinstance(regions_affected, list):
                                for region in regions_affected:
                                    if region not in practiced_by:
                                        practiced_by.append(region)
                            elif isinstance(regions_affected, str) and regions_affected not in practiced_by:
                                practiced_by.append(regions_affected)
                            
                            # Update the element in the database
                            await conn.execute("""
                                UPDATE CulturalElements
                                SET description = $1, practiced_by = $2
                                WHERE id = $3
                            """, full_description, practiced_by, element_id)
                            
                            evolution_data["update_successful"] = True
                            evolution_data["practiced_by"] = practiced_by
            except Exception as e:
                logging.error(f"Error updating cultural element: {e}")
                evolution_data["update_error"] = str(e)
        
        return evolution_data
        
    except Exception as e:
        logging.error(f"Error generating cultural evolution: {e}")
        return {"error": str(e)}

# These are any additional initialization or utility methods worth including 
# that might have been omitted from LoreEvolutionSystem, FaithSystem, 
# EmergentLoreSystem, or others

# Here are the missing implementations from the FaithSystem class:
# FaithSystem.add_deity, add_pantheon, add_religious_practice, etc.
# These methods would be implemented similarly to the other add_* methods 
# in the classes above


