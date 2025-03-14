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
