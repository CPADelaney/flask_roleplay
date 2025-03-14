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

# Initialize cache for lore items
LORE_CACHE = LoreCache(max_size=1000, ttl=7200)  # 2 hour TTL, larger cache

class MatriarchalPowerStructureFramework:
    """
    Defines core principles for power dynamics in femdom settings,
    ensuring consistency across all generated lore.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
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
        """
        Transform generic social structure description to a matriarchal one
        
        Args:
            social_structure: Original social structure description
            
        Returns:
            Matriarchal social structure description
        """
        # This is a simplified version - in a full implementation, 
        # you would use an LLM to transform the text more intelligently
        
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
        # This would use more sophisticated text transformation in a full implementation
        if "goddess" not in cosmology.lower():
            cosmology = cosmology.replace("god", "goddess")
            cosmology = cosmology.replace("God", "Goddess")
            
        # Add feminine cosmic principle
        feminine_principle = "\n\nAt the heart of all creation is the Feminine Principle, the source of all life and power. "
        feminine_principle += "The cosmos itself is understood as fundamentally feminine in nature, with masculine elements serving and supporting the greater feminine whole."
        
        return cosmology + feminine_principle
    
    def _gender_magic_system(self, magic_system: str) -> str:
        """Apply gendered dynamics to the magic system"""
        # In a full implementation, this would use more sophisticated text transformation
        
        gendered_magic = "\n\nThe flow and expression of magical energies reflect the natural order of feminine dominance. "
        gendered_magic += "Women typically possess greater innate magical potential and authority over the higher mysteries. "
        gendered_magic += "Male practitioners often specialize in supportive, protective, or enhancing magics that complement and serve the more powerful feminine magic traditions. "
        gendered_magic += "The most powerful spells and rituals often require a woman's touch to fully manifest."
        
        return magic_system + gendered_magic
    
    def _matriarchalize_history(self, history: str) -> str:
        """Ensure history reflects matriarchal development"""
        # In a full implementation, this would use more sophisticated text transformation
        
        matriarchal_history = "\n\nThroughout recorded history, women have held the reins of power. "
        matriarchal_history += "Great Empresses, Matriarchs, and female leaders have shaped the course of civilization. "
        matriarchal_history += "While there have been periods of conflict and attempts to upset the natural order, "
        matriarchal_history += "the fundamental principle of feminine authority has remained the consistent foundation of society."
        
        return history + matriarchal_history
    
    def _feminize_calendar(self, calendar_system: str) -> str:
        """Make the calendar system reflect feminine significance"""
        # In a full implementation, this would use more sophisticated text transformation
        
        feminine_calendar = "\n\nThe calendar marks significant events in feminine history, with important dates "
        feminine_calendar += "often corresponding to lunar cycles, feminine deities, or the reigns of great Matriarchs. "
        feminine_calendar += "The timing of festivals and holy days celebrates feminine power and the cycles of life that women control."
        
        return calendar_system + feminine_calendar
    
    def generate_hierarchical_constraints(self) -> Dict[str, Any]:
        """
        Generate consistent rules about power hierarchies.
        
        Returns:
            Dictionary of hierarchical constraints to maintain across lore
        """
        hierarchy_types = self.core_principles["power_dynamics"]["hierarchy_types"]
        chosen_type = random.choice(hierarchy_types)
        
        constraints = {
            "dominant_hierarchy_type": chosen_type,
            "power_expressions": random.sample(self.core_principles["power_dynamics"]["power_expression"], 3),
            "masculine_roles": random.sample(self.core_principles["power_dynamics"]["masculine_roles"], 3),
            "leadership_domains": random.sample(list(self.core_principles["societal_norms"]["female_leadership"]), 3),
            "property_rights": random.sample(list(self.core_principles["societal_norms"]["female_property_rights"]), 2),
            "status_markers": random.sample(list(self.core_principles["societal_norms"]["male_status_markers"]), 3),
            "relationship_structure": random.choice(list(self.core_principles["societal_norms"]["relationship_structures"])),
            "enforcement_mechanisms": random.sample(list(self.core_principles["societal_norms"]["enforcement_mechanisms"]), 2)
        }
        
        if chosen_type == "matrilineal":
            constraints["description"] = "Descent and inheritance pass through the maternal line, with women controlling family resources."
        elif chosen_type == "matrifocal":
            constraints["description"] = "Women are the center of family life and decision-making, with men in peripheral roles."
        elif chosen_type == "matriarchal":
            constraints["description"] = "Women hold formal political, economic, and social power over men in all domains."
        elif chosen_type == "gynocentric":
            constraints["description"] = "Society and culture are centered on feminine needs, values, and perspectives."
        
        return constraints
    
    def generate_power_expressions(self) -> List[Dict[str, Any]]:
        """
        Generate specific expressions of female power and male submission
        
        Returns:
            List of power expression descriptions
        """
        expressions = []
        
        # Political expressions
        expressions.append({
            "domain": "political",
            "title": "Council of Matriarchs",
            "description": "The ruling council composed exclusively of senior women who make all major decisions for the community.",
            "male_role": "Advisors and administrators who carry out the Matriarchs' decisions without question."
        })
        
        # Economic expressions
        expressions.append({
            "domain": "economic",
            "title": "Female Property Ownership",
            "description": "All significant property, businesses, and resources are owned and controlled by women.",
            "male_role": "Men manage resources only as agents of their female relatives or superiors."
        })
        
        # Religious expressions
        expressions.append({
            "domain": "religious",
            "title": "Priestesshood",
            "description": "Religious authority vested in female clergy who interpret the will of the Goddesses.",
            "male_role": "Temple servants who handle mundane tasks and participate in rituals as directed."
        })
        
        # Domestic expressions
        expressions.append({
            "domain": "domestic",
            "title": "Household Governance",
            "description": "Women control the household, making all significant decisions about family life.",
            "male_role": "Men handle domestic labor and childcare under female direction."
        })
        
        # Sexual expressions
        expressions.append({
            "domain": "sexual",
            "title": "Female Sexual Agency",
            "description": "Women determine when, how, and with whom sexual activity occurs.",
            "male_role": "Men's sexuality is considered a resource to be managed and directed by women."
        })
        
        # Military expressions
        expressions.append({
            "domain": "military",
            "title": "Feminine Command",
            "description": "Military leadership is exclusively female, with generals and officers all being women.",
            "male_role": "Men serve as foot soldiers, following orders from their female superiors."
        })
        
        return expressions

class LoreEvolutionSystem:
    """
    System responsible for evolving lore over time, both in response
    to specific events and through natural maturation processes.
    """
    
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
    
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Update world lore based on a significant narrative event
        
        Args:
            event_description: Description of the narrative event
            
        Returns:
            Dictionary with lore updates
        """
        # Check permissions with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="evolve_lore_with_event",
            action_details={"event_description": event_description}
        )
        
        if not permission["approved"]:
            logging.warning(f"Lore evolution not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
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
        """
        Identify lore elements that would be affected by the event
        
        Args:
            event_description: Description of the event
            
        Returns:
            List of affected lore elements with relevance scores
        """
        # Generate embedding for the event
        event_embedding = await generate_embedding(event_description)
        
        # Search all lore types for potentially affected elements
        affected_elements = []
        
        # List of lore types to check
        lore_types = [
            "WorldLore", 
            "Factions", 
            "CulturalElements", 
            "HistoricalEvents", 
            "GeographicRegions", 
            "LocationLore",
            "UrbanMyths",
            "LocalHistories",
            "Landmarks",
            "NotableFigures"
        ]
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for lore_type in lore_types:
                    try:
                        # Check if table exists
                        table_exists = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = '{lore_type.lower()}'
                            );
                        """)
                        
                        if not table_exists:
                            continue
                            
                        # Not all tables may have embedding - handle that case
                        has_embedding = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = '{lore_type.lower()}' AND column_name = 'embedding'
                            );
                        """)
                        
                        if not has_embedding:
                            continue
                        
                        # Perform similarity search
                        id_field = 'id'
                        if lore_type == 'LocationLore':
                            id_field = 'location_id'
                            
                        rows = await conn.fetch(f"""
                            SELECT {id_field} as id, name, description, 
                                   1 - (embedding <=> $1) as relevance
                            FROM {lore_type}
                            WHERE 1 - (embedding <=> $1) > 0.6
                            ORDER BY relevance DESC
                            LIMIT 5
                        """, event_embedding)
                        
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
        
        # Sort by relevance
        affected_elements.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Filter to most relevant (if too many)
        if len(affected_elements) > 15:
            affected_elements = affected_elements[:15]
            
        return affected_elements
    
    async def _generate_lore_updates(
        self, 
        affected_elements: List[Dict[str, Any]], 
        event_description: str
    ) -> List[Dict[str, Any]]:
        """
        Generate specific updates for affected lore elements
        
        Args:
            affected_elements: List of affected lore elements
            event_description: Description of the event
            
        Returns:
            List of updates to apply
        """
        updates = []
        
        # Use an LLM to generate updates for each element
        for element in affected_elements:
            # Create the run context
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
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
            
            # Create an agent for lore updates
            lore_update_agent = Agent(
                name="LoreUpdateAgent",
                instructions="You update existing lore elements based on narrative events while maintaining thematic consistency.",
                model="o3-mini"
            )
            
            # Get the response
            result = await Runner.run(lore_update_agent, prompt, context=run_ctx.context)
            response_text = result.final_output
            
            try:
                # Parse the JSON response
                update_data = json.loads(response_text)
                
                # Add the update
                updates.append({
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': update_data['new_description'],
                    'update_reason': update_data['update_reason'],
                    'impact_level': update_data['impact_level']
                })
            except json.JSONDecodeError:
                logging.error(f"Failed to parse LLM response: {response_text}")
                # Try to extract reasonable update even if JSON parsing failed
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
        """
        Apply the generated updates to the database
        
        Args:
            updates: List of updates to apply
        """
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for update in updates:
                    lore_type = update['lore_type']
                    lore_id = update['lore_id']
                    new_description = update['new_description']
                    
                    # Generate new embedding for the updated content
                    item_name = update.get('name', 'Unknown')
                    embedding_text = f"{item_name} {new_description}"
                    new_embedding = await generate_embedding(embedding_text)
                    
                    # Determine ID field name
                    id_field = 'id'
                    if lore_type == 'LocationLore':
                        id_field = 'location_id'
                    
                    try:
                        # Update the database
                        await conn.execute(f"""
                            UPDATE {lore_type}
                            SET description = $1, embedding = $2
                            WHERE {id_field} = $3
                        """, new_description, new_embedding, lore_id)
                        
                        # Record the update in a history table if it exists
                        history_table_exists = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'lorechangehistory'
                            );
                        """)
                        
                        if history_table_exists:
                            await conn.execute("""
                                INSERT INTO LoreChangeHistory 
                                (lore_type, lore_id, previous_description, new_description, change_reason)
                                VALUES ($1, $2, $3, $4, $5)
                            """, lore_type, lore_id, update['old_description'], new_description, update['update_reason'])
                    except Exception as e:
                        logging.error(f"Error updating {lore_type} ID {lore_id}: {e}")
                        
                    # Clear relevant cache
                    LORE_CACHE.invalidate_pattern(f"{lore_type}_{lore_id}")
    
    async def _generate_consequential_lore(
        self, 
        event_description: str, 
        affected_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate new lore elements that might emerge as a consequence of the event
        
        Args:
            event_description: Description of the event
            affected_elements: List of affected lore elements
            
        Returns:
            List of new lore elements
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
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
        
        # Create an agent for new lore generation
        lore_creation_agent = Agent(
            name="LoreCreationAgent",
            instructions="You create new lore elements that emerge from significant events.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(lore_creation_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        # Parse the JSON response
        try:
            new_elements = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(new_elements, list):
                if isinstance(new_elements, dict):
                    new_elements = [new_elements]
                else:
                    new_elements = []
                    
            # Save the new lore elements to appropriate tables
            await self._save_new_lore_elements(new_elements, event_description)
                
            return new_elements
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for new lore: {response_text}")
            return []
    
    async def _save_new_lore_elements(
        self, 
        new_elements: List[Dict[str, Any]], 
        event_description: str
    ) -> None:
        """
        Save newly generated lore elements to appropriate tables
        
        Args:
            new_elements: List of new lore elements
            event_description: Original event description
        """
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
                elif lore_type == "UrbanMyths":
                    # Use the urban myths manager if available
                    if hasattr(self, 'urban_myths_manager'):
                        await self.urban_myths_manager.add_urban_myth(
                            name=name,
                            description=description,
                            origin_event=event_description,
                            believability=element.get('believability', 6),
                            spread_rate=element.get('spread_rate', 7),
                            regions_known=element.get('regions_known', ['local area'])
                        )
                    else:
                        # Fall back to world lore
                        await self.lore_manager.add_world_lore(
                            name=name,
                            category='urban_myth',
                            description=description,
                            significance=significance,
                            tags=['urban_myth', 'event_consequence']
                        )
                elif lore_type == "NotableFigures":
                    # Use the notable figures manager if available
                    if hasattr(self, 'notable_figures_manager'):
                        await self.notable_figures_manager.add_notable_figure(
                            name=name,
                            description=description,
                            significance=significance,
                            areas_of_influence=element.get('areas_of_influence', ['local']),
                            connection_to_event=event_description
                        )
                    else:
                        # Fall back to world lore
                        await self.lore_manager.add_world_lore(
                            name=name,
                            category='notable_figure',
                            description=description,
                            significance=significance,
                            tags=['notable_figure', 'event_consequence']
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
    
    async def mature_lore_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        """
        Natural evolution of lore over time, simulating how history and culture develop
        
        Args:
            days_passed: Number of in-game days that have passed
            
        Returns:
            Summary of maturation changes
        """
        # Check permissions with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="mature_lore_over_time",
            action_details={"days_passed": days_passed}
        )
        
        if not permission["approved"]:
            logging.warning(f"Lore maturation not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
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
                                SET description = $1
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
                
                if shift_type == "alliance_change":
                    # Two factions change their relationship
                    if len(factions) >= 2:
                        faction1 = factions[0]
                        faction2 = factions[1]
                        
                        # Determine new relationship
                        current_allies1 = faction1['allies'] if faction1['allies'] else []
                        current_rivals1 = faction1['rivals'] if faction1['rivals'] else []
                        
                        if faction2['name'] in current_allies1:
                            # They were allies, now becoming rivals
                            new_relationship = "rivalry"
                            
                            # Update allies and rivals lists
                            if faction2['name'] in current_allies1:
                                current_allies1.remove(faction2['name'])
                            if faction2['name'] not in current_rivals1:
                                current_rivals1.append(faction2['name'])
                                
                            prompt = f"""
                            Two factions that were once allies are now becoming rivals:
                            
                            FACTION 1: {faction1['name']} ({faction1['type']})
                            Description: {faction1['description']}
                            
                            FACTION 2: {faction2['name']} ({faction2['type']})
                            Description: {faction2['description']}
                            
                            Update the description of Faction 1 to reflect this change in relationship.
                            Include information about why they've become rivals and what tensions exist now.
                            Maintain the matriarchal power dynamics in your description.
                            
                            Return only the new description, written in the same style as the original.
                            """
                            
                        elif faction2['name'] in current_rivals1:
                            # They were rivals, now becoming allies
                            new_relationship = "alliance"
                            
                            # Update allies and rivals lists
                            if faction2['name'] in current_rivals1:
                                current_rivals1.remove(faction2['name'])
                            if faction2['name'] not in current_allies1:
                                current_allies1.append(faction2['name'])
                                
                            prompt = f"""
                            Two factions that were once rivals are now becoming allies:
                            
                            FACTION 1: {faction1['name']} ({faction1['type']})
                            Description: {faction1['description']}
                            
                            FACTION 2: {faction2['name']} ({faction2['type']})
                            Description: {faction2['description']}
                            
                            Update the description of Faction 1 to reflect this change in relationship.
                            Include information about why they've become allies and what common interests they now share.
                            Maintain the matriarchal power dynamics in your description.
                            
                            Return only the new description, written in the same style as the original.
                            """
                            
                        else:
                            # They had no specific relationship, now becoming allies
                            new_relationship = "new alliance"
                            
                            # Update allies list
                            if faction2['name'] not in current_allies1:
                                current_allies1.append(faction2['name'])
                                
                            prompt = f"""
                            Two factions are forming a new alliance:
                            
                            FACTION 1: {faction1['name']} ({faction1['type']})
                            Description: {faction1['description']}
                            
                            FACTION 2: {faction2['name']} ({faction2['type']})
                            Description: {faction2['description']}
                            
                            Update the description of Faction 1 to reflect this new alliance.
                            Include information about why they've become allies and what mutual benefits exist.
                            Maintain the matriarchal power dynamics in your description.
                            
                            Return only the new description, written in the same style as the original.
                            """
                        
                        # Get updated description from LLM
                        geo_agent = Agent(
                            name="GeopoliticsAgent",
                            instructions="You evolve geopolitical relationships between factions.",
                            model="o3-mini"
                        )
                        
                        result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                        new_description = result.final_output
                        
                        # Apply the update to faction 1
                        try:
                            # Generate new embedding
                            embedding_text = f"{faction1['name']} {faction1['type']} {new_description}"
                            new_embedding = await generate_embedding(embedding_text)
                            
                            await conn.execute("""
                                UPDATE Factions
                                SET description = $1, 
                                    allies = $2,
                                    rivals = $3,
                                    embedding = $4
                                WHERE id = $5
                            """, new_description, current_allies1, current_rivals1, new_embedding, faction1['id'])
                            
                            # Need to also update faction 2's allies/rivals
                            current_allies2 = faction2['allies'] if faction2['allies'] else []
                            current_rivals2 = faction2['rivals'] if faction2['rivals'] else []
                            
                            if new_relationship == "alliance" or new_relationship == "new alliance":
                                if faction1['name'] in current_rivals2:
                                    current_rivals2.remove(faction1['name'])
                                if faction1['name'] not in current_allies2:
                                    current_allies2.append(faction1['name'])
                            else:  # rivalry
                                if faction1['name'] in current_allies2:
                                    current_allies2.remove(faction1['name'])
                                if faction1['name'] not in current_rivals2:
                                    current_rivals2.append(faction1['name'])
                            
                            await conn.execute("""
                                UPDATE Factions
                                SET allies = $1,
                                    rivals = $2
                                WHERE id = $3
                            """, current_allies2, current_rivals2, faction2['id'])
                            
                            # Record the change
                            changes.append({
                                "change_type": "alliance_change",
                                "faction1_id": faction1['id'],
                                "faction1_name": faction1['name'],
                                "faction2_id": faction2['id'],
                                "faction2_name": faction2['name'],
                                "new_relationship": new_relationship,
                                "old_description": faction1['description'],
                                "new_description": new_description
                            })
                            
                            # Clear cache
                            LORE_CACHE.invalidate_pattern(f"Factions_{faction1['id']}")
                            LORE_CACHE.invalidate_pattern(f"Factions_{faction2['id']}")
                        except Exception as e:
                            logging.error(f"Error updating faction relationship: {e}")
                
                elif shift_type == "territory_dispute":
                    # A faction is expanding its territory
                    faction = factions[0]
                    
                    prompt = f"""
                    This faction is expanding its territorial claims or influence:
                    
                    FACTION: {faction['name']} ({faction['type']})
                    Current Description: {faction['description']}
                    Current Territory: {faction['territory'] if faction['territory'] else "Unspecified areas"}
                    
                    Update the description to show how this faction is claiming new territory or expanding influence.
                    Include what areas they're moving into and any resistance or consequences this creates.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    # Get updated description from LLM
                    geo_agent = Agent(
                        name="GeopoliticsAgent",
                        instructions="You evolve geopolitical relationships between factions.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply the update
                    try:
                        # Generate new embedding
                        embedding_text = f"{faction['name']} {faction['type']} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        # Also update territory to include expansion
                        new_territory = faction['territory'] if faction['territory'] else []
                        new_areas = ["expanding into neighboring regions", "claiming new settlements"]
                        
                        if not isinstance(new_territory, list):
                            # Handle string case
                            new_territory = [new_territory, new_areas[0]]
                        else:
                            new_territory.append(random.choice(new_areas))
                        
                        await conn.execute("""
                            UPDATE Factions
                            SET description = $1,
                                territory = $2,
                                embedding = $3
                            WHERE id = $4
                        """, new_description, new_territory, new_embedding, faction['id'])
                        
                        # Record the change
                        changes.append({
                            "change_type": "territory_dispute",
                            "faction_id": faction['id'],
                            "faction_name": faction['name'],
                            "old_description": faction['description'],
                            "new_description": new_description,
                            "new_territory": new_territory
                        })
                        
                        # Clear cache
                        LORE_CACHE.invalidate_pattern(f"Factions_{faction['id']}")
                    except Exception as e:
                        logging.error(f"Error updating faction territory: {e}")
                
                elif shift_type == "regional_governance" and region:
                    # A region changes its governing faction
                    faction = factions[0]
                    
                    # Only proceed if this is a different faction than current governance
                    if region['governing_faction'] != faction['name']:
                        prompt = f"""
                        A geographic region is changing its governing faction:
                        
                        REGION: {region['name']}
                        Current Description: {region['description']}
                        Previous Governing Faction: {region['governing_faction'] if region['governing_faction'] else "None"}
                        New Governing Faction: {faction['name']} ({faction['type']})
                        
                        Update the region's description to reflect this change in governance.
                        Include how this transition occurred and what changes the new faction is implementing.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        # Get updated description from LLM
                        geo_agent = Agent(
                            name="GeopoliticsAgent",
                            instructions="You evolve geopolitical relationships between factions.",
                            model="o3-mini"
                        )
                        
                        result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                        new_description = result.final_output
                        
                        # Apply the update
                        try:
                            # Generate new embedding
                            embedding_text = f"{region['name']} {new_description}"
                            new_embedding = await generate_embedding(embedding_text)
                            
                            await conn.execute("""
                                UPDATE GeographicRegions
                                SET description = $1,
                                    governing_faction = $2,
                                    embedding = $3
                                WHERE id = $4
                            """, new_description, faction['name'], new_embedding, region['id'])
                            
                            # Record the change
                            changes.append({
                                "change_type": "regional_governance",
                                "region_id": region['id'],
                                "region_name": region['name'],
                                "old_governing_faction": region['governing_faction'] if region['governing_faction'] else "None",
                                "new_governing_faction": faction['name'],
                                "old_description": region['description'],
                                "new_description": new_description
                            })
                            
                            # Clear cache
                            LORE_CACHE.invalidate_pattern(f"GeographicRegions_{region['id']}")
                        except Exception as e:
                            logging.error(f"Error updating region governance: {e}")
                    
                else:  # influence_growth
                    # A faction is growing in influence
                    faction = factions[0]
                    
                    prompt = f"""
                    This faction is experiencing a significant growth in influence and power:
                    
                    FACTION: {faction['name']} ({faction['type']})
                    Current Description: {faction['description']}
                    
                    Update the description to show how this faction is becoming more influential.
                    Include new strategies they're employing, resources they're acquiring, or alliances they're leveraging.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    # Get updated description from LLM
                    geo_agent = Agent(
                        name="GeopoliticsAgent",
                        instructions="You evolve geopolitical relationships between factions.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply the update
                    try:
                        # Generate new embedding
                        embedding_text = f"{faction['name']} {faction['type']} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        await conn.execute("""
                            UPDATE Factions
                            SET description = $1,
                                embedding = $2
                            WHERE id = $3
                        """, new_description, new_embedding, faction['id'])
                        
                        # Record the change
                        changes.append({
                            "change_type": "influence_growth",
                            "faction_id": faction['id'],
                            "faction_name": faction['name'],
                            "old_description": faction['description'],
                            "new_description": new_description
                        })
                        
                        # Clear cache
                        LORE_CACHE.invalidate_pattern(f"Factions_{faction['id']}")
                    except Exception as e:
                        logging.error(f"Error updating faction influence: {e}")
        
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

class UrbanMythManager:
    """
    Manager for urban myths, local stories, and folk tales that develop organically
    across different regions and communities.
    """
    
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

class LocalHistoryManager:
    """
    Manager for local histories, events, and landmarks that are specific
    to particular locations rather than the broader world history.
    """
    
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
                    
                    # Create index
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
                    
                    # Create index
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

class EducationalSystemManager:
    """
    Manages how knowledge is taught and passed down across generations
    within the matriarchal society, including formal and informal systems.
    """
    
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

class GeopoliticalSystemManager:
    """
    Manages the geopolitical landscape of the world, including countries,
    regions, foreign relations, and international power dynamics.
    """
    
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

# Initialize cache for faiths
FAITH_CACHE = LoreCache(max_size=200, ttl=7200)  # 2 hour TTL

class FaithSystem:
    """
    Comprehensive system for managing religions, faiths, and belief systems
    within the matriarchal society with full Nyx governance integration.
    """
    
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
        
    async def initialize_tables(self):
        """Ensure faith system tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if Deities table exists
                deities_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'deities'
                    );
                """)
                
                if not deities_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE Deities (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            gender TEXT NOT NULL, -- female, male, non-binary, etc.
                            domain TEXT[] NOT NULL, -- love, war, knowledge, etc.
                            description TEXT NOT NULL,
                            iconography TEXT,
                            holy_symbol TEXT,
                            sacred_animals TEXT[],
                            sacred_colors TEXT[],
                            relationships JSONB, -- relationships with other deities
                            rank INTEGER CHECK (rank BETWEEN 1 AND 10), -- importance in pantheon
                            worshippers TEXT[], -- types of people who worship
                            pantheon_id INTEGER,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_deities_embedding 
                        ON Deities USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("Deities table created")
                
                # Check if Pantheons table exists
                pantheons_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'pantheons'
                    );
                """)
                
                if not pantheons_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE Pantheons (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            description TEXT NOT NULL,
                            origin_story TEXT NOT NULL,
                            major_holy_days TEXT[],
                            cosmic_structure TEXT, -- how the cosmos is organized
                            afterlife_beliefs TEXT,
                            creation_myth TEXT,
                            geographical_spread TEXT[], -- regions where worshipped
                            dominant_nations TEXT[], -- nations where dominant
                            primary_worshippers TEXT[], -- demographics who worship
                            matriarchal_elements TEXT NOT NULL, -- how it reinforces matriarchy
                            taboos TEXT[],
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_pantheons_embedding 
                        ON Pantheons USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("Pantheons table created")
                
                # Check if ReligiousPractices table exists
                practices_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'religiouspractices'
                    );
                """)
                
                if not practices_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ReligiousPractices (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            practice_type TEXT NOT NULL, -- ritual, ceremony, prayer, etc.
                            description TEXT NOT NULL,
                            frequency TEXT, -- daily, weekly, yearly, etc.
                            required_elements TEXT[], -- components needed
                            performed_by TEXT[], -- priests, all worshippers, etc.
                            purpose TEXT NOT NULL, -- blessing, protection, etc.
                            restricted_to TEXT[], -- if limited to certain people
                            deity_id INTEGER,
                            pantheon_id INTEGER,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_religiouspractices_embedding 
                        ON ReligiousPractices USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ReligiousPractices table created")
                
                # Check if HolySites table exists
                sites_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'holysites'
                    );
                """)
                
                if not sites_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE HolySites (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            site_type TEXT NOT NULL, -- temple, shrine, sacred grove, etc.
                            description TEXT NOT NULL,
                            location_id INTEGER, -- reference to Locations table
                            location_description TEXT, -- if not linked to location
                            deity_id INTEGER,
                            pantheon_id INTEGER,
                            clergy_type TEXT, -- priestesses, clerics, etc.
                            clergy_hierarchy TEXT[], -- ranks in order
                            pilgrimage_info TEXT,
                            miracles_reported TEXT[],
                            restrictions TEXT[], -- who can enter
                            architectural_features TEXT,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_holysites_embedding 
                        ON HolySites USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("HolySites table created")
                
                # Check if ReligiousTexts table exists
                texts_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'religioustexts'
                    );
                """)
                
                if not texts_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ReligiousTexts (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            text_type TEXT NOT NULL, -- scripture, hymnal, prayer book, etc.
                            description TEXT NOT NULL,
                            authorship TEXT, -- divine, prophetic, etc.
                            key_teachings TEXT[] NOT NULL,
                            restricted_to TEXT[], -- if access is limited
                            deity_id INTEGER,
                            pantheon_id INTEGER,
                            notable_passages TEXT[],
                            age_description TEXT, -- how old it is
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_religioustexts_embedding 
                        ON ReligiousTexts USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ReligiousTexts table created")
                
                # Check if ReligiousOrders table exists
                orders_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'religiousorders'
                    );
                """)
                
                if not orders_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ReligiousOrders (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            order_type TEXT NOT NULL, -- monastic, military, scholarly, etc.
                            description TEXT NOT NULL,
                            founding_story TEXT,
                            headquarters TEXT,
                            hierarchy_structure TEXT[],
                            vows TEXT[],
                            practices TEXT[],
                            deity_id INTEGER,
                            pantheon_id INTEGER,
                            gender_composition TEXT, -- female-only, primarily female, mixed, etc.
                            special_abilities TEXT[],
                            notable_members TEXT[],
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_religiousorders_embedding 
                        ON ReligiousOrders USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ReligiousOrders table created")
                
                # Check if ReligiousConflicts table exists
                conflicts_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'religiousconflicts'
                    );
                """)
                
                if not conflicts_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ReligiousConflicts (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            conflict_type TEXT NOT NULL, -- schism, holy war, theological debate, etc.
                            description TEXT NOT NULL,
                            beginning_date TEXT,
                            resolution_date TEXT,
                            status TEXT, -- ongoing, resolved, dormant, etc.
                            parties_involved TEXT[] NOT NULL,
                            core_disagreement TEXT NOT NULL,
                            casualties TEXT,
                            historical_impact TEXT,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_religiousconflicts_embedding 
                        ON ReligiousConflicts USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ReligiousConflicts table created")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_deity",
        action_description="Adding deity: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_deity(
        self, 
        ctx,
        name: str,
        gender: str,
        domain: List[str],
        description: str,
        pantheon_id: Optional[int] = None,
        iconography: Optional[str] = None,
        holy_symbol: Optional[str] = None,
        sacred_animals: Optional[List[str]] = None,
        sacred_colors: Optional[List[str]] = None,
        relationships: Optional[Dict[str, str]] = None,
        rank: int = 5,
        worshippers: Optional[List[str]] = None
    ) -> int:
        """
        Add a deity to the database with governance oversight.
        
        Args:
            name: Name of the deity
            gender: Gender of the deity (female, male, non-binary, etc.)
            domain: List of domains the deity controls
            description: Detailed description
            pantheon_id: Optional ID of the pantheon this deity belongs to
            iconography: Optional description of how the deity is depicted
            holy_symbol: Optional description of the deity's holy symbol
            sacred_animals: Optional list of sacred animals
            sacred_colors: Optional list of sacred colors
            relationships: Optional dict of relationships with other deities
            rank: Importance in pantheon (1-10)
            worshippers: Optional list of types of people who worship
            
        Returns:
            ID of the created deity
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        sacred_animals = sacred_animals or []
        sacred_colors = sacred_colors or []
        relationships = relationships or {}
        worshippers = worshippers or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {gender} {' '.join(domain)} {description}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                deity_id = await conn.fetchval("""
                    INSERT INTO Deities (
                        name, gender, domain, description, pantheon_id,
                        iconography, holy_symbol, sacred_animals, sacred_colors,
                        relationships, rank, worshippers, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id
                """, name, gender, domain, description, pantheon_id,
                     iconography, holy_symbol, sacred_animals, sacred_colors,
                     json.dumps(relationships), rank, worshippers, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("deity")
                
                return deity_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_pantheon",
        action_description="Adding pantheon: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_pantheon(
        self, 
        ctx,
        name: str,
        description: str,
        origin_story: str,
        matriarchal_elements: str,
        creation_myth: Optional[str] = None,
        afterlife_beliefs: Optional[str] = None,
        cosmic_structure: Optional[str] = None,
        major_holy_days: Optional[List[str]] = None,
        geographical_spread: Optional[List[str]] = None,
        dominant_nations: Optional[List[str]] = None,
        primary_worshippers: Optional[List[str]] = None,
        taboos: Optional[List[str]] = None
    ) -> int:
        """
        Add a pantheon to the database with governance oversight.
        
        Args:
            name: Name of the pantheon
            description: General description
            origin_story: How the pantheon came to be
            matriarchal_elements: How it reinforces matriarchy
            creation_myth: Optional creation myth
            afterlife_beliefs: Optional afterlife beliefs
            cosmic_structure: Optional cosmic structure
            major_holy_days: Optional list of major holy days
            geographical_spread: Optional list of regions where worshipped
            dominant_nations: Optional list of nations where dominant
            primary_worshippers: Optional list of demographics who worship
            taboos: Optional list of taboos
            
        Returns:
            ID of the created pantheon
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        major_holy_days = major_holy_days or []
        geographical_spread = geographical_spread or []
        dominant_nations = dominant_nations or []
        primary_worshippers = primary_worshippers or []
        taboos = taboos or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {description} {origin_story} {matriarchal_elements}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon_id = await conn.fetchval("""
                    INSERT INTO Pantheons (
                        name, description, origin_story, matriarchal_elements,
                        creation_myth, afterlife_beliefs, cosmic_structure,
                        major_holy_days, geographical_spread, dominant_nations,
                        primary_worshippers, taboos, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id
                """, name, description, origin_story, matriarchal_elements,
                     creation_myth, afterlife_beliefs, cosmic_structure,
                     major_holy_days, geographical_spread, dominant_nations,
                     primary_worshippers, taboos, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("pantheon")
                
                return pantheon_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_practice",
        action_description="Adding religious practice: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_religious_practice(
        self, 
        ctx,
        name: str,
        practice_type: str,
        description: str,
        purpose: str,
        frequency: Optional[str] = None,
        required_elements: Optional[List[str]] = None,
        performed_by: Optional[List[str]] = None,
        restricted_to: Optional[List[str]] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None
    ) -> int:
        """
        Add a religious practice to the database with governance oversight.
        
        Args:
            name: Name of the practice
            practice_type: Type of practice (ritual, ceremony, etc.)
            description: Detailed description
            purpose: Purpose of the practice
            frequency: Optional frequency (daily, yearly, etc.)
            required_elements: Optional list of components needed
            performed_by: Optional list of who performs it
            restricted_to: Optional list of who it's restricted to
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            
        Returns:
            ID of the created practice
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        required_elements = required_elements or []
        performed_by = performed_by or []
        restricted_to = restricted_to or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {practice_type} {description} {purpose}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                practice_id = await conn.fetchval("""
                    INSERT INTO ReligiousPractices (
                        name, practice_type, description, purpose,
                        frequency, required_elements, performed_by,
                        restricted_to, deity_id, pantheon_id, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, practice_type, description, purpose,
                     frequency, required_elements, performed_by,
                     restricted_to, deity_id, pantheon_id, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("practice")
                
                return practice_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_holy_site",
        action_description="Adding holy site: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_holy_site(
        self, 
        ctx,
        name: str,
        site_type: str,
        description: str,
        clergy_type: str,
        location_id: Optional[int] = None,
        location_description: Optional[str] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None,
        clergy_hierarchy: Optional[List[str]] = None,
        pilgrimage_info: Optional[str] = None,
        miracles_reported: Optional[List[str]] = None,
        restrictions: Optional[List[str]] = None,
        architectural_features: Optional[str] = None
    ) -> int:
        """
        Add a holy site to the database with governance oversight.
        
        Args:
            name: Name of the holy site
            site_type: Type of site (temple, shrine, etc.)
            description: Detailed description
            clergy_type: Type of clergy (priestesses, etc.)
            location_id: Optional ID in Locations table
            location_description: Optional description if no location_id
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            clergy_hierarchy: Optional list of ranks in hierarchy
            pilgrimage_info: Optional information on pilgrimages
            miracles_reported: Optional list of reported miracles
            restrictions: Optional list of restrictions on entry
            architectural_features: Optional description of features
            
        Returns:
            ID of the created holy site
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        clergy_hierarchy = clergy_hierarchy or []
        miracles_reported = miracles_reported or []
        restrictions = restrictions or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {site_type} {description} {clergy_type}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                site_id = await conn.fetchval("""
                    INSERT INTO HolySites (
                        name, site_type, description, clergy_type,
                        location_id, location_description, deity_id,
                        pantheon_id, clergy_hierarchy, pilgrimage_info,
                        miracles_reported, restrictions, architectural_features,
                        embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    RETURNING id
                """, name, site_type, description, clergy_type,
                     location_id, location_description, deity_id,
                     pantheon_id, clergy_hierarchy, pilgrimage_info,
                     miracles_reported, restrictions, architectural_features,
                     embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("site")
                
                return site_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_text",
        action_description="Adding religious text: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_religious_text(
        self, 
        ctx,
        name: str,
        text_type: str,
        description: str,
        key_teachings: List[str],
        authorship: Optional[str] = None,
        restricted_to: Optional[List[str]] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None,
        notable_passages: Optional[List[str]] = None,
        age_description: Optional[str] = None
    ) -> int:
        """
        Add a religious text to the database with governance oversight.
        
        Args:
            name: Name of the text
            text_type: Type of text (scripture, hymnal, etc.)
            description: Detailed description
            key_teachings: List of key teachings
            authorship: Optional description of authorship
            restricted_to: Optional list of who can access it
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            notable_passages: Optional list of notable passages
            age_description: Optional description of age
            
        Returns:
            ID of the created text
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        restricted_to = restricted_to or []
        notable_passages = notable_passages or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {text_type} {description} {' '.join(key_teachings)}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                text_id = await conn.fetchval("""
                    INSERT INTO ReligiousTexts (
                        name, text_type, description, key_teachings,
                        authorship, restricted_to, deity_id,
                        pantheon_id, notable_passages, age_description,
                        embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, text_type, description, key_teachings,
                     authorship, restricted_to, deity_id,
                     pantheon_id, notable_passages, age_description,
                     embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("text")
                
                return text_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_order",
        action_description="Adding religious order: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_religious_order(
        self, 
        ctx,
        name: str,
        order_type: str,
        description: str,
        gender_composition: str,
        founding_story: Optional[str] = None,
        headquarters: Optional[str] = None,
        hierarchy_structure: Optional[List[str]] = None,
        vows: Optional[List[str]] = None,
        practices: Optional[List[str]] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None,
        special_abilities: Optional[List[str]] = None,
        notable_members: Optional[List[str]] = None
    ) -> int:
        """
        Add a religious order to the database with governance oversight.
        
        Args:
            name: Name of the order
            order_type: Type of order (monastic, military, etc.)
            description: Detailed description
            gender_composition: Gender makeup (female-only, etc.)
            founding_story: Optional founding story
            headquarters: Optional headquarters location
            hierarchy_structure: Optional list of ranks
            vows: Optional list of vows taken
            practices: Optional list of practices
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            special_abilities: Optional list of special abilities
            notable_members: Optional list of notable members
            
        Returns:
            ID of the created order
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        hierarchy_structure = hierarchy_structure or []
        vows = vows or []
        practices = practices or []
        special_abilities = special_abilities or []
        notable_members = notable_members or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {order_type} {description} {gender_composition}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                order_id = await conn.fetchval("""
                    INSERT INTO ReligiousOrders (
                        name, order_type, description, gender_composition,
                        founding_story, headquarters, hierarchy_structure,
                        vows, practices, deity_id, pantheon_id,
                        special_abilities, notable_members, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    RETURNING id
                """, name, order_type, description, gender_composition,
                     founding_story, headquarters, hierarchy_structure,
                     vows, practices, deity_id, pantheon_id,
                     special_abilities, notable_members, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("order")
                
                return order_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_conflict",
        action_description="Adding religious conflict: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_religious_conflict(
        self, 
        ctx,
        name: str,
        conflict_type: str,
        description: str,
        parties_involved: List[str],
        core_disagreement: str,
        beginning_date: Optional[str] = None,
        resolution_date: Optional[str] = None,
        status: str = "ongoing",
        casualties: Optional[str] = None,
        historical_impact: Optional[str] = None
    ) -> int:
        """
        Add a religious conflict to the database with governance oversight.
        
        Args:
            name: Name of the conflict
            conflict_type: Type of conflict (schism, holy war, etc.)
            description: Detailed description
            parties_involved: List of parties involved
            core_disagreement: Central point of disagreement
            beginning_date: Optional textual beginning date
            resolution_date: Optional textual resolution date
            status: Status (ongoing, resolved, dormant, etc.)
            casualties: Optional description of casualties
            historical_impact: Optional description of impact
            
        Returns:
            ID of the created conflict
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {conflict_type} {description} {core_disagreement}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                conflict_id = await conn.fetchval("""
                    INSERT INTO ReligiousConflicts (
                        name, conflict_type, description, parties_involved,
                        core_disagreement, beginning_date, resolution_date,
                        status, casualties, historical_impact, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, conflict_type, description, parties_involved,
                     core_disagreement, beginning_date, resolution_date,
                     status, casualties, historical_impact, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("conflict")
                
                return conflict_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_pantheon",
        action_description="Generating pantheon for the world",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_pantheon(self, ctx) -> Dict[str, Any]:
        """
        Generate a complete pantheon for the world with governance oversight.
        
        Returns:
            Dictionary with the pantheon and its deities
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get world info for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get foundation lore for context
                foundation_lore = await conn.fetch("""
                    SELECT category, description FROM WorldLore
                    WHERE category in ('cosmology', 'magic_system', 'social_structure')
                """)
                
                foundation_context = {}
                for row in foundation_lore:
                    foundation_context[row['category']] = row['description']
                
                # Get some geographical regions for context
                regions = await conn.fetch("""
                    SELECT name FROM GeographicRegions
                    LIMIT 5
                """)
                
                region_names = [r['name'] for r in regions]
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, matriarchy_level FROM Nations
                    ORDER BY matriarchy_level DESC
                    LIMIT 5
                """)
                
                nation_context = ""
                for row in nations:
                    nation_context += f"{row['name']} (matriarchy level: {row['matriarchy_level']}), "
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a complete feminine-dominated pantheon for a matriarchal fantasy world.
        
        WORLD CONTEXT:
        Cosmology: {foundation_context.get('cosmology', 'Not available')}
        Magic System: {foundation_context.get('magic_system', 'Not available')}
        Social Structure: {foundation_context.get('social_structure', 'Not available')}
        
        Geographic Regions: {', '.join(region_names)}
        Nations: {nation_context}
        
        Create a pantheon that:
        1. Is predominantly female with goddesses in all major positions of power
        2. Includes a few male deities in subservient or specialized roles
        3. Has a clear hierarchical structure reinforcing feminine dominance
        4. Includes domains that reflect gender power dynamics
        5. Has a cosmic structure that reinforces matriarchal principles
        
        Return a JSON object with:
        1. "pantheon" - details about the overall pantheon
        2. "deities" - array of deity objects
        
        For the pantheon include:
        - name, description, origin_story, matriarchal_elements, creation_myth,
          afterlife_beliefs, cosmic_structure, major_holy_days, geographical_spread,
          dominant_nations, primary_worshippers, taboos
        
        For each deity include:
        - name, gender, domain (array), description, iconography, holy_symbol,
          sacred_animals (array), sacred_colors (array), rank (1-10),
          worshippers (array), relationships (to other deities as an object)
        """
        
        # Create an agent for pantheon generation
        pantheon_agent = Agent(
            name="PantheonGenerationAgent",
            instructions="You create religious pantheons for matriarchal fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(pantheon_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            pantheon_data = json.loads(response_text)
            
            # Ensure we have both pantheon and deities
            if not all([
                isinstance(pantheon_data, dict),
                "pantheon" in pantheon_data,
                "deities" in pantheon_data,
                isinstance(pantheon_data["deities"], list)
            ]):
                raise ValueError("Invalid pantheon structure in response")
            
            # Save the pantheon
            pantheon_info = pantheon_data["pantheon"]
            deities_info = pantheon_data["deities"]
            
            # Create the pantheon
            pantheon_id = await self.add_pantheon(
                run_ctx,
                name=pantheon_info.get("name", "The Pantheon"),
                description=pantheon_info.get("description", ""),
                origin_story=pantheon_info.get("origin_story", ""),
                matriarchal_elements=pantheon_info.get("matriarchal_elements", ""),
                creation_myth=pantheon_info.get("creation_myth"),
                afterlife_beliefs=pantheon_info.get("afterlife_beliefs"),
                cosmic_structure=pantheon_info.get("cosmic_structure"),
                major_holy_days=pantheon_info.get("major_holy_days"),
                geographical_spread=pantheon_info.get("geographical_spread"),
                dominant_nations=pantheon_info.get("dominant_nations"),
                primary_worshippers=pantheon_info.get("primary_worshippers"),
                taboos=pantheon_info.get("taboos")
            )
            
            # Create each deity
            created_deities = []
            for deity in deities_info:
                try:
                    deity_id = await self.add_deity(
                        run_ctx,
                        name=deity.get("name", "Unnamed Deity"),
                        gender=deity.get("gender", "female"),
                        domain=deity.get("domain", []),
                        description=deity.get("description", ""),
                        pantheon_id=pantheon_id,
                        iconography=deity.get("iconography"),
                        holy_symbol=deity.get("holy_symbol"),
                        sacred_animals=deity.get("sacred_animals"),
                        sacred_colors=deity.get("sacred_colors"),
                        relationships=deity.get("relationships", {}),
                        rank=deity.get("rank", 5),
                        worshippers=deity.get("worshippers")
                    )
                    
                    deity["id"] = deity_id
                    created_deities.append(deity)
                except Exception as e:
                    logging.error(f"Error creating deity {deity.get('name')}: {e}")
            
            # Return the created pantheon and deities
            return {
                "pantheon": {**pantheon_info, "id": pantheon_id},
                "deities": created_deities
            }
            
        except Exception as e:
            logging.error(f"Error generating pantheon: {e}")
            return {"error": str(e)}
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_religious_practices",
        action_description="Generating religious practices for pantheon: {pantheon_id}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_religious_practices(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Generate religious practices for a pantheon with governance oversight.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of generated religious practices
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                # Get the deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious practices for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        
        DEITIES:
        {json.dumps(deities_data, indent=2)}
        
        Create 5-7 religious practices that:
        1. Reinforce matriarchal dominance themes
        2. Include varied practice types (daily rituals, seasonal ceremonies, rites of passage, etc.)
        3. Provide specific details on how they're performed
        4. Show which deities they are associated with
        
        Return a JSON array where each practice has:
        - name: Name of the practice
        - practice_type: Type of practice (ritual, ceremony, prayer, etc.)
        - description: Detailed description of the practice
        - purpose: Purpose of the practice
        - frequency: How often it's performed
        - required_elements: Array of required components
        - performed_by: Array of who performs it
        - restricted_to: Array of who it's restricted to (if applicable)
        - deity_id: ID of the associated deity (use exact IDs from the provided deity list)
        """
        
        # Create an agent for practice generation
        practice_agent = Agent(
            name="ReligiousPracticeAgent",
            instructions="You create religious practices for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(practice_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            practices = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(practices, list):
                if isinstance(practices, dict):
                    practices = [practices]
                else:
                    practices = []
            
            # Save each practice
            created_practices = []
            for practice in practices:
                try:
                    practice_id = await self.add_religious_practice(
                        run_ctx,
                        name=practice.get("name", "Unnamed Practice"),
                        practice_type=practice.get("practice_type", "ritual"),
                        description=practice.get("description", ""),
                        purpose=practice.get("purpose", "worship"),
                        frequency=practice.get("frequency"),
                        required_elements=practice.get("required_elements"),
                        performed_by=practice.get("performed_by"),
                        restricted_to=practice.get("restricted_to"),
                        deity_id=practice.get("deity_id"),
                        pantheon_id=pantheon_id
                    )
                    
                    practice["id"] = practice_id
                    created_practices.append(practice)
                except Exception as e:
                    logging.error(f"Error creating religious practice {practice.get('name')}: {e}")
            
            return created_practices
        except Exception as e:
            logging.error(f"Error generating religious practices: {e}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_holy_sites",
        action_description="Generating holy sites for pantheon: {pantheon_id}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_holy_sites(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Generate holy sites for a pantheon with governance oversight.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of generated holy sites
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon and location info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, geographical_spread, dominant_nations
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                # Get the major deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain
                    FROM Deities
                    WHERE pantheon_id = $1 AND rank >= 6
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Get potential locations
                locations = await conn.fetch("""
                    SELECT id, location_name, description
                    FROM Locations
                    LIMIT 10
                """)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
                location_data = [dict(location) for location in locations]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate holy sites for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        
        MAJOR DEITIES:
        {json.dumps(deities_data, indent=2)}
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        Create 3-5 holy sites that:
        1. Reflect matriarchal dominance in architecture and function
        2. Include sites for major deities and some for the pantheon as a whole
        3. Have distinct clergy systems with feminine leadership
        4. Include varied site types (temples, shrines, sacred groves, etc.)
        
        Return a JSON array where each site has:
        - name: Name of the holy site
        - site_type: Type of site (temple, shrine, sacred grove, etc.)
        - description: Detailed description
        - clergy_type: Type of clergy (priestesses, etc.)
        - location_id: ID of the location (use exact IDs from the provided locations, or null)
        - location_description: Description of the location if no location_id provided
        - deity_id: ID of the associated deity (use exact IDs from the provided deities, or null)
        - clergy_hierarchy: Array of ranks in the clergy hierarchy
        - pilgrimage_info: Information about pilgrimages (if applicable)
        - miracles_reported: Array of reported miracles (if applicable)
        - restrictions: Array of restrictions on entry
        - architectural_features: Architectural features of the site
        """
        
        # Create an agent for holy site generation
        site_agent = Agent(
            name="HolySiteAgent",
            instructions="You create holy sites for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(site_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            sites = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(sites, list):
                if isinstance(sites, dict):
                    sites = [sites]
                else:
                    sites = []
            
            # Save each holy site
            created_sites = []
            for site in sites:
                try:
                    site_id = await self.add_holy_site(
                        run_ctx,
                        name=site.get("name", "Unnamed Site"),
                        site_type=site.get("site_type", "temple"),
                        description=site.get("description", ""),
                        clergy_type=site.get("clergy_type", "priestesses"),
                        location_id=site.get("location_id"),
                        location_description=site.get("location_description"),
                        deity_id=site.get("deity_id"),
                        pantheon_id=pantheon_id,
                        clergy_hierarchy=site.get("clergy_hierarchy"),
                        pilgrimage_info=site.get("pilgrimage_info"),
                        miracles_reported=site.get("miracles_reported"),
                        restrictions=site.get("restrictions"),
                        architectural_features=site.get("architectural_features")
                    )
                    
                    site["id"] = site_id
                    created_sites.append(site)
                except Exception as e:
                    logging.error(f"Error creating holy site {site.get('name')}: {e}")
            
            return created_sites
        except Exception as e:
            logging.error(f"Error generating holy sites: {e}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_faith_system",
        action_description="Generating complete faith system for the world",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_complete_faith_system(self, ctx) -> Dict[str, Any]:
        """
        Generate a complete faith system for the world with governance oversight.
        
        Returns:
            Dictionary with all faith system components
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # 1. Generate pantheon
        pantheon_data = await self.generate_pantheon(run_ctx)
        
        if "error" in pantheon_data:
            return pantheon_data
            
        pantheon_id = pantheon_data["pantheon"]["id"]
        
        # 2. Generate religious practices
        practices = await self.generate_religious_practices(run_ctx, pantheon_id)
        
        # 3. Generate holy sites
        holy_sites = await self.generate_holy_sites(run_ctx, pantheon_id)
        
        # 4. Generate religious texts
        religious_texts = await self._generate_religious_texts(run_ctx, pantheon_id)
        
        # 5. Generate religious orders
        religious_orders = await self._generate_religious_orders(run_ctx, pantheon_id)
        
        # 6. Generate religious conflicts
        religious_conflicts = await self._generate_religious_conflicts(run_ctx, pantheon_id)
        
        # Combine all results
        result = {
            "pantheon": pantheon_data["pantheon"],
            "deities": pantheon_data["deities"],
            "practices": practices,
            "holy_sites": holy_sites,
            "religious_texts": religious_texts,
            "religious_orders": religious_orders,
            "religious_conflicts": religious_conflicts
        }
        
        return result
    
    async def _generate_religious_texts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Helper method to generate religious texts for a pantheon"""
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, creation_myth
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return []
                
                # Get the deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious texts for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        CREATION MYTH: {pantheon_data.get('creation_myth')}
        
        DEITIES:
        {json.dumps(deities_data[:5], indent=2)}
        
        Create 3-5 religious texts that:
        1. Reinforce matriarchal principles and feminine divine superiority
        2. Include varied text types (core scripture, commentaries, prayers, etc.)
        3. Describe who has access to each text
        4. Include specific key teachings
        
        Return a JSON array where each text has:
        - name: Name of the text
        - text_type: Type of text (scripture, hymnal, prayer book, etc.)
        - description: Detailed description
        - key_teachings: Array of key teachings
        - authorship: Description of authorship
        - restricted_to: Array of who can access it (if applicable)
        - deity_id: ID of the associated deity (use exact IDs from the provided deities, or null)
        - notable_passages: Array of notable passages
        - age_description: Description of the text's age
        """
        
        # Create an agent for text generation
        text_agent = Agent(
            name="ReligiousTextAgent",
            instructions="You create religious texts for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(text_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            texts = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(texts, list):
                if isinstance(texts, dict):
                    texts = [texts]
                else:
                    texts = []
            
            # Save each religious text
            created_texts = []
            for text in texts:
                try:
                    text_id = await self.add_religious_text(
                        run_ctx,
                        name=text.get("name", "Unnamed Text"),
                        text_type=text.get("text_type", "scripture"),
                        description=text.get("description", ""),
                        key_teachings=text.get("key_teachings", []),
                        authorship=text.get("authorship"),
                        restricted_to=text.get("restricted_to"),
                        deity_id=text.get("deity_id"),
                        pantheon_id=pantheon_id,
                        notable_passages=text.get("notable_passages"),
                        age_description=text.get("age_description")
                    )
                    
                    text["id"] = text_id
                    created_texts.append(text)
                except Exception as e:
                    logging.error(f"Error creating religious text {text.get('name')}: {e}")
            
            return created_texts
        except Exception as e:
            logging.error(f"Error generating religious texts: {e}")
            return []
    
    async def _generate_religious_orders(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Helper method to generate religious orders for a pantheon"""
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, matriarchal_elements
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return []
                
                # Get the deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Get holy sites for potential headquarters
                holy_sites = await conn.fetch("""
                    SELECT id, name, site_type
                    FROM HolySites
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
                site_data = [dict(site) for site in holy_sites]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious orders for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        MATRIARCHAL ELEMENTS: {pantheon_data.get('matriarchal_elements')}
        
        DEITIES:
        {json.dumps(deities_data[:5], indent=2)}
        
        HOLY SITES (potential headquarters):
        {json.dumps(site_data, indent=2)}
        
        Create 3-4 religious orders that:
        1. Heavily emphasize female leadership and matriarchal structure
        2. Include varied order types (monastic, military, scholarly, etc.)
        3. Have clear gender compositions (most should be female-dominated)
        4. Include details on hierarchies and practices
        
        Return a JSON array where each order has:
        - name: Name of the order
        - order_type: Type of order (monastic, military, scholarly, etc.)
        - description: Detailed description
        - gender_composition: Gender makeup (female-only, primarily female, mixed, etc.)
        - founding_story: Founding story
        - headquarters: Headquarters location (can reference holy sites)
        - hierarchy_structure: Array of ranks in hierarchy (from highest to lowest)
        - vows: Array of vows taken by members
        - practices: Array of practices
        - deity_id: ID of the associated deity (use exact IDs from the provided deities, or null)
        - special_abilities: Array of special abilities (if applicable)
        - notable_members: Array of notable members (if applicable)
        """
        
        # Create an agent for order generation
        order_agent = Agent(
            name="ReligiousOrderAgent",
            instructions="You create religious orders for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(order_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            orders = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(orders, list):
                if isinstance(orders, dict):
                    orders = [orders]
                else:
                    orders = []
            
            # Save each religious order
            created_orders = []
            for order in orders:
                try:
                    order_id = await self.add_religious_order(
                        run_ctx,
                        name=order.get("name", "Unnamed Order"),
                        order_type=order.get("order_type", "monastic"),
                        description=order.get("description", ""),
                        gender_composition=order.get("gender_composition", "female-only"),
                        founding_story=order.get("founding_story"),
                        headquarters=order.get("headquarters"),
                        hierarchy_structure=order.get("hierarchy_structure"),
                        vows=order.get("vows"),
                        practices=order.get("practices"),
                        deity_id=order.get("deity_id"),
                        pantheon_id=pantheon_id,
                        special_abilities=order.get("special_abilities"),
                        notable_members=order.get("notable_members")
                    )
                    
                    order["id"] = order_id
                    created_orders.append(order)
                except Exception as e:
                    logging.error(f"Error creating religious order {order.get('name')}: {e}")
            
            return created_orders
        except Exception as e:
            logging.error(f"Error generating religious orders: {e}")
            return []
    
    async def _generate_religious_conflicts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Helper method to generate religious conflicts for a pantheon"""
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, matriarchal_elements
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return []
                
                # Get religious orders for potential conflict parties
                orders = await conn.fetch("""
                    SELECT id, name, order_type, gender_composition
                    FROM ReligiousOrders
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)
                
                # Get nations for potential conflicts
                nations = await conn.fetch("""
                    SELECT id, name, government_type, matriarchy_level
                    FROM Nations
                    LIMIT 5
                """)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                order_data = [dict(order) for order in orders]
                nation_data = [dict(nation) for nation in nations]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious conflicts for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        
        RELIGIOUS ORDERS (potential conflict parties):
        {json.dumps(order_data, indent=2)}
        
        NATIONS (potential conflict locations):
        {json.dumps(nation_data, indent=2)}
        
        Create 2-3 religious conflicts that:
        1. Show theological or power struggles within the faith
        2. Include conflicts that highlight gender dynamics (not just female vs male)
        3. Include different conflict types (schisms, theological debates, holy wars)
        4. Have realistic core disagreements
        
        Return a JSON array where each conflict has:
        - name: Name of the conflict
        - conflict_type: Type of conflict (schism, holy war, theological debate, etc.)
        - description: Detailed description
        - parties_involved: Array of parties involved
        - core_disagreement: Central point of disagreement
        - beginning_date: Textual beginning date
        - resolution_date: Textual resolution date (if resolved)
        - status: Status (ongoing, resolved, dormant)
        - casualties: Description of casualties (if applicable)
        - historical_impact: Description of historical impact
        """
        
        # Create an agent for conflict generation
        conflict_agent = Agent(
            name="ReligiousConflictAgent",
            instructions="You create religious conflicts for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(conflict_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            conflicts = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(conflicts, list):
                if isinstance(conflicts, dict):
                    conflicts = [conflicts]
                else:
                    conflicts = []
            
            # Save each religious conflict
            created_conflicts = []
            for conflict in conflicts:
                try:
                    conflict_id = await self.add_religious_conflict(
                        run_ctx,
                        name=conflict.get("name", "Unnamed Conflict"),
                        conflict_type=conflict.get("conflict_type", "schism"),
                        description=conflict.get("description", ""),
                        parties_involved=conflict.get("parties_involved", []),
                        core_disagreement=conflict.get("core_disagreement", ""),
                        beginning_date=conflict.get("beginning_date"),
                        resolution_date=conflict.get("resolution_date"),
                        status=conflict.get("status", "ongoing"),
                        casualties=conflict.get("casualties"),
                        historical_impact=conflict.get("historical_impact")
                    )
                    
                    conflict["id"] = conflict_id
                    created_conflicts.append(conflict)
                except Exception as e:
                    logging.error(f"Error creating religious conflict {conflict.get('name')}: {e}")
            
            return created_conflicts
        except Exception as e:
            logging.error(f"Error generating religious conflicts: {e}")
            return []
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await self.initialize_governance()
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="faith_system",
            agent_instance=self
        )
        
        # Issue a directive for faith system
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="faith_system",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Create and manage faith systems that emphasize feminine divine superiority.",
                "scope": "world_building"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"FaithSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")


class EmergentLoreSystem:
    """
    System for generating emergent lore, events, and developments
    to ensure the world evolves organically over time.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.faith_system = FaithSystem(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_emergent_event",
        action_description="Generating emergent world event",
        id_from_context=lambda ctx: "emergent_lore"
    )
    async def generate_emergent_event(self, ctx) -> Dict[str, Any]:
        """
        Generate a random emergent event in the world with governance oversight.
        
        Returns:
            Event details and lore impact
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get current world state for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get active factions
                factions = await conn.fetch("""
                    SELECT id, name, type, description
                    FROM Factions
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get some nations
                nations = await conn.fetch("""
                    SELECT id, name, government_type
                    FROM Nations
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get some locations
                locations = await conn.fetch("""
                    SELECT id, location_name, description
                    FROM Locations
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Convert to lists
                faction_data = [dict(faction) for faction in factions]
                nation_data = [dict(nation) for nation in nations]
                location_data = [dict(location) for location in locations]
        
        # Determine event type
        event_types = [
            "political_shift", "military_conflict", "natural_disaster",
            "cultural_development", "technological_advancement", "religious_event",
            "economic_change", "diplomatic_incident"
        ]
        
        # Weight certain events based on available data
        weights = [1] * len(event_types)
        
        # More likely to generate faction-related events if we have factions
        if faction_data:
            faction_event_indices = [0, 1, 3, 6, 7]  # political, military, cultural, economic, diplomatic
            for idx in faction_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # More likely to generate nation-related events if we have nations
        if nation_data:
            nation_event_indices = [0, 1, 7]  # political, military, diplomatic
            for idx in nation_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # More likely to generate location-related events if we have locations
        if location_data:
            location_event_indices = [2, 3, 5]  # natural, cultural, religious
            for idx in location_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # Select event type
        event_type = random.choices(event_types, weights=weights, k=1)[0]
        
        # Create a prompt based on event type
        if event_type == "political_shift":
            entities = faction_data if faction_data else nation_data
            prompt = f"""
            Generate a political shift event for this world:
            
            POTENTIAL ENTITIES INVOLVED:
            {json.dumps(entities, indent=2)}
            
            Create a significant political shift event that:
            1. Changes power dynamics in a meaningful way
            2. Is specific and detailed enough to impact the world
            3. Reinforces or challenges matriarchal power structures
            4. Could lead to interesting narrative developments
            
            Return a JSON object with:
            - event_name: Name of the event
            - event_type: "political_shift"
            - description: Detailed description
            - entities_involved: Array of entity names involved
            - instigator: The primary instigating entity
            - immediate_consequences: Array of immediate consequences
            - potential_long_term_effects: Array of potential long-term effects
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "military_conflict":
            prompt = f"""
            Generate a military conflict event for this world:
            
            POTENTIAL NATIONS INVOLVED:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS INVOLVED:
            {json.dumps(faction_data, indent=2)}
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            Create a military conflict that:
            1. Involves clear aggressor and defender sides
            2. Has a specific cause and objective
            3. Creates opportunities for heroism and character development
            4. Shows how warfare operates in a matriarchal society
            
            Return a JSON object with:
            - event_name: Name of the conflict
            - event_type: "military_conflict"
            - description: Detailed description
            - aggressor: Entity that initiated the conflict
            - defender: Entity defending against aggression
            - location: Where the conflict is primarily occurring
            - cause: What caused the conflict
            - scale: Scale of the conflict (skirmish, battle, war, etc.)
            - current_status: Current status of the conflict
            - casualties: Description of casualties
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "natural_disaster":
            prompt = f"""
            Generate a natural disaster event for this world:
            
            POTENTIAL LOCATIONS AFFECTED:
            {json.dumps(location_data, indent=2)}
            
            Create a natural disaster that:
            1. Has a significant impact on the affected area
            2. Could have supernatural or magical elements
            3. Creates opportunities for societal response that highlights values
            4. Changes the physical landscape in some way
            
            Return a JSON object with:
            - event_name: Name of the disaster
            - event_type: "natural_disaster"
            - description: Detailed description
            - disaster_type: Type of disaster (earthquake, flood, magical storm, etc.)
            - primary_location: Primary location affected
            - secondary_locations: Array of secondarily affected locations
            - severity: Severity level (1-10)
            - immediate_impact: Description of immediate impact
            - response: How communities and leadership are responding
            - supernatural_elements: Any supernatural aspects (if applicable)
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "cultural_development":
            prompt = f"""
            Generate a cultural development event for this world:
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a cultural development that:
            1. Introduces a new tradition, art form, or cultural practice
            2. Comes from a specific community or demographic
            3. Has meaning within the matriarchal power structure
            4. Could spread to other communities
            
            Return a JSON object with:
            - event_name: Name of the cultural development
            - event_type: "cultural_development"
            - description: Detailed description
            - development_type: Type of development (tradition, art form, practice, etc.)
            - originating_group: Group where it originated
            - significance: Social significance
            - symbolism: Symbolic meaning
            - reception: How different groups have received it
            - spread_potential: Likelihood of spreading (1-10)
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "technological_advancement":
            prompt = f"""
            Generate a technological or magical advancement event for this world:
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a technological or magical advancement that:
            1. Changes how something is done in the world
            2. Was developed by specific individuals or groups
            3. Has both benefits and potential drawbacks
            4. Reinforces matriarchal control of knowledge or resources
            
            Return a JSON object with:
            - event_name: Name of the advancement
            - event_type: "technological_advancement"
            - description: Detailed description
            - advancement_type: Type (technological, magical, alchemical, etc.)
            - creator: Who created or discovered it
            - applications: Potential applications
            - limitations: Current limitations
            - control: Who controls access to this advancement
            - societal_impact: How it impacts society
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "religious_event":
            prompt = f"""
            Generate a religious event for this world:
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            Create a religious event that:
            1. Has significance within an existing faith system
            2. Could be interpreted as divine intervention or revelation
            3. Affects religious practices or beliefs
            4. Reinforces the feminine divine nature of the world
            
            Return a JSON object with:
            - event_name: Name of the religious event
            - event_type: "religious_event"
            - description: Detailed description
            - event_category: Type of event (miracle, prophecy, divine manifestation, etc.)
            - location: Where it occurred
            - witnesses: Who witnessed it
            - religious_significance: Significance to believers
            - skeptic_explanation: How skeptics explain it
            - faith_impact: How it impacts faith practices
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "economic_change":
            prompt = f"""
            Generate an economic change event for this world:
            
            POTENTIAL NATIONS:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create an economic change that:
            1. Shifts resource distribution or trade patterns
            2. Affects multiple groups or regions
            3. Creates winners and losers
            4. Shows how economic power relates to matriarchal control
            
            Return a JSON object with:
            - event_name: Name of the economic change
            - event_type: "economic_change"
            - description: Detailed description
            - change_type: Type of change (trade shift, resource discovery, currency change, etc.)
            - primary_causes: What caused the change
            - beneficiaries: Who benefits
            - disadvantaged: Who is disadvantaged
            - resource_involved: Primary resource or commodity involved
            - wealth_redistribution: How wealth is being redistributed
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        else:  # diplomatic_incident
            prompt = f"""
            Generate a diplomatic incident for this world:
            
            POTENTIAL NATIONS:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a diplomatic incident that:
            1. Creates tension between two or more groups
            2. Stems from a specific action or misunderstanding
            3. Requires diplomatic resolution
            4. Highlights cultural differences or values
            
            Return a JSON object with:
            - event_name: Name of the diplomatic incident
            - event_type: "diplomatic_incident"
            - description: Detailed description
            - parties_involved: Array of involved parties
            - instigating_action: What triggered the incident
            - cultural_factors: Cultural factors at play
            - severity: Severity level (1-10)
            - potential_resolutions: Possible ways to resolve it
            - current_status: Current diplomatic status
            - affected_lore_categories: Array of lore categories this would affect
            """
        
        # Create an agent for event generation
        event_agent = Agent(
            name="EmergentEventAgent",
            instructions="You create emergent world events for fantasy settings.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(event_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            event_data = json.loads(response_text)
            
            # Create a unified event description for lore evolution
            event_description = event_data.get("description", "")
            event_name = event_data.get("event_name", "Unnamed Event")
            
            # Use the LoreEvolutionSystem to evolve the lore based on this event
            lore_evolution = LoreEvolutionSystem(self.user_id, self.conversation_id)
            lore_updates = await lore_evolution.evolve_lore_with_event(
                f"{event_name}: {event_description}"
            )
            
            # Add the lore updates to the event data
            event_data["lore_updates"] = lore_updates
            
            # Return the combined data
            return event_data
            
        except Exception as e:
            logging.error(f"Error generating emergent event: {e}")
            return {"error": str(e)}
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_cyclical_event",
        action_description="Generating cyclical seasonal event",
        id_from_context=lambda ctx: "emergent_lore"
    )
    async def generate_cyclical_event(self, ctx, season: str = None, is_solstice: bool = False, is_equinox: bool = False) -> Dict[str, Any]:
        """
        Generate a cyclical seasonal event with governance oversight.
        
        Args:
            season: Optional season (spring, summer, fall, winter)
            is_solstice: Whether this is a solstice event
            is_equinox: Whether this is an equinox event
            
        Returns:
            Cyclical event details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Set default season if not provided
        if not season:
            seasons = ["spring", "summer", "fall", "winter"]
            season = random.choice(seasons)
            
        # Get relevant data for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get cultural elements
                cultural_elements = await conn.fetch("""
                    SELECT name, type, description
                    FROM CulturalElements
                    WHERE type IN ('festival', 'tradition', 'ceremony', 'ritual')
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get religious practices
                religious_practices = await conn.fetch("""
                    SELECT name, practice_type, description, frequency
                    FROM ReligiousPractices
                    WHERE frequency LIKE '%annual%' OR frequency LIKE '%seasonal%'
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get suitable locations
                locations = await conn.fetch("""
                    SELECT location_name, description
                    FROM Locations
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Convert to lists
                cultural_data = [dict(element) for element in cultural_elements]
                religious_data = [dict(practice) for practice in religious_practices]
                location_data = [dict(location) for location in locations]
        
        # Determine event type and special modifiers
        event_subtypes = ["festival", "ritual", "harvest", "hunt", "pilgrimage", "market", "ceremony"]
        
        if is_solstice:
            event_subtypes.extend(["celestial alignment", "divine manifestation", "magical surge"])
        if is_equinox:
            event_subtypes.extend(["balance ritual", "transition ceremony", "world renewal"])
            
        if season == "spring":
            event_subtypes.extend(["planting", "fertility rite", "renewal ceremony"])
        elif season == "summer":
            event_subtypes.extend(["sun celebration", "competition", "coming of age"])
        elif season == "fall":
            event_subtypes.extend(["harvest festival", "ancestor veneration", "preparation ritual"])
        elif season == "winter":
            event_subtypes.extend(["endurance trial", "darkness ritual", "shelter ceremony"])
            
        event_subtype = random.choice(event_subtypes)
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a cyclical {season} {event_subtype} event for a matriarchal fantasy world.
        
        EXISTING CULTURAL ELEMENTS:
        {json.dumps(cultural_data, indent=2)}
        
        EXISTING RELIGIOUS PRACTICES:
        {json.dumps(religious_data, indent=2)}
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        SPECIAL MODIFIERS:
        Solstice Event: {"Yes" if is_solstice else "No"}
        Equinox Event: {"Yes" if is_equinox else "No"}
        
        Create a cyclical event that:
        1. Is tied to the {season} season specifically
        2. Reinforces matriarchal power and feminine divine connection
        3. Involves community participation with specific roles
        4. Has cultural and possibly religious significance
        
        Return a JSON object with:
        - event_name: Name of the cyclical event
        - event_type: "cyclical_event"
        - subtype: The specific type of event (festival, ritual, etc.)
        - season: The associated season
        - description: Detailed description
        - frequency: How often it occurs
        - duration: How long it lasts
        - locations: Where it's celebrated
        - primary_participants: Who leads or conducts the event
        - community_role: How the community participates
        - religious_significance: Any religious meaning
        - cultural_significance: Cultural meaning and importance
        - material_components: Required materials or preparations
        - traditions: Specific traditions associated with this event
        - gender_roles: How gender influences participation
        """
        
        # Create an agent for cyclical event generation
        cyclical_agent = Agent(
            name="CyclicalEventAgent",
            instructions="You create seasonal festivals and cyclical events for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(cyclical_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            event_data = json.loads(response_text)
            
            # Store this as a cultural element in the database
            try:
                # Prepare the data
                name = event_data.get("event_name", f"{season.capitalize()} {event_subtype.capitalize()}")
                description = event_data.get("description", "")
                
                cultural_description = description
                if "cultural_significance" in event_data:
                    cultural_description += f"\n\nCultural Significance: {event_data['cultural_significance']}"
                if "traditions" in event_data:
                    traditions = event_data.get("traditions", [])
                    if isinstance(traditions, list):
                        cultural_description += f"\n\nTraditions: {', '.join(traditions)}"
                    else:
                        cultural_description += f"\n\nTraditions: {traditions}"
                        
                event_data["stored_as_cultural_element"] = True
                
                # Add to CulturalElements
                element_id = await self.lore_manager.add_cultural_element(
                    name=name,
                    element_type=event_data.get("subtype", "festival"),
                    description=cultural_description,
                    practiced_by=event_data.get("locations", []),
                    significance=8,  # Cyclical events are quite significant
                    historical_origin=f"This {event_data.get('subtype', 'event')} developed as a {season} celebration."
                )
                
                event_data["cultural_element_id"] = element_id
                
                # If it has religious significance, also add as a religious practice
                if "religious_significance" in event_data and event_data.get("religious_significance"):
                    religious_description = description
                    religious_description += f"\n\nReligious Significance: {event_data['religious_significance']}"
                    
                    # Add to ReligiousPractices
                    practice_id = await self.faith_system.add_religious_practice(
                        run_ctx,
                        name=name,
                        practice_type=event_data.get("subtype", "festival"),
                        description=religious_description,
                        purpose="seasonal celebration",
                        frequency=f"Annual ({season})",
                        required_elements=event_data.get("material_components", [])
                    )
                    
                    event_data["religious_practice_id"] = practice_id
                    event_data["stored_as_religious_practice"] = True
            except Exception as e:
                logging.error(f"Error storing cyclical event: {e}")
                event_data["storage_error"] = str(e)
            
            return event_data
            
        except Exception as e:
            logging.error(f"Error generating cyclical event: {e}")
            return {"error": str(e)}
    
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
        # If we have more non-matriarchal nations, make this one

class FaithSystem:
    """
    Comprehensive system for managing religions, faiths, and belief systems
    within the matriarchal society with full Nyx governance integration.
    """
    
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
        
    async def initialize_tables(self):
        """Ensure faith system tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if Deities table exists
                deities_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'deities'
                    );
                """)
                
                if not deities_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE Deities (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            gender TEXT NOT NULL, -- female, male, non-binary, etc.
                            domain TEXT[] NOT NULL, -- love, war, knowledge, etc.
                            description TEXT NOT NULL,
                            iconography TEXT,
                            holy_symbol TEXT,
                            sacred_animals TEXT[],
                            sacred_colors TEXT[],
                            relationships JSONB, -- relationships with other deities
                            rank INTEGER CHECK (rank BETWEEN 1 AND 10), -- importance in pantheon
                            worshippers TEXT[], -- types of people who worship
                            pantheon_id INTEGER,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_deities_embedding 
                        ON Deities USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("Deities table created")
                
                # Check if Pantheons table exists
                pantheons_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'pantheons'
                    );
                """)
                
                if not pantheons_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE Pantheons (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            description TEXT NOT NULL,
                            origin_story TEXT NOT NULL,
                            major_holy_days TEXT[],
                            cosmic_structure TEXT, -- how the cosmos is organized
                            afterlife_beliefs TEXT,
                            creation_myth TEXT,
                            geographical_spread TEXT[], -- regions where worshipped
                            dominant_nations TEXT[], -- nations where dominant
                            primary_worshippers TEXT[], -- demographics who worship
                            matriarchal_elements TEXT NOT NULL, -- how it reinforces matriarchy
                            taboos TEXT[],
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_pantheons_embedding 
                        ON Pantheons USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("Pantheons table created")
                
                # Check if ReligiousPractices table exists
                practices_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'religiouspractices'
                    );
                """)
                
                if not practices_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ReligiousPractices (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            practice_type TEXT NOT NULL, -- ritual, ceremony, prayer, etc.
                            description TEXT NOT NULL,
                            frequency TEXT, -- daily, weekly, yearly, etc.
                            required_elements TEXT[], -- components needed
                            performed_by TEXT[], -- priests, all worshippers, etc.
                            purpose TEXT NOT NULL, -- blessing, protection, etc.
                            restricted_to TEXT[], -- if limited to certain people
                            deity_id INTEGER,
                            pantheon_id INTEGER,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_religiouspractices_embedding 
                        ON ReligiousPractices USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ReligiousPractices table created")
                
                # Check if HolySites table exists
                sites_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'holysites'
                    );
                """)
                
                if not sites_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE HolySites (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            site_type TEXT NOT NULL, -- temple, shrine, sacred grove, etc.
                            description TEXT NOT NULL,
                            location_id INTEGER, -- reference to Locations table
                            location_description TEXT, -- if not linked to location
                            deity_id INTEGER,
                            pantheon_id INTEGER,
                            clergy_type TEXT, -- priestesses, clerics, etc.
                            clergy_hierarchy TEXT[], -- ranks in order
                            pilgrimage_info TEXT,
                            miracles_reported TEXT[],
                            restrictions TEXT[], -- who can enter
                            architectural_features TEXT,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_holysites_embedding 
                        ON HolySites USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("HolySites table created")
                
                # Check if ReligiousTexts table exists
                texts_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'religioustexts'
                    );
                """)
                
                if not texts_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ReligiousTexts (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            text_type TEXT NOT NULL, -- scripture, hymnal, prayer book, etc.
                            description TEXT NOT NULL,
                            authorship TEXT, -- divine, prophetic, etc.
                            key_teachings TEXT[] NOT NULL,
                            restricted_to TEXT[], -- if access is limited
                            deity_id INTEGER,
                            pantheon_id INTEGER,
                            notable_passages TEXT[],
                            age_description TEXT, -- how old it is
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_religioustexts_embedding 
                        ON ReligiousTexts USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ReligiousTexts table created")
                
                # Check if ReligiousOrders table exists
                orders_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'religiousorders'
                    );
                """)
                
                if not orders_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ReligiousOrders (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            order_type TEXT NOT NULL, -- monastic, military, scholarly, etc.
                            description TEXT NOT NULL,
                            founding_story TEXT,
                            headquarters TEXT,
                            hierarchy_structure TEXT[],
                            vows TEXT[],
                            practices TEXT[],
                            deity_id INTEGER,
                            pantheon_id INTEGER,
                            gender_composition TEXT, -- female-only, primarily female, mixed, etc.
                            special_abilities TEXT[],
                            notable_members TEXT[],
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_religiousorders_embedding 
                        ON ReligiousOrders USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ReligiousOrders table created")
                
                # Check if ReligiousConflicts table exists
                conflicts_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'religiousconflicts'
                    );
                """)
                
                if not conflicts_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ReligiousConflicts (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            conflict_type TEXT NOT NULL, -- schism, holy war, theological debate, etc.
                            description TEXT NOT NULL,
                            beginning_date TEXT,
                            resolution_date TEXT,
                            status TEXT, -- ongoing, resolved, dormant, etc.
                            parties_involved TEXT[] NOT NULL,
                            core_disagreement TEXT NOT NULL,
                            casualties TEXT,
                            historical_impact TEXT,
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_religiousconflicts_embedding 
                        ON ReligiousConflicts USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ReligiousConflicts table created")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_deity",
        action_description="Adding deity: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_deity(
        self, 
        ctx,
        name: str,
        gender: str,
        domain: List[str],
        description: str,
        pantheon_id: Optional[int] = None,
        iconography: Optional[str] = None,
        holy_symbol: Optional[str] = None,
        sacred_animals: Optional[List[str]] = None,
        sacred_colors: Optional[List[str]] = None,
        relationships: Optional[Dict[str, str]] = None,
        rank: int = 5,
        worshippers: Optional[List[str]] = None
    ) -> int:
        """
        Add a deity to the database with governance oversight.
        
        Args:
            name: Name of the deity
            gender: Gender of the deity (female, male, non-binary, etc.)
            domain: List of domains the deity controls
            description: Detailed description
            pantheon_id: Optional ID of the pantheon this deity belongs to
            iconography: Optional description of how the deity is depicted
            holy_symbol: Optional description of the deity's holy symbol
            sacred_animals: Optional list of sacred animals
            sacred_colors: Optional list of sacred colors
            relationships: Optional dict of relationships with other deities
            rank: Importance in pantheon (1-10)
            worshippers: Optional list of types of people who worship
            
        Returns:
            ID of the created deity
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        sacred_animals = sacred_animals or []
        sacred_colors = sacred_colors or []
        relationships = relationships or {}
        worshippers = worshippers or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {gender} {' '.join(domain)} {description}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                deity_id = await conn.fetchval("""
                    INSERT INTO Deities (
                        name, gender, domain, description, pantheon_id,
                        iconography, holy_symbol, sacred_animals, sacred_colors,
                        relationships, rank, worshippers, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id
                """, name, gender, domain, description, pantheon_id,
                     iconography, holy_symbol, sacred_animals, sacred_colors,
                     json.dumps(relationships), rank, worshippers, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("deity")
                
                return deity_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_pantheon",
        action_description="Adding pantheon: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_pantheon(
        self, 
        ctx,
        name: str,
        description: str,
        origin_story: str,
        matriarchal_elements: str,
        creation_myth: Optional[str] = None,
        afterlife_beliefs: Optional[str] = None,
        cosmic_structure: Optional[str] = None,
        major_holy_days: Optional[List[str]] = None,
        geographical_spread: Optional[List[str]] = None,
        dominant_nations: Optional[List[str]] = None,
        primary_worshippers: Optional[List[str]] = None,
        taboos: Optional[List[str]] = None
    ) -> int:
        """
        Add a pantheon to the database with governance oversight.
        
        Args:
            name: Name of the pantheon
            description: General description
            origin_story: How the pantheon came to be
            matriarchal_elements: How it reinforces matriarchy
            creation_myth: Optional creation myth
            afterlife_beliefs: Optional afterlife beliefs
            cosmic_structure: Optional cosmic structure
            major_holy_days: Optional list of major holy days
            geographical_spread: Optional list of regions where worshipped
            dominant_nations: Optional list of nations where dominant
            primary_worshippers: Optional list of demographics who worship
            taboos: Optional list of taboos
            
        Returns:
            ID of the created pantheon
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        major_holy_days = major_holy_days or []
        geographical_spread = geographical_spread or []
        dominant_nations = dominant_nations or []
        primary_worshippers = primary_worshippers or []
        taboos = taboos or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {description} {origin_story} {matriarchal_elements}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon_id = await conn.fetchval("""
                    INSERT INTO Pantheons (
                        name, description, origin_story, matriarchal_elements,
                        creation_myth, afterlife_beliefs, cosmic_structure,
                        major_holy_days, geographical_spread, dominant_nations,
                        primary_worshippers, taboos, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id
                """, name, description, origin_story, matriarchal_elements,
                     creation_myth, afterlife_beliefs, cosmic_structure,
                     major_holy_days, geographical_spread, dominant_nations,
                     primary_worshippers, taboos, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("pantheon")
                
                return pantheon_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_practice",
        action_description="Adding religious practice: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_religious_practice(
        self, 
        ctx,
        name: str,
        practice_type: str,
        description: str,
        purpose: str,
        frequency: Optional[str] = None,
        required_elements: Optional[List[str]] = None,
        performed_by: Optional[List[str]] = None,
        restricted_to: Optional[List[str]] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None
    ) -> int:
        """
        Add a religious practice to the database with governance oversight.
        
        Args:
            name: Name of the practice
            practice_type: Type of practice (ritual, ceremony, etc.)
            description: Detailed description
            purpose: Purpose of the practice
            frequency: Optional frequency (daily, yearly, etc.)
            required_elements: Optional list of components needed
            performed_by: Optional list of who performs it
            restricted_to: Optional list of who it's restricted to
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            
        Returns:
            ID of the created practice
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        required_elements = required_elements or []
        performed_by = performed_by or []
        restricted_to = restricted_to or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {practice_type} {description} {purpose}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                practice_id = await conn.fetchval("""
                    INSERT INTO ReligiousPractices (
                        name, practice_type, description, purpose,
                        frequency, required_elements, performed_by,
                        restricted_to, deity_id, pantheon_id, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, practice_type, description, purpose,
                     frequency, required_elements, performed_by,
                     restricted_to, deity_id, pantheon_id, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("practice")
                
                return practice_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_holy_site",
        action_description="Adding holy site: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_holy_site(
        self, 
        ctx,
        name: str,
        site_type: str,
        description: str,
        clergy_type: str,
        location_id: Optional[int] = None,
        location_description: Optional[str] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None,
        clergy_hierarchy: Optional[List[str]] = None,
        pilgrimage_info: Optional[str] = None,
        miracles_reported: Optional[List[str]] = None,
        restrictions: Optional[List[str]] = None,
        architectural_features: Optional[str] = None
    ) -> int:
        """
        Add a holy site to the database with governance oversight.
        
        Args:
            name: Name of the holy site
            site_type: Type of site (temple, shrine, etc.)
            description: Detailed description
            clergy_type: Type of clergy (priestesses, etc.)
            location_id: Optional ID in Locations table
            location_description: Optional description if no location_id
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            clergy_hierarchy: Optional list of ranks in hierarchy
            pilgrimage_info: Optional information on pilgrimages
            miracles_reported: Optional list of reported miracles
            restrictions: Optional list of restrictions on entry
            architectural_features: Optional description of features
            
        Returns:
            ID of the created holy site
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        clergy_hierarchy = clergy_hierarchy or []
        miracles_reported = miracles_reported or []
        restrictions = restrictions or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {site_type} {description} {clergy_type}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                site_id = await conn.fetchval("""
                    INSERT INTO HolySites (
                        name, site_type, description, clergy_type,
                        location_id, location_description, deity_id,
                        pantheon_id, clergy_hierarchy, pilgrimage_info,
                        miracles_reported, restrictions, architectural_features,
                        embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    RETURNING id
                """, name, site_type, description, clergy_type,
                     location_id, location_description, deity_id,
                     pantheon_id, clergy_hierarchy, pilgrimage_info,
                     miracles_reported, restrictions, architectural_features,
                     embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("site")
                
                return site_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_text",
        action_description="Adding religious text: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_religious_text(
        self, 
        ctx,
        name: str,
        text_type: str,
        description: str,
        key_teachings: List[str],
        authorship: Optional[str] = None,
        restricted_to: Optional[List[str]] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None,
        notable_passages: Optional[List[str]] = None,
        age_description: Optional[str] = None
    ) -> int:
        """
        Add a religious text to the database with governance oversight.
        
        Args:
            name: Name of the text
            text_type: Type of text (scripture, hymnal, etc.)
            description: Detailed description
            key_teachings: List of key teachings
            authorship: Optional description of authorship
            restricted_to: Optional list of who can access it
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            notable_passages: Optional list of notable passages
            age_description: Optional description of age
            
        Returns:
            ID of the created text
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        restricted_to = restricted_to or []
        notable_passages = notable_passages or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {text_type} {description} {' '.join(key_teachings)}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                text_id = await conn.fetchval("""
                    INSERT INTO ReligiousTexts (
                        name, text_type, description, key_teachings,
                        authorship, restricted_to, deity_id,
                        pantheon_id, notable_passages, age_description,
                        embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, text_type, description, key_teachings,
                     authorship, restricted_to, deity_id,
                     pantheon_id, notable_passages, age_description,
                     embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("text")
                
                return text_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_order",
        action_description="Adding religious order: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_religious_order(
        self, 
        ctx,
        name: str,
        order_type: str,
        description: str,
        gender_composition: str,
        founding_story: Optional[str] = None,
        headquarters: Optional[str] = None,
        hierarchy_structure: Optional[List[str]] = None,
        vows: Optional[List[str]] = None,
        practices: Optional[List[str]] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None,
        special_abilities: Optional[List[str]] = None,
        notable_members: Optional[List[str]] = None
    ) -> int:
        """
        Add a religious order to the database with governance oversight.
        
        Args:
            name: Name of the order
            order_type: Type of order (monastic, military, etc.)
            description: Detailed description
            gender_composition: Gender makeup (female-only, etc.)
            founding_story: Optional founding story
            headquarters: Optional headquarters location
            hierarchy_structure: Optional list of ranks
            vows: Optional list of vows taken
            practices: Optional list of practices
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            special_abilities: Optional list of special abilities
            notable_members: Optional list of notable members
            
        Returns:
            ID of the created order
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        hierarchy_structure = hierarchy_structure or []
        vows = vows or []
        practices = practices or []
        special_abilities = special_abilities or []
        notable_members = notable_members or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {order_type} {description} {gender_composition}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                order_id = await conn.fetchval("""
                    INSERT INTO ReligiousOrders (
                        name, order_type, description, gender_composition,
                        founding_story, headquarters, hierarchy_structure,
                        vows, practices, deity_id, pantheon_id,
                        special_abilities, notable_members, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    RETURNING id
                """, name, order_type, description, gender_composition,
                     founding_story, headquarters, hierarchy_structure,
                     vows, practices, deity_id, pantheon_id,
                     special_abilities, notable_members, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("order")
                
                return order_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_conflict",
        action_description="Adding religious conflict: {name}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def add_religious_conflict(
        self, 
        ctx,
        name: str,
        conflict_type: str,
        description: str,
        parties_involved: List[str],
        core_disagreement: str,
        beginning_date: Optional[str] = None,
        resolution_date: Optional[str] = None,
        status: str = "ongoing",
        casualties: Optional[str] = None,
        historical_impact: Optional[str] = None
    ) -> int:
        """
        Add a religious conflict to the database with governance oversight.
        
        Args:
            name: Name of the conflict
            conflict_type: Type of conflict (schism, holy war, etc.)
            description: Detailed description
            parties_involved: List of parties involved
            core_disagreement: Central point of disagreement
            beginning_date: Optional textual beginning date
            resolution_date: Optional textual resolution date
            status: Status (ongoing, resolved, dormant, etc.)
            casualties: Optional description of casualties
            historical_impact: Optional description of impact
            
        Returns:
            ID of the created conflict
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {conflict_type} {description} {core_disagreement}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                conflict_id = await conn.fetchval("""
                    INSERT INTO ReligiousConflicts (
                        name, conflict_type, description, parties_involved,
                        core_disagreement, beginning_date, resolution_date,
                        status, casualties, historical_impact, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, conflict_type, description, parties_involved,
                     core_disagreement, beginning_date, resolution_date,
                     status, casualties, historical_impact, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("conflict")
                
                return conflict_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_pantheon",
        action_description="Generating pantheon for the world",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_pantheon(self, ctx) -> Dict[str, Any]:
        """
        Generate a complete pantheon for the world with governance oversight.
        
        Returns:
            Dictionary with the pantheon and its deities
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get world info for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get foundation lore for context
                foundation_lore = await conn.fetch("""
                    SELECT category, description FROM WorldLore
                    WHERE category in ('cosmology', 'magic_system', 'social_structure')
                """)
                
                foundation_context = {}
                for row in foundation_lore:
                    foundation_context[row['category']] = row['description']
                
                # Get some geographical regions for context
                regions = await conn.fetch("""
                    SELECT name FROM GeographicRegions
                    LIMIT 5
                """)
                
                region_names = [r['name'] for r in regions]
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, matriarchy_level FROM Nations
                    ORDER BY matriarchy_level DESC
                    LIMIT 5
                """)
                
                nation_context = ""
                for row in nations:
                    nation_context += f"{row['name']} (matriarchy level: {row['matriarchy_level']}), "
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a complete feminine-dominated pantheon for a matriarchal fantasy world.
        
        WORLD CONTEXT:
        Cosmology: {foundation_context.get('cosmology', 'Not available')}
        Magic System: {foundation_context.get('magic_system', 'Not available')}
        Social Structure: {foundation_context.get('social_structure', 'Not available')}
        
        Geographic Regions: {', '.join(region_names)}
        Nations: {nation_context}
        
        Create a pantheon that:
        1. Is predominantly female with goddesses in all major positions of power
        2. Includes a few male deities in subservient or specialized roles
        3. Has a clear hierarchical structure reinforcing feminine dominance
        4. Includes domains that reflect gender power dynamics
        5. Has a cosmic structure that reinforces matriarchal principles
        
        Return a JSON object with:
        1. "pantheon" - details about the overall pantheon
        2. "deities" - array of deity objects
        
        For the pantheon include:
        - name, description, origin_story, matriarchal_elements, creation_myth,
          afterlife_beliefs, cosmic_structure, major_holy_days, geographical_spread,
          dominant_nations, primary_worshippers, taboos
        
        For each deity include:
        - name, gender, domain (array), description, iconography, holy_symbol,
          sacred_animals (array), sacred_colors (array), rank (1-10),
          worshippers (array), relationships (to other deities as an object)
        """
        
        # Create an agent for pantheon generation
        pantheon_agent = Agent(
            name="PantheonGenerationAgent",
            instructions="You create religious pantheons for matriarchal fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(pantheon_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            pantheon_data = json.loads(response_text)
            
            # Ensure we have both pantheon and deities
            if not all([
                isinstance(pantheon_data, dict),
                "pantheon" in pantheon_data,
                "deities" in pantheon_data,
                isinstance(pantheon_data["deities"], list)
            ]):
                raise ValueError("Invalid pantheon structure in response")
            
            # Save the pantheon
            pantheon_info = pantheon_data["pantheon"]
            deities_info = pantheon_data["deities"]
            
            # Create the pantheon
            pantheon_id = await self.add_pantheon(
                run_ctx,
                name=pantheon_info.get("name", "The Pantheon"),
                description=pantheon_info.get("description", ""),
                origin_story=pantheon_info.get("origin_story", ""),
                matriarchal_elements=pantheon_info.get("matriarchal_elements", ""),
                creation_myth=pantheon_info.get("creation_myth"),
                afterlife_beliefs=pantheon_info.get("afterlife_beliefs"),
                cosmic_structure=pantheon_info.get("cosmic_structure"),
                major_holy_days=pantheon_info.get("major_holy_days"),
                geographical_spread=pantheon_info.get("geographical_spread"),
                dominant_nations=pantheon_info.get("dominant_nations"),
                primary_worshippers=pantheon_info.get("primary_worshippers"),
                taboos=pantheon_info.get("taboos")
            )
            
            # Create each deity
            created_deities = []
            for deity in deities_info:
                try:
                    deity_id = await self.add_deity(
                        run_ctx,
                        name=deity.get("name", "Unnamed Deity"),
                        gender=deity.get("gender", "female"),
                        domain=deity.get("domain", []),
                        description=deity.get("description", ""),
                        pantheon_id=pantheon_id,
                        iconography=deity.get("iconography"),
                        holy_symbol=deity.get("holy_symbol"),
                        sacred_animals=deity.get("sacred_animals"),
                        sacred_colors=deity.get("sacred_colors"),
                        relationships=deity.get("relationships", {}),
                        rank=deity.get("rank", 5),
                        worshippers=deity.get("worshippers")
                    )
                    
                    deity["id"] = deity_id
                    created_deities.append(deity)
                except Exception as e:
                    logging.error(f"Error creating deity {deity.get('name')}: {e}")
            
            # Return the created pantheon and deities
            return {
                "pantheon": {**pantheon_info, "id": pantheon_id},
                "deities": created_deities
            }
            
        except Exception as e:
            logging.error(f"Error generating pantheon: {e}")
            return {"error": str(e)}
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_religious_practices",
        action_description="Generating religious practices for pantheon: {pantheon_id}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_religious_practices(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Generate religious practices for a pantheon with governance oversight.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of generated religious practices
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                # Get the deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious practices for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        
        DEITIES:
        {json.dumps(deities_data, indent=2)}
        
        Create 5-7 religious practices that:
        1. Reinforce matriarchal dominance themes
        2. Include varied practice types (daily rituals, seasonal ceremonies, rites of passage, etc.)
        3. Provide specific details on how they're performed
        4. Show which deities they are associated with
        
        Return a JSON array where each practice has:
        - name: Name of the practice
        - practice_type: Type of practice (ritual, ceremony, prayer, etc.)
        - description: Detailed description of the practice
        - purpose: Purpose of the practice
        - frequency: How often it's performed
        - required_elements: Array of required components
        - performed_by: Array of who performs it
        - restricted_to: Array of who it's restricted to (if applicable)
        - deity_id: ID of the associated deity (use exact IDs from the provided deity list)
        """
        
        # Create an agent for practice generation
        practice_agent = Agent(
            name="ReligiousPracticeAgent",
            instructions="You create religious practices for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(practice_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            practices = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(practices, list):
                if isinstance(practices, dict):
                    practices = [practices]
                else:
                    practices = []
            
            # Save each practice
            created_practices = []
            for practice in practices:
                try:
                    practice_id = await self.add_religious_practice(
                        run_ctx,
                        name=practice.get("name", "Unnamed Practice"),
                        practice_type=practice.get("practice_type", "ritual"),
                        description=practice.get("description", ""),
                        purpose=practice.get("purpose", "worship"),
                        frequency=practice.get("frequency"),
                        required_elements=practice.get("required_elements"),
                        performed_by=practice.get("performed_by"),
                        restricted_to=practice.get("restricted_to"),
                        deity_id=practice.get("deity_id"),
                        pantheon_id=pantheon_id
                    )
                    
                    practice["id"] = practice_id
                    created_practices.append(practice)
                except Exception as e:
                    logging.error(f"Error creating religious practice {practice.get('name')}: {e}")
            
            return created_practices
        except Exception as e:
            logging.error(f"Error generating religious practices: {e}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_holy_sites",
        action_description="Generating holy sites for pantheon: {pantheon_id}",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_holy_sites(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Generate holy sites for a pantheon with governance oversight.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of generated holy sites
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon and location info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, geographical_spread, dominant_nations
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                # Get the major deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain
                    FROM Deities
                    WHERE pantheon_id = $1 AND rank >= 6
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Get potential locations
                locations = await conn.fetch("""
                    SELECT id, location_name, description
                    FROM Locations
                    LIMIT 10
                """)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
                location_data = [dict(location) for location in locations]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate holy sites for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        
        MAJOR DEITIES:
        {json.dumps(deities_data, indent=2)}
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        Create 3-5 holy sites that:
        1. Reflect matriarchal dominance in architecture and function
        2. Include sites for major deities and some for the pantheon as a whole
        3. Have distinct clergy systems with feminine leadership
        4. Include varied site types (temples, shrines, sacred groves, etc.)
        
        Return a JSON array where each site has:
        - name: Name of the holy site
        - site_type: Type of site (temple, shrine, sacred grove, etc.)
        - description: Detailed description
        - clergy_type: Type of clergy (priestesses, etc.)
        - location_id: ID of the location (use exact IDs from the provided locations, or null)
        - location_description: Description of the location if no location_id provided
        - deity_id: ID of the associated deity (use exact IDs from the provided deities, or null)
        - clergy_hierarchy: Array of ranks in the clergy hierarchy
        - pilgrimage_info: Information about pilgrimages (if applicable)
        - miracles_reported: Array of reported miracles (if applicable)
        - restrictions: Array of restrictions on entry
        - architectural_features: Architectural features of the site
        """
        
        # Create an agent for holy site generation
        site_agent = Agent(
            name="HolySiteAgent",
            instructions="You create holy sites for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(site_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            sites = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(sites, list):
                if isinstance(sites, dict):
                    sites = [sites]
                else:
                    sites = []
            
            # Save each holy site
            created_sites = []
            for site in sites:
                try:
                    site_id = await self.add_holy_site(
                        run_ctx,
                        name=site.get("name", "Unnamed Site"),
                        site_type=site.get("site_type", "temple"),
                        description=site.get("description", ""),
                        clergy_type=site.get("clergy_type", "priestesses"),
                        location_id=site.get("location_id"),
                        location_description=site.get("location_description"),
                        deity_id=site.get("deity_id"),
                        pantheon_id=pantheon_id,
                        clergy_hierarchy=site.get("clergy_hierarchy"),
                        pilgrimage_info=site.get("pilgrimage_info"),
                        miracles_reported=site.get("miracles_reported"),
                        restrictions=site.get("restrictions"),
                        architectural_features=site.get("architectural_features")
                    )
                    
                    site["id"] = site_id
                    created_sites.append(site)
                except Exception as e:
                    logging.error(f"Error creating holy site {site.get('name')}: {e}")
            
            return created_sites
        except Exception as e:
            logging.error(f"Error generating holy sites: {e}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_faith_system",
        action_description="Generating complete faith system for the world",
        id_from_context=lambda ctx: "faith_system"
    )
    async def generate_complete_faith_system(self, ctx) -> Dict[str, Any]:
        """
        Generate a complete faith system for the world with governance oversight.
        
        Returns:
            Dictionary with all faith system components
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # 1. Generate pantheon
        pantheon_data = await self.generate_pantheon(run_ctx)
        
        if "error" in pantheon_data:
            return pantheon_data
            
        pantheon_id = pantheon_data["pantheon"]["id"]
        
        # 2. Generate religious practices
        practices = await self.generate_religious_practices(run_ctx, pantheon_id)
        
        # 3. Generate holy sites
        holy_sites = await self.generate_holy_sites(run_ctx, pantheon_id)
        
        # 4. Generate religious texts
        religious_texts = await self._generate_religious_texts(run_ctx, pantheon_id)
        
        # 5. Generate religious orders
        religious_orders = await self._generate_religious_orders(run_ctx, pantheon_id)
        
        # 6. Generate religious conflicts
        religious_conflicts = await self._generate_religious_conflicts(run_ctx, pantheon_id)
        
        # Combine all results
        result = {
            "pantheon": pantheon_data["pantheon"],
            "deities": pantheon_data["deities"],
            "practices": practices,
            "holy_sites": holy_sites,
            "religious_texts": religious_texts,
            "religious_orders": religious_orders,
            "religious_conflicts": religious_conflicts
        }
        
        return result
    
    async def _generate_religious_texts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Helper method to generate religious texts for a pantheon"""
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, creation_myth
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return []
                
                # Get the deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious texts for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        CREATION MYTH: {pantheon_data.get('creation_myth')}
        
        DEITIES:
        {json.dumps(deities_data[:5], indent=2)}
        
        Create 3-5 religious texts that:
        1. Reinforce matriarchal principles and feminine divine superiority
        2. Include varied text types (core scripture, commentaries, prayers, etc.)
        3. Describe who has access to each text
        4. Include specific key teachings
        
        Return a JSON array where each text has:
        - name: Name of the text
        - text_type: Type of text (scripture, hymnal, prayer book, etc.)
        - description: Detailed description
        - key_teachings: Array of key teachings
        - authorship: Description of authorship
        - restricted_to: Array of who can access it (if applicable)
        - deity_id: ID of the associated deity (use exact IDs from the provided deities, or null)
        - notable_passages: Array of notable passages
        - age_description: Description of the text's age
        """
        
        # Create an agent for text generation
        text_agent = Agent(
            name="ReligiousTextAgent",
            instructions="You create religious texts for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(text_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            texts = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(texts, list):
                if isinstance(texts, dict):
                    texts = [texts]
                else:
                    texts = []
            
            # Save each religious text
            created_texts = []
            for text in texts:
                try:
                    text_id = await self.add_religious_text(
                        run_ctx,
                        name=text.get("name", "Unnamed Text"),
                        text_type=text.get("text_type", "scripture"),
                        description=text.get("description", ""),
                        key_teachings=text.get("key_teachings", []),
                        authorship=text.get("authorship"),
                        restricted_to=text.get("restricted_to"),
                        deity_id=text.get("deity_id"),
                        pantheon_id=pantheon_id,
                        notable_passages=text.get("notable_passages"),
                        age_description=text.get("age_description")
                    )
                    
                    text["id"] = text_id
                    created_texts.append(text)
                except Exception as e:
                    logging.error(f"Error creating religious text {text.get('name')}: {e}")
            
            return created_texts
        except Exception as e:
            logging.error(f"Error generating religious texts: {e}")
            return []
    
    async def _generate_religious_orders(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Helper method to generate religious orders for a pantheon"""
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, matriarchal_elements
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return []
                
                # Get the deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Get holy sites for potential headquarters
                holy_sites = await conn.fetch("""
                    SELECT id, name, site_type
                    FROM HolySites
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
                site_data = [dict(site) for site in holy_sites]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious orders for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        MATRIARCHAL ELEMENTS: {pantheon_data.get('matriarchal_elements')}
        
        DEITIES:
        {json.dumps(deities_data[:5], indent=2)}
        
        HOLY SITES (potential headquarters):
        {json.dumps(site_data, indent=2)}
        
        Create 3-4 religious orders that:
        1. Heavily emphasize female leadership and matriarchal structure
        2. Include varied order types (monastic, military, scholarly, etc.)
        3. Have clear gender compositions (most should be female-dominated)
        4. Include details on hierarchies and practices
        
        Return a JSON array where each order has:
        - name: Name of the order
        - order_type: Type of order (monastic, military, scholarly, etc.)
        - description: Detailed description
        - gender_composition: Gender makeup (female-only, primarily female, mixed, etc.)
        - founding_story: Founding story
        - headquarters: Headquarters location (can reference holy sites)
        - hierarchy_structure: Array of ranks in hierarchy (from highest to lowest)
        - vows: Array of vows taken by members
        - practices: Array of practices
        - deity_id: ID of the associated deity (use exact IDs from the provided deities, or null)
        - special_abilities: Array of special abilities (if applicable)
        - notable_members: Array of notable members (if applicable)
        """
        
        # Create an agent for order generation
        order_agent = Agent(
            name="ReligiousOrderAgent",
            instructions="You create religious orders for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(order_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            orders = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(orders, list):
                if isinstance(orders, dict):
                    orders = [orders]
                else:
                    orders = []
            
            # Save each religious order
            created_orders = []
            for order in orders:
                try:
                    order_id = await self.add_religious_order(
                        run_ctx,
                        name=order.get("name", "Unnamed Order"),
                        order_type=order.get("order_type", "monastic"),
                        description=order.get("description", ""),
                        gender_composition=order.get("gender_composition", "female-only"),
                        founding_story=order.get("founding_story"),
                        headquarters=order.get("headquarters"),
                        hierarchy_structure=order.get("hierarchy_structure"),
                        vows=order.get("vows"),
                        practices=order.get("practices"),
                        deity_id=order.get("deity_id"),
                        pantheon_id=pantheon_id,
                        special_abilities=order.get("special_abilities"),
                        notable_members=order.get("notable_members")
                    )
                    
                    order["id"] = order_id
                    created_orders.append(order)
                except Exception as e:
                    logging.error(f"Error creating religious order {order.get('name')}: {e}")
            
            return created_orders
        except Exception as e:
            logging.error(f"Error generating religious orders: {e}")
            return []
    
    async def _generate_religious_conflicts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Helper method to generate religious conflicts for a pantheon"""
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, matriarchal_elements
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return []
                
                # Get religious orders for potential conflict parties
                orders = await conn.fetch("""
                    SELECT id, name, order_type, gender_composition
                    FROM ReligiousOrders
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)
                
                # Get nations for potential conflicts
                nations = await conn.fetch("""
                    SELECT id, name, government_type, matriarchy_level
                    FROM Nations
                    LIMIT 5
                """)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                order_data = [dict(order) for order in orders]
                nation_data = [dict(nation) for nation in nations]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious conflicts for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        
        RELIGIOUS ORDERS (potential conflict parties):
        {json.dumps(order_data, indent=2)}
        
        NATIONS (potential conflict locations):
        {json.dumps(nation_data, indent=2)}
        
        Create 2-3 religious conflicts that:
        1. Show theological or power struggles within the faith
        2. Include conflicts that highlight gender dynamics (not just female vs male)
        3. Include different conflict types (schisms, theological debates, holy wars)
        4. Have realistic core disagreements
        
        Return a JSON array where each conflict has:
        - name: Name of the conflict
        - conflict_type: Type of conflict (schism, holy war, theological debate, etc.)
        - description: Detailed description
        - parties_involved: Array of parties involved
        - core_disagreement: Central point of disagreement
        - beginning_date: Textual beginning date
        - resolution_date: Textual resolution date (if resolved)
        - status: Status (ongoing, resolved, dormant)
        - casualties: Description of casualties (if applicable)
        - historical_impact: Description of historical impact
        """
        
        # Create an agent for conflict generation
        conflict_agent = Agent(
            name="ReligiousConflictAgent",
            instructions="You create religious conflicts for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(conflict_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            conflicts = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(conflicts, list):
                if isinstance(conflicts, dict):
                    conflicts = [conflicts]
                else:
                    conflicts = []
            
            # Save each religious conflict
            created_conflicts = []
            for conflict in conflicts:
                try:
                    conflict_id = await self.add_religious_conflict(
                        run_ctx,
                        name=conflict.get("name", "Unnamed Conflict"),
                        conflict_type=conflict.get("conflict_type", "schism"),
                        description=conflict.get("description", ""),
                        parties_involved=conflict.get("parties_involved", []),
                        core_disagreement=conflict.get("core_disagreement", ""),
                        beginning_date=conflict.get("beginning_date"),
                        resolution_date=conflict.get("resolution_date"),
                        status=conflict.get("status", "ongoing"),
                        casualties=conflict.get("casualties"),
                        historical_impact=conflict.get("historical_impact")
                    )
                    
                    conflict["id"] = conflict_id
                    created_conflicts.append(conflict)
                except Exception as e:
                    logging.error(f"Error creating religious conflict {conflict.get('name')}: {e}")
            
            return created_conflicts
        except Exception as e:
            logging.error(f"Error generating religious conflicts: {e}")
            return []
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await self.initialize_governance()
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="faith_system",
            agent_instance=self
        )
        
        # Issue a directive for faith system
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="faith_system",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Create and manage faith systems that emphasize feminine divine superiority.",
                "scope": "world_building"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"FaithSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")


# -------------------------------------------------------------------------------
# Enhanced Lore Integration Functions
# -------------------------------------------------------------------------------

async def register_all_enhanced_lore_systems(user_id: int, conversation_id: int) -> Dict[str, bool]:
    """
    Register all enhanced lore systems with Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary of registration results
    """
    # Get the Nyx governance system
    governance = await get_central_governance(user_id, conversation_id)
    
    # Track registration results
    registration_results = {}
    
    # Register faith system
    try:
        faith_system = FaithSystem(user_id, conversation_id)
        await faith_system.register_with_governance()
        registration_results["faith_system"] = True
        logging.info("Faith system registered with Nyx governance")
    except Exception as e:
        logging.error(f"Error registering faith system: {e}")
        registration_results["faith_system"] = False
    
    # Register lore evolution system
    try:
        lore_evolution = LoreEvolutionSystem(user_id, conversation_id)
        await lore_evolution.initialize_governance()
        registration_results["lore_evolution"] = True
        logging.info("Lore evolution system registered with Nyx governance")
    except Exception as e:
        logging.error(f"Error registering lore evolution system: {e}")
        registration_results["lore_evolution"] = False
    
    # Register emergent lore system
    try:
        emergent_lore = EmergentLoreSystem(user_id, conversation_id)
        await emergent_lore.register_with_governance()
        registration_results["emergent_lore"] = True
        logging.info("Emergent lore system registered with Nyx governance")
    except Exception as e:
        logging.error(f"Error registering emergent lore system: {e}")
        registration_results["emergent_lore"] = False
    
    # Issue directives for cooperative operation
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
    """
    Initialize all necessary database tables for enhanced lore systems.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Success status
    """
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
    """
    Generate initial enhanced lore for a new game world.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary of all generated lore
    """
    # Track generated lore
    generated_lore = {}
    
    # Generate faith system
    try:
        faith_system = FaithSystem(user_id, conversation_id)
        await faith_system.initialize_governance()
        
        # Create run context
        run_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        
        faith_data = await faith_system.generate_complete_faith_system(run_ctx)
        generated_lore["faith_system"] = faith_data
        logging.info(f"Generated faith system with {len(faith_data.get('deities', []))} deities")
    except Exception as e:
        logging.error(f"Error generating faith system: {e}")
        generated_lore["faith_system_error"] = str(e)
    
    # Generate cyclical events for the year
    try:
        emergent_lore = EmergentLoreSystem(user_id, conversation_id)
        await emergent_lore.initialize_governance()
        
        # Create run context
        run_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        
        # Generate seasonal events
        seasonal_events = []
        for season in ["spring", "summer", "fall", "winter"]:
            event = await emergent_lore.generate_cyclical_event(run_ctx, season=season)
            seasonal_events.append(event)
        
        # Generate solstice and equinox events
        solstice_event = await emergent_lore.generate_cyclical_event(run_ctx, season="summer", is_solstice=True)
        equinox_event = await emergent_lore.generate_cyclical_event(run_ctx, season="spring", is_equinox=True)
        
        seasonal_events.extend([solstice_event, equinox_event])
        generated_lore["seasonal_events"] = seasonal_events
        logging.info(f"Generated {len(seasonal_events)} seasonal events")
    except Exception as e:
        logging.error(f"Error generating seasonal events: {e}")
        generated_lore["seasonal_events_error"] = str(e)
    
    return generated_lore

async def evolve_world_over_time(user_id: int, conversation_id: int, days_passed: int = 30) -> Dict[str, Any]:
    """
    Evolve the world over time, simulating the passage of days or weeks.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        days_passed: Number of days to simulate
        
    Returns:
        Dictionary of evolution results
    """
    evolution_results = {}
    
    # Use LoreEvolutionSystem to mature lore
    try:
        lore_evolution = LoreEvolutionSystem(user_id, conversation_id)
        await lore_evolution.initialize_governance()
        
        maturation = await lore_evolution.mature_lore_over_time(days_passed)
        evolution_results["lore_maturation"] = maturation
        logging.info(f"Matured lore over {days_passed} days with {maturation.get('changes_applied', 0)} changes")
    except Exception as e:
        logging.error(f"Error maturing lore: {e}")
        evolution_results["lore_maturation_error"] = str(e)
    
    # Generate random emergent events based on time passed
    try:
        emergent_lore = EmergentLoreSystem(user_id, conversation_id)
        await emergent_lore.initialize_governance()
        
        # Create run context
        run_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        
        # Scale number of events with time passed, but randomize
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
    
    # Maybe add a new nation if enough time has passed
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


class EmergentLoreSystem:
    """
    System for generating emergent lore, events, and developments
    to ensure the world evolves organically over time.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.faith_system = FaithSystem(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_emergent_event",
        action_description="Generating emergent world event",
        id_from_context=lambda ctx: "emergent_lore"
    )
    async def generate_emergent_event(self, ctx) -> Dict[str, Any]:
        """
        Generate a random emergent event in the world with governance oversight.
        
        Returns:
            Event details and lore impact
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get current world state for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get active factions
                factions = await conn.fetch("""
                    SELECT id, name, type, description
                    FROM Factions
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get some nations
                nations = await conn.fetch("""
                    SELECT id, name, government_type
                    FROM Nations
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get some locations
                locations = await conn.fetch("""
                    SELECT id, location_name, description
                    FROM Locations
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Convert to lists
                faction_data = [dict(faction) for faction in factions]
                nation_data = [dict(nation) for nation in nations]
                location_data = [dict(location) for location in locations]
        
        # Determine event type
        event_types = [
            "political_shift", "military_conflict", "natural_disaster",
            "cultural_development", "technological_advancement", "religious_event",
            "economic_change", "diplomatic_incident"
        ]
        
        # Weight certain events based on available data
        weights = [1] * len(event_types)
        
        # More likely to generate faction-related events if we have factions
        if faction_data:
            faction_event_indices = [0, 1, 3, 6, 7]  # political, military, cultural, economic, diplomatic
            for idx in faction_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # More likely to generate nation-related events if we have nations
        if nation_data:
            nation_event_indices = [0, 1, 7]  # political, military, diplomatic
            for idx in nation_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # More likely to generate location-related events if we have locations
        if location_data:
            location_event_indices = [2, 3, 5]  # natural, cultural, religious
            for idx in location_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # Select event type
        event_type = random.choices(event_types, weights=weights, k=1)[0]
        
        # Create a prompt based on event type
        if event_type == "political_shift":
            entities = faction_data if faction_data else nation_data
            prompt = f"""
            Generate a political shift event for this world:
            
            POTENTIAL ENTITIES INVOLVED:
            {json.dumps(entities, indent=2)}
            
            Create a significant political shift event that:
            1. Changes power dynamics in a meaningful way
            2. Is specific and detailed enough to impact the world
            3. Reinforces or challenges matriarchal power structures
            4. Could lead to interesting narrative developments
            
            Return a JSON object with:
            - event_name: Name of the event
            - event_type: "political_shift"
            - description: Detailed description
            - entities_involved: Array of entity names involved
            - instigator: The primary instigating entity
            - immediate_consequences: Array of immediate consequences
            - potential_long_term_effects: Array of potential long-term effects
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "military_conflict":
            prompt = f"""
            Generate a military conflict event for this world:
            
            POTENTIAL NATIONS INVOLVED:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS INVOLVED:
            {json.dumps(faction_data, indent=2)}
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            Create a military conflict that:
            1. Involves clear aggressor and defender sides
            2. Has a specific cause and objective
            3. Creates opportunities for heroism and character development
            4. Shows how warfare operates in a matriarchal society
            
            Return a JSON object with:
            - event_name: Name of the conflict
            - event_type: "military_conflict"
            - description: Detailed description
            - aggressor: Entity that initiated the conflict
            - defender: Entity defending against aggression
            - location: Where the conflict is primarily occurring
            - cause: What caused the conflict
            - scale: Scale of the conflict (skirmish, battle, war, etc.)
            - current_status: Current status of the conflict
            - casualties: Description of casualties
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "natural_disaster":
            prompt = f"""
            Generate a natural disaster event for this world:
            
            POTENTIAL LOCATIONS AFFECTED:
            {json.dumps(location_data, indent=2)}
            
            Create a natural disaster that:
            1. Has a significant impact on the affected area
            2. Could have supernatural or magical elements
            3. Creates opportunities for societal response that highlights values
            4. Changes the physical landscape in some way
            
            Return a JSON object with:
            - event_name: Name of the disaster
            - event_type: "natural_disaster"
            - description: Detailed description
            - disaster_type: Type of disaster (earthquake, flood, magical storm, etc.)
            - primary_location: Primary location affected
            - secondary_locations: Array of secondarily affected locations
            - severity: Severity level (1-10)
            - immediate_impact: Description of immediate impact
            - response: How communities and leadership are responding
            - supernatural_elements: Any supernatural aspects (if applicable)
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "cultural_development":
            prompt = f"""
            Generate a cultural development event for this world:
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a cultural development that:
            1. Introduces a new tradition, art form, or cultural practice
            2. Comes from a specific community or demographic
            3. Has meaning within the matriarchal power structure
            4. Could spread to other communities
            
            Return a JSON object with:
            - event_name: Name of the cultural development
            - event_type: "cultural_development"
            - description: Detailed description
            - development_type: Type of development (tradition, art form, practice, etc.)
            - originating_group: Group where it originated
            - significance: Social significance
            - symbolism: Symbolic meaning
            - reception: How different groups have received it
            - spread_potential: Likelihood of spreading (1-10)
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "technological_advancement":
            prompt = f"""
            Generate a technological or magical advancement event for this world:
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a technological or magical advancement that:
            1. Changes how something is done in the world
            2. Was developed by specific individuals or groups
            3. Has both benefits and potential drawbacks
            4. Reinforces matriarchal control of knowledge or resources
            
            Return a JSON object with:
            - event_name: Name of the advancement
            - event_type: "technological_advancement"
            - description: Detailed description
            - advancement_type: Type (technological, magical, alchemical, etc.)
            - creator: Who created or discovered it
            - applications: Potential applications
            - limitations: Current limitations
            - control: Who controls access to this advancement
            - societal_impact: How it impacts society
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "religious_event":
            prompt = f"""
            Generate a religious event for this world:
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            Create a religious event that:
            1. Has significance within an existing faith system
            2. Could be interpreted as divine intervention or revelation
            3. Affects religious practices or beliefs
            4. Reinforces the feminine divine nature of the world
            
            Return a JSON object with:
            - event_name: Name of the religious event
            - event_type: "religious_event"
            - description: Detailed description
            - event_category: Type of event (miracle, prophecy, divine manifestation, etc.)
            - location: Where it occurred
            - witnesses: Who witnessed it
            - religious_significance: Significance to believers
            - skeptic_explanation: How skeptics explain it
            - faith_impact: How it impacts faith practices
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        elif event_type == "economic_change":
            prompt = f"""
            Generate an economic change event for this world:
            
            POTENTIAL NATIONS:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create an economic change that:
            1. Shifts resource distribution or trade patterns
            2. Affects multiple groups or regions
            3. Creates winners and losers
            4. Shows how economic power relates to matriarchal control
            
            Return a JSON object with:
            - event_name: Name of the economic change
            - event_type: "economic_change"
            - description: Detailed description
            - change_type: Type of change (trade shift, resource discovery, currency change, etc.)
            - primary_causes: What caused the change
            - beneficiaries: Who benefits
            - disadvantaged: Who is disadvantaged
            - resource_involved: Primary resource or commodity involved
            - wealth_redistribution: How wealth is being redistributed
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        else:  # diplomatic_incident
            prompt = f"""
            Generate a diplomatic incident for this world:
            
            POTENTIAL NATIONS:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a diplomatic incident that:
            1. Creates tension between two or more groups
            2. Stems from a specific action or misunderstanding
            3. Requires diplomatic resolution
            4. Highlights cultural differences or values
            
            Return a JSON object with:
            - event_name: Name of the diplomatic incident
            - event_type: "diplomatic_incident"
            - description: Detailed description
            - parties_involved: Array of involved parties
            - instigating_action: What triggered the incident
            - cultural_factors: Cultural factors at play
            - severity: Severity level (1-10)
            - potential_resolutions: Possible ways to resolve it
            - current_status: Current diplomatic status
            - affected_lore_categories: Array of lore categories this would affect
            """
        
        # Create an agent for event generation
        event_agent = Agent(
            name="EmergentEventAgent",
            instructions="You create emergent world events for fantasy settings.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(event_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            event_data = json.loads(response_text)
            
            # Create a unified event description for lore evolution
            event_description = event_data.get("description", "")
            event_name = event_data.get("event_name", "Unnamed Event")
            
            # Use the LoreEvolutionSystem to evolve the lore based on this event
            lore_evolution = LoreEvolutionSystem(self.user_id, self.conversation_id)
            lore_updates = await lore_evolution.evolve_lore_with_event(
                f"{event_name}: {event_description}"
            )
            
            # Add the lore updates to the event data
            event_data["lore_updates"] = lore_updates
            
            # Return the combined data
            return event_data
            
        except Exception as e:
            logging.error(f"Error generating emergent event: {e}")
            return {"error": str(e)}
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_cyclical_event",
        action_description="Generating cyclical seasonal event",
        id_from_context=lambda ctx: "emergent_lore"
    )
    async def generate_cyclical_event(self, ctx, season: str = None, is_solstice: bool = False, is_equinox: bool = False) -> Dict[str, Any]:
        """
        Generate a cyclical seasonal event with governance oversight.
        
        Args:
            season: Optional season (spring, summer, fall, winter)
            is_solstice: Whether this is a solstice event
            is_equinox: Whether this is an equinox event
            
        Returns:
            Cyclical event details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Set default season if not provided
        if not season:
            seasons = ["spring", "summer", "fall", "winter"]
            season = random.choice(seasons)
            
        # Get relevant data for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get cultural elements
                cultural_elements = await conn.fetch("""
                    SELECT name, type, description
                    FROM CulturalElements
                    WHERE type IN ('festival', 'tradition', 'ceremony', 'ritual')
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get religious practices
                religious_practices = await conn.fetch("""
                    SELECT name, practice_type, description, frequency
                    FROM ReligiousPractices
                    WHERE frequency LIKE '%annual%' OR frequency LIKE '%seasonal%'
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get suitable locations
                locations = await conn.fetch("""
                    SELECT location_name, description
                    FROM Locations
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Convert to lists
                cultural_data = [dict(element) for element in cultural_elements]
                religious_data = [dict(practice) for practice in religious_practices]
                location_data = [dict(location) for location in locations]
        
        # Determine event type and special modifiers
        event_subtypes = ["festival", "ritual", "harvest", "hunt", "pilgrimage", "market", "ceremony"]
        
        if is_solstice:
            event_subtypes.extend(["celestial alignment", "divine manifestation", "magical surge"])
        if is_equinox:
            event_subtypes.extend(["balance ritual", "transition ceremony", "world renewal"])
            
        if season == "spring":
            event_subtypes.extend(["planting", "fertility rite", "renewal ceremony"])
        elif season == "summer":
            event_subtypes.extend(["sun celebration", "competition", "coming of age"])
        elif season == "fall":
            event_subtypes.extend(["harvest festival", "ancestor veneration", "preparation ritual"])
        elif season == "winter":
            event_subtypes.extend(["endurance trial", "darkness ritual", "shelter ceremony"])
            
        event_subtype = random.choice(event_subtypes)
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a cyclical {season} {event_subtype} event for a matriarchal fantasy world.
        
        EXISTING CULTURAL ELEMENTS:
        {json.dumps(cultural_data, indent=2)}
        
        EXISTING RELIGIOUS PRACTICES:
        {json.dumps(religious_data, indent=2)}
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        SPECIAL MODIFIERS:
        Solstice Event: {"Yes" if is_solstice else "No"}
        Equinox Event: {"Yes" if is_equinox else "No"}
        
        Create a cyclical event that:
        1. Is tied to the {season} season specifically
        2. Reinforces matriarchal power and feminine divine connection
        3. Involves community participation with specific roles
        4. Has cultural and possibly religious significance
        
        Return a JSON object with:
        - event_name: Name of the cyclical event
        - event_type: "cyclical_event"
        - subtype: The specific type of event (festival, ritual, etc.)
        - season: The associated season
        - description: Detailed description
        - frequency: How often it occurs
        - duration: How long it lasts
        - locations: Where it's celebrated
        - primary_participants: Who leads or conducts the event
        - community_role: How the community participates
        - religious_significance: Any religious meaning
        - cultural_significance: Cultural meaning and importance
        - material_components: Required materials or preparations
        - traditions: Specific traditions associated with this event
        - gender_roles: How gender influences participation
        """
        
        # Create an agent for cyclical event generation
        cyclical_agent = Agent(
            name="CyclicalEventAgent",
            instructions="You create seasonal festivals and cyclical events for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(cyclical_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            event_data = json.loads(response_text)
            
            # Store this as a cultural element in the database
            try:
                # Prepare the data
                name = event_data.get("event_name", f"{season.capitalize()} {event_subtype.capitalize()}")
                description = event_data.get("description", "")
                
                cultural_description = description
                if "cultural_significance" in event_data:
                    cultural_description += f"\n\nCultural Significance: {event_data['cultural_significance']}"
                if "traditions" in event_data:
                    traditions = event_data.get("traditions", [])
                    if isinstance(traditions, list):
                        cultural_description += f"\n\nTraditions: {', '.join(traditions)}"
                    else:
                        cultural_description += f"\n\nTraditions: {traditions}"
                        
                event_data["stored_as_cultural_element"] = True
                
                # Add to CulturalElements
                element_id = await self.lore_manager.add_cultural_element(
                    name=name,
                    element_type=event_data.get("subtype", "festival"),
                    description=cultural_description,
                    practiced_by=event_data.get("locations", []),
                    significance=8,  # Cyclical events are quite significant
                    historical_origin=f"This {event_data.get('subtype', 'event')} developed as a {season} celebration."
                )
                
                event_data["cultural_element_id"] = element_id
                
                # If it has religious significance, also add as a religious practice
                if "religious_significance" in event_data and event_data.get("religious_significance"):
                    religious_description = description
                    religious_description += f"\n\nReligious Significance: {event_data['religious_significance']}"
                    
                    # Add to ReligiousPractices
                    practice_id = await self.faith_system.add_religious_practice(
                        run_ctx,
                        name=name,
                        practice_type=event_data.get("subtype", "festival"),
                        description=religious_description,
                        purpose="seasonal celebration",
                        frequency=f"Annual ({season})",
                        required_elements=event_data.get("material_components", [])
                    )
                    
                    event_data["religious_practice_id"] = practice_id
                    event_data["stored_as_religious_practice"] = True
            except Exception as e:
                logging.error(f"Error storing cyclical event: {e}")
                event_data["storage_error"] = str(e)
            
            return event_data
            
        except Exception as e:
            logging.error(f"Error generating cyclical event: {e}")
            return {"error": str(e)}
    
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
            nation_data = json.loads(response_text)
            
            # Create the nation in the geopolitical system
            from lore.enhanced_lore import GeopoliticalSystemManager
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
            figure_data = json.loads(response_text)
            
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
            evolution_data = json.loads(response_text)
            
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
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await self.initialize_governance()
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="emergent_lore",
            agent_instance=self
        )
        
        # Issue a directive for emergent lore
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="emergent_lore",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Generate emergent lore and events to ensure the world evolves organically over time.",
                "scope": "world_building"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"EmergentLoreSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")


class LoreExpansionSystem:
    """
    System for expanding lore beyond what was initially generated,
    adding new elements as needed for world coherence and depth.
    """
    
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
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_faction",
        action_description="Generating additional faction for the world",
        id_from_context=lambda ctx: "lore_expansion"
    )
    async def generate_additional_faction(self, ctx, faction_type: str = None) -> Dict[str, Any]:
        """
        Generate an additional faction with governance oversight.
        
        Args:
            faction_type: Optional type of faction to generate
            
        Returns:
            New faction details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get existing factions for context
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get existing factions
                existing_factions = await conn.fetch("""
                    SELECT name, type, description, values, goals
                    FROM Factions
                    LIMIT 10
                """)
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, government_type, matriarchy_level
                    FROM Nations
                    LIMIT 5
                """)
                
                # Get some cultural elements for context
                cultural_elements = await conn.fetch("""
                    SELECT name, type, description
                    FROM CulturalElements
                    ORDER BY RANDOM()
                    LIMIT 5
                """)
                
                # Convert to lists
                faction_data = [dict(faction) for faction in existing_factions]
                nation_data = [dict(nation) for nation in nations]
                cultural_data = [dict(element) for element in cultural_elements]
        
        # If faction_type not specified, select one that's underrepresented
        if not faction_type:
            faction_types = ["political", "religious", "criminal", "mercantile", "academic", "military", "social"]
            existing_types = [f.get("type", "").lower() for f in faction_data]
            
            # Count each type
            type_counts = {}
            for t in faction_types:
                type_counts[t] = sum(1 for et in existing_types if t in et)
            
            # Find underrepresented types
            min_count = min(type_counts.values()) if type_counts else 0
            underrepresented = [t for t, c in type_counts.items() if c == min_count]
            
            # Choose one randomly from underrepresented
            faction_type = random.choice(underrepresented if underrepresented else faction_types)
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a new {faction_type} faction for a matriarchal fantasy world:
        
        EXISTING FACTIONS:
        {json.dumps(faction_data, indent=2)}
        
        NATIONS:
        {json.dumps(nation_data, indent=2)}
        
        CULTURAL CONTEXT:
        {json.dumps(cultural_data, indent=2)}
        
        Create a {faction_type} faction that:
        1. Is distinct from existing factions
        2. Has a clear role and purpose in the world
        3. Has interesting values and goals
        4. Reflects matriarchal power dynamics
        5. Has potential for interesting stories and conflicts
        
        Return a JSON object with:
        - name: Name of the faction
        - type: Type of faction (should include "{faction_type}")
        - description: Detailed description
        - values: Array of core values
        - goals: Array of primary goals
        - headquarters: Main location or headquarters
        - founding_story: How the faction was founded
        - rivals: Array of rival factions or groups
        - allies: Array of allied factions or groups
        - territory: Areas where they operate
        - resources: Key resources they control
        - hierarchy_type: Structure of leadership
        - secret_knowledge: Any secret knowledge they possess
        - public_reputation: How they're viewed publicly
        - color_scheme: Colors associated with them
        - symbol_description: Their emblem or symbol
        """
        
        # Create an agent for faction generation
        faction_agent = Agent(
            name="FactionGenerationAgent",
            instructions="You create factions for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(faction_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            faction_data = json.loads(response_text)
            
            # Add faction to database
            faction_id = await self.lore_manager.add_faction(
                name=faction_data.get("name", f"Unnamed {faction_type.capitalize()} Faction"),
                faction_type=faction_data.get("type", faction_type),
                description=faction_data.get("description", ""),
                values=faction_data.get("values", []),
                goals=faction_data.get("goals", []),
                headquarters=faction_data.get("headquarters"),
                founding_story=faction_data.get("founding_story"),
                rivals=faction_data.get("rivals", []),
                allies=faction_data.get("allies", []),
                territory=faction_data.get("territory"),
                resources=faction_data.get("resources", []),
                hierarchy_type=faction_data.get("hierarchy_type"),
                secret_knowledge=faction_data.get("secret_knowledge"),
                public_reputation=faction_data.get("public_reputation"),
                color_scheme=faction_data.get("color_scheme"),
                symbol_description=faction_data.get("symbol_description")
            )
            
            # Update with the ID
            faction_data["id"] = faction_id
            
            return faction_data
            
        except Exception as e:
            logging.error(f"Error generating additional faction: {e}")
            return {"error": str(e)}
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_locations",
        action_description="Generating additional locations for the world",
        id_from_context=lambda ctx: "lore_expansion"
    )
    async def generate_additional_locations(self, ctx, location_types: List[str] = None, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate additional locations with governance oversight.
        
        Args:
            location_types: Optional types of locations to generate
            count: Number of locations to generate
            
        Returns:
            List of new location details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Set default location types if not provided
        if not location_types:
            location_types = ["settlement", "wilderness", "landmark", "dungeon", "ruin"]
        
        # Get context for location generation
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get existing locations
                existing_locations = await conn.fetch("""
                    SELECT location_name, description
                    FROM Locations
                    LIMIT 10
                """)
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, government_type
                    FROM Nations
                    LIMIT 5
                """)
                
                # Get factions for context
                factions = await conn.fetch("""
                    SELECT name, type
                    FROM Factions
                    LIMIT 5
                """)
                
                # Convert to lists
                location_data = [dict(location) for location in existing_locations]
                nation_data = [dict(nation) for nation in nations]
                faction_data = [dict(faction) for faction in factions]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate {count} new locations for a matriarchal fantasy world.
        
        EXISTING LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        NATIONS:
        {json.dumps(nation_data, indent=2)}
        
        FACTIONS:
        {json.dumps(faction_data, indent=2)}
        
        LOCATION TYPES TO INCLUDE:
        {json.dumps(location_types, indent=2)}
        
        Create {count} locations that:
        1. Are diverse in type and purpose
        2. Include detailed descriptions
        3. Have connections to factions, nations, or existing locations
        4. Reflect matriarchal power dynamics in their design or purpose
        5. Have potential for interesting stories and encounters
        
        Return a JSON array where each location has:
        - name: Name of the location
        - type: Type of location (should be one from the provided types)
        - description: Detailed description
        - controlling_faction: Which faction controls it (if any)
        - nation: Which nation it's in (if any)
        - notable_features: Array of notable features
        - hidden_secrets: Array of hidden secrets
        - strategic_importance: Any strategic importance
        - population: Description of population (if inhabited)
        - points_of_interest: Array of interesting places within
        """
        
        # Create an agent for location generation
        location_agent = Agent(
            name="LocationGenerationAgent",
            instructions="You create locations for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(location_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            locations = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(locations, list):
                if isinstance(locations, dict):
                    locations = [locations]
                else:
                    locations = []
            
            # Add each location
            created_locations = []
            for location in locations:
                try:
                    # Update the database in the standard way using the expected tables
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    
                    # Add to Locations table
                    cursor.execute("""
                        INSERT INTO Locations (user_id, conversation_id, location_name, description)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (self.user_id, self.conversation_id, location.get("name"), location.get("description")))
                    
                    location_id = cursor.fetchone()[0]
                    
                    # Add to LocationLore table
                    cursor.execute("""
                        INSERT INTO LocationLore (
                            location_id, founding_story, hidden_secrets, local_legends, historical_significance
                        )
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        location_id,
                        f"This {location.get('type', 'location')} was established as {location.get('name')}.",
                        location.get("hidden_secrets", []),
                        [],  # local_legends
                        location.get("strategic_importance", "")
                    ))
                    
                    conn.commit()
                    
                    # Update the location with its ID
                    location["id"] = location_id
                    created_locations.append(location)
                    
                except Exception as e:
                    logging.error(f"Error adding location {location.get('name')}: {e}")
                finally:
                    cursor.close()
                    conn.close()
            
            return created_locations
            
        except Exception as e:
            logging.error(f"Error generating additional locations: {e}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_cultural_elements",
        action_description="Generating additional cultural elements",
        id_from_context=lambda ctx: "lore_expansion"
    )
    async def generate_additional_cultural_elements(self, ctx, element_types: List[str] = None, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate additional cultural elements with governance oversight.
        
        Args:
            element_types: Optional types of elements to generate
            count: Number of elements to generate
            
        Returns:
            List of new cultural element details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Set default element types if not provided
        if not element_types:
            element_types = ["tradition", "custom", "taboo", "holiday", "ceremony", "social norm", "art form"]
        
        # Get context for cultural element generation
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get existing cultural elements
                existing_elements = await conn.fetch("""
                    SELECT name, type, description
                    FROM CulturalElements
                    LIMIT 10
                """)
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, cultural_traits
                    FROM Nations
                    LIMIT 5
                """)
                
                # Get factions for context
                factions = await conn.fetch("""
                    SELECT name, type, values
                    FROM Factions
                    LIMIT 5
                """)
                
                # Convert to lists
                element_data = [dict(element) for element in existing_elements]
                nation_data = [dict(nation) for nation in nations]
                faction_data = [dict(faction) for faction in factions]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate {count} new cultural elements for a matriarchal fantasy world.
        
        EXISTING CULTURAL ELEMENTS:
        {json.dumps(element_data, indent=2)}
        
        NATIONS AND THEIR CULTURAL TRAITS:
        {json.dumps(nation_data, indent=2)}
        
        FACTIONS AND THEIR VALUES:
        {json.dumps(faction_data, indent=2)}
        
        ELEMENT TYPES TO INCLUDE:
        {json.dumps(element_types, indent=2)}
        
        Create {count} cultural elements that:
        1. Are diverse in type and purpose
        2. Include detailed descriptions
        3. Clearly connect to groups that practice them
        4. Reflect matriarchal power dynamics
        5. Feel authentic to a richly developed culture
        
        Return a JSON array where each element has:
        - name: Name of the cultural element
        - type: Type of element (should be one from the provided types)
        - description: Detailed description
        - practiced_by: Array of groups who practice this element
        - historical_origin: How the element originated
        - significance: Importance from 1-10
        - related_elements: Any related cultural elements
        - gender_aspects: How gender influences this element
        """
        
        # Create an agent for cultural element generation
        cultural_agent = Agent(
            name="CulturalElementAgent",
            instructions="You create cultural elements for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(cultural_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            elements = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(elements, list):
                if isinstance(elements, dict):
                    elements = [elements]
                else:
                    elements = []
            
            # Add each cultural element
            created_elements = []
            for element in elements:
                try:
                    # Add to database using lore_manager
                    element_id = await self.lore_manager.add_cultural_element(
                        name=element.get("name", f"Unnamed {element.get('type', 'Custom')}"),
                        element_type=element.get("type", "custom"),
                        description=element.get("description", ""),
                        practiced_by=element.get("practiced_by", []),
                        significance=element.get("significance", 5),
                        historical_origin=element.get("historical_origin")
                    )
                    
                    # Update the element with its ID
                    element["id"] = element_id
                    created_elements.append(element)
                    
                except Exception as e:
                    logging.error(f"Error adding cultural element {element.get('name')}: {e}")
            
            return created_elements
            
        except Exception as e:
            logging.error(f"Error generating additional cultural elements: {e}")
            return []
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await self.initialize_governance()
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_expansion",
            agent_instance=self
        )
        
        # Issue a directive for lore expansion
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_expansion",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Expand lore with new elements as needed to maintain world coherence and depth.",
                "scope": "world_building"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"LoreExpansionSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")


# Update registration function to include the new system
async def register_all_enhanced_lore_systems(user_id: int, conversation_id: int) -> Dict[str, bool]:
    """
    Register all enhanced lore systems with Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary of registration results
    """
    # Get the Nyx governance system
    governance = await get_central_governance(user_id, conversation_id)
    
    # Track registration results
    registration_results = {}
    
    # Register faith system
    try:
        faith_system = FaithSystem(user_id, conversation_id)
        await faith_system.register_with_governance()
        registration_results["faith_system"] = True
        logging.info("Faith system registered with Nyx governance")
    except Exception as e:
        logging.error(f"Error registering faith system: {e}")
        registration_results["faith_system"] = False
    
    # Register lore evolution system
    try:
        lore_evolution = LoreEvolutionSystem(user_id, conversation_id)
        await lore_evolution.initialize_governance()
        registration_results["lore_evolution"] = True
        logging.info("Lore evolution system registered with Nyx governance")
    except Exception as e:
        logging.error(f"Error registering lore evolution system: {e}")
        registration_results["lore_evolution"] = False
    
    # Register emergent lore system
    try:
        emergent_lore = EmergentLoreSystem(user_id, conversation_id)
        await emergent_lore.register_with_governance()
        registration_results["emergent_lore"] = True
        logging.info("Emergent lore system registered with Nyx governance")
    except Exception as e:
        logging.error(f"Error registering emergent lore system: {e}")
        registration_results["emergent_lore"] = False
    
    # Register lore expansion system
    try:
        lore_expansion = LoreExpansionSystem(user_id, conversation_id)
        await lore_expansion.register_with_governance()
        registration_results["lore_expansion"] = True
        logging.info("Lore expansion system registered with Nyx governance")
    except Exception as e:
        logging.error(f"Error registering lore expansion system: {e}")
        registration_results["lore_expansion"] = False
    
    # Issue directives for cooperative operation
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
