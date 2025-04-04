# lore/managers/lore_update_manager.py

import logging
import json
import re
import random
from typing import Dict, List, Any, Optional, Tuple, Set
import asyncio
from datetime import datetime

from agents import Agent, Runner
from agents.run_context import RunContextWrapper

from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance

from embedding.vector_store import generate_embedding
from lore.core.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils

class LoreUpdateManager(BaseLoreManager):
    """
    Manager for updating lore elements based on events with sophisticated
    cascade effects and societal impact calculations.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "lore_update"
    
    async def _initialize_tables(self):
        """Initialize tables needed for lore updates"""
        table_definitions = {
            "LoreUpdates": """
                CREATE TABLE LoreUpdates (
                    id SERIAL PRIMARY KEY,
                    lore_id VARCHAR(255) NOT NULL,
                    old_description TEXT NOT NULL,
                    new_description TEXT NOT NULL,
                    update_reason TEXT NOT NULL,
                    impact_level INTEGER CHECK (impact_level BETWEEN 1 AND 10),
                    is_cascade_update BOOLEAN DEFAULT FALSE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                );
                
                CREATE INDEX IF NOT EXISTS idx_loreupdates_lore_id
                ON LoreUpdates(lore_id);
            """,
            
            "LoreUpdateErrors": """
                CREATE TABLE LoreUpdateErrors (
                    id SERIAL PRIMARY KEY,
                    lore_id VARCHAR(255) NOT NULL,
                    error_message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_data JSONB
                );
            """,
            
            "NarrativeInteractions": """
                CREATE TABLE NarrativeInteractions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    stability_impact INTEGER,
                    power_change TEXT,
                    public_perception TEXT,
                    affected_elements JSONB,
                    update_count INTEGER
                );
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="handle_narrative_event",
        action_description="Handling narrative event impacts",
        id_from_context=lambda ctx: "lore_update_manager"
    )
    async def handle_narrative_event(
        self, 
        ctx,
        event_description: str,
        affected_lore_ids: List[str] = None,
        player_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle impacts of a narrative event on the world
        
        Args:
            event_description: Description of the event that occurred
            affected_lore_ids: Optional list of specifically affected lore IDs
            player_data: Optional player character data
            
        Returns:
            Dictionary with all updates applied
        """
        # If no specific lore IDs provided, determine affected elements automatically
        if not affected_lore_ids:
            affected_lore_ids = await self._determine_affected_elements(event_description)
        
        # Fetch the lore elements
        affected_elements = await self._fetch_elements_by_ids(affected_lore_ids)
        
        # Generate updates for these elements
        updates = await self.generate_lore_updates(
            ctx,
            affected_elements=affected_elements,
            event_description=event_description,
            player_character=player_data
        )
        
        # Apply the updates to the database
        await self._apply_lore_updates(updates)
        
        # Check if the event should affect conflicts or domestic issues
        event_impact = await self._calculate_event_impact(event_description)
        
        # If significant impact, evolve conflicts and issues
        if event_impact > 6:
            # Get world politics manager for conflict evolution
            from lore.managers.politics import WorldPoliticsManager
            politics_manager = WorldPoliticsManager(self.user_id, self.conversation_id)
            await politics_manager.ensure_initialized()
            
            # Evolve conflicts and domestic issues
            evolution_results = await politics_manager.evolve_all_conflicts(ctx, days_passed=7)
            
            # Add these results to the updates
            updates.append({
                "type": "conflict_evolution",
                "results": evolution_results
            })
        
        return {
            "event": event_description,
            "updates": updates,
            "update_count": len(updates),
            "event_impact": event_impact
        }
    
    async def _determine_affected_elements(self, event_description: str) -> List[str]:
        """Determine which elements would be affected by this event"""
        # Use NLP or keyword matching to find relevant elements
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Simple keyword based search
                words = re.findall(r'\b\w+\b', event_description.lower())
                significant_words = [w for w in words if len(w) > 3]
                
                if not significant_words:
                    # Fallback to get some random important elements
                    elements = await conn.fetch("""
                        SELECT lore_id FROM LoreElements
                        WHERE importance > 7
                        LIMIT 3
                    """)
                    return [e['lore_id'] for e in elements]
                
                # Search for elements matching keywords
                placeholders = ', '.join(f'${i+1}' for i in range(len(significant_words)))
                query = f"""
                    SELECT DISTINCT lore_id FROM LoreElements
                    WHERE (name ILIKE ANY(ARRAY[{placeholders}]) 
                        OR description ILIKE ANY(ARRAY[{placeholders}]))
                    LIMIT 5
                """
                
                search_terms = [f'%{word}%' for word in significant_words] * 2
                elements = await conn.fetch(query, *search_terms)
                
                return [e['lore_id'] for e in elements]
    
    async def _fetch_elements_by_ids(self, element_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch lore elements by their IDs"""
        if not element_ids:
            return []
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                elements = await conn.fetch("""
                    SELECT lore_id, name, lore_type, description
                    FROM LoreElements
                    WHERE lore_id = ANY($1)
                """, element_ids)
                
                return [dict(elem) for elem in elements]
    
    async def _apply_lore_updates(self, updates: List[Dict[str, Any]]) -> None:
        """Apply updates to the lore database"""
        if not updates:
            return
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for update in updates:
                        if 'is_cascade_update' in update or 'lore_id' not in update:
                            continue  # Skip cascade updates or invalid updates
                            
                        # Update the element description
                        await conn.execute("""
                            UPDATE LoreElements 
                            SET description = $1
                            WHERE lore_id = $2
                        """, update['new_description'], update['lore_id'])
                        
                        # Add update record
                        await conn.execute("""
                            INSERT INTO LoreUpdates (
                                lore_id, old_description, new_description, 
                                update_reason, impact_level, timestamp
                            ) VALUES ($1, $2, $3, $4, $5, $6)
                        """, 
                        update['lore_id'],
                        update['old_description'],
                        update['new_description'],
                        update['update_reason'],
                        update['impact_level'],
                        update['timestamp'])
                        
                        # Update any type-specific tables
                        if update['lore_type'] == 'character':
                            # Example of character-specific updates
                            char_dev = update.get('character_development', {})
                            if char_dev:
                                await conn.execute("""
                                    UPDATE Characters
                                    SET confidence = $1, resolve = $2, ambition = $3
                                    WHERE character_id = $4
                                """,
                                char_dev.get('confidence', 5),
                                char_dev.get('resolve', 5),
                                char_dev.get('ambition', 5),
                                update['lore_id'])
    
    async def _calculate_event_impact(self, event_description: str) -> int:
        """Calculate the general impact level of an event"""
        # Simple keyword-based analysis
        # In a full implementation, this would use NLP
        
        high_impact = ['war', 'death', 'revolution', 'disaster', 'coronation', 'marriage', 'birth', 'conquest']
        medium_impact = ['conflict', 'dispute', 'challenge', 'treaty', 'alliance', 'ceremony', 'festival']
        low_impact = ['meeting', 'conversation', 'journey', 'meal', 'performance', 'minor']
        
        words = set(event_description.lower().split())
        
        high_count = sum(1 for word in high_impact if word in words)
        medium_count = sum(1 for word in medium_impact if word in words)
        low_count = sum(1 for word in low_impact if word in words)
        
        # Calculate base impact
        if high_count > 0:
            return 8 + min(high_count, 2)  # Max 10
        elif medium_count > 0:
            return 5 + min(medium_count, 2)  # Max 7
        elif low_count > 0:
            return 2 + min(low_count, 2)  # Max 4
        else:
            return 5  # Default medium impact
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_lore_updates",
        action_description="Generating lore updates for event",
        id_from_context=lambda ctx: "lore_update_manager"
    )
    async def generate_lore_updates(
        self, 
        ctx,
        affected_elements: List[Dict[str, Any]], 
        event_description: str,
        player_character: Dict[str, Any] = None,
        dominant_npcs: List[Dict[str, Any]] = None,
        world_state: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate sophisticated updates for affected lore elements
        
        Args:
            affected_elements: List of affected lore elements
            event_description: Description of the event
            player_character: Optional player character data to provide context
            dominant_npcs: Optional list of ruling NPCs relevant to the event
            world_state: Optional current world state data
            
        Returns:
            List of detailed updates to apply with cascading effects
        """
        # Create run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        updates = []
        relationship_changes = {}
        power_shifts = {}
        
        # Track elements that will need secondary updates due to relationships
        cascading_elements = set()
        
        # Retrieve world context from database if not provided
        if not world_state:
            world_state = await self._fetch_world_state()
        
        # Determine societal consequences of the event
        societal_impact = await self._calculate_societal_impact(
            event_description, 
            world_state.get('stability_index', 8),
            world_state.get('power_hierarchy', {})
        )
        
        # Generate updates for each element
        for element in affected_elements:
            # Retrieve relationship network for this element
            related_elements = await self._fetch_related_elements(element['lore_id'])
            
            # Determine element's position in power hierarchy
            hierarchy_position = await self._get_hierarchy_position(element)
            
            # Build contextual history of recent updates to this element
            update_history = await self._fetch_element_update_history(
                element['lore_id'], 
                limit=5
            )
            
            # Create agent for update generation
            update_agent = Agent(
                name="MatriarchalLoreAgent",
                instructions="""
                You update narrative elements in a matriarchal society RPG setting.
                Focus on power dynamics, authority shifts, and social consequences.
                Maintain internal consistency while allowing for character development.
                Ensure updates reflect the established hierarchy and social order.
                Consider how changes cascade through relationship networks.
                """,
                model="o3-mini"
            )
            
            # Create prompt for the agent
            prompt = await self._build_update_prompt(
                element=element,
                event_description=event_description,
                societal_impact=societal_impact,
                related_elements=related_elements,
                hierarchy_position=hierarchy_position,
                update_history=update_history,
                player_character=player_character,
                dominant_npcs=dominant_npcs
            )
            
            # Get response from agent
            try:
                result = await Runner.run(update_agent, prompt, context=run_ctx.context)
                
                # Parse the response
                update_data = await self._parse_update_response(result.final_output, element)
                
                # Calculate cascading effects
                cascade_effects = await self._calculate_cascade_effects(
                    element, 
                    update_data, 
                    related_elements
                )
                
                # Track relationship changes
                for rel_id, change in cascade_effects.get('relationship_changes', {}).items():
                    if rel_id in relationship_changes:
                        relationship_changes[rel_id] += change
                    else:
                        relationship_changes[rel_id] = change
                    
                    # Add affected relationships to cascade list
                    if abs(change) > 2:  # Only cascade significant changes
                        cascading_elements.add(rel_id)
                
                # Track power shifts
                for faction_id, shift in cascade_effects.get('power_shifts', {}).items():
                    if faction_id in power_shifts:
                        power_shifts[faction_id] += shift
                    else:
                        power_shifts[faction_id] = shift
                
                # Create the update record
                update_record = {
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': update_data['new_description'],
                    'update_reason': update_data['update_reason'],
                    'impact_level': update_data['impact_level'],
                    'timestamp': datetime.now().isoformat()
                }
                
                updates.append(update_record)
                
            except Exception as e:
                logging.error(f"Error generating update for {element['name']}: {str(e)}")
                await self._log_update_error(element['lore_id'], str(e), ctx.context)
        
        # Process cascading updates
        if cascading_elements:
            # Fetch the cascading elements
            cascade_element_data = await self._fetch_elements_by_ids(list(cascading_elements))
            
            # Generate cascade updates
            cascade_updates = await self._generate_cascade_updates(
                cascade_element_data,
                event_description,
                relationship_changes,
                power_shifts
            )
            
            # Add cascade updates to the main updates list
            updates.extend(cascade_updates)
        
        # Update world state with power shifts
        if power_shifts:
            await self._update_world_state(power_shifts)
        
        # Log narrative interactions for future reference
        await self._log_narrative_interactions(updates, societal_impact)
        
        return updates
    
    # Helper methods
    
    async def _build_update_prompt(
        self,
        element: Dict[str, Any],
        event_description: str,
        societal_impact: Dict[str, Any],
        related_elements: List[Dict[str, Any]],
        hierarchy_position: int,
        update_history: List[Dict[str, Any]],
        player_character: Dict[str, Any] = None,
        dominant_npcs: List[Dict[str, Any]] = None
    ) -> str:
        """Build a prompt for lore update generation"""
        # Format update history as context
        history_context = ""
        if update_history:
            history_items = []
            for update in update_history:
                history_items.append(f"- {update['timestamp']}: {update['update_reason']}")
            history_context = "UPDATE HISTORY:\n" + "\n".join(history_items)
        
        # Format related elements as context
        relationships_context = ""
        if related_elements:
            rel_items = []
            for rel in related_elements:
                rel_items.append(f"- {rel['name']} ({rel['lore_type']}): {rel['relationship_type']} - {rel['relationship_strength']}/10")
            relationships_context = "RELATIONSHIPS:\n" + "\n".join(rel_items)
        
        # Format player character context if available
        player_context = ""
        if player_character:
            player_context = f"""
            PLAYER CHARACTER CONTEXT:
            Name: {player_character['name']}
            Status: {player_character['status']}
            Recent Actions: {player_character['recent_actions']}
            Position in Hierarchy: {player_character.get('hierarchy_position', 'subordinate')}
            """
        
        # Format dominant NPCs context if available
        dominant_context = ""
        if dominant_npcs:
            dom_items = []
            for npc in dominant_npcs:
                dom_items.append(f"- {npc['name']}: {npc['position']} - {npc['attitude']} toward situation")
            dominant_context = "RELEVANT AUTHORITY FIGURES:\n" + "\n".join(dom_items)
        
        # Hierarchy-appropriate directive
        hierarchy_directive = await self._get_hierarchy_directive(hierarchy_position)
        
        # Build the complete prompt
        prompt = f"""
        The following lore element in our matriarchal-themed RPG world requires updating based on recent events:
        
        LORE ELEMENT:
        Type: {element['lore_type']}
        Name: {element['name']}
        Current Description: {element['description']}
        Position in Hierarchy: {hierarchy_position}/10 (lower number = higher authority)
        
        {relationships_context}
        
        {history_context}
        
        EVENT THAT OCCURRED:
        {event_description}
        
        SOCIETAL IMPACT ASSESSMENT:
        Stability Impact: {societal_impact['stability_impact']}/10
        Power Structure Change: {societal_impact['power_structure_change']}
        Public Perception Shift: {societal_impact['public_perception']}
        
        {player_context}
        
        {dominant_context}
        
        {hierarchy_directive}
        
        Generate a sophisticated update for this lore element that incorporates the impact of this event.
        The update should maintain narrative consistency while allowing for meaningful development.
        
        Return your response as a JSON object with:
        {{
            "new_description": "The updated description that reflects event impact",
            "update_reason": "Detailed explanation of why this update makes sense",
            "impact_level": A number from 1-10 indicating how significantly this event affects this element
        }}
        """
        
        return prompt
    
    async def _get_hierarchy_directive(self, hierarchy_position: int) -> str:
        """Get an appropriate directive based on the element's position in hierarchy"""
        if hierarchy_position <= 2:
            return """
            DIRECTIVE: This element represents a highest-tier authority figure. 
            Their decisions significantly impact the world. 
            They rarely change their core principles but may adjust strategies.
            They maintain control and authority in all situations.
            """
        elif hierarchy_position <= 4:
            return """
            DIRECTIVE: This element represents high authority.
            They have significant influence but answer to the highest tier.
            They strongly maintain the established order while pursuing their ambitions.
            They assert dominance in their domain but show deference to higher authority.
            """
        elif hierarchy_position <= 7:
            return """
            DIRECTIVE: This element has mid-level authority.
            They implement the will of higher authorities while managing those below.
            They may have personal aspirations but function within established boundaries.
            They balance compliance with higher authority against control of subordinates.
            """
        else:
            return """
            DIRECTIVE: This element has low authority in the hierarchy.
            They follow directives from above and have limited autonomy.
            They may seek to improve their position but must navigate carefully.
            They show appropriate deference to those of higher status.
            """
    
    async def _parse_update_response(self, response_text: str, element: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM response with error handling"""
        try:
            # First try to parse as JSON
            update_data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['new_description', 'update_reason', 'impact_level']
            for field in required_fields:
                if field not in update_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return update_data
            
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON response for {element['name']}")
            
            # Try regex extraction for common patterns
            patterns = {
                'new_description': r'"new_description"\s*:\s*"([^"]+)"',
                'update_reason': r'"update_reason"\s*:\s*"([^"]+)"',
                'impact_level': r'"impact_level"\s*:\s*(\d+)'
            }
            
            extracted_data = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    if key == 'impact_level':
                        extracted_data[key] = int(match.group(1))
                    else:
                        extracted_data[key] = match.group(1)
            
            # Fill in missing required fields with defaults
            if 'new_description' not in extracted_data:
                extracted_data['new_description'] = element['description']
            
            if 'update_reason' not in extracted_data:
                extracted_data['update_reason'] = "Event impact (extracted from unstructured response)"
                
            if 'impact_level' not in extracted_data:
                extracted_data['impact_level'] = 5
            
            return extracted_data
    
    async def _calculate_cascade_effects(
        self, 
        element: Dict[str, Any], 
        update_data: Dict[str, Any],
        related_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate how updates to one element affect related elements"""
        cascade_effects = {
            'relationship_changes': {},
            'power_shifts': {}
        }
        
        impact_level = update_data.get('impact_level', 5)
        
        # Calculate relationship changes based on impact
        for related in related_elements:
            rel_id = related['lore_id']
            rel_strength = related.get('relationship_strength', 5)
            rel_type = related.get('relationship_type', 'neutral')
            
            # Calculate relationship change based on event impact and current relationship
            # Higher impact events cause more relationship change
            if rel_type in ['subservient', 'loyal']:
                # Loyal/subservient relationships strengthen during impactful events
                change = (impact_level - 5) * 0.3
            elif rel_type in ['authority', 'dominant']:
                # Authority relationships may weaken slightly during high-impact events
                change = (5 - impact_level) * 0.2
            elif rel_type in ['rival', 'adversarial']:
                # Rivalries intensify during impactful events
                change = -abs(impact_level - 5) * 0.4
            else:
                # Neutral relationships shift based on impact direction
                change = (impact_level - 5) * 0.1
            
            # Adjust for hierarchy differences - larger gaps mean more significant changes
            hierarchy_diff = abs(
                element.get('hierarchy_position', 5) - 
                related.get('hierarchy_position', 5)
            )
            
            change *= (1 + (hierarchy_diff * 0.1))
            
            cascade_effects['relationship_changes'][rel_id] = round(change, 1)
        
        # Calculate power shifts for relevant factions
        if element['lore_type'] == 'faction':
            # Direct power shift for the affected faction
            faction_id = element['lore_id']
            power_shift = (impact_level - 5) * 0.5
            cascade_effects['power_shifts'][faction_id] = power_shift
        
        # Calculate power shifts for factions related to the element
        faction_relations = [r for r in related_elements if r['lore_type'] == 'faction']
        for faction in faction_relations:
            faction_id = faction['lore_id']
            rel_type = faction.get('relationship_type', 'neutral')
            
            # Calculate power shift based on relationship type
            if rel_type in ['allied', 'supportive']:
                # Allied factions shift in the same direction
                shift = (impact_level - 5) * 0.3
            elif rel_type in ['rival', 'opposed']:
                # Rival factions shift in the opposite direction
                shift = (5 - impact_level) * 0.3
            else:
                # Neutral factions have minimal shifts
                shift = (impact_level - 5) * 0.1
                
            cascade_effects['power_shifts'][faction_id] = round(shift, 1)
        
        return cascade_effects
    
    async def _generate_cascade_updates(
        self,
        cascade_elements: List[Dict[str, Any]],
        event_description: str,
        relationship_changes: Dict[str, float],
        power_shifts: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate simplified updates for elements affected indirectly"""
        cascade_updates = []
        
        for element in cascade_elements:
            element_id = element['lore_id']
            
            # Get the relationship change if any
            rel_change = relationship_changes.get(element_id, 0)
            
            # Get power shift if this is a faction
            power_shift = 0
            if element['lore_type'] == 'faction':
                power_shift = power_shifts.get(element_id, 0)
            
            # Calculate impact level based on relationship change and power shift
            impact_level = min(10, max(1, round(5 + abs(rel_change) * 2 + abs(power_shift) * 2)))
            
            # Generate a simplified update
            if abs(rel_change) > 1 or abs(power_shift) > 1:
                # Significant enough to warrant an update
                
                # Determine the nature of the update based on changes
                if element['lore_type'] == 'character':
                    if rel_change > 0:
                        update_reason = f"Strengthened position due to recent events"
                        description_modifier = "more confident and assured"
                    else:
                        update_reason = f"Position weakened by recent events"
                        description_modifier = "more cautious and reserved"
                
                elif element['lore_type'] == 'faction':
                    if power_shift > 0:
                        update_reason = f"Gained influence following recent events"
                        description_modifier = "increasing their authority and reach"
                    else:
                        update_reason = f"Lost influence due to recent events"
                        description_modifier = "adapting to their diminished standing"
                
                elif element['lore_type'] == 'location':
                    if rel_change > 0:
                        update_reason = f"Increased importance after recent events"
                        description_modifier = "now sees more activity and attention"
                    else:
                        update_reason = f"Decreased importance after recent events"
                        description_modifier = "now sees less activity and attention"
                
                else:
                    update_reason = f"Indirectly affected by recent events"
                    description_modifier = "subtly changed by recent developments"
                
                # Create a new description with the modifier
                new_description = element['description']
                if "." in new_description:
                    parts = new_description.split(".")
                    parts[-2] += f", {description_modifier}"
                    new_description = ".".join(parts)
                else:
                    new_description = f"{new_description} {description_modifier}."
                
                # Create update record
                cascade_updates.append({
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': new_description,
                    'update_reason': update_reason,
                    'impact_level': impact_level,
                    'is_cascade_update': True,
                    'timestamp': datetime.now().isoformat()
                })
        
        return cascade_updates
    
    async def _calculate_societal_impact(
        self,
        event_description: str,
        stability_index: int,
        power_hierarchy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate the societal impact of an event"""
        # Analyze event text for impact keywords
        impact_keywords = {
            'high_impact': [
                'overthrown', 'revolution', 'usurped', 'conquered', 'rebellion',
                'assassination', 'coup', 'catastrophe', 'disaster'
            ],
            'medium_impact': [
                'challenge', 'conflict', 'dispute', 'tension', 'unrest',
                'scandal', 'controversy', 'uprising', 'demonstration'
            ],
            'low_impact': [
                'minor', 'small', 'limited', 'isolated', 'contained',
                'private', 'personal', 'individual', 'trivial'
            ]
        }
        
        # Count keyword occurrences
        high_count = sum(1 for word in impact_keywords['high_impact'] if word.lower() in event_description.lower())
        medium_count = sum(1 for word in impact_keywords['medium_impact'] if word.lower() in event_description.lower())
        low_count = sum(1 for word in impact_keywords['low_impact'] if word.lower() in event_description.lower())
        
        # Calculate base stability impact
        if high_count > 0:
            base_stability_impact = 7 + min(high_count, 3)
        elif medium_count > 0:
            base_stability_impact = 4 + min(medium_count, 3)
        elif low_count > 0:
            base_stability_impact = 2 + min(low_count, 2)
        else:
            base_stability_impact = 3  # Default moderate impact
        
        # Adjust for current stability
        # Higher stability means events have less impact
        stability_modifier = (10 - stability_index) / 10
        adjusted_impact = base_stability_impact * (0.5 + stability_modifier)
        
        # Determine power structure change
        if adjusted_impact >= 8:
            power_change = "significant realignment of authority"
        elif adjusted_impact >= 6:
            power_change = "moderate shift in power dynamics"
        elif adjusted_impact >= 4:
            power_change = "subtle adjustments to authority structures"
        else:
            power_change = "minimal change to established order"
        
        # Determine public perception
        if adjusted_impact >= 7:
            if "rebellion" in event_description.lower() or "uprising" in event_description.lower():
                perception = "widespread questioning of authority"
            else:
                perception = "significant public concern"
        elif adjusted_impact >= 5:
            perception = "notable public interest and discussion"
        else:
            perception = "limited public awareness or interest"
        
        return {
            'stability_impact': round(adjusted_impact),
            'power_structure_change': power_change,
            'public_perception': perception
        }
    
    # Utility methods
    
    async def _fetch_world_state(self) -> Dict[str, Any]:
        """Fetch current world state from database"""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                world_state = await conn.fetchrow("""
                    SELECT * FROM WorldState 
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                
                if world_state:
                    return dict(world_state)
                else:
                    # Return default values if no world state found
                    return {
                        'stability_index': 8,
                        'narrative_tone': 'dramatic',
                        'power_dynamics': 'strict_hierarchy',
                        'power_hierarchy': {}
                    }
    
    async def _fetch_related_elements(self, lore_id: str) -> List[Dict[str, Any]]:
        """Fetch elements related to the given lore ID"""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                related = await conn.fetch("""
                    SELECT e.lore_id, e.name, e.lore_type, r.relationship_type, r.relationship_strength 
                    FROM LoreElements e
                    JOIN LoreRelationships r ON e.lore_id = r.target_id
                    WHERE r.source_id = $1
                """, lore_id)
                
                return [dict(rel) for rel in related]
    
    async def _get_hierarchy_position(self, element: Dict[str, Any]) -> int:
        """Determine element's position in the power hierarchy"""
        # If the element has a hierarchy position, use it
        if 'hierarchy_position' in element:
            return element['hierarchy_position']
        
        # Otherwise, make an educated guess based on lore type
        if element['lore_type'] == 'character':
            # Check name keywords for character
            name = element['name'].lower()
            if any(title in name for title in ['queen', 'empress', 'matriarch', 'high', 'supreme']):
                return 1
            elif any(title in name for title in ['princess', 'duchess', 'lady', 'noble']):
                return 3
            elif any(title in name for title in ['advisor', 'minister', 'council']):
                return 5
            else:
                return 8
        elif element['lore_type'] == 'faction':
            # Check faction importance
            if 'importance' in element:
                return max(1, 10 - element['importance'])
            else:
                return 4  # Default for factions
        elif element['lore_type'] == 'location':
            # Check location significance
            if 'significance' in element:
                return max(1, 10 - element['significance'])
            else:
                return 6  # Default for locations
        else:
            return 5  # Default middle position
    
    async def _fetch_element_update_history(self, lore_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent update history for an element"""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                history = await conn.fetch("""
                    SELECT timestamp, update_reason
                    FROM LoreUpdates
                    WHERE lore_id = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """, lore_id, limit)
                
                return [dict(update) for update in history]
    
    async def _log_update_error(self, lore_id: str, error: str, context: Dict[str, Any]) -> None:
        """Log an error that occurred during lore update"""
        logging.error(f"Error updating lore element {lore_id}: {error}")
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO LoreUpdateErrors (
                        lore_id, error_message, timestamp, context_data
                    ) VALUES ($1, $2, $3, $4)
                """, lore_id, error, datetime.now(), json.dumps(context))
    
    async def _update_world_state(self, power_shifts: Dict[str, float]) -> None:
        """Update the world state to reflect power shifts"""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get current power hierarchy
                world_state = await self._fetch_world_state()
                power_hierarchy = world_state.get('power_hierarchy', {})
                
                # Update power values
                for faction_id, shift in power_shifts.items():
                    current_power = power_hierarchy.get(faction_id, 5)
                    new_power = max(1, min(10, current_power + shift))
                    power_hierarchy[faction_id] = new_power
                
                # Update the database
                await conn.execute("""
                    UPDATE WorldState
                    SET power_hierarchy = $1,
                        last_updated = $2
                    WHERE user_id = $3 AND conversation_id = $4
                """, json.dumps(power_hierarchy), datetime.now(), 
                self.user_id, self.conversation_id)
    
    async def _log_narrative_interactions(self, updates: List[Dict[str, Any]], societal_impact: Dict[str, Any]) -> None:
        """Log the narrative interactions for future reference"""
        if not updates:
            return
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO NarrativeInteractions (
                        timestamp, stability_impact, power_change, 
                        public_perception, affected_elements, update_count
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                datetime.now(),
                societal_impact.get('stability_impact', 5),
                societal_impact.get('power_structure_change', 'minimal'),
                societal_impact.get('public_perception', 'limited'),
                json.dumps([u.get('lore_id') for u in updates]),
                len(updates))
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_update_manager",
            directive_text="Update lore elements based on events with sophisticated cascade effects.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
