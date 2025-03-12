import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

import asyncpg
from db.connection import get_db_connection
from logic.npc_creation import process_daily_npc_activities
from logic.calendar import load_calendar_names
from logic.time_cycle import get_current_time
from logic.chatgpt_integration import get_chatgpt_response
from logic.stats_logic import apply_stat_change
from utils.caching import CONFLICT_CACHE

logger = logging.getLogger(__name__)

class ConflictManager:
    """
    Manages the Dynamic Conflict System that generates, updates, and resolves
    layered conflicts within the game. This system creates an ongoing narrative 
    structure that responds to player choices and evolves the game world over time.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the Conflict Manager with user and conversation context."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Constants from the design document
        self.CONFLICT_TYPES = ["major", "minor", "standard", "catastrophic"]
        self.CONFLICT_PHASES = ["brewing", "active", "climax", "resolution"]
        self.INVOLVEMENT_LEVELS = ["none", "observing", "participating", "leading"]
        self.BASE_SUCCESS_RATES = {
            "none": 0,
            "observing": 30,
            "participating": 40,
            "leading": 50
        }
        
        # Configuration
        self.MAJOR_CONFLICT_INTERVAL = 7  # Generate major conflict every 7 days if none active
        self.MINOR_CONFLICT_INTERVAL = 3  # Generate minor conflict every 3 days during major conflicts
        self.MAX_STANDARD_CONFLICTS = 3   # Maximum number of standard conflicts at once
        self.CATASTROPHIC_CHANCE = 0.15   # 15% chance for catastrophic conflicts
        self.NPC_REQUEST_CHANCE = 0.3     # 30% chance per day for NPC request
    
    async def get_connection_pool(self) -> asyncpg.Pool:
        """Get a connection pool for database operations."""
        return await asyncpg.create_pool(dsn=get_db_connection())
    
    async def get_current_game_time(self) -> Tuple[int, int, int, str]:
        """Get the current game time."""
        current_year, current_month, current_day, time_of_day = get_current_time(
            self.user_id, self.conversation_id
        )
        return current_year, current_month, current_day, time_of_day
    
    async def generate_conflict(self, conflict_type: str = None) -> Dict[str, Any]:
        """
        Generate a new conflict of the specified type, or determine the appropriate type
        based on the current game state if none specified.
        """
        # Get current game time
        year, month, day, time_of_day = await self.get_current_game_time()
        
        # Determine conflict type if not specified
        if conflict_type is None:
            conflict_type = await self._determine_conflict_type(day)
        
        # Check for consequence chain triggers - conflicts that arise from past decisions
        consequence_chain = await self._check_consequence_chains()
        if consequence_chain and not conflict_type:
            conflict_type = consequence_chain.get('recommended_type', conflict_type)
            trigger_info = consequence_chain.get('trigger_info', {})
            logger.info(f"Conflict triggered by consequence chain: {trigger_info}")
        else:
            trigger_info = None
        
        # Get environment description for context
        environment_desc = await self._get_environment_description()
        
        # Get current active conflicts for context
        active_conflicts = await self.get_active_conflicts()
        
        # Build context for the GPT call
        context = {
            "environment_description": environment_desc,
            "year": year,
            "month": month,
            "day": day,
            "time_of_day": time_of_day,
            "conflict_type": conflict_type,
            "active_conflicts": active_conflicts,
            "triggered_by_consequence": trigger_info
        }
        
        # Generate conflict details using GPT
        conflict_details = await self._generate_conflict_details(context)
        
        # Save the conflict to the database
        conflict_id = await self._save_conflict(conflict_details, conflict_type, day)
        
        # Associate NPCs with the conflict
        await self._associate_npcs_with_conflict(conflict_id, conflict_details)
        
        # Create initial memory event for the conflict
        await self._create_conflict_memory(
            conflict_id, 
            f"A new {conflict_type} conflict has begun: {conflict_details['name']}",
            7  # significance
        )
        
        # Return complete conflict information
        return await self.get_conflict(conflict_id)
    
    async def _determine_conflict_type(self, current_day: int) -> str:
        """
        Determine what type of conflict should be generated based on current game state.
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check for active major conflicts
                major_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND conflict_type = 'major' AND is_active = TRUE
                """, self.user_id, self.conversation_id)
                
                # Check for active standard conflicts
                standard_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND conflict_type = 'standard' AND is_active = TRUE
                """, self.user_id, self.conversation_id)
                
                # Check when the last major conflict was created
                last_major_day = await conn.fetchval("""
                    SELECT start_day FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND conflict_type = 'major'
                    ORDER BY created_at DESC LIMIT 1
                """, self.user_id, self.conversation_id)
                
                # Logic to determine conflict type
                if major_count == 0 and (last_major_day is None or current_day - last_major_day >= self.MAJOR_CONFLICT_INTERVAL):
                    # Generate major conflict if none active and it's been long enough
                    if random.random() < self.CATASTROPHIC_CHANCE:
                        return "catastrophic"
                    else:
                        return "major"
                elif major_count > 0:
                    # If major conflict active, check if we should generate minor conflict
                    minor_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM Conflicts
                        WHERE user_id = $1 AND conversation_id = $2
                        AND conflict_type = 'minor' AND is_active = TRUE
                    """, self.user_id, self.conversation_id)
                    
                    # Check when the last minor conflict was created
                    last_minor_day = await conn.fetchval("""
                        SELECT start_day FROM Conflicts
                        WHERE user_id = $1 AND conversation_id = $2
                        AND conflict_type = 'minor'
                        ORDER BY created_at DESC LIMIT 1
                    """, self.user_id, self.conversation_id)
                    
                    if last_minor_day is None or current_day - last_minor_day >= self.MINOR_CONFLICT_INTERVAL:
                        return "minor"
                
                # If standard conflicts are below max, generate one
                if standard_count < self.MAX_STANDARD_CONFLICTS:
                    return "standard"
                
                # Default to standard if no other condition triggered
                return "standard"
    
    async def _check_consequence_chains(self) -> Optional[Dict[str, Any]]:
        """
        Check if past conflict resolutions should trigger a new conflict
        based on the Consequence Chain System.
        
        Returns:
            Optional dict with recommended conflict type and trigger info
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check for conflict history entries with grudges
                conflict_history = await conn.fetch("""
                    SELECT ch.id, ch.conflict_id, ch.affected_npc_id, ch.grudge_level,
                           ch.impact_type, ch.narrative_impact, ch.has_triggered_consequence,
                           n.npc_name, c.conflict_name
                    FROM ConflictHistory ch
                    JOIN NPCStats n ON ch.affected_npc_id = n.npc_id
                    JOIN Conflicts c ON ch.conflict_id = c.conflict_id
                    WHERE ch.user_id = $1 AND ch.conversation_id = $2
                    AND ch.grudge_level > 50
                    AND ch.has_triggered_consequence = FALSE
                    ORDER BY ch.grudge_level DESC
                """, self.user_id, self.conversation_id)
                
                if conflict_history:
                    # Pick the highest grudge event
                    history_entry = conflict_history[0]
                    
                    # Mark it as triggered
                    await conn.execute("""
                        UPDATE ConflictHistory
                        SET has_triggered_consequence = TRUE
                        WHERE id = $1
                    """, history_entry['id'])
                    
                    # Determine conflict type based on grudge level
                    if history_entry['grudge_level'] > 80:
                        conflict_type = "major"
                    else:
                        conflict_type = "minor"
                    
                    return {
                        'recommended_type': conflict_type,
                        'trigger_info': {
                            'type': 'grudge',
                            'npc_id': history_entry['affected_npc_id'],
                            'npc_name': history_entry['npc_name'],
                            'original_conflict': history_entry['conflict_name'],
                            'grudge_level': history_entry['grudge_level'],
                            'reason': history_entry['narrative_impact']
                        }
                    }
                
                # If no grudge found, check for multiple-conflict triggers
                # This represents scenarios where multiple past decisions combine
                # to trigger a single new conflict
                
                # Get resolved conflicts from the past 30 game days
                resolved_conflicts = await conn.fetch("""
                    SELECT conflict_id, conflict_name, outcome, 
                           conflict_type, faction_a_name, faction_b_name
                    FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND is_active = FALSE
                    AND outcome IN ('success', 'failure', 'partial_success')
                    AND updated_at > NOW() - INTERVAL '30 days'
                """, self.user_id, self.conversation_id)
                
                if len(resolved_conflicts) >= 3:
                    # Analyze pattern of conflicts
                    outcomes = [c['outcome'] for c in resolved_conflicts[:3]]
                    
                    # If player has had 3 successes in a row, generate a harder conflict
                    if outcomes.count('success') >= 3:
                        return {
                            'recommended_type': 'major',
                            'trigger_info': {
                                'type': 'escalation',
                                'reason': "Multiple successful conflict resolutions have attracted attention from more powerful figures",
                                'pattern': 'multiple_successes'
                            }
                        }
                    
                    # If player has had 3 failures in a row, generate an easier conflict
                    if outcomes.count('failure') >= 3:
                        return {
                            'recommended_type': 'standard',
                            'trigger_info': {
                                'type': 'mercy',
                                'reason': "After multiple failures, a more manageable conflict emerges",
                                'pattern': 'multiple_failures'
                            }
                        }
                
                # No consequence chain triggered
                return None
    
    async def _get_environment_description(self) -> str:
        """Get the current environment description from CurrentRoleplay."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'EnvironmentDesc'
                """, self.user_id, self.conversation_id)
                
                if row:
                    return row['value']
                else:
                    return "A dynamic femdom environment"
    
    async def _generate_conflict_details(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conflict details using GPT."""
        # Create prompt for GPT
        prompt = f"""
        Generate a {context['conflict_type']} conflict for the femdom roleplaying game.
        
        Environment: {context['environment_description']}
        Current date: Year {context['year']}, Month {context['month']}, Day {context['day']}, {context['time_of_day']}
        
        Active conflicts: {', '.join([c['conflict_name'] for c in context['active_conflicts']])}
        
        {"This conflict is triggered by a consequence from past events: " + str(context['triggered_by_consequence']) if context.get('triggered_by_consequence') else ""}
        
        Generate details for a new {context['conflict_type']} conflict, including:
        1. Conflict name
        2. Description
        3. Descriptions for each phase (brewing, active, climax, resolution)
        4. Two opposing factions or forces
        5. Estimated duration in days (2-3 weeks for major, 3-7 days for minor, 1-3 days for standard)
        6. Resources required for resolution (money, supplies, influence)
        7. Potential NPCs who would be involved
        8. Potential consequences for success and failure
        
        Format your response as a structured JSON object with these keys:
        - name: Conflict name (string)
        - description: Overall description (string)
        - brewing_description: Description of brewing phase (string)
        - active_description: Description of active phase (string)
        - climax_description: Description of climax phase (string)
        - resolution_description: Description of resolution phase (string)
        - faction_a: Name of first faction (string)
        - faction_b: Name of second faction (string)
        - estimated_duration: Days to resolve (number)
        - resources_required: Object with money, supplies, and influence values
        - involved_npcs: Array of NPC names who might be involved
        - success_consequences: Array of consequence descriptions for success
        - failure_consequences: Array of consequence descriptions for failure
        """
        
        # Call GPT to generate the conflict details
        response = await get_chatgpt_response(
            self.conversation_id,
            context['environment_description'],
            prompt
        )
        
        # Extract and parse JSON from response
        if response.get('type') == 'function_call':
            return response.get('function_args', {})
        else:
            # Try to extract JSON from text response
            text = response.get('response', '')
            try:
                # Find JSON block in text
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # Fallback to default structure if JSON parsing fails
                    return {
                        "name": f"Unnamed {context['conflict_type']} Conflict",
                        "description": "A mysterious conflict has arisen.",
                        "brewing_description": "Signs of brewing tension appear.",
                        "active_description": "The conflict is now in full swing.",
                        "climax_description": "The conflict reaches its critical point.",
                        "resolution_description": "The conflict concludes.",
                        "faction_a": "Faction A",
                        "faction_b": "Faction B",
                        "estimated_duration": self._get_default_duration(context['conflict_type']),
                        "resources_required": {
                            "money": 100,
                            "supplies": 5,
                            "influence": 20
                        },
                        "involved_npcs": [],
                        "success_consequences": ["The situation improves."],
                        "failure_consequences": ["The situation worsens."]
                    }
            except Exception as e:
                logger.error(f"Failed to parse conflict details from GPT response: {e}")
                # Return default structure
                return {
                    "name": f"Unnamed {context['conflict_type']} Conflict",
                    "description": "A mysterious conflict has arisen.",
                    "brewing_description": "Signs of brewing tension appear.",
                    "active_description": "The conflict is now in full swing.",
                    "climax_description": "The conflict reaches its critical point.",
                    "resolution_description": "The conflict concludes.",
                    "faction_a": "Faction A",
                    "faction_b": "Faction B",
                    "estimated_duration": self._get_default_duration(context['conflict_type']),
                    "resources_required": {
                        "money": 100,
                        "supplies": 5,
                        "influence": 20
                    },
                    "involved_npcs": [],
                    "success_consequences": ["The situation improves."],
                    "failure_consequences": ["The situation worsens."]
                }
    
    def _get_default_duration(self, conflict_type: str) -> int:
        """Get the default duration for a conflict type."""
        if conflict_type == "major":
            return random.randint(14, 21)  # 2-3 weeks
        elif conflict_type == "minor":
            return random.randint(3, 7)    # 3-7 days
        elif conflict_type == "standard":
            return random.randint(1, 3)    # 1-3 days
        elif conflict_type == "catastrophic":
            return random.randint(21, 28)  # 3-4 weeks
        else:
            return 7  # Default
    
    async def _save_conflict(self, conflict_details: Dict[str, Any], conflict_type: str, start_day: int) -> int:
        """Save the conflict to the database and return its ID."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Determine parent conflict ID for minor conflicts
                parent_id = None
                if conflict_type == "minor":
                    parent_row = await conn.fetchrow("""
                        SELECT conflict_id FROM Conflicts
                        WHERE user_id = $1 AND conversation_id = $2
                        AND conflict_type = 'major' AND is_active = TRUE
                        ORDER BY created_at DESC LIMIT 1
                    """, self.user_id, self.conversation_id)
                    
                    if parent_row:
                        parent_id = parent_row['conflict_id']
                
                # Set up default resources required based on conflict type
                default_resources = {
                    "major": {"money": 500, "supplies": 15, "influence": 50},
                    "minor": {"money": 200, "supplies": 8, "influence": 25},
                    "standard": {"money": 100, "supplies": 5, "influence": 10},
                    "catastrophic": {"money": 1000, "supplies": 30, "influence": 100}
                }
                
                # Get resources from conflict details or use defaults
                resources = conflict_details.get('resources_required', default_resources.get(conflict_type, {}))
                
                # Insert conflict record
                conflict_id = await conn.fetchval("""
                    INSERT INTO Conflicts (
                        user_id, conversation_id, conflict_name, conflict_type,
                        parent_conflict_id, description, brewing_description, active_description,
                        climax_description, resolution_description, progress, phase,
                        start_day, estimated_duration, faction_a_name, faction_b_name,
                        resources_required, is_active
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
                    ) RETURNING conflict_id
                """, 
                    self.user_id, self.conversation_id,
                    conflict_details.get('name', f"Unnamed {conflict_type} Conflict"),
                    conflict_type,
                    parent_id,
                    conflict_details.get('description', "A mysterious conflict has arisen."),
                    conflict_details.get('brewing_description', "Signs of brewing tension appear."),
                    conflict_details.get('active_description', "The conflict is now in full swing."),
                    conflict_details.get('climax_description', "The conflict reaches its critical point."),
                    conflict_details.get('resolution_description', "The conflict concludes."),
                    0,  # Initial progress is 0%
                    "brewing",  # Initial phase is 'brewing'
                    start_day,
                    conflict_details.get('estimated_duration', self._get_default_duration(conflict_type)),
                    conflict_details.get('faction_a', "Faction A"),
                    conflict_details.get('faction_b', "Faction B"),
                    json.dumps(resources),
                    True  # Is active
                )
                
                # Save consequences
                success_consequences = conflict_details.get('success_consequences', [])
                failure_consequences = conflict_details.get('failure_consequences', [])
                
                # Define consequence types
                consequence_types = ["relationship", "stat", "unlock", "permanent"]
                
                # Save success consequences
                for i, consequence in enumerate(success_consequences):
                    # Determine consequence type based on content
                    con_type = self._determine_consequence_type(consequence, consequence_types)
                    # Determine entity type and ID based on context
                    entity_type, entity_id = self._determine_entity(consequence)
                    
                    await conn.execute("""
                        INSERT INTO ConflictConsequences (
                            conflict_id, consequence_type, entity_type, entity_id, description
                        ) VALUES ($1, $2, $3, $4, $5)
                    """, conflict_id, con_type, entity_type, entity_id, f"Success consequence: {consequence}")
                
                # Save failure consequences
                for i, consequence in enumerate(failure_consequences):
                    # Determine consequence type based on content
                    con_type = self._determine_consequence_type(consequence, consequence_types)
                    # Determine entity type and ID based on context
                    entity_type, entity_id = self._determine_entity(consequence)
                    
                    await conn.execute("""
                        INSERT INTO ConflictConsequences (
                            conflict_id, consequence_type, entity_type, entity_id, description
                        ) VALUES ($1, $2, $3, $4, $5)
                    """, conflict_id, con_type, entity_type, entity_id, f"Failure consequence: {consequence}")
                
                return conflict_id
    
    def _determine_consequence_type(self, consequence: str, types: List[str]) -> str:
        """Determine the most likely consequence type based on description."""
        consequence = consequence.lower()
        
        # Check for relationship changes
        if any(term in consequence for term in ["relationship", "trust", "closeness", "bond", "alliance", "respect"]):
            return "relationship"
        
        # Check for stat changes
        if any(term in consequence for term in ["stat", "attribute", "skill", "corruption", "confidence", "willpower", "obedience", "dependency"]):
            return "stat"
        
        # Check for unlocks
        if any(term in consequence for term in ["unlock", "access", "reveal", "discover", "new", "location", "area"]):
            return "unlock"
        
        # Check for permanent changes
        if any(term in consequence for term in ["permanent", "destroy", "death", "die", "dissolve", "banish", "exile", "forever"]):
            return "permanent"
        
        # Default to relationship if can't determine
        return "relationship"
    
    def _determine_entity(self, consequence: str) -> Tuple[str, Optional[int]]:
        """Determine the entity type and ID based on consequence description."""
        consequence = consequence.lower()
        
        # Check for player references
        if any(term in consequence for term in ["you", "your", "player", "chase"]):
            return "player", None
        
        # Check for location references
        if any(term in consequence for term in ["location", "place", "area", "room", "building"]):
            return "location", None
        
        # Check for faction references
        if any(term in consequence for term in ["faction", "group", "organization", "alliance"]):
            return "faction", None
        
        # Default to NPC (we'll set the ID later when we know which NPCs are involved)
        return "npc", None
    
    async def _associate_npcs_with_conflict(self, conflict_id: int, conflict_details: Dict[str, Any]) -> None:
        """Associate NPCs with the conflict based on the generated details."""
        # Extract NPC names from conflict details
        involved_npcs = conflict_details.get('involved_npcs', [])
        if not involved_npcs:
            await self._auto_associate_npcs(conflict_id, conflict_details)
            return
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # For each NPC name, try to find the corresponding NPC in the database
                for npc_name in involved_npcs:
                    # Try to find the NPC by name
                    npc_row = await conn.fetchrow("""
                        SELECT npc_id FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_name = $3
                    """, self.user_id, self.conversation_id, npc_name)
                    
                    if npc_row:
                        npc_id = npc_row['npc_id']
                        
                        # Determine faction and role randomly
                        faction = random.choice(['a', 'b', 'neutral'])
                        role = random.choice(['leader', 'member', 'supporter', 'observer'])
                        influence = random.randint(30, 80)
                        
                        # Associate the NPC with the conflict
                        await conn.execute("""
                            INSERT INTO ConflictNPCs (
                                conflict_id, npc_id, faction, role, influence_level
                            ) VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (conflict_id, npc_id) DO UPDATE
                            SET faction = $3, role = $4, influence_level = $5
                        """, conflict_id, npc_id, faction, role, influence)
                        
                        # Update any conflict consequences targeting "npc" with this npc_id
                        await conn.execute("""
                            UPDATE ConflictConsequences
                            SET entity_id = $1
                            WHERE conflict_id = $2 AND entity_type = 'npc' AND entity_id IS NULL
                            LIMIT 1
                        """, npc_id, conflict_id)
    
    async def _auto_associate_npcs(self, conflict_id: int, conflict_details: Dict[str, Any]) -> None:
        """
        Automatically associate NPCs with the conflict if none were explicitly specified.
        Selects NPCs based on relevance to the conflict theme.
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get conflict details to analyze theme
                conflict_row = await conn.fetchrow("""
                    SELECT conflict_name, description, faction_a_name, faction_b_name, conflict_type
                    FROM Conflicts
                    WHERE conflict_id = $1
                """, conflict_id)
                
                if not conflict_row:
                    return
                
                # Combine text for analysis
                conflict_text = f"{conflict_row['conflict_name']} {conflict_row['description']} {conflict_row['faction_a_name']} {conflict_row['faction_b_name']}"
                conflict_text = conflict_text.lower()
                
                # Get all NPCs
                npc_rows = await conn.fetch("""
                    SELECT npc_id, npc_name, archetype_summary, dominance, cruelty, 
                           affiliations, personality_traits, hobbies
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
                    ORDER BY dominance DESC
                """, self.user_id, self.conversation_id)
                
                # Score NPCs by relevance
                npc_scores = []
                for npc in npc_rows:
                    score = 0
                    
                    # Base score from dominance 
                    score += npc['dominance'] * 0.1
                    
                    # Score from archetype
                    archetype = npc['archetype_summary'] or ""
                    if isinstance(archetype, str):
                        if "domme" in archetype.lower():
                            score += 20
                        if "cruel" in archetype.lower() and "cruel" in conflict_text:
                            score += 15
                        if "sadist" in archetype.lower() and ("punishment" in conflict_text or "pain" in conflict_text):
                            score += 15
                        if "mentor" in archetype.lower() and ("training" in conflict_text or "teaching" in conflict_text):
                            score += 15
                    
                    # Add randomness to avoid always picking the same NPCs
                    score += random.randint(1, 20)
                    
                    npc_scores.append((npc['npc_id'], npc['npc_name'], score))
                
                # Sort by score
                npc_scores.sort(key=lambda x: x[2], reverse=True)
                
                # Determine how many NPCs to associate based on conflict type
                num_npcs = 0
                if conflict_row['conflict_type'] == "major":
                    num_npcs = min(len(npc_scores), 4)
                elif conflict_row['conflict_type'] == "minor":
                    num_npcs = min(len(npc_scores), 3)
                elif conflict_row['conflict_type'] == "catastrophic":
                    num_npcs = min(len(npc_scores), 5)
                else:  # standard
                    num_npcs = min(len(npc_scores), 2)
                
                # Select top NPCs
                selected_npcs = npc_scores[:num_npcs]
                
                # Associate NPCs with conflict
                for i, (npc_id, npc_name, _) in enumerate(selected_npcs):
                    # Assign faction
                    if i % 2 == 0:
                        faction = 'a'
                    else:
                        faction = 'b'
                    
                    # First NPC in each faction is the leader
                    if i == 0 or i == 1:
                        role = 'leader'
                    else:
                        role = random.choice(['member', 'supporter'])
                    
                    # Influence level
                    influence = random.randint(50, 90) if role == 'leader' else random.randint(30, 70)
                    
                    # Associate with conflict
                    await conn.execute("""
                        INSERT INTO ConflictNPCs (
                            conflict_id, npc_id, faction, role, influence_level
                        ) VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (conflict_id, npc_id) DO UPDATE
                        SET faction = $3, role = $4, influence_level = $5
                    """, conflict_id, npc_id, faction, role, influence)
                    
                    # Associate with consequences
                    await conn.execute("""
                        UPDATE ConflictConsequences
                        SET entity_id = $1
                        WHERE conflict_id = $2 AND entity_type = 'npc' AND entity_id IS NULL
                        LIMIT 1
                    """, npc_id, conflict_id)
                    
                    # Create memory event
                    await self._create_conflict_memory(
                        conflict_id,
                        f"{npc_name} has become involved in the conflict as a {role} for faction {faction}",
                        5
                    )
    
    async def _create_conflict_memory(self, conflict_id: int, memory_text: str, significance: int = 5) -> None:
        """Create a memory event for the conflict."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ConflictMemoryEvents (
                        conflict_id, memory_text, significance, entity_type, entity_id
                    ) VALUES ($1, $2, $3, $4, $5)
                """, conflict_id, memory_text, significance, "player", None)
    
    async def get_conflict(self, conflict_id: int) -> Dict[str, Any]:
        """Get complete information about a specific conflict."""
        cache_key = f"conflict:{self.user_id}:{self.conversation_id}:{conflict_id}"
        cached = CONFLICT_CACHE.get(cache_key)
        if cached:
            return cached
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get conflict information
                conflict_row = await conn.fetchrow("""
                    SELECT * FROM Conflicts
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, self.user_id, self.conversation_id)
                
                if not conflict_row:
                    return None
                
                # Convert to dictionary
                conflict = dict(conflict_row)
                
                # Parse JSON fields
                if isinstance(conflict['resources_required'], str):
                    conflict['resources_required'] = json.loads(conflict['resources_required'])
                
                # Get involved NPCs
                conflict['involved_npcs'] = []
                npc_rows = await conn.fetch("""
                    SELECT cn.*, ns.npc_name 
                    FROM ConflictNPCs cn
                    JOIN NPCStats ns ON cn.npc_id = ns.npc_id
                    WHERE cn.conflict_id = $1
                """, conflict_id)
                
                for npc_row in npc_rows:
                    conflict['involved_npcs'].append(dict(npc_row))
                
                # Get consequences
                conflict['consequences'] = []
                consequence_rows = await conn.fetch("""
                    SELECT * FROM ConflictConsequences
                    WHERE conflict_id = $1
                """, conflict_id)
                
                for consequence_row in consequence_rows:
                    conflict['consequences'].append(dict(consequence_row))
                
                # Get player involvement
                involvement_row = await conn.fetchrow("""
                    SELECT * FROM PlayerConflictInvolvement
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, self.user_id, self.conversation_id)
                
                if involvement_row:
                    conflict['player_involvement'] = dict(involvement_row)
                    
                    # Parse JSON fields
                    if isinstance(conflict['player_involvement']['actions_taken'], str):
                        conflict['player_involvement']['actions_taken'] = json.loads(conflict['player_involvement']['actions_taken'])
                else:
                    conflict['player_involvement'] = {
                        "involvement_level": "none",
                        "faction": "neutral",
                        "money_committed": 0,
                        "supplies_committed": 0,
                        "influence_committed": 0,
                        "actions_taken": []
                    }
                
                # Get memory events
                conflict['memory_events'] = []
                memory_rows = await conn.fetch("""
                    SELECT * FROM ConflictMemoryEvents
                    WHERE conflict_id = $1
                    ORDER BY created_at DESC
                """, conflict_id)
                
                for memory_row in memory_rows:
                    conflict['memory_events'].append(dict(memory_row))
                
                # Cache result
                CONFLICT_CACHE.set(cache_key, conflict, 30)  # Cache for 30 seconds
                
                return conflict
    
    async def get_active_conflicts(self) -> List[Dict[str, Any]]:
        """Get all active conflicts."""
        cache_key = f"active_conflicts:{self.user_id}:{self.conversation_id}"
        cached = CONFLICT_CACHE.get(cache_key)
        if cached:
            return cached
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                conflict_rows = await conn.fetch("""
                    SELECT conflict_id, conflict_name, conflict_type, description, progress, phase,
                        start_day, estimated_duration, faction_a_name, faction_b_name,
                        is_active, created_at
                    FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2 AND is_active = TRUE
                    ORDER BY conflict_type, created_at
                """, self.user_id, self.conversation_id)
                
                conflicts = [dict(row) for row in conflict_rows]
                
                # Cache result
                CONFLICT_CACHE.set(cache_key, conflicts, 30)  # Cache for 30 seconds
                
                return conflicts
    
    async def update_conflict_progress(self, conflict_id: int, progress_increment: float) -> Dict[str, Any]:
        """
        Update the progress of a conflict and potentially change its phase.
        Returns the updated conflict information.
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get current conflict info
                conflict_row = await conn.fetchrow("""
                    SELECT progress, phase FROM Conflicts
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, self.user_id, self.conversation_id)
                
                if not conflict_row:
                    return None
                
                current_progress = conflict_row['progress']
                current_phase = conflict_row['phase']
                
                # Calculate new progress
                new_progress = min(100, current_progress + progress_increment)
                
                # Determine if phase should change
                new_phase = current_phase
                if current_phase == "brewing" and new_progress >= 25:
                    new_phase = "active"
                    await self._create_conflict_memory(
                        conflict_id, 
                        f"The conflict has moved from the brewing phase to the active phase.",
                        6
                    )
                elif current_phase == "active" and new_progress >= 75:
                    new_phase = "climax"
                    await self._create_conflict_memory(
                        conflict_id, 
                        f"The conflict has reached its climax phase.",
                        7
                    )
                elif current_phase == "climax" and new_progress >= 95:
                    new_phase = "resolution"
                    await self._create_conflict_memory(
                        conflict_id, 
                        f"The conflict is now in its resolution phase.",
                        8
                    )
                
                # Update the conflict
                await conn.execute("""
                    UPDATE Conflicts
                    SET progress = $1, phase = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE conflict_id = $3 AND user_id = $4 AND conversation_id = $5
                """, new_progress, new_phase, conflict_id, self.user_id, self.conversation_id)
                
                # Clear cache
                CONFLICT_CACHE.remove(f"conflict:{self.user_id}:{self.conversation_id}:{conflict_id}")
                CONFLICT_CACHE.remove(f"active_conflicts:{self.user_id}:{self.conversation_id}")
                
                # Return updated conflict
                return await self.get_conflict(conflict_id)
    
    async def set_player_involvement(
        self, 
        conflict_id: int, 
        involvement_level: str,
        faction: str = "neutral",
        money_committed: int = 0,
        supplies_committed: int = 0,
        influence_committed: int = 0,
        action_taken: str = None
    ) -> Dict[str, Any]:
        """
        Set the player's involvement in a conflict.
        Returns the updated conflict information.
        """
        if involvement_level not in self.INVOLVEMENT_LEVELS:
            return {"error": f"Invalid involvement level. Must be one of: {', '.join(self.INVOLVEMENT_LEVELS)}"}
        
        if faction not in ["a", "b", "neutral"]:
            return {"error": "Invalid faction. Must be 'a', 'b', or 'neutral'"}
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if involvement record exists
                involvement_row = await conn.fetchrow("""
                    SELECT id, actions_taken, involvement_level FROM PlayerConflictInvolvement
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, self.user_id, self.conversation_id)
                
                actions = []
                prev_involvement = None
                if involvement_row:
                    involvement_id = involvement_row['id']
                    prev_involvement = involvement_row['involvement_level']
                    
                    # Parse actions
                    if involvement_row['actions_taken']:
                        try:
                            if isinstance(involvement_row['actions_taken'], str):
                                actions = json.loads(involvement_row['actions_taken'])
                            else:
                                actions = involvement_row['actions_taken']
                        except:
                            actions = []
                    
                    # Update existing record
                    await conn.execute("""
                        UPDATE PlayerConflictInvolvement
                        SET involvement_level = $1, faction = $2,
                            money_committed = money_committed + $3,
                            supplies_committed = supplies_committed + $4,
                            influence_committed = influence_committed + $5,
                            actions_taken = $6,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = $7
                    """, 
                        involvement_level, faction,
                        money_committed, supplies_committed, influence_committed,
                        json.dumps(actions + ([action_taken] if action_taken else [])),
                        involvement_id
                    )
                else:
                    # Create new record
                    await conn.execute("""
                        INSERT INTO PlayerConflictInvolvement (
                            conflict_id, user_id, conversation_id, player_name,
                            involvement_level, faction, money_committed,
                            supplies_committed, influence_committed, actions_taken
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                        conflict_id, self.user_id, self.conversation_id, "Chase",
                        involvement_level, faction, money_committed,
                        supplies_committed, influence_committed,
                        json.dumps([action_taken] if action_taken else [])
                    )
                
                # Update conflict success rate
                await self._update_conflict_success_rate(
                    conn, conflict_id, involvement_level,
                    money_committed, supplies_committed, influence_committed
                )
                
                # Create memory event if action taken
                if action_taken:
                    await self._create_conflict_memory(
                        conflict_id, 
                        f"Player took action in conflict: {action_taken}",
                        5
                    )
                
                # If involvement level changed, create memory
                if prev_involvement and prev_involvement != involvement_level:
                    await self._create_conflict_memory(
                        conflict_id, 
                        f"Player changed involvement from {prev_involvement} to {involvement_level}",
                        6
                    )
                
                # Clear cache
                CONFLICT_CACHE.remove(f"conflict:{self.user_id}:{self.conversation_id}:{conflict_id}")
                
                # Return updated conflict
                return await self.get_conflict(conflict_id)
    
    async def _update_conflict_success_rate(
        self,
        conn,
        conflict_id: int,
        involvement_level: str,
        money_committed: int,
        supplies_committed: int,
        influence_committed: int
    ) -> None:
        """Update the success rate for a conflict based on player involvement and resources."""
        # Get conflict information
        conflict_row = await conn.fetchrow("""
            SELECT resources_required FROM Conflicts
            WHERE conflict_id = $1
        """, conflict_id)
        
        if not conflict_row:
            return
        
        # Parse resources required
        resources_required = conflict_row['resources_required']
        if isinstance(resources_required, str):
            resources_required = json.loads(resources_required)
        
        # Get player vital stats
        vitals_row = await conn.fetchrow("""
            SELECT energy, hunger FROM PlayerVitals
            WHERE user_id = $1 AND conversation_id = $2
            AND player_name = 'Chase'
        """, self.user_id, self.conversation_id)
        
        vitals_penalty = 0
        if vitals_row:
            energy = vitals_row['energy']
            hunger = vitals_row['hunger']
            
            # Apply penalties if below 30
            if energy < 30:
                vitals_penalty += (30 - energy) * 0.5
            if hunger < 30:
                vitals_penalty += (30 - hunger) * 0.5
        
        # Get NPC assistance - count NPCs in same faction as player
        player_faction = await conn.fetchval("""
            SELECT faction FROM PlayerConflictInvolvement
            WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
        """, conflict_id, self.user_id, self.conversation_id)
        
        npc_bonus = 0
        if player_faction and player_faction != 'neutral':
            npc_count = await conn.fetchval("""
                SELECT COUNT(*) FROM ConflictNPCs
                WHERE conflict_id = $1 AND faction = $2
            """, conflict_id, player_faction)
            
            npc_bonus = npc_count * 10
        
        # Get related minor conflict successes for major conflicts
        minor_success_bonus = 0
        conflict_type = await conn.fetchval("""
            SELECT conflict_type FROM Conflicts
            WHERE conflict_id = $1
        """, conflict_id)
        
        if conflict_type == "major":
            related_minors = await conn.fetch("""
                SELECT outcome FROM Conflicts
                WHERE parent_conflict_id = $1 AND conflict_type = 'minor'
            """, conflict_id)
            
            for minor in related_minors:
                if minor['outcome'] == "success":
                    minor_success_bonus += 15
                elif minor['outcome'] == "partial_success":
                    minor_success_bonus += 7
        
        # Calculate base success rate
        base_success_rate = self.BASE_SUCCESS_RATES.get(involvement_level, 0)
        
        # Calculate resource contribution rates
        money_required = resources_required.get('money', 0)
        supplies_required = resources_required.get('supplies', 0)
        influence_required = resources_required.get('influence', 0)
        
        money_rate = min(1, money_committed / max(1, money_required)) * 20 if money_required > 0 else 0
        supplies_rate = min(1, supplies_committed / max(1, supplies_required)) * 20 if supplies_required > 0 else 0
        influence_rate = min(1, influence_committed / max(1, influence_required)) * 20 if influence_required > 0 else 0
        
        # Calculate final success rate
        success_rate = (
            base_success_rate +
            money_rate +
            supplies_rate +
            influence_rate +
            npc_bonus +
            minor_success_bonus -
            vitals_penalty
        )
        
        # Update success rate
        await conn.execute("""
            UPDATE Conflicts
            SET success_rate = $1
            WHERE conflict_id = $2
        """, success_rate, conflict_id)
    
    async def resolve_conflict(self, conflict_id: int) -> Dict[str, Any]:
        """
        Resolve a conflict and apply consequences based on success rate.
        Returns the resolved conflict information.
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get conflict information
                conflict_row = await conn.fetchrow("""
                    SELECT conflict_name, conflict_type, progress, phase, success_rate, outcome
                    FROM Conflicts
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, self.user_id, self.conversation_id)
                
                if not conflict_row:
                    return {"error": "Conflict not found"}
                
                if conflict_row['outcome'] != "pending":
                    return {"error": "Conflict already resolved"}
                
                # Get player involvement
                involvement_row = await conn.fetchrow("""
                    SELECT involvement_level FROM PlayerConflictInvolvement
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, self.user_id, self.conversation_id)
                
                involvement_level = involvement_row['involvement_level'] if involvement_row else "none"
                
                # Determine outcome based on success rate and involvement
                outcome = "ignored"
                if involvement_level != "none":
                    success_rate = conflict_row['success_rate']
                    
                    if success_rate >= 80:
                        outcome = "success"
                    elif success_rate >= 40:
                        outcome = "partial_success"
                    else:
                        outcome = "failure"
                
                # Update conflict with outcome and mark as inactive
                await conn.execute("""
                    UPDATE Conflicts
                    SET outcome = $1, is_active = FALSE, progress = 100, phase = 'resolution',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE conflict_id = $2
                """, outcome, conflict_id)
                
                # Create resolution memory
                await self._create_conflict_memory(
                    conflict_id, 
                    f"The conflict '{conflict_row['conflict_name']}' has been resolved with outcome: {outcome}",
                    9  # High significance for resolution
                )
                
                # Apply consequences based on outcome
                if outcome == "success":
                    await self._apply_success_consequences(conn, conflict_id, outcome)
                elif outcome == "partial_success":
                    await self._apply_partial_success_consequences(conn, conflict_id, outcome)
                elif outcome == "failure":
                    await self._apply_failure_consequences(conn, conflict_id, outcome)
                
                # Update faction power shifts
                await self._update_faction_power(conn, conflict_id, outcome)
                
                # Record conflict history for consequence chain system
                await self._record_conflict_history(conn, conflict_id, outcome)
                
                # Clear cache
                CONFLICT_CACHE.remove(f"conflict:{self.user_id}:{self.conversation_id}:{conflict_id}")
                CONFLICT_CACHE.remove(f"active_conflicts:{self.user_id}:{self.conversation_id}")
                
                # Get final conflict state
                return await self.get_conflict(conflict_id)
    
    async def _apply_success_consequences(self, conn, conflict_id: int, outcome: str) -> None:
        """Apply success consequences for a conflict."""
        # Get success consequences
        consequence_rows = await conn.fetch("""
            SELECT id, description, consequence_type, entity_type, entity_id 
            FROM ConflictConsequences
            WHERE conflict_id = $1 AND description LIKE 'Success consequence:%'
        """, conflict_id)
        
        # Apply each consequence
        for consequence in consequence_rows:
            description = consequence['description']
            con_type = consequence['consequence_type']
            entity_type = consequence['entity_type']
            entity_id = consequence['entity_id']
            
            # Mark consequence as applied
            await conn.execute("""
                UPDATE ConflictConsequences
                SET applied = TRUE, applied_at = CURRENT_TIMESTAMP
                WHERE id = $1
            """, consequence['id'])
            
            # Create memory event for the consequence
            await self._create_conflict_memory(
                conflict_id, 
                f"Success consequence applied: {description}",
                7
            )
            
            # Apply actual game state changes based on consequence type
            if con_type == "relationship":
                await self._apply_relationship_consequence(conn, entity_type, entity_id, 15, description)
            elif con_type == "stat":
                await self._apply_stat_consequence(conn, entity_type, entity_id, 10, description)
            elif con_type == "unlock":
                await self._apply_unlock_consequence(conn, description)
            elif con_type == "permanent":
                await self._apply_permanent_consequence(conn, entity_type, entity_id, description)
    
    async def _apply_partial_success_consequences(self, conn, conflict_id: int, outcome: str) -> None:
        """Apply partial success consequences for a conflict."""
        # Get success consequences but apply them at reduced effect
        consequence_rows = await conn.fetch("""
            SELECT id, description, consequence_type, entity_type, entity_id 
            FROM ConflictConsequences
            WHERE conflict_id = $1 AND description LIKE 'Success consequence:%'
        """, conflict_id)
        
        # Apply each consequence at reduced effectiveness
        for consequence in consequence_rows:
            description = consequence['description']
            con_type = consequence['consequence_type']
            entity_type = consequence['entity_type']
            entity_id = consequence['entity_id']
            
            # Mark consequence as applied
            await conn.execute("""
                UPDATE ConflictConsequences
                SET applied = TRUE, applied_at = CURRENT_TIMESTAMP
                WHERE id = $1
            """, consequence['id'])
            
            # Create memory event for the consequence
            await self._create_conflict_memory(
                conflict_id, 
                f"Partial success consequence applied: {description}",
                6
            )
            
            # Apply actual game state changes at reduced effectiveness
            if con_type == "relationship":
                await self._apply_relationship_consequence(conn, entity_type, entity_id, 7, description)
            elif con_type == "stat":
                await self._apply_stat_consequence(conn, entity_type, entity_id, 5, description)
            elif con_type == "unlock":
                # 50% chance of unlock for partial success
                if random.random() < 0.5:
                    await self._apply_unlock_consequence(conn, description)
            # No permanent consequences for partial success
    
    async def _apply_failure_consequences(self, conn, conflict_id: int, outcome: str) -> None:
        """Apply failure consequences for a conflict."""
        # Get failure consequences
        consequence_rows = await conn.fetch("""
            SELECT id, description, consequence_type, entity_type, entity_id
            FROM ConflictConsequences
            WHERE conflict_id = $1 AND description LIKE 'Failure consequence:%'
        """, conflict_id)
        
        # Apply each consequence
        for consequence in consequence_rows:
            description = consequence['description']
            con_type = consequence['consequence_type']
            entity_type = consequence['entity_type']
            entity_id = consequence['entity_id']
            
            # Mark consequence as applied
            await conn.execute("""
                UPDATE ConflictConsequences
                SET applied = TRUE, applied_at = CURRENT_TIMESTAMP
                WHERE id = $1
            """, consequence['id'])
            
            # Create memory event for the consequence
            await self._create_conflict_memory(
                conflict_id, 
                f"Failure consequence applied: {description}",
                7
            )
            
            # Apply actual game state changes based on consequence type
            if con_type == "relationship":
                # Negative relationship change
                await self._apply_relationship_consequence(conn, entity_type, entity_id, -15, description)
            elif con_type == "stat":
                # Negative stat change
                await self._apply_stat_consequence(conn, entity_type, entity_id, -10, description)
            elif con_type == "permanent" and random.random() < 0.5:  # 50% chance for permanent failures
                await self._apply_permanent_consequence(conn, entity_type, entity_id, description)
    
    async def _apply_relationship_consequence(self, conn, entity_type: str, entity_id: Optional[int], value: int, description: str) -> None:
        """Apply a relationship change consequence."""
        if entity_type == "npc" and entity_id is not None:
            # Update player-NPC relationship
            await conn.execute("""
                UPDATE SocialLinks
                SET link_level = GREATEST(0, LEAST(100, link_level + $1))
                WHERE (entity1_type = 'player' AND entity1_id = 0 AND entity2_type = 'npc' AND entity2_id = $2)
                   OR (entity1_type = 'npc' AND entity1_id = $2 AND entity2_type = 'player' AND entity2_id = 0)
                AND user_id = $3 AND conversation_id = $4
            """, value, entity_id, self.user_id, self.conversation_id)
            
            # Create or update link history
            await conn.execute("""
                UPDATE SocialLinks
                SET link_history = link_history || $1::jsonb
                WHERE (entity1_type = 'player' AND entity1_id = 0 AND entity2_type = 'npc' AND entity2_id = $2)
                   OR (entity1_type = 'npc' AND entity1_id = $2 AND entity2_type = 'player' AND entity2_id = 0)
                AND user_id = $3 AND conversation_id = $4
            """, json.dumps([{"event": "conflict_consequence", "change": value, "description": description}]), 
            entity_id, self.user_id, self.conversation_id)
            
            # Update NPC stats
            await conn.execute("""
                UPDATE NPCStats
                SET closeness = GREATEST(0, LEAST(100, closeness + $1)),
                    trust = GREATEST(-100, LEAST(100, trust + $2))
                WHERE npc_id = $3 AND user_id = $4 AND conversation_id = $5
            """, value // 2, value, entity_id, self.user_id, self.conversation_id)
            
            # Also update NPC memory
            npc_name = await conn.fetchval("""
                SELECT npc_name FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, entity_id, self.user_id, self.conversation_id)
            
            if npc_name:
                memory_text = f"After a conflict, our relationship changed ({value:+d}). {description.split(':', 1)[1].strip()}"
                
                await conn.execute("""
                    INSERT INTO unified_memories (
                        entity_type, entity_id, user_id, conversation_id,
                        memory_text, memory_type, significance, emotional_intensity,
                        tags, metadata
                    ) VALUES (
                        'npc', $1, $2, $3, $4, 'relationship', $5, $6,
                        ARRAY['conflict', 'relationship_change'], $7
                    )
                """, entity_id, self.user_id, self.conversation_id, memory_text,
                 8 if abs(value) > 10 else 5,  # Significance based on magnitude
                 abs(value) * 5,  # Emotional intensity based on magnitude
                 json.dumps({"value_change": value, "source": "conflict_consequence"}))
    
    async def _apply_stat_consequence(self, conn, entity_type: str, entity_id: Optional[int], value: int, description: str) -> None:
        """Apply a stat change consequence."""
        # Extract the stat name from the description using AI or keyword matching
        stat_name = self._extract_stat_from_description(description)
        
        if entity_type == "player":
            # Apply stat change to player
            await apply_stat_change(
                self.user_id, 
                self.conversation_id, 
                {stat_name: value},
                f"Conflict consequence: {description}"
            )
            
            # Record stat history
            # Get current stat value first
            current_val = await conn.fetchval(f"""
                SELECT {stat_name} FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
            """, self.user_id, self.conversation_id)
            
            if current_val is not None:
                await conn.execute("""
                    INSERT INTO StatsHistory 
                    (user_id, conversation_id, player_name, stat_name, old_value, new_value, cause)
                    VALUES ($1, $2, 'Chase', $3, $4, $5, $6)
                """, self.user_id, self.conversation_id, stat_name, current_val, 
                current_val + value, f"Conflict consequence: {description}")
        
        elif entity_type == "npc" and entity_id is not None:
            # Determine which NPC stat to modify based on the player stat
            npc_stat = self._map_player_stat_to_npc_stat(stat_name)
            
            if npc_stat:
                # Apply stat change to NPC
                await conn.execute(f"""
                    UPDATE NPCStats
                    SET {npc_stat} = GREATEST(-100, LEAST(100, {npc_stat} + $1))
                    WHERE npc_id = $2 AND user_id = $3 AND conversation_id = $4
                """, value, entity_id, self.user_id, self.conversation_id)
                
                # Add NPC memory about stat change
                memory_text = f"My {npc_stat} changed by {value:+d} due to a conflict. {description.split(':', 1)[1].strip()}"
                
                await conn.execute("""
                    INSERT INTO unified_memories (
                        entity_type, entity_id, user_id, conversation_id,
                        memory_text, memory_type, significance, emotional_intensity,
                        tags, metadata
                    ) VALUES (
                        'npc', $1, $2, $3, $4, 'personal_development', $5, $6,
                        ARRAY['conflict', 'stat_change'], $7
                    )
                """, entity_id, self.user_id, self.conversation_id, memory_text,
                 7,  # Significance
                 abs(value) * 5,  # Emotional intensity
                 json.dumps({"stat": npc_stat, "value_change": value, "source": "conflict_consequence"}))
    
    def _extract_stat_from_description(self, description: str) -> str:
        """
        Extract the stat name from a consequence description.
        Uses keyword matching to identify the most likely stat.
        """
        description = description.lower()
        
        # Define mappings between keywords and stats
        stat_keywords = {
            "corruption": ["corrupt", "deprav", "degrad", "immoral", "debas", "control"],
            "confidence": ["confiden", "certainty", "self-assured", "assertive", "assured"],
            "willpower": ["willpower", "resolve", "determination", "resist", "defy", "strong-willed"],
            "obedience": ["obedien", "complian", "follow", "submit", "yield", "docile"],
            "dependency": ["dependen", "relian", "need", "require", "cling", "attachment"],
            "lust": ["lust", "arousal", "desire", "sexual", "passion", "want", "crave"],
            "mental_resilience": ["mental", "resilience", "psycholog", "trauma", "mind", "mental strength"],
            "physical_endurance": ["physical", "endurance", "stamina", "body", "endur", "strength"]
        }
        
        # Count keyword occurrences
        stat_scores = {stat: 0 for stat in stat_keywords}
        for stat, keywords in stat_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    stat_scores[stat] += 1
        
        # Find the stat with the highest score
        max_score = 0
        best_stat = "corruption"  # Default
        for stat, score in stat_scores.items():
            if score > max_score:
                max_score = score
                best_stat = stat
        
        return best_stat
    
    def _map_player_stat_to_npc_stat(self, player_stat: str) -> str:
        """Map a player stat to the corresponding NPC stat."""
        mapping = {
            "corruption": "dominance",
            "confidence": "respect",
            "willpower": "dominance",
            "obedience": "dominance",
            "dependency": "closeness",
            "lust": "intensity",
            "mental_resilience": "respect",
            "physical_endurance": "respect"
        }
        return mapping.get(player_stat, "dominance")
    
    async def _apply_unlock_consequence(self, conn, description: str) -> None:
        """
        Apply an unlock consequence.
        This could be unlocking a new location, activity, or other content.
        """
        # Check if this involves unlocking a location
        location_terms = ["location", "place", "area", "room", "building", "access"]
        if any(term in description.lower() for term in location_terms):
            # Extract the location name if possible
            location_name = self._extract_entity_name_from_description(description)
            
            if location_name:
                # Check if this location already exists
                existing = await conn.fetchval("""
                    SELECT id FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2 AND location_name = $3
                """, self.user_id, self.conversation_id, location_name)
                
                if not existing:
                    # Create a new location
                    await conn.execute("""
                        INSERT INTO Locations (user_id, conversation_id, location_name, description)
                        VALUES ($1, $2, $3, $4)
                    """, self.user_id, self.conversation_id, location_name, 
                    f"Location unlocked via conflict resolution. {description.split(':', 1)[1].strip()}")
                    
                    # Add a player memory about this new location
                    await conn.execute("""
                        INSERT INTO unified_memories (
                            entity_type, entity_id, user_id, conversation_id,
                            memory_text, memory_type, significance, emotional_intensity,
                            tags, metadata
                        ) VALUES (
                            'player', 0, $1, $2, $3, 'discovery', 8, 75,
                            ARRAY['conflict', 'unlock', 'location'], $4
                        )
                    """, self.user_id, self.conversation_id, 
                    f"I unlocked access to {location_name} after resolving a conflict. {description.split(':', 1)[1].strip()}",
                    json.dumps({"location": location_name, "source": "conflict_consequence"}))
        
        # Check if this involves unlocking an item
        item_terms = ["item", "object", "equipment", "gear", "tool", "weapon"]
        if any(term in description.lower() for term in item_terms):
            # Extract the item name if possible
            item_name = self._extract_entity_name_from_description(description)
            
            if item_name:
                # Add the item to the player's inventory
                await conn.execute("""
                    INSERT INTO PlayerInventory (
                        user_id, conversation_id, player_name, item_name, item_description, category
                    ) VALUES ($1, $2, 'Chase', $3, $4, 'Reward')
                    ON CONFLICT (user_id, conversation_id, player_name, item_name)
                    DO UPDATE SET quantity = PlayerInventory.quantity + 1
                """, self.user_id, self.conversation_id, item_name, 
                f"Obtained from conflict resolution. {description.split(':', 1)[1].strip()}")
        
        # Check if this involves unlocking a quest
        quest_terms = ["quest", "mission", "task", "assignment", "objective"]
        if any(term in description.lower() for term in quest_terms):
            # Extract the quest name if possible
            quest_name = self._extract_entity_name_from_description(description)
            
            if quest_name:
                # Create a new quest
                await conn.execute("""
                    INSERT INTO Quests (
                        user_id, conversation_id, quest_name, status, progress_detail
                    ) VALUES ($1, $2, $3, 'In Progress', $4)
                """, self.user_id, self.conversation_id, quest_name,
                f"Quest unlocked via conflict resolution. {description.split(':', 1)[1].strip()}")
    
    def _extract_entity_name_from_description(self, description: str) -> Optional[str]:
        """
        Extract an entity name from a description.
        Uses simple heuristics to identify likely entity names.
        """
        # Remove the "consequence: " prefix
        parts = description.split(':', 1)
        if len(parts) > 1:
            text = parts[1].strip()
        else:
            text = description
        
        # Look for quotation marks
        if '"' in text:
            # Extract text between first pair of quotation marks
            start = text.find('"')
            end = text.find('"', start + 1)
            if start != -1 and end != -1:
                return text[start + 1:end]
        
        # Look for 'to' or 'the' followed by a capitalized word or phrase
        import re
        matches = re.findall(r'(?:to|the)\s+([A-Z][A-Za-z\s\']+)(?:\.|,|\s|$)', text)
        if matches:
            return matches[0].strip()
        
        # Look for any capitalized phrase that might be a name (simple heuristic)
        matches = re.findall(r'([A-Z][A-Za-z\s\']+)(?:\.|,|\s|$)', text)
        if matches:
            # Filter out common sentence starters
            filtered = [m for m in matches if m.strip() not in ['I', 'You', 'The', 'A', 'An', 'This', 'That']]
            if filtered:
                return filtered[0].strip()
        
        return None
    
    async def _apply_permanent_consequence(self, conn, entity_type: str, entity_id: Optional[int], description: str) -> None:
        """
        Apply a permanent consequence to the game world.
        This could include NPC deaths, location destruction, faction changes, etc.
        """
        description_lower = description.lower()
        
        # Check if this is an NPC death/removal
        death_terms = ["death", "die", "kill", "perish", "demise", "fatal"]
        if entity_type == "npc" and entity_id is not None and any(term in description_lower for term in death_terms):
            # Get NPC name for the memory
            npc_name = await conn.fetchval("""
                SELECT npc_name FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, entity_id, self.user_id, self.conversation_id)
            
            if not npc_name:
                return
            
            # Create a global memory of this event
            await conn.execute("""
                INSERT INTO unified_memories (
                    entity_type, entity_id, user_id, conversation_id,
                    memory_text, memory_type, significance, emotional_intensity,
                    tags, metadata
                ) VALUES (
                    'player', 0, $1, $2, $3, 'permanent_change', 10, 100,
                    ARRAY['conflict', 'permanent', 'death'], $4
                )
            """, self.user_id, self.conversation_id, 
            f"{npc_name} has died as a result of the conflict. {description.split(':', 1)[1].strip()}",
            json.dumps({"npc_id": entity_id, "npc_name": npc_name, "type": "death", "source": "conflict_consequence"}))
            
            # Update NPC status - instead of deleting, mark them as inactive/deceased
            await conn.execute("""
                UPDATE NPCStats
                SET archetype_extras_summary = CASE
                    WHEN archetype_extras_summary IS NULL THEN 'Deceased due to conflict'
                    ELSE archetype_extras_summary || ' | Deceased due to conflict'
                END
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, entity_id, self.user_id, self.conversation_id)
            
            # Remove from future conflicts
            await conn.execute("""
                DELETE FROM ConflictNPCs
                WHERE npc_id = $1 AND conflict_id IN (
                    SELECT conflict_id FROM Conflicts
                    WHERE user_id = $2 AND conversation_id = $3 AND is_active = TRUE
                )
            """, entity_id, self.user_id, self.conversation_id)
        
        # Check if this is a location destruction
        destruction_terms = ["destroy", "destruction", "demolish", "ruin", "devastat", "obliterate"]
        if any(term in description_lower for term in destruction_terms):
            # Extract a location name
            location_name = self._extract_entity_name_from_description(description)
            
            if location_name:
                # Check if this location exists
                location_id = await conn.fetchval("""
                    SELECT id FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2 AND location_name = $3
                """, self.user_id, self.conversation_id, location_name)
                
                if location_id:
                    # Create a global memory of this event
                    await conn.execute("""
                        INSERT INTO unified_memories (
                            entity_type, entity_id, user_id, conversation_id,
                            memory_text, memory_type, significance, emotional_intensity,
                            tags, metadata
                        ) VALUES (
                            'player', 0, $1, $2, $3, 'permanent_change', 9, 90,
                            ARRAY['conflict', 'permanent', 'destruction'], $4
                        )
                    """, self.user_id, self.conversation_id, 
                    f"The location {location_name} has been destroyed. {description.split(':', 1)[1].strip()}",
                    json.dumps({"location_id": location_id, "location_name": location_name, "type": "destruction", "source": "conflict_consequence"}))
                    
                    # Update location description to reflect destruction
                    await conn.execute("""
                        UPDATE Locations
                        SET description = CASE
                            WHEN description IS NULL THEN 'Destroyed due to conflict'
                            ELSE description || ' | Destroyed due to conflict'
                        END
                        WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                    """, location_id, self.user_id, self.conversation_id)
        
        # Check if this is a faction dissolution/creation
        faction_terms = ["faction", "group", "alliance", "coalition", "organization"]
        if any(term in description_lower for term in faction_terms):
            # Create a global memory of this event
            await conn.execute("""
                INSERT INTO unified_memories (
                    entity_type, entity_id, user_id, conversation_id,
                    memory_text, memory_type, significance, emotional_intensity,
                    tags, metadata
                ) VALUES (
                    'player', 0, $1, $2, $3, 'permanent_change', 8, 80,
                    ARRAY['conflict', 'permanent', 'faction'], $4
                )
            """, self.user_id, self.conversation_id, 
            f"Faction dynamics have permanently changed. {description.split(':', 1)[1].strip()}",
            json.dumps({"type": "faction_change", "source": "conflict_consequence"}))
            
            # Get conflict details to determine which factions were involved
            conflict_factions = await conn.fetchrow("""
                SELECT faction_a_name, faction_b_name FROM Conflicts
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, self.user_id, self.conversation_id)
            
            if conflict_factions:
                # Record faction power shift
                creation = "creation" in description_lower or "new" in description_lower or "form" in description_lower
                dissolution = "dissolut" in description_lower or "destroy" in description_lower or "end" in description_lower
                
                faction_name = conflict_factions['faction_a_name'] if "faction a" in description_lower.replace(conflict_factions['faction_a_name'].lower(), "faction a") else conflict_factions['faction_b_name']
                
                power_change = 100 if creation else -100 if dissolution else 50
                
                await conn.execute("""
                    INSERT INTO FactionPowerShifts (
                        user_id, conversation_id, faction_name, power_level, change_amount, cause, conflict_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, self.user_id, self.conversation_id, faction_name, power_change, power_change, description, conflict_id)
    
    async def _update_faction_power(self, conn, conflict_id: int, outcome: str) -> None:
        """Update faction power based on conflict outcome."""
        # Get the conflict details
        conflict_row = await conn.fetchrow("""
            SELECT faction_a_name, faction_b_name, conflict_type FROM Conflicts
            WHERE conflict_id = $1
        """, conflict_id)
        
        if not conflict_row:
            return
        
        # Get player involvement
        involvement_row = await conn.fetchrow("""
            SELECT faction FROM PlayerConflictInvolvement
            WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
        """, conflict_id, self.user_id, self.conversation_id)
        
        player_faction = involvement_row['faction'] if involvement_row else "neutral"
        
        # Determine winning faction
        winning_faction = None
        if outcome == "success" or outcome == "partial_success":
            winning_faction = player_faction
        elif outcome == "failure":
            winning_faction = "b" if player_faction == "a" else "a"
        
        # Skip if player was neutral or no clear winner
        if winning_faction == "neutral" or winning_faction is None:
            return
        
        # Determine faction names
        faction_a_name = conflict_row['faction_a_name']
        faction_b_name = conflict_row['faction_b_name']
        
        winning_name = faction_a_name if winning_faction == "a" else faction_b_name
        losing_name = faction_b_name if winning_faction == "a" else faction_a_name
        
        # Calculate power shift based on conflict type
        power_values = {
            "major": {"win": 50, "loss": -50, "partial": 25},
            "minor": {"win": 25, "loss": -25, "partial": 10},
            "standard": {"win": 10, "loss": -10, "partial": 5},
            "catastrophic": {"win": 100, "loss": -100, "partial": 50}
        }
        
        conflict_type = conflict_row['conflict_type']
        win_value = power_values.get(conflict_type, {"win": 25, "loss": -25, "partial": 10})
        
        win_amount = win_value["win"] if outcome == "success" else win_value["partial"]
        loss_amount = win_value["loss"] if outcome == "success" else -win_value["partial"]
        
        # Record winning faction power increase
        await conn.execute("""
            INSERT INTO FactionPowerShifts (
                user_id, conversation_id, faction_name, power_level, change_amount, cause, conflict_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, self.user_id, self.conversation_id, winning_name, win_amount, win_amount, 
        f"Victory in conflict: {conflict_row['conflict_name'] if 'conflict_name' in conflict_row else 'Unknown conflict'}", conflict_id)
        
        # Record losing faction power decrease
        await conn.execute("""
            INSERT INTO FactionPowerShifts (
                user_id, conversation_id, faction_name, power_level, change_amount, cause, conflict_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, self.user_id, self.conversation_id, losing_name, loss_amount, loss_amount, 
        f"Defeat in conflict: {conflict_row['conflict_name'] if 'conflict_name' in conflict_row else 'Unknown conflict'}", conflict_id)
    
    async def _record_conflict_history(self, conn, conflict_id: int, outcome: str) -> None:
        """
        Record conflict history for the consequence chain system.
        This creates entries in ConflictHistory that can trigger future conflicts.
        """
        # Get the conflict details
        conflict_row = await conn.fetchrow("""
            SELECT conflict_name, conflict_type FROM Conflicts
            WHERE conflict_id = $1
        """, conflict_id)
        
        if not conflict_row:
            return
        
        # Get all NPCs involved in the conflict
        npc_rows = await conn.fetch("""
            SELECT cn.npc_id, cn.faction, cn.role, ns.npc_name, ns.dominance
            FROM ConflictNPCs cn
            JOIN NPCStats ns ON cn.npc_id = ns.npc_id
            WHERE cn.conflict_id = $1
        """, conflict_id)
        
        # Get player involvement
        player_row = await conn.fetchrow("""
            SELECT faction FROM PlayerConflictInvolvement
            WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
        """, conflict_id, self.user_id, self.conversation_id)
        
        player_faction = player_row['faction'] if player_row else "neutral"
        
        # For each NPC, create a history entry
        for npc in npc_rows:
            # Determine impact type based on outcome and factions
            impact_type = "neutral"
            grudge_level = 0
            narrative_impact = ""
            
            # If player and NPC were on the same side
            if player_faction == npc['faction']:
                if outcome == "success":
                    impact_type = "positive"
                    grudge_level = 0
                    narrative_impact = f"{npc['npc_name']} is pleased with the successful resolution of {conflict_row['conflict_name']}."
                elif outcome == "partial_success":
                    impact_type = "neutral"
                    grudge_level = 10
                    narrative_impact = f"{npc['npc_name']} is somewhat satisfied with the partial success in {conflict_row['conflict_name']}."
                else:  # failure
                    impact_type = "negative"
                    grudge_level = 30 + npc['dominance'] // 2  # Higher dominance = higher grudge
                    narrative_impact = f"{npc['npc_name']} blames you for the failure in {conflict_row['conflict_name']}."
            
            # If player and NPC were on opposite sides
            elif player_faction != "neutral" and npc['faction'] != "neutral" and player_faction != npc['faction']:
                if outcome == "success":
                    impact_type = "negative"
                    grudge_level = 50 + npc['dominance'] // 2  # Higher dominance = higher grudge
                    narrative_impact = f"{npc['npc_name']} holds a significant grudge after being defeated in {conflict_row['conflict_name']}."
                elif outcome == "partial_success":
                    impact_type = "negative"
                    grudge_level = 30 + npc['dominance'] // 3
                    narrative_impact = f"{npc['npc_name']} is displeased but not vengeful after the partial defeat in {conflict_row['conflict_name']}."
                else:  # failure
                    impact_type = "positive"
                    grudge_level = 0
                    narrative_impact = f"{npc['npc_name']} is satisfied with defeating you in {conflict_row['conflict_name']}."
            
            # Record the history entry
            await conn.execute("""
                INSERT INTO ConflictHistory (
                    user_id, conversation_id, conflict_id, affected_npc_id,
                    impact_type, grudge_level, narrative_impact
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, self.user_id, self.conversation_id, conflict_id, npc['npc_id'],
            impact_type, grudge_level, narrative_impact)
            
            # If there's a significant grudge, also create an NPC memory
            if grudge_level >= 30:
                await conn.execute("""
                    INSERT INTO unified_memories (
                        entity_type, entity_id, user_id, conversation_id,
                        memory_text, memory_type, significance, emotional_intensity,
                        tags, metadata
                    ) VALUES (
                        'npc', $1, $2, $3, $4, 'grudge', $5, $6,
                        ARRAY['conflict', 'grudge'], $7
                    )
                """, npc['npc_id'], self.user_id, self.conversation_id, 
                narrative_impact,
                min(10, 5 + grudge_level // 20),  # Significance based on grudge level
                grudge_level * 2,  # Emotional intensity
                json.dumps({"grudge_level": grudge_level, "conflict_id": conflict_id, "source": "conflict_outcome"}))
    
    async def recruit_npc_for_conflict(self, conflict_id: int, npc_id: int, faction: str = None) -> Dict[str, Any]:
        """
        Recruit an NPC to help with a conflict.
        Returns the updated conflict information.
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if NPC is already involved
                existing = await conn.fetchrow("""
                    SELECT faction, role FROM ConflictNPCs
                    WHERE conflict_id = $1 AND npc_id = $2
                """, conflict_id, npc_id)
                
                # Get player faction if not specified
                if faction is None:
                    player_faction_row = await conn.fetchrow("""
                        SELECT faction FROM PlayerConflictInvolvement
                        WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, conflict_id, self.user_id, self.conversation_id)
                    
                    faction = player_faction_row['faction'] if player_faction_row else "neutral"
                
                # Get NPC relationship level
                relationship_level = await conn.fetchval("""
                    SELECT link_level FROM SocialLinks
                    WHERE entity1_type = 'player' AND entity1_id = 0
                    AND entity2_type = 'npc' AND entity2_id = $1
                    AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                # Only allow recruitment if relationship level is 3+
                if relationship_level is None or relationship_level < 3:
                    return {
                        "success": False,
                        "message": "Your relationship with this NPC isn't strong enough for recruitment"
                    }
                
                # Create or update NPC involvement
                if existing:
                    # NPC is already involved, potentially changing sides
                    old_faction = existing['faction']
                    old_role = existing['role']
                    
                    if old_faction != faction:
                        # Changing factions is a significant event
                        await conn.execute("""
                            UPDATE ConflictNPCs
                            SET faction = $1
                            WHERE conflict_id = $2 AND npc_id = $3
                        """, faction, conflict_id, npc_id)
                        
                        # Create memory about faction change
                        npc_name = await conn.fetchval("""
                            SELECT npc_name FROM NPCStats
                            WHERE npc_id = $1
                        """, npc_id)
                        
                        await self._create_conflict_memory(
                            conflict_id, 
                            f"{npc_name} has switched from faction {old_faction} to faction {faction}",
                            7
                        )
                    
                    result = {
                        "success": True,
                        "message": f"NPC switched to faction {faction}",
                        "faction_change": old_faction != faction
                    }
                else:
                    # NPC is newly recruited
                    role = random.choice(['member', 'supporter'])
                    influence = random.randint(40, 70)
                    
                    await conn.execute("""
                        INSERT INTO ConflictNPCs (
                            conflict_id, npc_id, faction, role, influence_level
                        ) VALUES ($1, $2, $3, $4, $5)
                    """, conflict_id, npc_id, faction, role, influence)
                    
                    # Create memory about recruitment
                    npc_name = await conn.fetchval("""
                        SELECT npc_name FROM NPCStats
                        WHERE npc_id = $1
                    """, npc_id)
                    
                    await self._create_conflict_memory(
                        conflict_id, 
                        f"{npc_name} has been recruited to faction {faction}",
                        6
                    )
                    
                    result = {
                        "success": True,
                        "message": f"NPC recruited to faction {faction}",
                        "faction_change": False
                    }
                
                # Clear cache
                CONFLICT_CACHE.remove(f"conflict:{self.user_id}:{self.conversation_id}:{conflict_id}")
                
                return result
    
    async def daily_conflict_update(self) -> Dict[str, Any]:
        """
        Process daily updates for all active conflicts:
        - Increment progress
        - Check if any conflicts should be automatically generated
        - Update NPC involvement
        - Generate NPC requests
        
        Returns summary of updates
        """
        results = {
            "conflicts_updated": 0,
            "conflicts_generated": 0,
            "conflicts_resolved": 0,
            "phase_changes": 0,
            "npc_requests": 0
        }
        
        # Get current day
        _, _, current_day, _ = await self.get_current_game_time()
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get all active conflicts
                active_conflicts = await conn.fetch("""
                    SELECT conflict_id, conflict_type, progress, phase, estimated_duration, start_day
                    FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2 AND is_active = TRUE
                """, self.user_id, self.conversation_id)
                
                # Process each active conflict
                for conflict in active_conflicts:
                    # Determine daily progress increment based on estimated duration
                    estimated_duration = conflict['estimated_duration']
                    base_increment = 100 / max(1, estimated_duration)
                    
                    # Apply random variation
                    progress_increment = base_increment * random.uniform(0.8, 1.2)
                    
                    # Get current phase and progress
                    old_phase = conflict['phase']
                    old_progress = conflict['progress']
                    
                    # Update conflict progress
                    updated_conflict = await self.update_conflict_progress(
                        conflict['conflict_id'], progress_increment
                    )
                    
                    results["conflicts_updated"] += 1
                    
                    # Check if phase changed
                    if updated_conflict['phase'] != old_phase:
                        results["phase_changes"] += 1
                    
                    # Check if conflict should be resolved
                    if updated_conflict['progress'] >= 100 and updated_conflict['phase'] == 'resolution':
                        await self.resolve_conflict(conflict['conflict_id'])
                        results["conflicts_resolved"] += 1
                
                # Check if we should generate new conflicts
                should_generate_major = await self._should_generate_major_conflict(conn, current_day)
                should_generate_minor = await self._should_generate_minor_conflict(conn, current_day)
                should_generate_standard = await self._should_generate_standard_conflict(conn)
                
                if should_generate_major:
                    await self.generate_conflict("major")
                    results["conflicts_generated"] += 1
                
                if should_generate_minor:
                    await self.generate_conflict("minor")
                    results["conflicts_generated"] += 1
                
                if should_generate_standard:
                    await self.generate_conflict("standard")
                    results["conflicts_generated"] += 1
                
                # Process NPC proactive requests (30% chance per day per NPC)
                npc_rows = await conn.fetch("""
                    SELECT npc_id, npc_name, dominance FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
                """, self.user_id, self.conversation_id)
                
                for npc in npc_rows:
                    # 30% chance of making a request
                    if random.random() < self.NPC_REQUEST_CHANCE:
                        # Find an active conflict this NPC might care about
                        conflict_row = await conn.fetchrow("""
                            SELECT conflict_id, conflict_name, conflict_type FROM Conflicts
                            WHERE user_id = $1 AND conversation_id = $2 
                            AND is_active = TRUE
                            ORDER BY RANDOM() LIMIT 1
                        """, self.user_id, self.conversation_id)
                        
                        if conflict_row:
                            # Create a proactive request as a memory event
                            await self._create_conflict_memory(
                                conflict_row['conflict_id'],
                                f"{npc['npc_name']} has requested your assistance with {conflict_row['conflict_name']}",
                                7  # Higher significance for NPC requests
                            )
                            
                            # Also create an NPC memory
                            await conn.execute("""
                                INSERT INTO unified_memories (
                                    entity_type, entity_id, user_id, conversation_id,
                                    memory_text, memory_type, significance, emotional_intensity,
                                    tags, metadata
                                ) VALUES (
                                    'npc', $1, $2, $3, $4, 'request', 7, 70,
                                    ARRAY['conflict', 'request'], $5
                                )
                            """, npc['npc_id'], self.user_id, self.conversation_id,
                            f"I asked the player to help me with {conflict_row['conflict_name']}",
                            json.dumps({"conflict_id": conflict_row['conflict_id'], "type": "assistance_request"}))
                            
                            results["npc_requests"] += 1
                
        # Clear cache
        CONFLICT_CACHE.clear()
        
        return results
    
    async def _should_generate_major_conflict(self, conn, current_day: int) -> bool:
        """Determine if a major conflict should be generated."""
        # Check for active major conflicts
        major_count = await conn.fetchval("""
            SELECT COUNT(*) FROM Conflicts
            WHERE user_id = $1 AND conversation_id = $2
            AND conflict_type = 'major' AND is_active = TRUE
        """, self.user_id, self.conversation_id)
        
        if major_count > 0:
            return False
        
        # Check when the last major conflict was created
        last_major_day = await conn.fetchval("""
            SELECT start_day FROM Conflicts
            WHERE user_id = $1 AND conversation_id = $2
            AND conflict_type = 'major'
            ORDER BY created_at DESC LIMIT 1
        """, self.user_id, self.conversation_id)
        
        if last_major_day is None or current_day - last_major_day >= self.MAJOR_CONFLICT_INTERVAL:
            return True
        
        return False
    
    async def _should_generate_minor_conflict(self, conn, current_day: int) -> bool:
        """Determine if a minor conflict should be generated."""
        # Check for active major conflicts (need one for minor conflicts)
        major_count = await conn.fetchval("""
            SELECT COUNT(*) FROM Conflicts
            WHERE user_id = $1 AND conversation_id = $2
            AND conflict_type = 'major' AND is_active = TRUE
        """, self.user_id, self.conversation_id)
        
        if major_count == 0:
            return False
        
        # Check when the last minor conflict was created
        last_minor_day = await conn.fetchval("""
            SELECT start_day FROM Conflicts
            WHERE user_id = $1 AND conversation_id = $2
            AND conflict_type = 'minor'
            ORDER BY created_at DESC LIMIT 1
        """, self.user_id, self.conversation_id)
        
        if last_minor_day is None or current_day - last_minor_day >= self.MINOR_CONFLICT_INTERVAL:
            return True
        
        return False
    
    async def _should_generate_standard_conflict(self, conn) -> bool:
        """Determine if a standard conflict should be generated."""
        # Check for active standard conflicts
        standard_count = await conn.fetchval("""
            SELECT COUNT(*) FROM Conflicts
            WHERE user_id = $1 AND conversation_id = $2
            AND conflict_type = 'standard' AND is_active = TRUE
        """, self.user_id, self.conversation_id)
        
        if standard_count < self.MAX_STANDARD_CONFLICTS:
            # Random chance
            return random.random() < 0.3  # 30% chance per day
        
        return False
    
    async def update_player_vitals(self, activity_type: str = "standard") -> Dict[str, Any]:
        """
        Update player vitals (energy and hunger) based on activity type.
        Returns the updated vitals.
        """
        # Define vital changes based on activity type
        vital_changes = {
            "standard": {"energy": -5, "hunger": -3},
            "intense": {"energy": -15, "hunger": -10},
            "restful": {"energy": 10, "hunger": -5},
            "eating": {"energy": 0, "hunger": 30}
        }
        
        # Use standard if activity type not found
        changes = vital_changes.get(activity_type, vital_changes["standard"])
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get current vitals
                vitals_row = await conn.fetchrow("""
                    SELECT energy, hunger FROM PlayerVitals
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                """, self.user_id, self.conversation_id)
                
                if vitals_row:
                    # Calculate new values
                    energy = max(0, min(100, vitals_row['energy'] + changes["energy"]))
                    hunger = max(0, min(100, vitals_row['hunger'] + changes["hunger"]))
                    
                    # Update vitals
                    await conn.execute("""
                        UPDATE PlayerVitals
                        SET energy = $1, hunger = $2, last_update = CURRENT_TIMESTAMP
                        WHERE user_id = $3 AND conversation_id = $4 AND player_name = 'Chase'
                    """, energy, hunger, self.user_id, self.conversation_id)
                else:
                    # Create new vitals record with default values modified by activity
                    energy = max(0, min(100, 100 + changes["energy"]))
                    hunger = max(0, min(100, 100 + changes["hunger"]))
                    
                    await conn.execute("""
                        INSERT INTO PlayerVitals (user_id, conversation_id, player_name, energy, hunger)
                        VALUES ($1, $2, 'Chase', $3, $4)
                    """, self.user_id, self.conversation_id, energy, hunger)
                
                return {
                    "energy": energy,
                    "hunger": hunger,
                    "activity_type": activity_type
                }
    
    async def get_player_vitals(self) -> Dict[str, Any]:
        """Get the current player vitals."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                vitals_row = await conn.fetchrow("""
                    SELECT energy, hunger, last_update FROM PlayerVitals
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                """, self.user_id, self.conversation_id)
                
                if vitals_row:
                    return {
                        "energy": vitals_row['energy'],
                        "hunger": vitals_row['hunger'],
                        "last_update": vitals_row['last_update']
                    }
                else:
                    # Create default vitals
                    await conn.execute("""
                        INSERT INTO PlayerVitals (user_id, conversation_id, player_name, energy, hunger)
                        VALUES ($1, $2, 'Chase', 100, 100)
                    """, self.user_id, self.conversation_id)
                    
                    return {
                        "energy": 100,
                        "hunger": 100,
                        "last_update": datetime.now()
                    }
    
    async def add_conflict_to_narrative(self, narrative_text: str) -> Dict[str, Any]:
        """
        Analyze a narrative text to determine if it should trigger a conflict.
        If appropriate, generate a new conflict based on the narrative.
        
        Returns:
            Dict with analysis results and potentially a new conflict.
        """
        # Define conflict-related keywords and their weights
        conflict_keywords = {
            "brewing": 5,
            "tension": 5,
            "argument": 10,
            "disagreement": 10,
            "challenge": 15,
            "conflict": 20,
            "fight": 15,
            "battle": 15,
            "confrontation": 20,
            "opposition": 10,
            "struggle": 10,
            "strife": 15,
            "crisis": 20,
            "dilemma": 15,
            "problem": 5,
            "obstacle": 5,
            "hurdle": 5,
            "difficulty": 5,
            "complication": 10,
            "sabotage": 20,
            "betrayal": 25,
            "conspiracy": 25
        }
        
        # Calculate conflict intensity based on keywords
        narrative_lower = narrative_text.lower()
        conflict_intensity = 0
        matched_keywords = []
        
        for keyword, weight in conflict_keywords.items():
            if keyword in narrative_lower:
                conflict_intensity += weight
                matched_keywords.append(keyword)
        
        # Normalize conflict intensity (0-100)
        conflict_intensity = min(100, conflict_intensity)
        
        # Only create conflict if intensity is high enough
        result = {
            "analysis": {
                "conflict_intensity": conflict_intensity,
                "matched_keywords": matched_keywords
            },
            "conflict_generated": False
        }
        
        # Get active conflicts
        active_conflicts = await self.get_active_conflicts()
        
        # Generate a conflict if intensity is high enough and we don't have too many
        if conflict_intensity >= 50 and len(active_conflicts) < 5:  # Allow up to 5 total
            # Determine conflict type based on intensity
            conflict_type = "standard"
            if conflict_intensity >= 90:
                conflict_type = "catastrophic"
            elif conflict_intensity >= 70:
                conflict_type = "major"
            elif conflict_intensity >= 50:
                conflict_type = "minor"
            
            # 30% chance to generate a conflict when conditions are met
            if random.random() < 0.3:
                # Create custom prompt including the narrative
                custom_prompt = f"""
                The following narrative suggests a developing conflict:
                
                "{narrative_text}"
                
                Generate a {conflict_type} conflict for the femdom roleplaying game based on this narrative.
                Create unique conflict details that believably extend from this story.
                
                Format your response as a structured JSON object with these keys:
                - name: Conflict name (string)
                - description: Overall description (string)
                - brewing_description: Description of brewing phase (string)
                - active_description: Description of active phase (string)
                - climax_description: Description of climax phase (string)
                - resolution_description: Description of resolution phase (string)
                - faction_a: Name of first faction (string)
                - faction_b: Name of second faction (string)
                - estimated_duration: Days to resolve (number)
                - resources_required: Object with money, supplies, and influence values
                - involved_npcs: Array of NPC names who might be involved
                - success_consequences: Array of consequence descriptions for success
                - failure_consequences: Array of consequence descriptions for failure
                """
                
                # Get environment description for context
                environment_desc = await self._get_environment_description()
                
                # Call GPT with custom prompt
                gpt_response = await get_chatgpt_response(
                    self.conversation_id,
                    environment_desc,
                    custom_prompt
                )
                
                # Extract conflict details from response
                if gpt_response.get('type') == 'function_call':
                    conflict_details = gpt_response.get('function_args', {})
                else:
                    # Try to extract JSON from text response
                    text = gpt_response.get('response', '')
                    try:
                        # Find JSON block in text
                        json_start = text.find('{')
                        json_end = text.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = text[json_start:json_end]
                            conflict_details = json.loads(json_str)
                        else:
                            return result  # No valid JSON found
                    except Exception as e:
                        logger.error(f"Failed to parse conflict details from GPT response: {e}")
                        return result  # Error parsing JSON
                
                # Generate the conflict
                year, month, day, _ = await self.get_current_game_time()
                conflict_id = await self._save_conflict(conflict_details, conflict_type, day)
                await self._associate_npcs_with_conflict(conflict_id, conflict_details)
                await self._create_conflict_memory(
                    conflict_id,
                    f"A new {conflict_type} conflict has emerged from recent events: {conflict_details.get('name', 'Unnamed Conflict')}",
                    7
                )
                
                # Get the complete conflict
                new_conflict = await self.get_conflict(conflict_id)
                
                # Update result
                result["conflict_generated"] = True
                result["message"] = f"Generated a new {conflict_type} conflict based on narrative."
                result["conflict"] = new_conflict
        
        return result
    
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get a comprehensive state of the conflict system including:
        - Active conflicts
        - Player vitals
        - Conflict history
        - Faction power
        
        Returns:
            Dict with complete conflict system state
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get active conflicts
                active_conflicts = await self.get_active_conflicts()
                
                # Get player vitals
                vitals = await self.get_player_vitals()
                
                # Get conflict history
                conflict_history = await conn.fetch("""
                    SELECT ch.conflict_id, ch.affected_npc_id, ch.impact_type, 
                           ch.grudge_level, ch.narrative_impact, ch.has_triggered_consequence,
                           n.npc_name, c.conflict_name, c.outcome
                    FROM ConflictHistory ch
                    JOIN NPCStats n ON ch.affected_npc_id = n.npc_id
                    JOIN Conflicts c ON ch.conflict_id = c.conflict_id
                    WHERE ch.user_id = $1 AND ch.conversation_id = $2
                    ORDER BY ch.created_at DESC
                    LIMIT 20
                """, self.user_id, self.conversation_id)
                
                history = [dict(row) for row in conflict_history]
                
                # Get faction power shifts
                faction_rows = await conn.fetch("""
                    SELECT faction_name, SUM(power_level) as total_power,
                           MAX(created_at) as last_updated
                    FROM FactionPowerShifts
                    WHERE user_id = $1 AND conversation_id = $2
                    GROUP BY faction_name
                    ORDER BY total_power DESC
                """, self.user_id, self.conversation_id)
                
                factions = [dict(row) for row in faction_rows]
                
                # Get recent memory events
                memory_rows = await conn.fetch("""
                    SELECT conflict_id, memory_text, significance, entity_type, entity_id, created_at
                    FROM ConflictMemoryEvents
                    WHERE conflict_id IN (SELECT conflict_id FROM Conflicts WHERE user_id = $1 AND conversation_id = $2)
                    ORDER BY created_at DESC
                    LIMIT 30
                """, self.user_id, self.conversation_id)
                
                memories = [dict(row) for row in memory_rows]
                
                return {
                    "active_conflicts": active_conflicts,
                    "player_vitals": vitals,
                    "conflict_history": history,
                    "factions": factions,
                    "memories": memories
                }
                
