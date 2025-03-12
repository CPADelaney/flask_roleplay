# logic/conflict_system/conflict_manager.py

import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import asyncpg
from db.connection import get_db_connection
from logic.npc_creation import process_daily_npc_activities
from logic.calendar import load_calendar_names
from logic.time_cycle import get_current_time
from logic.chatgpt_integration import get_chatgpt_response
from utils.caching import CONFLICT_CACHE

logger = logging.getLogger(__name__)

class ConflictManager:
    """
    Manages the Dynamic Conflict System that generates, updates, and resolves
    layered conflicts within the game.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the Conflict Manager with user and conversation context."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Constants from your design document
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
            "active_conflicts": active_conflicts
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
        
        Generate details for a new {context['conflict_type']} conflict, including:
        1. Conflict name
        2. Description
        3. Descriptions for each phase (brewing, active, climax, resolution)
        4. Two opposing factions or forces
        5. Estimated duration in days (2-3 weeks for major, 3-7 days for minor, 1-3 days for standard)
        6. Resources required for resolution (money, supplies, influence)
        7. Potential NPCs who would be involved
        8. Potential consequences for success and failure
        
        Format your response as a structured JSON object.
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
                        "estimated_duration": 7,
                        "resources_required": {
                            "money": 100,
                            "supplies": 5,
                            "influence": 20
                        },
                        "involved_npcs": [],
                        "success_consequences": ["The situation improves."],
                        "failure_consequences": ["The situation worsens."]
                    }
            except:
                logger.error("Failed to parse conflict details from GPT response")
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
                    "estimated_duration": 7,
                    "resources_required": {
                        "money": 100,
                        "supplies": 5,
                        "influence": 20
                    },
                    "involved_npcs": [],
                    "success_consequences": ["The situation improves."],
                    "failure_consequences": ["The situation worsens."]
                }
    
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
                    conflict_details.get('estimated_duration', 7),
                    conflict_details.get('faction_a', "Faction A"),
                    conflict_details.get('faction_b', "Faction B"),
                    json.dumps(conflict_details.get('resources_required', {"money": 100, "supplies": 5, "influence": 20})),
                    True  # Is active
                )
                
                # Save consequences
                success_consequences = conflict_details.get('success_consequences', [])
                failure_consequences = conflict_details.get('failure_consequences', [])
                
                for consequence in success_consequences:
                    await conn.execute("""
                        INSERT INTO ConflictConsequences (
                            conflict_id, consequence_type, entity_type, entity_id, description
                        ) VALUES ($1, $2, $3, $4, $5)
                    """, conflict_id, "relationship", "player", None, f"Success consequence: {consequence}")
                
                for consequence in failure_consequences:
                    await conn.execute("""
                        INSERT INTO ConflictConsequences (
                            conflict_id, consequence_type, entity_type, entity_id, description
                        ) VALUES ($1, $2, $3, $4, $5)
                    """, conflict_id, "relationship", "player", None, f"Failure consequence: {consequence}")
                
                return conflict_id
    
    async def _associate_npcs_with_conflict(self, conflict_id: int, conflict_details: Dict[str, Any]) -> None:
        """Associate NPCs with the conflict based on the generated details."""
        # Extract NPC names from conflict details
        involved_npcs = conflict_details.get('involved_npcs', [])
        if not involved_npcs:
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
                    SET progress = $1, phase = $2
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
                    SELECT id, actions_taken FROM PlayerConflictInvolvement
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, self.user_id, self.conversation_id)
                
                actions = []
                if involvement_row:
                    involvement_id = involvement_row['id']
                    
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
                            actions_taken = $6
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
                if involvement_row and involvement_row.get('involvement_level') != involvement_level:
                    await self._create_conflict_memory(
                        conflict_id, 
                        f"Player changed involvement from {involvement_row.get('involvement_level')} to {involvement_level}",
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
        
        # Get NPC assistance
        npc_bonus = await conn.fetchval("""
            SELECT COUNT(*) * 10 FROM ConflictNPCs
            WHERE conflict_id = $1 AND faction = (
                SELECT faction FROM PlayerConflictInvolvement
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            )
        """, conflict_id, self.user_id, self.conversation_id)
        
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
                    SET outcome = $1, is_active = FALSE, progress = 100, phase = 'resolution'
                    WHERE conflict_id = $2
                """, outcome, conflict_id)
                
                # Create resolution memory
                await self._create_conflict_memory(
                    conflict_id, 
                    f"The conflict '{conflict_row['conflict_name']}' has been resolved with outcome: {outcome}",
                    9  # High significance for resolution
                )
                
                # Apply consequences based on outcome
                if outcome == "success" or outcome == "partial_success":
                    await self._apply_success_consequences(conn, conflict_id, outcome)
                elif outcome == "failure":
                    await self._apply_failure_consequences(conn, conflict_id)
                
                # Clear cache
                CONFLICT_CACHE.remove(f"conflict:{self.user_id}:{self.conversation_id}:{conflict_id}")
                CONFLICT_CACHE.remove(f"active_conflicts:{self.user_id}:{self.conversation_id}")
                
                # Get final conflict state
                return await self.get_conflict(conflict_id)
    
    async def _apply_success_consequences(self, conn, conflict_id: int, outcome: str) -> None:
        """Apply success consequences for a conflict."""
        # Get success consequences
        consequence_rows = await conn.fetch("""
            SELECT id, description FROM ConflictConsequences
            WHERE conflict_id = $1 AND description LIKE 'Success consequence:%'
        """, conflict_id)
        
        # Apply each consequence
        for consequence in consequence_rows:
            description = consequence['description']
            
            # Mark consequence as applied
            await conn.execute("""
                UPDATE ConflictConsequences
                SET applied = TRUE, applied_at = CURRENT_TIMESTAMP
                WHERE id = $1
            """, consequence['id'])
            
            # Create memory event for the consequence
            await self._create_conflict_memory(
                conflict_id, 
                f"Consequence applied: {description}",
                7
            )
            
            # TODO: Add actual game state changes based on consequence description
            # For now, just record the memory
    
    async def _apply_failure_consequences(self, conn, conflict_id: int) -> None:
        """Apply failure consequences for a conflict."""
        # Get failure consequences
        consequence_rows = await conn.fetch("""
            SELECT id, description FROM ConflictConsequences
            WHERE conflict_id = $1 AND description LIKE 'Failure consequence:%'
        """, conflict_id)
        
        # Apply each consequence
        for consequence in consequence_rows:
            description = consequence['description']
            
            # Mark consequence as applied
            await conn.execute("""
                UPDATE ConflictConsequences
                SET applied = TRUE, applied_at = CURRENT_TIMESTAMP
                WHERE id = $1
            """, consequence['id'])
            
            # Create memory event for the consequence
            await self._create_conflict_memory(
                conflict_id, 
                f"Consequence applied: {description}",
                7
            )
            
            # TODO: Add actual game state changes based on consequence description
            # For now, just record the memory
    
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
        
        Returns summary of updates
        """
        results = {
            "conflicts_updated": 0,
            "conflicts_generated": 0,
            "conflicts_resolved": 0,
            "phase_changes": 0
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
        
        # Clear cache
        CONFLICT_CACHE.remove(f"active_conflicts:{self.user_id}:{self.conversation_id}")
        
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
