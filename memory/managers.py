# memory/managers.py

import logging
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import asyncio
import random

from .core import (
    UnifiedMemoryManager, 
    Memory, 
    MemoryType, 
    MemoryStatus,
    MemorySignificance,
    with_transaction
)
from .connection import DBConnectionManager, TransactionContext

logger = logging.getLogger("memory_managers")

class NPCMemoryManager(UnifiedMemoryManager):
    """
    Specialized memory manager for NPC memories.
    Extends UnifiedMemoryManager with NPC-specific functionality.
    """
    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        super().__init__(
            entity_type="npc", 
            entity_id=npc_id, 
            user_id=user_id, 
            conversation_id=conversation_id
        )
        self.npc_id = npc_id
    
    @with_transaction
    async def get_npc_stats(self, conn=None) -> Dict[str, Any]:
        """
        Get NPC stats to help with memory processing.
        """
        row = await conn.fetchrow("""
            SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """, self.npc_id, self.user_id, self.conversation_id)
        
        if not row:
            return {}
            
        return dict(row)
    
    @with_transaction
    async def update_npc_relationship(self, 
                                     target_type: str, 
                                     target_id: int, 
                                     relationship_change: Dict[str, int],
                                     conn=None) -> bool:
        """
        Update NPC relationship with target based on memory.
        """
        # Get current relationship
        row = await conn.fetchrow("""
            SELECT link_id, link_type, link_level
            FROM SocialLinks
            WHERE user_id = $1
            AND conversation_id = $2
            AND ((entity1_type = 'npc' AND entity1_id = $3 AND entity2_type = $4 AND entity2_id = $5)
               OR (entity1_type = $4 AND entity1_id = $5 AND entity2_type = 'npc' AND entity2_id = $3))
        """, self.user_id, self.conversation_id, self.npc_id, target_type, target_id)
        
        if not row:
            # Relationship doesn't exist yet
            return False
            
        link_id = row['link_id']
        link_level = row['link_level']
        
        # Apply relationship changes
        level_change = relationship_change.get('level', 0)
        new_level = max(0, min(100, link_level + level_change))
        
        # Update relationship
        await conn.execute("""
            UPDATE SocialLinks
            SET link_level = $1
            WHERE link_id = $2
        """, new_level, link_id)
        
        # Add event to relationship history
        event_text = f"Memory-driven relationship change: {level_change:+d} (New level: {new_level})"
        await conn.execute("""
            UPDATE SocialLinks
            SET link_history = COALESCE(link_history, '[]'::jsonb) || $1::jsonb
            WHERE link_id = $2
        """, json.dumps([event_text]), link_id)
        
        return True
    
    @with_transaction
    async def add_observation_memory(self, 
                                    observation: str, 
                                    significance: int = MemorySignificance.MEDIUM,
                                    emotional_intensity: int = 0,
                                    tags: List[str] = None,
                                    affect_relationships: bool = False,
                                    conn=None) -> int:
        """
        Add an observation memory with optional relationship effects.
        """
        memory_id = await self.add_memory(
            memory=Memory(
                text=observation,
                memory_type=MemoryType.OBSERVATION,
                significance=significance,
                emotional_intensity=emotional_intensity,
                tags=tags or [],
                timestamp=datetime.now()
            ),
            conn=conn
        )
        
        # Apply relationship effects if needed
        if affect_relationships:
            # Simple relationship effect based on sentiment
            # In production, you would use a more sophisticated sentiment analysis
            sentiment = 0
            if "positive" in (tags or []):
                sentiment = 1
            elif "negative" in (tags or []):
                sentiment = -1
                
            # Apply effect to player relationship
            if sentiment != 0:
                await self.update_npc_relationship(
                    target_type="player",
                    target_id=self.user_id,
                    relationship_change={"level": sentiment * 2},
                    conn=conn
                )
        
        return memory_id
    
    @with_transaction
    async def recall_player_interactions(self, limit: int = 5, conn=None) -> List[Memory]:
        """
        Specifically recall memories about player interactions.
        """
        return await self.retrieve_memories(
            query="player Chase",
            tags=["player_related", "interaction"],
            limit=limit,
            conn=conn
        )
    
    @with_transaction
    async def get_personality_biased_memories(self, query: str, conn=None) -> List[Memory]:
        """
        Get memories with personality-based weighting.
        Different NPCs remember differently based on their stats.
        """
        # Get NPC stats
        stats = await self.get_npc_stats(conn=conn)
        if not stats:
            # Fallback to standard retrieval
            return await self.retrieve_memories(query=query, conn=conn)
            
        # Retrieve base memories
        memories = await self.retrieve_memories(
            query=query,
            limit=10,  # Get more than we need for filtering
            conn=conn
        )
        
        if not memories:
            return []
            
        # Apply personality bias
        dominance = stats.get('dominance', 0)
        cruelty = stats.get('cruelty', 0)
        
        for memory in memories:
            bias_score = 0
            
            # High dominance NPCs emphasize power dynamics
            if dominance > 50:
                if "command" in memory.text.lower() or "order" in memory.text.lower():
                    bias_score += (dominance - 50) / 10
                    
            # Cruel NPCs emphasize negative experiences
            if cruelty > 50:
                if memory.emotional_intensity < 0 or "negative" in memory.tags:
                    bias_score += (cruelty - 50) / 10
                    
            # Store bias in metadata
            memory.metadata["bias_score"] = bias_score
            
        # Sort by combined relevance and bias
        memories.sort(key=lambda m: 
            m.metadata.get("relevance_score", 0) + m.metadata.get("bias_score", 0), 
            reverse=True
        )
        
        # Return top memories after bias
        return memories[:5]
    
    @with_transaction
    async def get_recent_revelations(self, conn=None) -> List[Dict[str, Any]]:
        """
        Get recent revelations about this NPC.
        """
        rows = await conn.fetch("""
            SELECT id, narrative_stage, revelation_text, timestamp
            FROM NPCRevelations
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            ORDER BY timestamp DESC
            LIMIT 5
        """, self.user_id, self.conversation_id, self.npc_id)
        
        return [dict(row) for row in rows]
    
    @with_transaction
    async def get_mask_slippage_events(self, conn=None) -> List[Dict[str, Any]]:
        """
        Get mask slippage events for this NPC.
        """
        rows = await conn.fetch("""
            SELECT npc_id, mask_data->>'integrity' as integrity
            FROM NPCMasks
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
        """, self.user_id, self.conversation_id, self.npc_id)
        
        if not rows:
            return []
            
        mask_info = dict(rows[0])
        integrity = float(mask_info.get('integrity', 100))
        
        # Get actual slippage events if integrity is compromised
        if integrity < 90:
            evolution_rows = await conn.fetch("""
                SELECT id, mask_slippage_events
                FROM NPCEvolution
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                ORDER BY id DESC
                LIMIT 1
            """, self.user_id, self.conversation_id, self.npc_id)
            
            if evolution_rows and evolution_rows[0].get('mask_slippage_events'):
                events = evolution_rows[0]['mask_slippage_events']
                return events if isinstance(events, list) else json.loads(events)
        
        return []


class NyxMemoryManager(UnifiedMemoryManager):
    """
    Specialized memory manager for the Nyx DM character.
    Extends UnifiedMemoryManager with narrative and reflection capabilities.
    """
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(
            entity_type="nyx", 
            entity_id=0,  # Nyx is always entity_id 0 
            user_id=user_id, 
            conversation_id=conversation_id
        )
    
    @with_transaction
    async def add_reflection(self, 
                           reflection: str, 
                           reflection_type: str = "general",
                           significance: int = MemorySignificance.MEDIUM,
                           tags: List[str] = None,
                           conn=None) -> int:
        """
        Add a meta-level reflection about the narrative, player, or game state.
        """
        return await self.add_memory(
            memory=Memory(
                text=reflection,
                memory_type=MemoryType.REFLECTION,
                significance=significance,
                emotional_intensity=0,  # Reflections don't have emotional component
                tags=(tags or []) + ["reflection", reflection_type],
                timestamp=datetime.now()
            ),
            conn=conn
        )
    
    @with_transaction
    async def get_narrative_state(self, conn=None) -> Dict[str, Any]:
        """
        Get current narrative state.
        """
        # Try to get from NyxAgentState
        row = await conn.fetchrow("""
            SELECT current_goals, predicted_futures, narrative_assessment
            FROM NyxAgentState
            WHERE user_id = $1 AND conversation_id = $2
        """, self.user_id, self.conversation_id)
        
        if not row:
            return {
                "goals": [],
                "predictions": [],
                "arcs": [],
                "assessment": {}
            }
            
        # Parse JSON fields
        current_goals = row["current_goals"] if isinstance(row["current_goals"], dict) else json.loads(row["current_goals"] or "{}")
        predicted_futures = row["predicted_futures"] if isinstance(row["predicted_futures"], list) else json.loads(row["predicted_futures"] or "[]")
        narrative_assessment = row["narrative_assessment"] if isinstance(row["narrative_assessment"], dict) else json.loads(row["narrative_assessment"] or "{}")
        
        # Get narrative arcs from CurrentRoleplay
        arcs_row = await conn.fetchrow("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
        """, self.user_id, self.conversation_id)
        
        arcs = {}
        if arcs_row and arcs_row["value"]:
            arcs = json.loads(arcs_row["value"]) if isinstance(arcs_row["value"], str) else arcs_row["value"]
        
        return {
            "goals": current_goals.get("goals", []),
            "predictions": predicted_futures,
            "arcs": arcs,
            "assessment": narrative_assessment
        }
    
    @with_transaction
    async def update_narrative_assessment(self, 
                                        assessment: Dict[str, Any],
                                        conn=None) -> bool:
        """
        Update Nyx's assessment of the narrative.
        """
        # See if NyxAgentState exists
        exists = await conn.fetchval("""
            SELECT 1
            FROM NyxAgentState
            WHERE user_id = $1 AND conversation_id = $2
        """, self.user_id, self.conversation_id)
        
        if exists:
            # Update existing record
            await conn.execute("""
                UPDATE NyxAgentState
                SET narrative_assessment = $1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = $2 AND conversation_id = $3
            """, json.dumps(assessment), self.user_id, self.conversation_id)
        else:
            # Insert new record
            await conn.execute("""
                INSERT INTO NyxAgentState
                (user_id, conversation_id, narrative_assessment, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            """, self.user_id, self.conversation_id, json.dumps(assessment))
        
        return True
    
    @with_transaction
    async def compile_player_model(self, conn=None) -> Dict[str, Any]:
        """
        Create a comprehensive player model from various data sources.
        """
        # Get player stats
        player_row = await conn.fetchrow("""
            SELECT corruption, confidence, willpower, obedience, 
                   dependency, lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id = $1 AND conversation_id = $2
            LIMIT 1
        """, self.user_id, self.conversation_id)
        
        if not player_row:
            return {}
            
        # Get player memories to analyze behavior
        player_behavior_memories = await self.retrieve_memories(
            query="Chase player behavior",
            tags=["player_action"],
            limit=20,
            conn=conn
        )
        
        # Get player kink data
        kink_rows = await conn.fetch("""
            SELECT kink_type, level, intensity_preference, frequency
            FROM UserKinkProfile
            WHERE user_id = $1
        """, self.user_id)
        
        # Compile the model
        model = {
            "stats": dict(player_row),
            "kinks": {row["kink_type"]: {
                "level": row["level"],
                "intensity": row["intensity_preference"],
                "frequency": row["frequency"]
            } for row in kink_rows},
            "behavior_patterns": self._extract_behavior_patterns(player_behavior_memories),
            "predicted_responses": {},
            "development_status": {
                "corruption_stage": self._calculate_stage(player_row["corruption"]),
                "obedience_stage": self._calculate_stage(player_row["obedience"]),
                "dependency_stage": self._calculate_stage(player_row["dependency"])
            }
        }
        
        # Store in CurrentRoleplay
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'NyxPlayerModel', $3)
            ON CONFLICT (user_id, conversation_id, key) 
            DO UPDATE SET value = $3
        """, self.user_id, self.conversation_id, json.dumps(model))
        
        return model
    
    def _extract_behavior_patterns(self, memories: List[Memory]) -> Dict[str, Any]:
        """
        Extract behavior patterns from memories.
        """
        patterns = {
            "defiance_frequency": 0,
            "submission_frequency": 0,
            "curiosity_level": 0,
            "aggression_level": 0
        }
        
        for memory in memories:
            text = memory.text.lower()
            
            if any(word in text for word in ["defied", "refused", "rejected", "resisted"]):
                patterns["defiance_frequency"] += 1
                
            if any(word in text for word in ["obeyed", "submitted", "accepted", "complied"]):
                patterns["submission_frequency"] += 1
                
            if any(word in text for word in ["asked", "questioned", "explored", "investigated"]):
                patterns["curiosity_level"] += 1
                
            if any(word in text for word in ["attacked", "confronted", "threatened", "demanded"]):
                patterns["aggression_level"] += 1
        
        # Normalize
        total = len(memories) or 1
        for key in patterns:
            patterns[key] = int((patterns[key] / total) * 100)
            
        return patterns
    
    def _calculate_stage(self, value: int) -> str:
        """
        Calculate development stage based on stat value.
        """
        if value < 20:
            return "initial"
        elif value < 40:
            return "developing"
        elif value < 60:
            return "established"
        elif value < 80:
            return "advanced"
        else:
            return "complete"
    
    @with_transaction
    async def generate_plot_hooks(self, count: int = 3, conn=None) -> List[Dict[str, Any]]:
        """
        Generate potential plot hooks based on current game state.
        """
        # Get player kinks to incorporate
        kink_rows = await conn.fetch("""
            SELECT kink_type, level
            FROM UserKinkProfile
            WHERE user_id = $1 AND level >= 2
        """, self.user_id)
        
        kinks = [row["kink_type"] for row in kink_rows]
        
        # Get NPC data for potential hooks
        npc_rows = await conn.fetch("""
            SELECT npc_id, npc_name, dominance, cruelty, introduced
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2
            ORDER BY CASE WHEN introduced THEN 0 ELSE 1 END, dominance DESC
            LIMIT 5
        """, self.user_id, self.conversation_id)
        
        # Get current environment
        env_row = await conn.fetchrow("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id = $1 AND conversation_id = $2 AND key = 'EnvironmentDesc'
        """, self.user_id, self.conversation_id)
        
        environment = env_row["value"] if env_row else "Default environment"
        
        # Generate hooks
        hooks = []
        
        # Basic hook templates
        templates = [
            {"type": "challenge", "text": "Challenge from {npc} testing player's {stat}"},
            {"type": "temptation", "text": "Temptation by {npc} targeting player's {kink}"},
            {"type": "revelation", "text": "Revelation about {npc}'s true nature"},
            {"type": "quest", "text": "Quest involving {location} with {npc}"},
            {"type": "ritual", "text": "Ritual to increase player's {stat} orchestrated by {npc}"},
            {"type": "secret", "text": "Secret about the environment involving {npc}"},
            {"type": "choice", "text": "Meaningful choice between {npc1} and {npc2}"}
        ]
        
        # Get player stats
        player_row = await conn.fetchrow("""
            SELECT corruption, obedience, dependency, willpower
            FROM PlayerStats
            WHERE user_id = $1 AND conversation_id = $2
            LIMIT 1
        """, self.user_id, self.conversation_id)
        
        if not player_row:
            player_stats = ["corruption", "obedience", "dependency", "willpower"]
        else:
            # Find stats that are most interesting to develop
            player_stats = []
            for stat, value in player_row.items():
                # Focus on stats in the middle range
                if 30 <= value <= 70:
                    player_stats.append(stat)
                    
            if not player_stats:
                player_stats = ["corruption", "obedience", "dependency", "willpower"]
        
        # Get locations
        location_rows = await conn.fetch("""
            SELECT location_name 
            FROM Locations
            WHERE user_id = $1 AND conversation_id = $2
            ORDER BY RANDOM()
            LIMIT 3
        """, self.user_id, self.conversation_id)
        
        locations = [row["location_name"] for row in location_rows]
        if not locations:
            locations = ["mysterious location", "hidden room", "secret area"]
        
        # Generate hooks
        for i in range(min(count, len(templates))):
            template = random.choice(templates)
            templates.remove(template)  # Ensure variety
            
            hook = {
                "type": template["type"],
                "description": template["text"],
                "details": {}
            }
            
            # Fill in template
            if "{npc}" in hook["description"]:
                if npc_rows:
                    npc = random.choice(npc_rows)
                    hook["description"] = hook["description"].replace("{npc}", npc["npc_name"])
                    hook["details"]["npc_id"] = npc["npc_id"]
                    hook["details"]["npc_name"] = npc["npc_name"]
                else:
                    hook["description"] = hook["description"].replace("{npc}", "a mysterious character")
            
            if "{npc1}" in hook["description"] and "{npc2}" in hook["description"]:
                if len(npc_rows) >= 2:
                    npc1, npc2 = random.sample(npc_rows, 2)
                    hook["description"] = hook["description"].replace("{npc1}", npc1["npc_name"])
                    hook["description"] = hook["description"].replace("{npc2}", npc2["npc_name"])
                    hook["details"]["npc1_id"] = npc1["npc_id"]
                    hook["details"]["npc2_id"] = npc2["npc_id"]
                else:
                    hook["description"] = hook["description"].replace("{npc1}", "one character")
                    hook["description"] = hook["description"].replace("{npc2}", "another character")
            
            if "{stat}" in hook["description"]:
                stat = random.choice(player_stats)
                hook["description"] = hook["description"].replace("{stat}", stat)
                hook["details"]["stat"] = stat
            
            if "{kink}" in hook["description"]:
                if kinks:
                    kink = random.choice(kinks)
                    hook["description"] = hook["description"].replace("{kink}", kink)
                    hook["details"]["kink"] = kink
                else:
                    hook["description"] = hook["description"].replace("{kink}", "desire")
            
            if "{location}" in hook["description"]:
                location = random.choice(locations)
                hook["description"] = hook["description"].replace("{location}", location)
                hook["details"]["location"] = location
            
            hooks.append(hook)
        
        return hooks


class PlayerMemoryManager(UnifiedMemoryManager):
    """
    Specialized memory manager for player memories.
    """
    def __init__(self, player_name: str, user_id: int, conversation_id: int):
        super().__init__(
            entity_type="player", 
            entity_id=user_id,  # For player, entity_id is user_id 
            user_id=user_id, 
            conversation_id=conversation_id
        )
        self.player_name = player_name
    
    @with_transaction
    async def add_journal_entry(self, 
                              entry_text: str,
                              entry_type: str = "observation",
                              significance: int = MemorySignificance.MEDIUM,
                              fantasy_flag: bool = False,
                              intensity_level: int = 0,
                              conn=None) -> int:
        """
        Add a journal entry to player memory and the PlayerJournal table.
        """
        # First add to unified memory system
        memory_id = await self.add_memory(
            memory=Memory(
                text=entry_text,
                memory_type=MemoryType.OBSERVATION,
                significance=significance,
                emotional_intensity=intensity_level * 20,  # Scale 0-5 to 0-100
                tags=[entry_type, "journal"],
                metadata={
                    "fantasy_flag": fantasy_flag,
                    "intensity_level": intensity_level
                },
                timestamp=datetime.now()
            ),
            conn=conn
        )
        
        # Also add to PlayerJournal
        await conn.execute("""
            INSERT INTO PlayerJournal (
                user_id, conversation_id, entry_type, entry_text,
                fantasy_flag, intensity_level, timestamp
            )
            VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
        """, self.user_id, self.conversation_id, entry_type, entry_text, 
            fantasy_flag, intensity_level)
            
        return memory_id
    
    @with_transaction
    async def get_journal_history(self, 
                                entry_type: Optional[str] = None,
                                limit: int = 10,
                                conn=None) -> List[Dict[str, Any]]:
        """
        Get journal entries from PlayerJournal.
        """
        query = """
            SELECT id, entry_type, entry_text, fantasy_flag, intensity_level, timestamp
            FROM PlayerJournal
            WHERE user_id = $1 AND conversation_id = $2
        """
        params = [self.user_id, self.conversation_id]
        
        if entry_type:
            query += " AND entry_type = $3"
            params.append(entry_type)
            
        query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
        
        return [dict(row) for row in rows]
    
    @with_transaction
    async def record_stat_change(self,
                               stat_name: str,
                               old_value: int,
                               new_value: int,
                               cause: str,
                               conn=None) -> bool:
        """
        Record a stat change in StatsHistory and player memory.
        """
        # Add to StatsHistory
        await conn.execute("""
            INSERT INTO StatsHistory (
                user_id, conversation_id, player_name, stat_name,
                old_value, new_value, cause, timestamp
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
        """, self.user_id, self.conversation_id, self.player_name,
            stat_name, old_value, new_value, cause)
            
        # Create memory text
        memory_text = f"My {stat_name} changed from {old_value} to {new_value}. Cause: {cause}"
        
        # Add to unified memory
        await self.add_memory(
            memory=Memory(
                text=memory_text,
                memory_type=MemoryType.OBSERVATION,
                significance=MemorySignificance.MEDIUM,
                emotional_intensity=40,
                tags=["stat_change", stat_name],
                metadata={
                    "stat_name": stat_name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "change": new_value - old_value,
                    "cause": cause
                },
                timestamp=datetime.now()
            ),
            conn=conn
        )
        
        return True
    
    @with_transaction
    async def compile_player_profile(self, conn=None) -> Dict[str, Any]:
        """
        Generate a comprehensive player profile with stats, history, etc.
        """
        # Get current stats
        stats_row = await conn.fetchrow("""
            SELECT corruption, confidence, willpower, obedience, 
                   dependency, lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            LIMIT 1
        """, self.user_id, self.conversation_id, self.player_name)
        
        if not stats_row:
            return {}
            
        # Get stats history
        stat_history_rows = await conn.fetch("""
            SELECT stat_name, old_value, new_value, cause, timestamp
            FROM StatsHistory
            WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            ORDER BY timestamp DESC
            LIMIT 20
        """, self.user_id, self.conversation_id, self.player_name)
        
        # Get inventory
        inventory_rows = await conn.fetch("""
            SELECT item_name, item_description, item_effect, quantity, category
            FROM PlayerInventory
            WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
        """, self.user_id, self.conversation_id, self.player_name)
        
        # Get perks
        perk_rows = await conn.fetch("""
            SELECT perk_name, perk_description, perk_effect
            FROM PlayerPerks
            WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
        """, self.user_id, self.conversation_id, self.player_name)
        
        # Get kinks
        kink_rows = await conn.fetch("""
            SELECT kink_type, level, intensity_preference, frequency, trigger_context
            FROM UserKinkProfile
            WHERE user_id = $1
        """, self.user_id)
        
        # Get recent journal entries
        journal_rows = await conn.fetch("""
            SELECT entry_type, entry_text, fantasy_flag, intensity_level, timestamp
            FROM PlayerJournal
            WHERE user_id = $1 AND conversation_id = $2
            ORDER BY timestamp DESC
            LIMIT 10
        """, self.user_id, self.conversation_id)
        
        # Track major milestones
        milestones = []
        for row in stat_history_rows:
            if abs(row["new_value"] - row["old_value"]) >= 10:  # Big change
                milestones.append({
                    "type": "stat_change",
                    "stat": row["stat_name"],
                    "change": row["new_value"] - row["old_value"],
                    "cause": row["cause"],
                    "timestamp": row["timestamp"].isoformat()
                })
                
        # Check for significant journal entries
        for row in journal_rows:
            if row["intensity_level"] >= 3:
                milestones.append({
                    "type": "journal_entry",
                    "entry_type": row["entry_type"],
                    "text": row["entry_text"][:100] + "..." if len(row["entry_text"]) > 100 else row["entry_text"],
                    "intensity": row["intensity_level"],
                    "timestamp": row["timestamp"].isoformat()
                })
        
        # Compile profile
        profile = {
            "player_name": self.player_name,
            "stats": dict(stats_row),
            "inventory": [dict(row) for row in inventory_rows],
            "perks": [dict(row) for row in perk_rows],
            "kinks": {row["kink_type"]: {
                "level": row["level"],
                "intensity": row["intensity_preference"],
                "frequency": row["frequency"],
                "triggers": row["trigger_context"] if isinstance(row["trigger_context"], dict) 
                           else json.loads(row["trigger_context"] or "{}")
            } for row in kink_rows},
            "recent_journal": [dict(row) for row in journal_rows],
            "stat_changes": [dict(row) for row in stat_history_rows],
            "milestones": sorted(milestones, key=lambda x: x["timestamp"], reverse=True)
        }
        
        return profile
