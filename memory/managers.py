# memory/managers.py

import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple, Set
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
from agents import Agent, ModelSettings                 # already imported earlier?
from agents import Runner

logger = logging.getLogger("memory_managers")

plot_hook_generator = Agent(
    name="PlotHookGenerator",
    instructions="""
    You design concise, provocative plot hooks for a dark-erotic, femdom-themed
    RPG.  Given game state (player stats & kinks, key NPCs, environment, open
    locations) and a requested number N, respond with EXACTLY a JSON list of N
    objects.  For each hook return:
      - type: one of ["challenge","temptation","revelation","quest","ritual",
                      "secret","choice"]
      - description: 1-2 sentences written for the GM to drop into play
      - details:      any useful keys (npc_id, npc_name, stat, kink, location…)
    Keep text intriguing. Do not wrap the JSON in markdown fences.
    """,
    model="gpt-5-nano",
    model_settings=ModelSettings(temperature=0.8))

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
        Ask the PlotHookGenerator agent for <count> bespoke hooks that fit the
        current game state (kinks, NPCs, environment, stats, locations).
        """
        # ---------- Gather state ----------
        kink_rows = await conn.fetch(
            "SELECT kink_type, level FROM UserKinkProfile WHERE user_id=$1 AND level>=2",
            self.user_id
        )
        kinks = [row["kink_type"] for row in kink_rows]
    
        npc_rows = await conn.fetch(
            """
            SELECT npc_id, npc_name, dominance, cruelty, introduced
            FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY CASE WHEN introduced THEN 0 ELSE 1 END, dominance DESC
            LIMIT 5
            """,
            self.user_id, self.conversation_id
        )
    
        env_row = await conn.fetchrow(
            """
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc'
            """,
            self.user_id, self.conversation_id
        )
        environment = env_row["value"] if env_row else "a neutral setting"
    
        player_row = await conn.fetchrow(
            """
            SELECT corruption, obedience, dependency, willpower
            FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2
            LIMIT 1
            """,
            self.user_id, self.conversation_id
        )
        player_stats = dict(player_row) if player_row else {}
    
        location_rows = await conn.fetch(
            """
            SELECT location_name
            FROM Locations
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY RANDOM() LIMIT 5
            """,
            self.user_id, self.conversation_id
        )
        locations = [row["location_name"] for row in location_rows]
    
        # ---------- Build prompt ----------
        prompt = json.dumps({
            "request_count": count,
            "environment": environment,
            "player_stats": player_stats,
            "player_kinks": kinks,
            "locations": locations,
            "npcs": [
                {
                    "npc_id": r["npc_id"],
                    "npc_name": r["npc_name"],
                    "dominance": r["dominance"],
                    "cruelty": r["cruelty"],
                    "introduced": r["introduced"],
                }
                for r in npc_rows
            ],
        }, ensure_ascii=False)
    
        # ---------- Call the agent ----------
        try:
            result = await Runner.run(plot_hook_generator, prompt)
            hooks_raw = result.output.strip()
    
            # Make sure we’ve got valid JSON
            hooks: List[Dict[str, Any]] = json.loads(hooks_raw)
            # guard: truncate / pad to exactly `count`
            hooks = hooks[:count]
    
            return hooks
    
        except Exception as e:
            logger.error(f"Plot-hook generation failed: {e}", exc_info=True)
            # graceful fallback – keep old deterministic behaviour if desired
            return [{
                "type": "error",
                "description": "Failed to generate plot hooks.",
                "details": {"reason": str(e)}
            }]


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


class ConflictMemoryManager(UnifiedMemoryManager):
    """
    Specialized memory manager for handling conflict-related memories and their impact on relationships.
    """
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(
            entity_type="conflict",
            entity_id=0,  # Conflicts are managed at the conversation level
            user_id=user_id,
            conversation_id=conversation_id
        )
    
    @with_transaction
    async def record_conflict_resolution(self,
                                       conflict_id: int,
                                       resolution_data: Dict[str, Any],
                                       stakeholders: List[Dict[str, Any]],
                                       conn=None) -> Dict[str, Any]:
        """
        Record a conflict resolution and its impact on relationships.
        
        Args:
            conflict_id: ID of the resolved conflict
            resolution_data: Data about how the conflict was resolved
            stakeholders: List of stakeholders involved in the conflict
        """
        # Record the resolution as a memory
        resolution_memory = await self.add_memory(
            memory=Memory(
                text=f"Conflict {conflict_id} resolved: {resolution_data.get('resolution_type', 'unknown')}",
                memory_type=MemoryType.CONFLICT_RESOLUTION,
                significance=MemorySignificance.HIGH,
                emotional_intensity=resolution_data.get('emotional_intensity', 0.5),
                tags=["conflict", "resolution", resolution_data.get('resolution_type', 'unknown')],
                timestamp=datetime.now()
            ),
            conn=conn
        )
        
        # Update relationship memories for each stakeholder
        relationship_updates = []
        for stakeholder in stakeholders:
            entity_type = stakeholder.get('entity_type')
            entity_id = stakeholder.get('entity_id')
            outcome = stakeholder.get('outcome')
            
            # Record stakeholder-specific memory
            await self.add_memory(
                memory=Memory(
                    text=f"Stakeholder {entity_type} {entity_id} outcome: {outcome}",
                    memory_type=MemoryType.CONFLICT_OUTCOME,
                    significance=MemorySignificance.MEDIUM,
                    emotional_intensity=outcome.get('emotional_impact', 0.3),
                    tags=["conflict", "stakeholder", outcome.get('outcome_type', 'unknown')],
                    timestamp=datetime.now()
                ),
                conn=conn
            )
            
            # Update relationships with other stakeholders
            for other_stakeholder in stakeholders:
                if other_stakeholder['entity_id'] != entity_id:
                    relationship_change = self._calculate_relationship_change(
                        outcome,
                        other_stakeholder.get('outcome', {}),
                        resolution_data
                    )
                    
                    relationship_updates.append({
                        'entity1_type': entity_type,
                        'entity1_id': entity_id,
                        'entity2_type': other_stakeholder['entity_type'],
                        'entity2_id': other_stakeholder['entity_id'],
                        'change': relationship_change
                    })
        
        return {
            'resolution_memory_id': resolution_memory.get('memory_id'),
            'relationship_updates': relationship_updates
        }
    
    def _calculate_relationship_change(self,
                                    outcome1: Dict[str, Any],
                                    outcome2: Dict[str, Any],
                                    resolution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate how a conflict resolution affects relationships between stakeholders.
        """
        change = {
            'trust_change': 0,
            'respect_change': 0,
            'affinity_change': 0,
            'rivalry_change': 0
        }
        
        # Calculate changes based on outcomes
        if outcome1.get('outcome_type') == 'victory' and outcome2.get('outcome_type') == 'defeat':
            change['trust_change'] = -0.2
            change['respect_change'] = 0.1
            change['rivalry_change'] = 0.3
        elif outcome1.get('outcome_type') == 'compromise':
            change['trust_change'] = 0.1
            change['respect_change'] = 0.2
            change['affinity_change'] = 0.1
        elif outcome1.get('outcome_type') == 'cooperation':
            change['trust_change'] = 0.3
            change['respect_change'] = 0.2
            change['affinity_change'] = 0.2
        
        # Adjust based on resolution type
        resolution_type = resolution_data.get('resolution_type')
        if resolution_type == 'peaceful':
            change['trust_change'] += 0.1
            change['rivalry_change'] -= 0.1
        elif resolution_type == 'violent':
            change['trust_change'] -= 0.2
            change['rivalry_change'] += 0.2
        
        return change


class LoreMemoryManager(UnifiedMemoryManager):
    """
    Specialized memory manager for handling lore generation and relevance tracking.
    """
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(
            entity_type="lore",
            entity_id=0,  # Lore is managed at the conversation level
            user_id=user_id,
            conversation_id=conversation_id
        )
        self.pattern_cache = {}
        self.relevance_cache = {}
    
    @with_transaction
    async def generate_lore_from_memories(self,
                                        context: Dict[str, Any],
                                        conn=None) -> Dict[str, Any]:
        """
        Generate new lore based on memory patterns and current context.
        
        Args:
            context: Current game context
        """
        # Try to get from cache first
        cache_key = f"lore_patterns:{self.user_id}:{self.conversation_id}"
        cached_patterns = await get_cache(cache_key)
        
        if cached_patterns:
            patterns = cached_patterns
        else:
            # Get relevant memories for lore generation
            memories = await self.get_memories(
                memory_types=[MemoryType.EVENT, MemoryType.RELATIONSHIP, MemoryType.CONFLICT],
                limit=50,
                conn=conn
            )
            
            # Analyze memory patterns
            patterns = self._analyze_memory_patterns(memories)
            
            # Cache the patterns
            await set_cache(cache_key, patterns, ttl=300)  # Cache for 5 minutes
        
        # Generate lore based on patterns and context
        lore = self._generate_lore_from_patterns(patterns, context)
        
        # Record the generated lore as a memory
        lore_memory = await self.add_memory(
            memory=Memory(
                text=lore['text'],
                memory_type=MemoryType.LORE,
                significance=MemorySignificance.HIGH,
                emotional_intensity=0.3,
                tags=["lore", lore.get('category', 'general')] + lore.get('themes', []),
                metadata={
                    'patterns_used': patterns,
                    'context_snapshot': context,
                    'generation_timestamp': datetime.now().isoformat()
                },
                timestamp=datetime.now()
            ),
            conn=conn
        )
        
        # Update lore relevance with caching
        await self._update_lore_relevance(lore_memory.get('memory_id'), context, conn)
        
        return {
            'lore_id': lore_memory.get('memory_id'),
            'lore': lore,
            'patterns_used': patterns
        }
    
    def _analyze_memory_patterns(self, memories: List[Memory]) -> Dict[str, Any]:
        """
        Analyze patterns in memories that could form the basis for lore.
        """
        patterns = {
            'themes': {},
            'character_arcs': {},
            'relationships': {},
            'conflicts': {},
            'locations': {},
            'emotional_trends': {},
            'recurring_elements': {},
            'narrative_threads': []
        }
        
        # Track recurring elements and their connections
        element_connections = {}
        
        for memory in memories:
            # Extract themes with emotional context
            themes = self._extract_themes(memory.text)
            for theme in themes:
                if theme not in patterns['themes']:
                    patterns['themes'][theme] = {
                        'count': 0,
                        'emotional_context': [],
                        'related_elements': set()
                    }
                patterns['themes'][theme]['count'] += 1
                patterns['themes'][theme]['emotional_context'].append(memory.emotional_intensity)
            
            # Extract character arcs with progression
            if memory.memory_type == MemoryType.EVENT:
                character_arcs = self._extract_character_arcs(memory)
                for arc in character_arcs:
                    if arc not in patterns['character_arcs']:
                        patterns['character_arcs'][arc] = {
                            'stages': [],
                            'progression': 0,
                            'key_events': []
                        }
                    arc_data = patterns['character_arcs'][arc]
                    arc_data['stages'].append(memory.timestamp)
                    arc_data['key_events'].append(memory.text)
            
            # Extract relationships with dynamics
            if memory.memory_type == MemoryType.RELATIONSHIP:
                relationship = self._extract_relationship(memory)
                if relationship:
                    if relationship not in patterns['relationships']:
                        patterns['relationships'][relationship] = {
                            'dynamics': [],
                            'evolution': [],
                            'key_interactions': []
                        }
                    rel_data = patterns['relationships'][relationship]
                    rel_data['dynamics'].append(memory.metadata.get('relationship_dynamic', 'neutral'))
                    rel_data['evolution'].append({
                        'timestamp': memory.timestamp,
                        'state': memory.text
                    })
            
            # Track emotional trends
            emotion = memory.metadata.get('primary_emotion', 'neutral')
            if emotion not in patterns['emotional_trends']:
                patterns['emotional_trends'][emotion] = {
                    'count': 0,
                    'intensity_sum': 0,
                    'contexts': set()
                }
            patterns['emotional_trends'][emotion]['count'] += 1
            patterns['emotional_trends'][emotion]['intensity_sum'] += memory.emotional_intensity
            
            # Extract and connect recurring elements
            elements = self._extract_recurring_elements(memory.text)
            for elem in elements:
                if elem not in patterns['recurring_elements']:
                    patterns['recurring_elements'][elem] = {
                        'count': 0,
                        'contexts': set(),
                        'connections': set()
                    }
                patterns['recurring_elements'][elem]['count'] += 1
                patterns['recurring_elements'][elem]['contexts'].add(memory.memory_type)
                
                # Track connections between elements
                for other_elem in elements:
                    if other_elem != elem:
                        key = tuple(sorted([elem, other_elem]))
                        element_connections[key] = element_connections.get(key, 0) + 1
        
        # Identify narrative threads from element connections
        significant_connections = {k: v for k, v in element_connections.items() if v >= 2}
        patterns['narrative_threads'] = self._identify_narrative_threads(significant_connections)
        
        return patterns
    
    def _identify_narrative_threads(self, connections: Dict[Tuple[str, str], int]) -> List[Dict[str, Any]]:
        """Identify narrative threads from element connections."""
        threads = []
        visited = set()
        
        for (elem1, elem2), strength in sorted(connections.items(), key=lambda x: x[1], reverse=True):
            if elem1 not in visited or elem2 not in visited:
                thread = {
                    'elements': [elem1, elem2],
                    'strength': strength,
                    'potential_narrative': f"Connection between {elem1} and {elem2}"
                }
                threads.append(thread)
                visited.add(elem1)
                visited.add(elem2)
        
        return threads
    
    def _extract_recurring_elements(self, text: str) -> Set[str]:
        """Extract recurring elements from text."""
        elements = set()
        # Add implementation for extracting recurring elements
        # This could include named entities, key phrases, etc.
        return elements
    
    @with_transaction
    async def _update_lore_relevance(self,
                                   lore_id: int,
                                   context: Dict[str, Any],
                                   conn=None) -> None:
        """
        Update the relevance of lore based on current context with caching.
        """
        # Try to get from cache
        cache_key = f"lore_relevance:{lore_id}:{hash(str(context))}"
        cached_relevance = await get_cache(cache_key)
        
        if cached_relevance is not None:
            relevance_score = cached_relevance
        else:
            # Get the lore memory
            lore_memory = await self.get_memory(lore_id, conn=conn)
            if not lore_memory:
                return
            
            # Calculate relevance score
            relevance_score = self._calculate_lore_relevance(lore_memory, context)
            
            # Cache the result
            await set_cache(cache_key, relevance_score, ttl=300)  # Cache for 5 minutes
        
        # Update relevance in database
        await conn.execute("""
            UPDATE Memory
            SET relevance_score = $1,
                last_relevance_update = CURRENT_TIMESTAMP
            WHERE memory_id = $2
        """, relevance_score, lore_id)
    
    def _calculate_lore_relevance(self,
                                lore_memory: Memory,
                                context: Dict[str, Any]) -> float:
        """
        Calculate how relevant a piece of lore is to the current context.
        """
        relevance_score = 0.0
        
        # Check theme relevance
        lore_themes = set(lore_memory.tags)
        context_themes = set(context.get('themes', []))
        theme_overlap = len(lore_themes.intersection(context_themes))
        relevance_score += theme_overlap * 0.2
        
        # Check character relevance
        lore_characters = set(self._extract_characters(lore_memory.text))
        context_characters = set(context.get('characters', []))
        character_overlap = len(lore_characters.intersection(context_characters))
        relevance_score += character_overlap * 0.3
        
        # Check location relevance
        lore_locations = set(self._extract_locations(lore_memory.text))
        context_locations = set(context.get('locations', []))
        location_overlap = len(lore_locations.intersection(context_locations))
        relevance_score += location_overlap * 0.2
        
        # Check temporal relevance
        if self._is_temporally_relevant(lore_memory, context):
            relevance_score += 0.3
        
        return min(relevance_score, 1.0)

    def _generate_lore_from_patterns(self,
                                   patterns: Dict[str, Any],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate lore based on memory patterns and current context.
        """
        # Find dominant patterns with rich context
        dominant_themes = self._get_dominant_themes(patterns['themes'])
        dominant_arcs = self._get_dominant_arcs(patterns['character_arcs'])
        dominant_relationships = self._get_dominant_relationships(patterns['relationships'])
        emotional_trends = self._get_significant_emotional_trends(patterns['emotional_trends'])
        narrative_threads = patterns['narrative_threads'][:3]  # Top 3 threads
        
        # Generate lore text based on patterns
        lore_text = self._compose_lore_text(
            dominant_themes,
            dominant_arcs,
            dominant_relationships,
            emotional_trends,
            narrative_threads,
            context
        )
        
        # Determine lore category and metadata
        category = self._determine_lore_category(patterns, context)
        
        return {
            'text': lore_text,
            'category': category,
            'themes': [theme['name'] for theme in dominant_themes],
            'character_arcs': [arc['name'] for arc in dominant_arcs],
            'relationships': [rel['name'] for rel in dominant_relationships],
            'emotional_context': emotional_trends,
            'narrative_threads': [thread['potential_narrative'] for thread in narrative_threads],
            'metadata': {
                'pattern_confidence': self._calculate_pattern_confidence(patterns),
                'context_relevance': self._calculate_context_relevance(patterns, context),
                'emotional_intensity': sum(trend['average_intensity'] for trend in emotional_trends) / len(emotional_trends) if emotional_trends else 0
            }
        }

    def _get_dominant_themes(self, themes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get dominant themes with their context."""
        theme_list = []
        for name, data in themes.items():
            avg_emotional_intensity = sum(data['emotional_context']) / len(data['emotional_context']) if data['emotional_context'] else 0
            theme_list.append({
                'name': name,
                'count': data['count'],
                'average_emotional_intensity': avg_emotional_intensity,
                'related_elements': list(data['related_elements'])
            })
        
        return sorted(theme_list, key=lambda x: (x['count'], x['average_emotional_intensity']), reverse=True)[:3]

    def _get_dominant_arcs(self, arcs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get dominant character arcs with their progression."""
        arc_list = []
        for name, data in arcs.items():
            progression = len(data['stages'])
            arc_list.append({
                'name': name,
                'progression': progression,
                'stages': len(data['stages']),
                'key_events': data['key_events'][-3:]  # Last 3 key events
            })
        
        return sorted(arc_list, key=lambda x: (x['progression'], x['stages']), reverse=True)[:2]

    def _get_dominant_relationships(self, relationships: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get dominant relationships with their dynamics."""
        rel_list = []
        for name, data in relationships.items():
            rel_list.append({
                'name': name,
                'dynamics': data['dynamics'][-3:],  # Last 3 dynamics
                'evolution': data['evolution'][-3:],  # Last 3 evolution points
                'complexity': len(set(data['dynamics']))  # Unique dynamics count
            })
        
        return sorted(rel_list, key=lambda x: x['complexity'], reverse=True)[:2]

    def _get_significant_emotional_trends(self, trends: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get significant emotional trends."""
        trend_list = []
        for emotion, data in trends.items():
            if data['count'] >= 2:  # Only include trends with multiple occurrences
                trend_list.append({
                    'emotion': emotion,
                    'frequency': data['count'],
                    'average_intensity': data['intensity_sum'] / data['count'],
                    'contexts': list(data['contexts'])
                })
        
        return sorted(trend_list, key=lambda x: (x['frequency'], x['average_intensity']), reverse=True)[:3]

    def _compose_lore_text(self,
                          themes: List[Dict[str, Any]],
                          arcs: List[Dict[str, Any]],
                          relationships: List[Dict[str, Any]],
                          emotional_trends: List[Dict[str, Any]],
                          narrative_threads: List[Dict[str, Any]],
                          context: Dict[str, Any]) -> str:
        """
        Compose lore text from patterns and context.
        """
        lore_components = []
        
        # Add thematic elements
        if themes:
            theme_text = "The narrative weaves together themes of " + \
                        ", ".join(f"{t['name']} (with {t['average_emotional_intensity']:.1f} emotional intensity)" 
                                for t in themes[:-1])
            if len(themes) > 1:
                theme_text += f" and {themes[-1]['name']}"
            lore_components.append(theme_text)
        
        # Add character arc developments
        if arcs:
            for arc in arcs:
                arc_text = f"The arc of {arc['name']} has progressed through {arc['stages']} stages"
                if arc['key_events']:
                    arc_text += f", most recently: {arc['key_events'][-1]}"
                lore_components.append(arc_text)
        
        # Add relationship dynamics
        if relationships:
            for rel in relationships:
                rel_text = f"The relationship between {rel['name']} shows "
                if rel['dynamics']:
                    rel_text += f"recent {', '.join(rel['dynamics'][-2:])} dynamics"
                lore_components.append(rel_text)
        
        # Add emotional context
        if emotional_trends:
            emotion_text = "The emotional landscape is characterized by " + \
                         ", ".join(f"{t['emotion']} (intensity: {t['average_intensity']:.1f})" 
                                 for t in emotional_trends)
            lore_components.append(emotion_text)
        
        # Add narrative threads
        if narrative_threads:
            thread_text = "Key narrative threads include: " + \
                         "; ".join(thread['potential_narrative'] for thread in narrative_threads)
            lore_components.append(thread_text)
        
        # Combine all components
        return "\n\n".join(lore_components)

    def _determine_lore_category(self, patterns: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Determine the category of the generated lore."""
        # Count pattern types
        pattern_counts = {
            'character': len(patterns['character_arcs']),
            'relationship': len(patterns['relationships']),
            'location': len(patterns['locations']),
            'emotional': len(patterns['emotional_trends']),
            'narrative': len(patterns['narrative_threads'])
        }
        
        # Get the dominant pattern type
        dominant_type = max(pattern_counts.items(), key=lambda x: x[1])[0]
        
        # Map to lore categories
        category_mapping = {
            'character': 'character_development',
            'relationship': 'social_dynamics',
            'location': 'world_building',
            'emotional': 'psychological_insight',
            'narrative': 'plot_development'
        }
        
        return category_mapping.get(dominant_type, 'general')

    def _calculate_pattern_confidence(self, patterns: Dict[str, Any]) -> float:
        """Calculate confidence in the identified patterns."""
        confidence_scores = []
        
        # Theme confidence
        if patterns['themes']:
            max_theme_count = max(t['count'] for t in patterns['themes'].values())
            confidence_scores.append(min(max_theme_count / 10, 1.0))
        
        # Character arc confidence
        if patterns['character_arcs']:
            max_arc_stages = max(len(arc['stages']) for arc in patterns['character_arcs'].values())
            confidence_scores.append(min(max_arc_stages / 5, 1.0))
        
        # Relationship confidence
        if patterns['relationships']:
            max_rel_evolution = max(len(rel['evolution']) for rel in patterns['relationships'].values())
            confidence_scores.append(min(max_rel_evolution / 5, 1.0))
        
        # Emotional trend confidence
        if patterns['emotional_trends']:
            max_emotion_count = max(trend['count'] for trend in patterns['emotional_trends'].values())
            confidence_scores.append(min(max_emotion_count / 5, 1.0))
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    def _calculate_context_relevance(self, patterns: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how relevant the patterns are to the current context."""
        relevance_scores = []
        
        # Theme relevance
        context_themes = set(context.get('themes', []))
        pattern_themes = set(patterns['themes'].keys())
        theme_overlap = len(context_themes.intersection(pattern_themes))
        if context_themes:
            relevance_scores.append(theme_overlap / len(context_themes))
        
        # Character relevance
        context_characters = set(context.get('characters', []))
        pattern_characters = set(arc.split()[0] for arc in patterns['character_arcs'].keys())
        character_overlap = len(context_characters.intersection(pattern_characters))
        if context_characters:
            relevance_scores.append(character_overlap / len(context_characters))
        
        # Location relevance
        context_locations = set(context.get('locations', []))
        pattern_locations = set(patterns['locations'].keys())
        location_overlap = len(context_locations.intersection(pattern_locations))
        if context_locations:
            relevance_scores.append(location_overlap / len(context_locations))
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0


class ContextEvolutionManager(UnifiedMemoryManager):
    """
    Specialized memory manager for handling long-term context persistence and evolution.
    """
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(
            entity_type="context",
            entity_id=0,  # Context is managed at the conversation level
            user_id=user_id,
            conversation_id=conversation_id
        )
    
    @with_transaction
    async def update_context_evolution(self,
                                     current_context: Dict[str, Any],
                                     conn=None) -> Dict[str, Any]:
        """
        Update the evolution of context over time.
        
        Args:
            current_context: Current game context
        """
        # Get the latest context evolution
        latest_evolution = await self.get_latest_context_evolution(conn=conn)
        
        # Analyze changes in context
        changes = self._analyze_context_changes(latest_evolution, current_context)
        
        # Update context evolution
        evolution_id = await self._record_context_evolution(
            current_context,
            changes,
            conn=conn
        )
        
        # Update related memories
        await self._update_related_memories(evolution_id, changes, conn=conn)
        
        return {
            'evolution_id': evolution_id,
            'changes': changes,
            'context_snapshot': current_context
        }
    
    @with_transaction
    async def get_latest_context_evolution(self, conn=None) -> Dict[str, Any]:
        """
        Get the latest recorded context evolution.
        """
        result = await conn.fetchrow("""
            SELECT context_data, timestamp
            FROM ContextEvolution
            WHERE user_id = $1 AND conversation_id = $2
            ORDER BY timestamp DESC
            LIMIT 1
        """, self.user_id, self.conversation_id)
        
        if result:
            return {
                'context_data': result['context_data'],
                'timestamp': result['timestamp']
            }
        return None
    
    def _analyze_context_changes(self,
                               previous_context: Dict[str, Any],
                               current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze changes between previous and current context.
        """
        changes = {
            'themes': self._analyze_theme_changes(previous_context, current_context),
            'characters': self._analyze_character_changes(previous_context, current_context),
            'locations': self._analyze_location_changes(previous_context, current_context),
            'relationships': self._analyze_relationship_changes(previous_context, current_context),
            'conflicts': self._analyze_conflict_changes(previous_context, current_context)
        }
        
        # Calculate overall context shift
        changes['context_shift'] = self._calculate_context_shift(changes)
        
        return changes
    
    @with_transaction
    async def _record_context_evolution(self,
                                      current_context: Dict[str, Any],
                                      changes: Dict[str, Any],
                                      conn=None) -> int:
        """
        Record a new context evolution.
        """
        result = await conn.fetchrow("""
            INSERT INTO ContextEvolution (
                user_id, conversation_id, context_data, changes,
                context_shift, timestamp
            )
            VALUES ($1, $2, $3, $4, $5, NOW())
            RETURNING evolution_id
        """, self.user_id, self.conversation_id,
            json.dumps(current_context),
            json.dumps(changes),
            changes['context_shift'])
        
        return result['evolution_id']
    
    @with_transaction
    async def _update_related_memories(self,
                                     evolution_id: int,
                                     changes: Dict[str, Any],
                                     conn=None) -> None:
        """
        Update memories related to the context changes.
        """
        # Get memories that might be affected by the context changes
        affected_memories = await self._get_affected_memories(changes, conn=conn)
        
        for memory in affected_memories:
            # Update memory relevance
            new_relevance = self._calculate_new_memory_relevance(
                memory,
                changes
            )
            
            # Update memory in database
            await conn.execute("""
                UPDATE Memory
                SET relevance_score = $1,
                    last_context_update = NOW()
                WHERE memory_id = $2
            """, new_relevance, memory['memory_id'])
            
            # Record context evolution impact
            await conn.execute("""
                INSERT INTO MemoryContextEvolution (
                    memory_id, evolution_id, relevance_change
                )
                VALUES ($1, $2, $3)
            """, memory['memory_id'], evolution_id,
                new_relevance - memory.get('relevance_score', 0)) 
    
    def _calculate_context_shift(self, changes: Dict[str, Any]) -> float:
        """
        Calculate how significant the context shift is.
        """
        shift_score = 0.0
        
        # Theme changes
        theme_changes = changes['themes']
        shift_score += len(theme_changes.get('added', [])) * 0.2
        shift_score += len(theme_changes.get('removed', [])) * 0.2
        
        # Character changes
        character_changes = changes['characters']
        shift_score += len(character_changes.get('added', [])) * 0.3
        shift_score += len(character_changes.get('removed', [])) * 0.3
        
        # Location changes
        location_changes = changes['locations']
        shift_score += len(location_changes.get('added', [])) * 0.2
        shift_score += len(location_changes.get('removed', [])) * 0.2
        
        # Relationship changes
        relationship_changes = changes['relationships']
        shift_score += len(relationship_changes.get('added', [])) * 0.3
        shift_score += len(relationship_changes.get('removed', [])) * 0.3
        
        # Conflict changes
        conflict_changes = changes['conflicts']
        shift_score += len(conflict_changes.get('added', [])) * 0.4
        shift_score += len(conflict_changes.get('resolved', [])) * 0.4
        
        return min(shift_score, 1.0)
