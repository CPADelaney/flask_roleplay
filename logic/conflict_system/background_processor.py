# logic/conflict_system/background_processor.py
"""
Background processor for conflict system operations.
Moves heavy operations off the hot path into scheduled/on-demand execution.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from db.connection import get_db_connection_context
from logic.time_cycle import get_current_game_day

logger = logging.getLogger(__name__)

class ProcessingPriority(Enum):
    """Priority levels for background processing"""
    IMMEDIATE = 1  # Process if directly relevant to current scene
    HIGH = 2       # Process within current game day
    NORMAL = 3     # Process on next game day transition
    LOW = 4        # Process when convenient/idle

@dataclass
class ConflictContentLimits:
    """Limits for generated content per conflict"""
    max_news_items: int = 3
    news_generation_cooldown_days: int = 3  # Game days between news
    max_ripples_per_week: int = 5
    significant_event_threshold: float = 0.3  # 30% progress change triggers news

class BackgroundConflictProcessor:
    """
    Handles all background processing for conflicts.
    Replaces inline generation with smart, scheduled updates.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.limits = ConflictContentLimits()
        self._processing_queue: List[Dict[str, Any]] = []
        self._last_daily_process: Optional[int] = None
        
    async def should_generate_news(self, conflict_id: int) -> tuple[bool, str]:
        """
        Determine if we should generate news for a conflict.
        Returns (should_generate, reason)
        """
        async with get_db_connection_context() as conn:
            # Check existing news count
            news_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM BackgroundNews
                WHERE user_id = $1 AND conversation_id = $2 AND conflict_id = $3
                """,
                self.user_id, self.conversation_id, conflict_id
            )
            
            if news_count >= self.limits.max_news_items:
                return False, "max_news_reached"
            
            # Check last generation time
            last_news = await conn.fetchrow(
                """
                SELECT game_day FROM BackgroundNews
                WHERE user_id = $1 AND conversation_id = $2 AND conflict_id = $3
                ORDER BY game_day DESC LIMIT 1
                """,
                self.user_id, self.conversation_id, conflict_id
            )
            
            if last_news:
                current_day = await get_current_game_day(self.user_id, self.conversation_id)
                if current_day - last_news['game_day'] < self.limits.news_generation_cooldown_days:
                    return False, "cooldown_active"
            
            # Check for significant events
            conflict = await conn.fetchrow(
                """
                SELECT progress, last_significant_change FROM BackgroundConflicts
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                """,
                conflict_id, self.user_id, self.conversation_id
            )
            
            if conflict:
                if conflict['last_significant_change']:
                    last_change = json.loads(conflict['last_significant_change'])
                    if last_change.get('magnitude', 0) >= self.limits.significant_event_threshold:
                        return True, "significant_event"
            
            # Initial news for new conflicts
            if news_count == 0:
                return True, "initial_news"
                
            return False, "no_trigger"
    
    async def process_daily_updates(self, force: bool = False) -> Dict[str, Any]:
        """
        Process daily updates for all conflicts.
        Called on game day transitions, not every turn.
        """
        current_day = await get_current_game_day(self.user_id, self.conversation_id)
        
        # Skip if already processed today (unless forced)
        if not force and self._last_daily_process == current_day:
            return {"status": "already_processed", "day": current_day}
        
        results = {
            "day": current_day,
            "conflicts_updated": 0,
            "news_generated": [],
            "npcs_repositioned": 0,
            "opportunities_created": 0
        }
        
        async with get_db_connection_context() as conn:
            # Get all active conflicts
            conflicts = await conn.fetch(
                """
                SELECT id, name, progress, intensity, metadata
                FROM BackgroundConflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND status = 'active'
                """,
                self.user_id, self.conversation_id
            )
            
            for conflict in conflicts:
                conflict_id = conflict['id']
                
                # Determine if conflict needs attention
                should_gen, reason = await self.should_generate_news(conflict_id)
                
                if should_gen:
                    # Queue news generation (don't do it inline)
                    self._processing_queue.append({
                        'type': 'generate_news',
                        'conflict_id': conflict_id,
                        'priority': ProcessingPriority.HIGH if reason == 'significant_event' else ProcessingPriority.NORMAL,
                        'reason': reason
                    })
                    results['news_generated'].append(conflict['name'])
                
                # Update conflict progress (small daily tick)
                if conflict['intensity'] in ['visible_effects', 'ambient_tension']:
                    await self._tick_conflict_progress(conn, conflict_id, 0.5)  # Small daily progress
                    results['conflicts_updated'] += 1
        
        # Process NPC repositioning for the new day
        npc_count = await self._process_npc_daily_schedules()
        results['npcs_repositioned'] = npc_count
        
        # Check for opportunity windows closing
        opp_count = await self._check_expiring_opportunities()
        results['opportunities_created'] = opp_count
        
        self._last_daily_process = current_day
        return results
    
    async def process_scene_relevant_updates(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process only updates immediately relevant to current scene.
        Uses PostgreSQL JSON operators for metadata extraction.
        """
        location = scene_context.get('location', '')
        active_npcs = scene_context.get('npcs', [])
        topics = scene_context.get('conversation_topics', [])
    
        relevant_updates = {
            'immediate_news': None,
            'npc_updates': [],
            'ambient_effects': []
        }
    
        async with get_db_connection_context() as conn:
            # NOTE: (metadata -> 'key') returns JSON; cast to text to keep existing json.loads pipeline
            conflicts = await conn.fetch(
                """
                SELECT bc.*,
                       COALESCE((bc.metadata -> 'affected_locations')::text, '[]') AS locations,
                       COALESCE((bc.metadata -> 'key_figures')::text, '[]')       AS figures
                FROM BackgroundConflicts bc
                WHERE bc.user_id = $1
                  AND bc.conversation_id = $2
                  AND bc.status = 'active'
                  AND bc.intensity IN ('visible_effects', 'ambient_tension')
                """,
                self.user_id, self.conversation_id
            )
    
            for conflict in conflicts:
                # Check location relevance
                if self._is_location_affected(location, conflict):
                    # Only generate immediate context, not new content
                    ambient = await self._get_ambient_effect(conflict['id'])
                    if ambient:
                        relevant_updates['ambient_effects'].append(ambient)
    
                # Check NPC relevance
                for npc_id in active_npcs:
                    if self._is_npc_involved(npc_id, conflict):
                        # Queue NPC knowledge update for later
                        self._processing_queue.append({
                            'type': 'update_npc_knowledge',
                            'npc_id': npc_id,
                            'conflict_id': conflict['id'],
                            'priority': ProcessingPriority.HIGH
                        })
    
        return relevant_updates
    
    async def process_queued_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Process items from the background queue.
        Called by background workers, not in request path.
        """
        processed = []
        
        # Sort by priority
        self._processing_queue.sort(key=lambda x: x.get('priority', ProcessingPriority.NORMAL).value)
        
        items_to_process = self._processing_queue[:max_items]
        self._processing_queue = self._processing_queue[max_items:]
        
        for item in items_to_process:
            try:
                if item['type'] == 'generate_news':
                    result = await self._generate_single_news_item(item['conflict_id'])
                    processed.append({
                        'type': 'news',
                        'conflict_id': item['conflict_id'],
                        'result': result
                    })
                    
                elif item['type'] == 'update_npc_knowledge':
                    result = await self._update_npc_conflict_knowledge(
                        item['npc_id'], 
                        item['conflict_id']
                    )
                    processed.append({
                        'type': 'npc_knowledge',
                        'npc_id': item['npc_id'],
                        'result': result
                    })
                    
                elif item['type'] == 'generate_initial_conflicts':
                    # Generate initial background conflicts
                    from logic.conflict_system.background_grand_conflicts import BackgroundConflictOrchestrator
                    orchestrator = BackgroundConflictOrchestrator(self.user_id, self.conversation_id)
                    
                    conflicts_generated = 0
                    async with get_db_connection_context() as conn:
                        count = await conn.fetchval(
                            """
                            SELECT COUNT(*) FROM BackgroundConflicts
                            WHERE user_id = $1 AND conversation_id = $2 AND status = 'active'
                            """,
                            self.user_id, self.conversation_id
                        )
                        
                    # Generate 2-3 initial conflicts
                    while count < 2 and conflicts_generated < 3:
                        conflict = await orchestrator.generate_background_conflict()
                        if conflict:
                            conflicts_generated += 1
                            count += 1
                    
                    processed.append({
                        'type': 'initial_conflicts',
                        'generated': conflicts_generated,
                        'result': {'success': conflicts_generated > 0}
                    })
                    
            except Exception as e:
                logger.error(f"Error processing queued item {item}: {e}")
                
        return processed
    
    async def trigger_significant_event(self, conflict_id: int, event_data: Dict[str, Any]) -> bool:
        """
        Manually trigger news generation for significant events.
        Used when something important happens that warrants immediate news.
        """
        async with get_db_connection_context() as conn:
            # Record the significant change
            await conn.execute(
                """
                UPDATE BackgroundConflicts
                SET last_significant_change = $1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $2 AND user_id = $3 AND conversation_id = $4
                """,
                json.dumps({
                    'timestamp': datetime.utcnow().isoformat(),
                    'magnitude': event_data.get('magnitude', 0.5),
                    'description': event_data.get('description', '')
                }),
                conflict_id, self.user_id, self.conversation_id
            )
            
            # Add high priority news generation
            self._processing_queue.insert(0, {
                'type': 'generate_news',
                'conflict_id': conflict_id,
                'priority': ProcessingPriority.IMMEDIATE,
                'reason': 'manual_trigger',
                'event_data': event_data
            })
            
        return True
    
    # ========== Private Helper Methods ==========
    
    async def _tick_conflict_progress(self, conn, conflict_id: int, amount: float):
        """Small daily progress update for conflicts"""
        await conn.execute(
            """
            UPDATE BackgroundConflicts
            SET progress = LEAST(100.0, GREATEST(0.0, progress + $1)),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
            """,
            amount, conflict_id
        )
    
    async def _process_npc_daily_schedules(self) -> int:
        """
        Update NPC locations and activities for the new game day.
        This determines where NPCs can be found and what they're doing.
        """
        async with get_db_connection_context() as conn:
            # Get all active NPCs
            npcs = await conn.fetch(
                """
                SELECT npc_id, current_location, daily_schedule
                FROM NPCStates
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
                """,
                self.user_id, self.conversation_id
            )
            
            updated_count = 0
            current_day = await get_current_game_day(self.user_id, self.conversation_id)
            
            for npc in npcs:
                # Determine new location/activity based on schedule
                new_location = await self._determine_npc_daily_location(
                    npc['npc_id'], 
                    current_day,
                    json.loads(npc['daily_schedule'] or '{}')
                )
                
                if new_location != npc['current_location']:
                    await conn.execute(
                        """
                        UPDATE NPCStates
                        SET current_location = $1,
                            last_schedule_update = $2
                        WHERE npc_id = $3 AND user_id = $4 AND conversation_id = $5
                        """,
                        new_location, current_day,
                        npc['npc_id'], self.user_id, self.conversation_id
                    )
                    updated_count += 1
                    
        return updated_count
    
    async def _determine_npc_daily_location(self, npc_id: int, game_day: int, schedule: Dict) -> str:
        """
        Determine where an NPC should be for the given game day.
        This could be expanded with more complex logic.
        """
        # Simple rotation through scheduled locations
        locations = schedule.get('locations', ['home'])
        day_index = game_day % len(locations)
        return locations[day_index]
    
    async def _check_expiring_opportunities(self) -> int:
        """Check and process opportunities that are about to expire"""
        async with get_db_connection_context() as conn:
            current_day = await get_current_game_day(self.user_id, self.conversation_id)
            
            # Find expiring opportunities
            expiring = await conn.fetch(
                """
                SELECT id, conflict_id, expires_on
                FROM ConflictOpportunities
                WHERE user_id = $1 AND conversation_id = $2
                AND status = 'available'
                AND expires_on <= $3
                """,
                self.user_id, self.conversation_id, current_day + 1
            )
            
            for opp in expiring:
                # Mark as expired
                await conn.execute(
                    """
                    UPDATE ConflictOpportunities
                    SET status = 'expired'
                    WHERE id = $1
                    """,
                    opp['id']
                )
                
            return len(expiring)
    
    async def _generate_single_news_item(self, conflict_id: int) -> Dict[str, Any]:
        """Generate a single news item for a conflict (called from background)"""
        from logic.conflict_system.background_grand_conflicts import (
            BackgroundNewsGenerator, 
            BackgroundConflict,
            GrandConflictType,
            BackgroundIntensity
        )
        
        generator = BackgroundNewsGenerator(self.user_id, self.conversation_id)
        
        async with get_db_connection_context() as conn:
            conflict_data = await conn.fetchrow(
                """
                SELECT * FROM BackgroundConflicts
                WHERE id = $1
                """,
                conflict_id
            )
            
        if conflict_data:
            # Convert to conflict object
            conflict = BackgroundConflict(
                conflict_id=conflict_data['id'],
                conflict_type=GrandConflictType[conflict_data['conflict_type'].upper()],
                name=conflict_data['name'],
                description=conflict_data['description'],
                intensity=BackgroundIntensity[conflict_data['intensity'].upper()],
                progress=conflict_data['progress'],
                factions=json.loads(conflict_data['factions']),
                current_state=conflict_data['current_state'],
                recent_developments=json.loads(conflict_data.get('recent_developments', '[]')),
                impact_on_daily_life=json.loads(conflict_data.get('impacts', '[]')),
                player_awareness_level=conflict_data.get('awareness', 0.1),
                news_count=conflict_data.get('news_count', 0)
            )
            
            # Generate news
            news = await generator.generate_news_item(conflict)
            return {"generated": True, "conflict_id": conflict_id, "news": news}
            
        return {"generated": False, "conflict_id": conflict_id}
    
    async def _get_ambient_effect(self, conflict_id: int) -> Optional[str]:
        """Get ambient effect description for a conflict"""
        async with get_db_connection_context() as conn:
            effect = await conn.fetchval(
                """
                SELECT ambient_effect FROM BackgroundConflicts
                WHERE id = $1
                """,
                conflict_id
            )
        return effect
    
    def _is_location_affected(self, location: Any, conflict: Dict) -> bool:
        """Check if a location is affected by a conflict (robust to numeric/string)."""
        if not conflict.get('locations'):
            return False
    
        try:
            affected_locations = json.loads(conflict['locations'] or '[]')
        except Exception:
            affected_locations = []
    
        # Normalize both sides to lowercase strings for comparison
        loc_str = str(location or '').lower()
        affected_norm = [str(loc).lower() for loc in (affected_locations or [])]
    
        return bool(loc_str) and (loc_str in affected_norm)
    
    def _is_npc_involved(self, npc_id: int, conflict: Dict) -> bool:
        """Check if an NPC is involved in a conflict (figures is a JSON array)."""
        if not conflict.get('figures'):
            return False
    
        try:
            key_figures = json.loads(conflict['figures'] or '[]')
        except Exception:
            key_figures = []
    
        return npc_id in key_figures
    
    async def _update_npc_conflict_knowledge(self, npc_id: int, conflict_id: int) -> Dict[str, Any]:
        """Update what an NPC knows about a conflict"""
        async with get_db_connection_context() as conn:
            # Add or update NPC's knowledge of the conflict
            await conn.execute(
                """
                INSERT INTO NPCKnowledge (npc_id, conflict_id, knowledge_level, last_updated)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (npc_id, conflict_id) 
                DO UPDATE SET 
                    knowledge_level = EXCLUDED.knowledge_level,
                    last_updated = CURRENT_TIMESTAMP
                """,
                npc_id, conflict_id, 'aware'
            )
            
        return {"npc_id": npc_id, "conflict_id": conflict_id, "updated": True}


# ========== Background Task Scheduler ==========

class ConflictBackgroundScheduler:
    """
    Schedules and manages background processing tasks.
    Integrates with your existing task queue (Celery, etc.)
    """
    
    def __init__(self):
        self.processors: Dict[str, BackgroundConflictProcessor] = {}
        
    def get_processor(self, user_id: int, conversation_id: int) -> BackgroundConflictProcessor:
        """Get or create a processor for a conversation"""
        key = f"{user_id}_{conversation_id}"
        if key not in self.processors:
            self.processors[key] = BackgroundConflictProcessor(user_id, conversation_id)
        return self.processors[key]
    
    async def on_game_day_change(self, user_id: int, conversation_id: int, new_day: int):
        """
        Called when the game day changes.
        Triggers all daily background processes.
        """
        processor = self.get_processor(user_id, conversation_id)
        
        # Process daily updates in background
        result = await processor.process_daily_updates()
        
        # Process any high-priority queued items
        if processor._processing_queue:
            await processor.process_queued_items(max_items=5)
            
        logger.info(f"Daily conflict update for day {new_day}: {result}")
        return result
    
    async def on_scene_enter(self, user_id: int, conversation_id: int, scene_context: Dict[str, Any]):
        """
        Called when entering a new scene.
        Only processes immediately relevant updates.
        """
        processor = self.get_processor(user_id, conversation_id)
        return await processor.process_scene_relevant_updates(scene_context)
    
    async def process_background_queue(self, user_id: int, conversation_id: int):
        """
        Process background queue items.
        Called by worker processes periodically.
        """
        processor = self.get_processor(user_id, conversation_id)
        return await processor.process_queued_items()


# Global scheduler instance
_scheduler: Optional[ConflictBackgroundScheduler] = None

def get_conflict_scheduler() -> ConflictBackgroundScheduler:
    """Get the global conflict background scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = ConflictBackgroundScheduler()
    return _scheduler
