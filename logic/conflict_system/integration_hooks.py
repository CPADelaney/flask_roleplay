# logic/conflict_system/integration_hooks.py
"""
Integration hooks for the refactored conflict system.
Connects background processing to game events and Celery tasks.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from celery import Celery, Task
from celery.schedules import crontab

from db.connection import get_db_connection_context
from logic.time_cycle import get_current_game_day
from logic.conflict_system.background_processor import (
    get_conflict_scheduler,
    ProcessingPriority
)

logger = logging.getLogger(__name__)

# Assume you have a Celery app configured elsewhere
# from your_app import celery_app
# For this example, I'll create a placeholder
celery_app = Celery('conflict_tasks')

# ===============================================================================
# GAME EVENT HOOKS
# ===============================================================================

class ConflictEventHooks:
    """
    Hooks that integrate with your game's event system.
    Call these from your existing event handlers.
    """
    
    @staticmethod
    async def on_game_day_transition(user_id: int, conversation_id: int, new_day: int):
        """
        Called when the game day changes.
        Primary trigger for background updates.
        """
        logger.info(f"Game day transition to {new_day} for user {user_id}")
    
        scheduler = get_conflict_scheduler()
        try:
            # Run background model updates (do NOT call synthesizer.handle_day_transition to avoid recursion)
            result = await scheduler.on_game_day_change(user_id, conversation_id, new_day)
    
            # Notify orchestrator so subsystems can react
            try:
                from logic.conflict_system.conflict_synthesizer import get_synthesizer, SystemEvent, EventType, SubsystemType
                synthesizer = await get_synthesizer(user_id, conversation_id)
    
                await synthesizer.emit_event(SystemEvent(
                    event_id=f"day_{new_day}_{datetime.utcnow().timestamp()}",
                    event_type=EventType.DAY_TRANSITION,
                    source_subsystem=SubsystemType.ORCHESTRATOR,
                    payload={'new_day': new_day, 'processing_result': result},
                    requires_response=False,
                    priority=7,
                ))
    
                # Best-effort: process a few queued items (news, etc.) so side-effects are enqueued
                await synthesizer.process_background_queue(max_items=5)
            except Exception as e:
                logger.debug(f"Orchestrator day-transition notification failed (non-fatal): {e}")
    
            # Kick Celery if processor queue is still non-empty
            processor = scheduler.get_processor(user_id, conversation_id)
            if processor._processing_queue:
                process_conflict_queue.delay(user_id, conversation_id)
    
            return result
        except Exception as e:
            logger.error(f"Error in game day transition: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def on_scene_transition(
        user_id: int, 
        conversation_id: int,
        old_scene: Optional[Dict[str, Any]],
        new_scene: Dict[str, Any]
    ):
        """
        Called when transitioning between scenes.
        Asks the orchestrator for scene-relevant context (fast path) and lets it emit events.
        """
        try:
            from logic.conflict_system.conflict_synthesizer import get_synthesizer
            synth = await get_synthesizer(user_id, conversation_id)
    
            # Get scene-relevant context (this also enqueues background ambient events via orchestrator)
            context = await synth.conflict_context_for_scene(new_scene)
    
            # Best-effort: process a few high-priority background items now
            try:
                await synth.process_background_queue(max_items=5)
            except Exception:
                pass
    
            return context
        except Exception as e:
            logger.error(f"Error in scene transition: {e}")
            return {}
        
    @staticmethod
    async def on_significant_conflict_event(
        user_id: int,
        conversation_id: int,
        conflict_id: int,
        event_type: str,
        magnitude: float = 0.5
    ):
        """
        Called when something significant happens in a conflict.
        Uses orchestrator limits and nudges background processing.
        """
        try:
            from logic.conflict_system.conflict_synthesizer import get_synthesizer
            synth = await get_synthesizer(user_id, conversation_id)
    
            # Respect orchestrator limiters (e.g., news rate limiting)
            allowed, reason = await synth.should_generate_content('news', conflict_id)
            if not allowed:
                return {'triggered': False, 'reason': reason}
    
            # Use the background processor (owned by orchestrator) to enqueue the event
            try:
                result = await synth.processor.trigger_significant_event(
                    conflict_id,
                    {
                        'event_type': event_type,
                        'magnitude': magnitude,
                        'description': f"Significant {event_type} event",
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
            except Exception as e:
                logger.debug(f"Processor trigger failed (non-fatal): {e}")
                result = False
    
            # Nudge the queue to surface items and emit news state syncs if any
            try:
                await synth.process_background_queue(max_items=3)
            except Exception:
                pass
    
            return bool(result)
        except Exception as e:
            logger.error(f"Error triggering significant event: {e}")
            return False

# ===============================================================================
# CELERY BACKGROUND TASKS
# ===============================================================================

@celery_app.task(name='conflict.process_queue')
def process_conflict_queue(user_id: int, conversation_id: int):
    """
    Celery task to process background conflict queue via orchestrator.
    Ensures any generated items emit proper events.
    """
    async def _process():
        try:
            from logic.conflict_system.conflict_synthesizer import get_synthesizer
            synth = await get_synthesizer(user_id, conversation_id)
            results = await synth.process_background_queue(max_items=10)
            # Re-enqueue if there's more work (best-effort check)
            try:
                if synth.processor and synth.processor._processing_queue:
                    process_conflict_queue.apply_async(
                        args=[user_id, conversation_id],
                        countdown=60
                    )
            except Exception:
                pass
            return results
        except Exception as e:
            logger.error(f"Celery process_queue error: {e}")
            return []
    return asyncio.run(_process())

@celery_app.task(name='conflict.daily_maintenance')
def daily_conflict_maintenance():
    """
    Daily maintenance task for all active conflicts.
    Runs once per day across all conversations.
    """
    async def _maintain():
        async with get_db_connection_context() as conn:
            conversations = await conn.fetch(
                """
                SELECT DISTINCT user_id, conversation_id 
                FROM BackgroundConflicts
                WHERE status = 'active'
                  AND updated_at > NOW() - INTERVAL '7 days'
                """
            )
    
        processed = 0
        from logic.conflict_system.conflict_synthesizer import get_synthesizer
        for conv in conversations:
            try:
                user_id = int(conv['user_id'])
                conversation_id = int(conv['conversation_id'])
                game_day = await get_current_game_day(user_id, conversation_id)
                synth = await get_synthesizer(user_id, conversation_id)
                await synth.handle_day_transition(game_day)
                processed += 1
            except Exception as e:
                logger.error(f"Error in daily maintenance for {conv}: {e}")
    
        return {"processed": processed}
    
    return asyncio.run(_maintain())

@celery_app.task(name='conflict.cleanup_old_data')
def cleanup_old_conflict_data():
    """
    Clean up old conflict data that's no longer needed.
    Runs weekly.
    """
    async def _cleanup():
        async with get_db_connection_context() as conn:
            # Archive conflicts older than 30 game days
            archived = await conn.execute(
                """
                UPDATE BackgroundConflicts
                SET status = 'archived'
                WHERE status = 'resolved'
                AND updated_at < NOW() - INTERVAL '30 days'
                """
            )
            
            # Delete very old news items (keep max 10 per conflict)
            deleted = await conn.execute(
                """
                DELETE FROM BackgroundNews
                WHERE id IN (
                    SELECT id FROM (
                        SELECT id,
                               ROW_NUMBER() OVER (PARTITION BY conflict_id ORDER BY game_day DESC) as rn
                        FROM BackgroundNews
                    ) t
                    WHERE rn > 10
                )
                """
            )
            
            return {"archived": archived, "deleted_news": deleted}
    
    return asyncio.run(_cleanup())

# ===============================================================================
# CELERY BEAT SCHEDULE
# ===============================================================================

celery_app.conf.beat_schedule = {
    'daily-conflict-maintenance': {
        'task': 'conflict.daily_maintenance',
        'schedule': crontab(hour=3, minute=0),  # Run at 3 AM daily
    },
    'weekly-conflict-cleanup': {
        'task': 'conflict.cleanup_old_data', 
        'schedule': crontab(day_of_week=0, hour=4, minute=0),  # Sunday 4 AM
    }
}

# ===============================================================================
# TIME CYCLE INTEGRATION
# ===============================================================================

async def integrate_with_time_cycle(user_id: int, conversation_id: int):
    """
    Integrate conflict system with your existing time cycle.
    Call this when time advances in the game.
    """
    from logic.time_cycle import TimeManager, TimeState
    
    # This would hook into your existing time system
    # Example integration:
    
    async def on_time_advance(old_state: TimeState, new_state: TimeState):
        """Called when game time advances"""
        
        # Check if day changed
        if old_state.day != new_state.day:
            hooks = ConflictEventHooks()
            await hooks.on_game_day_transition(
                user_id,
                conversation_id, 
                new_state.day
            )
            
        # Check if week changed (for weekly events)
        if old_state.week != new_state.week:
            # Trigger weekly conflict events
            pass
            
    return on_time_advance

# ===============================================================================
# NPC SCHEDULE INTEGRATION
# ===============================================================================

class NPCScheduleManager:
    """
    Manages NPC schedules based on game day.
    Determines where NPCs can be found.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._schedule_cache = {}
    
    async def get_npc_location(self, npc_id: int, game_day: int) -> Dict[str, Any]:
        """
        Get where an NPC should be on a given game day.
        Uses pre-calculated schedules from background processing.
        """
        cache_key = f"{npc_id}_{game_day}"
        
        if cache_key in self._schedule_cache:
            return self._schedule_cache[cache_key]
        
        async with get_db_connection_context() as conn:
            # Get NPC's schedule for this day
            schedule = await conn.fetchrow(
                """
                SELECT current_location, current_activity, mood, available
                FROM NPCDailySchedules
                WHERE npc_id = $1 AND game_day = $2
                AND user_id = $3 AND conversation_id = $4
                """,
                npc_id, game_day, self.user_id, self.conversation_id
            )
            
            if not schedule:
                # Fallback to default location
                default = await conn.fetchrow(
                    """
                    SELECT default_location, default_activity
                    FROM NPCs
                    WHERE id = $1
                    """,
                    npc_id
                )
                
                schedule = {
                    'current_location': default['default_location'] if default else 'unknown',
                    'current_activity': default['default_activity'] if default else 'idle',
                    'mood': 'neutral',
                    'available': True
                }
            
            result = dict(schedule)
            self._schedule_cache[cache_key] = result
            return result
    
    async def update_npc_schedules_for_day(self, game_day: int):
        """
        Pre-calculate NPC schedules for a game day.
        Called by background processor.
        """
        async with get_db_connection_context() as conn:
            # Get all active NPCs
            npcs = await conn.fetch(
                """
                SELECT id, schedule_pattern, personality_traits
                FROM NPCs
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
                """,
                self.user_id, self.conversation_id
            )
            
            for npc in npcs:
                # Calculate location based on pattern and day
                location = self._calculate_location(
                    npc['schedule_pattern'],
                    game_day
                )
                
                activity = self._calculate_activity(
                    npc['personality_traits'],
                    location,
                    game_day
                )
                
                # Store pre-calculated schedule
                await conn.execute(
                    """
                    INSERT INTO NPCDailySchedules
                    (npc_id, game_day, current_location, current_activity, 
                     mood, available, user_id, conversation_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (npc_id, game_day, user_id, conversation_id)
                    DO UPDATE SET
                        current_location = EXCLUDED.current_location,
                        current_activity = EXCLUDED.current_activity,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    npc['id'], game_day, location, activity,
                    'neutral', True, self.user_id, self.conversation_id
                )
                
        # Clear cache for this day
        self._schedule_cache = {
            k: v for k, v in self._schedule_cache.items()
            if not k.endswith(f"_{game_day}")
        }
    
    def _calculate_location(self, pattern: Optional[str], day: int) -> str:
        """Calculate NPC location based on schedule pattern"""
        if not pattern:
            return 'home'
            
        patterns = {
            'merchant': ['market', 'shop', 'warehouse', 'home'],
            'guard': ['gate', 'barracks', 'patrol', 'tavern'],
            'noble': ['court', 'estate', 'gardens', 'salon'],
            'worker': ['workshop', 'market', 'tavern', 'home']
        }
        
        locations = patterns.get(pattern, ['home'])
        return locations[day % len(locations)]
    
    def _calculate_activity(self, traits: Optional[str], location: str, day: int) -> str:
        """Calculate what NPC is doing based on traits and location"""
        activities = {
            'market': ['trading', 'shopping', 'gossiping'],
            'tavern': ['drinking', 'talking', 'gambling'],
            'home': ['resting', 'working', 'entertaining'],
            'court': ['politicking', 'scheming', 'attending']
        }
        
        location_activities = activities.get(location, ['idle'])
        return location_activities[day % len(location_activities)]

# ===============================================================================
# MIGRATION HELPERS
# ===============================================================================

async def migrate_existing_conflicts(user_id: int, conversation_id: int):
    """
    Helper to migrate existing conflicts to new system.
    Adds tracking fields for news generation limits.
    """
    async with get_db_connection_context() as conn:
        # Add new tracking columns if they don't exist
        await conn.execute(
            """
            ALTER TABLE BackgroundConflicts
            ADD COLUMN IF NOT EXISTS last_news_generation INTEGER,
            ADD COLUMN IF NOT EXISTS news_count INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS last_significant_change JSONB
            """
        )
        
        # Update news counts for existing conflicts
        await conn.execute(
            """
            UPDATE BackgroundConflicts bc
            SET news_count = (
                SELECT COUNT(*) FROM BackgroundNews bn
                WHERE bn.conflict_id = bc.id
            ),
            last_news_generation = (
                SELECT MAX(game_day) FROM BackgroundNews bn  
                WHERE bn.conflict_id = bc.id
            )
            WHERE bc.user_id = $1 AND bc.conversation_id = $2
            """,
            user_id, conversation_id
        )
        
        # Create new tables if needed
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS NPCDailySchedules (
                npc_id INTEGER,
                game_day INTEGER,
                user_id INTEGER,
                conversation_id INTEGER,
                current_location TEXT,
                current_activity TEXT,
                mood TEXT,
                available BOOLEAN DEFAULT true,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (npc_id, game_day, user_id, conversation_id)
            )
            """
        )
        
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS BackgroundDevelopments (
                id SERIAL PRIMARY KEY,
                conflict_id INTEGER REFERENCES BackgroundConflicts(id),
                development TEXT,
                game_day INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ConflictOpportunities (
                id SERIAL PRIMARY KEY,
                conflict_id INTEGER,
                user_id INTEGER,
                conversation_id INTEGER,
                description TEXT,
                expires_on INTEGER,
                status TEXT DEFAULT 'available',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ConflictRipples (
                id SERIAL PRIMARY KEY,
                conflict_id INTEGER,
                user_id INTEGER,
                conversation_id INTEGER,
                ripple_data JSONB,
                game_day INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS NPCKnowledge (
                npc_id INTEGER,
                conflict_id INTEGER,
                knowledge_level TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (npc_id, conflict_id)
            )
            """
        )
        
    logger.info(f"Migration complete for user {user_id}, conversation {conversation_id}")
