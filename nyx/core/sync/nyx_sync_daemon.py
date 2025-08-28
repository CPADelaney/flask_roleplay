# nyx/core/sync/nyx_sync_daemon.py

import asyncio
import asyncpg
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Fixed imports - using the new nyx_agent module
from nyx.nyx_agent import (
    add_memory,
    AddMemoryInput,
)
from agents import RunContextWrapper
from nyx.user_model_sdk import UserModelManager

logger = logging.getLogger(__name__)

DB_DSN = "postgresql://nyx_user:your_password@localhost/nyx_db"

class NyxSyncDaemon:
    def __init__(self, db_dsn=DB_DSN):
        self.db_dsn = db_dsn
        self.interval_seconds = 60  # configurable polling time
        self._last_sync_time: Optional[datetime] = None
        self._sync_status: Dict[str, Any] = {
            "out_of_sync": False,
            "last_sync": None,
            "pending_strategies": 0,
            "pending_scenes": 0,
            "sync_errors": []
        }
        self._systems_synced = 0

    async def start(self):
        while True:
            try:
                await self.run_sync_cycle()
            except Exception as e:
                logger.error(f"Sync cycle failed: {e}")
                self._sync_status["sync_errors"].append({
                    "error": str(e),
                    "time": datetime.now().isoformat()
                })
                # Keep only last 10 errors
                self._sync_status["sync_errors"] = self._sync_status["sync_errors"][-10:]
            await asyncio.sleep(self.interval_seconds)

    async def run_sync_cycle(self):
        logger.info("Running Nyx Sync Daemon cycle...")
        conn = await asyncpg.connect(dsn=self.db_dsn)

        try:
            # Reset sync counters
            self._systems_synced = 0
            
            # --- Strategy Injection ---
            active_strategies = await conn.fetch("""
                SELECT * FROM nyx1_strategy_injections
                WHERE status = 'active'
                AND (expires_at IS NULL OR expires_at > NOW())
            """)
            
            self._sync_status["pending_strategies"] = len(active_strategies)

            for strategy in active_strategies:
                await self.inject_strategy(strategy, conn)
                self._systems_synced += 1

            # --- Scene Templates ---
            active_scenes = await conn.fetch("""
                SELECT * FROM nyx1_scene_templates
                WHERE active = TRUE
            """)
            
            self._sync_status["pending_scenes"] = len(active_scenes)

            for scene in active_scenes:
                await self.inject_scene(scene, conn)
                self._systems_synced += 1

            # Update sync status
            self._last_sync_time = datetime.now()
            self._sync_status["last_sync"] = self._last_sync_time.isoformat()
            self._sync_status["out_of_sync"] = False

        except Exception as e:
            self._sync_status["out_of_sync"] = True
            raise
        finally:
            await conn.close()

    async def get_sync_status(self) -> Dict[str, Any]:
        """Return current sync status for the workspace adapter"""
        # Check if we're out of sync (no sync in last 5 minutes)
        if self._last_sync_time:
            time_since_sync = datetime.now() - self._last_sync_time
            if time_since_sync.total_seconds() > 300:  # 5 minutes
                self._sync_status["out_of_sync"] = True
        else:
            self._sync_status["out_of_sync"] = True
        
        return self._sync_status.copy()

    async def background_sync(self) -> Dict[str, Any]:
        """Perform a background sync operation"""
        try:
            await self.run_sync_cycle()
            return {
                "synced_count": self._systems_synced,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Background sync failed: {e}")
            return {
                "synced_count": 0,
                "success": False,
                "error": str(e)
            }

    async def inject_strategy(self, strategy_row, conn):
        """
        Persist a strategy update as an abstraction‑level memory
        for every active user, and log the event.
        """
        strategy_id   = strategy_row["id"]
        strategy_type = strategy_row["strategy_type"]          # e.g. "rlhf‑policy"
        payload       = strategy_row["payload"] or {}
        strategy_name = strategy_row["strategy_name"]
    
        # pull every user that has a model‑state
        user_ids = await conn.fetch(
            "SELECT DISTINCT user_id FROM user_model_states"
        )
    
        for user in user_ids:
            user_id = user["user_id"]
            logger.info("Injecting strategy '%s' to user %s", strategy_name, user_id)
    
            # -------- build memory text --------
            memory_text = (
                f"Nyx's strategic layer updated: "
                f"{strategy_type.upper()} - {strategy_name}\n"
                f"{payload.get('description', '')}"
            )
    
            # -------- Get context and wrap it properly --------
            ctx = await self._get_context(user_id)
            wrapped_ctx = RunContextWrapper(context=ctx)
            
            # -------- Create memory input using new API --------
            memory_input = {
                'memory_text': memory_text,
                'memory_type': 'abstraction',
                'significance': 7,
                'entities': [strategy_id]  # Link to strategy entity
            }
            
            # -------- Store memory using new API --------
            await add_memory(wrapped_ctx, memory_input)
    
            # -------- log in DB --------
            await conn.execute(
                """
                INSERT INTO nyx1_strategy_logs
                       (strategy_id, user_id, event_type,
                        message_snippet, kink_profile)
                VALUES ($1, $2, 'triggered', $3, $4)
                """,
                strategy_id,
                user_id,
                memory_text[:250],
                await self._get_kink_profile(user_id),
            )

    async def inject_scene(self, scene_row, conn):
        """Inject scene template as memory for all active users"""
        prompt_template = scene_row['prompt_template']
        scene_title = scene_row['title'] or "Untitled Scene"
        intensity = scene_row['intensity_level'] or 5

        logger.info(f"Injecting scene template: {scene_title} (Intensity {intensity})")

        user_ids = await conn.fetch("SELECT DISTINCT user_id FROM user_model_states")

        for user in user_ids:
            user_id = user['user_id']
            
            # Get context and wrap it
            ctx = await self._get_context(user_id)
            wrapped_ctx = RunContextWrapper(context=ctx)
            
            # Create memory input
            memory_input = {
                'memory_text': f"New scene template loaded: {scene_title}\n{prompt_template}",
                'memory_type': 'template',
                'significance': 6,
                'entities': []
            }
            
            # Store memory
            await add_memory(wrapped_ctx, memory_input)

    async def _get_context(self, user_id):
        """Get or create NyxContext for a user"""
        # Since we need a conversation_id, we'll use a default one for system operations
        # In production, you might want to track this differently
        from nyx.nyx_agent.context import NyxContext
        
        conversation_id = 1  # System conversation ID
        ctx = NyxContext(user_id=user_id, conversation_id=conversation_id)
        await ctx.initialize()
        return ctx

    async def _get_kink_profile(self, user_id):
        """Get user's kink profile from UserModelManager"""
        try:
            model = await UserModelManager.get_instance(user_id, conversation_id=1)
            profile = await model.get_kink_profile()
            return profile
        except Exception as e:
            logger.warning(f"Could not get kink profile for user {user_id}: {e}")
            return {}

# Additional helper functions for noise classification (if needed)
async def run_noise_classification(conn):
    """Run noise classification on recent responses"""
    rows = await conn.fetch("""
        SELECT id, nyx_response FROM nyx1_response_noise
        WHERE marked_for_review = FALSE AND dismissed = FALSE
        ORDER BY created_at DESC LIMIT 50
    """)
    
    for row in rows:
        score = classify_noise(row['nyx_response'])

        if score > 0.8:
            await conn.execute("""
                UPDATE nyx1_response_noise
                SET marked_for_review = TRUE, score = $2
                WHERE id = $1
            """, row['id'], score)

def classify_noise(text: str) -> float:
    """Simple noise classification based on keywords"""
    keywords = ["uh", "maybe", "sorry", "idk", "unsure", "could", "perhaps"]
    matches = sum(1 for k in keywords if k in text.lower())
    return min(1.0, matches / 3.0)
