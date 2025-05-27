# nyx/core/sync/nyx_sync_daemon.py

import asyncio
import asyncpg
import logging
from datetime import datetime, timezone

from nyx.nyx_agent_sdk import add_memory
from nyx.user_model_sdk import UserModelManager
from memory.memory_nyx_integration import MemoryNyxBridge

from nyx.core.memory_core import MemoryCreateParams, add_memory
import datetime

logger = logging.getLogger(__name__)

DB_DSN = "postgresql://nyx_user:your_password@localhost/nyx_db"

class NyxSyncDaemon:
    def __init__(self, db_dsn=DB_DSN):
        self.db_dsn = db_dsn
        self.interval_seconds = 60  # configurable polling time

    async def start(self):
        while True:
            try:
                await self.run_sync_cycle()
            except Exception as e:
                logger.error(f"Sync cycle failed: {e}")
            await asyncio.sleep(self.interval_seconds)

    async def run_sync_cycle(self):
        logger.info("Running Nyx Sync Daemon cycle...")
        conn = await asyncpg.connect(dsn=self.db_dsn)

        try:
            # --- Strategy Injection ---
            active_strategies = await conn.fetch("""
                SELECT * FROM nyx1_strategy_injections
                WHERE status = 'active'
                AND (expires_at IS NULL OR expires_at > NOW())
            """)

            for strategy in active_strategies:
                await self.inject_strategy(strategy, conn)

            # --- Scene Templates ---
            active_scenes = await conn.fetch("""
                SELECT * FROM nyx1_scene_templates
                WHERE active = TRUE
            """)

            for scene in active_scenes:
                await self.inject_scene(scene, conn)

        finally:
            await conn.close()

    async def inject_strategy(self, strategy_row, conn):
        """
        Persist a strategy update as an abstraction‑level memory
        for every active user, and log the event.
        """
        strategy_id   = strategy_row["id"]
        strategy_type = strategy_row["strategy_type"]          # e.g. “rlhf‑policy”
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
    
            # -------- insert memory --------
            ctx = await self._get_context(user_id)   # returns RunContextWrapper
    
            params = MemoryCreateParams(
                memory_text  = memory_text,
                memory_type  = "abstraction",
                memory_scope = "game",
                memory_level = "abstraction",
                significance = 7,
                tags         = ["strategy", strategy_type],
                metadata     = {
                    "strategy_id"   : strategy_id,
                    "strategy_name" : strategy_name,
                    "timestamp"     : datetime.datetime.now().isoformat(),
                    "fidelity"      : 0.95,
                },
            )
            await add_memory(ctx, params)
    
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
        prompt_template = scene_row['prompt_template']
        scene_title = scene_row['title'] or "Untitled Scene"
        intensity = scene_row['intensity_level'] or 5

        logger.info(f"Injecting scene template: {scene_title} (Intensity {intensity})")

        user_ids = await conn.fetch("SELECT DISTINCT user_id FROM user_model_states")

        for user in user_ids:
            user_id = user['user_id']
            ctx = await self._get_context(user_id)

            await add_memory(
                ctx,
                f"New scene template loaded: {scene_title}\n{prompt_template}",
                memory_type="template",
                significance=6
            )

    async def _get_context(self, user_id):
        return await UserModelManager.get_instance(user_id, conversation_id=1)  # placeholder

    async def _get_kink_profile(self, user_id):
        model = await UserModelManager.get_instance(user_id, conversation_id=1)
        profile = await model.get_kink_profile()
        return profile
        
async def run_noise_classification(self, conn):
    rows = await conn.fetch("""
        SELECT id, nyx_response FROM nyx1_response_noise
        WHERE marked_for_review = FALSE AND dismissed = FALSE
        ORDER BY created_at DESC LIMIT 50
    """)
    
    for row in rows:
        score = self.classify_noise(row['nyx_response'])

        if score > 0.8:
            await conn.execute("""
                UPDATE nyx1_response_noise
                SET marked_for_review = TRUE, score = $2
                WHERE id = $1
            """, row['id'], score)

async def classify_noise(self, text: str) -> float:
    keywords = ["uh", "maybe", "sorry", "idk", "unsure", "could", "perhaps"]
    matches = sum(1 for k in keywords if k in text.lower())
    return min(1.0, matches / 3.0)
