# memory/init.py

import asyncio
import logging
import argparse
import os
import json
from typing import Dict, Any, List, Optional

from .config import load_config
from .connection import DBConnectionManager, TransactionContext
from .core import UnifiedMemoryManager
from .schemas import create_schema_tables
from .emotional import create_emotional_tables
from .flashbacks import create_flashback_tables
from .masks import create_mask_tables
from .reconsolidation import create_reconsolidation_tables
from .semantic import create_semantic_tables
from .integrated import init_memory_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_init")

async def initialize_database() -> bool:
    """Initialize all database tables for the memory system."""
    logger.info("Initializing memory system database...")
    try:
        # Create tables in proper order
        await UnifiedMemoryManager.create_tables()
        await create_schema_tables()
        await create_emotional_tables()
        await create_flashback_tables()
        await create_mask_tables()
        await create_reconsolidation_tables()
        await create_semantic_tables()
        
        logger.info("All memory system tables initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

async def setup_npc(user_id: int, conversation_id: int, npc_data: Dict[str, Any]) -> Optional[int]:
    """Set up an NPC with initial data."""
    try:
        from db.connection import get_db_connection_context
        
        async with get_db_connection_context() as conn:
            # Create NPC entry
            npc_id = npc_data.get("npc_id")
            if npc_id is None:
                # Generate new NPC ID
                npc_id = await conn.fetchval("""
                    SELECT COALESCE(MAX(npc_id), 0) + 1
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, user_id, conversation_id)
            
            # Insert or update NPC stats
            await conn.execute("""
                INSERT INTO NPCStats (
                    user_id, conversation_id, npc_id, npc_name,
                    dominance, cruelty, closeness, trust, respect, intensity,
                    personality_traits, archetype_summary, introduced
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (user_id, conversation_id, npc_id)
                DO UPDATE SET
                    npc_name = EXCLUDED.npc_name,
                    dominance = EXCLUDED.dominance,
                    cruelty = EXCLUDED.cruelty,
                    closeness = EXCLUDED.closeness,
                    trust = EXCLUDED.trust,
                    respect = EXCLUDED.respect,
                    intensity = EXCLUDED.intensity,
                    personality_traits = EXCLUDED.personality_traits,
                    archetype_summary = EXCLUDED.archetype_summary,
                    introduced = EXCLUDED.introduced
            """,
                user_id, conversation_id, npc_id, npc_data.get("npc_name", "Unnamed NPC"),
                npc_data.get("dominance", 50), npc_data.get("cruelty", 50),
                npc_data.get("closeness", 30), npc_data.get("trust", 30),
                npc_data.get("respect", 30), npc_data.get("intensity", 50),
                json.dumps(npc_data.get("personality_traits", [])),
                npc_data.get("archetype_summary", ""),
                npc_data.get("introduced", False)
            )
            
            # Initialize mask
            from .masks import ProgressiveRevealManager
            mask_manager = ProgressiveRevealManager(user_id, conversation_id)
            await mask_manager.initialize_npc_mask(npc_id)
            
            # Add initial memories if provided
            if "initial_memories" in npc_data and npc_data["initial_memories"]:
                from .wrapper import MemorySystem
                memory_system = await MemorySystem.get_instance(user_id, conversation_id)
                
                for memory_text in npc_data["initial_memories"]:
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=memory_text,
                        importance="medium"
                    )
            
            logger.info(f"NPC {npc_data.get('npc_name')} (ID: {npc_id}) set up successfully")
            return npc_id
    except Exception as e:
        logger.error(f"Error setting up NPC: {e}")
        return None

async def setup_player(user_id: int, conversation_id: int, player_data: Dict[str, Any]) -> bool:
    """Set up a player with initial data."""
    try:
        from db.connection import get_db_connection_context
        
        async with get_db_connection_context() as conn:
            player_name = player_data.get("player_name", "Chase")
            
            # Insert or update player stats
            await conn.execute("""
                INSERT INTO PlayerStats (
                    user_id, conversation_id, player_name,
                    corruption, confidence, willpower, obedience,
                    dependency, lust, mental_resilience, physical_endurance
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (user_id, conversation_id, player_name)
                DO UPDATE SET
                    corruption = EXCLUDED.corruption,
                    confidence = EXCLUDED.confidence,
                    willpower = EXCLUDED.willpower,
                    obedience = EXCLUDED.obedience,
                    dependency = EXCLUDED.dependency,
                    lust = EXCLUDED.lust,
                    mental_resilience = EXCLUDED.mental_resilience,
                    physical_endurance = EXCLUDED.physical_endurance
            """,
                user_id, conversation_id, player_name,
                player_data.get("corruption", 0),
                player_data.get("confidence", 50),
                player_data.get("willpower", 50),
                player_data.get("obedience", 30),
                player_data.get("dependency", 0),
                player_data.get("lust", 30),
                player_data.get("mental_resilience", 50),
                player_data.get("physical_endurance", 50)
            )
            
            # Add initial memories if provided
            if "initial_memories" in player_data and player_data["initial_memories"]:
                from .wrapper import MemorySystem
                memory_system = await MemorySystem.get_instance(user_id, conversation_id)
                
                for memory_text in player_data["initial_memories"]:
                    await memory_system.remember(
                        entity_type="player",
                        entity_id=user_id,  # For player, entity_id is user_id
                        memory_text=memory_text,
                        importance="medium"
                    )
            
            logger.info(f"Player {player_name} set up successfully")
            return True
    except Exception as e:
        logger.error(f"Error setting up player: {e}")
        return False

async def setup_nyx(user_id: int, conversation_id: int, nyx_data: Dict[str, Any]) -> bool:
    """Set up the Nyx DM with initial data."""
    try:
        from db.connection import get_db_connection_context
        
        async with get_db_connection_context() as conn:
            # Set up NyxAgentState
            await conn.execute("""
                INSERT INTO NyxAgentState (
                    user_id, conversation_id, current_goals, predicted_futures,
                    narrative_assessment, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, conversation_id)
                DO UPDATE SET
                    current_goals = EXCLUDED.current_goals,
                    predicted_futures = EXCLUDED.predicted_futures,
                    narrative_assessment = EXCLUDED.narrative_assessment,
                    updated_at = CURRENT_TIMESTAMP
            """,
                user_id, conversation_id,
                json.dumps(nyx_data.get("current_goals", {"goals": []})),
                json.dumps(nyx_data.get("predicted_futures", [])),
                json.dumps(nyx_data.get("narrative_assessment", {}))
            )
            
            # Set initial narrative arcs if provided
            if "narrative_arcs" in nyx_data:
                await conn.execute("""
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'NyxNarrativeArcs', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                """, user_id, conversation_id, json.dumps(nyx_data["narrative_arcs"]))
            
            # Add initial reflections if provided
            if "initial_reflections" in nyx_data and nyx_data["initial_reflections"]:
                from .wrapper import MemorySystem
                memory_system = await MemorySystem.get_instance(user_id, conversation_id)
                
                for reflection in nyx_data["initial_reflections"]:
                    await memory_system.add_narrative_reflection(
                        reflection=reflection,
                        reflection_type="setup",
                        importance="high"
                    )
            
            logger.info(f"Nyx DM set up successfully")
            return True
    except Exception as e:
        logger.error(f"Error setting up Nyx DM: {e}")
        return False

async def load_and_init_from_file(config_file: str, setup_file: str = None) -> bool:
    """
    Load configuration and initialize system from files.
    
    Args:
        config_file: Path to configuration file
        setup_file: Path to setup data file (optional)
        
    Returns:
        Success status
    """
    # Load configuration
    try:
        config = load_config(config_file)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return False
    
    # Initialize database
    db_initialized = await initialize_database()
    if not db_initialized:
        return False
    
    # If setup file provided, initialize entities
    if setup_file and os.path.exists(setup_file):
        try:
            with open(setup_file, 'r') as f:
                setup_data = json.load(f)
                
            user_id = setup_data.get("user_id", 1)
            conversation_id = setup_data.get("conversation_id", 1)
            
            # Initialize memory system
            memory_system = await init_memory_system(user_id, conversation_id)
            
            # Set up NPCs
            if "npcs" in setup_data:
                for npc_data in setup_data["npcs"]:
                    await setup_npc(user_id, conversation_id, npc_data)
            
            # Set up player
            if "player" in setup_data:
                await setup_player(user_id, conversation_id, setup_data["player"])
            
            # Set up Nyx
            if "nyx" in setup_data:
                await setup_nyx(user_id, conversation_id, setup_data["nyx"])
                
            logger.info("Entity setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up entities: {e}")
            return False
    
    return True

async def main():
    parser = argparse.ArgumentParser(description="Initialize the memory system")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--setup", help="Path to setup data file")
    args = parser.parse_args()
    
    success = await load_and_init_from_file(args.config, args.setup)
    
    if success:
        logger.info("Memory system initialization completed successfully")
    else:
        logger.error("Memory system initialization failed")
        exit(1)
    
    # Clean up connections
    await DBConnectionManager.close_pool()

if __name__ == "__main__":
    asyncio.run(main())
