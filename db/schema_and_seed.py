# db/schema_and_seed.py

import json
import logging
import re
from typing import List
# Import connection and pool management functions
from db.connection import (
    get_db_connection_context,
    initialize_connection_pool,
    close_connection_pool
)
# Assuming these are now async and use asyncpg internally
from routes.activities import insert_missing_activities
from routes.archetypes import insert_missing_archetypes
from routes.settings_routes import insert_missing_settings
from logic.seed_intensity_tiers import create_and_seed_intensity_tiers
from logic.seed_plot_triggers import create_and_seed_plot_triggers
from logic.seed_interactions import create_and_seed_interactions
from logic.stats_logic import (
    insert_stat_definitions,
    insert_or_update_game_rules,
)
import asyncpg
import asyncio
from typing import Dict, Any

from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME, MOTH_FLAME_POEMS, STORY_TONE_PROMPT

# Import system prompts for Nyx
from logic.prompts import SYSTEM_PROMPT, PRIVATE_REFLECTION_INSTRUCTIONS

# Configure logger for this module
logger = logging.getLogger(__name__)

def parse_system_prompt_to_memories(prompt: str, private_instructions: str = None) -> List[Dict[str, Any]]:
    """
    Parses the Nyx system prompt (and private instructions) into structured memory objects.
    Returns a list of dicts suitable for DB insertion.
    """
    memories = []
    section = None
    lines = prompt.splitlines()
    buffer = []
    
    def flush_buffer(buffer, section):
        "Helper to emit current buffered lines as memories."
        memories_list = []
        if buffer:
            content = "\n".join(buffer).strip()
            if content:
                memories_list.append({
                    "memory_text": content,
                    "memory_type": section or "core_identity",
                    "tags": [section] if section else [],
                })
        return memories_list

    for line in lines:
        # Detect bold/markdown headings (e.g., **Main Character:** or ## Main Character)
        heading_match = re.match(r"^(?:\*\*|#+)\s*(.+?)[\:\*]*\s*(?:\*\*|#+)?$", line.strip())
        if heading_match and len(line.strip()) < 40:
            # Flush old section
            memories.extend(flush_buffer(buffer, section))
            buffer = []
            section = heading_match.group(1).lower().replace(" ", "_")
        elif line.strip().startswith("• "):
            # Bullet point = separate memory (but keep section context)
            memories.extend(flush_buffer(buffer, section))
            buffer = [line.strip()]
            memories.extend(flush_buffer(buffer, section))
            buffer = []
        else:
            buffer.append(line)
    # Flush remainder
    memories.extend(flush_buffer(buffer, section))

    # Now, further split long memory_texts into smaller bites if desired
    more_memories = []
    for mem in memories:
        if len(mem["memory_text"]) > 600 and "\n• " in mem["memory_text"]:
            # Break up big blocks of bullets (for better recall granularity)
            for point in mem["memory_text"].split("\n• "):
                if point.strip():
                    text = "• " + point.strip() if not point.strip().startswith("•") else point.strip()
                    more_memories.append({**mem, "memory_text": text})
        else:
            more_memories.append(mem)
    memories = more_memories

    # --- Add PRIVATE_REFLECTION_INSTRUCTIONS as "private_rule" memories
    if private_instructions:
        # Split into individual rules by numbers or bullets
        priv_sections = re.split(r"\n\s*\d+\.\s*", private_instructions.strip())
        for section_text in priv_sections:
            if section_text.strip():
                # Take first line or phrase as the type, rest as content
                first_line, *rest = section_text.strip().splitlines()
                priv_type = first_line.split(":")[0].lower().replace(" ", "_")
                memories.append({
                    "memory_text": section_text.strip(),
                    "memory_type": f"private_{priv_type}",
                    "tags": ["private", priv_type]
                })
    return memories

async def seed_nyx_memories_from_prompt(prompt: str, private: str = None):
    """
    Seeds Nyx's system prompt and private reflection instructions as core memories.
    """
    logger.info("Seeding Nyx system prompt and rules as core memories...")
    chunks = parse_system_prompt_to_memories(prompt, private)
    async with get_db_connection_context() as conn:
        for i, mem in enumerate(chunks):
            exists = await conn.fetchval(
                """
                SELECT id FROM unified_memories
                WHERE entity_type = 'nyx' AND entity_id = 0 AND memory_text = $1
                """,
                mem["memory_text"]
            )
            if not exists:
                await conn.execute(
                    """
                    INSERT INTO unified_memories
                    (entity_type, entity_id, user_id, conversation_id, memory_text, memory_type, tags, significance)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    "nyx", 0, 0, 0, mem["memory_text"], mem.get("memory_type", "core_identity"), mem.get("tags", []), 10-i
                )
    logger.info(f"Seeded {len(chunks)} memories for Nyx from system prompt and rules.")

async def create_all_tables():
    """
    Asynchronously creates all database tables using asyncpg.
    Includes both legacy tables and new unified tables.
    """
    logger.info("Starting table creation process...")
    try:
        # Use a longer timeout for potentially long-running schema operations
        async with get_db_connection_context(timeout=180.0) as conn:
            logger.info("Acquired connection for table creation.")

            # Ensure vector extension is available
            logger.info("Ensuring 'vector' extension exists...")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("'vector' extension checked/created.")

            # Define all CREATE TABLE and CREATE INDEX statements
            sql_commands = [
                # ======================================
                # CORE USER AND CONVERSATION TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS folders (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    folder_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_name VARCHAR(100) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'processing',
                    created_at TIMESTAMP DEFAULT NOW(),
                    folder_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE SET NULL
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER NOT NULL,
                    sender VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    structured_content JSONB,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                
                # ======================================
                # GAME STATE AND SETTINGS
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS StateUpdates (
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    update_payload JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, conversation_id)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS CurrentRoleplay (
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (user_id, conversation_id, key),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS Settings (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    mood_tone TEXT NOT NULL,
                    enhanced_features JSONB NOT NULL,
                    stat_modifiers JSONB NOT NULL,
                    activity_examples JSONB NOT NULL
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS StatDefinitions (
                    id SERIAL PRIMARY KEY,
                    scope TEXT NOT NULL,
                    stat_name TEXT UNIQUE NOT NULL,
                    range_min INT NOT NULL,
                    range_max INT NOT NULL,
                    definition TEXT NOT NULL,
                    effects TEXT NOT NULL,
                    progression_triggers TEXT NOT NULL
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS GameRules (
                    id SERIAL PRIMARY KEY,
                    rule_name TEXT UNIQUE NOT NULL,
                    condition TEXT NOT NULL,
                    effect TEXT NOT NULL
                );
                ''',
                
                # ======================================
                # LOCATION AND WORLD TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS Locations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    location_name TEXT NOT NULL,
                    description TEXT,
                    location_type TEXT,
                    parent_location TEXT,
                    cultural_significance TEXT DEFAULT 'moderate',
                    economic_importance TEXT DEFAULT 'moderate',
                    strategic_value INTEGER DEFAULT 5,
                    population_density TEXT DEFAULT 'moderate',
                    notable_features JSONB DEFAULT '[]',  -- Changed from TEXT[]
                    hidden_aspects JSONB DEFAULT '[]',    -- Changed from TEXT[]
                    access_restrictions JSONB DEFAULT '[]', -- Changed from TEXT[]
                    local_customs JSONB DEFAULT '[]',     -- Changed from TEXT[]
                    open_hours JSONB,
                    embedding vector(384),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_locations_embedding_hnsw
                ON Locations USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS Events (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    event_name TEXT NOT NULL,
                    description TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    location TEXT NOT NULL,
                    year INT DEFAULT 1,
                    month INT DEFAULT 1,
                    day INT DEFAULT 1,
                    time_of_day TEXT DEFAULT 'Morning',
                    embedding vector(384),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_events_embedding_hnsw
                ON Events USING hnsw (embedding vector_cosine_ops);
                ''',
                
                # ======================================
                # NPC AND CHARACTER TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS NPCStats (
                    npc_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc_name TEXT NOT NULL,
                    introduced BOOLEAN DEFAULT FALSE,
                    archetypes JSONB,
                    archetype_summary TEXT,
                    archetype_extras_summary TEXT,
                    physical_description TEXT,
                    relationships JSONB,
                    dominance INT CHECK (dominance BETWEEN -100 AND 100),
                    cruelty INT CHECK (cruelty BETWEEN -100 AND 100),
                    closeness INT CHECK (closeness BETWEEN -100 AND 100),
                    trust INT CHECK (trust BETWEEN -100 AND 100),
                    respect INT CHECK (respect BETWEEN -100 AND 100),
                    intensity INT CHECK (intensity BETWEEN -100 AND 100),
                    memory JSONB,
                    monica_level INT DEFAULT 0,
                    monica_games_left INT DEFAULT 0,
                    sex TEXT DEFAULT 'female',
                    hobbies JSONB,
                    personality_traits JSONB,
                    likes JSONB,
                    dislikes JSONB,
                    affiliations JSONB,
                    schedule JSONB,
                    current_location TEXT,
                    age INT,
                    birthdate TEXT,
                    is_active BOOLEAN DEFAULT FALSE,
                    role TEXT,
                    embedding vector(384),
                    personality_patterns JSONB DEFAULT '[]',::jsonb,
                    trauma_triggers JSONB,
                    flashback_triggers JSONB,
                    revelation_plan JSONB,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                ALTER TABLE NPCStats ADD COLUMN IF NOT EXISTS role TEXT;
                ''',             
                '''
                CREATE INDEX IF NOT EXISTS idx_npcstats_embedding_hnsw
                ON NPCStats USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                ALTER TABLE NPCStats ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NPCGroups (
                    group_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    group_name TEXT NOT NULL,
                    group_data JSONB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    UNIQUE(user_id, conversation_id, group_name)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NPCEvolution (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc_id INTEGER NOT NULL,
                    mask_slippage_events JSONB,
                    evolution_events JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (npc_id) REFERENCES NPCStats(npc_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NPCRevelations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc_id INTEGER NOT NULL,
                    narrative_stage TEXT NOT NULL,
                    revelation_text TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (npc_id) REFERENCES NPCStats(npc_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NPCAgentState (
                    npc_id INT NOT NULL,
                    user_id INT NOT NULL,
                    conversation_id INT NOT NULL,
                    current_state JSONB,
                    last_decision JSONB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (npc_id, user_id, conversation_id),
                    FOREIGN KEY (npc_id) REFERENCES NPCStats(npc_id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NPCVisualAttributes (
                    id SERIAL PRIMARY KEY,
                    npc_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    hair_color TEXT,
                    hair_style TEXT,
                    eye_color TEXT,
                    skin_tone TEXT,
                    body_type TEXT,
                    height TEXT,
                    age_appearance TEXT,
                    default_outfit TEXT,
                    outfit_variations JSONB,
                    makeup_style TEXT,
                    accessories JSONB,
                    expressions JSONB,
                    poses JSONB,
                    visual_seed TEXT,
                    last_generated_image TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (npc_id) REFERENCES NPCStats(npc_id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NPCVisualEvolution (
                    id SERIAL PRIMARY KEY,
                    npc_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    event_type TEXT CHECK (event_type IN ('outfit_change','appearance_change','location_change','mood_change')),
                    event_description TEXT,
                    previous_state JSONB,
                    current_state JSONB,
                    scene_context TEXT,
                    image_generated TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (npc_id) REFERENCES NPCStats(npc_id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                
                # ======================================
                # PLAYER TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS PlayerStats (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL,
                    corruption INT CHECK (corruption BETWEEN 0 AND 100),
                    confidence INT CHECK (confidence BETWEEN 0 AND 100),
                    willpower INT CHECK (willpower BETWEEN 0 AND 100),
                    obedience INT CHECK (obedience BETWEEN 0 AND 100),
                    dependency INT CHECK (dependency BETWEEN 0 AND 100),
                    lust INT CHECK (lust BETWEEN 0 AND 100),
                    mental_resilience INT CHECK (mental_resilience BETWEEN 0 AND 100),
                    physical_endurance INT CHECK (physical_endurance BETWEEN 0 AND 100),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerInventory (
                    item_id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL,
                    conversation_id INT NOT NULL,
                    player_name VARCHAR(255) NOT NULL DEFAULT 'Chase',
                    item_name VARCHAR(100) NOT NULL,
                    item_description TEXT,
                    item_category VARCHAR(50) NOT NULL,
                    item_properties JSONB,
                    quantity INT DEFAULT 1,
                    equipped BOOLEAN DEFAULT FALSE,
                    date_acquired TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_player_inventory_user ON PlayerInventory(user_id, conversation_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerResources (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL DEFAULT 'Chase',
                    money INTEGER NOT NULL DEFAULT 100,
                    supplies INTEGER NOT NULL DEFAULT 20,
                    influence INTEGER NOT NULL DEFAULT 10,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, conversation_id, player_name),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerVitals (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    player_name VARCHAR(255) NOT NULL DEFAULT 'Chase',
                    energy INTEGER NOT NULL DEFAULT 100,
                    hunger INTEGER NOT NULL DEFAULT 100,
                    last_update TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, conversation_id, player_name)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerPerks (
                    perk_id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL,
                    conversation_id INT NOT NULL,
                    perk_name VARCHAR(100) NOT NULL,
                    perk_description TEXT,
                    perk_category VARCHAR(50) NOT NULL,
                    perk_tier INT DEFAULT 1,
                    perk_properties JSONB,
                    date_acquired TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_player_perks_user ON PlayerPerks(user_id, conversation_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerSpecialRewards (
                    reward_id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL,
                    conversation_id INT NOT NULL,
                    reward_name VARCHAR(100) NOT NULL,
                    reward_description TEXT,
                    reward_effect TEXT,
                    reward_category VARCHAR(50) NOT NULL,
                    reward_properties JSONB,
                    used BOOLEAN DEFAULT FALSE,
                    date_acquired TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_player_special_rewards_user ON PlayerSpecialRewards(user_id, conversation_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerJournal (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    entry_type TEXT NOT NULL,
                    entry_text TEXT NOT NULL,
                    revelation_types TEXT,
                    narrative_moment TEXT,
                    fantasy_flag BOOLEAN DEFAULT FALSE,
                    intensity_level INT CHECK (intensity_level BETWEEN 0 AND 4),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    entry_metadata JSONB DEFAULT '{}',
                    importance FLOAT DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags JSONB DEFAULT '[]',
                    consolidated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS player_journal_user_conversation_idx ON PlayerJournal(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS player_journal_importance_idx ON PlayerJournal(importance);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS player_journal_created_at_idx ON PlayerJournal(created_at);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS player_journal_entry_type_idx ON PlayerJournal(entry_type);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerAddictions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    player_name VARCHAR(255) NOT NULL,
                    addiction_type VARCHAR(50) NOT NULL,
                    level INTEGER NOT NULL DEFAULT 0,
                    target_npc_id INTEGER NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, conversation_id, player_name, addiction_type, target_npc_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_player_addictions_lookup
                ON PlayerAddictions(user_id, conversation_id, player_name);
                ''',
                
                # ======================================
                # MEMORY SYSTEM TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS NPCMemories (
                    id SERIAL PRIMARY KEY,
                    npc_id INT NOT NULL,
                    memory_text TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    tags JSONB DEFAULT '[]',
                    emotional_intensity INT DEFAULT 0,
                    times_recalled INT DEFAULT 0,
                    last_recalled TIMESTAMP,
                    embedding VECTOR(384),
                    memory_type TEXT DEFAULT 'observation',
                    associated_entities JSONB DEFAULT '{}'::jsonb,
                    is_consolidated BOOLEAN NOT NULL DEFAULT FALSE,
                    is_archived BOOLEAN DEFAULT FALSE,
                    significance INT NOT NULL DEFAULT 3,
                    status VARCHAR(20) NOT NULL DEFAULT 'active',
                    FOREIGN KEY (npc_id) REFERENCES NPCStats(npc_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_mem_npcid_status_ts
                ON NPCMemories (npc_id, status, timestamp);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS npc_memory_embedding_hnsw_idx
                ON NPCMemories
                USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_npc_memories_archived 
                ON NPCMemories(is_archived) 
                WHERE is_archived = FALSE;
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NPCMemoryAssociations (
                    id SERIAL PRIMARY KEY,
                    memory_id INT NOT NULL,
                    associated_memory_id INT NOT NULL,
                    association_strength FLOAT DEFAULT 0.0,
                    association_type TEXT,
                    FOREIGN KEY (memory_id) REFERENCES NPCMemories(id) ON DELETE CASCADE,
                    FOREIGN KEY (associated_memory_id) REFERENCES NPCMemories(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS unified_memories (
                    id SERIAL PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'observation',
                    significance INTEGER NOT NULL DEFAULT 3,
                    emotional_intensity INTEGER NOT NULL DEFAULT 0,
                    tags JSONB DEFAULT '[]',
                    embedding VECTOR(384),
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    times_recalled INTEGER NOT NULL DEFAULT 0,
                    last_recalled TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'active',
                    is_consolidated BOOLEAN NOT NULL DEFAULT FALSE,
                    is_archived BOOLEAN DEFAULT FALSE,
                    relevance_score FLOAT DEFAULT 0.0,
                    last_context_update TIMESTAMP
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_unified_memories_entity
                ON unified_memories(entity_type, entity_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_unified_memories_archived 
                ON unified_memories(is_archived) 
                WHERE is_archived = FALSE;
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_unified_memories_user_conv
                ON unified_memories(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_unified_memories_timestamp
                ON unified_memories(timestamp);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_unified_memories_embedding_hnsw
                ON unified_memories
                USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS MemoryMaintenanceSchedule (
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    maintenance_schedule JSONB NOT NULL,
                    next_maintenance_date TIMESTAMP NOT NULL,
                    last_maintenance_date TIMESTAMP,
                    PRIMARY KEY (user_id, conversation_id, entity_type, entity_id)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS memory_telemetry (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    operation TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    duration FLOAT NOT NULL,
                    data_size INTEGER,
                    error TEXT,
                    metadata JSONB
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_memory_telemetry_timestamp
                ON memory_telemetry(timestamp);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_memory_telemetry_operation
                ON memory_telemetry(operation);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_memory_telemetry_success
                ON memory_telemetry(success);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS SemanticNetworks (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id INTEGER NOT NULL,
                    central_topic TEXT NOT NULL,
                    network_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                
                # ======================================
                # ARCHETYPE AND ACTIVITY TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS Archetypes (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    baseline_stats JSONB NOT NULL,
                    progression_rules JSONB NOT NULL,
                    setting_examples JSONB NOT NULL,
                    unique_traits JSONB NOT NULL
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS Activities (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    purpose JSONB NOT NULL,
                    stat_integration JSONB,
                    intensity_tiers JSONB,
                    setting_variants JSONB,
                    fantasy_level TEXT DEFAULT 'realistic'
                        CHECK (fantasy_level IN ('realistic','fantastical','surreal'))
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ActivityEffects (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    activity_name TEXT NOT NULL,
                    activity_details TEXT,
                    setting_context TEXT,
                    effects JSONB NOT NULL,
                    description TEXT,
                    flags JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, conversation_id, activity_name, activity_details)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_activity_effects_lookup
                ON ActivityEffects(user_id, conversation_id, activity_name, activity_details);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS IntensityTiers (
                    id SERIAL PRIMARY KEY,
                    tier_name TEXT NOT NULL,
                    range_min INT NOT NULL,
                    range_max INT NOT NULL,
                    description TEXT NOT NULL,
                    key_features JSONB NOT NULL,
                    activity_examples JSONB,
                    permanent_effects JSONB,
                    fantasy_level TEXT DEFAULT 'realistic'
                        CHECK (fantasy_level IN ('realistic','fantastical','surreal'))
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS Interactions (
                    id SERIAL PRIMARY KEY,
                    interaction_name TEXT UNIQUE NOT NULL,
                    detailed_rules JSONB NOT NULL,
                    task_examples JSONB,
                    agency_overrides JSONB,
                    fantasy_level TEXT DEFAULT 'realistic'
                        CHECK (fantasy_level IN ('realistic','fantastical','surreal'))
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlotTriggers (
                    id SERIAL PRIMARY KEY,
                    trigger_name TEXT UNIQUE NOT NULL,
                    stage_name TEXT,
                    description TEXT,
                    key_features JSONB,
                    stat_dynamics JSONB,
                    examples JSONB,
                    triggers JSONB
                );
                ''',
                
                # ======================================
                # FACTION AND POLITICAL TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS Factions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT DEFAULT 'community' CHECK (type IN ('political', 'community', 'social', 'hobby', 'educational', 'professional', 'religious', 'criminal')),
                    description TEXT,
                    values JSONB DEFAULT '[]',           -- Changed from TEXT[]
                    goals JSONB DEFAULT '[]',            -- Changed from TEXT[]
                    hierarchy_type TEXT DEFAULT 'informal',
                    resources JSONB DEFAULT '[]',        -- Changed from TEXT[]
                    territory TEXT,
                    meeting_schedule TEXT,
                    membership_requirements JSONB DEFAULT '[]', -- Changed from TEXT[]
                    rivals JSONB DEFAULT '[]',           -- Changed from INTEGER[]
                    allies JSONB DEFAULT '[]',           -- Changed from INTEGER[]
                    public_reputation TEXT DEFAULT 'neutral',
                    secret_activities JSONB DEFAULT '[]', -- Changed from TEXT[]
                    power_level INTEGER DEFAULT 3 CHECK (power_level BETWEEN 1 AND 10),
                    influence_scope TEXT DEFAULT 'local' CHECK (influence_scope IN ('personal', 'local', 'regional', 'national', 'global')),
                    recruitment_methods JSONB DEFAULT '[]', -- Changed from TEXT[]
                    leadership_structure JSONB,
                    founding_story TEXT,
                    embedding Vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    UNIQUE(user_id, conversation_id, name)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_factions_name
                ON Factions(user_id, conversation_id, lower(name));
                ''',
                '''
                CREATE TABLE IF NOT EXISTS FactionPowerShifts (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    faction_name VARCHAR(255) NOT NULL,
                    power_level INTEGER NOT NULL,
                    change_amount INTEGER NOT NULL,
                    cause TEXT NOT NULL,
                    conflict_id INTEGER,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_faction_power_shifts
                ON FactionPowerShifts(user_id, conversation_id);
                ''',
                
                # ======================================
                # WORLD LORE TABLES (NEW)
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS Nations (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    government_type TEXT,
                    description TEXT,
                    relative_power INTEGER DEFAULT 5,
                    matriarchy_level INTEGER DEFAULT 5,
                    population_scale TEXT,
                    major_resources JSONB DEFAULT '[]',
                    major_cities JSONB DEFAULT '[]',
                    cultural_traits JSONB DEFAULT '[]',
                    neighboring_nations JSONB DEFAULT '[]',
                    notable_features TEXT,
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_nations_embedding_hnsw
                ON Nations USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_nations_name
                ON nations (lower(name));
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NationalConflicts (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    description TEXT,
                    severity INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'active',
                    start_date TEXT,
                    involved_nations JSONB DEFAULT '[]',
                    primary_aggressor INTEGER,
                    primary_defender INTEGER,
                    current_casualties TEXT,
                    economic_impact TEXT,
                    diplomatic_consequences TEXT,
                    public_opinion JSONB,
                    recent_developments JSONB DEFAULT '[]',
                    potential_resolution TEXT,
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_national_conflicts_embedding_hnsw
                ON NationalConflicts USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS CulturalElements (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    element_type TEXT NOT NULL,
                    description TEXT,
                    practiced_by JSONB DEFAULT '[]',
                    significance INTEGER DEFAULT 5,
                    historical_origin TEXT,
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_cultural_elements_embedding_hnsw
                ON CulturalElements USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS CulinaryTraditions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    nation_origin INTEGER,
                    description TEXT,
                    ingredients JSONB DEFAULT '[]',
                    preparation TEXT,
                    cultural_significance TEXT,
                    adopted_by JSONB DEFAULT '[]',
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_culinary_traditions_embedding_hnsw
                ON CulinaryTraditions USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS SocialCustoms (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    nation_origin INTEGER,
                    description TEXT,
                    context TEXT DEFAULT 'social',
                    formality_level TEXT DEFAULT 'medium',
                    adopted_by JSONB DEFAULT '[]',
                    adoption_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_social_customs_embedding_hnsw
                ON SocialCustoms USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS JointMemories (
                    memory_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    source_type VARCHAR(50) NOT NULL,
                    source_id INTEGER NOT NULL,
                    significance INTEGER DEFAULT 5,
                    tags JSONB DEFAULT '[]',::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_joint_memories_user_conv
                ON JointMemories(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_joint_memories_source
                ON JointMemories(source_type, source_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS JointMemorySharing (
                    id SERIAL PRIMARY KEY,
                    memory_id INTEGER NOT NULL,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES JointMemories(memory_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_joint_memory_sharing_memory
                ON JointMemorySharing(memory_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_joint_memory_sharing_entity
                ON JointMemorySharing(entity_type, entity_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS CulturalExchanges (
                    id SERIAL PRIMARY KEY,
                    nation1_id INTEGER NOT NULL,
                    nation2_id INTEGER NOT NULL,
                    exchange_type TEXT,
                    exchange_details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS GeographicRegions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    region_type TEXT NOT NULL,
                    description TEXT,
                    climate TEXT,
                    resources JSONB DEFAULT '[]',
                    governing_faction TEXT,
                    population_density TEXT,
                    major_settlements JSONB DEFAULT '[]',
                    cultural_traits JSONB DEFAULT '[]',
                    dangers JSONB DEFAULT '[]',
                    terrain_features JSONB DEFAULT '[]',
                    defensive_characteristics TEXT,
                    strategic_value INTEGER DEFAULT 5,
                    matriarchal_influence INTEGER DEFAULT 5,
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_geographic_regions_embedding_hnsw
                ON GeographicRegions USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PoliticalEntities (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    description TEXT,
                    region_id INTEGER,
                    governance_style TEXT,
                    leadership_structure TEXT,
                    population_scale TEXT,
                    cultural_identity TEXT,
                    economic_focus TEXT,
                    political_values TEXT,
                    matriarchy_level INTEGER DEFAULT 5,
                    relations JSONB,
                    military_strength INTEGER DEFAULT 5,
                    diplomatic_stance TEXT,
                    internal_conflicts JSONB DEFAULT '[]',
                    power_centers JSONB,
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_political_entities_embedding_hnsw
                ON PoliticalEntities USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ConflictSimulations (
                    id SERIAL PRIMARY KEY,
                    conflict_type TEXT NOT NULL,
                    primary_actors JSONB,
                    timeline JSONB,
                    intensity_progression TEXT,
                    diplomatic_events JSONB,
                    military_events JSONB,
                    civilian_impact JSONB,
                    resolution_scenarios JSONB,
                    most_likely_outcome JSONB,
                    duration_months INTEGER,
                    confidence_level FLOAT,
                    simulation_basis TEXT,
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_conflict_simulations_embedding_hnsw
                ON ConflictSimulations USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS BorderDisputes (
                    id SERIAL PRIMARY KEY,
                    region1_id INTEGER NOT NULL,
                    region2_id INTEGER NOT NULL,
                    dispute_type TEXT NOT NULL,
                    description TEXT,
                    severity INTEGER DEFAULT 5,
                    duration TEXT,
                    causal_factors TEXT,
                    status TEXT DEFAULT 'active',
                    resolution_attempts JSONB,
                    strategic_implications TEXT,
                    female_leaders_involved JSONB DEFAULT '[]',
                    gender_dynamics TEXT,
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_border_disputes_embedding_hnsw
                ON BorderDisputes USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS UrbanMyths (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    origin_location TEXT,
                    origin_event TEXT,
                    believability INTEGER DEFAULT 6,
                    spread_rate INTEGER DEFAULT 5,
                    regions_known JSONB DEFAULT '[]',
                    narrative_style TEXT DEFAULT 'folklore',
                    themes JSONB DEFAULT '[]',
                    matriarchal_elements JSONB DEFAULT '[]',
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_urban_myths_embedding_hnsw
                ON UrbanMyths USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS LocalHistories (
                    id SERIAL PRIMARY KEY,
                    location_id INTEGER NOT NULL,
                    event_name TEXT NOT NULL,
                    description TEXT,
                    date_description TEXT,
                    significance INTEGER DEFAULT 5,
                    impact_type TEXT,
                    notable_figures JSONB DEFAULT '[]',
                    current_relevance TEXT,
                    commemoration TEXT,
                    connected_myths JSONB DEFAULT '[]',
                    related_landmarks JSONB DEFAULT '[]',
                    narrative_category TEXT,
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_local_histories_embedding_hnsw
                ON LocalHistories USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS Landmarks (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    location_id INTEGER NOT NULL,
                    landmark_type TEXT NOT NULL,
                    description TEXT,
                    historical_significance TEXT,
                    current_use TEXT,
                    controlled_by TEXT,
                    legends JSONB DEFAULT '[]',
                    connected_histories JSONB DEFAULT '[]',
                    architectural_style TEXT,
                    symbolic_meaning TEXT,
                    matriarchal_significance TEXT DEFAULT 'moderate',
                    embedding vector(384)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_landmarks_embedding_hnsw
                ON Landmarks USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS HistoricalEvents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    date_description TEXT,
                    event_type TEXT DEFAULT 'political',
                    significance INTEGER DEFAULT 5,
                    involved_entities JSONB DEFAULT '[]',
                    location TEXT,
                    consequences JSONB DEFAULT '[]',
                    cultural_impact TEXT DEFAULT 'moderate',
                    disputed_facts JSONB DEFAULT '[]',
                    commemorations JSONB DEFAULT '[]',
                    primary_sources JSONB DEFAULT '[]',
                    embedding vector(384),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_historical_events_embedding_hnsw
                ON HistoricalEvents USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NotableFigures (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    birth_date TEXT,
                    death_date TEXT,
                    faction_affiliations JSONB DEFAULT '[]',
                    achievements JSONB DEFAULT '[]',
                    failures JSONB DEFAULT '[]',
                    personality_traits JSONB DEFAULT '[]',
                    public_image TEXT DEFAULT 'neutral',
                    hidden_aspects JSONB DEFAULT '[]',
                    influence_areas JSONB DEFAULT '[]',
                    legacy TEXT,
                    controversial_actions JSONB DEFAULT '[]',
                    relationships JSONB DEFAULT '[]',
                    current_status TEXT DEFAULT 'active',
                    reputation INTEGER DEFAULT 50,
                    significance INTEGER DEFAULT 5,
                    embedding vector(384),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_notable_figures_embedding_hnsw
                ON NotableFigures USING hnsw (embedding vector_cosine_ops);
                ''',
                
                # ======================================
                # RELATIONSHIP AND SOCIAL TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS SocialLinks (
                    link_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    entity1_type TEXT NOT NULL,
                    entity1_id INT NOT NULL,
                    entity2_type TEXT NOT NULL,
                    entity2_id INT NOT NULL,
                    link_type TEXT,
                    link_level INT DEFAULT 0,
                    link_history JSONB,
                    dynamics JSONB,
                    experienced_crossroads JSONB,
                    experienced_rituals JSONB,
                    relationship_stage TEXT,
                    group_interaction TEXT,
                    UNIQUE (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS RelationshipEvolution (
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc1_id INTEGER NOT NULL,
                    entity2_type TEXT NOT NULL,
                    entity2_id INTEGER NOT NULL,
                    relationship_type TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    progress_to_next INTEGER NOT NULL DEFAULT 0,
                    evolution_history JSONB NOT NULL DEFAULT '[]'::jsonb,
                    PRIMARY KEY (user_id, conversation_id, npc1_id, entity2_type, entity2_id)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS interaction_history (
                    id SERIAL PRIMARY KEY,
                    entity1_id VARCHAR(100) NOT NULL,
                    entity2_id VARCHAR(100) NOT NULL,
                    interaction_type VARCHAR(50) NOT NULL,
                    outcome VARCHAR(50) NOT NULL,
                    emotional_impact JSONB NOT NULL DEFAULT '{}',
                    duration INTEGER NOT NULL DEFAULT 0,
                    intensity FLOAT NOT NULL DEFAULT 0.5,
                    relationship_changes JSONB NOT NULL DEFAULT '{}',
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_interaction_history_entities 
                ON interaction_history(entity1_id, entity2_id, created_at);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_interaction_history_user_conv
                ON interaction_history(user_id, conversation_id, created_at);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_interaction_history_lookup
                ON interaction_history(user_id, conversation_id, entity1_id, entity2_id);
                ''',
                '''
                COMMENT ON TABLE interaction_history IS 'Tracks interactions between entities (NPCs, users) for relationship management';
                ''',
                
                # ======================================
                # CONFLICT SYSTEM TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS Conflicts (
                    conflict_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    conflict_name TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    description TEXT,
                    progress REAL DEFAULT 0.0,
                    phase TEXT DEFAULT 'brewing',
                    start_day INTEGER,
                    estimated_duration INTEGER,
                    success_rate REAL,
                    outcome TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_conflicts_user_conv
                ON Conflicts(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_active_conflicts
                ON Conflicts(is_active) WHERE is_active = TRUE;
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ConflictStakeholders (
                    id SERIAL PRIMARY KEY,
                    conflict_id INTEGER REFERENCES Conflicts(conflict_id),
                    npc_id INTEGER NOT NULL,
                    faction_id INTEGER,
                    faction_name TEXT,
                    faction_position TEXT,
                    public_motivation TEXT,
                    private_motivation TEXT,
                    desired_outcome TEXT,
                    involvement_level INTEGER DEFAULT 5,
                    alliances JSONB DEFAULT '{}',
                    rivalries JSONB DEFAULT '{}',
                    leadership_ambition INTEGER,
                    faction_standing INTEGER,
                    willing_to_betray_faction BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ResolutionPaths (
                    id SERIAL PRIMARY KEY,
                    conflict_id INTEGER REFERENCES Conflicts(conflict_id),
                    path_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    approach_type TEXT,
                    difficulty INTEGER,
                    requirements JSONB DEFAULT '{}',
                    stakeholders_involved JSONB DEFAULT '[]',
                    key_challenges JSONB DEFAULT '[]',
                    progress REAL DEFAULT 0.0,
                    is_completed BOOLEAN DEFAULT FALSE,
                    completion_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PathStoryBeats (
                    beat_id SERIAL PRIMARY KEY,
                    conflict_id INTEGER REFERENCES Conflicts(conflict_id),
                    path_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    involved_npcs JSONB DEFAULT '[]',
                    progress_value REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS StakeholderSecrets (
                    id SERIAL PRIMARY KEY,
                    conflict_id INTEGER REFERENCES Conflicts(conflict_id),
                    npc_id INTEGER NOT NULL,
                    secret_id TEXT NOT NULL,
                    secret_type TEXT,
                    content TEXT,
                    target_npc_id INTEGER,
                    is_revealed BOOLEAN DEFAULT FALSE,
                    revealed_to INTEGER,
                    is_public BOOLEAN DEFAULT FALSE,
                    revealed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerManipulationAttempts (
                    attempt_id SERIAL PRIMARY KEY,
                    conflict_id INTEGER REFERENCES Conflicts(conflict_id),
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc_id INTEGER NOT NULL,
                    manipulation_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    goal JSONB DEFAULT '{}',
                    success BOOLEAN DEFAULT FALSE,
                    player_response TEXT,
                    leverage_used JSONB DEFAULT '{}',
                    intimacy_level INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerConflictInvolvement (
                    id SERIAL PRIMARY KEY,
                    conflict_id INTEGER REFERENCES Conflicts(conflict_id),
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL,
                    involvement_level TEXT DEFAULT 'none',
                    faction TEXT DEFAULT 'neutral',
                    money_committed INTEGER DEFAULT 0,
                    supplies_committed INTEGER DEFAULT 0,
                    influence_committed INTEGER DEFAULT 0,
                    actions_taken JSONB DEFAULT '[]',
                    manipulated_by JSONB DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS InternalFactionConflicts (
                    struggle_id SERIAL PRIMARY KEY,
                    faction_id INTEGER NOT NULL,
                    conflict_name TEXT NOT NULL,
                    description TEXT,
                    primary_npc_id INTEGER NOT NULL,
                    target_npc_id INTEGER NOT NULL,
                    prize TEXT,
                    approach TEXT,
                    public_knowledge BOOLEAN DEFAULT FALSE,
                    current_phase TEXT DEFAULT 'brewing',
                    progress INTEGER DEFAULT 0,
                    parent_conflict_id INTEGER REFERENCES Conflicts(conflict_id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    FOREIGN KEY (parent_conflict_id) REFERENCES Conflicts(conflict_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS FactionStruggleMembers (
                    id SERIAL PRIMARY KEY,
                    struggle_id INTEGER REFERENCES InternalFactionConflicts(struggle_id),
                    npc_id INTEGER NOT NULL,
                    position TEXT,
                    side TEXT DEFAULT 'neutral',
                    standing INTEGER DEFAULT 50,
                    loyalty_strength INTEGER DEFAULT 50,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS FactionIdeologicalDifferences (
                    id SERIAL PRIMARY KEY,
                    struggle_id INTEGER REFERENCES InternalFactionConflicts(struggle_id),
                    issue TEXT NOT NULL,
                    incumbent_position TEXT,
                    challenger_position TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS FactionCoupAttempts (
                    id SERIAL PRIMARY KEY,
                    struggle_id INTEGER REFERENCES InternalFactionConflicts(struggle_id),
                    approach TEXT NOT NULL,
                    supporting_npcs JSONB DEFAULT '[]',
                    resources_committed JSONB DEFAULT '{}',
                    success BOOLEAN NOT NULL,
                    success_chance REAL NOT NULL,
                    result JSONB DEFAULT '{}',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ConflictMemoryEvents (
                    id SERIAL PRIMARY KEY,
                    conflict_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    significance INTEGER NOT NULL DEFAULT 5,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id INTEGER NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conflict_id) REFERENCES Conflicts(conflict_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ConflictConsequences (
                    id SERIAL PRIMARY KEY,
                    conflict_id INTEGER NOT NULL,
                    consequence_type VARCHAR(50) NOT NULL,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id INTEGER NULL,
                    description TEXT NOT NULL,
                    applied BOOLEAN NOT NULL DEFAULT FALSE,
                    applied_at TIMESTAMP NULL,
                    FOREIGN KEY (conflict_id) REFERENCES Conflicts(conflict_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ConflictNPCs (
                    conflict_id INTEGER NOT NULL,
                    npc_id INTEGER NOT NULL,
                    faction VARCHAR(10) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    influence_level INTEGER NOT NULL DEFAULT 50,
                    PRIMARY KEY (conflict_id, npc_id),
                    FOREIGN KEY (conflict_id) REFERENCES Conflicts(conflict_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_conflict_npcs
                ON ConflictNPCs(conflict_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ConflictHistory (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    conflict_id INTEGER NOT NULL,
                    affected_npc_id INTEGER NOT NULL,
                    impact_type VARCHAR(50) NOT NULL,
                    grudge_level INTEGER NOT NULL DEFAULT 0,
                    narrative_impact TEXT NOT NULL,
                    has_triggered_consequence BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conflict_id) REFERENCES Conflicts(conflict_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_conflict_history
                ON ConflictHistory(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_grudge_level
                ON ConflictHistory(grudge_level) WHERE grudge_level > 50;
                ''',
                
                # ======================================
                # QUEST AND EVENT TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS Quests (
                    quest_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    quest_name TEXT,
                    status TEXT NOT NULL DEFAULT 'In Progress',
                    progress_detail TEXT,
                    quest_giver TEXT,
                    reward JSONB DEFAULT '[]',
                    embedding vector(384),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_quests_embedding_hnsw
                ON Quests USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlannedEvents (
                    event_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc_id INT REFERENCES NPCStats(npc_id),
                    year INT DEFAULT 1,
                    month INT DEFAULT 1,
                    day INT NOT NULL,
                    time_of_day TEXT NOT NULL,
                    override_location TEXT NOT NULL,
                    UNIQUE(npc_id, year, month, day, time_of_day),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                
                # ======================================
                # CANONICAL EVENT TRACKING
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS CanonicalEvents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    event_text TEXT NOT NULL,
                    tags JSONB DEFAULT '[]',
                    significance INTEGER DEFAULT 5 CHECK (significance BETWEEN 1 AND 10),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_canonical_events_lookup
                ON CanonicalEvents(user_id, conversation_id, timestamp DESC);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_canonical_events_tags
                ON CanonicalEvents USING GIN (tags);
                ''',
                
                # ======================================
                # HISTORY AND TRACKING TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS ResourceHistoryLog (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL DEFAULT 'Chase',
                    resource_type TEXT NOT NULL,
                    old_value INTEGER NOT NULL,
                    new_value INTEGER NOT NULL,
                    amount_changed INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS StatsHistory (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL,
                    stat_name TEXT NOT NULL,
                    old_value INTEGER NOT NULL,
                    new_value INTEGER NOT NULL,
                    cause TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                
                # ======================================
                # NYX AGENT SYSTEM TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS nyx_brain_checkpoints (
                    id BIGSERIAL PRIMARY KEY,
                    nyx_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    checkpoint_time TIMESTAMP NOT NULL DEFAULT now(),
                    event TEXT,
                    serialized_state JSONB NOT NULL,
                    merged_from JSONB DEFAULT '[]',
                    notes TEXT,
                    UNIQUE(nyx_id, instance_id, checkpoint_time)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_nyx_brain_checkpoints_nyx_id_time 
                ON nyx_brain_checkpoints(nyx_id, checkpoint_time DESC);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS nyx_brain_events (
                    id BIGSERIAL PRIMARY KEY,
                    nyx_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    event_time TIMESTAMP NOT NULL DEFAULT now(),
                    event_type TEXT NOT NULL,
                    event_payload JSONB NOT NULL,
                    UNIQUE(nyx_id, instance_id, event_time, event_type)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_nyx_brain_events_nyx_id_time 
                ON nyx_brain_events(nyx_id, event_time desc);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxMemories (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(384),
                    significance INT NOT NULL DEFAULT 3,
                    times_recalled INT NOT NULL DEFAULT 0,
                    last_recalled TIMESTAMP,
                    memory_type TEXT DEFAULT 'reflection',
                    is_archived BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxAgentState (
                    user_id INT NOT NULL,
                    conversation_id INT NOT NULL,
                    current_goals JSONB,
                    predicted_futures JSONB,
                    reflection_notes TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, conversation_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NPCNarrativeProgression (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc_id INTEGER NOT NULL,
                    narrative_stage VARCHAR(50) NOT NULL DEFAULT 'Innocent Beginning',
                    corruption INTEGER DEFAULT 0,
                    dependency INTEGER DEFAULT 0,
                    realization_level INTEGER DEFAULT 0,
                    last_revelation TIMESTAMP,
                    stage_entered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    stage_history JSONB DEFAULT '[]'::jsonb,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (npc_id) REFERENCES NPCStats(npc_id) ON DELETE CASCADE,
                    UNIQUE(user_id, conversation_id, npc_id)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxConversations (
                    nyx_conv_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    nyx_conversation_name VARCHAR(100) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'processing',
                    created_at TIMESTAMP DEFAULT NOW(),
                    folder_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE SET NULL
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS nyx_dm_messages (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    message JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxNPCDirectives (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc_id INTEGER NOT NULL,
                    directive JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    priority INTEGER DEFAULT 5
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS JointMemoryGraph (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    source_type VARCHAR(50) NOT NULL,
                    source_id INTEGER NOT NULL,
                    significance INTEGER DEFAULT 5,
                    tags JSONB DEFAULT '[]',::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS JointMemoryEdges (
                    id SERIAL PRIMARY KEY,
                    memory_graph_id INTEGER REFERENCES JointMemoryGraph(id),
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_joint_memory_edges_entity
                ON JointMemoryEdges(entity_type, entity_id, user_id, conversation_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxAgentDirectives (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    agent_id VARCHAR(50) NOT NULL,
                    directive JSONB NOT NULL,
                    priority INTEGER DEFAULT 5,
                    expires_at TIMESTAMP WITH TIME ZONE,
                    scene_id VARCHAR(50),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_agent_directives_agent
                ON NyxAgentDirectives(user_id, conversation_id, agent_type, agent_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxActionTracking (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    agent_id VARCHAR(50) NOT NULL,
                    action_type VARCHAR(50),
                    action_data JSONB,
                    result_data JSONB,
                    status VARCHAR(20),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_action_tracking_agent
                ON NyxActionTracking(user_id, conversation_id, agent_type, agent_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxAgentCommunication (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    sender_type VARCHAR(50) NOT NULL,
                    sender_id VARCHAR(50) NOT NULL,
                    recipient_type VARCHAR(50) NOT NULL,
                    recipient_id VARCHAR(50) NOT NULL,
                    message_type VARCHAR(50) NOT NULL,
                    message_content JSONB NOT NULL,
                    response_content JSONB,
                    status VARCHAR(20) DEFAULT 'sent',
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_agent_communication_sender
                ON NyxAgentCommunication(user_id, conversation_id, sender_type, sender_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_agent_communication_recipient
                ON NyxAgentCommunication(user_id, conversation_id, recipient_type, recipient_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxJointMemoryGraph (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    memory_type VARCHAR(50) DEFAULT 'observation',
                    source_type VARCHAR(50) NOT NULL,
                    source_id VARCHAR(50) NOT NULL,
                    significance INTEGER DEFAULT 5,
                    tags JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxJointMemoryAccess (
                    id SERIAL PRIMARY KEY,
                    memory_id INTEGER NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    agent_id VARCHAR(50) NOT NULL,
                    access_level VARCHAR(20) DEFAULT 'read',
                    granted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES NyxJointMemoryGraph(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxNarrativeGovernance (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    narrative_stage VARCHAR(50) NOT NULL,
                    governance_policy JSONB NOT NULL,
                    theme_directives JSONB,
                    pacing_directives JSONB,
                    character_directives JSONB,
                    active_from TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    active_until TIMESTAMP WITH TIME ZONE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS nyx1_strategy_injections (
                    id SERIAL PRIMARY KEY,
                    strategy_type TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT now(),
                    expires_at TIMESTAMP,
                    created_by TEXT DEFAULT 'nyx_2'
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS nyx1_scene_templates (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    prompt_template TEXT NOT NULL,
                    intensity_level INTEGER DEFAULT 5,
                    active BOOLEAN DEFAULT TRUE,
                    tags JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT now()
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS nyx1_strategy_logs (
                    id SERIAL PRIMARY KEY,
                    strategy_id INTEGER REFERENCES nyx1_strategy_injections(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL,
                    event_type TEXT,
                    message_snippet TEXT,
                    kink_profile JSONB,
                    decision_meta JSONB,
                    created_at TIMESTAMP DEFAULT now()
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS nyx1_response_noise (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    conversation_id INTEGER,
                    nyx_response TEXT,
                    score FLOAT DEFAULT 0.0,
                    marked_for_review BOOLEAN DEFAULT FALSE,
                    dismissed BOOLEAN DEFAULT FALSE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NyxAgentRegistry (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    agent_id VARCHAR(50) NOT NULL,
                    agent_name VARCHAR(100),
                    capabilities JSONB,
                    status VARCHAR(20) DEFAULT 'active',
                    last_active TIMESTAMP WITH TIME ZONE,
                    first_registered TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    CONSTRAINT agent_registry_unique UNIQUE (user_id, conversation_id, agent_type, agent_id)
                );
                ''',
                
                # ======================================
                # MISC AND SUPPORTING TABLES
                # ======================================
                '''
                CREATE TABLE IF NOT EXISTS CurrencySystem (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    currency_name TEXT NOT NULL,
                    currency_plural TEXT NOT NULL,
                    minor_currency_name TEXT,
                    minor_currency_plural TEXT,
                    exchange_rate INTEGER DEFAULT 100,
                    currency_symbol TEXT,
                    format_template TEXT DEFAULT '{{amount}} {{currency}}',
                    description TEXT,
                    setting_context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, conversation_id)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PlayerReputation (
                    user_id INT,
                    npc_id INT,
                    reputation_score FLOAT DEFAULT 0,
                    PRIMARY KEY (user_id, npc_id)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ReflectionLogs (
                    id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL,
                    reflection_text TEXT NOT NULL,
                    was_accurate BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS AIReflectionSettings (
                    id SERIAL PRIMARY KEY,
                    temperature FLOAT DEFAULT 0.7,
                    max_tokens INT DEFAULT 4000
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS strategy_reviews (
                    id SERIAL PRIMARY KEY,
                    strategy_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    reviewed_at TIMESTAMP,
                    reviewer_notes TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_strategy_reviews_strategy
                ON strategy_reviews(strategy_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_strategy_reviews_user
                ON strategy_reviews(user_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_strategy_reviews_status
                ON strategy_reviews(status);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS MemorySchemas (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    schema_name TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    influence_strength FLOAT NOT NULL DEFAULT 0.5,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_memory_schemas_entity 
                ON MemorySchemas(user_id, conversation_id, entity_type, entity_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS scenario_states (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    state_data JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_scenario_states_user_conv
                ON scenario_states(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_scenario_states_created
                ON scenario_states(created_at DESC);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    metrics JSONB NOT NULL,
                    error_log JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_latest
                ON performance_metrics(user_id, conversation_id, created_at DESC);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    metrics JSONB NOT NULL,
                    learned_patterns JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_learning_metrics_latest
                ON learning_metrics(user_id, conversation_id, created_at DESC);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ImageFeedback (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    original_prompt TEXT NOT NULL,
                    npc_names JSONB NOT NULL,
                    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
                    feedback_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE OR REPLACE VIEW UserVisualPreferences AS
                SELECT
                    user_id,
                    npc_name,
                    AVG(rating) as avg_rating,
                    COUNT(*) as feedback_count
                FROM
                    ImageFeedback,
                    jsonb_array_elements_text(npc_names) as npc_name
                WHERE
                    rating >= 4
                GROUP BY
                    user_id, npc_name;
                ''',
                '''
                CREATE TABLE IF NOT EXISTS UserKinkProfile (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    kink_type TEXT NOT NULL,
                    level INTEGER CHECK (level BETWEEN 0 AND 4) DEFAULT 0,
                    discovery_source TEXT,
                    first_discovered TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    frequency INTEGER DEFAULT 0,
                    intensity_preference INTEGER CHECK (intensity_preference BETWEEN 0 AND 4) DEFAULT 0,
                    trigger_context JSONB,
                    confidence_score FLOAT DEFAULT 0.5,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE (user_id, kink_type)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS NPCMasks (
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc_id INTEGER NOT NULL,
                    mask_data JSONB NOT NULL,
                    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, conversation_id, npc_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (npc_id) REFERENCES NPCStats(npc_id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_npc_masks_lookup 
                ON NPCMasks(user_id, conversation_id, npc_id);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS KinkTeaseHistory (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    kink_id INTEGER NOT NULL,
                    tease_text TEXT NOT NULL,
                    tease_type TEXT CHECK (tease_type IN ('narrative','meta_commentary','punishment')),
                    narrative_context TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (kink_id) REFERENCES UserKinkProfile(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ContextEvolution (
                    evolution_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    context_data JSONB NOT NULL,
                    changes JSONB NOT NULL,
                    context_shift FLOAT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_context_evolution_user_conversation
                ON ContextEvolution(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_context_evolution_timestamp
                ON ContextEvolution(timestamp);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS WorldLore (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    significance INTEGER DEFAULT 5 CHECK (significance BETWEEN 1 AND 10),
                    tags JSONB DEFAULT '[]',              -- Changed from TEXT[]
                    metadata JSONB DEFAULT '{}',
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_worldlore_user_conv
                ON WorldLore(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_worldlore_category
                ON WorldLore(category);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_worldlore_embedding_hnsw
                ON WorldLore USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE TABLE IF NOT EXISTS MemoryContextEvolution (
                    memory_id INTEGER NOT NULL,
                    evolution_id INTEGER NOT NULL,
                    relevance_change FLOAT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (memory_id, evolution_id),
                    FOREIGN KEY (memory_id) REFERENCES unified_memories(id),
                    FOREIGN KEY (evolution_id) REFERENCES ContextEvolution(evolution_id)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_memory_context_evolution_memory
                ON MemoryContextEvolution(memory_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_memory_context_evolution_evolution
                ON MemoryContextEvolution(evolution_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_memory_relevance_score
                ON unified_memories(relevance_score);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_memory_last_context_update
                ON unified_memories(last_context_update);
                ''',
                '''
                CREATE EXTENSION IF NOT EXISTS vector;
                ''',
                '''         
                ALTER TABLE NPCStats ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE Locations ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE Events ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE Factions ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE Nations ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE NationalConflicts ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE CulturalElements ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE CulinaryTraditions ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE SocialCustoms ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE GeographicRegions ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE PoliticalEntities ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE ConflictSimulations ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE BorderDisputes ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE UrbanMyths ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE LocalHistories ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE Landmarks ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE HistoricalEvents ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE NotableFigures ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE Quests ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE NPCMemories ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''
                ALTER TABLE unified_memories ADD COLUMN IF NOT EXISTS embedding vector(384);
                ''',
                '''                
                CREATE INDEX IF NOT EXISTS idx_npcstats_embedding_hnsw ON NPCStats USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_locations_embedding_hnsw ON Locations USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_events_embedding_hnsw ON Events USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_nations_embedding_hnsw ON Nations USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_national_conflicts_embedding_hnsw ON NationalConflicts USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_cultural_elements_embedding_hnsw ON CulturalElements USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_culinary_traditions_embedding_hnsw ON CulinaryTraditions USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_social_customs_embedding_hnsw ON SocialCustoms USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_geographic_regions_embedding_hnsw ON GeographicRegions USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_political_entities_embedding_hnsw ON PoliticalEntities USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_conflict_simulations_embedding_hnsw ON ConflictSimulations USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_border_disputes_embedding_hnsw ON BorderDisputes USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_urban_myths_embedding_hnsw ON UrbanMyths USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_local_histories_embedding_hnsw ON LocalHistories USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_landmarks_embedding_hnsw ON Landmarks USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_historical_events_embedding_hnsw ON HistoricalEvents USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_notable_figures_embedding_hnsw ON NotableFigures USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_quests_embedding_hnsw ON Quests USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS npc_memory_embedding_hnsw_idx ON NPCMemories USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_unified_memories_embedding_hnsw ON unified_memories USING hnsw (embedding vector_cosine_ops);
                ''',
                '''
                ALTER TABLE Factions 
                ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL DEFAULT 0,
                ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL DEFAULT 0;
                ''',
                '''
                ALTER TABLE Factions 
                ADD CONSTRAINT fk_factions_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                ADD CONSTRAINT fk_factions_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE;
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_factions_name
                ON Factions(user_id, conversation_id, lower(name));
                ''',
                '''
                ALTER TABLE NPCStats ADD COLUMN trauma_triggers JSONB;
                ''',
                '''
                ALTER TABLE NPCStats ADD COLUMN IF NOT EXISTS personality_patterns JSONB DEFAULT '[]'::jsonb;
                ''',
                '''
                ALTER TABLE NPCStats ADD COLUMN flashback_triggers JSONB;
                ''',
                '''
                ALTER TABLE NPCStats ADD COLUMN revelation_plan JSONB;
                ''',
                '''
                ALTER TABLE NyxAgentDirectives 
                    ALTER COLUMN created_at TYPE TIMESTAMP,
                    ALTER COLUMN expires_at TYPE TIMESTAMP;
                ''',
                '''
                ALTER TABLE NyxAgentCommunication 
                    ALTER COLUMN timestamp TYPE TIMESTAMP;
                ''',
                '''
                ALTER TABLE NyxJointMemoryGraph 
                    ALTER COLUMN created_at TYPE TIMESTAMP;
                ''',
                '''
                ALTER TABLE NyxJointMemoryAccess 
                    ALTER COLUMN granted_at TYPE TIMESTAMP;
                ''',
                '''
                ALTER TABLE NyxNarrativeGovernance 
                    ALTER COLUMN active_from TYPE TIMESTAMP,
                    ALTER COLUMN active_until TYPE TIMESTAMP;
                ''',
                '''
                ALTER TABLE nyx_brain_checkpoints 
                    ALTER COLUMN checkpoint_time TYPE TIMESTAMP;
                ''',
                '''
                ALTER TABLE nyx_brain_events 
                    ALTER COLUMN event_time TYPE TIMESTAMP;
                ''',
                '''
                ALTER TABLE CulturalElements 
                ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL DEFAULT 0,
                ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL DEFAULT 0;
                ''',
                '''
                ALTER TABLE CulturalElements 
                ADD CONSTRAINT fk_culturalelements_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                ADD CONSTRAINT fk_culturalelements_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE;
                ''',
                '''
                ALTER TABLE LocalHistories 
                ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL DEFAULT 0,
                ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL DEFAULT 0;
                ''',
                '''
                ALTER TABLE LocalHistories 
                ADD CONSTRAINT fk_localhistories_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                ADD CONSTRAINT fk_localhistories_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE;
                ''',
                '''
                ALTER TABLE Locations 
                    ALTER COLUMN notable_features TYPE JSONB USING to_jsonb(notable_features),
                    ALTER COLUMN hidden_aspects TYPE JSONB USING to_jsonb(hidden_aspects),
                    ALTER COLUMN access_restrictions TYPE JSONB USING to_jsonb(access_restrictions),
                    ALTER COLUMN local_customs TYPE JSONB USING to_jsonb(local_customs);
                ''',
                '''
                ALTER TABLE Factions 
                    ALTER COLUMN values TYPE JSONB USING to_jsonb(values),
                    ALTER COLUMN goals TYPE JSONB USING to_jsonb(goals),
                    ALTER COLUMN membership_requirements TYPE JSONB USING to_jsonb(membership_requirements),
                    ALTER COLUMN rivals TYPE JSONB USING to_jsonb(rivals),
                    ALTER COLUMN allies TYPE JSONB USING to_jsonb(allies),
                    ALTER COLUMN secret_activities TYPE JSONB USING to_jsonb(secret_activities),
                    ALTER COLUMN recruitment_methods TYPE JSONB USING to_jsonb(recruitment_methods);
                ''',
                '''
                ALTER TABLE Nations 
                    ALTER COLUMN major_resources TYPE JSONB USING to_jsonb(major_resources),
                    ALTER COLUMN major_cities TYPE JSONB USING to_jsonb(major_cities),
                    ALTER COLUMN cultural_traits TYPE JSONB USING to_jsonb(cultural_traits),
                    ALTER COLUMN neighboring_nations TYPE JSONB USING to_jsonb(neighboring_nations);
                ''',
                '''
                ALTER TABLE NationalConflicts 
                    ALTER COLUMN involved_nations TYPE JSONB USING to_jsonb(involved_nations),
                    ALTER COLUMN recent_developments TYPE JSONB USING to_jsonb(recent_developments);
                ''',
                '''
                ALTER TABLE CulturalElements 
                    ALTER COLUMN practiced_by TYPE JSONB USING to_jsonb(practiced_by);
                ''',
                '''
                ALTER TABLE CulinaryTraditions 
                    ALTER COLUMN ingredients TYPE JSONB USING to_jsonb(ingredients),
                    ALTER COLUMN adopted_by TYPE JSONB USING to_jsonb(adopted_by);
                ''',
                '''
                ALTER TABLE SocialCustoms 
                    ALTER COLUMN adopted_by TYPE JSONB USING to_jsonb(adopted_by);
                ''',
                '''
                ALTER TABLE GeographicRegions 
                    ALTER COLUMN resources TYPE JSONB USING to_jsonb(resources),
                    ALTER COLUMN major_settlements TYPE JSONB USING to_jsonb(major_settlements),
                    ALTER COLUMN cultural_traits TYPE JSONB USING to_jsonb(cultural_traits),
                    ALTER COLUMN dangers TYPE JSONB USING to_jsonb(dangers),
                    ALTER COLUMN terrain_features TYPE JSONB USING to_jsonb(terrain_features);
                ''',
                '''
                ALTER TABLE PoliticalEntities 
                    ALTER COLUMN internal_conflicts TYPE JSONB USING to_jsonb(internal_conflicts);
                ''',
                '''
                ALTER TABLE BorderDisputes 
                    ALTER COLUMN female_leaders_involved TYPE JSONB USING to_jsonb(female_leaders_involved);
                ''',
                '''
                ALTER TABLE UrbanMyths 
                    ALTER COLUMN regions_known TYPE JSONB USING to_jsonb(regions_known),
                    ALTER COLUMN themes TYPE JSONB USING to_jsonb(themes),
                    ALTER COLUMN matriarchal_elements TYPE JSONB USING to_jsonb(matriarchal_elements);
                ''',
                '''
                ALTER TABLE LocalHistories 
                    ALTER COLUMN notable_figures TYPE JSONB USING to_jsonb(notable_figures),
                    ALTER COLUMN connected_myths TYPE JSONB USING to_jsonb(connected_myths),
                    ALTER COLUMN related_landmarks TYPE JSONB USING to_jsonb(related_landmarks);
                ''',
                '''
                ALTER TABLE Landmarks 
                    ALTER COLUMN legends TYPE JSONB USING to_jsonb(legends),
                    ALTER COLUMN connected_histories TYPE JSONB USING to_jsonb(connected_histories);
                ''',
                '''
                ALTER TABLE HistoricalEvents 
                    ALTER COLUMN involved_entities TYPE JSONB USING to_jsonb(involved_entities),
                    ALTER COLUMN consequences TYPE JSONB USING to_jsonb(consequences),
                    ALTER COLUMN disputed_facts TYPE JSONB USING to_jsonb(disputed_facts),
                    ALTER COLUMN commemorations TYPE JSONB USING to_jsonb(commemorations),
                    ALTER COLUMN primary_sources TYPE JSONB USING to_jsonb(primary_sources);
                ''',
                '''
                ALTER TABLE NotableFigures 
                    ALTER COLUMN faction_affiliations TYPE JSONB USING to_jsonb(faction_affiliations),
                    ALTER COLUMN achievements TYPE JSONB USING to_jsonb(achievements),
                    ALTER COLUMN failures TYPE JSONB USING to_jsonb(failures),
                    ALTER COLUMN personality_traits TYPE JSONB USING to_jsonb(personality_traits),
                    ALTER COLUMN hidden_aspects TYPE JSONB USING to_jsonb(hidden_aspects),
                    ALTER COLUMN influence_areas TYPE JSONB USING to_jsonb(influence_areas),
                    ALTER COLUMN controversial_actions TYPE JSONB USING to_jsonb(controversial_actions),
                    ALTER COLUMN relationships TYPE JSONB USING to_jsonb(relationships);
                ''',
                '''
                ALTER TABLE CanonicalEvents 
                    ALTER COLUMN tags TYPE JSONB USING to_jsonb(tags);
                ''',
                '''
                ALTER TABLE nyx_brain_checkpoints 
                    ALTER COLUMN merged_from TYPE JSONB USING to_jsonb(merged_from);
                ''',
                '''
                ALTER TABLE nyx1_scene_templates 
                    ALTER COLUMN tags TYPE JSONB USING to_jsonb(tags);
                ''',
                '''
                ALTER TABLE unified_memories 
                    ALTER COLUMN tags TYPE JSONB USING to_jsonb(tags);
                ''',
                '''
                ALTER TABLE NPCMemories 
                    ALTER COLUMN tags TYPE JSONB USING to_jsonb(tags);
                ''',
                '''
                ALTER TABLE WorldLore 
                    ALTER COLUMN tags TYPE JSONB USING to_jsonb(tags);
                ''',
                '''
                ALTER TABLE Quests
                    ALTER COLUMN reward TYPE JSONB USING to_jsonb(tags);
                ''',
                '''
                ALTER TABLE Factions
                    ADD COLUMN IF NOT EXISTS hierarchy_type TEXT DEFAULT 'informal';
                    ADD COLUMN IF NOT EXISTS resources TYPE JSONB USING to_jsonb(tags);
                ''',
                '''
                ALTER TABLE unified_memories 
                ADD COLUMN IF NOT EXISTS is_archived BOOLEAN DEFAULT FALSE;
                ''',
                '''
                ALTER TABLE NPCMemories 
                ADD COLUMN IF NOT EXISTS is_archived BOOLEAN DEFAULT FALSE;
                ''',       
                '''
                ALTER TABLE Factions DROP CONSTRAINT IF EXISTS factions_type_check;
                ''',
                '''
                ALTER TABLE Factions ADD CONSTRAINT factions_type_check 
                CHECK (type IN ('political', 'community', 'social', 'hobby', 'educational', 
                                'professional', 'religious', 'criminal', 'corporate', 
                                'cult', 'supernatural', 'mystical', 'elite', 'military', 'underground',
                                'faction', 'organization', 'group', 'other'));
                ''',
                '''
                ALTER TABLE Locations 
                ADD COLUMN IF NOT EXISTS controlling_faction TEXT;
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PresetStories (
                    id SERIAL PRIMARY KEY,
                    story_id TEXT UNIQUE NOT NULL,
                    story_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS PresetStoryProgress (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    story_id TEXT NOT NULL,
                    current_act INTEGER DEFAULT 1,
                    completed_beats JSONB DEFAULT '[]',
                    story_variables JSONB DEFAULT '{}',
                    last_beat_timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    UNIQUE(user_id, conversation_id)
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS StoryBeatHistory (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    story_id TEXT NOT NULL,
                    beat_id TEXT NOT NULL,
                    trigger_context JSONB,
                    outcomes_applied JSONB,
                    player_choices JSONB,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                '''

# --- Seeding and Initialization Functions ---

async def seed_initial_vitals():
    """Asynchronously seed initial player vitals using asyncpg."""
    logger.info("Seeding initial player vitals...")
    rows_affected = 0
    try:
        async with get_db_connection_context() as conn:
            # Use a transaction for the read-then-write pattern
            async with conn.transaction():
                # Fetch distinct players who need seeding
                player_rows = await conn.fetch("""
                    SELECT DISTINCT user_id, conversation_id FROM PlayerStats
                    WHERE player_name = 'Chase'
                """)

                if not player_rows:
                    logger.info("No players found requiring initial vitals seeding.")
                    return

                # Insert for each player directly without prepared statement
                count = 0
                for row in player_rows:
                    result = await conn.execute("""
                        INSERT INTO PlayerVitals (user_id, conversation_id, player_name, energy, hunger)
                        VALUES ($1, $2, 'Chase', 100, 100)
                        ON CONFLICT (user_id, conversation_id, player_name) DO NOTHING
                    """, row['user_id'], row['conversation_id'])
                    
                    # Check if rows were affected
                    if result.endswith(" 1"):
                        count += 1
                rows_affected = count

        logger.info(f"Seeded initial vitals for {rows_affected} players (others may have existed).")

    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error seeding initial vitals: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error seeding vitals: {e}", exc_info=True)

async def initialize_conflict_system():
    """Asynchronously initialize the conflict system by seeding initial data."""
    await seed_initial_vitals()  # Call the async version
    logger.info("Conflict system initialized (vitals seeded).")

async def seed_story_poems_as_memories(story_id: str = "the_moth_and_flame"):
    """
    Seeds story poems as core memories that the AI can reference for tone and imagery.
    Call this during database initialization for each story that has poems.
    """
    logger.info(f"Seeding poems for story: {story_id}")
    
    # Get poems from the story
    if story_id == "the_moth_and_flame":
        poems = MOTH_FLAME_POEMS
        tone_prompt = STORY_TONE_PROMPT
    else:
        logger.warning(f"No poems found for story: {story_id}")
        return
    
    async with get_db_connection_context() as conn:
        # Seed each poem as a core memory
        for poem_id, poem_text in poems.items():
            # Check if already exists
            exists = await conn.fetchval(
                """
                SELECT id FROM unified_memories
                WHERE entity_type = 'story_poem' 
                AND entity_id = 0 
                AND metadata->>'poem_id' = $1
                AND metadata->>'story_id' = $2
                """,
                poem_id, story_id
            )
            
            if not exists:
                # Extract title (first line)
                lines = poem_text.strip().split('\n')
                title = lines[0] if lines else "Untitled"
                
                # Insert full poem
                await conn.execute(
                    """
                    INSERT INTO unified_memories
                    (entity_type, entity_id, user_id, conversation_id, 
                     memory_text, memory_type, tags, significance, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    "story_poem",    # Entity type for story poems
                    0,               # Generic ID
                    0,               # System-level (user_id = 0)
                    0,               # System-level (conversation_id = 0)
                    poem_text,
                    "gothic_poem",
                    ["poetry", "tone_reference", story_id, poem_id, "gothic", "romantic"],
                    10,              # Maximum significance
                    {
                        "poem_id": poem_id,
                        "poem_title": title,
                        "story_id": story_id,
                        "usage": "tone_and_imagery_reference",
                        "themes": ["masks", "vulnerability", "devotion", "abandonment"]
                    }
                )
                
                # Extract and store key imagery separately for quick access
                key_imagery = extract_poetic_imagery(poem_text)
                for i, imagery in enumerate(key_imagery[:20]):  # Limit to 20 key images
                    await conn.execute(
                        """
                        INSERT INTO unified_memories
                        (entity_type, entity_id, user_id, conversation_id,
                         memory_text, memory_type, tags, significance, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        "story_imagery",
                        0, 0, 0,
                        imagery,
                        "poetic_imagery",
                        ["imagery", story_id, poem_id, "metaphor"],
                        9,  # High significance
                        {
                            "source_poem": poem_id,
                            "story_id": story_id,
                            "imagery_index": i,
                            "imagery_type": categorize_imagery(imagery)
                        }
                    )
        
        # Seed the tone instructions
        if tone_prompt:
            exists = await conn.fetchval(
                """
                SELECT id FROM unified_memories
                WHERE entity_type = 'story_instructions'
                AND metadata->>'story_id' = $1
                AND user_id = 0 AND conversation_id = 0
                """,
                story_id
            )
            
            if not exists:
                await conn.execute(
                    """
                    INSERT INTO unified_memories
                    (entity_type, entity_id, user_id, conversation_id,
                     memory_text, memory_type, tags, significance, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    "story_instructions",
                    0, 0, 0,
                    tone_prompt,
                    "writing_instructions",
                    ["instructions", "tone", "style", story_id, "gothic", "poetry"],
                    10,
                    {
                        "story_id": story_id,
                        "instruction_type": "tone_and_style_guide",
                        "apply_to": "all_story_content",
                        "priority": "high"
                    }
                )
        
        logger.info(f"Seeded {len(poems)} poems and tone instructions for {story_id}")


def extract_poetic_imagery(poem_text: str) -> List[str]:
    """Extract key imagery and metaphors from poem text"""
    import re
    
    imagery = []
    lines = poem_text.split('\n')
    
    # Patterns that often indicate metaphorical language
    metaphor_patterns = [
        r'(?:I am|You are|She is|We are)\s+(.+?)(?:\.|,|$)',
        r'like\s+(.+?)(?:\.|,|$)',
        r'as\s+(.+?)(?:\.|,|$)',
        r'(?:beneath|behind|within)\s+(.+?)(?:\.|,|$)',
    ]
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and titles
        if not line or line.isupper() or len(line) < 10:
            continue
            
        # Look for metaphorical patterns
        for pattern in metaphor_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            for match in matches:
                if len(match) > 5 and len(match) < 100:
                    imagery.append(match.strip())
        
        # Also capture short, powerful standalone lines
        if 15 < len(line) < 60 and any(word in line.lower() for word in 
            ['moth', 'flame', 'mask', 'porcelain', 'velvet', 'thorns', 'altar', 
             'temple', 'shadow', 'mirror', 'glass', 'broken', 'trembling']):
            imagery.append(line)
    
    # Also include specific powerful phrases
    powerful_phrases = [
        "porcelain curves", "painted smile", "queen of thorns",
        "fortress forged from glass", "altar of her throne",
        "rough geography of breaks", "moth with wings of broken glass",
        "invisible tattoos", "binding kiss", "velvet affliction",
        "three syllables tasting of burning stars", "binary stars",
        "lunar edict", "sanctified ruin", "lighthouse on my midnight tide"
    ]
    
    for phrase in powerful_phrases:
        if phrase.lower() in poem_text.lower():
            imagery.append(phrase)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_imagery = []
    for img in imagery:
        if img.lower() not in seen:
            seen.add(img.lower())
            unique_imagery.append(img)
    
    return unique_imagery


def categorize_imagery(imagery: str) -> str:
    """Categorize imagery for better retrieval"""
    imagery_lower = imagery.lower()
    
    if any(word in imagery_lower for word in ['mask', 'porcelain', 'mirror', 'facade']):
        return "masks_and_facades"
    elif any(word in imagery_lower for word in ['moth', 'flame', 'burn', 'fire']):
        return "moth_and_flame"
    elif any(word in imagery_lower for word in ['altar', 'temple', 'worship', 'prayer']):
        return "religious_devotion"
    elif any(word in imagery_lower for word in ['broken', 'shatter', 'trembling', 'tears']):
        return "vulnerability"
    elif any(word in imagery_lower for word in ['velvet', 'silk', 'thorns', 'blood']):
        return "sensual_danger"
    elif any(word in imagery_lower for word in ['shadow', 'dark', 'night', 'void']):
        return "darkness"
    elif any(word in imagery_lower for word in ['stars', 'moon', 'gravity', 'orbit']):
        return "celestial"
    else:
        return "general_imagery"


# Update your main initialization to include poem seeding
async def initialize_all_data():
    """
    Asynchronous convenience function to create tables + seed initial data.
    Updated to include poem seeding for preset stories.
    """
    logger.info("Starting full database initialization...")
    try:
        await create_all_tables()
        await seed_initial_data()
        await seed_initial_vitals()
        await seed_initial_resources()
        await initialize_conflict_system()
        
        # Seed story poems
        await seed_story_poems_as_memories("the_moth_and_flame")
        
        logger.info("All initialization steps completed successfully!")
    except Exception as e:
        logger.critical(f"Full database initialization failed: {e}", exc_info=True)
        raise


# Function to retrieve poem context for a specific conversation
async def get_poem_context_for_conversation(user_id: int, conversation_id: int, story_id: str) -> Dict[str, Any]:
    """
    Retrieve poem context for use in text generation.
    This would be called by your Nyx system when generating responses.
    """
    async with get_db_connection_context() as conn:
        # Get relevant poems
        poems = await conn.fetch(
            """
            SELECT memory_text, metadata
            FROM unified_memories
            WHERE entity_type = 'story_poem'
            AND metadata->>'story_id' = $1
            ORDER BY significance DESC
            """,
            story_id
        )
        
        # Get imagery for current context
        imagery = await conn.fetch(
            """
            SELECT memory_text, metadata
            FROM unified_memories
            WHERE entity_type = 'story_imagery'
            AND metadata->>'story_id' = $1
            ORDER BY significance DESC
            LIMIT 15
            """,
            story_id
        )
        
        # Get tone instructions
        instructions = await conn.fetchval(
            """
            SELECT memory_text
            FROM unified_memories
            WHERE entity_type = 'story_instructions'
            AND metadata->>'story_id' = $1
            """,
            story_id
        )
        
        return {
            "poems": [{"text": p["memory_text"], "title": p["metadata"].get("poem_title")} for p in poems],
            "imagery": [i["memory_text"] for i in imagery],
            "tone_instructions": instructions,
            "story_id": story_id
        }
            ]  # End of sql_commands list

            # Execute commands sequentially
            logger.info(f"Found {len(sql_commands)} SQL commands for schema creation.")
            for i, command in enumerate(sql_commands):
                # Clean up potential leading/trailing whitespace from multi-line strings
                cleaned_command = command.strip()
                if not cleaned_command:  # Skip empty strings if any accidentally got in
                    logger.warning(f"Skipping empty command at index {i}.")
                    continue
                try:
                    logger.debug(f"Executing schema command {i+1}/{len(sql_commands)}...")
                    await conn.execute(cleaned_command)
                except asyncpg.PostgresError as e:
                    logger.error(f"Error executing command {i+1}: {cleaned_command[:100]}... \nError: {e}", exc_info=True)
                    # Log and continue is generally better for IF NOT EXISTS
                    logger.warning(f"Continuing schema creation despite error on command {i+1}.")
                except Exception as e_generic:  # Catch other potential errors like syntax issues
                    logger.error(f"Non-DB error executing command {i+1}: {cleaned_command[:100]}... \nError: {e_generic}", exc_info=True)
                    logger.warning(f"Continuing schema creation despite non-DB error on command {i+1}.")

            logger.info("Finished executing schema commands loop.")

        # This log message indicates the 'async with' block completed
        logger.info("Schema creation process completed (connection released).")

    # Catch errors related to connection acquisition or timeout
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as e:
        logger.critical(f"Failed to acquire connection or timed out during table creation: {e}", exc_info=True)
        raise  # Re-raise the exception to indicate failure
    except Exception as e:
        logger.critical(f"An unexpected error occurred outside the connection block during table creation: {e}", exc_info=True)
        raise

async def seed_initial_data():
    """
    Asynchronously seeds default data for rules, stats, settings, etc.
    Ensures the DB has the minimal data set. Assumes called functions are async.
    """
    logger.info("Starting initial data seeding...")
    from routes.settings_routes import insert_missing_settings
    try:
        # Assuming these functions are defined elsewhere and are async
        await insert_or_update_game_rules()
        await insert_stat_definitions()
        await insert_missing_settings(is_initial_setup=True)
        await insert_missing_activities()
        await insert_missing_archetypes()
        await create_and_seed_intensity_tiers()
        await create_and_seed_plot_triggers()
        await create_and_seed_interactions()
        logger.info("All default data seeding tasks completed.")
    except Exception as e:
        logger.error(f"Error during initial data seeding: {e}", exc_info=True)
        raise  # Re-raise so initialize_all_data knows seeding failed

async def seed_initial_resources():
    """Asynchronously seed initial player resources using asyncpg."""
    logger.info("Seeding initial player resources...")
    rows_affected = 0
    try:
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                player_rows = await conn.fetch("""
                    SELECT DISTINCT user_id, conversation_id, player_name FROM PlayerStats
                """)

                if not player_rows:
                    logger.info("No players found requiring initial resource seeding.")
                    return

                # Insert for each player directly without prepared statement
                count = 0
                for row in player_rows:
                    result = await conn.execute("""
                        INSERT INTO PlayerResources (user_id, conversation_id, player_name, money, supplies, influence)
                        VALUES ($1, $2, $3, 100, 20, 10)
                        ON CONFLICT (user_id, conversation_id, player_name) DO NOTHING
                    """, row['user_id'], row['conversation_id'], row['player_name'])
                    
                    # Check if rows were affected
                    if result.endswith(" 1"):
                        count += 1
                rows_affected = count

        logger.info(f"Seeded initial resources for {rows_affected} players (others may have existed).")

    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error seeding initial resources: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error seeding resources: {e}", exc_info=True)

async def initialize_all_data():
    """
    Asynchronous convenience function to create tables + seed initial data.
    """
    logger.info("Starting full database initialization...")
    try:
        await create_all_tables()
        await seed_initial_data()
        # These seeding steps might depend on data created by seed_initial_data
        # or tables created by create_all_tables, so order matters.
        await seed_initial_vitals()
        await seed_initial_resources()
        # Conflict system init might have dependencies too
        await initialize_conflict_system()
        logger.info("All initialization steps completed successfully!")
    except Exception as e:
        # Catch error from any awaited step above
        logger.critical(f"Full database initialization failed during one of the steps: {e}", exc_info=True)
        raise  # Re-raise to signal failure to the caller (main)

async def main():
    """Main async function to initialize pool and run setup."""
    # Setup logging BEFORE doing anything else
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    logger.info("Application starting...")

    pool_initialized = await initialize_connection_pool()

    if not pool_initialized:
        logger.critical("Database pool could not be initialized. Exiting.")
        return  # Exit if pool fails

    try:
        # Run the full initialization
        await initialize_all_data()
        await seed_nyx_memories_from_prompt(SYSTEM_PROMPT, PRIVATE_REFLECTION_INSTRUCTIONS)
        logger.info("Application initialization successful.")
        
    except Exception as e:
        # initialize_all_data now re-raises, so main's try/except catches it
        logger.error(f"An error occurred during application initialization in main: {e}", exc_info=True)
    finally:
        # Ensure pool is closed on exit or error
        logger.info("Closing database connection pool...")
        await close_connection_pool()
        logger.info("Database connection pool closed.")

if __name__ == "__main__":
    # Load environment variables if needed (e.g., using python-dotenv)
    # from dotenv import load_dotenv
    # load_dotenv()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Use the logger configured in main()
        logger.info("Application interrupted by user. Shutting down.")
    except Exception as e:
        # Catch any unexpected errors during asyncio.run() itself
        print(f"CRITICAL: Unhandled exception during asyncio.run: {e}")
