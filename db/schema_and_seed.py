# db/schema_and_seed.py

import json
import logging
import re
from typing import List
# Import connection and pool management functions
from db.connection import (
    get_db_connection_context,
    initialize_connection_pool, # Added import
    close_connection_pool # Added import
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
    # insert_default_player_stats_chase # This might be synchronous or not needed if seeding covers it? Verify its nature.
)
import asyncpg
import asyncio # Added import for asyncio.run
from typing import Dict, Any

# Configure logger for this module
logger = logging.getLogger(__name__) # Define logger at module level

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
                # ---------- GLOBAL / CORE TABLES ----------
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
                CREATE TABLE IF NOT EXISTS nyx_brain_checkpoints (
                    id BIGSERIAL PRIMARY KEY,
                    nyx_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    checkpoint_time TIMESTAMPTZ NOT NULL DEFAULT now(),
                    event TEXT,
                    serialized_state JSONB NOT NULL,
                    merged_from TEXT[],
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
                    event_time TIMESTAMPTZ NOT NULL DEFAULT now(),
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
                CREATE TABLE IF NOT EXISTS GameRules (
                    id SERIAL PRIMARY KEY,
                    rule_name TEXT UNIQUE NOT NULL,
                    condition TEXT NOT NULL,
                    effect TEXT NOT NULL
                );
                ''',
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
                # ---------- PER-USER / CONVERSATION TABLES ----------
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
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
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
                CREATE TABLE IF NOT EXISTS PlayerInventory (
                    item_id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL,
                    conversation_id INT NOT NULL,
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
                CREATE TABLE IF NOT EXISTS Factions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT DEFAULT 'community' CHECK (type IN ('political', 'community', 'social', 'hobby', 'educational', 'professional', 'religious', 'criminal')),
                    description TEXT,
                    values TEXT[],
                    goals TEXT[],
                    hierarchy TEXT DEFAULT 'informal', 
                    resources TEXT[],
                    territory TEXT,  
                    meeting_schedule TEXT,  
                    membership_requirements TEXT[],
                    rivals INTEGER[],  
                    allies INTEGER[],
                    public_reputation TEXT DEFAULT 'neutral',
                    secret_activities TEXT[],
                    power_level INTEGER DEFAULT 3 CHECK (power_level BETWEEN 1 AND 10),
                    influence_scope TEXT DEFAULT 'local' CHECK (influence_scope IN ('personal', 'local', 'regional', 'national', 'global')),
                    recruitment_methods TEXT[],
                    leadership_structure JSONB,
                    founding_story TEXT,
                    embedding VECTOR(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    UNIQUE(user_id, conversation_id, name)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_factions_name
                ON Factions(user_id, conversation_id, lower(name));
                );
                ''',
                # ---------- RESOURCE HISTORY ----------
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
                # ---------- CONFLICT SYSTEM TABLES ----------
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
                # ---------- INTERNAL FACTION CONFLICTS ----------
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
                CREATE TABLE IF NOT EXISTS nations (
                  id SERIAL PRIMARY KEY,
                  name TEXT NOT NULL UNIQUE,
                  region_type TEXT,
                  climate TEXT,
                  resources JSONB,
                  strategic_value INT,
                  matriarchal_influence FLOAT DEFAULT 0.0,
                  embedding VECTOR(1536)
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_nations_name
                ON nations (lower(name)
                );
                ''',
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
                # ---------- CONFLICT MEMORY EVENTS ----------
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
                CREATE TABLE IF NOT EXISTS FactionPowerShifts (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    faction_name VARCHAR(255) NOT NULL,
                    power_level INTEGER NOT NULL,
                    change_amount INTEGER NOT NULL,
                    cause TEXT NOT NULL,
                    conflict_id INTEGER REFERENCES Conflicts(conflict_id),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                # ---------- CONFLICT SYSTEM INDEXES ----------
                '''
                CREATE INDEX IF NOT EXISTS idx_conflicts_user_conv
                ON Conflicts(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_conflict_npcs
                ON ConflictNPCs(conflict_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_active_conflicts
                ON Conflicts(is_active) WHERE is_active = TRUE;
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_conflict_history
                ON ConflictHistory(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_grudge_level
                ON ConflictHistory(grudge_level) WHERE grudge_level > 50;
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_faction_power_shifts
                ON FactionPowerShifts(user_id, conversation_id);
                ''',
                # ---------- LEGACY "NPCMemories" TABLE (OPTIONAL) ----------
                '''
                CREATE TABLE IF NOT EXISTS NPCMemories (
                    id SERIAL PRIMARY KEY,
                    npc_id INT NOT NULL,
                    memory_text TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT[],
                    emotional_intensity INT DEFAULT 0,
                    times_recalled INT DEFAULT 0,
                    last_recalled TIMESTAMP,
                    embedding VECTOR(1536),
                    memory_type TEXT DEFAULT 'observation',
                    associated_entities JSONB DEFAULT '{}'::jsonb,
                    is_consolidated BOOLEAN NOT NULL DEFAULT FALSE,
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
                # ---------- NEW "unified_memories" TABLE ----------
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
                    tags TEXT[] DEFAULT '{}',
                    embedding VECTOR(1536),
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    times_recalled INTEGER NOT NULL DEFAULT 0,
                    last_recalled TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'active',
                    is_consolidated BOOLEAN NOT NULL DEFAULT FALSE,
                    relevance_score FLOAT DEFAULT 0.0, -- Added previously
                    last_context_update TIMESTAMP      -- Added previously
                );
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_unified_memories_entity
                ON unified_memories(entity_type, entity_id);
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
                # ---------- MORE TABLES ----------
                '''
                CREATE TABLE IF NOT EXISTS Locations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    location_name TEXT NOT NULL,
                    description TEXT,
                    open_hours JSONB,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
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
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS Quests (
                    quest_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    quest_name TEXT,
                    status TEXT NOT NULL DEFAULT 'In Progress',
                    progress_detail TEXT,
                    quest_giver TEXT,
                    reward TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
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
                # ---------- ENHANCED SYSTEMS ----------
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
                '''
                CREATE TABLE IF NOT EXISTS CanonicalEvents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    event_text TEXT NOT NULL,
                    tags TEXT[] DEFAULT '{}',
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
                # ---------- TELEMETRY TABLE ----------
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
                # ---------- KINK DATA ----------
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
                # ---------- IMAGE GENERATION TABLES ----------
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
                # ---------- ADDITIONAL NPCMemory ASSOCIATIONS ----------
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
                # ---------- "NyxMemories" LEGACY TABLE (OPTIONAL) ----------
                '''
                CREATE TABLE IF NOT EXISTS NyxMemories (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(1536),
                    significance INT NOT NULL DEFAULT 3,
                    times_recalled INT NOT NULL DEFAULT 0,
                    last_recalled TIMESTAMP,
                    memory_type TEXT DEFAULT 'reflection',
                    is_archived BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                ''',
                # ---------- "NyxAgentState" FOR DM LOGIC ----------
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
                    tags JSONB DEFAULT '[]'::jsonb,
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
                    tags TEXT[],
                    created_at TIMESTAMP DEFAULT now()
                );
                ''',
                '''
                CREATE TABLE IF NOT EXISTS nyx1_strategy_logs (
                    id SERIAL PRIMARY KEY,
                    strategy_id INTEGER REFERENCES nyx1_strategy_injections(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL,
                    event_type TEXT, -- "triggered", "dismissed", "archived"
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
                "CREATE INDEX IF NOT EXISTS idx_player_inventory_user ON PlayerInventory(user_id, conversation_id);",
                "CREATE INDEX IF NOT EXISTS idx_player_perks_user ON PlayerPerks(user_id, conversation_id);",
                "CREATE INDEX IF NOT EXISTS idx_player_special_rewards_user ON PlayerSpecialRewards(user_id, conversation_id);",
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
                CREATE EXTENSION IF NOT EXISTS vector;
                ''',
                '''
                ALTER TABLE Events
                ADD COLUMN embedding vector(1536);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_events_embedding_hnsw
                ON Events USING hnsw (embedding vector_cosine_ops);
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
                # Alter Memory table to add missing columns if needed
                '''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'unified_memories'
                        AND column_name = 'relevance_score'
                    ) THEN
                        ALTER TABLE unified_memories ADD COLUMN relevance_score FLOAT DEFAULT 0.0;
                    END IF;
                END $$;
                ''',
                '''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'unified_memories'
                        AND column_name = 'last_context_update'
                    ) THEN
                        ALTER TABLE unified_memories ADD COLUMN last_context_update TIMESTAMP;
                    END IF;
                END $$;
                ''',
                # Corrected final block of CREATE INDEX statements
                '''
                CREATE INDEX IF NOT EXISTS idx_context_evolution_user_conversation
                ON ContextEvolution(user_id, conversation_id);
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_context_evolution_timestamp
                ON ContextEvolution(timestamp);
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
                '''
            ] # End of sql_commands list

            # Execute commands sequentially
            logger.info(f"Found {len(sql_commands)} SQL commands for schema creation.")
            for i, command in enumerate(sql_commands):
                # Clean up potential leading/trailing whitespace from multi-line strings
                cleaned_command = command.strip()
                if not cleaned_command: # Skip empty strings if any accidentally got in
                    logger.warning(f"Skipping empty command at index {i}.")
                    continue
                try:
                    logger.debug(f"Executing schema command {i+1}/{len(sql_commands)}...")
                    # Optional: Log part of the command for easier debugging
                    # logger.debug(f"SQL: {cleaned_command[:150]}...")
                    await conn.execute(cleaned_command)
                except asyncpg.PostgresError as e:
                    logger.error(f"Error executing command {i+1}: {cleaned_command[:100]}... \nError: {e}", exc_info=True)
                    # Option 2: Log and continue is generally better for IF NOT EXISTS
                    logger.warning(f"Continuing schema creation despite error on command {i+1}.")
                except Exception as e_generic: # Catch other potential errors like syntax issues
                    logger.error(f"Non-DB error executing command {i+1}: {cleaned_command[:100]}... \nError: {e_generic}", exc_info=True)
                    logger.warning(f"Continuing schema creation despite non-DB error on command {i+1}.")

            logger.info("Finished executing schema commands loop.")

        # This log message indicates the 'async with' block completed
        logger.info("Schema creation process completed (connection released).")

    # Catch errors related to connection acquisition or timeout
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as e:
        logger.critical(f"Failed to acquire connection or timed out during table creation: {e}", exc_info=True)
        raise # Re-raise the exception to indicate failure
    except Exception as e:
        logger.critical(f"An unexpected error occurred outside the connection block during table creation: {e}", exc_info=True)
        raise

# --- Seeding and Initialization Functions (Look correct from previous review) ---

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
    await seed_initial_vitals() # Call the async version
    logger.info("Conflict system initialized (vitals seeded).")


async def seed_initial_data():
    """
    Asynchronously seeds default data for rules, stats, settings, etc.
    Ensures the DB has the minimal data set. Assumes called functions are async.
    """
    logger.info("Starting initial data seeding...")
    from routes.settings_routes import insert_missing_settings
    try:
        # Assuming these functions are defined elsewhere and are async
        # Make sure insert_or_update_game_rules etc. are actually async!
        await insert_or_update_game_rules()
        await insert_stat_definitions()
        await insert_missing_settings(is_initial_setup=True)
        await insert_missing_activities()
        await insert_missing_archetypes()
        await create_and_seed_intensity_tiers()
        await create_and_seed_plot_triggers()
        await create_and_seed_interactions()
        # Note: insert_default_player_stats_chase was commented out previously,
        # ensure its purpose and async nature if re-enabled.
        # await insert_default_player_stats_chase()
        logger.info("All default data seeding tasks completed.")
    except Exception as e:
        logger.error(f"Error during initial data seeding: {e}", exc_info=True)
        raise # Re-raise so initialize_all_data knows seeding failed


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
        logger.info("All initialization steps completed successfully!") # Changed log message
    except Exception as e:
        # Catch error from any awaited step above
        logger.critical(f"Full database initialization failed during one of the steps: {e}", exc_info=True)
        raise # Re-raise to signal failure to the caller (main)


async def main():
    """Main async function to initialize pool and run setup."""
    # Setup logging BEFORE doing anything else
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s' # Added logger name
    )
    logger.info("Application starting...") # Use the module logger

    pool_initialized = await initialize_connection_pool()

    if not pool_initialized:
        logger.critical("Database pool could not be initialized. Exiting.")
        return # Exit if pool fails

    try:
        # Run the full initialization
        await initialize_all_data()
        await seed_nyx_memories_from_prompt(SYSTEM_PROMPT, PRIVATE_REFLECTION_INSTRUCTIONS)
        logger.info("Application initialization successful.") # Success message if all went well
        
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
        # Logger might not be fully configured if error happens very early
        print(f"CRITICAL: Unhandled exception during asyncio.run: {e}")
        # Optionally re-log if logger might be available
        # logger.critical(f"Critical error during application startup or shutdown: {e}", exc_info=True)
