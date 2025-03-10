# db/schema_and_seed.py

import json
import logging
from db.connection import get_db_connection
from routes.activities import insert_missing_activities
from routes.archetypes import insert_missing_archetypes
from routes.settings_routes import insert_missing_settings
from logic.seed_intensity_tiers import create_and_seed_intensity_tiers
from logic.seed_plot_triggers import create_and_seed_plot_triggers
from logic.seed_interactions import create_and_seed_interactions
from logic.stats_logic import (
    insert_stat_definitions,
    insert_or_update_game_rules,
    insert_default_player_stats_chase
)


def create_all_tables():
    """
    Creates all your database tables in one go. 
    Includes both old memory tables (NPCMemories, NyxMemories) 
    and the new 'unified_memories' plus 'memory_telemetry'.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Ensure vector extension is available before any tables use vector columns
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2) Create or ensure existence of all tables:

    #
    # ---------- GLOBAL / CORE TABLES ----------
    #
    cursor.execute('''    
        CREATE TABLE IF NOT EXISTS StateUpdates (
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            update_payload JSONB,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, conversation_id)
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Settings (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            mood_tone TEXT NOT NULL,
            enhanced_features JSONB NOT NULL,
            stat_modifiers JSONB NOT NULL,
            activity_examples JSONB NOT NULL
        );
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS GameRules (
            id SERIAL PRIMARY KEY,
            rule_name TEXT UNIQUE NOT NULL,
            condition TEXT NOT NULL,
            effect TEXT NOT NULL
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS folders (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            folder_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CurrentRoleplay (
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (user_id, conversation_id, key),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER NOT NULL,
            sender VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            structured_content JSONB,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    #
    # ---------- PER-USER / CONVERSATION TABLES ----------
    #
    cursor.execute('''
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
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
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
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Archetypes (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            baseline_stats JSONB NOT NULL,
            progression_rules JSONB NOT NULL,
            setting_examples JSONB NOT NULL,
            unique_traits JSONB NOT NULL
        );
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlayerInventory (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            item_name TEXT NOT NULL,
            item_description TEXT,
            item_effect TEXT,
            quantity INT DEFAULT 1,
            category TEXT,
            UNIQUE (user_id, conversation_id, player_name, item_name),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    #
    # ---------- LEGACY “NPCMemories” TABLE (OPTIONAL) ----------
    #
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS NPCMemories (
            id SERIAL PRIMARY KEY,
            npc_id INT NOT NULL,
            memory_text TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

            tags TEXT[],
            emotional_intensity INT DEFAULT 0,   -- 0..100
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
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_mem_npcid_status_ts
            ON NPCMemories (npc_id, status, timestamp);
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS npc_memory_embedding_hnsw_idx
            ON NPCMemories 
            USING hnsw (embedding vector_cosine_ops);
    ''')

    #
    # ---------- NEW “unified_memories” TABLE ----------
    #
    cursor.execute('''
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
            is_consolidated BOOLEAN NOT NULL DEFAULT FALSE
        );
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_unified_memories_entity
            ON unified_memories(entity_type, entity_id);
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_unified_memories_user_conv
            ON unified_memories(user_id, conversation_id);
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_unified_memories_timestamp
            ON unified_memories(timestamp);
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_unified_memories_embedding_hnsw
            ON unified_memories
            USING hnsw (embedding vector_cosine_ops);
    ''')

    #
    # ---------- MORE TABLES ----------
    #
    cursor.execute('''
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
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlayerPerks (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            perk_name TEXT NOT NULL,
            perk_description TEXT,
            perk_effect TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Interactions (
            id SERIAL PRIMARY KEY,
            interaction_name TEXT UNIQUE NOT NULL,
            detailed_rules JSONB NOT NULL,
            task_examples JSONB,
            agency_overrides JSONB,
            fantasy_level TEXT DEFAULT 'realistic'
                CHECK (fantasy_level IN ('realistic','fantastical','surreal'))
        );
    ''')

    cursor.execute('''
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
    ''')

    #
    # ---------- ENHANCED SYSTEMS ----------
    #
    cursor.execute('''
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
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
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
    ''')

    #
    # ---------- TELEMETRY TABLE ----------
    #
    cursor.execute('''
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
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_memory_telemetry_timestamp
            ON memory_telemetry(timestamp);
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_memory_telemetry_operation
            ON memory_telemetry(operation);
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_memory_telemetry_success
            ON memory_telemetry(success);
    ''')

    #
    # ---------- KINK DATA ----------
    #
    cursor.execute('''
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
    ''')

    cursor.execute('''
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
    ''')

    #
    # ---------- IMAGE GENERATION TABLES ----------
    #
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS NPCVisualAttributes (
            id SERIAL PRIMARY KEY,
            npc_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            conversation_id TEXT NOT NULL,
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
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ImageFeedback (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            conversation_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            original_prompt TEXT NOT NULL,
            npc_names JSONB NOT NULL,
            rating INTEGER CHECK (rating BETWEEN 1 AND 5),
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS NPCVisualEvolution (
            id SERIAL PRIMARY KEY,
            npc_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            conversation_id TEXT NOT NULL,
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
    ''')

    cursor.execute('''
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
            user_id, npc_name
    ''')

    #
    # ---------- (OPTIONAL) Additional NPCMemory Associations ----------
    #
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS NPCMemoryAssociations (
            id SERIAL PRIMARY KEY,
            memory_id INT NOT NULL,
            associated_memory_id INT NOT NULL,
            association_strength FLOAT DEFAULT 0.0,
            association_type TEXT,
            FOREIGN KEY (memory_id) REFERENCES NPCMemories(id) ON DELETE CASCADE,
            FOREIGN KEY (associated_memory_id) REFERENCES NPCMemories(id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
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
    ''')

    #
    # ---------- “NyxMemories” LEGACY TABLE (OPTIONAL) ----------
    #
    cursor.execute('''
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
    ''')

    #
    # ---------- “NyxAgentState” FOR DM LOGIC ----------
    #
    cursor.execute('''
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
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlayerReputation (
            user_id INT,
            npc_id INT,
            reputation_score FLOAT DEFAULT 0,
            PRIMARY KEY (user_id, npc_id)
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ReflectionLogs (
            id SERIAL PRIMARY KEY,
            user_id INT NOT NULL,
            reflection_text TEXT NOT NULL,
            was_accurate BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS AIReflectionSettings (
            id SERIAL PRIMARY KEY,
            temperature FLOAT DEFAULT 0.7,
            max_tokens INT DEFAULT 4000
        );
    ''')

    cursor.execute('''
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
    ''')

    cursor.execute('''
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
    ''')

    # Done creating everything:
    conn.commit()
    conn.close()
    logging.info("All tables created (old + new).")


def seed_initial_data():
    """
    Seeds default data for rules, stats, settings, etc.
    If you already have references to them in logic, 
    it ensures your DB has the minimal data set.
    """
    insert_or_update_game_rules()
    insert_stat_definitions()
    insert_missing_settings()
    insert_missing_activities()
    insert_missing_archetypes()
    create_and_seed_intensity_tiers()
    create_and_seed_plot_triggers()
    create_and_seed_interactions()
    insert_default_player_stats_chase()
    print("All default data seeded successfully.")


def initialize_all_data():
    """
    Convenience function to create tables + seed initial data 
    in one call.
    """
    create_all_tables()
    seed_initial_data()
    print("All tables created & default data seeded successfully!")
