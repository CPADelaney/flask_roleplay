import json
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
    Creates all core tables with user_id, conversation_id columns from the start,
    adjusted for fantastical punishments and image generation.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # StateUpdates
    cursor.execute('''    
        CREATE TABLE IF NOT EXISTS StateUpdates (
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            update_payload JSONB,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, conversation_id)
        );
    ''')

    # Global Tables (unchanged except where noted)
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

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS folders (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            folder_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    ''')

    # Per-User/Conversation Tables
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
            fantasy_level TEXT DEFAULT 'realistic' CHECK (fantasy_level IN ('realistic', 'fantastical', 'surreal'))  -- Added for surreal punishments
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
            group_interaction TEXT,  -- Added for group dynamics in punishments
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
            fantasy_level TEXT DEFAULT 'realistic' CHECK (fantasy_level IN ('realistic', 'fantastical', 'surreal'))  -- Added for surreal punishments
        );
    ''')

    # Removed incomplete ImageGenerations in favor of full image tables below

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Interactions (
            id SERIAL PRIMARY KEY,
            interaction_name TEXT UNIQUE NOT NULL,
            detailed_rules JSONB NOT NULL,
            task_examples JSONB,
            agency_overrides JSONB,
            fantasy_level TEXT DEFAULT 'realistic' CHECK (fantasy_level IN ('realistic', 'fantastical', 'surreal'))  -- Added for surreal interactions
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

    # Enhanced Systems
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlayerJournal (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            entry_type TEXT NOT NULL,
            entry_text TEXT NOT NULL,
            revelation_types TEXT,
            narrative_moment TEXT,
            fantasy_flag BOOLEAN DEFAULT FALSE,  -- Flags surreal events
            intensity_level INT CHECK (intensity_level BETWEEN 0 AND 4),  -- Ties to IntensityTiers
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS UserKinkProfile (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            kink_type TEXT NOT NULL,           -- e.g., "ass", "feet", "shrink_ray"
            level INTEGER CHECK (level BETWEEN 0 AND 4) DEFAULT 0,
            discovery_source TEXT,             -- e.g., "user_input", "narrative_response", "action_analysis"
            first_discovered TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            frequency INTEGER DEFAULT 0,       -- Times used across games
            intensity_preference INTEGER CHECK (intensity_preference BETWEEN 0 AND 4) DEFAULT 0,
            trigger_context JSONB,             -- e.g., {"location": "arcade", "npc": "Lila", "action": "staring"}
            confidence_score FLOAT DEFAULT 0.5, -- 0-1, how sure Nyx is of this kink (updates with evidence)
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
            tease_text TEXT NOT NULL,          -- e.g., "Face near her ass again, huh?"
            tease_type TEXT CHECK (tease_type IN ('narrative', 'meta_commentary', 'punishment')),
            narrative_context TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
            FOREIGN KEY (kink_id) REFERENCES UserKinkProfile(id) ON DELETE CASCADE
        );
    ''')

    # Image Generation Tables (from ai_image_generator.py)
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
            event_type TEXT CHECK (event_type IN ('outfit_change', 'appearance_change', 'location_change', 'mood_change')),
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

    conn.commit()
    conn.close()

def seed_initial_data():
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
    create_all_tables()
    seed_initial_data()
    print("All tables created & default data seeded successfully!")
