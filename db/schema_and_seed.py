import json
from db.connection import get_db_connection

# If you have these modules:
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
    so we don't need separate ALTER TABLE statements later.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Settings (Global or not? If truly global, omit user_id/conversation_id here)
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

    # 2) StatDefinitions (global, no user/conversation)
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

    # 3) GameRules (global as well)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS GameRules (
          id SERIAL PRIMARY KEY,
          rule_name TEXT UNIQUE NOT NULL,
          condition TEXT NOT NULL,
          effect TEXT NOT NULL
        );
    ''')

    # 4) users (so we can reference user_id below)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
          id SERIAL PRIMARY KEY,
          username VARCHAR(50) NOT NULL UNIQUE,
          password_hash TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    # 5) conversations (belongs to a user)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
          id SERIAL PRIMARY KEY,
          user_id INTEGER NOT NULL,
          conversation_name VARCHAR(100) NOT NULL,
          status VARCHAR(20) NOT NULL DEFAULT 'processing',
          created_at TIMESTAMP DEFAULT NOW(),
          FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    ''')

    # 6) CurrentRoleplay (with user_id, conversation_id so each user+conversation can store distinct keys)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CurrentRoleplay (
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (user_id, conversation_id, key),
            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
        );
    ''')

    # 7) messages (belongs to a conversation)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
          id SERIAL PRIMARY KEY,
          conversation_id INTEGER NOT NULL,
          sender VARCHAR(50) NOT NULL,
          content TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT NOW(),
          structured_content JSONB,
          FOREIGN KEY (conversation_id)
              REFERENCES conversations(id)
              ON DELETE CASCADE
        );
    ''')

    # 8) folders (optional)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS folders (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            folder_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    ''')

    # Add folder_id to conversations
    cursor.execute('''
        ALTER TABLE conversations
        ADD COLUMN IF NOT EXISTS folder_id INTEGER;
    ''')
    # Add a foreign key for folder_id
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE conversations DROP CONSTRAINT IF EXISTS fk_conversations_folder;
        EXCEPTION WHEN undefined_object THEN
        END;
        $$;
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE conversations
            ADD CONSTRAINT fk_conversations_folder
            FOREIGN KEY (folder_id)
            REFERENCES folders(id)
            ON DELETE CASCADE;
        EXCEPTION WHEN duplicate_object THEN
        END;
        $$;
    ''')

    # ----------------------------------------------------------------
    # Now define the per-user, per-conversation tables
    # ----------------------------------------------------------------

    # 9) NPCStats
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
            physical_description TEXT,       -- NEW: robust physical description
            relationships JSONB,
            dominance INT CHECK (dominance BETWEEN 0 AND 100),
            cruelty INT CHECK (cruelty BETWEEN 0 AND 100),
            closeness INT CHECK (closeness BETWEEN 0 AND 100),
            trust INT CHECK (trust BETWEEN -100 AND 100),
            respect INT CHECK (respect BETWEEN -100 AND 100),
            intensity INT CHECK (intensity BETWEEN 0 AND 100),
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
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    # 10) PlayerStats
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

    # 11) Archetypes (global, no user/conversation)
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

    # 12) Activities (global, no user/conversation)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Activities (
          id SERIAL PRIMARY KEY,
          name TEXT UNIQUE NOT NULL,
          purpose JSONB NOT NULL,
          stat_integration JSONB,
          intensity_tiers JSONB,
          setting_variants JSONB
        );
    ''')

    # 13) PlayerInventory
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

    # 14) Locations
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

    # 15) Events
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
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    # 16) Quests
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

    # 17) PlannedEvents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlannedEvents (
          event_id SERIAL PRIMARY KEY,
          user_id INTEGER NOT NULL,
          conversation_id INTEGER NOT NULL,
          npc_id INT REFERENCES NPCStats(npc_id),
          day INT NOT NULL,
          time_of_day TEXT NOT NULL,
          override_location TEXT NOT NULL,
          UNIQUE(npc_id, day, time_of_day),
          FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
          FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    # 18) SocialLinks
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
            UNIQUE (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    ''')

    # 19) PlayerPerks
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
            permanent_effects JSONB         
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Interactions (
            id SERIAL PRIMARY KEY,
            interaction_name TEXT UNIQUE NOT NULL,   
            detailed_rules JSONB NOT NULL,          
            task_examples JSONB,                    
            agency_overrides JSONB                  
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

    conn.commit()
    conn.close()

def seed_initial_data():
    """
    Inserts default data (stat definitions, game rules, settings, activities, archetypes, etc.).
    """
    insert_or_update_game_rules()
    insert_stat_definitions()
    insert_missing_settings() 
    insert_missing_activities()
    insert_missing_archetypes()
    create_and_seed_intensity_tiers()
    create_and_seed_plot_triggers()     
    create_and_seed_interactions()
    print("All default data seeded successfully.")

def initialize_all_data():
    create_all_tables()
    seed_initial_data()
    print("All tables created & default data seeded successfully!")
