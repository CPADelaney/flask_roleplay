import json
from db.connection import get_db_connection

# Import any "insert_missing_..." or "insert_stat_definitions" methods used for seeding
from routes.activities import insert_missing_activities
from routes.archetypes import insert_missing_archetypes
from routes.settings_routes import insert_missing_settings
from logic.stats_logic import insert_stat_definitions, insert_or_update_game_rules, insert_default_player_stats_chase


def create_all_tables():
    """
    Creates all the core tables if they do not exist yet.
    (Merged from db/initialization.py)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Settings
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

    # 2) CurrentRoleplay
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CurrentRoleplay (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    ''')

    # 3) StatDefinitions
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

    # 4) GameRules
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS GameRules (
          id SERIAL PRIMARY KEY,
          rule_name TEXT UNIQUE NOT NULL,
          condition TEXT NOT NULL,
          effect TEXT NOT NULL
        );
    ''')

    # 5) NPCStats
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS NPCStats (
            npc_id SERIAL PRIMARY KEY,
            npc_name TEXT NOT NULL,
            introduced BOOLEAN default FALSE,

            archetypes JSONB,

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
            current_location TEXT
        );
    ''')

    # 6) PlayerStats
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlayerStats (
           id SERIAL PRIMARY KEY,
           player_name TEXT NOT NULL,
           corruption INT CHECK (corruption BETWEEN 0 AND 100),
           confidence INT CHECK (confidence BETWEEN 0 AND 100),
           willpower INT CHECK (willpower BETWEEN 0 AND 100),
           obedience INT CHECK (obedience BETWEEN 0 AND 100),
           dependency INT CHECK (dependency BETWEEN 0 AND 100),
           lust INT CHECK (lust BETWEEN 0 AND 100),
           mental_resilience INT CHECK (mental_resilience BETWEEN 0 AND 100),
           physical_endurance INT CHECK (physical_endurance BETWEEN 0 AND 100)
        );
    ''')

    # 7) Archetypes
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

    # 8) Activities
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

    # 9) PlayerInventory
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlayerInventory (
            id SERIAL PRIMARY KEY,
            player_name TEXT NOT NULL,
            item_name TEXT NOT NULL,
            item_description TEXT,
            item_effect TEXT,
            quantity INT DEFAULT 1,
            category TEXT,
            UNIQUE (player_name, item_name)
        );
    ''')

    # 10) Locations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Locations (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            open_hours JSONB
        );
    ''')

    # 11) Events
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Events (
            id SERIAL PRIMARY KEY,
            event_name TEXT NOT NULL,
            description TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            location TEXT NOT NULL
        );
    ''')

    # 12) Quests
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Quests (
            quest_id SERIAL PRIMARY KEY,
            quest_name TEXT,
            status TEXT NOT NULL DEFAULT 'In Progress',
            progress_detail TEXT,
            quest_giver TEXT,
            reward TEXT
        );
    ''')

    # 13) PlannedEvents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlannedEvents (
          event_id SERIAL PRIMARY KEY,
          npc_id INT REFERENCES NPCStats(npc_id),
          day INT NOT NULL,          -- Day number or a 'Mon=1, Tue=2, etc.'
          time_of_day TEXT NOT NULL, -- "Morning", "Afternoon", etc.
          override_location TEXT NOT NULL,
          UNIQUE(npc_id, day, time_of_day)
        );
    ''')

    # 14) SocialLinks
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS SocialLinks (
            link_id SERIAL PRIMARY KEY,
            entity1_type TEXT NOT NULL,
            entity1_id INT NOT NULL,
            entity2_type TEXT NOT NULL,
            entity2_id INT NOT NULL,
    
            link_type TEXT,
            link_level INT DEFAULT 0,
            link_history JSONB,
            UNIQUE (entity1_type, entity1_id, entity2_type, entity2_id)
        );
    ''')

    # 15) PlayerPerks
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlayerPerks (
            id SERIAL PRIMARY KEY,
            player_name TEXT NOT NULL,
            perk_name TEXT NOT NULL,
            perk_description TEXT,
            perk_effect TEXT,
            UNIQUE (player_name, perk_name)
        );
    ''')

    # 16) Multiple User Support
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
          id SERIAL PRIMARY KEY,
          username VARCHAR(50) NOT NULL UNIQUE,
          password_hash TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    # conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
          id SERIAL PRIMARY KEY,
          user_id INTEGER NOT NULL REFERENCES users(id),
          conversation_name VARCHAR(100) NOT NULL,
          created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    # messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
          id SERIAL PRIMARY KEY,
          conversation_id INTEGER NOT NULL REFERENCES conversations(id),
          sender VARCHAR(50) NOT NULL,
          content TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    # Drop + re-add the foreign key on messages for ON DELETE CASCADE
    cursor.execute('''
        ALTER TABLE messages
        DROP CONSTRAINT IF EXISTS messages_conversation_id_fkey
    ''')
    cursor.execute('''
        ALTER TABLE messages
        ADD CONSTRAINT IF NOT EXISTS messages_conversation_id_fkey
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # Create the folders table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS folders (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            folder_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    # Add a folder_id column to conversations, referencing folders.id
    cursor.execute('''
        ALTER TABLE conversations
        ADD COLUMN IF NOT EXISTS folder_id INTEGER
    ''')

    # If the foreign key constraint might already exist, we can drop it first
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE conversations
            DROP CONSTRAINT IF EXISTS fk_conversations_folder;
        EXCEPTION WHEN undefined_object THEN
            -- no constraint yet
        END;
        $$
    ''')

    # Now re-add the constraint with ON DELETE CASCADE or SET NULL
    # We'll do CASCADE here, as requested
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE conversations
            ADD CONSTRAINT fk_conversations_folder
            FOREIGN KEY (folder_id)
            REFERENCES folders(id)
            ON DELETE CASCADE;
        EXCEPTION WHEN duplicate_object THEN
            -- constraint already there
        END;
        $$
    ''')

    conn.commit()
    conn.close()

def seed_initial_data():
    """
    Inserts default data (stat definitions, game rules, settings, activities, archetypes, etc.).
    (Merged from logic/initialization.py)
    """
    insert_or_update_game_rules()
    insert_stat_definitions()
    insert_missing_settings()
    insert_missing_activities()
    insert_missing_archetypes()
    insert_default_player_stats_chase()

    print("All default data seeded successfully.")


def initialize_all_data():
    create_all_tables()
    seed_initial_data()
    print("All tables created & default data seeded successfully!")
