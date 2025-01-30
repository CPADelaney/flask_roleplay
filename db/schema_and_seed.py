import json
from db.connection import get_db_connection

# If you have these modules:
from routes.activities import insert_missing_activities
from routes.archetypes import insert_missing_archetypes
from routes.settings_routes import insert_missing_settings
from logic.stats_logic import (
    insert_stat_definitions,
    insert_or_update_game_rules,
    insert_default_player_stats_chase
)

def create_all_tables():
    """
    Creates all the core tables if they do not exist yet.
    Then adds user_id and conversation_id columns to each table for multi-user,
    multi-conversation scoping, with ON DELETE CASCADE references.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) The base tables
    # (Same as your original, no changes needed in the CREATE TABLE ... IF NOT EXISTS)

    # --------------- EXAMPLE: CREATE TABLES IF NOT EXISTS ---------------
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
        CREATE TABLE IF NOT EXISTS CurrentRoleplay (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
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
          setting_variants JSONB
        );
    ''')

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

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Locations (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            open_hours JSONB
        );
    ''')

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

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlannedEvents (
          event_id SERIAL PRIMARY KEY,
          npc_id INT REFERENCES NPCStats(npc_id),
          day INT NOT NULL,
          time_of_day TEXT NOT NULL,
          override_location TEXT NOT NULL,
          UNIQUE(npc_id, day, time_of_day)
        );
    ''')

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
          user_id INTEGER NOT NULL REFERENCES users(id),
          conversation_name VARCHAR(100) NOT NULL,
          created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
          id SERIAL PRIMARY KEY,
          conversation_id INTEGER NOT NULL REFERENCES conversations(id),
          sender VARCHAR(50) NOT NULL,
          content TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    # -- Fix the problematic constraint code below --
    # Instead of "ADD CONSTRAINT IF NOT EXISTS", do "DROP ... IF EXISTS" then "ADD CONSTRAINT"
    # --------------
    # 1) Drop old constraint if present
    cursor.execute('''
        ALTER TABLE messages
        DROP CONSTRAINT IF EXISTS messages_conversation_id_fkey
    ''')

    # 2) Now add the constraint normally, without IF NOT EXISTS
    cursor.execute('''
        ALTER TABLE messages
        ADD CONSTRAINT messages_conversation_id_fkey
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # folders table + folder_id in conversations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS folders (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            folder_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    ''')

    cursor.execute('''
        ALTER TABLE conversations
        ADD COLUMN IF NOT EXISTS folder_id INTEGER
    ''')

    # We'll drop old constraint then add again
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE conversations DROP CONSTRAINT IF EXISTS fk_conversations_folder;
        EXCEPTION WHEN undefined_object THEN
            -- no constraint yet
        END;
        $$
    ''')

    # Add constraint plainly, no IF NOT EXISTS
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE conversations
            ADD CONSTRAINT fk_conversations_folder
            FOREIGN KEY (folder_id)
            REFERENCES folders(id)
            ON DELETE CASCADE;
        EXCEPTION WHEN duplicate_object THEN
            -- constraint might already exist
        END;
        $$
    ''')

    # --------------------------------------------------------------------
    # Now add user_id AND conversation_id to each table for multi-user & multi-conversation scoping

    # NPCStats:
    cursor.execute('''
        ALTER TABLE NPCStats
        ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        ALTER TABLE NPCStats
        ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL
    ''')
    # Drop then add constraint(s) plainly
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE NPCStats DROP CONSTRAINT IF EXISTS npcstats_user_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE NPCStats
        ADD CONSTRAINT npcstats_user_fk
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE NPCStats DROP CONSTRAINT IF EXISTS npcstats_conv_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE NPCStats
        ADD CONSTRAINT npcstats_conv_fk
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # PlayerStats:
    cursor.execute('''
        ALTER TABLE PlayerStats
        ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        ALTER TABLE PlayerStats
        ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE PlayerStats DROP CONSTRAINT IF EXISTS playerstats_user_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE PlayerStats
        ADD CONSTRAINT playerstats_user_fk
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE PlayerStats DROP CONSTRAINT IF EXISTS playerstats_conv_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE PlayerStats
        ADD CONSTRAINT playerstats_conv_fk
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # CurrentRoleplay:
    cursor.execute('''
        ALTER TABLE CurrentRoleplay
        ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        ALTER TABLE CurrentRoleplay
        ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE CurrentRoleplay DROP CONSTRAINT IF EXISTS currentroleplay_user_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE CurrentRoleplay
        ADD CONSTRAINT currentroleplay_user_fk
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE CurrentRoleplay DROP CONSTRAINT IF EXISTS currentroleplay_conv_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE CurrentRoleplay
        ADD CONSTRAINT currentroleplay_conv_fk
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # Events:
    cursor.execute('''
        ALTER TABLE Events
        ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        ALTER TABLE Events
        ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE Events DROP CONSTRAINT IF EXISTS events_user_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE Events
        ADD CONSTRAINT events_user_fk
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE Events DROP CONSTRAINT IF EXISTS events_conv_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE Events
        ADD CONSTRAINT events_conv_fk
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # PlannedEvents:
    cursor.execute('''
        ALTER TABLE PlannedEvents
        ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        ALTER TABLE PlannedEvents
        ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE PlannedEvents DROP CONSTRAINT IF EXISTS plannedevents_user_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE PlannedEvents
        ADD CONSTRAINT plannedevents_user_fk
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE PlannedEvents DROP CONSTRAINT IF EXISTS plannedevents_conv_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE PlannedEvents
        ADD CONSTRAINT plannedevents_conv_fk
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # PlayerInventory:
    cursor.execute('''
        ALTER TABLE PlayerInventory
        ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        ALTER TABLE PlayerInventory
        ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE PlayerInventory DROP CONSTRAINT IF EXISTS playerinventory_user_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE PlayerInventory
        ADD CONSTRAINT playerinventory_user_fk
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE PlayerInventory DROP CONSTRAINT IF EXISTS playerinventory_conv_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE PlayerInventory
        ADD CONSTRAINT playerinventory_conv_fk
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # Quests:
    cursor.execute('''
        ALTER TABLE Quests
        ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        ALTER TABLE Quests
        ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE Quests DROP CONSTRAINT IF EXISTS quests_user_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE Quests
        ADD CONSTRAINT quests_user_fk
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE Quests DROP CONSTRAINT IF EXISTS quests_conv_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE Quests
        ADD CONSTRAINT quests_conv_fk
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # Locations:
    cursor.execute('''
        ALTER TABLE Locations
        ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        ALTER TABLE Locations
        ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE Locations DROP CONSTRAINT IF EXISTS locations_user_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE Locations
        ADD CONSTRAINT locations_user_fk
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE Locations DROP CONSTRAINT IF EXISTS locations_conv_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE Locations
        ADD CONSTRAINT locations_conv_fk
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    # SocialLinks:
    cursor.execute('''
        ALTER TABLE SocialLinks
        ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        ALTER TABLE SocialLinks
        ADD COLUMN IF NOT EXISTS conversation_id INTEGER NOT NULL
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE SocialLinks DROP CONSTRAINT IF EXISTS sociallinks_user_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE SocialLinks
        ADD CONSTRAINT sociallinks_user_fk
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
    ''')
    cursor.execute('''
        DO $$
        BEGIN
            ALTER TABLE SocialLinks DROP CONSTRAINT IF EXISTS sociallinks_conv_fk;
        EXCEPTION WHEN undefined_object THEN END;
        $$
    ''')
    cursor.execute('''
        ALTER TABLE SocialLinks
        ADD CONSTRAINT sociallinks_conv_fk
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
    ''')

    conn.commit()
    conn.close()


def seed_initial_data():
    """
    Inserts default data (stat definitions, game rules, settings, activities, archetypes, etc.).
    """
    insert_or_update_game_rules()
    insert_stat_definitions()
    insert_missing_settings()  # Provide them here
    insert_missing_activities()
    insert_missing_archetypes()
    insert_default_player_stats_chase(user_id, conversation_id)
    print("All default data seeded successfully.")


def initialize_all_data():
    create_all_tables()
    seed_initial_data()
    print("All tables created & default data seeded successfully!")
