# db/initialization.py

from db.connection import get_db_connection

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Existing table creation
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
    # New or existing table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CurrentRoleplay (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    ''')
        # -- New: Create StatDefinitions
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
    
        # -- New: Create GameRules
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS GameRules (
          id SERIAL PRIMARY KEY,
          rule_name TEXT NOT NULL,
          condition TEXT NOT NULL,
          effect TEXT NOT NULL
        );
        ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS NPCStats (
            npc_id SERIAL PRIMARY KEY,
            npc_name TEXT NOT NULL,
            dominance INT CHECK (dominance BETWEEN 0 AND 100),
            cruelty INT CHECK (cruelty BETWEEN 0 AND 100),
            closeness INT CHECK (closeness BETWEEN 0 AND 100),
            trust INT CHECK (trust BETWEEN -100 AND 100),
            respect INT CHECK (respect BETWEEN -100 AND 100),
            intensity INT CHECK (intensity BETWEEN 0 AND 100),
            memory TEXT,
            monica_level INT DEFAULT 0,
            monica_games_left INT DEFAULT 0
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
    conn.commit()
    conn.close()
    
# @app.before_first_request
def init_tables_and_settings():
    initialize_database()
    insert_stat_definitions()
    insert_game_rules()
    insert_missing_settings()


app.teardown_appcontext(close_db) 
