# logic/initialization.py

from db.initialization import initialize_database
from routes.activities import insert_missing_activities
from routes.archetypes import insert_missing_archetypes
from routes.settings_routes import insert_missing_settings
from logic.stats_logic import insert_stat_definitions, insert_or_update_game_rules, insert_default_player_stats_chase
# (Import all relevant "insert_missing" or "insert_default" from your existing modules)

def initialize_all_data():
    """
    Creates all tables, then populates them with default data:
      - Stat definitions
      - Game rules
      - Settings
      - Activities
      - Archetypes
      - Possibly default player stats, etc.
    """
    # 1) Create tables if not exist
    initialize_database()

    # 2) Insert or update base data
    insert_or_update_game_rules()
    insert_stat_definitions()
    insert_missing_settings()
    insert_missing_activities()
    insert_missing_archetypes()
    insert_default_player_stats_chase()   # If you want a default 'Chase' player

    print("All data initialized successfully!")
