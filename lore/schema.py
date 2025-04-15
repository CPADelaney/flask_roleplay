# lore/schema.py

import logging
from db.connection import get_db_connection_context

async def create_lore_tables():
    """
    Creates database tables for the lore system using async operations.
    """
    logger = logging.getLogger(__name__)
    
    try:
        async with get_db_connection_context() as conn:
            # 1. World Lore - Core foundational elements of the world
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS WorldLore (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL, 
                    category TEXT NOT NULL,  -- creation_myth, cosmology, magic_system, etc.
                    description TEXT NOT NULL,
                    significance INTEGER CHECK (significance BETWEEN 1 AND 10),
                    embedding VECTOR(1536),
                    tags TEXT[]
                );
            ''')

            # 2. Factions - Political, religious, or social groups
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS Factions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL, -- political, religious, criminal, etc.
                    description TEXT NOT NULL,
                    headquarters TEXT,
                    founding_story TEXT,
                    values TEXT[] NOT NULL,
                    goals TEXT[] NOT NULL,
                    rivals TEXT[],
                    allies TEXT[],
                    territory TEXT,
                    resources TEXT[],
                    hierarchy_type TEXT, -- hierarchical, flat, democratic, etc.
                    secret_knowledge TEXT,
                    public_reputation TEXT,
                    embedding VECTOR(1536),
                    color_scheme TEXT,
                    symbol_description TEXT
                );
            ''')

            # 3. Cultural Elements - Traditions, customs, and beliefs
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS CulturalElements (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL, -- tradition, custom, taboo, holiday, etc.
                    description TEXT NOT NULL,
                    practiced_by TEXT[], -- factions, regions, or "universal"
                    historical_origin TEXT,
                    significance INTEGER CHECK (significance BETWEEN 1 AND 10),
                    related_elements TEXT[],
                    embedding VECTOR(1536)
                );
            ''')

            # 4. Historical Events - Past events that shaped the world
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS HistoricalEvents (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    date_description TEXT, -- "200 years ago," "during the reign of Emperor X"
                    description TEXT NOT NULL,
                    participating_factions TEXT[],
                    affected_locations TEXT[],
                    consequences TEXT[],
                    historical_figures TEXT[],
                    commemorated_by TEXT, -- holiday, monument, etc.
                    embedding VECTOR(1536),
                    significance INTEGER CHECK (significance BETWEEN 1 AND 10)
                );
            ''')

            # 5. Geographic Regions - Larger regions with distinct cultures
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS GeographicRegions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    climate TEXT,
                    notable_features TEXT[],
                    main_settlements TEXT[],
                    cultural_traits TEXT[],
                    governing_faction TEXT,
                    resources TEXT[],
                    strategic_importance TEXT,
                    population_description TEXT,
                    embedding VECTOR(1536)
                );
            ''')

            # 6. Location Lore - Additional lore-specific data for locations
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS LocationLore (
                    location_id INTEGER PRIMARY KEY,
                    founding_story TEXT,
                    hidden_secrets TEXT[],
                    supernatural_phenomena TEXT,
                    local_legends TEXT[],
                    historical_significance TEXT,
                    associated_factions TEXT[],
                    embedding VECTOR(1536),
                    FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                );
            ''')

            # 7. LoreConnections - Connections between lore elements
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS LoreConnections (
                    id SERIAL PRIMARY KEY,
                    source_type TEXT NOT NULL, -- WorldLore, Factions, etc.
                    source_id INTEGER NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id INTEGER NOT NULL,
                    connection_type TEXT NOT NULL, -- influences, conflicts_with, supports, etc.
                    description TEXT,
                    strength INTEGER CHECK (strength BETWEEN 1 AND 10),
                    UNIQUE (source_type, source_id, target_type, target_id, connection_type)
                );
            ''')

            # 8. LoreKnowledge - Who knows what lore
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS LoreKnowledge (
                    id SERIAL PRIMARY KEY,
                    entity_type TEXT NOT NULL, -- npc, player, faction
                    entity_id INTEGER NOT NULL,
                    lore_type TEXT NOT NULL, -- WorldLore, Factions, etc.
                    lore_id INTEGER NOT NULL,
                    knowledge_level INTEGER CHECK (knowledge_level BETWEEN 0 AND 10),
                    is_secret BOOLEAN DEFAULT FALSE,
                    discovery_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (entity_type, entity_id, lore_type, lore_id)
                );
            ''')

            # 9. LoreTags - For categorizing and searching lore
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS LoreTags (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    category TEXT
                );
            ''')

            # 10. LoreDiscoveryOpportunities - Places/events where lore can be discovered
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS LoreDiscoveryOpportunities (
                    id SERIAL PRIMARY KEY,
                    lore_type TEXT NOT NULL,
                    lore_id INTEGER NOT NULL,
                    location_id INTEGER,
                    event_id INTEGER,
                    npc_id INTEGER,
                    discovery_method TEXT NOT NULL, -- conversation, investigation, book, etc.
                    difficulty INTEGER CHECK (difficulty BETWEEN 1 AND 10),
                    prerequisites JSONB,
                    discovered BOOLEAN DEFAULT FALSE,
                    CHECK (
                        (location_id IS NOT NULL AND event_id IS NULL AND npc_id IS NULL) OR
                        (location_id IS NULL AND event_id IS NOT NULL AND npc_id IS NULL) OR
                        (location_id IS NULL AND event_id IS NULL AND npc_id IS NOT NULL)
                    )
                );
            ''')

            # 11. Update NPCStats table to include lore knowledge fields
            await conn.execute('''
                ALTER TABLE NPCStats
                ADD COLUMN IF NOT EXISTS lore_knowledge JSONB,
                ADD COLUMN IF NOT EXISTS cultural_background TEXT,
                ADD COLUMN IF NOT EXISTS faction_memberships TEXT[];
            ''')

            # Create necessary indexes
            # Note: These can be executed as a transaction or separately
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_worldlore_embedding ON WorldLore USING ivfflat (embedding vector_cosine_ops);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_factions_embedding ON Factions USING ivfflat (embedding vector_cosine_ops);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cultural_elements_embedding ON CulturalElements USING ivfflat (embedding vector_cosine_ops);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_historical_events_embedding ON HistoricalEvents USING ivfflat (embedding vector_cosine_ops);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_geographic_regions_embedding ON GeographicRegions USING ivfflat (embedding vector_cosine_ops);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_location_lore_embedding ON LocationLore USING ivfflat (embedding vector_cosine_ops);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_lore_connections_source ON LoreConnections(source_type, source_id);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_lore_connections_target ON LoreConnections(target_type, target_id);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_lore_knowledge_entity ON LoreKnowledge(entity_type, entity_id);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_lore_knowledge_lore ON LoreKnowledge(lore_type, lore_id);
            ''')

        logger.info("Lore tables created successfully.")

    except Exception as e:
        logger.error(f"Error creating lore tables: {e}")
        raise

# Modified main section to handle async
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(create_lore_tables())
