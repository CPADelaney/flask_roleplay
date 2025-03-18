# Lore System

A comprehensive narrative and relationship management system for roleplay applications.

## Overview

The Lore System is a modular framework that manages narrative progression and social relationships in roleplay applications. It provides:

- Dynamic narrative stages with progression based on player stats
- Complex social link system with multi-dimensional relationships
- Event management tied to narrative stages
- Efficient caching and database operations
- Robust error handling and logging

## Components

### Core Components

- `system.py`: Main LoreSystem class that integrates all components
- `narrative.py`: Narrative progression and stage management
- `social_links.py`: Social relationship management

### Utilities

- `db.py`: Database connection and query management
- `cache.py`: Redis-based caching system

### Configuration

- `settings.py`: System-wide configuration and constants

## Features

### Narrative Progression

- Multiple narrative stages with progression requirements
- Stage-specific events and content
- Automatic stage transitions based on player stats
- Event history tracking

### Social Links

- Multi-dimensional relationship tracking
- Relationship history and evolution
- Network analysis capabilities
- Support for various relationship types

### Performance

- Redis caching for frequently accessed data
- Connection pooling for database operations
- Efficient query optimization
- Background task support

## Usage

### Basic Usage

```python
from logic.lore.core.system import lore_system

# Get current state
state = lore_system.get_current_state(user_id, conversation_id)

# Update a social link
result = lore_system.update_social_link(
    user_id, conversation_id,
    entity_type="npc",
    entity_id=1,
    link_level=50,
    link_type="friendly"
)

# Get available events
events = lore_system.get_available_events(user_id, conversation_id)

# Get relationship network
network = lore_system.get_relationship_network(
    user_id, conversation_id,
    entity_type="npc",
    entity_id=1
)
```

### Configuration

The system can be configured through `settings.py`:

```python
from logic.lore.config.settings import config

# Modify settings
config.DB_POOL_SIZE = 10
config.CACHE_TTL = 7200
```

## Database Schema

### PlayerStats
- user_id: int
- conversation_id: int
- player_name: str
- corruption: int
- dependency: int
- other stats...

### SocialLinks
- link_id: int
- user_id: int
- conversation_id: int
- entity1_type: str
- entity1_id: int
- entity2_type: str
- entity2_id: int
- link_type: str
- link_level: int
- link_history: jsonb
- dimensions: jsonb
- last_updated: timestamp

### NarrativeTransitions
- transition_id: int
- user_id: int
- conversation_id: int
- old_stage: str
- new_stage: str
- transition_time: timestamp

### StageEvents
- event_id: int
- stage_name: str
- event_type: str
- description: text
- requirements: jsonb

## Error Handling

The system uses custom exceptions for different types of errors:

- `LoreError`: Base exception for lore system errors
- `NarrativeError`: Errors related to narrative progression
- `SocialLinkError`: Errors related to social links
- `DatabaseError`: Database operation errors
- `CacheError`: Caching operation errors

## Logging

The system uses Python's logging module with the following loggers:

- `logic.lore.core.system`
- `logic.lore.core.narrative`
- `logic.lore.core.social_links`
- `logic.lore.utils.db`
- `logic.lore.utils.cache`

## Dependencies

- SQLAlchemy: Database ORM
- Redis: Caching
- PostgreSQL: Database
- Python 3.8+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 