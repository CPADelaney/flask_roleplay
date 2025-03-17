# Data Access Method Consolidation

## Identified Data Access Method Redundancies

### 1. NPC Detail Retrieval

Multiple implementations of `get_npc_details` in:
- `npcs/npc_handler.py`
- `logic/fully_integrated_npc_system.py`
- `logic/conflict_system/conflict_tools.py`
- `npcs/new_npc_creation.py`
- `lore/lore_integration.py` (as `_get_npc_details`)

### 2. Location Data Retrieval

Multiple methods for getting location information:
- `get_location_details` and `_get_location_details` in different files
- `get_location_lore` and variants in multiple files
- `get_comprehensive_location_context` and helper methods

### 3. Cultural/Political/Religious Context Retrieval

- `_get_cultural_context_for_location`
- `_get_religious_context_for_location`
- `_get_political_context_for_location`

### 4. Conflict System Duplication

- Similar methods between `ConflictSystemIntegration` and `EnhancedConflictSystemIntegration`
- Redundant directive handling methods

## Consolidation Strategy

### 1. Create Data Access Layer (DAL)

Create dedicated data access classes:

```
data/
  ├── npc_dal.py
  │     └── NPCDataAccess class
  ├── location_dal.py
  │     └── LocationDataAccess class
  ├── lore_dal.py
  │     └── LoreDataAccess class
  └── conflict_dal.py
        └── ConflictDataAccess class
```

### 2. Standardized Method Signatures

For each data entity type, standardize method signatures:

#### NPCs
```python
async def get_npc_details(npc_id: int = None, npc_name: str = None) -> Dict[str, Any]
async def get_npc_relationships(npc_id: int) -> List[Dict[str, Any]]
```

#### Locations
```python
async def get_location_details(location_id: int = None, location_name: str = None) -> Dict[str, Any]
async def get_location_lore(location_id: int = None, location_name: str = None) -> Dict[str, Any]
async def get_location_context(location_id: int = None, location_name: str = None) -> Dict[str, Any]
```

#### Lore
```python
async def get_lore_by_id(lore_type: str, lore_id: int) -> Dict[str, Any]
async def get_relevant_lore(query_text: str, lore_types: List[str] = None) -> List[Dict[str, Any]]
```

### 3. Implementation Steps

1. Create the DAL classes with the standardized methods
2. Update existing methods to use the new DAL
3. Add deprecation warnings to the old methods
4. Update callers to use the new DAL directly

### 4. Specific Redundant Methods to Consolidate

| Current Multiple Implementations | Consolidated Method |
|----------------------------------|---------------------|
| `get_npc_details` variants | `NPCDataAccess.get_npc_details` |
| `get_location_details` variants | `LocationDataAccess.get_location_details` |
| `get_location_lore` variants | `LocationDataAccess.get_location_lore` |
| Location context methods | `LocationDataAccess.get_location_context` |
| Conflict detail methods | `ConflictDataAccess.get_conflict_details` |

### 5. Benefits

- Removes duplicate code
- Centralizes DB access logic
- Makes testing easier
- Standardizes error handling
- Reduces maintenance overhead

### 6. Handling Context and User/Conversation ID

All DAL methods should accept:
- `user_id`: Optional user ID 
- `conversation_id`: Optional conversation ID
- Passing `None` for either should retrieve global data (not specific to a user/conversation)

### 7. Connection Pooling

Use a single shared connection pool for all DAL classes to avoid connection overhead. 