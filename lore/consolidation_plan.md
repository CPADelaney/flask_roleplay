# Lore System Consolidation Plan

## Identified Redundancies

### 1. Multiple Lore-Related Classes with Overlapping Functionality

- `LoreSystem` (lore_system.py)
- `LoreManager` (lore_manager.py)
- `LoreIntegrationSystem` (lore_integration.py)
- `EnhancedLoreSystem` (enhanced_lore_consolidated.py)
- `DynamicLoreGenerator` (dynamic_lore_generator.py)
- `NPCLoreIntegration` (npc_lore_integration.py)

### 2. Conflict System Redundancy

- `ConflictSystemIntegration` and `EnhancedConflictSystemIntegration` in the same file with similar methods

### 3. Redundant Database Access Methods

- Multiple implementations of similar methods like `get_npc_details` across different files
- Redundant location/lore context retrieval methods

### 4. Duplicate Context Enhancement Methods

- Similar methods for enhancing context with lore across multiple classes

## Consolidation Approach

### 1. Core Lore System

Keep `LoreSystem` as the main entry point (as intended) but:
- Move all database access to `LoreManager` (keep this as the only class interacting with DB for lore)
- Merge `EnhancedLoreSystem` and `DynamicLoreGenerator` functionality into `LoreSystem`
- Keep `LoreIntegrationSystem` for integration with other systems only

### 2. Simplified Class Structure

```
LoreSystem (main entry point & API)
  ├── LoreManager (single DB access layer)
  │     └── Database operations (get_lore_by_id, etc.)
  ├── LoreIntegration (system integration functions)
  │     └── Integration with NPCs, conflicts, etc.
  └── NPCLoreIntegration (specialized NPC lore functions)
```

### 3. Remove Redundant Methods

- Standardize on single implementations of:
  - `get_npc_details`
  - `get_location_lore`
  - `get_location_details`
  - Location context methods
  - Conflict-related methods

### 4. Standardize Context Enhancement

- Create a single, unified method for enhancing contexts with lore

## Implementation Plan

1. First phase: Create adapter methods in `LoreSystem` to maintain API compatibility
2. Second phase: Update callers to use the consolidated APIs
3. Third phase: Remove redundant files & classes

## Files to be Consolidated/Removed

- `enhanced_lore_consolidated.py` (merge into `lore_system.py`)
- `dynamic_lore_generator.py` (merge into `lore_system.py`)
- Refactor `conflict_integration.py` to remove `EnhancedConflictSystemIntegration` redundancy
- Standardize data retrieval methods to avoid duplication

## Dependency Management

- Update imports to use the consolidated classes
- Ensure proper error handling during transition
- Add appropriate logging for deprecated methods 