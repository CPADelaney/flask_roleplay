# Hot Path vs Slow Path Architecture

## Overview

This document describes the architectural separation between **hot path** (fast, synchronous, player-facing) and **slow path** (background workers, LLM calls) operations in the Flask Roleplay application.

## Guiding Principle

> **The game loop must never wait for an LLM. It reads from a cache; background workers write to that cache.**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         HOT PATH                             │
│                  (Fast, Synchronous, <200ms)                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Event Handlers:                                              │
│  - _on_state_sync()          → Read cache, return hints       │
│  - _on_player_choice()       → Dispatch task, return ack      │
│  - _on_conflict_updated()    → Update numeric state only      │
│                                                               │
│  Operations:                                                  │
│  - DB queries (indexed, <50ms)                                │
│  - Redis cache reads                                          │
│  - Rule-based logic                                           │
│  - Numeric state updates                                      │
│                                                               │
│  NO:                                                          │
│  ❌ LLM calls (Runner.run, llm_json, etc.)                    │
│  ❌ Heavy computation                                         │
│  ❌ Waiting for I/O                                           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ dispatch tasks
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        SLOW PATH                             │
│              (Background Workers, 500ms-5s+)                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Celery Tasks:                                                │
│  - generate_stakeholder_action()  → LLM decision making       │
│  - generate_stakeholder_reaction() → LLM reaction generation  │
│  - narrate_phase_transition()     → LLM prose narration       │
│  - generate_social_bundle()       → LLM gossip + reputation   │
│  - canonize_conflict()            → LLM lore integration      │
│                                                               │
│  Operations:                                                  │
│  - OpenAI API calls (500ms-2s each)                           │
│  - Complex LLM prompts                                        │
│  - Heavy computation                                          │
│  - Batch processing                                           │
│                                                               │
│  Output:                                                      │
│  ✅ Write to planned_stakeholder_actions table                │
│  ✅ Cache results in Redis                                    │
│  ✅ Update DB with precomputed data                           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### Hot Path Modules

#### 1. Event Handlers
**Files:**
- `logic/conflict_system/autonomous_stakeholder_actions.py`
- `logic/conflict_system/conflict_flow.py`
- `logic/conflict_system/social_circle.py`
- `logic/conflict_system/conflict_canon.py`

**Requirements:**
- Must respond in <200ms
- Read from cache/DB only
- Dispatch background tasks when needed
- Return fallback values if cache miss

#### 2. Hot Path Helpers
**Files:**
- `logic/conflict_system/autonomous_stakeholder_actions_hotpath.py`
- `infra/cache.py`

**Functions:**
- `fetch_ready_actions_for_scene()` - Query precomputed actions
- `determine_scene_behavior()` - Rule-based NPC behavior
- `dispatch_action_generation()` - Non-blocking task dispatch
- `get_json()`, `set_json()` - Fast Redis cache access

### Slow Path Modules

#### 1. Background Tasks
**Files:**
- `nyx/tasks/background/stakeholder_tasks.py`
- `nyx/tasks/background/flow_tasks.py`
- `nyx/tasks/background/social_tasks.py`
- `nyx/tasks/background/canon_tasks.py`

**Tasks:**
- `generate_stakeholder_action` - LLM decision making
- `generate_stakeholder_reaction` - LLM reaction generation
- `narrate_phase_transition` - LLM prose for phase transitions
- `generate_beat_description` - LLM dramatic beat narration
- `generate_social_bundle` - LLM gossip + reputation
- `canonize_conflict` - LLM lore integration
- `generate_canon_references` - LLM NPC dialogue generation

#### 2. Storage
**Database Tables:**
- `planned_stakeholder_actions` - Precomputed actions ready for consumption
- `canon.events` - Canonical events for lore consistency

**Redis Keys:**
- `social_bundle:{scene_hash}` - Social dynamics cache
- `conflict:{id}:transition_narrative` - Phase transition prose
- `conflict:{id}:beat:{id}` - Dramatic beat prose

## Data Flow Examples

### Example 1: Player Choice Response

```
User clicks "Attack NPC"
    │
    ▼
┌──────────────────────────────────────┐
│ HOT PATH: _on_player_choice()        │
│ - Update conflict state (numeric)    │  <200ms
│ - Dispatch reaction tasks             │
│ - Return ACK to player                │
└──────────────────────────────────────┘
    │
    │ returns immediately
    ▼
User sees "NPC prepares to respond..."
    │
    │ (background)
    ▼
┌──────────────────────────────────────┐
│ SLOW PATH: generate_reaction()       │
│ - Call LLM for reaction               │  1-3s
│ - Store in planned_actions            │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ HOT PATH: Next _on_state_sync()      │
│ - Query planned_actions               │  <100ms
│ - Return NPC reaction                 │
└──────────────────────────────────────┘
    │
    ▼
User sees "NPC blocks your attack!"
```

### Example 2: Scene State Sync

```
GET /api/scene/sync
    │
    ▼
┌──────────────────────────────────────┐
│ HOT PATH: _on_state_sync()           │
│ - Query planned_actions (DB)         │  <50ms
│ - Read social_bundle (Redis)         │  <20ms
│ - Compute behavior hints (rules)     │  <10ms
│ - Return complete scene state        │
└──────────────────────────────────────┘
    │
    │ if cache miss for social bundle
    ▼
┌──────────────────────────────────────┐
│ HOT PATH: Acquire lock + dispatch    │
│ - redis_lock("social_bundle:lock")   │  <30ms
│ - dispatch generate_social_bundle()  │
│ - Return fallback: {gossip: []}      │
└──────────────────────────────────────┘
    │
    │ (background)
    ▼
┌──────────────────────────────────────┐
│ SLOW PATH: generate_social_bundle()  │
│ - LLM generate gossip                │  1-2s
│ - LLM calculate reputation           │  1-2s
│ - Cache result in Redis              │
└──────────────────────────────────────┘
    │
    ▼
Next sync: Returns full social data from cache
```

## Implementation Patterns

### Pattern 1: Query-Then-Dispatch

```python
async def _on_state_sync(event):
    # HOT PATH: Query precomputed data
    actions = await fetch_ready_actions_for_scene(scene_context)

    if not actions:
        # Cache miss: dispatch background generation
        dispatch_action_generation(stakeholder, scene_context)
        # Return fallback immediately
        return {"actions": [], "status": "generating"}

    # Cache hit: return precomputed data
    return {"actions": actions, "status": "ready"}
```

### Pattern 2: Numeric-Then-Prose

```python
async def _handle_phase_transition(conflict, to_phase):
    # HOT PATH: Update numeric state immediately
    conflict.current_phase = to_phase
    await db.save(conflict)

    # SLOW PATH: Dispatch prose generation
    narrate_phase_transition.delay({
        "conflict_id": conflict.id,
        "from_phase": from_phase,
        "to_phase": to_phase,
    })

    # Return immediately without prose
    return {"phase": to_phase, "narration_status": "pending"}
```

### Pattern 3: Cache-First with Lock

```python
async def get_social_bundle(scene_hash):
    # HOT PATH: Check cache first
    key = f"social_bundle:{scene_hash}"
    cached = redis_client.get_json(key)
    if cached:
        return cached

    # Cache miss: acquire lock to prevent task storm
    lock_key = f"{key}:lock"
    try:
        with redis_lock(lock_key, ttl=15, blocking=False):
            # Dispatch background generation
            generate_social_bundle.delay({"scene_hash": scene_hash})
    except RuntimeError:
        # Another worker is already generating
        pass

    # Return fallback immediately
    return {"gossip": [], "reputation_status": "pending"}
```

## Performance Metrics

### Before Refactoring
| Operation | Latency | Bottleneck |
|-----------|---------|------------|
| State sync | 2-5s | Multiple LLM calls in event handler |
| Player choice | 1-3s | Reaction generation blocking |
| Conflict creation | 2-4s | Flow initialization + stakeholder LLM calls |

### After Refactoring
| Operation | Latency | Mechanism |
|-----------|---------|-----------|
| State sync | <100ms | DB query + Redis cache read |
| Player choice | <200ms | Dispatch task, return ACK |
| Conflict creation | <300ms | Minimal DB insert, background enrichment |

**Improvement:** 10-25x latency reduction

## Testing Strategy

### Hot Path Tests
```python
# Test hot path is fast
async def test_state_sync_performance():
    start = time.time()
    result = await _on_state_sync(event)
    elapsed = time.time() - start
    assert elapsed < 0.2  # Must be <200ms

# Test fallback behavior
async def test_cache_miss_returns_fallback():
    # Clear cache
    redis_client.delete("social_bundle:test")

    # Should return fallback immediately
    result = await get_social_bundle("test")
    assert result["gossip"] == []
    assert result["reputation_status"] == "pending"
```

### Integration Tests
```python
# Test end-to-end flow
async def test_action_generation_flow():
    # Dispatch task
    dispatch_action_generation(stakeholder, context)

    # Wait for task to complete
    await asyncio.sleep(3)

    # Query should find precomputed action
    actions = await fetch_ready_actions_for_scene(context)
    assert len(actions) > 0
```

## CI Enforcement

### Automated Guard
The CI pipeline runs `scripts/ci/forbid_hotpath_llm.py` on every PR to detect:
- `Runner.run()` calls in hot path
- `llm_json()` calls in event handlers
- Blocking LLM functions in synchronous code

### GitHub Actions Workflow
`.github/workflows/hotpath-guard.yml` fails builds if violations are detected.

## Migration Checklist

### For Each Subsystem

- [ ] Identify blocking LLM calls in event handlers
- [ ] Create background task module in `nyx/tasks/background/`
- [ ] Create hot path helper module `*_hotpath.py`
- [ ] Add DB table or Redis cache for precomputed data
- [ ] Refactor event handlers to dispatch tasks + read cache
- [ ] Add fallback values for cache misses
- [ ] Write tests for hot and slow paths
- [ ] Update documentation

### Subsystem Status

| Subsystem | Hot Path Module | Slow Path Tasks | Status |
|-----------|----------------|-----------------|--------|
| Stakeholder Autonomy | `autonomous_stakeholder_actions_hotpath.py` | `stakeholder_tasks.py` | ✅ Created |
| Conflict Flow | TBD | `flow_tasks.py` | ✅ Tasks created |
| Social Circle | TBD | `social_tasks.py` | ✅ Tasks created |
| Canon | TBD | `canon_tasks.py` | ✅ Tasks created |

## Troubleshooting

### Issue: High cache miss rate
**Symptom:** Users see "generating" status frequently
**Fix:** Increase cache TTL, pregenerate common scenarios, batch task dispatch

### Issue: Task backlog growing
**Symptom:** Background tasks queue up, delays increase
**Fix:** Scale workers, add task priorities, implement rate limiting

### Issue: Stale cache data
**Symptom:** Users see outdated actions/narration
**Fix:** Add cache invalidation on state changes, reduce TTL

## References

- Architecture Plan: [consolidation_implementation_plan.md](../consolidation_implementation_plan.md)
- Blocking Patterns Report: [hot_path_blockers.md](./hot_path_blockers.md)
- Refactoring Guide: [logic/conflict_system/REFACTORING_GUIDE.md](../logic/conflict_system/REFACTORING_GUIDE.md)
- CI Guard: [scripts/ci/forbid_hotpath_llm.py](../scripts/ci/forbid_hotpath_llm.py)
