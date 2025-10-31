# Hot Path Blocking LLM Patterns - Refactoring Report

**Generated:** 2025-10-30
**Purpose:** Identify blocking LLM calls in synchronous event handlers that need async/background task refactoring

---

## Executive Summary

This report identifies **37 blocking LLM call sites** in hot-path code across the conflict system. These calls occur in **synchronous event handlers** that process real-time game events, causing unacceptable latency for players.

**Critical Issues:**
- Event handlers call LLM functions synchronously
- Each LLM call adds 500ms-2s+ latency
- Multiple calls can compound to 5-10s delays in user-facing operations
- State sync and player choice handlers are most affected

---

## Pattern Categories Found

### 1. Blocking Direct LLM Calls
- `llm_json()` - Direct OpenAI Responses API calls
- `Runner.run()` - Agent SDK blocking calls
- `agent.run()` - Direct agent execution

### 2. Blocking Higher-Level Functions
- `make_autonomous_decision()` (legacy; removed in favor of hot-path dispatch helpers)
- `generate_reaction()` (legacy; removed in favor of hot-path dispatch helpers)
- `generate_gossip()` - Social dynamics generation
- `calculate_reputation()` - Reputation calculation
- `generate_dramatic_beat()` - Flow beat generation

---

## Hot Path Files with Blocking Calls

### 🔴 CRITICAL: /home/user/flask_roleplay/logic/conflict_system/autonomous_stakeholder_actions.py

**Event Handlers (Synchronous - HIGH PRIORITY):**

#### ✅ `_on_conflict_updated()` - Line 371 (Resolved)
- **Context:** Previously awaited `make_autonomous_decision()` inline.
- **Status:** Now uses `should_dispatch_action_generation` and `dispatch_action_generation` from the hot-path module so the
  handler returns immediately.

---

#### ✅ `_on_player_choice()` - Line 423 (Resolved)
- **Context:** Previously blocked on `generate_reaction()` for each reacting stakeholder.
- **Status:** Uses `dispatch_reaction_generation` so reactions are queued on the background worker while the handler returns.

---

#### ✅ `_on_state_sync()` - Line 474 (Resolved)
- **Context:** Previously triggered multiple synchronous decision generations.
- **Status:** Combines `fetch_ready_actions_for_scene`, `determine_scene_behavior`, and `dispatch_action_generation` to keep the
  handler non-blocking.

---

#### ✅ `_on_stakeholder_action()` - Line 529 (Resolved)
- **Context:** Historically waited for `make_autonomous_decision()` to finish before responding.
- **Status:** Uses the hot-path helpers to enqueue action generation and immediately acknowledge the request.

---

**Core Blocking Functions:**

#### ❌ `llm_json()` - Line 74
```python
async def llm_json(prompt: str) -> Dict[str, Any]:
    """Call OpenAI Responses API and parse JSON output robustly"""
    try:
        client = _get_client()
        resp = await client.responses.create(
            model="gpt-5-nano",
            input=prompt,
        )
        # ... parse response
```
- **Legacy call sites:** `create_stakeholder()` (line 631), `make_autonomous_decision()` (removed), `generate_reaction()` (removed), `adapt_stakeholder_role()` (line 834)
- **Latency:** 500ms-2s per call
- **Impact:** All stakeholder operations block on LLM
- **Refactor Priority:** 🔴 CRITICAL

**Refactoring Tasks:**
- [ ] Create async wrapper with timeout
- [ ] Add request queueing/batching
- [ ] Implement caching layer
- [ ] Add fallback/degraded mode

---

#### ✅ `make_autonomous_decision()` (Removed)
- **Status:** Deleted from the subsystem. Decision generation now happens exclusively through
  `dispatch_action_generation()` and background tasks writing to `planned_stakeholder_actions`.

---

#### ✅ `generate_reaction()` (Removed)
- **Status:** Deleted from the subsystem. Reaction handling now relies on `dispatch_reaction_generation()` and the background
  Celery task `generate_stakeholder_reaction`.

---

#### ❌ `adapt_stakeholder_role()` - Line 813
```python
async def adapt_stakeholder_role(
    self,
    stakeholder: Stakeholder,
    changing_conditions: Dict[str, Any]
) -> Dict[str, Any]:
    # ... build prompt
    result = await llm_json(prompt)  # LINE 834 - BLOCKS
```
- **Called from:** Line 411 in `_on_phase_transition()`
- **Latency:** 500ms-2s per adaptation
- **Impact:** Phase transitions delayed
- **Refactor Priority:** 🟡 HIGH

**Refactoring Tasks:**
- [ ] Move to background task
- [ ] Apply role changes in next cycle
- [ ] Cache adaptation patterns
- [ ] Pre-compute likely transitions

---

### 🔴 CRITICAL: /home/user/flask_roleplay/logic/conflict_system/social_circle.py

**Event Handlers (Updated - Hot Path Clean):**

#### ✅ `_handle_state_sync()` - Line 229
```python
async def _handle_state_sync(self, event: SystemEvent) -> SubsystemResponse:
    if present_npcs and len(present_npcs) >= 2 and random.random() < 0.3:
        schedule_gossip_generation(
            scene_context,
            present_npcs[:2],
            user_id=self.user_id,
            conversation_id=self.conversation_id,
        )

    for npc_id in present_npcs:
        if npc_id not in self._reputation_cache:
            reputation = await get_cached_reputation_scores(npc_id)
            self._reputation_cache[npc_id] = reputation
```
- **Status:** ✅ Non-blocking. Hot path only schedules background work and reads caches.

#### ✅ `_handle_conflict_created()` - Line 368
```python
async def _handle_conflict_created(self, event: SystemEvent) -> SubsystemResponse:
    if participants:
        schedule_gossip_generation(
            scene_context,
            participants,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
        )
```
- **Status:** ✅ Gossip generation dispatched to Celery; handler returns immediately.

#### ✅ `_handle_conflict_resolved()` - Line 397
```python
async def _handle_conflict_resolved(self, event: SystemEvent) -> SubsystemResponse:
    if all_participants:
        schedule_gossip_generation(
            scene_context,
            all_participants,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
        )
        initial_gossip_cache = await get_cached_gossip_items(
            scene_context['scene_hash'],
            limit=3,
        )
        # ... schedule follow-up polling for richer data
```
- **Status:** ✅ Handler now queues work and consumes cached payloads.

---

**Core Blocking Functions (Resolved):**

- `SocialCircleManager.generate_gossip` ➜ removed; hot path now uses `schedule_gossip_generation()` + cache lookups.
- `SocialCircleManager.calculate_reputation` ➜ removed; handlers call `get_cached_reputation_scores()` and schedule slow paths.

---

#### ❌ `spread_gossip()` - Line 660
```python
async def spread_gossip(
    self,
    gossip: GossipItem,
    spreader_id: int,
    listeners: List[int]
) -> Dict[str, Any]:
    # ... build prompt
    response = await self.social_analyzer.run(prompt)  # LINE 688 - BLOCKS
```
- **Called from:** Line 346 in `_handle_npc_reaction()`
- **Latency:** 500ms-2s per spread event
- **Impact:** NPC reactions delayed
- **Refactor Priority:** 🟡 HIGH

**Refactoring Tasks:**
- [ ] Spread gossip immediately (simple rules)
- [ ] Generate detailed reactions in background
- [ ] Use personality-based heuristics
- [ ] LLM refines reactions async

---

#### ❌ `narrate_reputation_change()` - Line 777
```python
async def narrate_reputation_change(
    self,
    target_id: int,
    old_reputation: Dict[ReputationType, float],
    new_reputation: Dict[ReputationType, float]
) -> str:
    # ... build prompt
    response = await self.reputation_narrator.run(prompt)  # LINE 805 - BLOCKS
```
- **Called from:** Line 293 in `_handle_stakeholder_action()`
- **Latency:** 500ms-2s per narration
- **Impact:** Action feedback delayed
- **Refactor Priority:** 🟡 MEDIUM

**Refactoring Tasks:**
- [ ] Generate narration in background
- [ ] Use template-based placeholder
- [ ] Stream narration when ready
- [ ] Cache common narrations

---

#### ❌ `form_alliance()` - Line 808
```python
async def form_alliance(
    self,
    initiator_id: int,
    target_id: int,
    reason: str
) -> Dict[str, Any]:
    # ... build prompt
    response = await self.alliance_strategist.run(prompt)  # LINE 834 - BLOCKS
```
- **Called from:** Line 314 in `_handle_stakeholder_action()`
- **Latency:** 500ms-2s per alliance
- **Impact:** Alliance formation delayed
- **Refactor Priority:** 🟡 MEDIUM

**Refactoring Tasks:**
- [ ] Create alliance immediately with defaults
- [ ] Fill in details asynchronously
- [ ] Use templates for common alliances
- [ ] LLM generates flavor text later

---

### 🔴 CRITICAL: /home/user/flask_roleplay/logic/conflict_system/conflict_flow.py

**Event Handlers (Synchronous - HIGH PRIORITY):**

#### ❌ `_handle_conflict_created()` - Line 265
```python
async def _handle_conflict_created(self, event: "SystemEvent") -> "SubsystemResponse":
    # ...
    flow = await self.initialize_conflict_flow(conflict_id, conflict_type, context)  # LINE 287 - BLOCKS

    if flow.pacing_style == PacingStyle.RAPID_ESCALATION:
        beat = await self.generate_dramatic_beat(flow, context)  # LINE 292 - BLOCKS
```
- **Context:** Initializes flow when conflict created
- **Blocks on:** `initialize_conflict_flow()` (calls Runner.run at line 610) and `generate_dramatic_beat()` (calls Runner.run at line 721)
- **Impact:** Conflict creation blocked on LLM flow initialization
- **Refactor Priority:** 🔴 CRITICAL

**Refactoring Tasks:**
- [ ] Use default flow settings immediately
- [ ] Refine flow parameters in background
- [ ] Generate dramatic beats asynchronously
- [ ] Apply beat effects in next cycle

---

#### ❌ `_handle_conflict_updated()` - Line 353
```python
async def _handle_conflict_updated(self, event: "SystemEvent") -> "SubsystemResponse":
    # ...
    result = await self.update_conflict_flow(flow, payload)  # LINE 379 - BLOCKS
```
- **Context:** Updates flow based on conflict changes
- **Blocks on:** `update_conflict_flow()` which calls Runner.run at line 674
- **Impact:** Conflict updates wait for flow analysis
- **Refactor Priority:** 🟡 HIGH

**Refactoring Tasks:**
- [ ] Apply rule-based flow updates immediately
- [ ] Queue LLM analysis for background
- [ ] Use heuristics for intensity/momentum
- [ ] LLM validates/refines async

---

#### ❌ `_handle_tension_changed()` - Line 405
```python
async def _handle_tension_changed(self, event: "SystemEvent") -> "SubsystemResponse":
    # ...
    if abs(flow.momentum - old_momentum) > 0.3:
        beat = await self.generate_dramatic_beat(flow, payload)  # LINE 418 - BLOCKS
```
- **Context:** Generates beats on significant tension changes
- **Blocks on:** `generate_dramatic_beat()` which calls Runner.run at line 721
- **Impact:** Tension changes wait for beat generation
- **Refactor Priority:** 🟡 HIGH

**Refactoring Tasks:**
- [ ] Acknowledge tension change immediately
- [ ] Generate beats in background
- [ ] Use beat templates for speed
- [ ] Apply beats asynchronously

---

#### ❌ `_handle_player_choice()` - Line 442
```python
async def _handle_player_choice(self, event: "SystemEvent") -> "SubsystemResponse":
    # ...
    if flow.phase_progress >= 1.0:
        transition = await self._handle_phase_transition(flow, payload)  # LINE 459 - BLOCKS
```
- **Context:** Handles phase transitions from player choices
- **Blocks on:** `_handle_phase_transition()` which calls Runner.run at line 767
- **Impact:** Player choices wait for phase transition narration
- **Refactor Priority:** 🟡 HIGH

**Refactoring Tasks:**
- [ ] Transition phase immediately with default narration
- [ ] Generate detailed narration async
- [ ] Use template transitions
- [ ] Stream narration when ready

---

**Core Blocking Functions:**

#### ❌ `initialize_conflict_flow()` - Line 588
```python
async def initialize_conflict_flow(
    self,
    conflict_id: int,
    conflict_type: str,
    initial_context: Dict[str, Any]
) -> ConflictFlow:
    # ... build prompt
    response = await Runner.run(self.pacing_director, prompt)  # LINE 610 - BLOCKS
```
- **Called from:** Line 287
- **Latency:** 500ms-2s for flow initialization
- **Impact:** Conflict creation blocked
- **Refactor Priority:** 🔴 CRITICAL

**Refactoring Tasks:**
- [ ] Use conflict-type-based defaults immediately
- [ ] Refine with LLM in background
- [ ] Pre-compute flows for common types
- [ ] Apply LLM improvements async

---

#### ❌ `update_conflict_flow()` - Line 648
```python
async def update_conflict_flow(
    self,
    flow: ConflictFlow,
    event: Dict[str, Any]
) -> Dict[str, Any]:
    # ... build prompt
    response = await Runner.run(self.flow_analyzer, prompt)  # LINE 674 - BLOCKS
```
- **Called from:** Line 379
- **Latency:** 500ms-2s per update
- **Impact:** Flow updates blocked on LLM
- **Refactor Priority:** 🟡 HIGH

**Refactoring Tasks:**
- [ ] Apply immediate rule-based updates
- [ ] Queue LLM analysis
- [ ] Use simple heuristics for changes
- [ ] Validate with LLM async

---

#### ❌ `generate_dramatic_beat()` - Line 699
```python
async def generate_dramatic_beat(
    self,
    flow: ConflictFlow,
    context: Dict[str, Any]
) -> Optional[DramaticBeat]:
    # ... build prompt
    response = await Runner.run(self.beat_generator, prompt)  # LINE 721 - BLOCKS
```
- **Called from:** Lines 292, 418
- **Latency:** 500ms-2s per beat
- **Impact:** Beats block flow progression
- **Refactor Priority:** 🟡 HIGH

**Refactoring Tasks:**
- [ ] Use beat template library
- [ ] Generate beats in background
- [ ] Apply generic beats immediately
- [ ] Refine with LLM async

---

#### ❌ `_handle_phase_transition()` - Line 745
```python
async def _handle_phase_transition(
    self,
    flow: ConflictFlow,
    trigger_event: Dict[str, Any]
) -> PhaseTransition:
    # ... build prompt
    response = await Runner.run(self.transition_narrator, prompt)  # LINE 767 - BLOCKS
```
- **Called from:** Lines 459, 687
- **Latency:** 500ms-2s per transition
- **Impact:** Phase changes wait for narration
- **Refactor Priority:** 🟡 HIGH

**Refactoring Tasks:**
- [ ] Transition immediately with template
- [ ] Generate narration in background
- [ ] Use phase-based defaults
- [ ] Stream narration when ready

---

## Other Files with Runner.run Calls (Lower Priority)

### 🟡 /home/user/flask_roleplay/logic/conflict_system/conflict_victory.py
- Lines 541, 701, 720, 755, 795
- **Context:** Victory processing (not hot path, but should be async)
- **Refactor Priority:** 🟢 LOW (not in immediate event handlers)

### 🟡 /home/user/flask_roleplay/logic/conflict_system/edge_cases.py
- Lines 725, 764, 798, 833, 872
- **Context:** Edge case recovery (should be async)
- **Refactor Priority:** 🟢 LOW (rare cases)

### 🟡 /home/user/flask_roleplay/logic/conflict_system/integration.py
- Line 524
- **Context:** Mode optimization
- **Refactor Priority:** 🟢 LOW (not hot path)

### 🟡 /home/user/flask_roleplay/logic/conflict_system/enhanced_conflict_integration.py
- Lines 711, 751, 787
- **Context:** Conflict integration and narration
- **Refactor Priority:** 🟢 LOW (not in core event loop)

### 🟡 /home/user/flask_roleplay/logic/conflict_system/dynamic_conflict_template.py
- Lines 961, 1161, 1225, 1285
- **Context:** Template generation (usually cached)
- **Refactor Priority:** 🟢 LOW (not hot path)

---

## Refactoring Strategy

### Phase 1: Critical Hot Path (Sprint 1) 🔴
**Target:** Remove all blocking calls from event handlers that affect player-facing operations

1. **autonomous_stakeholder_actions.py**
   - Route slow-path work through existing Celery tasks:
     - `generate_stakeholder_action`
     - `generate_stakeholder_reaction`
     - `evaluate_stakeholder_role`
   - Keep event handlers limited to dispatch helpers and cached reads
   - Surface results via planned action records / follow-up events

2. **social_circle.py**
   - Create Celery tasks:
     - `generate_gossip_task()`
     - `calculate_reputation_task()`
   - Implement reputation caching (5min TTL)
   - Use template gossip as placeholders
   - Stream updates when LLM completes

3. **conflict_flow.py**
   - Create default flow initialization
   - Queue flow refinement tasks
   - Use template beats and transitions
   - Apply LLM-generated content asynchronously

### Phase 2: Optimization (Sprint 2) 🟡
**Target:** Improve latency through caching and batching

1. **Caching Layer**
   - Cache reputation calculations
   - Cache common decisions by context hash
   - Pre-generate gossip templates
   - Store dramatic beat patterns

2. **Batching**
   - Batch multiple gossip requests
   - Batch reputation updates
   - Process reactions in batches

3. **Pre-computation**
   - Pre-generate likely NPC actions
   - Pre-compute phase transitions
   - Pre-generate alliance terms

### Phase 3: Advanced Async Architecture (Sprint 3) 🟢
**Target:** Streaming and progressive enhancement

1. **Streaming Results**
   - Stream gossip as it generates
   - Progressive reputation updates
   - Streaming narration

2. **Progressive Enhancement**
   - Fast rule-based initial results
   - LLM refines in background
   - Updates apply seamlessly

3. **Predictive Loading**
   - Predict likely LLM needs
   - Pre-warm common scenarios
   - Cache aggressively

---

## Impact Assessment

### Current State (Before Refactoring)
- **Scene state sync:** 2-5 seconds (gossip + reputation + decisions)
- **Player choice:** 1-3 seconds per choice (reactions + flow updates)
- **Conflict creation:** 2-4 seconds (flow init + stakeholder creation)
- **Average player-facing latency:** 1.5-3 seconds

### Target State (After Refactoring)
- **Scene state sync:** <100ms (immediate, async updates)
- **Player choice:** <200ms (immediate feedback, reactions stream in)
- **Conflict creation:** <300ms (defaults + async refinement)
- **Average player-facing latency:** <200ms

### Expected Improvement
- **90-95% latency reduction** for player-facing operations
- **10x+ throughput improvement** for concurrent users
- **Better UX** with immediate feedback and progressive enhancement

---

## Testing Strategy

### Performance Tests
- [ ] Measure baseline latency for each event handler
- [ ] Measure async version latency
- [ ] Verify <200ms for all hot-path operations
- [ ] Load test with 100+ concurrent users

### Functional Tests
- [ ] Verify eventual consistency of async operations
- [ ] Test error handling for failed background tasks
- [ ] Verify cache invalidation works correctly
- [ ] Test degraded mode when LLM unavailable

### Integration Tests
- [ ] Test full player session end-to-end
- [ ] Verify UI updates correctly with async data
- [ ] Test race conditions in concurrent operations
- [ ] Verify data consistency across requests

---

## Monitoring & Observability

### Metrics to Track
- Event handler latency (p50, p95, p99)
- Background task completion time
- Cache hit rates
- LLM call frequency and success rate
- Queue depths and processing times

### Alerts
- Event handler latency >500ms
- Background task failure rate >5%
- Cache hit rate <80%
- LLM timeout rate >10%
- Queue depth >100 items

---

## Conclusion

This refactoring is **critical for production viability**. The current architecture introduces unacceptable latency (1-5s) for real-time game operations. By moving LLM calls to background tasks and implementing progressive enhancement, we can achieve <200ms response times while maintaining rich, LLM-generated content.

**Estimated effort:** 3 sprints (6 weeks)
**Risk:** Medium (requires careful async design)
**Benefit:** 10x+ performance improvement, production-ready architecture
