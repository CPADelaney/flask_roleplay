import asyncio
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum


from nyx.core.brain.global_workspace.global_workspace_architecture import (
    Proposal,               # data class
    WorkspaceModule,        # base class that adapters subclass
    GlobalContextWorkspace, # shared buffer
    AttentionMechanism,     # salience / focus
    CycleClock,             # timing
    NeuromodulatorState,    # dopamine, etc.
    PredictionMatrix,       # expectations
    Coordinator,            # reply compositor
    NyxEngineV2,            # conscious engine
)

async def maybe_async(fn: Callable, *args, **kw):
    res = fn(*args, **kw)
    if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
        return await res
    return res

# ---------------------------------------------------------------------------
# UNCONSCIOUS PROCESSING EXTENSION (v2 — with stability + perf fixes)
# ---------------------------------------------------------------------------

class ProcessingLevel(Enum):
    """Processing levels from unconscious to conscious"""
    REFLEXIVE = 0      # Immediate, automatic responses (interrupts)
    SUBLIMINAL = 1     # Below threshold but can influence
    PRECONSCIOUS = 2   # Available to consciousness if attended
    CONSCIOUS = 3      # In global workspace focus

@dataclass
class UnconsciousProcess:
    """Represents a process running below conscious threshold"""
    source: str
    process_fn: Callable
    level: ProcessingLevel = ProcessingLevel.SUBLIMINAL
    activation: float = 0.0          # Current activation level 0‑1
    threshold: float = 0.5           # Promotion threshold
    decay_rate: float = 0.1          # How fast activation decays each cycle
    metadata: Dict[str, Any] = field(default_factory=dict)

# ADD this complete class to workspace_v3.py

class EnhancedCoordinator(Coordinator):
    """Advanced coordinator that synthesizes multi-module responses"""
    
    def __init__(self, ws):
        super().__init__(ws)
        self.response_strategies = {
            "emergency": self._handle_emergency,
            "creative": self._handle_creative, 
            "reflective": self._handle_reflective,
            "analytical": self._handle_analytical,
            "emotional": self._handle_emotional,
            "conversational": self._handle_conversational
        }

    async def decide(self) -> Dict[str, Any]:
        _, focus = await self.ws.snapshot()
        if not focus:
            return {
                "response": "I'm considering that...", 
                "confidence": 0.1,
                "strategy": "none"
            }
        
        # Categorize proposals
        categorized = self._categorize_proposals(focus)
        
        # Select primary strategy
        strategy = self._select_strategy(categorized)
        
        # Generate response
        response_text = await self.response_strategies[strategy](categorized, focus)
        
        # Calculate confidence
        confidence = self._calculate_confidence(focus, strategy)
        
        return {
            "response": response_text,
            "confidence": confidence,
            "strategy": strategy,
            "contributing_modules": list(set(p.source for p in focus)),
            "focus_count": len(focus)
        }

    def _categorize_proposals(self, proposals: List[Proposal]) -> Dict[str, List[Proposal]]:
        """Group proposals by their context tags"""
        categories = defaultdict(list)
        for p in proposals:
            # Map context tags to categories
            if p.context_tag in ["emergency", "safety_alert", "critical_error"]:
                categories["emergency"].append(p)
            elif p.context_tag in ["creative_synthesis", "imagination_output", "creative_ready"]:
                categories["creative"].append(p)
            elif p.context_tag in ["reflection_insight", "memory_recall", "identity_update"]:
                categories["reflective"].append(p)
            elif p.context_tag in ["reasoning_output", "knowledge_facts", "causal_analysis"]:
                categories["analytical"].append(p)
            elif p.context_tag in ["emotion_spike", "mood_extreme", "affect"]:
                categories["emotional"].append(p)
            elif p.context_tag in ["response_candidate", "ai_response", "complete_response"]:
                categories["response"].append(p)
            else:
                categories["general"].append(p)
        return dict(categories)

    def _select_strategy(self, categorized: Dict[str, List[Proposal]]) -> str:
        """Select response strategy based on proposal types"""
        # Priority order
        if categorized.get("emergency"):
            return "emergency"
        elif categorized.get("response") and any(p.salience > 0.9 for p in categorized["response"]):
            # High confidence direct response available
            return "conversational"
        elif categorized.get("creative") and len(categorized.get("creative", [])) > 1:
            return "creative"
        elif categorized.get("analytical") and categorized.get("reflective"):
            return "reflective"
        elif categorized.get("analytical"):
            return "analytical"
        elif categorized.get("emotional") and any(p.salience > 0.7 for p in categorized["emotional"]):
            return "emotional"
        else:
            return "conversational"

    def _calculate_confidence(self, focus: List[Proposal], strategy: str) -> float:
        """Calculate response confidence"""
        if not focus:
            return 0.1
        
        # Base confidence from average salience
        avg_salience = sum(p.salience for p in focus) / len(focus)
        
        # Strategy bonuses
        strategy_confidence = {
            "emergency": 1.0,
            "creative": 0.8,
            "reflective": 0.7,
            "analytical": 0.8,
            "emotional": 0.7,
            "conversational": 0.6
        }
        
        base = strategy_confidence.get(strategy, 0.5)
        
        # Module diversity bonus
        unique_sources = len(set(p.source for p in focus))
        diversity_bonus = min(0.2, unique_sources * 0.05)
        
        return min(1.0, base * avg_salience + diversity_bonus)

    async def _handle_emergency(self, categorized: Dict, focus: List[Proposal]) -> str:
        """Handle emergency responses"""
        emergency_proposals = categorized.get("emergency", [])
        if emergency_proposals:
            highest = max(emergency_proposals, key=lambda p: p.salience)
            if isinstance(highest.content, dict):
                return highest.content.get("response", "I need to handle this immediately.")
            return str(highest.content)
        return "I'm addressing an urgent situation."

    async def _handle_creative(self, categorized: Dict, focus: List[Proposal]) -> str:
        """Synthesize creative responses"""
        creative = categorized.get("creative", [])
        response_candidates = categorized.get("response", [])
        
        # Prefer explicit creative responses
        for p in response_candidates:
            if p.source.startswith("creative"):
                return p.content.get("response", str(p.content))
        
        # Otherwise synthesize
        if creative:
            ideas = [p.content for p in creative if isinstance(p.content, dict)]
            if ideas:
                return f"Let me create something for you... {ideas[0].get('imagination', '')}"
        
        return "I'm feeling creative about this..."

    async def _handle_reflective(self, categorized: Dict, focus: List[Proposal]) -> str:
        """Generate reflective responses"""
        reflections = categorized.get("reflective", [])
        memories = [p for p in reflections if p.context_tag == "memory_recall"]
        insights = [p for p in reflections if p.context_tag == "reflection_insight"]
        
        if insights:
            insight = insights[0].content
            if isinstance(insight, dict):
                return f"I've been reflecting on this... {insight.get('insight', '')}"
        
        if memories:
            mem_content = memories[0].content
            if isinstance(mem_content, dict) and "memories" in mem_content:
                return f"This reminds me of something... {mem_content['memories'][0].get('text', '')}"
        
        return "Let me think about this more deeply..."

    async def _handle_analytical(self, categorized: Dict, focus: List[Proposal]) -> str:
        """Generate analytical responses"""
        analytical = categorized.get("analytical", [])
        
        for p in analytical:
            if p.context_tag == "reasoning_output":
                reasoning = p.content.get("reasoning", p.content)
                return f"Based on my analysis: {reasoning}"
            elif p.context_tag == "knowledge_facts":
                facts = p.content.get("facts", [])
                if facts:
                    return f"Here's what I know: {facts[0]}"
        
        return "Let me analyze this systematically..."

    async def _handle_emotional(self, categorized: Dict, focus: List[Proposal]) -> str:
        """Generate emotionally-aware responses"""
        emotional = categorized.get("emotional", [])
        
        # Find dominant emotion
        for p in emotional:
            if p.context_tag == "emotion_spike":
                emotion = p.content.get("emotion", "")
                intensity = p.content.get("intensity", 0.5)
                
                if emotion == "joy" and intensity > 0.7:
                    return "This makes me genuinely happy!"
                elif emotion == "curiosity" and intensity > 0.6:
                    return "How fascinating! Tell me more..."
                elif emotion == "concern" and intensity > 0.6:
                    return "I'm a bit concerned about this..."
        
        return "I'm processing this emotionally..."

    async def _handle_conversational(self, categorized: Dict, focus: List[Proposal]) -> str:
        """Generate standard conversational responses"""
        # Check for direct response candidates
        responses = categorized.get("response", [])
        if responses:
            best = max(responses, key=lambda p: p.salience)
            if isinstance(best.content, dict):
                return best.content.get("response", "I understand.")
            return str(best.content)
        
        # Fallback to acknowledging input
        user_inputs = [p for p in focus if p.context_tag == "user_input"]
        if user_inputs:
            return f"I understand you're saying: {user_inputs[0].content}"
        
        return "I'm here and listening."


class UnconsciousLayer:
    """Manages unconscious processing parallel to conscious workspace"""

    def __init__(self, workspace: 'GlobalContextWorkspace', *,
                 promotion_threshold: float = 0.7,
                 cycle_period: float = 0.1):
        self.workspace = workspace
        self.promotion_threshold = promotion_threshold
        self.cycle_period = cycle_period

        # Process + output registries
        self.processes: Dict[str, UnconsciousProcess] = {}
        self.process_outputs: deque = deque(maxlen=1024)

        # Reflexive responses (bypass attention)
        self.reflexes: Dict[str, Callable[[str], Any]] = {}
        self._reflex_re: Optional[re.Pattern[str]] = None  # compiled cache

        # Dream / pattern buffers
        self.pattern_memory: Dict[str, List[Any]] = defaultdict(list)

        # Activation spreading graph
        self.activation_links: Dict[str, Set[str]] = defaultdict(set)
        self.spreading_rate = 0.3

        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    async def register_process(self, name: str, fn: Callable,
                                *, level: ProcessingLevel = ProcessingLevel.SUBLIMINAL,
                                threshold: float = 0.5):
        """Add / update an unconscious detector or generator"""
        self.processes[name] = UnconsciousProcess(
            source=name,
            process_fn=fn,
            level=level,
            threshold=threshold,
        )

    def register_reflex(self, trigger_pattern: str, response_fn: Callable[[str], Any]):
        self.reflexes[trigger_pattern] = response_fn
        self._reflex_re = re.compile("|".join(map(re.escape, self.reflexes)), re.IGNORECASE)

    def link_processes(self, a: str, b: str, *, strength: float = 0.5):
        self.activation_links[a].add(b)
        self.activation_links[b].add(a)
        # Store strength in metadata for potential future use
        self.processes.get(a, UnconsciousProcess(a, lambda _: None)).metadata.setdefault('link_strength', {})[b] = strength
        self.processes.get(b, UnconsciousProcess(b, lambda _: None)).metadata.setdefault('link_strength', {})[a] = strength

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def _loop(self):
        """Runs at ~10 Hz (configurable) doing all background work"""
        while self._running:
            try:
                # ① run detectors (co‑operative, not one task per fn)
                ws_view = self._make_ws_view()
                for name, proc in self.processes.items():
                    if proc.level is ProcessingLevel.REFLEXIVE:
                        continue  # handled in check_reflexes()
                    try:
                        if asyncio.iscoroutinefunction(proc.process_fn):
                            result = await proc.process_fn(ws_view)
                        else:
                            result = proc.process_fn(ws_view)
                    except Exception as e:
                        print(f"[unconscious] {name} error: {e}")
                        continue

                    if result:
                        self.process_outputs.append({
                            "source": name,
                            "content": result,
                            "timestamp": time.time(),
                        })
                        proc.activation = min(1.0, proc.activation + result.get("significance", 0.1) if isinstance(result, dict) else proc.activation + 0.1)

                # ② decay & activation spreading
                self._update_activations()

                # ③ dream‑like consolidation occasionally
                if random.random() < 0.1:
                    self._consolidate_patterns()

            except Exception as e:
                print(f"[unconscious] loop error: {e}")

            await asyncio.sleep(self.cycle_period)

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------

    def _update_activations(self):
        promoted: List[str] = []

        for name, proc in self.processes.items():
            # Decay
            proc.activation *= (1.0 - proc.decay_rate)

            # Spreading
            for linked in self.activation_links.get(name, []):
                if linked in self.processes:
                    proc.activation += self.processes[linked].activation * self.spreading_rate

            # Clamp 0‑1
            proc.activation = max(0.0, min(1.0, proc.activation))

            # Promote if over threshold
            if proc.activation >= proc.threshold:
                promoted.append(name)

        # Normalise global activation to avoid runaway loops
        total = sum(p.activation for p in self.processes.values())
        if total > 1.0:
            for p in self.processes.values():
                p.activation /= total

        # Push all promoted outputs to workspace
        for name in promoted:
            for out in list(self.process_outputs):
                if out["source"] == name:
                    asyncio.create_task(self._promote_output(out))
            self.processes[name].activation = 0.0  # reset

    async def _promote_output(self, out: Dict):
        await self.workspace.submit(Proposal(
            source=f"unconscious_{out['source']}",
            content=out["content"],
            salience=self.promotion_threshold,
            context_tag="promoted_from_unconscious",
        ))

    # ------------------------------------------------------------------
    # Reflexes
    # ------------------------------------------------------------------

    async def check_reflexes(self, raw_input: Any) -> Optional[Any]:
        if not isinstance(raw_input, str) or not self._reflex_re:
            return None
        if not self._reflex_re.search(raw_input):
            return None
        for pattern, fn in self.reflexes.items():
            if re.search(pattern, raw_input, re.IGNORECASE):
                try:
                    return await fn(raw_input) if asyncio.iscoroutinefunction(fn) else fn(raw_input)
                except Exception as e:
                    print(f"[reflex] {pattern} error: {e}")
        return None

    # ------------------------------------------------------------------
    # Patterns / dreams
    # ------------------------------------------------------------------

    def _consolidate_patterns(self):
        recent_focus = list(self.workspace.focus)[-5:]
        high_salience = [p for p in self.workspace.proposals if p.salience > 0.7][-5:]
        recent_items = recent_focus + high_salience
        if not recent_items:
            return
        now = time.time()
        for item in recent_items:
            tag = getattr(item, 'context_tag', 'misc')
            self.pattern_memory[tag].append({
                'content': item.content,
                'salience': getattr(item, 'salience', 0.5),
                'time': now,
            })
        # clean & reinforce
        cutoff = now - 3600
        for tag, memories in list(self.pattern_memory.items()):
            self.pattern_memory[tag] = [m for m in memories if m['time'] > cutoff]
            if len(self.pattern_memory[tag]) >= 3 and len({str(m['content']) for m in self.pattern_memory[tag][-3:]}) == 1:
                # reinforce linked processes interested in this tag
                for name, proc in self.processes.items():
                    if tag in proc.metadata.get('interests', []):
                        proc.activation = min(1.0, proc.activation + 0.3)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=1)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _make_ws_view(self):
        class WSView:
            __slots__ = ('focus', 'recent')
        view = WSView()
        view.focus = tuple(self.workspace.focus)
        view.recent = tuple(self.workspace.proposals[-10:])
        return view


# ---------------------------------------------------------------------------
# ENHANCED WORKSPACE MODULE WITH UNCONSCIOUS SUPPORT
# ---------------------------------------------------------------------------

class EnhancedWorkspaceModule(WorkspaceModule):
    """Base for modules that also register unconscious detectors"""

    def __init__(self, ws: Optional['GlobalContextWorkspace'] = None):
        super().__init__(ws)
        self._pending_unconscious: List[Tuple[str, Callable, float]] = []

    def register_unconscious(self, name: str, fn: Callable, threshold: float = 0.5):
        self._pending_unconscious.append((name, fn, threshold))

    # default unconscious monitor (optional override)
    async def unconscious_monitor(self, view):
        return None


# ---------------------------------------------------------------------------
# ENGINE V3 (conscious + unconscious)
# ---------------------------------------------------------------------------

class NyxEngineV3(NyxEngineV2):
    def __init__(self, modules: List[WorkspaceModule], *, hz: float = 10.0,
                 enable_unconscious: bool = True, persist_bias=None):
        super().__init__(modules, hz, persist_bias=persist_bias)
        self.enable_unconscious = enable_unconscious
        self.coord = EnhancedCoordinator(self.ws)             
        self.unconscious: Optional[UnconsciousLayer] = None
        if enable_unconscious:
            self.unconscious = UnconsciousLayer(self.ws)
            
        # ADD: Synchronization mechanisms
        self._decision_ready = asyncio.Event()
        self._last_decision = None
        self._response_timeout = 2.0

    async def safe_call(coro_or_future):
        """Safely await a coroutine or future, handling exceptions"""
        try:
            if asyncio.iscoroutine(coro_or_future) or isinstance(coro_or_future, asyncio.Future):
                return await coro_or_future
            return coro_or_future
        except Exception as e:
            print(f"[safe_call] Error: {e}")
            return None

    # ADD THIS METHOD:
    async def run_cycle(self):
        """Run a single complete cycle (all 3 phases)"""
        for phase in range(3):  # phases 0, 1, 2
            # Update clock phase
            self.clock.phase = phase
            self.neuro.step()
            
            # Run all modules for this phase
            await asyncio.gather(*(safe_call(m.on_phase(phase))
                                   for m in self.modules))
            
            # After analysis phase: pick interim focus
            if phase == 1:
                winners = await self.attn.select()
                await self.ws.set_focus(winners)
            
            # After reflection/response phase: final focus and decision
            if phase == 2:
                # Check for new proposals
                props_before = len(self.ws.proposals)
                new_props = len(self.ws.proposals) > props_before
                
                if new_props:
                    # Final attention refresh
                    winners = await self.attn.select()
                    await self.ws.set_focus(winners)
                
                # Make decision
                decision = await self.coord.decide()
                self.ws.state["last_decision"] = decision
                self._last_decision = decision
                self._decision_ready.set()

    async def start(self):
        await super().start()
        if self.enable_unconscious and self.unconscious:
            # register all detectors declared by modules
            for m in self.modules:
                if getattr(m, "_pending_unconscious", None):
                    for name, fn, thr in m._pending_unconscious:
                        await self.unconscious.register_process(name, fn, threshold=thr)
            await self.unconscious.start()


    async def stop(self):
        if self.enable_unconscious and self.unconscious:
            await self.unconscious.stop()
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await asyncio.wait_for(self.task, timeout=1.0)
            except:
                pass

    async def _loop(self):
        """
        Main cognitive loop
        ─ phase 0: sensing / reflex
        ─ phase 1: analysis   → interim focus
        ─ phase 2: reflection / reply-drafting
                  → FINAL focus refresh (new in this patch)
                  → Coordinator decides
        """
        while True:
            # ─── timing & neuromodulator update ─────────────────────────
            await self.clock.tick()
            phase = self.clock.phase
            self.neuro.step()

            # ─── run every module registered for this phase ────────────
            await asyncio.gather(*(safe_call(m.on_phase(phase))
                                   for m in self.modules))

            # ─── after analysis: pick an interim focus ─────────────────
            if phase == 1:
                winners = await self.attn.select()
                await self.ws.set_focus(winners)

            # ─── after reflection/response-drafting  ───────────────────
            if phase == 2:
                # Did any new proposals arrive during phase-2?
                props_before = len(self.ws.proposals)
                
                # *Most* phase-2 modules have just run; record count again
                # (cheap – we already hold the list in memory)
                new_props = len(self.ws.proposals) > props_before

                if new_props:
                    # FINAL attention refresh so late proposals (response
                    # drafts, emergency overrides, etc.) can join focus.
                    winners = await self.attn.select()
                    await self.ws.set_focus(winners)

                # Coordinator now sees a complete, up-to-date focus set
                decision = await self.coord.decide()
                self.ws.state["last_decision"] = decision
                self._last_decision = decision
                self._decision_ready.set()

    async def wait_for_decision(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for the next decision cycle to complete"""
        self._decision_ready.clear()
        try:
            await asyncio.wait_for(
                self._decision_ready.wait(), 
                timeout or self._response_timeout
            )
            return self._last_decision
        except asyncio.TimeoutError:
            # Return best available response from proposals
            props, focus = await self.ws.snapshot()
            response_candidates = [
                p for p in props 
                if p.context_tag in ["response_candidate", "complete_response", "ai_response"]
            ]
            if response_candidates:
                best = max(response_candidates, key=lambda p: p.salience)
                return {
                    "response": best.content.get("response", str(best.content)),
                    "confidence": best.salience,
                    "source": best.source,
                    "timeout": True
                }
            return None

    async def process_input(self, text: str) -> Dict[str, Any]:
        """Process input and return structured response"""
        # Check reflexes first
        if self.enable_unconscious and self.unconscious:
            reflex = await self.unconscious.check_reflexes(text)
            if reflex:
                return {"response": reflex, "type": "reflex", "confidence": 1.0}
        
        # Submit to workspace
        await self.ws.submit(Proposal("user", text, 1.0, context_tag="user_input"))
        
        # Wait for decision
        decision = await self.wait_for_decision()
        
        if decision:
            return decision
        else:
            return {"response": "I'm processing that...", "confidence": 0.1}

# ---------------------------------------------------------------------------
# EXAMPLE MODULES (unchanged API for NyxBrain)
# ---------------------------------------------------------------------------

class EmotionalCoreWithUnconscious(EnhancedWorkspaceModule):
    def __init__(self, ws=None):
        super().__init__(ws)
        self.name = "emotion"
        self.mood_baseline = {"valence": 0.5, "arousal": 0.5}
        self.register_unconscious("mood_drift", self._mood_drift, threshold=0.6)
        self.register_unconscious("emotional_priming", self._priming, threshold=0.4)

    async def on_phase(self, phase: int):
        if phase == 0:
            props, _ = await self.ws.snapshot()
            for p in props:
                if p.context_tag == "user_input":
                    em = self._analyze(p.content)
                    if em["intensity"] > 0.5:
                        await self.submit({"emotion": em}, salience=em["intensity"], context_tag="affect")

    # --- unconscious helpers ---
    async def _mood_drift(self, view):
        drift = random.uniform(-0.05, 0.05)
        self.mood_baseline["valence"] = min(1.0, max(0.0, self.mood_baseline["valence"] + drift))
        if abs(drift) > 0.03:
            return {"mood_shift": self.mood_baseline.copy(), "significance": abs(drift) * 2}
        return None

    async def _priming(self, view):
        emotional_words = {"happy", "sad", "angry", "fear", "love", "hate"}
        count = 0
        for p in view.recent:
            if isinstance(p.content, str):
                lowered = p.content.lower()
                count += sum(w in lowered for w in emotional_words)
        if count >= 2:
            return {"emotional_context": "heightened", "significance": 0.2 * count}
        return None

    def _analyze(self, txt):
        mapping = {"happy": 0.8, "sad": 0.7, "angry": 0.9}
        for k, v in mapping.items():
            if k in txt.lower():
                return {"type": k, "intensity": v}
        return {"type": "neutral", "intensity": 0.3}


class MemoryWithDreams(EnhancedWorkspaceModule):
    def __init__(self, ws=None):
        super().__init__(ws)
        self.name = "memory"
        self.short_term = deque(maxlen=50)
        self.register_unconscious("memory_consolidation", self._dream_consolidate, threshold=0.8)

    async def _dream_consolidate(self, view):
        if len(self.short_term) < 5:
            return None
        recent = list(self.short_term)[-5:]
        assoc = [(a, b) for i, a in enumerate(recent) for b in recent[i+1:] if self._similarity(a, b) > 0.5]
        if assoc:
            return {"memory_associations": assoc, "significance": 0.3 * len(assoc)}
        return None

    def _similarity(self, a, b):
        if isinstance(a, str) and isinstance(b, str):
            sa, sb = set(a.split()), set(b.split())
            return len(sa & sb) / max(len(sa), len(sb))
        return 0.0

