import asyncio
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Iterable
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

    async def maybe_async(fn: Callable, *args, **kw):
        res = fn(*args, **kw)
        if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
            return await res
        return res

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

    def register_unconscious(self, name: str, fn: Callable, *, threshold: float = 0.5):
        self._pending_unconscious.append((name, fn, threshold))

    # default unconscious monitor (optional override)
    async def unconscious_monitor(self, view):
        return None


# ---------------------------------------------------------------------------
# ENGINE V3 (conscious + unconscious)
# ---------------------------------------------------------------------------

class NyxEngineV3(NyxEngineV2):
    def __init__(self, modules: List[WorkspaceModule], *, hz: float = 10.0,
                 enable_unconscious: bool = True, **kw):
        super().__init__(modules, hz, **kw)
        self.enable_unconscious = enable_unconscious
        self.unconscious: Optional[UnconsciousLayer] = None
        if enable_unconscious:
            self.unconscious = UnconsciousLayer(self.ws)

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
        await super().stop()

    async def process_input(self, text: str) -> str:
        # reflex first
        if self.enable_unconscious and self.unconscious:
            reflex = await self.unconscious.check_reflexes(text)
            if reflex:
                return reflex
        # otherwise conscious path
        return await super().process_input(text)


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

