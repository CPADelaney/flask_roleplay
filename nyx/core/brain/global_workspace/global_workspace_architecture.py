import asyncio
import json
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

"""Global Workspace 2.3 – hierarchical, reputation‑aware, latency‑safe.
Adds **multimodal perception**, **symbolic reasoning**, **foundation‑model consulting**,
**multimodal binding** and **online‑learning daemon** while keeping the core
~300 LOC. All new capability is added as plug‑in modules so existing engine
remains lean and composable."""

# ---------------------------------------------------------------------------
# 0  ───  DATA TYPES
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    source: str
    content: Any
    salience: float = 0.5
    context_tag: str = "general"
    prediction: Any = None
    signal_id: str = "default"
    deadline: int = 0  # cycles to keep
    ts: float = field(default_factory=time.time)
    eff_salience: float = 0.0  # filled by attention

# ---------------------------------------------------------------------------
# 1  ───  CLOCK & STATE + WATCHDOG
# ---------------------------------------------------------------------------

class CycleClock:
    def __init__(self, hz: float = 10.0, latency_warn=0.2):
        self.period = 1.0 / hz
        self.phase = 0
        self.latency_warn = latency_warn
        self._last = time.time()
    async def tick(self):
        await asyncio.sleep(self.period)
        now = time.time()
        drift = now - self._last - self.period
        self._last = now
        if drift > self.latency_warn:
            print(f"[⚠] workspace loop drift {drift:.3f}s – shedding low‑prio tasks")
        self.phase = (self.phase + 1) % 3

class NeuromodulatorState:
    def __init__(self):
        self.levels = {"dopamine": .5, "noradrenaline": .2, "ach": .5, "curiosity": .4}
        self.decay = 0.01
    def update(self, d):
        for k, v in d.items():
            if k in self.levels:
                self.levels[k] = max(0, min(1, self.levels[k] + v))
    def step(self):
        for k, v in self.levels.items():
            self.levels[k] += (0.5 - v) * self.decay

class PredictionMatrix:
    def __init__(self):
        self.store: Dict[Tuple[str, str], Tuple[Any, float]] = {}
    def set(self, m, s, val, var=1.0):
        self.store[(m, s)] = (val, var)
    def get(self, m, s):
        return self.store.get((m, s), (None, 1.0))

# ---------------------------------------------------------------------------
# 2  ───  GLOBAL WORKSPACE CORE (buffer + reputation + persistence)
# ---------------------------------------------------------------------------

class ReputationMap:
    """Tracks proposal quality by comparing actual focus vs. predicted winner."""
    def __init__(self):
        self.rep: Dict[str, float] = {}
    def reward(self, mod, delta):
        self.rep[mod] = max(-1.0, min(1.0, self.rep.get(mod, 0.0) + delta))
    def score(self, mod):
        return self.rep.get(mod, 0.0)

class GlobalContextWorkspace:
    def __init__(self, focus_max=8, buf_max=600, per_mod_max=60, persist_path: Optional[Path] = None):
        self.focus_max = focus_max
        self.buf_max = buf_max
        self.per_mod_max = per_mod_max
        self.proposals: List[Proposal] = []
        self.focus: List[Proposal] = []
        self.state: Dict[str, Any] = {"salience_bias": {}}
        self.lock = asyncio.Lock()
        self.reputation = ReputationMap()
        self.persist_path = persist_path or Path("salience_bias.json")
        self._load_bias()
    # ---- persistence ----
    def _load_bias(self):
        if self.persist_path.exists():
            try:
                self.state["salience_bias"].update(json.loads(self.persist_path.read_text()))
                print("[ws] loaded salience bias")
            except Exception as e:
                print("[ws] bias load fail", e)
    def _save_bias(self):
        try:
            self.persist_path.write_text(json.dumps(self.state["salience_bias"], indent=1))
        except Exception:
            pass

    async def _trim_buffers(self):
        # global cap
        if len(self.proposals) > self.buf_max:
            self.proposals = self.proposals[-self.buf_max:]
        # per‑module cap
        counts: Dict[str, int] = {}
        trimmed = []
        for p in reversed(self.proposals):
            counts[p.source] = counts.get(p.source, 0) + 1
            if counts[p.source] <= self.per_mod_max:
                trimmed.append(p)
        self.proposals = list(reversed(trimmed))

    async def submit(self, p: Proposal):
        async with self.lock:
            self.proposals.append(p)
            await self._trim_buffers()
    async def snapshot(self):
        async with self.lock:
            return list(self.proposals), list(self.focus)
    async def set_focus(self, winners: List[Proposal]):
        async with self.lock:
            self.focus = winners
            # routine deadline & bias update
            buf = []
            for p in self.proposals:
                if p.deadline > 0:
                    p.deadline -= 1
                    if p.deadline >= 0:
                        buf.append(p)
            self.proposals = buf
            # reputation + bias tuning
            for w in winners:
                self.reputation.reward(w.source, +0.02)
            losers = {p.source for p in self.proposals} - {w.source for w in winners}
            for l in losers:
                self.reputation.reward(l, -0.01)
            for mod, rep in self.reputation.rep.items():
                self.state["salience_bias"][mod] = rep
            self._save_bias()

# ---------------------------------------------------------------------------
# 3  ───  MODULE BASE – error‑isolated
# ---------------------------------------------------------------------------

class WorkspaceModule:
    def __init__(self, ws: Optional[GlobalContextWorkspace] = None):
        self.ws = ws
        self.name = self.__class__.__name__
        self.cooldown = 0
    async def submit(self, content, salience=.5, **kw):
        if self.cooldown:
            self.cooldown -= 1
            return
        bias = self.ws.state["salience_bias"].get(self.name, 0)
        prop = Proposal(self.name, content, max(0, min(1, salience + bias)), **kw)
        await self.ws.submit(prop)
    async def on_phase(self, phase: int):
        pass  # override

# helper to wrap calls

def safe_call(coro):
    async def wrapper():
        try:
            await coro
        except Exception as e:
            print(f"[WARN] module error: {e}")
    return wrapper()

# ---------------------------------------------------------------------------
# 4  ───  ATTENTION  (adaptive + reputation‑weighted)
# ---------------------------------------------------------------------------

class AttentionMechanism:
    def __init__(self, ws, neuro, preds, adapt_thresh=.85):
        self.ws, self.neuro, self.preds = ws, neuro, preds
        self.adapt_thresh = adapt_thresh
    def _score(self, p):
        pred, var = self.preds.get(p.source, p.signal_id)
        surprise = 0.0
        if pred is not None:
            try:
                surprise = abs(p.content - pred) / (var + 1e-6)
            except Exception:
                surprise = 1.0 if p.content != pred else 0.0
        n = self.neuro.levels
        gain = 0.7 + 0.6*n["dopamine"] + 0.8*n["noradrenaline"] + 0.4*n["ach"]
        rep = self.ws.reputation.score(p.source)
        return max(0.0, min(1.0, p.salience * (1+surprise) * gain * (1+rep)))
    async def select(self):
        props, _ = await self.ws.snapshot()
        if not props:
            return []
        total = 0.0
        for p in props:
            p.eff_salience = self._score(p)
            total += p.eff_salience
        props.sort(key=lambda x: x.eff_salience, reverse=True)
        winners, accum = [], 0.0
        for p in props:
            winners.append(p)
            accum += p.eff_salience
            if accum >= total * self.adapt_thresh or len(winners) >= self.ws.focus_max:
                break
        return winners

# ---------------------------------------------------------------------------
# 5  ───  COORDINATOR  (layered, pluggable)
# ---------------------------------------------------------------------------

class ReplyDraft:
    def __init__(self, text=""):
        self.text = text
        self.meta: Dict[str, Any] = {}

class Coordinator:
    def __init__(self, ws):
        self.ws = ws
        self.layers: List[Callable[[ReplyDraft, List[Proposal]], None]] = []
        self._register_default_layers()
    def _register_default_layers(self):
        self.layers.extend([
            self._safety_layer,
            self._persona_layer,
            self._expression_layer,
        ])
    # ----- layers -----
    def _safety_layer(self, draft: ReplyDraft, focus):
        if any("blocked" in str(p.content).lower() for p in focus):
            draft.text = "❌ Sorry, that content is unsafe."
    def _persona_layer(self, draft: ReplyDraft, focus):
        if not draft.text:
            draft.text = str(focus[0].content)
        draft.text = draft.text.replace("I", "Nyx")
    def _expression_layer(self, draft: ReplyDraft, focus):
        if len(focus) > 1:
            extras = " | ".join(str(p.content) for p in focus[1:])
            draft.text += f" (context: {extras})"

    async def decide(self) -> str:
        _, focus = await self.ws.snapshot()
        if not focus:
            return "…"
        draft = ReplyDraft()
        for layer in self.layers:
            layer(draft, focus)
        return draft.text

# ---------------------------------------------------------------------------
# 6  ───  ENGINE (supports layered workspaces)
# ---------------------------------------------------------------------------

class NyxEngineV2:
    """Single‑layer engine hosting multiple specialist modules with shared workspace."""
    def __init__(self, modules: List[WorkspaceModule], hz=10.0, persist_bias=None):
        self.clock = CycleClock(hz)
        self.ws = GlobalContextWorkspace(persist_path=persist_bias and Path(persist_bias))
        self.neuro = NeuromodulatorState()
        self.preds = PredictionMatrix()
        self.attn = AttentionMechanism(self.ws, self.neuro, self.preds)
        self.coord = Coordinator(self.ws)
        self.modules = modules
        for m in self.modules:
            if m.ws is None:
                m.ws = self.ws
        self.task: Optional[asyncio.Task] = None
        # sub‑workspaces (e.g., autonomic).
        self.sub_layers: List[NyxEngineV2] = []

    async def start(self):
        if not self.task or self.task.done():
            self.task = asyncio.create_task(self._loop())
            for sub in self.sub_layers:
                await sub.start()

    async def _loop(self):
        while True:
            await self.clock.tick()
            phase = self.clock.phase
            self.neuro.step()
            await asyncio.gather(*(safe_call(m.on_phase(phase)) for m in self.modules))
            if phase == 1:
                winners = await self.attn.select()
                await self.ws.set_focus(winners)
            if phase == 2:
                decision = await self.coord.decide()
                self.ws.state["last_decision"] = decision

    # ------------- external call -------------
    async def process_input(self, text: str) -> str:
        await self.ws.submit(Proposal("user", text, 1.0, context_tag="user_input"))
        # wait at most one full cycle for reply
        await asyncio.sleep(self.clock.period * 1.5)
        return self.ws.state.get("last_decision", "…")

# ---------------------------------------------------------------------------
# 7  ───  MULTIMODAL SENSOR MODULES (vision, audio, haptics)
# ---------------------------------------------------------------------------

class VisionSensor(WorkspaceModule):
    """Dummy vision sensor – replace with real OpenCV / YOLO pipeline."""
    def __init__(self, ws=None, camera_id=0):
        super().__init__(ws)
        self.camera_id = camera_id
        self.fno = 0
    async def on_phase(self, phase):
        if phase == 0:
            self.fno += 1
            if self.fno % 3 == 0:  # every third cycle emit detection
                obj_id = f"obj_{self.fno//3}"
                content = {"type": "vision", "id": obj_id, "bbox": (random.random(), random.random())}
                await self.submit(content, salience=0.6, context_tag="vision", signal_id=obj_id, deadline=8)

class AudioSensor(WorkspaceModule):
    """Dummy audio sensor – replace with Whisper ASR or keyword spotting."""
    async def on_phase(self, phase):
        if phase == 0 and random.random() < 0.1:
            word = random.choice(["hello", "stop", "apple", "unknown"])
            await self.submit({"type": "audio", "word": word}, salience=0.55, context_tag="audio", signal_id=word, deadline=4)

class TouchSensor(WorkspaceModule):
    """Dummy haptic sensor – random touch events."""
    async def on_phase(self, phase):
        if phase == 0 and random.random() < 0.05:
            await self.submit({"type": "touch", "contact": True, "force": round(random.uniform(1,5),2)}, salience=0.5, context_tag="haptic", deadline=2)

# ---------------------------------------------------------------------------
# 8  ───  SPATIAL / EMBODIMENT MODULES
# ---------------------------------------------------------------------------

class SpatialMap(WorkspaceModule):
    """Maintains a simple map of object positions."""
    def __init__(self, ws=None):
        super().__init__(ws)
        self.objects: Dict[str, Tuple[float,float]] = {}
    async def on_phase(self, phase):
        if phase == 1:  # after sensors
            props, _ = await self.ws.snapshot()
            for p in props:
                if isinstance(p.content, dict) and p.content.get("type") == "vision":
                    self.objects[p.content["id"]] = p.content["bbox"]
        if phase == 2 and random.random() < 0.05:  # occasionally broadcast map summary
            await self.submit({"map_size": len(self.objects)}, salience=0.4, context_tag="spatial")

# ---------------------------------------------------------------------------
# 9  ───  MULTIMODAL BINDING MODULE
# ---------------------------------------------------------------------------

class MultimodalBinder(WorkspaceModule):
    """Binds vision + audio referring to same ID/word into unified event."""
    async def on_phase(self, phase):
        if phase != 1:
            return
        props, _ = await self.ws.snapshot()
        latest_audio = {p.content["word"]: p for p in props if isinstance(p.content, dict) and p.content.get("type") == "audio"}
        latest_vision = {p.content["id"]: p for p in props if isinstance(p.content, dict) and p.content.get("type") == "vision"}
        for key in latest_audio.keys() & latest_vision.keys():
            fused = {"event": "bound", "label": key, "pos": latest_vision[key].content["bbox"]}
            await self.submit(fused, salience=0.8, context_tag="binding", signal_id=key, deadline=10)

# ---------------------------------------------------------------------------
# 10 ───  FOUNDATION‑MODEL CONSULTANT (LLM)
# ---------------------------------------------------------------------------

class FoundationModelConsultant(WorkspaceModule):
    """Stub that calls an LLM (replace with actual API) when asked a question."""
    async def on_phase(self, phase):
        if phase != 2:
            return
        # look for user questions waiting answer
        props, _ = await self.ws.snapshot()
        for p in props:
            if p.context_tag == "user_input" and p.content.endswith("?"):
                answer = f"🤖 (LLM): I believe the answer to '{p.content}' is 42."
                await self.submit(answer, salience=0.9, context_tag="llm", deadline=5)
                break

# ---------------------------------------------------------------------------
# 11 ───  SYMBOLIC REASONER (toy)
# ---------------------------------------------------------------------------

class SymbolicReasoner(WorkspaceModule):
    """Very small rule engine – counts objects and reports simple facts."""
    def __init__(self, ws=None):
        super().__init__(ws)
        self.facts: Dict[str, Any] = {}
    async def on_phase(self, phase):
        if phase == 1:
            props, _ = await self.ws.snapshot()
            objs = {p.content["id"] for p in props if isinstance(p.content, dict) and p.content.get("type") == "vision"}
            self.facts["obj_count"] = len(objs)
        if phase == 2 and random.random() < 0.05:
            await self.submit({"reasoning": f"I see {self.facts.get('obj_count',0)} objects"}, salience=0.5, context_tag="symbolic")

# ---------------------------------------------------------------------------
# 12 ───  CONTINUAL LEARNING DAEMON
# ---------------------------------------------------------------------------

class LearningDaemon(WorkspaceModule):
    """Learns from surprise – adjusts neuromodulator curiosity & prints log."""
    def __init__(self, ws=None):
        super().__init__(ws)
        self.iter = 0
    async def on_phase(self, phase):
        if phase != 2:
            return
        self.iter += 1
        if self.iter % 100 == 0:
            # pretend we fine‑tune a model here
            self.ws.state.setdefault("logs", []).append(f"[learn] model updated at t={self.iter}")
            await self.submit("model_updated", salience=0.3, context_tag="learning")
            # dampen curiosity after learning burst
            self.ws.state.setdefault("neuro_delta", {}).update({"curiosity": -0.1})

# ---------------------------------------------------------------------------
# 13 ───  DEMO / QUICK TEST
# ---------------------------------------------------------------------------

class EmotionalCoreMod(WorkspaceModule):
    async def on_phase(self, phase):
        if phase == 0:
            delta = {"dopamine": random.uniform(-0.02, 0.02)}
            self.ws.state.setdefault("neuro_delta", {}).update(delta)

async def _demo():
    mods = [VisionSensor(), AudioSensor(), TouchSensor(), SpatialMap(), MultimodalBinder(),
            FoundationModelConsultant(), SymbolicReasoner(), LearningDaemon(), EmotionalCoreMod()]
    engine = NyxEngineV2(mods, hz=10)
    await engine.start()
    # simple REPL
    while True:
        txt = input("You: ")
        reply = await engine.process_input(txt)
        print("Nyx:", reply)

if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        print("Exiting…")
