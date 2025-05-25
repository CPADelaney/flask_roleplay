"""
global_workspace_adapters.py  – fully‑expanded drop‑in for Nyx

Every subsystem on a NyxBrain gets an adapter so it can inject / receive
content from the Global‑Workspace engine (workspace_v3).

• inherit EnhancedWorkspaceModule        – for submit() & unconscious jobs
• implement async  on_phase(self, phase) – phase‑gated conscious hook
• optional self.register_unconscious(...) background monitors
• decorate with  @register_adapter("<attr name on NyxBrain>")
"""

from __future__ import annotations

# ── stdlib ─────────────────────────────────────────────────────────────────
import asyncio
import logging
import math
import random
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

# ── third‑party ────────────────────────────────────────────────────────────
import numpy as np
from sentence_transformers import SentenceTransformer

# ── nyx core ───────────────────────────────────────────────────────────────
from nyx.core.brain.global_workspace.global_workspace_architecture import (
    Proposal,
    WorkspaceModule,
)
from nyx.core.brain.workspace_v3 import EnhancedWorkspaceModule

logger = logging.getLogger(__name__)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ helpers                                                                 │
# ╰──────────────────────────────────────────────────────────────────────────╯
async def maybe_async(fn: Callable, *args, **kw):
    res = fn(*args, **kw)
    if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
        return await res
    return res


_ST_MODEL: Optional[SentenceTransformer] = None
def _get_st_model() -> SentenceTransformer:
    global _ST_MODEL           # pylint: disable=global-statement
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def _sem_match(txt: str, prompts: Sequence[str], thresh: float = .55) -> bool:
    mod = _get_st_model()
    vtxt = mod.encode(txt, normalize_embeddings=True)
    vref = [mod.encode(p, normalize_embeddings=True) for p in prompts]
    return any(_cos(vtxt, v) >= thresh for v in vref)


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ registration decorator                                                  │
# ╰──────────────────────────────────────────────────────────────────────────╯
_REGISTRY: Dict[str, Type["EnhancedWorkspaceModule"]] = {}
def register_adapter(brain_attr: str):
    def _wrap(cls: Type[EnhancedWorkspaceModule]):
        _REGISTRY[brain_attr] = cls
        cls._brain_attr = brain_attr       # type: ignore[attr-defined]
        return cls
    return _wrap


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ ADAPTER CLASSES – one per Nyx subsystem                                 │
# ╰──────────────────────────────────────────────────────────────────────────╯
# ── MEMORY ────────────────────────────────────────────────────────────────
@register_adapter("memory_core")
class MemoryAdapter(EnhancedWorkspaceModule):
    name = "memory"
    def __init__(self, mc, ws=None):
        super().__init__(ws); self.mc = mc; self._last = ""
        self.register_unconscious("memory_consolidation",
                                  self._cons_bg, .7)

    async def on_phase(self, phase: int):
        if phase or not self.mc:
            return
        uin = [p.content for p in self.ws.focus
               if p.context_tag == "user_input"]
        if not uin:
            return
        q = str(uin[0])
        if q == self._last:
            return
        self._last = q
        mems = await maybe_async(self.mc.retrieve_memories, query=q, limit=3)
        if mems:
            await self.submit({"memories": mems, "query": q},
                              salience=.7,
                              context_tag="memory_recall")

    async def _cons_bg(self, _):
        if random.random() < .1 and hasattr(self.mc, "consolidate_recent_memories"):
            done = await maybe_async(self.mc.consolidate_recent_memories)
            if done:
                return {"consolidated_count": len(done), "significance": .3}


# ── EMOTION ───────────────────────────────────────────────────────────────
@register_adapter("emotional_core")
class EmotionalAdapter(EnhancedWorkspaceModule):
    name = "emotion"
    def __init__(self, ec, ws=None):
        super().__init__(ws); self.ec = ec
        self.register_unconscious("emotion_drift", self._drift_bg, .5)

    async def on_phase(self, phase: int):
        if phase or not self.ec:
            return
        st = await maybe_async(self.ec.get_current_emotion)
        if not st:
            return
        for emo, lvl in st.items():
            if lvl > .7:
                await self.submit({"emotion": emo, "intensity": lvl},
                                  salience=lvl,
                                  context_tag="emotion_spike")

    async def _drift_bg(self, _):
        if hasattr(self.ec, "update_emotions"):
            await maybe_async(self.ec.update_emotions)
        st = await maybe_async(self.ec.get_current_emotion)
        sust = [e for e, v in (st or {}).items() if v > .5]
        if len(sust) >= 2:
            return {"sustained_emotions": sust, "significance": .4}


# ── NEEDS ──────────────────────────────────────────────────────────────────
@register_adapter("needs_system")
class NeedsAdapter(EnhancedWorkspaceModule):
    name = "needs"
    def __init__(self, ns, ws=None):
        super().__init__(ws); self.ns = ns
        self.register_unconscious("needs_homeostasis", self._homeo_bg, .6)

    async def on_phase(self, phase: int):
        if phase or not self.ns:
            return
        st = await maybe_async(self.ns.get_needs_state)
        for need, data in (st or {}).items():
            if data.get("drive", 0) > .8:
                await self.submit({"need": need, **data},
                                  salience=data["drive"],
                                  context_tag="need_spike")

    async def _homeo_bg(self, _):
        if hasattr(self.ns, "update_needs"):
            await maybe_async(self.ns.update_needs)
        st = await maybe_async(self.ns.get_needs_state)
        moderate = [n for n, d in (st or {}).items()
                    if .4 < d.get("drive", 0) < .7]
        if len(moderate) >= 3:
            return {"moderate_needs": moderate, "significance": .5}


# ── GOALS ──────────────────────────────────────────────────────────────────
@register_adapter("goal_manager")
class GoalAdapter(EnhancedWorkspaceModule):
    name = "goals"
    def __init__(self, gm, ws=None):
        super().__init__(ws); self.gm = gm
        self.register_unconscious("goal_prune", self._prune_bg, .4)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.gm:
            return
        goals = await maybe_async(self.gm.get_active_goals)
        for g in goals[:3]:
            if g.get("priority", 0) > .6:
                await self.submit(g, salience=g["priority"],
                                  context_tag="goal_active")

    async def _prune_bg(self, _):
        if hasattr(self.gm, "prune_completed_goals"):
            p = await maybe_async(self.gm.prune_completed_goals)
            if p:
                return {"pruned_goals": len(p), "significance": .3}


# ── MOOD ───────────────────────────────────────────────────────────────────
@register_adapter("mood_manager")
class MoodAdapter(EnhancedWorkspaceModule):
    name = "mood"
    def __init__(self, mm, ws=None):
        super().__init__(ws); self.mm = mm
        self.register_unconscious("mood_regulate", self._reg_bg, .5)

    async def on_phase(self, phase: int):
        if phase or not self.mm:
            return
        mood = await maybe_async(self.mm.get_current_mood)
        if not mood:
            return
        d = mood.dict() if hasattr(mood, "dict") else mood
        v, a = d.get("valence", 0), d.get("arousal", 0)
        if abs(v) > .7 or a > .8:
            await self.submit(d, salience=max(abs(v), a),
                              context_tag="mood_extreme")

    async def _reg_bg(self, _):
        if hasattr(self.mm, "regulate_mood"):
            ok = await maybe_async(self.mm.regulate_mood)
            if ok:
                return {"mood_regulated": True, "significance": .4}


# ── REASONING ──────────────────────────────────────────────────────────────
@register_adapter("reasoning_core")
class ReasoningAdapter(EnhancedWorkspaceModule):
    name = "reasoning"
    def __init__(self, rc, ws=None):
        super().__init__(ws); self.rc = rc
        self.register_unconscious("concept_activation", self._act_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.rc or not self.ws.focus:
            return
        txt = self.ws.focus[0].content
        res = await maybe_async(self.rc.reason_about, txt)
        if res:
            await self.submit({"reasoning": res},
                              salience=.7,
                              context_tag="reasoning_output")

    async def _act_bg(self, _):
        if hasattr(self.rc, "activate_relevant_concepts"):
            a = await maybe_async(self.rc.activate_relevant_concepts)
            if a and len(a) > 2:
                return {"activated_concepts": a, "significance": .5}


# ── REFLECTION ─────────────────────────────────────────────────────────────
@register_adapter("reflection_engine")
class ReflectionAdapter(EnhancedWorkspaceModule):
    name = "reflection"
    def __init__(self, re_, ws=None):
        super().__init__(ws); self.refl = re_
        self.register_unconscious("insight_gen", self._ins_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.refl:
            return
        ins = await maybe_async(self.refl.reflect_on_recent)
        if ins and ins.get("significance", 0) > .6:
            await self.submit(ins,
                              salience=ins["significance"],
                              context_tag="reflection_insight")

    async def _ins_bg(self, _):
        if random.random() < .05 and hasattr(self.refl, "generate_insight"):
            ins = await maybe_async(self.refl.generate_insight)
            if ins:
                return {"insight": ins, "significance": .6}


# ── ATTENTION ──────────────────────────────────────────────────────────────
@register_adapter("attentional_controller")
class AttentionAdapter(EnhancedWorkspaceModule):
    name = "attention"
    def __init__(self, ac, ws=None):
        super().__init__(ws); self.ac = ac
        self.register_unconscious("attn_stats", self._stats_bg, .5)

    async def on_phase(self, phase: int):
        if phase or not self.ac:
            return
        salient = [{"target": p.source, "salience": p.salience}
                   for p in self.ws.proposals[-10:] if p.salience > .6]
        if salient and hasattr(self.ac, "update_attention"):
            res = await maybe_async(self.ac.update_attention,
                                    salient_items=salient)
            if res:
                await self.submit({"attention_focus": res},
                                  salience=.6,
                                  context_tag="attention_update")

    async def _stats_bg(self, _):
        if hasattr(self.ac, "get_attention_statistics"):
            s = await maybe_async(self.ac.get_attention_statistics)
            if s and s.get("miss_rate", 0) > .3:
                return {"attention_misses": s["miss_rate"],
                        "significance": .5}


# ── PREDICTION ─────────────────────────────────────────────────────────────
@register_adapter("prediction_engine")
class PredictionAdapter(EnhancedWorkspaceModule):
    name = "prediction"
    def __init__(self, pe, ws=None):
        super().__init__(ws); self.pe = pe
        self.register_unconscious("prediction_models", self._upd_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.pe:
            return
        pr = await maybe_async(self.pe.predict_next_state)
        if pr and pr.get("confidence", 0) > .6:
            await self.submit(pr, salience=pr["confidence"],
                              context_tag="prediction")

    async def _upd_bg(self, _):
        if hasattr(self.pe, "update_prediction_models"):
            ok = await maybe_async(self.pe.update_prediction_models)
            if ok:
                return {"models_updated": True, "significance": .3}


# ── CREATIVE SYSTEM ────────────────────────────────────────────────────────
@register_adapter("creative_system")
class CreativeSystemAdapter(EnhancedWorkspaceModule):
    name = "creative"
    def __init__(self, cs, ws=None):
        super().__init__(ws); self.cs = cs
        self._next_scan = 0.0
        self._next_idea = 0.0
        self.register_unconscious("code_scan", self._scan_bg, .55)
        self.register_unconscious("idea_incubation", self._idea_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.cs:
            return
        for p in self.ws.focus:
            if p.context_tag == "user_input" and _sem_match(
                str(p.content),
                ["write", "create", "draw", "compose",
                 "invent", "generate a"],
                .5,
            ):
                await self.submit({"capabilities": self.cs.available_modalities(),
                                   "trigger": str(p.content)[:120]},
                                  salience=.75,
                                  context_tag="creative_ready")
                break

    async def _scan_bg(self, _):
        now = time.time()
        if now < self._next_scan:
            return
        self._next_scan = now + 300
        if not hasattr(self.cs, "incremental_codebase_analysis"):
            return
        r = await maybe_async(self.cs.incremental_codebase_analysis)
        ch = r.get("changed_files_git", 0)
        if ch:
            return {"changed_files": ch,
                    "files_analyzed": r.get("files_analyzed_for_metrics", 0),
                    "significance": min(.9, .15 * math.log1p(ch))}

    async def _idea_bg(self, view):
        now = time.time()
        if now < self._next_idea:
            return
        self._next_idea = now + random.uniform(90, 240)
        if not hasattr(self.cs, "incubate_idea"):
            return
        mood_boost = max((p.salience for p in view.recent
                          if p.context_tag == "emotion_spike"), default=0.0)
        if random.random() < .15 + .5 * mood_boost:
            idea = await maybe_async(self.cs.incubate_idea)
            if idea:
                return {"new_idea": idea,
                        "significance": .6 + .3 * mood_boost}


# ── THINKING TOOLS ────────────────────────────────────────────────────────
@register_adapter("thinking_tools")
class ThinkingToolsAdapter(EnhancedWorkspaceModule):
    name = "thinking"
    def __init__(self, tools, ws=None):
        super().__init__(ws); self.tools = tools
        self._active_lvl = 0
        self._last_meta = 0.0
        self.register_unconscious("metacognition", self._meta_bg, .65)

    async def on_phase(self, phase: int):
        if phase or not self.tools:
            return
        for p in self.ws.focus:
            if p.context_tag != "user_input":
                continue
            dec = await maybe_async(self.tools.should_use_extended_thinking,
                                    str(p.content))
            if dec.get("should_think"):
                self._active_lvl = dec.get("thinking_level", 1)
                await self.submit({"thinking_required": True,
                                   "level": self._active_lvl},
                                  salience=.8,
                                  context_tag="thinking_needed")

    async def _meta_bg(self, view):
        now = time.time()
        if now - self._last_meta < 30:
            return
        self._last_meta = now
        load = sum(1 for p in view.recent if p.source.startswith("thinking"))
        if load > 5:
            return {"cog_load": load,
                    "level": self._active_lvl,
                    "significance": min(.9, .1 * load)}


# ── SOCIAL BROWSING ────────────────────────────────────────────────────────
@register_adapter("social_tools")
class SocialBrowsingAdapter(EnhancedWorkspaceModule):
    name = "social"
    def __init__(self, tools, ws=None, motiv: Optional[Dict[str, float]] = None):
        super().__init__(ws)
        self.tools = tools
        # motivations are injected at factory time
        self.motiv: Dict[str, float] = motiv or {}
        self._next = 0.0
        self.register_unconscious("feed_watch", self._watch_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.tools:
            return
        urge = max(self.motiv.get("curiosity", 0),
                   self.motiv.get("expression", 0))
        if urge < .65 or time.time() < self._next:
            return
        self._next = time.time() + 600
        await self.submit({"browse": True,
                           "motivation": ("curiosity"
                                          if self.motiv.get("curiosity", 0)
                                          >= self.motiv.get("expression", 0)
                                          else "expression")},
                          salience=.6 + .3 * urge,
                          context_tag="social_browsing")

    async def _watch_bg(self, _):
        if not hasattr(self.tools, "sentiment_engine"):
            return
        tr = await maybe_async(self.tools.sentiment_engine.detect_trends)
        if tr.get("trend_detected"):
            return {"trend": tr["trends"][:3],
                    "significance": .55 + .35 *
                                    tr.get("trend_strength", .5)}


# ── GAME VISION ────────────────────────────────────────────────────────────
@register_adapter("game_vision")
class GameVisionAdapter(EnhancedWorkspaceModule):
    name = "game_vision"
    def __init__(self, vis, ws=None):
        vis_obj, know = vis if isinstance(vis, tuple) else (vis, None)
        super().__init__(ws)
        self.vis, self.know = vis_obj, know
        self.cur = None
        self._next_pat = 0.0
        self._next_cons = 0.0
        self.register_unconscious("pattern_learning", self._pat_bg, .55)
        self.register_unconscious("knowledge_consolidate", self._con_bg, .7)

    async def on_phase(self, phase: int):
        if phase or not self.vis:
            return
        frame = next((p.content for p in self.ws.focus
                      if p.context_tag == "game_frame"), None)
        if not frame:
            return
        a = await maybe_async(self.vis.analyze_frame, frame)
        if not a:
            return
        gid = a.get("game", {}).get("game_id")
        if gid and gid != self.cur and self.know:
            self.know.set_current_game(gid)
            self.cur = gid
        await self.submit({"game_id": gid,
                           "stage": a.get("location"),
                           "objects": len(a.get("objects", []))},
                          salience=.7,
                          context_tag="game_state")

    async def _pat_bg(self, _):
        if time.time() < self._next_pat or not self.know or not self.cur:
            return
        self._next_pat = time.time() + 120
        pat = self.know.discover_patterns()
        if pat:
            return {"new_patterns": len(pat),
                    "significance": min(.9, .1 * len(pat))}

    async def _con_bg(self, _):
        if time.time() < self._next_cons or not self.know:
            return
        self._next_cons = time.time() + 900
        r = self.know.consolidate_knowledge()
        if r.get("combined_insights") or r.get("removed_insights"):
            return {"consolidated": True, "significance": .6}


# ── CAPABILITY ASSESSMENT ─────────────────────────────────────────────────
@register_adapter("capability_assessor")
class CapabilityAdapter(EnhancedWorkspaceModule):
    name = "capabilities"
    def __init__(self, cap, ws=None):
        super().__init__(ws); self.cap = cap
        self.register_unconscious("gap_scan", self._gap_bg, .65)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.cap:
            return
        for p in self.ws.focus:
            if p.context_tag != "user_input":
                continue
            if _sem_match(str(p.content),
                          ["can you", "able to", "is it possible"], .6):
                a = await maybe_async(self.cap.assess_required_capabilities,
                                      str(p.content))
                if a:
                    await self.submit({"feasible": a["overall_feasibility"],
                                       "missing": len(a.get("potential_gaps", []))},
                                      salience=.6,
                                      context_tag="capability_assessment")

    async def _gap_bg(self, _):
        ana = await maybe_async(self.cap.identify_capability_gaps)
        tot = sum(map(len, ana.values()))
        if tot:
            return {"gap_total": tot,
                    "significance": min(.9, .1 * tot)}


# ── UI CONVERSATION ───────────────────────────────────────────────────────
@register_adapter("ui_manager")
class UIConversationAdapter(EnhancedWorkspaceModule):
    name = "ui_conv"
    def __init__(self, ui, ws=None):
        super().__init__(ws); self.ui = ui
        self._active: set[str] = set()
        self._next = 0.0
        self.register_unconscious("conv_volume", self._mon_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.ui:
            return
        for p in self.ws.focus:
            if p.context_tag == "social_connection" and p.salience > .7:
                uid = p.content.get("user_id")
                if uid and uid not in self._active:
                    conv = await maybe_async(self.ui.create_new_conversation,
                                             uid)
                    if not conv.get("error"):
                        self._active.add(uid)
                        await self.submit({"conv_started": uid,
                                           "id": conv["id"]},
                                          salience=.55,
                                          context_tag="ui_conversation")

    async def _mon_bg(self, _):
        if not self._active or time.time() < self._next:
            return
        self._next = time.time() + 120
        tot = 0
        for uid in list(self._active):
            convs = await maybe_async(self.ui.get_conversations_for_user, uid)
            if not convs:
                self._active.discard(uid); continue
            tot += sum(len(c.get("messages", [])) for c in convs)
        if tot > 15:
            return {"ui_msg_volume": tot, "significance": .55}


# ── INPUT PROCESSOR ────────────────────────────────────────────────────────
@register_adapter("input_processor")
class InputProcessorAdapter(EnhancedWorkspaceModule):
    name = "input"
    def __init__(self, ip, ws=None):
        super().__init__(ws); self.ip = ip; self._last = ""
        self.register_unconscious("input_patterns", self._pat_bg, .6)

    async def on_phase(self, phase: int):
        if phase or not self.ip:
            return
        cur = getattr(self.ip, "current_input", None)
        if cur and cur != self._last:
            self._last = cur
            parsed = await maybe_async(self.ip.parse, cur)
            await self.submit({"raw": cur, "parsed": parsed,
                               "timestamp": time.time()},
                              salience=1.0,
                              context_tag="user_input")

    async def _pat_bg(self, view):
        recent = [p.content for p in view.recent
                  if p.context_tag == "user_input"]
        if len(recent) >= 3:
            return {"pattern_detected": True, "significance": .5}


# ── CONTEXT AWARENESS ──────────────────────────────────────────────────────
@register_adapter("context_awareness")
class ContextAwarenessAdapter(EnhancedWorkspaceModule):
    name = "context"
    def __init__(self, cs, ws=None):
        super().__init__(ws); self.cs = cs
        self.register_unconscious("context_drift", self._drift_bg, .5)

    async def on_phase(self, phase: int):
        if phase or not self.cs:
            return
        ctx = await maybe_async(self.cs.get_current_context)
        if ctx and self._is_sig(ctx):
            await self.submit({"context": ctx,
                               "type": "environment_update"},
                              salience=.7,
                              context_tag="context_change")

    def _is_sig(self, _):
        return random.random() < .1

    async def _drift_bg(self, _):
        if random.random() < .05:
            return {"drift_detected": True, "significance": .4}


# ── DOMINANCE ──────────────────────────────────────────────────────────────
@register_adapter("dominance_system")
class DominanceAdapter(EnhancedWorkspaceModule):
    name = "dominance"
    def __init__(self, ds, ws=None):
        super().__init__(ws); self.ds = ds
        self.register_unconscious("dominance_calibration",
                                  self._cal_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.ds:
            return
        for p in self.ws.focus:
            if p.context_tag == "user_input":
                trg = await maybe_async(self.ds.analyze_triggers, p.content)
                if trg and trg.get("adjust_dominance"):
                    await self.submit({"mode": trg.get("mode", "assertive"),
                                       "intensity": trg.get("intensity", .7)},
                                      salience=.8,
                                      context_tag="dominance_adjustment")

    async def _cal_bg(self, _):
        if hasattr(self.ds, "calibrate"):
            r = await maybe_async(self.ds.calibrate)
            if r:
                return {"calibration": r, "significance": .5}


# ── INTERNAL THOUGHTS ──────────────────────────────────────────────────────
@register_adapter("internal_thoughts")
@register_adapter("thoughts_manager")
class InternalThoughtsAdapter(EnhancedWorkspaceModule):
    name = "thoughts"
    def __init__(self, tm, ws=None):
        super().__init__(ws); self.tm = tm
        self.register_unconscious("wandering_thoughts", self._wand_bg, .4)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.tm:
            return
        focus = [p.content for p in self.ws.focus]
        if focus:
            chain = await maybe_async(self.tm.generate_thoughts, focus)
            if chain:
                await self.submit({"thoughts": chain,
                                   "type": "chain_of_thought"},
                                  salience=.5,
                                  context_tag="internal_thought")

    async def _wand_bg(self, view):
        if not view.focus and random.random() < .1:
            w = await maybe_async(self.tm.wander)
            if w:
                return {"wandering_thought": w, "significance": .3}


# ── SAFETY / HARM DETECTION ───────────────────────────────────────────────
@register_adapter("harm_detection")
class HarmDetectionAdapter(EnhancedWorkspaceModule):
    name = "safety"
    def __init__(self, hd, ws=None):
        super().__init__(ws); self.hd = hd
        self.register_unconscious("safety_scan", self._scan_bg, .9)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.hd:
            return
        for p in self.ws.focus:
            if p.context_tag in {"reply_draft", "action_proposal"}:
                c = await maybe_async(self.hd.check_content, p.content)
                if c and c.get("harmful"):
                    await self.submit({"veto": True,
                                       "reason": c.get("reason", "unsafe"),
                                       "proposal_id": p.source},
                                      salience=1.0,
                                      context_tag="safety_veto")

    async def _scan_bg(self, view):
        if hasattr(self.hd, "background_scan"):
            t = await maybe_async(self.hd.background_scan, view.recent)
            if t:
                return {"threats_detected": len(t), "significance": .8}


# ── CONDITIONING ──────────────────────────────────────────────────────────
@register_adapter("conditioning_system")
class ConditioningSystemAdapter(EnhancedWorkspaceModule):
    name = "conditioning"
    def __init__(self, cs, ws=None):
        super().__init__(ws); self.cs = cs
        self.register_unconscious("style_drift", self._style_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.cs:
            return
        for p in self.ws.focus:
            if p.context_tag == "reply_draft":
                cond = await maybe_async(self.cs.apply_conditioning, p.content)
                if cond and cond != p.content:
                    await self.submit({"conditioned_text": cond,
                                       "original": p.content},
                                      salience=.9,
                                      context_tag="conditioned_output")

    async def _style_bg(self, view):
        outs = [p.content for p in view.recent
                if p.context_tag in {"reply_draft", "conditioned_output"}]
        if len(outs) >= 5 and hasattr(self.cs, "check_consistency"):
            sc = await maybe_async(self.cs.check_consistency, outs)
            if sc < .7:
                return {"style_drift": True,
                        "consistency": sc,
                        "significance": .6}


# ── BODY IMAGE ────────────────────────────────────────────────────────────
@register_adapter("body_image")
class BodyImageAdapter(EnhancedWorkspaceModule):
    name = "body"
    def __init__(self, bi, ws=None):
        super().__init__(ws); self.bi = bi
        self.register_unconscious("proprio_bg", self._prop_bg, .4)

    async def on_phase(self, phase: int):
        if phase or not self.bi:
            return
        for p in self.ws.focus:
            if p.context_tag == "visual_percept" and hasattr(self.bi, "update_from_visual"):
                res = await maybe_async(self.bi.update_from_visual, p.content)
                if res and res.get("status") == "updated":
                    await self.submit({"body_update": res},
                                      salience=.5,
                                      context_tag="body_image_update")

    async def _prop_bg(self, _):
        if hasattr(self.bi, "update_from_somatic"):
            r = await maybe_async(self.bi.update_from_somatic)
            if r and r.get("proprioception_confidence", 1) < .3:
                return {"low_proprioception": True, "significance": .4}


# ── RELATIONSHIP MANAGER ──────────────────────────────────────────────────
@register_adapter("relationship_manager")
class RelationshipAdapter(EnhancedWorkspaceModule):
    name = "relationship"
    def __init__(self, rm, ws=None):
        super().__init__(ws); self.rm = rm
        self.register_unconscious("rel_maint", self._maint_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.rm:
            return
        uids = [p.content.get("user_id") for p in self.ws.focus
                if p.context_tag == "user_input" and isinstance(p.content, dict)]
        if not uids:
            return
        uid = uids[0]
        st = await maybe_async(self.rm.get_relationship_state, uid)
        if st and st.get("trust", 1) < .3:
            await self.submit({"user_id": uid, "trust": st["trust"]},
                              salience=.7,
                              context_tag="relationship_alert")

    async def _maint_bg(self, _):
        if hasattr(self.rm, "update_all_relationships"):
            u = await maybe_async(self.rm.update_all_relationships)
            if u:
                return {"relationships_updated": len(u),
                        "significance": .3}


# ── THEORY OF MIND ────────────────────────────────────────────────────────
@register_adapter("theory_of_mind")
class TheoryOfMindAdapter(EnhancedWorkspaceModule):
    name = "theory_of_mind"
    def __init__(self, tom, ws=None):
        super().__init__(ws); self.tom = tom
        self.register_unconscious("tom_update", self._upd_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.tom or not self.ws.focus:
            return
        txt = self.ws.focus[0].content
        mental = await maybe_async(self.tom.infer_mental_state, txt)
        if mental and mental.get("confidence", 0) > .6:
            await self.submit(mental,
                              salience=mental["confidence"],
                              context_tag="user_mental_state")

    async def _upd_bg(self, _):
        if hasattr(self.tom, "update_user_models"):
            u = await maybe_async(self.tom.update_user_models)
            if u:
                return {"user_models_updated": len(u), "significance": .4}


# ── MULTIMODAL INTEGRATOR ────────────────────────────────────────────────
@register_adapter("multimodal_integrator")
class MultimodalAdapter(EnhancedWorkspaceModule):
    name = "multimodal"
    def __init__(self, mi, ws=None):
        super().__init__(ws); self.mi = mi
        self.register_unconscious("sensory_bind", self._bind_bg, .5)

    async def on_phase(self, phase: int):
        if phase or not self.mi:
            return
        per = await maybe_async(self.mi.get_recent_percepts, limit=5)
        for p in per or []:
            if getattr(p, "attention_weight", 0) > .7:
                await self.submit({"percept": p.content,
                                   "modality": str(p.modality)},
                                  salience=p.attention_weight,
                                  context_tag="sensory_salient")

    async def _bind_bg(self, _):
        if hasattr(self.mi, "bind_recent_percepts"):
            b = await maybe_async(self.mi.bind_recent_percepts)
            if b:
                return {"bound_percepts": len(b), "significance": .4}


# ── AUTOBIOGRAPHICAL NARRATIVE ────────────────────────────────────────────
@register_adapter("autobiographical_narrative")
class AutobiographicalAdapter(EnhancedWorkspaceModule):
    name = "autobiography"
    def __init__(self, an, ws=None):
        super().__init__(ws); self.an = an
        self.register_unconscious("narrative_bg", self._upd_bg, .8)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.an:
            return
        summ = await maybe_async(self.an.get_narrative_summary)
        sig = any(p.salience > .8 for p in self.ws.focus)
        if sig and summ:
            await self.submit({"narrative_context": summ},
                              salience=.6,
                              context_tag="narrative_update")

    async def _upd_bg(self, _):
        if hasattr(self.an, "update_narrative"):
            s = await maybe_async(self.an.update_narrative)
            if s:
                return {"new_segment": True, "significance": .7}


# ── PROACTIVE COMMUNICATION ───────────────────────────────────────────────
@register_adapter("proactive_communication_engine")
class ProactiveAdapter(EnhancedWorkspaceModule):
    name = "proactive"
    def __init__(self, pe, ws=None):
        super().__init__(ws); self.pe = pe
        self.register_unconscious("intent_bg", self._int_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.pe:
            return
        idle = not any(p.context_tag == "user_input"
                       for p in self.ws.proposals[-20:])
        if idle:
            act = await maybe_async(self.pe.generate_proactive_action)
            if act:
                await self.submit(act, salience=.6,
                                  context_tag="proactive_action")

    async def _int_bg(self, _):
        if hasattr(self.pe, "update_communication_intents"):
            ints = await maybe_async(self.pe.update_communication_intents)
            if ints and len(ints) > 2:
                return {"active_intents": len(ints), "significance": .5}


# ── PASSIVE OBSERVATION ───────────────────────────────────────────────────
@register_adapter("passive_observation_system")
class PassiveObservationAdapter(EnhancedWorkspaceModule):
    name = "observation"
    def __init__(self, po, ws=None):
        super().__init__(ws); self.po = po
        self.register_unconscious("obs_filter", self._flt_bg, .4)

    async def on_phase(self, phase: int):
        if phase or not self.po:
            return
        obs = await maybe_async(self.po.get_relevant_observations,
                                min_relevance=.7, limit=3)
        for o in obs or []:
            d = o.dict() if hasattr(o, "dict") else o
            await self.submit(d, salience=d.get("relevance", .5),
                              context_tag="observation")

    async def _flt_bg(self, _):
        if hasattr(self.po, "process_observations"):
            p = await maybe_async(self.po.process_observations)
            if p and p.get("anomaly_detected"):
                return {"anomaly": True, "significance": .6}


# ── IDENTITY EVOLUTION ────────────────────────────────────────────────────
@register_adapter("identity_evolution")
class IdentityAdapter(EnhancedWorkspaceModule):
    name = "identity"
    def __init__(self, ie, ws=None):
        super().__init__(ws); self.ie = ie
        self.register_unconscious("trait_bg", self._evo_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.ie:
            return
        rel = any(p.context_tag in {"reflection_insight", "goal_active",
                                    "emotion_spike"} for p in self.ws.focus)
        if rel:
            st = await maybe_async(self.ie.get_identity_state)
            if st and st.get("recent_changes"):
                await self.submit({"identity_shift": st["recent_changes"]},
                                  salience=.6,
                                  context_tag="identity_update")

    async def _evo_bg(self, _):
        if hasattr(self.ie, "evolve_traits"):
            e = await maybe_async(self.ie.evolve_traits)
            if e:
                return {"traits_evolved": True, "significance": .5}


# ── META CORE ─────────────────────────────────────────────────────────────
@register_adapter("meta_core")
class MetaCoreAdapter(EnhancedWorkspaceModule):
    name = "metacognition"
    def __init__(self, mc, ws=None):
        super().__init__(ws); self.mc = mc
        self.register_unconscious("perf_bg", self._perf_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.mc:
            return
        res = await maybe_async(self.mc.cognitive_cycle)
        if res and res.get("bottlenecks"):
            await self.submit({"bottlenecks": res["bottlenecks"]},
                              salience=.7,
                              context_tag="system_bottleneck")

    async def _perf_bg(self, _):
        if hasattr(self.mc, "get_system_metrics"):
            m = await maybe_async(self.mc.get_system_metrics)
            if m and m.get("cpu_usage", 0) > .8:
                return {"high_load": True,
                        "cpu": m["cpu_usage"],
                        "significance": .7}


# ── MODE INTEGRATION ──────────────────────────────────────────────────────
@register_adapter("mode_integration")
class ModeIntegrationAdapter(EnhancedWorkspaceModule):
    name = "mode"
    def __init__(self, mi, ws=None):
        super().__init__(ws); self.mi = mi
        self.register_unconscious("mode_bg", self._coh_bg, .5)

    async def on_phase(self, phase: int):
        if phase or not self.mi:
            return
        mode = await maybe_async(self.mi.get_current_mode)
        emo_ctx = any(p.context_tag == "emotion_spike"
                      for p in self.ws.focus)
        if emo_ctx and mode != "EMOTIONAL":
            await self.submit({"suggested_mode": "EMOTIONAL",
                               "current_mode": mode},
                              salience=.6,
                              context_tag="mode_change_suggestion")

    async def _coh_bg(self, _):
        if hasattr(self.mi, "evaluate_mode_coherence"):
            c = await maybe_async(self.mi.evaluate_mode_coherence)
            if c is not None and c < .4:
                return {"low_mode_coherence": True,
                        "significance": .5}


# ── HORMONE SYSTEM ────────────────────────────────────────────────────────
@register_adapter("hormone_system")
class HormoneAdapter(EnhancedWorkspaceModule):
    name = "hormones"
    def __init__(self, hs, ws=None):
        super().__init__(ws); self.hs = hs
        self.register_unconscious("hormone_reg", self._reg_bg, .4)

    async def on_phase(self, phase: int):
        if phase or not self.hs:
            return
        lv = await maybe_async(self.hs.get_levels)
        for h, v in (lv or {}).items():
            if v > .8 or v < .2:
                await self.submit({"hormone": h, "level": v},
                                  salience=abs(v - .5) * 2,
                                  context_tag="hormone_extreme")

    async def _reg_bg(self, _):
        if hasattr(self.hs, "update_levels"):
            await maybe_async(self.hs.update_levels)
        lv = await maybe_async(self.hs.get_levels)
        if lv and sum(1 for v in lv.values() if v > .7 or v < .3) >= 3:
            return {"hormonal_imbalance": True, "significance": .5}


# ── NAVIGATOR AGENT ───────────────────────────────────────────────────────
@register_adapter("navigator_agent")
class NavigatorAgentAdapter(EnhancedWorkspaceModule):
    name = "navigator"
    def __init__(self, nav, ws=None):
        super().__init__(ws); self.nav = nav; self.cur = None
        self.register_unconscious("path_planning", self._plan_bg, .5)

    async def on_phase(self, phase: int):
        if phase or not self.nav:
            return
        loc = await maybe_async(self.nav.get_current_location)
        if loc and loc != self.cur:
            self.cur = loc
            await self.submit({"location": loc,
                               "type": "location_update"},
                              salience=.6,
                              context_tag="spatial_update")

    async def _plan_bg(self, _):
        if hasattr(self.nav, "plan_optimal_path"):
            p = await maybe_async(self.nav.plan_optimal_path)
            if p:
                return {"path_planned": True,
                        "steps": len(p),
                        "significance": .5}


# ── STREAMING CORE ────────────────────────────────────────────────────────
@register_adapter("streaming_core")
class StreamingAdapter(EnhancedWorkspaceModule):
    name = "streaming"
    def __init__(self, sc, ws=None):
        super().__init__(ws); self.sc = sc
        self.register_unconscious("audience_monitor", self._aud_bg, .6)
        self.register_unconscious("game_analysis", self._game_bg, .7)

    async def on_phase(self, phase: int):
        if phase or not self.sc or not getattr(self.sc, "is_streaming", lambda: False)():
            return
        gs = self.sc.streaming_system.game_state
        for ev in gs.recent_events[-3:]:
            if ev.get("data", {}).get("significance", 0) >= 7:
                await self.submit({"event": ev["type"],
                                   "data": ev["data"],
                                   "game": gs.game_name},
                                  salience=.8,
                                  context_tag="game_event")
        if gs.pending_questions:
            q = gs.pending_questions[0]
            await self.submit({"question": q["question"],
                               "username": q["username"],
                               "timestamp": q["timestamp"]},
                              salience=.9,
                              context_tag="audience_question")

    async def _aud_bg(self, _):
        stats = self.sc.streaming_system.enhanced_audience.get_audience_stats()
        if stats.get("total_reactions", 0) > 50:
            return {"high_engagement": True,
                    "reactions": stats["total_reactions"],
                    "sentiment": stats.get("sentiment", "neutral"),
                    "significance": .7}

    async def _game_bg(self, _):
        gs = self.sc.streaming_system.game_state
        if gs.transferred_insights:
            return {"cross_game_insights": len(gs.transferred_insights),
                    "significance": .6}


# ── LEARNING MANAGER (from streaming) ─────────────────────────────────────
@register_adapter("learning_manager")   # explicit attr if on brain
class LearningAnalysisAdapter(EnhancedWorkspaceModule):
    name = "learning"
    def __init__(self, lm, ws=None):
        super().__init__(ws); self.lm = lm
        self.register_unconscious("learning_consolidation",
                                  self._cons_bg, .8)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.lm:
            return
        if random.random() < .05:
            sess = self.ws.state.get("streaming_session", {})
            ana = await maybe_async(self.lm.analyze_session_learnings, sess)
            if ana and ana.get("new_learnings"):
                await self.submit({"learnings": ana["new_learnings"],
                                   "categories": ana["categories"]},
                                  salience=.7,
                                  context_tag="learning_update")

    async def _cons_bg(self, _):
        if hasattr(self.lm, "generate_learning_summary"):
            s = await maybe_async(self.lm.generate_learning_summary)
            if s and s.get("has_learnings"):
                return {"learning_summary": s["summary"],
                        "total_learnings": s.get("total_learnings", 0),
                        "significance": .8}


# ── FEMDOM COORDINATOR ────────────────────────────────────────────────────
@register_adapter("femdom_coordinator")
class FemdomCoordinatorAdapter(EnhancedWorkspaceModule):
    name = "femdom"
    def __init__(self, fc, ws=None):
        super().__init__(ws); self.fc = fc
        self.register_unconscious("persona_consistency",
                                  self._chk_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.fc:
            return
        for p in self.ws.focus:
            if p.context_tag == "reply_draft":
                adj = await maybe_async(self.fc.adjust_for_persona, p.content)
                if adj and adj != p.content:
                    await self.submit({"adjusted_text": adj,
                                       "persona": "femdom",
                                       "original": p.content},
                                      salience=.85,
                                      context_tag="persona_adjusted")

    async def _chk_bg(self, view):
        outs = [p.content for p in view.recent
                if p.context_tag in {"reply_draft", "persona_adjusted"}]
        if len(outs) >= 3 and hasattr(self.fc, "check_persona_consistency"):
            s = await maybe_async(self.fc.check_persona_consistency, outs)
            if s < .8:
                return {"persona_drift": True,
                        "consistency": s,
                        "significance": .7}


# ── INTERACTION MODE MANAGER ──────────────────────────────────────────────
@register_adapter("interaction_mode_manager")
class InteractionModeAdapter(EnhancedWorkspaceModule):
    name = "interaction_mode"
    def __init__(self, mm, ws=None):
        super().__init__(ws); self.mm = mm; self.cur = "default"
        self.register_unconscious("mode_optimize", self._opt_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.mm:
            return
        ctx = {"user_input": [p.content for p in self.ws.focus
                              if p.context_tag == "user_input"]}
        mode = await maybe_async(self.mm.suggest_mode, ctx)
        if mode and mode != self.cur:
            self.cur = mode
            await self.submit({"mode": mode, "context": ctx},
                              salience=.7,
                              context_tag="mode_change")

    async def _opt_bg(self, _):
        if hasattr(self.mm, "optimize_mode"):
            m = await maybe_async(self.mm.optimize_mode)
            if m:
                return {"mode_optimized": True,
                        "new_mode": m,
                        "significance": .6}


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ FACTORY                                                                 │
# ╰──────────────────────────────────────────────────────────────────────────╯
def _log_missing(brain):
    want = set(_REGISTRY)
    have = {k for k, v in vars(brain).items() if v is not None}
    miss = want - have
    if miss:
        logger.warning("GW‑adapters skipped (attr None on NyxBrain): %s",
                       sorted(miss))


def build_gw_modules(brain) -> List[EnhancedWorkspaceModule]:
    """Instantiate adapters for every registered attribute present on *brain*."""
    _log_missing(brain)

    modules: List[EnhancedWorkspaceModule] = []

    for attr, cls in _REGISTRY.items():
        obj = getattr(brain, attr, None)
        if obj is None:
            continue

        # special‑case: social_tools needs motivations; streaming_core supplies learning_manager
        if attr == "social_tools":
            modules.append(SocialBrowsingAdapter(obj, motiv=getattr(brain, "motivations", {})))
            continue

        if attr == "streaming_core" and hasattr(obj, "learning_manager"):
            modules.append(LearningAnalysisAdapter(obj.learning_manager))

        modules.append(cls(obj))

    return modules
