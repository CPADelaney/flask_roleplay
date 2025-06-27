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
from nyx.core.brain.global_workspace.workspace_v3 import EnhancedWorkspaceModule

from nyx.core.agentic_action_generator import ActionContext

logger = logging.getLogger(__name__)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ helpers                                                                 │
# ╰──────────────────────────────────────────────────────────────────────────╯

# clamp salience to [0, 1] in one place
def _clamp(val: float) -> float:
    if val < 0:
        return 0.0
    return 1.0 if val > 1 else val
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

@register_adapter("response_synthesizer")
class ResponseSynthesizerAdapter(EnhancedWorkspaceModule):
    """
    Promote the best reply-draft (or generate a last-minute one) to
    a ‘complete_response’ so downstream guardrails have something
    concrete to vet instead of falling back to boiler-plate.
    """
    name = "response_synthesizer"

    def __init__(self, brain, ws=None):
        super().__init__(ws)
        self.brain = brain

    # ------------------------------------------------------------------ #
    async def on_phase(self, phase: int):
        if phase != 2:                       # final conscious pass
            return

        # ------------------------------------------
        # 1. Gather every reply-like proposal so far
        # ------------------------------------------
        REPLY_TAGS = {
            "persona_adjusted",
            "conditioned_output",
            "reply_draft",
            "creative_synthesis",
            "imagination_output",
            "response_candidate",
        }
        candidates = [
            p for p in self.ws.proposals
            if p.context_tag in REPLY_TAGS
        ]

        # ------------------------------------------
        # 2. If we already have a COMPLETE response,
        #    don’t touch anything.
        # ------------------------------------------
        if any(p.context_tag == "complete_response" for p in candidates):
            return

        # ------------------------------------------
        # 3. Choose the strongest candidate or build
        #    a bare-bones reply from scratch.
        # ------------------------------------------
        if candidates:
            best = max(candidates, key=lambda p: p.salience)
            text = _extract_text(best.content) or "…"        # fallback text
            confidence = min(1.0, best.salience)

        else:  # nothing at all – fabricate something minimal
            text = "I'm still thinking about that…"
            confidence = 0.3

        # ------------------------------------------
        # 4. Publish COMPLETE response                 (context_tag key!)
        # ------------------------------------------
        await self.submit(
            {
                "response": text,
                "confidence": confidence,
                "sources": [
                    p.source for p in candidates if p.salience > 0.6
                ],
            },
            salience=1.0,                    # make sure it wins attention
            context_tag="complete_response",
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_text(content: Any) -> str | None:
        """Best-effort text extraction from various draft payload shapes."""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            for key in (
                "adjusted_text", "conditioned_text", "response",
                "message", "imagination"
            ):
                if key in content and content[key]:
                    return str(content[key])
            # last resort – stringify the whole dict
            return str(content)
        return None


@register_adapter("fallback_responder") 
class FallbackResponderAdapter(EnhancedWorkspaceModule):
    """Ensures we always have some response"""
    name = "fallback"
    
    def __init__(self, brain, ws=None):
        super().__init__(ws)
        self.brain = brain
    
    async def on_phase(self, phase: int):
        if phase != 2:
            return
            
        props, _ = await self.ws.snapshot()
        has_any_response = any(p.context_tag in ["response_candidate", "complete_response", "ai_response"]
                              for p in props[-50:])
        
        if not has_any_response:
            await self.submit({
                "response": "I'm processing that thought...",
                "confidence": 0.3,
                "fallback": True
            }, salience=0.5, context_tag="response_candidate")
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
        # FIX: Use get_emotional_state instead of get_current_emotion
        st = await maybe_async(self.ec.get_emotional_state)
        if st and 'emotional_state_matrix' in st:
            matrix = st['emotional_state_matrix']
            primary = matrix.get('primary_emotion', {})
            if primary.get('intensity', 0) > .7:
                await self.submit({"emotion": primary.get('name', 'unknown'), 
                                  "intensity": primary.get('intensity', 0)},
                                  salience=primary.get('intensity', 0),
                                  context_tag="emotion_spike")

    async def _drift_bg(self, _):
        if hasattr(self.ec, "update_emotions"):
            await maybe_async(self.ec.update_emotions)
        # FIX: Use get_emotional_state and extract emotions properly
        st = await maybe_async(self.ec.get_emotional_state)
        emotions = {}
        if st and 'emotional_state_matrix' in st:
            matrix = st['emotional_state_matrix']
            primary = matrix.get('primary_emotion', {})
            emotions[primary.get('name', 'neutral')] = primary.get('intensity', 0)
            # Include secondary emotions
            for name, data in matrix.get('secondary_emotions', {}).items():
                emotions[name] = data.get('intensity', 0)
        
        sust = [e for e, v in emotions.items() if v > .5]
        if len(sust) >= 2:
            return {"sustained_emotions": sust, "significance": .4}



# ── NEEDS ──────────────────────────────────────────────────────────────────
@register_adapter("needs_system")
class NeedsAdapter(EnhancedWorkspaceModule):
    name = "needs"

    # --------------------------------------------------------------------- #
    # INITIALISATION                                                        #
    # --------------------------------------------------------------------- #
    def __init__(self, ns, ws: Optional[GlobalWorkspace] = None):
        super().__init__(ws)
        self.ns = ns
        # run every unconscious cycle; 0.6 = moderate attention bonus
        self.register_unconscious("needs_homeostasis", self._homeo_bg, 0.6)

    # --------------------------------------------------------------------- #
    # HELPERS                                                               #
    # --------------------------------------------------------------------- #
    async def _call_ns(self, *attr_names, **kw):
        """
        Try the given attribute names in order (`foo_async`, then `foo`, …),
        invoke the first one that exists, and await the result if necessary.
        """
        for attr in attr_names:
            fn = getattr(self.ns, attr, None)
            if fn:
                return await maybe_async(fn, **kw)
        return None


    async def on_phase(self, phase: int):
        # We only fire in the first (0) conscious phase so we don’t starve others
        if phase != 0 or not self.ns:
            return

        state = await self._call_ns("get_needs_state_async", "get_needs_state")
        if not state:
            return

        # ---- new schema: NeedsStateResponse.needs ---------------------- #
        if hasattr(state, "needs"):
            for need in state.needs:
                if need.drive_strength > 0.8:
                    await self.submit(
                        {
                            "need":  need.name,
                            "drive": need.drive_strength,
                            "level": need.level,
                        },
                        salience=need.drive_strength,
                        context_tag="need_spike",
                    )
        # ---- legacy schema: {name: {...}} ------------------------------ #
        else:
            for name, data in state.items():
                drive = data.get("drive", 0)
                if drive > 0.8:
                    await self.submit(
                        {"need": name, **data},
                        salience=drive,
                        context_tag="need_spike",
                    )


    async def _homeo_bg(self, _):
        # 1) apply decay + goal integration ------------------------------ #
        await self._call_ns("update_needs_async", "update_needs")

        # 2) fetch state again ------------------------------------------- #
        state = await self._call_ns("get_needs_state_async", "get_needs_state")
        if not state:
            return

        # 3) collect moderately-urgent drives (.4 - .7) ------------------ #
        if hasattr(state, "needs"):
            moderate = [n.name for n in state.needs if 0.4 < n.drive_strength < 0.7]
        else:
            moderate = [n for n, d in state.items() if 0.4 < d.get("drive", 0) < 0.7]

        # 4) publish summary note if multiple drives hover in the “itch” zone
        if len(moderate) >= 3:
            return {"moderate_needs": moderate, "significance": 0.5}


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

@register_adapter("emergency_response")
class EmergencyResponseAdapter(EnhancedWorkspaceModule):
    """Handles emergency situations requiring immediate response"""
    name = "emergency_response"
    
    def __init__(self, brain, ws=None):
        super().__init__(ws)
        self.brain = brain
    
    async def on_phase(self, phase: int):
        if phase != 2:
            return
            
        # Check for emergency signals
        emergencies = [p for p in self.ws.focus 
                      if p.context_tag in ["emergency", "critical_error", "system_failure"]]
        
        if emergencies:
            highest_priority = max(emergencies, key=lambda p: p.salience)
            response = await self._handle_emergency(highest_priority)
            
            await self.submit(
                {
                    "response": response,
                    "confidence": 1.0,
                    "override_reason": "emergency",
                    "priority": "critical"
                },
                salience=1.0,
                context_tag="complete_response"
            )
    
    async def _handle_emergency(self, emergency):
        # Generate appropriate emergency response
        return f"Emergency detected. Taking immediate action: {emergency.content}"

@register_adapter("creative_response")
class CreativeResponseAdapter(EnhancedWorkspaceModule):
    """Can generate complete creative responses when inspiration strikes"""
    name = "creative_response"
    
    def __init__(self, cs, ws=None):
        super().__init__(ws)
        self.cs = cs  # Creative system reference
        
    async def on_phase(self, phase: int):
        if phase != 2:
            return
            
        # Check if we have strong creative inspiration
        creative_signals = [p for p in self.ws.focus 
                           if p.context_tag in ["creative_synthesis", "imagination_output"]]
        
        if creative_signals and any(p.salience > 0.8 for p in creative_signals):
            # Check if user asked for creative content
            user_wants_creative = any(
                p.context_tag == "user_input" and 
                any(word in str(p.content).lower() 
                    for word in ["write", "create", "imagine", "story", "poem"])
                for p in self.ws.focus
            )
            
            if user_wants_creative:
                best_creative = max(creative_signals, key=lambda p: p.salience)
                await self.submit(
                    {
                        "response": best_creative.content.get("imagination", "Let me create something for you..."),
                        "confidence": best_creative.salience,
                        "override_reason": "creative_inspiration"
                    },
                    salience=best_creative.salience,
                    context_tag="complete_response"
                )


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
        if focus and hasattr(self.tm, 'generate_thought'):
            # Import ThoughtSource from the module
            try:
                from nyx.core.internal_thoughts import ThoughtSource
                
                # Generate a reasoning thought about the current focus
                context = {
                    "workspace_focus": focus,
                    "focus_count": len(focus),
                    "reasoning_target": "understanding_context"
                }
                
                thought = await maybe_async(self.tm.generate_thought, 
                                          ThoughtSource.REASONING, 
                                          context)
                
                if thought and hasattr(thought, 'content'):
                    await self.submit({"thoughts": [thought.content],
                                       "thought_id": thought.thought_id,
                                       "priority": thought.priority.value,
                                       "type": "reasoning_thought"},
                                      salience=.5,
                                      context_tag="internal_thought")
                    
                    # Generate additional perception thought if high activity
                    if len(focus) > 2:
                        perception_thought = await maybe_async(
                            self.tm.generate_thought,
                            ThoughtSource.PERCEPTION,
                            {"observation": "Multiple items in focus", "items": focus}
                        )
                        if perception_thought and hasattr(perception_thought, 'content'):
                            await self.submit({"thoughts": [perception_thought.content],
                                               "thought_id": perception_thought.thought_id,
                                               "type": "perception_thought"},
                                              salience=.4,
                                              context_tag="internal_thought")
            except ImportError:
                logger.error("Could not import ThoughtSource from internal_thoughts module")

    async def _wand_bg(self, view):
        if not view.focus and random.random() < .1 and hasattr(self.tm, 'generate_thought'):
            try:
                from nyx.core.internal_thoughts import ThoughtSource
                
                # Generate wandering/imagination thought
                thought = await maybe_async(self.tm.generate_thought,
                                          ThoughtSource.IMAGINATION,
                                          {"context": "idle_wandering", 
                                           "prompt": "What comes to mind when nothing is happening?"})
                
                if thought and hasattr(thought, 'content'):
                    return {"wandering_thought": thought.content,
                            "thought_id": thought.thought_id,
                            "significance": .3}
            except ImportError:
                pass
        return None

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
        # FIX: Use get_hormone_levels instead of get_levels
        lv = await maybe_async(self.hs.get_hormone_levels)
        for h, data in (lv or {}).items():
            # Handle the hormone data structure properly
            v = data.get('value', 0.5) if isinstance(data, dict) else data
            if v > .8 or v < .2:
                await self.submit({"hormone": h, "level": v},
                                  salience=abs(v - .5) * 2,
                                  context_tag="hormone_extreme")

    async def _reg_bg(self, _):
        # FIX: Use update_hormone_cycles with context
        if hasattr(self.hs, "update_hormone_cycles"):
            # Create a minimal context for the hormone system
            from nyx.core.emotions.context import EmotionalContext
            ctx = EmotionalContext()
            await maybe_async(self.hs.update_hormone_cycles, ctx)
        
        # FIX: Use get_hormone_levels
        lv = await maybe_async(self.hs.get_hormone_levels)
        # Count hormones out of normal range
        imbalanced = 0
        for h, data in (lv or {}).items():
            v = data.get('value', 0.5) if isinstance(data, dict) else data
            if v > .7 or v < .3:
                imbalanced += 1
        
        if imbalanced >= 3:
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

# ── KNOWLEDGE CORE ────────────────────────────────────────────────────────
@register_adapter("knowledge_core")
class KnowledgeCoreAdapter(EnhancedWorkspaceModule):
    name = "knowledge"
    def __init__(self, kc, ws=None):
        super().__init__(ws); self.kc = kc
        self.register_unconscious("knowledge_update", self._update_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.kc:
            return
        # Look for queries that need knowledge
        for p in self.ws.focus:
            if p.context_tag == "user_input":
                facts = await maybe_async(self.kc.retrieve_relevant_facts, str(p.content))
                if facts:
                    await self.submit({"facts": facts, "query": str(p.content)},
                                      salience=.7,
                                      context_tag="knowledge_facts")

    async def _update_bg(self, _):
        if hasattr(self.kc, "update_knowledge_graph"):
            updated = await maybe_async(self.kc.update_knowledge_graph)
            if updated:
                return {"knowledge_updated": len(updated), "significance": .4}


# ── MEMORY ORCHESTRATOR ───────────────────────────────────────────────────
@register_adapter("memory_orchestrator")
class MemoryOrchestratorAdapter(EnhancedWorkspaceModule):
    name = "memory_orchestrator"
    def __init__(self, mo, ws=None):
        super().__init__(ws); self.mo = mo
        self.register_unconscious("memory_coordination", self._coord_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.mo:
            return
        # Coordinate memory operations across systems
        active_queries = [p.content for p in self.ws.focus if p.context_tag == "memory_query"]
        if active_queries:
            coord_result = await maybe_async(self.mo.coordinate_retrieval, active_queries)
            if coord_result:
                await self.submit({"coordinated_memories": coord_result},
                                  salience=.8,
                                  context_tag="memory_coordination")

    async def _coord_bg(self, _):
        if hasattr(self.mo, "synchronize_memory_systems"):
            sync_result = await maybe_async(self.mo.synchronize_memory_systems)
            if sync_result:
                return {"memory_systems_synced": True, "significance": .5}


# ── REWARD SYSTEM ─────────────────────────────────────────────────────────
@register_adapter("reward_system")
class RewardSystemAdapter(EnhancedWorkspaceModule):
    name = "reward"
    def __init__(self, rs, ws=None):
        super().__init__(ws); self.rs = rs; self._next = 0.0
        self.register_unconscious("reward_processing", self._process_bg, .8)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.rs:
            return
        # Check for rewarding stimuli
        if time.time() < self._next:
            return
        self._next = time.time() + 1.0     # 1 s cooldown
        for p in self.ws.focus:
            if p.salience > .7:
                reward = await maybe_async(self.rs.evaluate_stimulus, p.content)
                if reward and reward.get("value", 0) > .5:
                    await self.submit({"reward": reward, "stimulus": p.source},
                                      salience=reward["value"],
                                      context_tag="reward_signal")

    async def _process_bg(self, view):
        if hasattr(self.rs, "process_delayed_rewards"):
            processed = await maybe_async(self.rs.process_delayed_rewards)
            if processed:
                return {"delayed_rewards": len(processed), "significance": .6}


# ── DIGITAL SOMATOSENSORY ─────────────────────────────────────────────────
@register_adapter("digital_somatosensory_system")
class DigitalSomatosensoryAdapter(EnhancedWorkspaceModule):
    name = "somatosensory"
    def __init__(self, dss, ws=None):
        super().__init__(ws); self.dss = dss
        self.register_unconscious("somatic_monitoring", self._monitor_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.dss:
            return
        # Process somatic signals
        signals = await maybe_async(self.dss.get_current_sensations)
        if signals:
            for signal in signals:
                if signal.get("intensity", 0) > .6:
                    await self.submit(signal,
                                      salience=signal["intensity"],
                                      context_tag="somatic_signal")

    async def _monitor_bg(self, _):
        if hasattr(self.dss, "update_body_state"):
            state = await maybe_async(self.dss.update_body_state)
            if state and state.get("arousal", 0) > .7:
                return {"high_arousal": True, "level": state["arousal"], "significance": .7}


# ── EXPERIENCE INTERFACE ──────────────────────────────────────────────────
@register_adapter("experience_interface")
class ExperienceInterfaceAdapter(EnhancedWorkspaceModule):
    name = "experience"
    def __init__(self, ei, ws=None):
        super().__init__(ws); self.ei = ei
        self.register_unconscious("experience_processing", self._process_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.ei:
            return
        # Process significant workspace content into experiences
        significant = [p.content for p in self.ws.focus if p.salience > .7]
        if significant:
            exp = await maybe_async(self.ei.create_experience, significant)
            if exp:
                await self.submit({"experience": exp},
                                  salience=.8,
                                  context_tag="new_experience")

    async def _process_bg(self, _):
        if hasattr(self.ei, "consolidate_experiences"):
            consolidated = await maybe_async(self.ei.consolidate_experiences)
            if consolidated:
                return {"experiences_consolidated": len(consolidated), "significance": .6}


# ── TEMPORAL PERCEPTION ───────────────────────────────────────────────────
@register_adapter("temporal_perception")
class TemporalPerceptionAdapter(EnhancedWorkspaceModule):
    name = "temporal"
    def __init__(self, tp, ws=None):
        super().__init__(ws); self.tp = tp
        self.register_unconscious("time_tracking", self._track_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.tp:
            return
        # Update temporal context
        temporal_context = await maybe_async(self.tp.get_temporal_context)
        if temporal_context and temporal_context.get("significant_change"):
            await self.submit(temporal_context,
                              salience=.6,
                              context_tag="temporal_shift")

    async def _track_bg(self, _):
        if hasattr(self.tp, "update_time_perception"):
            perception = await maybe_async(self.tp.update_time_perception)
            if perception and perception.get("time_distortion", 0) > .3:
                return {"time_distortion": perception["time_distortion"], "significance": .5}


# ── AGENTIC ACTION GENERATOR ──────────────────────────────────────────────
@register_adapter("agentic_action_generator")
class AgenticActionAdapter(EnhancedWorkspaceModule):
    name = "action_generator"
    def __init__(self, aag, brain, ws=None):  # Need brain reference!
        super().__init__(ws)
        self.aag = aag
        self.brain = brain
        
    async def on_phase(self, phase: int):
        if phase != 1:  # Earlier phase to provide context
            return
            
        # Don't generate responses, just provide action context
        user_inputs = [p for p in self.ws.focus if p.context_tag == "user_input"]
        if not user_inputs:
            return
        
        # Analyze what kind of action might be appropriate
        action_context = await self._analyze_action_context()
        
        if action_context:
            await self.submit(
                action_context,
                salience=0.7,
                context_tag="action_context"
            )

    async def _analyze_action_context(self):
        """Analyze workspace to suggest action context"""
        # Look at emotional state, memories, etc. to suggest action type
        emotions = [p for p in self.ws.focus if p.context_tag == "emotion_spike"]
        memories = [p for p in self.ws.focus if p.context_tag == "memory_recall"]
        
        context = {
            "suggested_action_type": "conversational",  # default
            "emotional_tone": "neutral",
            "relevant_capabilities": []
        }
        
        # Adjust based on workspace state
        if any(e.content.get("emotion") == "curiosity" for e in emotions):
            context["suggested_action_type"] = "exploratory"
            context["relevant_capabilities"].append("knowledge_retrieval")
        
        if memories and len(memories) > 2:
            context["suggested_action_type"] = "reflective"
            context["relevant_capabilities"].append("memory_synthesis")
        
        return context

    async def _plan_bg(self, _):
        if hasattr(self.aag, "update_action_models"):
            updated = await maybe_async(self.aag.update_action_models)
            if updated:
                return {"action_models_updated": True, "significance": .6}


# ── INTERNAL FEEDBACK ─────────────────────────────────────────────────────
@register_adapter("internal_feedback")
class InternalFeedbackAdapter(EnhancedWorkspaceModule):
    name = "feedback"
    def __init__(self, ifs, ws=None):
        super().__init__(ws); self.ifs = ifs
        self.register_unconscious("feedback_loop", self._loop_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.ifs:
            return
        # Generate feedback on recent actions
        recent_actions = [p for p in self.ws.proposals[-10:] if p.context_tag == "action_proposal"]
        if recent_actions:
            feedback = await maybe_async(self.ifs.evaluate_actions, recent_actions)
            if feedback:
                await self.submit(feedback,
                                  salience=.7,
                                  context_tag="internal_feedback")

    async def _loop_bg(self, _):
        if hasattr(self.ifs, "process_feedback_queue"):
            processed = await maybe_async(self.ifs.process_feedback_queue)
            if processed:
                return {"feedback_processed": len(processed), "significance": .5}


# ── DYNAMIC ADAPTATION ────────────────────────────────────────────────────
@register_adapter("dynamic_adaptation")
class DynamicAdaptationAdapter(EnhancedWorkspaceModule):
    name = "adaptation"
    def __init__(self, da, ws=None):
        super().__init__(ws); self.da = da; self._next = 0.0
        self.register_unconscious("adaptation_check", self._adapt_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.da:
            return
        # Check for adaptation opportunities
        if time.time() < self._next:
            return
        self._next = time.time() + 2.0      # 2 s cooldown
        performance = self.ws.state.get("performance_metrics", {})
        if performance:
            adaptation = await maybe_async(self.da.suggest_adaptation, performance)
            if adaptation:
                await self.submit(adaptation,
                                  salience=.7,
                                  context_tag="adaptation_suggestion")

    async def _adapt_bg(self, view):
        if hasattr(self.da, "continuous_adaptation"):
            adapted = await maybe_async(self.da.continuous_adaptation)
            if adapted:
                return {"adaptations_made": len(adapted), "significance": .6}


# ── SPATIAL MAPPER ────────────────────────────────────────────────────────
@register_adapter("spatial_mapper")
class SpatialMapperAdapter(EnhancedWorkspaceModule):
    name = "spatial_mapper"
    def __init__(self, sm, ws=None):
        super().__init__(ws); self.sm = sm
        self.register_unconscious("map_update", self._update_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.sm:
            return
        # Process spatial information
        for p in self.ws.focus:
            if p.context_tag == "location_update":
                mapping = await maybe_async(self.sm.update_map, p.content)
                if mapping:
                    await self.submit({"spatial_update": mapping},
                                      salience=.6,
                                      context_tag="spatial_mapping")

    async def _update_bg(self, _):
        if hasattr(self.sm, "consolidate_spatial_memory"):
            consolidated = await maybe_async(self.sm.consolidate_spatial_memory)
            if consolidated:
                return {"spatial_memory_updated": True, "significance": .4}


# ── SPATIAL MEMORY ────────────────────────────────────────────────────────
@register_adapter("spatial_memory")
class SpatialMemoryAdapter(EnhancedWorkspaceModule):
    name = "spatial_memory"
    def __init__(self, sm, ws=None):
        super().__init__(ws); self.sm = sm
        self.register_unconscious("spatial_recall", self._recall_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.sm:
            return
        # Retrieve spatial memories
        # pass raw **content** (list[str|dict]) to recall_locations
        queries = [p.content for p in self.ws.focus
                   if "location" in str(p.content).lower()]
        if queries:
            memories = await maybe_async(self.sm.recall_locations, queries)
            if memories:
                await self.submit({"spatial_memories": memories},
                                  salience=.6,
                                  context_tag="spatial_recall")

    async def _recall_bg(self, _):
        if hasattr(self.sm, "prune_old_locations"):
            pruned = await maybe_async(self.sm.prune_old_locations)
            if pruned:
                return {"locations_pruned": pruned, "significance": .3}


# ── NOVELTY ENGINE ────────────────────────────────────────────────────────
@register_adapter("novelty_engine")
class NoveltyEngineAdapter(EnhancedWorkspaceModule):
    name = "novelty"
    def __init__(self, ne, ws=None):
        super().__init__(ws); self.ne = ne
        self.register_unconscious("novelty_detection", self._detect_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.ne:
            return
        # Detect novel patterns
        for p in self.ws.focus:
            novelty = await maybe_async(self.ne.assess_novelty, p.content)
            if novelty and novelty.get("score", 0) > .7:
                await self.submit({"novel_content": p.content,
                                   "novelty": novelty},
                                  salience=_clamp(novelty["score"]),
                                  context_tag="novelty_detected")

    async def _detect_bg(self, view):
        if hasattr(self.ne, "update_novelty_baseline"):
            updated = await maybe_async(self.ne.update_novelty_baseline, view.recent)
            if updated:
                return {"novelty_baseline_updated": True, "significance": .5}


# ── RECOGNITION MEMORY ────────────────────────────────────────────────────
@register_adapter("recognition_memory")
class RecognitionMemoryAdapter(EnhancedWorkspaceModule):
    name = "recognition"
    def __init__(self, rm, ws=None):
        super().__init__(ws); self.rm = rm
        self.register_unconscious("pattern_recognition", self._recognize_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.rm:
            return
        # Check for familiar patterns
        for p in self.ws.focus:
            recognition = await maybe_async(self.rm.recognize_pattern, p.content)
            if recognition and recognition.get("confidence", 0) > .6:
                await self.submit(recognition,
                                  salience=recognition["confidence"],
                                  context_tag="pattern_recognized")

    async def _recognize_bg(self, _):
        if hasattr(self.rm, "consolidate_patterns"):
            consolidated = await maybe_async(self.rm.consolidate_patterns)
            if consolidated:
                return {"patterns_consolidated": len(consolidated), "significance": .5}


# ── CREATIVE MEMORY ───────────────────────────────────────────────────────
@register_adapter("creative_memory")
class CreativeMemoryAdapter(EnhancedWorkspaceModule):
    name = "creative_memory"
    def __init__(self, cm, ws=None):
        super().__init__(ws); self.cm = cm
        self.register_unconscious("creative_synthesis", self._synth_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.cm:
            return
        # Look for creative opportunities
        if any(p.context_tag == "novelty_detected" for p in self.ws.focus):
            synthesis = await maybe_async(self.cm.synthesize_creative_memory)
            if synthesis:
                await self.submit(synthesis,
                                  salience=.8,
                                  context_tag="creative_synthesis")

    async def _synth_bg(self, _):
        if hasattr(self.cm, "incubate_ideas"):
            ideas = await maybe_async(self.cm.incubate_ideas)
            if ideas:
                return {"ideas_incubated": len(ideas), "significance": .6}


# ── EXPERIENCE CONSOLIDATION ──────────────────────────────────────────────
@register_adapter("experience_consolidation")
class ExperienceConsolidationAdapter(EnhancedWorkspaceModule):
    name = "consolidation"
    def __init__(self, ec, ws=None):
        super().__init__(ws); self.ec = ec
        self.register_unconscious("consolidation_cycle", self._consolidate_bg, .8)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.ec:
            return
        # Check if consolidation needed based on time
        if hasattr(self.ec, 'get_consolidation_insights'):
            insights = await maybe_async(self.ec.get_consolidation_insights)
            if insights and insights.ready_for_consolidation:
                await self.submit({"ready_for_consolidation": True,
                                   "total_consolidations": insights.total_consolidations},
                                  salience=.7,
                                  context_tag="consolidation_ready")

    async def _consolidate_bg(self, _):
        if random.random() < .1:  # 10% chance per cycle
            # Use run_consolidation_cycle which is the actual method
            if hasattr(self.ec, 'run_consolidation_cycle'):
                result = await maybe_async(self.ec.run_consolidation_cycle)
                if result and hasattr(result, 'consolidations_created'):
                    if result.consolidations_created > 0:
                        return {"consolidations_created": result.consolidations_created,
                                "memories_processed": result.source_memories_processed,
                                "significance": .8}
        return None


# ── CROSS USER MANAGER ────────────────────────────────────────────────────
@register_adapter("cross_user_manager")
class CrossUserManagerAdapter(EnhancedWorkspaceModule):
    name = "cross_user"
    def __init__(self, cum, ws=None):
        super().__init__(ws); self.cum = cum
        self.register_unconscious("cross_user_sync", self._sync_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.cum:
            return
        # Share significant experiences
        # send only the payload, not whole Proposal objects
        significant = [p.content for p in self.ws.focus
                       if p.salience > .8 and p.context_tag == "new_experience"]
        if significant:
            shared = await maybe_async(self.cum.share_experiences, significant)
            if shared:
                await self.submit({"shared_experiences": len(shared)},
                                  salience=.6,
                                  context_tag="cross_user_share")

    async def _sync_bg(self, _):
        if hasattr(self.cum, "sync_cross_user_knowledge"):
            synced = await maybe_async(self.cum.sync_cross_user_knowledge)
            if synced:
                return {"cross_user_synced": True, "significance": .5}


# ── REFLEXIVE SYSTEM ──────────────────────────────────────────────────────
@register_adapter("reflexive_system")
class ReflexiveSystemAdapter(EnhancedWorkspaceModule):
    name = "reflexive"
    def __init__(self, rs, ws=None):
        super().__init__(ws); self.rs = rs
        self.register_unconscious("reflex_monitoring", self._monitor_bg, .9)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.rs:
            return
        # Check for reflex triggers
        for p in self.ws.focus:
            if p.salience > .9:  # High salience triggers
                reflex = await maybe_async(self.rs.check_reflexes, p.content)
                if reflex:
                    await self.submit({"reflex_action": reflex},
                                      salience=1.0,
                                      context_tag="reflex_response")

    async def _monitor_bg(self, _):
        if hasattr(self.rs, "update_reflex_patterns"):
            updated = await maybe_async(self.rs.update_reflex_patterns)
            if updated:
                return {"reflexes_updated": len(updated), "significance": .7}

@register_adapter("reflexive_override")  # New adapter, different brain attribute
class ReflexiveOverrideAdapter(EnhancedWorkspaceModule):
    """Can generate complete responses for reflexive/emergency situations"""
    name = "reflexive_override"
    
    def __init__(self, rs, ws=None):
        super().__init__(ws)
        self.rs = rs  # Reference to reflexive system if needed
        self.register_unconscious("safety_monitor", self._safety_bg)
    
    async def on_phase(self, phase: int):
        if phase != 2:  # Late phase
            return
            
        # Check for emergency/reflexive situations
        for p in self.ws.focus:
            if p.context_tag == "safety_alert" and p.salience > 0.9:
                response = await self._generate_safety_response(p.content)
                await self.submit(
                    {
                        "response": response,
                        "confidence": 0.95,
                        "override_reason": "safety"
                    },
                    salience=1.0,
                    context_tag="complete_response"
                )
            elif p.context_tag == "reflex_response" and p.salience == 1.0:
                # Handle reflex responses that need immediate output
                await self.submit(
                    {
                        "response": p.content.get("reflex_action", "I need to respond immediately."),
                        "confidence": 0.9,
                        "override_reason": "reflex"
                    },
                    salience=0.95,
                    context_tag="complete_response"
                )
    
    async def _generate_safety_response(self, alert_content):
        """Generate appropriate safety response based on alert type"""
        if "harmful_content" in str(alert_content):
            return "I cannot engage with that type of content."
        elif "protocol_violation" in str(alert_content):
            return "That would violate our established protocols."
        else:
            return "I need to handle this carefully for safety reasons."
    
    async def _safety_bg(self, view):
        """Background safety monitoring"""
        unsafe_patterns = ["harm", "danger", "emergency", "unsafe"]
        for p in view.recent:
            if any(pattern in str(p.content).lower() for pattern in unsafe_patterns):
                return {"safety_concern": True, "pattern": p.content, "significance": .8}
        return None


# ── PROCEDURAL MEMORY ─────────────────────────────────────────────────────
@register_adapter("procedural_memory_manager")
class ProceduralMemoryAdapter(EnhancedWorkspaceModule):
    name = "procedural"
    def __init__(self, pm, ws=None):
        super().__init__(ws); self.pm = pm
        self.register_unconscious("skill_practice", self._practice_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.pm:
            return
        # Retrieve relevant procedures
        for p in self.ws.focus:
            if p.context_tag == "action_proposal":
                procedures = await maybe_async(self.pm.get_relevant_procedures, p.content)
                if procedures:
                    await self.submit({"procedures": procedures},
                                      salience=.7,
                                      context_tag="procedural_knowledge")

    async def _practice_bg(self, _):
        if hasattr(self.pm, "practice_procedures"):
            practiced = await maybe_async(self.pm.practice_procedures)
            if practiced:
                return {"procedures_practiced": len(practiced), "significance": .4}


# ── SYNC DAEMON ───────────────────────────────────────────────────────────
@register_adapter("sync_daemon")
class SyncDaemonAdapter(EnhancedWorkspaceModule):
    name = "sync"
    def __init__(self, sd, ws=None):
        super().__init__(ws); self.sd = sd
        self.register_unconscious("sync_check", self._sync_bg, .8)

    async def on_phase(self, phase: int):
        # Sync daemon works across all phases
        if not self.sd:
            return
        if phase == 2:  # End of cycle
            sync_status = await maybe_async(self.sd.get_sync_status)
            if sync_status and sync_status.get("out_of_sync"):
                await self.submit(sync_status,
                                  salience=.9,
                                  context_tag="sync_required")

    async def _sync_bg(self, _):
        if hasattr(self.sd, "background_sync"):
            result = await maybe_async(self.sd.background_sync)
            if result:
                return {"systems_synced": result.get("synced_count", 0), "significance": .7}


# ── AGENT EVALUATOR ───────────────────────────────────────────────────────
@register_adapter("agent_evaluator")
class AgentEvaluatorAdapter(EnhancedWorkspaceModule):
    name = "evaluator"
    def __init__(self, ae, ws=None):
        super().__init__(ws); self.ae = ae
        self.register_unconscious("agent_performance", self._eval_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.ae:
            return
        # Evaluate agent actions
        agent_actions = [p for p in self.ws.proposals[-20:] if "agent" in p.source]
        if agent_actions:
            evaluation = await maybe_async(self.ae.evaluate_agents, agent_actions)
            if evaluation:
                await self.submit(evaluation,
                                  salience=.6,
                                  context_tag="agent_evaluation")

    async def _eval_bg(self, _):
        if hasattr(self.ae, "compile_performance_report"):
            report = await maybe_async(self.ae.compile_performance_report)
            if report and report.get("underperforming_agents"):
                return {"underperforming_agents": len(report["underperforming_agents"]), 
                        "significance": .6}


# ── ISSUE TRACKING ────────────────────────────────────────────────────────
@register_adapter("issue_tracking_system")
class IssueTrackingAdapter(EnhancedWorkspaceModule):
    name = "issues"
    def __init__(self, its, ws=None):
        super().__init__(ws); self.its = its
        self.register_unconscious("issue_monitor", self._monitor_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.its:
            return
        # Check for new issues from errors
        errors = [p for p in self.ws.focus if p.context_tag == "error" or "error" in str(p.content)]
        if errors:
            for error in errors:
                issue = await maybe_async(self.its.create_issue_from_error, error.content)
                if issue:
                    await self.submit({"new_issue": issue},
                                      salience=.8,
                                      context_tag="issue_created")

    async def _monitor_bg(self, _):
        if hasattr(self.its, "get_critical_issues"):
            critical = await maybe_async(self.its.get_critical_issues)
            if critical:
                return {"critical_issues": len(critical), "significance": .8}


# ── MODULE OPTIMIZER ──────────────────────────────────────────────────────
@register_adapter("module_optimizer")
class ModuleOptimizerAdapter(EnhancedWorkspaceModule):
    name = "optimizer"
    def __init__(self, mo, ws=None):
        super().__init__(ws); self.mo = mo
        self.register_unconscious("optimization_cycle", self._optimize_bg, .5)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.mo:
            return
        # Check performance metrics
        metrics = self.ws.state.get("module_metrics", {})
        if metrics:
            optimization = await maybe_async(self.mo.suggest_optimizations, metrics)
            if optimization:
                await self.submit(optimization,
                                  salience=.6,
                                  context_tag="optimization_suggestion")

    async def _optimize_bg(self, _):
        if hasattr(self.mo, "auto_optimize"):
            optimized = await maybe_async(self.mo.auto_optimize)
            if optimized:
                return {"modules_optimized": len(optimized), "significance": .5}


# ── SYSTEM HEALTH CHECKER ─────────────────────────────────────────────────
@register_adapter("system_health_checker")
class SystemHealthAdapter(EnhancedWorkspaceModule):
    name = "health"
    def __init__(self, shc, ws=None):
        super().__init__(ws); self.shc = shc
        self.register_unconscious("health_monitor", self._monitor_bg, .9)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.shc:
            return
        # Quick health check at start of cycle
        health = await maybe_async(self.shc.quick_health_check)
        if health and health.get("status") != "healthy":
            await self.submit(health,
                              salience=.9,
                              context_tag="health_alert")

    async def _monitor_bg(self, _):
        if hasattr(self.shc, "deep_health_check"):
            issues = await maybe_async(self.shc.deep_health_check)
            if issues:
                return {"health_issues": len(issues), "significance": .8}


# ── IMAGINATION SIMULATOR ─────────────────────────────────────────────────
@register_adapter("imagination_simulator")
class ImaginationAdapter(EnhancedWorkspaceModule):
    name = "imagination"
    def __init__(self, ims, ws=None):
        super().__init__(ws); self.ims = ims
        self.register_unconscious("imaginative_drift", self._drift_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.ims:
            return
        # Use imagine_scenario which is the actual method
        prompts = [p for p in self.ws.focus if p.salience > .6]
        if prompts and hasattr(self.ims, 'imagine_scenario'):
            # Extract text from first high-salience prompt
            prompt_content = prompts[0].content
            description = ""
            if isinstance(prompt_content, dict):
                description = prompt_content.get("message", str(prompt_content))
            else:
                description = str(prompt_content)
            
            # Get brain state from workspace
            brain_state = {
                "workspace_focus": [p.content for p in self.ws.focus],
                "emotional_state": self.ws.state.get("emotional_state", {}),
                "context": "imagination_from_workspace"
            }
            
            result = await maybe_async(self.ims.imagine_scenario, description, brain_state)
            if result and result.get('success'):
                await self.submit({"imagination": result.get('reflection', ''),
                                   "key_insights": result.get('key_insights', []),
                                   "confidence": result.get('confidence', 0.5)},
                                  salience=.7,
                                  context_tag="imagination_output")

    async def _drift_bg(self, _):
        if random.random() < .2 and hasattr(self.ims, 'imagine_scenario'):
            # Create a daydream using imagine_scenario
            prompts = [
                "What if I could experience something completely new?",
                "Imagining a moment of perfect clarity",
                "What would happen if everything changed?",
                "Exploring the edges of possibility"
            ]
            description = random.choice(prompts)
            brain_state = {
                "context": "daydreaming",
                "emotional_state": {"valence": 0.5, "arousal": 0.3}
            }
            
            result = await maybe_async(self.ims.imagine_scenario, description, brain_state)
            if result and result.get('success'):
                return {"daydream": result.get('reflection', description),
                        "insights": result.get('key_insights', []),
                        "significance": .5}
        return None


# ── PROCESSING MANAGER ────────────────────────────────────────────────────
@register_adapter("processing_manager")
class ProcessingManagerAdapter(EnhancedWorkspaceModule):
    name = "processing"
    def __init__(self, pm, ws=None):
        super().__init__(ws); self.pm = pm
        self.register_unconscious("load_balancing", self._balance_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.pm:
            return
        # Manage processing load
        load = len(self.ws.focus) + len(self.ws.proposals)
        if load > 20:
            adjustment = await maybe_async(self.pm.adjust_processing, load)
            if adjustment:
                await self.submit(adjustment,
                                  salience=.8,
                                  context_tag="processing_adjustment")

    async def _balance_bg(self, _):
        if hasattr(self.pm, "balance_load"):
            balanced = await maybe_async(self.pm.balance_load)
            if balanced:
                return {"load_balanced": True, "significance": .6}


# ── CHECKPOINT PLANNER ────────────────────────────────────────────────────
@register_adapter("checkpoint_planner")
class CheckpointPlannerAdapter(EnhancedWorkspaceModule):
    name = "checkpoint"
    def __init__(self, cp, ws=None):
        super().__init__(ws); self.cp = cp
        self.register_unconscious("checkpoint_timing", self._check_bg, .8)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.cp:
            return
        # Check if checkpoint needed
        changes = self.ws.state.get("state_changes", 0)
        if changes > 50:
            plan = await maybe_async(self.cp.plan_checkpoint)
            if plan:
                await self.submit(plan,
                                  salience=.9,
                                  context_tag="checkpoint_needed")

    async def _check_bg(self, _):
        if hasattr(self.cp, "is_checkpoint_due"):
            due = await maybe_async(self.cp.is_checkpoint_due)
            if due:
                return {"checkpoint_due": True, "significance": .8}


# ── FEMDOM SYSTEMS ────────────────────────────────────────────────────────
# Note: Many femdom systems already have adapters, adding the missing ones

@register_adapter("protocol_enforcement")
class ProtocolEnforcementAdapter(EnhancedWorkspaceModule):
    name = "protocol"
    def __init__(self, pe, ws=None):
        super().__init__(ws); self.pe = pe
        self.register_unconscious("protocol_monitor", self._monitor_bg, .8)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.pe:
            return
        # Check for protocol violations
        for p in self.ws.focus:
            if p.context_tag == "user_input":
                violation = await maybe_async(self.pe.check_protocol, p.content)
                if violation:
                    await self.submit(violation,
                                      salience=.9,
                                      context_tag="protocol_violation")

    async def _monitor_bg(self, _):
        if hasattr(self.pe, "update_protocols"):
            updated = await maybe_async(self.pe.update_protocols)
            if updated:
                return {"protocols_updated": len(updated), "significance": .6}


@register_adapter("body_service_system")
class BodyServiceAdapter(EnhancedWorkspaceModule):
    name = "body_service"
    def __init__(self, bss, ws=None):
        super().__init__(ws); self.bss = bss
        self.register_unconscious("service_tracking", self._track_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.bss:
            return
        # Track service acts
        for p in self.ws.focus:
            if "service" in str(p.content).lower():
                tracked = await maybe_async(self.bss.track_service, p.content)
                if tracked:
                    await self.submit(tracked,
                                      salience=.8,
                                      context_tag="service_tracked")

    async def _track_bg(self, _):
        if hasattr(self.bss, "evaluate_service_quality"):
            quality = await maybe_async(self.bss.evaluate_service_quality)
            if quality and quality.get("score", 0) > .8:
                return {"excellent_service": True, "score": quality["score"], "significance": .7}


@register_adapter("psychological_dominance")
class PsychologicalDominanceAdapter(EnhancedWorkspaceModule):
    name = "psych_dom"
    def __init__(self, pd, ws=None):
        super().__init__(ws); self.pd = pd
        self.register_unconscious("psych_analysis", self._analyze_bg, .8)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.pd:
            return
        # Analyze psychological state
        user_states = [p for p in self.ws.focus if p.context_tag == "user_mental_state"]
        if user_states:
            analysis = await maybe_async(self.pd.analyze_submission_depth, user_states)
            if analysis:
                await self.submit(analysis,
                                  salience=.8,
                                  context_tag="psychological_analysis")

    async def _analyze_bg(self, _):
        if hasattr(self.pd, "update_psychological_models"):
            updated = await maybe_async(self.pd.update_psychological_models)
            if updated:
                return {"psych_models_updated": True, "significance": .7}


@register_adapter("orgasm_control_system")
class OrgasmControlAdapter(EnhancedWorkspaceModule):
    name = "orgasm_control"
    def __init__(self, ocs, ws=None):
        super().__init__(ws); self.ocs = ocs
        self.register_unconscious("arousal_monitor", self._monitor_bg, .9)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.ocs:
            return
        # Monitor arousal signals
        arousal_signals = [p for p in self.ws.focus if p.context_tag == "somatic_signal" 
                          and p.content.get("type") == "arousal"]
        if arousal_signals:
            control = await maybe_async(self.ocs.evaluate_control_needed, arousal_signals)
            if control:
                await self.submit(control,
                                  salience=.9,
                                  context_tag="orgasm_control")

    async def _monitor_bg(self, _):
        if hasattr(self.ocs, "update_arousal_patterns"):
            patterns = await maybe_async(self.ocs.update_arousal_patterns)
            if patterns:
                return {"arousal_patterns_updated": True, "significance": .8}


@register_adapter("sadistic_response_system")
class SadisticResponseAdapter(EnhancedWorkspaceModule):
    name = "sadistic"
    def __init__(self, srs, ws=None):
        super().__init__(ws); self.srs = srs
        self.register_unconscious("sadistic_calibration", self._calibrate_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.srs:
            return
        # Generate sadistic responses
        opportunities = [p for p in self.ws.focus if p.context_tag in 
                        ["protocol_violation", "submission_detected"]]
        if opportunities:
            response = await maybe_async(self.srs.generate_response, opportunities)
            if response:
                await self.submit(response,
                                  salience=.8,
                                  context_tag="sadistic_response")

    async def _calibrate_bg(self, _):
        if hasattr(self.srs, "calibrate_intensity"):
            calibrated = await maybe_async(self.srs.calibrate_intensity)
            if calibrated:
                return {"sadistic_calibration": calibrated, "significance": .6}


# ── STREAMING SYSTEMS ─────────────────────────────────────────────────────
@register_adapter("streaming_hormone_system")
class StreamingHormoneAdapter(EnhancedWorkspaceModule):
    name = "stream_hormones"
    def __init__(self, shs, ws=None):
        super().__init__(ws); self.shs = shs
        self.register_unconscious("stream_hormone_update", self._update_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.shs:
            return
        # Update hormones based on streaming events
        events = [p.content for p in self.ws.focus
                  if p.context_tag == "game_event"]
        if events:
            update = await maybe_async(self.shs.process_stream_events, events)
            if update:
                await self.submit(update,
                                  salience=.7,
                                  context_tag="stream_hormone_update")

    async def _update_bg(self, _):
        if hasattr(self.shs, "balance_stream_hormones"):
            balanced = await maybe_async(self.shs.balance_stream_hormones)
            if balanced:
                return {"stream_hormones_balanced": True, "significance": .5}


@register_adapter("streaming_reflection_engine")
class StreamingReflectionAdapter(EnhancedWorkspaceModule):
    name = "stream_reflection"
    def __init__(self, sre, ws=None):
        super().__init__(ws); self.sre = sre
        self.register_unconscious("stream_insight", self._insight_bg, .7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.sre:
            return
        # Reflect on streaming session
        session_data = self.ws.state.get("streaming_session")
        if session_data:
            reflection = await maybe_async(self.sre.reflect_on_session, session_data)
            if reflection:
                await self.submit(reflection,
                                  salience=.7,
                                  context_tag="stream_reflection")

    async def _insight_bg(self, _):
        if hasattr(self.sre, "generate_streaming_insights"):
            insights = await maybe_async(self.sre.generate_streaming_insights)
            if insights:
                return {"streaming_insights": len(insights), "significance": .6}


@register_adapter("cross_game_knowledge")
class CrossGameKnowledgeAdapter(EnhancedWorkspaceModule):
    name = "cross_game"
    def __init__(self, cgk, ws=None):
        super().__init__(ws); self.cgk = cgk
        self.register_unconscious("knowledge_transfer", self._transfer_bg, .6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.cgk:
            return
        # Apply cross-game knowledge
        game_context = [p for p in self.ws.focus if p.context_tag == "game_state"]
        if game_context:
            knowledge = await maybe_async(self.cgk.get_relevant_knowledge, game_context)
            if knowledge:
                await self.submit(knowledge,
                                  salience=.7,
                                  context_tag="cross_game_knowledge")

    async def _transfer_bg(self, _):
        if hasattr(self.cgk, "consolidate_game_knowledge"):
            consolidated = await maybe_async(self.cgk.consolidate_game_knowledge)
            if consolidated:
                return {"game_knowledge_consolidated": True, "significance": .6}


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ FACTORY                                                                 │
# ╰──────────────────────────────────────────────────────────────────────────╯
def _log_missing(brain):
    want = set(_REGISTRY)
    have = {k for k, v in vars(brain).items() if v is not None}
    miss = want - have
    if miss:
        logger.info("GW‑adapters skipped (attr None on NyxBrain): %s",
                       sorted(miss))


def build_gw_modules(brain) -> List[EnhancedWorkspaceModule]:
    modules: List[EnhancedWorkspaceModule] = []
    
    # Add standard modules
    for attr, cls in _REGISTRY.items():
        obj = getattr(brain, attr, None)
        if obj is None:
            continue
            
        # Special cases that need brain reference
        if attr == "agentic_action_generator":
            modules.append(cls(obj, brain))
            continue

        # Special case for social_tools
        if attr == "social_tools":
            modules.append(SocialBrowsingAdapter(obj, motiv=getattr(brain, "motivations", {})))
            continue
            
        # Standard case
        try:
            modules.append(cls(obj))
        except Exception as e:
            logger.error(f"Failed to create adapter for {attr}: {e}")
    
    # Add special response handling modules
    modules.extend([
        ResponseSynthesizerAdapter(brain),
        FallbackResponderAdapter(brain),
        ReflexiveOverrideAdapter(
            brain.reflexive_system if hasattr(brain, 'reflexive_system') else None
        ),
        EmergencyResponseAdapter(brain),
        CreativeResponseAdapter(
            brain.creative_system if hasattr(brain, 'creative_system') else None
        )
    ])
    
    # Log what we've loaded
    logger.info(f"Loaded {len(modules)} GWA modules: {[m.name for m in modules]}")
    
    return modules
