"""global_workspace_adapters.py

Adapters that let each of Nyx’s big subsystems plug straight into the
Global‑Workspace / Attention engine.  Every adapter:
  • inherits EnhancedWorkspaceModule (so it can register unconscious jobs)
  • exposes a conscious `on_phase` hook that injects salient content into
    the workspace focus
  • registers at least one unconscious background coroutine (dream‑like
    consolidation, drift, maintenance …)
  • uses the helper `maybe_async()` so it doesn’t matter whether the
    underlying subsystem API is sync or async.

Add or remove adapters freely – just mimic the pattern.
"""
from __future__ import annotations

import asyncio, random, time, math, functools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
from sentence_transformers import SentenceTransformer

_ST_MODEL: Optional[SentenceTransformer] = None        # lazy‑load to keep boot fast
def _get_st_model() -> SentenceTransformer:
    global _ST_MODEL         # pylint: disable=global-statement
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def _sem_match(txt: str, prompts: Sequence[str], thresh: float = .55) -> bool:
    """Semantic yes/no using a cheap SBERT embedding cache."""
    model = _get_st_model()
    vec_txt = model.encode(txt, normalize_embeddings=True)
    vec_ref = [model.encode(p, normalize_embeddings=True) for p in prompts]
    return any(_cos(vec_txt, v) >= thresh for v in vec_ref)

# NOTE: adjust this import if you moved the workspace implementation
from nyx.core.brain.workspace_v3 import EnhancedWorkspaceModule

# ---------------------------------------------------------------------------
# utility: run sync OR async functions transparently
# ---------------------------------------------------------------------------
async def maybe_async(fn: Callable, *args, **kwargs):
    """Call *fn* – await if it returns a coroutine, else return value directly."""
    res = fn(*args, **kwargs)
    if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
        return await res
    return res

# ---------------------------------------------------------------------------
# ADAPTER CLASSES – one per Nyx subsystem
# ---------------------------------------------------------------------------
class MemoryAdapter(EnhancedWorkspaceModule):
    """Surfaces relevant memories and runs consolidation in background."""
    def __init__(self, memory_core, ws=None):
        super().__init__(ws)
        self.name = "memory"
        self.mc = memory_core
        self.last_query = ""
        self.register_unconscious("memory_consolidation", self._consolidate_bg, threshold=0.7)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.mc:
            return
        # look for fresh user input in focus
        user_inputs = [p.content for p in self.ws.focus if p.context_tag == "user_input"]
        if not user_inputs:
            return
        query = str(user_inputs[0])
        if query == self.last_query:
            return
        self.last_query = query
        memories = await maybe_async(self.mc.retrieve_memories, query=query, limit=3)
        if memories:
            await self.submit({"memories": memories, "query": query}, salience=0.7, context_tag="memory_recall")

    async def _consolidate_bg(self, _view):
        if random.random() < 0.1 and hasattr(self.mc, "consolidate_recent_memories"):
            consolidated = await maybe_async(self.mc.consolidate_recent_memories)
            if consolidated:
                return {"consolidated_count": len(consolidated), "significance": 0.3}
        return None

# ---------------------------------------------------------------------------
class EmotionalAdapter(EnhancedWorkspaceModule):
    """Tracks emotional spikes and slow drift."""
    def __init__(self, emotional_core, ws=None):
        super().__init__(ws)
        self.name = "emotion"
        self.ec = emotional_core
        self.register_unconscious("emotion_drift", self._emotion_drift_bg, threshold=0.5)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.ec:
            return
        state = await maybe_async(self.ec.get_current_emotion)
        if not state:
            return
        for emo, level in state.items():
            if isinstance(level, (float, int)) and level > 0.7:
                await self.submit({"emotion": emo, "intensity": level}, salience=level, context_tag="emotion_spike")

    async def _emotion_drift_bg(self, _):
        if hasattr(self.ec, "update_emotions"):
            await maybe_async(self.ec.update_emotions)
        state = await maybe_async(self.ec.get_current_emotion)
        sustained = [e for e, v in (state or {}).items() if v > 0.5]
        if len(sustained) >= 2:
            return {"sustained_emotions": sustained, "significance": 0.4}
        return None

# ---------------------------------------------------------------------------
class NeedsAdapter(EnhancedWorkspaceModule):
    """Promotes urgent physiological / psychological needs."""
    def __init__(self, needs_system, ws=None):
        super().__init__(ws)
        self.name = "needs"
        self.ns = needs_system
        self.register_unconscious("needs_homeostasis", self._homeostasis_bg, threshold=0.6)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.ns:
            return
        state: Dict[str, Dict[str, float]] = await maybe_async(self.ns.get_needs_state)
        for need, data in (state or {}).items():
            if data.get("drive", 0) > 0.8:
                await self.submit({"need": need, **data}, salience=data["drive"], context_tag="need_spike")

    async def _homeostasis_bg(self, _):
        if hasattr(self.ns, "update_needs"):
            await maybe_async(self.ns.update_needs)
        state = await maybe_async(self.ns.get_needs_state)
        moderate = [n for n, d in (state or {}).items() if 0.4 < d.get("drive", 0) < 0.7]
        if len(moderate) >= 3:
            return {"moderate_needs": moderate, "significance": 0.5}
        return None

# ---------------------------------------------------------------------------
class GoalAdapter(EnhancedWorkspaceModule):
    def __init__(self, goal_manager, ws=None):
        super().__init__(ws)
        self.name = "goals"
        self.gm = goal_manager
        self.register_unconscious("goal_prune", self._prune_bg, threshold=0.4)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.gm:
            return
        goals = await maybe_async(self.gm.get_active_goals)
        for g in goals[:3]:
            if g.get("priority", 0) > 0.6:
                await self.submit(g, salience=g["priority"], context_tag="goal_active")

    async def _prune_bg(self, _):
        if hasattr(self.gm, "prune_completed_goals"):
            pruned = await maybe_async(self.gm.prune_completed_goals)
            if pruned:
                return {"pruned_goals": len(pruned), "significance": 0.3}
        return None

# ---------------------------------------------------------------------------
class MoodAdapter(EnhancedWorkspaceModule):
    def __init__(self, mood_manager, ws=None):
        super().__init__(ws)
        self.name = "mood"
        self.mm = mood_manager
        self.register_unconscious("mood_regulation", self._regulate_bg, threshold=0.5)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.mm:
            return
        mood = await maybe_async(self.mm.get_current_mood)
        if not mood:
            return
        data = mood.dict() if hasattr(mood, "dict") else mood
        val, ar = data.get("valence", 0), data.get("arousal", 0)
        if abs(val) > 0.7 or ar > 0.8:
            await self.submit(data, salience=max(abs(val), ar), context_tag="mood_extreme")

    async def _regulate_bg(self, _):
        if hasattr(self.mm, "regulate_mood"):
            regulated = await maybe_async(self.mm.regulate_mood)
            if regulated:
                return {"mood_regulated": True, "significance": 0.4}
        return None

# ---------------------------------------------------------------------------
class ReasoningAdapter(EnhancedWorkspaceModule):
    def __init__(self, reasoning_core, ws=None):
        super().__init__(ws)
        self.name = "reasoning"
        self.rc = reasoning_core
        self.register_unconscious("concept_activation", self._concept_bg, threshold=0.6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.rc:
            return
        if not self.ws.focus:
            return
        user_text = self.ws.focus[0].content
        reasoning = await maybe_async(self.rc.reason_about, user_text)
        if reasoning:
            await self.submit({"reasoning": reasoning}, salience=0.7, context_tag="reasoning_output")

    async def _concept_bg(self, _):
        if hasattr(self.rc, "activate_relevant_concepts"):
            act = await maybe_async(self.rc.activate_relevant_concepts)
            if act and len(act) > 2:
                return {"activated_concepts": act, "significance": 0.5}
        return None

# ---------------------------------------------------------------------------
class ReflectionAdapter(EnhancedWorkspaceModule):
    def __init__(self, reflection_engine, ws=None):
        super().__init__(ws)
        self.name = "reflection"
        self.refl = reflection_engine
        self.register_unconscious("insight_gen", self._insight_bg, threshold=0.7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.refl:
            return
        insight = await maybe_async(self.refl.reflect_on_recent)
        if insight and insight.get("significance", 0) > 0.6:
            await self.submit(insight, salience=insight["significance"], context_tag="reflection_insight")

    async def _insight_bg(self, _):
        if random.random() < 0.05 and hasattr(self.refl, "generate_insight"):
            ins = await maybe_async(self.refl.generate_insight)
            if ins:
                return {"insight": ins, "significance": 0.6}
        return None

# ---------------------------------------------------------------------------
class AttentionAdapter(EnhancedWorkspaceModule):
    def __init__(self, attentional_controller, ws=None):
        super().__init__(ws)
        self.name = "attention"
        self.ac = attentional_controller
        self.register_unconscious("attn_stats", self._stats_bg, threshold=0.5)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.ac:
            return
        salient = [{"target": p.source, "salience": p.salience} for p in self.ws.proposals[-10:] if p.salience > 0.6]
        if salient and hasattr(self.ac, "update_attention"):
            res = await maybe_async(self.ac.update_attention, salient_items=salient)
            if res:
                await self.submit({"attention_focus": res}, salience=0.6, context_tag="attention_update")

    async def _stats_bg(self, _):
        if hasattr(self.ac, "get_attention_statistics"):
            stats = await maybe_async(self.ac.get_attention_statistics)
            if stats and stats.get("miss_rate", 0) > 0.3:
                return {"attention_misses": stats["miss_rate"], "significance": 0.5}
        return None

# ---------------------------------------------------------------------------
class BodyImageAdapter(EnhancedWorkspaceModule):
    def __init__(self, body_image, ws=None):
        super().__init__(ws)
        self.name = "body"
        self.bi = body_image
        self.register_unconscious("proprio_bg", self._prop_bg, threshold=0.4)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.bi:
            return
        for p in self.ws.focus:
            if p.context_tag == "visual_percept" and hasattr(self.bi, "update_from_visual"):
                res = await maybe_async(self.bi.update_from_visual, p.content)
                if res and res.get("status") == "updated":
                    await self.submit({"body_update": res}, salience=0.5, context_tag="body_image_update")

    async def _prop_bg(self, _):
        if hasattr(self.bi, "update_from_somatic"):
            res = await maybe_async(self.bi.update_from_somatic)
            if res and res.get("proprioception_confidence", 1) < 0.3:
                return {"low_proprioception": True, "significance": 0.4}
        return None

# ---------------------------------------------------------------------------
class RelationshipAdapter(EnhancedWorkspaceModule):
    def __init__(self, relationship_manager, ws=None):
        super().__init__(ws)
        self.name = "relationship"
        self.rm = relationship_manager
        self.register_unconscious("rel_maint", self._maintain_bg, threshold=0.5)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.rm:
            return
        user_ids = [p.content.get("user_id") for p in self.ws.focus if p.context_tag == "user_input" and isinstance(p.content, dict)]
        if not user_ids:
            return
        uid = user_ids[0]
        state = await maybe_async(self.rm.get_relationship_state, uid)
        if state and state.get("trust", 1) < 0.3:
            await self.submit({"user_id": uid, "trust": state["trust"]}, salience=0.7, context_tag="relationship_alert")

    async def _maintain_bg(self, _):
        if hasattr(self.rm, "update_all_relationships"):
            upd = await maybe_async(self.rm.update_all_relationships)
            if upd:
                return {"relationships_updated": len(upd), "significance": 0.3}
        return None

# ---------------------------------------------------------------------------
class TheoryOfMindAdapter(EnhancedWorkspaceModule):
    def __init__(self, theory_of_mind, ws=None):
        super().__init__(ws)
        self.name = "theory_of_mind"
        self.tom = theory_of_mind
        self.register_unconscious("tom_bg", self._update_bg, threshold=0.6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.tom or not self.ws.focus:
            return
        user_text = self.ws.focus[0].content
        mental = await maybe_async(self.tom.infer_mental_state, user_text)
        if mental and mental.get("confidence", 0) > 0.6:
            await self.submit(mental, salience=mental["confidence"], context_tag="user_mental_state")

    async def _update_bg(self, _):
        if hasattr(self.tom, "update_user_models"):
            upd = await maybe_async(self.tom.update_user_models)
            if upd:
                return {"user_models_updated": len(upd), "significance": 0.4}
        return None

# ---------------------------------------------------------------------------
class MultimodalAdapter(EnhancedWorkspaceModule):
    def __init__(self, multimodal_integrator, ws=None):
        super().__init__(ws)
        self.name = "multimodal"
        self.mi = multimodal_integrator
        self.register_unconscious("sensory_bind", self._bind_bg, threshold=0.5)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.mi:
            return
        percepts = await maybe_async(self.mi.get_recent_percepts, limit=5)
        for p in percepts or []:
            if getattr(p, "attention_weight", 0) > 0.7:
                await self.submit({"percept": p.content, "modality": str(p.modality)}, salience=p.attention_weight, context_tag="sensory_salient")

    async def _bind_bg(self, _):
        if hasattr(self.mi, "bind_recent_percepts"):
            b = await maybe_async(self.mi.bind_recent_percepts)
            if b:
                return {"bound_percepts": len(b), "significance": 0.4}
        return None

# ---------------------------------------------------------------------------
class AutobiographicalAdapter(EnhancedWorkspaceModule):
    def __init__(self, autobiographical_narrative, ws=None):
        super().__init__(ws)
        self.name = "autobiography"
        self.an = autobiographical_narrative
        self.register_unconscious("narrative_bg", self._update_bg, threshold=0.8)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.an:
            return
        summary = await maybe_async(self.an.get_narrative_summary)
        significant_event = any(p.salience > 0.8 for p in self.ws.focus)
        if significant_event and summary:
            await self.submit({"narrative_context": summary}, salience=0.6, context_tag="narrative_update")

    async def _update_bg(self, _):
        if hasattr(self.an, "update_narrative"):
            seg = await maybe_async(self.an.update_narrative)
            if seg:
                return {"new_segment": True, "significance": 0.7}
        return None

# ---------------------------------------------------------------------------
class CreativeAdapter(EnhancedWorkspaceModule):
    def __init__(self, creative_system, ws=None):
        super().__init__(ws)
        self.name = "creative"
        self.cs = creative_system
        self.register_unconscious("idea_incubation", self._incubate_bg, threshold=0.6)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.cs:
            return
        triggers = [p for p in self.ws.focus if p.context_tag in ("user_input", "goal_active") and "create" in str(p.content).lower()]
        if not triggers:
            return
        creative = await maybe_async(self.cs.generate_creative_response, triggers[0].content)
        if creative:
            await self.submit(creative, salience=0.7, context_tag="creative_output")

    async def _incubate_bg(self, _):
        if random.random() < 0.1 and hasattr(self.cs, "incubate_idea"):
            idea = await maybe_async(self.cs.incubate_idea)
            if idea:
                return {"new_idea": idea, "significance": 0.5}
        return None

# ---------------------------------------------------------------------------
class PredictionAdapter(EnhancedWorkspaceModule):
    def __init__(self, prediction_engine, ws=None):
        super().__init__(ws)
        self.name = "prediction"
        self.pe = prediction_engine
        self.register_unconscious("prediction_update", self._update_bg, threshold=0.5)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.pe:
            return
        pred = await maybe_async(self.pe.predict_next_state)
        if pred and pred.get("confidence", 0) > 0.6:
            await self.submit(pred, salience=pred["confidence"], context_tag="prediction", prediction=pred.get("predicted_state"))

    async def _update_bg(self, _):
        if hasattr(self.pe, "update_prediction_models"):
            upd = await maybe_async(self.pe.update_prediction_models)
            if upd:
                return {"models_updated": True, "significance": 0.3}
        return None

# ---------------------------------------------------------------------------
class ProactiveAdapter(EnhancedWorkspaceModule):
    def __init__(self, proactive_engine, ws=None):
        super().__init__(ws)
        self.name = "proactive"
        self.pe = proactive_engine
        self.register_unconscious("intent_bg", self._intent_bg, threshold=0.6)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.pe:
            return
        idle = not any(p.context_tag == "user_input" for p in self.ws.proposals[-20:])
        if idle:
            action = await maybe_async(self.pe.generate_proactive_action)
            if action:
                await self.submit(action, salience=0.6, context_tag="proactive_action")

    async def _intent_bg(self, _):
        if hasattr(self.pe, "update_communication_intents"):
            intents = await maybe_async(self.pe.update_communication_intents)
            if intents and len(intents) > 2:
                return {"active_intents": len(intents), "significance": 0.5}
        return None

# ---------------------------------------------------------------------------
class PassiveObservationAdapter(EnhancedWorkspaceModule):
    def __init__(self, passive_obs, ws=None):
        super().__init__(ws)
        self.name = "observation"
        self.po = passive_obs
        self.register_unconscious("obs_filter", self._filter_bg, threshold=0.4)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.po:
            return
        obs = await maybe_async(self.po.get_relevant_observations, min_relevance=0.7, limit=3)
        for o in obs or []:
            data = o.dict() if hasattr(o, "dict") else o
            await self.submit(data, salience=data.get("relevance", 0.5), context_tag="observation")

    async def _filter_bg(self, _):
        if hasattr(self.po, "process_observations"):
            proc = await maybe_async(self.po.process_observations)
            if proc and proc.get("anomaly_detected"):
                return {"anomaly": True, "significance": 0.6}
        return None

# ---------------------------------------------------------------------------
class IdentityAdapter(EnhancedWorkspaceModule):
    def __init__(self, identity_evolution, ws=None):
        super().__init__(ws)
        self.name = "identity"
        self.ie = identity_evolution
        self.register_unconscious("trait_bg", self._evolve_bg, threshold=0.7)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.ie:
            return
        relevant = any(p.context_tag in ("reflection_insight", "goal_active", "emotion_spike") for p in self.ws.focus)
        if relevant:
            state = await maybe_async(self.ie.get_identity_state)
            if state and state.get("recent_changes"):
                await self.submit({"identity_shift": state["recent_changes"]}, salience=0.6, context_tag="identity_update")

    async def _evolve_bg(self, _):
        if hasattr(self.ie, "evolve_traits"):
            evo = await maybe_async(self.ie.evolve_traits)
            if evo:
                return {"traits_evolved": True, "significance": 0.5}
        return None

# ---------------------------------------------------------------------------
class MetaCoreAdapter(EnhancedWorkspaceModule):
    def __init__(self, meta_core, ws=None):
        super().__init__(ws)
        self.name = "metacognition"
        self.mc = meta_core
        self.register_unconscious("perf_bg", self._perf_bg, threshold=0.6)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.mc:
            return
        res = await maybe_async(self.mc.cognitive_cycle)
        if res and res.get("bottlenecks"):
            await self.submit({"bottlenecks": res["bottlenecks"]}, salience=0.7, context_tag="system_bottleneck")

    async def _perf_bg(self, _):
        if hasattr(self.mc, "get_system_metrics"):
            m = await maybe_async(self.mc.get_system_metrics)
            if m and m.get("cpu_usage", 0) > 0.8:
                return {"high_load": True, "cpu": m["cpu_usage"], "significance": 0.7}
        return None

# ---------------------------------------------------------------------------
class ModeIntegrationAdapter(EnhancedWorkspaceModule):
    def __init__(self, mode_integration, ws=None):
        super().__init__(ws)
        self.name = "mode"
        self.mi = mode_integration
        self.register_unconscious("mode_bg", self._coherence_bg, threshold=0.5)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.mi:
            return
        mode = await maybe_async(self.mi.get_current_mode)
        emotional_context = any(p.context_tag == "emotion_spike" for p in self.ws.focus)
        if emotional_context and mode != "EMOTIONAL":
            await self.submit({"suggested_mode": "EMOTIONAL", "current_mode": mode}, salience=0.6, context_tag="mode_change_suggestion")

    async def _coherence_bg(self, _):
        if hasattr(self.mi, "evaluate_mode_coherence"):
            coh = await maybe_async(self.mi.evaluate_mode_coherence)
            if coh is not None and coh < 0.4:
                return {"low_mode_coherence": True, "significance": 0.5}
        return None

# ---------------------------------------------------------------------------
class HormoneAdapter(EnhancedWorkspaceModule):
    def __init__(self, hormone_system, ws=None):
        super().__init__(ws)
        self.name = "hormones"
        self.hs = hormone_system
        self.register_unconscious("hormone_reg", self._reg_bg, threshold=0.4)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.hs:
            return
        levels = await maybe_async(self.hs.get_levels)
        for h, lvl in (levels or {}).items():
            if lvl > 0.8 or lvl < 0.2:
                await self.submit({"hormone": h, "level": lvl}, salience=abs(lvl - 0.5) * 2, context_tag="hormone_extreme")

    async def _reg_bg(self, _):
        if hasattr(self.hs, "update_levels"):
            await maybe_async(self.hs.update_levels)
        levels = await maybe_async(self.hs.get_levels)
        if levels and sum(1 for l in levels.values() if l > 0.7 or l < 0.3) >= 3:
            return {"hormonal_imbalance": True, "significance": 0.5}
        return None


# ────────────────────────────────────────────────────────────────────────────
# 1) CreativeSystemAdapter  – semantic trigger + adaptive code scan
# ────────────────────────────────────────────────────────────────────────────
class CreativeSystemAdapter(EnhancedWorkspaceModule):
    def __init__(self, creative_system, ws=None) -> None:
        super().__init__(ws)
        self.name = "creative"
        self.cs = creative_system
        self._next_scan: float = 0.0          # unix‑ts
        self._next_idea: float = 0.0

        self.register_unconscious("code_scan", self._scan_codebase_bg, threshold=.55)
        self.register_unconscious("idea_incubation", self._generate_ideas_bg, threshold=.6)

    # ---------- conscious -------------
    async def on_phase(self, phase: int) -> None:
        if phase != 1 or not self.cs:
            return
        for p in self.ws.focus:
            if p.context_tag == "user_input" and _sem_match(
                str(p.content),
                ["write", "create", "draw", "compose", "invent", "generate a"],
                .5,
            ):
                await self.submit(
                    {
                        "capabilities": self.cs.available_modalities(),
                        "trigger": str(p.content)[:120],
                    },
                    salience=.75,
                    context_tag="creative_ready",
                )
                break

    # ---------- unconscious -------------
    async def _scan_codebase_bg(self, view):
        now = time.time()
        if now < self._next_scan:
            return None
        self._next_scan = now + 300                       # 5‑min cool‑down
        if not hasattr(self.cs, "incremental_codebase_analysis"):
            return None
        result: dict = await maybe_async(self.cs.incremental_codebase_analysis)
        changed = result.get("changed_files_git", 0)
        if changed:
            return {
                "changed_files": changed,
                "files_analyzed": result.get("files_analyzed_for_metrics", 0),
                "significance": min(.9, .15 * math.log1p(changed)),
            }
        return None

    async def _generate_ideas_bg(self, view):
        now = time.time()
        if now < self._next_idea:
            return None
        self._next_idea = now + random.uniform(90, 240)    # 1.5‑4 min
        if not hasattr(self.cs, "incubate_idea"):
            return None
        # emotional priming boosts chance
        mood_boost = max(
            (p.salience for p in view.recent if p.context_tag == "emotion_spike"),
            default=0.0,
        )
        if random.random() < .15 + .5 * mood_boost:
            idea = await maybe_async(self.cs.incubate_idea)
            if idea:
                return {"new_idea": idea, "significance": .6 + .3 * mood_boost}
        return None

# ────────────────────────────────────────────────────────────────────────────
# 2) ThinkingToolsAdapter – smarter gating & metacog signal
# ────────────────────────────────────────────────────────────────────────────
class ThinkingToolsAdapter(EnhancedWorkspaceModule):
    def __init__(self, tools, ws=None):
        super().__init__(ws)
        self.name = "thinking"
        self.tools = tools
        self._active_level: int = 0
        self._last_metacog: float = 0.0

        self.register_unconscious("metacognition", self._metacog_bg, threshold=.65)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.tools:
            return
        for p in self.ws.focus:
            if p.context_tag != "user_input":
                continue
            decision = await maybe_async(self.tools.should_use_extended_thinking, str(p.content))
            if decision.get("should_think"):
                self._active_level = decision.get("thinking_level", 1)
                await self.submit(
                    {"thinking_required": True, "level": self._active_level},
                    salience=.8,
                    context_tag="thinking_needed",
                )

    async def _metacog_bg(self, view):
        now = time.time()
        if now - self._last_metacog < 30:                   # every 30 s tops
            return None
        self._last_metacog = now
        load = sum(1 for p in view.recent if p.source.startswith("thinking"))
        if load > 5:
            return {
                "cog_load": load,
                "level": self._active_level,
                "significance": min(.9, .1 * load),
            }
        return None

# ────────────────────────────────────────────────────────────────────────────
# 3) SocialBrowsingAdapter – motivation‑weighted & trend semantic
# ────────────────────────────────────────────────────────────────────────────
class SocialBrowsingAdapter(EnhancedWorkspaceModule):
    def __init__(self, social_tools, motivations: Dict[str, float], ws=None):
        super().__init__(ws)
        self.name = "social"
        self.tools = social_tools
        self.motiv = motivations
        self._next_browse = 0.0
        self.register_unconscious("feed_watch", self._watch_feed_bg, threshold=.5)

    async def on_phase(self, phase: int):
        if phase != 2 or not self.tools:
            return
        urge = max(self.motiv.get("curiosity", 0), self.motiv.get("expression", 0))
        if urge < .65:
            return
        if time.time() < self._next_browse:
            return
        self._next_browse = time.time() + 600              # 10‑min cool‑down
        await self.submit(
            {"browse": True, "motivation": "curiosity" if self.motiv["curiosity"] >= self.motiv["expression"] else "expression"},
            salience=.6 + .3 * urge,
            context_tag="social_browsing",
        )

    async def _watch_feed_bg(self, view):
        if not hasattr(self.tools, "sentiment_engine"):
            return None
        trends = await maybe_async(self.tools.sentiment_engine.detect_trends)
        if trends.get("trend_detected"):
            return {
                "trend": trends["trends"][:3],
                "significance": .55 + .35 * trends.get("trend_strength", .5),
            }
        return None

# ────────────────────────────────────────────────────────────────────────────
# 4) GameVisionAdapter – basic embedding & adaptive cadence
# ────────────────────────────────────────────────────────────────────────────
class GameVisionAdapter(EnhancedWorkspaceModule):
    def __init__(self, vision, knowledge, ws=None):
        super().__init__(ws)
        self.name = "game_vision"
        self.vision, self.knowledge = vision, knowledge
        self.cur_game: Optional[str] = None
        self._next_pattern = 0.0
        self._next_consolidate = 0.0

        self.register_unconscious("pattern_learning", self._pattern_bg, threshold=.55)
        self.register_unconscious("knowledge_consolidate", self._consolidate_bg, threshold=.7)

    async def on_phase(self, phase: int):
        if phase != 0 or not self.vision:
            return
        frame = next((p.content for p in self.ws.focus if p.context_tag == "game_frame"), None)
        if not frame:
            return
        analysis = await maybe_async(self.vision.analyze_frame, frame)
        if not analysis:
            return
        ginfo = analysis.get("game", {})
        gid = ginfo.get("game_id")
        if gid and gid != self.cur_game and self.knowledge:
            self.knowledge.set_current_game(gid)
            self.cur_game = gid
        await self.submit(
            {
                "game_id": gid,
                "stage": analysis.get("location"),
                "objects": len(analysis.get("objects", [])),
            },
            salience=.7,
            context_tag="game_state",
        )

    # ­­­background
    async def _pattern_bg(self, _):
        if time.time() < self._next_pattern or not self.knowledge or not self.cur_game:
            return None
        self._next_pattern = time.time() + 120            # every 2 min
        pat = self.knowledge.discover_patterns()
        if pat:
            return {"new_patterns": len(pat), "significance": min(.9, .1 * len(pat))}
        return None

    async def _consolidate_bg(self, _):
        if time.time() < self._next_consolidate or not self.knowledge:
            return None
        self._next_consolidate = time.time() + 900        # 15 min
        res = self.knowledge.consolidate_knowledge()
        if res.get("combined_insights", 0) or res.get("removed_insights", 0):
            return {"consolidated": True, "significance": .6}
        return None

# ────────────────────────────────────────────────────────────────────────────
# 5) CapabilityAdapter – vector‑based similarity to user request
# ────────────────────────────────────────────────────────────────────────────
class CapabilityAdapter(EnhancedWorkspaceModule):
    def __init__(self, cap_sys, ws=None):
        super().__init__(ws)
        self.name = "capabilities"
        self.cap = cap_sys
        self.register_unconscious("gap_scan", self._gap_bg, threshold=.65)

    async def on_phase(self, phase: int):
        if phase != 1 or not self.cap:
            return
        for p in self.ws.focus:
            if p.context_tag != "user_input":
                continue
            if _sem_match(str(p.content), ["can you", "able to", "is it possible"], .6):
                assess = await maybe_async(self.cap.assess_required_capabilities, str(p.content))
                if assess:
                    await self.submit(
                        {
                            "feasible": assess["overall_feasibility"],
                            "missing": len(assess.get("potential_gaps", [])),
                        },
                        salience=.6,
                        context_tag="capability_assessment",
                    )

    async def _gap_bg(self, _):
        analyse = await maybe_async(self.cap.identify_capability_gaps)
        total = sum(map(len, analyse.values()))
        if total:
            return {
                "gap_total": total,
                "significance": min(.9, .1 * total),
            }
        return None

# ────────────────────────────────────────────────────────────────────────────
# 6) UIConversationAdapter – activity‑aware ping
# ────────────────────────────────────────────────────────────────────────────
class UIConversationAdapter(EnhancedWorkspaceModule):
    def __init__(self, ui_mgr, ws=None):
        super().__init__(ws)
        self.name = "ui_conv"
        self.ui = ui_mgr
        self._active: set[str] = set()
        self._next_scan = 0.0

    async def on_phase(self, phase: int):
        if phase != 2 or not self.ui:
            return
        # trigger from social connection proposals
        for p in self.ws.focus:
            if p.context_tag == "social_connection" and p.salience > .7:
                uid = p.content.get("user_id")
                if uid and uid not in self._active:
                    conv = await maybe_async(self.ui.create_new_conversation, uid)
                    if not conv.get("error"):
                        self._active.add(uid)
                        await self.submit(
                            {"conv_started": uid, "id": conv["id"]},
                            salience=.55,
                            context_tag="ui_conversation",
                        )

    async def _monitor_conv_bg(self, _):
        if not self._active or time.time() < self._next_scan:
            return None
        self._next_scan = time.time() + 120              # every 2 min
        total = 0
        for uid in list(self._active):
            convs = await maybe_async(self.ui.get_conversations_for_user, uid)
            if not convs:
                self._active.discard(uid)
                continue
            total += sum(len(c.get("messages", [])) for c in convs)
        if total > 15:
            return {"ui_msg_volume": total, "significance": .55}
        return None

# ---------------------------------------------------------------------------
# FACTORY -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_gw_modules(brain) -> List[EnhancedWorkspaceModule]:
    """Create adapters for every subsystem actually present in *brain*."""
    adapters: List[EnhancedWorkspaceModule] = []
    add = adapters.append
    has = lambda attr: getattr(brain, attr, None) is not None   # noqa: E731

    # ── core cognitive ------------------------------------------------------
    if has("memory_core"):              add(MemoryAdapter(brain.memory_core))
    if has("emotional_core"):           add(EmotionalAdapter(brain.emotional_core))
    if has("needs_system"):             add(NeedsAdapter(brain.needs_system))
    if has("goal_manager"):             add(GoalAdapter(brain.goal_manager))
    if has("mood_manager"):             add(MoodAdapter(brain.mood_manager))
    if has("reasoning_core"):           add(ReasoningAdapter(brain.reasoning_core))
    if has("reflection_engine"):        add(ReflectionAdapter(brain.reflection_engine))
    if has("attentional_controller"):   add(AttentionAdapter(brain.attentional_controller))
    if has("prediction_engine"):        add(PredictionAdapter(brain.prediction_engine))
    if has("meta_core"):                add(MetaCoreAdapter(brain.meta_core))
    if has("mode_integration"):         add(ModeIntegrationAdapter(brain.mode_integration))
    if has("hormone_system"):           add(HormoneAdapter(brain.hormone_system))

    # ── embodiment / perception --------------------------------------------
    if has("body_image"):               add(BodyImageAdapter(brain.body_image))
    if has("multimodal_integrator"):    add(MultimodalAdapter(brain.multimodal_integrator))
    if has("passive_observation_system"): add(PassiveObservationAdapter(brain.passive_observation_system))

    # ── social / relational -------------------------------------------------
    if has("relationship_manager"):     add(RelationshipAdapter(brain.relationship_manager))
    if has("theory_of_mind"):           add(TheoryOfMindAdapter(brain.theory_of_mind))
    if has("autobiographical_narrative"): add(AutobiographicalAdapter(brain.autobiographical_narrative))
    if has("identity_evolution"):       add(IdentityAdapter(brain.identity_evolution))

    # ── creative & higher‑order thinking ------------------------------------
    if has("creative_system"):          add(CreativeSystemAdapter(brain.creative_system))
    if has("thinking_tools"):           add(ThinkingToolsAdapter(brain.thinking_tools))
    if has("capability_assessor"):      add(CapabilityAdapter(brain.capability_assessor))

    # ── proactive / interactive --------------------------------------------
    if has("proactive_communication_engine"):
        add(ProactiveAdapter(brain.proactive_communication_engine))

    # ── social browsing (needs tools *and* motivation dict) -----------------
    if has("social_tools") and has("motivations"):
        add(SocialBrowsingAdapter(brain.social_tools, brain.motivations))

    # ── game‑vision integration --------------------------------------------
    if has("game_vision") or has("cross_game_knowledge"):
        add(
            GameVisionAdapter(
                getattr(brain, "game_vision", None),
                getattr(brain, "cross_game_knowledge", None),
            )
        )

    # ── UI conversation manager --------------------------------------------
    if has("ui_manager"):
        add(UIConversationAdapter(brain.ui_manager))

    # ---- done --------------------------------------------------------------
    return adapters
