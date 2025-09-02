# story_agent/story_orchestrator.py

"""
Story Orchestrator: Unifies world state, narration, activities, tasks, patterns, and summaries
into a single "story packet" ready for the final response generator (lore/npcs/etc.) to consume.

Changelog (highlights):
- Added robust world bundle fallback (no hard dependency on context.get_world_bundle)
- Corrected DailyTaskGenerator invocation pattern
- Integrated conflict synthesizer awareness (+ narrative adjustments)
- Canonical logging of significant packets
- Added module registry (_get_module) for standardized lazy imports
- Added short-lived caching for packets
- Added better memory integration (relevant_memories)
- Added health_check() to verify subsystem status
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =========================
# Utility helpers
# =========================

def _now_iso() -> str:
    return datetime.now().isoformat()

def _safe_dump(obj: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable Dict (or passthrough).
    Handles pydantic v2 models, dataclasses, dicts, lists, known result wrappers.
    """
    try:
        data = getattr(obj, "data", None)
        if data is not None:
            return _safe_dump(data)
    except Exception:
        pass

    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass

    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        try:
            return {k: _safe_dump(v) for k, v in obj.items()}
        except Exception:
            return obj
    if isinstance(obj, list):
        try:
            return [_safe_dump(v) for v in obj]
        except Exception:
            return obj

    return str(obj)

def _ws_brief(ws: Any) -> Dict[str, Any]:
    brief: Dict[str, Any] = {"has_world_state": ws is not None}
    try:
        def get(path, default=None):
            cur = ws
            for p in path:
                if isinstance(cur, dict):
                    cur = cur.get(p, None)
                else:
                    cur = getattr(cur, p, None)
                if cur is None:
                    return default
            return cur

        tod = get(["current_time", "time_of_day"])
        brief["time_of_day"] = getattr(tod, "value", tod)

        mood = get(["world_mood"])
        brief["world_mood"] = getattr(mood, "value", mood)

        wt = get(["world_tension"])
        if isinstance(wt, dict):
            brief["tension"] = {
                "overall": wt.get("overall_tension", 0.0),
                "power": wt.get("power_tension", 0.0),
                "social": wt.get("social_tension", 0.0),
                "sexual": wt.get("sexual_tension", 0.0),
                "emotional": wt.get("emotional_tension", 0.0),
            }
        else:
            brief["tension"] = {
                "overall": getattr(wt, "overall_tension", 0.0),
                "power": getattr(wt, "power_tension", 0.0),
                "social": getattr(wt, "social_tension", 0.0),
                "sexual": getattr(wt, "sexual_tension", 0.0),
                "emotional": getattr(wt, "emotional_tension", 0.0),
            }

        active = get(["active_npcs"], []) or []
        brief["active_npc_count"] = len(active) if isinstance(active, list) else 0

        loc = get(["location_data"], "")
        brief["location"] = loc.get("current_location") if isinstance(loc, dict) else loc

    except Exception as e:
        brief["summary_error"] = str(e)
    return brief


# =========================
# Packet Model
# =========================

@dataclass
class StoryPacket:
    user_id: int
    conversation_id: int
    timestamp: str

    # Primary narrative
    primary_narrative: Optional[str] = None
    primary_narration_obj: Optional[Dict[str, Any]] = None
    action_result: Optional[Dict[str, Any]] = None

    # Scene elements
    scene: Optional[Dict[str, Any]] = None
    dialogues: List[Dict[str, Any]] = field(default_factory=list)
    ambient: Optional[Dict[str, Any]] = None

    # World/components
    world_state_brief: Dict[str, Any] = field(default_factory=dict)
    world_patterns: Optional[Dict[str, Any]] = None
    emergent_threads: Optional[Dict[str, Any]] = None
    active_conflicts: Optional[List[Dict[str, Any]]] = None

    # Recommendations/tasks
    activity_recommendations: Optional[Dict[str, Any]] = None
    daily_task: Optional[Dict[str, Any]] = None
    creative_task: Optional[Dict[str, Any]] = None

    # Memory/context
    narrative_context: Optional[Dict[str, Any]] = None
    relevant_memories: Optional[List[Dict[str, Any]]] = None

    # Canonical IDs (optional)
    canonical_ids: Dict[str, Any] = field(default_factory=dict)

    # Diagnostics
    performance: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _safe_dump(self)


# =========================
# Options
# =========================

@dataclass
class OrchestratorOptions:
    include_activity_recs: bool = True
    include_daily_task: bool = True
    include_creative_task: bool = False
    include_emergent_patterns: bool = True
    include_ambient: bool = True
    include_dialogues: bool = True
    include_world_patterns_from_bundle: bool = True
    max_dialogue_npcs: int = 2
    scene_type: str = "routine"
    prefer_fast_world_bundle: bool = False
    timeout_seconds: int = 30


# =========================
# Orchestrator
# =========================

class StoryOrchestrator:
    def __init__(self, user_id: int, conversation_id: int, options: Optional[OrchestratorOptions] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.options = options or OrchestratorOptions()

        self._director = None
        self._narrator = None
        self._summarizer = None
        self._perf_monitor = None

        # Module registry map
        self._module_registry: Dict[str, Any] = {}

        # Cache for short-lived packets
        self._packet_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl: int = 30  # seconds

        # Current world
        self._world_state = None
        self._world_bundle = None

    # --------- Module registry ---------
    async def _get_module(self, module_name: str):
        if module_name in self._module_registry:
            return self._module_registry[module_name]

        if module_name == "activity_recommender":
            import story_agent.activity_recommender as mod
        elif module_name == "daily_task_generator":
            from story_agent import daily_task_generator as mod
        elif module_name == "creative_task_agent":
            from story_agent import creative_task_agent as mod
        elif module_name == "slice_of_life_narrator":
            from story_agent import slice_of_life_narrator as mod
        else:
            mod = None

        if mod is not None:
            self._module_registry[module_name] = mod
        return mod

    # --------- Lazy loaders ---------
    async def _ensure_director(self):
        if self._director is None:
            from story_agent.world_director_agent import CompleteWorldDirector
            self._director = CompleteWorldDirector(self.user_id, self.conversation_id)
            await self._director.initialize()
            self._perf_monitor = getattr(self._director.context, "performance_monitor", None)

    async def _ensure_narrator(self):
        if self._narrator is None:
            from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
            self._narrator = SliceOfLifeNarrator(self.user_id, self.conversation_id)
            await self._narrator.initialize()

    async def _ensure_summarizer(self):
        if self._summarizer is None:
            from story_agent.progressive_summarization import RPGNarrativeManager
            self._summarizer = RPGNarrativeManager(self.user_id, self.conversation_id)
            await self._summarizer.initialize()

    # --------- World helpers ---------
    async def _get_world_bundle(self) -> Dict[str, Any]:
        await self._ensure_director()
        try:
            ctx = self._director.context
            if hasattr(ctx, "get_world_bundle"):
                self._world_bundle = await ctx.get_world_bundle(
                    fast=self.options.prefer_fast_world_bundle
                )
                self._world_state = self._world_bundle.get("world_state")
            else:
                # Fallback: basic world state
                self._world_state = await self._director.get_world_state()
                self._world_bundle = {
                    "world_state": self._world_state,
                    "patterns": None,
                    "summary": _ws_brief(self._world_state)
                }
        except Exception as e:
            logger.warning(f"get_world_bundle failed: {e}")
            self._world_bundle = {"world_state": None}
            self._world_state = None
        return self._world_bundle

    # =========================
    # Public API
    # =========================

    async def assemble_story_packet(
        self,
        user_input: Optional[str] = None,
        mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Assemble and return a JSON-serializable dict (cache-aware).
        """
        # Cache key: time-bucketed by TTL
        bucket = int(time.time() // self._cache_ttl)
        cache_key = f"{self.user_id}:{self.conversation_id}:{mode}:{(user_input or 'auto')}:{bucket}"

        # Fast path: return cached dict
        cached = self._packet_cache.get(cache_key)
        if cached:
            return cached

        # Build packet
        packet = StoryPacket(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            timestamp=_now_iso()
        )

        try:
            await asyncio.wait_for(
                self._assemble(packet, user_input, mode),
                timeout=self.options.timeout_seconds
            )
        except asyncio.TimeoutError:
            packet.errors.append("Orchestration timed out")
        except Exception as e:
            logger.error(f"assemble_story_packet error: {e}", exc_info=True)
            packet.errors.append(str(e))

        # Store in cache
        packet_dict = packet.to_dict()
        self._packet_cache[cache_key] = packet_dict

        # Cleanup old cache entries
        try:
            cutoff_bucket = int(time.time() // self._cache_ttl) - 2
            self._packet_cache = {
                k: v for k, v in self._packet_cache.items()
                if int(k.split(":")[-1]) >= cutoff_bucket
            }
        except Exception:
            pass

        return packet_dict

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of subsystems.
        """
        health = {"director": False, "narrator": False, "summarizer": False, "activity": False, "errors": []}
        try:
            await self._ensure_director()
            health["director"] = self._director is not None
        except Exception as e:
            health["errors"].append(f"Director: {e}")

        try:
            await self._ensure_narrator()
            health["narrator"] = self._narrator is not None
        except Exception as e:
            health["errors"].append(f"Narrator: {e}")

        try:
            await self._ensure_summarizer()
            health["summarizer"] = self._summarizer is not None
        except Exception as e:
            health["errors"].append(f"Summarizer: {e}")

        try:
            mod = await self._get_module("activity_recommender")
            health["activity"] = mod is not None
        except Exception as e:
            health["errors"].append(f"Activity: {e}")

        health["healthy"] = all([health["director"], health["narrator"], health["summarizer"]])
        return health

    # =========================
    # Internals
    # =========================

    async def _assemble(self, packet: StoryPacket, user_input: Optional[str], mode: str):
        # Ensure systems
        await asyncio.gather(
            self._ensure_director(),
            self._ensure_narrator(),
            self._ensure_summarizer()
        )

        # World
        bundle = await self._get_world_bundle()
        ws = self._world_state
        packet.world_state_brief = _ws_brief(ws)
        if self.options.include_world_patterns_from_bundle and isinstance(bundle, dict):
            packet.world_patterns = _safe_dump(bundle.get("patterns"))

        # Conflict awareness
        await self._attach_conflict_awareness(packet)

        # Branch
        if user_input:
            await self._process_player_input_path(packet, user_input)
        else:
            await self._autonomous_path(packet)

        # Activities & tasks
        if self.options.include_activity_recs:
            await self._attach_activity_recommendations(packet)
        if self.options.include_daily_task:
            await self._attach_daily_task(packet)
        if self.options.include_creative_task:
            await self._attach_creative_task(packet)

        # Emergent patterns (full scan)
        if self.options.include_emergent_patterns:
            await self._attach_emergent_patterns(packet)

        # Narrative context + relevant memories
        await self._attach_narrative_context(packet, user_input or packet.primary_narrative or "")

        # Performance
        await self._attach_metrics(packet)

        # Canonical logging for significant packets
        await self._canonical_log_packet(packet, mode)

    # --------- Conflict awareness ---------
    async def _attach_conflict_awareness(self, packet: StoryPacket):
        try:
            ctx = self._director.context
            synth = getattr(ctx, "conflict_synthesizer", None)
            if synth:
                conflict_state = await synth.get_system_state()
                active = conflict_state.get("active_conflicts", [])
                if active:
                    packet.active_conflicts = _safe_dump(active)
                    await self._adjust_for_conflicts(packet, conflict_state)
        except Exception as e:
            logger.warning(f"Conflict integration failed: {e}")

    async def _adjust_for_conflicts(self, packet: StoryPacket, conflict_state: Dict[str, Any]):
        """Nudge narrative content based on conflicts being active."""
        try:
            severity = float(conflict_state.get("metrics", {}).get("complexity_score", 0.0))
            
            # More nuanced adjustments based on conflict type
            active_conflicts = conflict_state.get("active_conflicts", [])
            for conflict in active_conflicts:
                conflict_type = conflict.get("type", "unknown")
                
                # Type-specific narrative adjustments
                if conflict_type == "social" and packet.dialogues:
                    # Add tension to dialogues
                    for dialogue in packet.dialogues:
                        if isinstance(dialogue, dict):
                            dialogue["subtext"] = "Unspoken tensions color the conversation"
                
                elif conflict_type == "power" and packet.primary_narrative:
                    # Adjust narrative tone
                    if severity > 0.7:
                        packet.primary_narrative = packet.primary_narrative.replace(
                            "relaxed", "charged with subtle tension"
                        )
        except Exception as e:
            logger.debug(f"Conflict adjustment failed: {e}")

    # --------- Player Input Path ---------
    async def _process_player_input_path(self, packet: StoryPacket, user_input: str):
        try:
            result = await self._narrator.process_player_input(user_input)
            result_dict = _safe_dump(result)
            packet.action_result = result_dict
            packet.primary_narrative = result_dict.get("narrative") or result_dict.get("response") or ""
            if self.options.include_dialogues:
                npc_responses = result_dict.get("npc_responses") or result_dict.get("npc_dialogues")
                if npc_responses:
                    packet.dialogues = [_safe_dump(d) for d in npc_responses if d]
        except Exception as e:
            logger.error(f"process_player_input failed: {e}", exc_info=True)
            packet.errors.append(f"process_input: {e}")

        if self.options.include_ambient:
            await self._attach_ambient(packet)

    # --------- Autonomous Path ---------
    async def _autonomous_path(self, packet: StoryPacket):
        try:
            narration_text = await self._narrator.narrate_world_state()
            packet.primary_narrative = narration_text
            # Try to attach a structured narration object (best-effort)
            try:
                sln = await self._get_module("slice_of_life_narrator")
                # Build a Run call using the narrator's agent (scene_narrator) and tool
                result = await self._narrator.scene_narrator.run(
                    messages=[{"role": "user", "content": f"Narrate a {self.options.scene_type} scene"}],
                    context=self._narrator.context,
                    tool_calls=[{
                        "tool": sln.narrate_slice_of_life_scene,
                        "kwargs": {
                            "payload": sln.NarrateSliceOfLifeInput(
                                scene_type=self.options.scene_type,
                                world_state=self._world_state
                            ).model_dump()
                        }
                    }]
                )
                packet.primary_narration_obj = _safe_dump(getattr(result, "data", result))
            except Exception:
                pass
        except Exception as e:
            logger.error(f"narrate_world_state failed: {e}", exc_info=True)
            packet.errors.append(f"narrate_world: {e}")

        if self.options.include_dialogues:
            await self._attach_dialogues(packet, max_npcs=self.options.max_dialogue_npcs)
        if self.options.include_ambient:
            await self._attach_ambient(packet)

    # --------- Attachments ---------
    async def _attach_dialogues(self, packet: StoryPacket, max_npcs: int = 2):
        try:
            ws = self._world_state
            chosen: List[int] = []
            if ws is not None:
                active = getattr(ws, "active_npcs", None) or []
                for item in active:
                    nid = None
                    if isinstance(item, int):
                        nid = item
                    elif isinstance(item, dict):
                        nid = item.get("npc_id") or item.get("id")
                    else:
                        nid = getattr(item, "npc_id", None) or getattr(item, "id", None)
                    if isinstance(nid, int):
                        chosen.append(nid)
                    if len(chosen) >= max_npcs:
                        break

            for npc_id in chosen:
                try:
                    dialogue = await self._narrator.generate_npc_dialogue(
                        npc_id=npc_id,
                        situation="ambient interaction",
                        world_state=ws,
                        player_input=None
                    )
                    packet.dialogues.append(_safe_dump(dialogue))
                except Exception as de:
                    logger.warning(f"dialogue generation failed for npc {npc_id}: {de}")
        except Exception as e:
            logger.error(f"_attach_dialogues failed: {e}", exc_info=True)
            packet.errors.append(f"dialogues: {e}")

    async def _attach_ambient(self, packet: StoryPacket):
        try:
            sln = await self._get_module("slice_of_life_narrator")
            result = await self._narrator.scene_narrator.run(
                messages=[{"role": "user", "content": "Ambient generation"}],
                context=self._narrator.context,
                tool_calls=[{
                    "tool": sln.generate_ambient_narration,
                    "kwargs": {
                        "focus": "ambient",
                        "world_state": self._world_state.model_dump() if hasattr(self._world_state, "model_dump") else self._world_state,
                        "intensity": 0.5
                    }
                }]
            )
            packet.ambient = _safe_dump(getattr(result, "data", result))
        except Exception as e:
            logger.warning(f"ambient attachment failed: {e}")

    async def _attach_activity_recommendations(self, packet: StoryPacket):
        try:
            mod = await self._get_module("activity_recommender")
            recs = await mod.recommend_activities(
                self.user_id,
                self.conversation_id,
                context=None,
                num_recommendations=4
            )
            packet.activity_recommendations = _safe_dump(recs)
        except Exception as e:
            logger.warning(f"activity recommendations failed: {e}")

    async def _attach_daily_task(self, packet: StoryPacket):
        try:
            # Correct pattern: DailyTaskGenerator is already an Agent instance
            from story_agent.daily_task_generator import DailyTaskGenerator
            from agents import Runner
            result = await Runner.run(
                DailyTaskGenerator,
                "Generate a contextual daily task for the current world and relationships"
            )
            packet.daily_task = _safe_dump(getattr(result, "final_output", None) or getattr(result, "data", None) or result)
        except Exception as e:
            logger.warning(f"daily task generation failed: {e}")

    async def _attach_creative_task(self, packet: StoryPacket):
        try:
            from story_agent.creative_task_agent import femdom_task_agent
            from agents import Runner
            npc_id = None
            ws = self._world_state
            if ws is not None:
                active = getattr(ws, "active_npcs", None) or []
                for it in active:
                    if isinstance(it, dict) and it.get("dominance", 0) > 60:
                        npc_id = it.get("npc_id") or it.get("id")
                        if npc_id:
                            break
                if npc_id is None and active:
                    first = active[0]
                    npc_id = first.get("npc_id") if isinstance(first, dict) else getattr(first, "npc_id", None)

            msg = "Build a contextual creative femdom task."
            if npc_id:
                msg = f"Build a contextual creative femdom task centered on NPC {npc_id}."
            res = await Runner.run(femdom_task_agent, messages=[{"role": "user", "content": msg}])
            packet.creative_task = _safe_dump(getattr(res, "data", None) or getattr(res, "final_output", None) or res)
        except Exception as e:
            logger.info(f"creative task generation skipped/failed: {e}")

    async def _attach_emergent_patterns(self, packet: StoryPacket):
        try:
            from story_agent.world_director_agent import check_all_emergent_patterns
            from agents import RunContextWrapper
            result = await check_all_emergent_patterns(RunContextWrapper(self._director.context))
            packet.emergent_threads = _safe_dump(result)
        except Exception as e:
            logger.info(f"emergent patterns failed: {e}")

    async def _attach_narrative_context(self, packet: StoryPacket, input_text: str):
        try:
            context = await self._summarizer.get_current_narrative_context(
                input_text=input_text,
                max_tokens=2000
            )
            packet.narrative_context = _safe_dump(context)
        except Exception as e:
            logger.info(f"narrative context summarization failed: {e}")

        # Relevant memories via director.context.memory_manager (best-effort)
        try:
            mm = getattr(self._director.context, "memory_manager", None)
            if mm:
                query = input_text or packet.primary_narrative or ""
                mems = await mm.search_memories(
                    query_text=query[:256] or "current moment",
                    memory_types=["scene", "interaction", "power_exchange"],
                    limit=5,
                    use_vector=True
                )
                # Depending on manager, result may be MemorySearchResult with .memories
                items = getattr(mems, "memories", None) or mems
                packet.relevant_memories = [_safe_dump(m) for m in (items or [])]
        except Exception as e:
            logger.info(f"relevant memory search failed: {e}")

    async def _attach_metrics(self, packet: StoryPacket):
        try:
            perf = {}
            if self._perf_monitor:
                try:
                    perf["world_director"] = self._perf_monitor.get_metrics()
                except Exception:
                    pass
            try:
                perf["narrator"] = await self._narrator.get_performance_metrics()
            except Exception:
                pass
            packet.performance = _safe_dump(perf)
        except Exception as e:
            logger.debug(f"metrics collection failed: {e}")

    # --------- Canonical logging ---------
    def _calculate_packet_significance(self, packet: StoryPacket) -> int:
        """
        Heuristic significance score (1-10) for canonical logging.
        """
        score = 5
        brief = packet.world_state_brief or {}
        tension = (brief.get("tension") or {}).get("power", 0.0)
        if tension and tension > 0.6:
            score += 1
        if packet.active_conflicts:
            score += 2
        if packet.dialogues and len(packet.dialogues) > 1:
            score += 1
        if packet.emergent_threads:
            score += 1
        return max(1, min(score, 10))

    async def _canonical_log_packet(self, packet: StoryPacket, mode: str):
        if not packet.primary_narrative:
            return
        
        try:
            from lore.core.canon import log_canonical_event, ensure_canonical_context
            from db.connection import get_db_connection_context
            
            canonical_ctx = ensure_canonical_context({
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            })
            
            significance = self._calculate_packet_significance(packet)
            
            if significance >= 7:
                async with get_db_connection_context() as conn:
                    # Batch multiple canonical operations
                    async with conn.transaction():
                        await log_canonical_event(
                            canonical_ctx, conn,
                            f"Story packet: {packet.primary_narrative[:100]}",
                            tags=["story_packet", mode],
                            significance=significance
                        )
                        
                        # Log related events in same transaction
                        if packet.active_conflicts:
                            await log_canonical_event(
                                canonical_ctx, conn,
                                f"Conflicts active during packet: {len(packet.active_conflicts)}",
                                tags=["story_packet", "conflicts"],
                                significance=significance - 1
                            )
        except Exception as e:
            logger.warning(f"Canonical logging failed: {e}")


# =========================
# Convenience function (returns dict)
# =========================

async def assemble_story_packet(
    user_id: int,
    conversation_id: int,
    user_input: Optional[str] = None,
    options: Optional[OrchestratorOptions] = None,
    mode: str = "auto"
) -> Dict[str, Any]:
    """
    One-shot convenience function returning a JSON-serializable dict.
    Uses orchestrator-internal caching automatically.
    """
    orchestrator = StoryOrchestrator(user_id, conversation_id, options)
    # Now the orchestrator method already returns a dict and handles caching.
    return await orchestrator.assemble_story_packet(user_input=user_input, mode=mode)
