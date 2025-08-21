# nyx/nyx_agent/utils.py
"""Utility functions for Nyx Agent SDK"""

import json
import time
import os
import logging
import statistics
import asyncio
from enum import Enum
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union
from contextlib import suppress

logger = logging.getLogger(__name__)

def safe_psutil(func_name: str, *args, default=None, **kwargs):
    """Safe wrapper for psutil calls that may fail on certain platforms"""
    try:
        import psutil
        func = getattr(psutil, func_name)
        return func(*args, **kwargs)
    except (AttributeError, OSError, RuntimeError) as e:
        logger.debug(f"psutil.{func_name} failed (platform compatibility): {e}")
        return default

def safe_process_metric(process, metric_name: str, default=0):
    """Safe wrapper for process-specific metrics"""
    try:
        metric_func = getattr(process, metric_name)
        result = metric_func()
        if hasattr(result, 'rss'):  # memory_info returns a named tuple
            return result.rss
        return result
    except (AttributeError, OSError, RuntimeError) as e:
        logger.debug(f"Process metric {metric_name} failed: {e}")
        return default

def get_process_info() -> Optional[Any]:
    """Get current process info safely"""
    try:
        import psutil
        return psutil.Process(os.getpid())
    except Exception as e:
        logger.debug(f"Failed to get process info: {e}")
        return None

def bytes_to_mb(value: Optional[Union[int, float]]) -> float:
    """Convert bytes to megabytes safely"""
    return (value or 0) / (1024 * 1024)

def extract_token_usage(result: Any) -> int:
    """Extract token usage from various result formats"""
    try:
        if hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
            return result.usage.total_tokens
        elif hasattr(result, 'trace') and hasattr(result.trace, 'final_usage'):
            return result.trace.final_usage.get('total_tokens', 0)
        else:
            logger.debug("Token usage not found in result object")
            return 0
    except Exception as e:
        logger.debug(f"Failed to retrieve token usage: {e}")
        return 0

def get_context_text_lower(context: Dict[str, Any]) -> str:
    """Extract text from context and convert to lowercase for analysis"""
    text_parts = []
    for key, value in context.items():
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, (list, dict)):
            text_parts.append(str(value))
    return " ".join(text_parts).lower()

def _prune_list(lst: List[Any], max_len: int) -> None:
    """Prune a list to maximum length in-place"""
    if len(lst) > max_len:
        del lst[:-max_len]

def _calculate_avg_response_time(response_times: List[float]) -> float:
    """Calculate average response time safely"""
    if not response_times:
        return 0.0
    try:
        return statistics.fmean(response_times)
    except Exception:
        return sum(response_times) / len(response_times)

def _calculate_variance(values: List[float]) -> float:
    """Calculate variance of a list of values"""
    if len(values) < 2:
        return 0.0
    try:
        return statistics.variance(values)
    except Exception:
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

def _json_safe(value, *, _depth=0, _max_depth=4):
    """Best-effort conversion of arbitrary Python objects to JSON-safe primitives."""
    if _depth > _max_depth:
        return str(value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    try:
        from datetime import datetime, date
        if isinstance(value, (datetime, date)):
            return value.isoformat()
    except Exception:
        pass

    try:
        from enum import Enum
        if isinstance(value, Enum):
            return _json_safe(getattr(value, "value", str(value)), _depth=_depth+1, _max_depth=_max_depth)
    except Exception:
        pass

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _depth=_depth+1, _max_depth=_max_depth) for v in value]

    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth=_depth+1, _max_depth=_max_depth) for k, v in value.items()}

    try:
        import dataclasses
        if dataclasses.is_dataclass(value):
            return _json_safe(dataclasses.asdict(value), _depth=_depth+1, _max_depth=_max_depth)
    except Exception:
        pass

    for attr in ("model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _json_safe(fn(), _depth=_depth+1, _max_depth=_max_depth)
            except Exception:
                break

    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return _json_safe(data, _depth=_depth+1, _max_depth=_max_depth)
    return str(value)

def _preview(text: Optional[str], n: int = 240) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    return cleaned[:n] + ("…" if len(cleaned) > n else "")

def _js(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"

def _jsonable(x):
    """Convert any object to JSON-serializable format"""
    if x is None:
        return None
    if isinstance(x, Enum):
        return x.value
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    return x

def _default_json_encoder(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")

def extract_runner_response(result: Any) -> str:
    """Best-effort extraction of model output from Runner.run result."""
    try:
        for attr in ("final_output", "output_text", "text"):
            v = getattr(result, attr, None)
            if isinstance(v, str) and v.strip():
                return v
        data = getattr(result, "data", None)
        if data is not None:
            try:
                return json.dumps(_json_safe(data), ensure_ascii=False)
            except Exception:
                return str(data)
        msgs = getattr(result, "messages", None)
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    parts = m.get("content") or []
                    for part in reversed(parts):
                        if isinstance(part, dict) and part.get("type") == "output_text" and part.get("text"):
                            return part["text"]
        return str(result)
    except Exception:
        return str(result)

def _get_tool_schema_dict(tool):
    for attr in ("parameters", "_parameters", "schema", "_schema", "openai_schema"):
        val = getattr(tool, attr, None)
        if isinstance(val, dict):
            return attr, val
        try:
            from pydantic import BaseModel as _BM
            if isinstance(val, type) and issubclass(val, _BM):
                return attr, val.model_json_schema()
        except Exception:
            pass
    return None, None

def _set_tool_schema_dict(tool, attr_name, schema):
    try:
        if hasattr(tool, "parameters_model"):
            try: delattr(tool, "parameters_model")
            except Exception: setattr(tool, "parameters_model", None)
        setattr(tool, attr_name or "parameters", schema)
    except Exception:
        try: tool.parameters = schema
        except Exception: pass

def _find_paths_for_property(schema_dict, target_key):
    """Find where a key appears as a nested property for debugging."""
    paths = []
    def walk(node, path=""):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and target_key in props:
                paths.append(path + (".properties" if path else "properties"))
            for k, v in node.items():
                walk(v, path + ("." if path else "") + k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
    walk(schema_dict)
    return paths

def _tool_payload(tool):
    name = getattr(tool, "name", getattr(tool, "__name__", "unnamed_tool"))
    desc = (getattr(tool, "__doc__", "") or "")[:1000]

    params = None
    for attr in ("parameters", "_parameters", "openai_schema", "schema", "_schema"):
        v = getattr(tool, attr, None)
        if isinstance(v, dict):
            params = v
            break
        try:
            from pydantic import BaseModel as _BM
            if isinstance(v, type) and issubclass(v, _BM):
                params = v.model_json_schema()
                break
        except Exception:
            pass

    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params,
        },
    }

def _log_tool(tool, log):
    payload = _tool_payload(tool)
    fn = payload["function"]
    params = fn.get("parameters") or {}
    props = list((params.get("properties") or {}).keys())
    req = list((params.get("required") or []))

    extra = sorted(set(req) - set(props))
    missing = sorted(set(props) - set(req))

    log.debug("[tool:%s] root.properties=%s", fn["name"], props)
    log.debug("[tool:%s] root.required=%s", fn["name"], req)
    if extra:
        log.error("[tool:%s] ✗ EXTRA keys in required (not in properties): %s", fn["name"], extra)
    if missing:
        log.error("[tool:%s] ✗ MISSING keys in required (present in properties): %s", fn["name"], missing)

def assert_no_required_leaks(tool, log=None):
    payload = _tool_payload(tool)
    fn = payload["function"]
    params = fn.get("parameters") or {}
    props = list((params.get("properties") or {}).keys())
    req = list((params.get("required") or []))

    extra = sorted(set(req) - set(props))
    missing = sorted(set(props) - set(req))

    if extra or missing:
        msg = (f"Tool '{fn['name']}' has invalid root schema: "
               f"extra_in_required={extra} missing_in_required={missing}")
        if log:
            log.error(msg)
        raise ValueError(msg)

def _unwrap_tool_ctx(ctx_like):
    """Return the real NyxContext even if we have nested RunContextWrapper(...) layers."""
    c = getattr(ctx_like, "context", ctx_like)
    expected = ("user_id", "conversation_id", "world_director", "slice_of_life_narrator", "current_context")
    seen = set()
    while hasattr(c, "context") and not any(hasattr(c, k) for k in expected) and id(c) not in seen:
        seen.add(id(c))
        c = getattr(c, "context")
    return c

def _resolve_app_ctx(ctx_like):
    """Accept either RunContextWrapper[NyxContext] or NyxContext and return the NyxContext."""
    return getattr(ctx_like, "context", ctx_like)

def _get_app_ctx(ctx: Any) -> Any:
    """Robustly unwrap nested RunContextWrapper -> ... -> NyxContext."""
    c = getattr(ctx, "context", ctx)
    seen = set()
    while True:
        if id(c) in seen:
            break
        seen.add(id(c))

        if any(hasattr(c, k) for k in ("user_id", "conversation_id", "world_director", "current_world_state")):
            break

        nxt = getattr(c, "context", None)
        if nxt is None:
            break
        c = nxt

    return c

def _ensure_context_map(app_ctx):
    """Ensure app_ctx.current_context exists and is a dict.
    Returns (context_map, writable) where writable=False means we couldn't set it."""
    context_map = getattr(app_ctx, "current_context", None)
    if not isinstance(context_map, dict):
        try:
            setattr(app_ctx, "current_context", {})
            context_map = app_ctx.current_context
            return context_map, True
        except Exception:
            return {}, False
    return context_map, True

async def _ensure_world_state(app_ctx):
    """Return a world_state object if available; try to initialize WorldDirector if needed."""
    ws = getattr(app_ctx, "current_world_state", None)
    if ws is not None:
        return ws

    wd = getattr(app_ctx, "world_director", None)
    if wd is None:
        return None

    try:
        if hasattr(wd, "initialize") and asyncio.iscoroutinefunction(wd.initialize):
            if not getattr(wd, "_initialized", False):
                await wd.initialize()
    except Exception:
        pass

    ctx2 = getattr(wd, "context", None)
    return getattr(ctx2, "current_world_state", None) if ctx2 else None

async def _ensure_world_state_from_ctx(app_ctx: Any):
    """Best-effort world_state fetch that never throws."""
    ws = getattr(app_ctx, "current_world_state", None)
    if ws is not None:
        return ws

    wd = getattr(app_ctx, "world_director", None)
    if not wd:
        return None

    try:
        if hasattr(wd, "_initialized") and not getattr(wd, "_initialized", False):
            if hasattr(wd, "initialize") and asyncio.iscoroutinefunction(wd.initialize):
                await wd.initialize()
        ctx2 = getattr(wd, "context", None)
        return getattr(ctx2, "current_world_state", None) if ctx2 else None
    except Exception:
        return None

def _extract_last_assistant_text(resp) -> str:
    """Extract the last assistant message text from response"""
    msgs = [c for c in resp if c.get("type") == "message" and c.get("role") == "assistant"]
    if not msgs:
        return ""
    parts = msgs[-1].get("content") or []
    for part in reversed(parts):
        if part.get("type") == "output_text" and part.get("text"):
            return part["text"]
    return ""

def _did_call_tool(resp, tool_name: str) -> bool:
    """Check if a specific tool was called in the response"""
    for c in resp:
        if c.get("type") in ("function_call", "function_call_output") and c.get("name") == tool_name:
            return True
    return False

def _tool_output(resp, name: str):
    """Extract tool output data from response, return {} if not found"""
    outs = [c for c in resp if c.get("type") == "function_call_output" and c.get("name") == name]
    if not outs:
        return {}
    try:
        raw = outs[-1].get("output")
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        return {}

def _score_scene_text(scene_text: str) -> float:
    s = (scene_text or "").lower()
    score = 0.0
    for kw in ("dramatic","intense","beautiful","transformation","reveal","climax","pivotal"):
        if kw in s:
            score += 0.2
    if any(w in s for w in ("enter","arrive","transform","change","shift")):
        score += 0.15
    if any(w in s for w in ("gasp","shock","awe","breathtaking","stunning")):
        score += 0.25
    if any(w in s for w in ("landscape","environment","setting","atmosphere")):
        score += 0.1
    return min(1.0, score)

def _build_image_prompt(scene_text: str) -> str:
    s = (scene_text or "").lower()
    elems = []
    if "dramatic" in s: elems.append("dramatic lighting")
    if "intense"  in s: elems.append("intense atmosphere")
    if "beautiful" in s: elems.append("beautiful composition")
    return f"Scene depicting: {', '.join(elems) if elems else 'atmospheric scene'}"

def run_compat(agent, *, instruction=None, messages=None, context=None):
    if instruction is None and messages:
        parts = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") in (None, "user", "system"):
                parts.append(m.get("content", ""))
        instruction = "\n".join(p for p in parts if p)
    if instruction is None:
        instruction = ""
    from agents import Runner
    return Runner.run(agent, instruction, context=context)

def force_fix_tool_parameters(agent, *, tool_name="narrate_slice_of_life_scene"):
    """0.2.x-safe hardening for a single tool."""
    try:
        tools = getattr(agent, "tools", []) or []
        for t in tools:
            if getattr(t, "name", "") != tool_name:
                continue

            for attr in ("parameters", "_parameters", "_schema", "schema", "openai_schema"):
                params = getattr(t, attr, None)
                if not isinstance(params, dict):
                    continue

                props = params.get("properties")
                if not isinstance(props, dict):
                    continue

                params.pop("additionalProperties", None)
                params.pop("unevaluatedProperties", None)

                for k in ("scene", "world_state"):
                    pv = props.get(k)
                    if isinstance(pv, dict):
                        desc = pv.get("description") or f"{k} payload"
                        props[k] = {"type": "object", "description": desc}

                root_keys = list(props.keys())
                params["required"] = root_keys

                try:
                    setattr(t, attr, params)
                except Exception:
                    pass
    except Exception:
        logger.exception("force_fix_tool_parameters failed")

def sanitize_agent_tools_in_place(agent):
    """If any tools were created before our patches, sanitize their schema dicts in-place."""
    from .models import sanitize_json_schema
    try:
        tools = []
        
        if hasattr(agent, "tools"):
            tools = getattr(agent, "tools", []) or []
        elif hasattr(agent, "get_all_tools") and callable(agent.get_all_tools):
            try:
                tools = agent.get_all_tools() or []
            except TypeError as e:
                if "run_context" in str(e):
                    tools = getattr(agent, "tools", []) or []
                else:
                    raise
        elif hasattr(agent, "get_tools") and callable(agent.get_tools):
            try:
                tools = agent.get_tools() or []
            except TypeError:
                tools = getattr(agent, "tools", []) or []

        for t in tools:
            for attr in ("parameters", "_parameters", "_schema", "schema", "openai_schema"):
                val = getattr(t, attr, None)
                if isinstance(val, dict):
                    try:
                        setattr(t, attr, sanitize_json_schema(val))
                    except Exception:
                        logger.debug("Could not sanitize tool attr %s on %r", attr, t)
    except Exception:
        logger.exception("sanitize_agent_tools_in_place failed")

def debug_strict_schema_for_agent(agent: Any, log: logging.Logger = logger) -> None:
    """Log sanitized tool schemas at DEBUG. Never raises if a tool is wrapped."""
    try:
        tools = []
        
        if hasattr(agent, "tools"):
            tools = getattr(agent, "tools", []) or []
        elif hasattr(agent, "get_all_tools"):
            try:
                tools = agent.get_all_tools() or []
            except TypeError as e:
                if "run_context" in str(e):
                    log.debug("[strict] get_all_tools requires run_context, using direct attribute access")
                    tools = getattr(agent, "tools", []) or []
                else:
                    raise
        elif hasattr(agent, "get_tools") and callable(agent.get_tools):
            try:
                tools = agent.get_tools() or []
            except TypeError:
                tools = getattr(agent, "tools", []) or []

        log.debug("[strict] inspecting %d tools on agent %s", len(tools), getattr(agent, "name", agent))
        for i, t in enumerate(tools):
            try:
                name = getattr(t, "name", getattr(t, "__name__", f"tool_{i}"))
                log.debug("[strict] tool=%s", name)
            except Exception:
                name = f"tool_{i}"
                log.error("Could not inspect tool %s", name, exc_info=True)
    except Exception:
        log.exception("debug_strict_schema_for_agent: top-level failure")

def log_strict_hits(agent: Any) -> None:
    """Backwards-compatible alias."""
    debug_strict_schema_for_agent(agent, logger)

def get_canonical_context(ctx_obj) -> Any:
    """Convert various context objects to canonical context"""
    from lore.core.canon import ensure_canonical_context
    if hasattr(ctx_obj, 'user_id') and hasattr(ctx_obj, 'conversation_id'):
        return ensure_canonical_context({
            'user_id': ctx_obj.user_id,
            'conversation_id': ctx_obj.conversation_id
        })
    elif hasattr(ctx_obj, 'context'):
        return get_canonical_context(ctx_obj.context)
    else:
        raise ValueError("Cannot extract canonical context from object")

# Decision scoring helpers
def _calculate_context_relevance(option: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Calculate how relevant an option is to context"""
    score = 0.5
    
    option_keywords = set(str(option).lower().split())
    context_keywords = set(str(context).lower().split())
    
    overlap = len(option_keywords.intersection(context_keywords))
    if overlap > 0:
        score += min(0.3, overlap * 0.1)
    
    if context.get("scenario_type") and context["scenario_type"] in str(option):
        score += 0.2
    
    return min(1.0, score)

def _calculate_emotional_alignment(option: Dict[str, Any], emotional_state: Dict[str, float]) -> float:
    """Calculate emotional alignment score"""
    if "command" in str(option).lower() or "control" in str(option).lower():
        return emotional_state.get("dominance", 0.5)
    
    if "intense" in str(option).lower() or "extreme" in str(option).lower():
        return emotional_state.get("arousal", 0.5)
    
    if "reward" in str(option).lower() or "praise" in str(option).lower():
        return (emotional_state.get("valence", 0) + 1) / 2
    
    return 0.5

def _calculate_pattern_score(option: Dict[str, Any], learned_patterns: Dict[str, Any]) -> float:
    """Calculate score based on learned patterns"""
    if not learned_patterns:
        return 0.5
    
    option_str = str(option).lower()
    relevant_scores = []
    
    patterns_copy = dict(learned_patterns)
    for pattern_key, pattern_data in patterns_copy.items():
        if any(keyword in option_str for keyword in pattern_key.split("_")):
            success_rate = pattern_data.get("success_rate", 0.5)
            recency_factor = 1.0 / (1 + (time.time() - pattern_data.get("last_seen", 0)) / 3600)
            relevant_scores.append(success_rate * recency_factor)
    
    return sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.5

def _calculate_relationship_impact(option: Dict[str, Any], relationship_states: Dict[str, Dict[str, Any]]) -> float:
    """Calculate relationship impact score"""
    if not relationship_states:
        return 0.5
    
    avg_trust = sum(rel.get("trust", 0.5) for rel in relationship_states.values()) / len(relationship_states)
    
    if "risk" in str(option).lower() or "challenge" in str(option).lower():
        return avg_trust
    
    return 0.5 + (avg_trust * 0.5)

def _get_fallback_decision(options: List) -> Any:
    """Get a safe fallback decision"""
    from .models import DecisionOption, DecisionMetadata
    safe_words = ["talk", "observe", "wait", "consider", "listen", "pause"]
    for option in options:
        if any(safe_word in str(option).lower() for safe_word in safe_words):
            return option
    
    return options[0] if options else DecisionOption(
        id="fallback",
        description="Take a moment to assess",
        metadata=DecisionMetadata()
    )

async def _get_memory_emotional_impact(ctx, context: Dict[str, Any]) -> Dict[str, float]:
    """Get emotional impact from related memories"""
    return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

def should_generate_task(context: Dict[str, Any]) -> bool:
    """Determine if we should generate a creative task.
    DEPRECATED: Use NyxContext.should_generate_task() instead."""
    from .config import Config
    if not context.get("active_npc_id"):
        return False
    scenario_type = context.get("scenario_type", "").lower()
    task_scenarios = ["training", "challenge", "service", "discipline"]
    if not any(t in scenario_type for t in task_scenarios):
        return False
    npc_relationship = context.get("npc_relationship_level", 0)
    if npc_relationship < Config.MIN_NPC_RELATIONSHIP_FOR_TASK:
        return False
    return True

def enhance_context_with_memories(context: Dict[str, Any], memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add memories to context for better decision making."""
    enhanced_context = context.copy()
    enhanced_context['relevant_memories'] = memories
    return enhanced_context

def get_available_activities(scenario_type: str, relationship_states: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get available activities based on scenario type and relationships."""
    activities = []
    
    if "training" in scenario_type.lower() or any(rel.get("type") == "submissive" 
        for rel in relationship_states.values()):
        activities.extend([
            {
                "name": "Obedience Training",
                "description": "Test and improve submission through structured exercises",
                "requirements": ["trust > 0.4", "submission tendency"],
                "duration": "15-30 minutes",
                "intensity": "medium"
            },
            {
                "name": "Position Practice",
                "description": "Learn and perfect submissive positions",
                "requirements": ["trust > 0.5"],
                "duration": "10-20 minutes",
                "intensity": "low-medium"
            }
        ])
    
    for entity_id, rel in relationship_states.items():
        if rel.get("type") == "intimate" and rel.get("trust", 0) > 0.7:
            activities.append({
                "name": "Intimate Scene",
                "description": f"Deepen connection with trusted partner",
                "requirements": ["high trust", "intimate relationship"],
                "duration": "30-60 minutes",
                "intensity": "high",
                "partner_id": entity_id
            })
            break
    
    activities.extend([
        {
            "name": "Exploration",
            "description": "Discover new areas or items",
            "requirements": [],
            "duration": "10-30 minutes",
            "intensity": "low"
        },
        {
            "name": "Conversation",
            "description": "Engage in meaningful dialogue",
            "requirements": [],
            "duration": "5-15 minutes",
            "intensity": "low"
        }
    ])
    
    return activities

async def add_nyx_hosting_style(narrator_response: str, world_state: Any) -> Dict[str, str]:
    """Enhance narrator response with Nyx's hosting personality"""
    from logic.chatgpt_integration import generate_text_completion

    prompt = (
        "As Nyx, the AI Dominant host, respond to this slice-of-life moment.\n"
        f"World mood: {getattr(getattr(world_state, 'world_mood', None), 'value', '')}\n"
        f"Time of day: {getattr(getattr(getattr(world_state, 'current_time', None), 'time_of_day', None), 'value', '')}\n"
        f"Narration: {narrator_response}"
    )
    narrative = await generate_text_completion(
        system_prompt="You are Nyx, the AI Dominant host of this simulation",
        user_prompt=prompt,
    )
    return {"narrative": narrative}

def calculate_world_tension(world_state: Any) -> int:
    """Derive a tension level from world state"""
    return int(getattr(world_state, "tension", 0))

def should_generate_image_for_scene(world_state: Any) -> bool:
    """Placeholder logic for image generation decision"""
    return False

def detect_emergent_opportunities(world_state: Any) -> List[str]:
    """Return emergent narrative opportunities from world state"""
    opportunities = getattr(world_state, "emergent_opportunities", [])
    return [getattr(o, "description", str(o)) for o in opportunities]
