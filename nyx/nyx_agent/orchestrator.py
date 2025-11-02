# nyx/nyx_agent/orchestrator.py
"""Main orchestration and runtime functions for Nyx Agent SDK with enhanced reality enforcement"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Iterable, TYPE_CHECKING
from contextlib import asynccontextmanager
from types import SimpleNamespace

from agents import Agent, RunConfig, RunContextWrapper, ModelSettings
from db.connection import get_db_connection_context

# Import enhanced feasibility functions
from nyx.nyx_agent.feasibility import (
    assess_action_feasibility,
    record_impossibility,
    record_possibility,
    detect_setting_type,
    assess_action_feasibility_fast
)

from nyx.gateway.llm_gateway import execute, execute_stream, LLMRequest, LLMOperation
from nyx.telemetry.metrics import (
    REQUEST_LATENCY,
    TASK_FAILURES,
    record_queue_delay_from_context,
)
from nyx.telemetry.tracing import trace_step
from .config import Config
from nyx.config import flags

if TYPE_CHECKING:
    from nyx.nyx_agent_sdk import NyxSDKConfig
from .context import NyxContext, PackedContext
from ._feasibility_helpers import (
    DeferPromptContext,
    build_defer_fallback_text,
    build_defer_prompt,
    coalesce_agent_output_text,
    extract_defer_details,
)
from .models import *
from .agents import nyx_main_agent, nyx_defer_agent, reflection_agent, DEFAULT_MODEL_SETTINGS
from .assembly import assemble_nyx_response, resolve_scene_requests
from .tools import (
    update_relationship_state,
    generate_universal_updates_impl,
)
from .utils import (
    _did_call_tool,
    _extract_last_assistant_text,
    _js,
    sanitize_agent_tools_in_place,
    log_strict_hits,
    extract_runner_response,
)

# ---- optional punishment enforcer (refactored to accept meta) ---------------
try:
    # expects signature: enforce_all_rules_on_player(player_name, user_id, conversation_id, metadata)
    from logic.rule_enforcement import enforce_all_rules_on_player  # type: ignore
except Exception:  # pragma: no cover
    enforce_all_rules_on_player = None  # type: ignore

logger = logging.getLogger(__name__)


DEFAULT_DEFER_RUN_TIMEOUT_SECONDS: float = getattr(
    Config, "DEFER_RUN_TIMEOUT_SECONDS", 45.0
)
DEFER_RUN_TIMEOUT_SECONDS: float = DEFAULT_DEFER_RUN_TIMEOUT_SECONDS


def get_defer_run_timeout_seconds(config: Optional["NyxSDKConfig"] = None) -> float:
    """Return the defer agent timeout, preferring SDK configuration when provided."""

    if config is not None:
        timeout = getattr(config, "request_timeout_seconds", None)
        try:
            parsed = float(timeout)
        except (TypeError, ValueError):
            parsed = 0.0
        if parsed > 0:
            return parsed
    return DEFER_RUN_TIMEOUT_SECONDS


async def _generate_defer_taunt(
    context: DeferPromptContext,
    trace_id: str,
    nyx_context: Optional[NyxContext] = None,
) -> Optional[str]:
    """Ask Nyx to craft a defer response; fall back to None on failure."""

    if nyx_defer_agent is None:
        return None

    prompt = build_defer_prompt(context)
    if not prompt.strip():
        return None

    run_kwargs = {"max_turns": 2}
    if RunContextWrapper is not None and nyx_context is not None:
        try:
            run_kwargs["context"] = RunContextWrapper(nyx_context)
        except Exception:
            logger.debug(f"[{trace_id}] Failed to wrap Nyx context for defer taunt", exc_info=True)

    try:
        # Use remaining turn budget if available; else fall back to default
        remaining = None
        try:
            import time as _t
            dl = (getattr(nyx_context, "current_context", {}) or {}).get("_deadline")
            if dl:
                remaining = max(0.25, dl - _t.monotonic())
        except Exception:
            remaining = None
        request = LLMRequest(
            prompt=prompt,
            agent=nyx_defer_agent,
            metadata={"operation": LLMOperation.ORCHESTRATION.value},
            runner_kwargs=run_kwargs or None,
        )
        result_wrapper = await asyncio.wait_for(
            _execute_llm(request),
            timeout=remaining if remaining is not None else get_defer_run_timeout_seconds(),
        )
        result = result_wrapper.raw
    except asyncio.TimeoutError:
        logger.debug(
            f"[{trace_id}] Nyx defer taunt generation timed out",
            exc_info=True,
        )
        return None
    except Exception:
        logger.debug(f"[{trace_id}] Nyx defer taunt generation failed", exc_info=True)
        return None

    return coalesce_agent_output_text(result)


def _is_meaningful(value: Any) -> bool:
    """Return True if the value carries information (non-empty/None)."""
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, set, dict)):
        return bool(value)
    return True


def _ensure_list(value: Any) -> List[Any]:
    """Coerce arbitrary metadata into a list for scene storage."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (set, tuple)):
        return list(value)
    return [value]


def _normalize_scene_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize legacy scene metadata keys onto canonical fields."""
    normalized: Dict[str, Any] = dict(context or {})

    def adopt(canonical: str, legacy_keys: Iterable[str], default_factory: Optional[Any] = None,
              coerce_list: bool = False) -> None:
        if not _is_meaningful(normalized.get(canonical)):
            for key in legacy_keys:
                if key in normalized and _is_meaningful(normalized.get(key)):
                    value = normalized[key]
                    normalized[canonical] = _ensure_list(value) if coerce_list else value
                    break

        if canonical not in normalized:
            if callable(default_factory):
                normalized[canonical] = default_factory()
            elif default_factory is not None:
                normalized[canonical] = default_factory
        elif coerce_list:
            normalized[canonical] = _ensure_list(normalized.get(canonical))

    adopt("current_location", ("location", "active_location", "scene_location"), default_factory=lambda: {})
    adopt(
        "present_npcs",
        ("npc_present", "npcs", "present_entities", "participants", "active_npcs"),
        default_factory=list,
        coerce_list=True,
    )
    adopt(
        "available_items",
        ("items", "inventory", "inventory_items", "scene_items"),
        default_factory=list,
        coerce_list=True,
    )
    adopt(
        "recent_interactions",
        ("recent_turns", "recent_dialogue", "recent_messages"),
        default_factory=list,
        coerce_list=True,
    )

    if isinstance(normalized.get("recent_interactions"), list):
        canonical_turns: List[Dict[str, Any]] = []
        for entry in normalized.get("recent_interactions", []):
            if not isinstance(entry, dict):
                continue
            sender = entry.get("sender")
            content = entry.get("content")
            if sender is None and content is None:
                continue
            turn: Dict[str, Any] = {}
            if sender is not None:
                turn["sender"] = sender
            if content is not None:
                turn["content"] = content
            canonical_turns.append(turn)
        normalized["recent_interactions"] = canonical_turns
        if not _is_meaningful(normalized.get("recent_turns")):
            normalized["recent_turns"] = canonical_turns

    if not _is_meaningful(normalized.get("present_entities")):
        normalized["present_entities"] = list(normalized.get("present_npcs", []))

    return normalized


async def _execute_llm(request: LLMRequest):
    """Execute an LLM request honouring the rollout flag."""

    if flags.llm_gateway_enabled():
        return await execute(request)

    from agents import Runner  # local import to avoid circular dependency at import time

    runner_kwargs = dict(request.runner_kwargs or {})
    raw_result = await Runner.run(
        request.agent,
        request.prompt,
        context=request.context,
        **runner_kwargs,
    )

    agent_name: Optional[str] = None
    agent_spec = request.agent
    if hasattr(agent_spec, "name"):
        agent_name = getattr(agent_spec, "name")
    elif isinstance(agent_spec, str):
        agent_name = agent_spec

    text = (
        getattr(raw_result, "final_output", None)
        or getattr(raw_result, "output_text", None)
        or ""
    )

    return SimpleNamespace(
        text=text,
        raw=raw_result,
        agent_name=agent_name,
        metadata=dict(request.metadata or {}),
        attempts=1,
        used_fallback=False,
        duration=None,
    )


def _preserve_hydrated_location(target: Dict[str, Any], location: Any) -> None:
    """Ensure hydrated location metadata survives context merges."""

    if not _is_meaningful(location):
        return

    for key in ("current_location", "location", "location_name", "location_id"):
        if not _is_meaningful(target.get(key)):
            target[key] = location

# ===== Logging Helper =====
@asynccontextmanager
async def _log_step(name: str, trace_id: str, **meta):
    """Async context manager for logging step execution"""
    t0 = time.time()
    logger.debug(f"[{trace_id}] ▶ START {name} meta={_js(meta)}")
    with trace_step(f"nyx.orchestrator.{name}", trace_id, **meta):
        try:
            yield
            dt = time.time() - t0
            logger.info(f"[{trace_id}] ✔ DONE  {name} in {dt:.3f}s")
        except Exception:
            dt = time.time() - t0
            logger.exception(f"[{trace_id}] ✖ FAIL  {name} after {dt:.3f}s meta={_js(meta)}")
            raise


# ===== Error Handling =====
async def run_agent_safely(
    agent: Agent,
    input_data: Any,
    context: Any,
    run_config: Optional[RunConfig] = None,
    fallback_response: Any = None
) -> Any:
    """Run agent with automatic fallback on strict schema errors"""
    try:
        # First attempt with the agent as-is
        base_runner_kwargs = {"context": context}
        if run_config is not None:
            base_runner_kwargs["run_config"] = run_config
        request = LLMRequest(
            prompt=input_data,
            agent=agent,
            metadata={"operation": LLMOperation.ORCHESTRATION.value},
            runner_kwargs=base_runner_kwargs,
        )
        result_wrapper = await _execute_llm(request)
        return result_wrapper.raw
    except Exception as e:
        error_msg = str(e).lower()
        if "additionalproperties" in error_msg or "strict schema" in error_msg:
            logger.warning(f"Strict schema error, attempting without structured output: {e}")

            # Create a simple text-only agent
            fallback_agent = Agent(
                name=f"{getattr(agent, 'name', 'Agent')} (Fallback)",
                instructions=getattr(agent, 'instructions', ''),
                model=getattr(agent, 'model', None),
                model_settings=DEFAULT_MODEL_SETTINGS,
            )


            try:
                fallback_runner_kwargs = {"context": context}
                if run_config is not None:
                    fallback_runner_kwargs["run_config"] = run_config
                fallback_request = LLMRequest(
                    prompt=input_data,
                    agent=fallback_agent,
                    metadata={
                        "operation": LLMOperation.ORCHESTRATION.value,
                        "fallback": True,
                    },
                    runner_kwargs=fallback_runner_kwargs,
                )
                fallback_result = await _execute_llm(fallback_request)
                return fallback_result.raw
            except Exception as e2:
                logger.error(f"Fallback agent also failed: {e2}")
                if fallback_response is not None:
                    return fallback_response
                raise
        else:
            # Not a schema error, re-raise
            raise

async def run_agent_with_error_handling(
    agent: Agent,
    input_data: Any,
    context: NyxContext,
    output_type: Optional[type] = None,
    fallback_value: Any = None
) -> Any:
    """Legacy compatibility wrapper for running agents with error handling"""
    try:
        result = await run_agent_safely(
            agent,
            input_data,
            context,
            run_config=RunConfig(workflow_name=f"Nyx {getattr(agent, 'name', 'Agent')}"),
            fallback_response=fallback_value
        )
        if output_type:
            return result.final_output_as(output_type)
        return getattr(result, "final_output", None) or getattr(result, "output_text", None)
    except Exception as e:
        logger.error(f"Error running agent {getattr(agent, 'name', 'unknown')}: {e}")
        if fallback_value is not None:
            return fallback_value
        raise


async def decide_image_generation_standalone(ctx: NyxContext, scene_text: str) -> str:
    """Standalone image generation decision without tool context"""
    from nyx.nyx_agent.models import ImageGenerationDecision
    from nyx.nyx_agent.utils import _score_scene_text, _build_image_prompt
    
    # Ensure we have the actual NyxContext, not a wrapper
    if hasattr(ctx, 'context'):
        ctx = ctx.context
    
    score = _score_scene_text(scene_text)
    recent_images = ctx.current_context.get("recent_image_count", 0)
    threshold = 0.7 if recent_images > 3 else 0.6 if recent_images > 1 else 0.5

    should_generate = score > threshold
    image_prompt = _build_image_prompt(scene_text) if should_generate else None

    if should_generate:
        ctx.current_context["recent_image_count"] = recent_images + 1

    return ImageGenerationDecision(
        should_generate=should_generate,
        score=score,
        image_prompt=image_prompt,
        reasoning=f"Scene has visual impact score of {score:.2f} (threshold: {threshold:.2f})",
    ).model_dump_json()


# ===== Main Process Function with Enhanced Reality Enforcement =====
async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Process user input with non-blocking context initialization, background writes,
    and enhanced reality enforcement.
    """
    trace_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    monotonic = time.monotonic
    request_started = time.perf_counter()
    success = True
    failure_reason: Optional[str] = None
    record_queue_delay_from_context(context_data, queue="orchestrator")
    # Adopt a single absolute deadline from the SDK or create a reasonable one
    deadline = None
    if isinstance(context_data, dict):
        deadline = context_data.get("_deadline") or context_data.get("deadline")
    if deadline is None:
        deadline = monotonic() + get_defer_run_timeout_seconds()
    def time_left(floor: float = 0.25) -> float:
        return max(floor, deadline - monotonic())
    nyx_context: Optional[NyxContext] = None
    packed_context: Optional[PackedContext] = None

    logger.info(f"[{trace_id}] ========== PROCESS START ==========")
    logger.info(f"[{trace_id}] user_id={user_id} conversation_id={conversation_id}")
    logger.info(f"[{trace_id}] user_input={user_input[:200]}")

    try:
        # ---- STEP 0: Mandatory fast feasibility (dynamic) ---------------------
        logger.info(f"[{trace_id}] Running mandatory fast feasibility check")
        fast = None
        try:
            from nyx.nyx_agent.feasibility import assess_action_feasibility_fast
            fast = await assess_action_feasibility_fast(user_id, conversation_id, user_input)
            logger.info(f"[{trace_id}] Fast feasibility: {fast.get('overall', {})}")
        except Exception as e:
            logger.error(f"[{trace_id}] Fast feasibility failed softly: {e}", exc_info=True)

        if isinstance(fast, dict):
            overall = fast.get("overall", {})
            if overall.get("feasible") is False and (overall.get("strategy") or "").lower() == "deny":
                # [Existing early return logic - this is correct]
                per = fast.get("per_intent") or []
                first = per[0] if per and isinstance(per[0], dict) else {}
                guidance = first.get("narrator_guidance") or "That can't happen here. Try a grounded approach that fits the setting."
                options = [{"text": o} for o in (first.get("suggested_alternatives") or [])]

                logger.warning(f"[{trace_id}] ACTION BLOCKED (fast gate). Reason: {first.get('violations', [])}")
                return {
                    "success": True,
                    "response": guidance,
                    "metadata": {
                        "choices": options[:4],
                        "universal_updates": False,
                        "feasibility": fast,
                        "action_blocked": True,
                        "block_reason": (first.get("violations") or [{}])[0].get("reason", "setting constraints"),
                        "reality_maintained": True
                    },
                    "trace_id": trace_id,
                    "processing_time": time.time() - start_time,
                }

        # ---- STEP 1: Context initialization -----------------------------------
        async with _log_step("context_init", trace_id):
            nyx_context = NyxContext(user_id, conversation_id)
            # *** CHANGE 1: THIS CALL IS NOW NON-BLOCKING AND RETURNS IMMEDIATELY ***
            await nyx_context.initialize() 
            
            base_context = _normalize_scene_context(context_data or {})
            base_context["_deadline"] = deadline
            base_context["user_input"] = user_input
            base_context["last_user_input"] = user_input
            nyx_context.last_user_input = user_input
            nyx_context.current_context = base_context
            _preserve_hydrated_location(nyx_context.current_context, nyx_context.current_location)

            # This will now internally await the necessary subsystems "just-in-time"
            packed_context = await nyx_context.build_context_for_input(
                user_input,
                base_context,
            )
            nyx_context.last_packed_context = packed_context

        # ---- STEP 2: World state integration ----------------------------------
        async with _log_step("world_state", trace_id):
            # *** CHANGE 2: AWAIT THE 'WORLD' SUBSYSTEM BEFORE USING IT ***
            if nyx_context and await nyx_context.await_orchestrator("world"):
                if nyx_context.world_director and getattr(nyx_context.world_director, "context", None):
                    nyx_context.current_world_state = (
                        nyx_context.world_director.context.current_world_state
                    )

        # ---- STEP 3: Full feasibility (dynamic) --------------------------------
        logger.info(f"[{trace_id}] Running full feasibility assessment")
        feas = None
        enhanced_input = user_input  # Initialize with original input
        
        try:
            from nyx.nyx_agent.feasibility import assess_action_feasibility, record_impossibility, record_possibility
            feas = await assess_action_feasibility(nyx_context, user_input)
            nyx_context.current_context["feasibility"] = feas
            logger.info(f"[{trace_id}] Full feasibility: {feas.get('overall', {})}")
        except ImportError:
            logger.warning(f"[{trace_id}] Full feasibility not available; proceeding without it.")
        except Exception as e:
            logger.warning(f"[{trace_id}] Full feasibility failed softly: {e}", exc_info=True)

        if isinstance(feas, dict):
            # [This entire block of feasibility logic is fine and remains unchanged]
            overall = feas.get("overall", {})
            feasible_flag = overall.get("feasible")
            strategy = (overall.get("strategy") or "").lower()

            if feasible_flag is False and strategy == "deny":
                per = feas.get("per_intent") or []
                first = per[0] if per and isinstance(per[0], dict) else {}
                violations = first.get("violations", [])
                violation_text = violations[0]["reason"] if violations else "That violates the laws of this reality"

                try:
                    await record_impossibility(nyx_context, user_input, violation_text)
                except Exception:
                    logger.debug(f"[{trace_id}] record_impossibility failed softly", exc_info=True)

                rejection_narrative = f"*{first.get('reality_response', 'Reality ripples and refuses.')}*\n\n"
                rejection_narrative += first.get('narrator_guidance', 'The world itself resists your attempt.')
                alternatives = first.get("suggested_alternatives", [])
                if alternatives:
                    rejection_narrative += f"\n\n*Perhaps you could {alternatives[0]} instead.*"
                choices = [{"text": alt, "description": "A possible action within this reality", "feasible": True}
                           for alt in alternatives[:4]]

                return {
                    'success': True, 'response': rejection_narrative,
                    'metadata': {
                        'choices': choices, 'universal_updates': False, 'feasibility': feas,
                        'action_blocked': True, 'block_reason': violation_text, 'reality_maintained': True,
                    },
                    'trace_id': trace_id, 'processing_time': time.time() - start_time,
                }
            elif feasible_flag is False and strategy == "ask":
                constraints = feas.get("per_intent", [{}])[0].get("violations", [])
                constraint_text = "[REALITY CHECK: This action pushes boundaries. Consider: " + ", ".join(
                    v.get("reason", "") for v in constraints
                ) + ". Describe attempt with appropriate limitations.]"
                enhanced_input = f"{constraint_text}\n\n{user_input}"
            elif feasible_flag is False and strategy == "defer":
                defer_context, extra_meta = extract_defer_details(feas)
                leads = extra_meta.get("leads", [])
                guidance = None
                if defer_context:
                    guidance = await _generate_defer_taunt(defer_context, trace_id, nyx_context)
                if not guidance:
                    guidance = build_defer_fallback_text(defer_context) if defer_context else "Oh, pet, slow down. Reality keeps its heel on you until you ground that attempt."

                logger.info(f"[{trace_id}] ACTION DEFERRED (full feasibility)")
                metadata = {
                    "choices": [{"text": lead} for lead in leads[:4]], "universal_updates": False,
                    "feasibility": feas, "action_blocked": True, "action_deferred": True, "reality_maintained": True,
                }
                metadata.update(extra_meta)
                return {
                    "success": True, "response": guidance, "metadata": metadata,
                    "trace_id": trace_id, "processing_time": time.time() - start_time,
                }
            elif feasible_flag is True:
                try:
                    intents = feas.get("per_intent", [])
                    if intents and (cats := intents[0].get("categories")):
                        await record_possibility(nyx_context, user_input, cats)
                except Exception:
                    logger.debug(f"[{trace_id}] record_possibility failed softly", exc_info=True)
                enhanced_input = f"[REALITY CHECK: Action is feasible within universe laws.]\n\n{user_input}"

        # ---- STEP 4: Tool sanitization ----------------------------------------
        async with _log_step("tool_sanitization", trace_id):
            sanitize_agent_tools_in_place(nyx_main_agent)
            log_strict_hits(nyx_main_agent)

        # ---- STEP 5: Run main agent with enhanced input ----------------------
        async with _log_step("agent_run", trace_id):
            runner_context = RunContextWrapper(nyx_context)
            safe_settings = ModelSettings(strict_tools=False)
            run_config = RunConfig(model_settings=safe_settings)

            request = LLMRequest(
                prompt=enhanced_input,
                agent=nyx_main_agent,
                context=runner_context,
                runner_kwargs={"run_config": run_config},
                metadata={
                    "trace_id": trace_id,
                    "operation": LLMOperation.ORCHESTRATION.value,
                    "stream": False,
                    "tags": ["nyx", "orchestrator", "main"],
                },
            )
            result = await asyncio.wait_for(
                _execute_llm(request),
                timeout=time_left()
            )

            resp_stream = extract_runner_response(result.raw)

        # ---- STEP 6: Post-run enforcement (updates/image hooks + punishment) ---
        async with _log_step("post_run_enforcement", trace_id):
            # [This block of logic is fine and remains unchanged]
            if time_left() < 2.0:
                logger.info(f"[{trace_id}] Skipping post-run extras due to low budget")
            elif not _did_call_tool(resp_stream, "generate_universal_updates"):
                narrative = _extract_last_assistant_text(resp_stream)
                if narrative and len(narrative) > 20:
                    try:
                        update_result = await generate_universal_updates_impl(nyx_context, narrative)
                        resp_stream.append({
                            "type": "function_call_output", "name": "generate_universal_updates",
                            "output": json.dumps({
                                "success": update_result.success, "updates_generated": update_result.updates_generated,
                                "source": "post_run_injection"
                            })
                        })
                    except Exception as e:
                        logger.debug(f"[{trace_id}] Post-run universal updates failed softly: {e}")

            if time_left() >= 2.0 and not _did_call_tool(resp_stream, "decide_image_generation"):
                narrative = _extract_last_assistant_text(resp_stream)
                if narrative and len(narrative) > 20:
                    try:
                        image_result = await decide_image_generation_standalone(nyx_context, narrative)
                        resp_stream.append({
                            "type": "function_call_output", "name": "decide_image_generation",
                            "output": image_result
                        })
                    except Exception as e:
                        logger.debug(f"[{trace_id}] Post-run image decision failed softly: {e}")

            if time_left() >= 2.0 and enforce_all_rules_on_player:
                try:
                    player_name = (nyx_context.current_context.get("player_name") or "Chase")
                    punishment_meta = {
                        "scene_tags": nyx_context.current_context.get("scene_tags", []),
                        "stimuli": nyx_context.current_context.get("stimuli", []),
                        "feasibility": (feas or fast),
                        "turn_index": nyx_context.current_context.get("turn_index", 0),
                    }
                    punishment_result = await enforce_all_rules_on_player(
                        player_name=player_name, user_id=user_id,
                        conversation_id=conversation_id, metadata=punishment_meta,
                    )
                    resp_stream.append({
                        "type": "function_call_output", "name": "enforce_punishments",
                        "output": json.dumps(punishment_result)
                    })
                    nyx_context.current_context["punishment"] = punishment_result
                except Exception as e:
                    logger.debug(f"[{trace_id}] Punishment enforcement failed softly: {e}")
            else:
                logger.debug(f"[{trace_id}] punishment module unavailable; skipping enforcement")

        # ---- STEP 7: Response assembly (feasibility-aware) ---------------------
        async with _log_step("response_assembly", trace_id):
            resp_stream = await resolve_scene_requests(resp_stream, nyx_context)
            assembled = await assemble_nyx_response(
                agent_output=resp_stream,
                processing_metadata={
                    "feasibility": (feas or fast),
                    "punishment": nyx_context.current_context.get("punishment"),
                },
                user_input=user_input,
                conversation_id=str(conversation_id),
            )

            out = {
                "success": True, "response": assembled.narrative,
                "metadata": {
                    "world": getattr(assembled, "world_state", {}),
                    "choices": getattr(assembled, "choices", []),
                    "emergent": getattr(assembled, "emergent_events", []),
                    "image": getattr(assembled, "image", None),
                    "telemetry": (assembled.metadata or {}).get("performance", {}),
                    "nyx_commentary": (assembled.metadata or {}).get("nyx_commentary"),
                    "universal_updates": (assembled.metadata or {}).get("universal_updates", False),
                    "reality_maintained": True,
                    "punishment": nyx_context.current_context.get("punishment"),
                },
                "trace_id": trace_id, "processing_time": time.time() - start_time,
            }

            # *** CHANGE 3: CREATE AND LAUNCH THE BACKGROUND WRITE TASK ***
            async def perform_background_writes():
                try:
                    # Save the final context state (emotional, scene, etc.)
                    await _save_context_state(nyx_context)
                    
                    # Persist any universal updates (like location changes)
                    if nyx_context and nyx_context.context_broker:
                        bundle = getattr(nyx_context.context_broker, '_last_bundle', None)
                        if bundle:
                            pending_updates = await nyx_context.context_broker.collect_universal_updates(bundle)
                            if pending_updates:
                                await nyx_context.context_broker.apply_updates_async(pending_updates)
                    
                    logger.info(f"[{trace_id}] ✔ Background writes completed.")
                except Exception as e:
                    logger.error(f"[{trace_id}] ✖ Background write task failed: {e}", exc_info=True)

            # Schedule the writes to run in the background without blocking the response
            if nyx_context:
                task = asyncio.create_task(perform_background_writes())
                # Use the helper method from the context to track the task
                nyx_context._track_background_task(task, task_name="post_turn_writes")

            logger.info(f"[{trace_id}] ========== PROCESS COMPLETE (Response Sent) ==========")
            logger.info(f"[{trace_id}] Response length: {len(assembled.narrative or '')}")
            logger.info(f"[{trace_id}] Processing time: {out['processing_time']:.2f}s")
            
            return out

    except asyncio.CancelledError:
        success = False
        failure_reason = "cancelled"
        logger.info(f"[{trace_id}] ========== PROCESS CANCELLED ==========")
        raise
    except Exception as e:
        success = False
        failure_reason = type(e).__name__
        logger.error(f"[{trace_id}] ========== PROCESS FAILED ==========", exc_info=True)
        return {
            'success': False,
            'response': "I encountered an error processing your request. Please try again.",
            'error': str(e),
            'trace_id': trace_id,
            'processing_time': time.time() - start_time,
        }
    finally:
        duration = time.perf_counter() - request_started
        REQUEST_LATENCY.labels(component="orchestrator").observe(duration)
        if not success:
            TASK_FAILURES.labels(task="nyx_orchestrator", reason=failure_reason or "unknown").inc()

# ===== State Management =====
async def _save_context_state(ctx: NyxContext):
    """Save context state to database"""
    async with get_db_connection_context() as conn:
        try:
            normalized_context = _normalize_scene_context(getattr(ctx, "current_context", {}))
            if isinstance(getattr(ctx, "current_context", None), dict):
                ctx.current_context.clear()
                ctx.current_context.update(normalized_context)
            else:
                ctx.current_context = normalized_context

            # Get emotional state from current_context or provide default
            emotional_state = ctx.current_context.get('emotional_state', {
                'valence': 0.0,
                'arousal': 0.5,
                'dominance': 0.7
            })

            # Save emotional state
            await conn.execute("""
                INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, conversation_id) 
                DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
            """, ctx.user_id, ctx.conversation_id, json.dumps(emotional_state, ensure_ascii=False))
            
            # Save current scene state for future feasibility checks
            scene_state = {
                "location": ctx.current_context.get("current_location"),
                "items": ctx.current_context.get("available_items", []),
                "npcs": ctx.current_context.get("present_npcs", [])
            }
            
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'CurrentScene', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, ctx.user_id, ctx.conversation_id, json.dumps(scene_state))
            
            # Save scenario state if active
            if ctx.scenario_state and ctx.scenario_state.get("active") and ctx._tables_available.get("scenario_states", True):
                should_save_heartbeat = ctx.should_run_task("scenario_heartbeat")
                
                try:
                    if should_save_heartbeat:
                        await conn.execute("""
                            INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                        """, ctx.user_id, ctx.conversation_id, 
                        json.dumps(ctx.scenario_state, ensure_ascii=False))
                        
                        ctx.record_task_run("scenario_heartbeat")
                    else:
                        await conn.execute("""
                            INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                            ON CONFLICT (user_id, conversation_id) 
                            DO UPDATE SET state_data = $3, created_at = CURRENT_TIMESTAMP
                        """, ctx.user_id, ctx.conversation_id, 
                        json.dumps(ctx.scenario_state, ensure_ascii=False))
                except Exception as e:
                    if "does not exist" in str(e) or "no such table" in str(e).lower():
                        ctx._tables_available["scenario_states"] = False
                        logger.warning("scenario_states table not available - skipping save")
                    else:
                        raise
            
            # Save learning metrics periodically
            if ctx.should_run_task("learning_save"):
                await conn.execute("""
                    INSERT INTO learning_metrics (user_id, conversation_id, metrics, learned_patterns, created_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                """, ctx.user_id, ctx.conversation_id, 
                json.dumps(ctx.learning_metrics, ensure_ascii=False), 
                json.dumps(dict(list(ctx.learned_patterns.items())[-Config.MAX_LEARNED_PATTERNS:]), ensure_ascii=False))
                
                ctx.record_task_run("learning_save")
            
            # Save performance metrics periodically
            if ctx.should_run_task("performance_save"):
                bounded_metrics = ctx.performance_metrics.copy()
                if "response_times" in bounded_metrics:
                    bounded_metrics["response_times"] = bounded_metrics["response_times"][-Config.MAX_RESPONSE_TIMES:]
                
                await conn.execute("""
                    INSERT INTO performance_metrics (user_id, conversation_id, metrics, error_log, created_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                """, ctx.user_id, ctx.conversation_id,
                json.dumps(bounded_metrics, ensure_ascii=False),
                json.dumps(ctx.error_log[-Config.MAX_ERROR_LOG_ENTRIES:], ensure_ascii=False))
                
                ctx.record_task_run("performance_save")
                
        except Exception as e:
            logger.error(f"Error saving context state: {e}")

# ===== High-Level Operations =====
async def generate_reflection(
    user_id: int,
    conversation_id: int,
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a reflection from Nyx on a specific topic"""
    try:
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()

        prompt = f"Create a reflection about: {topic}" if topic else \
                 "Create a reflection about the user based on your memories"

        result = await run_agent_safely(
            reflection_agent,
            prompt,
            context=nyx_context,
            run_config=RunConfig(workflow_name="Nyx Reflection"),
        )

        reflection = result.final_output_as(MemoryReflection)
        return {
            "reflection": reflection.reflection,
            "confidence": reflection.confidence,
            "topic": reflection.topic or topic,
        }
    except Exception as e:
        logger.error(f"Error generating reflection: {e}")
        return {"reflection": "Unable to generate reflection at this time.", "confidence": 0.0, "topic": topic}

async def manage_scenario(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """DEPRECATED - Replace with emergent scenario management"""
    try:
        user_id = scenario_data.get("user_id")
        conversation_id = scenario_data.get("conversation_id")

        from story_agent.world_director_agent import CompleteWorldDirector

        director = CompleteWorldDirector(user_id, conversation_id)
        await director.initialize()

        next_moment = await director.generate_next_moment()

        return {
            "success": True,
            "emergent_scenario": next_moment.get("moment"),
            "world_state": next_moment.get("world_state"),
            "patterns": next_moment.get("patterns"),
            "linear_progression": None
        }
    except Exception as e:
        logger.error(f"Error managing scenario: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def manage_relationships(interaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """Manage and update relationships between entities."""
    nyx_context = None
    
    try:
        user_id = interaction_data.get("user_id")
        conversation_id = interaction_data.get("conversation_id")
        
        if not user_id or not conversation_id:
            raise ValueError("interaction_data must include user_id and conversation_id")
        
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        participants = interaction_data.get("participants", [])
        relationship_updates = {}
        
        for i, p1 in enumerate(participants):
            for p2 in participants[i+1:]:
                from .models import kvlist_to_dict
                p1_dict = kvlist_to_dict(p1) if not isinstance(p1, dict) else p1
                p2_dict = kvlist_to_dict(p2) if not isinstance(p2, dict) else p2
                
                entity_key = "_".join(sorted([str(p1_dict.get('id', p1)), str(p2_dict.get('id', p2))]))
                
                trust_change = 0.1 if interaction_data.get("outcome") == "success" else -0.05
                bond_change = 0.05 if interaction_data.get("emotional_impact", {}).get("positive", 0) > 0 else 0
                power_change = 0.0
                 
                if interaction_data.get("interaction_type") == "training":
                    power_change = 0.05
                elif interaction_data.get("interaction_type") == "conflict":
                    power_change = -0.05
                
                result = await update_relationship_state(
                    RunContextWrapper(context=nyx_context),
                    UpdateRelationshipStateInput(
                        entity_id=entity_key,
                        trust_change=trust_change,
                        power_change=power_change,
                        bond_change=bond_change
                    )
                )
                
                relationship_updates[entity_key] = json.loads(result)
        
        logger.warning("interaction_history table not found in schema - skipping interaction storage")
        
        for pair, updates in relationship_updates.items():
            await nyx_context.learn_from_interaction(
                action=f"relationship_{interaction_data.get('interaction_type', 'general')}",
                outcome=interaction_data.get("outcome", "unknown"),
                success=updates.get("changes", {}).get("trust", 0) > 0
            )
        
        return {
            "success": True,
            "relationship_updates": relationship_updates,
            "analysis": {
                "total_relationships_updated": len(relationship_updates),
                "interaction_type": interaction_data.get("interaction_type"),
                "outcome": interaction_data.get("outcome"),
                "stored_in_history": False
            }
        }
        
    except Exception as e:
        logger.error(f"Error managing relationships: {e}")
        if nyx_context:
            nyx_context.log_error(e, interaction_data)
        return {
            "success": False,
            "error": str(e)
        }

async def store_messages(user_id: int, conversation_id: int, user_input: str, nyx_response: str):
    """Store user and Nyx messages in database"""
    async with get_db_connection_context() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "user", user_input
        )
        
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "Nyx", nyx_response
        )
