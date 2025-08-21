# nyx/nyx_agent/orchestrator.py
"""Main orchestration and runtime functions for Nyx Agent SDK"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from agents import Agent, Runner, RunConfig, RunContextWrapper, ModelSettings
from db.connection import get_db_connection_context

from .config import Config
from .context import NyxContext
from .models import *
from .agents import nyx_main_agent, reflection_agent, DEFAULT_MODEL_SETTINGS
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

logger = logging.getLogger(__name__)

# ===== Logging Helper =====
@asynccontextmanager
async def _log_step(name: str, trace_id: str, **meta):
    """Async context manager for logging step execution"""
    t0 = time.time()
    logger.debug(f"[{trace_id}] ▶ START {name} meta={_js(meta)}")
    try:
        yield
        dt = time.time() - t0
        logger.info(f"[{trace_id}] ✔ DONE  {name} in {dt:.3f}s")
    except Exception:
        dt = time.time() - t0
        logger.exception(f"[{trace_id}] ✖ FAIL  {name} after {dt:.3f}s meta={_js(meta)}")
        raise

@asynccontextmanager
async def _log_step(name: str, trace_id: str, **meta):
    """Async context manager for logging step execution"""
    t0 = time.time()
    logger.debug(f"[{trace_id}] ▶ START {name} meta={_js(meta)}")
    try:
        yield
        dt = time.time() - t0
        logger.info(f"[{trace_id}] ✔ DONE  {name} in {dt:.3f}s")
    except Exception:
        dt = time.time() - t0
        logger.exception(f"[{trace_id}] ✖ FAIL  {name} after {dt:.3f}s meta={_js(meta)}")
        raise

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
        result = await Runner.run(
            agent,
            input_data,
            context=context,
            run_config=run_config
        )
        return result
    except Exception as e:
        error_msg = str(e).lower()
        if "additionalproperties" in error_msg or "strict schema" in error_msg:
            logger.warning(f"Strict schema error, attempting without structured output: {e}")
            
            # Create a simple text-only agent
            fallback_agent = Agent[type(context)](
                name=f"{agent.name} (Fallback)",
                instructions=agent.instructions,
                model=agent.model,
                model_settings=DEFAULT_MODEL_SETTINGS,
                # No tools, no structured output
            )
            
            try:
                result = await Runner.run(
                    fallback_agent,
                    input_data,
                    context=context,
                    run_config=run_config
                )
                return result
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
    from .models import ImageGenerationDecision
    from .utils import _score_scene_text, _build_image_prompt
    
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
        result = await Runner.run(
            agent,
            input_data,
            context=context,
            run_config=run_config
        )
        return result
    except Exception as e:
        error_msg = str(e).lower()
        if "additionalproperties" in error_msg or "strict schema" in error_msg:
            logger.warning(f"Strict schema error, attempting without structured output: {e}")
            
            # Create a simple text-only agent
            fallback_agent = Agent[type(context)](
                name=f"{agent.name} (Fallback)",
                instructions=agent.instructions,
                model=agent.model,
                model_settings=DEFAULT_MODEL_SETTINGS,
                # No tools, no structured output
            )
            
            try:
                result = await Runner.run(
                    fallback_agent,
                    input_data,
                    context=context,
                    run_config=run_config
                )
                return result
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

# ===== Main Process Function =====
async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Process user input with post-run enforcement and robust assembly"""
    trace_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    nyx_context = None

    logger.info(f"[{trace_id}] ========== PROCESS START ==========")
    logger.info(f"[{trace_id}] user_id={user_id} conversation_id={conversation_id}")
    logger.info(f"[{trace_id}] user_input={user_input[:100]}")
    logger.info(f"[{trace_id}] context_data keys: {list(context_data.keys()) if context_data else 'None'}")

    try:
        # ===== STEP 1: Context initialization =====
        async with _log_step("context_init", trace_id):
            nyx_context = NyxContext(user_id, conversation_id)
            await nyx_context.initialize()
            nyx_context.current_context = (context_data or {}).copy()
            nyx_context.current_context["user_input"] = user_input

        # ===== STEP 2: World state integration =====
        async with _log_step("world_state", trace_id):
            if nyx_context.world_director and nyx_context.world_director.context:
                world_state = nyx_context.world_director.context.current_world_state
                nyx_context.current_world_state = world_state

        # ===== STEP 3: Tool sanitization =====
        async with _log_step("tool_sanitization", trace_id):
            sanitize_agent_tools_in_place(nyx_main_agent)
            log_strict_hits(nyx_main_agent)

        # ===== STEP 4: Running the agent =====
        async with _log_step("agent_run", trace_id):
            runner_context = RunContextWrapper(nyx_context)
            safe_settings = ModelSettings(strict_tools=False, response_format=None)
            run_config = RunConfig(model_settings=safe_settings)
            
            result = await Runner.run(
                nyx_main_agent,
                user_input,
                context=runner_context,
                run_config=run_config
            )
            
            # Convert result to list format for processing
            resp = []
            
            # Try different ways to extract the response history
            if hasattr(result, 'messages'):
                resp = result.messages
            elif hasattr(result, 'history'):
                resp = result.history
            elif hasattr(result, 'events'):
                resp = result.events
            elif hasattr(result, '__iter__'):
                try:
                    resp = list(result)
                except:
                    pass
            
            # If we still don't have a response list, create minimal structure
            if not resp:
                response_text = ""
                if hasattr(result, 'final_output'):
                    response_text = str(result.final_output)
                elif hasattr(result, 'output'):
                    response_text = str(result.output)
                elif hasattr(result, 'text'):
                    response_text = str(result.text)
                else:
                    response_text = str(result)
                
                resp = [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": response_text}]
                    }
                ]

        # ===== STEP 5: Post-run enforcement =====
        async with _log_step("post_run_enforcement", trace_id):
            # Check and inject generate_universal_updates if missing
            if not _did_call_tool(resp, "generate_universal_updates"):
                narrative = _extract_last_assistant_text(resp)
                if narrative and len(narrative) > 20:
                    try:
                        update_result = await generate_universal_updates_impl(nyx_context, narrative)
                        resp.append({
                            "type": "function_call_output",
                            "name": "generate_universal_updates",
                            "output": json.dumps({
                                "success": update_result.success,
                                "updates_generated": update_result.updates_generated,
                                "source": "post_run_injection"
                            })
                        })
                    except Exception as e:
                        logger.exception(f"[{trace_id}] Post-run universal updates failed: {e}")
            
            # Check and inject decide_image_generation if missing
            if not _did_call_tool(resp, "decide_image_generation"):
                narrative = _extract_last_assistant_text(resp)
                if narrative and len(narrative) > 20:
                    try:
                        image_result = await decide_image_generation_standalone(nyx_context, narrative)
                        resp.append({
                            "type": "function_call_output",
                            "name": "decide_image_generation",
                            "output": image_result
                        })
                    except Exception as e:
                        logger.exception(f"[{trace_id}] Post-run image decision failed: {e}")

        # ===== STEP 6: Response assembly =====
        async with _log_step("response_assembly", trace_id):
            # Resolve any scene requests
            resp = await resolve_scene_requests(resp, nyx_context)
            
            # Assemble final response
            assembled = assemble_nyx_response(resp)
            
            # Save state changes
            await _save_context_state(nyx_context)
            
            result = {
                'success': True,
                'response': assembled['narrative'],
                'metadata': {
                    'world': assembled['world'],
                    'choices': assembled['choices'],
                    'emergent': assembled['emergent'],
                    'image': assembled['image'],
                    'telemetry': assembled['telemetry'],
                    'nyx_commentary': assembled.get('nyx_commentary'),
                    'universal_updates': assembled.get('universal_updates', False)
                },
                'trace_id': trace_id,
                'processing_time': time.time() - start_time,
            }
            
            # Track performance metrics
            nyx_context.update_performance("response_times", result['processing_time'])
            nyx_context.update_performance("successful_actions", nyx_context.performance_metrics.get("successful_actions", 0) + 1)
            nyx_context.update_performance("total_actions", nyx_context.performance_metrics.get("total_actions", 0) + 1)
            
            logger.info(f"[{trace_id}] ========== PROCESS COMPLETE ==========")
            logger.info(f"[{trace_id}] Response length: {len(assembled['narrative'])}")
            logger.info(f"[{trace_id}] Processing time: {result['processing_time']:.2f}s")
            
            return result
            
    except Exception as e:
        logger.error(f"[{trace_id}] ========== PROCESS FAILED ==========")
        logger.error(f"[{trace_id}] Fatal error in process_user_input", exc_info=True)
        
        # Track error
        if nyx_context:
            nyx_context.log_error(e, {"user_input": user_input, "context_data": context_data})
            nyx_context.update_performance("failed_actions", nyx_context.performance_metrics.get("failed_actions", 0) + 1)
            nyx_context.update_performance("total_actions", nyx_context.performance_metrics.get("total_actions", 0) + 1)
        
        return {
            'success': False,
            'response': "I encountered an error processing your request. Please try again.",
            'error': str(e),
            'trace_id': trace_id,
            'processing_time': time.time() - start_time,
        }

# ===== State Management =====
async def _save_context_state(ctx: NyxContext):
    """Save context state to database"""
    async with get_db_connection_context() as conn:
        try:
            # Save emotional state
            await conn.execute("""
                INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, conversation_id) 
                DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
            """, ctx.user_id, ctx.conversation_id, json.dumps(ctx.emotional_state, ensure_ascii=False))
            
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
