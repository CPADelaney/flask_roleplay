# nyx/nyx_agent_sdk.py

"""
Nyx Agent SDK - Main Orchestrator

Enhanced version that fully leverages all integrated orchestrators:
- NPC Orchestrator for character management
- Memory Orchestrator for rich memory systems
- Conflict Synthesizer for dynamic tensions
- Lore Orchestrator for world-building
- World Director for state management
- Narrator for immersive storytelling
- Game Logic Systems for mechanics
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

# Import all modularized components
from .nyx_agent import *
from .nyx_agent.config import Config
from .nyx_agent.context import NyxContext
from .nyx_agent.models import *
from .nyx_agent.tools import *
from .nyx_agent.agents import nyx_main_agent, DEFAULT_MODEL_SETTINGS
from .nyx_agent.assembly import assemble_nyx_response, resolve_scene_requests
from .nyx_agent.utils import (
    _did_call_tool, _extract_last_assistant_text, _js,
    sanitize_agent_tools_in_place, log_strict_hits,
    extract_runner_response
)
from .nyx_agent.compatibility import (
    AgentContext,
    enhance_context_with_strategies,
    determine_image_generation,
    process_user_input_with_openai,
    process_user_input_standalone,
    mark_strategy_for_review,
    generate_base_response,
    get_emotional_state,
    update_emotional_state
)

# Import agents SDK components
from agents import Agent, Runner, RunConfig, RunContextWrapper, GuardrailFunctionOutput, ModelSettings

# Import database connection
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===== Logging Helper =====
@asynccontextmanager
async def _log_step(name: str, trace_id: str, **meta):
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
            
            fallback_agent = Agent[type(context)](
                name=f"{agent.name} (Fallback)",
                instructions=agent.instructions,
                model=agent.model,
                model_settings=DEFAULT_MODEL_SETTINGS,
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
            raise

# ===== Main Processing Function =====
async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Process user input with full orchestrator integration"""
    trace_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    nyx_context = None

    logger.info(f"[{trace_id}] ========== PROCESS START ==========")
    logger.info(f"[{trace_id}] user_id={user_id} conversation_id={conversation_id}")
    logger.info(f"[{trace_id}] user_input={user_input[:100]}")

    try:
        # ===== STEP 1: Full Context Initialization =====
        async with _log_step("full_context_init", trace_id):
            nyx_context = NyxContext(user_id, conversation_id)
            await nyx_context.initialize()
            
            # Set initial context
            nyx_context.current_context = (context_data or {}).copy()
            nyx_context.current_context["user_input"] = user_input

        # ===== STEP 2: Synchronize All Orchestrators =====
        async with _log_step("orchestrator_sync", trace_id):
            # Sync World Director (provides world state, NPCs, tensions)
            await nyx_context.sync_world_director_data(force=True)
            
            # Sync Narrator (provides narrative context, atmosphere)
            await nyx_context.sync_narrator_data(force=True)
            
            # Sync Lore (provides world lore, cultures, geopolitics)
            await nyx_context.sync_lore_data(force=True)
            
            # Sync Game Systems
            await nyx_context.sync_game_time(force=True)
            await nyx_context.sync_player_stats(force=True)
            await nyx_context.sync_inventory_data(force=True)
            await nyx_context.sync_addiction_status(force=True)
            await nyx_context.sync_calendar_events(force=True)
            await nyx_context.sync_narrative_progression(force=True)
            await nyx_context.sync_activity_data(force=True)

        # ===== STEP 3: Update Scene Context =====
        async with _log_step("scene_context_update", trace_id):
            # Load NPC context for current scene
            await nyx_context._load_npc_context()
            
            # Update conflict context for current scene
            location = nyx_context.current_location or nyx_context.current_context.get('location')
            scene_type = nyx_context.current_context.get('scene_type', 'interaction')
            
            await nyx_context.update_conflict_context(
                location=location,
                scene_type=scene_type,
                npc_ids=list(nyx_context.current_scene_npcs)
            )
            
            # Process scene for conflicts
            scene_result = await nyx_context.process_scene_conflicts({
                'user_input': user_input,
                'scene_type': scene_type,
                'location': location
            })

        # ===== STEP 4: Enrich Context with Memories =====
        async with _log_step("memory_enrichment", trace_id):
            # Enrich context with relevant memories
            await nyx_context.enrich_context_with_memories()
            
            # Get narrative memory context
            memory_narrative = await nyx_context.get_narrative_memory_context(
                include_npcs=True,
                include_player=True,
                include_canon=True,
                time_window_hours=24
            )
            
            # Analyze memory patterns for predictions
            memory_patterns = await nyx_context.analyze_memory_patterns(
                topic=user_input[:50]  # Use truncated input as topic
            )

        # ===== STEP 5: Run Periodic Maintenance Tasks =====
        async with _log_step("maintenance_tasks", trace_id):
            # NPC maintenance
            if nyx_context.should_run_task("npc_perception_update"):
                await nyx_context.update_npc_perceptions()
            
            if nyx_context.should_run_task("npc_decision_cycle"):
                npc_decisions = await nyx_context.trigger_npc_decisions({
                    "user_input": user_input,
                    "scene_context": scene_type
                })
            
            if nyx_context.should_run_task("npc_scheming_check"):
                scheming_results = await nyx_context.check_npc_scheming()
            
            # Conflict maintenance
            if nyx_context.should_run_task("tension_calculation"):
                await nyx_context.calculate_conflict_tensions()
            
            if nyx_context.should_run_task("conflict_resolution_check"):
                resolution_opportunities = await nyx_context.check_conflict_resolution_opportunities()
            
            # Lore maintenance
            if nyx_context.should_run_task("lore_evolution"):
                await nyx_context.check_lore_evolution()
            
            if nyx_context.should_run_task("cultural_update"):
                await nyx_context.update_cultural_context()
            
            if nyx_context.should_run_task("geopolitical_check"):
                await nyx_context.check_geopolitical_state()
            
            # Memory maintenance
            if nyx_context.should_run_task("memory_maintenance"):
                await nyx_context.run_memory_maintenance(
                    operations=["consolidate", "cleanup", "pattern_analysis"]
                )
            
            # Check for narrative moments
            await nyx_context.check_narrative_moments()

        # ===== STEP 6: Build Comprehensive Context =====
        async with _log_step("comprehensive_context", trace_id):
            # Get the fully integrated context
            comprehensive_context = nyx_context.get_comprehensive_context_for_response()
            
            # Add specific sub-contexts
            comprehensive_context['npc_detail'] = nyx_context.get_npc_context_for_response()
            comprehensive_context['conflict_detail'] = nyx_context.get_conflict_context_for_response()
            comprehensive_context['lore_detail'] = nyx_context.get_lore_context_for_response()
            
            # Update agent's current context with comprehensive data
            nyx_context.current_context.update(comprehensive_context)
            
            logger.debug(f"[{trace_id}] Context includes: {len(comprehensive_context['npcs_present'])} NPCs, "
                        f"{len(comprehensive_context['active_conflicts'])} conflicts, "
                        f"{len(comprehensive_context['lore']['active_nations'])} nations")

        # ===== STEP 7: Tool Sanitization =====
        async with _log_step("tool_sanitization", trace_id):
            sanitize_agent_tools_in_place(nyx_main_agent)
            log_strict_hits(nyx_main_agent)

        # ===== STEP 8: Run Main Agent with Rich Context =====
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

        # ===== STEP 9: Process Agent Response =====
        async with _log_step("response_processing", trace_id):
            # Resolve any scene requests
            resp = await resolve_scene_requests(resp, nyx_context)
            
            # Do early assembly to extract narrative
            assembled = assemble_nyx_response(resp)
            narrative = assembled.get('narrative') or assembled.get('full_text') or ""
            
            logger.debug(f"[{trace_id}] Narrative extracted: {len(narrative)} chars")

        # ===== STEP 10: Post-Run Updates =====
        async with _log_step("post_run_updates", trace_id):
            # Ensure universal updates were called
            if not _did_call_tool(resp, "generate_universal_updates"):
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
                        logger.error(f"[{trace_id}] Post-run universal updates failed: {e}")
            
            # Process any NPC interactions that occurred
            if assembled.get('npc') and assembled['npc'].get('last_dialogue'):
                dialogue_data = assembled['npc']['last_dialogue']
                if dialogue_data.get('npc_id'):
                    interaction_result = await nyx_context.process_npc_interaction(
                        npc_id=dialogue_data['npc_id'],
                        interaction_type='dialogue',
                        player_input=user_input,
                        context={
                            'npc_response': dialogue_data.get('text'),
                            'tone': dialogue_data.get('tone')
                        }
                    )
            
            # Create memories for significant events
            if narrative and len(narrative) > 50:
                # Store player memory
                await nyx_context.store_memory(
                    entity_type="player",
                    entity_id=user_id,
                    memory_text=f"User said: {user_input[:200]}. Nyx responded with: {narrative[:200]}",
                    importance="medium" if len(narrative) > 100 else "low",
                    emotional=any(word in narrative.lower() for word in ['feel', 'emotion', 'love', 'hate', 'fear']),
                    tags=["interaction", scene_type]
                )
                
                # Store NPC memories for involved NPCs
                for npc_id in list(nyx_context.current_scene_npcs)[:3]:  # Limit to 3 NPCs
                    await nyx_context.create_npc_memory(
                        npc_id=npc_id,
                        memory_text=f"Witnessed interaction: {user_input[:100]}",
                        memory_type="observation",
                        significance=3,
                        emotional_valence=0
                    )
            
            # Check for world evolution based on significant events
            if assembled.get('emergent') and len(assembled['emergent']) > 0:
                for emergent_event in assembled['emergent'][:1]:  # Process first emergent event
                    await nyx_context.evolve_world_lore(emergent_event)
            
            # Update beliefs based on interactions
            if nyx_context.current_scene_npcs:
                for npc_id in list(nyx_context.current_scene_npcs)[:2]:  # Limit to 2 NPCs
                    await nyx_context.create_belief(
                        entity_type="npc",
                        entity_id=npc_id,
                        belief_text=f"The player seems to be {nyx_context.emotional_state.get('dominance', 0.5) > 0.5 and 'assertive' or 'submissive'}",
                        confidence=0.6
                    )

        # ===== STEP 11: Final Assembly =====
        async with _log_step("final_assembly", trace_id):
            # Re-assemble with all updates
            assembled = assemble_nyx_response(resp)
            
            # Add orchestrator metrics
            assembled['orchestrator_metrics'] = {
                'npcs_active': len(nyx_context.current_scene_npcs),
                'conflicts_active': len(nyx_context.active_conflicts),
                'memories_recent': sum(len(mems) for mems in nyx_context.recent_memories.values()),
                'lore_nations': len(nyx_context.active_nations),
                'lore_religions': len(nyx_context.active_religions),
                'world_synced': nyx_context.last_world_sync > 0,
                'narrator_synced': nyx_context.last_narrative_sync > 0,
                'lore_synced': nyx_context.last_lore_sync > 0
            }
            
            # Save state changes
            await _save_context_state(nyx_context)
            
            # Build final response
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
                    'universal_updates': assembled.get('universal_updates', False),
                    'orchestrator_metrics': assembled['orchestrator_metrics'],
                    'conflicts': nyx_context.get_conflict_context_for_response(),
                    'npcs': nyx_context.get_npc_context_for_response(),
                    'lore_context': nyx_context.get_lore_context_for_response()
                },
                'trace_id': trace_id,
                'processing_time': time.time() - start_time,
            }
            
            # Track performance metrics
            nyx_context.update_performance("response_times", result['processing_time'])
            nyx_context.update_performance("successful_actions", nyx_context.performance_metrics.get("successful_actions", 0) + 1)
            nyx_context.update_performance("total_actions", nyx_context.performance_metrics.get("total_actions", 0) + 1)
            
            # Handle high memory usage if needed
            memory_usage = nyx_context.performance_metrics.get("memory_usage", 0)
            if memory_usage > Config.MAX_MEMORY_MB * 0.8:
                await nyx_context.handle_high_memory_usage()
            
            logger.info(f"[{trace_id}] ========== PROCESS COMPLETE ==========")
            logger.info(f"[{trace_id}] Response length: {len(assembled['narrative'])}")
            logger.info(f"[{trace_id}] Processing time: {result['processing_time']:.2f}s")
            logger.info(f"[{trace_id}] Orchestrators engaged: {assembled['orchestrator_metrics']}")
            
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
            'response': "I encountered an error processing your request. Let me try to recover...",
            'error': str(e),
            'trace_id': trace_id,
            'processing_time': time.time() - start_time,
        }

async def _save_context_state(ctx: NyxContext):
    """Enhanced state saving with all orchestrator data"""
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
            if ctx.scenario_state and ctx.scenario_state.get("active"):
                if ctx.should_run_task("scenario_heartbeat"):
                    await conn.execute("""
                        INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                    """, ctx.user_id, ctx.conversation_id, 
                    json.dumps(ctx.scenario_state, ensure_ascii=False))
                    ctx.record_task_run("scenario_heartbeat")
            
            # Save orchestrator state periodically
            if ctx.should_run_task("orchestrator_state_save"):
                orchestrator_state = {
                    'npc_snapshots': {str(k): v for k, v in ctx.npc_snapshots.items()},
                    'active_conflicts': {str(k): v for k, v in ctx.active_conflicts.items()},
                    'conflict_tensions': ctx.conflict_tensions,
                    'world_lore': ctx.world_lore,
                    'active_nations': {str(k): v for k, v in ctx.active_nations.items()},
                    'active_religions': {str(k): v for k, v in ctx.active_religions.items()},
                    'recent_memories': ctx.recent_memories,
                    'belief_systems': ctx.belief_systems,
                    'memory_predictions': ctx.memory_predictions,
                    'emergent_narratives': ctx.emergent_narratives[-5:],  # Keep last 5
                    'last_sync_times': {
                        'world': ctx.last_world_sync,
                        'narrator': ctx.last_narrative_sync,
                        'lore': ctx.last_lore_sync
                    }
                }
                
                await conn.execute("""
                    INSERT INTO orchestrator_state (user_id, conversation_id, state_data, created_at)
                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, conversation_id)
                    DO UPDATE SET state_data = $3, created_at = CURRENT_TIMESTAMP
                """, ctx.user_id, ctx.conversation_id,
                json.dumps(orchestrator_state, ensure_ascii=False))
                
                ctx.record_task_run("orchestrator_state_save")
            
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

# ===== Enhanced Helper Functions =====

async def generate_contextual_reflection(
    user_id: int,
    conversation_id: int,
    topic: Optional[str] = None,
    include_orchestrators: bool = True
) -> Dict[str, Any]:
    """Generate a reflection with full orchestrator context"""
    try:
        from .nyx_agent.agents import reflection_agent
        
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        # Gather orchestrator context if requested
        reflection_context = {}
        if include_orchestrators:
            await nyx_context.refresh_all_context("reflection_generation")
            reflection_context = {
                'npcs': nyx_context.get_npc_context_for_response(),
                'conflicts': nyx_context.get_conflict_context_for_response(),
                'lore': nyx_context.get_lore_context_for_response(),
                'memories': await nyx_context.get_narrative_memory_context()
            }

        prompt = f"Create a reflection about: {topic}" if topic else \
                 f"Create a reflection based on this context: {json.dumps(reflection_context, indent=2)[:1000]}"

        result = await run_agent_safely(
            reflection_agent,
            prompt,
            context=nyx_context,
            run_config=RunConfig(workflow_name="Nyx Contextual Reflection"),
        )

        reflection = result.final_output_as(MemoryReflection)
        
        # Store the reflection as a memory
        if reflection.confidence > 0.5:
            await nyx_context.store_memory(
                entity_type="system",
                entity_id=0,
                memory_text=reflection.reflection,
                importance="high" if reflection.confidence > 0.8 else "medium",
                tags=["reflection", "nyx_insight"],
                emotional=True
            )
        
        return {
            "reflection": reflection.reflection,
            "confidence": reflection.confidence,
            "topic": reflection.topic or topic,
            "context_used": reflection_context if include_orchestrators else None
        }
    except Exception as e:
        logger.error(f"Error generating contextual reflection: {e}")
        return {"reflection": "Unable to generate reflection at this time.", "confidence": 0.0, "topic": topic}

async def orchestrate_scene_transition(
    user_id: int,
    conversation_id: int,
    from_location: str,
    to_location: str,
    transition_type: str = "movement"
) -> Dict[str, Any]:
    """Orchestrate a scene transition with all systems"""
    try:
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        # Update location in context
        nyx_context.current_location = to_location
        nyx_context.current_context['location'] = to_location
        nyx_context.current_context['previous_location'] = from_location
        
        # Update conflict context for new location
        await nyx_context.update_conflict_context(location=to_location)
        
        # Get NPCs at new location
        if nyx_context.npc_orchestrator:
            npcs_at_location = await nyx_context.npc_orchestrator.get_npcs_at_location(to_location)
            nyx_context.current_scene_npcs = [npc.npc_id for npc in npcs_at_location]
        
        # Get lore for new location
        if nyx_context.lore_orchestrator:
            location_lore = await nyx_context.lore_orchestrator.get_location_lore(
                nyx_context.lore_orchestrator.create_run_context(),
                nyx_context.current_context.get('location_id')
            )
            nyx_context.current_location_lore = location_lore
        
        # Generate transition narrative
        if nyx_context.slice_of_life_narrator:
            transition_narrative = await nyx_context.slice_of_life_narrator.narrate_transition(
                from_location, to_location, transition_type
            )
        else:
            transition_narrative = f"You move from {from_location} to {to_location}."
        
        # Store transition as memory
        await nyx_context.store_memory(
            entity_type="player",
            entity_id=user_id,
            memory_text=f"Moved from {from_location} to {to_location}",
            importance="low",
            tags=["movement", "transition"]
        )
        
        return {
            "success": True,
            "narrative": transition_narrative,
            "new_location": to_location,
            "npcs_present": nyx_context.current_scene_npcs,
            "location_lore": nyx_context.current_location_lore,
            "active_conflicts": list(nyx_context.scene_conflicts)
        }
        
    except Exception as e:
        logger.error(f"Error orchestrating scene transition: {e}")
        return {"success": False, "error": str(e)}

async def trigger_emergent_event(
    user_id: int,
    conversation_id: int,
    event_type: Optional[str] = None
) -> Dict[str, Any]:
    """Trigger an emergent event using all orchestrators"""
    try:
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        await nyx_context.refresh_all_context("emergent_event")
        
        emergent_data = {}
        
        # Generate conflict-based event
        if event_type == "conflict" or (not event_type and len(nyx_context.active_conflicts) > 0):
            conflict_id = list(nyx_context.active_conflicts.keys())[0] if nyx_context.active_conflicts else None
            if conflict_id:
                emergent_data['conflict_escalation'] = await nyx_context.update_conflict(
                    conflict_id, "escalate", {"reason": "emergent_event"}
                )
        
        # Generate NPC-based event
        if event_type == "npc" or (not event_type and nyx_context.current_scene_npcs):
            if nyx_context.npc_orchestrator:
                npc_id = nyx_context.current_scene_npcs[0] if nyx_context.current_scene_npcs else None
                if npc_id:
                    decision = await nyx_context.npc_orchestrator.make_npc_decision(
                        npc_id, {"trigger": "emergent_event"}
                    )
                    emergent_data['npc_action'] = decision
        
        # Generate lore-based event
        if event_type == "lore" or not event_type:
            if nyx_context.lore_orchestrator:
                lore_event = await nyx_context.lore_orchestrator.generate_emergent_event(
                    nyx_context.lore_orchestrator.create_run_context()
                )
                emergent_data['lore_event'] = lore_event
        
        # Generate world-based event
        if nyx_context.world_director:
            world_event = await nyx_context.world_director.generate_emergent_event()
            emergent_data['world_event'] = world_event
        
        # Create narrative for the event
        if nyx_context.slice_of_life_narrator and emergent_data:
            event_narrative = await nyx_context.slice_of_life_narrator.narrate_emergent_event(emergent_data)
        else:
            event_narrative = "Something unexpected happens..."
        
        # Store event as memory
        await nyx_context.store_memory(
            entity_type="system",
            entity_id=0,
            memory_text=f"Emergent event occurred: {event_type or 'mixed'}",
            importance="high",
            tags=["emergent", "event"],
            emotional=True
        )
        
        return {
            "success": True,
            "narrative": event_narrative,
            "event_data": emergent_data,
            "event_type": event_type or "mixed"
        }
        
    except Exception as e:
        logger.error(f"Error triggering emergent event: {e}")
        return {"success": False, "error": str(e)}

# ===== Preserve existing functions =====
async def initialize_agents():
    """Initialize necessary resources for the agents system"""
    pass

async def generate_reflection(user_id: int, conversation_id: int, topic: Optional[str] = None) -> Dict[str, Any]:
    """Legacy wrapper for reflection generation"""
    return await generate_contextual_reflection(user_id, conversation_id, topic, include_orchestrators=False)

async def manage_scenario(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced scenario management with orchestrators"""
    try:
        user_id = scenario_data.get("user_id")
        conversation_id = scenario_data.get("conversation_id")
        
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        await nyx_context.refresh_all_context("scenario_management")
        
        # Let world director generate next moment
        if nyx_context.world_director:
            next_moment = await nyx_context.world_director.generate_next_moment()
        else:
            next_moment = {"moment": "The scenario continues..."}
        
        # Check for conflict opportunities
        if nyx_context.conflict_synthesizer:
            opportunities = await nyx_context.check_conflict_resolution_opportunities()
            if opportunities:
                next_moment['conflict_opportunities'] = opportunities
        
        # Get NPC suggestions
        if nyx_context.npc_orchestrator:
            npc_suggestions = await nyx_context.npc_orchestrator.get_narrative_context(
                focus_npc_ids=nyx_context.current_scene_npcs,
                include_next_steps=True
            )
            next_moment['npc_suggestions'] = npc_suggestions
        
        return {
            "success": True,
            "next_moment": next_moment,
            "orchestrator_data": {
                "active_npcs": len(nyx_context.current_scene_npcs),
                "active_conflicts": len(nyx_context.active_conflicts),
                "lore_context": bool(nyx_context.current_location_lore)
            }
        }
    except Exception as e:
        logger.error(f"Error managing scenario: {e}")
        return {"success": False, "error": str(e)}

async def manage_relationships(interaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced relationship management with memory integration"""
    nyx_context = None
    
    try:
        user_id = interaction_data.get("user_id")
        conversation_id = interaction_data.get("conversation_id")
        
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        participants = interaction_data.get("participants", [])
        relationship_updates = {}
        
        for i, p1 in enumerate(participants):
            for p2 in participants[i+1:]:
                entity_key = "_".join(sorted([str(p1.get('id', p1)), str(p2.get('id', p2))]))
                
                # Calculate relationship changes
                trust_change = 0.1 if interaction_data.get("outcome") == "success" else -0.05
                bond_change = 0.05 if interaction_data.get("emotional_impact", {}).get("positive", 0) > 0 else 0
                power_change = 0.05 if interaction_data.get("interaction_type") == "training" else 0
                
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
                
                # Store relationship change as memory
                await nyx_context.store_memory(
                    entity_type="relationship",
                    entity_id=hash(entity_key) % 1000000,  # Generate pseudo-ID
                    memory_text=f"Relationship between {p1.get('name', p1)} and {p2.get('name', p2)} changed: trust {trust_change:+.2f}",
                    importance="medium",
                    tags=["relationship", interaction_data.get("interaction_type", "interaction")]
                )
        
        # Learn from the interaction
        for pair, updates in relationship_updates.items():
            await nyx_context.learn_from_interaction(
                action=f"relationship_{interaction_data.get('interaction_type', 'general')}",
                outcome=interaction_data.get("outcome", "unknown"),
                success=updates.get("changes", {}).get("trust", 0) > 0
            )
        
        return {
            "success": True,
            "relationship_updates": relationship_updates,
            "memories_created": len(relationship_updates),
            "analysis": {
                "total_relationships_updated": len(relationship_updates),
                "interaction_type": interaction_data.get("interaction_type"),
                "outcome": interaction_data.get("outcome")
            }
        }
        
    except Exception as e:
        logger.error(f"Error managing relationships: {e}")
        if nyx_context:
            nyx_context.log_error(e, interaction_data)
        return {"success": False, "error": str(e)}

async def store_messages(user_id: int, conversation_id: int, user_input: str, nyx_response: str):
    """Store messages with enhanced metadata"""
    async with get_db_connection_context() as conn:
        # Store user message
        await conn.execute(
            """INSERT INTO messages (conversation_id, sender, content, metadata) 
               VALUES ($1, $2, $3, $4)""",
            conversation_id, "user", user_input, json.dumps({"timestamp": time.time()})
        )
        
        # Store Nyx response with metadata
        await conn.execute(
            """INSERT INTO messages (conversation_id, sender, content, metadata) 
               VALUES ($1, $2, $3, $4)""",
            conversation_id, "Nyx", nyx_response, 
            json.dumps({"timestamp": time.time(), "orchestrators_active": True})
        )

# Preserve remaining functions and exports...
async def create_nyx_agent_with_prompt(system_prompt: str, private_reflection: str = "") -> Agent[NyxContext]:
    """Create agent with preset story awareness"""
    from .nyx_agent.agent_factory import create_nyx_agent_with_prompt as factory_create
    return await factory_create(system_prompt, private_reflection)

async def create_preset_aware_nyx_agent(
    conversation_id: int,
    system_prompt: str, 
    private_reflection: str = ""
) -> Agent[NyxContext]:
    """Create agent with automatic preset detection"""
    from .nyx_agent.agent_factory import create_preset_aware_nyx_agent as factory_create
    return await factory_create(conversation_id, system_prompt, private_reflection)

async def decide_image_generation_standalone(ctx: NyxContext, scene_text: str) -> str:
    """Standalone image generation decision"""
    from .nyx_agent.models import ImageGenerationDecision
    from .nyx_agent.utils import _score_scene_text, _build_image_prompt
    
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

# Re-export everything for backward compatibility
__all__ = [
    'Config',
    'initialize_agents',
    'process_user_input',
    'generate_reflection',
    'generate_contextual_reflection',
    'orchestrate_scene_transition',
    'trigger_emergent_event',
    'manage_scenario',
    'manage_relationships',
    'store_messages',
    'create_nyx_agent_with_prompt',
    'create_preset_aware_nyx_agent',
    'NyxContext',
    'AgentContext',
    'NarrativeResponse',
    'ImageGenerationDecision',
    'get_emotional_state',
    'update_emotional_state',
    'retrieve_memories',
    'add_memory',
    'get_user_model_guidance',
    'detect_user_revelations',
    'generate_image_from_scene',
    'decide_image_generation',
    'decide_image_generation_standalone',
    'calculate_emotional_impact',
    'calculate_and_update_emotional_state',
    'update_relationship_state',
    'check_performance_metrics',
    'get_activity_recommendations',
    'manage_beliefs',
    'score_decision_options',
    'detect_conflicts_and_instability',
    'generate_universal_updates',
    'orchestrate_slice_scene',
    'check_world_state',
    'generate_emergent_event',
    'simulate_npc_autonomy',
    'run_agent_with_error_handling',
    'enhance_context_with_memories',
    'get_available_activities',
    'enhance_context_with_strategies',
    'determine_image_generation',
    'process_user_input_with_openai',
    'process_user_input_standalone',
    'mark_strategy_for_review',
]
