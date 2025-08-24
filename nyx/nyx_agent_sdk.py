# nyx/nyx_agent_sdk.py
"""
Nyx Agent SDK - Refactored for Scene-Scoped Context Assembly
Main changes:
- Parallel orchestrator fetching via scene bundles
- Context persistence and caching
- Background task offloading
- ContextBundle-based agent invocation
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .context import NyxContext, SceneScope, ContextBundle, PackedContext
from .models import NyxResponse, ContextMetrics
from .assembly import assemble_nyx_response
from .prompts import get_system_prompt
from .config import NyxConfig
from ..utils.performance import log_performance_metrics
from ..utils.background import enqueue_task  # Background task queue

logger = logging.getLogger(__name__)


class NyxAgentSDK:
    """
    Refactored SDK with scene-scoped context assembly and performance optimizations.
    """
    
    def __init__(self, config: NyxConfig):
        self.config = config
        self.agent = None  # Lazy load
        self.persistent_contexts: Dict[str, NyxContext] = {}
        
    async def initialize_agent(self):
        """Lazy initialize the agent when first needed."""
        if self.agent is None:
            from .agent import create_nyx_agent
            self.agent = await create_nyx_agent(self.config)
    
    async def process_user_input(
        self,
        message: str,
        conversation_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NyxResponse:
        """
        Main entry point - now with scene-scoped parallel context assembly.
        """
        start_time = time.time()
        
        try:
            # Initialize agent if needed
            await self.initialize_agent()
            
            # Get or create persistent context for this conversation
            context = await self._get_or_create_context(
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # PHASE 1: Scene-Scoped Context Assembly (Parallel)
            logger.info(f"Computing scene scope for input: {message[:100]}...")
            
            # Compute what's relevant for this turn
            scene_scope = await context.context_broker.compute_scene_scope(
                user_input=message,
                current_state=context.current_context  # Use current_context, not get_current_state()
            )
            
            # Log scope for debugging
            logger.debug(f"Scene scope: {scene_scope}")
            
            # PHASE 2: Fetch or Load Context Bundle (Cached + Parallel)
            # ContextBroker handles all caching/staleness internally
            bundle = await context.context_broker.load_or_fetch_bundle(
                scene_scope=scene_scope
            )
            
            # PHASE 3: Pack Context for LLM (Canon-First, Token-Aware)
            packed_context = await self._pack_context_for_agent(
                bundle=bundle,
                message=message,
                token_budget=self.config.max_context_tokens
            )
            
            # PHASE 4: Invoke Agent with Packed Context
            agent_response = await self._invoke_agent(
                message=message,
                packed_context=packed_context,
                conversation_id=conversation_id,
                context=context  # Pass for tool access
            )
            
            # PHASE 5: Assemble Final Response
            nyx_response = await self._assemble_response(
                agent_response=agent_response,
                bundle=bundle,
                context=context,
                user_input=message
            )
            
            # PHASE 6: Background Tasks (Non-Blocking)
            await self._schedule_background_maintenance(
                context=context,
                bundle=bundle,
                response=nyx_response,
                conversation_id=conversation_id
            )
            
            # Performance metrics
            elapsed = time.time() - start_time
            logger.info(f"Response generated in {elapsed:.2f}s")
            
            await self._log_metrics(
                elapsed=elapsed,
                bundle_size=(
                    len(packed_context.canonical)
                    + len(packed_context.optional)
                    + len(packed_context.summarized)
                )
            )
            
            return nyx_response
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
            return self._create_error_response(str(e))
    
    async def _get_or_create_context(
        self,
        conversation_id: str,
        user_id: str
    ) -> NyxContext:
        """
        Retrieve persistent context or create new one.
        Key optimization: Reuse context across turns instead of recreating.
        """
        if conversation_id in self.persistent_contexts:
            context = self.persistent_contexts[conversation_id]
            logger.debug(f"Reusing persistent context for {conversation_id}")
        else:
            # First time - full initialization
            context = NyxContext(
                conversation_id=int(conversation_id),
                user_id=int(user_id)
            )
            await context.initialize()
            self.persistent_contexts[conversation_id] = context
            logger.info(f"Created new context for {conversation_id}")
        
        return context
    

    
    async def _pack_context_for_agent(
        self,
        bundle: ContextBundle,
        message: str,
        token_budget: int
    ) -> PackedContext:
        """
        Pack the context bundle for the LLM with canon-first priority.
        """
        # Use bundle's pack method with token budget (sync)
        packed = bundle.pack(token_budget=token_budget)
        
        # Add the user input as canonical
        packed.add_canonical('user_input', message)
        
        # Add any additional critical context
        world_dict = bundle.world.data if isinstance(bundle.world.data, dict) else {}
        npcs_dict = (
            bundle.npcs.data.to_dict() if hasattr(bundle.npcs.data, "to_dict") else bundle.npcs.data
        ) or {}
        packed.add_canonical('scene_state', {
            "time": world_dict.get("time"),
            "mood": world_dict.get("mood"),
            "weather": world_dict.get("weather"),
            "events": world_dict.get("events", []),
            "active_npcs": sorted(list(bundle.scene_scope.npc_ids)) if bundle.scene_scope else [],
        })
        
        logger.debug(f"Packed context with canonical/optional/summarized sections")
        return packed
    
    async def _invoke_agent(
        self,
        message: str,
        packed_context: PackedContext,
        conversation_id: str,
        context: NyxContext
    ) -> Dict[str, Any]:
        """
        Invoke the main agent with packed context.
        """
        # Build system prompt with context
        system_prompt = self._build_system_prompt(packed_context)
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        
        # Configure agent with tools that can fetch more context on-demand
        config = RunnableConfig(
            configurable={
                "conversation_id": conversation_id,
                "context_broker": context.context_broker,  # For on-demand expansion
                "packed_context": packed_context
            },
            callbacks=[],
            metadata={"sdk_version": "2.0"}
        )
        
        # Invoke agent
        logger.info("Invoking Nyx agent...")
        result = await self.agent.ainvoke(
            {"messages": messages},
            config=config
        )
        
        return result
    
    def _build_system_prompt(self, packed_context: PackedContext) -> str:
        """
        Build the system prompt with packed context sections.
        """
        base_prompt = get_system_prompt()
        
        # Format context compactly for the prompt
        import json
        context_str = json.dumps(packed_context.to_dict(), ensure_ascii=False)
        
        full_prompt = f"{base_prompt}\n\n{context_str}"
        return full_prompt
    
    async def _assemble_response(
        self,
        agent_response: Dict[str, Any],
        bundle: ContextBundle,
        context: NyxContext,
        user_input: str
    ) -> NyxResponse:
        """
        Assemble the final response with all enrichments.
        """
        # Use existing assembler with bundle instead of dict
        response = await assemble_nyx_response(
            agent_output=agent_response,
            context_bundle=bundle,
            context=context,
            user_input=user_input
        )
        
        # Add performance metadata
        response.metadata["scene_key"] = bundle.scene_scope.to_key() if hasattr(bundle, 'scene_scope') else None
        
        return response
    
    async def _schedule_background_maintenance(
        self,
        context: NyxContext,
        bundle: ContextBundle,
        response: NyxResponse,
        conversation_id: str
    ):
        """
        Schedule heavy maintenance tasks to run after response is sent.
        These don't block the user response!
        """
        maintenance_tasks = []
        
        # Memory consolidation and pattern analysis
        if getattr(context, "memory_orchestrator", None) and hasattr(context.memory_orchestrator, "needs_consolidation") and context.memory_orchestrator.needs_consolidation():
            maintenance_tasks.append({
                "task": "memory.consolidate",
                "params": {
                    "conversation_id": conversation_id,
                    "recent_memories": (
                        (bundle.memories.data.to_dict() if hasattr(bundle.memories.data, "to_dict") else bundle.memories.data) or {}
                    ).get("recent", [])
                }
            })
        
        # NPC background thinking/scheming
        active_npcs = sorted(list(bundle.scene_scope.npc_ids)) if bundle.scene_scope else []
        for npc_id in active_npcs:
            maintenance_tasks.append({
                "task": "npc.background_think",
                "params": {
                    "npc_id": npc_id,
                    # Workers can refetch full NPC context by id; keep payload tiny.
                    "context": {}
                }
            })
        
        # Lore evolution checks
        lore_dict = (
            bundle.lore.data.to_dict() if hasattr(bundle.lore.data, "to_dict") else bundle.lore.data
        ) or {}
        if lore_dict.get("evolution_pending"):
            maintenance_tasks.append({
                "task": "lore.evolve",
                "params": {
                    "affected_entities": lore_dict.get("affected_entities", [])
                }
            })
        
        # Conflict tension updates
        conflicts_dict = bundle.conflicts.data or {}
        if context.should_run_task("conflict_tension_calculation"):
            active_ids = [c["id"] for c in conflicts_dict.get("active", []) if isinstance(c, dict) and "id" in c]
            maintenance_tasks.append({
                "task": "conflict.update_tensions",
                "params": {"active_conflicts": active_ids}
            })
        
        # Universal state updates
        maintenance_tasks.append({
            "task": "world.update_universal",
            "params": {
                "response": response.to_dict(),
                "conversation_id": conversation_id
            }
        })
        
        # Enqueue all tasks (non-blocking)
        for task in maintenance_tasks:
            await enqueue_task(
                task_name=task["task"],
                params=task["params"],
                priority="low",
                delay_seconds=1  # Small delay to ensure response is sent first
            )
        
        logger.info(f"Scheduled {len(maintenance_tasks)} background maintenance tasks")
    
    async def _log_metrics(
        self,
        elapsed: float,
        bundle_size: int
    ):
        """Log performance metrics for monitoring."""
        metrics = ContextMetrics(
            total_time=elapsed,
            bundle_sections=bundle_size,
            timestamp=datetime.utcnow()
        )
        
        await log_performance_metrics(metrics)
        
        # Alert if response time exceeds threshold
        if elapsed > 30:
            logger.warning(f"Response time ({elapsed:.2f}s) exceeded 30s threshold")
        elif elapsed < 10:
            logger.info(f"🚀 Fast response achieved: {elapsed:.2f}s")
    
    def _create_error_response(self, error_message: str) -> NyxResponse:
        """Create a fallback error response."""
        return NyxResponse(
            narrative="*Nyx's form flickers momentarily* Something interfered with our connection...",
            choices=[],
            metadata={"error": error_message},
            world_state={},
            success=False
        )
    
    async def cleanup_conversation(self, conversation_id: str):
        """
        Clean up persistent context when conversation ends.
        """
        if conversation_id in self.persistent_contexts:
            context = self.persistent_contexts[conversation_id]
            if hasattr(context, "cleanup") and callable(getattr(context, "cleanup")):
                await context.cleanup()
            del self.persistent_contexts[conversation_id]
            logger.info(f"Cleaned up context for {conversation_id}")
    
    async def warmup_cache(self, conversation_id: str, location: str):
        """
        Pre-warm caches for a location to make first response faster.
        """
        # Create a temporary context for warmup
        context = NyxContext(
            conversation_id=int(conversation_id),
            user_id=0  # Use 0 for warmup context
        )
        await context.initialize()
        
        # Pre-fetch common scene scopes
        base_scope = SceneScope(
            location_id=location,
            npc_ids=set(),
            topics=set(),
            lore_tags=set()
        )
        
        # Fetch and cache (ContextBroker handles caching internally)
        await context.context_broker.load_or_fetch_bundle(scene_scope=base_scope)
        
        logger.info(f"Cache warmed for location: {location}")


# Backward compatibility wrapper
class NyxAgentRunner:
    """Legacy interface wrapper for backward compatibility."""
    
    def __init__(self, config: Optional[NyxConfig] = None):
        self.sdk = NyxAgentSDK(config or NyxConfig())
    
    async def run(
        self,
        user_input: str,
        conversation_id: str,
        user_id: str,
        **kwargs
    ) -> NyxResponse:
        """Legacy run method."""
        return await self.sdk.process_user_input(
            message=user_input,
            conversation_id=conversation_id,
            user_id=user_id,
            metadata=kwargs
        )
    
    async def initialize(self):
        """Legacy initialization."""
        await self.sdk.initialize_agent()


# Module exports
__all__ = [
    'NyxAgentSDK',
    'NyxAgentRunner',  # For backward compatibility
]
