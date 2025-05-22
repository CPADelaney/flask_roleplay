# nyx/core/a2a/context_aware_memory_orchestrator.py

import logging
from typing import Dict, List, Any, Optional
import asyncio

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareMemoryOrchestrator(ContextAwareModule):
    """
    Enhanced MemoryOrchestrator with full context distribution capabilities
    Coordinates memory operations across the system with context awareness
    """
    
    def __init__(self, original_orchestrator):
        super().__init__("memory_orchestrator")
        self.original_orchestrator = original_orchestrator
        self.context_subscriptions = [
            "memory_request", "consolidation_trigger", "reflection_trigger",
            "narrative_request", "memory_health_check", "cross_module_memory_need",
            "experience_synthesis_needed", "memory_pattern_alert"
        ]
        
    async def on_context_received(self, context: SharedContext):
        """Initialize memory orchestration for this context"""
        logger.debug(f"MemoryOrchestrator received context for user: {context.user_id}")
        
        # Determine memory operation needs
        operation_needs = await self._assess_memory_operation_needs(context)
        
        # Check system-wide memory health
        system_health = await self._assess_system_memory_health(context)
        
        # Identify cross-module memory coordination needs
        coordination_needs = await self._identify_coordination_needs(context)
        
        # Send orchestration context
        await self.send_context_update(
            update_type="memory_orchestration_ready",
            data={
                "operation_needs": operation_needs,
                "system_health": system_health,
                "coordination_needs": coordination_needs,
                "orchestration_priority": await self._calculate_orchestration_priority(context)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates requiring memory orchestration"""
        
        if update.update_type == "memory_request":
            # Route memory request to appropriate handler
            await self._route_memory_request(update.data)
            
        elif update.update_type == "consolidation_trigger":
            # Coordinate memory consolidation
            await self._coordinate_consolidation(update.data)
            
        elif update.update_type == "reflection_trigger":
            # Orchestrate reflection creation
            await self._orchestrate_reflection(update.data)
            
        elif update.update_type == "narrative_request":
            # Coordinate narrative construction
            await self._orchestrate_narrative(update.data)
            
        elif update.update_type == "cross_module_memory_need":
            # Handle cross-module memory requests
            await self._handle_cross_module_need(update.data)
            
        elif update.update_type == "memory_pattern_alert":
            # Process detected memory patterns
            await self._process_memory_pattern(update.data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with memory orchestration"""
        # Determine memory operation type
        operation_type = await self._determine_operation_type(context)
        
        # Execute appropriate memory operations
        operation_results = await self._execute_memory_operations(context, operation_type)
        
        # Coordinate with other modules
        coordination_results = await self._coordinate_with_modules(context, operation_results)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Send orchestration update
        await self.send_context_update(
            update_type="memory_operations_complete",
            data={
                "operation_type": operation_type,
                "operation_results": operation_results,
                "coordination_results": coordination_results,
                "cross_module_coordination": len(messages)
            }
        )
        
        return {
            "orchestration_complete": True,
            "operation_type": operation_type,
            "results": operation_results,
            "coordination": coordination_results
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze memory system state and needs"""
        # Comprehensive memory system analysis
        system_analysis = await self._analyze_memory_system(context)
        
        # Identify optimization opportunities
        optimization_opportunities = await self._identify_optimizations(context)
        
        # Analyze cross-module memory usage
        usage_analysis = await self._analyze_cross_module_usage(context)
        
        # Assess memory quality and coverage
        quality_assessment = await self._assess_memory_quality(context)
        
        # Generate maintenance recommendations
        maintenance_recommendations = await self._generate_maintenance_recommendations(
            system_analysis, optimization_opportunities
        )
        
        return {
            "system_analysis": system_analysis,
            "optimization_opportunities": optimization_opportunities,
            "usage_analysis": usage_analysis,
            "quality_assessment": quality_assessment,
            "maintenance_recommendations": maintenance_recommendations
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize memory orchestration for response"""
        messages = await self.get_cross_module_messages()
        
        # Create orchestration synthesis
        orchestration_synthesis = {
            "memory_coordination_status": await self._get_coordination_status(context),
            "recommended_operations": await self._recommend_operations(context, messages),
            "memory_insights": await self._synthesize_memory_insights(context),
            "narrative_opportunities": await self._identify_narrative_opportunities(context),
            "system_recommendations": await self._generate_system_recommendations(context)
        }
        
        # Check for urgent memory operations
        urgent_operations = await self._check_urgent_operations(context, messages)
        if urgent_operations:
            await self.send_context_update(
                update_type="urgent_memory_operations",
                data=urgent_operations,
                priority=ContextPriority.CRITICAL
            )
        
        return orchestration_synthesis
    
    # ========================================================================================
    # ORCHESTRATION-SPECIFIC HELPER METHODS
    # ========================================================================================
    
    async def _assess_memory_operation_needs(self, context: SharedContext) -> Dict[str, Any]:
        """Assess what memory operations are needed"""
        needs = {
            "retrieval_needed": False,
            "storage_needed": False,
            "reflection_needed": False,
            "consolidation_needed": False,
            "narrative_needed": False
        }
        
        # Check for retrieval needs
        if "?" in context.user_input or any(q in context.user_input.lower() for q in ["remember", "recall"]):
            needs["retrieval_needed"] = True
        
        # Check for storage needs
        if len(context.context_updates) > 3:  # Multiple module updates suggest significant event
            needs["storage_needed"] = True
        
        # Check for reflection needs
        if context.processing_stage == "synthesis" and len(context.module_outputs) > 2:
            needs["reflection_needed"] = True
        
        # Check for consolidation needs (based on memory stats)
        memory_stats = await self.original_orchestrator.memory_core.get_memory_stats()
        if memory_stats.get("total_memories", 0) > 100:
            needs["consolidation_needed"] = True
        
        # Check for narrative needs
        if any(word in context.user_input.lower() for word in ["story", "tell me about", "what happened"]):
            needs["narrative_needed"] = True
        
        return needs
    
    async def _route_memory_request(self, request_data: Dict[str, Any]):
        """Route memory requests to appropriate handlers"""
        request_type = request_data.get("type", "retrieve")
        
        if request_type == "retrieve":
            # Use parallel retrieval for efficiency
            results = await self.original_orchestrator.retrieve_memories_parallel(
                query=request_data.get("query", ""),
                memory_types=request_data.get("memory_types"),
                limit_per_type=request_data.get("limit", 3)
            )
            
            # Send results back
            await self.send_context_update(
                update_type="memory_retrieval_results",
                data={"memories": results, "request_id": request_data.get("request_id")},
                target_modules=[request_data.get("source_module", "reasoning_core")]
            )
            
        elif request_type == "store":
            # Create new memory
            memory_id = await self.original_orchestrator.create_memory(
                memory_text=request_data.get("text", ""),
                memory_type=request_data.get("memory_type", "observation"),
                tags=request_data.get("tags", []),
                significance=request_data.get("significance", 5)
            )
            
            # Notify completion
            await self.send_context_update(
                update_type="memory_storage_complete",
                data={"memory_id": memory_id, "request_id": request_data.get("request_id")}
            )
    
    async def _coordinate_consolidation(self, trigger_data: Dict[str, Any]):
        """Coordinate memory consolidation across modules"""
        # Notify modules of impending consolidation
        await self.send_context_update(
            update_type="consolidation_starting",
            data={"reason": trigger_data.get("reason", "scheduled")},
            scope=ContextScope.GLOBAL
        )
        
        # Run consolidation
        consolidation_result = await self.original_orchestrator.memory_core.consolidate_memory_clusters()
        
        # Run decay
        decay_result = await self.original_orchestrator.memory_core.apply_memory_decay()
        
        # Notify completion
        await self.send_context_update(
            update_type="consolidation_complete",
            data={
                "clusters_consolidated": consolidation_result.get("clusters_consolidated", 0),
                "memories_decayed": decay_result.get("memories_decayed", 0),
                "memories_archived": decay_result.get("memories_archived", 0)
            },
            scope=ContextScope.GLOBAL
        )
    
    async def _orchestrate_reflection(self, trigger_data: Dict[str, Any]):
        """Orchestrate reflection creation with cross-module input"""
        topic = trigger_data.get("topic")
        
        # Gather context from other modules
        await self.send_context_update(
            update_type="reflection_context_request",
            data={"topic": topic},
            scope=ContextScope.GLOBAL
        )
        
        # Wait for responses (simplified - in production use proper async coordination)
        await asyncio.sleep(0.5)
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Create reflection with context
        reflection_result = await self.original_orchestrator.create_reflection(topic=topic)
        
        # Enhance reflection with cross-module insights
        if messages:
            enhanced_reflection = await self._enhance_reflection_with_context(
                reflection_result, messages
            )
            reflection_result = enhanced_reflection
        
        # Send reflection result
        await self.send_context_update(
            update_type="reflection_created",
            data=reflection_result,
            scope=ContextScope.GLOBAL
        )
    
    async def _orchestrate_narrative(self, request_data: Dict[str, Any]):
        """Orchestrate narrative construction"""
        topic = request_data.get("topic", "")
        chronological = request_data.get("chronological", True)
        
        # Retrieve relevant memories with prioritization
        prioritized_memories = await self.original_orchestrator.retrieve_memories_with_prioritization(
            query=topic,
            memory_types=["experience", "observation", "reflection"],
            prioritization={
                "experience": 0.4,
                "observation": 0.3,
                "reflection": 0.3
            },
            limit=10
        )
        
        # Construct narrative
        narrative_result = await self.original_orchestrator.memory_core.construct_narrative(
            topic=topic,
            chronological=chronological,
            limit=len(prioritized_memories)
        )
        
        # Send narrative
        await self.send_context_update(
            update_type="narrative_constructed",
            data=narrative_result,
            target_modules=[request_data.get("source_module", "response_generator")]
        )
    
    async def _execute_memory_operations(self, context: SharedContext, operation_type: str) -> Dict[str, Any]:
        """Execute determined memory operations"""
        results = {}
        
        if operation_type == "retrieve":
            # Retrieve with context awareness
            memories = await self.original_orchestrator.retrieve_memories(
                query=context.user_input,
                memory_types=None,  # All types
                limit=5
            )
            results["retrieved_memories"] = memories
            
        elif operation_type == "store":
            # Create contextual memory
            memory_id = await self.original_orchestrator.create_memory(
                memory_text=context.user_input,
                memory_type="observation",
                significance=5
            )
            results["created_memory"] = memory_id
            
        elif operation_type == "reflect":
            # Create reflection
            reflection = await self.original_orchestrator.create_reflection()
            results["reflection"] = reflection
            
        elif operation_type == "maintain":
            # Run maintenance
            maintenance = await self.original_orchestrator.run_maintenance()
            results["maintenance"] = maintenance
            
        elif operation_type == "complex":
            # Complex operation involving multiple steps
            results = await self._execute_complex_operation(context)
        
        return results
    
    async def _coordinate_with_modules(self, context: SharedContext, operation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate memory results with other modules"""
        coordination = {}
        
        # Share retrieved memories with relevant modules
        if "retrieved_memories" in operation_results:
            await self.send_context_update(
                update_type="memories_available",
                data={
                    "memories": operation_results["retrieved_memories"],
                    "context_id": context.conversation_id
                },
                scope=ContextScope.GLOBAL
            )
            coordination["shared_memories"] = len(operation_results["retrieved_memories"])
        
        # Notify about new memories
        if "created_memory" in operation_results:
            await self.send_context_update(
                update_type="new_memory_created",
                data={
                    "memory_id": operation_results["created_memory"],
                    "context_id": context.conversation_id
                },
                scope=ContextScope.GLOBAL
            )
            coordination["notified_creation"] = True
        
        return coordination
    
    async def _check_urgent_operations(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Check for urgent memory operations needed"""
        urgent_ops = []
        
        # Check memory health
        stats = await self.original_orchestrator.memory_core.get_memory_stats()
        
        # High memory count needs consolidation
        if stats.get("total_memories", 0) > 1000:
            urgent_ops.append({
                "operation": "consolidation",
                "reason": "high_memory_count",
                "urgency": 0.8
            })
        
        # Low average fidelity needs quality check
        if stats.get("avg_fidelity", 1.0) < 0.5:
            urgent_ops.append({
                "operation": "quality_check",
                "reason": "low_fidelity",
                "urgency": 0.7
            })
        
        # Check for module requests
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg.get("type") == "urgent_memory_need":
                    urgent_ops.append({
                        "operation": "retrieve",
                        "reason": f"urgent_request_from_{module_name}",
                        "urgency": 0.9,
                        "details": msg.get("data", {})
                    })
        
        if urgent_ops:
            # Sort by urgency
            urgent_ops.sort(key=lambda x: x["urgency"], reverse=True)
            return {
                "operations": urgent_ops,
                "most_urgent": urgent_ops[0]
            }
        
        return None
    
    async def _generate_system_recommendations(self, context: SharedContext) -> List[str]:
        """Generate recommendations for memory system optimization"""
        recommendations = []
        
        # Get current stats
        stats = await self.original_orchestrator.memory_core.get_memory_stats()
        
        # Memory distribution recommendations
        type_counts = stats.get("type_counts", {})
        if type_counts.get("observation", 0) > type_counts.get("reflection", 0) * 10:
            recommendations.append("Consider creating more reflections to balance memory types")
        
        # Consolidation recommendations
        if stats.get("total_memories", 0) > 500 and stats.get("consolidated_count", 0) < 50:
            recommendations.append("Run memory consolidation to optimize storage")
        
        # Quality recommendations
        if stats.get("avg_fidelity", 1.0) < 0.6:
            recommendations.append("Review and potentially crystallize important low-fidelity memories")
        
        # Schema recommendations
        if stats.get("total_schemas", 0) < 5 and stats.get("total_memories", 0) > 100:
            recommendations.append("Analyze memories for pattern detection and schema creation")
        
        return recommendations
    
    # ========================================================================================
    # UTILITY METHODS
    # ========================================================================================
    
    async def _determine_operation_type(self, context: SharedContext) -> str:
        """Determine the type of memory operation needed"""
        user_input_lower = context.user_input.lower()
        
        # Check for explicit memory operations
        if any(word in user_input_lower for word in ["remember", "recall", "what was"]):
            return "retrieve"
        elif any(word in user_input_lower for word in ["note", "store", "save"]):
            return "store"
        elif any(word in user_input_lower for word in ["reflect", "think about"]):
            return "reflect"
        elif any(word in user_input_lower for word in ["maintain", "optimize", "clean"]):
            return "maintain"
        
        # Check context for implicit operations
        if len(context.context_updates) > 5:
            return "complex"  # Multiple updates suggest complex operation
        
        # Default to retrieval
        return "retrieve"
    
    async def _enhance_reflection_with_context(self, reflection: Dict[str, Any], messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Enhance reflection with cross-module context"""
        enhanced = reflection.copy()
        
        # Add emotional context
        for msg in messages.get("emotional_core", []):
            if msg["type"] == "emotional_state_update":
                enhanced["emotional_context"] = msg["data"]
                break
        
        # Add goal context
        for msg in messages.get("goal_manager", []):
            if msg["type"] == "goal_context_available":
                enhanced["goal_alignment"] = msg["data"].get("active_goals", [])
                break
        
        # Add knowledge context
        for msg in messages.get("knowledge_core", []):
            if msg["type"] == "knowledge_context_available":
                enhanced["knowledge_gaps"] = msg["data"].get("knowledge_gaps", [])
                break
        
        return enhanced
    
    async def _execute_complex_operation(self, context: SharedContext) -> Dict[str, Any]:
        """Execute complex multi-step memory operation"""
        results = {}
        
        # Step 1: Retrieve relevant memories
        memories = await self.original_orchestrator.retrieve_memories_parallel(
            query=context.user_input,
            limit_per_type=3
        )
        results["retrieved"] = memories
        
        # Step 2: Check for patterns
        all_memories = []
        for mem_type, mems in memories.items():
            all_memories.extend(mems)
        
        if len(all_memories) > 5:
            # Create abstraction from patterns
            abstraction = await self.original_orchestrator.memory_core.create_abstraction_from_memories(
                memory_ids=[m["id"] for m in all_memories[:5]],
                pattern_type="behavioral"
            )
            results["abstraction"] = abstraction
        
        # Step 3: Create new memory if significant
        if context.emotional_state and max(context.emotional_state.values()) > 0.7:
            memory_id = await self.original_orchestrator.create_memory(
                memory_text=f"Complex interaction: {context.user_input[:100]}",
                memory_type="experience",
                significance=7
            )
            results["created"] = memory_id
        
        return results
    
    # Delegate to original orchestrator
    def __getattr__(self, name):
        """Delegate any missing methods to the original orchestrator"""
        return getattr(self.original_orchestrator, name)
