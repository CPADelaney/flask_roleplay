# nyx/core/a2a/context_aware_streaming_integration.py

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareStreamingIntegration(ContextAwareModule):
    """
    Advanced StreamingIntegration that coordinates all streaming-related modules
    with full context distribution capabilities
    """
    
    def __init__(self, brain, video_source=0, audio_source=None):
        super().__init__("streaming_integration")
        self.brain = brain
        self.video_source = video_source
        self.audio_source = audio_source
        
        # This will hold references to all streaming components
        self.streaming_core = None
        self.hormone_system = None
        self.reflection_engine = None
        self.cross_game_knowledge = None
        self.learning_manager = None
        
        # Track integration state
        self.integration_state = {
            "initialized": False,
            "components_loaded": {},
            "active_integrations": [],
            "performance_mode": "balanced"  # balanced, performance, quality
        }
        
        self.context_subscriptions = [
            "module_initialization", "streaming_request",
            "integration_request", "performance_adjustment",
            "cross_system_event", "learning_opportunity"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize streaming integration for this context"""
        logger.debug(f"StreamingIntegration received context")
        
        # Check if streaming setup is needed
        if await self._needs_streaming_setup(context):
            await self._setup_streaming_components()
        
        # Send integration status
        await self.send_context_update(
            update_type="streaming_integration_status",
            data={
                "initialized": self.integration_state["initialized"],
                "components": list(self.integration_state["components_loaded"].keys()),
                "ready": self._is_ready_for_streaming()
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that affect streaming integration"""
        
        if update.update_type == "module_initialization":
            # Track module initialization
            module_data = update.data
            module_name = module_data.get("module_name")
            if module_name in ["streaming_core", "hormone_system", "reflection_engine"]:
                self.integration_state["components_loaded"][module_name] = True
        
        elif update.update_type == "streaming_request":
            # Handle streaming requests
            request_data = update.data
            await self._handle_streaming_request(request_data)
        
        elif update.update_type == "integration_request":
            # Handle integration requests between components
            integration_data = update.data
            await self._handle_integration_request(integration_data)
        
        elif update.update_type == "performance_adjustment":
            # Adjust performance settings
            performance_data = update.data
            await self._adjust_performance_settings(performance_data)
        
        elif update.update_type == "cross_system_event":
            # Handle events that span multiple systems
            event_data = update.data
            await self._handle_cross_system_event(event_data)
        
        elif update.update_type == "learning_opportunity":
            # Process learning opportunities
            learning_data = update.data
            await self._process_learning_opportunity(learning_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with streaming integration awareness"""
        # Check for integration commands
        integration_command = await self._parse_integration_command(context.user_input)
        
        result = {}
        if integration_command:
            result = await self._execute_integration_command(integration_command, context)
        
        # Coordinate streaming components if active
        if self._is_streaming_active():
            coordination_result = await self._coordinate_streaming_components(context)
            result["coordination"] = coordination_result
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        integration_effects = await self._process_integration_messages(messages)
        
        return {
            "integration_processed": True,
            "command": integration_command,
            "result": result,
            "cross_module_effects": integration_effects,
            "streaming_active": self._is_streaming_active()
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze streaming integration state"""
        # Analyze component health
        component_health = await self._analyze_component_health()
        
        # Analyze integration effectiveness
        integration_effectiveness = await self._analyze_integration_effectiveness()
        
        # Analyze performance metrics
        performance_analysis = await self._analyze_performance_metrics()
        
        # Analyze learning progress
        learning_analysis = await self._analyze_learning_progress()
        
        return {
            "component_health": component_health,
            "integration_effectiveness": integration_effectiveness,
            "performance_analysis": performance_analysis,
            "learning_analysis": learning_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize streaming integration for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate integration synthesis
        integration_synthesis = {
            "streaming_readiness": self._assess_streaming_readiness(),
            "component_coordination": await self._synthesize_component_coordination(context),
            "performance_recommendations": await self._generate_performance_recommendations(),
            "integration_insights": await self._generate_integration_insights(context, messages),
            "next_steps": await self._suggest_next_steps(context)
        }
        
        # Send integration synthesis update
        await self.send_context_update(
            update_type="integration_synthesis_complete",
            data=integration_synthesis,
            priority=ContextPriority.NORMAL
        )
        
        return {
            "integration_synthesis": integration_synthesis,
            "synthesis_complete": True
        }
    
    # ========================================================================================
    # INTEGRATION-SPECIFIC METHODS
    # ========================================================================================
    
    async def _needs_streaming_setup(self, context: SharedContext) -> bool:
        """Determine if streaming setup is needed"""
        # Check if components are initialized
        if not self.integration_state["initialized"]:
            return True
        
        # Check if user is requesting streaming
        if "stream" in context.user_input.lower():
            return True
        
        # Check if modules are requesting setup
        if context.session_context.get("streaming_setup_requested"):
            return True
        
        return False
    
    async def _setup_streaming_components(self):
        """Set up all streaming components with context awareness"""
        try:
            # Import context-aware versions
            from nyx.core.a2a.context_aware_streaming_core import ContextAwareStreamingCore
            from nyx.core.a2a.context_aware_streaming_hormone_system import ContextAwareStreamingHormoneSystem
            from nyx.core.a2a.context_aware_streaming_reflection import ContextAwareStreamingReflectionEngine
            
            # Create base streaming core
            from nyx.streamer.nyx_streaming_core import StreamingCore
            base_streaming_core = StreamingCore(self.brain, self.video_source, self.audio_source)
            
            # Wrap with context-aware version
            self.streaming_core = ContextAwareStreamingCore(base_streaming_core)
            
            # Create hormone system
            from nyx.streamer.streaming_hormone_system import StreamingHormoneSystem
            base_hormone_system = StreamingHormoneSystem(self.brain)
            self.hormone_system = ContextAwareStreamingHormoneSystem(base_hormone_system)
            
            # Create reflection engine
            from nyx.streamer.streaming_reflection import StreamingReflectionEngine
            base_reflection_engine = StreamingReflectionEngine(self.brain, base_streaming_core)
            self.reflection_engine = ContextAwareStreamingReflectionEngine(base_reflection_engine)
            
            # Make components available to each other
            self.streaming_core.hormone_system = self.hormone_system
            self.streaming_core.reflection_engine = self.reflection_engine
            
            # Set context systems
            for component in [self.streaming_core, self.hormone_system, self.reflection_engine]:
                component.set_context_system(self._context_system)
            
            # Update state
            self.integration_state["initialized"] = True
            self.integration_state["components_loaded"] = {
                "streaming_core": True,
                "hormone_system": True,
                "reflection_engine": True
            }
            
            # Register functions with brain
            self._register_brain_functions()
            
            logger.info("Streaming components initialized with context awareness")
            
        except Exception as e:
            logger.error(f"Error setting up streaming components: {e}")
            self.integration_state["initialized"] = False
    
    def _is_ready_for_streaming(self) -> bool:
        """Check if all components are ready for streaming"""
        required_components = ["streaming_core", "hormone_system", "reflection_engine"]
        
        for component in required_components:
            if not self.integration_state["components_loaded"].get(component, False):
                return False
        
        return self.integration_state["initialized"]
    
    async def _handle_streaming_request(self, request_data: Dict[str, Any]):
        """Handle streaming requests"""
        request_type = request_data.get("type")
        
        if request_type == "start" and self._is_ready_for_streaming():
            # Start streaming with all components
            result = await self.streaming_core.start_streaming()
            
            # Notify all components
            await self.send_context_update(
                update_type="streaming_started",
                data=result,
                priority=ContextPriority.HIGH
            )
        
        elif request_type == "stop" and self._is_streaming_active():
            # Stop streaming
            result = await self.streaming_core.stop_streaming()
            
            # Generate final reflection
            if self.reflection_engine:
                reflection = await self.reflection_engine.run_periodic_reflection(force=True)
            
            # Notify all components
            await self.send_context_update(
                update_type="streaming_stopped",
                data={**result, "final_reflection": reflection},
                priority=ContextPriority.HIGH
            )
    
    async def _handle_integration_request(self, integration_data: Dict[str, Any]):
        """Handle integration requests between components"""
        source = integration_data.get("source")
        target = integration_data.get("target")
        action = integration_data.get("action")
        
        # Route integration requests
        if source == "streaming_core" and target == "hormone_system":
            if action == "sync_hormones":
                await self.hormone_system.sync_with_brain_hormone_system()
        
        elif source == "reflection_engine" and target == "streaming_core":
            if action == "consolidate_experiences":
                result = await self.reflection_engine.consolidate_streaming_experiences()
                await self.streaming_core.send_context_update(
                    update_type="experiences_consolidated",
                    data=result
                )
    
    async def _adjust_performance_settings(self, performance_data: Dict[str, Any]):
        """Adjust performance settings across components"""
        mode = performance_data.get("mode", "balanced")
        self.integration_state["performance_mode"] = mode
        
        # Adjust streaming core settings
        if hasattr(self.streaming_core, "original_core"):
            if mode == "performance":
                self.streaming_core.original_core.skip_frames = 3
                self.streaming_core.original_core.parallel_processing = True
            elif mode == "quality":
                self.streaming_core.original_core.skip_frames = 1
                self.streaming_core.original_core.parallel_processing = True
            else:  # balanced
                self.streaming_core.original_core.skip_frames = 2
                self.streaming_core.original_core.parallel_processing = True
    
    async def _handle_cross_system_event(self, event_data: Dict[str, Any]):
        """Handle events that affect multiple streaming systems"""
        event_type = event_data.get("type")
        
        if event_type == "significant_moment":
            # Process through all systems
            if self.streaming_core:
                await self.streaming_core.process_significant_moment(
                    game_name=event_data.get("game_name", "Unknown"),
                    event_type=event_data.get("event_subtype", "moment"),
                    event_data=event_data.get("data", {}),
                    significance=event_data.get("significance", 7.0)
                )
            
            if self.hormone_system:
                await self.hormone_system.update_from_event(
                    game_name=event_data.get("game_name", "Unknown"),
                    event_type=event_data.get("event_subtype", "moment"),
                    event_description=event_data.get("description", ""),
                    event_intensity=event_data.get("intensity", 0.7)
                )
    
    async def _process_learning_opportunity(self, learning_data: Dict[str, Any]):
        """Process learning opportunities from streaming"""
        if hasattr(self, "learning_manager") and self.learning_manager:
            await self.learning_manager.process_learning_opportunity(learning_data)
        else:
            # Store for later processing
            self.integration_state.setdefault("pending_learning", []).append(learning_data)
    
    async def _parse_integration_command(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Parse integration commands from user input"""
        input_lower = user_input.lower()
        
        if "setup streaming" in input_lower:
            return {"action": "setup_streaming"}
        elif "streaming performance" in input_lower:
            if "high" in input_lower or "best" in input_lower:
                return {"action": "set_performance", "mode": "quality"}
            elif "fast" in input_lower or "quick" in input_lower:
                return {"action": "set_performance", "mode": "performance"}
        elif "integrate" in input_lower and "streaming" in input_lower:
            return {"action": "check_integration"}
        
        return None
    
    async def _execute_integration_command(self, command: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Execute integration commands"""
        action = command.get("action")
        
        if action == "setup_streaming":
            await self._setup_streaming_components()
            return {"status": "setup_complete", "ready": self._is_ready_for_streaming()}
        
        elif action == "set_performance":
            mode = command.get("mode", "balanced")
            await self._adjust_performance_settings({"mode": mode})
            return {"status": "performance_adjusted", "mode": mode}
        
        elif action == "check_integration":
            return {
                "status": "integration_checked",
                "components": self.integration_state["components_loaded"],
                "active": self._is_streaming_active()
            }
        
        return {"error": "Unknown integration command"}
    
    def _is_streaming_active(self) -> bool:
        """Check if streaming is currently active"""
        if self.streaming_core and hasattr(self.streaming_core, "is_streaming"):
            return self.streaming_core.is_streaming()
        return False
    
    async def _coordinate_streaming_components(self, context: SharedContext) -> Dict[str, Any]:
        """Coordinate all streaming components"""
        coordination_results = {}
        
        # Get current game state
        if self.streaming_core and hasattr(self.streaming_core.original_core, "streaming_system"):
            game_state = self.streaming_core.original_core.streaming_system.game_state
            
            # Share game state with all components
            game_context = {
                "game_name": game_state.game_name,
                "frame_count": game_state.frame_count,
                "recent_events": list(game_state.recent_events)[-5:]
            }
            
            # Update hormone system with game context
            if self.hormone_system:
                hormone_update = self.hormone_system.update_environmental_factors({
                    "game_pacing": 0.7 if game_state.detected_action else 0.3
                })
                coordination_results["hormone_sync"] = hormone_update
            
            # Check for reflection opportunities
            if self.reflection_engine and game_state.frame_count % 1800 == 0:  # Every minute
                reflection_check = await self.reflection_engine._should_generate_deep_reflection(context)
                coordination_results["reflection_due"] = reflection_check
        
        return coordination_results
    
    async def _process_integration_messages(self, messages: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Process integration-related messages from other modules"""
        effects = []
        
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg["type"] == "request_streaming_setup":
                    await self._setup_streaming_components()
                    effects.append({
                        "module": module_name,
                        "effect": "streaming_setup_triggered"
                    })
                elif msg["type"] == "performance_issue":
                    await self._adjust_performance_settings({"mode": "performance"})
                    effects.append({
                        "module": module_name,
                        "effect": "performance_adjusted"
                    })
        
        return effects
    
    async def _analyze_component_health(self) -> Dict[str, Any]:
        """Analyze health of streaming components"""
        health = {}
        
        # Check streaming core
        if self.streaming_core:
            health["streaming_core"] = {
                "active": self._is_streaming_active(),
                "initialized": True
            }
        else:
            health["streaming_core"] = {"active": False, "initialized": False}
        
        # Check hormone system
        if self.hormone_system:
            emotional_state = self.hormone_system.get_emotional_state()
            health["hormone_system"] = {
                "active": True,
                "stability": emotional_state.get("stability", 0.5)
            }
        else:
            health["hormone_system"] = {"active": False}
        
        # Check reflection engine
        if self.reflection_engine:
            health["reflection_engine"] = {
                "active": True,
                "pending_reflections": len(self.reflection_engine.reflection_context["pending_reflections"])
            }
        else:
            health["reflection_engine"] = {"active": False}
        
        return health
    
    async def _analyze_integration_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effectively components are integrated"""
        effectiveness = {
            "component_communication": 0.0,
            "data_flow": 0.0,
            "coordination_quality": 0.0
        }
        
        # Check if components are communicating
        if self.streaming_core and self.hormone_system:
            effectiveness["component_communication"] += 0.33
        if self.streaming_core and self.reflection_engine:
            effectiveness["component_communication"] += 0.33
        if self.hormone_system and self.reflection_engine:
            effectiveness["component_communication"] += 0.34
        
        # Check data flow
        if self._is_streaming_active():
            effectiveness["data_flow"] = 0.8
        
        # Check coordination quality
        if self.integration_state["initialized"]:
            effectiveness["coordination_quality"] = 0.9
        
        return effectiveness
    
    async def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze streaming performance metrics"""
        if self.streaming_core and hasattr(self.streaming_core, "get_performance_metrics"):
            return self.streaming_core.get_performance_metrics()
        
        return {
            "fps": 0,
            "resource_usage": {"cpu": 0, "memory": 0},
            "performance_mode": self.integration_state["performance_mode"]
        }
    
    async def _analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze learning progress from streaming"""
        learning_progress = {
            "sessions_completed": 0,
            "insights_discovered": 0,
            "patterns_identified": 0
        }
        
        # Get reflection history
        if self.reflection_engine:
            history = self.reflection_engine.original_engine.reflection_history
            learning_progress["sessions_completed"] = len(history)
            
            # Count insights
            insights = self.reflection_engine.reflection_context["recent_insights"]
            learning_progress["insights_discovered"] = len(insights)
        
        return learning_progress
    
    def _assess_streaming_readiness(self) -> Dict[str, Any]:
        """Assess readiness for streaming"""
        readiness = {
            "components_ready": self._is_ready_for_streaming(),
            "performance_optimized": self.integration_state["performance_mode"] != "quality",
            "integration_complete": self.integration_state["initialized"]
        }
        
        readiness["overall_readiness"] = all(readiness.values())
        
        return readiness
    
    async def _synthesize_component_coordination(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize coordination between components"""
        coordination = {
            "active_components": list(self.integration_state["components_loaded"].keys()),
            "coordination_mode": "distributed" if len(self.integration_state["components_loaded"]) > 2 else "simple",
            "synchronization_status": "active" if self._is_streaming_active() else "standby"
        }
        
        return coordination
    
    async def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if self.integration_state["performance_mode"] == "quality":
            recommendations.append("Consider switching to balanced mode for better performance")
        
        if self._is_streaming_active():
            metrics = await self._analyze_performance_metrics()
            if metrics.get("fps", 30) < 25:
                recommendations.append("Frame rate is low, consider reducing processing load")
        
        return recommendations
    
    async def _generate_integration_insights(self, context: SharedContext, messages: Dict) -> List[str]:
        """Generate insights about streaming integration"""
        insights = []
        
        if self._is_ready_for_streaming():
            insights.append("All streaming components are integrated and ready")
        
        if self._is_streaming_active():
            insights.append("Streaming is actively coordinating across all systems")
        
        # Check for cross-system patterns
        if len(messages) > 3:
            insights.append(f"Observing coordination across {len(messages)} systems")
        
        return insights
    
    async def _suggest_next_steps(self, context: SharedContext) -> List[str]:
        """Suggest next steps for streaming"""
        suggestions = []
        
        if not self._is_ready_for_streaming():
            suggestions.append("Set up streaming components to begin")
        elif not self._is_streaming_active():
            suggestions.append("Start streaming to engage all systems")
        else:
            suggestions.append("Continue streaming to build more experiences")
            
            # Check for pending reflections
            if self.reflection_engine and len(self.reflection_engine.reflection_context["pending_reflections"]) > 3:
                suggestions.append("Process pending reflections for deeper insights")
        
        return suggestions
    
    def _register_brain_functions(self):
        """Register streaming functions with the brain"""
        if not self.brain:
            return
        
        # Basic streaming functions
        self.brain.stream = self.streaming_core.start_streaming
        self.brain.stop_stream = self.streaming_core.stop_streaming
        self.brain.add_stream_question = self.streaming_core.add_audience_question
        self.brain.get_stream_stats = self.streaming_core.get_streaming_stats
        
        # Enhanced functions
        self.brain.process_streaming_frame = self.streaming_core.process_frame_optimized
        self.brain.get_streaming_performance = self.streaming_core.get_performance_metrics
        
        # Hormone functions
        self.brain.get_streaming_emotional_state = self.hormone_system.get_emotional_state
        self.brain.update_streaming_hormones = self.hormone_system.update_from_event
        
        # Reflection functions
        self.brain.generate_streaming_reflection = self.reflection_engine.generate_deep_reflection
        self.brain.consolidate_streaming = self.reflection_engine.consolidate_streaming_experiences
        
        logger.info("Registered streaming functions with brain")
    
    # Integration helper for setup
    @classmethod
    async def create_integrated_streaming(cls, brain, video_source=0, audio_source=None):
        """Create a fully integrated streaming system"""
        integration = cls(brain, video_source, audio_source)
        
        # Set up components
        await integration._setup_streaming_components()
        
        # Register with brain's context system if available
        if hasattr(brain, "context_distribution"):
            integration.set_context_system(brain.context_distribution)
        
        return integration
