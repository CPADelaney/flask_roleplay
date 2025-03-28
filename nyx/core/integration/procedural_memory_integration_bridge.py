# nyx/core/integration/procedural_memory_integration_bridge.py

import asyncio
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class ProceduralMemoryIntegrationBridge:
    """
    Central integration bridge for the procedural memory system.
    
    Provides unified access to procedural knowledge across modules and enables
    cross-module functionality for learning, executing, and adapting procedures.
    
    Key functions:
    1. Routes procedural memory events throughout the system
    2. Enables cross-module procedural learning from observations
    3. Coordinates procedural execution with other cognitive systems
    4. Facilitates transfer of procedural knowledge between domains
    5. Integrates emotional and sensory context with procedural knowledge
    """
    
    def __init__(self, 
                brain_reference=None,
                procedural_memory=None,
                memory_core=None,
                emotional_core=None,
                planning_system=None,
                attention_system=None,
                motor_system=None,
                knowledge_core=None):
        """Initialize the procedural memory integration bridge."""
        self.brain = brain_reference
        self.procedural_memory = procedural_memory
        self.memory_core = memory_core
        self.emotional_core = emotional_core
        self.planning_system = planning_system
        self.attention_system = attention_system
        self.motor_system = motor_system
        self.knowledge_core = knowledge_core
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.learning_threshold = 0.6  # Threshold for auto-learning procedures
        self.execution_confidence_threshold = 0.7  # Min confidence for automatic execution
        self.memory_integration_weight = 0.3  # Weight for memory integration in procedural learning
        self.emotion_influence_weight = 0.2  # How much emotions influence procedure selection
        
        # Tracking variables
        self._subscribed = False
        self.pending_observations = []
        self.max_observations = 100
        self.recent_executions = []
        self.max_executions = 50
        
        # Cross-domain mapping cache
        self.domain_mappings = {}  # Cache for domain mappings
        
        logger.info("ProceduralMemoryIntegrationBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Initialize core systems if provided as references only
            if self.procedural_memory is None and hasattr(self.brain, "procedural_memory"):
                self.procedural_memory = self.brain.procedural_memory
                
            if self.memory_core is None and hasattr(self.brain, "memory_core"):
                self.memory_core = self.brain.memory_core
                
            if self.emotional_core is None and hasattr(self.brain, "emotional_core"):
                self.emotional_core = self.brain.emotional_core
                
            if self.planning_system is None and hasattr(self.brain, "planning_system"):
                self.planning_system = self.brain.planning_system
                
            if self.attention_system is None and hasattr(self.brain, "attention_system"):
                self.attention_system = self.brain.attention_system
                
            if self.motor_system is None and hasattr(self.brain, "motor_system"):
                self.motor_system = self.brain.motor_system
                
            if self.knowledge_core is None and hasattr(self.brain, "knowledge_core"):
                self.knowledge_core = self.brain.knowledge_core
            
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("procedural_observation", self._handle_procedural_observation)
                self.event_bus.subscribe("execute_procedure_request", self._handle_execution_request)
                self.event_bus.subscribe("emotional_state_change", self._handle_emotional_change)
                self.event_bus.subscribe("memory_retrieved", self._handle_memory_retrieved)
                self.event_bus.subscribe("learning_opportunity", self._handle_learning_opportunity)
                self.event_bus.subscribe("decision_made", self._handle_decision_event)
                self.event_bus.subscribe("attention_focus_changed", self._handle_attention_change)
                self._subscribed = True
            
            # Initialize procedural memory if necessary
            if self.procedural_memory and hasattr(self.procedural_memory, "initialize_enhanced_components"):
                await self.procedural_memory.initialize_enhanced_components()
            
            # Schedule background maintenance
            asyncio.create_task(self._schedule_maintenance())
            
            logger.info("ProceduralMemoryIntegrationBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing ProceduralMemoryIntegrationBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="ProceduralMemory")
    async def learn_from_observations(self, 
                                   observations: List[Dict[str, Any]],
                                   domain: str,
                                   name: Optional[str] = None) -> Dict[str, Any]:
        """
        Learn a new procedure from observations with integrated context.
        
        Args:
            observations: Sequence of observed actions
            domain: Domain for the new procedure
            name: Optional name for the procedure
            
        Returns:
            Result of procedure learning with cross-module context
        """
        if not self.procedural_memory:
            return {"status": "error", "message": "Procedural memory system not available"}
        
        try:
            # Add emotional context if available
            enhanced_observations = []
            
            for obs in observations:
                enhanced_obs = obs.copy()
                
                # Add emotional context if available
                if self.emotional_core and "state" in enhanced_obs:
                    try:
                        if hasattr(self.emotional_core, 'get_emotional_state_matrix'):
                            emotional_state = await self.emotional_core.get_emotional_state_matrix()
                            
                            # Add emotional state to observation context
                            enhanced_obs["state"]["emotional_state"] = emotional_state
                    except Exception as e:
                        logger.warning(f"Error retrieving emotional context: {e}")
                
                # Add attention context if available
                if self.attention_system and "state" in enhanced_obs:
                    try:
                        attention_focus = await self.attention_system.get_current_focus()
                        
                        # Add attention focus to observation context
                        enhanced_obs["state"]["attention_focus"] = attention_focus
                    except Exception as e:
                        logger.warning(f"Error retrieving attention context: {e}")
                
                enhanced_observations.append(enhanced_obs)
            
            # Use the enhanced memory manager if available
            if hasattr(self.procedural_memory, "learn_from_demonstration"):
                # Learn procedure with enhanced observations
                procedure_result = await self.procedural_memory.learn_from_demonstration(
                    observation_sequence=enhanced_observations,
                    domain=domain,
                    name=name
                )
                
                # Store learning experience in episodic memory if available
                if self.memory_core and hasattr(self.memory_core, "add_memory"):
                    memory_text = f"Learned a new procedure for {procedure_result.get('name')} in the {domain} domain"
                    
                    await self.memory_core.add_memory(
                        memory_text=memory_text,
                        memory_type="procedural_learning",
                        significance=7,  # High significance
                        tags=["procedural_memory", "learning", domain],
                        metadata={
                            "procedure_name": procedure_result.get("name"),
                            "domain": domain,
                            "steps_count": procedure_result.get("steps_count", 0),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                
                # Broadcast event for other modules
                event = Event(
                    event_type="procedure_learned",
                    source="procedural_memory_integration_bridge",
                    data={
                        "procedure_name": procedure_result.get("name"),
                        "domain": domain,
                        "steps_count": procedure_result.get("steps_count", 0),
                        "confidence": procedure_result.get("confidence", 0.0)
                    }
                )
                await self.event_bus.publish(event)
                
                return procedure_result
            else:
                # Fallback to basic learning
                return {
                    "status": "error", 
                    "message": "Enhanced procedural learning not available",
                    "observations_count": len(observations)
                }
        except Exception as e:
            logger.error(f"Error learning from observations: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="ProceduralMemory")
    async def execute_procedure_with_context(self,
                                         name: str,
                                         execution_context: Dict[str, Any] = None,
                                         force_conscious: bool = False) -> Dict[str, Any]:
        """
        Execute a procedure with integrated cross-module context.
        
        Args:
            name: Name of the procedure to execute
            execution_context: Optional execution context
            force_conscious: Whether to force conscious execution
            
        Returns:
            Execution results with integrated context
        """
        if not self.procedural_memory:
            return {"status": "error", "message": "Procedural memory system not available"}
        
        try:
            # Initialize context if needed
            context = execution_context.copy() if execution_context else {}
            
            # Add emotional context if available
            if self.emotional_core:
                try:
                    if hasattr(self.emotional_core, 'get_emotional_state_matrix'):
                        emotional_state = await self.emotional_core.get_emotional_state_matrix()
                        
                        # Add to context
                        context["emotional_state"] = emotional_state
                        
                        # Set conscious execution based on emotional intensity
                        if not force_conscious and emotional_state:
                            if isinstance(emotional_state, dict) and "arousal" in emotional_state:
                                # High arousal suggests more conscious processing
                                if emotional_state["arousal"] > 0.7:
                                    force_conscious = True
                except Exception as e:
                    logger.warning(f"Error retrieving emotional context: {e}")
            
            # Add attention context if available
            if self.attention_system:
                try:
                    attention_focus = await self.attention_system.get_current_focus()
                    
                    # Add to context
                    context["attention_focus"] = attention_focus
                    
                    # Direct attention to procedure execution
                    if hasattr(self.attention_system, 'focus_attention'):
                        await self.attention_system.focus_attention(
                            target=f"executing procedure: {name}",
                            target_type="procedural_execution",
                            attention_level=0.8,  # High attention
                            source="procedural_memory_integration_bridge"
                        )
                except Exception as e:
                    logger.warning(f"Error setting attention context: {e}")
            
            # Add memory context if available
            if self.memory_core and hasattr(self.memory_core, "retrieve_memories"):
                try:
                    # Get relevant memories for this procedure
                    relevant_memories = await self.memory_core.retrieve_memories(
                        query=f"procedure execution {name}",
                        memory_types=["experience", "observation"],
                        limit=3,
                        min_significance=5
                    )
                    
                    # Add to context
                    if relevant_memories:
                        context["relevant_memories"] = relevant_memories
                except Exception as e:
                    logger.warning(f"Error retrieving memory context: {e}")
            
            # Execute procedure with enhanced context
            if hasattr(self.procedural_memory, "execute_procedure"):
                # Use the enhanced execution
                execution_result = await self.procedural_memory.execute_procedure(
                    name=name,
                    context=context,
                    force_conscious=force_conscious
                )
                
                # Record execution in memory if significant or exceptional
                if self.memory_core and (
                    execution_result.get("exceptional", False) or 
                    not execution_result.get("success", True)
                ):
                    success = execution_result.get("success", False)
                    significance = 7 if not success else 5  # Failed executions are more notable
                    
                    memory_text = f"{'Successfully executed' if success else 'Failed to execute'} procedure '{name}'"
                    
                    await self.memory_core.add_memory(
                        memory_text=memory_text,
                        memory_type="procedural_execution",
                        significance=significance,
                        tags=["procedural_memory", "execution", "success" if success else "failure"],
                        metadata={
                            "procedure_name": name,
                            "success": success,
                            "execution_time": execution_result.get("execution_time", 0.0),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                
                # Broadcast execution completion event
                event = Event(
                    event_type="procedure_executed",
                    source="procedural_memory_integration_bridge",
                    data={
                        "procedure_name": name,
                        "success": execution_result.get("success", False),
                        "execution_time": execution_result.get("execution_time", 0.0)
                    }
                )
                await self.event_bus.publish(event)
                
                # Update recent executions
                self.recent_executions.append({
                    "procedure_name": name,
                    "success": execution_result.get("success", False),
                    "execution_time": execution_result.get("execution_time", 0.0),
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Trim history if needed
                if len(self.recent_executions) > self.max_executions:
                    self.recent_executions = self.recent_executions[-self.max_executions:]
                
                return execution_result
            else:
                # Fallback to basic execution
                return {
                    "status": "error", 
                    "message": "Enhanced procedural execution not available"
                }
        except Exception as e:
            logger.error(f"Error executing procedure: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="ProceduralMemory")
    async def transfer_procedure_knowledge(self,
                                        source_procedure: str,
                                        target_domain: str,
                                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transfer procedural knowledge from one domain to another with cross-module context.
        
        Args:
            source_procedure: Name of the source procedure
            target_domain: Target domain to transfer to
            context: Optional additional context
            
        Returns:
            Results of the knowledge transfer
        """
        if not self.procedural_memory:
            return {"status": "error", "message": "Procedural memory system not available"}
        
        try:
            # Generate a good name for the target procedure
            target_name = f"{source_procedure}_{target_domain}"
            
            # Add knowledge context if available
            if self.knowledge_core and hasattr(self.knowledge_core, "get_domain_knowledge"):
                try:
                    # Get knowledge about target domain
                    domain_knowledge = await self.knowledge_core.get_domain_knowledge(target_domain)
                    
                    # Create context if not provided
                    if context is None:
                        context = {}
                    
                    # Add domain knowledge
                    if domain_knowledge:
                        context["domain_knowledge"] = domain_knowledge
                except Exception as e:
                    logger.warning(f"Error retrieving domain knowledge: {e}")
            
            # Use transfer optimizer if available
            if hasattr(self.procedural_memory, "optimize_procedure_transfer"):
                # First optimize the transfer
                transfer_plan = await self.procedural_memory.optimize_procedure_transfer(
                    source_procedure=source_procedure,
                    target_domain=target_domain
                )
                
                # Execute the transfer plan
                transfer_result = await self.procedural_memory.execute_transfer_plan(
                    transfer_plan=transfer_plan,
                    target_name=target_name
                )
            elif hasattr(self.procedural_memory, "transfer_procedure"):
                # Use simple transfer
                transfer_result = await self.procedural_memory.transfer_procedure(
                    source_name=source_procedure,
                    target_name=target_name,
                    target_domain=target_domain
                )
            else:
                return {
                    "status": "error", 
                    "message": "Procedure transfer functionality not available"
                }
            
            # Add to knowledge core if transfer was successful
            if transfer_result.get("success", False) and self.knowledge_core:
                await self.share_procedural_knowledge(target_domain)
            
            # Broadcast transfer event
            event = Event(
                event_type="procedure_transferred",
                source="procedural_memory_integration_bridge",
                data={
                    "source_procedure": source_procedure,
                    "target_procedure": target_name,
                    "source_domain": transfer_result.get("source_domain", "unknown"),
                    "target_domain": target_domain,
                    "success": transfer_result.get("success", False)
                }
            )
            await self.event_bus.publish(event)
            
            return transfer_result
        except Exception as e:
            logger.error(f"Error transferring procedure: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="ProceduralMemory")
    async def integrate_with_planning(self,
                                   goal: Dict[str, Any],
                                   available_procedures: List[str] = None) -> Dict[str, Any]:
        """
        Integrate procedural knowledge with planning for goal achievement.
        
        Args:
            goal: Goal state to achieve
            available_procedures: Optional list of procedures to consider
            
        Returns:
            Planning results with procedural steps
        """
        if not self.procedural_memory or not self.planning_system:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Get available procedures if not provided
            if not available_procedures and hasattr(self.procedural_memory, "list_procedures"):
                procedures_list = await self.procedural_memory.list_procedures(None)
                available_procedures = [p["name"] for p in procedures_list]
            
            # Find suitable procedures for goal
            suitable_procedures = []
            
            for proc_name in available_procedures:
                # Check if procedure exists
                if hasattr(self.procedural_memory.procedures, proc_name):
                    procedure = self.procedural_memory.procedures[proc_name]
                    
                    # Check if procedure helps achieve goal
                    relevance_score = self._calculate_goal_relevance(procedure, goal)
                    
                    if relevance_score > 0.6:  # Threshold for relevance
                        suitable_procedures.append({
                            "name": proc_name,
                            "relevance": relevance_score,
                            "proficiency": procedure.proficiency
                        })
            
            # Sort by relevance
            suitable_procedures.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Create planning steps
            planning_steps = []
            
            if suitable_procedures:
                # Use most relevant procedure
                best_procedure = suitable_procedures[0]["name"]
                
                # Create plan step for procedure execution
                planning_steps.append({
                    "action": f"execute_procedure:{best_procedure}",
                    "description": f"Execute procedure '{best_procedure}' to achieve goal",
                    "preconditions": {},
                    "postconditions": goal,
                    "estimated_success": suitable_procedures[0]["relevance"] * suitable_procedures[0]["proficiency"]
                })
            else:
                # No suitable procedures, suggest learning
                planning_steps.append({
                    "action": "learn_new_procedure",
                    "description": "No suitable procedures found, need to learn a new one",
                    "preconditions": {},
                    "postconditions": {"new_procedure_learned": True},
                    "estimated_success": 0.5
                })
            
            # Integrate with planning system
            if hasattr(self.planning_system, "integrate_procedural_steps"):
                planning_result = await self.planning_system.integrate_procedural_steps(
                    goal=goal,
                    procedural_steps=planning_steps
                )
            else:
                # Return simple planning result
                planning_result = {
                    "goal": goal,
                    "procedural_steps": planning_steps,
                    "suitable_procedures": suitable_procedures
                }
            
            return planning_result
        except Exception as e:
            logger.error(f"Error integrating with planning: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="ProceduralMemory")
    async def create_integrated_procedure(self,
                                       steps: List[Dict[str, Any]],
                                       domain: str,
                                       name: str,
                                       description: str = None,
                                       goal_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new procedure with cross-module integration.
        
        Args:
            steps: Procedure steps
            domain: Domain for the procedure
            name: Name for the procedure
            description: Optional description
            goal_state: Optional goal state
            
        Returns:
            Created procedure information
        """
        if not self.procedural_memory:
            return {"status": "error", "message": "Procedural memory system not available"}
        
        try:
            # Enhance steps with cross-module context
            enhanced_steps = []
            
            for step in steps:
                # Create enhanced copy
                enhanced_step = step.copy()
                
                # Add emotional tagging if not present
                if "emotional_state" not in enhanced_step and self.emotional_core:
                    try:
                        # Tag with current emotional state
                        emotional_state = await self.emotional_core.get_current_emotional_state()
                        enhanced_step["emotional_context"] = emotional_state
                    except Exception as e:
                        logger.warning(f"Error adding emotional context to step: {e}")
                
                # Add attention requirements if not present
                if "attention_level" not in enhanced_step and "focus_requirements" not in enhanced_step:
                    enhanced_step["attention_level"] = 0.7  # Default high attention
                
                enhanced_steps.append(enhanced_step)
            
            # Create procedure based on available method
            if hasattr(self.procedural_memory, "create_hierarchical_procedure") and goal_state:
                # Create hierarchical procedure with goal state
                procedure_result = await self.procedural_memory.create_hierarchical_procedure(
                    name=name,
                    description=description or f"Procedure for {name}",
                    domain=domain,
                    steps=enhanced_steps,
                    goal_state=goal_state
                )
            elif hasattr(self.procedural_memory, "create_procedure"):
                # Create standard procedure
                procedure_result = await self.procedural_memory.create_procedure(
                    name=name,
                    steps=enhanced_steps,
                    description=description or f"Procedure for {name}",
                    domain=domain
                )
            else:
                return {"status": "error", "message": "Procedure creation not available"}
            
            # Add to knowledge core
            if self.knowledge_core and hasattr(self.knowledge_core, "add_procedural_knowledge"):
                knowledge_item = {
                    "procedure_name": name,
                    "domain": domain,
                    "description": description or f"Procedure for {name}",
                    "steps_count": len(enhanced_steps)
                }
                
                await self.knowledge_core.add_procedural_knowledge(knowledge_item)
            
            # Broadcast creation event
            event = Event(
                event_type="procedure_created",
                source="procedural_memory_integration_bridge",
                data={
                    "procedure_name": name,
                    "domain": domain,
                    "steps_count": len(enhanced_steps)
                }
            )
            await self.event_bus.publish(event)
            
            return procedure_result
        except Exception as e:
            logger.error(f"Error creating integrated procedure: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="ProceduralMemory")
    async def share_procedural_knowledge(self, domain: str = None) -> Dict[str, Any]:
        """
        Share procedural knowledge with the knowledge core.
        
        Args:
            domain: Optional domain to filter by
            
        Returns:
            Results of knowledge sharing
        """
        if not self.procedural_memory or not self.knowledge_core:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Get all procedures for the domain
            if hasattr(self.procedural_memory, "list_procedures"):
                procedures_list = await self.procedural_memory.list_procedures(domain)
            else:
                return {"status": "error", "message": "Procedure listing not available"}
            
            # Share each procedure with the knowledge core
            shared_count = 0
            for procedure in procedures_list:
                if hasattr(self.knowledge_core, "add_procedural_knowledge"):
                    knowledge_item = {
                        "procedure_name": procedure["name"],
                        "domain": procedure["domain"],
                        "description": procedure["description"],
                        "proficiency": procedure["proficiency"],
                        "steps_count": procedure["steps_count"]
                    }
                    
                    await self.knowledge_core.add_procedural_knowledge(knowledge_item)
                    shared_count += 1
            
            return {
                "status": "success",
                "procedures_shared": shared_count,
                "domain": domain or "all"
            }
        except Exception as e:
            logger.error(f"Error sharing procedural knowledge: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="ProceduralMemory")
    async def optimize_procedural_memory(self) -> Dict[str, Any]:
        """
        Optimize procedural memory for better performance.
        
        Returns:
            Optimization results
        """
        if not self.procedural_memory:
            return {"status": "error", "message": "Procedural memory system not available"}
        
        try:
            # Use enhanced optimizer if available
            if hasattr(self.procedural_memory, "optimize_procedural_memory"):
                optimization_result = await self.procedural_memory.optimize_procedural_memory()
                
                # Broadcast optimization event
                event = Event(
                    event_type="procedural_memory_optimized",
                    source="procedural_memory_integration_bridge",
                    data={
                        "memory_saved": optimization_result.get("memory_saved", 0),
                        "procedures_cleaned": optimization_result.get("procedures_cleaned", 0)
                    }
                )
                await self.event_bus.publish(event)
                
                return optimization_result
            else:
                return {"status": "error", "message": "Memory optimization not available"}
        except Exception as e:
            logger.error(f"Error optimizing procedural memory: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="ProceduralMemory")
    async def handle_execution_error(self,
                                  error: Dict[str, Any],
                                  procedure_name: str,
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle execution errors with cross-module integration.
        
        Args:
            error: Error details
            procedure_name: Name of the procedure that failed
            context: Optional execution context
            
        Returns:
            Error handling results
        """
        if not self.procedural_memory:
            return {"status": "error", "message": "Procedural memory system not available"}
        
        try:
            # Use sophisticated error handling if available
            if hasattr(self.procedural_memory, "handle_execution_error"):
                error_handling = await self.procedural_memory.handle_execution_error(
                    error=error,
                    context=context
                )
                
                # Record error in memory
                if self.memory_core and hasattr(self.memory_core, "add_memory"):
                    error_text = f"Error executing procedure '{procedure_name}': {error.get('message', 'Unknown error')}"
                    
                    await self.memory_core.add_memory(
                        memory_text=error_text,
                        memory_type="execution_error",
                        significance=6,  # High significance for errors
                        tags=["procedural_memory", "error", procedure_name],
                        metadata={
                            "procedure_name": procedure_name,
                            "error_type": error.get("type", "unknown"),
                            "error_message": error.get("message", "Unknown error"),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                
                # Update emotional state if error is significant
                if self.emotional_core and error.get("severity", 0) > 0.7:
                    try:
                        if hasattr(self.emotional_core, 'update_emotion'):
                            await self.emotional_core.update_emotion(
                                "frustration", 
                                error.get("severity", 0.5)
                            )
                    except Exception as e:
                        logger.warning(f"Error updating emotional state: {e}")
                
                # Broadcast error event
                event = Event(
                    event_type="procedure_execution_error",
                    source="procedural_memory_integration_bridge",
                    data={
                        "procedure_name": procedure_name,
                        "error_type": error.get("type", "unknown"),
                        "error_message": error.get("message", "Unknown error"),
                        "recoverable": error_handling.get("recoverable", False)
                    }
                )
                await self.event_bus.publish(event)
                
                return error_handling
            else:
                # Basic error reporting
                return {
                    "status": "handled",
                    "procedure_name": procedure_name,
                    "error": error,
                    "recoverable": False
                }
        except Exception as e:
            logger.error(f"Error handling execution error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_goal_relevance(self, procedure: Any, goal: Dict[str, Any]) -> float:
        """Calculate relevance of a procedure to a goal."""
        relevance = 0.0
        
        # Check for goal_state attribute
        if hasattr(procedure, "goal_state") and procedure.goal_state:
            goal_state = procedure.goal_state
            
            # Count matching goal items
            matches = 0
            for key, value in goal.items():
                if key in goal_state and goal_state[key] == value:
                    matches += 1
            
            if len(goal) > 0:
                # Calculate match percentage
                relevance = matches / len(goal)
        else:
            # Check step postconditions
            for step in procedure.steps:
                postconditions = step.get("postconditions", {})
                
                for key, value in goal.items():
                    if key in postconditions and postconditions[key] == value:
                        relevance += 0.2  # Partial match
            
            # Cap at 1.0
            relevance = min(1.0, relevance)
        
        return relevance
    
    async def _handle_procedural_observation(self, event: Event) -> None:
        """
        Handle procedural observation events.
        
        Args:
            event: Procedural observation event
        """
        # Extract observation data
        observations = event.data.get("observations", [])
        domain = event.data.get("domain", "general")
        action_sequence = event.data.get("action_sequence", [])
        
        # Combine all observations
        all_observations = observations
        if action_sequence:
            # Convert action sequence to observations format
            for action in action_sequence:
                all_observations.append({
                    "action": action.get("action"),
                    "state": action.get("state", {}),
                    "timestamp": action.get("timestamp", datetime.datetime.now().isoformat())
                })
        
        # Add to pending observations
        self.pending_observations.extend(all_observations)
        
        # Trim if needed
        if len(self.pending_observations) > self.max_observations:
            self.pending_observations = self.pending_observations[-self.max_observations:]
        
        # Check if we should learn automatically
        if len(all_observations) >= 3:  # Need at least 3 observations
            # Auto-learn procedures if observations are consistent
            consistent_pattern = self._check_for_consistent_pattern(all_observations)
            
            if consistent_pattern and consistent_pattern.get("consistency", 0) > self.learning_threshold:
                # Auto-learn a new procedure
                await self.learn_from_observations(
                    observations=all_observations,
                    domain=domain,
                    name=consistent_pattern.get("suggested_name")
                )
    
    async def _handle_execution_request(self, event: Event) -> None:
        """
        Handle procedure execution requests.
        
        Args:
            event: Execution request event
        """
        # Extract request data
        procedure_name = event.data.get("procedure_name")
        context = event.data.get("context", {})
        force_conscious = event.data.get("force_conscious", False)
        
        if not procedure_name:
            return
        
        # Execute the procedure
        execution_result = await self.execute_procedure_with_context(
            name=procedure_name,
            execution_context=context,
            force_conscious=force_conscious
        )
        
        # Send response event
        response_event = Event(
            event_type="procedure_execution_response",
            source="procedural_memory_integration_bridge",
            data={
                "procedure_name": procedure_name,
                "success": execution_result.get("success", False),
                "execution_time": execution_result.get("execution_time", 0.0),
                "request_id": event.data.get("request_id")
            }
        )
        await self.event_bus.publish(response_event)
    
    async def _handle_emotional_change(self, event: Event) -> None:
        """
        Handle emotional state change events.
        
        Args:
            event: Emotional state change event
        """
        # Extract emotional data
        emotion = event.data.get("emotion")
        intensity = event.data.get("intensity", 0.5)
        
        # Only handle significant emotional changes
        if intensity < 0.7:
            return
        
        # Check if this should affect procedure execution
        if emotion in ["fear", "panic", "anxiety"]:
            # These emotions suggest more deliberate processing
            self.execution_confidence_threshold = 0.9  # Temporarily increase threshold
            
            # Schedule reset after a delay
            async def reset_threshold():
                await asyncio.sleep(60)  # Reset after 1 minute
                self.execution_confidence_threshold = 0.7  # Back to default
                
            asyncio.create_task(reset_threshold())
    
    async def _handle_memory_retrieved(self, event: Event) -> None:
        """
        Handle memory retrieval events for procedural integration.
        
        Args:
            event: Memory retrieved event
        """
        # Extract memory data
        memory = event.data.get("memory", {})
        
        # Check if this is a procedural memory
        if memory.get("memory_type") in ["procedural_learning", "procedural_execution"]:
            # Update system context with this memory
            self.system_context.set_value(
                "recent_procedural_memory", 
                {
                    "memory_id": memory.get("id"),
                    "memory_text": memory.get("memory_text"),
                    "procedure_name": memory.get("metadata", {}).get("procedure_name"),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
    
    async def _handle_learning_opportunity(self, event: Event) -> None:
        """
        Handle learning opportunity events.
        
        Args:
            event: Learning opportunity event
        """
        # Extract learning data
        domain = event.data.get("domain", "general")
        observations = event.data.get("observations", [])
        
        if not observations:
            return
        
        # Check if this is a good learning opportunity
        if len(observations) >= 3:  # Need at least 3 steps to learn
            # Create name suggestion
            action_types = [obs.get("action", "").split("_")[0] for obs in observations 
                          if obs.get("action")]
            name_suggestion = f"{action_types[0]}_{domain}"
            
            # Learn new procedure
            await self.learn_from_observations(
                observations=observations,
                domain=domain,
                name=name_suggestion
            )
    
    async def _handle_decision_event(self, event: Event) -> None:
        """
        Handle decision events for procedural integration.
        
        Args:
            event: Decision event
        """
        # Extract decision data
        decision_type = event.data.get("decision_type")
        selected_option = event.data.get("selected_option", {})
        
        # Check if decision relates to procedure execution
        if decision_type == "action_selection" and "execute_procedure" in str(selected_option):
            # Extract procedure name
            procedure_name = None
            action = selected_option.get("action", "")
            
            # Parse action string
            if ":" in action:
                parts = action.split(":")
                if len(parts) >= 2 and parts[0] == "execute_procedure":
                    procedure_name = parts[1]
            
            if procedure_name:
                # Get context from decision
                context = selected_option.get("context", {})
                
                # Execute the procedure
                await self.execute_procedure_with_context(
                    name=procedure_name,
                    execution_context=context
                )
    
    async def _handle_attention_change(self, event: Event) -> None:
        """
        Handle attention focus change events.
        
        Args:
            event: Attention change event
        """
        # Only used to update system context
        pass
    
    def _check_for_consistent_pattern(self, observations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check for consistent patterns in observations."""
        if len(observations) < 3:
            return None
        
        # Extract action sequences
        actions = [obs.get("action") for obs in observations if obs.get("action")]
        
        if not actions:
            return None
        
        # Count sequences
        sequences = []
        for i in range(len(actions) - 2):
            # Look at three consecutive actions
            seq = (actions[i], actions[i+1], actions[i+2])
            sequences.append(seq)
        
        # Check for repetitions
        if not sequences:
            return None
            
        # Count occurrences
        seq_counts = {}
        for seq in sequences:
            seq_str = str(seq)
            if seq_str not in seq_counts:
                seq_counts[seq_str] = 0
            seq_counts[seq_str] += 1
        
        # Find most common sequence
        if not seq_counts:
            return None
            
        most_common = max(seq_counts.items(), key=lambda x: x[1])
        most_common_seq = eval(most_common[0])  # Convert back to tuple
        consistency = most_common[1] / len(sequences)
        
        # Generate name suggestion
        first_action = most_common_seq[0].split("_")[0] if most_common_seq[0] else "procedure"
        suggested_name = f"{first_action}_sequence"
        
        return {
            "consistency": consistency,
            "suggested_name": suggested_name,
            "sequence": most_common_seq
        }
    
    async def _schedule_maintenance(self) -> None:
        """Schedule periodic maintenance."""
        while True:
            # Run maintenance every hour
            await asyncio.sleep(3600)
            
            try:
                # Run optimization
                await self.optimize_procedural_memory()
                
                # Share knowledge with knowledge core
                if self.knowledge_core:
                    await self.share_procedural_knowledge()
                
                logger.info("Completed procedural memory maintenance")
            except Exception as e:
                logger.error(f"Error during scheduled maintenance: {e}")

# Factory function to create the bridge
def create_procedural_memory_integration_bridge(nyx_brain):
    """Create a procedural memory integration bridge for the given brain."""
    return ProceduralMemoryIntegrationBridge(
        brain_reference=nyx_brain,
        procedural_memory=nyx_brain.procedural_memory if hasattr(nyx_brain, "procedural_memory") else None,
        memory_core=nyx_brain.memory_core if hasattr(nyx_brain, "memory_core") else None,
        emotional_core=nyx_brain.emotional_core if hasattr(nyx_brain, "emotional_core") else None,
        planning_system=nyx_brain.planning_system if hasattr(nyx_brain, "planning_system") else None,
        attention_system=nyx_brain.attention_system if hasattr(nyx_brain, "attention_system") else None,
        motor_system=nyx_brain.motor_system if hasattr(nyx_brain, "motor_system") else None,
        knowledge_core=nyx_brain.knowledge_core if hasattr(nyx_brain, "knowledge_core") else None
    )
