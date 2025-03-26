# nyx/core/brain/agent_integration.py
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from agents import Agent, Runner, trace, function_tool, handoff, RunContextWrapper

logger = logging.getLogger(__name__)

class AgentIntegration:
    """
    Integration system for agent-based processing and meta-tone adjustments
    in the Nyx brain system
    """
    
    def __init__(self, brain=None):
        """
        Initialize the agent integration system
        
        Args:
            brain: Reference to the NyxBrain instance
        """
        self.brain = brain
        
        # Agent registry
        self.agents = {}
        self.agent_capabilities = {}
        
        # Agent meta-tone adjustments
        self.meta_tone_registry = {}
        self.current_meta_tone = "standard"
        self.meta_tone_history = []
        
        # Agent combination patterns
        self.combination_patterns = {}
        
        # Integration metrics
        self.integration_metrics = {
            "agent_invocations": {},
            "meta_tone_switches": [],
            "combination_uses": {},
            "processing_times": []
        }
        
        # Performance metrics
        self.performance_metrics = {
            "avg_response_time": 0.0,
            "token_usage": {
                "total": 0,
                "by_agent": {}
            },
            "success_rate": 0.95  # Starting assumption
        }
        
        # Reflection data
        self.reflection_registry = {
            "agent_reflections": {},
            "meta_tone_reflections": {},
            "combination_reflections": {}
        }
        
        logger.info("Agent integration system initialized")
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the system and register default agents"""
        # Set up default meta-tones
        self._setup_default_meta_tones()
        
        # Set up default combination patterns
        self._setup_default_combination_patterns()
        
        # Register core agents if brain is available
        if self.brain:
            await self._register_core_agents()
        
        return {
            "meta_tones_available": list(self.meta_tone_registry.keys()),
            "combination_patterns_available": list(self.combination_patterns.keys()),
            "agents_registered": len(self.agents)
        }
    
    def _setup_default_meta_tones(self) -> None:
        """Set up default meta-tone adjustments"""
        self.meta_tone_registry = {
            "standard": {
                "description": "Standard balanced tone",
                "modifiers": {
                    "formality": 0.5,
                    "warmth": 0.5,
                    "conciseness": 0.5,
                    "detail_orientation": 0.5,
                    "technical_level": 0.5,
                    "empathy": 0.5
                }
            },
            "academic": {
                "description": "Formal, technical, and detail-oriented",
                "modifiers": {
                    "formality": 0.8,
                    "warmth": 0.3,
                    "conciseness": 0.4,
                    "detail_orientation": 0.9,
                    "technical_level": 0.8,
                    "empathy": 0.3
                }
            },
            "casual": {
                "description": "Warm, conversational, and concise",
                "modifiers": {
                    "formality": 0.2,
                    "warmth": 0.8,
                    "conciseness": 0.7,
                    "detail_orientation": 0.4,
                    "technical_level": 0.3,
                    "empathy": 0.7
                }
            },
            "empathetic": {
                "description": "Highly empathetic and supportive",
                "modifiers": {
                    "formality": 0.3,
                    "warmth": 0.9,
                    "conciseness": 0.5,
                    "detail_orientation": 0.6,
                    "technical_level": 0.4,
                    "empathy": 0.9
                }
            },
            "technical": {
                "description": "Highly technical and precise",
                "modifiers": {
                    "formality": 0.7,
                    "warmth": 0.3,
                    "conciseness": 0.6,
                    "detail_orientation": 0.8,
                    "technical_level": 0.9,
                    "empathy": 0.2
                }
            },
            "concise": {
                "description": "Brief and to the point",
                "modifiers": {
                    "formality": 0.5,
                    "warmth": 0.4,
                    "conciseness": 0.9,
                    "detail_orientation": 0.3,
                    "technical_level": 0.5,
                    "empathy": 0.4
                }
            }
        }
    
    def _setup_default_combination_patterns(self) -> None:
        """Set up default agent combination patterns"""
        self.combination_patterns = {
            "reasoning_enhanced": {
                "description": "Enhanced reasoning combining multiple reasoning approaches",
                "agents": ["analytical_reasoner", "creative_reasoner", "critical_reasoner"],
                "workflow": "parallel_synthesis",
                "weights": {"analytical_reasoner": 0.4, "creative_reasoner": 0.3, "critical_reasoner": 0.3}
            },
            "identity_reflection": {
                "description": "Deep reflection on identity combining internal and external perspectives",
                "agents": ["identity_reflector", "external_observer", "consistency_checker"],
                "workflow": "sequential",
                "sequence": ["identity_reflector", "external_observer", "consistency_checker"]
            },
            "emotional_multimodal": {
                "description": "Emotional processing with multimodal integration",
                "agents": ["emotional_analyst", "multimodal_integrator", "response_generator"],
                "workflow": "hierarchical",
                "hierarchy": {
                    "root": "emotional_analyst",
                    "children": {
                        "emotional_analyst": ["multimodal_integrator"],
                        "multimodal_integrator": ["response_generator"]
                    }
                }
            },
            "streaming_specialist": {
                "description": "Specialized processing for streaming content",
                "agents": ["content_analyzer", "streaming_memory_integrator", "streaming_response_generator"],
                "workflow": "parallel_with_coordinator",
                "coordinator": "streaming_coordinator"
            }
        }
    
    async def _register_core_agents(self) -> None:
        """Register core agents from the brain system"""
        # Register main brain agent if available
        if hasattr(self.brain, "brain_agent"):
            self.register_agent(
                name="main_brain_agent",
                agent=self.brain.brain_agent,
                capabilities=["general_processing", "system_coordination"],
                description="Main coordinating agent for the Nyx brain system"
            )
        
        # Register reasoning agents if available
        if hasattr(self.brain, "reasoning_core"):
            self.register_agent(
                name="reasoning_core",
                agent=self.brain.reasoning_core,
                capabilities=["reasoning", "analysis", "problem_solving"],
                description="Core reasoning agent for complex analysis"
            )
        
        if hasattr(self.brain, "reasoning_triage_agent"):
            self.register_agent(
                name="reasoning_triage",
                agent=self.brain.reasoning_triage_agent,
                capabilities=["reasoning_triage", "task_routing"],
                description="Triage agent for routing reasoning tasks"
            )
        
        # Register memory agent if available
        if hasattr(self.brain, "memory_agent"):
            self.register_agent(
                name="memory_agent",
                agent=self.brain.memory_agent,
                capabilities=["memory_retrieval", "memory_integration"],
                description="Agent for memory operations"
            )
        
        # Register reflection agent if available
        if hasattr(self.brain, "reflection_agent"):
            self.register_agent(
                name="reflection_agent",
                agent=self.brain.reflection_agent,
                capabilities=["reflection", "self_improvement"],
                description="Agent for reflective processing"
            )
        
        # Register any other specialized agents
        if hasattr(self.brain, "nyx_main_agent"):
            self.register_agent(
                name="nyx_role_agent",
                agent=self.brain.nyx_main_agent,
                capabilities=["roleplay", "narrative", "creative"],
                description="Agent for roleplay and narrative generation"
            )
    
    def register_agent(self, 
                     name: str, 
                     agent: Agent, 
                     capabilities: List[str] = None,
                     description: str = None) -> None:
        """
        Register an agent with the integration system
        
        Args:
            name: Name to register the agent under
            agent: The Agent object
            capabilities: List of capability tags
            description: Description of the agent
        """
        self.agents[name] = agent
        self.agent_capabilities[name] = {
            "capabilities": capabilities or [],
            "description": description or "",
            "registration_time": datetime.datetime.now().isoformat()
        }
        
        # Initialize metrics
        self.integration_metrics["agent_invocations"][name] = 0
        self.performance_metrics["token_usage"]["by_agent"][name] = 0
        
        logger.info(f"Registered agent '{name}' with capabilities: {capabilities}")
    
    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent from the integration system
        
        Args:
            name: Name of the agent to unregister
            
        Returns:
            Success status
        """
        if name in self.agents:
            del self.agents[name]
            del self.agent_capabilities[name]
            logger.info(f"Unregistered agent '{name}'")
            return True
        return False
    
    def register_meta_tone(self, 
                         name: str, 
                         description: str,
                         modifiers: Dict[str, float]) -> None:
        """
        Register a new meta-tone adjustment profile
        
        Args:
            name: Name of the meta-tone
            description: Description of the meta-tone
            modifiers: Dictionary of tone modifiers (0.0-1.0)
        """
        self.meta_tone_registry[name] = {
            "description": description,
            "modifiers": modifiers,
            "registration_time": datetime.datetime.now().isoformat()
        }
        logger.info(f"Registered meta-tone '{name}': {description}")
    
    def register_combination_pattern(self,
                                   name: str,
                                   description: str,
                                   agents: List[str],
                                   workflow: str,
                                   workflow_config: Dict[str, Any]) -> None:
        """
        Register a new agent combination pattern
        
        Args:
            name: Name of the combination pattern
            description: Description of the pattern
            agents: List of agent names involved
            workflow: Type of workflow ("parallel", "sequential", etc.)
            workflow_config: Configuration for the workflow
        """
        self.combination_patterns[name] = {
            "description": description,
            "agents": agents,
            "workflow": workflow,
            **workflow_config,
            "registration_time": datetime.datetime.now().isoformat()
        }
        
        # Initialize metrics for this combination
        self.integration_metrics["combination_uses"][name] = 0
        
        logger.info(f"Registered combination pattern '{name}' with {len(agents)} agents")
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Get a registered agent by name
        
        Args:
            name: Name of the agent
            
        Returns:
            Agent object or None if not found
        """
        return self.agents.get(name)
    
    def get_agents_by_capability(self, capability: str) -> Dict[str, Agent]:
        """
        Get agents that have a specific capability
        
        Args:
            capability: Capability to filter for
            
        Returns:
            Dictionary of matching agents
        """
        matching_agents = {}
        for name, agent in self.agents.items():
            if capability in self.agent_capabilities[name]["capabilities"]:
                matching_agents[name] = agent
        return matching_agents
    
    async def set_meta_tone(self, tone_name: str, reason: str = None) -> Dict[str, Any]:
        """
        Set the current meta-tone for agent responses
        
        Args:
            tone_name: Name of the meta-tone to use
            reason: Optional reason for the tone change
            
        Returns:
            Status of the tone change
        """
        if tone_name not in self.meta_tone_registry:
            return {
                "success": False,
                "error": f"Meta-tone '{tone_name}' not found"
            }
        
        previous_tone = self.current_meta_tone
        self.current_meta_tone = tone_name
        
        # Record the tone change
        change_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "from": previous_tone,
            "to": tone_name,
            "reason": reason
        }
        self.meta_tone_history.append(change_record)
        
        # Record a reflection on this tone change if appropriate
        if self.brain and hasattr(self.brain, "reflection_engine"):
            try:
                reflection = await self.brain.reflection_engine.generate_reflection(
                    topic=f"meta_tone_change_{previous_tone}_to_{tone_name}",
                    context={
                        "previous_tone": previous_tone,
                        "new_tone": tone_name,
                        "reason": reason,
                        "tone_details": self.meta_tone_registry[tone_name]
                    }
                )
                
                self.reflection_registry["meta_tone_reflections"][f"{previous_tone}_to_{tone_name}"] = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "reflection": reflection
                }
            except Exception as e:
                logger.error(f"Error generating meta-tone change reflection: {str(e)}")
        
        return {
            "success": True,
            "previous_tone": previous_tone,
            "new_tone": tone_name,
            "tone_details": self.meta_tone_registry[tone_name]
        }
    
    def get_current_meta_tone(self) -> Dict[str, Any]:
        """
        Get the current meta-tone configuration
        
        Returns:
            Current meta-tone details
        """
        return {
            "tone": self.current_meta_tone,
            "details": self.meta_tone_registry.get(self.current_meta_tone, {})
        }
    
    def _apply_meta_tone_to_instructions(self, instructions: str, tone: str = None) -> str:
        """
        Apply meta-tone adjustments to agent instructions
        
        Args:
            instructions: Original instructions
            tone: Meta-tone to apply (defaults to current)
            
        Returns:
            Modified instructions
        """
        # Use current tone if none specified
        tone_name = tone or self.current_meta_tone
        
        # If tone doesn't exist or is "standard", return original
        if tone_name not in self.meta_tone_registry or tone_name == "standard":
            return instructions
        
        # Get tone modifiers
        modifiers = self.meta_tone_registry[tone_name]["modifiers"]
        
        # Add tone guidance to instructions
        tone_guidance = f"\n\nMeta-tone adjustment: {self.meta_tone_registry[tone_name]['description']}.\n"
        
        # Add specific guidance based on modifiers
        if modifiers.get("formality", 0.5) > 0.7:
            tone_guidance += "Use a formal, professional tone with proper language. "
        elif modifiers.get("formality", 0.5) < 0.3:
            tone_guidance += "Use a casual, conversational tone. "
        
        if modifiers.get("conciseness", 0.5) > 0.7:
            tone_guidance += "Be concise and to the point. Prioritize brevity. "
        elif modifiers.get("conciseness", 0.5) < 0.3:
            tone_guidance += "Provide comprehensive and thorough information. "
        
        if modifiers.get("technical_level", 0.5) > 0.7:
            tone_guidance += "Use technical language and precise terminology. "
        elif modifiers.get("technical_level", 0.5) < 0.3:
            tone_guidance += "Use simple, accessible language avoiding technical jargon. "
        
        if modifiers.get("empathy", 0.5) > 0.7:
            tone_guidance += "Show strong empathy and emotional understanding. "
        
        if modifiers.get("detail_orientation", 0.5) > 0.7:
            tone_guidance += "Include specific details and nuanced information. "
        elif modifiers.get("detail_orientation", 0.5) < 0.3:
            tone_guidance += "Focus on the big picture rather than specific details. "
        
        # Combine with original instructions
        modified_instructions = instructions + tone_guidance
        
        return modified_instructions
    
    async def run_agent(self, 
                     agent_name: str, 
                     input_text: str,
                     context: Dict[str, Any] = None,
                     apply_meta_tone: bool = True) -> Dict[str, Any]:
        """
        Run a specific agent with the given input
        
        Args:
            agent_name: Name of the agent to run
            input_text: Input text for the agent
            context: Additional context
            apply_meta_tone: Whether to apply meta-tone adjustments
            
        Returns:
            Result of the agent run
        """
        # Get the agent
        agent = self.get_agent(agent_name)
        if not agent:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found"
            }
        
        # Create a trace group ID for this run
        trace_id = f"agent_integration_{agent_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        with trace(workflow_name=f"run_agent_{agent_name}", group_id=trace_id):
            start_time = datetime.datetime.now()
            
            # Apply meta-tone if requested
            if apply_meta_tone:
                # Clone the agent with modified instructions
                if hasattr(agent, "instructions") and agent.instructions:
                    modified_instructions = self._apply_meta_tone_to_instructions(agent.instructions)
                    agent = agent.clone(instructions=modified_instructions)
            
            # Initialize run context
            run_context = {
                "brain": self.brain,
                "meta_tone": self.current_meta_tone,
                **(context or {})
            }
            
            # Run the agent
            try:
                result = await Runner.run(agent, input_text, context=run_context)
                
                # Track metrics
                self.integration_metrics["agent_invocations"][agent_name] += 1
                
                end_time = datetime.datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                self.integration_metrics["processing_times"].append(execution_time)
                
                # Update token usage if available
                if hasattr(result, "raw_responses"):
                    for response in result.raw_responses:
                        if hasattr(response, "usage"):
                            usage = response.usage
                            if hasattr(usage, "total_tokens"):
                                tokens = usage.total_tokens
                                self.performance_metrics["token_usage"]["total"] += tokens
                                self.performance_metrics["token_usage"]["by_agent"][agent_name] += tokens
                
                # Extract the final output
                if hasattr(result, "final_output"):
                    output = result.final_output
                else:
                    output = str(result)
                
                return {
                    "success": True,
                    "output": output,
                    "execution_time": execution_time,
                    "result_object": result
                }
            except Exception as e:
                logger.error(f"Error running agent '{agent_name}': {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "agent": agent_name
                }
    
    async def run_combination(self, 
                           combination_name: str, 
                           input_text: str,
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a combination of agents according to a registered pattern
        
        Args:
            combination_name: Name of the combination pattern
            input_text: Input text for the agents
            context: Additional context
            
        Returns:
            Combined result of the agent runs
        """
        # Get the combination pattern
        if combination_name not in self.combination_patterns:
            return {
                "success": False,
                "error": f"Combination pattern '{combination_name}' not found"
            }
        
        pattern = self.combination_patterns[combination_name]
        
        # Create a trace group ID for this run
        trace_id = f"agent_combination_{combination_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        with trace(workflow_name=f"run_combination_{combination_name}", group_id=trace_id):
            start_time = datetime.datetime.now()
            
            # Initialize context
            run_context = {
                "brain": self.brain,
                "meta_tone": self.current_meta_tone,
                "combination": combination_name,
                **(context or {})
            }
            
            # Select workflow based on pattern type
            workflow = pattern["workflow"]
            
            try:
                # Execute the appropriate workflow
                if workflow == "parallel_synthesis":
                    result = await self._execute_parallel_synthesis(pattern, input_text, run_context)
                elif workflow == "sequential":
                    result = await self._execute_sequential(pattern, input_text, run_context)
                elif workflow == "hierarchical":
                    result = await self._execute_hierarchical(pattern, input_text, run_context)
                elif workflow == "parallel_with_coordinator":
                    result = await self._execute_parallel_with_coordinator(pattern, input_text, run_context)
                else:
                    return {
                        "success": False,
                        "error": f"Unknown workflow type: {workflow}"
                    }
                
                # Track metrics
                self.integration_metrics["combination_uses"][combination_name] += 1
                
                end_time = datetime.datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Generate reflection if appropriate
                if self.brain and hasattr(self.brain, "reflection_engine"):
                    try:
                        reflection = await self.brain.reflection_engine.generate_reflection(
                            topic=f"agent_combination_{combination_name}",
                            context={
                                "combination": combination_name,
                                "input": input_text,
                                "result": result,
                                "execution_time": execution_time
                            }
                        )
                        
                        self.reflection_registry["combination_reflections"][combination_name] = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "reflection": reflection
                        }
                    except Exception as e:
                        logger.error(f"Error generating combination reflection: {str(e)}")
                
                # Include execution information in result
                result["execution_time"] = execution_time
                result["combination"] = combination_name
                
                return result
            
            except Exception as e:
                logger.error(f"Error running combination '{combination_name}': {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "combination": combination_name
                }
    
    async def _execute_parallel_synthesis(self,
                                      pattern: Dict[str, Any],
                                      input_text: str,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a parallel synthesis workflow
        
        Args:
            pattern: Combination pattern configuration
            input_text: Input text
            context: Run context
            
        Returns:
            Combined result
        """
        # Get agents
        agent_names = pattern["agents"]
        weights = pattern.get("weights", {})
        
        # Set default weights if not specified
        for agent_name in agent_names:
            if agent_name not in weights:
                weights[agent_name] = 1.0 / len(agent_names)
        
        # Run agents in parallel
        tasks = {}
        for agent_name in agent_names:
            tasks[agent_name] = asyncio.create_task(
                self.run_agent(agent_name, input_text, context)
            )
        
        # Wait for all tasks to complete
        agent_results = {}
        for agent_name, task in tasks.items():
            try:
                agent_results[agent_name] = await task
            except Exception as e:
                logger.error(f"Error in parallel agent '{agent_name}': {str(e)}")
                agent_results[agent_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Check if all agents failed
        all_failed = all(not result.get("success", False) for result in agent_results.values())
        if all_failed:
            return {
                "success": False,
                "error": "All parallel agents failed",
                "agent_results": agent_results
            }
        
        # Combine successful results with synthesis
        successful_outputs = {}
        for agent_name, result in agent_results.items():
            if result.get("success", True):
                successful_outputs[agent_name] = {
                    "output": result.get("output", ""),
                    "weight": weights.get(agent_name, 1.0 / len(agent_names))
                }
        
        # Try to synthesize results
        synthesized = await self._synthesize_outputs(successful_outputs, input_text, context)
        
        return {
            "success": True,
            "output": synthesized,
            "individual_results": agent_results
        }
    
    async def _execute_sequential(self,
                              pattern: Dict[str, Any],
                              input_text: str,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a sequential workflow
        
        Args:
            pattern: Combination pattern configuration
            input_text: Input text
            context: Run context
            
        Returns:
            Combined result
        """
        # Get the sequence of agents
        sequence = pattern.get("sequence", pattern["agents"])
        
        # Process each agent in sequence
        current_input = input_text
        results = []
        all_successful = True
        
        for agent_name in sequence:
            # Run the current agent
            result = await self.run_agent(agent_name, current_input, context)
            results.append({
                "agent": agent_name,
                "result": result
            })
            
            # Check for success
            if not result.get("success", False):
                all_successful = False
                break
            
            # Use this output as input for the next agent
            current_input = result["output"]
        
        return {
            "success": all_successful,
            "output": current_input,  # Final output from the last agent
            "sequential_results": results
        }
    
    async def _execute_hierarchical(self,
                                pattern: Dict[str, Any],
                                input_text: str,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a hierarchical workflow
        
        Args:
            pattern: Combination pattern configuration
            input_text: Input text
            context: Run context
            
        Returns:
            Combined result
        """
        # Get hierarchy configuration
        hierarchy = pattern.get("hierarchy", {})
        root_agent = hierarchy.get("root")
        
        if not root_agent:
            return {
                "success": False,
                "error": "No root agent specified in hierarchy"
            }
        
        # Initialize tracking
        processed_agents = set()
        agent_outputs = {}
        
        # Process the hierarchy starting with the root
        result = await self._process_hierarchical_node(
            root_agent, input_text, context, hierarchy, processed_agents, agent_outputs
        )
        
        return {
            "success": result.get("success", False),
            "output": result.get("output", ""),
            "hierarchical_results": agent_outputs
        }
    
    async def _process_hierarchical_node(self,
                                     agent_name: str,
                                     input_text: str,
                                     context: Dict[str, Any],
                                     hierarchy: Dict[str, Any],
                                     processed_agents: set,
                                     agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a node in the hierarchical workflow
        
        Args:
            agent_name: Name of the agent to process
            input_text: Input text
            context: Run context
            hierarchy: Hierarchy configuration
            processed_agents: Set of already processed agents
            agent_outputs: Dictionary to store outputs
            
        Returns:
            Result of processing this node
        """
        # Run the current agent
        result = await self.run_agent(agent_name, input_text, context)
        agent_outputs[agent_name] = result
        processed_agents.add(agent_name)
        
        # Stop if the agent failed
        if not result.get("success", False):
            return result
        
        # Get child agents
        children = hierarchy.get("children", {}).get(agent_name, [])
        
        # Process each child
        child_results = []
        for child_name in children:
            # Skip if already processed (avoid cycles)
            if child_name in processed_agents:
                continue
                
            # Process the child
            child_result = await self._process_hierarchical_node(
                child_name, result["output"], context, hierarchy, processed_agents, agent_outputs
            )
            child_results.append(child_result)
        
        # If we have child results, combine them
        if child_results:
            # Use the last child's output as our result
            # (In a true hierarchical system, we might implement a more sophisticated combining strategy)
            return child_results[-1]
        else:
            # No children, just return our result
            return result
    
    async def _execute_parallel_with_coordinator(self,
                                            pattern: Dict[str, Any],
                                            input_text: str,
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a parallel with coordinator workflow
        
        Args:
            pattern: Combination pattern configuration
            input_text: Input text
            context: Run context
            
        Returns:
            Combined result
        """
        # Get configuration
        agent_names = pattern["agents"]
        coordinator_name = pattern.get("coordinator")
        
        if not coordinator_name:
            return {
                "success": False,
                "error": "No coordinator agent specified"
            }
        
        # Run agents in parallel
        tasks = {}
        for agent_name in agent_names:
            tasks[agent_name] = asyncio.create_task(
                self.run_agent(agent_name, input_text, context)
            )
        
        # Wait for all tasks to complete
        agent_results = {}
        for agent_name, task in tasks.items():
            try:
                agent_results[agent_name] = await task
            except Exception as e:
                logger.error(f"Error in parallel agent '{agent_name}': {str(e)}")
                agent_results[agent_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Format results for coordinator
        coordinator_input = f"Input: {input_text}\n\nAgent results:\n"
        for agent_name, result in agent_results.items():
            if result.get("success", False):
                coordinator_input += f"\n{agent_name}: {result.get('output', '')}\n"
            else:
                coordinator_input += f"\n{agent_name}: FAILED - {result.get('error', 'Unknown error')}\n"
        
        # Run coordinator
        coordinator_result = await self.run_agent(coordinator_name, coordinator_input, context)
        
        return {
            "success": coordinator_result.get("success", False),
            "output": coordinator_result.get("output", ""),
            "agent_results": agent_results,
            "coordinator_result": coordinator_result
        }
    
    async def _synthesize_outputs(self,
                              outputs: Dict[str, Dict[str, Any]],
                              original_input: str,
                              context: Dict[str, Any]) -> str:
        """
        Synthesize outputs from multiple agents
        
        Args:
            outputs: Dictionary of agent outputs with weights
            original_input: Original input text
            context: Additional context
            
        Returns:
            Synthesized output
        """
        # If we have a dedicated synthesizer agent, use it
        synthesizer = self.get_agent("output_synthesizer")
        if not synthesizer and self.brain and hasattr(self.brain, "brain_agent"):
            # Fallback to main brain agent
            synthesizer = self.brain.brain_agent
        
        if synthesizer:
            # Format input for synthesizer
            synthesizer_input = f"Original input: {original_input}\n\nAgent outputs to synthesize:\n"
            for agent_name, data in outputs.items():
                weight = data.get("weight", 1.0)
                output = data.get("output", "")
                synthesizer_input += f"\n{agent_name} (weight: {weight}):\n{output}\n"
            
            # Run synthesizer
            try:
                result = await Runner.run(synthesizer, synthesizer_input, context=context)
                
                # Extract result
                if hasattr(result, "final_output"):
                    return result.final_output
                else:
                    return str(result)
            except Exception as e:
                logger.error(f"Error in synthesizer: {str(e)}")
                # Fallback to weighted concatenation
                pass
        
        # Simple fallback: concatenate with weight indicators
        synthesized = "Combined response:\n\n"
        for agent_name, data in sorted(outputs.items(), key=lambda x: x[1].get("weight", 0), reverse=True):
            weight = data.get("weight", 1.0)
            output = data.get("output", "")
            if weight > 0.3:  # Only include significant contributions
                synthesized += f"[{agent_name}] {output}\n\n"
        
        return synthesized
    
    async def generate_agent_reflection(self, agent_name: str) -> Dict[str, Any]:
        """
        Generate a reflection on an agent's performance and characteristics
        
        Args:
            agent_name: Name of the agent to reflect on
            
        Returns:
            Reflection data
        """
        if agent_name not in self.agents:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found"
            }
        
        # Get agent data
        agent = self.agents[agent_name]
        capabilities = self.agent_capabilities[agent_name]["capabilities"]
        invocations = self.integration_metrics["agent_invocations"].get(agent_name, 0)
        token_usage = self.performance_metrics["token_usage"]["by_agent"].get(agent_name, 0)
        
        # Generate reflection
        if self.brain and hasattr(self.brain, "reflection_engine"):
            try:
                reflection = await self.brain.reflection_engine.generate_reflection(
                    topic=f"agent_{agent_name}_reflection",
                    context={
                        "agent_name": agent_name,
                        "capabilities": capabilities,
                        "invocations": invocations,
                        "token_usage": token_usage
                    }
                )
                
                # Store the reflection
                self.reflection_registry["agent_reflections"][agent_name] = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "reflection": reflection
                }
                
                return {
                    "success": True,
                    "agent": agent_name,
                    "reflection": reflection,
                    "metrics": {
                        "invocations": invocations,
                        "token_usage": token_usage
                    }
                }
            except Exception as e:
                logger.error(f"Error generating agent reflection: {str(e)}")
                return {
                    "success": False,
                    "error": str(e)
                }
        else:
            return {
                "success": False,
                "error": "Reflection engine not available"
            }
    
    async def generate_meta_tone_recommendation(self, 
                                            user_input: str, 
                                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a recommendation for which meta-tone to use
        
        Args:
            user_input: User input text
            context: Additional context
            
        Returns:
            Tone recommendation with reasoning
        """
        # Get available tones
        available_tones = list(self.meta_tone_registry.keys())
        
        # Analyze input characteristics
        input_characteristics = await self._analyze_input_characteristics(user_input, context)
        
        # Define basic mapping rules for tone selection
        rules = {
            "technical_content": {
                "condition": lambda c: c.get("technical_content_likelihood", 0) > 0.7,
                "tone": "technical",
                "reason": "High likelihood of technical content"
            },
            "emotional_content": {
                "condition": lambda c: c.get("emotional_content", 0) > 0.7,
                "tone": "empathetic",
                "reason": "High emotional content detected"
            },
            "brevity_requested": {
                "condition": lambda c: c.get("brevity_requested", 0) > 0.7,
                "tone": "concise",
                "reason": "User appears to be requesting brevity"
            },
            "formal_context": {
                "condition": lambda c: c.get("formal_context", 0) > 0.7,
                "tone": "academic",
                "reason": "Formal context detected"
            },
            "casual_context": {
                "condition": lambda c: c.get("casual_context", 0) > 0.7,
                "tone": "casual",
                "reason": "Casual context detected"
            }
        }
        
        # Apply rules
        matched_tone = None
        match_reason = None
        
        for rule_name, rule in rules.items():
            if rule["condition"](input_characteristics):
                matched_tone = rule["tone"]
                match_reason = rule["reason"]
                break
        
        # Default to standard if no match
        if not matched_tone:
            matched_tone = "standard"
            match_reason = "No specific tone indicators detected"
        
        # Get tone details
        tone_details = self.meta_tone_registry.get(matched_tone, {})
        
        return {
            "recommended_tone": matched_tone,
            "reason": match_reason,
            "current_tone": self.current_meta_tone,
            "tone_details": tone_details,
            "input_characteristics": input_characteristics
        }
    
    async def _analyze_input_characteristics(self, 
                                        input_text: str, 
                                        context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Analyze input text for characteristics that inform tone selection
        
        Args:
            input_text: User input text
            context: Additional context
            
        Returns:
            Characteristics scores
        """
        # Initialize default characteristics
        characteristics = {
            "technical_content_likelihood": 0.0,
            "emotional_content": 0.0,
            "brevity_requested": 0.0,
            "formal_context": 0.0,
            "casual_context": 0.0
        }
        
        # Simple text-based heuristics
        text_lower = input_text.lower()
        
        # Technical content indicators
        technical_terms = ["code", "implement", "algorithm", "function", "system", 
                        "technical", "analyze", "data", "problem", "solution"]
        technical_count = sum(1 for term in technical_terms if term in text_lower)
        characteristics["technical_content_likelihood"] = min(1.0, technical_count / 5)
        
        # Emotional content indicators
        emotional_terms = ["feel", "emotion", "happy", "sad", "angry", "worried", 
                         "concerned", "love", "hate", "excited", "afraid"]
        emotional_count = sum(1 for term in emotional_terms if term in text_lower)
        characteristics["emotional_content"] = min(1.0, emotional_count / 3)
        
        # Brevity indicators
        brevity_terms = ["brief", "quick", "short", "summarize", "concise", 
                       "briefly", "tl;dr", "summary"]
        brevity_count = sum(1 for term in brevity_terms if term in text_lower)
        characteristics["brevity_requested"] = min(1.0, brevity_count / 2)
        
        # Formal context indicators
        formal_terms = ["formal", "academic", "professional", "research", "analysis", 
                      "thesis", "paper", "report", "study"]
        formal_count = sum(1 for term in formal_terms if term in text_lower)
        characteristics["formal_context"] = min(1.0, formal_count / 3)
        
        # Casual context indicators
        casual_terms = ["hey", "hi", "chat", "talk", "casual", "friendly", 
                      "just wondering", "curious", "what's up"]
        casual_count = sum(1 for term in casual_terms if term in text_lower)
        characteristics["casual_context"] = min(1.0, casual_count / 3)
        
        # If the brain has an emotional core, use it for more sophisticated analysis
        if self.brain and hasattr(self.brain, "emotional_core"):
            try:
                emotional_stimuli = self.brain.emotional_core.analyze_text_sentiment(input_text)
                characteristics["emotional_content"] = max(
                    characteristics["emotional_content"],
                    emotional_stimuli.get("intensity", 0.0)
                )
            except Exception as e:
                logger.error(f"Error analyzing emotional content: {str(e)}")
        
        # Use more sophisticated analysis if available
        # If we have a dedicated analyzer agent, use it
        analyzer = self.get_agent("input_analyzer")
        if analyzer:
            try:
                # Format input for analyzer
                analyzer_input = f"Analyze the following input for characteristics that inform tone selection:\n\n{input_text}"
                
                # Run analyzer
                result = await Runner.run(analyzer, analyzer_input)
                
                # Try to parse the result as JSON
                if hasattr(result, "final_output"):
                    try:
                        import json
                        parsed = json.loads(result.final_output)
                        if isinstance(parsed, dict):
                            # Update our characteristics with the agent's analysis
                            for key, value in parsed.items():
                                if key in characteristics and isinstance(value, (int, float)):
                                    characteristics[key] = value
                    except (json.JSONDecodeError, ValueError):
                        # Not valid JSON, just ignore
                        pass
            except Exception as e:
                logger.error(f"Error in input analyzer: {str(e)}")
        
        return characteristics
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the agent integration system
        
        Returns:
            Statistics data
        """
        # Calculate average response time
        avg_time = 0.0
        if self.integration_metrics["processing_times"]:
            avg_time = sum(self.integration_metrics["processing_times"]) / len(self.integration_metrics["processing_times"])
            self.performance_metrics["avg_response_time"] = avg_time
        
        # Get most used agents and combinations
        most_used_agents = sorted(
            self.integration_metrics["agent_invocations"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        most_used_combinations = sorted(
            self.integration_metrics["combination_uses"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get most recent meta-tone changes
        recent_tone_changes = self.meta_tone_history[-5:] if self.meta_tone_history else []
        
        # Get token efficiency stats
        token_stats = {
            "total": self.performance_metrics["token_usage"]["total"],
            "by_agent": {k: v for k, v in sorted(
                self.performance_metrics["token_usage"]["by_agent"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]}  # Top 5 agents by token usage
        }
        
        return {
            "registered_agents": len(self.agents),
            "available_meta_tones": list(self.meta_tone_registry.keys()),
            "current_meta_tone": self.current_meta_tone,
            "available_combinations": list(self.combination_patterns.keys()),
            "performance": {
                "avg_response_time": avg_time,
                "success_rate": self.performance_metrics["success_rate"],
                "token_usage": token_stats,
                "total_invocations": sum(self.integration_metrics["agent_invocations"].values()),
                "total_combinations": sum(self.integration_metrics["combination_uses"].values()),
                "tone_changes": len(self.meta_tone_history)
            },
            "top_agents": most_used_agents[:5],
            "top_combinations": most_used_combinations[:5],
            "recent_tone_changes": recent_tone_changes
        }
