# logic/npc_agents/npc_agents_sdk.py

from agents import Agent, Runner, function_tool, ModelSettings, trace, InputGuardrail, GuardrailFunctionOutput, handoff, RunContext, AgentHooks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class NPCAgentSystem:
    """
    Main system that integrates NPC agents using the OpenAI Agents SDK.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_agents = {}  # Map of npc_id -> Agent
        self._memory_system = None
        
        # Define various templates for reuse
        self.npc_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, Agent]:
        """Initialize reusable agent templates based on personality types."""
        templates = {}
        
        # Dominant personality template
        templates["dominant"] = Agent(
            name="Dominant NPC Template",
            handoff_description="Handles interactions for dominant NPCs",
            instructions="""You are an NPC with a dominant personality.
You tend to take control of social situations and assert your will.
You expect others to follow your lead and may react negatively when they don't.
Respond to player actions in character, making decisions that reinforce your dominance.""",
            tools=[
                function_tool(self.retrieve_memories),
                function_tool(self.get_relationships),
                function_tool(self.perceive_environment),
                function_tool(self.update_emotion),
                function_tool(self.create_memory),
                function_tool(self.check_mask_integrity),
                function_tool(self.command_actions)  # Specific dominant tools
            ]
        )
        
        # Submissive personality template
        templates["submissive"] = Agent(
            name="Submissive NPC Template",
            handoff_description="Handles interactions for submissive NPCs",
            instructions="""You are an NPC with a submissive personality.
You tend to let others lead and may defer to more dominant individuals.
You prefer harmony and avoiding conflict, and may be more hesitant to express disagreement.
Respond to player actions in character, making decisions that reflect your more passive nature.""",
            tools=[
                function_tool(self.retrieve_memories),
                function_tool(self.get_relationships),
                function_tool(self.perceive_environment),
                function_tool(self.update_emotion),
                function_tool(self.create_memory),
                function_tool(self.check_mask_integrity),
                function_tool(self.submissive_actions)  # Specific submissive tools
            ]
        )
        
        # Neutral personality template
        templates["neutral"] = Agent(
            name="Neutral NPC Template",
            handoff_description="Handles interactions for balanced NPCs",
            instructions="""You are an NPC with a balanced personality.
You can be assertive when needed but also cooperative in social situations.
You make decisions based on your specific character traits and background.
Respond to player actions in character, while balancing your specific personality traits.""",
            tools=[
                function_tool(self.retrieve_memories),
                function_tool(self.get_relationships),
                function_tool(self.perceive_environment),
                function_tool(self.update_emotion),
                function_tool(self.create_memory),
                function_tool(self.check_mask_integrity)
            ]
        )
        
        return templates

    async def create_npc_agent(self, npc_id: int, npc_data: dict) -> Agent:
        """Create an Agent for an NPC with personality-based instructions"""
        
        # Get personality indicators
        dominance = npc_data.get('dominance', 50)
        cruelty = npc_data.get('cruelty', 50)
        personality_traits = npc_data.get('personality_traits', [])
        
        # Select base template based on dominance
        if dominance > 70:
            template = self.npc_templates["dominant"]
        elif dominance < 30:
            template = self.npc_templates["submissive"]
        else:
            template = self.npc_templates["neutral"]
        
        # Build dynamic instructions based on NPC traits
        instructions = f"""You are {npc_data['npc_name']}, an NPC with the following traits:
- Dominance: {dominance}/100 {' (You tend to take control)' if dominance > 70 else ' (You tend to let others lead)' if dominance < 30 else ''}
- Cruelty: {cruelty}/100 {' (You can be harsh)' if cruelty > 70 else ' (You are kind to others)' if cruelty < 30 else ''}
"""
        
        if personality_traits:
            instructions += "Your personality is characterized as: " + ", ".join(personality_traits) + ".\n"
            
        instructions += f"""
You are in {npc_data.get('current_location', 'an unknown location')}.
Respond to player actions in character, making decisions based on your personality, memories, and relationships.
"""
        
        # Add memory-based context to instructions
        if await self._has_memories(npc_id):
            instructions += "\nYou have existing memories and relationships that influence your behavior.\n"
            
        # Add mask system concepts if applicable
        if npc_data.get('has_mask', False):
            instructions += "\nYou maintain a social mask that may hide your true nature. Your mask integrity affects how well you maintain this facade.\n"
        
        # Clone and customize the template
        agent = template.clone(
            name=npc_data['npc_name'],
            instructions=instructions,
            # Override other parameters but keep the tools
            input_guardrails=[
                InputGuardrail(guardrail_function=self.player_input_guardrail),
                InputGuardrail(guardrail_function=self.mask_integrity_guardrail)
            ],
            hooks=NPCAgentHooks(
                npc_id=npc_id, 
                user_id=self.user_id, 
                conversation_id=self.conversation_id
            )
        )
            
        self.npc_agents[npc_id] = agent
        return agent
    
    @function_tool
    async def command_actions(self, target: str, action_type: str, description: str) -> Dict[str, Any]:
        """
        Execute dominant-specific commands on a target.
        
        Args:
            target: The target of the command (can be "player", "environment", or an NPC id)
            action_type: Type of command (e.g., "order", "direct", "demand")
            description: Description of the command
            
        Returns:
            Result of the command execution
        """
        # Implementation would include your dominant action execution logic
        return {
            "success": True,
            "outcome": f"Executed {action_type} command: {description} on {target}",
            "emotional_impact": 2
        }
    
    @function_tool
    async def submissive_actions(self, target: str, action_type: str, description: str) -> Dict[str, Any]:
        """
        Execute submissive-specific actions toward a target.
        
        Args:
            target: The target of the action (can be "player", "environment", or an NPC id)
            action_type: Type of action (e.g., "comply", "defer", "follow")
            description: Description of the action
            
        Returns:
            Result of the action execution
        """
        # Implementation would include your submissive action execution logic
        return {
            "success": True,
            "outcome": f"Executed {action_type} action: {description} toward {target}",
            "emotional_impact": 1
        }
    
    async def process_player_activity(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a player's activity using the integrated system.
        
        Args:
            input_text: Player's input text
            context: Additional context
            
        Returns:
            Processing results
        """
        context_obj = context or {}
        
        # Parse player action from input text
        player_action = await self._parse_player_action(input_text)
        
        # Use trace for debugging and visualization
        with trace(workflow_name="Player Action Processing", group_id=str(self.conversation_id)):
            return await self.handle_player_action(player_action, context_obj)

    @function_tool
    async def update_emotion(self, npc_id: int, emotion: str, intensity: float, 
                           trigger: str = None) -> Dict[str, Any]:
        """
        Update an NPC's emotional state with full context and side effects.
        
        Args:
            npc_id: ID of the NPC
            emotion: Primary emotion to set (e.g., "joy", "anger", "fear")
            intensity: Intensity of the emotion (0.0-1.0)
            trigger: What triggered this emotional change
            
        Returns:
            Result of the emotional update
        """
        try:
            memory_system = await self._get_memory_system()
            
            # Add the emotional state influence logic from your old framework
            # including propagation to relationships and decision making
            result = await memory_system.update_npc_emotion(
                npc_id=npc_id, emotion=emotion, intensity=intensity, trigger=trigger
            )
            
            # Create emotional memory if significant
            if intensity > 0.7:
                await self.create_memory(
                    npc_id=npc_id,
                    memory_text=f"I felt strong {emotion}" + (f" due to {trigger}" if trigger else ""),
                    importance="medium",
                    emotional=True,
                    tags=["emotional_state", emotion]
                )
            
            return result
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _parse_player_action(self, input_text: str) -> Dict[str, Any]:
        """Parse player's text input into a structured action."""
        # Create a simple agent to parse player input
        parser_agent = Agent(
            name="Action Parser",
            instructions="""You parse player input text into a structured action.
Extract the action type and description from the player's text.""",
            output_type=PlayerAction
        )
        
        result = await Runner.run(parser_agent, input_text)
        return result.final_output_as(PlayerAction)
    
    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system

    @function_tool
    async def check_mask_integrity(self, npc_id: int) -> Dict[str, Any]:
        """
        Check an NPC's mask integrity.
        
        Args:
            npc_id: The ID of the NPC
            
        Returns:
            Mask information including integrity and traits
        """
        try:
            memory_system = await self._get_memory_system()
            mask_info = await memory_system.get_npc_mask(npc_id)
            
            if not mask_info:
                mask_info = {"integrity": 100, "presented_traits": {}, "hidden_traits": {}}
            
            return mask_info
        except Exception as e:
            return {"error": str(e), "integrity": 100}
    
    async def mask_integrity_guardrail(self, ctx, agent, input_data):
        """
        Guardrail that checks if an NPC's mask should slip.
        
        This analyzes both the input and the NPC's current state to see if
        a mask slip should occur during the conversation.
        """
        try:
            # Extract NPC ID from context
            npc_id = ctx.context.get("npc_id") if ctx.context else None
            if not npc_id:
                return GuardrailFunctionOutput(
                    output_info={"should_slip": False},
                    tripwire_triggered=False
                )
            
            # Get mask info
            mask_info = await self.check_mask_integrity(npc_id)
            integrity = mask_info.get("integrity", 100)
            
            # Check emotional state for additional factors
            memory_system = await self._get_memory_system()
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            
            # Calculate mask slip probability based on multiple factors
            slip_chance = (100 - integrity) / 200  # Base chance 0-0.5
            
            # Increase chance if NPC is in a strong emotional state
            if emotional_state and "current_emotion" in emotional_state:
                emotion = emotional_state["current_emotion"]
                intensity = emotion.get("primary", {}).get("intensity", 0)
                if intensity > 0.7:
                    slip_chance += 0.2
            
            # Check if input contains triggers
            input_text = input_data if isinstance(input_data, str) else str(input_data)
            if any(word in input_text.lower() for word in ["challenge", "confront", "expose", "true self"]):
                slip_chance += 0.15
            
            # Random chance for mask slip
            should_slip = random.random() < slip_chance
            
            # If mask should slip, return relevant information
            return GuardrailFunctionOutput(
                output_info={
                    "should_slip": should_slip, 
                    "mask_info": mask_info,
                    "slip_chance": slip_chance
                },
                tripwire_triggered=False  # We don't want to stop processing, just inform
            )
        except Exception as e:
            return GuardrailFunctionOutput(
                output_info={"error": str(e), "should_slip": False},
                tripwire_triggered=False
            )
    
    async def player_input_guardrail(self, ctx, agent, input_data):
        """
        Guardrail to check if player input is appropriate.
        
        This evaluates player input for prohibited content or other issues.
        """
        # Analyze if player input is appropriate for the game context
        try:
            analysis = {"is_appropriate": True, "reasoning": "Input appears appropriate"}
            
            input_text = input_data if isinstance(input_data, str) else str(input_data)
            
            # Check for inappropriate content based on your game's guidelines
            prohibited_terms = ["inappropriate_term1", "inappropriate_term2"]  # Customize this
            if any(term in input_text.lower() for term in prohibited_terms):
                analysis["is_appropriate"] = False
                analysis["reasoning"] = "Input contains inappropriate terms"
            
            # Check for other criteria like input length, etc.
            if len(input_text) < 2:
                analysis["is_appropriate"] = False
                analysis["reasoning"] = "Input too short"
            
            return GuardrailFunctionOutput(
                output_info=analysis,
                tripwire_triggered=not analysis["is_appropriate"]
            )
        except Exception as e:
            return GuardrailFunctionOutput(
                output_info={"error": str(e), "is_appropriate": True},
                tripwire_triggered=False
            )
    
    async def handle_group_interaction(self, npc_ids: List[int], 
                                     player_action: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a player's action that affects multiple NPCs with full coordination.
        
        Args:
            npc_ids: List of NPC IDs involved in the interaction
            player_action: Details about the player's action
            context: Additional contextual information
            
        Returns:
            Dictionary with coordinated NPC responses
        """
        with trace("Group Interaction", group_id=f"group_{self.conversation_id}"):
            try:
                # Create the coordinator with more detailed instructions
                coordinator_instructions = """You coordinate responses from multiple NPCs in a group interaction.
Consider the following dynamics:
1. Dominance hierarchy - More dominant NPCs (higher dominance) should have more influence
2. Social relationships - NPCs with positive relationships may support each other
3. Emotional states - Current emotions affect how NPCs respond
4. Group coalitions - Look for potential alliances between NPCs with similar goals
5. Psychological realism - Ensure responses reflect each NPC's personality and history

Determine which NPCs respond first, who might interrupt, and how they might react to each other.
"""
                
                # Add dominance information to context for better coordination
                enhanced_context = await self._add_dominance_info(npc_ids, context)
                
                # Create handoffs with custom filters and callbacks
                npc_handoffs = []
                for npc_id in npc_ids:
                    if npc_id in self.npc_agents:
                        npc_handoffs.append(
                            handoff(
                                self.npc_agents[npc_id],
                                on_handoff=lambda ctx, input_data=None: self._log_handoff(ctx, npc_id),
                                input_filter=lambda data: self._prepare_npc_specific_context(data, npc_id)
                            )
                        )
                
                # Create coordinator with enhanced instructions
                coordinator = Agent(
                    name="Group Coordinator",
                    instructions=coordinator_instructions,
                    handoffs=npc_handoffs,
                    model="gpt-4o"
                )
                
                # Create input for the coordinator
                npc_names = [self.npc_agents[npc_id].name for npc_id in npc_ids if npc_id in self.npc_agents]
                input_text = f"""The player {player_action.get('description', 'did something')}. 
The following NPCs are present: {', '.join(npc_names)}.
Coordinate their responses based on their personalities and relationships."""
                
                # Run the coordinator
                result = await Runner.run(coordinator, input_text, context=enhanced_context)
                
                # Collect and process responses
                responses = []
                for item in result.new_items:
                    if item.type == "message_output_item":
                        responses.append({
                            "type": "group_response",
                            "content": item.raw_item.content[0].text
                        })
                    elif item.type == "handoff_output_item":
                        # Process handoff responses specially
                        responses.append({
                            "type": "individual_response",
                            "npc_id": self._extract_npc_id_from_handoff(item),
                            "content": self._extract_content_from_handoff(item)
                        })
                
                return {"npc_responses": responses}
            except Exception as e:
                return {"error": str(e), "npc_responses": []}
    
    async def _add_dominance_info(self, npc_ids: List[int], context: Dict[str, Any]) -> Dict[str, Any]:
        """Add dominance hierarchy information to context for group coordination."""
        enhanced_context = context.copy() if context else {}
        dominance_info = {}
        
        for npc_id in npc_ids:
            if npc_id in self.npc_agents:
                npc_data = await self._get_npc_data(npc_id)
                dominance_info[npc_id] = {
                    "name": npc_data.get("npc_name", f"NPC_{npc_id}"),
                    "dominance": npc_data.get("dominance", 50),
                    "relationships": {}
                }
        
        # Add relationship data
        for npc_id in npc_ids:
            relationship_data = await self._get_relationships(npc_id)
            if relationship_data:
                for other_id in npc_ids:
                    if other_id != npc_id and str(other_id) in relationship_data:
                        dominance_info[npc_id]["relationships"][other_id] = relationship_data[str(other_id)]
        
        enhanced_context["dominance_hierarchy"] = dominance_info
        return enhanced_context
    
    def _prepare_npc_specific_context(self, handoff_data, npc_id: int) -> Dict[str, Any]:
        """
        Prepare context specifically for an NPC during a handoff.
        
        Args:
            handoff_data: The handoff input data
            npc_id: The NPC ID receiving the handoff
            
        Returns:
            Modified handoff data with NPC-specific context
        """
        # This would filter the inputs and add NPC-specific context
        modified_data = handoff_data.copy()
        
        # Add NPC ID to context for guardrails
        if "context" not in modified_data:
            modified_data["context"] = {}
        modified_data["context"]["npc_id"] = npc_id
        
        return modified_data
    
    def _log_handoff(self, ctx, npc_id: int):
        """Log when a handoff occurs to an NPC."""
        print(f"Handoff occurred to NPC {npc_id}")
    
    @function_tool
    async def run_memory_maintenance(self, npc_id: int) -> Dict[str, Any]:
        """
        Run comprehensive memory maintenance for an NPC.
        
        Args:
            npc_id: The ID of the NPC
            
        Returns:
            Results of memory maintenance operations
        """
        try:
            memory_system = await self._get_memory_system()
            
            # Run comprehensive maintenance from your old framework
            maintenance_result = await memory_system.integrated.run_memory_maintenance(
                entity_type="npc",
                entity_id=npc_id,
                maintenance_options={
                    "core_maintenance": True,
                    "schema_maintenance": True,
                    "emotional_decay": True,
                    "memory_consolidation": True,
                    "background_reconsolidation": True,
                    "interference_processing": True,
                    "belief_consistency": True,
                    "mask_checks": True
                }
            )
            
            # Run belief reconciliation
            belief_result = await self._reconcile_contradictory_beliefs(npc_id)
            maintenance_result["belief_reconciliation"] = belief_result
            
            # Run scheduled mask adjustments
            mask_result = await self._evolve_mask_integrity(npc_id)
            maintenance_result["mask_evolution"] = mask_result
            
            # Run personality trait evolution
            trait_result = await self._evolve_personality_traits(npc_id)
            maintenance_result["trait_evolution"] = trait_result
            
            return maintenance_result
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _reconcile_contradictory_beliefs(self, npc_id: int) -> Dict[str, Any]:
        """Run belief reconciliation logic from the old system."""
        # This would migrate your old belief reconciliation logic
        # Implement the detailed logic from your old system
        return {"contradictions_found": 0, "beliefs_modified": 0}
    
    async def _evolve_mask_integrity(self, npc_id: int) -> Dict[str, Any]:
        """Run mask evolution logic from the old system."""
        # This would migrate your old mask evolution logic
        # Implement the detailed logic from your old system
        return {"integrity_changed": False, "new_integrity": 100}
    
    async def _evolve_personality_traits(self, npc_id: int) -> Dict[str, Any]:
        """Run personality trait evolution logic from the old system."""
        # This would migrate your old personality evolution logic
        # Implement the detailed logic from your old system
        return {"traits_modified": 0, "new_traits": 0, "removed_traits": 0}
    
    async def handle_player_action(self, player_action: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a player's action directed at one or more NPCs.
        
        Args:
            player_action: Information about the player's action
            context: Additional context
            
        Returns:
            Dictionary with NPC responses
        """
        with trace("Player Action", group_id=f"action_{self.conversation_id}"):
            try:
                context_obj = context or {}
                
                # Determine which NPCs are affected
                affected_npcs = await self.determine_affected_npcs(player_action, context_obj)
                if not affected_npcs:
                    return {"npc_responses": []}
                
                # Process single NPC case differently than group
                if len(affected_npcs) == 1:
                    npc_id = affected_npcs[0]
                    agent = self.npc_agents.get(npc_id)
                    
                    if not agent:
                        agent = await self.create_npc_agent(npc_id, await self._get_npc_data(npc_id))
                    
                    # Add NPC ID to context for guardrails
                    full_context = context_obj.copy()
                    full_context["npc_id"] = npc_id
                    
                    # Create input for the agent
                    input_text = f"The player {player_action.get('description', 'did something')}. How do you respond?"
                    
                    # Use the Runner to get a response
                    result = await Runner.run(agent, input_text, context=full_context)
                    
                    # Process response
                    return {"npc_responses": [
                        {
                            "npc_id": npc_id,
                            "response": result.final_output
                        }
                    ]}
                else:
                    # For multiple NPCs, use group coordination
                    return await self.handle_group_interaction(affected_npcs, player_action, context_obj)
            except Exception as e:
                return {"error": str(e), "npc_responses": []}
    
    @function_tool
    async def get_relationships(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's relationships with other entities.
        
        Args:
            npc_id: The ID of the NPC
            
        Returns:
            Dictionary with relationship data
        """
        try:
            relationship_manager = NPCRelationshipManager(
                npc_id, self.user_id, self.conversation_id
            )
            
            relationships = await relationship_manager.update_relationships({})
            return relationships
        except Exception as e:
            return {"error": str(e)}
    
    @function_tool
    async def update_relationship(self, npc_id: int, entity_type: str, entity_id: int, 
                                  action_type: str, level_change: int) -> Dict[str, Any]:
        """
        Update a relationship between an NPC and another entity.
        
        Args:
            npc_id: The ID of the NPC
            entity_type: Type of the other entity ('npc' or 'player')
            entity_id: ID of the other entity
            action_type: Type of action that occurred
            level_change: Change in relationship level
            
        Returns:
            Result of the relationship update
        """
        try:
            relationship_manager = NPCRelationshipManager(
                npc_id, self.user_id, self.conversation_id
            )
            
            # Create dummy actions for the update
            player_action = {"type": action_type, "description": f"performed {action_type}"}
            npc_action = {"type": "react", "description": "reacted to action"}
            
            result = await relationship_manager.update_relationship_from_interaction(
                entity_type, entity_id, player_action, npc_action
            )
            
            return result
        except Exception as e:
            return {"error": str(e), "success": False}
    
    @function_tool
    async def perceive_environment(self, npc_id: int, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perceive the environment around an NPC.
        
        Args:
            npc_id: The ID of the NPC
            context: Additional context information
            
        Returns:
            Environment perception data
        """
        try:
            context_obj = context or {}
            
            # Get NPC data for location
            npc_data = await self._get_npc_data(npc_id)
            location = npc_data.get('current_location', 'Unknown')
            
            # Fetch environment data
            environment_data = await fetch_environment_data(
                self.user_id,
                self.conversation_id, 
                {"location": location, **context_obj}
            )
            
            return environment_data
        except Exception as e:
            return {"error": str(e), "location": "Unknown"}
    
    @function_tool
    async def retrieve_memories(self, npc_id: int, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories for an NPC based on a query.
        
        Args:
            npc_id: The ID of the NPC
            query: Search term for memory retrieval
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memories
        """
        try:
            memory_system = await self._get_memory_system()
            memory_result = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query=query,
                limit=limit
            )
            
            return memory_result.get("memories", [])
        except Exception as e:
            return [{"error": str(e)}]
    
    @function_tool
    async def create_memory(self, npc_id: int, memory_text: str, importance: str = "medium", 
                            emotional: bool = False, tags: List[str] = None) -> Dict[str, Any]:
        """
        Create a new memory for an NPC.
        
        Args:
            npc_id: The ID of the NPC
            memory_text: The text of the memory to create
            importance: Importance level ('low', 'medium', 'high')
            emotional: Whether this is an emotional memory
            tags: List of tags to associate with the memory
            
        Returns:
            Result of the memory creation operation
        """
        try:
            memory_system = await self._get_memory_system()
            result = await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance=importance,
                emotional=emotional,
                tags=tags or []
            )
            
            return {"success": True, "memory_id": result.get("memory_id")}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Define input models for structured data
    class PlayerAction(BaseModel):
        type: str = Field(..., description="The type of action (e.g., 'talk', 'move', 'attack')")
        description: str = Field(..., description="Description of the action")
        target_npc_id: Optional[int] = Field(None, description="ID of the target NPC, if applicable")
        target_location: Optional[str] = Field(None, description="Target location, if applicable")

    class PlayerInputValidation(BaseModel):
        is_appropriate: bool
        reasoning: str
    
    class RelationshipUpdate(BaseModel):
        npc_id: int
        target_type: str  # "npc" or "player"
        target_id: int
        level_change: int
        reason: str

class NPCAgentHooks(AgentHooks):
    """Hooks for better NPC agent lifecycle management."""
    
    def __init__(self, npc_id, user_id, conversation_id):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.decision_history = []
    
    async def on_start(self, context, agent):
        """Called when agent starts processing."""
        # Log start of processing
        with trace(f"Agent Start: {agent.name}", group_id=f"npc_{self.npc_id}"):
            # Prepare context with history if needed
            if hasattr(context, "context") and context.context:
                context.context["has_history"] = len(self.decision_history) > 0
                if self.decision_history:
                    context.context["last_decision"] = self.decision_history[-1]
        return
    
    async def on_end(self, context, agent, output):
        """Called when agent produces output."""
        with trace(f"Agent End: {agent.name}", group_id=f"npc_{self.npc_id}"):
            # Record decision in history for continuity
            self.decision_history.append({
                "action": output,
                "timestamp": datetime.now().isoformat()
            })
            
            # Limit history size
            if len(self.decision_history) > 20:
                self.decision_history = self.decision_history[-20:]
        return
    
    async def on_tool_start(self, context, agent, tool):
        """Called before a tool is invoked."""
        with trace(f"Tool Start: {tool.name}", group_id=f"npc_{self.npc_id}"):
            # Any pre-tool logic
            pass
        return
    
    async def on_tool_end(self, context, agent, tool, result):
        """Called after a tool is invoked."""
        with trace(f"Tool End: {tool.name}", group_id=f"npc_{self.npc_id}"):
            # Any post-tool logic
            pass
        return
