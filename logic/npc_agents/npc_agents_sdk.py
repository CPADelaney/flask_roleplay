# logic/npc_agents/npc_agents_sdk.py

from agents import Agent, function_tool, Runner, ModelSettings, trace, InputGuardrail
from agents import GuardrailFunctionOutput, handoff
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class NPCAgentSystem:
    """
    Main system that integrates NPC agents using the OpenAI Agents SDK.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_agents = {}  # Map of npc_id -> Agent
        self._memory_system = None

    async def create_npc_agent(self, npc_id: int, npc_data: dict):
        """Create an Agent for an NPC with personality-based instructions"""
        
        # Create personalized instructions based on NPC traits
        dominance = npc_data.get('dominance', 50)
        cruelty = npc_data.get('cruelty', 50)
        personality_traits = npc_data.get('personality_traits', [])
        
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
            instructions += "\nYou have existing memories and relationships that influence your behavior. Use the retrieve_memories tool to access them.\n"
            
        # Add mask system concepts if applicable
        if npc_data.get('has_mask', False):
            instructions += "\nYou maintain a social mask that may hide your true nature. Your mask integrity affects how well you maintain this facade.\n"
        
        agent = Agent(
            name=npc_data['npc_name'],
            instructions=instructions,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.7),
            tools=[
                function_tool(self.retrieve_memories),
                function_tool(self.get_relationships),
                function_tool(self.perceive_environment),
                function_tool(self.update_emotion),
                function_tool(self.create_memory),
                function_tool(self.check_mask_integrity)
            ],
            # Add hooks for better lifecycle management
            hooks=NPCAgentHooks(npc_id=npc_id, user_id=self.user_id, conversation_id=self.conversation_id)
        )
            
            self.npc_agents[npc_id] = agent
            return agent
    
    async def initialize_agents(self) -> None:
        """Initialize agents for all NPCs in this conversation."""
        npc_ids = await self._get_all_npc_ids()
        for npc_id in npc_ids:
            npc_data = await self._get_npc_data(npc_id)
            agent = await self.create_npc_agent(npc_id, npc_data)
            self.npc_agents[npc_id] = agent

    dominant_agent_template = Agent(
        name="Dominant NPC Template",
        handoff_description="Handles interactions for dominant NPCs",
        instructions="You are a dominant personality NPC...",
        tools=[function_tool(self.command_tools)]
    )
    
    submissive_agent_template = Agent(
        name="Submissive NPC Template",
        handoff_description="Handles interactions for submissive NPCs",
        instructions="You are a submissive personality NPC...",
        tools=[function_tool(self.obedience_tools)]
    )
    
    # Then use these templates as a base and customize per NPC:
    def create_specialized_agent(self, npc_id, npc_data):
        base_template = dominant_agent_template if npc_data.get('dominance', 50) > 60 else submissive_agent_template
        
        # Clone and customize the template
        npc_agent = base_template.clone(
            name=npc_data['npc_name'],
            # Override other parameters as needed
        )
        
        return npc_agent
    
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
    async def check_mask_integrity(self, npc_id: int) -> str:
        """
        Check an NPC's mask integrity.
        
        Args:
            npc_id: The ID of the NPC
            
        Returns:
            JSON string with mask information
        """
        memory_system = await self._get_memory_system()
        mask_info = await memory_system.get_npc_mask(npc_id)
        
        if not mask_info:
            mask_info = {"integrity": 100, "presented_traits": {}, "hidden_traits": {}}
        
        return json.dumps(mask_info)
    
    async def mask_integrity_guardrail(ctx, agent, input_data):
        """Guardrail that checks if an NPC's mask should slip."""
        mask_info = json.loads(await self.check_mask_integrity(npc_id))
        integrity = mask_info.get("integrity", 100)
        
        # Calculate mask slip probability based on emotional state and context
        slip_chance = (100 - integrity) / 200  # Base chance 0-0.5
        
        # Random chance for mask slip
        should_slip = random.random() < slip_chance
        
        # If mask should slip, return relevant information
        return GuardrailFunctionOutput(
            output_info={"should_slip": should_slip, "mask_info": mask_info},
            tripwire_triggered=False  # We don't want to stop processing, just inform
        )
    
      async def handle_group_interaction(self, npc_ids: list[int], 
                                      player_action: dict, 
                                      context: dict) -> dict:
        """
        Handle a player's action that affects multiple NPCs.
        
        Args:
            npc_ids: List of affected NPC IDs
            player_action: Player action data
            context: Additional context
            
        Returns:
            Dictionary with group responses
        """
        # Create or get all required agents
        agents = []
        for npc_id in npc_ids:
            if npc_id in self.npc_agents:
                agents.append(self.npc_agents[npc_id])
            else:
                npc_data = await self._get_npc_data(npc_id)
                agent = await self.create_npc_agent(npc_id, npc_data)
                agents.append(agent)
        
        # Create a coordinator agent that will delegate to NPC agents
        coordinator = Agent(
            name="Group Coordinator",
            instructions="""You coordinate responses from multiple NPCs to a player action.
    Consider each NPC's personality and determine which NPCs should respond and in what order.
    Use handoffs to let each relevant NPC respond in character.""",
            handoffs=agents,
            model="gpt-4o"
        )
        
        # Create input for the coordinator
        npc_names = [agent.name for agent in agents]
        input_text = f"""The player {player_action.get('description', 'did something')}. 
    The following NPCs are present: {', '.join(npc_names)}.
    Coordinate their responses based on their personalities and relationships."""
        
        # Run the coordinator
        result = await Runner.run(coordinator, input_text, context=context)
        
        # Collect and process responses
        responses = []
        for item in result.new_items:
            if item.type == "message_output_item":
                responses.append({
                    "type": "group_response",
                    "content": item.raw_item.content[0].text
                })
        
        return {"npc_responses": responses}
    
      async def handle_player_action(self, player_action: dict, context: dict = None) -> dict:
        """
        Handle a player's action directed at one or more NPCs.
        
        Args:
            player_action: Information about the player's action
            context: Additional context
            
        Returns:
            Dictionary with NPC responses
        """
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
            
            # Create input for the agent
            input_text = f"The player {player_action.get('description', 'did something')}. How do you respond?"
            
            # Use the Runner to get a response
            result = await Runner.run(agent, input_text, context=context_obj)
            
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
    
    @function_tool
    async def get_relationships(self, npc_id: int) -> str:
        """
        Get an NPC's relationships with other entities.
        
        Args:
            npc_id: The ID of the NPC
            
        Returns:
            JSON string with relationship data
        """
        relationship_manager = NPCRelationshipManager(
            npc_id, self.user_id, self.conversation_id
        )
        
        relationships = await relationship_manager.update_relationships({})
        return json.dumps(relationships)
    
    @function_tool
    async def update_relationship(self, npc_id: int, entity_type: str, entity_id: int, 
                                  action_type: str, level_change: int) -> str:
        """
        Update a relationship between an NPC and another entity.
        
        Args:
            npc_id: The ID of the NPC
            entity_type: Type of the other entity ('npc' or 'player')
            entity_id: ID of the other entity
            action_type: Type of action that occurred
            level_change: Change in relationship level
            
        Returns:
            JSON string with update result
        """
        relationship_manager = NPCRelationshipManager(
            npc_id, self.user_id, self.conversation_id
        )
        
        # Create dummy actions for the update
        player_action = {"type": action_type, "description": f"performed {action_type}"}
        npc_action = {"type": "react", "description": "reacted to action"}
        
        result = await relationship_manager.update_relationship_from_interaction(
            entity_type, entity_id, player_action, npc_action
        )
        
        return json.dumps(result)
    
    @function_tool
    async def perceive_environment(self, npc_id: int, context: dict = None) -> str:
        """
        Perceive the environment around an NPC.
        
        Args:
            npc_id: The ID of the NPC
            context: Additional context information
            
        Returns:
            JSON string with environment data
        """
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
        
        return json.dumps(environment_data)
    
    @function_tool
    async def retrieve_memories(self, npc_id: int, query: str, limit: int = 5) -> str:
        """
        Retrieve memories for an NPC based on a query.
        
        Args:
            npc_id: The ID of the NPC
            query: Search term for memory retrieval
            limit: Maximum number of memories to return
            
        Returns:
            JSON string of relevant memories
        """
        memory_system = await self._get_memory_system()
        memory_result = await memory_system.recall(
            entity_type="npc",
            entity_id=npc_id,
            query=query,
            limit=limit
        )
        
        return json.dumps(memory_result.get("memories", []))
    
    @function_tool
    async def create_memory(self, npc_id: int, memory_text: str, importance: str = "medium", 
                            emotional: bool = False, tags: list = None) -> str:
        """
        Create a new memory for an NPC.
        
        Args:
            npc_id: The ID of the NPC
            memory_text: The text of the memory to create
            importance: Importance level ('low', 'medium', 'high')
            emotional: Whether this is an emotional memory
            tags: List of tags to associate with the memory
            
        Returns:
            JSON string with the memory creation result
        """
        memory_system = await self._get_memory_system()
        result = await memory_system.remember(
            entity_type="npc",
            entity_id=npc_id,
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags or []
        )
        
        return json.dumps({"success": True, "memory_id": result.get("memory_id")})
    
    # Define input models for structured data
    class PlayerAction(BaseModel):
        type: str
        description: str
        target_npc_id: Optional[int] = None
        target_location: Optional[str] = None

    class PlayerInputValidation(BaseModel):
        is_appropriate: bool
        reasoning: str
    
    def player_input_guardrail(ctx, agent, input_data):
        # Analyze if player input is appropriate for the game context
        # This replaces some of your safety checks from the old system
        analysis = {"is_appropriate": True, "reasoning": "Input appears appropriate"}
        
        # Check for inappropriate content based on your game's guidelines
        if "inappropriate_terms" in input_data.lower():
            analysis["is_appropriate"] = False
            analysis["reasoning"] = "Input contains inappropriate terms"
        
        return GuardrailFunctionOutput(
            output_info=analysis,
            tripwire_triggered=not analysis["is_appropriate"]
        )
