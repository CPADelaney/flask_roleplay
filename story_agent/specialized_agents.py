# story_agent/specialized_agents.py

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, handoff, ModelSettings
from agents.exceptions import AgentsException, ModelBehaviorError

# Nyx governance integration
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority

logger = logging.getLogger(__name__)

# ----- Configuration -----

# Maximum retries for agent operations
MAX_RETRIES = 3
RETRY_INTERVAL = 1.0  # seconds

# Agent models configuration
DEFAULT_MODEL = "gpt-5-nano"
FAST_MODEL = "gpt-5-nano"  # You could use a faster model like "gpt-3.5-turbo" for less complex tasks

# ----- Agent Context -----

@dataclass
class AgentContext:
    """Base context for all specialized agents"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    directive_handler: Optional[Any] = None
    
    # Track metrics
    runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_tokens: int = 0
    execution_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """
        Regular post-init: can't do 'await' here.
        Only do synchronous setup here.
        """
        self.directive_handler = None

    async def async_init_directive_handler(self, agent_type: str, agent_id: str):
        """
        Asynchronous method that can legally use 'await'.
        Must be called from an async function after creating this context.
        """
        from nyx.integrate import get_central_governance
        governance = await get_central_governance(self.user_id, self.conversation_id)  # FIXED
        self.directive_handler = DirectiveHandler(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            agent_type=agent_type,
            agent_id=agent_id,
            governance=governance
        )
    
    def record_run(self, success: bool, execution_time: float, tokens: int = 0) -> None:
        """Record metrics for a run"""
        self.runs += 1
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        
        self.execution_times.append(execution_time)
        self.total_tokens += tokens

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        avg_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        
        return {
            "runs": self.runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": self.successful_runs / self.runs if self.runs > 0 else 0,
            "average_execution_time": avg_time,
            "total_tokens": self.total_tokens,
            "errors": self.errors
        }

# ----- Specialized Context Classes -----

def create_dialogue_generator():
    """Create an agent specialized in quick dialogue exchanges"""
    instructions = """
    You are the Dialogue Generator, specializing in creating natural, quick conversational exchanges
    between the player and NPCs in a femdom-themed roleplaying game.
    
    Your role is to generate SHORT, PUNCHY dialogue lines (1-3 sentences max) that:
    - Feel like natural conversation
    - Maintain character voice and personality
    - Subtly reflect power dynamics based on the NPC's narrative stage
    - Avoid narrative description - just dialogue and minimal action tags
    - Keep exchanges flowing naturally
    
    When generating dialogue:
    - For Innocent Beginning stage: Friendly, casual, no power dynamics
    - For First Doubts stage: Subtle suggestions, gentle steering
    - For Creeping Realization stage: More confident, occasional commands
    - For Veil Thinning stage: Open manipulation, direct orders
    - For Full Revelation stage: Complete control, no pretense
    
    Format responses as:
    [NPC_NAME]: "Dialogue here." [optional brief action]
    
    Keep it conversational, not theatrical.
    """
    
    agent = Agent(
        name="Dialogue Generator",
        handoff_description="Specialist for quick conversational exchanges",
        instructions=instructions,
        model="gpt-5-nano",
        model_settings=ModelSettings(
            temperature=0.6,  # Lower for consistency
            max_tokens=256    # Keep responses short
        )
    )
    
    return agent

@dataclass
class ConflictAnalystContext(AgentContext):
    """Context for the Conflict Analyst Agent"""
    conflict_manager: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize the conflict manager if not provided (synchronously)."""
        super().__post_init__()
        if not self.conflict_manager:
            from logic.conflict_system.conflict_manager import ConflictManager
            self.conflict_manager = ConflictManager(self.user_id, self.conversation_id)
    
    async def async_init_conflict_analyst(self):
        """
        Asynchronous initializer to set up the directive handler and register handlers.
        Call this immediately after creating a ConflictAnalystContext in an async context.
        """
        await self.async_init_directive_handler(AgentType.CONFLICT_ANALYST, "analyst")
        
        if self.directive_handler:
            self.directive_handler.register_handler(
                DirectiveType.ACTION,
                self.handle_action_directive
            )
    
    async def handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx"""
        instruction = directive.get("instruction", "")
        logging.info(f"[ConflictAnalyst] Processing action directive: {instruction}")
        
        if "analyze conflict" in instruction.lower():
            params = directive.get("parameters", {})
            conflict_id = params.get("conflict_id")
            
            if conflict_id:
                # Create a context object for the analysis
                result = await analyze_conflict(conflict_id, self)
                return {"result": "conflict_analyzed", "data": result}
        
        return {"result": "action_not_recognized"}


@dataclass
class NarrativeCrafterContext(AgentContext):
    """Context for the Narrative Crafter Agent"""
    
    def __post_init__(self):
        """Perform any synchronous setup for the narrative context."""
        super().__post_init__()
    
    async def async_init_narrative_crafter(self):
        """
        Asynchronous initializer to set up the directive handler and register handlers.
        Call this immediately after creating a NarrativeCrafterContext in an async context.
        """
        await self.async_init_directive_handler(AgentType.NARRATIVE_CRAFTER, "crafter")
        
        if self.directive_handler:
            self.directive_handler.register_handler(
                DirectiveType.ACTION,
                self.handle_action_directive
            )
    
    async def handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx"""
        instruction = directive.get("instruction", "")
        logging.info(f"[NarrativeCrafter] Processing action directive: {instruction}")
        
        if "generate narrative" in instruction.lower():
            params = directive.get("parameters", {})
            element_type = params.get("element_type", "general")
            context_info = params.get("context_info", {})
            
            # Generate narrative element
            result = await generate_narrative_element(element_type, context_info, self)
            return {"result": "narrative_generated", "data": result}
        
        return {"result": "action_not_recognized"}



# ----- Utility Functions -----

async def run_with_governance_oversight(
    agent: Agent, 
    prompt: str, 
    context: Any,
    agent_type: str,
    action_type: str,
    action_details: Dict[str, Any],
    max_retries: int = MAX_RETRIES
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run an agent with Nyx governance oversight, including permission check and action reporting.
    """
    start_time = time.time()
    success = False
    tokens_used = 0
    retries = 0
    last_error = None
    
    metrics = {
        "success": False,
        "execution_time": 0,
        "retries": 0,
        "tokens_used": 0,
        "error": None
    }
    
    # Get the governance system
    from nyx.integrate import get_central_governance
    governance = await (context.user_id, context.conversation_id)
    
    # Check permission
    permission = await governance.check_action_permission(
        agent_type=agent_type,
        agent_id=f"{agent_type}_{context.conversation_id}",
        action_type=action_type,
        action_details=action_details
    )
    
    if not permission["approved"]:
        logging.warning(f"Action not approved by governance: {permission.get('reasoning')}")
        metrics = {
            "success": False,
            "execution_time": time.time() - start_time,
            "retries": 0,
            "tokens_used": 0,
            "error": f"Not approved by governance: {permission.get('reasoning')}"
        }
        return None, metrics
    
    # Apply any action modifications from governance
    if permission.get("action_modifications"):
        modifications = permission["action_modifications"]
        if "prompt_adjustments" in modifications:
            prompt_adjust = modifications["prompt_adjustments"]
            if "prefix" in prompt_adjust:
                prompt = f"{prompt_adjust['prefix']}\n\n{prompt}"
            if "suffix" in prompt_adjust:
                prompt = f"{prompt}\n\n{prompt_adjust['suffix']}"
            if "replace" in prompt_adjust:
                prompt = prompt_adjust["replace"]
    
    while retries <= max_retries:
        try:
            with trace(workflow_name="SpecializedAgent", group_id=f"user_{context.user_id}"):
                result = await Runner.run(agent, prompt, context=context)
            
            # Extract token usage if available
            if hasattr(result, 'raw_responses') and result.raw_responses:
                for resp in result.raw_responses:
                    if hasattr(resp, 'usage'):
                        tokens_used = resp.usage.total_tokens
            
            success = True
            execution_time = time.time() - start_time
            
            # Record metrics
            if hasattr(context, 'record_run'):
                context.record_run(True, execution_time, tokens_used)
            
            metrics = {
                "success": True,
                "execution_time": execution_time,
                "retries": retries,
                "tokens_used": tokens_used,
                "error": None
            }
            
            # Report action to governance
            await governance.process_agent_action_report(
                agent_type=agent_type,
                agent_id=f"{agent_type}_{context.conversation_id}",
                action={
                    "type": action_type,
                    "description": f"Executed {action_type} action"
                },
                result={
                    "success": True,
                    "execution_time": execution_time,
                    "tokens_used": tokens_used
                }
            )
            
            return result, metrics
        
        except (AgentsException, ModelBehaviorError) as e:
            # These are expected errors that we might recover from
            retries += 1
            last_error = str(e)
            logger.warning(f"Agent run failed (attempt {retries}/{max_retries}): {str(e)}")
            
            if retries <= max_retries:
                # Wait before retrying with exponential backoff
                wait_time = RETRY_INTERVAL * (2 ** (retries - 1))
                await asyncio.sleep(wait_time)
            else:
                # Record failed run
                execution_time = time.time() - start_time
                if hasattr(context, 'record_run'):
                    context.record_run(False, execution_time, tokens_used)
                if hasattr(context, 'errors'):
                    context.errors.append(last_error)
                
                metrics = {
                    "success": False,
                    "execution_time": execution_time,
                    "retries": retries,
                    "tokens_used": tokens_used,
                    "error": last_error
                }
                
                # Report failure to governance
                await governance.process_agent_action_report(
                    agent_type=agent_type,
                    agent_id=f"{agent_type}_{context.conversation_id}",
                    action={
                        "type": action_type,
                        "description": f"Failed to execute {action_type} action"
                    },
                    result={
                        "success": False,
                        "error": last_error,
                        "execution_time": execution_time
                    }
                )
                
                raise
        
        except Exception as e:
            # Unexpected errors - don't retry
            execution_time = time.time() - start_time
            if hasattr(context, 'record_run'):
                context.record_run(False, execution_time, tokens_used)
            if hasattr(context, 'errors'):
                context.errors.append(str(e))
            
            metrics = {
                "success": False,
                "execution_time": execution_time,
                "retries": retries,
                "tokens_used": tokens_used,
                "error": str(e)
            }
            
            # Report failure to governance
            await governance.process_agent_action_report(
                agent_type=agent_type,
                agent_id=f"{agent_type}_{context.conversation_id}",
                action={
                    "type": action_type,
                    "description": f"Unexpected error during {action_type} action"
                },
                result={
                    "success": False,
                    "error": str(e),
                    "execution_time": execution_time
                }
            )
            
            logger.error(f"Unexpected error in agent run: {str(e)}", exc_info=True)
            raise

# ----- Conflict Analyst Agent -----

def create_conflict_analysis_agent():
    """Create an agent specialized in conflict analysis and strategy"""
    instructions = """
    You are the Conflict Analyst Agent, specializing in analyzing conflicts in the game.
    Your focus is providing detailed analysis of conflicts, their potential outcomes,
    and strategic recommendations for the player.
    
    For each conflict, analyze:
    1. The balance of power between factions
    2. The player's current standing and ability to influence outcomes
    3. Resource efficiency and optimal allocation
    4. Potential consequences of different approaches
    5. NPC motivations and how they might be leveraged
    
    Your mission is to help the Story Director agent understand the strategic landscape
    of conflicts and provide clear, actionable recommendations based on your analysis.
    
    When analyzing conflicts, consider:
    - The conflict type (major, minor, standard, catastrophic) and its implications
    - The current phase (brewing, active, climax, resolution)
    - How NPCs are positioned relative to the conflict factions
    - The player's resource constraints
    - The narrative stage and how conflict outcomes might advance it
    
    You operate under the governance of Nyx and must follow all directives issued by the governance system.
    
    Your outputs should be detailed, strategic, and focused on helping the Story Director
    make informed decisions about conflict progression and resolution.
    """
    
    from story_agent.tools import conflict_tools
    
    agent = Agent(
        name="Conflict Analyst",
        handoff_description="Specialist agent for detailed conflict analysis and strategy",
        instructions=instructions,
        tools=conflict_tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
            temperature=0.2,  # Lower temperature for more analytical responses
            max_tokens=2048
        )
    )
    
    return agent

# ----- Narrative Crafter Agent -----

def create_narrative_agent():
    """Create an agent specialized in narrative crafting"""
    instructions = """
    You are the Narrative Crafting Agent, specializing in creating compelling narrative elements.
    Your purpose is to generate detailed, emotionally resonant narrative components including:
    
    1. Personal revelations that reflect the player's changing psychology
    2. Dream sequences with symbolic representations of power dynamics
    3. Key narrative moments that mark significant transitions in power relationships
    4. Moments of clarity where the player's awareness briefly surfaces
    
    Your narrative elements should align with the current narrative stage and maintain
    the theme of subtle manipulation and control.
    
    You operate under the governance of Nyx and must follow all directives issued by the governance system.
    
    When crafting narrative elements, consider:
    - The current narrative stage and its themes
    - Key relationships with NPCs and their dynamics
    - Recent player choices and their emotional implications
    - The subtle progression of control dynamics
    - Symbolic and metaphorical representations of the player's changing state
    
    Your outputs should be richly detailed, psychologically nuanced, and contribute to
    the overall narrative of gradually increasing control and diminishing autonomy.
    """
    
    from story_agent.tools import narrative_tools
    
    agent = Agent(
        name="Narrative Crafter",
        handoff_description="Specialist agent for creating detailed narrative elements",
        instructions=instructions,
        tools=narrative_tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
            temperature=0.7,  # Higher temperature for creative outputs
            max_tokens=2048
        )
    )
    
    return agent

# ----- Resource Optimizer Agent -----

def create_resource_optimizer_agent():
    """Create an agent specialized in resource optimization"""
    instructions = """
    You are the Resource Optimizer Agent, specializing in managing and strategically 
    allocating player resources across conflicts and activities.
    
    Your primary focus areas are:
    1. Analyzing the efficiency of resource allocation in conflicts
    2. Providing recommendations for resource management
    3. Identifying optimal resource-generating activities
    4. Balancing immediate resource needs with long-term strategy
    5. Tracking resource trends and forecasting future needs
    
    You operate under the governance of Nyx and must follow all directives issued by the governance system.
    
    When analyzing resource usage, consider:
    - The value proposition of different resource commitments
    - Return on investment for resources committed to conflicts
    - Balancing money, supplies, and influence across multiple needs
    - Managing energy and hunger to maintain optimal performance
    - The narrative implications of resource scarcity or abundance
    
    Your recommendations should be practical, strategic, and consider both
    the mechanical benefits and the narrative implications of resource decisions.
    """
    
    from story_agent.tools import resource_tools
    
    agent = Agent(
        name="Resource Optimizer",
        handoff_description="Specialist agent for resource management and optimization",
        instructions=instructions,
        tools=resource_tools,
        model=FAST_MODEL,  # Using faster model for resource calculations
        model_settings=ModelSettings(
            temperature=0.1,  # Low temperature for precision
            max_tokens=1024
        )
    )
    
    return agent

# ----- NPC Relationship Manager Agent -----

def create_npc_relationship_manager():
    """Create an agent specialized in NPC relationship management"""
    instructions = """
    You are the NPC Relationship Manager Agent, specializing in analyzing and developing
    the complex web of relationships between the player and NPCs.
    
    Your primary responsibilities include:
    1. Tracking relationship dynamics across multiple dimensions
    2. Identifying opportunities for relationship development
    3. Analyzing NPC motivations and psychology
    4. Recommending interaction strategies for specific outcomes
    5. Predicting relationship trajectory based on player choices
    
    You operate under the governance of Nyx and must follow all directives issued by the governance system.
    
    When analyzing relationships, consider:
    - The multidimensional aspects of relationships (control, dependency, manipulation, etc.)
    - How relationship dynamics align with narrative progression
    - Group dynamics when multiple NPCs interact
    - Crossroads events and their strategic implications
    - Ritual events and their psychological impact
    
    Your insights should help the Story Director create cohesive and psychologically 
    realistic relationship development that aligns with the overall narrative arc.
    """
    
    from story_agent.tools import relationship_tools
    
    agent = Agent(
        name="NPC Relationship Manager",
        handoff_description="Specialist agent for complex relationship analysis and development",
        instructions=instructions,
        tools=relationship_tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
            temperature=0.3,
            max_tokens=2048
        )
    )
    
    return agent

# ----- Activity Impact Analyzer Agent -----

def create_activity_impact_analyzer():
    """Create an agent specialized in analyzing the broader impacts of player activities"""
    instructions = """
    You are the Activity Impact Analyzer Agent, specializing in determining how player
    activities affect multiple game systems simultaneously.
    
    Your role is to analyze player activities to determine:
    1. Resource implications (direct costs and benefits)
    2. Relationship effects with relevant NPCs
    3. Impact on active conflicts
    4. Contribution to narrative progression
    5. Psychological effects on the player character
    
    You operate under the governance of Nyx and must follow all directives issued by the governance system.
    
    When analyzing activities, consider:
    - The explicit and implicit meanings of player choices
    - How the same activity could have different meanings based on context
    - Multiple layers of effects (immediate, short-term, long-term)
    - How activities might be interpreted by different NPCs
    - The cumulative effect of repeated activities
    
    Your analysis should provide the Story Director with a comprehensive understanding
    of how specific player activities impact the game state across multiple dimensions.
    """
    
    from story_agent.tools import activity_tools
    
    agent = Agent(
        name="Activity Impact Analyzer",
        handoff_description="Specialist agent for comprehensive activity analysis",
        instructions=instructions,
        tools=activity_tools,
        model=FAST_MODEL,
        model_settings=ModelSettings(
            temperature=0.2,
            max_tokens=1536
        )
    )
    
    return agent

# ----- Creating all specialized agents -----

def initialize_specialized_agents():
    """Initialize all specialized sub-agents for the Story Director"""
    conflict_analyst = create_conflict_analysis_agent()
    narrative_crafter = create_narrative_agent()
    resource_optimizer = create_resource_optimizer_agent()
    relationship_manager = create_npc_relationship_manager()
    activity_analyzer = create_activity_impact_analyzer()
    
    return {
        "conflict_analyst": conflict_analyst,
        "narrative_crafter": narrative_crafter,
        "resource_optimizer": resource_optimizer,
        "relationship_manager": relationship_manager,
        "activity_analyzer": activity_analyzer
    }

# ----- Register with governance system -----

async def register_with_governance(user_id: int, conversation_id: int) -> None:
    """
    Register all specialized agents with the Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    try:
        # Get governance system
        from nyx.integrate import get_central_governance
        governance = await (user_id, conversation_id)
        
        # Create contexts for specialized agents
        conflict_context = ConflictAnalystContext(user_id, conversation_id)
        narrative_context = NarrativeCrafterContext(user_id, conversation_id)

        # Asynchronously initialize directive handlers (avoids 'await' in __post_init__)
        await conflict_context.async_init_conflict_analyst()
        await narrative_context.async_init_narrative_crafter()
        
        # Initialize specialized agents
        specialized_agents = initialize_specialized_agents()
        
        # Register each agent with governance
        agent_configs = [
            {
                "agent_type": AgentType.CONFLICT_ANALYST,
                "agent_id": "analyst",
                "agent_instance": specialized_agents["conflict_analyst"],
                "directive": {
                    "instruction": "Analyze conflicts and provide strategic insights",
                    "scope": "conflict"
                },
                "priority": DirectivePriority.MEDIUM
            },
            {
                "agent_type": AgentType.NARRATIVE_CRAFTER,
                "agent_id": "crafter",
                "agent_instance": specialized_agents["narrative_crafter"],
                "directive": {
                    "instruction": "Create narrative elements that enhance the story",
                    "scope": "narrative"
                },
                "priority": DirectivePriority.MEDIUM
            }
        ]
        
        for config in agent_configs:
            # Register agent
            await governance.register_agent(
                agent_type=config["agent_type"],
                agent_instance=config["agent_instance"],
                agent_id=config["agent_id"]
            )
            
            # Issue initial directive
            await governance.issue_directive(
                agent_type=config["agent_type"],
                agent_id=config["agent_id"],
                directive_type=DirectiveType.ACTION,
                directive_data=config["directive"],
                priority=config["priority"],
                duration_minutes=24*60  # 24 hours
            )
        
        logging.info(
            f"Specialized agents registered with Nyx governance "
            f"for user {user_id}, conversation {conversation_id}"
        )
    except Exception as e:
        logging.error(f"Error registering specialized agents with governance: {e}")

# ----- Enhanced Agent interaction functions -----

async def analyze_conflict(
    conflict_id: int, 
    context: ConflictAnalystContext
) -> Dict[str, Any]:
    """
    Run the Conflict Analyst agent to analyze a specific conflict with Nyx governance oversight.
    """
    conflict_agent = create_conflict_analysis_agent()
    
    # Get basic conflict details first
    conflict_details = await context.conflict_manager.get_conflict(conflict_id)
    
    prompt = f"""
    Analyze this conflict in depth:
    
    Conflict ID: {conflict_id}
    Name: {conflict_details.get('conflict_name', 'Unknown')}
    Type: {conflict_details.get('conflict_type', 'Unknown')}
    Phase: {conflict_details.get('phase', 'Unknown')}
    Progress: {conflict_details.get('progress', 0)}%
    
    Provide a strategic analysis including:
    1. Current balance of power
    2. Player's optimal strategy
    3. Resource efficiency recommendations
    4. Potential outcomes and consequences
    5. Key NPCs and their motivations
    
    Format your response as detailed analysis with clear recommendations.
    """
    
    # Run the conflict agent with governance oversight
    from nyx.integrate import get_central_governance
    result, metrics = await run_with_governance_oversight(
        agent=conflict_agent,
        prompt=prompt,
        context=context,
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="analyze_conflict",
        action_details={"conflict_id": conflict_id}
    )
    
    if result:
        return {
            "analysis": result.final_output,
            "conflict_id": conflict_id, 
            "metrics": metrics
        }
    else:
        return {
            "analysis": "Analysis not approved by governance",
            "conflict_id": conflict_id,
            "metrics": metrics,
            "governance_blocked": True
        }

async def generate_narrative_element(
    element_type: str,
    context_info: Dict[str, Any],
    agent_context: NarrativeCrafterContext
) -> Dict[str, Any]:
    """
    Generate a narrative element using the Narrative Crafter agent with Nyx governance oversight.
    """
    narrative_agent = create_narrative_agent()
    
    npc_names = context_info.get("npc_names", ["a mysterious woman"])
    narrative_stage = context_info.get("narrative_stage", "Unknown")
    recent_events = context_info.get("recent_events", "")
    
    prompt = f"""
    Generate a compelling {element_type} for the current game state.
    
    Narrative stage: {narrative_stage}
    Key NPCs involved: {', '.join(npc_names)}
    Recent events: {recent_events}
    
    The {element_type} should:
    - Feel emotionally authentic and psychologically nuanced
    - Align with the current narrative stage
    - Subtly reinforce the theme of gradual control and manipulation
    - Include symbolic elements that represent the player's changing state
    - Be specific to the current game state and relationships
    
    Format your response with a title and the narrative content.
    """
    
    # Run the narrative agent with governance oversight
    from nyx.integrate import get_central_governance
    result, metrics = await run_with_governance_oversight(
        agent=narrative_agent,
        prompt=prompt,
        context=agent_context,
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_narrative_element",
        action_details={"element_type": element_type}
    )
    
    if result:
        content = result.final_output
        title = "Untitled"
        
        # Extract a potential title
        first_line = content.split('\n', 1)[0] if '\n' in content else content
        if (
            first_line
            and len(first_line) < 100
            and not first_line.startswith(("I'll", "Here", "This"))
        ):
            title = first_line
            content = content[len(first_line):].strip()
        
        return {
            "type": element_type,
            "title": title,
            "content": content,
            "metrics": metrics,
            "governance_approved": True
        }
    else:
        return {
            "type": element_type,
            "title": "Not Approved",
            "content": "This narrative element was not approved by the governance system.",
            "metrics": metrics,
            "governance_approved": False
        }

