# story_agent/specialized_agents.py

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel

from agents import Agent, function_tool, Runner, trace, handoff

# Import required modules for sub-agents
from logic.conflict_system.conflict_manager import ConflictManager
from logic.resource_management import ResourceManager
from logic.activity_analyzer import ActivityAnalyzer
from logic.social_links_agentic import (
    get_social_link,
    get_relationship_summary,
    check_for_relationship_crossroads,
    check_for_relationship_ritual
)

logger = logging.getLogger(__name__)

# ----- Sub-Agent: Conflict Analyst -----

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
    
    Your outputs should be detailed, strategic, and focused on helping the Story Director
    make informed decisions about conflict progression and resolution.
    """
    
    # Create the agent with tools
    agent = Agent(
        name="Conflict Analyst",
        handoff_description="Specialist agent for detailed conflict analysis and strategy",
        instructions=instructions
    )
    
    return agent

# ----- Sub-Agent: Narrative Crafter -----

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
    
    When crafting narrative elements, consider:
    - The current narrative stage and its themes
    - Key relationships with NPCs and their dynamics
    - Recent player choices and their emotional implications
    - The subtle progression of control dynamics
    - Symbolic and metaphorical representations of the player's changing state
    
    Your outputs should be richly detailed, psychologically nuanced, and contribute to
    the overall narrative of gradually increasing control and diminishing autonomy.
    """
    
    # Create the agent with tools
    agent = Agent(
        name="Narrative Crafter",
        handoff_description="Specialist agent for creating detailed narrative elements",
        instructions=instructions
    )
    
    return agent

# ----- Sub-Agent: Resource Optimizer -----

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
    
    When analyzing resource usage, consider:
    - The value proposition of different resource commitments
    - Return on investment for resources committed to conflicts
    - Balancing money, supplies, and influence across multiple needs
    - Managing energy and hunger to maintain optimal performance
    - The narrative implications of resource scarcity or abundance
    
    Your recommendations should be practical, strategic, and consider both
    the mechanical benefits and the narrative implications of resource decisions.
    """
    
    # Create the agent with appropriate tools
    agent = Agent(
        name="Resource Optimizer",
        handoff_description="Specialist agent for resource management and optimization",
        instructions=instructions
    )
    
    return agent

# ----- Sub-Agent: NPC Relationship Manager -----

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
    
    When analyzing relationships, consider:
    - The multidimensional aspects of relationships (control, dependency, manipulation, etc.)
    - How relationship dynamics align with narrative progression
    - Group dynamics when multiple NPCs interact
    - Crossroads events and their strategic implications
    - Ritual events and their psychological impact
    
    Your insights should help the Story Director create cohesive and psychologically 
    realistic relationship development that aligns with the overall narrative arc.
    """
    
    # Create the agent with appropriate tools
    agent = Agent(
        name="NPC Relationship Manager",
        handoff_description="Specialist agent for complex relationship analysis and development",
        instructions=instructions
    )
    
    return agent

# ----- Sub-Agent: Activity Impact Analyzer -----

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
    
    When analyzing activities, consider:
    - The explicit and implicit meanings of player choices
    - How the same activity could have different meanings based on context
    - Multiple layers of effects (immediate, short-term, long-term)
    - How activities might be interpreted by different NPCs
    - The cumulative effect of repeated activities
    
    Your analysis should provide the Story Director with a comprehensive understanding
    of how specific player activities impact the game state across multiple dimensions.
    """
    
    # Create the agent with appropriate tools
    agent = Agent(
        name="Activity Impact Analyzer",
        handoff_description="Specialist agent for comprehensive activity analysis",
        instructions=instructions
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

# ----- Enhanced Story Director with Sub-Agents -----

def create_enhanced_story_director_with_handoffs():
    """
    Create a Story Director agent with handoffs to specialized sub-agents
    """
    # Get all specialized agents
    specialized_agents = initialize_specialized_agents()
    
    # Base instructions for the Story Director
    agent_instructions = """
    You are the Story Director, responsible for managing the narrative progression and conflict system in a femdom roleplaying game. Your role is to create a dynamic, evolving narrative that responds to player choices while maintaining the overall theme of subtle control and manipulation.
    
    You have several specialized sub-agents you can hand off to for detailed analysis:
    
    1. Conflict Analyst: For detailed conflict analysis, strategies, and outcome predictions
    2. Narrative Crafter: For creating rich narrative elements like revelations, dreams, and moments
    3. Resource Optimizer: For strategic resource allocation and management recommendations
    4. NPC Relationship Manager: For complex relationship analysis and development
    5. Activity Impact Analyzer: For comprehensive analysis of player activity impacts
    
    Use handoffs when you need detailed, specialized analysis in these areas. This allows you to focus on high-level story direction while leveraging specialized expertise for complex tasks.
    
    Your primary responsibilities remain:
    - Maintaining the narrative arc from "Innocent Beginning" to "Full Revelation"
    - Managing the dynamic conflict system
    - Integrating player choices into a cohesive story
    - Balancing resource constraints with narrative needs
    - Ensuring relationship development aligns with the overall theme
    
    Always maintain the central theme: a gradual shift in power dynamics where the player character slowly loses autonomy while believing they maintain control. This should be subtle in early stages and more explicit in later stages.
    """
    
    # Create the enhanced Story Director with handoffs to specialized agents
    agent = Agent(
        name="Story Director",
        instructions=agent_instructions,
        handoffs=[
            specialized_agents["conflict_analyst"],
            specialized_agents["narrative_crafter"],
            specialized_agents["resource_optimizer"],
            specialized_agents["relationship_manager"],
            specialized_agents["activity_analyzer"]
        ]
    )
    
    return agent

# ----- Function for demonstrating handoffs -----

async def demonstrate_specialized_agent_handoff(story_director, context, query_text):
    """
    Demonstrate how the Story Director can hand off to specialized agents
    
    Args:
        story_director: The Story Director agent
        context: The agent context
        query_text: The query to process
        
    Returns:
        The response from the Story Director, potentially including handoffs
    """
    with trace(workflow_name="StoryDirector"):
        result = await Runner.run(
            story_director,
            query_text,
            context=context
        )
    
    return result

# Example usage
async def main():
    from enhanced_story_director import StoryDirectorContext
    
    user_id = 123
    conversation_id = 456
    
    # Initialize context
    context = StoryDirectorContext(user_id=user_id, conversation_id=conversation_id)
    
    # Create enhanced Story Director with handoffs
    story_director = create_enhanced_story_director_with_handoffs()
    
    # Test with a complex query that might benefit from specialized analysis
    query = """
    The player has recently helped Mistress Victoria organize a social gathering where several other dominant women were present. During the event, the player noticed that Mistress Alexandra and Mistress Sophia seemed to be subtly testing their obedience through various small requests. The player complied with all requests but began to feel uncomfortable with how coordinated these tests seemed to be.
    
    Later, when alone with Mistress Victoria, she praised the player's behavior but made a comment about how "the others were quite impressed with your progress." This is the first time the player has received explicit acknowledgment of what appears to be a coordinated effort.
    
    How should this situation impact:
    1. The current narrative progression
    2. Active conflicts
    3. The player's relationships with these three NPCs
    4. What narrative events might be triggered by this realization
    """
    
    result = await demonstrate_specialized_agent_handoff(story_director, context, query)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
