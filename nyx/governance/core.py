# nyx/governance/core.py
"""
Core governance class that combines all mixins.
"""
import logging
import json
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
from datetime import datetime

# Import all the mixins
from .story import StoryGovernanceMixin
from .npc import NPCGovernanceMixin
from .conflict import ConflictGovernanceMixin
from .world import WorldGovernanceMixin
from .agents import AgentGovernanceMixin
from .player import PlayerGovernanceMixin
from .constants import DirectiveType, DirectivePriority, AgentType

# Other imports
from utils.caching import CACHE_TTL
from db.connection import get_db_connection_context

if TYPE_CHECKING:
    from lore.core.lore_system import LoreSystem

logger = logging.getLogger(__name__)


class NyxUnifiedGovernor(
    StoryGovernanceMixin,
    NPCGovernanceMixin,
    ConflictGovernanceMixin,
    WorldGovernanceMixin,
    AgentGovernanceMixin,
    PlayerGovernanceMixin
):
    """
    Enhanced unified governance system for Nyx to control all agents with agentic capabilities.
    
    This class combines all governance functionality through mixins while maintaining
    the same public API for backward compatibility.
    
    This class provides:
      1. Central authority over all agents with enhanced coordination
      2. Goal-oriented agent coordination and planning
      3. Adaptive decision making with learning capabilities
      4. Performance monitoring and feedback loops
      5. Cross-agent communication and collaboration
      6. Dynamic goal prioritization and resource allocation
      7. Agent learning and adaptation tracking
      8. Enhanced conflict resolution with context awareness
      9. Temporal consistency enforcement
      10. User preference integration
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Will be initialized in _initialize_systems() to avoid circular dependency
        self.lore_system: Optional[Any] = None

        # Core systems and state
        self.memory_system = None
        self.game_state = None
        self.registered_agents: Dict[str, Dict[str, Any]] = {}     # {agent_type: {agent_id: instance}}

        # Multi-agent analytics
        self.active_goals: List[Dict[str, Any]] = []
        self.agent_goals: Dict[str, Dict[str, Any]] = {}           # {agent_type: {agent_id: ...}}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}     # {agent_type: {agent_id: ...}}
        self.agent_learning: Dict[str, Dict[str, Any]] = {}        # {agent_type: {agent_id: ...}}
        self.coordination_history: List[Dict[str, Any]] = []
        self.memory_integration = None
        self.memory_graph = None
    
        # Learning state
        self.strategy_effectiveness: Dict[str, Any] = {}
        self.adaptation_patterns: Dict[str, Any] = {}
        self.collaboration_success: Dict[str, Dict[str, Any]] = {}

        # Disagreement history
        self.disagreement_history: Dict[str, List[Dict[str, Any]]] = {}
        self.disagreement_thresholds: Dict[str, float] = {
            "narrative_impact": 0.7,
            "character_consistency": 0.8,
            "world_integrity": 0.9,
            "player_experience": 0.6
        }

        # Directive/action reports
        self.directives: Dict[str, Dict[str, Any]] = {}
        self.action_reports: Dict[str, Dict[str, Any]] = {}
        
        # Flag to track initialization
        self._initialized = False

    async def initialize(self) -> "NyxUnifiedGovernor":
        """
        Initialize the governance system asynchronously.
        This must be called after creating a new NyxUnifiedGovernor instance.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return self

        # Initialize other systems
        await self._initialize_systems()
        self._initialized = True
        return self

    async def _initialize_systems(self):
        """Initialize memory system, game state, and discover agents."""
        # Import LoreSystem locally to avoid circular import
        from lore.core.lore_system import LoreSystem
        
        # Get an instance of the LoreSystem
        # FIX: await the async get_instance call
        self.lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
        
        # Set the governor on the lore system (dependency injection)
        # FIX: set_governor is not async, so don't await it
        self.lore_system.set_governor(self)
        
        # Initialize the lore system WITH the governor reference
        await self.lore_system.initialize(governor=self)  # Pass self as governor
    
        # Initialize other systems
        from memory.memory_nyx_integration import get_memory_nyx_bridge
        self.memory_system = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
        
        # Initialize memory integration
        from memory.memory_integration import MemoryIntegration
        self.memory_integration = MemoryIntegration(self.user_id, self.conversation_id)
        await self.memory_integration.initialize()
    
        from nyx.integrate import JointMemoryGraph
        self.memory_graph = JointMemoryGraph(self.user_id, self.conversation_id)
        
        self.game_state = await self.initialize_game_state()
        await self.discover_and_register_agents()
        await self._load_initial_state()

    async def _load_initial_state(self):
        """Load goals and agent state from memory."""
        goal_memories = await self.memory_system.recall(
            entity_type="nyx",
            entity_id=self.conversation_id,
            query="active goals",
            context="system goals",
            limit=10
        )
        self.active_goals = await self._extract_goals_from_memories(goal_memories.get("memories", []))

        # Load all agents' goals
        for agent_type, agents_dict in self.registered_agents.items():
            for agent_id, agent in agents_dict.items():
                if hasattr(agent, "get_active_goals"):
                    self.agent_goals.setdefault(agent_type, {})[agent_id] = await agent.get_active_goals()
                else:
                    self.agent_goals.setdefault(agent_type, {})[agent_id] = []

        await self._update_performance_metrics()
        await self._load_learning_state()

    async def initialize_game_state(self, *, force: bool = False) -> Dict[str, Any]:
        """
        Fetch and return the game-state snapshot for this user/conversation.
    
        If it has already been loaded during this governor's lifetime, return
        the cached copy unless `force=True`.
        """
        if getattr(self, "game_state", None) and not force:
            logger.info(
                f"[GAME-STATE] Already initialized for {self.user_id}:{self.conversation_id}; skipping."
            )
            return self.game_state
    
        logger.info(
            f"[GAME-STATE] Initializing for {self.user_id}, conversation {self.conversation_id}"
        )
    
        game_state = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "current_location": None,
            "current_npcs": [],
            "current_time": None,
            "active_quests": [],
            "player_stats": {},
            "narrative_state": {},
            "world_state": {},
        }
    
        try:
            async with get_db_connection_context() as conn:
                # Get current location
                row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                if row: game_state["current_location"] = row["value"]
                
                # Get current time
                row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentTime'
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                if row: game_state["current_time"] = row["value"]

                row = await conn.fetchrow("""
                    SELECT * FROM PlayerStats
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                if row: game_state["player_stats"] = dict(row)

                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                game_state["current_npcs"] = [dict(r) for r in rows]

                rows = await conn.fetch("""
                    SELECT * FROM Quests
                    WHERE user_id = $1 AND conversation_id = $2 AND status = 'In Progress'
                """, self.user_id, self.conversation_id)
                game_state["active_quests"] = [dict(r) for r in rows]

                logger.info(
                    f"Game state initialized with "
                    f"{len(game_state['current_npcs'])} NPCs and "
                    f"{len(game_state['active_quests'])} quests."
                )
        except Exception as e:
            logger.error(f"Error initializing game state: {e}")
    
        # cache it so the next call can short-circuit
        self.game_state = game_state
        return game_state

    async def get_current_state(self, user_id: int, conversation_id: int) -> Dict[str, Any]:
        """
        Get the current state of the game world including narrative, character, and world state.
        """
        # Use the existing game state if available
        if hasattr(self, 'game_state') and self.game_state:
            base_state = self.game_state
        else:
            base_state = await self.initialize_game_state(force=True)
        
        # Get additional narrative context
        narrative_context = {}
        character_state = {}
        world_state = {}
        
        async with get_db_connection_context() as conn:
            # Get current narrative stage/arc
            narrative_stage = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NarrativeStage'
            """, user_id, conversation_id)
            
            if narrative_stage:
                narrative_context["current_arc"] = narrative_stage
                
            # Get active plot points
            active_quests = await conn.fetch("""
                SELECT quest_name, progress_detail FROM Quests
                WHERE user_id = $1 AND conversation_id = $2 AND status = 'In Progress'
            """, user_id, conversation_id)
            
            narrative_context["plot_points"] = [
                {"name": q["quest_name"], "details": q["progress_detail"]} 
                for q in active_quests
            ]
            
            # Get player character state
            player_stats = await conn.fetchrow("""
                SELECT * FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
            """, user_id, conversation_id)
            
            if player_stats:
                character_state = dict(player_stats)
                
            # Get player relationships
            relationships = await conn.fetch("""
                SELECT sl.*, ns.npc_name 
                FROM SocialLinks sl
                JOIN NPCStats ns ON sl.entity2_id = ns.npc_id
                WHERE sl.user_id = $1 AND sl.conversation_id = $2 
                AND sl.entity1_type = 'player' AND sl.entity2_type = 'npc'
            """, user_id, conversation_id)
            
            character_state["relationships"] = {
                r["npc_name"]: {
                    "type": r["link_type"],
                    "level": r["link_level"],
                    "stage": r["relationship_stage"]
                } for r in relationships
            }
            
            # Get world rules and systems
            world_rules = await conn.fetch("""
                SELECT rule_name, condition, effect FROM GameRules
            """)
            
            world_state["rules"] = {
                r["rule_name"]: {
                    "condition": r["condition"],
                    "effect": r["effect"]
                } for r in world_rules
            }
            
            # Get current setting info
            setting_name = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSetting'
            """, user_id, conversation_id)
            
            world_state["setting"] = setting_name or "Unknown"
        
        return {
            "game_state": base_state,
            "narrative_context": narrative_context,
            "character_state": character_state,
            "world_state": world_state
        }
