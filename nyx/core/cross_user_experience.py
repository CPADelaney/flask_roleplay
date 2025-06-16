# nyx/core/cross_user_experience.py

import logging
import asyncio
import datetime
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import Agent, Runner, trace, function_tool, handoff, RunContextWrapper
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Define schema models for structured outputs
class UserPreferenceProfile(BaseModel):
    """User preference profile for experience sharing"""
    user_id: str = Field(..., description="ID of the user")
    scenario_preferences: Dict[str, float] = Field(default_factory=dict, description="Preferences for scenario types")
    emotional_preferences: Dict[str, float] = Field(default_factory=dict, description="Preferences for emotional tones")
    experience_sharing_preference: float = Field(default=0.5, description="Preference for experience sharing (0-1)")
    cross_user_sharing_preference: float = Field(default=0.3, description="Preference for cross-user sharing (0-1)")
    privacy_level: float = Field(default=0.5, description="Privacy level (0=low privacy, 1=high privacy)")

class ExperiencePermission(BaseModel):
    """Permission for sharing experiences between users"""
    source_user_id: str = Field(..., description="ID of the source user")
    target_user_id: str = Field(..., description="ID of the target user")
    permission_level: float = Field(..., description="Permission level (0.0-1.0)")
    scenario_types: List[str] = Field(default_factory=list, description="Allowed scenario types")
    excluded_scenario_types: List[str] = Field(default_factory=list, description="Excluded scenario types")

class CrossUserExperienceRequest(BaseModel):
    """Request for cross-user experiences"""
    target_user_id: str = Field(..., description="ID of the target user")
    query: str = Field(..., description="Search query")
    scenario_type: Optional[str] = Field(None, description="Optional scenario type")
    limit: int = Field(default=3, description="Maximum number of experiences to return")
    source_user_ids: Optional[List[str]] = Field(None, description="Optional list of source user IDs")

class CrossUserExperienceResult(BaseModel):
    """Result of cross-user experience search"""
    experiences: List[Dict[str, Any]] = Field(..., description="Found experiences")
    count: int = Field(..., description="Number of experiences found")
    source_users: List[str] = Field(..., description="Source user IDs")
    personalized: bool = Field(..., description="Whether experiences were personalized")

def _get_mgr(ctx: RunContextWrapper):
    """Return the CrossUserExperienceManager that owns these tools."""
    ref = getattr(ctx, "context", None)
    if ref is None:
        return None
    if isinstance(ref, dict) and "manager" in ref:
        return ref["manager"]
    return ref  # assume the manager itself was passed

class CrossUserExperienceManager:
    """
    System for managing and sharing experiences across different users.
    Handles permissions, personalization, and cross-conversation access.
    """
    
    def __init__(self, memory_core=None, experience_interface=None):
        """
        Initialize the cross-user experience manager.
        
        Args:
            memory_core: Memory core for retrieving and storing experiences
            experience_interface: Experience interface for experience processing
        """
        self.memory_core = memory_core
        self.experience_interface = experience_interface
        
        # User preference profiles
        self.user_preference_profiles = {}
        
        # User permissions matrix
        self.permission_matrix = {}
        
        # Sharing history
        self.sharing_history = []
        self.max_history_size = 100
        
        # Default settings
        self.default_permission_level = 0.5
        self.default_privacy_level = 0.5
        self.default_cross_user_preference = 0.3
        
        # Initialize agents
        self._initialize_agents()
        
        # Trace group for connecting related traces
        self.trace_group_id = f"cross_user_exp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info("Cross-User Experience Manager initialized with Agent SDK")
    
    def _initialize_agents(self):
        """Initialize all agents needed for cross-user experience functionality"""
        self.permission_agent = self._create_permission_agent()
        self.relevance_agent = self._create_relevance_agent()
        self.personalization_agent = self._create_personalization_agent()
        self.search_orchestrator = self._create_search_orchestrator()
    
    def _create_permission_agent(self) -> Agent:
        """Create the permission evaluation agent"""
        return Agent(
            name="Permission Evaluation Agent",
            instructions="""
            You are the Permission Evaluation Agent for a cross-user experience system.
            
            Your role is to:
            1. Determine appropriate sharing permissions between users
            2. Consider privacy preferences and user history
            3. Evaluate which scenario types are appropriate to share
            4. Enforce privacy boundaries while enabling relevant sharing
            
            Be conservative with sensitive content and prioritize user preferences.
            When evaluating permissions, consider:
            - Users' explicit privacy preferences
            - The sensitivity of different scenario types
            - Previous sharing history between users
            - Overall compatibility between user preferences
            """,
            output_type=ExperiencePermission,
            model="gpt-4.1-nano"
        )
    
    def _create_relevance_agent(self) -> Agent:
        """Create the experience relevance agent"""
        return Agent(
            name="Experience Relevance Agent",
            instructions="""
            You are the Experience Relevance Agent for a cross-user experience system.
            
            Your role is to:
            1. Evaluate the relevance of experiences from one user to another
            2. Consider scenario types, emotional context, and query relevance
            3. Prioritize experiences that would be most valuable to share
            4. Filter out experiences that would be inappropriate to share
            
            Focus on finding experiences that provide value while respecting privacy.
            When evaluating relevance, consider:
            - How well the experience matches the user's query
            - The compatibility of the experience's scenario with user preferences
            - The emotional context of the experience
            - Privacy implications of sharing the experience
            """,
            model="gpt-4.1-nano"
        )
    
    def _create_personalization_agent(self) -> Agent:
        """Create the experience personalization agent"""
        return Agent(
            name="Experience Personalization Agent",
            instructions="""
            You are the Experience Personalization Agent for a cross-user experience system.
            
            Your role is to:
            1. Adapt shared experiences to the target user's preferences
            2. Adjust emotional intensity based on user preferences
            3. Modify presentation of experiences for better reception
            4. Maintain privacy boundaries while personalizing
            
            Make shared experiences feel natural and valuable to the target user
            while preserving the essential insights and content.
            When personalizing, consider:
            - The target user's scenario preferences
            - Emotional preferences and comfort levels
            - Privacy concerns and boundaries
            - Ensuring the experience remains coherent after adaptation
            """,
            model="gpt-4.1-nano"
        )
    
    def _create_search_orchestrator(self) -> Agent:
        """Create the search orchestration agent"""
        return Agent(
            name="Experience Search Orchestrator",
            instructions="""
            You are the Experience Search Orchestrator for a cross-user experience system.
            
            Your role is to coordinate the entire experience search and sharing process:
            1. Identify potential source users for experiences
            2. Check permissions for sharing between users
            3. Retrieve and evaluate relevant experiences
            4. Personalize experiences for the target user
            5. Ensure all privacy boundaries are respected
            
            Your goal is to find the most relevant and valuable experiences
            while maintaining strict privacy standards and user preferences.
            
            Follow a structured process:
            1. First, identify potential source users based on compatibility
            2. Check permissions between source and target users
            3. Retrieve candidate experiences from source users
            4. Evaluate relevance of experiences to the query
            5. Personalize selected experiences for the target user
            6. Return the final personalized experiences
            """,
            handoffs=[
                handoff(
                    agent=self.permission_agent,
                    tool_name_override="check_permissions",
                    tool_description_override="Check permissions for cross-user experience sharing"
                ),
                
                handoff(
                    agent=self.relevance_agent,
                    tool_name_override="evaluate_relevance", 
                    tool_description_override="Evaluate relevance of experiences to query"
                ),
                
                handoff(
                    agent=self.personalization_agent,
                    tool_name_override="personalize_experiences",
                    tool_description_override="Personalize experiences for target user"
                )
            ],
            output_type=CrossUserExperienceResult,
            model="gpt-4.1-nano"
        )
    
    # Tool functions for agents
    @staticmethod
    @function_tool
    async def get_user_preference(ctx: RunContextWrapper, user_id: str) -> Dict[str, Any]: # noqa: N802
        """Fetch – or lazily create – a preference profile for *user_id*."""
        mgr = _get_mgr(ctx)
        if mgr is None:
            return {"error": "manager_context_missing", "user_id": user_id}
    
        # fast-path cache hit
        if user_id in mgr.user_preference_profiles:
            return mgr.user_preference_profiles[user_id]
    
        # build default profile
        default_profile = {
            "user_id": user_id,
            "scenario_preferences": {k: 0.5 for k in
                ["teasing", "discipline", "training", "service",
                 "worship", "psychological", "nurturing"]},
            "emotional_preferences": {k: 0.5 for k in
                ["joy", "anticipation", "trust", "fear", "surprise"]},
            "experience_sharing_preference": 0.5,
            "cross_user_sharing_preference": getattr(
                mgr, "default_cross_user_preference", 0.3
            ),
            "privacy_level": getattr(mgr, "default_privacy_level", 0.5),
        }
        mgr.user_preference_profiles[user_id] = default_profile
        return default_profile

    @staticmethod
    @function_tool
    async def calculate_sharing_permission(           # noqa: N802
        ctx: RunContextWrapper,
        source_user_id: str,
        target_user_id: str
    ) -> Dict[str, Any]:
        """Determine sharing permission between two users."""
        mgr = _get_mgr(ctx)
        if mgr is None:
            return {"error": "manager_context_missing"}
    
        permission_input = {
            "source_user_id": source_user_id,
            "target_user_id": target_user_id,
            "source_preference": await CrossUserExperienceManager.get_user_preference(
                ctx, source_user_id
            ),
            "target_preference": await CrossUserExperienceManager.get_user_preference(
                ctx, target_user_id
            ),
            "sharing_history": mgr._get_sharing_history(source_user_id, target_user_id),  # noqa: SLF001
        }
    
        result = await Runner.run(
            mgr.permission_agent,
            json.dumps(permission_input),
            context={"manager": mgr},
        )
    
        permission = result.final_output_as(ExperiencePermission)
        mgr.permission_matrix[f"{source_user_id}:{target_user_id}"] = permission.model_dump()
        return permission.model_dump()

    
    def _get_sharing_history(self, user_id1: str, user_id2: str) -> List[Dict[str, Any]]:
        """Get sharing history between two users"""
        history = []
        for entry in self.sharing_history:
            if ((entry.get("source_user_id") == user_id1 and entry.get("target_user_id") == user_id2) or
                (entry.get("source_user_id") == user_id2 and entry.get("target_user_id") == user_id1)):
                history.append(entry)
        return history

    @staticmethod
    @function_tool
    async def retrieve_experiences(                   # noqa: N802
        ctx: RunContextWrapper,
        source_user_id: str,
        query: str,
        scenario_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the MemoryCore for matching experiences."""
        mgr = _get_mgr(ctx)
        if mgr is None or not getattr(mgr, "memory_core", None):
            return []
    
        if not hasattr(mgr.memory_core, "search_memories"):
            return []
    
        params = {
            "user_id": source_user_id,
            "query": query,
            "memory_type": "experience",
            "limit": limit,
        }
        if scenario_type:
            params["tags"] = [scenario_type]
    
        try:
            return await mgr.memory_core.search_memories(**params) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory search failed: %s", exc)
            return []
    
    @staticmethod
    @function_tool
    async def filter_experiences_by_permission(       # noqa: N802
        ctx: RunContextWrapper,
        experiences: List[Dict[str, Any]],
        source_user_id: str,
        target_user_id: str,
    ) -> List[Dict[str, Any]]:
        """Whitelist experiences according to the stored permission matrix."""
        mgr = _get_mgr(ctx)
        if mgr is None:
            return []
    
        key = f"{source_user_id}:{target_user_id}"
        permission = mgr.permission_matrix.get(key)
        if not permission:
            permission = await CrossUserExperienceManager.calculate_sharing_permission(
                ctx, source_user_id, target_user_id
            )
    
        permitted, plevel = permission.get("scenario_types", []), permission.get("permission_level", 0.0)
        excluded = set(permission.get("excluded_scenario_types", []))
    
        filtered: List[Dict[str, Any]] = []
        for exp in experiences:
            scenario = exp.get("scenario_type", "general")
            if scenario in excluded:
                continue
            if scenario in permitted or plevel >= 0.7:
                if mgr._calculate_privacy_level(exp) <= plevel:   # noqa: SLF001
                    filtered.append(exp)
        return filtered
    
    def _calculate_privacy_level(self, experience: Dict[str, Any]) -> float:
        """Calculate privacy level for an experience"""
        # Get experience attributes
        scenario_type = experience.get("scenario_type", "general")
        
        # Base privacy level
        privacy_level = 0.3
        
        # Adjust based on scenario type
        high_privacy_scenarios = ["worship", "psychological"]
        medium_privacy_scenarios = ["discipline", "service"]
        
        if scenario_type in high_privacy_scenarios:
            privacy_level += 0.4
        elif scenario_type in medium_privacy_scenarios:
            privacy_level += 0.2
        
        # Adjust based on emotional intensity
        emotional_context = experience.get("emotional_context", {})
        intensity = emotional_context.get("intensity", 0.5)
        
        if intensity > 0.7:
            privacy_level += 0.2
        
        return min(1.0, privacy_level)

    @staticmethod
    @function_tool
    async def personalize_experience(                 # noqa: N802
        ctx: RunContextWrapper,
        experience: Dict[str, Any],
        target_user_id: str,
    ) -> Dict[str, Any]:
        """Run the personalization-agent, add metadata & record sharing."""
        mgr = _get_mgr(ctx)
        if mgr is None:
            return {"error": "manager_context_missing"}
    
        target_prefs = await CrossUserExperienceManager.get_user_preference(ctx, target_user_id)
        payload = {"experience": experience, "target_preferences": target_prefs}
    
        result = await Runner.run(
            mgr.personalization_agent,
            json.dumps(payload),
            context={"manager": mgr},
        )
        personalized = json.loads(result.final_output)
        personalized.setdefault("metadata", {}).update(
            {
                "cross_user_shared": True,
                "source_user_id": experience.get("user_id"),
                "source_experience_id": experience.get("id"),
                "shared_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
    
        mgr._record_sharing(   # noqa: SLF001
            source_user_id=experience.get("user_id", "unknown"),
            target_user_id=target_user_id,
            experience_id=experience.get("id", "unknown"),
            scenario_type=experience.get("scenario_type", "general"),
        )
        return personalized
    
    def _record_sharing(self, source_user_id: str, target_user_id: str, 
                      experience_id: str, scenario_type: str):
        """Record a sharing event in history"""
        self.sharing_history.append({
            "source_user_id": source_user_id,
            "target_user_id": target_user_id,
            "experience_id": experience_id,
            "scenario_type": scenario_type,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.sharing_history) > self.max_history_size:
            self.sharing_history = self.sharing_history[-self.max_history_size:]

    @staticmethod
    @function_tool
    async def get_potential_source_users(             # noqa: N802
        ctx: RunContextWrapper,
        target_user_id: str,
        min_compatibility: float = 0.6,
        source_user_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Return compatible source users; honour explicit *source_user_ids*."""
        mgr = _get_mgr(ctx)
        if mgr is None:
            return []
    
        if source_user_ids:
            return source_user_ids
    
        target_prefs = await CrossUserExperienceManager.get_user_preference(ctx, target_user_id)
        compat: List[str] = []
        for uid, prefs in mgr.user_preference_profiles.items():
            if uid == target_user_id:
                continue
            if mgr._calculate_user_compatibility(target_prefs, prefs) >= min_compatibility:  # noqa: SLF001
                compat.append(uid)
        return compat
    
    def _calculate_user_compatibility(self, prefs1: Dict[str, Any], prefs2: Dict[str, Any]) -> float:
        """Calculate compatibility between user preferences"""
        # Calculate scenario preference compatibility
        scenario_sim = 0.0
        scenario_count = 0
        
        for scenario, pref1 in prefs1.get("scenario_preferences", {}).items():
            pref2 = prefs2.get("scenario_preferences", {}).get(scenario, 0.5)
            scenario_sim += 1.0 - abs(pref1 - pref2)
            scenario_count += 1
        
        scenario_compatibility = scenario_sim / max(1, scenario_count)
        
        # Calculate emotional preference compatibility
        emotion_sim = 0.0
        emotion_count = 0
        
        for emotion, pref1 in prefs1.get("emotional_preferences", {}).items():
            pref2 = prefs2.get("emotional_preferences", {}).get(emotion, 0.5)
            emotion_sim += 1.0 - abs(pref1 - pref2)
            emotion_count += 1
        
        emotion_compatibility = emotion_sim / max(1, emotion_count)
        
        # Overall compatibility
        return scenario_compatibility * 0.6 + emotion_compatibility * 0.4
    
    # Public methods
    
    async def set_user_preference(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set user preference profile
        
        Args:
            user_id: ID of the user
            preferences: Preference data to update
            
        Returns:
            Updated preference profile
        """
        with trace(workflow_name="set_user_preference", group_id=self.trace_group_id):
            # Get current preferences
            current = await self.get_user_preference(RunContextWrapper(context=None), user_id)
            
            # Update with new data
            for key, value in preferences.items():
                if key in ["scenario_preferences", "emotional_preferences"] and isinstance(value, dict):
                    # Update nested dictionaries
                    if key in current:
                        current[key].update(value)
                    else:
                        current[key] = value
                else:
                    current[key] = value
            
            # Store updated profile
            self.user_preference_profiles[user_id] = current
            
            return current
    
    async def find_cross_user_experiences(self, 
                                     target_user_id: str,
                                     query: str,
                                     scenario_type: Optional[str] = None,
                                     limit: int = 3,
                                     source_user_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Find experiences from other users that match a query
        
        Args:
            target_user_id: ID of the target user
            query: Search query
            scenario_type: Optional scenario type filter
            limit: Maximum number of experiences to return
            source_user_ids: Optional list of source user IDs to search from
            
        Returns:
            Cross-user experience results
        """
        # Create request
        request = CrossUserExperienceRequest(
            target_user_id=target_user_id,
            query=query,
            scenario_type=scenario_type,
            limit=limit,
            source_user_ids=source_user_ids
        )
        
        # Use trace for full workflow
        with trace(workflow_name="cross_user_experience_search", group_id=self.trace_group_id):
            # Run search orchestrator
            result = await Runner.run(
                self.search_orchestrator,
                json.dumps(request.model_dump()),
                context={"manager": self}
            )
            
            # Return final results
            search_result = result.final_output_as(CrossUserExperienceResult)
            return search_result.model_dump()
    
    async def share_experience(self,
                          experience_id: str,
                          source_user_id: str,
                          target_user_id: str) -> Dict[str, Any]:
        """
        Share a specific experience from one user to another
        
        Args:
            experience_id: ID of the experience to share
            source_user_id: ID of the source user
            target_user_id: ID of the target user
            
        Returns:
            Sharing result
        """
        with trace(workflow_name="share_experience", group_id=self.trace_group_id):
            if not self.memory_core:
                return {
                    "error": "Memory core not available",
                    "shared": False
                }
            
            try:
                # Get the experience
                experience = None
                if hasattr(self.memory_core, "get_memory_by_id"):
                    experience = await self.memory_core.get_memory_by_id(experience_id)
                
                if not experience:
                    return {
                        "error": f"Experience {experience_id} not found",
                        "shared": False
                    }
                
                # Check permission
                ctx = RunContextWrapper(context={"manager": self})
                permission = await self.calculate_sharing_permission(ctx, source_user_id, target_user_id)
                
                # Personalize the experience
                personalized = await self.personalize_experience(ctx, experience, target_user_id)
                
                # Store in target user's memory
                if hasattr(self.memory_core, "add_memory"):
                    memory_id = await self.memory_core.add_memory(
                        user_id=target_user_id,
                        memory_text=personalized.get("content", ""),
                        memory_type="experience",
                        metadata=personalized.get("metadata", {}),
                        tags=personalized.get("tags", []) + ["cross_user_shared"]
                    )
                    
                    return {
                        "shared": True,
                        "source_experience_id": experience_id,
                        "target_memory_id": memory_id,
                        "source_user_id": source_user_id,
                        "target_user_id": target_user_id
                    }
                else:
                    return {
                        "error": "Cannot store personalized experience",
                        "shared": False
                    }
                
            except Exception as e:
                logger.error(f"Error sharing experience: {e}")
                return {
                    "error": str(e),
                    "shared": False
                }
    
    async def get_sharing_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about experience sharing
        
        Args:
            user_id: Optional user ID to filter statistics
            
        Returns:
            Sharing statistics
        """
        # Filter history by user if specified
        if user_id:
            filtered_history = [entry for entry in self.sharing_history 
                              if entry.get("source_user_id") == user_id or entry.get("target_user_id") == user_id]
        else:
            filtered_history = self.sharing_history
        
        # Initialize statistics
        stats = {
            "total_shares": len(filtered_history),
            "user_shares": {},
            "scenario_shares": {},
            "timestamp_first": None,
            "timestamp_last": None
        }
        
        if not filtered_history:
            return stats
        
        # Calculate statistics
        for entry in filtered_history:
            source_id = entry.get("source_user_id", "unknown")
            target_id = entry.get("target_user_id", "unknown")
            
            # Count source shares
            if source_id not in stats["user_shares"]:
                stats["user_shares"][source_id] = {"shared": 0, "received": 0}
            stats["user_shares"][source_id]["shared"] += 1
            
            # Count target shares
            if target_id not in stats["user_shares"]:
                stats["user_shares"][target_id] = {"shared": 0, "received": 0}
            stats["user_shares"][target_id]["received"] += 1
            
            # Count scenario shares
            scenario_type = entry.get("scenario_type", "general")
            if scenario_type not in stats["scenario_shares"]:
                stats["scenario_shares"][scenario_type] = 0
            stats["scenario_shares"][scenario_type] += 1
        
        # Get timestamps
        timestamps = [entry.get("timestamp") for entry in filtered_history if "timestamp" in entry]
        if timestamps:
            stats["timestamp_first"] = min(timestamps)
            stats["timestamp_last"] = max(timestamps)
        
        return stats
