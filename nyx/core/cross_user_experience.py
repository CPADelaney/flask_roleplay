# nyx/core/cross_user_experience.py

import logging
import asyncio
import datetime
import json
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from agents import Agent, Runner, trace, function_tool, RunContextWrapper
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

class ExperienceSharingPermission(BaseModel):
    """Permission for sharing experiences between users"""
    source_user_id: str = Field(..., description="ID of the source user")
    target_user_id: str = Field(..., description="ID of the target user")
    permission_level: float = Field(..., description="Permission level (0.0-1.0)")
    scenario_types: List[str] = Field(default_factory=list, description="Allowed scenario types")
    excluded_scenario_types: List[str] = Field(default_factory=list, description="Excluded scenario types")
    timestamp: str = Field(..., description="ISO timestamp of permission setting")

class CrossUserExperienceRequest(BaseModel):
    """Request for cross-user experiences"""
    source_user_id: str = Field(..., description="ID of the source user")
    target_user_id: str = Field(..., description="ID of the target user")
    query: str = Field(..., description="Search query")
    scenario_type: Optional[str] = Field(None, description="Optional scenario type")
    limit: int = Field(default=3, description="Maximum number of experiences to return")
    min_relevance: float = Field(default=0.7, description="Minimum relevance score (0.0-1.0)")

class CrossUserExperienceResult(BaseModel):
    """Result of cross-user experience request"""
    experiences: List[Dict[str, Any]] = Field(..., description="Found experiences")
    count: int = Field(..., description="Number of experiences found")
    source_user_id: str = Field(..., description="ID of the source user")
    target_user_id: str = Field(..., description="ID of the target user")
    relevance_scores: List[float] = Field(..., description="Relevance scores for found experiences")
    personalized: bool = Field(..., description="Whether experiences were personalized")

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
        
        # Initialize agents
        self.permission_agent = self._create_permission_agent()
        self.relevance_agent = self._create_relevance_agent()
        self.personalization_agent = self._create_personalization_agent()
        
        # User preference profiles
        self.user_preference_profiles = {}
        
        # User permissions matrix
        self.permission_matrix = {}
        
        # Sharing history
        self.sharing_history = []
        self.max_history_size = 100
        
        # User clusters
        self.user_clusters = {}
        self.cluster_similarity = {}
        
        # Default settings
        self.default_permission_level = 0.5
        self.default_privacy_level = 0.5
        self.default_cross_user_preference = 0.3
        self.min_relevance_threshold = 0.7
        
        # Trace ID for connecting traces
        self.trace_group_id = f"cross_user_exp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info("Cross-User Experience Manager initialized")
    
    def _create_permission_agent(self) -> Agent:
        """Create the permission agent"""
        return Agent(
            name="Experience Permission Agent",
            instructions="""
            You are the Experience Permission Agent for Nyx's cross-user experience system.
            
            Your role is to:
            1. Determine appropriate sharing permissions between users
            2. Consider privacy preferences and user history
            3. Update permission settings based on user interactions
            4. Enforce privacy boundaries while enabling relevant sharing
            5. Handle permission requests and adjustments
            
            Balance the benefits of sharing experiences with privacy concerns.
            Be conservative with sensitive content and prioritize user preferences.
            """,
            tools=[
                function_tool(self._get_user_preference),
                function_tool(self._calculate_sharing_permission),
                function_tool(self._check_permission_compatibility),
                function_tool(self._update_permission_matrix)
            ]
        )
    
    def _create_relevance_agent(self) -> Agent:
        """Create the relevance agent"""
        return Agent(
            name="Cross-User Relevance Agent",
            instructions="""
            You are the Cross-User Relevance Agent for Nyx's cross-user experience system.
            
            Your role is to:
            1. Evaluate the relevance of experiences from one user to another
            2. Consider scenario types, emotional context, and query relevance
            3. Prioritize experiences that would be most valuable to share
            4. Filter out experiences that would be inappropriate to share
            5. Balance relevance with novelty and diversity
            
            Focus on finding experiences that provide value and insight while
            respecting privacy and sharing preferences.
            """,
            tools=[
                function_tool(self._calculate_cross_user_relevance),
                function_tool(self._filter_experiences_by_permission),
                function_tool(self._calculate_experience_privacy_level)
            ]
        )
    
    def _create_personalization_agent(self) -> Agent:
        """Create the personalization agent"""
        return Agent(
            name="Experience Personalization Agent",
            instructions="""
            You are the Experience Personalization Agent for Nyx's cross-user experience system.
            
            Your role is to:
            1. Adapt shared experiences to the target user's preferences
            2. Adjust emotional intensity based on user preferences
            3. Modify presentation of experiences for better reception
            4. Maintain privacy boundaries while personalizing
            5. Create a coherent narrative for the shared experience
            
            Make shared experiences feel natural and valuable to the target user
            while preserving the essential insights and content.
            """,
            tools=[
                function_tool(self._get_user_preference),
                function_tool(self._personalize_experience),
                function_tool(self._modify_emotional_context),
                function_tool(self._calculate_personalization_level)
            ]
        )
    
    # Tool functions
    
    @function_tool
    async def _get_user_preference(self, ctx: RunContextWrapper, user_id: str) -> Dict[str, Any]:
        """
        Get user preference profile for experience sharing
        
        Args:
            user_id: ID of the user
            
        Returns:
            User preference profile
        """
        # Check if profile exists
        if user_id in self.user_preference_profiles:
            return self.user_preference_profiles[user_id]
        
        # Create default profile
        default_profile = {
            "user_id": user_id,
            "scenario_preferences": {
                "teasing": 0.5,
                "dark": 0.5,
                "indulgent": 0.5,
                "psychological": 0.5,
                "nurturing": 0.5,
                "discipline": 0.5,
                "training": 0.5,
                "service": 0.5,
                "worship": 0.5
            },
            "emotional_preferences": {
                "joy": 0.5,
                "sadness": 0.5,
                "anger": 0.5,
                "fear": 0.5,
                "trust": 0.5,
                "disgust": 0.5,
                "anticipation": 0.5,
                "surprise": 0.5,
                "love": 0.5
            },
            "experience_sharing_preference": 0.5,
            "cross_user_sharing_preference": self.default_cross_user_preference,
            "privacy_level": self.default_privacy_level
        }
        
        # Store and return
        self.user_preference_profiles[user_id] = default_profile
        return default_profile
    
    @function_tool
    async def _calculate_sharing_permission(self, ctx: RunContextWrapper,
                                       source_user_id: str,
                                       target_user_id: str) -> float:
        """
        Calculate appropriate sharing permission level between users
        
        Args:
            source_user_id: ID of the source user
            target_user_id: ID of the target user
            
        Returns:
            Permission level (0.0-1.0)
        """
        # Get user preferences
        source_profile = await self._get_user_preference(ctx, source_user_id)
        target_profile = await self._get_user_preference(ctx, target_user_id)
        
        # Get privacy level (higher = less sharing)
        source_privacy = source_profile.get("privacy_level", self.default_privacy_level)
        
        # Get cross-user sharing preference
        target_preference = target_profile.get("cross_user_sharing_preference", self.default_cross_user_preference)
        
        # Calculate base permission level
        # Lower privacy and higher preference = higher permission
        base_permission = (1.0 - source_privacy) * 0.5 + target_preference * 0.5
        
        # Get historical interaction level
        interaction_level = self._calculate_user_interaction_level(source_user_id, target_user_id)
        
        # Adjust permission based on interaction level
        adjusted_permission = base_permission * 0.7 + interaction_level * 0.3
        
        # Enforce minimum permission level
        min_permission = 0.1
        
        return max(min_permission, min(1.0, adjusted_permission))
    
    def _calculate_user_interaction_level(self, user_id1: str, user_id2: str) -> float:
        """
        Calculate interaction level between users based on sharing history
        
        Args:
            user_id1: First user ID
            user_id2: Second user ID
            
        Returns:
            Interaction level (0.0-1.0)
        """
        # Count shared experiences in both directions
        share_count = 0
        for entry in self.sharing_history:
            if (entry.get("source_user_id") == user_id1 and entry.get("target_user_id") == user_id2) or \
               (entry.get("source_user_id") == user_id2 and entry.get("target_user_id") == user_id1):
                share_count += 1
        
        # Calculate level (caps at 20 shares)
        return min(1.0, share_count / 20.0)
    
    @function_tool
    async def _check_permission_compatibility(self, ctx: RunContextWrapper,
                                         source_user_id: str,
                                         target_user_id: str) -> Dict[str, Any]:
        """
        Check compatibility between users for experience sharing
        
        Args:
            source_user_id: ID of the source user
            target_user_id: ID of the target user
            
        Returns:
            Compatibility assessment
        """
        # Get user preferences
        source_profile = await self._get_user_preference(ctx, source_user_id)
        target_profile = await self._get_user_preference(ctx, target_user_id)
        
        # Calculate scenario preference compatibility
        scenario_compatibility = {}
        
        for scenario, source_pref in source_profile.get("scenario_preferences", {}).items():
            target_pref = target_profile.get("scenario_preferences", {}).get(scenario, 0.5)
            
            # Higher preference on both sides = higher compatibility
            compatibility = (source_pref + target_pref) / 2
            scenario_compatibility[scenario] = compatibility
        
        # Calculate emotional preference compatibility
        emotional_compatibility = {}
        
        for emotion, source_pref in source_profile.get("emotional_preferences", {}).items():
            target_pref = target_profile.get("emotional_preferences", {}).get(emotion, 0.5)
            
            # Higher preference on both sides = higher compatibility
            compatibility = (source_pref + target_pref) / 2
            emotional_compatibility[emotion] = compatibility
        
        # Calculate overall compatibility
        overall_compatibility = (
            sum(scenario_compatibility.values()) / len(scenario_compatibility) if scenario_compatibility else 0.5
        ) * 0.5 + (
            sum(emotional_compatibility.values()) / len(emotional_compatibility) if emotional_compatibility else 0.5
        ) * 0.5
        
        # Get recommended scenarios to share (highest compatibility)
        recommended_scenarios = sorted(
            scenario_compatibility.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Get excluded scenarios (lowest compatibility)
        excluded_scenarios = sorted(
            scenario_compatibility.items(),
            key=lambda x: x[1]
        )[:2]
        
        return {
            "overall_compatibility": overall_compatibility,
            "scenario_compatibility": scenario_compatibility,
            "emotional_compatibility": emotional_compatibility,
            "recommended_scenarios": [s for s, _ in recommended_scenarios],
            "excluded_scenarios": [s for s, _ in excluded_scenarios if _ < 0.3]  # Only exclude if low compatibility
        }
    
    @function_tool
    async def _update_permission_matrix(self, ctx: RunContextWrapper,
                                   source_user_id: str,
                                   target_user_id: str,
                                   permission_level: float,
                                   allowed_scenarios: List[str],
                                   excluded_scenarios: List[str]) -> Dict[str, Any]:
        """
        Update permission matrix with new permission settings
        
        Args:
            source_user_id: ID of the source user
            target_user_id: ID of the target user
            permission_level: Permission level (0.0-1.0)
            allowed_scenarios: List of allowed scenario types
            excluded_scenarios: List of excluded scenario types
            
        Returns:
            Update results
        """
        # Create matrix entry key
        key = f"{source_user_id}:{target_user_id}"
        
        # Create permission entry
        permission = {
            "source_user_id": source_user_id,
            "target_user_id": target_user_id,
            "permission_level": max(0.0, min(1.0, permission_level)),
            "scenario_types": allowed_scenarios,
            "excluded_scenario_types": excluded_scenarios,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Store in matrix
        self.permission_matrix[key] = permission
        
        return {
            "updated": True,
            "permission": permission
        }
    
    @function_tool
    async def _calculate_cross_user_relevance(self, ctx: RunContextWrapper,
                                         experience: Dict[str, Any],
                                         target_user_id: str,
                                         query: str) -> float:
        """
        Calculate relevance of an experience for cross-user sharing
        
        Args:
            experience: Experience to evaluate
            target_user_id: ID of the target user
            query: Search query
            
        Returns:
            Relevance score (0.0-1.0)
        """
        # Get target user preferences
        target_profile = await self._get_user_preference(ctx, target_user_id)
        
        # Get experience details
        scenario_type = experience.get("scenario_type", "general")
        emotional_context = experience.get("emotional_context", {})
        significance = experience.get("significance", 5) / 10  # Convert to 0-1 scale
        
        # Calculate query relevance
        query_relevance = experience.get("relevance_score", 0.5)
        
        # Calculate scenario preference match
        scenario_preference = target_profile.get("scenario_preferences", {}).get(scenario_type, 0.5)
        
        # Calculate emotional preference match
        primary_emotion = emotional_context.get("primary_emotion", "neutral").lower()
        emotional_preference = target_profile.get("emotional_preferences", {}).get(primary_emotion, 0.5)
        
        # Calculate privacy level (higher = less sharing)
        privacy_level = await self._calculate_experience_privacy_level(ctx, experience)
        privacy_factor = 1.0 - privacy_level  # Invert for relevance calculation
        
        # Calculate overall relevance
        relevance = (
            query_relevance * 0.4 +
            scenario_preference * 0.2 +
            emotional_preference * 0.2 +
            significance * 0.1 +
            privacy_factor * 0.1
        )
        
        return max(0.0, min(1.0, relevance))
    
    @function_tool
    async def _filter_experiences_by_permission(self, ctx: RunContextWrapper,
                                          experiences: List[Dict[str, Any]],
                                          source_user_id: str,
                                          target_user_id: str) -> List[Dict[str, Any]]:
        """
        Filter experiences based on sharing permissions
        
        Args:
            experiences: List of experiences to filter
            source_user_id: ID of the source user
            target_user_id: ID of the target user
            
        Returns:
            Filtered experiences
        """
        # Get permission settings
        key = f"{source_user_id}:{target_user_id}"
        permission = self.permission_matrix.get(key)
        
        if not permission:
            # Calculate permission if not in matrix
            permission_level = await self._calculate_sharing_permission(ctx, source_user_id, target_user_id)
            
            # Get compatibility
            compatibility = await self._check_permission_compatibility(ctx, source_user_id, target_user_id)
            
            # Create and store permission
            permission = await self._update_permission_matrix(
                ctx,
                source_user_id=source_user_id,
                target_user_id=target_user_id,
                permission_level=permission_level,
                allowed_scenarios=compatibility.get("recommended_scenarios", []),
                excluded_scenarios=compatibility.get("excluded_scenarios", [])
            )
            
            permission = permission.get("permission", {})
        
        # Get permission settings
        permission_level = permission.get("permission_level", self.default_permission_level)
        allowed_scenarios = permission.get("scenario_types", [])
        excluded_scenarios = permission.get("excluded_scenario_types", [])
        
        # Filter experiences
        filtered_experiences = []
        
        for experience in experiences:
            # Get scenario type
            scenario_type = experience.get("scenario_type", "general")
            
            # Check if scenario is explicitly excluded
            if scenario_type in excluded_scenarios:
                continue
            
            # Check if scenario is allowed or permission level is high enough
            if scenario_type in allowed_scenarios or permission_level >= 0.7:
                # Calculate privacy level
                privacy_level = await self._calculate_experience_privacy_level(ctx, experience)
                
                # Only share if privacy level is acceptable
                if privacy_level <= permission_level:
                    filtered_experiences.append(experience)
        
        return filtered_experiences
    
    @function_tool
    async def _calculate_experience_privacy_level(self, ctx: RunContextWrapper,
                                            experience: Dict[str, Any]) -> float:
        """
        Calculate privacy level for an experience
        
        Args:
            experience: Experience to evaluate
            
        Returns:
            Privacy level (0.0-1.0, higher = more private)
        """
        # Get experience content
        content = experience.get("content", "")
        scenario_type = experience.get("scenario_type", "general")
        significance = experience.get("significance", 5) / 10  # Convert to 0-1 scale
        
        # Base privacy level
        privacy_level = 0.3
        
        # Adjust based on scenario type
        high_privacy_scenarios = ["dark", "punishment", "worship"]
        medium_privacy_scenarios = ["discipline", "psychological", "service"]
        
        if scenario_type in high_privacy_scenarios:
            privacy_level += 0.4
        elif scenario_type in medium_privacy_scenarios:
            privacy_level += 0.2
        
        # Adjust based on emotional intensity
        emotional_context = experience.get("emotional_context", {})
        primary_intensity = emotional_context.get("primary_intensity", 0.5)
        
        if primary_intensity > 0.7:
            privacy_level += 0.2
        
        # Adjust based on content analysis
        private_keywords = ["personal", "private", "intimate", "secret", "confidential"]
        if any(keyword in content.lower() for keyword in private_keywords):
            privacy_level += 0.3
        
        # Adjust based on significance
        if significance > 0.8:
            privacy_level += 0.1
        
        return max(0.0, min(1.0, privacy_level))
    
    @function_tool
    async def _personalize_experience(self, ctx: RunContextWrapper,
                                 experience: Dict[str, Any],
                                 target_user_id: str) -> Dict[str, Any]:
        """
        Personalize an experience for a target user
        
        Args:
            experience: Experience to personalize
            target_user_id: ID of the target user
            
        Returns:
            Personalized experience
        """
        # Get target user preferences
        target_profile = await self._get_user_preference(ctx, target_user_id)
        
        # Make a copy of the experience to modify
        personalized = experience.copy()
        
        # Remove user-specific information
        if "metadata" in personalized:
            metadata = personalized["metadata"].copy()
            # Remove original user info
            if "user_id" in metadata:
                metadata["original_user_id"] = metadata.pop("user_id")
            
            # Add sharing metadata
            metadata["cross_user_shared"] = True
            metadata["shared_with"] = target_user_id
            metadata["shared_timestamp"] = datetime.datetime.now().isoformat()
            
            personalized["metadata"] = metadata
        
        # Adjust emotional context based on user preferences
        emotional_context = personalized.get("emotional_context", {})
        
        if emotional_context:
            # Calculate personalization level
            personalization_level = await self._calculate_personalization_level(
                ctx, target_user_id, experience.get("scenario_type", "general")
            )
            
            # Modify emotional context
            personalized["emotional_context"] = await self._modify_emotional_context(
                ctx, emotional_context, target_profile, personalization_level
            )
        
        # Record sharing in history
        self.sharing_history.append({
            "source_user_id": experience.get("user_id", "unknown"),
            "target_user_id": target_user_id,
            "experience_id": experience.get("id", "unknown"),
            "scenario_type": experience.get("scenario_type", "general"),
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.sharing_history) > self.max_history_size:
            self.sharing_history = self.sharing_history[-self.max_history_size:]
        
        return personalized
    
    @function_tool
    async def _modify_emotional_context(self, ctx: RunContextWrapper,
                                   emotional_context: Dict[str, Any],
                                   target_profile: Dict[str, Any],
                                   personalization_level: float) -> Dict[str, Any]:
        """
        Modify emotional context based on target user preferences
        
        Args:
            emotional_context: Emotional context to modify
            target_profile: Target user preference profile
            personalization_level: Level of personalization to apply
            
        Returns:
            Modified emotional context
        """
        # Make a copy of the emotional context
        modified = emotional_context.copy()
        
        # Get primary emotion
        primary_emotion = modified.get("primary_emotion", "neutral")
        primary_intensity = modified.get("primary_intensity", 0.5)
        
        if primary_emotion and primary_emotion.lower() in target_profile.get("emotional_preferences", {}):
            # Get target preference for this emotion
            emotion_preference = target_profile["emotional_preferences"][primary_emotion.lower()]
            
            # Calculate adjustment factor
            adjustment = (emotion_preference - 0.5) * personalization_level
            
            # Adjust intensity based on preference (higher preference = higher intensity)
            adjusted_intensity = primary_intensity + adjustment
            modified["primary_intensity"] = max(0.1, min(1.0, adjusted_intensity))
            
            # Adjust valence if present
            if "valence" in modified:
                valence = modified["valence"]
                
                # Amplify positive valence for high preference, negative for low
                if emotion_preference > 0.7 and valence > 0:
                    modified["valence"] = min(1.0, valence + 0.2 * personalization_level)
                elif emotion_preference < 0.3 and valence < 0:
                    modified["valence"] = max(-1.0, valence - 0.2 * personalization_level)
        
        return modified
    
    @function_tool
    async def _calculate_personalization_level(self, ctx: RunContextWrapper,
                                         target_user_id: str,
                                         scenario_type: str) -> float:
        """
        Calculate appropriate personalization level for a user and scenario
        
        Args:
            target_user_id: ID of the target user
            scenario_type: Type of scenario
            
        Returns:
            Personalization level (0.0-1.0)
        """
        # Get target user preferences
        target_profile = await self._get_user_preference(ctx, target_user_id)
        
        # Get overall experience sharing preference
        sharing_preference = target_profile.get("experience_sharing_preference", 0.5)
        
        # Get scenario preference
        scenario_preference = target_profile.get("scenario_preferences", {}).get(scenario_type, 0.5)
        
        # Higher preferences = higher personalization
        personalization_level = sharing_preference * 0.4 + scenario_preference * 0.6
        
        return max(0.0, min(1.0, personalization_level))
    
    # Public methods
    
    async def set_user_preference(self, 
                              user_id: str, 
                              preference_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set user preference profile for experience sharing
        
        Args:
            user_id: ID of the user
            preference_data: Preference data to update
            
        Returns:
            Updated preference profile
        """
        with trace(workflow_name="set_user_preference", group_id=self.trace_group_id):
            # Get current profile
            current_profile = await self._get_user_preference(RunContextWrapper(context=None), user_id)
            
            # Update with new data
            for key, value in preference_data.items():
                if key == "scenario_preferences" or key == "emotional_preferences":
                    if key in current_profile:
                        current_profile[key].update(value)
                else:
                    current_profile[key] = value
            
            # Store updated profile
            self.user_preference_profiles[user_id] = current_profile
            
            return current_profile
    
    async def get_user_preference(self, user_id: str) -> Dict[str, Any]:
        """
        Get user preference profile for experience sharing
        
        Args:
            user_id: ID of the user
            
        Returns:
            User preference profile
        """
        return await self._get_user_preference(RunContextWrapper(context=None), user_id)
    
    async def set_sharing_permission(self,
                                source_user_id: str,
                                target_user_id: str,
                                permission_level: Optional[float] = None,
                                allowed_scenarios: Optional[List[str]] = None,
                                excluded_scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Set sharing permission between users
        
        Args:
            source_user_id: ID of the source user
            target_user_id: ID of the target user
            permission_level: Optional permission level (0.0-1.0)
            allowed_scenarios: Optional list of allowed scenario types
            excluded_scenarios: Optional list of excluded scenario types
            
        Returns:
            Updated permission settings
        """
        with trace(workflow_name="set_sharing_permission", group_id=self.trace_group_id):
            # Calculate permission level if not provided
            if permission_level is None:
                permission_level = await self._calculate_sharing_permission(
                    RunContextWrapper(context=None),
                    source_user_id=source_user_id,
                    target_user_id=target_user_id
                )
            
            # Get compatibility for scenarios if not provided
            if allowed_scenarios is None or excluded_scenarios is None:
                compatibility = await self._check_permission_compatibility(
                    RunContextWrapper(context=None),
                    source_user_id=source_user_id,
                    target_user_id=target_user_id
                )
                
                if allowed_scenarios is None:
                    allowed_scenarios = compatibility.get("recommended_scenarios", [])
                
                if excluded_scenarios is None:
                    excluded_scenarios = compatibility.get("excluded_scenarios", [])
            
            # Update permission matrix
            result = await self._update_permission_matrix(
                RunContextWrapper(context=None),
                source_user_id=source_user_id,
                target_user_id=target_user_id,
                permission_level=permission_level,
                allowed_scenarios=allowed_scenarios,
                excluded_scenarios=excluded_scenarios
            )
            
            return result.get("permission", {})
    
    async def get_sharing_permission(self, source_user_id: str, target_user_id: str) -> Dict[str, Any]:
        """
        Get sharing permission between users
        
        Args:
            source_user_id: ID of the source user
            target_user_id: ID of the target user
            
        Returns:
            Permission settings
        """
        # Check if permission exists
        key = f"{source_user_id}:{target_user_id}"
        
        if key in self.permission_matrix:
            return self.permission_matrix[key]
        
        # Calculate and set permission if not exists
        return await self.set_sharing_permission(source_user_id, target_user_id)
    
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
        with trace(workflow_name="find_cross_user_experiences", group_id=self.trace_group_id):
            if not self.experience_interface:
                return {
                    "error": "Experience interface not available",
                    "experiences": [],
                    "count": 0
                }
            
            # Get all potential source users if not specified
            if not source_user_ids:
                source_user_ids = []
                # Get users from permission matrix
                for key in self.permission_matrix:
                    source_id, target_id = key.split(":")
                    if target_id == target_user_id and source_id != target_user_id:
                        source_user_ids.append(source_id)
                
                # Add users from sharing history
                for entry in self.sharing_history:
                    if entry.get("target_user_id") == target_user_id and entry.get("source_user_id") != target_user_id:
                        source_user_ids.append(entry.get("source_user_id"))
                
                # Get unique user IDs
                source_user_ids = list(set(source_user_ids))
                
                # Add users from same cluster if available
                if target_user_id in self.user_clusters:
                    cluster = self.user_clusters[target_user_id]
                    for user_id in cluster:
                        if user_id != target_user_id and user_id not in source_user_ids:
                            source_user_ids.append(user_id)
            
            # If no source users, return empty result
            if not source_user_ids:
                return {
                    "experiences": [],
                    "count": 0,
                    "source_user_ids": [],
                    "target_user_id": target_user_id
                }
            
            # Get experiences from each source user
            all_experiences = []
            
            for source_user_id in source_user_ids:
                # Check permission
                permission = await self.get_sharing_permission(source_user_id, target_user_id)
                permission_level = permission.get("permission_level", self.default_permission_level)
                
                # Skip if permission level is too low
                if permission_level < 0.3:
                    continue
                
                try:
                    # Search for experiences from this user
                    if hasattr(self.experience_interface, "retrieve_experiences_enhanced"):
                        experiences = await self.experience_interface.retrieve_experiences_enhanced(
                            query=query,
                            scenario_type=scenario_type,
                            limit=limit * 2,  # Get more to filter
                            user_id=source_user_id,
                            include_cross_user=False  # Only from this user
                        )
                    else:
                        experiences = []
                    
                    # Filter by permission
                    filtered_experiences = await self._filter_experiences_by_permission(
                        RunContextWrapper(context=None),
                        experiences=experiences,
                        source_user_id=source_user_id,
                        target_user_id=target_user_id
                    )
                    
                    # Add source user ID to each experience
                    for exp in filtered_experiences:
                        exp["source_user_id"] = source_user_id
                    
                    all_experiences.extend(filtered_experiences)
                except Exception as e:
                    logger.error(f"Error getting experiences from user {source_user_id}: {e}")
            
            # Score experiences for relevance
            scored_experiences = []
            
            for exp in all_experiences:
                # Calculate relevance
                relevance = await self._calculate_cross_user_relevance(
                    RunContextWrapper(context=None),
                    experience=exp,
                    target_user_id=target_user_id,
                    query=query
                )
                
                # Add to scored experiences
                scored_experiences.append((exp, relevance))
            
            # Sort by relevance
            scored_experiences.sort(key=lambda x: x[1], reverse=True)
            
            # Get top experiences
            top_experiences = scored_experiences[:limit]
            
            # Personalize experiences
            personalized_experiences = []
            relevance_scores = []
            
            for exp, relevance in top_experiences:
                if relevance >= self.min_relevance_threshold:
                    # Personalize experience
                    personalized = await self._personalize_experience(
                        RunContextWrapper(context=None),
                        experience=exp,
                        target_user_id=target_user_id
                    )
                    
                    personalized_experiences.append(personalized)
                    relevance_scores.append(relevance)
            
            return {
                "experiences": personalized_experiences,
                "count": len(personalized_experiences),
                "source_user_ids": list(set(exp["source_user_id"] for exp, _ in top_experiences)),
                "target_user_id": target_user_id,
                "relevance_scores": relevance_scores,
                "personalized": True
            }
    
    async def find_similar_users(self, user_id: str, min_similarity: float = 0.6) -> List[Dict[str, Any]]:
        """
        Find users with similar preferences
        
        Args:
            user_id: ID of the user
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar users with similarity scores
        """
        with trace(workflow_name="find_similar_users", group_id=self.trace_group_id):
            # Get user preference
            user_profile = await self.get_user_preference(user_id)
            
            # Calculate similarity with all other users
            similar_users = []
            
            for other_id, other_profile in self.user_preference_profiles.items():
                if other_id == user_id:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_user_similarity(user_profile, other_profile)
                
                if similarity >= min_similarity:
                    similar_users.append({
                        "user_id": other_id,
                        "similarity": similarity,
                        "compatible_scenarios": self._get_compatible_scenarios(user_profile, other_profile)
                    })
            
            # Sort by similarity
            similar_users.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similar_users
    
    def _calculate_user_similarity(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> float:
        """
        Calculate similarity between user profiles
        
        Args:
            profile1: First user profile
            profile2: Second user profile
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Calculate scenario preference similarity
        scenario_sim = 0.0
        scenario_count = 0
        
        for scenario, pref1 in profile1.get("scenario_preferences", {}).items():
            pref2 = profile2.get("scenario_preferences", {}).get(scenario, 0.5)
            scenario_sim += 1.0 - abs(pref1 - pref2)
            scenario_count += 1
        
        scenario_similarity = scenario_sim / scenario_count if scenario_count > 0 else 0.5
        
        # Calculate emotional preference similarity
        emotion_sim = 0.0
        emotion_count = 0
        
        for emotion, pref1 in profile1.get("emotional_preferences", {}).items():
            pref2 = profile2.get("emotional_preferences", {}).get(emotion, 0.5)
            emotion_sim += 1.0 - abs(pref1 - pref2)
            emotion_count += 1
        
        emotion_similarity = emotion_sim / emotion_count if emotion_count > 0 else 0.5
        
        # Calculate sharing preference similarity
        share_pref1 = profile1.get("experience_sharing_preference", 0.5)
        share_pref2 = profile2.get("experience_sharing_preference", 0.5)
        sharing_similarity = 1.0 - abs(share_pref1 - share_pref2)
        
        # Calculate overall similarity
        similarity = (
            scenario_similarity * 0.5 +
            emotion_similarity * 0.3 +
            sharing_similarity * 0.2
        )
        
        return similarity
    
    def _get_compatible_scenarios(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> List[str]:
        """
        Get most compatible scenario types between users
        
        Args:
            profile1: First user profile
            profile2: Second user profile
            
        Returns:
            List of compatible scenario types
        """
        compatibility = []
        
        for scenario, pref1 in profile1.get("scenario_preferences", {}).items():
            pref2 = profile2.get("scenario_preferences", {}).get(scenario, 0.5)
            
            # Both preferences high = compatible
            if pref1 >= 0.6 and pref2 >= 0.6:
                compatibility.append(scenario)
        
        return compatibility
    
    async def update_user_clusters(self) -> Dict[str, Any]:
        """
        Update user clusters based on similarity
        
        Returns:
            Updated cluster information
        """
        with trace(workflow_name="update_user_clusters", group_id=self.trace_group_id):
            # Get all users
            all_users = list(self.user_preference_profiles.keys())
            
            # Calculate similarity matrix
            similarity_matrix = {}
            
            for user1 in all_users:
                similarity_matrix[user1] = {}
                profile1 = await self.get_user_preference(user1)
                
                for user2 in all_users:
                    if user1 == user2:
                        similarity_matrix[user1][user2] = 1.0
                        continue
                    
                    profile2 = await self.get_user_preference(user2)
                    similarity = self._calculate_user_similarity(profile1, profile2)
                    similarity_matrix[user1][user2] = similarity
            
            # Store similarity matrix
            self.cluster_similarity = similarity_matrix
            
            # Create clusters
            clusters = {}
            clustered_users = set()
            
            # For each user, find similar users
            for user_id in all_users:
                if user_id in clustered_users:
                    continue
                
                # Find similar users
                similar = []
                for other_id, similarity in similarity_matrix[user_id].items():
                    if other_id != user_id and similarity >= 0.7:
                        similar.append(other_id)
                
                # Create cluster
                cluster = [user_id] + similar
                cluster_id = f"cluster_{len(clusters) + 1}"
                clusters[cluster_id] = cluster
                
                # Mark users as clustered
                clustered_users.update(cluster)
            
            # For remaining users, assign to most similar cluster
            for user_id in all_users:
                if user_id not in clustered_users:
                    # Find most similar user
                    most_similar = max(
                        [(other_id, similarity_matrix[user_id][other_id]) for other_id in all_users if other_id != user_id],
                        key=lambda x: x[1]
                    )
                    
                    # Find cluster containing most similar user
                    for cluster_id, cluster in clusters.items():
                        if most_similar[0] in cluster:
                            cluster.append(user_id)
                            clustered_users.add(user_id)
                            break
            
            # Create user to cluster mapping
            user_clusters = {}
            for cluster_id, cluster in clusters.items():
                for user_id in cluster:
                    user_clusters[user_id] = cluster
            
            # Store mapping
            self.user_clusters = user_clusters
            
            return {
                "clusters": clusters,
                "user_clusters": user_clusters,
                "cluster_count": len(clusters)
            }
    
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
            if not self.experience_interface or not self.memory_core:
                return {
                    "error": "Experience interface or memory core not available",
                    "shared": False
                }
            
            try:
                # Get the experience
                if hasattr(self.memory_core, "get_memory_by_id"):
                    experience = await self.memory_core.get_memory_by_id(experience_id)
                else:
                    return {
                        "error": "Cannot retrieve experience",
                        "shared": False
                    }
                
                if not experience:
                    return {
                        "error": f"Experience {experience_id} not found",
                        "shared": False
                    }
                
                # Check permission
                permission = await self.get_sharing_permission(source_user_id, target_user_id)
                permission_level = permission.get("permission_level", self.default_permission_level)
                
                # Calculate privacy level
                privacy_level = await self._calculate_experience_privacy_level(
                    RunContextWrapper(context=None),
                    experience
                )
                
                # Check if sharing is allowed
                if privacy_level > permission_level:
                    return {
                        "error": "Privacy level too high for sharing",
                        "shared": False,
                        "privacy_level": privacy_level,
                        "permission_level": permission_level
                    }
                
                # Check scenario type
                scenario_type = experience.get("scenario_type", "general")
                excluded_scenarios = permission.get("excluded_scenario_types", [])
                
                if scenario_type in excluded_scenarios:
                    return {
                        "error": f"Scenario type {scenario_type} is excluded from sharing",
                        "shared": False
                    }
                
                # Personalize the experience
                personalized = await self._personalize_experience(
                    RunContextWrapper(context=None),
                    experience=experience,
                    target_user_id=target_user_id
                )
                
                # Store personalized experience in target's memory
                if hasattr(self.memory_core, "add_memory"):
                    # Create metadata
                    metadata = personalized.get("metadata", {}).copy()
                    metadata["cross_user_shared"] = True
                    metadata["source_user_id"] = source_user_id
                    metadata["source_experience_id"] = experience_id
                    metadata["shared_timestamp"] = datetime.datetime.now().isoformat()
                    
                    # Store in memory
                    memory_id = await self.memory_core.add_memory(
                        memory_text=personalized.get("content", ""),
                        memory_type="experience",
                        memory_scope="game",
                        significance=personalized.get("significance", 5),
                        tags=personalized.get("tags", []) + ["cross_user_shared"],
                        metadata=metadata
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
        # Initialize statistics
        stats = {
            "total_shares": 0,
            "user_shares": {},
            "scenario_shares": {},
            "timestamp_first": None,
            "timestamp_last": None
        }
        
        # Filter history by user if specified
        if user_id:
            filtered_history = [entry for entry in self.sharing_history 
                              if entry.get("source_user_id") == user_id or entry.get("target_user_id") == user_id]
        else:
            filtered_history = self.sharing_history
        
        if not filtered_history:
            return stats
        
        # Calculate statistics
        stats["total_shares"] = len(filtered_history)
        
        # Count shares by user
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
        
        # Calculate most active users
        if stats["user_shares"]:
            # Most shared
            stats["most_active_sharer"] = max(
                stats["user_shares"].items(),
                key=lambda x: x[1]["shared"]
            )[0]
            
            # Most received
            stats["most_active_receiver"] = max(
                stats["user_shares"].items(),
                key=lambda x: x[1]["received"]
            )[0]
        
        # Calculate most shared scenario type
        if stats["scenario_shares"]:
            stats["most_shared_scenario"] = max(
                stats["scenario_shares"].items(),
                key=lambda x: x[1]
            )[0]
        
        return stats
