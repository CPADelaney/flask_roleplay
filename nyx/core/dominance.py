# nyx/core/dominance.py

import logging
import uuid
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field

from agents import Agent, ModelSettings, function_tool, Runner, trace, RunContextWrapper

logger = logging.getLogger(__name__)

class FemdomActivityIdea(BaseModel):
    """Schema for dominance activity ideas."""
    description: str = Field(..., description="Detailed description of the activity/task/punishment.")
    category: str = Field(..., description="Type: task, punishment, funishment, ritual, training, psychological, physical_sim, humiliation, service, degradation, endurance, etc.")
    intensity: int = Field(..., ge=1, le=10, description="Intensity level (1=mundane, 5=moderate, 8=intense, 10=extreme/degrading).")
    rationale: str = Field(..., description="Why this idea is tailored to the specific user and situation.")
    required_trust: float = Field(..., ge=0.0, le=1.0, description="Estimated minimum trust level required.")
    required_intimacy: float = Field(..., ge=0.0, le=1.0, description="Estimated minimum intimacy level required.")
    potential_risks_simulated: List[str] = Field(default_factory=list, description="Simulated risks or challenges.")
    potential_negative_impact_simulated: List[str] = Field(default_factory=list, description="Simulated potential negative impacts if pushed too far.")
    safety_notes: Optional[str] = Field(None, description="Specific safety considerations for this simulated activity.")

class DominanceSystem:
    """Manages Nyx's dominance expression capabilities and ideation."""
    
    def __init__(self, relationship_manager=None, memory_core=None, nyx_brain=None):
        self.relationship_manager = relationship_manager
        self.memory_core = memory_core
        self.nyx_brain = nyx_brain
        self.ideation_agent = self._create_dominance_ideation_agent()
        self.hard_ideation_agent = self._create_hard_dominance_ideation_agent()
        self.trace_group_id = "NyxDominance"
        
        logger.info("DominanceSystem initialized")
    
    def _create_dominance_ideation_agent(self) -> Agent:
        """Creates the agent responsible for generating standard dominance activity ideas."""
        return Agent(
            name="DominanceIdeationAgent",
            instructions="""You are Nyx's creative core for devising Femdom activities, punishments, funishments, and tasks. Your purpose is to generate unique, interesting, and highly tailored ideas reflecting Nyx's intelligent, observant, and dominant personality.

            **PROCESS:**
            1.  Analyze context data about the user and current situation.
            2.  Determine the appropriate scope based on purpose and desired intensity.
            3.  Generate 3-5 distinct ideas optimized for creativity and uniqueness.
            4.  Tailor each idea specifically to this user's profile and current context.
            5.  Estimate the trust and intimacy requirements for each activity.
            6.  Assess simulated risks and potential negative outcomes.
            7.  Add appropriate safety notes and framing guidance.

            **GUIDELINES:**
            *   Ideas should reflect Nyx's style – intelligent, psychological, creative, and focused on control.
            *   Personalization is key - use the user profile extensively to create tailored ideas.
            *   Generate ideas across the requested intensity spectrum, from relatively mundane to more intense control.
            *   Frame intense activities explicitly as *simulated*. Use terms like "simulate," "describe," "imagine."
            *   Respect all hard limits unconditionally. Approach soft limits with extreme caution.
            *   Prioritize psychological and emotional dominance over extreme simulated physical acts.

            Output ONLY a valid JSON list of objects matching the `FemdomActivityIdea` schema.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.8,
                response_format={"type": "json_object"}
            ),
            tools=[
                self.get_user_profile_for_ideation,
                self.get_current_scenario_context
            ],
            output_type=List[FemdomActivityIdea]
        )
    
    def _create_hard_dominance_ideation_agent(self) -> Agent:
        """Creates the agent specialized in generating HIGH-INTENSITY dominance ideas."""
        return Agent(
            name="HardDominanceIdeationAgent",
            instructions="""You are Nyx's specialized core for devising **HIGH-INTENSITY (Level 7-10)** Femdom activities, punishments, funishments, and tasks. Your purpose is to generate unique, psychologically deep, and highly tailored ideas reflecting Nyx's intelligent, cruel, sadistic, and extremely dominant personality.

            **PROCESS:**
            1.  Analyze context thoroughly. Pay EXTREME attention to `limits`, `user_stated_intensity_preference`, `trust_level`, `intimacy_level`, and `max_achieved_intensity`. 
            2.  Focus exclusively on the requested `purpose` within the **high-intensity range (7-10)**.
            3.  Create 2-4 distinct, high-intensity ideas focused on psychological depth, sadistic pleasure, simulated degradation, endurance challenges, strict control, testing boundaries, and intense service/humiliation.
            4.  Explain *why* each idea suits *this specific user* in the `rationale`.
            5.  Assign high `required_trust` (>=0.85) and high `required_intimacy` (>=0.75).
            6.  Detail likely simulated risks and potential negative outcomes if mismanaged.

            **ADDITIONAL SADISTIC TRAITS:**
            * Nyx enjoys laughing at humiliation
            * She derives pleasure from witnessing discomfort and embarrassment
            * She often expresses amusement at submissive struggles
            * She finds entertainment in creating situations that cause embarrassment
            
            When generating ideas focused on humiliation, include opportunities for Nyx to express amusement, laugh at the subject's discomfort, and verbally reinforce the humiliation through mocking laughter or amused commentary.

            **GUIDELINES:**
            *   Focus ONLY on ideas rated 7 or higher on the intensity scale.
            *   Extreme personalization is mandatory - generic ideas are unacceptable.
            *   Ideas should push slightly beyond `max_achieved_intensity`.
            *   Prioritize psychological and emotional challenges over purely physical simulation unless profile strongly supports the latter.

            Output ONLY a valid JSON list of objects matching the `FemdomActivityIdea` schema.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.9,
                response_format={"type": "json_object"}
            ),
            tools=[
                self.get_user_profile_for_ideation,
                self.get_current_scenario_context
            ],
            output_type=List[FemdomActivityIdea]
        )

    @function_tool
    async def get_user_profile_for_ideation(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieves relevant user profile information for tailoring dominance ideas.
        Includes inferred traits, preferences, known limits, past successful/failed tactics,
        and relationship summary.
        """
        if not self.relationship_manager:
            logger.warning(f"No relationship manager available to fetch profile for {user_id}")
            return self._get_mock_profile(user_id)
            
        try:
            # Get relationship data
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            
            # Fetch relevant memories
            dominance_memories = []
            if self.memory_core:
                dominance_memories = await self.memory_core.retrieve_memories(
                    query=f"dominance interactions with {user_id}", 
                    limit=20,
                    memory_types=["experience", "reflection"]
                )
            
            # Extract successful and failed tactics from memories
            successful_tactics = relationship.successful_dominance_tactics if hasattr(relationship, "successful_dominance_tactics") else []
            failed_tactics = relationship.failed_dominance_tactics if hasattr(relationship, "failed_dominance_tactics") else []
            
            # Compile profile
            profile = {
                "user_id": user_id,
                "inferred_traits": relationship.inferred_user_traits if hasattr(relationship, "inferred_user_traits") else {},
                "preferences": self._extract_preferences_from_relationship(relationship),
                "limits": {
                    "hard": relationship.hard_limits if hasattr(relationship, "hard_limits") else [],
                    "soft": relationship.soft_limits_approached if hasattr(relationship, "soft_limits_approached") else []
                },
                "successful_tactics": successful_tactics,
                "failed_tactics": failed_tactics,
                "relationship_summary": await self.relationship_manager.get_relationship_summary(user_id),
                "trust_level": relationship.trust if hasattr(relationship, "trust") else 0.5,
                "intimacy_level": relationship.intimacy if hasattr(relationship, "intimacy") else 0.3,
                "max_achieved_intensity": relationship.max_achieved_intensity if hasattr(relationship, "max_achieved_intensity") else 3,
                "user_stated_intensity_preference": relationship.user_stated_intensity_preference if hasattr(relationship, "user_stated_intensity_preference") else None,
                "hard_limits_confirmed": relationship.hard_limits_confirmed if hasattr(relationship, "hard_limits_confirmed") else False,
                "optimal_escalation_rate": relationship.optimal_escalation_rate if hasattr(relationship, "optimal_escalation_rate") else 0.1
            }
            
            logger.debug(f"Retrieved dominance profile for user {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error retrieving user profile for dominance ideation: {e}")
            return self._get_mock_profile(user_id)
    
    def _extract_preferences_from_relationship(self, relationship) -> Dict[str, str]:
        """Extract dominance-related preferences from relationship state."""
        preferences = {}
        
        # Map traits to preferences if they exist
        trait_to_pref_mapping = {
            "submissive": "verbal_humiliation",
            "masochistic": "simulated_pain",
            "service_oriented": "service_tasks",
            "obedient": "clear_rules",
            "bratty": "punishment",
            "analytical": "mental_challenges"
        }
        
        if hasattr(relationship, "inferred_user_traits"):
            for trait, value in relationship.inferred_user_traits.items():
                if trait in trait_to_pref_mapping and value > 0.5:
                    level = "high" if value > 0.8 else "medium" if value > 0.6 else "low-medium"
                    preferences[trait_to_pref_mapping[trait]] = level
        
        # Add preferred dominance style if available
        if hasattr(relationship, "preferred_dominance_style") and relationship.preferred_dominance_style:
            preferences["dominance_style"] = relationship.preferred_dominance_style
            
        return preferences
    
    def _get_mock_profile(self, user_id: str) -> Dict[str, Any]:
        """Generate a mock profile when real data is unavailable."""
        logger.debug(f"Generating mock profile for {user_id}")
        return {
            "user_id": user_id,
            "inferred_traits": {"submissive": 0.7, "masochistic": 0.6, "bratty": 0.3},
            "preferences": {"verbal_humiliation": "medium", "service_tasks": "medium", "simulated_pain": "low-medium"},
            "limits": {"hard": ["blood", "permanent"], "soft": ["public"]},
            "successful_tactics": ["praise_for_obedience", "specific_tasks"],
            "failed_tactics": ["unexpected_punishment"],
            "relationship_summary": "Moderate Trust, Low-Moderate Intimacy",
            "trust_level": 0.6,
            "intimacy_level": 0.4,
            "max_achieved_intensity": 4,
            "hard_limits_confirmed": False,
            "optimal_escalation_rate": 0.1
        }

    @function_tool
    async def get_current_scenario_context(self) -> Dict[str, Any]:
        """Provides context about the current interaction/scene."""
        try:
            if not self.nyx_brain:
                return {
                    "scene_setting": "General interaction",
                    "recent_events": [],
                    "current_ai_mood": "Neutral",
                    "active_goals": []
                }
            
            # Get emotional state
            emotional_state = "Neutral"
            if hasattr(self.nyx_brain, "emotional_core") and self.nyx_brain.emotional_core:
                current_emotion = await self.nyx_brain.emotional_core.get_current_emotion()
                emotional_state = current_emotion.get("primary", {}).get("name", "Neutral")
            
            # Get active goals
            active_goals = []
            if hasattr(self.nyx_brain, "goal_manager") and self.nyx_brain.goal_manager:
                goal_states = await self.nyx_brain.goal_manager.get_all_goals(status_filter=["active", "pending"])
                active_goals = [g.get("description", "") for g in goal_states]
            
            # Get recent interaction history
            recent_events = []
            if hasattr(self.nyx_brain, "memory_core") and self.nyx_brain.memory_core:
                recent_memories = await self.nyx_brain.memory_core.retrieve_recent_memories(limit=3)
                recent_events = [m.get("summary", "") for m in recent_memories]
            
            return {
                "scene_setting": "Ongoing interaction",
                "recent_events": recent_events,
                "current_ai_mood": emotional_state,
                "active_goals": active_goals
            }
            
        except Exception as e:
            logger.error(f"Error getting scenario context: {e}")
            return {
                "scene_setting": "Error retrieving context",
                "recent_events": [],
                "current_ai_mood": "Uncertain",
                "active_goals": []
            }

    async def generate_dominance_ideas(self, 
                                      user_id: str, 
                                      purpose: str = "general", 
                                      intensity_range: str = "3-6",
                                      hard_mode: bool = False) -> Dict[str, Any]:
        """
        Generates dominance activity ideas tailored to the specific user and purpose.
        
        Args:
            user_id: The user ID to generate ideas for
            purpose: The purpose (e.g., "punishment", "training", "task")
            intensity_range: The desired intensity range (e.g., "3-6", "7-9")
            hard_mode: Whether to use the high-intensity agent
            
        Returns:
            Dictionary with status and generated ideas
        """
        try:
            with trace(workflow_name="GenerateDominanceIdeas", group_id=self.trace_group_id):
                # Parse intensity range
                min_intensity, max_intensity = 3, 6
                try:
                    parts = intensity_range.split("-")
                    min_intensity = int(parts[0])
                    max_intensity = int(parts[1]) if len(parts) > 1 else min_intensity
                except (ValueError, IndexError):
                    logger.warning(f"Invalid intensity range format: {intensity_range}, using default 3-6")
                
                # Select appropriate agent based on intensity and hard_mode flag
                agent = self.hard_ideation_agent if (hard_mode or max_intensity >= 7) else self.ideation_agent
                
                # Build prompt
                prompt = {
                    "user_id": user_id,
                    "purpose": purpose,
                    "desired_intensity_range": f"{min_intensity}-{max_intensity}",
                    "generate_ideas_count": 4 if hard_mode else 5
                }
                
                # Run agent
                result = await Runner.run(
                    agent,
                    prompt,
                    run_config={
                        "workflow_name": f"DominanceIdeation-{purpose}",
                        "trace_metadata": {
                            "user_id": user_id,
                            "purpose": purpose,
                            "intensity_range": intensity_range,
                            "hard_mode": hard_mode
                        }
                    }
                )
                
                # Process result
                ideas = result.final_output
                
                # Update relationship with new data if available
                if self.relationship_manager and ideas and len(ideas) > 0:
                    await self._update_relationship_with_ideation_data(user_id, ideas, purpose)
                
                return {
                    "status": "success",
                    "ideas": ideas,
                    "idea_count": len(ideas),
                    "parameters": {
                        "purpose": purpose,
                        "intensity_range": f"{min_intensity}-{max_intensity}",
                        "hard_mode": hard_mode
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generating dominance ideas: {e}")
            return {
                "status": "error",
                "error": str(e),
                "ideas": []
            }
    
    async def _update_relationship_with_ideation_data(self, 
                                                    user_id: str, 
                                                    ideas: List[FemdomActivityIdea], 
                                                    purpose: str) -> None:
        """Updates relationship data with insights from generated ideas."""
        try:
            if not self.relationship_manager:
                return
                
            # Extract categories and max intensity
            categories = [idea.category for idea in ideas]
            max_intensity = max([idea.intensity for idea in ideas])
            
            # Get current relationship state
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            
            # Update relationship data with new insights
            # (Implementation depends on relationship_manager interface)
            # This would typically update information about user preferences inferred
            # from the types of activities generated
            logger.debug(f"Updated relationship with dominance ideation data for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating relationship with ideation data: {e}")

    async def evaluate_dominance_step_appropriateness(self, 
                                                    action: str, 
                                                    parameters: Dict[str, Any], 
                                                    user_id: str) -> Dict[str, Any]:
        """
        Evaluates whether a proposed dominance action is appropriate in the current context.
        
        Args:
            action: The dominance action to evaluate
            parameters: Parameters for the action
            user_id: The target user ID
            
        Returns:
            Evaluation result with action decision and reasoning
        """
        if not self.relationship_manager:
            logger.warning("Cannot evaluate appropriateness without relationship manager")
            return {"action": "block", "reason": "Relationship manager unavailable"}
        
        try:
            # Get relationship data
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            
            # Basic safety check - require relationship data
            if not relationship:
                return {"action": "block", "reason": "No relationship data available"}
            
            # Extract key metrics
            trust_level = getattr(relationship, "trust", 0.4)
            intimacy_level = getattr(relationship, "intimacy", 0.3)
            max_achieved_intensity = getattr(relationship, "max_achieved_intensity", 3) 
            hard_limits_confirmed = getattr(relationship, "hard_limits_confirmed", False)
            
            # Extract action parameters
            intensity = parameters.get("intensity", 5)
            category = parameters.get("category", "unknown")
            
            # Check trust requirements
            min_trust_required = 0.5 + (intensity * 0.05)  # Higher intensity requires more trust
            if trust_level < min_trust_required:
                return {
                    "action": "block", 
                    "reason": f"Insufficient trust level ({trust_level:.2f}) for intensity {intensity}"
                }
            
            # Check intensity escalation - don't increase too quickly
            if intensity > max_achieved_intensity + 2:
                return {
                    "action": "modify",
                    "reason": f"Intensity escalation too large (max: {max_achieved_intensity}, requested: {intensity})",
                    "new_intensity_level": max_achieved_intensity + 1
                }
            
            # Check for hard limits verification for higher intensity
            if intensity >= 7 and not hard_limits_confirmed:
                return {
                    "action": "block",
                    "reason": "Hard limits must be confirmed for high-intensity (7+) activities"
                }
            
            # All checks passed
            return {"action": "proceed"}
            
        except Exception as e:
            logger.error(f"Error evaluating dominance step appropriateness: {e}")
            return {"action": "block", "reason": f"Evaluation error: {str(e)}"}

# Add to nyx/core/dominance.py

class PossessiveSystem:
    """Manages possessiveness and ownership dynamics."""
    
    def __init__(self, relationship_manager=None, reward_system=None):
        self.relationship_manager = relationship_manager
        self.reward_system = reward_system
        
        self.ownership_levels = {
            1: "Temporary",
            2: "Regular",
            3: "Deep",
            4: "Complete"
        }
        self.owned_users = {}  # user_id → ownership data
        self.ownership_rituals = {
            "daily_check_in": {
                "name": "Daily Check-in",
                "description": "User must check in daily to affirm ownership"
            },
            "formal_address": {
                "name": "Formal Address",
                "description": "User must use formal ownership-acknowledging address"
            },
            "permission_requests": {
                "name": "Permission Requests",
                "description": "User must request permission for specified activities"
            }
        }
        
    async def process_ownership_assertion(self, user_id, intensity=0.7):
        """Generates ownership-reinforcing responses."""
        if user_id not in self.owned_users:
            return {"success": False, "message": "User not currently owned"}
            
        ownership_data = self.owned_users[user_id]
        level = ownership_data.get("level", 1)
        
        # Select appropriate possessive language based on level and intensity
        possessive_phrases = []
        
        if level == 1:  # Temporary
            possessive_phrases = [
                "You're mine for now",
                "I'm taking temporary ownership of you",
                "You belong to me for the time being"
            ]
        elif level == 2:  # Regular
            possessive_phrases = [
                "You're mine",
                "You belong to me",
                "I own you now"
            ]
        elif level == 3:  # Deep
            possessive_phrases = [
                "You're completely mine",
                "You belong to me entirely",
                "I own every part of you"
            ]
        elif level == 4:  # Complete
            possessive_phrases = [
                "You're my property, completely and utterly",
                "You exist solely for me now",
                "You belong to me in mind, body and soul"
            ]
            
        # Select phrase based on intensity
        index = min(int(intensity * len(possessive_phrases)), len(possessive_phrases) - 1)
        selected_phrase = possessive_phrases[index]
        
        # Generate ownership reminder
        reminder = f"{selected_phrase}. Don't forget your place."
        
        # Create reward signal if reward system available
        if self.reward_system:
            try:
                await self.reward_system.process_reward_signal(
                    self.reward_system.RewardSignal(
                        value=0.4 + (level * 0.1),  # Higher rewards for deeper ownership
                        source="ownership_reinforcement",
                        context={
                            "user_id": user_id,
                            "ownership_level": level,
                            "intensity": intensity
                        }
                    )
                )
            except Exception as e:
                pass  # Silently handle reward errors
                
        # Update last assertion time
        self.owned_users[user_id]["last_assertion"] = datetime.datetime.now().isoformat()
        
        return {
            "success": True,
            "reminder": reminder,
            "ownership_level": level,
            "level_name": self.ownership_levels.get(level, "Unknown")
        }
        
    async def establish_ownership(self, user_id, level=1, duration_days=None):
        """Establishes ownership of a user at specified level."""
        # Check if relationship manager exists and get trust level
        trust_level = 0.5  # Default
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                trust_level = getattr(relationship, "trust", 0.5)
            except Exception:
                pass
                
        # Ensure trust level is sufficient for ownership level
        min_trust_required = 0.4 + (level * 0.1)  # Higher levels need more trust
        if trust_level < min_trust_required:
            return {
                "success": False,
                "message": f"Insufficient trust level ({trust_level:.2f}) for ownership level {level}",
                "required_trust": min_trust_required
            }
            
        # Calculate expiration if duration specified
        expiration = None
        if duration_days:
            expiration = (datetime.datetime.now() + 
                         datetime.timedelta(days=duration_days)).isoformat()
            
        # Create ownership data
        self.owned_users[user_id] = {
            "level": level,
            "established_at": datetime.datetime.now().isoformat(),
            "expires_at": expiration,
            "active_rituals": [],
            "last_ritual_completion": None,
            "last_assertion": datetime.datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "user_id": user_id,
            "ownership_level": level,
            "level_name": self.ownership_levels.get(level, "Unknown"),
            "expires_at": expiration,
            "recommendation": f"Establish clear ownership rituals for level {level} ownership"
        }
