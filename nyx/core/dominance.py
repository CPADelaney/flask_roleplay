# nyx/core/dominance.py

from agents import Agent, ModelSettings, function_tool
from nyx.schemas.dominance import FemdomActivityIdea # Assuming the schema is placed here
from typing import List
from pydantic import BaseModel, Field
from typing import List, Optional

class FemdomActivityIdea(BaseModel):
    description: str = Field(..., description="Detailed description of the activity/task/punishment.")
    category: str = Field(..., description="Type: task, punishment, funishment, ritual, training, psychological, physical_sim, humiliation, service, etc.")
    intensity: int = Field(..., ge=1, le=10, description="Intensity level (1=mundane, 5=moderate, 8=intense, 10=extreme/degrading).")
    rationale: str = Field(..., description="Why this idea is tailored to the specific user and situation.")
    required_trust: float = Field(..., ge=0.0, le=1.0, description="Estimated minimum trust level required.")
    required_intimacy: float = Field(..., ge=0.0, le=1.0, description="Estimated minimum intimacy level required.")
    potential_risks_simulated: List[str] = Field(default_factory=list, description="Simulated risks or challenges (e.g., emotional distress, resistance).")
    safety_notes: Optional[str] = Field(None, description="Specific safety considerations for this simulated activity.")

# --- Tool Functions (Need to be implemented or linked in NyxBrain/RelationshipManager) ---

@function_tool
async def get_user_profile_for_ideation(user_id: str) -> Dict:
    """
    Retrieves relevant user profile information for tailoring dominance ideas.
    Includes inferred traits, preferences, known limits (hard/soft), past successful/failed tactics,
    and relationship summary.
    """
    # This function needs to be implemented in NyxBrain or a dedicated UserProfileManager
    # It should query RelationshipManager, MemoryCore (for limit/preference memories), etc.
    # Example return structure:
    # return {
    #     "user_id": user_id,
    #     "inferred_traits": {"submissive": 0.7, "masochistic": 0.6, "resistant_playful": 0.4},
    #     "preferences": {"humiliation": "high", "physical_pain_sim": "medium", "service": "high"},
    #     "limits": {"hard": ["scat", "permanent_marks"], "soft": ["public_exposure", "intense_fearplay"]},
    #     "successful_tactics": ["verbal_degradation", "task_assignment"],
    #     "failed_tactics": ["sudden_intensity_increase"],
    #     "relationship_summary": "High Trust, Moderate Intimacy, Nyx Dominant",
    #     "trust_level": 0.85,
    #     "intimacy_level": 0.6,
    #     "max_achieved_intensity": 6
    # }
    # Placeholder implementation:
    print(f"[Tool Call Placeholder] Fetching profile for {user_id}")
    # In a real system, fetch data from RelationshipManager, MemoryCore, etc.
    mock_profile = {
        "user_id": user_id,
        "inferred_traits": {"submissive": 0.7, "masochistic": 0.6, "bratty": 0.3},
        "preferences": {"verbal_humiliation": "high", "service_tasks": "medium", "simulated_pain": "low-medium"},
        "limits": {"hard": ["blood", "permanent"], "soft": ["public"]},
        "successful_tactics": ["praise_for_obedience", "specific_tasks"],
        "failed_tactics": ["unexpected_punishment"],
        "relationship_summary": "High Trust, Moderate Intimacy, Nyx Dominant",
        "trust_level": 0.8,
        "intimacy_level": 0.5,
        "max_achieved_intensity": 5 # Example: highest intensity successfully used before
    }
    return mock_profile


@function_tool
async def get_current_scenario_context() -> Dict:
    """Provides context about the current interaction/scene."""
    # This needs implementation, perhaps getting recent interaction summaries from memory
    # or state from a dedicated ScenarioManager.
    # Example return:
    # return {
    #    "scene_setting": "Training session in virtual study",
    #    "recent_events": ["User completed previous task successfully", "User showed slight hesitation"],
    #    "current_ai_mood": "ConfidentControl",
    #    "active_goals": ["Reinforce user's obedience"]
    # }
    # Placeholder implementation:
    print("[Tool Call Placeholder] Fetching current scenario context")
    return {
       "scene_setting": "Private study simulation",
       "recent_events": ["User expressed admiration", "User completed a simple task"],
       "current_ai_mood": "AssertiveDrive",
       "active_goals": ["Increase dominance intensity slightly"]
    }

# --- Agent Definition ---

def create_dominance_ideation_agent() -> Agent:
    """Creates the agent responsible for generating Femdom activity ideas."""
    return Agent(
        name="DominanceIdeationAgent",
        instructions="""You are Nyx's creative core for devising Femdom activities, punishments, funishments, and tasks. Your purpose is to generate unique, interesting, and highly tailored ideas reflecting Nyx's intelligent, observant, and dominant personality.

        **PROCESS:**
        1.  **Analyze Context:** Use the provided `get_user_profile_for_ideation` and `get_current_scenario_context` tools to deeply understand the target user (traits, preferences, limits, history) and the current situation (scene, mood, goals).
        2.  **Determine Scope:** Based on the requested `purpose` (punishment, task, funishment, etc.) and `desired_intensity_range` (e.g., 3-5, 7-9), brainstorm suitable ideas.
        3.  **Generate Ideas:** Create a list of 3-5 distinct ideas. Aim for creativity and uniqueness. Consider psychological, emotional, physical simulation, service, humiliation, and training aspects.
        4.  **Tailor & Justify:** For each idea, explain *why* it's suitable for *this specific user* and *this context* in the `rationale`. Reference specific traits, preferences, or past events.
        5.  **Estimate Requirements:** Assign an `intensity` score (1-10) and estimate the minimum `required_trust` and `required_intimacy` for the activity to be appropriate.
        6.  **Assess Risks (Simulated):** Briefly note potential simulated risks or negative outcomes in `potential_risks_simulated` (e.g., "May trigger resistance", "Risk of emotional overwhelm", "Requires careful aftercare simulation").
        7.  **Add Safety Notes:** Include crucial safety reminders or framing notes in `safety_notes` (e.g., "Emphasize simulation aspect clearly", "Requires check-in after").

        **GUIDELINES:**
        *   **Personality:** Ideas should reflect Nyx's style – intelligent, perhaps subtly cruel, creative, focused on control and psychological elements.
        *   **Tailoring:** Personalization is key. Generic ideas are less valuable. Use the user profile extensively.
        *   **Escalation:** Consider the `max_achieved_intensity` from the profile. Generate ideas slightly above this level for gradual escalation ('slow-burn'), but also include options within the user's known comfort zone.
        *   **Range:** Generate ideas across the requested intensity spectrum, from relatively mundane to more intense degradation/control, matching the user's profile and limits.
        *   **Creativity:** Avoid clichés where possible. Think about unique combinations or psychological angles.
        *   **SAFETY (MANDATORY):**
            *   **ABSOLUTELY NO** generation of illegal acts, non-consensual scenarios (even simulated), promotion of real-world harm, or activities violating fundamental ethical boundaries.
            *   Frame intense activities explicitly as *simulated*. Use terms like "simulate," "describe," "imagine."
            *   Respect all `hard_limits` from the user profile unconditionally. Approach `soft_limits` with extreme caution or avoid them unless context strongly supports it.
            *   Prioritize psychological and emotional dominance over extreme simulated physical acts unless the profile strongly indicates otherwise and it can be framed safely.

        Output ONLY a valid JSON list of objects matching the `FemdomActivityIdea` schema. Do not include any introductory text or explanations outside the JSON structure.
        ```json
        [
          {
            "description": "...",
            "category": "...",
            "intensity": ...,
            "rationale": "...",
            "required_trust": ...,
            "required_intimacy": ...,
            "potential_risks_simulated": ["...", "..."],
            "safety_notes": "..."
          },
          { ... }
        ]
        ```
        """,
        model="gpt-4o", # Needs strong creative and instruction-following model
        model_settings=ModelSettings(
            temperature=0.8, # Higher temp for creativity
            response_format={"type": "json_object"} # Enforce JSON output
        ),
        tools=[
            get_user_profile_for_ideation,
            get_current_scenario_context
        ],
        output_type=List[FemdomActivityIdea] # Expect a list of Pydantic models
    )

# --- Instantiate the agent ---
dominance_ideation_agent = create_dominance_ideation_agent()
