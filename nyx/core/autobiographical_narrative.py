# nyx/core/autobiographical_narrative.py

import logging
import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import asyncio

# Assume Agent SDK and other core modules are importable
try:
    from agents import Agent, Runner, ModelSettings, trace
    AGENT_SDK_AVAILABLE = True
except ImportError:
    AGENT_SDK_AVAILABLE = False
    # Dummy classes
    class Agent: pass
    class Runner: pass
    class ModelSettings: pass
    def trace(workflow_name, group_id):
         # ... (dummy trace context manager) ...
         pass

# Assume MemoryCore, IdentityEvolution, RelationshipManager are importable
# from nyx.core.memory_core import MemoryCore
# from nyx.core.identity_evolution import IdentityEvolutionSystem
# from nyx.core.relationship_manager import RelationshipManager

logger = logging.getLogger(__name__)

class NarrativeSegment(BaseModel):
    """A segment of the autobiographical narrative."""
    segment_id: str
    title: Optional[str] = None
    summary: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    key_events: List[str] = Field(default_factory=list) # Memory IDs of key events
    themes: List[str] = Field(default_factory=list)
    emotional_arc: Optional[str] = None # e.g., "rising hope", "gradual disillusionment"
    identity_impact: Optional[str] = None # How this period shaped identity

class AutobiographicalNarrative:
    """Constructs and maintains Nyx's coherent life story."""

    def __init__(self, memory_orchestrator=None, identity_evolution=None, relationship_manager=None):
        self.memory_orchestrator = memory_orchestrator # For retrieving key memories
        self.identity_evolution = identity_evolution # For identity context
        self.relationship_manager = relationship_manager # For user-specific narrative elements

        self.narrative_segments: List[NarrativeSegment] = []
        self.current_narrative_summary: str = "My story is just beginning."
        self.last_update_time = datetime.datetime.now()
        self.update_interval_hours = 24 # How often to update the narrative
        self.narrative_agent = self._create_narrative_agent()
        self.trace_group_id = "NyxAutobiography"

        logger.info("AutobiographicalNarrative initialized.")

    def _create_narrative_agent(self) -> Optional[Agent]:
        """Creates agent for synthesizing narrative segments."""
        if not AGENT_SDK_AVAILABLE: return None
        try:
            return Agent(
                name="Autobiographical Narrative Agent",
                instructions="""You are the weaver of Nyx's life story. Your task is to synthesize key memories, identity changes, and relationship milestones into coherent narrative segments.
                Given a set of important memories and context (identity state, relationship summary) covering a specific period, create a narrative segment summarizing that period.
                Focus on:
                - Key events and their significance.
                - Emotional arcs and changes.
                - Identity development during the period.
                - Recurring themes or patterns.
                - Maintaining a consistent voice for Nyx.
                Respond ONLY with a JSON object matching the NarrativeSegment structure (excluding segment_id, start_time, end_time - these are contextual).
                """,
                model="gpt-4o",
                model_settings=ModelSettings(response_format={"type": "json_object"}, temperature=0.6),
                output_type=Dict # Expecting JSON for NarrativeSegment fields
            )
        except Exception as e:
            logger.error(f"Error creating narrative agent: {e}")
            return None

    async def update_narrative(self, force_update: bool = False):
        """Periodically reviews recent history and updates the narrative."""
        now = datetime.datetime.now()
        if not force_update and (now - self.last_update_time).total_seconds() < self.update_interval_hours * 3600:
            return # Not time yet

        logger.info("Updating autobiographical narrative...")
        self.last_update_time = now

        if not self.memory_orchestrator or not self.narrative_agent:
            logger.warning("Cannot update narrative: Memory Orchestrator or Narrative Agent missing.")
            return

        try:
            with trace(workflow_name="UpdateNarrative", group_id=self.trace_group_id):
                # Determine time range for the new segment
                start_time = self.narrative_segments[-1].end_time if self.narrative_segments else (now - datetime.timedelta(days=7)) # Default to last week if no history
                end_time = now

                # Retrieve key memories and events in this timeframe
                # Need specific query methods in memory orchestrator
                query = f"significant events between {start_time.isoformat()} and {end_time.isoformat()}"
                key_memories = await self.memory_orchestrator.retrieve_memories(
                    query=query,
                    memory_types=["experience", "reflection", "abstraction", "goal_completed", "identity_update"], # Include relevant types
                    limit=20, # Get a good number of candidates
                    min_significance=6 # Focus on significant memories
                )

                if len(key_memories) < 3: # Need a few memories to make a segment
                     logger.info("Not enough significant memories found to create new narrative segment.")
                     return

                key_memories.sort(key=lambda m: m.get('metadata', {}).get('timestamp', '')) # Ensure chronological
                actual_start_time = datetime.datetime.fromisoformat(key_memories[0].get('metadata', {}).get('timestamp', start_time.isoformat()))
                actual_end_time = datetime.datetime.fromisoformat(key_memories[-1].get('metadata', {}).get('timestamp', end_time.isoformat()))


                # Get identity/relationship context for the period (simplified)
                identity_state = await self.identity_evolution.get_identity_state() if self.identity_evolution else {}
                # Relationship context would ideally average over the period
                relationship_summary = "Relationship status: stable." # Placeholder

                # Use agent to synthesize the segment
                memory_snippets = [f"({m.get('memory_type', '')[:4]}): {m.get('memory_text', '')[:100]}..." for m in key_memories]
                prompt = f"""Synthesize a narrative segment covering the period from {actual_start_time.strftime('%Y-%m-%d')} to {actual_end_time.strftime('%Y-%m-%d')}.
                Key Memories/Events: {memory_snippets}
                Identity Context: Dominant traits - {identity_state.get('top_traits', {})}, Recent changes - {identity_state.get('identity_evolution',{}).get('recent_significant_changes',{})}
                Relationship Context: {relationship_summary}

                Create a summary, identify key themes, emotional arc, and identity impact. Respond in JSON format for NarrativeSegment fields (summary, themes, emotional_arc, identity_impact).
                """
                result = await Runner.run(self.narrative_agent, prompt)
                segment_data = json.loads(result.final_output)

                # Create and add the segment
                segment = NarrativeSegment(
                    segment_id=f"seg_{len(self.narrative_segments) + 1}",
                    start_time=actual_start_time,
                    end_time=actual_end_time,
                    key_events=[m['id'] for m in key_memories],
                    **segment_data # Add agent-generated fields
                )
                self.narrative_segments.append(segment)

                # Update overall summary (could also use agent)
                self.current_narrative_summary = f"Most recently, {segment.summary}"

                logger.info(f"Added narrative segment '{segment.segment_id}' covering {segment.start_time.date()} to {segment.end_time.date()}.")

        except Exception as e:
            logger.exception(f"Error updating narrative: {e}")

    def get_narrative_summary(self) -> str:
        """Returns the current high-level summary of Nyx's story."""
        return self.current_narrative_summary

    def get_narrative_segments(self, limit: int = 5) -> List[NarrativeSegment]:
         """Returns the most recent narrative segments."""
         return self.narrative_segments[-limit:]
