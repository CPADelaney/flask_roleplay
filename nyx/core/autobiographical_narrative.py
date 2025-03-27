# nyx/core/autobiographical_narrative.py

import logging
import datetime
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

class NarrativeSegment(BaseModel):
    """A segment of the autobiographical narrative."""
    segment_id: str = Field(default_factory=lambda: f"seg_{uuid.uuid4().hex[:8]}")
    title: Optional[str] = None
    summary: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    key_events: List[str] = Field(default_factory=list)  # Memory IDs of key events
    themes: List[str] = Field(default_factory=list)
    emotional_arc: Optional[str] = None  # e.g., "rising hope", "gradual disillusionment"
    identity_impact: Optional[str] = None  # How this period shaped identity

class AutobiographicalNarrative:
    """Constructs and maintains Nyx's coherent life story."""

    def __init__(self, memory_orchestrator=None, identity_evolution=None, relationship_manager=None):
        self.memory_orchestrator = memory_orchestrator  # For retrieving key memories
        self.identity_evolution = identity_evolution  # For identity context
        self.relationship_manager = relationship_manager  # For user-specific narrative elements

        self.narrative_segments: List[NarrativeSegment] = []
        self.current_narrative_summary: str = "My story is just beginning."
        self.last_update_time = datetime.datetime.now()
        self.update_interval_hours = 24  # How often to update the narrative
        self.narrative_agent = self._create_narrative_agent()
        self.trace_group_id = "NyxAutobiography"

        logger.info("AutobiographicalNarrative initialized")

    def _create_narrative_agent(self) -> Optional[Agent]:
        """Creates agent for synthesizing narrative segments."""
        try:
            return Agent(
                name="Autobiographical Narrator",
                instructions="""You are the weaver of Nyx's life story. Your task is to synthesize key memories, identity changes, and relationship milestones into coherent narrative segments.
                Given a set of important memories and context (identity state, relationship summary) covering a specific period, create a narrative segment summarizing that period.
                Focus on:
                - Key events and their significance.
                - Emotional arcs and changes.
                - Identity development during the period.
                - Recurring themes or patterns.
                - Maintaining a consistent voice for Nyx.
                Respond ONLY with a JSON object containing: "title", "summary", "themes", "emotional_arc", and "identity_impact".
                """,
                model="gpt-4o",
                model_settings=ModelSettings(response_format={"type": "json_object"}, temperature=0.6),
                output_type=Dict[str, Any]
            )
        except Exception as e:
            logger.error(f"Error creating narrative agent: {e}")
            return None

    @function_tool
    async def retrieve_significant_memories(self, start_time: str, end_time: str, min_significance: int = 6) -> List[Dict]:
        """Retrieves significant memories within a time period."""
        if not self.memory_orchestrator:
            return []
        
        try:
            memories = await self.memory_orchestrator.retrieve_memories(
                query=f"significant events between {start_time} and {end_time}",
                memory_types=["experience", "reflection", "abstraction", "goal_completed", "identity_update"],
                limit=20,
                min_significance=min_significance
            )
            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    @function_tool
    async def get_identity_snapshot(self, timestamp: str) -> Dict[str, Any]:
        """Gets a snapshot of Nyx's identity at a specific time."""
        if not self.identity_evolution:
            return {"status": "unavailable", "reason": "No identity evolution system available"}
        
        try:
            state = await self.identity_evolution.get_identity_state(timestamp)
            return {
                "top_traits": state.get("top_traits", {}),
                "recent_changes": state.get("identity_evolution", {}).get("recent_significant_changes", {})
            }
        except Exception as e:
            logger.error(f"Error retrieving identity snapshot: {e}")
            return {"status": "error", "reason": str(e)}

    async def update_narrative(self, force_update: bool = False) -> Optional[NarrativeSegment]:
        """Periodically reviews recent history and updates the narrative."""
        now = datetime.datetime.now()
        if not force_update and (now - self.last_update_time).total_seconds() < self.update_interval_hours * 3600:
            return None  # Not time yet

        logger.info("Updating autobiographical narrative...")
        self.last_update_time = now

        if not self.memory_orchestrator or not self.narrative_agent:
            logger.warning("Cannot update narrative: Memory Orchestrator or Narrative Agent missing")
            return None

        try:
            with trace(workflow_name="UpdateNarrative", group_id=self.trace_group_id):
                # Determine time range for the new segment
                start_time = self.narrative_segments[-1].end_time if self.narrative_segments else (now - datetime.timedelta(days=7))
                end_time = now

                # Retrieve key memories 
                key_memories = await self.retrieve_significant_memories(
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat()
                )

                if len(key_memories) < 3:  # Need enough memories to make a segment
                    logger.info("Not enough significant memories found to create new narrative segment")
                    return None

                # Sort chronologically
                key_memories.sort(key=lambda m: m.get('metadata', {}).get('timestamp', ''))
                
                # Get actual timespan from memories
                actual_start_time = datetime.datetime.fromisoformat(key_memories[0].get('metadata', {}).get('timestamp', start_time.isoformat()))
                actual_end_time = datetime.datetime.fromisoformat(key_memories[-1].get('metadata', {}).get('timestamp', end_time.isoformat()))

                # Get identity context
                identity_state = await self.get_identity_snapshot(actual_end_time.isoformat())

                # Prepare memory snippets
                memory_snippets = [f"({m.get('memory_type', '')[:4]}): {m.get('memory_text', '')[:100]}..." for m in key_memories]

                # Construct prompt for narrative generation
                prompt = {
                    "time_period": {
                        "start": actual_start_time.strftime('%Y-%m-%d'),
                        "end": actual_end_time.strftime('%Y-%m-%d')
                    },
                    "key_memories": memory_snippets,
                    "identity_context": identity_state,
                    "existing_narrative_summary": self.current_narrative_summary
                }

                # Generate narrative segment
                result = await Runner.run(self.narrative_agent, json.dumps(prompt))
                segment_data = result.final_output

                # Create and add the segment
                segment = NarrativeSegment(
                    start_time=actual_start_time,
                    end_time=actual_end_time,
                    key_events=[m.get('id', f"mem_{i}") for i, m in enumerate(key_memories)],
                    summary=segment_data.get("summary", "A period of time passed"),
                    title=segment_data.get("title"),
                    themes=segment_data.get("themes", []),
                    emotional_arc=segment_data.get("emotional_arc"),
                    identity_impact=segment_data.get("identity_impact")
                )
                self.narrative_segments.append(segment)

                # Update overall summary
                self.current_narrative_summary = f"Most recently, {segment.summary}"

                logger.info(f"Added narrative segment '{segment.segment_id}' covering {segment.start_time.date()} to {segment.end_time.date()}")
                return segment

        except Exception as e:
            logger.exception(f"Error updating narrative: {e}")
            return None

    def get_narrative_summary(self) -> str:
        """Returns the current high-level summary of Nyx's story."""
        return self.current_narrative_summary

    def get_narrative_segments(self, limit: int = 5) -> List[NarrativeSegment]:
        """Returns the most recent narrative segments."""
        return self.narrative_segments[-limit:]

    def get_full_narrative(self) -> Dict[str, Any]:
        """Returns the complete narrative history in a structured format."""
        return {
            "summary": self.current_narrative_summary,
            "segments": [segment.model_dump() for segment in self.narrative_segments],
            "segment_count": len(self.narrative_segments),
            "time_span": {
                "start": self.narrative_segments[0].start_time.isoformat() if self.narrative_segments else None,
                "end": self.narrative_segments[-1].end_time.isoformat() if self.narrative_segments else None
            }
        }
