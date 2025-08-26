"""
Nyx Response Assembly Module - Refactored for ContextBundle Architecture
Assembles final responses from agent outputs with canon-first priority and on-demand expansion
"""
from __future__ import annotations  # optional but nice to have
from typing import Any, Dict, List, Optional

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict
from uuid import uuid4

from agents import Agent, Runner
from agents.models.openai_responses import OpenAIResponsesModel, ModelSettings
from agents.guardrail import JSONSchema

from nyx.nyx_agent.models import (
    NyxResponse,
    WorldState,
    MemoryHighlight,
    EmergentEvent,
    NPCDialogue,
    Choice,
    ContextBundle,
    SceneScope,
    BundleMetadata,
)
from nyx.nyx_agent.context import ContextBroker

logger = logging.getLogger(__name__)


@dataclass
class AssemblyConfig:
    """Configuration for response assembly"""
    max_narrative_tokens: int = 2000
    max_context_tokens: int = 8000
    enable_emergent_connections: bool = True
    enable_on_demand_expansion: bool = True
    enable_live_validation: bool = False  # Set True for full validation (adds latency)
    include_performance_metrics: bool = True
    llm_timeout_seconds: float = 5.0  # Timeout for validation calls


class NyxResponseAssembler:
    """
    Assembles rich, canon-consistent responses from agent outputs and context bundles.
    Supports on-demand context expansion and emergent narrative connections.
    """
    
    def __init__(
        self,
        context_broker: ContextBroker,
        config: Optional[AssemblyConfig] = None,
        llm_model: Optional[str] = None
    ):
        self.context_broker = context_broker
        self.config = config or AssemblyConfig()
        self._expansion_cache = OrderedDict()  # Cache for on-demand expansions with predictable FIFO
        
        # Only initialize validation agents if enabled (adds latency)
        if self.config.enable_live_validation:
            llm_model = llm_model or "gpt-4o-mini"
            
            # Canon validation agent for narrative consistency
            self._canon_validator = self._create_canon_validation_agent(llm_model)
            
            # Character consistency analyst for dialogue validation
            self._character_analyst = self._create_character_analysis_agent(llm_model)
            
            # Pattern detection agent for emergent connections
            self._pattern_detector = self._create_pattern_detection_agent(llm_model)
            
            # World state validator for consistency checks
            self._world_validator = self._create_world_validation_agent(llm_model)
        else:
            # Set to None when validation is disabled
            self._canon_validator = None
            self._character_analyst = None
            self._pattern_detector = None
            self._world_validator = None
    
    def _iter_active_conflicts(self, conflict_bundle: Optional[BundleMetadata]):
        """
        Adapter for conflict bundle shape variations.
        Handles both new (data.active: List) and legacy (data.active_conflicts: Dict) formats.
        """
        if not conflict_bundle:
            return
            
        data = conflict_bundle.data or {}
        
        # New synthesizer shape
        if isinstance(data.get("active"), list):
            for i, c in enumerate(data["active"]):
                # Use existing ID or stable fallback based on index
                cid = str(c.get("id", f"conflict_{i}"))
                yield cid, c
        # Back-compat
        elif isinstance(data.get("active_conflicts"), dict):
            for cid, c in data["active_conflicts"].items():
                yield str(cid), c
    
    def _create_canon_validation_agent(self, model_name: str):
        """Create agent for validating canon consistency."""
        system_prompt = """You are a canon validation specialist for the Nyx roleplay system.
        
Your role is to:
1. Check narratives for contradictions with established canon facts
2. Identify violations of world rules, character traits, or historical events
3. Suggest minimal corrections that preserve narrative flow while fixing violations
4. Rate severity of violations (minor/moderate/major)

When validating, consider:
- World lore and established history
- Character backstories and immutable traits
- Physical laws and magic system rules
- Political relationships and power structures
- Timeline consistency

Output format:
{
    "has_violations": boolean,
    "violations": [
        {
            "type": "character_inconsistency|world_lore|timeline|physics",
            "description": "specific violation",
            "severity": "minor|moderate|major",
            "narrative_segment": "the problematic text",
            "canon_fact": "the contradicted fact"
        }
    ],
    "corrected_narrative": "full corrected text if auto_correct=true and fixes possible",
    "suggestions": ["list of suggested changes if not auto-correcting"]
}"""
        
        return create_validation_agent(
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=0.3  # Low temperature for consistency
        )
    
    def _create_character_analysis_agent(self, model_name: str):
        """Create agent for analyzing character dialogue consistency."""
        system_prompt = """You are a character consistency analyst for the Nyx roleplay system.

Your role is to analyze dialogue and ensure it matches established character profiles.

Check for consistency in:
1. Vocabulary style (formal/casual/archaic/modern)
2. Personality traits (shy/bold, honest/deceptive, kind/cruel)
3. Knowledge boundaries (what the character should/shouldn't know)
4. Emotional patterns (how they express feelings)
5. Speech patterns (catchphrases, verbal tics, accents)
6. Relationship dynamics (how they address different people)

Output format:
{
    "overall_consistency_score": 0.0-1.0,
    "aspect_scores": {
        "vocabulary_style": 0.0-1.0,
        "personality_traits": 0.0-1.0,
        "knowledge_boundaries": 0.0-1.0,
        "emotional_patterns": 0.0-1.0,
        "speech_patterns": 0.0-1.0
    },
    "violations": [
        {
            "type": "aspect violated",
            "severity": "minor|moderate|major",
            "detail": "specific issue"
        }
    ],
    "character_voice_preserved": boolean,
    "suggestions": ["improvements if needed"]
}"""
        
        return create_analysis_agent(
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=0.3
        )
    
    def _create_pattern_detection_agent(self, model_name: str):
        """Create agent for detecting emergent patterns and connections."""
        system_prompt = """You are a pattern detection specialist for the Nyx roleplay system.

Your role is to identify emergent themes, connections, and patterns across different narrative elements.

Look for:
1. Recurring themes across memories and events
2. Hidden connections between characters and conflicts
3. Foreshadowing and narrative threads coming together
4. Cause-and-effect chains across subsystems
5. Symbolic or thematic resonances
6. Character arc convergences

Evaluate emergence potential (0.0-1.0) based on:
- Narrative impact if revealed now
- Number of connected elements
- Thematic significance
- Player engagement potential

Output format:
{
    "patterns": [
        {
            "type": "theme|connection|foreshadowing|convergence",
            "description": "what pattern was found",
            "emergence_potential": 0.0-1.0,
            "entities": ["involved entities"],
            "narrative_impact": "how this could affect the story",
            "suggested_reveal": "how to naturally introduce this"
        }
    ],
    "cross_system_links": [
        {
            "source_system": "memory|npc|conflict|lore",
            "target_system": "memory|npc|conflict|lore",
            "connection": "description of link",
            "strength": 0.0-1.0
        }
    ]
}"""
        
        # Use PATTERN_JSON_SCHEMA for structured output
        runner = _make_runner(system_prompt, model_name, 0.7, PATTERN_JSON_SCHEMA)
        
        class PatternAgent:
            def __init__(self, runner: Runner):
                self.runner = runner
            
            async def analyze_patterns(self, **kwargs):
                run = await self.runner.run(input=f"Detect patterns in: {json.dumps(kwargs)}")
                payload = (getattr(run, "output_parsed", None)
                          or getattr(run, "output_text", None)
                          or getattr(run, "output", ""))
                return payload if isinstance(payload, dict) else _safe_json(payload, {})
            
            async def evaluate_connection(self, **kwargs):
                run = await self.runner.run(input=f"Evaluate connection strength: {json.dumps(kwargs)}")
                payload = (getattr(run, "output_parsed", None)
                          or getattr(run, "output_text", None)
                          or getattr(run, "output", ""))
                return payload if isinstance(payload, dict) else _safe_json(payload, {})
        
        return PatternAgent(runner)
    
    def _create_world_validation_agent(self, model_name: str):
        """Create agent for validating world state consistency."""
        system_prompt = """You are a world state validation specialist for the Nyx roleplay system.

Your role is to ensure world state updates don't violate established rules.

Validate:
1. Geographic consistency (can't be in two places at once)
2. Temporal consistency (time flows forward, events in sequence)
3. Resource consistency (conservation of matter/energy/currency)
4. Political consistency (power structures, allegiances)
5. Environmental consistency (weather patterns, seasons)
6. Causal consistency (effects follow causes)

Output format:
{
    "is_valid": boolean,
    "consistency_score": 0.0-1.0,
    "violations": [
        {
            "type": "geographic|temporal|resource|political|environmental|causal",
            "description": "specific issue",
            "severity": "minor|moderate|major"
        }
    ],
    "warnings": ["potential issues that aren't violations"],
    "ripple_effects": ["consequences of these changes"]
}"""
        
        # Define world validation schema
        world_validation_schema = {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "consistency_score": {"type": "number"},
                "violations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "severity": {"type": "string"}
                        }
                    }
                },
                "warnings": {"type": "array", "items": {"type": "string"}},
                "ripple_effects": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["is_valid", "consistency_score"]
        }
        
        runner = _make_runner(system_prompt, model_name, 0.2, world_validation_schema)
        
        class WorldValidator:
            def __init__(self, runner: Runner):
                self.runner = runner
            
            async def validate_choice(self, **kwargs):
                run = await self.runner.run(input=f"Validate choice: {json.dumps(kwargs)}")
                payload = (getattr(run, "output_parsed", None)
                          or getattr(run, "output_text", None)
                          or getattr(run, "output", ""))
                return payload if isinstance(payload, dict) else _safe_json(payload, {})
            
            async def validate_updates(self, **kwargs):
                run = await self.runner.run(input=f"Validate world updates: {json.dumps(kwargs)}")
                payload = (getattr(run, "output_parsed", None)
                          or getattr(run, "output_text", None)
                          or getattr(run, "output", ""))
                return payload if isinstance(payload, dict) else _safe_json(payload, {})
            
            async def analyze_ripple_effects(self, **kwargs):
                run = await self.runner.run(input=f"Analyze ripple effects: {json.dumps(kwargs)}")
                payload = (getattr(run, "output_parsed", None)
                          or getattr(run, "output_text", None)
                          or getattr(run, "output", ""))
                result = payload if isinstance(payload, dict) else _safe_json(payload, {})
                # Ensure is_significant field for ripple analysis
                if "is_significant" not in result:
                    result["is_significant"] = result.get("consistency_score", 0) > 0.5
                return result
        
        return WorldValidator(runner)
        
    async def assemble_nyx_response(
        self,
        agent_output: Dict[str, Any],
        context_bundle: ContextBundle,
        scene_scope: SceneScope,
        conversation_id: str,
        user_input: str,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> NyxResponse:
        """
        Main assembly function - transforms agent output and context into final response.
        
        Args:
            agent_output: Raw output from the Nyx agent
            context_bundle: Pre-assembled context bundle with canon data
            scene_scope: Current scene scope for focused context
            conversation_id: Current conversation identifier
            user_input: Original user input for reference
            processing_metadata: Performance and debug metadata
            
        Returns:
            Fully assembled NyxResponse with narrative, world state, and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # 1. Extract and validate core narrative
            narrative_content = await self._extract_narrative(
                agent_output, 
                context_bundle,
                scene_scope
            )
            
            # 2. Build world state with canon-first priority
            world_state = await self._build_world_state(
                context_bundle.world_bundle,
                context_bundle.narrator_bundle,
                agent_output.get("world_updates", {})
            )
            
            # 3. Process NPC dialogue and interactions
            npc_dialogues = await self._process_npc_dialogues(
                agent_output.get("npc_interactions", []),
                context_bundle.npc_bundle,
                scene_scope
            )
            
            # 4. Extract memory highlights and emergent connections
            memory_highlights = await self._extract_memory_highlights(
                context_bundle.memory_bundle,
                agent_output.get("memory_references", []),
                scene_scope
            )
            
            # 5. Identify emergent events and connections
            emergent_events = []
            if self.config.enable_emergent_connections:
                emergent_events = await self._identify_emergent_events(
                    context_bundle,
                    agent_output,
                    scene_scope
                )
            
            # 6. Generate player choices
            choices = await self._generate_choices(
                agent_output.get("choices", []),
                context_bundle.conflict_bundle,
                world_state,
                context_bundle
            )
            
            # 7. Apply on-demand expansions if requested
            expansion_results = {}
            if self.config.enable_on_demand_expansion:
                expansion_results = await self._apply_expansions(
                    agent_output.get("expansion_requests", []),
                    context_bundle,
                    scene_scope
                )
                narrative_content = self._integrate_expansions(
                    narrative_content,
                    expansion_results
                )
            
            # 8. Calculate performance metrics
            assembly_time = (datetime.utcnow() - start_time).total_seconds()
            performance_metrics = self._build_performance_metrics(
                assembly_time,
                processing_metadata,
                context_bundle
            )
            
            # 9. Construct final response
            # Safely serialize scene_scope
            try:
                scope_meta = scene_scope.to_dict()
            except Exception:
                scope_meta = {k: v for k, v in vars(scene_scope).items() if not k.startswith("_")}
            
            response = NyxResponse(
                id=str(uuid4()),
                conversation_id=conversation_id,
                narrative=narrative_content,
                world_state=world_state,
                npc_dialogues=npc_dialogues,
                memory_highlights=memory_highlights,
                emergent_events=emergent_events,
                choices=choices,
                metadata={
                    "scene_scope": scope_meta,
                    "context_stats": self._get_context_stats(context_bundle),
                    "performance": performance_metrics,
                    "timestamp": datetime.utcnow().isoformat(),
                    "canon_adherence_score": await self._calculate_canon_adherence(
                        agent_output,
                        context_bundle
                    ),
                    "expansions": expansion_results if self.config.enable_on_demand_expansion else {},
                }
            )
            
            # 10. Log and validate
            await self._validate_response(response, context_bundle)
            logger.info(
                f"Assembled response in {assembly_time:.2f}s "
                f"(narrative: {len(narrative_content)} chars, "
                f"choices: {len(choices)}, "
                f"emergent: {len(emergent_events)})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error assembling response: {str(e)}", exc_info=True)
            return self._create_fallback_response(
                conversation_id,
                user_input,
                error=str(e)
            )
    
    async def _extract_narrative(
        self,
        agent_output: Dict[str, Any],
        context_bundle: ContextBundle,
        scene_scope: SceneScope
    ) -> str:
        """
        Extract and enhance narrative from agent output.
        Prioritizes narrator tool output, falls back to direct narrative.
        """
        # Check for narrator tool output first (highest quality)
        narrator_tool_names = {"narrate_slice_of_life_scene", "tool_narrate_slice_of_life_scene", "narrate_scene"}
        
        if "tool_outputs" in agent_output:
            for tool_output in agent_output["tool_outputs"]:
                if tool_output.get("tool") in narrator_tool_names:
                    narrative = tool_output.get("narrative", "")
                    if narrative:
                        return await self._enhance_narrative_with_canon(
                            narrative,
                            context_bundle.narrator_bundle
                        )
        
        # Fall back to direct narrative field
        narrative = agent_output.get("narrative", "")
        if not narrative:
            narrative = agent_output.get("response", "")
        
        # Ensure canon elements are reflected
        if narrative and context_bundle.narrator_bundle and (context_bundle.narrator_bundle.canon or {}):
            narrative = await self._ensure_canon_in_narrative(
                narrative,
                context_bundle.narrator_bundle.canon or {}
            )
        
        return narrative or "..."
    
    async def _enhance_narrative_with_canon(
        self,
        narrative: str,
        narrator_bundle: Optional[BundleMetadata]
    ) -> str:
        """
        Enhance narrative with canonical narrator elements.
        """
        if not narrator_bundle or not narrator_bundle.canon:
            return narrative
        
        # Validate against canon
        validated = await self._ensure_canon_in_narrative(
            narrative,
            narrator_bundle.canon
        )
        
        # Could add atmospheric elements from canon if needed
        # For now, return the validated narrative
        return validated
    
    async def _build_world_state(
        self,
        world_bundle: Optional[BundleMetadata],
        narrator_bundle: Optional[BundleMetadata],
        updates: Dict[str, Any]
    ) -> WorldState:
        """
        Build world state with canon-first priority.
        Canon world facts override any dynamic updates.
        """
        # Start with canonical world facts
        world_state = WorldState()
        
        # Apply canon first (immutable)
        if world_bundle and world_bundle.canon:
            world_state.update_from_canon(world_bundle.canon)
        
        # Layer dynamic world data
        if world_bundle:
            world_data = world_bundle.data or {}
            if world_data:
                world_state.time_of_day = world_data.get("time_of_day", "unknown")
                world_state.weather = world_data.get("weather", {})
                world_state.location = world_data.get("current_location", {})
                world_state.mood = world_data.get("mood", "neutral")
        
        # Apply narrator atmosphere
        if narrator_bundle:
            narrator_data = narrator_bundle.data or {}
            if narrator_data:
                world_state.atmosphere = narrator_data.get("atmosphere", {})
                world_state.tension_level = narrator_data.get("tension", 0.5)
        
        # Finally apply any agent updates (lowest priority)
        if updates:
            world_state.apply_updates(updates, validate_against_canon=True)
        
        return world_state
    
    async def _process_npc_dialogues(
        self,
        interactions: List[Dict[str, Any]],
        npc_bundle: Optional[BundleMetadata],
        scene_scope: SceneScope
    ) -> List[NPCDialogue]:
        """
        Process NPC dialogues ensuring canonical character consistency.
        """
        dialogues = []
        
        if not npc_bundle:
            return dialogues
        
        # Parallelize dialogue checks if validation is enabled
        if self.config.enable_live_validation and self._character_analyst:
            # Collect all interactions that need validation
            validation_tasks = []
            interaction_data = []
            
            for interaction in interactions:
                npc_id = interaction.get("npc_id")
                if not npc_id or npc_id not in scene_scope.npc_ids:
                    continue
                
                # Normalize key to string
                key = str(npc_id)
                npc_canon = (npc_bundle.canon or {}).get(key, {})
                
                validation_tasks.append(
                    self._check_dialogue_canon_adherence(
                        interaction.get("dialogue", ""),
                        npc_canon
                    )
                )
                interaction_data.append((interaction, npc_id, key, npc_canon))
            
            # Run all validations in parallel (with exception handling)
            if validation_tasks:
                adherence_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                # Convert exceptions to default scores
                adherence_scores = []
                for result in adherence_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Dialogue validation failed: {result}")
                        adherence_scores.append(0.9)
                    else:
                        adherence_scores.append(result)
            else:
                adherence_scores = []
            
            # Build dialogues with validation results
            for i, (interaction, npc_id, key, npc_canon) in enumerate(interaction_data):
                npc_dynamic_map = npc_bundle.data or {}
                npc_dynamic = npc_dynamic_map.get(key) or npc_dynamic_map.get(npc_id, {})
                adherence_score = adherence_scores[i] if i < len(adherence_scores) else 0.9
                
                dialogue = NPCDialogue(
                    npc_id=npc_id,
                    npc_name=npc_canon.get("name", f"NPC_{npc_id}"),
                    text=interaction.get("dialogue", ""),
                    emotion=interaction.get("emotion", "neutral"),
                    intent=interaction.get("intent", npc_dynamic.get("current_intent", "unknown")),
                    canonical_traits=npc_canon.get("traits", []),
                    relationship_to_player=npc_canon.get("relationship_to_player", "neutral"),
                    metadata={
                        "canon_adherence": adherence_score,
                        "scene_relevance": self._calculate_scene_relevance(
                            interaction,
                            scene_scope
                        )
                    }
                )
                dialogues.append(dialogue)
        else:
            # Fast path without validation
            for interaction in interactions:
                npc_id = interaction.get("npc_id")
                if not npc_id or npc_id not in scene_scope.npc_ids:
                    continue
                
                # Normalize key to string
                key = str(npc_id)
                npc_canon = (npc_bundle.canon or {}).get(key, {})
                npc_dynamic_map = npc_bundle.data or {}
                npc_dynamic = npc_dynamic_map.get(key) or npc_dynamic_map.get(npc_id, {})
                
                dialogue = NPCDialogue(
                    npc_id=npc_id,
                    npc_name=npc_canon.get("name", f"NPC_{npc_id}"),
                    text=interaction.get("dialogue", ""),
                    emotion=interaction.get("emotion", "neutral"),
                    intent=interaction.get("intent", npc_dynamic.get("current_intent", "unknown")),
                    canonical_traits=npc_canon.get("traits", []),
                    relationship_to_player=npc_canon.get("relationship_to_player", "neutral"),
                    metadata={
                        "canon_adherence": 0.9,  # High default score when not validating
                        "scene_relevance": self._calculate_scene_relevance(
                            interaction,
                            scene_scope
                        )
                    }
                )
                dialogues.append(dialogue)
        
        return dialogues
    
    async def _extract_memory_highlights(
        self,
        memory_bundle: Optional[BundleMetadata],
        referenced_memories: List[str],
        scene_scope: SceneScope
    ) -> List[MemoryHighlight]:
        """
        Extract relevant memories with emphasis on canonical events.
        """
        highlights = []
        
        # Always include canonical memories first
        if memory_bundle and memory_bundle.canon:
            for memory_id, memory_data in memory_bundle.canon.items():
                if self._is_memory_relevant(memory_data, scene_scope):
                    highlights.append(MemoryHighlight(
                        id=memory_id,
                        content=memory_data.get("content", ""),
                        relevance_score=1.0,  # Canon memories get max relevance
                        is_canonical=True,
                        timestamp=memory_data.get("timestamp"),
                        entities_involved=memory_data.get("entities", [])
                    ))
        
        # Add agent-referenced memories
        memory_data_map = memory_bundle.data or {} if memory_bundle else {}
        for ref in referenced_memories:
            if ref not in [h.id for h in highlights]:  # Avoid duplicates
                memory_data = memory_data_map.get(ref, {})
                if memory_data:
                    highlights.append(MemoryHighlight(
                        id=ref,
                        content=memory_data.get("content", ""),
                        relevance_score=memory_data.get("relevance_score", 0.5),
                        is_canonical=False,
                        timestamp=memory_data.get("timestamp"),
                        entities_involved=memory_data.get("entities", [])
                    ))
        
        # Sort by relevance but keep canon at top
        highlights.sort(key=lambda h: (h.is_canonical, h.relevance_score), reverse=True)
        
        return highlights[:10]  # Limit to top 10 to avoid bloat
    
    async def _identify_emergent_events(
        self,
        context_bundle: ContextBundle,
        agent_output: Dict[str, Any],
        scene_scope: SceneScope
    ) -> List[EmergentEvent]:
        """
        Identify emergent narrative connections across subsystems.
        This is where the magic happens - finding unexpected links.
        """
        events = []
        
        # Check for memory-lore connections
        if context_bundle.linked_concepts:
            for link in context_bundle.linked_concepts:
                if link["type"] == "memory_lore_connection":
                    events.append(EmergentEvent(
                        type="revelation",
                        description=f"Connection discovered: {link['description']}",
                        impact_score=link.get("impact", 0.7),
                        entities_involved=link.get("entities", []),
                        triggers_future_event=link.get("triggers_future", None)
                    ))
        
        # Check for NPC-conflict emergent behaviors
        npc_conflicts = self._find_npc_conflict_emergence(
            context_bundle.npc_bundle,
            context_bundle.conflict_bundle
        )
        for emergence in npc_conflicts:
            events.append(EmergentEvent(
                type="character_development",
                description=emergence["description"],
                impact_score=emergence["impact"],
                entities_involved=emergence["npcs"],
                triggers_future_event=emergence.get("consequence")
            ))
        
        # Check for world-event ripples
        if "world_ripples" in agent_output:
            for ripple in agent_output["world_ripples"]:
                events.append(EmergentEvent(
                    type="world_consequence",
                    description=ripple["description"],
                    impact_score=ripple.get("magnitude", 0.5),
                    entities_involved=ripple.get("affected_entities", []),
                    triggers_future_event=ripple.get("future_impact")
                ))
        
        # Pattern detection from memory analysis
        mem_data = (context_bundle.memory_bundle.data or {}) if context_bundle.memory_bundle else {}
        for pattern in mem_data.get("patterns", []):
            if pattern.get("emergence_potential", 0) > 0.7:
                events.append(EmergentEvent(
                    type="pattern_emergence",
                    description=f"Pattern emerging: {pattern['description']}",
                    impact_score=pattern["emergence_potential"],
                    entities_involved=pattern.get("entities", []),
                    triggers_future_event=pattern.get("predicted_outcome"),
                ))
        
        return events
    
    async def _generate_choices(
        self,
        agent_choices: List[Dict[str, Any]],
        conflict_bundle: Optional[BundleMetadata],
        world_state: WorldState,
        context_bundle: ContextBundle
    ) -> List[Choice]:
        """
        Generate player choices with awareness of conflicts and world state.
        """
        choices = []
        
        # Add agent-generated choices
        for choice_data in agent_choices:
            choice = Choice(
                id=str(uuid4()),
                text=choice_data.get("text", ""),
                category=choice_data.get("category", "action"),
                requirements=choice_data.get("requirements", {}),
                consequences=choice_data.get("consequences", {}),
                canon_alignment=await self._calculate_canon_alignment(choice_data, context_bundle)
            )
            choices.append(choice)
        
        # Add conflict-driven choices using adapter for shape variations
        if conflict_bundle:
            for cid, c in self._iter_active_conflicts(conflict_bundle):
                if c.get("requires_player_choice"):
                    options = c.get("options", [])
                    
                    # If no options provided, synthesize generic resolution choices
                    if not options:
                        options = [
                            {"id": "deescalate", "text": "Attempt to de-escalate the situation"},
                            {"id": "escalate", "text": "Escalate the conflict"},
                            {"id": "negotiate", "text": "Try to negotiate a resolution"}
                        ]
                    
                    for option in options:
                        choice = Choice(
                            id=f"conflict_{cid}_{option.get('id', uuid4())}",
                            text=option.get("text", "Resolve conflict"),
                            category="conflict_resolution",
                            requirements=option.get("requirements", {}),
                            consequences=option.get("consequences", {}),
                            canon_alignment=1.0  # Conflict choices are canon-driven
                        )
                        choices.append(choice)
        
        # Ensure at least one viable choice
        if not choices:
            choices.append(self._create_default_choice(world_state))
        
        return choices
    
    def _expansion_result_key(self, expansion_type: str, request: Dict[str, Any]) -> str:
        """Get consistent key for expansion result regardless of cache status."""
        if expansion_type == "npc_detail":
            return f"npc_{request.get('npc_id')}"
        if expansion_type == "memory_depth":
            return "additional_memories"
        if expansion_type == "lore_context":
            return "lore_expansion"
        if expansion_type == "conflict_details":
            return "conflict_details"
        return expansion_type  # fallback
    
    async def _apply_expansions(
        self,
        expansion_requests: List[Dict[str, Any]],
        context_bundle: ContextBundle,
        scene_scope: SceneScope
    ) -> Dict[str, Any]:
        """
        Handle on-demand context expansions requested by the agent.
        Uses caching to avoid redundant expansions within the same scene.
        """
        expansions = {}
        expansions_meta = {}
        
        # Generate cache key based on scene scope
        scene_key = f"{scene_scope.location_id}_{hash(frozenset(scene_scope.npc_ids))}"
        
        for request in expansion_requests:
            expansion_type = request.get("type")
            
            # Create cache key for this specific request
            cache_key = f"{scene_key}_{expansion_type}_{json.dumps(request, sort_keys=True, default=str)}"
            result_key = self._expansion_result_key(expansion_type, request)
            
            # Check cache first
            if cache_key in self._expansion_cache:
                self._expansion_cache.move_to_end(cache_key)  # Move to end for LRU
                expansions[result_key] = self._expansion_cache[cache_key]
                expansions_meta[result_key] = {"cached": True}
                continue
            
            # Perform expansion
            result = None
            
            if expansion_type == "npc_detail":
                result = await self.context_broker.expand_npc(
                    request.get("npc_id"),
                    request.get("fields", []),
                    scene_scope
                )
            
            elif expansion_type == "memory_depth":
                result = await self.context_broker.expand_memories(
                    entity_ids=request.get("entity_ids", []),
                    k=request.get("k", 5),
                    scene_scope=scene_scope
                )
            
            elif expansion_type == "lore_context":
                entities = request.get("entities", [])
                depth = request.get("depth", 1)
                # Try with include_history first, fall back if not supported
                try:
                    result = await self.context_broker.expand_lore(
                        entities,
                        depth,
                        True,  # include_history
                        scene_scope
                    )
                except TypeError:
                    result = await self.context_broker.expand_lore(
                        entities,
                        depth,
                        scene_scope
                    )
            
            elif expansion_type == "conflict_details":
                conflict_ids = request.get("conflict_ids", [])
                # Try with include_stakes first, fall back if not supported
                try:
                    result = await self.context_broker.expand_conflicts(
                        conflict_ids,
                        True,  # include_stakes
                        scene_scope
                    )
                except TypeError:
                    result = await self.context_broker.expand_conflicts(
                        conflict_ids,
                        scene_scope
                    )
            
            # Store result with consistent key
            if result is not None:
                self._expansion_cache[cache_key] = result
                expansions[result_key] = result
                expansions_meta[result_key] = {"cached": False}
                
                # Limit cache size to prevent memory bloat
                while len(self._expansion_cache) > 100:
                    self._expansion_cache.popitem(last=False)  # Remove oldest (FIFO)
        
        # Store meta in expansions for optional use
        if expansions_meta:
            expansions["_meta"] = expansions_meta
        
        return expansions
    
    def _integrate_expansions(
        self,
        narrative: str,
        expansions: Dict[str, Any]
    ) -> str:
        """
        Integrate expansion results into the narrative if needed.
        """
        # Check for significant lore expansions that could clarify plot twists
        lore = expansions.get("lore_expansion") or {}
        canonical_facts = lore.get("canonical_facts") or lore.get("canon") or {}
        
        if canonical_facts and any(
            (v or {}).get("plot_critical") for v in canonical_facts.values()
        ):
            # Add a brief atmospheric hint rather than breaking narrative flow
            narrative += "\n\n*[The shadows seem to whisper of deeper truths...]*"
        
        # Expansions are also available in metadata for UI rendering
        return narrative
    
    def _find_npc_conflict_emergence(
        self,
        npc_bundle: Optional[BundleMetadata],
        conflict_bundle: Optional[BundleMetadata]
    ) -> List[Dict[str, Any]]:
        """
        Find emergent behaviors from NPC-conflict interactions.
        """
        emergences: List[Dict[str, Any]] = []
        npc_data_map = npc_bundle.data or {} if npc_bundle else {}
        
        if not npc_data_map or not conflict_bundle:
            return emergences
        
        # Normalize active conflicts via adapter
        active = dict(self._iter_active_conflicts(conflict_bundle))
        
        for raw_npc_id, npc_data in npc_data_map.items():
            # Normalize id both as int and str for membership checks
            npc_id_str = str(raw_npc_id)
            npc_id_int = None
            try:
                npc_id_int = int(raw_npc_id)
            except Exception:
                pass
            
            npc_conflicts = []
            for c in active.values():
                parts = set((c.get("participants") or []) + (c.get("stakeholders") or []))
                if npc_id_str in parts or (npc_id_int is not None and npc_id_int in parts):
                    npc_conflicts.append(c)
            
            if len(npc_conflicts) > 1:
                emergences.append({
                    "description": f"{npc_data.get('name', 'Someone')} torn between conflicting loyalties",
                    "impact": 0.8,
                    "npcs": [raw_npc_id],
                    "consequence": "loyalty_shift",
                })
            
            # Pacifist tension
            npc_canon_map = npc_bundle.canon or {} if npc_bundle else {}
            npc_canon = npc_canon_map.get(npc_id_str, {})
            if npc_canon.get("pacifist") and any((c.get("type") == "combat") for c in npc_conflicts):
                emergences.append({
                    "description": f"{npc_data.get('name', 'Someone')} forced to question their pacifist beliefs",
                    "impact": 0.9,
                    "npcs": [raw_npc_id],
                    "consequence": "character_growth",
                })
        
        return emergences
    
    async def _ensure_canon_in_narrative(
        self, 
        narrative: str,
        canon_facts: Dict[str, Any]
    ) -> str:
        """
        Ensure canonical facts are respected in narrative using validation agent.
        """
        # Skip validation if disabled or validator not available
        if not self.config.enable_live_validation or not self._canon_validator:
            return narrative
        
        try:
            # Use the canon validation agent with timeout
            validation_result = await asyncio.wait_for(
                self._canon_validator.validate_narrative(
                    narrative=narrative,
                    canon_facts=canon_facts,
                    auto_correct=True  # Allow minor corrections
                ),
                timeout=self.config.llm_timeout_seconds
            )
            
            if validation_result["has_violations"]:
                logger.warning(
                    f"Canon violations detected: {validation_result['violations']}"
                )
                
                if validation_result.get("corrected_narrative"):
                    # Return the corrected version if available
                    return validation_result["corrected_narrative"]
                
                # If no correction available, at least log the issues
                for violation in validation_result["violations"]:
                    logger.error(f"Canon violation: {violation['description']}")
        
        except asyncio.TimeoutError:
            logger.warning("Canon validation timed out, using original narrative")
        except Exception as e:
            logger.error(f"Canon validation error: {e}")
        
        return narrative
    
    async def _check_dialogue_canon_adherence(
        self,
        dialogue: str,
        npc_canon: Dict[str, Any]
    ) -> float:
        """
        Score how well dialogue adheres to character canon using analysis agent.
        """
        # If validation disabled, return neutral score
        if not self.config.enable_live_validation or not self._character_analyst:
            return 0.9  # High default when not validating
        
        try:
            # Use the character consistency agent with timeout
            consistency_result = await asyncio.wait_for(
                self._character_analyst.analyze_dialogue(
                    dialogue=dialogue,
                    character_profile=npc_canon,
                    check_aspects=[
                        "vocabulary_style",
                        "personality_traits", 
                        "knowledge_boundaries",
                        "emotional_patterns",
                        "speech_patterns"
                    ]
                ),
                timeout=self.config.llm_timeout_seconds
            )
            
            score = consistency_result.get("overall_consistency_score", 1.0)
            
            # Apply specific penalties based on violations
            violations = consistency_result.get("violations", [])
            for violation in violations:
                severity = violation.get("severity", "minor")
                if severity == "major":
                    score *= 0.5
                elif severity == "moderate":
                    score *= 0.7
                else:  # minor
                    score *= 0.9
            
            # Log significant deviations for monitoring
            if score < 0.7:
                logger.warning(
                    f"Low canon adherence ({score:.2f}) for NPC {npc_canon.get('name', 'Unknown')}: "
                    f"{', '.join(v['type'] for v in violations)}"
                )
            
            return max(score, 0.0)  # Ensure non-negative
            
        except asyncio.TimeoutError:
            logger.warning("Dialogue validation timed out")
            return 0.9
        except Exception as e:
            logger.error(f"Dialogue validation error: {e}")
            return 0.9
    
    def _calculate_scene_relevance(
        self,
        interaction: Dict[str, Any],
        scene_scope: SceneScope
    ) -> float:
        """
        Calculate how relevant an interaction is to the current scene.
        """
        score = 0.5  # Base relevance
        
        # Check NPC relevance
        if interaction.get("npc_id") in scene_scope.npc_ids:
            score += 0.3
        
        # Check topic relevance
        interaction_topics = set(interaction.get("topics", []))
        if interaction_topics & scene_scope.topics:
            score += 0.2
        
        return min(score, 1.0)
    
    def _is_memory_relevant(
        self,
        memory_data: Dict[str, Any],
        scene_scope: SceneScope
    ) -> bool:
        """
        Check if a memory is relevant to the current scene.
        """
        # Check entity overlap (avoid None in set)
        memory_entities = set(memory_data.get("entities", []))
        scene_entities = set(scene_scope.npc_ids)
        if scene_scope.location_id is not None:
            scene_entities.add(scene_scope.location_id)
        
        if memory_entities & scene_entities:
            return True
        
        # Check topic overlap
        memory_topics = set(memory_data.get("topics", []))
        if memory_topics & scene_scope.topics:
            return True
        
        # Check lore tag overlap
        memory_lore = set(memory_data.get("lore_tags", []))
        if memory_lore & scene_scope.lore_tags:
            return True
        
        return False
    
    async def _calculate_canon_alignment(
        self,
        choice_data: Dict[str, Any],
        context_bundle: Optional[ContextBundle] = None
    ) -> float:
        """
        Calculate how well a choice aligns with world canon using validation agent.
        """
        if choice_data.get("canon_approved"):
            return 1.0
        
        # Skip validation if disabled or validator not available  
        if not self.config.enable_live_validation or not self._world_validator:
            # Simple heuristic when validation disabled
            if choice_data.get("violates_canon"):
                return 0.0
            return 0.9  # High default when not validating
            
        # Use world validator to check the choice
        validation_input = {
            "proposed_action": choice_data.get("text", ""),
            "requirements": choice_data.get("requirements", {}),
            "consequences": choice_data.get("consequences", {}),
            "world_canon": context_bundle.world_bundle.canon if context_bundle and context_bundle.world_bundle else {},
            "active_rules": (context_bundle.lore_bundle.canon or {}).get("rules", {}) if context_bundle and context_bundle.lore_bundle else {}
        }
        
        try:
            validation_result = await asyncio.wait_for(
                self._world_validator.validate_choice(
                    choice_data=validation_input
                ),
                timeout=self.config.llm_timeout_seconds
            )
            
            # Check for explicit violations
            if validation_result.get("violations"):
                major_violations = [v for v in validation_result["violations"] 
                                   if v.get("severity") == "major"]
                if major_violations:
                    choice_data["violates_canon"] = True
                    return 0.0
                
                # Minor violations reduce score
                return max(0.3, 1.0 - (0.2 * len(validation_result["violations"])))
            
            # Return consistency score if no violations
            return validation_result.get("consistency_score", 0.5)
            
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Choice validation failed: {e}")
            return 0.9  # High default on error
    
    async def _calculate_canon_adherence(
        self,
        agent_output: Dict[str, Any],
        context_bundle: ContextBundle
    ) -> float:
        """
        Overall canon adherence score for the response using validation agents.
        """
        # Skip if validation disabled
        if not self.config.enable_live_validation:
            return 0.95  # High default score when not validating
        
        parts = {}
        
        # Check narrative adherence with canon validator
        if agent_output.get("narrative") and self._canon_validator:
            try:
                narrative_validation = await asyncio.wait_for(
                    self._canon_validator.validate_narrative(
                        narrative=agent_output["narrative"],
                        canon_facts={
                            "world": context_bundle.world_bundle.canon or {} if context_bundle.world_bundle else {},
                            "lore": context_bundle.lore_bundle.canon or {} if context_bundle.lore_bundle else {},
                            "npcs": context_bundle.npc_bundle.canon or {} if context_bundle.npc_bundle else {}
                        },
                        auto_correct=False
                    ),
                    timeout=self.config.llm_timeout_seconds
                )
                
                if not narrative_validation.get("has_violations"):
                    parts["narrative"] = 1.0
                else:
                    v = narrative_validation.get("violations", [])
                    score = 1.0 - (0.3 * sum(x.get("severity") == "major" for x in v)
                                   + 0.15 * sum(x.get("severity") == "moderate" for x in v)
                                   + 0.05 * sum(x.get("severity") == "minor" for x in v))
                    parts["narrative"] = max(0.0, score)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Narrative validation failed: {e}")
                parts["narrative"] = 0.9
        
        # Check NPC dialogue adherence with character analyst  
        if agent_output.get("npc_interactions") and self._character_analyst and context_bundle.npc_bundle:
            ds = []
            for inter in agent_output["npc_interactions"]:
                nid = str(inter.get("npc_id"))
                profile = (context_bundle.npc_bundle.canon or {}).get(nid)
                if profile:
                    try:
                        score = await self._check_dialogue_canon_adherence(
                            inter.get("dialogue", ""), 
                            profile
                        )
                        ds.append(score)
                    except Exception:
                        ds.append(0.9)
            if ds:
                parts["dialogue"] = sum(ds) / len(ds)
        
        # Check world state adherence with world validator
        if agent_output.get("world_updates") and self._world_validator and context_bundle.world_bundle:
            try:
                world_validation = await asyncio.wait_for(
                    self._world_validator.validate_updates(
                        updates=agent_output["world_updates"],
                        current_state=context_bundle.world_bundle.data or {},
                        canon_rules=context_bundle.world_bundle.canon or {}
                    ),
                    timeout=self.config.llm_timeout_seconds
                )
                parts["world"] = world_validation.get("consistency_score", 0.9)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"World validation failed: {e}")
                parts["world"] = 0.9
        
        # Calculate weighted average
        if not parts:
            return 1.0
        
        weights = {"narrative": 0.4, "dialogue": 0.3, "world": 0.3}
        weighted_sum = sum(parts[k] * weights[k] for k in parts)
        total_weight = sum(weights[k] for k in parts)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.95
    
    def _get_context_stats(self, context_bundle: ContextBundle) -> Dict[str, Any]:
        """
        Get statistics about the context bundle for debugging.
        """
        active_count = 0
        if context_bundle.conflict_bundle and context_bundle.conflict_bundle.data:
            active_count = sum(1 for _ in self._iter_active_conflicts(context_bundle.conflict_bundle))
        
        return {
            "total_npcs": len(context_bundle.npc_bundle.data or {}) if context_bundle.npc_bundle else 0,
            "canon_npcs": len(context_bundle.npc_bundle.canon or {}) if context_bundle.npc_bundle else 0,
            "memories": len(context_bundle.memory_bundle.data or {}) if context_bundle.memory_bundle else 0,
            "active_conflicts": active_count,
            "linked_concepts": len(context_bundle.linked_concepts or []),
            "lore_entities": len(context_bundle.lore_bundle.data or {}) if context_bundle.lore_bundle else 0,
        }
    
    def _build_performance_metrics(
        self,
        assembly_time: float,
        processing_metadata: Optional[Dict[str, Any]],
        context_bundle: ContextBundle
    ) -> Dict[str, Any]:
        """
        Build performance metrics for monitoring.
        """
        metrics = {
            "assembly_time_seconds": assembly_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if processing_metadata:
            metrics.update({
                "context_fetch_time": processing_metadata.get("context_fetch_time", 0),
                "agent_processing_time": processing_metadata.get("agent_time", 0),
                "total_time": processing_metadata.get("total_time", 0),
            })
        
        # Add bundle timing if available
        for bundle_name in ["npc", "lore", "memory", "conflict", "world", "narrator"]:
            bundle = getattr(context_bundle, f"{bundle_name}_bundle", None)
            if bundle and hasattr(bundle, "fetch_time"):
                metrics[f"{bundle_name}_fetch_ms"] = bundle.fetch_time
        
        return metrics
    
    def _create_default_choice(self, world_state: WorldState) -> Choice:
        """
        Create a default choice when none are available.
        """
        return Choice(
            id="default_continue",
            text="Continue...",
            category="continuation",
            requirements={},
            consequences={},
            canon_alignment=1.0
        )
    
    async def _validate_response(
        self,
        response: NyxResponse,
        context_bundle: ContextBundle
    ) -> None:
        """
        Validate the response for consistency and completeness.
        """
        # Check narrative exists
        if not response.narrative or len(response.narrative) < 10:
            logger.warning("Response has minimal or no narrative")
        
        # Check canon adherence
        if response.metadata.get("canon_adherence_score", 1.0) < 0.5:
            logger.warning(
                f"Low canon adherence score: {response.metadata['canon_adherence_score']}"
            )
        
        # Check for required elements based on scene
        if context_bundle.npc_bundle and context_bundle.npc_bundle.data and not response.npc_dialogues:
            logger.info("NPCs present but no dialogue generated")
        
        # Validate choices
        if not response.choices:
            logger.warning("No choices provided in response")
    
    def _create_fallback_response(
        self,
        conversation_id: str,
        user_input: str,
        error: str
    ) -> NyxResponse:
        """
        Create a fallback response when assembly fails.
        """
        return NyxResponse(
            id=str(uuid4()),
            conversation_id=conversation_id,
            narrative="[The shadows ripple with uncertainty...]",
            world_state=WorldState(),
            npc_dialogues=[],
            memory_highlights=[],
            emergent_events=[],
            choices=[
                Choice(
                    id="retry",
                    text="Try again",
                    category="system",
                    requirements={},
                    consequences={},
                    canon_alignment=1.0
                )
            ],
            metadata={
                "error": error,
                "fallback": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        )


# On-demand expansion tools for the agent
class ExpansionTools:
    """
    Tools that the agent can call to expand context on-demand.
    These are registered with the agent and called during generation.
    """
    
    def __init__(self, context_broker: ContextBroker, assembler: Optional['NyxResponseAssembler'] = None):
        self.context_broker = context_broker
        self._expansion_history = []
        self._assembler = assembler  # Reference to assembler for pattern detection
    
    async def expand_npc_details(
        self, 
        npc_id: int,
        fields: Optional[List[str]] = None,
        scene_scope: Optional[SceneScope] = None
    ) -> Dict[str, Any]:
        """
        Expand details about a specific NPC.
        
        Args:
            npc_id: ID of the NPC to expand
            fields: Specific fields to fetch (backstory, goals, secrets, etc.)
            scene_scope: Current scene context
            
        Returns:
            Expanded NPC information
        """
        result = await self.context_broker.expand_npc(
            npc_id,
            fields or ["full_backstory", "current_goals", "secrets", "relationships"],
            scene_scope
        )
        
        self._expansion_history.append({
            "type": "npc_detail",
            "npc_id": npc_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "npc_id": npc_id,
            "expanded_data": result,
            "canon_priority": True  # Signal this is canon data
        }
    
    async def get_additional_memories(
        self,
        entity_ids: List[int],
        k: int = 5,
        memory_types: Optional[List[str]] = None,
        scene_scope: Optional[SceneScope] = None
    ) -> Dict[str, Any]:
        """
        Retrieve additional memories related to specific entities.
        
        Args:
            entity_ids: IDs of entities to find memories about
            k: Number of memories to retrieve
            memory_types: Types of memories to prioritize
            scene_scope: Current scene context
            
        Returns:
            Additional memory context
        """
        # Pass along memory_types if broker supports it
        kwargs = {
            "entity_ids": entity_ids,
            "k": k,
            "scene_scope": scene_scope
        }
        if memory_types:
            kwargs["memory_types"] = memory_types
            
        try:
            memories = await self.context_broker.expand_memories(**kwargs)
        except TypeError:
            # Fallback if broker doesn't support memory_types
            memories = await self.context_broker.expand_memories(
                entity_ids=entity_ids,
                k=k,
                scene_scope=scene_scope
            )
        
        self._expansion_history.append({
            "type": "memory_depth",
            "entities": entity_ids,
            "count": len(memories) if isinstance(memories, list) else 0,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "memories": memories,
            "entities": entity_ids,
            "patterns": await self._detect_patterns(memories) if isinstance(memories, list) else []
        }
    
    async def expand_lore_context(
        self,
        entities: List[str],
        depth: int = 1,
        include_history: bool = True,
        scene_scope: Optional[SceneScope] = None
    ) -> Dict[str, Any]:
        """
        Expand lore and world context for specific entities.
        
        Args:
            entities: Lore entities to expand (locations, factions, concepts)
            depth: How many relationship hops to include
            include_history: Whether to include historical context
            scene_scope: Current scene context
            
        Returns:
            Expanded lore information
        """
        lore_data = await self.context_broker.expand_lore(
            entities,
            depth,
            include_history,
            scene_scope
        )
        
        self._expansion_history.append({
            "type": "lore_context",
            "entities": entities,
            "depth": depth,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "lore": lore_data,
            "entities": entities,
            "canonical_facts": lore_data.get("canon", {}),
            "relationships": lore_data.get("relationships", {})
        }
    
    async def check_world_state(
        self,
        aspects: List[str] = None,
        scene_scope: Optional[SceneScope] = None
    ) -> Dict[str, Any]:
        """
        Check specific aspects of the world state.
        
        Args:
            aspects: Specific aspects to check (time, weather, mood, etc.)
            scene_scope: Current scene context
            
        Returns:
            Current world state information
        """
        aspects = aspects or ["time", "weather", "mood", "location", "tension"]
        
        world_state = await self.context_broker.get_world_state(aspects, scene_scope)
        
        return {
            "world_state": world_state,
            "canonical_rules": world_state.get("canon", {}),
            "dynamic_state": world_state.get("current", {})
        }
    
    async def get_conflict_details(
        self,
        conflict_ids: Optional[List[str]] = None,
        include_stakes: bool = True,
        scene_scope: Optional[SceneScope] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about active conflicts.
        
        Args:
            conflict_ids: Specific conflicts to detail (None for all active)
            include_stakes: Whether to include stakes and consequences
            scene_scope: Current scene context
            
        Returns:
            Detailed conflict information
        """
        conflicts = await self.context_broker.expand_conflicts(
            conflict_ids,
            include_stakes,
            scene_scope
        )
        
        return {
            "conflicts": conflicts,
            "active_count": len(conflicts),
            "player_involved": any(c.get("player_involved") for c in conflicts.values()),
            "highest_tension": max(
                (c.get("tension", 0) for c in conflicts.values()),
                default=0
            )
        }
    
    async def _detect_patterns(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use pattern detection agent to find meaningful patterns across memories.
        """
        if not memories:
            return []
        
        # Use the assembler's pattern detector if available
        asm = getattr(self, "_assembler", None)
        detector = getattr(asm, "_pattern_detector", None) if asm else None
        if detector:
            # Get timeout from assembler config if available
            timeout = getattr(asm, "config", AssemblyConfig()).llm_timeout_seconds
            try:
                pattern_result = await asyncio.wait_for(
                    detector.analyze_patterns(
                        context_data={
                            "memories": {str(i): m for i, m in enumerate(memories)},
                            "memory_count": len(memories),
                            "time_span": self._calculate_time_span(memories),
                            "entities": self._extract_all_entities(memories)
                        }
                    ),
                    timeout=timeout
                )
                
                patterns = []
                for p in pattern_result.get("patterns", []):
                    patterns.append({
                        "type": p["type"],
                        "description": p.get("description", ""),
                        "significance": p.get("emergence_potential", 0.5),
                        "entities": p.get("entities", []),
                        "suggested_narrative": p.get("suggested_reveal", "")
                    })
                
                return patterns
            except asyncio.TimeoutError:
                logger.warning("Pattern detection timed out, using fallback")
            except Exception as e:
                logger.warning(f"Pattern detection failed: {e}, using fallback")
        
        # Fallback to simple pattern detection
        patterns = []
        
        # Check for recurring entities
        entity_counts = {}
        for memory in memories:
            for entity in memory.get("entities", []):
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        for entity, count in entity_counts.items():
            if count >= 3:
                patterns.append({
                    "type": "recurring_entity",
                    "entity": entity,
                    "frequency": count,
                    "significance": min(count / len(memories), 1.0)
                })
        
        # Check for emotional patterns
        emotions = [m.get("emotion") for m in memories if m.get("emotion")]
        if emotions:
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            if emotion_counts[dominant_emotion] >= len(emotions) * 0.4:
                patterns.append({
                    "type": "emotional_pattern",
                    "emotion": dominant_emotion,
                    "frequency": emotion_counts[dominant_emotion],
                    "significance": emotion_counts[dominant_emotion] / len(emotions)
                })
        
        # Check for temporal clustering
        if all(m.get("timestamp") for m in memories):
            timestamps = [m["timestamp"] for m in memories]
            # Simple clustering - check if most memories are from a specific period
            if self._check_temporal_clustering(timestamps):
                patterns.append({
                    "type": "temporal_cluster",
                    "description": "Memories concentrated in specific time period",
                    "significance": 0.7
                })
        
        return patterns
    
    def _calculate_time_span(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate the time span covered by memories."""
        timestamps = [m.get("timestamp") for m in memories if m.get("timestamp")]
        if not timestamps:
            return {"span": "unknown"}
        
        sorted_ts = sorted(timestamps)
        return {
            "earliest": sorted_ts[0],
            "latest": sorted_ts[-1],
            "span": "variable"  # Could calculate actual duration if timestamp format known
        }
    
    def _extract_all_entities(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Extract all unique entities from memories."""
        entities = set()
        for memory in memories:
            entities.update(memory.get("entities", []))
        return list(entities)
    
    def _check_temporal_clustering(self, timestamps: List[Any]) -> bool:
        """Check if timestamps show clustering pattern."""
        # Simplified check - in production would use proper clustering algorithm
        if len(timestamps) < 3:
            return False
        
        # This is a placeholder - implement actual temporal clustering logic
        return len(set(timestamps)) < len(timestamps) * 0.7


# Agent creation helpers using OpenAI Agents SDK

# JSON schemas for structured outputs
VALIDATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "has_violations": {"type": "boolean"},
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string"},
                    "narrative_segment": {"type": "string"},
                    "canon_fact": {"type": "string"},
                },
                "required": ["type", "description", "severity"]
            }
        },
        "corrected_narrative": {"type": "string"},
        "suggestions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["has_violations"]
}

ANALYSIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_consistency_score": {"type": "number"},
        "aspect_scores": {"type": "object"},
        "violations": {"type": "array"},
        "character_voice_preserved": {"type": "boolean"},
        "suggestions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["overall_consistency_score"]
}

PATTERN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "emergence_potential": {"type": "number"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "narrative_impact": {"type": "string"},
                    "suggested_reveal": {"type": "string"}
                }
            }
        },
        "cross_system_links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_system": {"type": "string"},
                    "target_system": {"type": "string"},
                    "connection": {"type": "string"},
                    "strength": {"type": "number"}
                }
            }
        }
    }
}

def _safe_json(text: str, default=None) -> Dict[str, Any]:
    """Safely parse JSON from LLM response with fallback."""
    try:
        return json.loads(text)
    except Exception:
        # Best-effort extraction of first {...} block
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return default if default is not None else {}


def _make_runner(system_prompt: str, model_name: str, temperature: float, json_schema: Optional[Dict] = None) -> Runner:
    """Common helper to build an Agents SDK runner on OpenAI Responses."""
    model = OpenAIResponsesModel(
        model=model_name,
        settings=ModelSettings(
            temperature=temperature,
            max_output_tokens=4096,
            # Force JSON response format if no schema provided
            extra_body={"response_format": {"type": "json_object"}} if not json_schema else {}
        ),
    )
    agent = Agent(
        name="nyx_aux_validator",
        instructions=system_prompt,
        model=model,
        # Strong JSON guarantee via schema
        output_schema=JSONSchema(json_schema) if json_schema else None
    )
    return Runner(agent)


def create_validation_agent(
    model_name: str,
    system_prompt: str,
    temperature: float = 0.3
):
    """Create a validation agent with the specified configuration."""
    runner = _make_runner(system_prompt, model_name, temperature, VALIDATION_JSON_SCHEMA)
    
    class ValidationAgent:
        def __init__(self, runner: Runner):
            self.runner = runner
        
        async def validate_narrative(self, **kwargs):
            run = await self.runner.run(input=json.dumps(kwargs))
            payload = (getattr(run, "output_parsed", None)
                      or getattr(run, "output_text", None)
                      or getattr(run, "output", ""))
            return payload if isinstance(payload, dict) else _safe_json(payload, {})
        
        async def validate_choice(self, **kwargs):
            run = await self.runner.run(input=f"Validate choice: {json.dumps(kwargs)}")
            payload = (getattr(run, "output_parsed", None)
                      or getattr(run, "output_text", None)
                      or getattr(run, "output", ""))
            return payload if isinstance(payload, dict) else _safe_json(payload, {})
        
        async def validate_updates(self, **kwargs):
            run = await self.runner.run(input=f"Validate world updates: {json.dumps(kwargs)}")
            payload = (getattr(run, "output_parsed", None)
                      or getattr(run, "output_text", None)
                      or getattr(run, "output", ""))
            return payload if isinstance(payload, dict) else _safe_json(payload, {})
        
        async def analyze_ripple_effects(self, **kwargs):
            run = await self.runner.run(input=f"Analyze ripple effects: {json.dumps(kwargs)}")
            payload = (getattr(run, "output_parsed", None)
                      or getattr(run, "output_text", None)
                      or getattr(run, "output", ""))
            return payload if isinstance(payload, dict) else _safe_json(payload, {})
    
    return ValidationAgent(runner)


def create_analysis_agent(
    model_name: str,
    system_prompt: str,
    temperature: float = 0.5
):
    """Create an analysis agent with the specified configuration."""
    runner = _make_runner(system_prompt, model_name, temperature, ANALYSIS_JSON_SCHEMA)
    
    class AnalysisAgent:
        def __init__(self, runner: Runner):
            self.runner = runner
        
        async def analyze_dialogue(self, **kwargs):
            run = await self.runner.run(input=f"Analyze dialogue consistency: {json.dumps(kwargs)}")
            payload = (getattr(run, "output_parsed", None)
                      or getattr(run, "output_text", None)
                      or getattr(run, "output", ""))
            return payload if isinstance(payload, dict) else _safe_json(payload, {})
        
        async def analyze_conflict_impact(self, **kwargs):
            run = await self.runner.run(input=f"Analyze conflict impact on character: {json.dumps(kwargs)}")
            payload = (getattr(run, "output_parsed", None)
                      or getattr(run, "output_text", None)
                      or getattr(run, "output", ""))
            return payload if isinstance(payload, dict) else _safe_json(payload, {})
        
        async def analyze_patterns(self, **kwargs):
            run = await self.runner.run(input=f"Detect patterns in: {json.dumps(kwargs)}")
            payload = (getattr(run, "output_parsed", None)
                      or getattr(run, "output_text", None)
                      or getattr(run, "output", ""))
            return payload if isinstance(payload, dict) else _safe_json(payload, {})
        
        async def evaluate_connection(self, **kwargs):
            run = await self.runner.run(input=f"Evaluate connection strength: {json.dumps(kwargs)}")
            payload = (getattr(run, "output_parsed", None)
                      or getattr(run, "output_text", None)
                      or getattr(run, "output", ""))
            return payload if isinstance(payload, dict) else _safe_json(payload, {})
    
    return AnalysisAgent(runner)


# Convenience factory function
def create_assembler(
    context_broker: ContextBroker,
    config: Optional[AssemblyConfig] = None,
    llm_model: Optional[str] = None
) -> NyxResponseAssembler:
    """
    Create a configured response assembler.
    
    Args:
        context_broker: The context broker for data access
        config: Optional configuration overrides
        llm_model: Optional LLM model name
        
    Returns:
        Configured NyxResponseAssembler
    """
    return NyxResponseAssembler(context_broker, config, llm_model)


# Agent tool registration helpers
def register_expansion_tools(
    agent, 
    context_broker: ContextBroker,
    assembler: Optional[NyxResponseAssembler] = None
) -> None:
    """
    Register expansion tools with the agent for on-demand use.
    
    Args:
        agent: The Nyx agent instance
        context_broker: The context broker for expansions
        assembler: Optional reference to the response assembler
    """
    tools = ExpansionTools(context_broker, assembler)
    
    agent.register_tool("expand_npc", tools.expand_npc_details)
    agent.register_tool("get_more_memories", tools.get_additional_memories)
    agent.register_tool("expand_lore", tools.expand_lore_context)
    agent.register_tool("check_world_state", tools.check_world_state)
    agent.register_tool("get_conflict_details", tools.get_conflict_details)
