# npcs/new_npc_creation.py
"""
Unified NPC creation functionality with Dynamic Relationships integration.
"""
from __future__ import annotations
import logging
import json
import asyncio
import random
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import os
import asyncpg
from datetime import datetime
from agents.models.openai_responses import OpenAIResponsesModel

from agents import Agent, Runner, function_tool, GuardrailFunctionOutput, InputGuardrail, RunContextWrapper, input_guardrail, output_guardrail, ModelSettings
from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem
from memory.core import Memory, MemoryType, MemorySignificance
from memory.managers import NPCMemoryManager
from memory.emotional import EmotionalMemoryManager
from memory.schemas import MemorySchemaManager
from textwrap import dedent
from memory.flashbacks import FlashbackManager
from memory.semantic import SemanticMemoryManager
from memory.masks import ProgressiveRevealManager, RevealType, RevealSeverity
from memory.reconsolidation import ReconsolidationManager

from logic.chatgpt_integration import get_openai_client, get_chatgpt_response, get_async_openai_client
from logic.gpt_utils import spaced_gpt_call
from logic.gpt_helpers import fetch_npc_name
from logic.calendar import load_calendar_names
from memory.memory_nyx_integration import remember_through_nyx
from openai import AsyncOpenAI
from pydantic import ValidationError

# Import the new dynamic relationships system
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    process_relationship_interaction_tool,
    get_relationship_summary_tool,
    update_relationship_context_tool
)

from npcs.dynamic_templates import (
    get_mask_slippage_triggers,
    get_relationship_stages,
    generate_core_beliefs,
    get_semantic_seed_topics,
    get_calendar_day_names,
    get_trauma_keywords,
)

logger = logging.getLogger(__name__)

# Configuration
DB_DSN = os.getenv("DB_DSN")

# Allow env overrides per-call
_DEFAULT_NAME_MODEL = os.getenv("OPENAI_NAME_MODEL", "gpt-4.1-nano")
_DEFAULT_DESC_MODEL = os.getenv("OPENAI_DESC_MODEL", "gpt-4.1-nano")
_DEFAULT_SCHED_MODEL = os.getenv("OPENAI_SCHED_MODEL", "gpt-4.1-nano")
_DEFAULT_PERS_MODEL = os.getenv("OPENAI_PERS_MODEL", "gpt-4.1-nano")
_DEFAULT_STATS_MODEL = os.getenv("OPENAI_STATS_MODEL", "gpt-4.1-nano")
_DEFAULT_ARCH_MODEL  = os.getenv("OPENAI_ARCH_MODEL", "gpt-4.1-nano")


async def _responses_json_call(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_output_tokens: int | None = None,
    response_format: dict | None = None,          # ➟ keep, default=None
    previous_response_id: str | None = None,      # ➟ new, optional thread-continue
) -> str:
    """
    Wrapper for the new `client.responses.create()` endpoint.
    Returns the best-effort STRING representation of whatever the
    model produced: JSON (stringified), plain text, or arguments for a tool call.
    """
    client = get_async_openai_client()

    params = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if response_format:
        params["response_format"] = response_format            # e.g. {"type":"json_object"}
    if previous_response_id:
        params["previous_response_id"] = previous_response_id  # thread continuity

    try:
        resp = await client.responses.create(**params)

        # ── 1️⃣  preferred: look at the generic output list ──────────────────
        if getattr(resp, "output", None):            # SDK ≥1.14
            for piece in resp.output:                # keep first usable
                if piece.type == "output_json":
                    # piece.json is already a Python dict
                    return json.dumps(piece.json, ensure_ascii=False)
                if piece.type == "output_text" and piece.text.strip():
                    return piece.text.strip()

        # ── 2️⃣  legacy shims kept by SDK for a while ────────────────────────
        if getattr(resp, "output_json", None):
            return json.dumps(resp.output_json, ensure_ascii=False)
        if getattr(resp, "output_text", None):
            txt = resp.output_text.strip()
            if txt:
                return txt

        # ── 3️⃣  tool / function calls ───────────────────────────────────────
        for accessor in ("tool_calls", "function_call"):        # SDK exposes both
            tc = getattr(resp, accessor, None)
            if not tc:
                continue
            if isinstance(tc, list):                            # tool_calls
                for call in tc:
                    if getattr(call, "function", None):
                        args = call.function.arguments
                        if args:                                # args already dict
                            return json.dumps(args, ensure_ascii=False)
            else:                                               # single function_call
                if getattr(tc, "arguments", None):
                    return json.dumps(tc.arguments, ensure_ascii=False)

        # ── 4️⃣  truly empty / error case ────────────────────────────────────
        raise ValueError("Empty model response.")

    except Exception as e:
        logging.error(f"_responses_json_call failed (model={model}): {e}", exc_info=True)
        raise

def _json_first_obj(text: str) -> dict | None:
    """
    Attempt to parse the *first* JSON object found in `text`.
    Uses same heuristics as safe_json_loads but smaller & fast.
    """
    if not text:
        return None
    # Direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # find {...}
    import re
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # code block
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # single->double quote brute
    try:
        return json.loads(text.replace("'", '"'))
    except Exception:
        return None

def _safe_list(value, *, fallback=None):
    if isinstance(value, list):
        return value
    return fallback if fallback is not None else []

def _strip_or(value, default=""):
    if isinstance(value, str):
        return value.strip()
    return default

# --- Per-output coercers ------------------------------------------------------

def _coerce_name(raw_txt: str, *, forbidden: list[str], existing: list[str]) -> str:
    """
    Extract a *single name string* from model output.
    The model is instructed to return just a name, but we defend anyway.
    """
    # quick cut: 1st line w/ letters
    line = raw_txt.splitlines()[0].strip()
    # Remove wrapping quotes/backticks if present
    if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
        line = line[1:-1].strip()
    # Extremely defensive: if JSON present with key 'name'
    data = _json_first_obj(raw_txt)
    if data:
        cand = data.get("name") or data.get("npc_name")
        if isinstance(cand, str) and cand.strip():
            line = cand.strip()

    # Fallback if punctuation heavy: grab 2 tokens
    tokens = [t for t in re.split(r'[\s,]+', line) if t]
    if len(tokens) >= 2:
        cand_name = f"{tokens[0].title()} {tokens[1].title()}"
    else:
        cand_name = tokens[0].title() if tokens else "Unnamed NPC"

    # uniqueness enforcement reusing external helper pattern
    unwanted = {"seraphina"}
    lower_forbidden = {n.lower() for n in forbidden} | {n.lower() for n in existing} | unwanted
    if cand_name.lower() in lower_forbidden:
        suffix = 2
        base = cand_name
        while f"{base} {suffix}".lower() in lower_forbidden:
            suffix += 1
        cand_name = f"{base} {suffix}"
    return cand_name

async def create_preset_npc(self, ctx, npc_data: dict, environment_context: str) -> int:
    """Create an NPC from preset story data"""
    user_id = ctx.context["user_id"]
    conversation_id = ctx.context["conversation_id"]
    
    # Build the NPC creation prompt
    prompt = f"""
    Create an NPC for this environment: {environment_context}
    
    Required NPC specifications:
    Name: {npc_data['name']}
    Role: {npc_data['role']}
    Archetype: {npc_data['archetype']}
    Traits: {', '.join(npc_data['traits'])}
    
    Stats requirements:
    {json.dumps(npc_data.get('stats', {}), indent=2)}
    
    Generate a compelling NPC that fits these specifications while adding rich details like:
    - Physical appearance
    - Backstory that explains their role
    - Personality quirks
    - Speaking style
    - Personal interests/hobbies
    - Schedule: {json.dumps(npc_data.get('schedule', {}), indent=2)}
    """
    
    # Use your existing NPC creation logic
    result = await self.create_single_npc(
        ctx=ctx,
        environment_desc=environment_context,
        custom_prompt=prompt,
        override_stats=npc_data.get('stats', {})
    )
    
    return result["npc_id"]

def _coerce_personality(data: dict) -> "NPCPersonalityData":
    return NPCPersonalityData(
        personality_traits=_safe_list(data.get("personality_traits"), fallback=[]),
        likes=_safe_list(data.get("likes"), fallback=[]),
        dislikes=_safe_list(data.get("dislikes"), fallback=[]),
        hobbies=_safe_list(data.get("hobbies"), fallback=[]),
    )

def _coerce_stats(data: dict) -> "NPCStatsData":
    def _iv(v, default):  # int value
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default
    return NPCStatsData(
        dominance=_iv(data.get("dominance"), 0),
        cruelty=_iv(data.get("cruelty"), 0),
        closeness=_iv(data.get("closeness"), 0),
        trust=_iv(data.get("trust"), 0),
        respect=_iv(data.get("respect"), 0),
        intensity=_iv(data.get("intensity"), 0),
    )

def _coerce_archetype(data: dict, *, provided_names: list[str] | None = None) -> "NPCArchetypeData":
    names = data.get("archetype_names") or provided_names or []
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    return NPCArchetypeData(
        archetype_names=_safe_list(names, fallback=[]),
        archetype_summary=_strip_or(data.get("archetype_summary"), ""),
        archetype_extras_summary=_strip_or(data.get("archetype_extras_summary"), ""),
    )

def _coerce_schedule(data: dict, *, day_names: list[str]) -> dict:
    # expect {"schedule": {...}} else data = direct schedule
    sched = data.get("schedule", data)
    if not isinstance(sched, dict):
        return {}
    # ensure required keys
    out = {}
    for day in day_names:
        day_block = sched.get(day, {})
        if not isinstance(day_block, dict):
            day_block = {}
        out[day] = {
            "Morning": _strip_or(day_block.get("Morning"), f"{day}: Morning routine."),
            "Afternoon": _strip_or(day_block.get("Afternoon"), f"{day}: Afternoon tasks."),
            "Evening": _strip_or(day_block.get("Evening"), f"{day}: Evening engagements."),
            "Night": _strip_or(day_block.get("Night"), f"{day}: Retires for the night."),
        }
    return out

def _coerce_physical_desc(data: dict, *, npc_name: str) -> str:
    desc = data.get("physical_description") or data.get("description") or ""
    desc = _strip_or(desc, "")
    if len(desc) < 40:
        desc = f"{npc_name} has an appearance that reflects her role in the setting."
    return desc

async def _await_logged(label: str, awaitable):
    """
    Await `awaitable` with detailed debug logging.

    Logs:
      • label starting
      • object type, repr, id()
      • whether object is a coroutine, Task, or Future
      • completion or exception (including stack)
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[npc_init] %s: awaiting %s (type=%s, id=%s, iscoroutine=%s, isfuture=%s, istask=%s)",
            label,
            awaitable,
            type(awaitable).__name__,
            id(awaitable),
            asyncio.iscoroutine(awaitable),
            asyncio.isfuture(awaitable),
            isinstance(awaitable, asyncio.Task),
        )
    try:
        result = await awaitable
    except Exception as e:  # catch to add context, then re-raise
        if logger.isEnabledFor(logging.ERROR):
            logger.error(
                "[npc_init] %s: await FAILED: %s\n%s",
                label,
                e,
                "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            )
        raise
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[npc_init] %s: completed ok (%r)", label, result)
        return result

# Models for input/output
class NPCCreationContext(BaseModel):
    user_id: int
    conversation_id: int
    db_dsn: str = DB_DSN

class NPCPersonalityData(BaseModel):
    personality_traits: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)

class NPCStatsData(BaseModel):
    dominance: int = 50
    cruelty: int = 30
    closeness: int = 50
    trust: int = 0
    respect: int = 0
    intensity: int = 40

class NPCArchetypeData(BaseModel):
    archetype_names: List[str] = Field(default_factory=list)
    archetype_summary: str = ""
    archetype_extras_summary: str = ""

class NPCCreationInput(BaseModel):
    npc_name: str
    sex: str = "female"
    archetype_names: List[str] = Field(default_factory=list)
    physical_description: str = ""
    personality_traits: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    introduced: bool = False
    dominance: int = 50
    cruelty: int = 50
    closeness: int = 50
    trust: int = 50
    respect: int = 50
    intensity: int = 50

class NPCCreationResult(BaseModel):
    npc_id: int
    npc_name: str
    physical_description: str
    personality: NPCPersonalityData
    stats: NPCStatsData
    archetypes: NPCArchetypeData
    schedule: Dict[str, Any] = Field(default_factory=dict)
    memories: List[str] = Field(default_factory=list)
    current_location: str = ""

class EnvironmentGuardrailOutput(BaseModel):
    is_valid: bool = True
    reasoning: str = ""

class NPCMemories(BaseModel):
    memories: List[str]


def build_memory_system_prompt() -> str:
    """
    System-level *contract* for the memory generator.
    Brief by design; we put creative richness in the user message.
    """
    return dedent("""
        You are an RPG memory-generation model.
        Return BETWEEN 3 AND 5 memories, first-person, each a concrete past event.
        Output MUST be valid JSON:
            {"memories": ["memory1", "memory2", ...]}
        - Each memory string may contain multiple sentences.
        - No markdown, no backticks, no additional keys.
        - Escape internal quotes properly for JSON.
    """).strip()


def build_memory_user_prompt(
    npc_name: str,
    environment_desc: str,
    archetype: str,
    dominance: int,
    cruelty: int,
    personality_traits: List[str],
    relationships: List[dict],
) -> str:
    """
    Rich creative brief (your original content, lightly edited for token sanity).
    """
    traits_str = ", ".join(personality_traits) if personality_traits else "complex personality"
    rel_lines = []
    for rel in relationships[:12]:  # cap for context
        lbl = rel.get("relationship_label", "relation")
        ety = rel.get("entity_type", "?")
        eid = rel.get("entity_id", "?")
        rel_lines.append(f"- {lbl} → {ety}:{eid}")
    rel_block = "\n".join(rel_lines) if rel_lines else "None documented."

    return dedent(f"""
        Create 3-5 vivid, detailed *memories* for **{npc_name}**.

        ## Environment Context
        {environment_desc}

        ## NPC Snapshot
        - Name: {npc_name}
        - Archetype: {archetype}
        - Dominance: {dominance}/100
        - Cruelty: {cruelty}/100
        - Personality: {traits_str}

        ## Known Relationships
        {rel_block}

        ## MEMORY REQUIREMENTS
        1. Each memory = a SPECIFIC EVENT (time/place/action), not a vague mood.
        2. Rich sensory detail (sight, sound, smell, touch, body feel).
        3. Include **quoted dialogue** at least once across the set; more is fine.
        4. First-person voice from {npc_name}'s perspective ("I...").
        5. 3–5 sentences per memory (short paragraphs ok).
        6. Subtle control / influence / boundary-testing beats.
        7. Emotional truth: show how I felt, noticed power shifts, or chose to conceal something.
        8. Vary tone & stakes (casual, tense, formative, bittersweet, revealing).

        ## OUTPUT CONTRACT
        Return ONLY valid JSON with key "memories" containing an array of strings.
        Do not include markdown fences or commentary.
    """).strip()


class NPCCreationHandler:
    """
    Unified handler for NPC creation and management.
    Combines functionality from NPCCreationAgent and logic/npc_creation.py.
    """
    
    def __init__(self):
        # Initialize input validation guardrail
        self._memory_agent = None
        @input_guardrail
        async def environment_guardrail(ctx, agent, input_str):
            """Validate that the environment description is appropriate for NPC creation"""
            try:
                # Check if the input has minimum required information
                if len(input_str) < 50:
                    return GuardrailFunctionOutput(
                        output_info=EnvironmentGuardrailOutput(
                            is_valid=False, 
                            reasoning="Environment description is too short for effective NPC creation"
                        )
                # Check for required elements
                required_elements = ["setting", "environment", "world", "location"]
                if not any(element in input_str.lower() for element in required_elements):
                    return GuardrailFunctionOutput(
                        output_info=EnvironmentGuardrailOutput(
                            is_valid=False, 
                            reasoning="Environment description lacks essential setting information"
                        ),
                        tripwire_triggered=True
                    )
                
                return GuardrailFunctionOutput(
                    output_info=EnvironmentGuardrailOutput(
                        is_valid=True, 
                        reasoning="Environment description is valid"
                    ),
                    tripwire_triggered=False
                )
            except Exception as e:
                logging.error(f"Error in environment guardrail: {e}")
                return GuardrailFunctionOutput(
                    output_info=EnvironmentGuardrailOutput(
                        is_valid=False, 
                        reasoning=f"Error validating environment: {str(e)}"
                    ),
                    tripwire_triggered=True
                )
                
        # Personality designer agent
        self.personality_designer = Agent(
            name="NPCPersonalityDesigner",
            instructions="""
            You are a specialist in designing unique and consistent NPC personalities.
            
            Create personalities with:
            - 3-5 distinct personality traits that form a coherent character
            - 3-5 likes that align with the personality
            - 3-5 dislikes that create interesting tension
            - 2-4 hobbies or interests that make the character feel three-dimensional
            
            The personalities should feel like real individuals with subtle psychological
            depth. Include traits that suggest hidden layers, secret motivations, or
            potential for character growth.
            
            For femdom-themed worlds, incorporate subtle traits related to control,
            authority, or psychological dominance.
            These should be woven naturally into the personality.
            """,
            output_type=NPCPersonalityData,
            model="gpt-4.1-nano",
        )
        
        # Stats calibrator agent
        self.stats_calibrator = Agent(
            name="NPCStatsCalibrator",
            instructions="""
            You calibrate NPC stats to match their personality and archetypes.
            
            Determine appropriate values (0-100) for:
            - dominance: How naturally controlling/authoritative the NPC is
            - cruelty: How willing the NPC is to cause discomfort/distress
            - closeness: How emotionally available/connected the NPC is
            - trust: Trust toward the player (-100 to 100)
            - respect: Respect toward the player (-100 to 100)
            - intensity: Overall emotional/psychological intensity
            
            The stats should align coherently with the personality traits and archetypes.
            For femdom-themed NPCs, calibrate dominance higher (50-90) while ensuring
            other stats create a balanced, nuanced character. Cruelty can vary widely
            depending on personality (10-80).
            """,
            output_type=NPCStatsData,
            model="gpt-4.1-nano",
        )
        
        # Archetype synthesizer agent
        self.archetype_synthesizer = Agent(
            name="NPCArchetypeSynthesizer",
            instructions="""
            You synthesize multiple archetypes into a coherent character concept.
            
            Given multiple archetypes, create:
            - A cohesive archetype summary that blends the archetypes
            - An extras summary explaining how the archetype fusion affects the character
            
            The synthesis should feel natural rather than forced, identifying common
            themes and resolving contradictions between archetypes. Focus on how the
            archetypes interact to create a unique character foundation.
            
            For femdom-themed archetypes, emphasize subtle dominance dynamics while
            maintaining psychological realism and depth.
            """,
            output_type=NPCArchetypeData,
            model="gpt-4.1-nano",
            tools=[function_tool(self.get_available_archetypes)]
        )
        
        # Main NPC creator agent 
        self.npc_creator = Agent(
            name="NPCCreator",
            instructions="""
            You are a specialized agent for creating detailed NPCs for a roleplaying game with subtle femdom elements.
            
            Create NPCs with:
            - Consistent and coherent personalities
            - Realistic motivations and backgrounds
            - Subtle dominance traits hidden behind friendly facades
            - Detailed physical and personality descriptions
            - Appropriate archetypes that fit the game's themes
            
            The NPCs should feel like real individuals with complex personalities and hidden agendas,
            while maintaining a balance between mundane everyday characteristics and subtle control dynamics.
            """,
            output_type=NPCCreationInput,
            model="gpt-4.1-nano",
            tools=[
                function_tool(self.suggest_archetypes),
                function_tool(self.get_environment_details)
            ]
        )
        
        # Schedule creator agent
        self.schedule_creator = Agent(
            name="ScheduleCreator",
            instructions="""
            You create detailed, realistic daily schedules for NPCs in a roleplaying game.
            
            Each schedule should:
            - Fit the NPC's personality, interests, and social status
            - Follow a realistic pattern throughout the week
            - Include variations for different days
            - Place the NPC in appropriate locations at appropriate times
            - Include opportunities for player interactions
            
            The schedules should feel natural and mundane while creating opportunities for
            subtle power dynamics to emerge during player encounters.
            """,
            model="gpt-4.1-nano",
            tools=[
                function_tool(self.get_locations),
                function_tool(self.get_npc_details)
            ]
        )
        
        # Memory creator agent
        self.memory_creator = Agent(
            name="MemoryCreator",
            instructions="""
            You create psychologically rich memories for NPCs that establish their background, relationships, and personality.
            
            Each memory should:
            - Be written in first-person perspective
            - Include sensory details and emotional responses
            - Reveal aspects of the NPC's true nature and personality
            - Establish connections to the environment and other characters
            - For femdom-themed worlds, subtly hint at control dynamics and authority patterns
            
            Memories should feel authentic and provide insight into how the NPC views themselves and others.
            """,
            model="gpt-4.1-nano",
            tools=[
                function_tool(self.get_npc_details),
                function_tool(self.get_environment_details)
            ]
        )
        
        # Main agent with input guardrail
        self.agent = Agent(
            name="NPCCreator",
            instructions="""
            You are an expert NPC creator for immersive, psychologically realistic role-playing games.
            
            Create detailed NPCs with:
            - Unique names and physical descriptions
            - Consistent personalities and motivations
            - Appropriate stat calibrations
            - Coherent archetype synthesis
            - Detailed schedules and memories
            - Subtle elements of control and influence where appropriate
            
            The NPCs should feel like real people with histories, quirks, and hidden depths.
            Focus on psychological realism and subtle complexity rather than explicit themes.
            
            For femdom-themed worlds, incorporate subtle dominance dynamics
            into the character design without being heavy-handed or explicit.
            These elements should be woven naturally into the character's psychology.
            """,
            model="gpt-4.1-nano",
            tools=[
                function_tool(self.generate_npc_name),
                function_tool(self.generate_physical_description),
                function_tool(self.design_personality),
                function_tool(self.calibrate_stats),
                function_tool(self.synthesize_archetypes),
                function_tool(self.generate_schedule),
                function_tool(self.generate_memories),
                function_tool(self.create_npc_in_database),
                function_tool(self.get_environment_details),
                function_tool(self.get_day_names)
            ],
            input_guardrails=[environment_guardrail]
        )
        
        # Keep track of existing NPC names to ensure uniqueness
        self.existing_npc_names = set()
    
    async def initialize_npc_emotional_state(self, user_id, conversation_id, npc_id, npc_data, memories):
        """
        Initialize emotional state based on NPC traits and memories.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
            memories: List of memory strings
        
        Returns:
            Boolean indicating success
        """
        try:
            emotional_manager = EmotionalMemoryManager(user_id, conversation_id)
            
            # Determine base emotional state from NPC traits
            dominance = npc_data.get("dominance", 50)
            cruelty = npc_data.get("cruelty", 30)
            
            # Higher dominance tends toward confident emotions
            # Higher cruelty tends toward colder emotions
            primary_emotion = "neutral"
            if dominance > 70:
                primary_emotion = "confidence" if cruelty < 50 else "pride"
            elif cruelty > 70:
                primary_emotion = "contempt"
            elif dominance > 50 and cruelty > 50:
                primary_emotion = "satisfaction"
            
            # Set intensity based on traits
            intensity = ((dominance + cruelty) / 200) + 0.3  # 0.3-0.8 range
            
            # Create emotional state
            current_emotion = {
                "primary_emotion": primary_emotion,
                "intensity": intensity,
                "secondary_emotions": {},
                "valence": 0.1 if cruelty > 50 else 0.3,  # Slight positive bias
                "arousal": dominance / 200  # Higher dominance = higher arousal
            }
            
            await emotional_manager.update_entity_emotional_state(
                entity_type="npc",
                entity_id=npc_id,
                current_emotion=current_emotion
            )
            
            # Additionally, analyze the emotional content of the first memory
            if memories:
                await emotional_manager.add_emotional_memory(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=memories[0],
                    primary_emotion=primary_emotion,
                    emotion_intensity=intensity
                )
            
            return True
        except Exception as e:
            logging.error(f"Error initializing emotional state for NPC {npc_id}: {e}")
            return False
    
    async def generate_npc_beliefs(
        self,
        user_id: int,
        conversation_id: int,
        npc_id: int,
        npc_data: Dict[str, Any],
        *,
        n: int = 5,
    ) -> bool:
        """Create core beliefs via GPT and store them in the memory system."""
        try:
            from memory.wrapper import MemorySystem  # local import to avoid cycles
    
            memory_system = await MemorySystem.get_instance(user_id, conversation_id)
    
            # Convert personality_traits list to string for caching
            personality_traits = npc_data.get("personality_traits", [])
            personality_traits_str = ", ".join(personality_traits) if personality_traits else ""
    
            beliefs = await generate_core_beliefs(
                npc_data.get("archetype_summary", ""),
                personality_traits_str,  # Pass as string instead of list
                npc_data.get("environment_desc", ""),
                n=n,
            )
    
            for text in beliefs:
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=npc_id,
                    belief_text=text,
                    confidence=0.75,
                )
            return True
        except Exception as e:  # pragma: no cover
            logger.error("Belief generation failed for NPC %s: %s", npc_id, e)
            return False
    
    async def initialize_npc_memory_schemas(self, user_id, conversation_id, npc_id, npc_data):
        """
        Initialize basic memory schemas based on NPC archetype.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
        
        Returns:
            Boolean indicating success
        """
        try:
            schema_manager = MemorySchemaManager(user_id, conversation_id)
            archetype_summary = npc_data.get("archetype_summary", "")
            
            # Create a basic schema for interactions with the player
            await schema_manager.create_schema(
                entity_type="npc",
                entity_id=npc_id,
                schema_name="Player Interactions",
                description="Patterns in how the player behaves toward me",
                category="social",
                attributes={
                    "compliance_level": "unknown",
                    "respect_shown": "moderate",
                    "vulnerability_signs": "to be observed"
                }
            )
            
            # Create archetype-specific schemas
            if npc_data.get("dominance", 50) > 60:
                await schema_manager.create_schema(
                    entity_type="npc",
                    entity_id=npc_id,
                    schema_name="Control Dynamics",
                    description="Patterns of establishing and maintaining control",
                    category="power",
                    attributes={
                        "submission_triggers": "to be identified",
                        "resistance_patterns": "to be analyzed",
                        "effective_techniques": "to be developed"
                    }
                )
            
            return True
        except Exception as e:
            logging.error(f"Error initializing memory schemas for NPC {npc_id}: {e}")
            return False
    
    async def setup_npc_trauma_model(self, user_id, conversation_id, npc_id, npc_data, memories):
        """
        Set up trauma model for NPCs with traumatic backgrounds.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID  
            npc_id: NPC ID
            npc_data: Dict containing NPC data
            memories: List of memory strings
        
        Returns:
            Dict with trauma model information
        """
        try:
            # Only setup trauma for NPCs with higher cruelty or intense backgrounds
            cruelty = npc_data.get("cruelty", 0)
            archetype_summary = npc_data.get("archetype_summary", "").lower()
            
            has_traumatic_background = (
                cruelty > 70 or
                "trauma" in archetype_summary or 
                "tragic" in archetype_summary or
                "abused" in archetype_summary
            )
            
            if not has_traumatic_background:
                # Check memories for trauma indicators
                trauma_keywords = ["hurt", "pain", "suffer", "trauma", "abuse", "betray", "abandon"]
                memory_has_trauma = any(
                    any(keyword in memory.lower() for keyword in trauma_keywords)
                    for memory in memories
                )
                
                if not memory_has_trauma:
                    return {"trauma_model_needed": False}
            
            # At this point, we've determined trauma modeling is appropriate
            emotional_manager = EmotionalMemoryManager(user_id, conversation_id)
            
            # Create traumatic event record
            traumatic_memory = next(
                (memory for memory in memories 
                 if any(keyword in memory.lower() for keyword in ["hurt", "pain", "suffer", "trauma", "abuse", "betray", "abandon"])),
                memories[0] if memories else "A traumatic experience from the past that still affects me today."
            )
            
            # Analyze emotional content
            emotion_analysis = await emotional_manager.analyze_emotional_content(traumatic_memory)
            
            # Create trauma event
            trauma_event = {
                "memory_text": traumatic_memory,
                "emotion": emotion_analysis.get("primary_emotion", "fear"),
                "intensity": emotion_analysis.get("intensity", 0.8),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update emotional state with trauma
            await emotional_manager.update_entity_emotional_state(
                entity_type="npc",
                entity_id=npc_id,
                trauma_event=trauma_event
            )
            
            # Generate trauma triggers
            words = traumatic_memory.split()
            significant_words = [w for w in words if len(w) > 4 and w.isalpha()]
            triggers = random.sample(significant_words, min(3, len(significant_words)))
            
            # Store triggers in database
            async with get_db_connection_context() as conn:
                query = """
                    UPDATE NPCStats
                    SET trauma_triggers = $1
                    WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
                """
                
                await conn.execute(query, json.dumps(triggers), user_id, conversation_id, npc_id)
            
            return {
                "trauma_model_created": True,
                "trauma_triggers": triggers,
                "primary_emotion": emotion_analysis.get("primary_emotion")
            }
        except Exception as e:
            logging.error(f"Error setting up trauma model for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def setup_relationship_evolution_tracking(self, user_id, conversation_id, npc_id, relationships):
        """
        Set up tracking for relationship evolution using dynamic system insights.
        Since dynamic relationships already track evolution, this mainly sets up additional metadata.
        """
        try:
            from lore.core import canon
            
            ctx = type("CanonCtx", (), {
                "user_id": user_id,
                "conversation_id": conversation_id
            })()
            
            # For compatibility, we'll still create RelationshipEvolution entries
            # but they'll be lighter weight since the dynamic system handles most tracking
            if not relationships:
                relationships = []
                
                # Get relationships from NPCStats
                async with get_db_connection_context() as conn:
                    query = """
                        SELECT relationships FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    """
                    row = await conn.fetchrow(query, user_id, conversation_id, npc_id)
                    
                    if row and row['relationships']:
                        if isinstance(row['relationships'], str):
                            try:
                                relationships = json.loads(row['relationships'])
                            except:
                                relationships = []
                        else:
                            relationships = row['relationships'] or []
            
            tracked_count = 0
            
            for relationship in relationships:
                entity_type = relationship.get("entity_type")
                entity_id = relationship.get("entity_id")
                relationship_label = relationship.get("relationship_label", "associate")
                
                if not entity_type or not entity_id:
                    continue
                
                # Create a lightweight evolution tracker
                async with get_db_connection_context() as conn:
                    # Check if relationship evolution entry exists
                    check_query = """
                        SELECT 1 FROM RelationshipEvolution
                        WHERE user_id=$1 AND conversation_id=$2 AND npc1_id=$3 
                        AND entity2_type=$4 AND entity2_id=$5
                    """
                    
                    exists = await conn.fetchrow(
                        check_query,
                        user_id, conversation_id, npc_id, entity_type, entity_id
                    )
                    
                    if not exists:
                        # Create new relationship evolution record
                        insert_query = """
                            INSERT INTO RelationshipEvolution (
                                user_id, conversation_id, npc1_id, entity2_type, entity2_id, 
                                relationship_type, current_stage, progress_to_next, evolution_history
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """
                        
                        await conn.execute(
                            insert_query,
                            user_id, conversation_id, npc_id, entity_type, entity_id,
                            relationship_label, "dynamic", 0,  # Mark as "dynamic" stage
                            json.dumps([{
                                "stage": "dynamic",
                                "date": datetime.now().isoformat(),
                                "note": f"Relationship tracked by dynamic system as {relationship_label}"
                            }])
                        )
                        
                        tracked_count += 1
                        
                        # Log canonical event
                        target_name = "player" if entity_type == "player" else f"entity {entity_id}"
                        await canon.log_canonical_event(
                            ctx, conn,
                            f"Dynamic relationship tracking established between NPC {npc_id} and {target_name}",
                            tags=["relationship", "evolution", "dynamic"],
                            significance=3
                        )
            
            return {"relationships_tracked": tracked_count}
        except Exception as e:
            logging.error(f"Error setting up relationship evolution for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def build_initial_semantic_network(
        self,
        user_id: int,
        conversation_id: int,
        npc_id: int,
        npc_data: Dict[str, Any],
    ):
        """Generate seed topics dynamically then build networks (depth‑1)."""
        from memory.semantic import SemanticMemoryManager
        from db.connection import get_db_connection_context
    
        topics = await get_semantic_seed_topics(
            npc_data.get("archetype_summary", ""),
            npc_data.get("environment_desc", ""),
        )
    
        semantic_manager = SemanticMemoryManager(user_id, conversation_id)
        results: List[Dict[str, Any]] = []
    
        for topic in topics:
            net = await semantic_manager.build_semantic_network(
                entity_type="npc", entity_id=npc_id, central_topic=topic, depth=1
            )
            # Persist (same as original implementation, but condensed)
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO SemanticNetworks (user_id, conversation_id, entity_type, entity_id,
                                                   central_topic, network_data, created_at)
                    VALUES ($1,$2,'npc',$3,$4,$5,CURRENT_TIMESTAMP)
                    """,
                    user_id,
                    conversation_id,
                    npc_id,
                    topic,
                    json.dumps(net),
                )
            results.append({"topic": topic, "nodes": len(net["nodes"]), "edges": len(net["edges"])})
        return {"semantic_networks_created": len(results), "networks": results}

    
    async def detect_memory_patterns(self, user_id, conversation_id, npc_id):
        """
        Detect patterns in memories to establish consistent personality traits.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
        
        Returns:
            Dict with pattern detection results
        """
        try:
            schema_manager = MemorySchemaManager(user_id, conversation_id)
            
            # Attempt to detect schemas from existing memories
            result = await schema_manager.detect_schema_from_memories(
                entity_type="npc",
                entity_id=npc_id,
                min_memories=2  # Lower threshold for initial detection
            )
            
            if result.get("schema_detected", False):
                # Schema detected - we have a pattern to work with
                schema_id = result.get("schema_id")
                schema_name = result.get("schema_name")
                
                # Store this as a personality pattern
                async with get_db_connection_context() as conn:
                    query = """
                        UPDATE NPCStats
                        SET personality_patterns = personality_patterns || $1::jsonb
                        WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
                    """
                    
                    pattern_data = json.dumps([{
                        "pattern_name": schema_name,
                        "schema_id": schema_id,
                        "confidence": result.get("confidence", 0.7),
                        "detected_at": datetime.now().isoformat()
                    }])
                    
                    await conn.execute(query, pattern_data, user_id, conversation_id, npc_id)
                
                return {
                    "pattern_detected": True,
                    "pattern_name": schema_name,
                    "schema_id": schema_id
                }
            
            return {"pattern_detected": False}
        except Exception as e:
            logging.error(f"Error detecting memory patterns for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def schedule_npc_memory_maintenance(self, user_id, conversation_id, npc_id):
        try:
            from lore.core import canon
            
            # Create context - FIX
            ctx = type("CanonCtx", (), {
                "user_id": user_id,
                "conversation_id": conversation_id
            })()
            
            # Create maintenance schedule entry in database
            async with get_db_connection_context() as conn:
                # Check if we already have a schedule
                check_query = """
                    SELECT 1 FROM MemoryMaintenanceSchedule
                    WHERE user_id=$1 AND conversation_id=$2 AND entity_type='npc' AND entity_id=$3
                """
                
                exists = await conn.fetchrow(check_query, user_id, conversation_id, npc_id)
                
                if exists:
                    # Already scheduled
                    return {"already_scheduled": True}
                
                # Create maintenance schedule
                maintenance_types = [
                    {
                        "type": "consolidation",
                        "description": "Consolidate related memories",
                        "interval_days": 3,
                        "last_run": None
                    },
                    {
                        "type": "decay",
                        "description": "Apply memory decay to old memories",
                        "interval_days": 7,
                        "last_run": None
                    },
                    {
                        "type": "schema_update",
                        "description": "Update memory schemas based on new experiences",
                        "interval_days": 5,
                        "last_run": None
                    },
                    {
                        "type": "mask_update",
                        "description": "Evolve mask integrity based on interactions",
                        "interval_days": 2,
                        "last_run": None
                    },
                    {
                        "type": "relationship_sync",
                        "description": "Sync relationship evolution with dynamic system",
                        "interval_days": 1,
                        "last_run": None
                    }
                ]
                
                insert_query = """
                    INSERT INTO MemoryMaintenanceSchedule (
                        user_id, conversation_id, entity_type, entity_id,
                        maintenance_schedule, next_maintenance_date
                    )
                    VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP + INTERVAL '1 day')
                """
                
                await conn.execute(
                    insert_query,
                    user_id, conversation_id, "npc", npc_id, json.dumps(maintenance_types)
                )
                
                # Log canonical event
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Memory maintenance scheduled for NPC {npc_id}",
                    tags=["memory", "maintenance", "schedule"],
                    significance=2
                )
            
            return {
                "maintenance_scheduled": True,
                "maintenance_types": len(maintenance_types)
            }
        except Exception as e:
            logging.error(f"Error scheduling memory maintenance for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def setup_npc_flashback_triggers(
        self,
        user_id: int,
        conversation_id: int,
        npc_id: int,
        npc_data: Dict[str, Any],
    ):
        """Choose trigger words via GPT then store them canonically."""
        from lore.core import canon
        from memory.flashbacks import FlashbackManager
        from db.connection import get_db_connection_context
    
        ctx = type("CanonCtx", (), {"user_id": user_id, "conversation_id": conversation_id})()
    
        flashback_manager = FlashbackManager(user_id, conversation_id)
    
        # 1) Try pulling high‑intensity memory keywords (original logic)…
        trigger_words: List[str] = []
        try:
            from memory.wrapper import MemorySystem
            memory_system = await MemorySystem.get_instance(user_id, conversation_id)
            mems = await memory_system.recall(entity_type="npc", entity_id=npc_id, limit=10)
            intense = [m for m in mems.get("memories", []) if m.get("emotional_intensity", 0) > 60]
            for m in intense:
                trigger_words += [w for w in m["text"].split() if len(w) > 4][:2]
        except Exception:
            pass
    
        # 2) Supplement with GPT‑derived trauma keywords if still sparse
        if len(trigger_words) < 3:
            trigger_words += await get_trauma_keywords(npc_data.get("environment_desc", ""))
            trigger_words = trigger_words[:5]
    
        if not trigger_words:
            return {"triggers_established": 0}
    
        # Make sure at least one flashback path can run immediately (chance=1.0 for test)
        await flashback_manager.check_for_triggered_flashback(
            entity_type="npc", entity_id=npc_id, trigger_words=trigger_words, chance=1.0
        )
    
        async with get_db_connection_context() as conn:
            await canon.update_entity_canonically(
                ctx,
                conn,
                "NPCStats",
                npc_id,
                {"flashback_triggers": json.dumps(trigger_words)},
                f"Set dynamic flashback triggers for NPC {npc_id}",
            )
        return {"triggers_established": len(trigger_words), "trigger_words": trigger_words}

    
    async def generate_counterfactual_memories(self, user_id, conversation_id, npc_id, npc_data):
        """
        Generate 'what-if' alternative versions of key memories to deepen personality.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
        
        Returns:
            Dict with counterfactual generation results
        """
        try:
            semantic_manager = SemanticMemoryManager(user_id, conversation_id)
            
            # Get NPC's existing memories
            memory_system = await MemorySystem.get_instance(user_id, conversation_id)
            memories_result = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                limit=5
            )
            
            # Find a significant memory for counterfactual generation
            significant_memories = [m for m in memories_result.get("memories", []) 
                                  if m.get("significance", 0) >= 3]
            
            if not significant_memories:
                return {"counterfactuals_generated": 0}
                
            # Generate counterfactuals for the most significant memory
            target_memory = significant_memories[0]
            
            # Generate opposite outcome counterfactual
            opposite_cf = await semantic_manager.generate_counterfactual(
                memory_id=target_memory["id"],
                entity_type="npc",
                entity_id=npc_id,
                variation_type="opposite"
            )
            
            # Generate exaggerated outcome counterfactual
            exaggerated_cf = await semantic_manager.generate_counterfactual(
                memory_id=target_memory["id"],
                entity_type="npc",
                entity_id=npc_id,
                variation_type="exaggeration"
            )
            
            return {
                "counterfactuals_generated": 2,
                "based_on_memory": target_memory["text"],
                "counterfactuals": [
                    opposite_cf.get("counterfactual_text"),
                    exaggerated_cf.get("counterfactual_text")
                ]
            }
        except Exception as e:
            logging.error(f"Error generating counterfactuals for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def plan_mask_revelations(self, user_id, conversation_id, npc_id, npc_data):
        try:
            from lore.core import canon
            
            # Create context - FIX
            ctx = type("CanonCtx", (), {
                "user_id": user_id,
                "conversation_id": conversation_id
            })()
            
            mask_manager = ProgressiveRevealManager(user_id, conversation_id)
            
            # Get current mask info
            mask_info = await mask_manager.get_npc_mask(npc_id)
            if "error" in mask_info:
                return {"error": mask_info["error"]}
            
            # Hidden traits that need to be revealed
            hidden_traits = mask_info.get("hidden_traits", {})
            if not hidden_traits:
                return {"revelation_plan_needed": False}
            
            # Create a progressive revelation plan
            revelation_plan = []
            
            # Plan subtle revelations for early encounters
            for trait_name in hidden_traits.keys():
                # Early stage - subtle hints through physical tells
                revelation_plan.append({
                    "trait": trait_name,
                    "severity": RevealSeverity.SUBTLE,
                    "type": RevealType.PHYSICAL,
                    "stage": "early",
                    "integrity_threshold": 90,
                    "trigger_contexts": ["stress", "unexpected", "authority challenged"]
                })
                
                # Mid stage - verbal slips
                revelation_plan.append({
                    "trait": trait_name,
                    "severity": RevealSeverity.MINOR,
                    "type": RevealType.VERBAL_SLIP,
                    "stage": "mid",
                    "integrity_threshold": 70,
                    "trigger_contexts": ["command", "confrontation", "private conversation"]
                })
                
                # Later stage - behavioral inconsistencies
                revelation_plan.append({
                    "trait": trait_name,
                    "severity": RevealSeverity.MODERATE,
                    "type": RevealType.BEHAVIOR,
                    "stage": "later",
                    "integrity_threshold": 50,
                    "trigger_contexts": ["frustration", "opportunity", "unexpected behavior"]
                })
                
                # Final stage - direct revelation
                revelation_plan.append({
                    "trait": trait_name,
                    "severity": RevealSeverity.MAJOR,
                    "type": RevealType.EMOTIONAL,
                    "stage": "final",
                    "integrity_threshold": 30,
                    "trigger_contexts": ["confronted", "cornered", "powerful position"]
                })
            
            # Store the revelation plan using canon system
            async with get_db_connection_context() as conn:
                await canon.update_entity_canonically(
                    ctx, conn, "NPCStats", npc_id,
                    {"revelation_plan": json.dumps(revelation_plan)},
                    f"Planning progressive mask revelations for NPC {npc_data.get('npc_name', npc_id)}"
                )
            
            return {
                "revelation_plan_created": True,
                "planned_revelations": len(revelation_plan),
                "traits_covered": list(hidden_traits.keys())
            }
        except Exception as e:
            logging.error(f"Error planning mask revelations for NPC {npc_id}: {e}")
            return {"error": str(e)},
                        tripwire_triggered=True
                    )
                
    def _get_memory_generation_agent(self) -> Agent:
        if self._memory_agent is None:
            self._memory_agent = Agent(
                name="NPCMemoryGenerator",
                instructions=build_memory_system_prompt(),
                output_type=NPCMemories,
                model=OpenAIResponsesModel(
                    model="gpt-4.1-nano",
                    openai_client=get_async_openai_client(),
                    # optional structured-output hints (if wrapper supports)
                ),
                model_settings=ModelSettings(temperature=0.8),
            )
        return self._memory_agent
    
    # --- Helper methods for robust parsing ---
    
    def safe_json_loads(self, text, default=None):
        """
        Safely parse JSON with multiple fallback methods.
        
        Args:
            text: Text to parse as JSON
            default: Default value to return if parsing fails
            
        Returns:
            Parsed JSON or default value
        """
        if not text:
            return default if default is not None else {}
        
        # Method 1: Direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Look for JSON object within text
        try:
            json_match = re.search(r'(\{[\s\S]*\})', text)
            if json_match:
                return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
        
        # Method 3: Try to fix common JSON syntax errors
        try:
            # Replace single quotes with double quotes
            fixed_text = text.replace("'", '"')
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
        
        # Method 4: Extract field from markdown code block
        try:
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_block_match:
                return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
        
        # Return default if all parsing attempts fail
        return default if default is not None else {}
    
    def extract_field_from_text(self, text, field_name):
        """
        Extract a specific field from text that might contain JSON or key-value patterns.
        
        Args:
            text: Text to extract field from
            field_name: Name of the field to extract
            
        Returns:
            The field value or empty string if not found
        """
        # Try parsing as JSON first
        data = self.safe_json_loads(text)
        if data and field_name in data:
            return data[field_name]
        
        # Try regex patterns for field extraction
        patterns = [
            rf'"{field_name}"\s*:\s*"([^"]*)"',      # For string values: "field": "value"
            rf'"{field_name}"\s*:\s*(\[[^\]]*\])',     # For array values: "field": [...]
            rf'"{field_name}"\s*:\s*(\{{[^}}]*\}})',  # For object values: "field": {...}
            rf'{field_name}:\s*(.*?)(?:\n|$)',          # For plain text: field: value
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def get_unique_npc_name(self, proposed_name: str, existing_names=None) -> str:
        """
        Ensures a name is unique by checking against existing names and unwanted names.
        
        Args:
            proposed_name: The suggested name
            existing_names: List of names already in use
            
        Returns:
            A unique name
        """
        # If the name is "Seraphina" or already exists, choose an alternative from a predefined list.
        unwanted_names = {"seraphina"}
        existing_set = set(existing_names) if existing_names else self.existing_npc_names
        
        if proposed_name.strip().lower() in unwanted_names or proposed_name in existing_set:
            alternatives = ["Aurora", "Celeste", "Luna", "Nova", "Ivy", "Evelyn", "Isolde", "Marina"]
            # Filter out any alternatives already in use
            available = [name for name in alternatives if name not in existing_set and name.lower() not in unwanted_names]
            if available:
                new_name = random.choice(available)
            else:
                # If none available, simply append a random number
                new_name = f"{proposed_name}{random.randint(2, 99)}"
            return new_name
        
        # Add the name to our tracking set
        self.existing_npc_names.add(proposed_name)
        return proposed_name
    
    def clamp(self, value, min_val, max_val):
        """
        Clamp a value between a minimum and maximum.
        
        Args:
            value: Value to clamp
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Clamped value
        """
        return max(min_val, min(value, max_val))
    
    def dynamic_reciprocal_relationship(self, rel_type: str, archetype_summary: str = "") -> str:
        """
        Given a relationship label, return a reciprocal label in a context‐sensitive way.
        
        Args:
            rel_type: Relationship type (e.g., "thrall", "underling", "friend")
            archetype_summary: Summary of the target's archetype for context
            
        Returns:
            Reciprocal relationship label
        """
        fixed = {
            "mother": "child",
            "sister": "younger sibling",
            "aunt": "nephew/niece"
        }
        rel_lower = rel_type.lower()
        if rel_lower in fixed:
            return fixed[rel_lower]
        if rel_lower in ["friend", "best friend"]:
            return rel_type  # mutual relationship
        dynamic_options = {
            "underling": ["boss", "leader", "overseer"],
            "thrall": ["master", "controller", "dominator"],
            "enemy": ["rival", "adversary"],
            "lover": ["lover", "beloved"],
            "colleague": ["colleague"],
            "neighbor": ["neighbor"],
            "classmate": ["classmate"],
            "teammate": ["teammate"],
            "rival": ["rival", "competitor"],
        }
        if rel_lower in dynamic_options:
            if "dominant" in archetype_summary.lower() or "domina" in archetype_summary.lower():
                if rel_lower in ["underling", "thrall"]:
                    return "boss"
            return random.choice(dynamic_options[rel_lower])
        return "associate"
    
    async def get_entity_name(self, conn, entity_type, entity_id, user_id, conversation_id):
        """
        Get the name of an entity (NPC or player).
        
        Args:
            conn: Database connection
            entity_type: Type of entity ("npc" or "player")
            entity_id: ID of the entity
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Name of the entity
        """
        if entity_type == 'player':
            return "Chase"
        
        query = """
            SELECT npc_name FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
        """
        
        row = await conn.fetchrow(query, user_id, conversation_id, entity_id)
        
        return row['npc_name'] if row else "Unknown"
    
    def get_reciprocal_description(self, description):
        """
        Generate a reciprocal description from the perspective of the other entity.
        
        Args:
            description: Original description
            
        Returns:
            Reciprocal description
        """
        # Simple replacements for now
        replacements = {
            "control": "being controlled",
            "dominance": "submission",
            "manipulating": "being influenced",
            "assessing vulnerabilities": "being evaluated",
            "control and submission": "submission and control"
        }
        
        result = description
        for original, replacement in replacements.items():
            result = result.replace(original, replacement)
        
        return result
    
    async def add_npc_memory(self, conn, user_id, conversation_id, npc_id, memory_text):
        """
        Add a memory entry for an NPC using the canon system.
        """
        try:
            from lore.core import canon
            
            # Create context
            ctx = type("CanonCtx", (), {
                "user_id": user_id,
                "conversation_id": conversation_id
            })()
            
            query = """
                SELECT memory FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
            """
            
            row = await conn.fetchrow(query, user_id, conversation_id, npc_id)
            if row and row['memory']:
                if isinstance(row['memory'], str):
                    try:
                        memory = json.loads(row['memory'])
                    except:
                        memory = []
                else:
                    memory = row['memory']
            else:
                memory = []
            
            memory.append(memory_text)
            
            # Update through canon system
            await canon.update_entity_canonically(
                ctx, conn, "NPCStats", npc_id,
                {"memory": json.dumps(memory)},
                f"Adding new memory to NPC {npc_id}"
            )
            
        except Exception as e:
            logging.error(f"Error adding NPC memory: {e}")
    
    # --- NPC data retrieval methods ---
    
    async def get_available_archetypes(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Get available archetypes from the database.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
        Returns:
            List of archetype data
        """
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT id, name, baseline_stats, progression_rules, 
                           setting_examples, unique_traits
                    FROM Archetypes
                    ORDER BY name
                """
                
                rows = await conn.fetch(query)
                archetypes = []
                
                for row in rows:
                    archetype = {
                        "id": row['id'],
                        "name": row['name']
                    }
                    
                    # Add detailed information if available
                    if row['baseline_stats']:  # baseline_stats
                        try:
                            if isinstance(row['baseline_stats'], str):
                                archetype["baseline_stats"] = json.loads(row['baseline_stats'])
                            else:
                                archetype["baseline_stats"] = row['baseline_stats']
                        except:
                            pass
                    
                    if row['unique_traits']:  # unique_traits
                        try:
                            if isinstance(row['unique_traits'], str):
                                archetype["unique_traits"] = json.loads(row['unique_traits'])
                            else:
                                archetype["unique_traits"] = row['unique_traits']
                        except:
                            pass
                    
                    archetypes.append(archetype)
                
                return archetypes
        except Exception as e:
            logging.error(f"Error getting archetypes: {e}")
            return []
    
    async def suggest_archetypes(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Suggest appropriate archetypes for NPCs.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
        Returns:
            List of archetype objects with id and name
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")
            
            async with get_db_connection_context() as conn:
                query = """
                    SELECT id, name
                    FROM Archetypes
                    ORDER BY id
                """
                
                rows = await conn.fetch(query)
                archetypes = []
                
                for row in rows:
                    archetypes.append({
                        "id": row['id'],
                        "name": row['name']
                    })
                
                return archetypes
        except Exception as e:
            logging.error(f"Error suggesting archetypes: {e}")
            return []
    
    async def get_environment_details(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get details about the game environment.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
        Returns:
            Dictionary with environment details
        """
        user_id = ctx.context.get("user_id")
        conversation_id = ctx.context.get("conversation_id")
        
        if not user_id or not conversation_id:
            return {
                "environment_desc": "A detailed, immersive world with subtle layers of control and influence.",
                "setting_name": "Default Setting",
                "locations": []
            }
        
        try:
            async with get_db_connection_context() as conn:
                # Get environment description
                env_query = """
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc'
                """
                
                env_row = await conn.fetchrow(env_query, user_id, conversation_id)
                environment_desc = env_row['value'] if env_row else "No environment description available"
                
                # Get current setting
                setting_query = """
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentSetting'
                """
                
                setting_row = await conn.fetchrow(setting_query, user_id, conversation_id)
                setting_name = setting_row['value'] if setting_row else "Unknown Setting"
                
                # Get locations
                locations_query = """
                    SELECT location_name, description FROM Locations
                    WHERE user_id=$1 AND conversation_id=$2
                    LIMIT 10
                """
                
                location_rows = await conn.fetch(locations_query, user_id, conversation_id)
                locations = []
                
                for row in location_rows:
                    locations.append({
                        "name": row['location_name'],
                        "description": row['description']
                    })
                
                return {
                    "environment_desc": environment_desc,
                    "setting_name": setting_name,
                    "locations": locations
                }
        except Exception as e:
            logging.error(f"Error getting environment details: {e}")
            return {
                "environment_desc": "Error retrieving environment",
                "setting_name": "Unknown",
                "locations": []
            }
    
    async def get_locations(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Get all locations in the game world.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
        Returns:
            List of location objects
        """
        user_id = ctx.context.get("user_id")
        conversation_id = ctx.context.get("conversation_id")
        
        if not user_id or not conversation_id:
            return []
        
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT id, location_name, description, open_hours
                    FROM Locations
                    WHERE user_id=$1 AND conversation_id=$2
                    ORDER BY id
                """
                
                rows = await conn.fetch(query, user_id, conversation_id)
                locations = []
                
                for row in rows:
                    location = {
                        "id": row['id'],
                        "location_name": row['location_name'],
                        "description": row['description']
                    }
                    
                    # Parse open_hours if available
                    if row['open_hours']:
                        try:
                            if isinstance(row['open_hours'], str):
                                location["open_hours"] = json.loads(row['open_hours'])
                            else:
                                location["open_hours"] = row['open_hours']
                        except:
                            location["open_hours"] = []
                    else:
                        location["open_hours"] = []
                    
                    locations.append(location)
                
                return locations
        except Exception as e:
            logging.error(f"Error getting locations: {e}")
            return []
    
    async def get_day_names(self, ctx: RunContextWrapper) -> List[str]:
        """
        Get custom day names from the calendar system.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
        Returns:
            List of day names
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")
            
            if not user_id or not conversation_id:
                return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            calendar_data = await load_calendar_names(user_id, conversation_id)
            day_names = calendar_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            
            return day_names
        except Exception as e:
            logging.error(f"Error getting day names: {e}")
            return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    async def get_npc_details(self, ctx: RunContextWrapper, npc_id=None, npc_name=None) -> Dict[str, Any]:
        """
        Get details about a specific NPC.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
            npc_id: ID of the NPC to get details for (optional)
            npc_name: Name of the NPC to get details for (optional)
            
        Returns:
            Dictionary with NPC details
        """
        user_id = ctx.context.get("user_id")
        conversation_id = ctx.context.get("conversation_id")
        
        if not user_id or not conversation_id:
            return {"error": "Missing user_id or conversation_id"}
        
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT npc_id, npc_name, introduced, archetypes, archetype_summary, 
                           archetype_extras_summary, physical_description, relationships,
                           dominance, cruelty, closeness, trust, respect, intensity,
                           hobbies, personality_traits, likes, dislikes, affiliations,
                           schedule, current_location, sex, age, memory
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                """
                
                params = [user_id, conversation_id]
                
                if npc_id is not None:
                    query += " AND npc_id=$3"
                    params.append(npc_id)
                elif npc_name is not None:
                    query += " AND LOWER(npc_name)=LOWER($3)"
                    params.append(npc_name)
                else:
                    return {"error": "No NPC ID or name provided"}
                
                query += " LIMIT 1"
                
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    return {"error": "NPC not found"}
                
                # Process JSON fields
                def parse_json_field(field):
                    if field is None:
                        return []
                    if isinstance(field, str):
                        try:
                            return json.loads(field)
                        except:
                            return []
                    return field
                
                return {
                    "npc_id": row['npc_id'],
                    "npc_name": row['npc_name'],
                    "introduced": row['introduced'],
                    "archetypes": parse_json_field(row['archetypes']),
                    "archetype_summary": row['archetype_summary'],
                    "archetype_extras_summary": row['archetype_extras_summary'],
                    "physical_description": row['physical_description'],
                    "relationships": parse_json_field(row['relationships']),
                    "dominance": row['dominance'],
                    "cruelty": row['cruelty'],
                    "closeness": row['closeness'],
                    "trust": row['trust'],
                    "respect": row['respect'],
                    "intensity": row['intensity'],
                    "hobbies": parse_json_field(row['hobbies']),
                    "personality_traits": parse_json_field(row['personality_traits']),
                    "likes": parse_json_field(row['likes']),
                    "dislikes": parse_json_field(row['dislikes']),
                    "affiliations": parse_json_field(row['affiliations']),
                    "schedule": parse_json_field(row['schedule']),
                    "current_location": row['current_location'],
                    "sex": row['sex'],
                    "age": row['age'],
                    "memories": parse_json_field(row['memory'])
                }
        except Exception as e:
            logging.error(f"Error getting NPC details: {e}")
            return {"error": f"Error retrieving NPC details: {str(e)}"}
    
    # --- Helper methods for NPC generation ---
    
    async def integrate_femdom_elements(self, npc_data):
        """
        Analyze NPC data and subtly integrate femdom elements based on dominance level.
        This doesn't add overt femdom content, but rather plants seeds through traits.
        
        Args:
            npc_data: NPC data to enhance
            
        Returns:
            Updated NPC data with subtle femdom elements
        """
        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 30)
        
        # Don't add femdom tendencies if dominance is very low
        if dominance < 20:
            return npc_data
            
        # Determine femdom intensity based on dominance and cruelty
        femdom_intensity = (dominance + cruelty) / 2
        is_high_intensity = femdom_intensity > 70
        is_medium_intensity = 40 <= femdom_intensity <= 70
        
        # Copy existing traits
        personality_traits = npc_data.get("personality_traits", [])
        likes = npc_data.get("likes", [])
        dislikes = npc_data.get("dislikes", [])
        
        # Add subtle femdom personality traits based on intensity
        potential_traits = []
        
        if is_high_intensity:
            potential_traits += [
                "enjoys being obeyed",
                "naturally commanding",
                "good at reading weaknesses",
                "expects compliance",
                "subtle manipulator",
                "finds pleasure in control",
                "notices when others defer to her",
            ]
        elif is_medium_intensity:
            potential_traits += [
                "prefers making decisions",
                "naturally takes charge",
                "surprisingly assertive at times",
                "notices small power dynamics",
                "enjoys being respected",
                "secretly enjoys having influence",
                "feels comfortable setting rules"
            ]
        else:
            potential_traits += [
                "occasionally assertive",
                "selective with permissions",
                "expects politeness",
                "notices disrespect quickly",
                "sometimes tests boundaries",
                "appreciates deference"
            ]
        
        # Select 1-2 traits to add without being too obvious
        traits_to_add = random.sample(potential_traits, min(2, len(potential_traits)))
        for trait in traits_to_add:
            if trait not in personality_traits:
                personality_traits.append(trait)
        
        # Add subtle femdom likes/dislikes based on intensity
        potential_likes = []
        potential_dislikes = []
        
        if is_high_intensity:
            potential_likes += [
                "seeing others follow her lead",
                "making important decisions",
                "being the center of attention",
                "quiet acknowledgment of her authority",
                "setting clear expectations"
            ]
            potential_dislikes += [
                "being interrupted",
                "unexpected defiance",
                "having her judgment questioned",
                "people who overstep boundaries",
                "being ignored"
            ]
        elif is_medium_intensity:
            potential_likes += [
                "receiving prompt responses",
                "being asked for permission",
                "planning events for others",
                "mentoring those who listen well",
                "being consulted on decisions"
            ]
            potential_dislikes += [
                "tardiness",
                "people who speak over her",
                "having to repeat herself",
                "casual dismissals of her opinions"
            ]
        else:
            potential_likes += [
                "well-mannered individuals",
                "being respected in conversations",
                "when others remember her preferences",
                "thoughtful attentiveness"
            ]
            potential_dislikes += [
                "poor etiquette",
                "being contradicted publicly",
                "presumptuous behavior"
            ]
        
        # Add 1 like and 1 dislike
        if potential_likes:
            new_like = random.choice(potential_likes)
            if new_like not in likes:
                likes.append(new_like)
                
        if potential_dislikes:
            new_dislike = random.choice(potential_dislikes)
            if new_dislike not in dislikes:
                dislikes.append(new_dislike)
        
        # Update the data and return
        updated_data = npc_data.copy()
        updated_data["personality_traits"] = personality_traits
        updated_data["likes"] = likes
        updated_data["dislikes"] = dislikes
        
        return updated_data
    
    # --- NPC generation methods ---
                  
    async def generate_npc_name(
        self,
        ctx: RunContextWrapper,
        desired_gender: str = "female",
        style: str = "unique",
        forbidden_names=None,
        *,
        model: str = _DEFAULT_NAME_MODEL,
        temperature: float = 0.8,
        max_output_tokens: int = 50,
    ) -> str:
        """
        Generate a unique NPC name using the Responses API.
        Returns a *single* name string; falls back to heuristic list on failure.
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")

            # Environment context
            env_details = await self.get_environment_details(ctx)
            env_desc = env_details.get("environment_desc", "")

            # Existing names
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT npc_name FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                    """,
                    user_id,
                    conversation_id,
                )
            existing_names = [r["npc_name"] for r in rows] if rows else []
            if forbidden_names:
                existing_names.extend(forbidden_names)

            system_prompt = (
                "You generate *one* in-setting NPC name.\n"
                "Return ONLY the name OR valid JSON: {\"name\": \"Full Name\"}.\n"
                "No commentary."
            )

            # include context JSON block to discourage duplicates
            context_block = json.dumps(
                {
                    "desired_gender": desired_gender,
                    "style": style,
                    "existing_names": existing_names,
                    "environment_excerpt": env_desc[:500],
                },
                ensure_ascii=False,
            )
            user_prompt = (
                f"Generate a unique {desired_gender} NPC name styled as {style}.\n"
                "Avoid existing names listed.\n"
                "Setting excerpt provided.\n"
                "Return ONLY the name or JSON with 'name'.\n"
                f"\n### CONTEXT_JSON\n{context_block}\n"
            )

            raw_txt = await _responses_json_call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                
            )

            name = _coerce_name(
                raw_txt,
                forbidden=forbidden_names or [],
                existing=existing_names,
            )
            # Track for uniqueness enforcement in instance list
            name = self.get_unique_npc_name(name, existing_names)
            return name

        except Exception as e:
            logging.error(f"Error generating NPC name: {e}", exc_info=True)
            # fallback
            first_names = [
                "Elara",
                "Thalia",
                "Vespera",
                "Lyra",
                "Nadia",
                "Corin",
                "Isadora",
                "Maren",
                "Octavia",
                "Quinn",
            ]
            last_names = [
                "Valen",
                "Nightshade",
                "Wolfe",
                "Thorn",
                "Blackwood",
                "Frost",
                "Stone",
                "Rivers",
                "Skye",
                "Ash",
            ]
            if forbidden_names:
                first_names = [n for n in first_names if n not in forbidden_names]
            return f"{random.choice(first_names)} {random.choice(last_names)}"
            
    async def generate_physical_description(
        self,
        ctx: RunContextWrapper,
        npc_name: str,
        archetype_summary: str = "",
        environment_desc: str | None = None,
        *,
        model: str = _DEFAULT_DESC_MODEL,
        temperature: float = 0.7,
        max_output_tokens: int | None = None,
    ) -> str:
        """
        Generate a detailed physical description for an NPC (2–3 paragraphs).
        Returns a string.
        """
        try:
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]

            # Rich guidance stays, but trimmed for clarity; we instruct JSON.
            system_prompt = (
                "You write vivid *third-person* physical descriptions for NPCs in a mature, "
                "femdom-toned RPG. Evocative, sensory, explicit detail\n"
                "Respond as JSON: {\"physical_description\": \"...\"}."
            )

            user_payload = {
                "npc_name": npc_name,
                "archetype_summary": archetype_summary,
                "environment_desc": environment_desc,
                "requirements": {
                    "paragraphs": "2-3",
                    "sensory": True,
                    "style": "evocative, immersive, erotic",
                    "show_archetype_in_appearance": True,
                    "include_mannerisms": True,
                    "include_voice_scent_presence": True,
                },
            }
            user_prompt = (
                "Generate a detailed physical description meeting the requirements below.\n"
                "Return JSON with key physical_description (string).\n"
                f"\n### CONTEXT_JSON\n{json.dumps(user_payload, ensure_ascii=False)}\n"
            )

            raw_txt = await _responses_json_call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                
            )

            data = _json_first_obj(raw_txt) or {}
            return _coerce_physical_desc(data, npc_name=npc_name)

        except Exception as e:
            logging.error(f"Error generating physical description for {npc_name}: {e}", exc_info=True)
            return f"{npc_name} has an appearance that matches her role in this environment."
    
    async def design_personality(
        self,
        ctx: RunContextWrapper,
        npc_name: str,
        archetype_summary: str = "",
        environment_desc: str | None = None,
        *,
        model: str = _DEFAULT_PERS_MODEL,
        temperature: float = 0.7,
        max_output_tokens: int | None = None,
    ) -> NPCPersonalityData:
        """
        Create a coherent personality profile (traits/likes/dislikes/hobbies).
        Returns NPCPersonalityData.
        """
        try:
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]

            system_prompt = (
                "You design psychologically coherent NPC personalities for an RPG.\n"
                "Output JSON: {personality_traits:[], likes:[], dislikes:[], hobbies:[]}.\n"
                "3-5 items per list; concise natural language phrases."
            )
            payload = {
                "npc_name": npc_name,
                "environment_desc": environment_desc,
                "archetype_summary": archetype_summary,
                "guidance": {
                    "min_traits": 3,
                    "max_traits": 5,
                    "min_likes": 3,
                    "max_likes": 5,
                    "min_dislikes": 3,
                    "max_dislikes": 5,
                    "min_hobbies": 2,
                    "max_hobbies": 4,
                    "subtle_femdom": True,
                },
            }
            user_prompt = (
                "Design a personality profile meeting the guidance below.\n"
                "Return JSON only.\n"
                f"\n### CONTEXT_JSON\n{json.dumps(payload, ensure_ascii=False)}\n"
            )

            raw_txt = await _responses_json_call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                
            )
            data = _json_first_obj(raw_txt) or {}
            return _coerce_personality(data)

        except Exception as e:
            logging.error(f"Error designing personality: {e}", exc_info=True)
            return NPCPersonalityData(
                personality_traits=["confident", "observant", "private"],
                likes=["structure", "competence", "subtle control"],
                dislikes=["vulnerability", "unpredictability", "unnecessary conflict"],
                hobbies=["psychology", "strategic games"],
            )

    
    async def calibrate_stats(
        self,
        ctx: RunContextWrapper,
        npc_name: str,
        personality: NPCPersonalityData | None = None,
        archetype_summary: str = "",
        *,
        model: str = _DEFAULT_STATS_MODEL,
        temperature: float = 0.4,
        max_output_tokens: int | None = None,
    ) -> NPCStatsData:
        """
        Calibrate numeric stat block aligned w/ personality + archetype.
        Returns NPCStatsData.
        """
        try:
            pers = personality.dict() if personality else {}
            system_prompt = (
                "You assign calibrated numeric stats (0-100 except trust/respect -100..100) "
                "for RPG NPCs based on personality + archetype.\n"
                "Return JSON: {dominance:int, cruelty:int, closeness:int, trust:int, respect:int, intensity:int}."
            )
            payload = {
                "npc_name": npc_name,
                "archetype_summary": archetype_summary,
                "personality": pers,
                "scales": {
                    "dominance": "0=passive,100=commanding",
                    "cruelty": "0=gentle,100=delights in harm",
                    "closeness": "0=distant,100=deeply bonded",
                    "trust": "-100=active distrust,0=neutral,100=complete trust",
                    "respect": "-100=contempt,0=neutral,100=deep respect",
                    "intensity": "0=muted presence,100=overwhelming force",
                },
            }
            user_prompt = (
                "Calibrate stats using the context below.\n"
                "Return JSON only.\n"
                f"\n### CONTEXT_JSON\n{json.dumps(payload, ensure_ascii=False)}\n"
            )

            raw_txt = await _responses_json_call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                
            )
            data = _json_first_obj(raw_txt) or {}
            return _coerce_stats(data)

        except Exception as e:
            logging.error(f"Error calibrating stats: {e}", exc_info=True)
            return NPCStatsData(
                dominance=60,
                cruelty=40,
                closeness=50,
                trust=20,
                respect=30,
                intensity=55,
            )

    
    async def synthesize_archetypes(
        self,
        ctx: RunContextWrapper,
        archetype_names: list[str] | None = None,
        npc_name: str = "",
        *,
        model: str = _DEFAULT_ARCH_MODEL,
        temperature: float = 0.8,
        max_output_tokens: int | None = None,
    ) -> NPCArchetypeData:
        """
        Blend multiple archetypes into a cohesive concept & extras summary.
        Returns NPCArchetypeData.
        """
        try:
            if not archetype_names:
                # fallback: sample from DB
                available_archetypes = await self.get_available_archetypes(ctx)
                if available_archetypes:
                    selected = random.sample(available_archetypes, min(3, len(available_archetypes)))
                    archetype_names = [arch["name"] for arch in selected]
                else:
                    archetype_names = ["Mentor", "Authority Figure", "Hidden Depth"]

            system_prompt = (
                "You synthesize multiple RPG archetypes into one coherent character concept.\n"
                "Return JSON: {archetype_names:[], archetype_summary:str, archetype_extras_summary:str}."
            )
            payload = {
                "npc_name": npc_name,
                "archetypes": archetype_names,
                "guidelines": {
                    "resolve_conflicts": True,
                    "highlight_common_themes": True,
                    "subtle_femdom": True,
                    "max_summary_chars": 400,
                    "max_extras_chars": 400,
                },
            }
            user_prompt = (
                "Blend the listed archetypes into a single coherent character concept.\n"
                "Return JSON only.\n"
                f"\n### CONTEXT_JSON\n{json.dumps(payload, ensure_ascii=False)}\n"
            )

            raw_txt = await _responses_json_call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                
            )
            data = _json_first_obj(raw_txt) or {}
            return _coerce_archetype(data, provided_names=archetype_names)

        except Exception as e:
            logging.error(f"Error synthesizing archetypes: {e}", exc_info=True)
            return NPCArchetypeData(
                archetype_names=archetype_names or ["Authority Figure"],
                archetype_summary="A complex character with layers of authority and hidden depth.",
                archetype_extras_summary="Authority expressed through subtle psychological control rather than overt dominance.",
            )

    
    async def generate_schedule(
        self,
        ctx: RunContextWrapper,
        npc_name: str,
        environment_desc: str | None = None,
        day_names: list[str] | None = None,
        *,
        model: str = _DEFAULT_SCHED_MODEL,
        temperature: float = 0.7,
        max_output_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """
        Generate a weekly schedule keyed by day -> {Morning/Afternoon/Evening/Night}.
        Returns dict[str, dict[str,str]].
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")

            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]

            if not day_names:
                day_names = await self.get_day_names(ctx)

            # Pull NPC row (optional: errors tolerated)
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT npc_id, archetypes, hobbies, personality_traits
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_name=$3
                    LIMIT 1
                    """,
                    user_id,
                    conversation_id,
                    npc_name,
                )
            archetypes = []
            hobbies = []
            personality_traits = []
            if row:
                def _parse(v):
                    if not v:
                        return []
                    if isinstance(v, str):
                        try:
                            return json.loads(v)
                        except Exception:
                            return []
                    return v
                archetypes = _parse(row["archetypes"])
                hobbies = _parse(row["hobbies"])
                personality_traits = _parse(row["personality_traits"])

            # shorter list of archetype names for prompt readability
            archetype_names = [a.get("name", "") for a in archetypes if isinstance(a, dict)]

            system_prompt = (
                "You create structured weekly schedules for RPG NPCs.\n"
                "Output JSON: {\"schedule\": {DAY: {Morning:\"\",Afternoon:\"\",Evening:\"\",Night:\"\"}, ...}}.\n"
                "Activities must reflect personality, archetypes, environment, and offer interaction hooks."
            )
            payload = {
                "npc_name": npc_name,
                "environment_desc": environment_desc,
                "day_names": day_names,
                "archetypes": archetype_names,
                "hobbies": hobbies,
                "personality_traits": personality_traits,
                "style_guidelines": {
                    "tie_to_environment": True,
                    "vary_by_day": True,
                    "include_social_hooks": True,
                    "subtle_power_dynamics": True,
                    "keep_entries_short": True,
                },
            }
            user_prompt = (
                "Generate the schedule described. Use all listed days.\n"
                "Return JSON ONLY.\n"
                f"\n### CONTEXT_JSON\n{json.dumps(payload, ensure_ascii=False)}\n"
            )

            raw_txt = await _responses_json_call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                
            )
            data = _json_first_obj(raw_txt) or {}
            return _coerce_schedule(data, day_names=day_names)

        except Exception as e:
            logging.error(f"Error generating schedule: {e}", exc_info=True)
            # Fallback simple schedule
            if not day_names:
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            schedule = {}
            for day in day_names:
                schedule[day] = {
                    "Morning": f"{npc_name} starts the day with personal routines.",
                    "Afternoon": f"{npc_name} attends to responsibilities.",
                    "Evening": f"{npc_name} engages in social or hobby activities.",
                    "Night": f"{npc_name} returns home and rests.",
                }
            return schedule
    
    async def generate_memories(
        self,
        ctx: RunContextWrapper,
        npc_name,
        environment_desc=None,
    ) -> List[str]:
        """
        Generate detailed memories for an NPC (rich-prompt + structured JSON).
        """
        try:
            user_id        = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")

            # --- Environment ---------------------------------------------------
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]

            # --- NPC row -------------------------------------------------------
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT npc_id, archetypes, archetype_summary, relationships,
                           dominance, cruelty, personality_traits
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_name=$3
                    LIMIT 1
                    """,
                    user_id,
                    conversation_id,
                    npc_name,
                )

            if not row:
                raise RuntimeError(f"NPC '{npc_name}' not found.")

            dominance = row["dominance"]
            cruelty   = row["cruelty"]

            # local JSON decoders
            def _parse(field):
                if not field:
                    return []
                if isinstance(field, str):
                    try:
                        return json.loads(field)
                    except Exception:
                        return []
                return field

            relationships      = _parse(row["relationships"])
            personality_traits = _parse(row["personality_traits"])

            # --- Build creative prompt text -----------------------------------
            user_prompt = build_memory_user_prompt(
                npc_name=npc_name,
                environment_desc=environment_desc,
                archetype=row["archetype_summary"] or "Unknown",
                dominance=dominance,
                cruelty=cruelty,
                personality_traits=personality_traits,
                relationships=relationships,
            )

            # --- Run via Agent SDK --------------------------------------------
            mem_agent  = self._get_memory_generation_agent()
            run_result = await Runner.run(
                mem_agent,
                user_prompt,            # send the rich creative brief
                context=ctx.context,    # pass through metadata (user_id, etc.)
            )

            # parse -> Pydantic
            mem_obj = run_result.final_output_as(NPCMemories)

            if mem_obj and mem_obj.memories:
                mems = [m.strip() for m in mem_obj.memories if isinstance(m, str) and m.strip()]
                if mems:
                    return mems

        except Exception as e:
            logging.error(f"Error generating memories: {e}", exc_info=True)

        # ----------------------------------------------------------------------
        # Fallback memories (unchanged)
        # ----------------------------------------------------------------------
        return [
            (
                f"I remember when I first arrived in this place. The atmosphere was "
                f"both familiar and strange, like I belonged here but didn't yet know why."
            ),
            (
                "There was that conversation last month where I realized how easily "
                "people shared their secrets with me. It was fascinating how a simple "
                "question, asked the right way, could reveal so much."
            ),
            (
                "Sometimes I think about my position here and the subtle influence I've cultivated. "
                "Few realize how carefully I've positioned myself within the social dynamics."
            ),
        ]
    
    async def create_npc_in_database(self, ctx: RunContextWrapper, npc_data) -> Dict[str, Any]:
        """
        Create an NPC in the database using the canon system for consistency.
        """
        try:
            # Fix: Access context attributes correctly
            if hasattr(ctx, 'context') and isinstance(ctx.context, dict):
                user_id = ctx.context.get("user_id")
                conversation_id = ctx.context.get("conversation_id")
            else:
                user_id = getattr(ctx, 'user_id', None)
                conversation_id = getattr(ctx, 'conversation_id', None)
                
            if not user_id or not conversation_id:
                raise ValueError("Missing user_id or conversation_id in context")
            
            # Import canon system
            from lore.core import canon
            
            # Create a proper canonical context EARLY
            canon_ctx = type('CanonicalContext', (), {
                'user_id': user_id,
                'conversation_id': conversation_id
            })()
            
            # Get day names for scheduling
            day_names = await self.get_day_names(ctx)
            
            # Get environment details
            env_details = await self.get_environment_details(ctx)
            environment_desc = env_details["environment_desc"]
            
            # Extract values from npc_data
            npc_name = npc_data.get("npc_name", "Unnamed NPC")
            physical_description = npc_data.get("physical_description", "")
            introduced = npc_data.get("introduced", False)
            sex = npc_data.get("sex", "female")
            
            # Extract personality data
            personality = npc_data.get("personality", {})
            if isinstance(personality, dict):
                personality_traits = personality.get("personality_traits", [])
                likes = personality.get("likes", [])
                dislikes = personality.get("dislikes", [])
                hobbies = personality.get("hobbies", [])
            else:
                personality_traits = getattr(personality, "personality_traits", [])
                likes = getattr(personality, "likes", [])
                dislikes = getattr(personality, "dislikes", [])
                hobbies = getattr(personality, "hobbies", [])
            
            # Extract stats data
            stats = npc_data.get("stats", {})
            if isinstance(stats, dict):
                dominance = stats.get("dominance", 50)
                cruelty = stats.get("cruelty", 30)
                closeness = stats.get("closeness", 50)
                trust = stats.get("trust", 0)
                respect = stats.get("respect", 0)
                intensity = stats.get("intensity", 40)
            else:
                dominance = getattr(stats, "dominance", 50)
                cruelty = getattr(stats, "cruelty", 30)
                closeness = getattr(stats, "closeness", 50)
                trust = getattr(stats, "trust", 0)
                respect = getattr(stats, "respect", 0)
                intensity = getattr(stats, "intensity", 40)
            
            # Extract archetype data
            archetypes = npc_data.get("archetypes", {})
            if isinstance(archetypes, dict):
                archetype_names = archetypes.get("archetype_names", [])
                archetype_summary = archetypes.get("archetype_summary", "")
                archetype_extras_summary = archetypes.get("archetype_extras_summary", "")
            else:
                archetype_names = getattr(archetypes, "archetype_names", [])
                archetype_summary = getattr(archetypes, "archetype_summary", "")
                archetype_extras_summary = getattr(archetypes, "archetype_extras_summary", "")
            
            # Format archetypes for database
            archetype_objs = [{"name": name} for name in archetype_names]
            
            # Apply subtle femdom elements
            npc_data_with_femdom = await self.integrate_femdom_elements({
                "npc_name": npc_name,
                "dominance": dominance,
                "cruelty": cruelty,
                "personality_traits": personality_traits,
                "likes": likes,
                "dislikes": dislikes
            })
            
            # Get updated traits
            personality_traits = npc_data_with_femdom.get("personality_traits", personality_traits)
            likes = npc_data_with_femdom.get("likes", likes)
            dislikes = npc_data_with_femdom.get("dislikes", dislikes)
            
            # Generate age and birthdate
            age = random.randint(20, 50)
            calendar_data = await load_calendar_names(user_id, conversation_id)
            months_list = calendar_data.get("months", [
                "Frostmoon", "Windspeak", "Bloomrise", "Dawnsveil",
                "Emberlight", "Goldencrest", "Shadowleaf", "Harvesttide",
                "Stormcall", "Nightwhisper", "Snowbound", "Yearsend"
            ])
            birth_month = random.choice(months_list)
            birth_day = random.randint(1, 28)
            birthdate = f"{birth_month} {birth_day}"
            
            # =====================================================
            # STEP 1: CREATE THE NPC FIRST (with minimal data)
            # =====================================================
            async with get_db_connection_context() as conn:
                # Create NPC canonically with just the basic info
                npc_id = await canon.find_or_create_npc(
                    canon_ctx, conn, npc_name,
                    role=archetype_summary,
                    affiliations=npc_data.get("affiliations", [])
                )
                
                if not npc_id:
                    raise ValueError(f"Failed to create NPC {npc_name} - no ID returned")
                
                logger.info(f"Created NPC {npc_name} with ID {npc_id}")
                
                # Update with basic stats first
                basic_updates = {
                    "introduced": introduced,
                    "sex": sex,
                    "dominance": dominance if dominance is not None else 0,
                    "cruelty": cruelty if cruelty is not None else 0,
                    "closeness": closeness if closeness is not None else 0,
                    "trust": trust if trust is not None else 0,
                    "respect": respect if respect is not None else 0,
                    "intensity": intensity if intensity is not None else 0,
                    "archetypes": json.dumps(archetype_objs),
                    "archetype_summary": archetype_summary,
                    "archetype_extras_summary": archetype_extras_summary,
                    "likes": json.dumps(likes),
                    "dislikes": json.dumps(dislikes),
                    "hobbies": json.dumps(hobbies),
                    "personality_traits": json.dumps(personality_traits),
                    "age": age,
                    "birthdate": birthdate,
                    "relationships": '[]',
                    "memory": '[]',  # Empty for now
                    "schedule": '{}',  # Empty for now
                    "physical_description": physical_description,
                    "current_location": ""  # Will determine later
                }
                
                await canon.update_entity_canonically(
                    canon_ctx, conn, "NPCStats", npc_id, basic_updates,
                    f"Initial setup for NPC {npc_name}"
                )
            
            # =====================================================
            # STEP 2: GENERATE SCHEDULE (requires NPC to exist)
            # =====================================================
            schedule = npc_data.get("schedule", {})
            if not schedule:
                schedule = await self.generate_schedule(
                    ctx, npc_name, environment_desc, day_names
                )
                
                # Update the NPC with the schedule
                async with get_db_connection_context() as conn:
                    await canon.update_entity_canonically(
                        canon_ctx, conn, "NPCStats", npc_id,
                        {"schedule": json.dumps(schedule)},
                        f"Adding schedule to NPC {npc_name}"
                    )
            
            # =====================================================
            # STEP 3: DETERMINE CURRENT LOCATION (requires schedule)
            # =====================================================
            current_location = npc_data.get("current_location", "")
            if not current_location:
                # Determine current time of day and day
                async with get_db_connection_context() as conn:
                    time_query = """
                        SELECT value FROM CurrentRoleplay 
                        WHERE user_id=$1 AND conversation_id=$2 AND key='TimeOfDay'
                    """
                    
                    time_row = await conn.fetchrow(time_query, user_id, conversation_id)
                    time_of_day = time_row['value'] if time_row else "Morning"
                    
                    day_query = """
                        SELECT value FROM CurrentRoleplay 
                        WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentDay'
                    """
                    
                    day_row = await conn.fetchrow(day_query, user_id, conversation_id)
                    current_day_num = int(day_row['value']) if day_row and day_row['value'].isdigit() else 1
                
                # Calculate day index
                day_index = (current_day_num - 1) % len(day_names)
                current_day = day_names[day_index]
                
                # Extract current location from schedule
                if schedule and current_day in schedule and time_of_day in schedule[current_day]:
                    activity = schedule[current_day][time_of_day]
                    # Extract location from activity description
                    location_keywords = ["at the", "in the", "at", "in"]
                    for keyword in location_keywords:
                        if keyword in activity:
                            parts = activity.split(keyword, 1)
                            if len(parts) > 1:
                                potential_location = parts[1].split(".")[0].split(",")[0].strip()
                                if len(potential_location) > 3:
                                    current_location = potential_location
                                    break
                
                # If we couldn't extract a location, use a canonical location
                if not current_location:
                    async with get_db_connection_context() as conn:
                        current_location = await canon.find_or_create_location(
                            canon_ctx, conn, "Town Square"
                        )
                
                # Update location
                async with get_db_connection_context() as conn:
                    await canon.update_entity_canonically(
                        canon_ctx, conn, "NPCStats", npc_id,
                        {"current_location": current_location},
                        f"Setting current location for NPC {npc_name}"
                    )
            
            # =====================================================
            # STEP 4: GENERATE MEMORIES (requires NPC to exist)
            # =====================================================
            memories = npc_data.get("memories", [])
            if not memories:
                memories = await self.generate_memories(
                    ctx, npc_name, environment_desc
                )
                
                # Update the NPC with the generated memories
                async with get_db_connection_context() as conn:
                    await canon.update_entity_canonically(
                        canon_ctx, conn, "NPCStats", npc_id,
                        {"memory": json.dumps(memories)},
                        f"Adding generated memories to NPC {npc_name}"
                    )
            
            # =====================================================
            # STEP 5: ASSIGN RELATIONSHIPS (requires NPC to exist)
            # =====================================================
            try:
                await self.assign_random_relationships_dynamic(
                    user_id, conversation_id, npc_id, npc_name, archetype_objs
                )
            except Exception as e:
                logging.error(f"Error assigning relationships for NPC {npc_id}: {e}")
            
            # =====================================================
            # STEP 6: INITIALIZE MEMORY SYSTEM (requires NPC & memories)
            # =====================================================
            try:
                memory_system = await _await_logged(
                    "MemorySystem.get_instance",
                    MemorySystem.get_instance(user_id, conversation_id),
                )
            
                await _await_logged(
                    "store_npc_memories",
                    self.store_npc_memories(user_id, conversation_id, npc_id, memories),
                )
            
                await _await_logged(
                    "initialize_npc_emotional_state",
                    self.initialize_npc_emotional_state(
                        user_id,
                        conversation_id,
                        npc_id,
                        {
                            "npc_name": npc_name,
                            "dominance": dominance,
                            "cruelty": cruelty,
                            "archetype_summary": archetype_summary,
                        },
                        memories,
                    ),
                )
            
                await _await_logged(
                    "generate_npc_beliefs",
                    self.generate_npc_beliefs(
                        user_id,
                        conversation_id,
                        npc_id,
                        {
                            "npc_name": npc_name,
                            "dominance": dominance,
                            "cruelty": cruelty,
                            "archetype_summary": archetype_summary,
                        },
                    ),
                )
            
                await _await_logged(
                    "initialize_npc_memory_schemas",
                    self.initialize_npc_memory_schemas(
                        user_id,
                        conversation_id,
                        npc_id,
                        {
                            "npc_name": npc_name,
                            "dominance": dominance,
                            "archetype_summary": archetype_summary,
                        },
                    ),
                )
            
                await _await_logged(
                    "setup_npc_trauma_model",
                    self.setup_npc_trauma_model(
                        user_id,
                        conversation_id,
                        npc_id,
                        {
                            "npc_name": npc_name,
                            "dominance": dominance,
                            "cruelty": cruelty,
                            "archetype_summary": archetype_summary,
                        },
                        memories,
                    ),
                )
            
                await _await_logged(
                    "setup_npc_flashback_triggers",
                    self.setup_npc_flashback_triggers(
                        user_id,
                        conversation_id,
                        npc_id,
                        {
                            "npc_name": npc_name,
                            "dominance": dominance,
                            "archetype_summary": archetype_summary,
                        },
                    ),
                )
            
                await _await_logged(
                    "generate_counterfactual_memories",
                    self.generate_counterfactual_memories(
                        user_id,
                        conversation_id,
                        npc_id,
                        {
                            "npc_name": npc_name,
                            "dominance": dominance,
                            "archetype_summary": archetype_summary,
                        },
                    ),
                )
            
                await _await_logged(
                    "plan_mask_revelations",
                    self.plan_mask_revelations(
                        user_id,
                        conversation_id,
                        npc_id,
                        {
                            "npc_name": npc_name,
                            "dominance": dominance,
                            "cruelty": cruelty,
                            "archetype_summary": archetype_summary,
                        },
                    ),
                )
            
                await _await_logged(
                    "build_initial_semantic_network",
                    self.build_initial_semantic_network(
                        user_id,
                        conversation_id,
                        npc_id,
                        {
                            "npc_name": npc_name,
                            "archetype_summary": archetype_summary,
                        },
                    ),
                )
            
                await _await_logged(
                    "detect_memory_patterns",
                    self.detect_memory_patterns(user_id, conversation_id, npc_id),
                )
            
                await _await_logged(
                    "schedule_npc_memory_maintenance",
                    self.schedule_npc_memory_maintenance(user_id, conversation_id, npc_id),
                )
            
            except Exception as e:
                logging.error(
                    f"Error initializing memory system for NPC {npc_id}: {e}",
                    exc_info=True,
                )
            
            # =====================================================
            # STEP 7: INFORM NYX (requires NPC to exist)
            # =====================================================
            try:
                from nyx.integrate import remember_with_governance, add_joint_memory_with_governance
                
                # Notify Nyx about the new NPC using the governance memory system
                await remember_with_governance(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    entity_type="nyx",
                    entity_id=0,
                    memory_text=f"A new NPC named {npc_name} has been created. {npc_name} is {physical_description[:100]}...",
                    importance="high",
                    emotional=False,
                    tags=["npc_creation", f"npc_{npc_id}"]
                )
                
                # Create a joint memory for the NPC creation
                await add_joint_memory_with_governance(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_text=f"NPC {npc_name} has been created with {archetype_summary}",
                    source_type="system",
                    source_id=0,
                    shared_with=[
                        {"entity_type": "nyx", "entity_id": 0},
                        {"entity_type": "npc", "entity_id": npc_id}
                    ],
                    significance=7,
                    tags=["npc_creation", "system_event"],
                    metadata={
                        "npc_id": npc_id,
                        "npc_name": npc_name,
                        "archetype_summary": archetype_summary,
                        "dominance": dominance,
                        "cruelty": cruelty
                    }
                )
            except Exception as e:
                logging.error(f"Error informing Nyx about new NPC: {e}")
            
            # =====================================================
            # STEP 8: CREATE RESULT DICTIONARY
            # =====================================================
            npc_details = {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "physical_description": physical_description,
                "personality": {
                    "personality_traits": personality_traits,
                    "likes": likes,
                    "dislikes": dislikes,
                    "hobbies": hobbies
                },
                "stats": {
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "closeness": closeness,
                    "trust": trust,
                    "respect": respect,
                    "intensity": intensity
                },
                "archetypes": {
                    "archetype_names": archetype_names,
                    "archetype_summary": archetype_summary,
                    "archetype_extras_summary": archetype_extras_summary
                },
                "schedule": schedule,
                "memories": memories,
                "current_location": current_location
            }
    
            # Return the NPC details
            return npc_details
            
        except Exception as e:
            logging.error(f"Error creating NPC in database: {e}", exc_info=True)
            # Return error but don't crash
            return {"error": f"Failed to create NPC: {str(e)}", "npc_id": None}
    
    # --- Main NPC creation method ---
    
    async def create_npc(self, ctx: RunContextWrapper, archetype_names=None, physical_desc=None, starting_traits=None) -> Dict[str, Any]:
        """
        Create a new NPC with detailed characteristics.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
            archetype_names: List of archetype names to use (optional)
            physical_desc: Physical description of the NPC (optional)
            starting_traits: Initial personality traits (optional)
            
        Returns:
            Dictionary with the created NPC details
        """
        user_id = ctx.context.get("user_id")
        conversation_id = ctx.context.get("conversation_id")
        
        # Get environment details for context
        env_details = await self.get_environment_details(ctx)
        
        # Create prompt for the NPC creator
        archetypes_str = ", ".join(archetype_names) if archetype_names else "to be determined"
        physical_desc_str = physical_desc if physical_desc else "to be determined"
        traits_str = ", ".join(starting_traits) if starting_traits else "to be determined"
        
        prompt = f"""
        Create a detailed NPC for a roleplaying game set in:
        {env_details['setting_name']}: {env_details['environment_desc']}
        
        The NPC should have:
        - Suggested archetypes: {archetypes_str}
        - Physical description: {physical_desc_str}
        - Suggested personality traits: {traits_str}
        
        Create a complete, coherent NPC with appropriate:
        - Name (if not already provided)
        - Sex (default to female)
        - Physical description
        - Personality traits
        - Likes and dislikes
        - Hobbies and interests
        - Affiliations and connections
        - Stats (dominance, cruelty, etc.)
        
        The NPC should feel like a real person with complex motivations,
        while incorporating subtle elements of control and influence.
        """
        
        # Run the NPC creator
        result = await Runner.run(
            self.npc_creator,
            prompt,
            context=ctx.context
        )
        
        npc_data = result.final_output
        
        # Now create the NPC in the database
        created_npc = await self.create_npc_in_database(ctx, npc_data)
        
        return created_npc
    
    async def create_npc_with_context(self, environment_desc=None, archetype_names=None, specific_traits=None, user_id=None, conversation_id=None, db_dsn=None) -> NPCCreationResult:
            """
            Main function to create a complete NPC using the agent.
            
            Args:
                environment_desc: Description of the environment (optional)
                archetype_names: List of desired archetype names (optional)
                specific_traits: Dictionary with specific traits to incorporate (optional)
                user_id: User ID (required)
                conversation_id: Conversation ID (required)
                db_dsn: Database connection string (optional)
                
            Returns:
                NPCCreationResult object
            """
            if not user_id or not conversation_id:
                raise ValueError("user_id and conversation_id are required")
            
            # Create context
            context = NPCCreationContext(
                user_id=user_id,
                conversation_id=conversation_id,
                db_dsn=db_dsn or DB_DSN
            )
            
            ctx = RunContextWrapper(context.dict())
            
            # Get environment description if not provided
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]
            
            # Build prompt
            archetypes_str = ", ".join(archetype_names) if archetype_names else "to be determined based on setting"
            traits_str = ""
            if specific_traits:
                traits_str = "Please incorporate these specific traits:\n"
                for trait_type, traits in specific_traits.items():
                    if isinstance(traits, list):
                        traits_str += f"- {trait_type}: {', '.join(traits)}\n"
                    else:
                        traits_str += f"- {trait_type}: {traits}\n"
            
            prompt = f"""
            Create a detailed, psychologically realistic NPC for this environment:
            
            {environment_desc}
            
            Desired archetypes: {archetypes_str}
            
            {traits_str}
            
            Generate a complete NPC with:
            1. A unique name and physical description
            2. A coherent personality with traits, likes, dislikes, and hobbies
            3. Appropriate stats (dominance, cruelty, etc.)
            4. A synthesis of the desired archetypes
            5. A detailed weekly schedule
            6. Rich, diverse memories
            
            The NPC should feel like a real person with psychological depth and subtle complexity.
            For femdom-themed worlds, incorporate natural elements of control and influence
            that feel organic to the character rather than forced or explicit.
            """
            
            # Generate a name
            npc_name = await self.generate_npc_name(ctx)
            
            # Synthesize archetypes
            archetypes = await self.synthesize_archetypes(ctx, archetype_names, npc_name)
            
            # Generate a physical description
            physical_description = await self.generate_physical_description(
                ctx, npc_name, archetypes.archetype_summary, environment_desc
            )
            
            # Design personality
            personality = await self.design_personality(
                ctx, npc_name, archetypes.archetype_summary, environment_desc
            )
            
            # Calibrate stats
            stats = await self.calibrate_stats(
                ctx, npc_name, personality, archetypes.archetype_summary
            )
            
            # Create the NPC in the database
            npc_data = {
                "npc_name": npc_name,
                "physical_description": physical_description,
                "personality": personality.dict(),
                "stats": stats.dict(),
                "archetypes": archetypes.dict(),
                "environment_desc": environment_desc
            }
            
            created_npc = await self.create_npc_in_database(ctx, npc_data)
            
            # Check if there was an error during creation
            if "error" in created_npc:
                raise RuntimeError(f"Failed to create NPC in database: {created_npc['error']}")
            
            # Extract the NPC ID
            npc_id = created_npc.get("npc_id")
            
            # Ensure we have a valid NPC ID
            if not npc_id:
                raise RuntimeError("NPC creation succeeded but no ID was returned")
            
            # Generate schedule
            schedule = await self.generate_schedule(ctx, npc_name, environment_desc)
            
            # Generate memories
            memories = await self.generate_memories(ctx, npc_name, environment_desc)
            
            # Update schedule and memories in the database
            async with get_db_connection_context() as conn:
                query = """
                    UPDATE NPCStats
                    SET schedule = $1, memory = $2
                    WHERE npc_id = $3
                """
                
                await conn.execute(query, json.dumps(schedule), json.dumps(memories), npc_id)
            
            # Return the final NPC
            return NPCCreationResult(
                npc_id=npc_id,
                npc_name=npc_name,
                physical_description=physical_description,
                personality=personality,
                stats=stats,
                archetypes=archetypes,
                schedule=schedule,
                memories=memories,
                current_location=created_npc.get("current_location", "")
            )
        
    # --- Multiple NPC creation ---
    
    async def spawn_multiple_npcs(self, ctx: RunContextWrapper, count=3) -> List[int]:
        """
        Spawn multiple NPCs for the game world.
        """
        # Fix: Handle context correctly
        if hasattr(ctx, 'context') and isinstance(ctx.context, dict):
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")
        else:
            # Fallback for other context types
            user_id = getattr(ctx, 'user_id', None)
            conversation_id = getattr(ctx, 'conversation_id', None)
        
        if not user_id or not conversation_id:
            raise ValueError("Missing user_id or conversation_id in context")
        
        # Get environment description
        env_details = await self.get_environment_details(ctx)
        
        # Get day names
        day_names = await self.get_day_names(ctx)
        
        # Spawn NPCs one by one
        npc_ids = []
        for i in range(count):
            try:
                # Create context with proper structure for create_npc_with_context
                result = await self.create_npc_with_context(
                    environment_desc=env_details["environment_desc"],
                    user_id=user_id,
                    conversation_id=conversation_id
                )
                
                # Fix: Check the result properly
                if result and hasattr(result, 'npc_id') and result.npc_id is not None:
                    npc_ids.append(result.npc_id)
                    logger.info(f"Successfully created NPC {i+1}/{count} with ID {result.npc_id}")
                else:
                    logger.error(f"Failed to create NPC {i+1}/{count}: Invalid result - {result}")
                    # Continue trying to create other NPCs instead of failing completely
                    
            except Exception as e:
                logger.error(f"Error creating NPC {i+1}/{count}: {e}", exc_info=True)
                # Continue with other NPCs
                continue
            
            # Add a small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        if not npc_ids:
            raise RuntimeError(f"Failed to create any NPCs out of {count} requested")
        
        logger.info(f"Successfully created {len(npc_ids)} out of {count} requested NPCs")
        return npc_ids
    
    # --- Memory system methods ---
    
    async def store_npc_memories(
        self,
        user_id: int,
        conversation_id: int,
        npc_id: int,
        memories: list[str]
    ) -> None:
        """
        Store a batch of memories for an NPC through the canon system.
        """
        if not memories:
            return
    
        # Import canon
        from lore.core import canon
        
        # Create context
        ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": conversation_id
        })
    
        for i, memory_text in enumerate(memories):
            # Decide significance
            significance = "high" if i < 2 else "medium"
    
            # Basic tags
            tags = ["npc_creation", "initial_memory"]
            text_lower = memory_text.lower()
    
            if "childhood" in text_lower or "young" in text_lower:
                tags.append("childhood")
            if "power" in text_lower or "control" in text_lower:
                tags.append("power_dynamics")
            if "family" in text_lower or "parent" in text_lower:
                tags.append("family")
    
            # Call remember_through_nyx for governance
            result = await remember_through_nyx(
                user_id=user_id,
                conversation_id=conversation_id,
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance=significance,
                emotional=True,
                tags=tags
            )
    
            # Check for error / block
            if "error" in result:
                logging.warning(
                    f"NPC={npc_id} memory blocked by Nyx, reason: {result['error']}"
                )
                continue
    
            stored_id = result.get("memory_id")
            logging.info(
                f"NPC={npc_id} memory stored via Nyx. significance={significance}, memory_id={stored_id}"
            )
    
            # Optional post-approval steps:
            if significance == "high" and random.random() < 0.7:
                semantic_manager = SemanticMemoryManager(user_id, conversation_id)
    
                try:
                    await semantic_manager.generate_semantic_memory(
                        source_memory_id=stored_id,
                        entity_type="npc",
                        entity_id=npc_id,
                        abstraction_level=0.7
                    )
                except Exception as e:
                    logging.error(f"Error generating semantic memory for NPC {npc_id}: {e}")

    
    async def create_reciprocal_memory(
        self,
        original_memory: str,
        npc1_name: str,
        npc2_name: str,
        relationship_type: str,
        reciprocal_type: str,
        *,
        model: str = "gpt-4.1-nano",
        temperature: float = 0.7,
        max_output_tokens: int = 200,
    ) -> Optional[str]:
        """
        Use an LLM to rewrite `original_memory` (npc1's POV) into npc2's POV.
    
        Returns the rewritten memory text (string) on success, or a heuristic
        fallback transformation on error.
        """
        # --- Build prompts ----------------------------------------------------
        system_msg = (
            "You are a narrative rewriter for an RPG memory system. "
            "Given an event described from Character-A's perspective, "
            "rewrite it from Character-B's point of view. "
            "Preserve the underlying event, emotional tone, and significance. "
            "Return ONLY the rewritten memory text with no JSON, no prefix, "
            "no quotes, no explanations."
        )
    
        user_msg = f"""
    Convert this memory from {npc1_name}'s perspective to {npc2_name}'s perspective.
    
    Original memory from {npc1_name}: "{original_memory}"
    
    Relationship context:
    - {npc1_name} is {npc2_name}'s {relationship_type}.
    - {npc2_name} is {npc1_name}'s {reciprocal_type}.
    
    Rewrite this memory from {npc2_name}'s perspective, keeping the same event,
    emotional tone, and significance. Adjust pronouns appropriately.
    Return ONLY the rewritten memory text.
    """.strip()
    
        client = get_async_openai_client()
    
        # --- Call Responses API w/ retry -------------------------------------
        last_err = None
        for attempt in range(3):
            try:
                resp = await client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
    
                # Preferred convenience accessor (unifies across content parts).
                text = (resp.output_text or "").strip()
    
                if not text:
                    raise ValueError("Empty model response.")
    
                return text
    
            except Exception as e:  # pragma: no cover
                last_err = e
                logging.warning(
                    "create_reciprocal_memory: attempt %s failed: %s",
                    attempt + 1,
                    e,
                )
                # simple backoff
                await asyncio.sleep(1.5 * (attempt + 1))
    
        # --- Fallback heuristic ------------------------------------------------
        logging.error(
            "create_reciprocal_memory: all attempts failed; falling back. Last error: %s",
            last_err,
        )
        # crude perspective flip (best-effort)
        memory = original_memory.replace(npc1_name, "___OTHER___")
        memory = memory.replace(npc2_name, "___SELF___")
        memory = memory.replace("___OTHER___", npc1_name)
        # very naive pronoun shift
        memory = memory.replace("___SELF___", "I")
        return memory
    
    # --- API Methods ---
    
    async def create_npc_api(self, request):
        """
        API endpoint to create a new NPC.

        Args:
            request: NPCCreationRequest object

        Returns:
            Dict with basic NPC info
        """
        try:
            # Extract request data
            user_id = request.user_id
            conversation_id = request.conversation_id
            environment_desc = request.environment_desc
            archetype_names = request.archetype_names
            specific_traits = request.specific_traits

            # Create context wrapper
            ctx = RunContextWrapper({
                "user_id": user_id,
                "conversation_id": conversation_id
            })

            # Use existing function to create NPC
            result = await self.create_npc(
                ctx,
                archetype_names=archetype_names,
                physical_desc=None,
                starting_traits=specific_traits.get("personality_traits") if specific_traits else None
            )

            # Extract basic info for immediate return
            npc_id = result.get("npc_id")
            npc_name = result.get("npc_name")
            physical_description = result.get("physical_description", "")

            # Get personality traits from result
            personality = result.get("personality", {})
            if isinstance(personality, dict):
                personality_traits = personality.get("personality_traits", [])
            else:
                personality_traits = []

            # Get archetypes from result
            archetypes = result.get("archetypes", {})
            if isinstance(archetypes, dict):
                archetype_names = archetypes.get("archetype_names", [])
            else:
                archetype_names = []

            return {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "physical_description": (
                    physical_description[:100] + "..."
                    if len(physical_description) > 100
                    else physical_description
                ),
                "personality_traits": personality_traits,
                "archetypes": archetype_names,
                "current_location": result.get("current_location", ""),
                "message": "NPC creation in progress. Basic information is available now."
            }
        except Exception as e:
            logging.error(f"Error in create_npc_api: {e}")
            return {"error": str(e)}
    
    async def spawn_multiple_npcs_api(self, request):
        """
        API endpoint to spawn multiple NPCs.
        
        Args:
            request: MultipleNPCRequest object
            
        Returns:
            Dict with NPC IDs
        """
        try:
            # Extract request data
            user_id = request.user_id
            conversation_id = request.conversation_id
            count = request.count
            environment_desc = request.environment_desc
            
            # Create context wrapper
            ctx = RunContextWrapper({
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            # Use existing function to spawn NPCs
            npc_ids = await self.spawn_multiple_npcs(ctx, count)
            
            return {
                "npc_ids": npc_ids,
                "message": f"Successfully spawned {len(npc_ids)} NPCs."
            }
        except Exception as e:
            logging.error(f"Error in spawn_multiple_npcs_api: {e}")
            return {"error": str(e)}
    
    async def get_npc_api(self, npc_id, user_id, conversation_id):
        """
        API endpoint to get NPC details.
        
        Args:
            npc_id: NPC ID
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Dict with NPC details
        """
        try:
            # Create context wrapper
            ctx = RunContextWrapper({
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            # Use existing function to get NPC details
            result = await self.get_npc_details(ctx, npc_id=npc_id)
            
            if "error" in result:
                return result
            
            # Extract relevant info for API response
            return {
                "npc_id": result["npc_id"],
                "npc_name": result["npc_name"],
                "physical_description": result["physical_description"],
                "personality_traits": result["personality_traits"],
                "archetypes": [a.get("name", "") for a in result["archetypes"]],
                "current_location": result["current_location"]
            }
        except Exception as e:
            logging.error(f"Error in get_npc_api: {e}")
            return {"error": str(e)}

    async def check_for_mask_slippage(
        self,
        user_id: int,
        conversation_id: int,
        npc_id: int,
        *,
        overwrite_description: bool = True,
    ) -> list[dict] | None:
        """
        Evaluate whether the NPC's *facade* cracks for the **player** based on the
        *current* values of dominance / cruelty / intensity.
    
        • Uses the environment‑aware templates supplied by `dynamic_templates`.
        • Writes *cue* events to **NPCEvolution.mask_slippage_events** so later
          logic (e.g. dialogue colouring) can react.
        • No longer creates canned memories – mask slippage is a *criterion*, not
          a retrospective thought.
        • Optionally appends a short physical tell to `physical_description`
          the first time each cue fires (can be disabled).
    
        Returns
        -------
        list[dict] | None
            The list of newly triggered cue dictionaries, or `None` on error.
        """
        try:
            from lore.core import canon
    
            # ── gather context ──────────────────────────────────────────────────
            ctx = type("CanonCtx", (), {"user_id": user_id, "conversation_id": conversation_id})()
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT npc_name, dominance, cruelty, intensity, memory,
                           (SELECT value FROM CurrentRoleplay
                            WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc') AS env
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    """,
                    user_id,
                    conversation_id,
                    npc_id,
                )
                if not row:
                    return None
    
                npc_name = row["npc_name"]
                env_desc = row["env"] or ""
                
                # FIX: Handle None values with defaults, but preserve 0 as valid
                stats_current = {
                    "dominance": row["dominance"] if row["dominance"] is not None else 50,
                    "cruelty": row["cruelty"] if row["cruelty"] is not None else 30,
                    "intensity": row["intensity"] if row["intensity"] is not None else 40,
                }
    
            # ── fetch *dynamic* trigger tables ──────────────────────────────────
            trigger_map: dict[str, list[dict]] = {}
            for stat in stats_current:
                trigger_map[stat] = await get_mask_slippage_triggers(stat, env_desc)
    
            # ── load prior history ──────────────────────────────────────────────
            async with get_db_connection_context() as conn:
                hist_row = await conn.fetchrow(
                    """
                    SELECT mask_slippage_events
                    FROM NPCEvolution
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    """,
                    user_id,
                    conversation_id,
                    npc_id,
                )
                history: list[dict] = (
                    json.loads(hist_row["mask_slippage_events"])
                    if hist_row and hist_row["mask_slippage_events"]
                    else []
                )
    
            history_cues = {h["cue"] for h in history}
            newly_triggered: list[dict] = []
    
            # ── evaluate each stat's ladder ─────────────────────────────────────
            for stat, value in stats_current.items():
                # Additional safety check (though shouldn't be needed with the fix above)
                if value is None:
                    logging.warning(f"Stat {stat} is None for NPC {npc_id}, skipping")
                    continue
                    
                for step in trigger_map[stat]:
                    cue = step["cue"]
                    if cue in history_cues:
                        continue                       # already fired
    
                    if value >= step["threshold"]:
                        newly_triggered.append(
                            {
                                "cue": cue,
                                "stat": stat,
                                "threshold": step["threshold"],
                                "description": step["description"],
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                        history_cues.add(cue)
    
            if not newly_triggered:
                return []                              # nothing new this tick
    
            # ── persist to NPCEvolution ─────────────────────────────────────────
            history.extend(newly_triggered)
    
            async with get_db_connection_context() as conn:
                await canon.update_entity_canonically(
                    ctx,
                    conn,
                    "NPCEvolution",
                    npc_id,
                    {"mask_slippage_events": json.dumps(history)},
                    f"Mask‑slippage cues fired for {npc_name}: {', '.join(e['cue'] for e in newly_triggered)}",
                )
    
                # ── optionally tweak physical description the first time a cue fires
                if overwrite_description:
                    desc_row = await conn.fetchrow(
                        """
                        SELECT physical_description
                        FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                        """,
                        user_id,
                        conversation_id,
                        npc_id,
                    )
                    current_desc = desc_row["physical_description"] if desc_row else ""
                    # only add snippets that are brand‑new
                    additions = [
                        f" {e['description']}" for e in newly_triggered
                        if e["description"] and e["description"] not in current_desc
                    ]
                    if additions:
                        await canon.update_entity_canonically(
                            ctx,
                            conn,
                            "NPCStats",
                            npc_id,
                            {"physical_description": current_desc + "".join(additions)},
                            f"Physical tells added after mask slippage for {npc_name}",
                        )
    
            return newly_triggered
    
        except Exception as err:
            logging.error("check_for_mask_slippage failed (NPC %s): %s", npc_id, err, exc_info=True)
            return None
