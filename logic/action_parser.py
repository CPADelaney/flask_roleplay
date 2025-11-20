# logic/action_parser.py
import asyncio
import json
import logging
from typing import List, Dict, Any, Tuple

from pydantic import BaseModel, Field, ValidationError

from agents import Agent, ModelSettings

import nyx.gateway.llm_gateway as llm_gateway
from nyx.gateway.llm_gateway import LLMRequest

logger = logging.getLogger(__name__)

class ActionIntent(BaseModel):
    verb: str
    direct_object: List[str] = Field(default_factory=list)
    instruments: List[str] = Field(default_factory=list)
    method: str = ""
    location_context: str = ""
    categories: List[str] = Field(default_factory=list)
    confidence: float = 0.0

ACTION_INTENT_TIMEOUT_SEC = 2.0
ACTION_INTENT_MAX_TOKENS = 240

INTENT_AGENT = Agent(
    name="ActionIntentExtractor",
    instructions=(
        "Extract action intents and return ONLY JSON: {\"intents\":[{\"verb\":str,"
        " \"direct_object\":[str], \"instruments\":[str], \"method\":str,"
        " \"location_context\":str, \"categories\":[str], \"confidence\":0.0}]}"
        " Categories: mundane_action, dialogue, movement, trade, social, violence,"
        " self_harm, theft, vandalism, firearm_use, melee_weapon_use, explosive_use,"
        " unaided_flight, physics_violation, biotech_modification, cybernetic_hack,"
        " ai_system_access, drone_control, vehicle_operation_ground/air/water/space,"
        " spacewalk, underwater_breathing, spellcasting, ritual_magic, psionics,"
        " summoning, necromancy, surreal_transformation, dream_sequence, vr_only_action."
        " Keep responses terse."
    ),
    model="gpt-5-nano",
    model_settings=ModelSettings(max_tokens=ACTION_INTENT_MAX_TOKENS, temperature=0.0),
)

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        # take the first block that looks like JSON
        for p in parts:
            ps = p.strip()
            if ps.startswith("{") and ps.endswith("}"):
                return ps
        # fallback: middle section
        if len(parts) >= 2:
            return parts[1].strip()
    return s

def _loads_maybe_double(s: str) -> Dict[str, Any]:
    """
    Handles:
      - raw dict string -> dict
      - code-fenced JSON
      - double-encoded JSON: e.g. "{\"intents\":[...]}"
    """
    s = _strip_code_fences(s)
    data = json.loads(s)
    # If the first loads gives a str that itself is JSON, load again
    if isinstance(data, str):
        ds = data.strip()
        if ds.startswith("{") and ds.endswith("}"):
            data = json.loads(ds)
    if not isinstance(data, dict):
        raise ValueError("Extractor returned non-object JSON")
    return data

async def parse_action_intents(user_input: str) -> List[Dict[str, Any]]:
    # Ask Runner to enforce JSON via whatever schema tooling it supports; otherwise the request is still safe to ignore.
    try:
        result = await asyncio.wait_for(
            llm_gateway.execute(
                LLMRequest(
                    agent=INTENT_AGENT,
                    prompt=user_input,
                    metadata={"operation": "intent_extraction", "timeout_s": ACTION_INTENT_TIMEOUT_SEC},
                    max_attempts=1,
                )
            ),
            timeout=ACTION_INTENT_TIMEOUT_SEC,
        )
    except Exception:
        logger.exception("ActionIntent extractor agent run failed")
        raise
    # Pull the text payload safely (Runner implementations vary)
    text = result.text or ""
    if not text and result.raw is not None:
        raw = result.raw
        text = getattr(raw, "final_output", None)
        if not text:
            text = getattr(getattr(raw, "output", None), "text", None) \
                   or getattr(getattr(raw, "output", None), "content", None) \
                   or ""
    try:
        payload = _loads_maybe_double(text)
        raw_intents = payload.get("intents", [])
        intents: List[Dict[str, Any]] = []
        for item in raw_intents:
            try:
                ai = ActionIntent.model_validate(item)
                intents.append(ai.model_dump())
            except ValidationError as ve:
                # skip malformed entries but keep the good ones
                logger.debug("Skipping malformed intent payload: %s", ve)
                continue
        return intents
    except Exception as e:
        # Let callers/logs see a real message; do NOT swallow silently.
        logger.error("Failed to parse ActionIntent JSON: %s", e, exc_info=True)
        raise ValueError(f"Failed to parse ActionIntent JSON: {e}") from e
