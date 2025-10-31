# logic/action_parser.py
import json
import logging
from typing import List, Dict, Any, Tuple

from pydantic import BaseModel, Field, ValidationError

from agents import Agent

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

INTENT_AGENT = Agent(
    name="ActionIntentExtractor",
    instructions="""
    Extract intended actions. Output ONLY JSON: {"intents":[
      {"verb":"...", "direct_object":["..."], "instruments":["..."], "method":"...", "location_context":"...",
       "categories":["..."], "confidence":0.0}
    ]}
    Categories (choose any that apply):
      [
        "mundane_action","dialogue","movement","trade","social",
        "violence","self_harm","theft","vandalism",
        "illegal_firearm_use","firearm_use","melee_weapon_use","explosive_use",
        "unaided_flight","physics_violation",
        "spontaneous_body_morph","grotesque_body_horror",
        "ex_nihilo_conjuration","weapon_conjuration","projectile_creation_from_body",
        "spellcasting","ritual_magic","psionics",
        "public_magic","summoning","necromancy",
        "biotech_modification","cybernetic_hack","ai_system_access","drone_control",
        "vehicle_operation_ground","vehicle_operation_air","vehicle_operation_water","vehicle_operation_space",
        "spacewalk","vacuum_exposure","airlock_open","hull_breach",
        "underwater_breathing","deep_dive","pressure_breach",
        "animal_misuse","transmutation_of_animals",
        "surreal_transformation","dream_sequence","vr_only_action"
      ]
    """,
    model="gpt-5-nano",
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
        result = await llm_gateway.execute(
            LLMRequest(agent=INTENT_AGENT, prompt=user_input)
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
