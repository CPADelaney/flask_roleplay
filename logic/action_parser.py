# logic/action_parser.py
import json
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field, ValidationError
from agents import Agent, Runner

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
    # If your Runner supports it, this guarantees a JSON object string:
    response_format={"type": "json_object"},
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
    # Ask Runner to enforce JSON if it supports response_format; otherwise it’s ignored harmlessly.
    run = await Runner.run(INTENT_AGENT, user_input, response_format={"type": "json_object"})
    # Pull the text payload safely (Runner implementations vary)
    text = getattr(run, "final_output", None)
    if not text:
        # try common shapes
        text = getattr(getattr(run, "output", None), "text", None) \
               or getattr(getattr(run, "output", None), "content", None) \
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
                continue
        return intents
    except Exception as e:
        # Let callers/logs see a real message; do NOT swallow silently.
        raise ValueError(f"Failed to parse ActionIntent JSON: {e}") from e
