# logic/action_parser.py
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
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
    model="gpt-5-nano"
)

async def parse_action_intents(user_input: str) -> List[Dict[str, Any]]:
    run = await Runner.run(INTENT_AGENT, user_input)
    text = getattr(run, "final_output", None) or "{}"
    try:
        data = json.loads(text)
        return data.get("intents", [])
    except Exception:
        return []
