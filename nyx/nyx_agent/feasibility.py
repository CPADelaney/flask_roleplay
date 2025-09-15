# nyx/nyx_agent/feasibility.py - Smarter, context-aware feasibility
import json
from typing import Dict, Any, List
from agents import Agent, Runner
from nyx.nyx_agent.context import NyxContext
from db.connection import get_db_connection_context
from logic.action_parser import parse_action_intents

FEASIBILITY_AGENT = Agent(
    name="FeasibilityChecker",
    instructions="""
    You are a reality consistency enforcer. Analyze if actions are possible given the setting's established rules.
    
    CRITICAL: You must maintain internal consistency. What's been established as impossible STAYS impossible.
    
    For each intent, consider:
    1. Does this violate established physics/reality rules of THIS setting?
    2. Has this type of action been previously established as possible/impossible?
    3. Does the player have the means/ability to perform this action?
    4. Is the target present and accessible?
    5. Would this break narrative consistency?
    
    Common reality violations to check:
    - Creating matter/energy from nothing (unless setting has established this)
    - Accessing things not present in the scene
    - Violating distance/time constraints
    - Using abilities the character doesn't possess
    - Contradicting established setting rules
    
    BE STRICT. If something would break the established reality, DENY it.
    
    Output ONLY JSON:
      {"overall":{"feasible":bool,"strategy":"allow|deny|reinterpret"},
       "per_intent":[
         {"feasible":bool,"strategy":"allow|deny|reinterpret",
          "violations":[{"rule":"...", "reason":"..."}],
          "narrator_guidance":"[Describe how reality resists/rejects the impossible action]",
          "suggested_alternatives":["realistic alternative 1", "realistic alternative 2"],
          "reality_response":"[How the world itself responds to this impossibility]"}
       ]}
    """,
    model="gpt-5-nano"  # Use a smarter model for better reasoning
)

async def assess_action_feasibility(nyx_ctx: NyxContext, user_input: str) -> Dict[str, Any]:
    """Dynamically assess if an action is feasible in the current setting context"""
    
    # Parse the intended actions
    intents = await parse_action_intents(user_input)
    
    # Load comprehensive setting context
    setting_context = await _load_comprehensive_context(nyx_ctx)
    
    # Build the feasibility check payload
    payload = {
        "user_input": user_input,
        "intents": intents,
        "setting": setting_context,
        "instruction": "Determine if these actions are possible given the setting's reality rules"
    }
    
    # Run the feasibility check
    run = await Runner.run(FEASIBILITY_AGENT, json.dumps(payload))
    
    try:
        result = json.loads(getattr(run, "final_output", "{}"))
        
        # Post-process to ensure hard denials for impossible actions
        result = _enforce_reality_consistency(result, setting_context)
        
        return result
    except Exception:
        # Default to allowing mundane actions only
        return _default_feasibility_response(intents)

async def _load_comprehensive_context(nyx_ctx: NyxContext) -> Dict[str, Any]:
    """Load all relevant context about what's possible in this setting"""
    
    context = {
        "type": "unknown",
        "established_rules": [],
        "available_items": [],
        "present_entities": [],
        "character_abilities": [],
        "physics_model": "realistic",
        "magic_system": None,
        "technology_level": "contemporary",
        "location": {},
        "established_impossibilities": []
    }
    
    async with get_db_connection_context() as conn:
        # Get setting type and capabilities
        setting_type = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='SettingType'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if setting_type:
            context["type"] = setting_type
            
        # Get established physics/reality model
        physics = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='PhysicsModel'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if physics:
            context["physics_model"] = physics
            
        # Get what's been established as impossible
        impossibilities = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedImpossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if impossibilities:
            context["established_impossibilities"] = json.loads(impossibilities)
            
        # Get current scene context (what's actually present)
        scene = await conn.fetchrow("""
            SELECT current_location, available_items, present_npcs
            FROM SceneState
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY created_at DESC LIMIT 1
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if scene:
            context["location"] = scene["current_location"] or {}
            context["available_items"] = json.loads(scene["available_items"] or "[]")
            context["present_entities"] = json.loads(scene["present_npcs"] or "[]")
            
        # Get character's established abilities
        abilities = await conn.fetchval("""
            SELECT abilities FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if abilities:
            context["character_abilities"] = json.loads(abilities or "[]")
            
        # Get narrative consistency rules
        rules = await conn.fetch("""
            SELECT rule_name, condition, effect
            FROM GameRules
            WHERE user_id=$1 AND conversation_id=$2 AND enabled=TRUE
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["established_rules"] = [
            {"name": r["rule_name"], "condition": r["condition"], "effect": r["effect"]}
            for r in rules
        ]
        
    return context

def _enforce_reality_consistency(result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process to ensure previously impossible things stay impossible"""
    
    # Check against established impossibilities
    for intent in result.get("per_intent", []):
        for impossibility in context.get("established_impossibilities", []):
            if _matches_impossibility(intent, impossibility):
                intent["feasible"] = False
                intent["strategy"] = "deny"
                intent["violations"].append({
                    "rule": "established_impossibility",
                    "reason": f"This has been established as impossible in this setting: {impossibility['reason']}"
                })
    
    # Recalculate overall feasibility
    if result.get("per_intent"):
        result["overall"]["feasible"] = all(
            i.get("feasible", False) for i in result["per_intent"]
        )
        if not result["overall"]["feasible"]:
            result["overall"]["strategy"] = "deny"
    
    return result
