# nyx/nyx_agent/feasibility.py
"""
Dynamic feasibility system that learns what's possible/impossible in each unique setting.
Maintains reality consistency without hard-coded rules.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from agents import Agent, Runner
from nyx.nyx_agent.context import NyxContext
from db.connection import get_db_connection_context
from logic.action_parser import parse_action_intents

# Enhanced feasibility agent with dynamic reasoning
FEASIBILITY_AGENT = Agent(
    name="FeasibilityChecker",
    instructions="""
    You are a reality consistency enforcer. Analyze if actions are possible given the setting's established rules.
    
    CRITICAL: You must maintain internal consistency. What's been established as impossible STAYS impossible.
    
    For each intent, consider:
    1. Does this violate established physics/reality rules of THIS setting?
    2. Has this type of action been previously established as possible/impossible?
    3. Does the player have the means/ability to perform this action?
    4. Is the target present and accessible in the current scene?
    5. Would this break narrative consistency?
    6. Does the player's current state allow this action?
    
    Common reality violations to check:
    - Creating matter/energy from nothing (unless setting has established this)
    - Accessing things not present in the scene
    - Violating distance/time constraints  
    - Using abilities the character doesn't possess
    - Contradicting established setting rules
    - Affecting targets not physically present
    - Breaking cause-and-effect chains
    
    BE STRICT. If something would break the established reality, DENY it.
    For denied actions, provide immersive narrator guidance that shows reality itself rejecting the attempt.
    
    Output ONLY JSON:
      {"overall":{"feasible":bool,"strategy":"allow|deny|reinterpret"},
       "per_intent":[
         {"feasible":bool,"strategy":"allow|deny|reinterpret",
          "violations":[{"rule":"...", "reason":"..."}],
          "narrator_guidance":"[Immersive description of how reality prevents this]",
          "suggested_alternatives":["realistic alternative 1", "realistic alternative 2"],
          "reality_response":"[How the world itself responds - e.g., 'The air shimmers and resists']",
          "categories":["..."]}
       ]}
    """,
    model="gpt-5-nano"  # Use smarter model for better reasoning
)

# Setting detective agent for auto-detecting setting type
SETTING_DETECTIVE_AGENT = Agent(
    name="SettingDetective",
    instructions="""
    Analyze the established narrative elements to determine the setting type.
    
    Consider:
    - Technology level (medieval, modern, futuristic, etc.)
    - Presence of magic or supernatural elements
    - Physics model (realistic, soft sci-fi, fantasy, surreal)
    - Genre markers (noir, cyberpunk, high fantasy, etc.)
    - Established world rules and limitations
    
    Output one of:
    - "realistic_modern": Contemporary real world, normal physics
    - "realistic_historical": Historical period, normal physics  
    - "soft_scifi": Some tech advances, mostly normal physics
    - "hard_scifi": Advanced tech, scientific physics
    - "urban_fantasy": Modern world with hidden magic
    - "high_fantasy": Magic is common and known
    - "surreal": Dream-like, inconsistent physics
    - "horror": Dark, possibly supernatural elements
    - "cyberpunk": High tech, low life, neural interfaces
    - "post_apocalyptic": After societal collapse
    - "custom": Unique rules (specify in details)
    
    Also output confidence level (0.0-1.0) and key indicators.
    
    Output JSON:
    {"setting_type": "...", "confidence": 0.X, "indicators": ["...", "..."], "details": "..."}
    """,
    model="gpt-5-nano"
)

async def assess_action_feasibility(nyx_ctx: NyxContext, user_input: str) -> Dict[str, Any]:
    """
    Dynamically assess if an action is feasible in the current setting context.
    Learns from previous denials to maintain consistency.
    """
    # Parse the intended actions
    intents = await parse_action_intents(user_input)
    
    # Load comprehensive setting context including impossibilities
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
        result = _enforce_reality_consistency(result, setting_context, intents)
        
        # Add immersive rejection narratives if needed
        result = _enhance_rejection_narratives(result, setting_context)
        
        return result
    except Exception as e:
        # Fallback to safe defaults
        return _default_feasibility_response(intents)

async def _load_comprehensive_context(nyx_ctx: NyxContext) -> Dict[str, Any]:
    """Load all relevant context about what's possible in this setting"""
    
    context = {
        "type": "unknown",
        "established_rules": [],
        "available_items": [],
        "present_entities": [],
        "character_abilities": [],
        "character_state": {},
        "physics_model": "realistic",
        "magic_system": None,
        "technology_level": "contemporary",
        "location": {},
        "established_impossibilities": [],
        "established_possibilities": [],
        "narrative_history": []
    }
    
    async with get_db_connection_context() as conn:
        # Get setting type
        setting_type = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='SettingType'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if setting_type:
            context["type"] = setting_type
        else:
            # Auto-detect if not set
            context["type"] = await detect_setting_type(nyx_ctx)
            
        # Get physics model
        physics = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='PhysicsModel'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["physics_model"] = physics or "realistic"
            
        # Get established impossibilities (what we've learned can't happen)
        impossibilities = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedImpossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if impossibilities:
            context["established_impossibilities"] = json.loads(impossibilities)
            
        # Get established possibilities (what we've learned CAN happen)
        possibilities = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedPossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if possibilities:
            context["established_possibilities"] = json.loads(possibilities)
            
        # Get current scene state
        scene = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentScene'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if scene and scene["value"]:
            scene_data = json.loads(scene["value"])
            context["location"] = scene_data.get("location", {})
            context["available_items"] = scene_data.get("items", [])
            context["present_entities"] = scene_data.get("npcs", [])
            
        # Get player's current location
        location = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentLocation'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if location:
            context["location"]["name"] = location
            
        # Get character abilities and state
        player_stats = await conn.fetchrow("""
            SELECT * FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2
            LIMIT 1
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if player_stats:
            context["character_state"] = dict(player_stats)
            # Extract abilities if stored
            if "abilities" in player_stats:
                context["character_abilities"] = json.loads(player_stats["abilities"] or "[]")
                
        # Get inventory
        inventory = await conn.fetchval("""
            SELECT items FROM Inventory
            WHERE user_id=$1 AND conversation_id=$2
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if inventory:
            context["available_items"].extend(json.loads(inventory or "[]"))
            
        # Get active NPCs
        npcs = await conn.fetch("""
            SELECT npc_name, current_location 
            FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        for npc in npcs:
            if npc["current_location"] == context["location"].get("name"):
                context["present_entities"].append(npc["npc_name"])
                
        # Get narrative rules
        rules = await conn.fetch("""
            SELECT rule_name, condition, effect
            FROM GameRules
            WHERE user_id=$1 AND conversation_id=$2 AND enabled=TRUE
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["established_rules"] = [
            {"name": r["rule_name"], "condition": r["condition"], "effect": r["effect"]}
            for r in rules
        ]
        
        # Get recent narrative for context
        recent = await conn.fetch("""
            SELECT narrative_text FROM StoryEvents
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY created_at DESC LIMIT 5
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["narrative_history"] = [r["narrative_text"] for r in recent if r["narrative_text"]]
        
    return context

def _enforce_reality_consistency(result: Dict[str, Any], context: Dict[str, Any], intents: List[Dict]) -> Dict[str, Any]:
    """Post-process to ensure previously impossible things stay impossible"""
    
    # Check against established impossibilities
    for i, intent in enumerate(result.get("per_intent", [])):
        # Get intent categories
        intent_categories = set(intents[i].get("categories", []) if i < len(intents) else [])
        
        # Check established impossibilities
        for impossibility in context.get("established_impossibilities", []):
            if _matches_impossibility(intent, impossibility, intent_categories):
                intent["feasible"] = False
                intent["strategy"] = "deny"
                if "violations" not in intent:
                    intent["violations"] = []
                intent["violations"].append({
                    "rule": "established_impossibility",
                    "reason": f"This has been established as impossible: {impossibility['reason']}"
                })
                
        # Check if action requires something not present
        if not _check_prerequisites(intent, context):
            intent["feasible"] = False
            intent["strategy"] = "deny"
            if "violations" not in intent:
                intent["violations"] = []
            intent["violations"].append({
                "rule": "missing_prerequisites",
                "reason": "Required elements are not present in the scene"
            })
    
    # Recalculate overall feasibility
    if result.get("per_intent"):
        all_feasible = all(i.get("feasible", False) for i in result["per_intent"])
        result["overall"]["feasible"] = all_feasible
        if not all_feasible:
            result["overall"]["strategy"] = "deny"
    
    return result

def _enhance_rejection_narratives(result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Add immersive rejection narratives that fit the setting"""
    
    setting_type = context.get("type", "realistic_modern")
    
    for intent in result.get("per_intent", []):
        if not intent.get("feasible"):
            # Generate setting-appropriate rejection
            if not intent.get("reality_response"):
                intent["reality_response"] = _get_reality_response(setting_type, intent)
            
            if not intent.get("narrator_guidance"):
                intent["narrator_guidance"] = _get_narrator_guidance(setting_type, intent)
                
            # Add alternatives if not present
            if not intent.get("suggested_alternatives"):
                intent["suggested_alternatives"] = _generate_alternatives(context, intent)
    
    return result

def _matches_impossibility(intent: Dict, impossibility: Dict, intent_categories: set) -> bool:
    """Check if an intent matches a recorded impossibility"""
    
    # Check category overlap
    imp_categories = set(impossibility.get("categories", []))
    if intent_categories & imp_categories:
        return True
        
    # Check action similarity
    if "action" in impossibility:
        intent_text = json.dumps(intent).lower()
        if impossibility["action"].lower() in intent_text:
            return True
            
    return False

def _check_prerequisites(intent: Dict, context: Dict) -> bool:
    """Check if required elements for the action are present"""
    
    # Check if required items are available
    if "instruments" in intent:
        for item in intent["instruments"]:
            if item and item not in context.get("available_items", []):
                return False
                
    # Check if targets are present
    if "direct_object" in intent:
        for target in intent["direct_object"]:
            if target and target not in context.get("present_entities", []):
                # Check if it's a location target
                if target != context.get("location", {}).get("name"):
                    return False
                    
    return True

def _get_reality_response(setting_type: str, intent: Dict) -> str:
    """Get setting-appropriate reality response"""
    
    responses = {
        "realistic_modern": [
            "The laws of physics assert themselves",
            "Reality remains stubbornly mundane",
            "The world refuses to bend to imagination"
        ],
        "high_fantasy": [
            "The weave of magic recoils from your intent",
            "The arcane energies refuse to coalesce",
            "The mystical forces remain inert"
        ],
        "hard_scifi": [
            "The conservation laws remain inviolate",
            "Quantum probability collapses against you",
            "The physics simulation rejects the parameters"
        ],
        "surreal": [
            "Even dreams have their own logic",
            "The narrative threads tangle and resist",
            "The story refuses this particular madness"
        ]
    }
    
    import random
    response_list = responses.get(setting_type, responses["realistic_modern"])
    return random.choice(response_list)

def _get_narrator_guidance(setting_type: str, intent: Dict) -> str:
    """Generate narrator guidance for the rejection"""
    
    violations = intent.get("violations", [])
    if violations:
        reason = violations[0].get("reason", "impossible")
    else:
        reason = "impossible in this reality"
        
    return f"You reach for the impossible, but {reason}. The moment passes, leaving only what is real."

def _generate_alternatives(context: Dict, intent: Dict) -> List[str]:
    """Generate realistic alternatives based on what IS available"""
    
    alternatives = []
    
    # Suggest using available items
    if context.get("available_items"):
        alternatives.append(f"use the {context['available_items'][0]} instead")
        
    # Suggest interacting with present entities
    if context.get("present_entities"):
        alternatives.append(f"speak with {context['present_entities'][0]}")
        
    # Generic alternatives
    alternatives.extend([
        "look for another way",
        "try a more conventional approach",
        "work within the constraints of reality"
    ])
    
    return alternatives[:3]  # Return top 3

def _default_feasibility_response(intents: List[Dict]) -> Dict[str, Any]:
    """Default response when feasibility check fails"""
    
    # Allow only mundane actions by default
    mundane = all(
        "mundane_action" in i.get("categories", []) or 
        "dialogue" in i.get("categories", []) or
        "movement" in i.get("categories", [])
        for i in intents
    )
    
    return {
        "overall": {
            "feasible": mundane,
            "strategy": "allow" if mundane else "deny"
        },
        "per_intent": [
            {
                "feasible": mundane,
                "strategy": "allow" if mundane else "deny",
                "violations": [] if mundane else [{"rule": "unknown", "reason": "Cannot verify feasibility"}]
            }
            for _ in intents
        ]
    }

async def record_impossibility(nyx_ctx: NyxContext, action: str, reason: str):
    """Record that something has been established as impossible in this setting"""
    
    async with get_db_connection_context() as conn:
        # Get existing impossibilities
        current = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedImpossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        impossibilities = json.loads(current) if current else []
        
        # Extract categories for this action
        categories = await _extract_action_categories(action)
        
        # Check if similar impossibility already exists
        exists = any(
            imp["action"] == action or 
            set(imp.get("categories", [])) & set(categories)
            for imp in impossibilities
        )
        
        if not exists:
            # Add new impossibility
            impossibilities.append({
                "action": action,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "categories": categories
            })
            
            # Keep only recent impossibilities (last 100)
            impossibilities = impossibilities[-100:]
            
            # Store updated list
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'EstablishedImpossibilities', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, json.dumps(impossibilities))

async def record_possibility(nyx_ctx: NyxContext, action: str, categories: List[str]):
    """Record that something has been established as possible in this setting"""
    
    async with get_db_connection_context() as conn:
        # Get existing possibilities
        current = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedPossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        possibilities = json.loads(current) if current else []
        
        # Add if not already recorded
        exists = any(
            p["action"] == action or 
            set(p.get("categories", [])) & set(categories)
            for p in possibilities
        )
        
        if not exists:
            possibilities.append({
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "categories": categories
            })
            
            # Keep only recent (last 100)
            possibilities = possibilities[-100:]
            
            # Store updated list
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'EstablishedPossibilities', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, json.dumps(possibilities))

async def detect_setting_type(nyx_ctx: NyxContext) -> str:
    """Intelligently detect what kind of setting this is based on established elements"""
    
    # Gather narrative context
    context = {"recent_narrative": [], "elements": {}}
    
    async with get_db_connection_context() as conn:
        # Get recent narrative
        recent_events = await conn.fetch("""
            SELECT narrative_text FROM StoryEvents
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY created_at DESC LIMIT 10
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["recent_narrative"] = [
            e["narrative_text"] for e in recent_events 
            if e["narrative_text"]
        ]
        
        # Get established elements
        elements = await conn.fetch("""
            SELECT key, value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 
            AND key IN ('Technology', 'Magic', 'Location', 'TimePeriod')
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        for elem in elements:
            context["elements"][elem["key"]] = elem["value"]
    
    # Run detection
    run = await Runner.run(SETTING_DETECTIVE_AGENT, json.dumps(context))
    
    try:
        result = json.loads(getattr(run, "final_output", "{}"))
        setting_type = result.get("setting_type", "realistic_modern")
        
        # Store the detected type
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'SettingType', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, setting_type)
            
            # Also store confidence and indicators
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'SettingTypeConfidence', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, str(result.get("confidence", 0.5)))
        
        return setting_type
        
    except Exception:
        return "realistic_modern"  # Safe default

async def _extract_action_categories(action: str) -> List[str]:
    """Extract categories from an action string for impossibility tracking"""
    try:
        intents = await parse_action_intents(action)
        categories = set()
        for intent in intents:
            categories.update(intent.get("categories", []))
        return list(categories)
    except Exception:
        return []

# Fast context-free feasibility check for pre-gate
async def assess_action_feasibility_fast(user_id: int, conversation_id: int, text: str) -> Dict[str, Any]:
    """Quick feasibility check without full context initialization"""
    
    # Parse intents
    intents = await parse_action_intents(text)
    
    # Quick load of critical context only
    async with get_db_connection_context() as conn:
        # Get setting type
        setting_type = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='SettingType'
        """, user_id, conversation_id) or "realistic_modern"
        
        # Get impossibilities
        imp_raw = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedImpossibilities'
        """, user_id, conversation_id)
        
        impossibilities = json.loads(imp_raw) if imp_raw else []
    
    # Quick check for obvious violations
    for intent in intents:
        categories = set(intent.get("categories", []))
        
        # Check established impossibilities
        for imp in impossibilities:
            if set(imp.get("categories", [])) & categories:
                return {
                    "overall": {"feasible": False, "strategy": "deny"},
                    "per_intent": [{
                        "feasible": False,
                        "strategy": "deny",
                        "violations": [{"rule": "established", "reason": imp["reason"]}],
                        "narrator_guidance": "Reality resists your attempt.",
                        "suggested_alternatives": ["Try something more grounded"]
                    }]
                }
        
        # Quick rules for realistic settings
        if setting_type in ["realistic_modern", "realistic_historical"]:
            forbidden = {
                "dimensional_portal", "extradimensional_access",
                "ex_nihilo_conjuration", "weapon_conjuration",
                "spellcasting", "ritual_magic", "psionics",
                "physics_violation", "reality_warping",
                "unaided_flight", "teleportation"
            }
            
            if categories & forbidden:
                return {
                    "overall": {"feasible": False, "strategy": "deny"},
                    "per_intent": [{
                        "feasible": False,
                        "strategy": "deny",
                        "violations": [{"rule": "physics", "reason": "Violates realistic physics"}],
                        "narrator_guidance": "The laws of reality remain firm.",
                        "suggested_alternatives": ["Work within realistic constraints"]
                    }]
                }
    
    # Default allow for non-violating actions
    return {
        "overall": {"feasible": True, "strategy": "allow"},
        "per_intent": [{"feasible": True, "strategy": "allow"} for _ in intents]
    }
