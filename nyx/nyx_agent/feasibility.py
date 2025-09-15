# nyx/nyx_agent/feasibility.py
"""
Dynamic feasibility system that learns what's possible/impossible in each unique setting.
Maintains reality consistency without hard-coded rules or repetitive responses.
"""

import json
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from agents import Agent, Runner
from nyx.nyx_agent.context import NyxContext
from db.connection import get_db_connection_context
from logic.action_parser import parse_action_intents

import logging
logger = logging.getLogger(__name__)

# Dynamic rejection narrator for unique, contextual rejections
REJECTION_NARRATOR_AGENT = Agent(
    name="RejectionNarrator",
    instructions="""
    You are the voice of reality itself, explaining why certain actions cannot occur.
    
    Given the setting context and attempted action, create a UNIQUE, immersive rejection that:
    1. Fits the setting's tone and genre perfectly
    2. Never repeats previous rejections (check rejection_history)
    3. Feels like the world itself is responding, not a game system
    4. Maintains the narrative flow without breaking immersion
    5. Subtly guides toward what IS possible
    
    Consider:
    - Setting atmosphere and mood from environment_desc
    - The specific physics/magic rules of this world
    - Recent narrative events for continuity
    - The exact nature of what was attempted
    - Previous rejections to avoid repetition
    - Current scene details (NPCs, items, location)
    
    Generate THREE elements:
    1. reality_response: A visceral, sensory description of reality resisting (1-2 sentences)
    2. narrator_guidance: Poetic explanation of why this cannot be (2-3 sentences)  
    3. suggested_alternatives: 2-3 contextual alternatives that fit the current scene
    4. metaphor: A unique metaphor for this specific rejection
    
    Make each rejection feel unique to THIS moment in THIS world.
    
    Output JSON:
    {
        "reality_response": "...",
        "narrator_guidance": "...",
        "suggested_alternatives": ["...", "...", "..."],
        "metaphor": "..."
    }
    """,
    model="gpt-5-nano"
)

# Alternative suggestion generator
ALTERNATIVE_GENERATOR_AGENT = Agent(
    name="AlternativeGenerator",
    instructions="""
    Given what the player tried to do and what's actually available in the scene,
    suggest 3 creative alternatives that:
    1. Achieve a similar narrative goal
    2. Use only what's present in the scene
    3. Fit the setting's capabilities
    4. Feel like natural choices, not system suggestions
    5. Vary in approach (physical, social, environmental)
    
    Make suggestions feel organic and enticing, not like consolation prizes.
    Each alternative should be specific to the current moment and scene.
    
    Output JSON array of strings: ["alternative 1", "alternative 2", "alternative 3"]
    """,
    model="gpt-5-nano"
)

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
    7. The setting kind (realistic, fantasy, sci-fi, etc.) and its capabilities
    8. Hard rules that MUST be enforced vs soft rules that guide behavior
    
    Consider the dynamic context provided, including:
    - Setting capabilities and limitations
    - Current scene state and available elements
    - Previously established possibilities/impossibilities
    - The specific nature and context of this world
    
    BE STRICT but CONTEXTUAL. Enforce the world's unique logic.
    
    Output ONLY JSON:
      {"overall":{"feasible":bool,"strategy":"allow|deny|reinterpret"},
       "per_intent":[
         {"feasible":bool,"strategy":"allow|deny|reinterpret",
          "violations":[{"rule":"...", "reason":"..."}],
          "categories":["..."]}
       ]}
    """,
    model="gpt-5-nano"
)

# Setting detective agent for auto-detecting setting type
SETTING_DETECTIVE_AGENT = Agent(
    name="SettingDetective",
    instructions="""
    Analyze the established narrative elements to determine the setting type and capabilities.
    
    Consider:
    - Technology level (medieval, modern, futuristic, etc.)
    - Presence of magic or supernatural elements
    - Physics model (realistic, soft sci-fi, fantasy, surreal)
    - Genre markers (noir, cyberpunk, high fantasy, etc.)
    - Established world rules and limitations
    - Environmental descriptions and atmosphere
    
    Determine:
    1. Setting type (e.g., "realistic_modern", "high_fantasy", "cyberpunk")
    2. Setting kind (broader category)
    3. Key capabilities (what's possible in this world)
    4. Confidence level
    
    Output JSON:
    {
        "setting_type": "...",
        "setting_kind": "...",
        "confidence": 0.X,
        "indicators": ["...", "..."],
        "capabilities": {
            "magic": "none|limited|common|ubiquitous",
            "technology": "primitive|medieval|modern|advanced|futuristic",
            "physics": "realistic|flexible|surreal",
            "supernatural": "none|hidden|known|common"
        },
        "details": "..."
    }
    """,
    model="gpt-5-nano"
)

async def assess_action_feasibility(nyx_ctx: NyxContext, user_input: str) -> Dict[str, Any]:
    """
    Dynamically assess if an action is feasible in the current setting context.
    Generates unique, contextual responses for every rejection.
    """
    # Parse the intended actions
    intents = await parse_action_intents(user_input)
    
    # Load comprehensive setting context
    setting_context = await _load_comprehensive_context(nyx_ctx)
    
    # Quick check against hard rules
    quick_check = await _quick_feasibility_check(setting_context, intents)
    
    if quick_check.get("hard_blocked"):
        # Generate dynamic, unique rejections for blocked actions
        per_intent = []
        for i, intent in enumerate(intents):
            if not quick_check["intent_feasible"][i]:
                # Generate completely unique rejection
                rejection = await generate_dynamic_rejection(
                    setting_context, 
                    {**intent, "raw_text": user_input, "violations": quick_check["violations"][i]},
                    nyx_ctx
                )
                
                per_intent.append({
                    "feasible": False,
                    "strategy": "deny",
                    "violations": quick_check["violations"][i],
                    "reality_response": rejection["reality_response"],
                    "narrator_guidance": rejection["narrator_guidance"],
                    "suggested_alternatives": rejection["suggested_alternatives"],
                    "metaphor": rejection.get("metaphor", ""),
                    "categories": intent.get("categories", [])
                })
            else:
                per_intent.append({
                    "feasible": True,
                    "strategy": "allow",
                    "categories": intent.get("categories", [])
                })
        
        return {
            "overall": {"feasible": False, "strategy": "deny"},
            "per_intent": per_intent
        }
    
    # Full AI-powered assessment for nuanced cases
    return await _full_dynamic_assessment(nyx_ctx, user_input, intents, setting_context)

async def _load_comprehensive_context(nyx_ctx: NyxContext) -> Dict[str, Any]:
    """Load all relevant context about what's possible in this setting"""
    
    context = {
        "type": "unknown",
        "kind": "modern_realistic",
        "capabilities": {},
        "reality_context": "normal",
        "established_rules": [],
        "hard_rules": [],
        "soft_rules": [],
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
        "narrative_history": [],
        "environment_desc": "",
        "setting_name": "",
        "stat_modifiers": {}
    }
    
    async with get_db_connection_context() as conn:
        # Get comprehensive setting information from new_game_agent storage
        setting_keys = [
            'SettingType', 'SettingKind', 'SettingCapabilities', 
            'RealityContext', 'PhysicsModel', 'EnvironmentDesc', 
            'CurrentSetting', 'SettingStatModifiers', 'EnvironmentHistory',
            'ScenarioName', 'CurrentLocation', 'CurrentTime'
        ]
        
        setting_data = await conn.fetch("""
            SELECT key, value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 
            AND key = ANY($3)
        """, nyx_ctx.user_id, nyx_ctx.conversation_id, setting_keys)
        
        for row in setting_data:
            key = row['key']
            value = row['value']
            
            if key == 'SettingType':
                context["type"] = value
            elif key == 'SettingKind':
                context["kind"] = value
            elif key == 'SettingCapabilities':
                try:
                    context["capabilities"] = json.loads(value)
                    # Extract specific capabilities
                    if "magic" in context["capabilities"]:
                        context["magic_system"] = context["capabilities"]["magic"]
                    if "technology" in context["capabilities"]:
                        context["technology_level"] = context["capabilities"]["technology"]
                except:
                    pass
            elif key == 'RealityContext':
                context["reality_context"] = value
            elif key == 'PhysicsModel':
                context["physics_model"] = value
            elif key == 'EnvironmentDesc':
                context["environment_desc"] = value
            elif key == 'CurrentSetting':
                context["setting_name"] = value
            elif key == 'SettingStatModifiers':
                try:
                    context["stat_modifiers"] = json.loads(value)
                except:
                    pass
            elif key == 'EnvironmentHistory':
                context["environment_history"] = value
            elif key == 'CurrentLocation':
                context["location"]["name"] = value
            elif key == 'CurrentTime':
                context["current_time"] = value
        
        # Auto-detect if not set
        if context["type"] == "unknown":
            detected = await detect_setting_type(nyx_ctx)
            context["type"] = detected["setting_type"]
            context["kind"] = detected.get("setting_kind", "modern_realistic")
            context["capabilities"] = detected.get("capabilities", {})
            
        # Get established impossibilities
        impossibilities = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedImpossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if impossibilities:
            context["established_impossibilities"] = json.loads(impossibilities)
            
        # Get established possibilities
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
            context["location"].update(scene_data.get("location", {}))
            context["available_items"] = scene_data.get("items", [])
            context["present_entities"] = scene_data.get("npcs", [])
            
        # Get game rules with categorization
        rules = await conn.fetch("""
            SELECT rule_name, condition, effect
            FROM GameRules
            WHERE user_id=$1 AND conversation_id=$2
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        for r in rules:
            rule_data = {
                "name": r["rule_name"], 
                "condition": r["condition"], 
                "effect": r["effect"]
            }
            
            # Categorize rules
            if r["rule_name"].startswith("hard_"):
                context["hard_rules"].append(rule_data)
            elif r["rule_name"].startswith("soft_"):
                context["soft_rules"].append(rule_data)
            else:
                context["established_rules"].append(rule_data)
                
        # Get character state
        player_stats = await conn.fetchrow("""
            SELECT * FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2
            LIMIT 1
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if player_stats:
            context["character_state"] = dict(player_stats)
            
        # Get inventory items
        inventory = await conn.fetch("""
            SELECT item_name, equipped FROM PlayerInventory
            WHERE user_id=$1 AND conversation_id=$2
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["available_items"].extend([item["item_name"] for item in inventory])
        
        # Get active NPCs in current location
        if context["location"].get("name"):
            npcs = await conn.fetch("""
                SELECT npc_name FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 
                AND current_location=$3
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, context["location"]["name"])
            
            context["present_entities"].extend([npc["npc_name"] for npc in npcs])
            
        # Get recent narrative for context
        recent = await conn.fetch("""
            SELECT content FROM messages
            WHERE conversation_id=$1 AND sender='Nyx'
            ORDER BY created_at DESC LIMIT 5
        """, nyx_ctx.conversation_id)
        
        context["narrative_history"] = [r["content"][:500] for r in recent if r["content"]]
        
    return context

async def generate_dynamic_rejection(
    setting_context: Dict[str, Any], 
    intent: Dict[str, Any],
    nyx_ctx: NyxContext
) -> Dict[str, Any]:
    """Generate completely unique, contextual rejection narratives"""
    
    # Load rejection history to avoid repetition
    rejection_history = await _load_rejection_history(nyx_ctx)
    
    # Get current scene details
    current_scene = await _load_current_scene(nyx_ctx)
    
    # Build dynamic context
    rejection_context = {
        "setting": {
            "name": setting_context.get("setting_name"),
            "kind": setting_context.get("kind"),
            "atmosphere": setting_context.get("environment_desc", "")[:500],
            "reality_type": setting_context.get("reality_context"),
            "capabilities": setting_context.get("capabilities", {}),
            "current_location": setting_context.get("location", {}).get("name"),
            "physics_model": setting_context.get("physics_model"),
            "time": setting_context.get("current_time", "unknown time")
        },
        "attempted_action": {
            "raw_input": intent.get("raw_text", ""),
            "categories": intent.get("categories", []),
            "violations": intent.get("violations", []),
            "specific_reason": intent.get("violations", [{}])[0].get("reason", "") if intent.get("violations") else ""
        },
        "scene_context": {
            "present_npcs": current_scene.get("npcs", []),
            "available_items": current_scene.get("items", []),
            "recent_events": current_scene.get("recent_narrative", [])[-3:],
            "location_features": current_scene.get("location_features", []),
            "time_phase": current_scene.get("time_phase", "day")
        },
        "rejection_history": rejection_history[-5:],  # Last 5 to avoid repetition
        "instruction": "Generate a unique rejection that has never been used before. Make it specific to this exact moment and action."
    }
    
    # Generate unique rejection
    run = await Runner.run(
        REJECTION_NARRATOR_AGENT,
        json.dumps(rejection_context)
    )
    
    try:
        result = json.loads(getattr(run, "final_output", "{}"))
        
        # Store this rejection to avoid future repetition
        await _store_rejection(nyx_ctx, result)
        
        # Generate contextual alternatives
        result["suggested_alternatives"] = await _generate_contextual_alternatives(
            nyx_ctx, setting_context, current_scene, intent
        )
        
        return result
    except Exception as e:
        # Dynamic fallback
        return await _generate_fallback_rejection(setting_context, intent, current_scene)

async def _generate_contextual_alternatives(
    nyx_ctx: NyxContext,
    setting_context: Dict,
    current_scene: Dict,
    failed_intent: Dict
) -> List[str]:
    """Generate alternatives based on what's actually available in the scene"""
    
    context = {
        "failed_attempt": {
            "action": failed_intent.get("raw_text", ""),
            "categories": failed_intent.get("categories", []),
            "goal": "What the player was trying to achieve"
        },
        "scene_state": {
            "npcs": current_scene.get("npcs", []),
            "items": current_scene.get("items", []),
            "location": setting_context.get("location", {}).get("name", "unknown"),
            "location_features": current_scene.get("location_features", []),
            "time": current_scene.get("time_phase", "day")
        },
        "player_state": {
            "abilities": setting_context.get("character_abilities", []),
            "inventory": setting_context.get("available_items", []),
            "stats": setting_context.get("character_state", {})
        },
        "world_rules": {
            "capabilities": setting_context.get("capabilities", {}),
            "kind": setting_context.get("kind", "realistic"),
            "established_possibilities": setting_context.get("established_possibilities", [])[-5:]
        }
    }
    
    run = await Runner.run(ALTERNATIVE_GENERATOR_AGENT, json.dumps(context))
    
    try:
        alternatives = json.loads(getattr(run, "final_output", "[]"))
        return alternatives[:3]
    except:
        # Dynamic fallback based on actual scene
        return await _generate_scene_based_alternatives(current_scene, setting_context)

async def _generate_scene_based_alternatives(current_scene: Dict, setting_context: Dict) -> List[str]:
    """Generate fallback alternatives based on scene elements"""
    alternatives = []
    
    # NPC interactions
    if current_scene.get("npcs"):
        npc = random.choice(current_scene["npcs"])
        alternatives.append(f"approach {npc} for assistance")
    
    # Item usage
    if current_scene.get("items"):
        item = random.choice(current_scene["items"])
        alternatives.append(f"examine the {item} more closely")
    
    # Location features
    if current_scene.get("location_features"):
        feature = random.choice(current_scene["location_features"])
        alternatives.append(f"investigate the {feature}")
    
    # Time-based alternatives
    time_phase = current_scene.get("time_phase", "day")
    if time_phase == "night":
        alternatives.append("wait until dawn for better visibility")
    elif time_phase == "day":
        alternatives.append("search for a different approach")
    
    # Setting-specific alternatives
    if setting_context.get("kind") == "high_fantasy":
        alternatives.append("seek magical guidance")
    elif setting_context.get("kind") == "cyberpunk":
        alternatives.append("access the local network for information")
    else:
        alternatives.append("reconsider your approach")
    
    return alternatives[:3]

async def _quick_feasibility_check(setting_context: Dict, intents: List[Dict]) -> Dict:
    """Quick check against hard rules without repetitive responses"""
    blocked = False
    intent_feasible = []
    violations = []
    
    for intent in intents:
        intent_violations = []
        feasible = True
        
        # Check hard rules dynamically
        for rule in setting_context.get("hard_rules", []):
            if await _rule_applies_to_intent(rule, intent, setting_context):
                feasible = False
                intent_violations.append({
                    "rule": rule["name"],
                    "reason": rule["effect"]
                })
        
        # Check established impossibilities with fuzzy matching
        for imp in setting_context.get("established_impossibilities", []):
            if _matches_impossibility_dynamic(intent, imp):
                feasible = False
                intent_violations.append({
                    "rule": "established_impossibility",
                    "reason": imp["reason"]
                })
        
        # Check prerequisites
        if not await _check_prerequisites(intent, setting_context):
            feasible = False
            intent_violations.append({
                "rule": "missing_prerequisites",
                "reason": "Required elements are not present"
            })
        
        intent_feasible.append(feasible)
        violations.append(intent_violations)
        if not feasible:
            blocked = True
    
    return {
        "hard_blocked": blocked,
        "intent_feasible": intent_feasible,
        "violations": violations
    }

async def _rule_applies_to_intent(rule: Dict, intent: Dict, context: Dict) -> bool:
    """Dynamically check if a rule applies to an intent"""
    condition = rule.get("condition", "").lower()
    
    # Check intent categories
    if "categories" in intent:
        for cat in intent["categories"]:
            if cat.lower() in condition:
                return True
    
    # Check action keywords
    intent_str = json.dumps(intent).lower()
    condition_keywords = set(condition.split())
    intent_keywords = set(intent_str.split())
    
    # Require significant overlap
    overlap = len(condition_keywords & intent_keywords)
    if overlap >= max(2, len(condition_keywords) * 0.3):
        return True
    
    return False

def _matches_impossibility_dynamic(intent: Dict, impossibility: Dict) -> bool:
    """Dynamic matching with fuzzy logic"""
    # Category-based matching with threshold
    imp_categories = set(impossibility.get("categories", []))
    intent_categories = set(intent.get("categories", []))
    
    if imp_categories and intent_categories:
        overlap = len(imp_categories & intent_categories)
        if overlap >= max(1, min(len(imp_categories), len(intent_categories)) * 0.5):
            return True
    
    # Semantic similarity for actions
    if "action" in impossibility:
        imp_action = impossibility["action"].lower()
        intent_text = json.dumps(intent).lower()
        
        # Key phrase matching
        if len(imp_action) > 10 and imp_action in intent_text:
            return True
        
        # Word overlap threshold
        imp_words = set(imp_action.split())
        intent_words = set(intent_text.split())
        
        common_words = imp_words & intent_words
        # Filter out common words
        meaningful_overlap = common_words - {"the", "a", "an", "to", "from", "with", "at", "in", "on"}
        
        if len(meaningful_overlap) >= min(3, len(imp_words) * 0.4):
            return True
    
    return False

async def _check_prerequisites(intent: Dict, context: Dict) -> bool:
    """Check if required elements for the action are present"""
    
    # Check required items
    if "instruments" in intent:
        for item in intent["instruments"]:
            if item and item not in context.get("available_items", []):
                return False
    
    # Check target presence
    if "direct_object" in intent:
        all_present_entities = set(context.get("present_entities", []))
        all_present_entities.add(context.get("location", {}).get("name", ""))
        
        for target in intent["direct_object"]:
            if target and target not in all_present_entities:
                # Check if it's an item
                if target not in context.get("available_items", []):
                    return False
    
    # Check ability requirements based on categories
    categories = intent.get("categories", [])
    if "spellcasting" in categories and context.get("magic_system") == "none":
        return False
    if "hacking" in categories and context.get("technology_level") in ["primitive", "medieval"]:
        return False
    
    return True

async def _full_dynamic_assessment(
    nyx_ctx: NyxContext,
    user_input: str,
    intents: List[Dict],
    setting_context: Dict
) -> Dict[str, Any]:
    """Full AI-powered assessment for nuanced cases"""
    
    # Build comprehensive assessment context
    assessment_context = {
        "user_input": user_input,
        "intents": intents,
        "setting": {
            "name": setting_context.get("setting_name"),
            "kind": setting_context.get("kind"),
            "capabilities": setting_context.get("capabilities"),
            "reality_level": setting_context.get("reality_context"),
            "environment": setting_context.get("environment_desc", "")[:500],
            "hard_rules": setting_context.get("hard_rules"),
            "soft_rules": setting_context.get("soft_rules")
        },
        "context": {
            "location": setting_context.get("location"),
            "present_npcs": setting_context.get("present_entities"),
            "available_items": setting_context.get("available_items"),
            "player_state": setting_context.get("character_state"),
            "recent_narrative": setting_context.get("narrative_history", [])[-3:]
        },
        "history": {
            "established_impossibilities": setting_context.get("established_impossibilities", [])[-10:],
            "established_possibilities": setting_context.get("established_possibilities", [])[-10:]
        }
    }
    
    # Run full assessment
    run = await Runner.run(FEASIBILITY_AGENT, json.dumps(assessment_context))
    
    try:
        result = json.loads(getattr(run, "final_output", "{}"))
        
        # Enhance denied intents with dynamic rejections
        for i, intent_result in enumerate(result.get("per_intent", [])):
            if not intent_result.get("feasible"):
                # Generate unique rejection
                rejection = await generate_dynamic_rejection(
                    setting_context,
                    {**intents[i], "raw_text": user_input, "violations": intent_result.get("violations", [])},
                    nyx_ctx
                )
                
                intent_result.update(rejection)
        
        return result
    except Exception as e:
        logger.error(f"Dynamic assessment failed: {e}")
        return _default_feasibility_response(intents)

async def _load_rejection_history(nyx_ctx: NyxContext) -> List[Dict]:
    """Load recent rejection narratives to avoid repetition"""
    async with get_db_connection_context() as conn:
        history = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='RejectionHistory'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        return json.loads(history) if history else []

async def _store_rejection(nyx_ctx: NyxContext, rejection: Dict):
    """Store rejection for future reference"""
    history = await _load_rejection_history(nyx_ctx)
    
    # Add timestamp and context
    rejection["timestamp"] = datetime.now().isoformat()
    rejection["context_hash"] = hash(json.dumps(rejection, sort_keys=True))
    
    # Add to history and limit size
    history.append(rejection)
    history = history[-30:]  # Keep last 30 rejections
    
    async with get_db_connection_context() as conn:
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'RejectionHistory', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value = EXCLUDED.value
        """, nyx_ctx.user_id, nyx_ctx.conversation_id, json.dumps(history))

async def _load_current_scene(nyx_ctx: NyxContext) -> Dict:
    """Load comprehensive current scene data"""
    scene = {
        "npcs": [],
        "items": [],
        "location_features": [],
        "recent_narrative": [],
        "time_phase": "unknown"
    }
    
    async with get_db_connection_context() as conn:
        # Get scene state
        scene_data = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentScene'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if scene_data:
            parsed = json.loads(scene_data)
            scene.update(parsed)
        
        # Get current time phase
        time_data = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentTime'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if time_data:
            import re
            time_match = re.search(r'(Morning|Afternoon|Evening|Night)', time_data)
            if time_match:
                scene["time_phase"] = time_match.group(1).lower()
        
        # Get location details
        location_name = scene.get("location") or await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentLocation'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if location_name:
            location = await conn.fetchrow("""
                SELECT notable_features, hidden_aspects, description
                FROM Locations
                WHERE user_id=$1 AND conversation_id=$2 AND location_name=$3
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, location_name)
            
            if location:
                scene["location_features"] = location.get("notable_features", []) or []
                scene["location_description"] = location.get("description", "")
        
        # Get recent narrative
        recent = await conn.fetch("""
            SELECT content FROM messages
            WHERE conversation_id=$1 AND sender='Nyx'
            ORDER BY created_at DESC LIMIT 3
        """, nyx_ctx.conversation_id)
        
        scene["recent_narrative"] = [r["content"][:200] for r in recent]
    
    return scene

async def _generate_fallback_rejection(
    setting_context: Dict, 
    intent: Dict,
    current_scene: Dict
) -> Dict[str, Any]:
    """Generate dynamic fallback rejection when AI fails"""
    
    # Build contextual response based on setting
    setting_kind = setting_context.get("kind", "realistic")
    
    reality_responses = {
        "realistic": [
            f"The world remains bound by familiar laws",
            f"Reality offers no exception here",
            f"The universe maintains its steady rhythm"
        ],
        "fantasy": [
            f"Even magic has boundaries it cannot cross",
            f"The weave resists this particular thread",
            f"Ancient laws hold firm against your will"
        ],
        "scifi": [
            f"The system parameters reject this input",
            f"Quantum mechanics forbid this outcome",
            f"The simulation constraints remain absolute"
        ]
    }
    
    base_type = "realistic"
    if "fantasy" in setting_kind or "magic" in setting_kind:
        base_type = "fantasy"
    elif "sci" in setting_kind or "cyber" in setting_kind or "tech" in setting_kind:
        base_type = "scifi"
    
    return {
        "reality_response": random.choice(reality_responses.get(base_type, reality_responses["realistic"])),
        "narrator_guidance": f"What you attempt slips beyond reach, the world's fabric unchanged by will alone.",
        "suggested_alternatives": await _generate_scene_based_alternatives(current_scene, setting_context),
        "metaphor": "like trying to paint with shadows on water"
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
        
        # Extract categories
        categories = await _extract_action_categories(action)
        
        # Check for duplicates with fuzzy matching
        is_duplicate = False
        for imp in impossibilities:
            if _similar_impossibility(imp, action, categories):
                is_duplicate = True
                break
        
        if not is_duplicate:
            impossibilities.append({
                "action": action,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "categories": categories,
                "hash": hash(action + reason)
            })
            
            # Keep limited history
            impossibilities = impossibilities[-50:]
            
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
        current = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedPossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        possibilities = json.loads(current) if current else []
        
        # Check for duplicates
        is_duplicate = any(
            p.get("hash") == hash(action) for p in possibilities
        )
        
        if not is_duplicate:
            possibilities.append({
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "categories": categories,
                "hash": hash(action)
            })
            
            possibilities = possibilities[-50:]
            
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'EstablishedPossibilities', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, json.dumps(possibilities))

async def detect_setting_type(nyx_ctx: NyxContext) -> Dict[str, Any]:
    """Intelligently detect what kind of setting this is based on established elements"""
    
    # Gather comprehensive context
    context = {"narrative": [], "elements": {}, "npcs": [], "locations": [], "items": []}
    
    async with get_db_connection_context() as conn:
        # Get recent narrative
        recent_msgs = await conn.fetch("""
            SELECT content FROM messages
            WHERE conversation_id=$1 
            ORDER BY created_at DESC LIMIT 10
        """, nyx_ctx.conversation_id)
        
        context["narrative"] = [msg["content"][:500] for msg in recent_msgs if msg["content"]]
        
        # Get roleplay elements
        elements = await conn.fetch("""
            SELECT key, value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 
            AND key IN ('EnvironmentDesc', 'EnvironmentHistory', 'CurrentSetting', 'ScenarioName')
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        for elem in elements:
            context["elements"][elem["key"]] = elem["value"]
        
        # Sample NPCs
        npcs = await conn.fetch("""
            SELECT npc_name, role, archetypes FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2
            LIMIT 5
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["npcs"] = [{"name": n["npc_name"], "role": n["role"]} for n in npcs]
        
        # Sample locations
        locations = await conn.fetch("""
            SELECT location_name, location_type FROM Locations
            WHERE user_id=$1 AND conversation_id=$2
            LIMIT 5
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["locations"] = [{"name": l["location_name"], "type": l["location_type"]} for l in locations]
    
    # Run detection
    run = await Runner.run(SETTING_DETECTIVE_AGENT, json.dumps(context))
    
    try:
        result = json.loads(getattr(run, "final_output", "{}"))
        
        # Store the detected settings
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'SettingType', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, result.get("setting_type", "realistic_modern"))
            
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'DetectedCapabilities', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, json.dumps(result.get("capabilities", {})))
        
        return result
        
    except Exception:
        return {
            "setting_type": "realistic_modern",
            "setting_kind": "modern_realistic", 
            "confidence": 0.5,
            "capabilities": {}
        }

async def _extract_action_categories(action: str) -> List[str]:
    """Extract categories from an action string"""
    try:
        intents = await parse_action_intents(action)
        categories = set()
        for intent in intents:
            categories.update(intent.get("categories", []))
        return list(categories)
    except Exception:
        return []

def _similar_impossibility(existing: Dict, new_action: str, new_categories: List[str]) -> bool:
    """Check if an impossibility is similar to an existing one"""
    # Check hash first
    if existing.get("hash") == hash(new_action + existing.get("reason", "")):
        return True
    
    # Check category overlap
    existing_cats = set(existing.get("categories", []))
    new_cats = set(new_categories)
    
    if existing_cats and new_cats:
        overlap = len(existing_cats & new_cats)
        if overlap >= max(1, min(len(existing_cats), len(new_cats)) * 0.7):
            return True
    
    # Check action similarity
    if "action" in existing:
        existing_action = existing["action"].lower()
        new_action_lower = new_action.lower()
        
        # Direct substring match
        if len(existing_action) > 20 and len(new_action_lower) > 20:
            if existing_action in new_action_lower or new_action_lower in existing_action:
                return True
    
    return False

def _default_feasibility_response(intents: List[Dict]) -> Dict[str, Any]:
    """Default response when feasibility check fails"""
    
    # Allow only clearly mundane actions by default
    mundane_categories = {"movement", "dialogue", "observation", "mundane_action", "interaction"}
    
    per_intent = []
    all_feasible = True
    
    for intent in intents:
        intent_categories = set(intent.get("categories", []))
        is_mundane = bool(intent_categories & mundane_categories) or not intent_categories
        
        per_intent.append({
            "feasible": is_mundane,
            "strategy": "allow" if is_mundane else "deny",
            "violations": [] if is_mundane else [{"rule": "unknown", "reason": "Cannot verify feasibility"}],
            "categories": list(intent_categories)
        })
        
        if not is_mundane:
            all_feasible = False
    
    return {
        "overall": {
            "feasible": all_feasible,
            "strategy": "allow" if all_feasible else "deny"
        },
        "per_intent": per_intent
    }

# Fast context-free feasibility check for pre-gate
# nyx/nyx_agent/feasibility.py
import json
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple

from .db import get_db_connection_context  # adjust import path if different
from .action_parser import parse_action_intents  # your existing parser
from .logging import logger  # your logger

# Optional helpers (kept soft — only used if present in your codebase)
try:
    from .helpers import _infer_categories_from_text, _scene_alternatives, _compose_guidance
except Exception:
    def _infer_categories_from_text(text_l: str) -> Set[str]:
        # ultra-light fallback; intentionally weak (keeps system dynamic)
        hits = set()
        if any(k in text_l for k in ("summon", "conjure", "spawn", "manifest", "materialize")):
            hits.add("ex_nihilo_conjuration")
        if any(k in text_l for k in ("fly", "levitate", "hover")):
            hits.add("physics_violation")
        if any(k in text_l for k in ("spaceship", "laser", "plasma", "warp")):
            hits.add("scifi_setpiece")
        if any(k in text_l for k in ("hack drone", "access ai", "drone")):
            hits.add("ai_system_access")
        return hits

    def _scene_alternatives(npcs, items, features, time_phase) -> List[str]:
        alts = []
        if items:
            alts.append(f"use the {items[0]}")
        if features:
            alts.append(f"interact with {features[0]}")
        if npcs:
            alts.append(f"ask {npcs[0].get('name','someone')} for help")
        if not alts:
            alts.append("try a simpler, grounded action that uses something visible in the scene")
        return alts

    def _compose_guidance(setting_kind: str, location_name: Optional[str], blocking: Set[str]) -> str:
        loc = f" in {location_name}" if location_name else ""
        cats = ", ".join(sorted(blocking))
        return f"Reality{loc} doesn’t support that ({cats}). Try something that fits what’s actually present."

def _normalize_bool(v: Any) -> bool:
    if isinstance(v, bool): return v
    if v is None: return False
    s = str(v).strip().lower()
    return s in {"1","true","yes","y","on","allowed","enable","enabled"}

def _safe_json_loads(s: Optional[str], default):
    try:
        return json.loads(s or "") if s else default
    except Exception:
        return default

async def assess_action_feasibility_fast(user_id: int, conversation_id: int, text: str) -> Dict[str, Any]:
    """
    Conversation/scene-aware quick feasibility gate with LOUD logging and dynamic judgments.
    - Uses per-conversation GameRules + CurrentRoleplay to decide.
    - Explicit rules/scene bans/EstablishedImpossibilities can deny.
    - Capability mismatches only downgrade to ASK (soft block) rather than hard DENY.
    - If context is missing, prefer ALLOW (keep world flexible).
    """
    logger.info(f"[FEASIBILITY] Checking: {text[:160]!r}")
    text_l = (text or "").lower()

    # ---- 1) Parse intents (never hard-block on parse errors) -----------------
    parse_error: Optional[str] = None
    try:
        intents = await parse_action_intents(text or "")
        logger.info(f"[FEASIBILITY] Parsed {len(intents)} intents "
                    f"-> cats: {[i.get('categories', []) for i in intents]}")
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"
        logger.error(f"[FEASIBILITY] Intent parsing FAILED: {parse_error}", exc_info=True)
        intents = []

    # Fallback single-pass intent so we still produce a decision
    if not intents:
        intents = [{"categories": list(_infer_categories_from_text(text_l)) or []}]

    # ---- 2) Load dynamic context ------------------------------------------------
    setting_kind = "modern_realistic"
    setting_type = "realistic_modern"
    reality_context = "normal"
    physics_model = "realistic"

    capabilities: Dict[str, Any] = {}
    scene: Dict[str, Any] = {}
    location_name: Optional[str] = None
    established_impossibilities: List[Dict[str, Any]] = []
    rules: List[Dict[str, Any]] = []

    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(
                """
                SELECT key, value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2
                  AND key = ANY($3)
                """,
                user_id,
                conversation_id,
                [
                    "SettingType",
                    "SettingKind",
                    "RealityContext",
                    "PhysicsModel",
                    "SettingCapabilities",
                    "CurrentScene",
                    "CurrentLocation",
                    "EstablishedImpossibilities",
                ],
            )
            kv = {r["key"]: r["value"] for r in rows}

            setting_type = kv.get("SettingType") or setting_type
            setting_kind = kv.get("SettingKind") or setting_kind
            reality_context = kv.get("RealityContext") or reality_context
            physics_model = kv.get("PhysicsModel") or physics_model

            capabilities = _safe_json_loads(kv.get("SettingCapabilities"), {}) or {}
            scene = _safe_json_loads(kv.get("CurrentScene"), {}) or {}
            location_name = kv.get("CurrentLocation") or (scene.get("location") if isinstance(scene, dict) else None)
            established_impossibilities = _safe_json_loads(kv.get("EstablishedImpossibilities"), []) or []

            rules = await conn.fetch(
                """
                SELECT condition, effect
                FROM GameRules
                WHERE user_id=$1 AND conversation_id=$2 AND enabled=TRUE
                """,
                user_id,
                conversation_id,
            )
    except Exception as e:
        logger.error(f"[FEASIBILITY] DB read failed (soft): {e}", exc_info=True)
        # Keep defaults; remain permissive

    logger.info("[FEASIBILITY] Setting "
                f"type={setting_type} kind={setting_kind} reality={reality_context} physics={physics_model}")
    logger.debug(f"[FEASIBILITY] Capabilities: {capabilities}")
    logger.debug(f"[FEASIBILITY] Scene keys: {list(scene.keys()) if isinstance(scene, dict) else 'n/a'}")

    # ---- 3) Build dynamic allow/deny sets from conversation rules/scene --------
    hard_deny_cats: Set[str] = set()
    allow_cats: Set[str] = set()

    for r in (rules or []):
        cond = (r.get("condition") or "").strip().lower()
        eff = (r.get("effect") or "").strip().lower()
        if cond.startswith("category:"):
            name = cond.split(":", 1)[1].strip()
            # explicit allows/denies ONLY from DB rules (dynamic)
            if any(k in eff for k in ("prohibit", "forbid", "cannot", "not allowed", "disallow", "deny")):
                hard_deny_cats.add(name)
            if any(k in eff for k in ("allow", "permitted", "can", "enabled", "allowed")):
                allow_cats.add(name)

    scene_banned = set((scene.get("banned_categories") or []) if isinstance(scene, dict) else [])
    scene_allowed = set((scene.get("allowed_categories") or []) if isinstance(scene, dict) else [])

    # ---- 4) Soft constraints from capabilities/physics (NO hard rules) ---------
    # We compute "mismatch" categories to nudge toward ASK rather than DENY.
    # You can define your own mapping from capabilities to category clusters in CurrentRoleplay.SettingCapabilities.
    # Example capabilities you might store: { "magic":"none|limited|common", "physics":"realistic|flexible|surreal",
    #   "technology":"primitive|medieval|modern|advanced|futuristic", "supernatural":"none|hidden|known|common" }
    caps_magic = str((capabilities.get("magic") or "")).lower()
    caps_physics = str((capabilities.get("physics") or physics_model or "")).lower()
    caps_tech = str((capabilities.get("technology") or "")).lower()
    caps_supernatural = str((capabilities.get("supernatural") or "")).lower()

    # Dynamic feature flags that creators can set per conversation:
    # e.g. {"feature_flags": {"public_magic_ok": true, "sci_fi_elements_ok": false}}
    feature_flags = (capabilities.get("feature_flags") or {}) if isinstance(capabilities, dict) else {}

    # Category groups to *soft*-warn on when mismatch:
    soft_map: List[Tuple[bool, Set[str], str]] = []

    # Magic-sensitive categories
    if caps_magic in {"none", ""} and not _normalize_bool(feature_flags.get("magic_ok")):
        soft_map.append((
            True,
            {"spellcasting", "ritual_magic", "summoning", "necromancy", "ex_nihilo_conjuration", "psionics", "public_magic"},
            "magic_limited_by_setting"
        ))

    # Physics-sensitive categories
    if caps_physics in {"realistic", ""} and not _normalize_bool(feature_flags.get("loose_physics_ok")):
        soft_map.append((
            True,
            {"physics_violation", "reality_warping", "unaided_flight", "time_travel", "teleportation", "ex_nihilo_conjuration"},
            "physics_constrained"
        ))

    # Tech/scifi-sensitive categories
    if caps_tech in {"primitive", "medieval", "modern", ""} and not _normalize_bool(feature_flags.get("sci_fi_elements_ok")):
        soft_map.append((
            True,
            {"vehicle_operation_space", "ai_system_access", "drone_control", "scifi_setpiece", "vacuum_exposure", "spacewalk"},
            "tech_level_constrained"
        ))

    # Supernatural-sensitive
    if caps_supernatural in {"none", ""} and not _normalize_bool(feature_flags.get("supernatural_ok")):
        soft_map.append((
            True,
            {"undead_control", "spirit_binding", "demon_summoning", "necromancy"},
            "supernatural_constrained"
        ))

    soft_constraints_map: Dict[str, str] = {}
    for _active, cats, tag in soft_map:
        if _active:
            for c in cats:
                # only soft when *not* explicitly allowed
                if c not in allow_cats and c not in scene_allowed:
                    soft_constraints_map[c] = tag

    # ---- 5) Per-intent evaluation ---------------------------------------------
    per_intent: List[Dict[str, Any]] = []
    any_hard_block = False

    # Quick scene affordances for alternatives
    scene_npcs = (scene.get("npcs") or scene.get("present_npcs") or []) if isinstance(scene, dict) else []
    scene_items = (scene.get("items") or scene.get("available_items") or []) if isinstance(scene, dict) else []
    location_features = (scene.get("location_features") or []) if isinstance(scene, dict) else []
    time_phase = (scene.get("time_phase") or scene.get("time_of_day") or "day") if isinstance(scene, dict) else "day"

    # Use only the last few impossibilities (most recent canon)
    last_imps = (established_impossibilities or [])[-12:]

    def reasons_for(category_hits: Set[str]) -> List[Dict[str, str]]:
        reasons: List[Dict[str, str]] = []
        for c in sorted(category_hits):
            if c in hard_deny_cats:
                reasons.append({"rule": f"category:{c}", "reason": "Prohibited by world rule"})
            elif c in scene_banned:
                reasons.append({"rule": f"scene:{c}", "reason": "Not available in this scene"})
            else:
                # established impossibility covered elsewhere; default:
                reasons.append({"rule": f"unavailable:{c}", "reason": "Unavailable here"})
        return reasons

    for intent in intents or [{}]:
        cats = set(intent.get("categories") or [])
        if not cats:
            inferred = _infer_categories_from_text(text_l)
            if inferred:
                cats = inferred

        # (A) Established Impossibilities (hard deny if categories overlap)
        hit_imposs = []
        if last_imps and cats:
            for imp in last_imps:
                imp_cats = set((imp or {}).get("categories", []) or [])
                if imp_cats & cats:
                    hit_imposs.append(imp)
        if hit_imposs:
            logger.info(f"[FEASIBILITY] Hard deny by EstablishedImpossibilities -> {cats}")
            per_intent.append({
                "feasible": False,
                "strategy": "deny",
                "violations": [{"rule": "established_impossibility", "reason": (hit_imposs[-1].get("reason") or "Previously established as impossible")}],
                "narrator_guidance": _compose_guidance(setting_kind, location_name, cats),
                "suggested_alternatives": _scene_alternatives(scene_npcs, scene_items, location_features, time_phase),
                "categories": sorted(cats),
            })
            any_hard_block = True
            continue

        # (B) Explicit world/scene rule bans (hard deny)
        hard_bans = (cats & (hard_deny_cats | scene_banned)) - (allow_cats | scene_allowed)
        if hard_bans:
            logger.info(f"[FEASIBILITY] Hard deny by explicit rule/scene -> {hard_bans}")
            per_intent.append({
                "feasible": False,
                "strategy": "deny",
                "violations": reasons_for(hard_bans),
                "narrator_guidance": _compose_guidance(setting_kind, location_name, hard_bans),
                "suggested_alternatives": _scene_alternatives(scene_npcs, scene_items, location_features, time_phase),
                "categories": sorted(cats),
            })
            any_hard_block = True
            continue

        # (C) Soft constraints (ASK for clarification or propose grounded rewrite)
        soft_hits = {c for c in cats if c in soft_constraints_map}
        if soft_hits:
            tag = soft_constraints_map[next(iter(soft_hits))]
            logger.info(f"[FEASIBILITY] Soft constraint (ASK) -> cats={soft_hits} tag={tag}")
            per_intent.append({
                "feasible": False,
                "strategy": "ask",
                "violations": [{"rule": tag, "reason": "May not be supported here without setup"}],
                "narrator_guidance": (
                    "That might stretch this setting. Want to adapt it to what's already present, "
                    "or describe how your character attempts it within realistic bounds?"
                ),
                "suggested_alternatives": _scene_alternatives(scene_npcs, scene_items, location_features, time_phase),
                "categories": sorted(cats),
            })
            # ASK is not a hard block; we won’t set any_hard_block = True
            continue

        # (D) No issues => allow
        per_intent.append({
            "feasible": True,
            "strategy": "allow",
            "categories": sorted(cats),
        })

    overall = {"feasible": not any_hard_block, "strategy": "deny" if any_hard_block else "allow"}

    # If parse failed and we had literally no signal, prefer ASK rather than deny
    if parse_error and all((not i.get("categories") for i in intents)):
        logger.info("[FEASIBILITY] Parse failed & no categories inferred -> soft ASK")
        overall = {"feasible": False, "strategy": "ask"}
        per_intent = [{
            "feasible": False,
            "strategy": "ask",
            "violations": [{"rule": "parse_error", "reason": "Unclear intent"}],
            "narrator_guidance": "I didn’t quite follow that. Say it as a single, concrete action or break it into steps.",
            "suggested_alternatives": ["Describe one action you take", "Name an object in the scene you use"],
            "categories": []
        }]

    logger.info(f"[FEASIBILITY] overall={overall}")
    return {
        "overall": overall,
        "per_intent": per_intent
    }


def _infer_categories_from_text(text_l: str) -> Set[str]:
    """
    Very light, capability-gated hints — only used when the parser gives us nothing.
    We map obvious phrases to canonical categories the rest of the system understands.
    """
    mapping = [
        ({"cast", "spell", "ritual", "incantation"}, "spellcasting"),
        ({"teleport", "blink", "warp"}, "teleportation"),
        ({"time travel", "go back in time", "rewind"}, "time_travel"),
        ({"fly unaided", "take off myself", "levitate"}, "unaided_flight"),
        ({"spaceship", "rocket", "shuttle", "orbit", "space", "plasma", "laser"}, "spaceflight"),
        ({"summon", "conjure", "from nothing"}, "ex_nihilo_conjuration"),
        ({"mind control", "telepathy", "psychic"}, "psionics"),
        ({"reality warp", "bend physics"}, "physics_violation"),
    ]
    cats: Set[str] = set()
    for keys, cat in mapping:
        if any(k in text_l for k in keys):
            cats.add(cat)
    return cats


def _scene_alternatives(npcs: List[str], items: List[str], features: List[str], time_phase: str) -> List[str]:
    alts: List[str] = []
    if npcs:
        alts.append(f"approach {npcs[0]} for help")
    if items:
        alts.append(f"examine the {items[0]} more closely")
    if features:
        alts.append(f"investigate the {features[0]}")
    if time_phase == "night":
        alts.append("wait until dawn for better visibility")
    elif time_phase == "day":
        alts.append("survey the area for a grounded advantage")
    # keep it tight
    return alts[:3]


def _compose_guidance(setting_kind: str, location_name: Optional[str], blocking: Set[str]) -> str:
    loc = f" in {location_name}" if location_name else ""
    # Pick one dominant block to speak to
    if {"spaceflight", "orbital_travel", "spaceship_piloting"} & blocking:
        return f"This isn’t a spacefaring world{loc}; the sky stays near and no engines like that exist here."
    if {"spellcasting", "ritual_magic", "conjuration", "summoning"} & blocking:
        return f"Whatever power hums here, it isn’t magic you can wield{loc}."
    if {"physics_violation", "reality_warping", "ex_nihilo_conjuration"} & blocking:
        return f"The world keeps its seams tight{loc}; physics don’t bend that way."
    if {"unaided_flight"} & blocking:
        return f"Gravity still owns the air{loc}; you can’t take wing without help."
    if {"time_travel", "teleportation"} & blocking:
        return f"Time and distance refuse shortcuts{loc}."
    return f"This setting{loc} doesn’t support that move; try a grounded approach."
