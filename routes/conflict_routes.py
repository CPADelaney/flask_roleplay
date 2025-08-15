# routes/conflict_routes.py

"""
Routes for interacting with the open-world conflict system through the synthesizer.
Integrated with NYX governance for permission checking and action reporting.
"""

import logging
import json
from functools import wraps
from typing import Dict, Any, Optional
from quart import Blueprint, request, jsonify, session

from logic.conflict_system.conflict_synthesizer import (
    get_synthesizer,
    orchestrate_conflict_creation,
    orchestrate_scene_processing,
    orchestrate_conflict_resolution,
    get_orchestrated_system_state
)
from agents import RunContextWrapper
from db.connection import get_db_connection_context

# Import governance integration
from nyx.governance import AgentType
from nyx.governance_helpers import (
    check_permission,
    report_action,
    propose_canonical_change,
    with_governance
)
from nyx.integrate import get_central_governance

logger = logging.getLogger(__name__)

# Create Flask blueprint
conflict_bp = Blueprint("conflict", __name__)

# ----- Middleware -----

def require_login(f):
    """Decorator to require login."""
    @wraps(f)
    async def decorated(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        return await f(user_id, *args, **kwargs)
    return decorated

async def check_conflict_permission(
    user_id: int,
    conversation_id: int,
    action_type: str,
    conflict_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check with governance before performing conflict operations.
    """
    permission = await check_permission(
        user_id=user_id,
        conversation_id=conversation_id,
        agent_type=AgentType.CONFLICT_ANALYST,
        agent_id=f"conflict_system_{conversation_id}",
        action_type=action_type,
        action_details=conflict_data
    )
    return permission

# ----- Helper Functions -----

def create_context(user_id: int, conversation_id: int) -> RunContextWrapper:
    """Create a context wrapper for the given user and conversation."""
    return RunContextWrapper({
        "user_id": user_id,
        "conversation_id": conversation_id
    })

async def validate_request_data(data: Dict[str, Any], required_fields: list) -> Optional[Dict[str, Any]]:
    """Validate that required fields are present in request data."""
    missing_fields = [field for field in required_fields if not data.get(field)]
    
    if missing_fields:
        return {
            "error": f"Missing required fields: {', '.join(missing_fields)}",
            "status": 400
        }
    
    # Convert conversation_id to int if it's a string
    if "conversation_id" in data and isinstance(data["conversation_id"], str):
        try:
            data["conversation_id"] = int(data["conversation_id"])
        except ValueError:
            return {
                "error": "conversation_id must be a valid integer",
                "status": 400
            }
    
    return None

# ----- API Routes with Governance Integration -----

@conflict_bp.route("/api/conflict/create", methods=["POST"])
@require_login
async def create_conflict(user_id):
    """Create a new conflict through the synthesizer with governance approval."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        # Get conflict parameters
        conflict_type = data.get("conflict_type", "slice")
        
        # Validate conflict type (multiparty is no longer a type)
        valid_types = ["social", "slice", "background", "political", "economic", "resource", "ideological"]
        if conflict_type not in valid_types:
            return jsonify({
                "error": f"Invalid conflict type. Must be one of: {', '.join(valid_types)}"
            }), 400
        
        context_data = data.get("context", {})
        
        # Extract multiparty flag from context or participants
        participants = context_data.get("participants", [])
        is_multiparty = context_data.get("is_multiparty", len(participants) > 2)
        
        # Add multiparty metadata to context
        if is_multiparty:
            context_data["is_multiparty"] = True
            context_data["party_count"] = len(participants) if participants else context_data.get("party_count", 3)
            context_data["multiparty_dynamics"] = context_data.get("multiparty_dynamics", {
                "alliance_potential": True,
                "shifting_sides": True,
                "faction_formation": len(participants) > 4
            })
        
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="create_conflict",
            conflict_data={
                "conflict_type": conflict_type,
                "is_multiparty": is_multiparty,
                "context": context_data
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Conflict creation denied by governance: {permission.get('reasoning')}")
            return jsonify({
                "success": False,
                "error": "Governance denied conflict creation",
                "reasoning": permission.get("reasoning"),
                "governance_blocked": True
            }), 403
        
        # Apply any governance overrides
        if permission.get("override_action"):
            override = permission["override_action"]
            if "conflict_type" in override:
                conflict_type = override["conflict_type"]
            if "context" in override:
                context_data.update(override["context"])
        
        # Create context for the orchestration
        ctx = create_context(user_id, conversation_id)
        
        # Check if we should use governance's conflict system instead
        governor = await get_central_governance(user_id, conversation_id)
        if governor and hasattr(governor, 'create_conflict'):
            # Use governance's conflict creation for better integration
            result = await governor.create_conflict(
                conflict_data={
                    "name": context_data.get("name", f"{conflict_type} conflict"),
                    "conflict_type": conflict_type,
                    **context_data
                },
                reason=context_data.get("reason", "API request")
            )
            
            # Format response
            conflict_data = {
                "conflict_id": result.get("conflict_id"),
                "status": result.get("status", "created"),
                "conflict_type": conflict_type,
                "subsystem_data": result.get("subsystem_data", {})
            }
        else:
            # Fallback to direct synthesizer usage
            result = await orchestrate_conflict_creation(
                ctx,
                conflict_type=conflict_type,
                context_json=json.dumps(context_data)
            )
            
            # Parse the result
            conflict_data = json.loads(result) if isinstance(result, str) else result
        
        # Report the action to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_system_{conversation_id}",
            action={
                "type": "create_conflict",
                "conflict_type": conflict_type,
                "context": context_data
            },
            result={
                "success": True,
                "conflict_id": conflict_data.get("conflict_id")
            }
        )
        
        return jsonify({
            "success": True,
            "conflict": conflict_data,
            "governance_metadata": {
                "permission_tracking_id": permission.get("tracking_id", -1),
                "directive_applied": permission.get("directive_applied", False)
            }
        })
    except Exception as e:
        logger.error(f"Error creating conflict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/process_scene", methods=["POST"])
@require_login
async def process_scene(user_id):
    """Process a scene through the conflict system with governance oversight."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        # Get scene context
        scene_context = data.get("scene_context", {})
        
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="process_scene",
            conflict_data={
                "scene_context": scene_context
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Scene processing denied by governance: {permission.get('reasoning')}")
            return jsonify({
                "success": False,
                "error": "Governance denied scene processing",
                "reasoning": permission.get("reasoning"),
                "governance_blocked": True
            }), 403
        
        # Apply any governance overrides
        if permission.get("override_action"):
            override = permission["override_action"]
            if "scene_context" in override:
                scene_context.update(override["scene_context"])
        
        # Create context
        ctx = create_context(user_id, conversation_id)
        
        # Process scene through orchestrator
        result = await orchestrate_scene_processing(
            ctx,
            scene_context_json=json.dumps(scene_context)
        )
        
        # Parse the result
        scene_data = json.loads(result) if isinstance(result, str) else result
        
        # Report the action to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.SCENE_MANAGER,
            agent_id=f"scene_processor_{conversation_id}",
            action={
                "type": "process_scene",
                "scene_context": scene_context
            },
            result={
                "success": True,
                "conflicts_detected": scene_data.get("conflicts_detected", []),
                "events_triggered": scene_data.get("events_triggered", [])
            }
        )
        
        return jsonify({
            "success": True,
            "scene_result": scene_data,
            "governance_metadata": {
                "permission_tracking_id": permission.get("tracking_id", -1),
                "directive_applied": permission.get("directive_applied", False)
            }
        })
    except Exception as e:
        logger.error(f"Error processing scene: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/resolve", methods=["POST"])
@require_login
async def resolve_conflict(user_id):
    """Resolve a conflict through the synthesizer with governance approval."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id", "conflict_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    conflict_id = data["conflict_id"]
    
    try:
        # Get resolution parameters
        resolution_type = data.get("resolution_type", "negotiated")
        context_data = data.get("context", {})
        
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="resolve_conflict",
            conflict_data={
                "conflict_id": conflict_id,
                "resolution_type": resolution_type,
                "context": context_data
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Conflict resolution denied by governance: {permission.get('reasoning')}")
            return jsonify({
                "success": False,
                "error": "Governance denied conflict resolution",
                "reasoning": permission.get("reasoning"),
                "governance_blocked": True
            }), 403
        
        # Apply any governance overrides
        if permission.get("override_action"):
            override = permission["override_action"]
            if "resolution_type" in override:
                resolution_type = override["resolution_type"]
            if "context" in override:
                context_data.update(override["context"])
        
        # Create context
        ctx = create_context(user_id, conversation_id)
        
        # Resolve through orchestrator
        result = await orchestrate_conflict_resolution(
            ctx,
            conflict_id=conflict_id,
            resolution_type=resolution_type,
            context_json=json.dumps(context_data)
        )
        
        # Parse the result
        resolution_data = json.loads(result) if isinstance(result, str) else result
        
        # Report the action to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_resolver_{conversation_id}",
            action={
                "type": "resolve_conflict",
                "conflict_id": conflict_id,
                "resolution_type": resolution_type,
                "context": context_data
            },
            result={
                "success": resolution_data.get("resolved", False),
                "outcome": resolution_data.get("outcome"),
                "became_canonical": resolution_data.get("became_canonical", False)
            }
        )
        
        # If resolution had significant impact, record it canonically
        if resolution_data.get("became_canonical"):
            from lore.core.context import CanonicalContext
            from lore.core.canon import log_canonical_event
            
            canonical_ctx = CanonicalContext(user_id, conversation_id)
            async with get_db_connection_context() as conn:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Conflict {conflict_id} resolved via {resolution_type}",
                    tags=["conflict", "resolution", resolution_type],
                    significance=8
                )
        
        return jsonify({
            "success": True,
            "resolution": resolution_data,
            "governance_metadata": {
                "permission_tracking_id": permission.get("tracking_id", -1),
                "directive_applied": permission.get("directive_applied", False)
            }
        })
    except Exception as e:
        logger.error(f"Error resolving conflict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/system_state", methods=["GET"])
@require_login
async def get_system_state(user_id):
    """Get the overall conflict system state."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        # No governance check needed for read-only operation
        # Create context
        ctx = create_context(user_id, int(conversation_id))
        
        # Get system state through orchestrator
        result = await get_orchestrated_system_state(ctx)
        
        # Parse the result
        state_data = json.loads(result) if isinstance(result, str) else result
        
        return jsonify({
            "success": True,
            "system_state": state_data
        })
    except Exception as e:
        logger.error(f"Error getting system state: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/active", methods=["GET"])
@require_login
async def get_active_conflicts(user_id):
    """Get all active conflicts."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        # Get synthesizer
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        
        # Get system state which includes active conflicts
        state = await synthesizer.get_system_state()
        
        # Extract active conflicts
        active_conflicts = []
        for conflict_id in state.get("active_conflicts", []):
            conflict_state = await synthesizer.get_conflict_state(conflict_id)
            active_conflicts.append({
                "conflict_id": conflict_id,
                "type": conflict_state.get("conflict_type", "unknown"),
                "status": conflict_state.get("status", "active"),
                "participants": conflict_state.get("participants", []),
                "tension_level": conflict_state.get("tension", {}).get("current_level", 0),
                "phase": conflict_state.get("flow", {}).get("current_phase", "unknown")
            })
        
        return jsonify({
            "success": True,
            "conflicts": active_conflicts,
            "count": len(active_conflicts)
        })
    except Exception as e:
        logger.error(f"Error getting active conflicts: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/details/<int:conflict_id>", methods=["GET"])
@require_login
async def get_conflict_details(user_id, conflict_id):
    """Get details for a specific conflict."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        # Get synthesizer
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        
        # Get conflict state
        conflict_state = await synthesizer.get_conflict_state(conflict_id)
        
        if not conflict_state:
            return jsonify({"error": "Conflict not found"}), 404
        
        return jsonify({
            "success": True,
            "conflict": conflict_state
        })
    except Exception as e:
        logger.error(f"Error getting conflict details: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Updated conflict generation in the route
@conflict_bp.route("/api/conflict/generate", methods=["POST"])
@require_login
async def generate_emergent_conflict(user_id):
    """Generate an emergent conflict based on current world state with governance approval."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        # Check governance permission for generating conflicts
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="generate_emergent_conflict",
            conflict_data={
                "intensity": data.get("intensity", "moderate"),
                "participants": data.get("participants", []),
                "location": data.get("location", "current"),
                "trigger": data.get("trigger", "natural_progression")
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Emergent conflict generation denied by governance: {permission.get('reasoning')}")
            return jsonify({
                "success": False,
                "error": "Governance denied emergent conflict generation",
                "reasoning": permission.get("reasoning"),
                "governance_blocked": True
            }), 403
        
        # Get synthesizer
        synthesizer = await get_synthesizer(user_id, conversation_id)
        
        # Determine conflict type based on world state
        state = await synthesizer.get_system_state()
        
        # Base conflict types (WHAT the conflict is about)
        base_conflict_types = ["social", "slice", "background"]
        weights = [0.35, 0.45, 0.2]  # Favor slice-of-life conflicts
        
        import random
        conflict_type = random.choices(base_conflict_types, weights=weights)[0]
        
        # Apply governance override if present
        if permission.get("override_action") and "conflict_type" in permission["override_action"]:
            conflict_type = permission["override_action"]["conflict_type"]
        
        # Determine if this should be multiparty based on context
        participants = data.get("participants", [])
        is_multiparty = len(participants) > 2 or random.random() < 0.3  # 30% chance of multiparty
        
        # Generate context based on active events
        context = {
            "emergent": True,
            "intensity": data.get("intensity", "moderate"),
            "participants": participants,
            "location": data.get("location", "current"),
            "trigger": data.get("trigger", "natural_progression"),
            "is_multiparty": is_multiparty,  # Add as a flag
            "party_count": len(participants) if participants else (random.randint(3, 5) if is_multiparty else 2)
        }
        
        # Add type-specific context
        if conflict_type == "social":
            context["social_dynamics"] = {
                "relationship_type": random.choice(["friendship", "romantic", "family", "professional"]),
                "stakes": random.choice(["trust", "loyalty", "respect", "affection"])
            }
        elif conflict_type == "slice":
            context["slice_context"] = {
                "daily_issue": random.choice(["resources", "scheduling", "boundaries", "responsibilities"]),
                "urgency": random.choice(["low", "medium", "high"])
            }
        elif conflict_type == "background":
            context["background_context"] = {
                "scope": random.choice(["neighborhood", "district", "city", "region"]),
                "visibility": random.choice(["subtle", "noticeable", "prominent"])
            }
        
        # Create the conflict
        result = await synthesizer.create_conflict(conflict_type, context)
        
        # Report the action to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_generator_{conversation_id}",
            action={
                "type": "generate_emergent_conflict",
                "conflict_type": conflict_type,
                "is_multiparty": is_multiparty,
                "context": context
            },
            result={
                "success": True,
                "conflict_id": result.get("conflict_id")
            }
        )
        
        return jsonify({
            "success": True,
            "conflict": result,
            "governance_metadata": {
                "permission_tracking_id": permission.get("tracking_id", -1),
                "directive_applied": permission.get("directive_applied", False)
            }
        })
    except Exception as e:
        logger.error(f"Error generating emergent conflict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- Player Rewards Routes (with governance tracking) -----

@conflict_bp.route("/api/conflict/rewards/inventory", methods=["GET"])
@require_login
async def get_player_inventory(user_id):
    """Get the player's inventory items from resolved conflicts."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        async with get_db_connection_context() as conn:
            # Get inventory items
            items = await conn.fetch("""
                SELECT item_id, item_name, item_description, item_category, 
                       item_properties, quantity, equipped, date_acquired
                FROM PlayerInventory
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY date_acquired DESC
            """, user_id, int(conversation_id))
            
            formatted_items = []
            for item in items:
                # Parse JSON properties safely
                try:
                    props = json.loads(item['item_properties']) if isinstance(item['item_properties'], str) else item['item_properties'] or {}
                except (json.JSONDecodeError, TypeError):
                    props = {}
                
                formatted_items.append({
                    "item_id": item['item_id'],
                    "name": item['item_name'],
                    "description": item['item_description'],
                    "category": item['item_category'],
                    "properties": props,
                    "quantity": item['quantity'],
                    "equipped": item['equipped'],
                    "date_acquired": item['date_acquired'].isoformat() if item['date_acquired'] else None
                })
        
        return jsonify({
            "success": True,
            "inventory_items": formatted_items,
            "count": len(formatted_items)
        })
    except Exception as e:
        logger.error(f"Error getting player inventory: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/rewards/perks", methods=["GET"])
@require_login
async def get_player_perks(user_id):
    """Get the player's perks from resolved conflicts."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        async with get_db_connection_context() as conn:
            # Get perks
            perks = await conn.fetch("""
                SELECT perk_id, perk_name, perk_description, perk_category, 
                       perk_tier, perk_properties, date_acquired
                FROM PlayerPerks
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY perk_tier DESC, date_acquired DESC
            """, user_id, int(conversation_id))
            
            formatted_perks = []
            for perk in perks:
                # Parse JSON properties safely
                try:
                    props = json.loads(perk['perk_properties']) if isinstance(perk['perk_properties'], str) else perk['perk_properties'] or {}
                except (json.JSONDecodeError, TypeError):
                    props = {}
                
                formatted_perks.append({
                    "perk_id": perk['perk_id'],
                    "name": perk['perk_name'],
                    "description": perk['perk_description'],
                    "category": perk['perk_category'],
                    "tier": perk['perk_tier'],
                    "properties": props,
                    "date_acquired": perk['date_acquired'].isoformat() if perk['date_acquired'] else None
                })
        
        return jsonify({
            "success": True,
            "perks": formatted_perks,
            "count": len(formatted_perks)
        })
    except Exception as e:
        logger.error(f"Error getting player perks: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/rewards/special", methods=["GET"])
@require_login
async def get_player_special_rewards(user_id):
    """Get the player's special rewards from resolved conflicts."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        async with get_db_connection_context() as conn:
            # Get special rewards
            rewards = await conn.fetch("""
                SELECT reward_id, reward_name, reward_description, reward_effect, 
                       reward_category, reward_properties, used, date_acquired
                FROM PlayerSpecialRewards
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY date_acquired DESC
            """, user_id, int(conversation_id))
            
            formatted_rewards = []
            for reward in rewards:
                # Parse JSON properties safely
                try:
                    props = json.loads(reward['reward_properties']) if isinstance(reward['reward_properties'], str) else reward['reward_properties'] or {}
                except (json.JSONDecodeError, TypeError):
                    props = {}
                
                formatted_rewards.append({
                    "reward_id": reward['reward_id'],
                    "name": reward['reward_name'],
                    "description": reward['reward_description'],
                    "effect": reward['reward_effect'],
                    "category": reward['reward_category'],
                    "properties": props,
                    "used": reward['used'],
                    "date_acquired": reward['date_acquired'].isoformat() if reward['date_acquired'] else None
                })
        
        return jsonify({
            "success": True,
            "special_rewards": formatted_rewards,
            "count": len(formatted_rewards)
        })
    except Exception as e:
        logger.error(f"Error getting player special rewards: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/rewards/use_special", methods=["POST"])
@require_login
async def use_special_reward(user_id):
    """Use a special reward from a resolved conflict with governance tracking."""
    data = await request.get_json() or {}
    
    # Validate request
    validation_error = await validate_request_data(data, ["conversation_id", "reward_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    reward_id = data["reward_id"]
    
    try:
        # Check governance permission for using rewards
        permission = await check_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"reward_system_{conversation_id}",
            action_type="use_special_reward",
            action_details={
                "reward_id": reward_id
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Special reward usage denied by governance: {permission.get('reasoning')}")
            return jsonify({
                "success": False,
                "error": "Governance denied reward usage",
                "reasoning": permission.get("reasoning"),
                "governance_blocked": True
            }), 403
        
        async with get_db_connection_context() as conn:
            # Check if reward exists and belongs to user
            reward = await conn.fetchrow("""
                SELECT reward_name, reward_effect, used
                FROM PlayerSpecialRewards
                WHERE reward_id = $1 AND user_id = $2 AND conversation_id = $3
            """, reward_id, user_id, conversation_id)
            
            if not reward:
                return jsonify({
                    "error": "Special reward not found or doesn't belong to this user"
                }), 404
            
            # Check if already used
            if reward['used']:
                return jsonify({
                    "success": False,
                    "message": "This special reward has already been used"
                }), 400
            
            # Mark as used
            await conn.execute("""
                UPDATE PlayerSpecialRewards
                SET used = TRUE
                WHERE reward_id = $1
            """, reward_id)
        
        # Report the action to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"reward_system_{conversation_id}",
            action={
                "type": "use_special_reward",
                "reward_id": reward_id,
                "reward_name": reward['reward_name']
            },
            result={
                "success": True,
                "reward_effect": reward['reward_effect']
            }
        )
        
        # Log canonically if significant
        if reward['reward_effect'] and 'permanent' in reward['reward_effect'].lower():
            from lore.core.context import CanonicalContext
            from lore.core.canon import log_canonical_event
            
            canonical_ctx = CanonicalContext(user_id, conversation_id)
            async with get_db_connection_context() as conn:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Player used special reward: {reward['reward_name']}",
                    tags=["reward", "conflict", "player_action"],
                    significance=6
                )
        
        # Return the effect that should be applied
        return jsonify({
            "success": True,
            "message": f"Successfully used special reward: {reward['reward_name']}",
            "reward_name": reward['reward_name'],
            "reward_effect": reward['reward_effect'],
            "governance_metadata": {
                "permission_tracking_id": permission.get("tracking_id", -1),
                "directive_applied": permission.get("directive_applied", False)
            }
        })
    except Exception as e:
        logger.error(f"Error using special reward: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
