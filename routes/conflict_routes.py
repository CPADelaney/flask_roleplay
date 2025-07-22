# routes/conflict_routes.py

"""
Routes for interacting with the conflict system through Nyx governance.
"""

import logging
import json
from functools import wraps
from typing import Dict, Any
from quart import Blueprint, request, jsonify, session

from agents import RunContextWrapper  # Added import for context wrapper
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from logic.conflict_system.conflict_integration import ConflictSystemIntegration
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# Create Flask blueprint
conflict_bp = Blueprint("conflict", __name__)

# ----- Middleware -----

def require_login(f):
    """Decorator to require login."""
    @wraps(f)  # This preserves the original function's name and attributes
    async def decorated(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        return await f(user_id, *args, **kwargs)
    return decorated

# ----- API Routes -----

@conflict_bp.route("/api/conflict/register", methods=["POST"])
@require_login
async def register_conflict_system(user_id):
    """Register the conflict system with Nyx governance."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        # Register the enhanced conflict system
        result = await register_enhanced_integration(user_id, int(conversation_id))
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error registering conflict system: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/generate", methods=["POST"])
@require_login
async def generate_conflict(user_id):
    """Generate a new conflict."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Get the conflict system agent
        conflict_system = governance.registered_agents.get(AgentType.CONFLICT_ANALYST)
        
        if not conflict_system:
            return jsonify({
                "error": "Conflict system not registered", 
                "message": "Please register the conflict system first"
            }), 400
        
        # Generate the conflict
        conflict_data = data.get("conflict_data", {})
        
        # Create context wrapper with user and conversation IDs
        conflict_ctx = RunContextWrapper({
            "user_id": user_id, 
            "conversation_id": int(conversation_id)
        })
        
        # Pass ctx to generate_conflict
        result = await conflict_system.generate_conflict(conflict_data, ctx=conflict_ctx)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error generating conflict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/resolve", methods=["POST"])
@require_login
async def resolve_conflict(user_id):
    """Resolve an existing conflict."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    conflict_id = data.get("conflict_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    if not conflict_id:
        return jsonify({"error": "Missing conflict_id parameter"}), 400
    
    try:
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Get the conflict system agent
        conflict_system = governance.registered_agents.get(AgentType.CONFLICT_ANALYST)
        
        if not conflict_system:
            return jsonify({
                "error": "Conflict system not registered", 
                "message": "Please register the conflict system first"
            }), 400
        
        # Create context wrapper
        conflict_ctx = RunContextWrapper({
            "user_id": user_id, 
            "conversation_id": int(conversation_id)
        })
        
        # Resolve the conflict
        resolution_data = data.get("resolution_data", {})
        resolution_data["conflict_id"] = conflict_id
        
        # Pass ctx to resolve_conflict
        result = await conflict_system.resolve_conflict(resolution_data, ctx=conflict_ctx)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error resolving conflict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/update_stakeholders", methods=["POST"])
@require_login
async def update_stakeholders(user_id):
    """Update stakeholders for an existing conflict."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    conflict_id = data.get("conflict_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    if not conflict_id:
        return jsonify({"error": "Missing conflict_id parameter"}), 400
    
    try:
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Get the conflict system agent
        conflict_system = governance.registered_agents.get(AgentType.CONFLICT_ANALYST)
        
        if not conflict_system:
            return jsonify({
                "error": "Conflict system not registered", 
                "message": "Please register the conflict system first"
            }), 400
        
        # Create context wrapper
        conflict_ctx = RunContextWrapper({
            "user_id": user_id, 
            "conversation_id": int(conversation_id)
        })
        
        # Update stakeholders
        stakeholder_data = data.get("stakeholder_data", {})
        stakeholder_data["conflict_id"] = conflict_id
        
        # Pass ctx to update_stakeholders
        result = await conflict_system.update_stakeholders(stakeholder_data, ctx=conflict_ctx)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error updating stakeholders: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/manipulate", methods=["POST"])
@require_login
async def manage_manipulation(user_id):
    """Manage a manipulation attempt in a conflict."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    conflict_id = data.get("conflict_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    if not conflict_id:
        return jsonify({"error": "Missing conflict_id parameter"}), 400
    
    try:
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Get the conflict system agent
        conflict_system = governance.registered_agents.get(AgentType.CONFLICT_ANALYST)
        
        if not conflict_system:
            return jsonify({
                "error": "Conflict system not registered", 
                "message": "Please register the conflict system first"
            }), 400
        
        # Create context wrapper
        conflict_ctx = RunContextWrapper({
            "user_id": user_id, 
            "conversation_id": int(conversation_id)
        })
        
        # Manage manipulation
        manipulation_data = data.get("manipulation_data", {})
        manipulation_data["conflict_id"] = conflict_id
        
        # Pass ctx to manage_manipulation
        result = await conflict_system.manage_manipulation(manipulation_data, ctx=conflict_ctx)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error managing manipulation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/issue_directive", methods=["POST"])
@require_login
async def issue_conflict_directive(user_id):
    """Issue a directive to the conflict system."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Prepare directive data
        directive_type = data.get("directive_type", DirectiveType.ACTION_REQUEST)
        directive_data = data.get("directive_data", {})
        priority = data.get("priority", DirectivePriority.MEDIUM)
        duration_minutes = data.get("duration_minutes", 60)
        
        # Issue directive (no context needed here as we're using governance directly)
        directive_id = await governance.governor.issue_directive(
            agent_type=AgentType.CONFLICT_ANALYST,
            directive_type=directive_type,
            directive_data=directive_data,
            priority=priority,
            duration_minutes=duration_minutes
        )
        
        return jsonify({
            "success": True,
            "directive_id": directive_id,
            "message": "Directive issued to conflict system"
        })
    except Exception as e:
        logger.error(f"Error issuing directive: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- Player Rewards Routes -----

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
                    props = item['item_properties'] if isinstance(item['item_properties'], dict) else {}
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
                    props = perk['perk_properties'] if isinstance(perk['perk_properties'], dict) else {}
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
                    props = reward['reward_properties'] if isinstance(reward['reward_properties'], dict) else {}
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
    """Use a special reward from a resolved conflict."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    reward_id = data.get("reward_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    if not reward_id:
        return jsonify({"error": "Missing reward_id parameter"}), 400
    
    try:
        async with get_db_connection_context() as conn:
            # Check if reward exists and belongs to user
            reward = await conn.fetchrow("""
                SELECT reward_name, reward_effect, used
                FROM PlayerSpecialRewards
                WHERE reward_id = $1 AND user_id = $2 AND conversation_id = $3
            """, reward_id, user_id, int(conversation_id))
            
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
        
        # Return the effect that should be applied
        return jsonify({
            "success": True,
            "message": f"Successfully used special reward: {reward['reward_name']}",
            "reward_name": reward['reward_name'],
            "reward_effect": reward['reward_effect']
        })
    except Exception as e:
        logger.error(f"Error using special reward: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
