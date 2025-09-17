# lore/core/gate_integration.py
"""
Integration points for the existence gate system.
Hooks into all entity creation paths to ensure reality consistency.
"""

import logging
from typing import Dict, Any, Optional
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# Import the gate components
from .existence_gate import ExistenceGate, GateDecision
from .physics_validator import PhysicsValidator, quick_physics_check

async def setup_world_rules(user_id: int, conversation_id: int, setting_profile: str = 'realistic'):
    """
    Initialize world rules and physics caps for a conversation.
    Should be called during world creation or setting initialization.
    """
    profiles = {
        'realistic': {
            'physics': {
                'max_jump_height_m': 1.5,
                'max_throw_speed_ms': 45,
                'max_safe_fall_m': 3,
                'max_carry_weight_kg': 100,
                'max_running_speed_ms': 12,
                'teleportation_allowed': False,
                'time_travel_allowed': False,
                'ex_nihilo_creation': False,
                'gravity_constant': 9.8
            },
            'infrastructure': {
                'electricity': True,
                'refrigeration': True,
                'mass_packaging': True,
                'night_economy': True,
                'plumbing': True,
                'printing': True,
                'global_trade': True,
                'instant_communication': True
            },
            'technology': 'modern',
            'era': 'contemporary'
        },
        'fantasy_medieval': {
            'physics': {
                'max_jump_height_m': 2,
                'max_throw_speed_ms': 50,
                'max_safe_fall_m': 4,
                'max_carry_weight_kg': 120,
                'max_running_speed_ms': 15,
                'teleportation_allowed': False,
                'time_travel_allowed': False,
                'ex_nihilo_creation': False,
                'gravity_constant': 9.8,
                'magic_system': 'limited'
            },
            'infrastructure': {
                'electricity': False,
                'refrigeration': False,
                'mass_packaging': False,
                'night_economy': False,
                'plumbing': False,
                'printing': False,
                'global_trade': False,
                'instant_communication': False,
                'magic_communication': True
            },
            'technology': 'medieval',
            'era': 'medieval'
        },
        'high_fantasy': {
            'physics': {
                'max_jump_height_m': 50,
                'max_throw_speed_ms': 200,
                'max_safe_fall_m': 100,
                'max_carry_weight_kg': 1000,
                'max_running_speed_ms': 50,
                'teleportation_allowed': True,
                'time_travel_allowed': False,
                'ex_nihilo_creation': True,
                'gravity_constant': 9.8,
                'magic_system': 'extensive',
                'spell_resource': 'mana'
            },
            'infrastructure': {
                'electricity': False,
                'refrigeration': True,  # Magic ice
                'mass_packaging': False,
                'night_economy': True,  # Magic lights
                'plumbing': True,  # Magic water
                'printing': True,
                'global_trade': True,
                'instant_communication': True,  # Scrying
                'magic_infrastructure': True
            },
            'technology': 'magical',
            'era': 'timeless'
        },
        'cyberpunk': {
            'physics': {
                'max_jump_height_m': 5,
                'max_throw_speed_ms': 80,
                'max_safe_fall_m': 8,
                'max_carry_weight_kg': 300,
                'max_running_speed_ms': 20,
                'teleportation_allowed': False,
                'time_travel_allowed': False,
                'ex_nihilo_creation': False,
                'gravity_constant': 9.8,
                'tech_requirement': 'advanced',
                'cybernetics_allowed': True
            },
            'infrastructure': {
                'electricity': True,
                'refrigeration': True,
                'mass_packaging': True,
                'night_economy': True,
                'plumbing': True,
                'printing': True,
                'global_trade': True,
                'instant_communication': True,
                'ai_systems': True,
                'neural_networks': True,
                'augmented_reality': True
            },
            'technology': 'advanced',
            'era': 'near_future'
        },
        'space_opera': {
            'physics': {
                'max_jump_height_m': 10,
                'max_throw_speed_ms': 100,
                'max_safe_fall_m': 15,
                'max_carry_weight_kg': 500,
                'max_running_speed_ms': 25,
                'teleportation_allowed': True,
                'time_travel_allowed': False,
                'ex_nihilo_creation': False,
                'gravity_constant': 'variable',
                'tech_requirement': 'futuristic',
                'ftl_travel': True
            },
            'infrastructure': {
                'electricity': True,
                'refrigeration': True,
                'mass_packaging': True,
                'night_economy': True,
                'plumbing': True,
                'printing': True,
                'global_trade': True,
                'instant_communication': True,
                'quantum_computing': True,
                'ai_systems': True,
                'nano_fabrication': True,
                'matter_replication': True,
                'interstellar_travel': True
            },
            'technology': 'futuristic',
            'era': 'far_future'
        }
    }
    
    profile_data = profiles.get(setting_profile, profiles['realistic'])
    
    async with get_db_connection_context() as conn:
        # Store physics profile
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'PhysicsProfile', $3)
            ON CONFLICT (user_id, conversation_id, key) 
            DO UPDATE SET value = EXCLUDED.value
        """, user_id, conversation_id, setting_profile)
        
        # Store physics caps
        import json
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'PhysicsCaps', $3)
            ON CONFLICT (user_id, conversation_id, key) 
            DO UPDATE SET value = EXCLUDED.value
        """, user_id, conversation_id, json.dumps(profile_data['physics']))
        
        # Store infrastructure flags
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'InfrastructureFlags', $3)
            ON CONFLICT (user_id, conversation_id, key) 
            DO UPDATE SET value = EXCLUDED.value
        """, user_id, conversation_id, json.dumps(profile_data['infrastructure']))
        
        # Store technology level
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'TechnologyLevel', $3)
            ON CONFLICT (user_id, conversation_id, key) 
            DO UPDATE SET value = EXCLUDED.value
        """, user_id, conversation_id, profile_data['technology'])
        
        # Store era
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'SettingEra', $3)
            ON CONFLICT (user_id, conversation_id, key) 
            DO UPDATE SET value = EXCLUDED.value
        """, user_id, conversation_id, profile_data['era'])
        
        logger.info(f"Initialized world rules for conversation {conversation_id} with profile: {setting_profile}")


async def validate_entity_creation(
    ctx,
    entity_type: str,
    entity_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main validation function called before any entity creation.
    
    Returns:
        {
            'allowed': bool,
            'decision': GateDecision,
            'details': dict with reasoning,
            'modified_data': dict with any transformations
        }
    """
    gate = ExistenceGate(ctx)
    
    # Get current scene context
    async with get_db_connection_context() as conn:
        scene_row = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentScene'
        """, ctx.user_id, ctx.conversation_id)
    
    scene_context = None
    if scene_row and scene_row['value']:
        import json
        scene_context = json.loads(scene_row['value'])
    
    # Run assessment
    decision, details = await gate.assess_entity(entity_type, entity_data, scene_context)
    
    # Prepare response
    result = {
        'allowed': decision == GateDecision.ALLOW,
        'decision': decision,
        'details': details,
        'modified_data': entity_data.copy()
    }
    
    # Handle transformations
    if decision == GateDecision.ANALOG:
        result['modified_data'] = details.get('analog_data', entity_data)
        result['analog_used'] = details.get('analog')
    
    return result


async def validate_action_physics(
    ctx,
    action_type: str,
    parameters: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate an action against physics caps.
    
    Returns:
        {
            'valid': bool,
            'violations': list of violations,
            'alternative': suggested alternative if invalid,
            'resources_required': resources needed to make valid
        }
    """
    # Load physics caps
    async with get_db_connection_context() as conn:
        caps_row = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='PhysicsCaps'
        """, ctx.user_id, ctx.conversation_id)
    
    if caps_row and caps_row['value']:
        import json
        caps = json.loads(caps_row['value'])
    else:
        # Default realistic caps
        caps = {
            'max_jump_height_m': 1.5,
            'max_throw_speed_ms': 45,
            'max_safe_fall_m': 3,
            'max_carry_weight_kg': 100,
            'max_running_speed_ms': 12,
            'teleportation_allowed': False,
            'time_travel_allowed': False,
            'ex_nihilo_creation': False,
            'gravity_constant': 9.8
        }
    
    validator = PhysicsValidator(caps)
    
    # Validate action
    valid, violations = validator.validate_action(action_type, parameters, context)
    
    result = {
        'valid': valid,
        'violations': [
            {
                'type': v.violation_type.value,
                'description': v.description,
                'severity': v.severity,
                'consequence': v.consequence
            }
            for v in violations
        ]
    }
    
    # Get alternative if invalid
    if not valid:
        result['alternative'] = validator.suggest_alternative(
            action_type, parameters, violations
        )
        
        # Check resource requirements
        resources = validator.check_resource_requirement(action_type, parameters)
        if resources:
            result['resources_required'] = resources
    
    return result


async def log_gate_decision(
    ctx,
    entity_type: str,
    decision: GateDecision,
    details: Dict[str, Any]
):
    """Log gate decisions for debugging and learning."""
    async with get_db_connection_context() as conn:
        import json
        from datetime import datetime
        
        await conn.execute("""
            INSERT INTO GateDecisions (
                user_id, conversation_id, entity_type, decision,
                details, timestamp
            )
            VALUES ($1, $2, $3, $4, $5, $6)
        """, ctx.user_id, ctx.conversation_id, entity_type,
            decision.value, json.dumps(details), datetime.utcnow())
        
        # Also log as canonical event if it was a denial
        if decision == GateDecision.DENY:
            from .canon import log_canonical_event
            await log_canonical_event(
                ctx, conn,
                f"Reality prevented creation of {entity_type}: {details.get('reason', 'Unknown')}",
                tags=['gate', 'denial', entity_type],
                significance=3
            )


async def check_timeline_consistency(
    ctx,
    event_data: Dict[str, Any]
) -> bool:
    """
    Check if an event would violate timeline consistency.
    
    Returns:
        True if consistent, False if it would create paradox
    """
    async with get_db_connection_context() as conn:
        from datetime import datetime
        import json
        
        # Get all future events
        future_events = await conn.fetch("""
            SELECT id, event_text, tags, timestamp
            FROM CanonicalEvents
            WHERE user_id=$1 AND conversation_id=$2
            AND timestamp > $3
            AND significance >= 5
            ORDER BY timestamp
        """, ctx.user_id, ctx.conversation_id, 
            event_data.get('timestamp', datetime.utcnow()))
        
        # Check each future event for dependencies
        for future in future_events:
            future_text = future['event_text'].lower()
            
            # Check if event would prevent future event
            if 'kills' in event_data:
                if event_data['kills'].lower() in future_text:
                    logger.warning(f"Event would create paradox - {event_data['kills']} needed for future event")
                    return False
            
            if 'destroys' in event_data:
                if event_data['destroys'].lower() in future_text:
                    logger.warning(f"Event would create paradox - {event_data['destroys']} needed for future event")
                    return False
            
            if 'prevents' in event_data:
                tags = json.loads(future['tags']) if future['tags'] else []
                if any(tag in event_data['prevents'] for tag in tags):
                    logger.warning(f"Event would prevent future event with tags: {tags}")
                    return False
        
        return True


async def enforce_resource_conservation(
    ctx,
    resource_type: str,
    amount_change: float,
    source: str
) -> bool:
    """
    Enforce conservation laws for resources (matter, energy, currency).
    
    Returns:
        True if change is valid, False if it violates conservation
    """
    # Check if creating resources from nothing
    if amount_change > 0 and source == 'ex_nihilo':
        # Check if ex nihilo creation is allowed
        async with get_db_connection_context() as conn:
            caps_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='PhysicsCaps'
            """, ctx.user_id, ctx.conversation_id)
        
        if caps_row and caps_row['value']:
            import json
            caps = json.loads(caps_row['value'])
            if not caps.get('ex_nihilo_creation', False):
                logger.warning(f"Cannot create {resource_type} from nothing")
                return False
    
    # Check if total resources exceed reasonable limits
    if resource_type == 'currency':
        # Check economy constraints
        pass  # Implement economy checks
    
    return True


# Hook for the orchestrator to call during processing
async def apply_existence_gate(ctx, tool_name: str, tool_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Hook called by orchestrator before entity creation tools.
    
    Returns:
        None if allowed, or error dict if blocked
    """
    # Map tool names to entity types
    entity_map = {
        'find_or_create_location': 'location',
        'find_or_create_npc': 'npc',
        'find_or_create_inventory_item': 'item',
        'log_event_creation': 'event',
        'create_magical_effect': 'magic',
        'spawn_creature': 'creature'
    }
    
    entity_type = entity_map.get(tool_name)
    if not entity_type:
        return None  # Not a creation tool
    
    # Validate the creation
    validation = await validate_entity_creation(ctx, entity_type, tool_args)
    
    # Log the decision
    await log_gate_decision(ctx, entity_type, validation['decision'], validation['details'])
    
    if validation['allowed']:
        # Transform arguments if analog was suggested
        if 'analog_used' in validation:
            return validation['modified_data']
        return None
    
    # Block the creation
    return {
        'error': 'ExistenceGateViolation',
        'message': validation['details'].get('reason', 'Entity cannot exist'),
        'decision': validation['decision'].value,
        'details': validation['details']
    }


# Table creation for logging
async def create_gate_tables(conn):
    """Create tables for gate decision logging."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS GateDecisions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            entity_type VARCHAR(50),
            decision VARCHAR(20),
            details JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_gate_decisions_lookup
        ON GateDecisions(user_id, conversation_id, entity_type)
    """)
