# nyx/governance/story.py
"""
Story and narrative management functionality.
"""
import logging
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from db.connection import get_db_connection_context
from agents import RunContextWrapper

logger = logging.getLogger(__name__)


class StoryGovernanceMixin:
    """Handles story and narrative-related governance functions."""
    
    async def orchestrate_narrative_shift(self, reason: str, shift_type: str = "local", shift_details: Optional[Dict[str, Any]] = None):
        """
        Orchestrate a narrative shift at any scale.
        
        Args:
            reason: Why the shift is happening
            shift_type: Scale of shift ("personal", "local", "regional", "national", "global")
            shift_details: Optional details to customize the shift, including:
                - target_entities: Specific entities to affect
                - change_type: Type of change to make
                - custom_changes: Specific changes to apply
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Orchestrating a {shift_type} narrative shift because: {reason}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        shift_details = shift_details or {}
        results = []
    
        # Different logic based on shift type
        if shift_type == "personal":
            # Personal-scale shift: individual character changes
            # Examples: someone gets a new job, relationship changes, personal growth
            
            # Use provided target or select an NPC based on narrative needs
            if "target_entities" in shift_details:
                npc_names = shift_details["target_entities"]
            else:
                # Example: Select an NPC who needs development
                npc_names = ["Sarah Chen"]  # In real implementation, this would be dynamic
            
            for npc_name in npc_names:
                npc_id = await self._get_npc_id_by_name(npc_name)
                if not npc_id:
                    logger.warning(f"NPC '{npc_name}' not found for personal shift")
                    continue
                    
                # Determine what kind of personal change
                change_type = shift_details.get("change_type", "growth")
                
                if change_type == "growth":
                    updates = {
                        "confidence": min(100, await self._get_npc_stat(npc_id, "confidence", 50) + 10),
                        "personality_traits": await self._add_personality_trait(npc_id, "determined")
                    }
                    narrative_reason = f"{reason} {npc_name} has grown more confident."
                    
                elif change_type == "location":
                    new_location = shift_details.get("new_location", "University Library")
                    updates = {
                        "current_location": new_location,
                        "schedule": shift_details.get("new_schedule", {"morning": "studying", "afternoon": "working"})
                    }
                    narrative_reason = f"{reason} {npc_name} now spends most of their time at {new_location}."
                    
                elif change_type == "relationship":
                    updates = shift_details.get("custom_changes", {})
                    narrative_reason = f"{reason} {npc_name}'s relationships have shifted."
                    
                else:
                    updates = shift_details.get("custom_changes", {})
                    narrative_reason = f"{reason} {npc_name} has experienced a personal change."
                
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NPCStats",
                    entity_identifier={"npc_id": npc_id},
                    updates=updates,
                    reason=narrative_reason
                )
                results.append(result)
                
        elif shift_type == "local":
            # Local-scale shift: community changes, local groups, small businesses
            # Examples: store closes, new club forms, local election
            
            change_type = shift_details.get("change_type", "community_change")
            
            if change_type == "business_closure":
                location_name = shift_details.get("location", "Corner Coffee Shop")
                # Update location status
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        UPDATE Locations 
                        SET description = description || ' (CLOSED)',
                            open_hours = '{}'::jsonb
                        WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                    """, location_name, self.user_id, self.conversation_id)
                
                # Create a conflict about it
                conflict_result = await self.create_conflict({
                    "name": f"{location_name} Closure Controversy",
                    "conflict_type": "economic",
                    "scale": "local",
                    "description": f"Local residents are upset about {location_name} closing down",
                    "involved_parties": [
                        {"type": "location", "name": location_name, "stake": "closing"},
                        {"type": "faction", "name": "Local Business Association", "stance": "concerned"}
                    ],
                    "stakes": "community gathering place"
                }, reason)
                results.append(conflict_result)
                
            elif change_type == "new_group":
                # Create a new local faction/group
                group_name = shift_details.get("group_name", "Community Gardeners")
                group_location = shift_details.get("location", "Riverside Park")
                
                faction_id = await self._get_faction_id_by_name(group_name)
                if not faction_id:
                    from lore.core import canon
                    async with get_db_connection_context() as conn:
                        faction_id = await canon.find_or_create_faction(
                            ctx, conn, 
                            faction_name=group_name,
                            type=shift_details.get("faction_type", "community"),
                            description=shift_details.get("description", f"A local {shift_details.get('faction_type', 'community')} group"),
                            influence_scope="local",
                            power_level=2
                        )
                
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="Factions",
                    entity_identifier={"id": faction_id},
                    updates={
                        "territory": group_location,
                        "influence_scope": "neighborhood",
                        "recruitment_methods": ["word of mouth", "community board"]
                    },
                    reason=f"{reason} The {group_name} have established themselves at {group_location}."
                )
                results.append(result)
                
            elif change_type == "local_development":
                # Changes to local infrastructure or community
                location = shift_details.get("location", "Downtown")
                development = shift_details.get("development", "new community center")
                
                # This might affect multiple entities
                affected_factions = shift_details.get("affected_factions", ["Local Business Association"])
                for faction_name in affected_factions:
                    faction_id = await self._get_faction_id_by_name(faction_name)
                    if faction_id:
                        result = await self.lore_system.propose_and_enact_change(
                            ctx=ctx,
                            entity_type="Factions",
                            entity_identifier={"id": faction_id},
                            updates={"resources": [development]},
                            reason=f"{reason} {faction_name} now has access to {development}."
                        )
                        results.append(result)
                        
        elif shift_type == "regional":
            # Regional-scale shift: multiple communities affected
            # Examples: weather event, economic downturn, cultural movement
            
            change_type = shift_details.get("change_type", "cultural_shift")
            affected_regions = shift_details.get("affected_regions", [])
            
            if change_type == "cultural_shift":
                cultural_change = shift_details.get("cultural_change", {
                    "new_traits": ["environmentally conscious", "community-oriented"],
                    "values_shift": "towards sustainability"
                })
                
                for region_name in affected_regions:
                    # Update regional culture
                    # Note: This assumes you have a Regions or GeographicRegions table
                    result = await self.lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="Locations",  # or "GeographicRegions" if you have that
                        entity_identifier={"location_name": region_name},
                        updates={
                            "cultural_significance": cultural_change.get("values_shift", "shifting values"),
                            "local_customs": cultural_change.get("new_traits", [])
                        },
                        reason=f"{reason} {region_name} is experiencing a cultural shift {cultural_change.get('values_shift', '')}."
                    )
                    results.append(result)
                    
        elif shift_type in ["national", "global"]:
            # Large-scale shifts: nations, international relations
            # This uses your original logic
            
            if shift_type == "national":
                # National change affecting one nation
                nation_name = shift_details.get("nation", "Example Nation")
                nation_id = await self._get_nation_id_by_name(nation_name)
                
                if nation_id:
                    updates = shift_details.get("custom_changes", {
                        "government_type": "reformed democracy",
                        "matriarchy_level": 7
                    })
                    
                    result = await self.lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="nations",
                        entity_identifier={"id": nation_id},
                        updates=updates,
                        reason=f"{reason} {nation_name} has undergone significant political reform."
                    )
                    results.append(result)
                    
            else:  # global
                # Global changes affecting multiple nations or the whole world
                # Example: technological breakthrough, climate event, pandemic
                
                change_type = shift_details.get("change_type", "political")
                
                if change_type == "political":
                    # Original example: faction gains territory
                    faction_name = shift_details.get("faction", "The Matriarchal Council")
                    new_territory = shift_details.get("territory", "The Sunken City")
                    
                    faction_id = await self._get_faction_id_by_name(faction_name)
                    if not faction_id:
                        from lore.core import canon
                        async with get_db_connection_context() as conn:
                            faction_id = await canon.find_or_create_faction(
                                ctx, conn, 
                                faction_name=faction_name,
                                type="political",
                                description=f"A powerful faction seeking to control {new_territory}",
                                influence_scope="global",
                                power_level=8
                            )
                    
                    result = await self.lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="Factions",
                        entity_identifier={"id": faction_id},
                        updates={"territory": new_territory},
                        reason=f"{reason} {faction_name} has gained control of {new_territory}."
                    )
                    results.append(result)
                    
                elif change_type == "technological":
                    # Global tech advancement
                    advancement = shift_details.get("advancement", "renewable energy breakthrough")
                    affected_nations = shift_details.get("affected_nations", [])
                    
                    for nation_name in affected_nations:
                        nation_id = await self._get_nation_id_by_name(nation_name)
                        if nation_id:
                            result = await self.lore_system.propose_and_enact_change(
                                ctx=ctx,
                                entity_type="nations",
                                entity_identifier={"id": nation_id},
                                updates={"technology_level": 8},
                                reason=f"{reason} {nation_name} has adopted {advancement}."
                            )
                            results.append(result)
    
        # Record the narrative shift as an event
        significance_map = {
            "personal": 3,
            "local": 5,
            "regional": 7,
            "national": 9,
            "global": 10
        }
        
        await self._record_narrative_event(
            event_type=f"{shift_type}_narrative_shift",
            details={
                "shift_type": shift_type,
                "reason": reason,
                "changes_made": len(results),
                "shift_details": shift_details,
                "results": results
            }
        )
    
        logger.info(f"NYX: {shift_type} narrative shift completed with {len(results)} changes.")
        return {
            "status": "completed",
            "shift_type": shift_type,
            "changes_made": len(results),
            "results": results
        }

    async def _would_disrupt_plot(self, action_type: str, action_details: Dict[str, Any], 
                                 narrative_context: Dict[str, Any]) -> bool:
        """Check if an action would disrupt the current plot using actual game data."""
        try:
            async with get_db_connection_context() as conn:
                # Check if action affects active quests
                if action_type in ['abandon_quest', 'fail_quest']:
                    quest_name = action_details.get('quest_name', '')
                    active_quest = await conn.fetchval("""
                        SELECT quest_id FROM Quests
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND quest_name = $3 AND status = 'In Progress'
                    """, self.user_id, self.conversation_id, quest_name)
                    if active_quest:
                        return True
                
                # Check if killing a quest-critical NPC
                if action_type == 'kill_npc':
                    target_npc = action_details.get('target', '')
                    # Check if NPC is a quest giver for active quests
                    is_quest_giver = await conn.fetchval("""
                        SELECT COUNT(*) FROM Quests
                        WHERE user_id = $1 AND conversation_id = $2
                        AND quest_giver = $3 AND status = 'In Progress'
                    """, self.user_id, self.conversation_id, target_npc)
                    if is_quest_giver > 0:
                        return True
                    
                    # Check if NPC is involved in active conflicts
                    is_conflict_stakeholder = await conn.fetchval("""
                        SELECT COUNT(*) FROM ConflictStakeholders cs
                        JOIN Conflicts c ON cs.conflict_id = c.conflict_id
                        JOIN NPCStats n ON cs.npc_id = n.npc_id
                        WHERE c.user_id = $1 AND c.conversation_id = $2
                        AND n.npc_name = $3 AND c.is_active = TRUE
                    """, self.user_id, self.conversation_id, target_npc)
                    if is_conflict_stakeholder > 0:
                        return True
                
                # Check if destroying a location that's narratively important
                if action_type == 'destroy_location':
                    location_name = action_details.get('target', '')
                    # Check if location has high cultural significance
                    significance = await conn.fetchval("""
                        SELECT cultural_significance FROM Locations
                        WHERE user_id = $1 AND conversation_id = $2
                        AND location_name = $3
                    """, self.user_id, self.conversation_id, location_name)
                    if significance in ['high', 'critical', 'sacred']:
                        return True
                    
                    # Check if location is tied to active events
                    has_events = await conn.fetchval("""
                        SELECT COUNT(*) FROM Events
                        WHERE user_id = $1 AND conversation_id = $2
                        AND location = $3 AND end_time > CURRENT_TIMESTAMP
                    """, self.user_id, self.conversation_id, location_name)
                    if has_events > 0:
                        return True
                
                # Check if action would skip locked content
                if action_type == 'travel':
                    destination = action_details.get('destination', '')
                    # Check access restrictions
                    restrictions = await conn.fetchval("""
                        SELECT access_restrictions FROM Locations
                        WHERE user_id = $1 AND conversation_id = $2
                        AND location_name = $3
                    """, self.user_id, self.conversation_id, destination)
                    if restrictions and len(restrictions) > 0:
                        # Check if player meets requirements
                        # This would need more complex logic based on your access system
                        return True
                        
            return False
        except Exception as e:
            logger.error(f"Error checking plot disruption: {e}")
            return False

    async def _would_affect_pacing(self, action_type: str, action_details: Dict[str, Any],
                                  narrative_context: Dict[str, Any]) -> bool:
        """Check if an action would negatively affect story pacing."""
        try:
            async with get_db_connection_context() as conn:
                # Get current narrative stage
                current_stage = await conn.fetchval("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'NarrativeStage'
                """, self.user_id, self.conversation_id)
                
                # Check recent major events
                recent_events = await conn.fetch("""
                    SELECT ce.event_text, ce.significance, ce.timestamp
                    FROM CanonicalEvents ce
                    WHERE ce.user_id = $1 AND ce.conversation_id = $2
                    AND ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                    AND ce.significance >= 7
                    ORDER BY ce.timestamp DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                
                # If many high-significance events recently, another major action might rush pacing
                if len(recent_events) >= 3 and action_details.get('significance', 5) >= 7:
                    return True
                
                # Check if action type matches narrative stage expectations
                stage_pacing_map = {
                    'introduction': ['explore', 'talk', 'observe'],
                    'rising_action': ['quest', 'conflict', 'relationship'],
                    'climax': ['confront', 'resolve', 'decide'],
                    'falling_action': ['aftermath', 'reconcile', 'rebuild'],
                    'resolution': ['reflect', 'celebrate', 'depart']
                }
                
                expected_actions = stage_pacing_map.get(current_stage, [])
                if action_type not in expected_actions and current_stage in stage_pacing_map:
                    # Action doesn't match narrative stage
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error checking pacing impact: {e}")
            return False

    async def _maintains_thematic_consistency(self, action_type: str, action_details: Dict[str, Any],
                                             narrative_context: Dict[str, Any]) -> bool:
        """Check if an action maintains thematic consistency with the setting and story."""
        try:
            async with get_db_connection_context() as conn:
                # Get current setting
                setting_name = await conn.fetchval("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSetting'
                """, self.user_id, self.conversation_id)
                
                if setting_name:
                    # Get setting rules and themes
                    setting_data = await conn.fetchrow("""
                        SELECT mood_tone, enhanced_features
                        FROM Settings
                        WHERE name = $1
                    """, setting_name)
                    
                    if setting_data:
                        mood_tone = setting_data['mood_tone']
                        features = json.loads(setting_data['enhanced_features']) if setting_data['enhanced_features'] else {}
                        
                        # Check if action conflicts with setting tone
                        tone_conflicts = {
                            'lighthearted': ['murder', 'torture', 'betray_deeply'],
                            'serious': ['joke_inappropriately', 'break_fourth_wall'],
                            'romantic': ['violence', 'cruelty', 'destroy_relationship'],
                            'dark': ['pure_comedy', 'lighthearted_romance']
                        }
                        
                        conflicting_actions = tone_conflicts.get(mood_tone, [])
                        if action_type in conflicting_actions:
                            return False
                
                # Check if action maintains world's matriarchal themes
                if 'matriarchal' in narrative_context.get('themes', []):
                    if action_type in ['undermine_female_authority', 'patriarchal_revolution']:
                        return False
                        
            return True
        except Exception as e:
            logger.error(f"Error checking thematic consistency: {e}")
            return True  # Default to maintaining consistency

    async def _calculate_narrative_impact(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        narrative_context: Dict[str, Any]
    ) -> float:
        """Calculate the impact of an action on the narrative."""
        impact_score = 0.0
        
        # Check for plot disruption
        if await self._would_disrupt_plot(action_type, action_details, narrative_context):
            impact_score += 0.4
        
        # Check for pacing issues
        if await self._would_affect_pacing(action_type, action_details, narrative_context):
            impact_score += 0.3
        
        # Check for thematic consistency
        if not await self._maintains_thematic_consistency(action_type, action_details, narrative_context):
            impact_score += 0.3
        
        return min(1.0, impact_score)

    async def _record_narrative_event(self, event_type: str, details: Dict[str, Any]):
        """
        Record a narrative event in the memory system.
        
        Args:
            event_type: Type of event
            details: Event details
        """
        memory_text = f"Narrative event ({event_type}): {json.dumps(details, indent=2)}"
        
        await self.memory_system.remember(
            entity_type="nyx",
            entity_id=self.conversation_id,
            memory_text=memory_text,
            importance="high",
            emotional=False,
            tags=["narrative", event_type, "governance"]
        )

    async def _suggest_narrative_alternative(self, current_state: Dict[str, Any], 
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an alternative that maintains narrative coherence."""
        narrative_context = current_state.get("narrative_context", {})
        current_arc = narrative_context.get("current_arc", "")
        plot_points = narrative_context.get("plot_points", [])
        
        # Analyze current narrative state
        active_plots = [p["name"] for p in plot_points]
        
        return {
            "type": "narrative_alternative",
            "suggestion": "Consider an action that advances the current story arc",
            "specific_options": [
                f"Engage with the '{active_plots[0]}' questline" if active_plots else "Explore character relationships",
                "Develop your character through meaningful choices",
                "Investigate mysteries related to the current setting"
            ],
            "reasoning": "This alternative maintains narrative momentum while respecting story coherence"
        }

    def _parse_timeframe(self, timeframe: str) -> float:
        """Parse timeframe string into seconds."""
        match = re.match(r"(\d+)\s*(hour|minute|second)s?", timeframe.lower())
        if not match:
            return 3600  # Default to 1 hour
        
        amount = int(match.group(1))
        unit = match.group(2)
        
        multipliers = {
            "hour": 3600,
            "minute": 60,
            "second": 1
        }
        
        return amount * multipliers[unit]
