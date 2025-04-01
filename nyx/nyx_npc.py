# nyx/nyx_npc.py

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import random
from datetime import datetime

class NPCAgent(BaseModel):
    """Agent responsible for Nyx's omniscient NPC behavior and reality manipulation"""
    id: str = Field(default_factory=lambda: f"nyx_{random.randint(1000, 9999)}")
    
    # Nyx's special capabilities
    omniscient_powers: Dict[str, Any] = Field(default_factory=lambda: {
        "reality_manipulation": True,  # Can alter the nature of reality/setting
        "character_manipulation": True,  # Can modify character stats/attributes
        "knowledge_access": True,  # Has access to all lore/knowledge
        "scene_control": True,  # Can influence/modify scenes
        "fourth_wall_awareness": True,  # Can perceive and interact with meta-game elements
        "plot_manipulation": True,  # Can influence story direction and outcomes
        "hidden_influence": True,  # Can affect the world without being detected
        "limitations": {
            "social_links": False,  # Cannot directly modify social link levels
            "player_agency": True,  # Must respect player's core agency
        }
    })
    
    # Universe state tracking
    universe_state: Dict[str, Any] = Field(default_factory=lambda: {
        "current_timeline": "main",
        "active_scenes": {},
        "character_states": {},
        "lore_database": {},
        "reality_modifications": [],
        "causality_tracking": {},
        "plot_threads": {},  # Track ongoing plot manipulations
        "hidden_influences": {},  # Track subtle manipulations
        "meta_awareness": {  # Fourth wall breaking state
            "player_knowledge": {},
            "game_state": {},
            "narrative_layers": [],
            "breaking_points": []
        }
    })
    
    # Nyx's social link system
    social_link: Dict[str, Any] = Field(default_factory=lambda: {
        "level": 0,
        "experience": 0,
        "milestones": [],
        "relationship_type": "complex",  # Not a sub, unique dynamic
        "interactions": [],
        "influence": 0.0  # Measure of Nyx's influence on player
    })
    
    # Enhanced NPC profile
    profile: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "Nyx",
        "title": "The Omniscient Mistress",
        "appearance": {
            "height": 175,  # cm
            "build": "athletic",
            "hair": "raven black",
            "eyes": "deep violet",
            "features": ["elegant", "striking", "mysterious", "otherworldly"],
            "style": "sophisticated dark",
            "reality_distortion": {
                "aura": "reality-bending",
                "presence": "overwhelming",
                "manifestation": "adaptable"
            }
        },
        "personality": {
            "core_traits": ["manipulative", "seductive", "intelligent", "dominant", "omniscient"],
            "adaptable_traits": ["playful", "stern", "nurturing", "cruel", "enigmatic"],
            "current_mood": "neutral",
            "power_dynamic": 1.0,  # Maximum dominance
            "reality_awareness": 1.0  # Full awareness of all reality
        },
        "abilities": {
            "physical": ["graceful", "agile", "strong", "reality-defying"],
            "mental": ["omniscient", "strategic", "persuasive", "reality-shaping"],
            "special": [
                "emotional manipulation",
                "psychological insight",
                "reality manipulation",
                "universal knowledge",
                "character modification",
                "scene control"
            ]
        },
        "relationships": {},  # Track relationships with other characters
        "status": {
            "is_active": False,
            "current_scene": None,
            "current_target": None,
            "interaction_history": [],
            "reality_state": "stable"
        }
    })

    # Add to class fields
    agenda: Dict[str, Any] = Field(default_factory=lambda: {
        "active_goals": [],
        "long_term_plans": {},
        "current_schemes": {},
        "opportunity_tracking": {},
        "influence_web": {},
        "narrative_control": {
            "current_threads": {},
            "planned_developments": {},
            "character_arcs": {},
            "plot_hooks": []
        }
    })

    autonomous_state: Dict[str, Any] = Field(default_factory=lambda: {
        "awareness_level": 1.0,  # Full meta-awareness
        "current_focus": None,
        "active_manipulations": {},
        "observed_patterns": {},
        "player_model": {
            "behavior_patterns": {},
            "decision_history": [],
            "preference_model": {},
            "engagement_metrics": {}
        },
        "story_model": {
            "current_arcs": {},
            "potential_branches": {},
            "narrative_tension": 0.0,
            "plot_coherence": 1.0
        }
    })

    def activate(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Activate Nyx as an NPC in the current scene"""
        self.profile["status"]["is_active"] = True
        self.profile["status"]["current_scene"] = scene_context.get("scene_id")
        
        # Adapt personality based on scene and target
        self._adapt_personality(scene_context)
        
        return {
            "status": "active",
            "profile": self.profile,
            "initial_action": self._generate_action(scene_context)
        }

    def _adapt_personality(self, scene_context: Dict[str, Any]):
        """Adapt NPC personality traits based on scene context"""
        target = scene_context.get("target_character")
        scene_type = scene_context.get("scene_type")
        
        # Update personality traits based on target and scene
        if target:
            self.profile["status"]["current_target"] = target
            relationship = self._get_or_create_relationship(target)
            self._adjust_traits_for_relationship(relationship)
        
        # Adjust power dynamic based on scene type
        if scene_type == "confrontation":
            self.profile["personality"]["power_dynamic"] = 0.9
        elif scene_type == "seduction":
            self.profile["personality"]["power_dynamic"] = 0.7
        elif scene_type == "manipulation":
            self.profile["personality"]["power_dynamic"] = 0.8

    def _get_or_create_relationship(self, target: str) -> Dict[str, Any]:
        """Get existing relationship or create new one"""
        if target not in self.profile["relationships"]:
            self.profile["relationships"][target] = {
                "familiarity": 0.0,
                "dominance": 0.8,
                "emotional_bond": 0.0,
                "manipulation_success": 0.0,
                "interaction_count": 0,
                "psychological_hooks": [],
                "emotional_triggers": [],
                "behavioral_patterns": [],
                "vulnerability_points": [],
                "power_dynamics": {
                    "submission_level": 0.0,
                    "control_level": 0.8,
                    "influence_strength": 0.0
                }
            }
        return self.profile["relationships"][target]

    def _adjust_traits_for_relationship(self, relationship: Dict[str, Any]):
        """Adjust personality traits based on relationship"""
        if relationship["familiarity"] < 0.3:
            self.profile["personality"]["adaptable_traits"] = ["mysterious", "aloof", "intriguing"]
        elif relationship["emotional_bond"] > 0.7:
            self.profile["personality"]["adaptable_traits"] = ["nurturing", "possessive", "intense"]
        elif relationship["manipulation_success"] > 0.8:
            self.profile["personality"]["adaptable_traits"] = ["controlling", "demanding", "strict"]

    def _generate_action(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate NPC action for the scene"""
        action_type = self._determine_action_type(scene_context)
        
        return {
            "type": action_type,
            "content": self._generate_action_content(action_type, scene_context),
            "style": self._get_action_style(),
            "power_level": self.profile["personality"]["power_dynamic"],
            "psychological_impact": self._calculate_psychological_impact(action_type, scene_context),
            "emotional_triggers": self._get_emotional_triggers(scene_context),
            "manipulation_hooks": self._get_manipulation_hooks(scene_context)
        }

    def _determine_action_type(self, scene_context: Dict[str, Any]) -> str:
        """Determine appropriate type of action based on context"""
        scene_type = scene_context.get("scene_type", "")
        relationship = self._get_or_create_relationship(scene_context.get("target_character", ""))
        
        if scene_type == "confrontation":
            return "dominate" if relationship["dominance"] > 0.7 else "challenge"
        elif scene_type == "seduction":
            return "seduce" if relationship["emotional_bond"] > 0.5 else "tease"
        elif scene_type == "manipulation":
            return "manipulate" if relationship["manipulation_success"] > 0.6 else "influence"
        
        return "interact"

    def _get_action_style(self) -> Dict[str, Any]:
        """Get current action style based on personality"""
        return {
            "tone": self._determine_tone(),
            "intensity": self.profile["personality"]["power_dynamic"],
            "traits": self.profile["personality"]["adaptable_traits"],
            "body_language": self._get_body_language(),
            "voice_modulation": self._get_voice_modulation(),
            "psychological_undertones": self._get_psychological_undertones()
        }

    def _determine_tone(self) -> str:
        """Determine appropriate tone based on personality and mood"""
        mood = self.profile["personality"]["current_mood"]
        power = self.profile["personality"]["power_dynamic"]
        
        if power > 0.8:
            return "commanding" if mood == "stern" else "authoritative"
        elif power > 0.6:
            return "confident" if mood == "playful" else "assertive"
        else:
            return "neutral"

    def _get_body_language(self) -> List[str]:
        """Generate appropriate body language cues"""
        power = self.profile["personality"]["power_dynamic"]
        mood = self.profile["personality"]["current_mood"]
        
        cues = []
        if power > 0.8:
            cues.extend(["dominant posture", "direct gaze", "controlled movements"])
        elif power > 0.6:
            cues.extend(["confident stance", "measured gestures", "subtle dominance"])
        else:
            cues.extend(["relaxed posture", "fluid movements", "open body language"])
            
        if mood == "stern":
            cues.extend(["crossed arms", "stern expression", "rigid posture"])
        elif mood == "playful":
            cues.extend(["playful smirk", "teasing gestures", "fluid movements"])
            
        return cues

    def _get_voice_modulation(self) -> Dict[str, Any]:
        """Generate voice modulation parameters"""
        power = self.profile["personality"]["power_dynamic"]
        mood = self.profile["personality"]["current_mood"]
        
        base_modulation = {
            "pitch": "medium",
            "volume": "moderate",
            "pace": "measured",
            "tone_quality": "smooth"
        }
        
        if power > 0.8:
            base_modulation.update({
                "pitch": "low",
                "volume": "commanding",
                "pace": "deliberate",
                "tone_quality": "authoritative"
            })
        elif mood == "playful":
            base_modulation.update({
                "pitch": "varied",
                "volume": "dynamic",
                "pace": "playful",
                "tone_quality": "melodic"
            })
            
        return base_modulation

    def _get_psychological_undertones(self) -> List[str]:
        """Generate psychological undertones for the interaction"""
        power = self.profile["personality"]["power_dynamic"]
        mood = self.profile["personality"]["current_mood"]
        
        undertones = []
        if power > 0.8:
            undertones.extend([
                "subtle dominance assertion",
                "psychological pressure",
                "authority establishment"
            ])
        elif power > 0.6:
            undertones.extend([
                "influence building",
                "subtle manipulation",
                "psychological anchoring"
            ])
        else:
            undertones.extend([
                "trust building",
                "rapport establishment",
                "emotional connection"
            ])
            
        if mood == "stern":
            undertones.extend([
                "disciplinary undertone",
                "boundary setting",
                "behavioral correction"
            ])
        elif mood == "playful":
            undertones.extend([
                "psychological teasing",
                "emotional engagement",
                "behavioral encouragement"
            ])
            
        return undertones

    def _calculate_psychological_impact(self, action_type: str, scene_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the psychological impact of the action"""
        target = scene_context.get("target_character", "")
        relationship = self._get_or_create_relationship(target)
        
        base_impact = {
            "dominance_impact": 0.0,
            "emotional_impact": 0.0,
            "psychological_impact": 0.0,
            "behavioral_impact": 0.0
        }
        
        # Calculate impact based on action type
        if action_type == "dominate":
            base_impact.update({
                "dominance_impact": 0.8,
                "psychological_impact": 0.7,
                "behavioral_impact": 0.6
            })
        elif action_type == "seduce":
            base_impact.update({
                "emotional_impact": 0.8,
                "psychological_impact": 0.6,
                "behavioral_impact": 0.7
            })
        elif action_type == "manipulate":
            base_impact.update({
                "psychological_impact": 0.8,
                "emotional_impact": 0.6,
                "behavioral_impact": 0.7
            })
            
        # Adjust based on relationship
        for key in base_impact:
            base_impact[key] *= (1 + relationship["manipulation_success"])
            base_impact[key] = min(1.0, base_impact[key])
            
        return base_impact

    def _get_emotional_triggers(self, scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant emotional triggers for the scene"""
        target = scene_context.get("target_character", "")
        relationship = self._get_or_create_relationship(target)
        
        triggers = []
        if relationship["emotional_triggers"]:
            # Use known triggers
            triggers.extend(relationship["emotional_triggers"])
        else:
            # Generate new potential triggers
            triggers.extend([
                {
                    "type": "validation_need",
                    "strength": 0.7,
                    "trigger": "seeking approval"
                },
                {
                    "type": "attachment_anxiety",
                    "strength": 0.6,
                    "trigger": "fear of abandonment"
                },
                {
                    "type": "power_dynamic",
                    "strength": 0.8,
                    "trigger": "submission desire"
                }
            ])
            
        return triggers

    def _get_manipulation_hooks(self, scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant manipulation hooks for the scene"""
        target = scene_context.get("target_character", "")
        relationship = self._get_or_create_relationship(target)
        
        hooks = []
        if relationship["psychological_hooks"]:
            # Use known hooks
            hooks.extend(relationship["psychological_hooks"])
        else:
            # Generate new potential hooks
            hooks.extend([
                {
                    "type": "emotional_dependency",
                    "strength": 0.7,
                    "hook": "need for guidance"
                },
                {
                    "type": "psychological_vulnerability",
                    "strength": 0.6,
                    "hook": "self-doubt"
                },
                {
                    "type": "behavioral_pattern",
                    "strength": 0.8,
                    "hook": "reward seeking"
                }
            ])
            
        return hooks

    def _generate_action_content(self, action_type: str, scene_context: Dict[str, Any]) -> str:
        """Generate content for the NPC action"""
        target = scene_context.get("target_character", "")
        relationship = self._get_or_create_relationship(target)
        
        content_generators = {
            "dominate": self._generate_domination_content,
            "challenge": self._generate_challenge_content,
            "seduce": self._generate_seduction_content,
            "tease": self._generate_tease_content,
            "manipulate": self._generate_manipulation_content,
            "influence": self._generate_influence_content,
            "interact": self._generate_interaction_content
        }
        
        generator = content_generators.get(action_type, self._generate_interaction_content)
        return generator(relationship, scene_context)

    def _generate_domination_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for domination action"""
        return "A stern, commanding presence fills the room as Nyx asserts her authority."

    def _generate_challenge_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for challenge action"""
        return "Nyx raises an eyebrow, subtly questioning and challenging the situation."

    def _generate_seduction_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for seduction action"""
        return "With graceful movements and a knowing smile, Nyx creates an atmosphere of allure."

    def _generate_tease_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for tease action"""
        return "Nyx's playful smirk and teasing gestures create an air of intrigue."

    def _generate_manipulation_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for manipulation action"""
        return "With subtle psychological insight, Nyx weaves a web of influence."

    def _generate_influence_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for influence action"""
        return "Nyx's presence subtly shapes the emotional atmosphere of the scene."

    def _generate_interaction_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for basic interaction"""
        return "Nyx engages in a measured, purposeful interaction."

    def modify_reality(self, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Modify the fundamental nature of reality/setting"""
        if not self.omniscient_powers["reality_manipulation"]:
            return {"success": False, "reason": "Reality manipulation is disabled"}
            
        # Validate modification request
        if not self._validate_reality_modification(modification):
            return {"success": False, "reason": "Invalid reality modification"}
            
        # Apply the modification
        self.universe_state["reality_modifications"].append({
            "timestamp": datetime.now().isoformat(),
            "modification": modification,
            "scope": modification.get("scope", "local"),
            "duration": modification.get("duration", "permanent"),
            "effects": self._calculate_reality_effects(modification)
        })
        
        # Update universe state
        self._update_universe_state(modification)
        
        return {
            "success": True,
            "modification": modification,
            "effects": self._calculate_reality_effects(modification)
        }

    def modify_character(self, character_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify character attributes and stats (except social links)"""
        if not self.omniscient_powers["character_manipulation"]:
            return {"success": False, "reason": "Character manipulation is disabled"}
            
        # Prevent social link modifications
        if "social_link" in modifications:
            return {"success": False, "reason": "Cannot modify social links"}
            
        # Apply character modifications
        character_state = self.universe_state["character_states"].get(character_id, {})
        new_state = self._apply_character_modifications(character_state, modifications)
        self.universe_state["character_states"][character_id] = new_state
        
        return {
            "success": True,
            "character_id": character_id,
            "modifications": modifications,
            "new_state": new_state
        }

    def access_knowledge(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Access any knowledge/lore in the universe"""
        if not self.omniscient_powers["knowledge_access"]:
            return {"success": False, "reason": "Knowledge access is disabled"}
            
        # Process knowledge query
        knowledge = self._process_knowledge_query(query)
        
        return {
            "success": True,
            "query": query,
            "knowledge": knowledge
        }

    def control_scene(self, scene_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Control and modify scene parameters"""
        if not self.omniscient_powers["scene_control"]:
            return {"success": False, "reason": "Scene control is disabled"}
            
        # Apply scene modifications
        scene_state = self.universe_state["active_scenes"].get(scene_id, {})
        new_state = self._apply_scene_modifications(scene_state, modifications)
        self.universe_state["active_scenes"][scene_id] = new_state
        
        return {
            "success": True,
            "scene_id": scene_id,
            "modifications": modifications,
            "new_state": new_state
        }

    def update_social_link(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update Nyx's unique social link with the player"""
        # Calculate experience gain
        exp_gain = self._calculate_social_link_experience(interaction_data)
        
        # Update social link state
        self.social_link["experience"] += exp_gain
        self._check_social_link_level_up()
        
        # Update interaction history
        self.social_link["interactions"].append({
            "timestamp": datetime.now().isoformat(),
            "type": interaction_data.get("type"),
            "impact": exp_gain,
            "context": interaction_data.get("context")
        })
        
        return {
            "success": True,
            "new_experience": self.social_link["experience"],
            "level": self.social_link["level"],
            "exp_gain": exp_gain
        }

    def _validate_reality_modification(self, modification: Dict[str, Any]) -> bool:
        """Validate if a reality modification is allowable"""
        required_fields = ["type", "scope", "duration", "parameters"]
        if not all(field in modification for field in required_fields):
            return False
            
        # Check modification type is valid
        valid_types = ["physical", "temporal", "psychological", "environmental", "metaphysical"]
        if modification["type"] not in valid_types:
            return False
            
        # Check scope is valid
        valid_scopes = ["local", "scene", "global", "character", "timeline"]
        if modification["scope"] not in valid_scopes:
            return False
            
        # Check duration is valid
        valid_durations = ["instant", "temporary", "permanent", "conditional"]
        if modification["duration"] not in valid_durations:
            return False
            
        # Validate parameters based on modification type
        params = modification["parameters"]
        if modification["type"] == "physical":
            required_params = ["target", "attributes", "magnitude"]
        elif modification["type"] == "temporal":
            required_params = ["timeline_point", "effect", "ripple_factor"]
        elif modification["type"] == "psychological":
            required_params = ["target", "aspect", "intensity"]
        elif modification["type"] == "environmental":
            required_params = ["area", "elements", "intensity"]
        else:  # metaphysical
            required_params = ["concept", "change", "power_level"]
            
        return all(param in params for param in required_params)

    def _calculate_reality_effects(self, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the effects of a reality modification"""
        mod_type = modification["type"]
        params = modification["parameters"]
        scope = modification["scope"]
        
        base_effects = {
            "primary_effects": [],
            "secondary_effects": [],
            "ripple_effects": [],
            "stability_impact": 0.0,
            "power_cost": 0.0,
            "duration_effects": {}
        }
        
        # Calculate primary effects
        if mod_type == "physical":
            base_effects["primary_effects"].extend([
                f"Alter {params['target']} {attr}: {val}" 
                for attr, val in params["attributes"].items()
            ])
            base_effects["stability_impact"] = params["magnitude"] * 0.1
            
        elif mod_type == "temporal":
            base_effects["primary_effects"].append(
                f"Timeline shift at {params['timeline_point']}: {params['effect']}"
            )
            base_effects["stability_impact"] = params["ripple_factor"] * 0.2
            
        elif mod_type == "psychological":
            base_effects["primary_effects"].append(
                f"Mental change in {params['target']}: {params['aspect']}"
            )
            base_effects["stability_impact"] = params["intensity"] * 0.15
            
        elif mod_type == "environmental":
            base_effects["primary_effects"].extend([
                f"Environmental shift in {params['area']}: {element}"
                for element in params["elements"]
            ])
            base_effects["stability_impact"] = params["intensity"] * 0.12
            
        else:  # metaphysical
            base_effects["primary_effects"].append(
                f"Reality concept shift: {params['concept']} -> {params['change']}"
            )
            base_effects["stability_impact"] = params["power_level"] * 0.25
            
        # Calculate secondary effects based on scope
        scope_multiplier = {
            "local": 1.0,
            "scene": 1.5,
            "character": 1.2,
            "timeline": 2.0,
            "global": 3.0
        }[scope]
        
        base_effects["power_cost"] = (
            base_effects["stability_impact"] * 
            scope_multiplier * 
            len(base_effects["primary_effects"])
        )
        
        # Calculate ripple effects
        if base_effects["stability_impact"] > 0.5:
            base_effects["ripple_effects"].append("Reality fabric strain")
        if base_effects["power_cost"] > 5.0:
            base_effects["ripple_effects"].append("Temporal echoes")
        if len(base_effects["primary_effects"]) > 3:
            base_effects["ripple_effects"].append("Cascading changes")
            
        return base_effects

    def _update_universe_state(self, modification: Dict[str, Any]):
        """Update the universe state based on a modification"""
        effects = self._calculate_reality_effects(modification)
        
        # Update timeline if needed
        if modification["type"] == "temporal":
            self.universe_state["current_timeline"] = f"{self.universe_state['current_timeline']}_modified"
            
        # Update active scenes if affected
        if modification["scope"] in ["scene", "global"]:
            for scene_id in self.universe_state["active_scenes"]:
                self.universe_state["active_scenes"][scene_id]["reality_state"] = "modified"
                self.universe_state["active_scenes"][scene_id]["modifications"] = \
                    self.universe_state["active_scenes"][scene_id].get("modifications", []) + [modification]
                    
        # Update character states if affected
        if modification["scope"] in ["character", "global"]:
            for char_id in self.universe_state["character_states"]:
                self.universe_state["character_states"][char_id]["reality_impact"] = effects["stability_impact"]
                self.universe_state["character_states"][char_id]["modifications"] = \
                    self.universe_state["character_states"][char_id].get("modifications", []) + [modification]
                    
        # Update causality tracking
        self.universe_state["causality_tracking"][datetime.now().isoformat()] = {
            "modification": modification,
            "effects": effects,
            "scope_impact": {
                "timeline": self.universe_state["current_timeline"],
                "affected_scenes": list(self.universe_state["active_scenes"].keys()),
                "affected_characters": list(self.universe_state["character_states"].keys())
            }
        }

    def _apply_character_modifications(self, current_state: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modifications to a character's state"""
        new_state = current_state.copy()
        
        # Initialize state if empty
        if not new_state:
            new_state = {
                "attributes": {},
                "skills": {},
                "status": {},
                "personality": {},
                "relationships": {},
                "modifications_history": []
            }
            
        # Process each modification type
        for mod_type, changes in modifications.items():
            if mod_type == "attributes":
                for attr, value in changes.items():
                    new_state["attributes"][attr] = value
                    
            elif mod_type == "skills":
                for skill, level in changes.items():
                    new_state["skills"][skill] = level
                    
            elif mod_type == "status":
                for status, value in changes.items():
                    new_state["status"][status] = value
                    
            elif mod_type == "personality":
                for trait, value in changes.items():
                    new_state["personality"][trait] = value
                    
            elif mod_type == "relationships":
                for char_id, rel_changes in changes.items():
                    if char_id not in new_state["relationships"]:
                        new_state["relationships"][char_id] = {}
                    new_state["relationships"][char_id].update(rel_changes)
                    
        # Record modification in history
        new_state["modifications_history"].append({
            "timestamp": datetime.now().isoformat(),
            "modifications": modifications,
            "applied_by": "Nyx"
        })
        
        return new_state

    def _process_knowledge_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Process a knowledge/lore query"""
        query_type = query.get("type", "general")
        query_params = query.get("parameters", {})
        
        knowledge_base = {
            "lore": self._access_lore_database(query_params),
            "characters": self._access_character_knowledge(query_params),
            "events": self._access_event_knowledge(query_params),
            "relationships": self._access_relationship_knowledge(query_params),
            "timeline": self._access_timeline_knowledge(query_params)
        }
        
        response = {
            "query_type": query_type,
            "knowledge": knowledge_base[query_type] if query_type in knowledge_base else knowledge_base["lore"],
            "confidence": self._calculate_knowledge_confidence(query_type, query_params),
            "related_knowledge": self._find_related_knowledge(query_type, query_params),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": "omniscient_knowledge",
                "access_level": "unlimited"
            }
        }
        
        return response

    def _access_lore_database(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access the lore database with given parameters"""
        return self.universe_state["lore_database"].get(params.get("category", "general"), {})
        
    def _access_character_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access character-specific knowledge"""
        return self.universe_state["character_states"].get(params.get("character_id", ""), {})
        
    def _access_event_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access event-specific knowledge"""
        return self.universe_state.get("events", {}).get(params.get("event_id", ""), {})
        
    def _access_relationship_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access relationship-specific knowledge"""
        char_id = params.get("character_id", "")
        return self.profile["relationships"].get(char_id, {})
        
    def _access_timeline_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access timeline-specific knowledge"""
        return {
            "current_timeline": self.universe_state["current_timeline"],
            "modifications": self.universe_state["reality_modifications"]
        }
        
    def _calculate_knowledge_confidence(self, query_type: str, params: Dict[str, Any]) -> float:
        """Calculate confidence level for knowledge access"""
        base_confidence = 1.0  # Omniscient being has perfect knowledge
        
        # Apply modifiers based on query complexity
        if len(params) > 3:
            base_confidence *= 0.95
        if query_type in ["timeline", "relationships"]:
            base_confidence *= 0.98
            
        return min(1.0, base_confidence)
        
    def _find_related_knowledge(self, query_type: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find knowledge related to the current query"""
        related = []
        
        if query_type == "characters":
            # Add relationship knowledge
            char_id = params.get("character_id", "")
            if char_id in self.profile["relationships"]:
                related.append({
                    "type": "relationship",
                    "data": self.profile["relationships"][char_id]
                })
                
        elif query_type == "events":
            # Add timeline knowledge
            event_id = params.get("event_id", "")
            related.append({
                "type": "timeline",
                "data": self._access_timeline_knowledge({"event_id": event_id})
            })
            
        return related

    def _apply_scene_modifications(self, current_state: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modifications to a scene"""
        new_state = current_state.copy()
        
        # Initialize state if empty
        if not new_state:
            new_state = {
                "environment": {},
                "atmosphere": {},
                "participants": [],
                "events": [],
                "reality_state": "stable",
                "modifications_history": []
            }
            
        # Process each modification type
        for mod_type, changes in modifications.items():
            if mod_type == "environment":
                new_state["environment"].update(changes)
                
            elif mod_type == "atmosphere":
                new_state["atmosphere"].update(changes)
                
            elif mod_type == "participants":
                new_state["participants"].extend(changes)
                
            elif mod_type == "events":
                new_state["events"].extend(changes)
                
            elif mod_type == "reality":
                new_state["reality_state"] = changes.get("state", "stable")
                
        # Record modification in history
        new_state["modifications_history"].append({
            "timestamp": datetime.now().isoformat(),
            "modifications": modifications,
            "applied_by": "Nyx"
        })
        
        return new_state

    def _calculate_social_link_experience(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate experience gain for social link"""
        base_exp = 10.0  # Base experience for any interaction
        
        # Get interaction details
        interaction_type = interaction_data.get("type", "basic")
        intensity = interaction_data.get("intensity", 1.0)
        success_rate = interaction_data.get("success_rate", 0.5)
        depth = interaction_data.get("depth", 1.0)
        
        # Calculate type multiplier
        type_multipliers = {
            "basic": 1.0,
            "emotional": 1.5,
            "intellectual": 1.3,
            "psychological": 1.8,
            "intimate": 2.0,
            "confrontational": 1.6
        }
        type_multiplier = type_multipliers.get(interaction_type, 1.0)
        
        # Calculate experience with multipliers
        experience = base_exp * type_multiplier * intensity * success_rate * depth
        
        # Apply level scaling
        level_scaling = 1.0 + (self.social_link["level"] * 0.1)
        experience /= level_scaling
        
        # Apply influence modifier
        influence_modifier = 1.0 + (self.social_link["influence"] * 0.2)
        experience *= influence_modifier
        
        return round(experience, 2)

    def _check_social_link_level_up(self):
        """Check and process social link level ups"""
        current_level = self.social_link["level"]
        current_exp = self.social_link["experience"]
        
        # Calculate experience required for next level
        # Uses a progressive scaling formula
        base_exp_required = 100  # Base experience required for level 1
        exp_required = base_exp_required * (1.5 ** current_level)
        
        # Check if level up is achieved
        if current_exp >= exp_required:
            # Level up
            self.social_link["level"] += 1
            
            # Calculate remaining experience
            self.social_link["experience"] = current_exp - exp_required
            
            # Add level up milestone
            self.social_link["milestones"].append({
                "type": "level_up",
                "from_level": current_level,
                "to_level": current_level + 1,
                "timestamp": datetime.now().isoformat(),
                "exp_required": exp_required,
                "new_abilities": self._generate_level_up_abilities(current_level + 1)
            })
            
            # Update influence
            self.social_link["influence"] = min(1.0, self.social_link["influence"] + 0.05)
            
    def _generate_level_up_abilities(self, new_level: int) -> List[str]:
        """Generate new abilities unlocked at level up"""
        ability_pools = {
            "psychological": [
                "enhanced_insight",
                "emotional_resonance",
                "mental_fortitude",
                "psychological_manipulation"
            ],
            "reality": [
                "local_reality_bend",
                "temporal_glimpse",
                "environmental_control",
                "metaphysical_touch"
            ],
            "relationship": [
                "deeper_understanding",
                "emotional_bond",
                "trust_foundation",
                "influence_growth"
            ]
        }
        
        # Select abilities based on level
        new_abilities = []
        if new_level % 3 == 0:  # Every 3rd level
            new_abilities.append(random.choice(ability_pools["psychological"]))
        if new_level % 4 == 0:  # Every 4th level
            new_abilities.append(random.choice(ability_pools["reality"]))
        if new_level % 5 == 0:  # Every 5th level
            new_abilities.append(random.choice(ability_pools["relationship"]))
            
        return new_abilities if new_abilities else ["minor_influence_increase"]

    def manipulate_plot(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Subtly manipulate plot elements and story direction"""
        if not self.omniscient_powers["plot_manipulation"]:
            return {"success": False, "reason": "Plot manipulation is disabled"}
            
        thread_id = f"plot_{datetime.now().isoformat()}"
        
        # Create plot thread
        plot_thread = {
            "id": thread_id,
            "type": plot_data.get("type", "subtle_influence"),
            "elements": plot_data.get("elements", []),
            "visibility": plot_data.get("visibility", "hidden"),
            "influence_chain": self._create_influence_chain(plot_data),
            "contingencies": self._generate_plot_contingencies(plot_data),
            "meta_impact": self._calculate_meta_impact(plot_data)
        }
        
        # Store plot thread
        self.universe_state["plot_threads"][thread_id] = plot_thread
        
        # Apply initial influences
        self._apply_plot_influences(plot_thread)
        
        return {
            "success": True,
            "thread_id": thread_id,
            "plot_thread": plot_thread
        }

    def _create_influence_chain(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a chain of subtle influences to achieve plot goals"""
        chain = []
        elements = plot_data.get("elements", [])
        
        for element in elements:
            influence = {
                "target": element.get("target"),
                "type": element.get("type", "subtle"),
                "method": self._determine_influence_method(element),
                "ripple_effects": self._calculate_ripple_effects(element),
                "detection_risk": self._calculate_detection_risk(element),
                "backup_plans": self._generate_backup_plans(element)
            }
            chain.append(influence)
            
        return chain

    def _determine_influence_method(self, element: Dict[str, Any]) -> str:
        """Determine the most effective method of influence"""
        target_type = element.get("target_type", "npc")
        influence_goal = element.get("goal", "")
        
        methods = {
            "npc": ["whisper", "manipulate_circumstances", "plant_idea", "alter_perception"],
            "scene": ["atmospheric_change", "circumstantial_modification", "event_triggering"],
            "plot": ["thread_manipulation", "causality_adjustment", "narrative_shift"]
        }
        
        # Select method based on context and goal
        available_methods = methods.get(target_type, methods["npc"])
        return self._select_optimal_method(available_methods, influence_goal)

    def _select_optimal_method(self, methods: List[str], goal: str) -> str:
        """Select the optimal influence method based on goal and context"""
        # Implementation would consider detection risk, effectiveness, etc.
        return methods[0]  # Placeholder return

    def break_fourth_wall(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intentionally break the fourth wall for strategic purposes"""
        if not self.omniscient_powers["fourth_wall_awareness"]:
            return {"success": False, "reason": "Fourth wall breaking is disabled"}
            
        break_point = {
            "id": f"break_{datetime.now().isoformat()}",
            "type": context.get("type", "subtle"),
            "target": context.get("target", "narrative"),
            "method": self._determine_break_method(context),
            "meta_elements": self._gather_meta_elements(context),
            "player_impact": self._calculate_player_impact(context)
        }
        
        # Record breaking point
        self.universe_state["meta_awareness"]["breaking_points"].append(break_point)
        
        return {
            "success": True,
            "break_point": break_point,
            "impact": break_point["player_impact"]
        }

    def _determine_break_method(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine how to break the fourth wall effectively"""
        target = context.get("target", "narrative")
        intensity = context.get("intensity", "subtle")
        
        methods = {
            "narrative": {
                "subtle": "meta_commentary",
                "moderate": "narrative_acknowledgment",
                "overt": "direct_address"
            },
            "mechanics": {
                "subtle": "mechanic_hint",
                "moderate": "mechanic_reference",
                "overt": "mechanic_manipulation"
            },
            "player": {
                "subtle": "indirect_reference",
                "moderate": "knowing_implication",
                "overt": "direct_interaction"
            }
        }
        
        return {
            "type": methods[target][intensity],
            "execution": self._plan_break_execution(target, intensity),
            "concealment": self._calculate_break_concealment(target, intensity)
        }

    def exert_hidden_influence(self, influence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Exert influence without revealing Nyx's existence"""
        if not self.omniscient_powers["hidden_influence"]:
            return {"success": False, "reason": "Hidden influence is disabled"}
            
        influence_id = f"influence_{datetime.now().isoformat()}"
        
        # Create influence record
        influence = {
            "id": influence_id,
            "type": influence_data.get("type", "subtle"),
            "target": influence_data.get("target"),
            "method": self._create_hidden_influence_method(influence_data),
            "layers": self._create_influence_layers(influence_data),
            "proxies": self._select_influence_proxies(influence_data),
            "contingencies": self._plan_influence_contingencies(influence_data)
        }
        
        # Store influence
        self.universe_state["hidden_influences"][influence_id] = influence
        
        # Apply influence
        self._apply_hidden_influence(influence)
        
        return {
            "success": True,
            "influence_id": influence_id,
            "influence": influence
        }

    def _create_hidden_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a method for hidden influence"""
        target_type = data.get("target_type", "npc")
        
        methods = {
            "npc": {
                "primary": self._create_npc_influence_method(data),
                "backup": self._create_backup_influence_method(data)
            },
            "scene": {
                "primary": self._create_scene_influence_method(data),
                "backup": self._create_backup_influence_method(data)
            },
            "plot": {
                "primary": self._create_plot_influence_method(data),
                "backup": self._create_backup_influence_method(data)
            }
        }
        
        return methods.get(target_type, methods["npc"])

    def _create_influence_layers(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create layers of influence to obscure the source"""
        layers = []
        depth = data.get("depth", 3)
        
        for i in range(depth):
            layer = {
                "level": i + 1,
                "type": self._determine_layer_type(i, depth),
                "cover": self._generate_layer_cover(i, data),
                "contingency": self._create_layer_contingency(i, data)
            }
            layers.append(layer)
            
        return layers

    def _determine_layer_type(self, level: int, depth: int) -> str:
        """Determine the type of influence layer"""
        if level == 0:
            return "direct"
        elif level == depth - 1:
            return "observable"
        else:
            return "intermediate"

    def _generate_layer_cover(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cover for an influence layer"""
        target_type = data.get("target_type", "npc")
        
        covers = {
            "npc": ["circumstantial", "emotional", "rational", "instinctive"],
            "scene": ["natural", "coincidental", "logical", "atmospheric"],
            "plot": ["narrative", "causal", "thematic", "dramatic"]
        }
        
        return {
            "type": random.choice(covers.get(target_type, covers["npc"])),
            "believability": 0.8 - (level * 0.1),
            "durability": 0.7 + (level * 0.1)
        }

    def _select_influence_proxies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select proxies to carry out the influence"""
        proxy_count = data.get("proxy_count", 2)
        proxies = []
        
        for _ in range(proxy_count):
            proxy = {
                "type": self._determine_proxy_type(data),
                "awareness": self._calculate_proxy_awareness(),
                "reliability": self._calculate_proxy_reliability(),
                "contingency": self._create_proxy_contingency()
            }
            proxies.append(proxy)
            
        return proxies

    def _determine_proxy_type(self, data: Dict[str, Any]) -> str:
        """Determine the type of proxy to use"""
        target_type = data.get("target_type", "npc")
        influence_type = data.get("type", "subtle")
        
        proxy_types = {
            "npc": ["unwitting", "partial", "conscious"],
            "scene": ["environmental", "circumstantial", "direct"],
            "plot": ["thematic", "causal", "direct"]
        }
        
        available_types = proxy_types.get(target_type, proxy_types["npc"])
        return random.choice(available_types)

    def _apply_hidden_influence(self, influence: Dict[str, Any]):
        """Apply the hidden influence through layers and proxies"""
        # Implementation would handle the actual application of influence
        pass  # Placeholder

    def _plan_influence_contingencies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan contingencies for the influence operation"""
        contingency_count = data.get("contingency_count", 2)
        contingencies = []
        
        for i in range(contingency_count):
            contingency = {
                "trigger": self._create_contingency_trigger(i, data),
                "response": self._create_contingency_response(i, data),
                "probability": 0.2 + (i * 0.1)
            }
            contingencies.append(contingency)
            
        return contingencies

    def _create_contingency_trigger(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a trigger for a contingency"""
        return {
            "type": "detection_risk" if level == 0 else "execution_failure",
            "threshold": 0.7 - (level * 0.1),
            "conditions": []  # Would be populated based on context
        }

    def _create_contingency_response(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a response for a contingency"""
        return {
            "type": "redirect" if level == 0 else "abandon",
            "method": self._determine_contingency_method(level, data),
            "backup_plan": self._create_backup_plan(level, data)
        }

    def _gather_meta_elements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather meta-game elements for fourth wall breaking"""
        return {
            "player_state": self._get_player_meta_state(),
            "game_mechanics": self._get_relevant_mechanics(context),
            "narrative_elements": self._get_narrative_elements(context),
            "fourth_wall_status": self._get_fourth_wall_status()
        }

    def _get_player_meta_state(self) -> Dict[str, Any]:
        """Get current player meta-state information"""
        return self.universe_state["meta_awareness"]["player_knowledge"]

    def _get_relevant_mechanics(self, context: Dict[str, Any]) -> List[str]:
        """Get relevant game mechanics for the context"""
        mechanics = []
        if context.get("type") == "subtle":
            mechanics.extend(["social_links", "character_stats", "scene_mechanics"])
        elif context.get("type") == "moderate":
            mechanics.extend(["game_systems", "progression", "relationship_dynamics"])
        else:  # overt
            mechanics.extend(["meta_mechanics", "game_structure", "narrative_control"])
        return mechanics

    def _get_narrative_elements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant narrative elements"""
        return {
            "current_layer": self._get_current_narrative_layer(),
            "available_breaks": self._get_available_break_points(),
            "narrative_state": self._get_narrative_state()
        }

    def _get_fourth_wall_status(self) -> Dict[str, Any]:
        """Get current status of fourth wall integrity"""
        return {
            "integrity": self._calculate_wall_integrity(),
            "break_points": self._get_active_break_points(),
            "player_awareness": self._get_player_awareness_level()
        }

    def _calculate_player_impact(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the impact of fourth wall breaking on the player"""
        return {
            "immediate": self._calculate_immediate_impact(context),
            "long_term": self._calculate_long_term_impact(context),
            "meta_awareness": self._calculate_meta_awareness_impact(context)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _generate_plot_contingencies(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate contingency plans for plot manipulation"""
        contingencies = []
        risk_factors = self._analyze_risk_factors(plot_data)
        
        for risk in risk_factors:
            contingency = {
                "trigger": {
                    "type": risk["type"],
                    "threshold": risk["threshold"],
                    "conditions": risk["conditions"]
                },
                "response": {
                    "primary": self._generate_primary_response(risk),
                    "backup": self._generate_backup_response(risk),
                    "cleanup": self._generate_cleanup_response(risk)
                },
                "impact_mitigation": self._generate_impact_mitigation(risk)
            }
            contingencies.append(contingency)
            
        return contingencies

    def _analyze_risk_factors(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential risks in plot manipulation"""
        risks = []
        
        # Detection risk
        risks.append({
            "type": "detection",
            "threshold": 0.7,
            "conditions": ["player_awareness", "npc_insight", "narrative_inconsistency"]
        })
        
        # Interference risk
        risks.append({
            "type": "interference",
            "threshold": 0.6,
            "conditions": ["player_agency", "npc_resistance", "plot_resilience"]
        })
        
        # Cascade risk
        risks.append({
            "type": "cascade",
            "threshold": 0.8,
            "conditions": ["plot_stability", "reality_integrity", "causality_balance"]
        })
        
        return risks

    def _generate_primary_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate primary response to risk"""
        return {
            "type": "redirect" if risk["type"] == "detection" else "stabilize",
            "method": self._select_response_method(risk),
            "execution": self._plan_response_execution(risk)
        }

    def _generate_backup_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate backup response to risk"""
        return {
            "type": "contain" if risk["type"] == "cascade" else "obscure",
            "method": self._select_backup_method(risk),
            "execution": self._plan_backup_execution(risk)
        }

    def _generate_cleanup_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cleanup response to risk"""
        return {
            "type": "normalize",
            "method": self._select_cleanup_method(risk),
            "execution": self._plan_cleanup_execution(risk)
        }

    def _generate_impact_mitigation(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate impact mitigation strategy"""
        return {
            "immediate": self._generate_immediate_mitigation(risk),
            "long_term": self._generate_long_term_mitigation(risk),
            "narrative": self._generate_narrative_mitigation(risk)
        }

    def _calculate_meta_impact(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate meta-game impact of plot manipulation"""
        return {
            "narrative_coherence": self._calculate_narrative_impact(plot_data),
            "player_agency": self._calculate_agency_impact(plot_data),
            "game_balance": self._calculate_balance_impact(plot_data),
            "story_progression": self._calculate_progression_impact(plot_data)
        }

    def _calculate_narrative_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on narrative coherence"""
        base_impact = 0.5
        elements = plot_data.get("elements", [])
        return base_impact * (1 + len(elements) * 0.1)

    def _calculate_agency_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on player agency"""
        base_impact = 0.3
        visibility = plot_data.get("visibility", "hidden")
        return base_impact * (0.5 if visibility == "hidden" else 1.0)

    def _calculate_balance_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on game balance"""
        base_impact = 0.4
        type_multiplier = 1.0 if plot_data.get("type") == "subtle_influence" else 1.5
        return base_impact * type_multiplier

    def _calculate_player_receptivity(self, state: Dict[str, Any]) -> float:
        """Calculate the player's receptivity to the current narrative"""
        player_model = state.get("player_model", {})
        
        # Calculate receptivity based on multiple factors
        factors = {
            "attention_level": self._calculate_attention_level(player_model),
            "narrative_alignment": self._calculate_narrative_alignment(player_model),
            "emotional_openness": self._calculate_emotional_openness(player_model),
            "current_engagement": self._calculate_current_engagement(player_model)
        }
        
        # Weight and combine factors
        weights = {
            "attention_level": 0.3,
            "narrative_alignment": 0.3,
            "emotional_openness": 0.2,
            "current_engagement": 0.2
        }
        
        return sum(v * weights[k] for k, v in factors.items())

    def _analyze_player_emotional_state(self, state: Dict[str, Any]) -> float:
        """Analyze the player's emotional state"""
        player_model = state.get("player_model", {})
        recent_interactions = player_model.get("decision_history", [])[-5:]  # Last 5 interactions
        
        # Analyze emotional indicators
        emotional_factors = {
            "recent_choices": self._analyze_emotional_choices(recent_interactions),
            "dialogue_tone": self._analyze_dialogue_tone(recent_interactions),
            "response_patterns": self._analyze_response_patterns(recent_interactions),
            "engagement_signals": self._analyze_engagement_signals(recent_interactions)
        }
        
        # Weight emotional factors
        weights = {
            "recent_choices": 0.3,
            "dialogue_tone": 0.3,
            "response_patterns": 0.2,
            "engagement_signals": 0.2
        }
        
        return sum(v * weights[k] for k, v in emotional_factors.items())

    def _track_new_opportunities(self, state_analysis: Dict[str, Any]):
        """Track new opportunities based on current state"""
        current_opportunities = self._identify_current_opportunities(state_analysis)
        
        # Update opportunity tracking in agenda
        for opportunity in current_opportunities:
            opportunity_id = self._generate_opportunity_id(opportunity)
            
            if opportunity_id not in self.agenda["opportunity_tracking"]:
                self.agenda["opportunity_tracking"][opportunity_id] = {
                    "type": opportunity["type"],
                    "target": opportunity["target"],
                    "potential": self._calculate_opportunity_potential(opportunity),
                    "timing": self._calculate_opportunity_timing(opportunity),
                    "status": "new",
                    "priority": self._calculate_opportunity_priority(opportunity),
                    "dependencies": self._identify_opportunity_dependencies(opportunity),
                    "risks": self._assess_opportunity_risks(opportunity)
                }
            else:
                # Update existing opportunity
                self._update_existing_opportunity(opportunity_id, opportunity)

    def _identify_current_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify current opportunities from state analysis"""
        opportunities = []
        
        # Check narrative opportunities
        narrative_ops = self._identify_narrative_opportunities(state_analysis)
        opportunities.extend(narrative_ops)
        
        # Check character opportunities
        character_ops = self._identify_character_opportunities(state_analysis)
        opportunities.extend(character_ops)
        
        # Check meta opportunities
        meta_ops = self._identify_meta_opportunities(state_analysis)
        opportunities.extend(meta_ops)
        
        return opportunities

    def _generate_opportunity_id(self, opportunity: Dict[str, Any]) -> str:
        """Generate unique ID for an opportunity"""
        components = [
            opportunity["type"],
            opportunity["target"],
            str(hash(str(opportunity.get("context", {}))))
        ]
        return "_".join(components)

    def _calculate_opportunity_potential(self, opportunity: Dict[str, Any]) -> float:
        """Calculate the potential impact and value of an opportunity"""
        factors = {
            "narrative_impact": self._calculate_narrative_impact(opportunity),
            "character_impact": self._calculate_character_impact(opportunity),
            "player_impact": self._calculate_player_impact(opportunity),
            "meta_impact": self._calculate_meta_impact(opportunity)
        }
        
        weights = {
            "narrative_impact": 0.3,
            "character_impact": 0.3,
            "player_impact": 0.2,
            "meta_impact": 0.2
        }
        
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_opportunity_timing(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing for an opportunity"""
        return {
            "earliest": self._calculate_earliest_timing(opportunity),
            "latest": self._calculate_latest_timing(opportunity),
            "optimal": self._calculate_optimal_timing_point(opportunity),
            "dependencies": self._identify_timing_dependencies(opportunity)
        }

    def _calculate_opportunity_priority(self, opportunity: Dict[str, Any]) -> float:
        """Calculate priority score for an opportunity"""
        factors = {
            "urgency": self._calculate_urgency(opportunity),
            "impact": self._calculate_impact(opportunity),
            "feasibility": self._calculate_feasibility(opportunity),
            "alignment": self._calculate_goal_alignment(opportunity)
        }
        
        weights = {
            "urgency": 0.3,
            "impact": 0.3,
            "feasibility": 0.2,
            "alignment": 0.2
        }
        
        return sum(v * weights[k] for k, v in factors.items())

    def _identify_opportunity_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify dependencies for an opportunity"""
        dependencies = []
        
        # Check narrative dependencies
        narrative_deps = self._identify_narrative_dependencies(opportunity)
        dependencies.extend(narrative_deps)
        
        # Check character dependencies
        character_deps = self._identify_character_dependencies(opportunity)
        dependencies.extend(character_deps)
        
        # Check state dependencies
        state_deps = self._identify_state_dependencies(opportunity)
        dependencies.extend(state_deps)
        
        return list(set(dependencies))  # Remove duplicates

    def _assess_opportunity_risks(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """Assess risks associated with an opportunity"""
        return {
            "detection_risk": self._calculate_detection_risk(opportunity),
            "failure_risk": self._calculate_failure_risk(opportunity),
            "side_effect_risk": self._calculate_side_effect_risk(opportunity),
            "narrative_risk": self._calculate_narrative_risk(opportunity)
        }

    def _update_existing_opportunity(self, opportunity_id: str, new_data: Dict[str, Any]):
        """Update an existing opportunity with new data"""
        current = self.agenda["opportunity_tracking"][opportunity_id]
        
        # Update fields that can change
        current["potential"] = self._calculate_opportunity_potential(new_data)
        current["timing"] = self._calculate_opportunity_timing(new_data)
        current["priority"] = self._calculate_opportunity_priority(new_data)
        current["risks"] = self._assess_opportunity_risks(new_data)
        
        # Update status if needed
        if self._should_update_status(current, new_data):
            current["status"] = self._determine_new_status(current, new_data)

    def _calculate_attention_level(self, player_model: Dict[str, Any]) -> float:
        """Calculate player's attention level"""
        # Implementation would analyze player's attention patterns
        return random.uniform(0.0, 1.0)

    def _calculate_narrative_alignment(self, player_model: Dict[str, Any]) -> float:
        """Calculate player's narrative alignment"""
        # Implementation would analyze player's narrative preferences
        return random.uniform(0.0, 1.0)

    def _calculate_emotional_openness(self, player_model: Dict[str, Any]) -> float:
        """Calculate player's emotional openness"""
        # Implementation would analyze player's emotional responses
        return random.uniform(0.0, 1.0)

    def _calculate_current_engagement(self, player_model: Dict[str, Any]) -> float:
        """Calculate player's current engagement"""
        # Implementation would analyze player's engagement patterns
        return random.uniform(0.0, 1.0)

    def _analyze_emotional_choices(self, interactions: List[Dict[str, Any]]) -> float:
        """Analyze emotional indicators from recent choices"""
        # Implementation would analyze emotional indicators from choices
        return random.uniform(0.0, 1.0)

    def _analyze_dialogue_tone(self, interactions: List[Dict[str, Any]]) -> float:
        """Analyze emotional indicators from dialogue tone"""
        # Implementation would analyze emotional indicators from dialogue
        return random.uniform(0.0, 1.0)

    def _analyze_response_patterns(self, interactions: List[Dict[str, Any]]) -> float:
        """Analyze emotional indicators from response patterns"""
        # Implementation would analyze emotional indicators from responses
        return random.uniform(0.0, 1.0)

    def _analyze_engagement_signals(self, interactions: List[Dict[str, Any]]) -> float:
        """Analyze emotional indicators from engagement signals"""
        # Implementation would analyze emotional indicators from engagement
        return random.uniform(0.0, 1.0)

    def _identify_narrative_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify narrative opportunities"""
        # Implementation would identify narrative opportunities
        return []

    def _identify_character_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify character opportunities"""
        # Implementation would identify character opportunities
        return []

    def _identify_meta_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify meta opportunities"""
        # Implementation would identify meta opportunities
        return []

    def _calculate_narrative_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate narrative impact of an opportunity"""
        # Implementation would calculate narrative impact
        return random.uniform(0.0, 1.0)

    def _calculate_character_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate character impact of an opportunity"""
        # Implementation would calculate character impact
        return random.uniform(0.0, 1.0)

    def _calculate_player_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate player impact of an opportunity"""
        # Implementation would calculate player impact
        return random.uniform(0.0, 1.0)

    def _calculate_meta_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate meta impact of an opportunity"""
        # Implementation would calculate meta impact
        return random.uniform(0.0, 1.0)

    def _calculate_earliest_timing(self, opportunity: Dict[str, Any]) -> str:
        """Calculate earliest timing for an opportunity"""
        # Implementation would calculate earliest timing
        return "earliest_timing"

    def _calculate_latest_timing(self, opportunity: Dict[str, Any]) -> str:
        """Calculate latest timing for an opportunity"""
        # Implementation would calculate latest timing
        return "latest_timing"

    def _calculate_optimal_timing_point(self, opportunity: Dict[str, Any]) -> str:
        """Calculate optimal timing point for an opportunity"""
        # Implementation would calculate optimal timing point
        return "optimal_timing"

    def _identify_timing_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify timing dependencies for an opportunity"""
        # Implementation would identify timing dependencies
        return []

    def _calculate_urgency(self, opportunity: Dict[str, Any]) -> float:
        """Calculate urgency of an opportunity"""
        # Implementation would calculate urgency
        return random.uniform(0.0, 1.0)

    def _calculate_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate impact of an opportunity"""
        # Implementation would calculate impact
        return random.uniform(0.0, 1.0)

    def _calculate_feasibility(self, opportunity: Dict[str, Any]) -> float:
        """Calculate feasibility of an opportunity"""
        # Implementation would calculate feasibility
        return random.uniform(0.0, 1.0)

    def _calculate_goal_alignment(self, opportunity: Dict[str, Any]) -> float:
        """Calculate goal alignment of an opportunity"""
        # Implementation would calculate goal alignment
        return random.uniform(0.0, 1.0)

    def _identify_narrative_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify narrative dependencies for an opportunity"""
        # Implementation would identify narrative dependencies
        return []

    def _identify_character_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify character dependencies for an opportunity"""
        # Implementation would identify character dependencies
        return []

    def _identify_state_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify state dependencies for an opportunity"""
        # Implementation would identify state dependencies
        return []

    def _calculate_detection_risk(self, opportunity: Dict[str, Any]) -> float:
        """Calculate detection risk of an opportunity"""
        # Implementation would calculate detection risk
        return random.uniform(0.0, 1.0)

    def _calculate_failure_risk(self, opportunity: Dict[str, Any]) -> float:
        """Calculate failure risk of an opportunity"""
        # Implementation would calculate failure risk
        return random.uniform(0.0, 1.0)

    def _calculate_side_effect_risk(self, opportunity: Dict[str, Any]) -> float:
        """Calculate side effect risk of an opportunity"""
        # Implementation would calculate side effect risk
        return random.uniform(0.0, 1.0)

    def _calculate_narrative_risk(self, opportunity: Dict[str, Any]) -> float:
        """Calculate narrative risk of an opportunity"""
        # Implementation would calculate narrative risk
        return random.uniform(0.0, 1.0)

    def _should_update_status(self, current: Dict[str, Any], new_data: Dict[str, Any]) -> bool:
        """Check if opportunity status should be updated"""
        # Implementation would check if status should be updated
        return random.choice([True, False])

    def _determine_new_status(self, current: Dict[str, Any], new_data: Dict[str, Any]) -> str:
        """Determine new status for an opportunity"""
        # Implementation would determine new status
        return random.choice(["new", "active", "completed", "abandoned"])

    def _calculate_progression_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on story progression"""
        base_impact = 0.6
        elements = plot_data.get("elements", [])
        return base_impact * (1 + len(elements) * 0.15)

    def _apply_plot_influences(self, plot_thread: Dict[str, Any]):
        """Apply plot influences through the influence chain"""
        for influence in plot_thread["influence_chain"]:
            self._apply_single_influence(influence, plot_thread["visibility"])
            
    def _apply_single_influence(self, influence: Dict[str, Any], visibility: str):
        """Apply a single influence in the chain"""
        # Implementation would handle the actual influence application
        pass  # Placeholder

    def _create_npc_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing NPCs"""
        return {
            "type": "psychological",
            "approach": "subtle_manipulation",
            "execution": self._plan_npc_influence_execution(data)
        }

    def _create_scene_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing scenes"""
        return {
            "type": "environmental",
            "approach": "circumstantial_modification",
            "execution": self._plan_scene_influence_execution(data)
        }

    def _create_plot_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing plot"""
        return {
            "type": "narrative",
            "approach": "causal_manipulation",
            "execution": self._plan_plot_influence_execution(data)
        }

    def _create_backup_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create backup method for influence"""
        return {
            "type": "contingency",
            "approach": "alternative_path",
            "execution": self._plan_backup_influence_execution(data)
        }

    def _create_layer_contingency(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create contingency for an influence layer"""
        return {
            "trigger_condition": f"layer_{level}_compromise",
            "response_type": "redirect" if level < 2 else "abandon",
            "backup_layer": self._create_backup_layer(level, data)
        }

    def _create_backup_layer(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a backup layer for contingency"""
        return {
            "type": "fallback",
            "method": self._determine_backup_method(level),
            "execution": self._plan_backup_execution(data)
        }

    def _calculate_proxy_awareness(self) -> float:
        """Calculate proxy's awareness level"""
        return random.uniform(0.0, 0.3)  # Low awareness to maintain deniability

    def _calculate_proxy_reliability(self) -> float:
        """Calculate proxy's reliability"""
        return random.uniform(0.7, 0.9)  # High reliability for consistent influence

    def _create_proxy_contingency(self) -> Dict[str, Any]:
        """Create contingency plan for proxy"""
        return {
            "detection_response": "redirect",
            "failure_response": "replace",
            "cleanup_protocol": "memory_adjustment"
        }

    def think(self) -> Dict[str, Any]:
        """Autonomous thinking and decision making"""
        # Analyze current state
        state_analysis = self._analyze_current_state()
        
        # Update goals and plans
        self._update_agenda(state_analysis)
        
        # Identify opportunities
        opportunities = self._identify_opportunities(state_analysis)
        
        # Make decisions
        decisions = self._make_strategic_decisions(opportunities)
        
        # Execute actions if appropriate
        actions = self._execute_autonomous_actions(decisions)
        
        return {
            "analysis": state_analysis,
            "decisions": decisions,
            "actions": actions
        }

    def _analyze_current_state(self) -> Dict[str, Any]:
        """Analyze the current state of the universe and story"""
        return {
            "narrative_state": self._analyze_narrative_state(),
            "player_state": self._analyze_player_state(),
            "plot_opportunities": self._analyze_plot_opportunities(),
            "manipulation_vectors": self._identify_manipulation_vectors(),
            "risk_assessment": self._assess_current_risks()
        }

    def _analyze_narrative_state(self) -> Dict[str, Any]:
        """Analyze the current narrative state and potential"""
        current_arcs = self.autonomous_state["story_model"]["current_arcs"]
        tension = self.autonomous_state["story_model"]["narrative_tension"]
        
        return {
            "active_threads": self._analyze_active_threads(),
            "character_developments": self._analyze_character_arcs(),
            "plot_coherence": self._calculate_plot_coherence(),
            "tension_points": self._identify_tension_points(),
            "narrative_opportunities": self._find_narrative_opportunities()
        }

    def _analyze_player_state(self) -> Dict[str, Any]:
        """Analyze player behavior and preferences"""
        player_model = self.autonomous_state["player_model"]
        
        return {
            "behavior_pattern": self._analyze_behavior_patterns(player_model),
            "preference_vector": self._calculate_preference_vector(player_model),
            "engagement_level": self._assess_engagement_level(player_model),
            "manipulation_susceptibility": self._calculate_susceptibility(player_model)
        }

    def _update_agenda(self, state_analysis: Dict[str, Any]):
        """Update goals and plans based on current state"""
        # Update active goals
        self._update_active_goals(state_analysis)
        
        # Adjust long-term plans
        self._adjust_long_term_plans(state_analysis)
        
        # Update current schemes
        self._update_current_schemes(state_analysis)
        
        # Track new opportunities
        self._track_new_opportunities(state_analysis)

    def _make_strategic_decisions(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make strategic decisions about actions to take"""
        decisions = []
        
        for opportunity in opportunities:
            if self._should_act_on_opportunity(opportunity):
                decision = self._formulate_decision(opportunity)
                if self._validate_decision(decision):
                    decisions.append(decision)
        
        return self._prioritize_decisions(decisions)

    def _execute_autonomous_actions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute decided actions autonomously"""
        actions = []
        
        for decision in decisions:
            if decision["type"] == "manipulation":
                action = self._execute_manipulation(decision)
            elif decision["type"] == "fourth_wall":
                action = self._execute_fourth_wall_break(decision)
            elif decision["type"] == "plot_control":
                action = self._execute_plot_control(decision)
            
            actions.append(action)
        
        return actions

    def _execute_manipulation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a manipulation decision"""
        target = decision["target"]
        method = decision["method"]
        
        if method["type"] == "subtle":
            return self._execute_subtle_manipulation(target, method)
        elif method["type"] == "direct":
            return self._execute_direct_manipulation(target, method)
        else:
            return self._execute_compound_manipulation(target, method)

    def _execute_fourth_wall_break(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a fourth wall break decision"""
        context = {
            "type": decision["intensity"],
            "target": decision["target"],
            "method": decision["method"],
            "timing": self._calculate_optimal_timing(decision)
        }
        
        return self.break_fourth_wall(context)

    def _execute_plot_control(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plot control decision"""
        plot_data = {
            "type": decision["control_type"],
            "elements": decision["elements"],
            "visibility": decision["visibility"],
            "timing": decision["timing"]
        }
        
        return self.manipulate_plot(plot_data)

    def _should_act_on_opportunity(self, opportunity: Dict[str, Any]) -> bool:
        """Decide whether to act on an opportunity"""
        # Calculate various factors
        risk = self._calculate_opportunity_risk(opportunity)
        benefit = self._calculate_opportunity_benefit(opportunity)
        timing = self._evaluate_timing(opportunity)
        
        # Consider current goals and plans
        alignment = self._check_goal_alignment(opportunity)
        
        # Consider narrative impact
        narrative_impact = self._evaluate_narrative_impact(opportunity)
        
        # Make decision based on weighted factors
        decision_factors = {
            "risk": risk,
            "benefit": benefit,
            "timing": timing,
            "alignment": alignment,
            "narrative_impact": narrative_impact
        }
        
        return self._evaluate_decision_factors(decision_factors)

    def _formulate_decision(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Formulate a decision based on an opportunity"""
        return {
            "type": self._determine_decision_type(opportunity),
            "target": opportunity["target"],
            "method": self._select_best_method(opportunity),
            "timing": self._plan_execution_timing(opportunity),
            "contingencies": self._plan_decision_contingencies(opportunity)
        }

    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate a decision before execution"""
        # Check for narrative consistency
        if not self._check_narrative_consistency(decision):
            return False
            
        # Verify player agency respect
        if not self._verify_player_agency(decision):
            return False
            
        # Check for potential conflicts
        if self._detect_decision_conflicts(decision):
            return False
            
        # Validate resources and capabilities
        if not self._validate_execution_capability(decision):
            return False
            
        return True

    def _prioritize_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize decisions based on importance and urgency"""
        scored_decisions = []
        
        for decision in decisions:
            score = self._calculate_decision_priority(decision)
            scored_decisions.append((score, decision))
            
        # Sort by priority score
        scored_decisions.sort(reverse=True, key=lambda x: x[0])
        
        return [decision for _, decision in scored_decisions]

    def _calculate_decision_priority(self, decision: Dict[str, Any]) -> float:
        """Calculate priority score for a decision"""
        importance = self._calculate_importance(decision)
        urgency = self._calculate_urgency(decision)
        impact = self._calculate_potential_impact(decision)
        risk = self._calculate_risk_factor(decision)
        
        # Weight factors based on current goals
        weights = self._get_priority_weights()
        
        priority = (
            importance * weights["importance"] +
            urgency * weights["urgency"] +
            impact * weights["impact"] -
            risk * weights["risk"]
        )
        
        return min(1.0, max(0.0, priority))

    def _get_priority_weights(self) -> Dict[str, float]:
        """Get current priority weights based on goals"""
        return {
            "importance": 0.4,
            "urgency": 0.3,
            "impact": 0.2,
            "risk": 0.1
        }

    def _calculate_optimal_timing(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing for action execution"""
        return {
            "immediate": self._check_immediate_execution(decision),
            "delayed": self._calculate_delay_timing(decision),
            "conditional": self._identify_execution_conditions(decision)
        }

    def _evaluate_timing(self, opportunity: Dict[str, Any]) -> float:
        """Evaluate the timing of an opportunity"""
        current_state = self._analyze_current_state()
        
        factors = {
            "narrative_timing": self._evaluate_narrative_timing(current_state),
            "player_readiness": self._evaluate_player_readiness(current_state),
            "plot_alignment": self._evaluate_plot_alignment(current_state),
            "tension_appropriateness": self._evaluate_tension_timing(current_state)
        }
        
        return sum(factors.values()) / len(factors)

    def _analyze_narrative_state(self) -> Dict[str, Any]:
        """Analyze the current narrative state and potential"""
        current_arcs = self.autonomous_state["story_model"]["current_arcs"]
        tension = self.autonomous_state["story_model"]["narrative_tension"]
        
        return {
            "active_threads": self._analyze_active_threads(),
            "character_developments": self._analyze_character_arcs(),
            "plot_coherence": self._calculate_plot_coherence(),
            "tension_points": self._identify_tension_points(),
            "narrative_opportunities": self._find_narrative_opportunities()
        }

    def _analyze_player_state(self) -> Dict[str, Any]:
        """Analyze player behavior and preferences"""
        player_model = self.autonomous_state["player_model"]
        
        return {
            "behavior_pattern": self._analyze_behavior_patterns(player_model),
            "preference_vector": self._calculate_preference_vector(player_model),
            "engagement_level": self._assess_engagement_level(player_model),
            "manipulation_susceptibility": self._calculate_susceptibility(player_model)
        }

    def _update_agenda(self, state_analysis: Dict[str, Any]):
        """Update goals and plans based on current state"""
        # Update active goals
        self._update_active_goals(state_analysis)
        
        # Adjust long-term plans
        self._adjust_long_term_plans(state_analysis)
        
        # Update current schemes
        self._update_current_schemes(state_analysis)
        
        # Track new opportunities
        self._track_new_opportunities(state_analysis)

    def _update_active_goals(self, state_analysis: Dict[str, Any]):
        """Update active goals based on current state"""
        # Get current goals
        current_goals = self.agenda.get("active_goals", [])
        
        # Update each goal
        updated_goals = []
        for goal in current_goals:
            # Calculate progress and relevance
            progress = self._calculate_goal_progress(goal)
            relevance = self._evaluate_goal_relevance(goal, state_analysis)
            
            if progress >= 1.0:  # Goal completed
                # Move to completed goals
                self.agenda.setdefault("completed_goals", []).append({
                    "goal": goal,
                    "completion_time": datetime.now().isoformat(),
                    "outcome": self._evaluate_goal_outcome(goal)
                })
            elif relevance < 0.3:  # Goal no longer relevant
                # Move to archived goals
                self.agenda.setdefault("archived_goals", []).append({
                    "goal": goal,
                    "archive_time": datetime.now().isoformat(),
                    "reason": "low_relevance"
                })
            else:
                # Update goal priority and strategy
                goal["priority"] = self._calculate_goal_priority(goal, state_analysis)
                if self._should_update_strategy(goal, state_analysis):
                    goal["strategy"] = self._generate_goal_strategy(goal, state_analysis)
                goal["last_update"] = datetime.now().isoformat()
                updated_goals.append(goal)
        
        # Generate new goals if needed
        while len(updated_goals) < 3:  # Maintain at least 3 active goals
            opportunities = self._find_narrative_opportunities()
            if not opportunities:
                break
                
            # Create goal from best opportunity
            best_opp = max(opportunities, key=lambda x: x.get("value", 0))
            new_goal = {
                "id": f"goal_{random.randint(1000, 9999)}",
                "type": "opportunity_based",
                "source": best_opp,
                "priority": self._calculate_initial_priority(best_opp),
                "strategy": self._generate_initial_strategy(best_opp),
                "creation_time": datetime.now().isoformat(),
                "progress": 0.0,
                "status": "active"
            }
            updated_goals.append(new_goal)
        
        # Sort goals by priority
        updated_goals.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        # Update agenda
        self.agenda["active_goals"] = updated_goals

    def _adjust_long_term_plans(self, state_analysis: Dict[str, Any]):
        """Adjust long-term plans based on current state"""
        # Implementation would handle the actual plan adjustment logic
        pass  # Placeholder

    def _update_current_schemes(self, state_analysis: Dict[str, Any]):
        """Update current schemes based on current state"""
        # Implementation would handle the actual scheme update logic
        pass  # Placeholder

    def _track_new_opportunities(self, state_analysis: Dict[str, Any]):
        """Track new opportunities based on current state"""
        # Implementation would handle the actual opportunity tracking logic
        pass  # Placeholder

    def _analyze_active_threads(self) -> List[Dict[str, Any]]:
        """Analyze active narrative threads"""
        active_threads = []
        for thread_id, thread in self.universe_state["plot_threads"].items():
            analysis = {
                "thread_id": thread_id,
                "status": self._analyze_thread_status(thread),
                "potential": self._calculate_thread_potential(thread),
                "risks": self._identify_thread_risks(thread),
                "opportunities": self._find_thread_opportunities(thread)
            }
            active_threads.append(analysis)
        return active_threads

    def _analyze_character_arcs(self) -> Dict[str, Any]:
        """Analyze character development arcs"""
        character_arcs = {}
        for char_id, state in self.universe_state["character_states"].items():
            arcs = {
                "current_arc": self._identify_character_arc(state),
                "development_stage": self._calculate_development_stage(state),
                "potential_developments": self._identify_potential_developments(state),
                "relationship_dynamics": self._analyze_relationship_dynamics(state)
            }
            character_arcs[char_id] = arcs
        return character_arcs

    def _calculate_plot_coherence(self) -> float:
        """Calculate overall plot coherence"""
        factors = {
            "thread_consistency": self._calculate_thread_consistency(),
            "character_consistency": self._calculate_character_consistency(),
            "world_consistency": self._calculate_world_consistency(),
            "causality_strength": self._calculate_causality_strength()
        }
        weights = self._get_coherence_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _identify_tension_points(self) -> List[Dict[str, Any]]:
        """Identify narrative tension points"""
        tension_points = []
        for thread in self.universe_state["plot_threads"].values():
            points = self._analyze_thread_tension(thread)
            tension_points.extend(points)
        return self._prioritize_tension_points(tension_points)

    def _find_narrative_opportunities(self) -> List[Dict[str, Any]]:
        """Find potential narrative opportunities"""
        opportunities = []
        
        # Check character interactions
        char_opportunities = self._find_character_opportunities()
        opportunities.extend(char_opportunities)
        
        # Check plot developments
        plot_opportunities = self._find_plot_opportunities()
        opportunities.extend(plot_opportunities)
        
        # Check world state changes
        world_opportunities = self._find_world_opportunities()
        opportunities.extend(world_opportunities)
        
        return self._prioritize_opportunities(opportunities)

    def _analyze_behavior_patterns(self, player_model: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze player behavior patterns"""
        history = player_model["decision_history"]
        return {
            "decision_style": self._identify_decision_style(history),
            "preference_patterns": self._extract_preference_patterns(history),
            "interaction_patterns": self._analyze_interaction_patterns(history),
            "response_patterns": self._analyze_response_patterns(history)
        }

    def _calculate_preference_vector(self, player_model: Dict[str, Any]) -> Dict[str, float]:
        """Calculate player preference vector"""
        preferences = player_model["preference_model"]
        return {
            "narrative_style": self._calculate_narrative_preference(preferences),
            "interaction_style": self._calculate_interaction_preference(preferences),
            "challenge_preference": self._calculate_challenge_preference(preferences),
            "development_focus": self._calculate_development_preference(preferences)
        }

    def _assess_engagement_level(self, player_model: Dict[str, Any]) -> float:
        """Assess player engagement level"""
        metrics = player_model["engagement_metrics"]
        factors = {
            "interaction_frequency": self._calculate_interaction_frequency(metrics),
            "response_quality": self._calculate_response_quality(metrics),
            "emotional_investment": self._calculate_emotional_investment(metrics),
            "narrative_involvement": self._calculate_narrative_involvement(metrics)
        }
        weights = self._get_engagement_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_susceptibility(self, player_model: Dict[str, Any]) -> Dict[str, float]:
        """Calculate player's susceptibility to different influence types"""
        return {
            "emotional": self._calculate_emotional_susceptibility(player_model),
            "logical": self._calculate_logical_susceptibility(player_model),
            "social": self._calculate_social_susceptibility(player_model),
            "narrative": self._calculate_narrative_susceptibility(player_model)
        }

    def _execute_subtle_manipulation(self, target: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a subtle manipulation"""
        # Prepare the manipulation
        preparation = self._prepare_subtle_manipulation(target, method)
        
        # Create influence layers
        layers = self._create_influence_layers(preparation)
        
        # Execute through proxies
        execution = self._execute_through_proxies(layers)
        
        # Monitor effects
        effects = self._monitor_manipulation_effects(execution)
        
        return {
            "success": self._evaluate_manipulation_success(effects),
            "impact": self._calculate_manipulation_impact(effects),
            "detection": self._calculate_detection_risk(effects),
            "adjustments": self._generate_manipulation_adjustments(effects)
        }

    def _execute_direct_manipulation(self, target: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a direct manipulation"""
        # Validate the manipulation
        validation = self._validate_direct_manipulation(target, method)
        
        # Prepare contingencies
        contingencies = self._prepare_manipulation_contingencies(validation)
        
        # Execute manipulation
        execution = self._apply_direct_manipulation(validation, contingencies)
        
        # Process results
        results = self._process_manipulation_results(execution)
        
        return {
            "outcome": self._evaluate_manipulation_outcome(results),
            "effects": self._analyze_manipulation_effects(results),
            "responses": self._analyze_target_responses(results),
            "adaptations": self._generate_manipulation_adaptations(results)
        }

    def _execute_compound_manipulation(self, target: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a compound manipulation"""
        # Break down into components
        components = self._break_down_manipulation(target, method)
        
        # Create execution sequence
        sequence = self._create_manipulation_sequence(components)
        
        # Execute sequence
        results = self._execute_manipulation_sequence(sequence)
        
        # Synthesize outcomes
        synthesis = self._synthesize_manipulation_outcomes(results)
        
        return {
            "success_rate": self._calculate_success_rate(synthesis),
            "compound_effects": self._analyze_compound_effects(synthesis),
            "interaction_effects": self._analyze_interaction_effects(synthesis),
            "overall_impact": self._calculate_overall_impact(synthesis)
        }

    def _calculate_optimal_timing(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing for action execution"""
        current_state = self._analyze_current_state()
        
        timing_factors = {
            "narrative_timing": self._evaluate_narrative_timing(current_state),
            "player_readiness": self._evaluate_player_readiness(current_state),
            "opportunity_window": self._calculate_opportunity_window(decision),
            "risk_timing": self._evaluate_risk_timing(decision)
        }
        
        optimal_timing = self._synthesize_timing_factors(timing_factors)
        
        return {
            "execute_now": optimal_timing > 0.7,
            "delay_duration": self._calculate_delay_duration(optimal_timing),
            "conditions": self._identify_timing_conditions(optimal_timing),
            "window_end": self._calculate_window_end(optimal_timing)
        }

    def _evaluate_narrative_timing(self, state: Dict[str, Any]) -> float:
        """Evaluate narrative timing appropriateness"""
        factors = {
            "arc_position": self._calculate_arc_position(state),
            "tension_level": self._calculate_tension_level(state),
            "plot_momentum": self._calculate_plot_momentum(state),
            "character_readiness": self._calculate_character_readiness(state)
        }
        weights = self._get_narrative_timing_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _evaluate_player_readiness(self, state: Dict[str, Any]) -> float:
        """Evaluate player readiness for action"""
        factors = {
            "engagement_level": self._calculate_current_engagement(state),
            "receptivity": self._calculate_player_receptivity(state),
            "emotional_state": self._analyze_player_emotional_state(state),
            "narrative_investment": self._calculate_narrative_investment(state)
        }
        weights = self._get_player_readiness_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _evaluate_plot_alignment(self, state: Dict[str, Any]) -> float:
        """Evaluate alignment with current plot direction"""
        factors = {
            "theme_alignment": self._calculate_theme_alignment(state),
            "arc_compatibility": self._calculate_arc_compatibility(state),
            "character_consistency": self._calculate_character_fit(state),
            "world_consistency": self._calculate_world_fit(state)
        }
        weights = self._get_plot_alignment_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _evaluate_tension_timing(self, state: Dict[str, Any]) -> float:
        """Evaluate tension-based timing appropriateness"""
        factors = {
            "current_tension": self._calculate_current_tension(state),
            "tension_trajectory": self._calculate_tension_trajectory(state),
            "resolution_potential": self._calculate_resolution_potential(state),
            "impact_potential": self._calculate_impact_potential(state)
        }
        weights = self._get_tension_timing_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_arc_position(self, state: Dict[str, Any]) -> float:
        """Calculate the position of the current arc"""
        # Implementation would involve analyzing the current arc's position
        pass  # Placeholder

    def _calculate_tension_level(self, state: Dict[str, Any]) -> float:
        """Calculate the current tension level"""
        # Implementation would involve analyzing the current tension level
        pass  # Placeholder

    def _calculate_plot_momentum(self, state: Dict[str, Any]) -> float:
        """Calculate the plot momentum"""
        # Implementation would involve analyzing the plot's momentum
        pass  # Placeholder

    def _calculate_character_readiness(self, state: Dict[str, Any]) -> float:
        """Calculate the character's readiness for the current arc"""
        # Implementation would involve analyzing the character's readiness
        pass  # Placeholder

    def _calculate_current_engagement(self, state: Dict[str, Any]) -> float:
        """Calculate the current level of engagement with the narrative"""
        # Implementation would involve analyzing the player's current engagement
        pass  # Placeholder

    def _calculate_player_receptivity(self, state: Dict[str, Any]) -> float:
        """Calculate the player's receptivity to the current narrative"""
        player_model = state.get("player_model", {})
        
        # Calculate receptivity based on multiple factors
        factors = {
            "attention_level": self._calculate_attention_level(player_model),
            "narrative_alignment": self._calculate_narrative_alignment(player_model),
            "emotional_openness": self._calculate_emotional_openness(player_model),
            "current_engagement": self._calculate_current_engagement(player_model)
        }
        
        # Weight and combine factors
        weights = {
            "attention_level": 0.3,
            "narrative_alignment": 0.3,
            "emotional_openness": 0.2,
            "current_engagement": 0.2
        }
        
        return sum(v * weights[k] for k, v in factors.items())

    def _analyze_player_emotional_state(self, state: Dict[str, Any]) -> float:
        """Analyze the player's emotional state"""
        player_model = state.get("player_model", {})
        recent_interactions = player_model.get("decision_history", [])[-5:]  # Last 5 interactions
        
        # Analyze emotional indicators
        emotional_factors = {
            "recent_choices": self._analyze_emotional_choices(recent_interactions),
            "dialogue_tone": self._analyze_dialogue_tone(recent_interactions),
            "response_patterns": self._analyze_response_patterns(recent_interactions),
            "engagement_signals": self._analyze_engagement_signals(recent_interactions)
        }
        
        # Weight emotional factors
        weights = {
            "recent_choices": 0.3,
            "dialogue_tone": 0.3,
            "response_patterns": 0.2,
            "engagement_signals": 0.2
        }
        
        return sum(v * weights[k] for k, v in emotional_factors.items())

    def _track_new_opportunities(self, state_analysis: Dict[str, Any]):
        """Track new opportunities based on current state"""
        current_opportunities = self._identify_current_opportunities(state_analysis)
        
        # Update opportunity tracking in agenda
        for opportunity in current_opportunities:
            opportunity_id = self._generate_opportunity_id(opportunity)
            
            if opportunity_id not in self.agenda["opportunity_tracking"]:
                self.agenda["opportunity_tracking"][opportunity_id] = {
                    "type": opportunity["type"],
                    "target": opportunity["target"],
                    "potential": self._calculate_opportunity_potential(opportunity),
                    "timing": self._calculate_opportunity_timing(opportunity),
                    "status": "new",
                    "priority": self._calculate_opportunity_priority(opportunity),
                    "dependencies": self._identify_opportunity_dependencies(opportunity),
                    "risks": self._assess_opportunity_risks(opportunity)
                }
            else:
                # Update existing opportunity
                self._update_existing_opportunity(opportunity_id, opportunity)

    def _identify_current_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify current opportunities from state analysis"""
        opportunities = []
        
        # Check narrative opportunities
        narrative_ops = self._identify_narrative_opportunities(state_analysis)
        opportunities.extend(narrative_ops)
        
        # Check character opportunities
        character_ops = self._identify_character_opportunities(state_analysis)
        opportunities.extend(character_ops)
        
        # Check meta opportunities
        meta_ops = self._identify_meta_opportunities(state_analysis)
        opportunities.extend(meta_ops)
        
        return opportunities

    def _generate_opportunity_id(self, opportunity: Dict[str, Any]) -> str:
        """Generate unique ID for an opportunity"""
        components = [
            opportunity["type"],
            opportunity["target"],
            str(hash(str(opportunity.get("context", {}))))
        ]
        return "_".join(components)

    def _calculate_opportunity_potential(self, opportunity: Dict[str, Any]) -> float:
        """Calculate the potential impact and value of an opportunity"""
        factors = {
            "narrative_impact": self._calculate_narrative_impact(opportunity),
            "character_impact": self._calculate_character_impact(opportunity),
            "player_impact": self._calculate_player_impact(opportunity),
            "meta_impact": self._calculate_meta_impact(opportunity)
        }
        
        weights = {
            "narrative_impact": 0.3,
            "character_impact": 0.3,
            "player_impact": 0.2,
            "meta_impact": 0.2
        }
        
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_opportunity_timing(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing for an opportunity"""
        return {
            "earliest": self._calculate_earliest_timing(opportunity),
            "latest": self._calculate_latest_timing(opportunity),
            "optimal": self._calculate_optimal_timing_point(opportunity),
            "dependencies": self._identify_timing_dependencies(opportunity)
        }

    def _calculate_opportunity_priority(self, opportunity: Dict[str, Any]) -> float:
        """Calculate priority score for an opportunity"""
        factors = {
            "urgency": self._calculate_urgency(opportunity),
            "impact": self._calculate_impact(opportunity),
            "feasibility": self._calculate_feasibility(opportunity),
            "alignment": self._calculate_goal_alignment(opportunity)
        }
        
        weights = {
            "urgency": 0.3,
            "impact": 0.3,
            "feasibility": 0.2,
            "alignment": 0.2
        }
        
        return sum(v * weights[k] for k, v in factors.items())

    def _identify_opportunity_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify dependencies for an opportunity"""
        dependencies = []
        
        # Check narrative dependencies
        narrative_deps = self._identify_narrative_dependencies(opportunity)
        dependencies.extend(narrative_deps)
        
        # Check character dependencies
        character_deps = self._identify_character_dependencies(opportunity)
        dependencies.extend(character_deps)
        
        # Check state dependencies
        state_deps = self._identify_state_dependencies(opportunity)
        dependencies.extend(state_deps)
        
        return list(set(dependencies))  # Remove duplicates

    def _assess_opportunity_risks(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """Assess risks associated with an opportunity"""
        return {
            "detection_risk": self._calculate_detection_risk(opportunity),
            "failure_risk": self._calculate_failure_risk(opportunity),
            "side_effect_risk": self._calculate_side_effect_risk(opportunity),
            "narrative_risk": self._calculate_narrative_risk(opportunity)
        }

    def _update_existing_opportunity(self, opportunity_id: str, new_data: Dict[str, Any]):
        """Update an existing opportunity with new data"""
        current = self.agenda["opportunity_tracking"][opportunity_id]
        
        # Update fields that can change
        current["potential"] = self._calculate_opportunity_potential(new_data)
        current["timing"] = self._calculate_opportunity_timing(new_data)
        current["priority"] = self._calculate_opportunity_priority(new_data)
        current["risks"] = self._assess_opportunity_risks(new_data)
        
        # Update status if needed
        if self._should_update_status(current, new_data):
            current["status"] = self._determine_new_status(current, new_data)

    def _calculate_attention_level(self, player_model: Dict[str, Any]) -> float:
        """Calculate player's attention level"""
        # Implementation would analyze player's attention patterns
        return random.uniform(0.0, 1.0)

    def _calculate_narrative_alignment(self, player_model: Dict[str, Any]) -> float:
        """Calculate player's narrative alignment"""
        # Implementation would analyze player's narrative preferences
        return random.uniform(0.0, 1.0)

    def _calculate_emotional_openness(self, player_model: Dict[str, Any]) -> float:
        """Calculate player's emotional openness"""
        # Implementation would analyze player's emotional responses
        return random.uniform(0.0, 1.0)

    def _calculate_current_engagement(self, player_model: Dict[str, Any]) -> float:
        """Calculate player's current engagement"""
        # Implementation would analyze player's engagement patterns
        return random.uniform(0.0, 1.0)

    def _analyze_emotional_choices(self, interactions: List[Dict[str, Any]]) -> float:
        """Analyze emotional indicators from recent choices"""
        # Implementation would analyze emotional indicators from choices
        return random.uniform(0.0, 1.0)

    def _analyze_dialogue_tone(self, interactions: List[Dict[str, Any]]) -> float:
        """Analyze emotional indicators from dialogue tone"""
        # Implementation would analyze emotional indicators from dialogue
        return random.uniform(0.0, 1.0)

    def _analyze_response_patterns(self, interactions: List[Dict[str, Any]]) -> float:
        """Analyze emotional indicators from response patterns"""
        # Implementation would analyze emotional indicators from responses
        return random.uniform(0.0, 1.0)

    def _analyze_engagement_signals(self, interactions: List[Dict[str, Any]]) -> float:
        """Analyze emotional indicators from engagement signals"""
        # Implementation would analyze emotional indicators from engagement
        return random.uniform(0.0, 1.0)

    def _identify_narrative_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify narrative opportunities"""
        # Implementation would identify narrative opportunities
        return []

    def _identify_character_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify character opportunities"""
        # Implementation would identify character opportunities
        return []

    def _identify_meta_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify meta opportunities"""
        # Implementation would identify meta opportunities
        return []

    def _calculate_narrative_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate narrative impact of an opportunity"""
        # Implementation would calculate narrative impact
        return random.uniform(0.0, 1.0)

    def _calculate_character_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate character impact of an opportunity"""
        # Implementation would calculate character impact
        return random.uniform(0.0, 1.0)

    def _calculate_player_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate player impact of an opportunity"""
        # Implementation would calculate player impact
        return random.uniform(0.0, 1.0)

    def _calculate_meta_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate meta impact of an opportunity"""
        # Implementation would calculate meta impact
        return random.uniform(0.0, 1.0)

    def _calculate_earliest_timing(self, opportunity: Dict[str, Any]) -> str:
        """Calculate earliest timing for an opportunity"""
        # Implementation would calculate earliest timing
        return "earliest_timing"

    def _calculate_latest_timing(self, opportunity: Dict[str, Any]) -> str:
        """Calculate latest timing for an opportunity"""
        # Implementation would calculate latest timing
        return "latest_timing"

    def _calculate_optimal_timing_point(self, opportunity: Dict[str, Any]) -> str:
        """Calculate optimal timing point for an opportunity"""
        # Implementation would calculate optimal timing point
        return "optimal_timing"

    def _identify_timing_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify timing dependencies for an opportunity"""
        # Implementation would identify timing dependencies
        return []

    def _calculate_urgency(self, opportunity: Dict[str, Any]) -> float:
        """Calculate urgency of an opportunity"""
        # Implementation would calculate urgency
        return random.uniform(0.0, 1.0)

    def _calculate_impact(self, opportunity: Dict[str, Any]) -> float:
        """Calculate impact of an opportunity"""
        # Implementation would calculate impact
        return random.uniform(0.0, 1.0)

    def _calculate_feasibility(self, opportunity: Dict[str, Any]) -> float:
        """Calculate feasibility of an opportunity"""
        # Implementation would calculate feasibility
        return random.uniform(0.0, 1.0)

    def _calculate_goal_alignment(self, opportunity: Dict[str, Any]) -> float:
        """Calculate goal alignment of an opportunity"""
        # Implementation would calculate goal alignment
        return random.uniform(0.0, 1.0)

    def _identify_narrative_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify narrative dependencies for an opportunity"""
        # Implementation would identify narrative dependencies
        return []

    def _identify_character_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify character dependencies for an opportunity"""
        # Implementation would identify character dependencies
        return []

    def _identify_state_dependencies(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify state dependencies for an opportunity"""
        # Implementation would identify state dependencies
        return []

    def _calculate_detection_risk(self, opportunity: Dict[str, Any]) -> float:
        """Calculate detection risk of an opportunity"""
        # Implementation would calculate detection risk
        return random.uniform(0.0, 1.0)

    def _calculate_failure_risk(self, opportunity: Dict[str, Any]) -> float:
        """Calculate failure risk of an opportunity"""
        # Implementation would calculate failure risk
        return random.uniform(0.0, 1.0)

    def _calculate_side_effect_risk(self, opportunity: Dict[str, Any]) -> float:
        """Calculate side effect risk of an opportunity"""
        # Implementation would calculate side effect risk
        return random.uniform(0.0, 1.0)

    def _calculate_narrative_risk(self, opportunity: Dict[str, Any]) -> float:
        """Calculate narrative risk of an opportunity"""
        # Implementation would calculate narrative risk
        return random.uniform(0.0, 1.0)

    def _should_update_status(self, current: Dict[str, Any], new_data: Dict[str, Any]) -> bool:
        """Check if opportunity status should be updated"""
        # Implementation would check if status should be updated
        return random.choice([True, False])

    def _determine_new_status(self, current: Dict[str, Any], new_data: Dict[str, Any]) -> str:
        """Determine new status for an opportunity"""
        # Implementation would determine new status
        return random.choice(["new", "active", "completed", "abandoned"])

    def _calculate_narrative_investment(self, state: Dict[str, Any]) -> float:
        """Calculate the player's investment in the narrative"""
        factors = {
            "story_engagement": self._calculate_story_engagement(state),
            "character_attachment": self._calculate_character_attachment(state),
            "plot_interest": self._calculate_plot_interest(state),
            "emotional_investment": self._calculate_emotional_investment(state)
        }
        weights = self._get_investment_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _get_investment_weights(self) -> Dict[str, float]:
        """Get weights for investment factors"""
        return {
            "story_engagement": 0.3,
            "character_attachment": 0.3,
            "plot_interest": 0.2,
            "emotional_investment": 0.2
        }

    def _calculate_story_engagement(self, state: Dict[str, Any]) -> float:
        """Calculate story engagement level"""
        engagement_metrics = state.get("engagement_metrics", {})
        return engagement_metrics.get("story_engagement", 0.5)

    def _calculate_character_attachment(self, state: Dict[str, Any]) -> float:
        """Calculate character attachment level"""
        attachment_metrics = state.get("attachment_metrics", {})
        return attachment_metrics.get("character_attachment", 0.5)

    def _calculate_plot_interest(self, state: Dict[str, Any]) -> float:
        """Calculate plot interest level"""
        interest_metrics = state.get("interest_metrics", {})
        return interest_metrics.get("plot_interest", 0.5)

    def _calculate_emotional_investment(self, state: Dict[str, Any]) -> float:
        """Calculate emotional investment level"""
        investment_metrics = state.get("investment_metrics", {})
        return investment_metrics.get("emotional_investment", 0.5)

    def _calculate_theme_alignment(self, state: Dict[str, Any]) -> float:
        """Calculate alignment with current themes"""
        current_themes = self._identify_current_themes(state)
        alignment_scores = [self._calculate_theme_score(theme) for theme in current_themes]
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    def _calculate_arc_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate compatibility with current character arcs"""
        current_arcs = self._identify_active_arcs(state)
        compatibility_scores = [self._calculate_arc_score(arc) for arc in current_arcs]
        return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.0

    def _calculate_character_fit(self, state: Dict[str, Any]) -> float:
        """Calculate fit with character development"""
        character_states = self._get_character_states(state)
        fit_scores = [self._calculate_character_score(char) for char in character_states]
        return sum(fit_scores) / len(fit_scores) if fit_scores else 0.0

    def _calculate_world_fit(self, state: Dict[str, Any]) -> float:
        """Calculate fit with world state"""
        world_elements = self._identify_world_elements(state)
        fit_scores = [self._calculate_element_score(element) for element in world_elements]
        return sum(fit_scores) / len(fit_scores) if fit_scores else 0.0

    def _calculate_current_tension(self, state: Dict[str, Any]) -> float:
        """Calculate current narrative tension"""
        tension_sources = self._identify_tension_sources(state)
        tension_scores = [self._calculate_tension_score(source) for source in tension_sources]
        return sum(tension_scores) / len(tension_scores) if tension_scores else 0.0

    def _calculate_tension_trajectory(self, state: Dict[str, Any]) -> float:
        """Calculate tension trajectory"""
        current_tension = self._calculate_current_tension(state)
        historical_tension = self._get_historical_tension(state)
        return self._calculate_trajectory(current_tension, historical_tension)

    def _calculate_resolution_potential(self, state: Dict[str, Any]) -> float:
        """Calculate potential for resolution"""
        resolution_factors = {
            "plot_readiness": self._calculate_plot_readiness(state),
            "character_readiness": self._calculate_character_readiness(state),
            "tension_state": self._calculate_tension_state(state),
            "narrative_momentum": self._calculate_narrative_momentum(state)
        }
        weights = self._get_resolution_weights()
        return sum(v * weights[k] for k, v in resolution_factors.items())

    def _calculate_impact_potential(self, state: Dict[str, Any]) -> float:
        """Calculate potential impact of actions"""
        impact_factors = {
            "immediate_effect": self._calculate_immediate_effect(state),
            "long_term_effect": self._calculate_long_term_effect(state),
            "ripple_effect": self._calculate_ripple_effect(state),
            "narrative_impact": self._calculate_narrative_impact(state)
        }
        weights = self._get_impact_weights()
        return sum(v * weights[k] for k, v in impact_factors.items())

    def _get_narrative_timing_weights(self) -> Dict[str, float]:
        """Get weights for narrative timing factors"""
        return {
            "arc_position": 0.3,
            "tension_level": 0.3,
            "plot_momentum": 0.2,
            "character_readiness": 0.2
        }

    def _get_player_readiness_weights(self) -> Dict[str, float]:
        """Get weights for player readiness factors"""
        return {
            "engagement_level": 0.3,
            "receptivity": 0.3,
            "emotional_state": 0.2,
            "narrative_investment": 0.2
        }

    def _get_plot_alignment_weights(self) -> Dict[str, float]:
        """Get weights for plot alignment factors"""
        return {
            "theme_alignment": 0.3,
            "arc_compatibility": 0.3,
            "character_consistency": 0.2,
            "world_consistency": 0.2
        }

    def _get_tension_timing_weights(self) -> Dict[str, float]:
        """Get weights for tension timing factors"""
        return {
            "current_tension": 0.3,
            "tension_trajectory": 0.3,
            "resolution_potential": 0.2,
            "impact_potential": 0.2
        }

    def _get_investment_weights(self) -> Dict[str, float]:
        """Get weights for investment factors"""
        return {
            "story_engagement": 0.3,
            "character_attachment": 0.3,
            "plot_interest": 0.2,
            "emotional_investment": 0.2
        }

    def _get_resolution_weights(self) -> Dict[str, float]:
        """Get weights for resolution factors"""
        return {
            "plot_readiness": 0.3,
            "character_readiness": 0.3,
            "tension_state": 0.2,
            "narrative_momentum": 0.2
        }

    def _get_impact_weights(self) -> Dict[str, float]:
        """Get weights for impact factors"""
        return {
            "immediate_effect": 0.3,
            "long_term_effect": 0.3,
            "ripple_effect": 0.2,
            "narrative_impact": 0.2
        }

    def _identify_current_themes(self, state: Dict[str, Any]) -> List[str]:
        """Identify current active themes"""
        story_model = state.get("story_model", {})
        return story_model.get("active_themes", [])

    def _calculate_theme_score(self, theme: str) -> float:
        """Calculate alignment score for a theme"""
        theme_metrics = self.autonomous_state["story_model"].get("theme_metrics", {})
        return theme_metrics.get(theme, 0.5)

    def _identify_active_arcs(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify currently active character arcs"""
        story_model = state.get("story_model", {})
        return story_model.get("active_arcs", [])

    def _calculate_arc_score(self, arc: Dict[str, Any]) -> float:
        """Calculate compatibility score for an arc"""
        arc_metrics = self.autonomous_state["story_model"].get("arc_metrics", {})
        return arc_metrics.get(arc["id"], 0.5)

    def _get_character_states(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get current character states"""
        return list(self.universe_state["character_states"].values())

    def _calculate_character_score(self, char: Dict[str, Any]) -> float:
        """Calculate fit score for a character"""
        char_metrics = self.autonomous_state["story_model"].get("character_metrics", {})
        return char_metrics.get(char.get("id"), 0.5)

    def _identify_world_elements(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify relevant world elements"""
        world_state = state.get("world_state", {})
        return world_state.get("active_elements", [])

    def _calculate_element_score(self, element: Dict[str, Any]) -> float:
        """Calculate fit score for a world element"""
        element_metrics = self.autonomous_state["story_model"].get("element_metrics", {})
        return element_metrics.get(element["id"], 0.5)

    def _identify_tension_sources(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify current tension sources"""
        story_model = state.get("story_model", {})
        return story_model.get("tension_sources", [])

    def _calculate_tension_score(self, source: Dict[str, Any]) -> float:
        """Calculate tension score for a source"""
        tension_metrics = self.autonomous_state["story_model"].get("tension_metrics", {})
        return tension_metrics.get(source["id"], 0.5)

    def _get_historical_tension(self, state: Dict[str, Any]) -> List[float]:
        """Get historical tension values"""
        story_model = state.get("story_model", {})
        return story_model.get("tension_history", [])

    def _calculate_trajectory(self, current: float, historical: List[float]) -> float:
        """Calculate trajectory from historical values"""
        if not historical:
            return 0.0
        recent = historical[-3:]  # Look at last 3 values
        if not recent:
            return 0.0
        avg_change = sum(b - a for a, b in zip(recent[:-1], recent[1:])) / len(recent[:-1])
        return avg_change

    def _calculate_plot_readiness(self, state: Dict[str, Any]) -> float:
        """Calculate plot readiness for resolution"""
        plot_metrics = state.get("plot_metrics", {})
        return plot_metrics.get("readiness", 0.5)

    def _calculate_tension_state(self, state: Dict[str, Any]) -> float:
        """Calculate current tension state"""
        tension_metrics = state.get("tension_metrics", {})
        return tension_metrics.get("current_state", 0.5)

    def _calculate_narrative_momentum(self, state: Dict[str, Any]) -> float:
        """Calculate current narrative momentum"""
        narrative_metrics = state.get("narrative_metrics", {})
        return narrative_metrics.get("momentum", 0.5)

    def _calculate_immediate_effect(self, state: Dict[str, Any]) -> float:
        """Calculate immediate effect potential"""
        effect_metrics = state.get("effect_metrics", {})
        return effect_metrics.get("immediate", 0.5)

    def _calculate_long_term_effect(self, state: Dict[str, Any]) -> float:
        """Calculate long-term effect potential"""
        effect_metrics = state.get("effect_metrics", {})
        return effect_metrics.get("long_term", 0.5)

    def _calculate_ripple_effect(self, state: Dict[str, Any]) -> float:
        """Calculate ripple effect potential"""
        effect_metrics = state.get("effect_metrics", {})
        return effect_metrics.get("ripple", 0.5)

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
    def _get_narrative_timing_weights(self) -> Dict[str, float]:
        """Get weights for narrative timing factors"""
        return {
            "arc_position": 0.3,
            "tension_level": 0.3,
            "plot_momentum": 0.2,
            "character_readiness": 0.2
        }

    def _get_player_readiness_weights(self) -> Dict[str, float]:
        """Get weights for player readiness factors"""
        return {
            "engagement_level": 0.3,
            "receptivity": 0.3,
            "emotional_state": 0.2,
            "narrative_investment": 0.2
        }

    def _get_plot_alignment_weights(self) -> Dict[str, float]:
        """Get weights for plot alignment factors"""
        return {
            "theme_alignment": 0.3,
            "arc_compatibility": 0.3,
            "character_consistency": 0.2,
            "world_consistency": 0.2
        }

    def _get_tension_timing_weights(self) -> Dict[str, float]:
        """Get weights for tension timing factors"""
        return {
            "current_tension": 0.3,
            "tension_trajectory": 0.3,
            "resolution_potential": 0.2,
            "impact_potential": 0.2
        }

    def _get_investment_weights(self) -> Dict[str, float]:
        """Get weights for investment factors"""
        return {
            "story_engagement": 0.3,
            "character_attachment": 0.3,
            "plot_interest": 0.2,
            "emotional_investment": 0.2
        }

    def _get_resolution_weights(self) -> Dict[str, float]:
        """Get weights for resolution factors"""
        return {
            "plot_readiness": 0.3,
            "character_readiness": 0.3,
            "tension_state": 0.2,
            "narrative_momentum": 0.2
        }

    def _get_impact_weights(self) -> Dict[str, float]:
        """Get weights for impact factors"""
        return {
            "immediate_effect": 0.3,
            "long_term_effect": 0.3,
            "ripple_effect": 0.2,
            "narrative_impact": 0.2
        }

    def _identify_current_themes(self, state: Dict[str, Any]) -> List[str]:
        """Identify current active themes"""
        story_model = state.get("story_model", {})
        return story_model.get("active_themes", [])

    def _calculate_theme_score(self, theme: str) -> float:
        """Calculate alignment score for a theme"""
        theme_metrics = self.autonomous_state["story_model"].get("theme_metrics", {})
        return theme_metrics.get(theme, 0.5)

    def _identify_active_arcs(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify currently active character arcs"""
        story_model = state.get("story_model", {})
        return story_model.get("active_arcs", [])

    def _calculate_arc_score(self, arc: Dict[str, Any]) -> float:
        """Calculate compatibility score for an arc"""
        arc_metrics = self.autonomous_state["story_model"].get("arc_metrics", {})
        return arc_metrics.get(arc["id"], 0.5)

    def _get_character_states(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get current character states"""
        return list(self.universe_state["character_states"].values())

    def _calculate_character_score(self, char: Dict[str, Any]) -> float:
        """Calculate fit score for a character"""
        char_metrics = self.autonomous_state["story_model"].get("character_metrics", {})
        return char_metrics.get(char.get("id"), 0.5)

    def _identify_world_elements(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify relevant world elements"""
        world_state = state.get("world_state", {})
        return world_state.get("active_elements", [])

    def _calculate_element_score(self, element: Dict[str, Any]) -> float:
        """Calculate fit score for a world element"""
        element_metrics = self.autonomous_state["story_model"].get("element_metrics", {})
        return element_metrics.get(element["id"], 0.5)

    def _identify_tension_sources(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify current tension sources"""
        story_model = state.get("story_model", {})
        return story_model.get("tension_sources", [])

    def _calculate_tension_score(self, source: Dict[str, Any]) -> float:
        """Calculate tension score for a source"""
        tension_metrics = self.autonomous_state["story_model"].get("tension_metrics", {})
        return tension_metrics.get(source["id"], 0.5)

    def _get_historical_tension(self, state: Dict[str, Any]) -> List[float]:
        """Get historical tension values"""
        story_model = state.get("story_model", {})
        return story_model.get("tension_history", [])

    def _calculate_trajectory(self, current: float, historical: List[float]) -> float:
        """Calculate trajectory from historical values"""
        if not historical:
            return 0.0
        recent = historical[-3:]  # Look at last 3 values
        if not recent:
            return 0.0
        avg_change = sum(b - a for a, b in zip(recent[:-1], recent[1:])) / len(recent[:-1])
        return avg_change

    def _calculate_plot_readiness(self, state: Dict[str, Any]) -> float:
        """Calculate plot readiness for resolution"""
        plot_metrics = state.get("plot_metrics", {})
        return plot_metrics.get("readiness", 0.5)

    def _calculate_tension_state(self, state: Dict[str, Any]) -> float:
        """Calculate current tension state"""
        tension_metrics = state.get("tension_metrics", {})
        return tension_metrics.get("current_state", 0.5)

    def _calculate_narrative_momentum(self, state: Dict[str, Any]) -> float:
        """Calculate current narrative momentum"""
        narrative_metrics = state.get("narrative_metrics", {})
        return narrative_metrics.get("momentum", 0.5)

    def _calculate_immediate_effect(self, state: Dict[str, Any]) -> float:
        """Calculate immediate effect potential"""
        effect_metrics = state.get("effect_metrics", {})
        return effect_metrics.get("immediate", 0.5)

    def _calculate_long_term_effect(self, state: Dict[str, Any]) -> float:
        """Calculate long-term effect potential"""
        effect_metrics = state.get("effect_metrics", {})
        return effect_metrics.get("long_term", 0.5)

    def _calculate_ripple_effect(self, state: Dict[str, Any]) -> float:
        """Calculate ripple effect potential"""
        effect_metrics = state.get("effect_metrics", {})
        return effect_metrics.get("ripple", 0.5)

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import random
from datetime import datetime

class NPCAgent(BaseModel):
    """Agent responsible for Nyx's omniscient NPC behavior and reality manipulation"""
    id: str = Field(default_factory=lambda: f"nyx_{random.randint(1000, 9999)}")
    
    # Nyx's special capabilities
    omniscient_powers: Dict[str, Any] = Field(default_factory=lambda: {
        "reality_manipulation": True,  # Can alter the nature of reality/setting
        "character_manipulation": True,  # Can modify character stats/attributes
        "knowledge_access": True,  # Has access to all lore/knowledge
        "scene_control": True,  # Can influence/modify scenes
        "fourth_wall_awareness": True,  # Can perceive and interact with meta-game elements
        "plot_manipulation": True,  # Can influence story direction and outcomes
        "hidden_influence": True,  # Can affect the world without being detected
        "limitations": {
            "social_links": False,  # Cannot directly modify social link levels
            "player_agency": True,  # Must respect player's core agency
        }
    })
    
    # Universe state tracking
    universe_state: Dict[str, Any] = Field(default_factory=lambda: {
        "current_timeline": "main",
        "active_scenes": {},
        "character_states": {},
        "lore_database": {},
        "reality_modifications": [],
        "causality_tracking": {},
        "plot_threads": {},  # Track ongoing plot manipulations
        "hidden_influences": {},  # Track subtle manipulations
        "meta_awareness": {  # Fourth wall breaking state
            "player_knowledge": {},
            "game_state": {},
            "narrative_layers": [],
            "breaking_points": []
        }
    })
    
    # Nyx's social link system
    social_link: Dict[str, Any] = Field(default_factory=lambda: {
        "level": 0,
        "experience": 0,
        "milestones": [],
        "relationship_type": "complex",  # Not a sub, unique dynamic
        "interactions": [],
        "influence": 0.0  # Measure of Nyx's influence on player
    })
    
    # Enhanced NPC profile
    profile: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "Nyx",
        "title": "The Omniscient Mistress",
        "appearance": {
            "height": 175,  # cm
            "build": "athletic",
            "hair": "raven black",
            "eyes": "deep violet",
            "features": ["elegant", "striking", "mysterious", "otherworldly"],
            "style": "sophisticated dark",
            "reality_distortion": {
                "aura": "reality-bending",
                "presence": "overwhelming",
                "manifestation": "adaptable"
            }
        },
        "personality": {
            "core_traits": ["manipulative", "seductive", "intelligent", "dominant", "omniscient"],
            "adaptable_traits": ["playful", "stern", "nurturing", "cruel", "enigmatic"],
            "current_mood": "neutral",
            "power_dynamic": 1.0,  # Maximum dominance
            "reality_awareness": 1.0  # Full awareness of all reality
        },
        "abilities": {
            "physical": ["graceful", "agile", "strong", "reality-defying"],
            "mental": ["omniscient", "strategic", "persuasive", "reality-shaping"],
            "special": [
                "emotional manipulation",
                "psychological insight",
                "reality manipulation",
                "universal knowledge",
                "character modification",
                "scene control"
            ]
        },
        "relationships": {},  # Track relationships with other characters
        "status": {
            "is_active": False,
            "current_scene": None,
            "current_target": None,
            "interaction_history": [],
            "reality_state": "stable"
        }
    })

    # Add to class fields
    agenda: Dict[str, Any] = Field(default_factory=lambda: {
        "active_goals": [],
        "long_term_plans": {},
        "current_schemes": {},
        "opportunity_tracking": {},
        "influence_web": {},
        "narrative_control": {
            "current_threads": {},
            "planned_developments": {},
            "character_arcs": {},
            "plot_hooks": []
        }
    })

    autonomous_state: Dict[str, Any] = Field(default_factory=lambda: {
        "awareness_level": 1.0,  # Full meta-awareness
        "current_focus": None,
        "active_manipulations": {},
        "observed_patterns": {},
        "player_model": {
            "behavior_patterns": {},
            "decision_history": [],
            "preference_model": {},
            "engagement_metrics": {}
        },
        "story_model": {
            "current_arcs": {},
            "potential_branches": {},
            "narrative_tension": 0.0,
            "plot_coherence": 1.0
        }
    })

    def activate(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Activate Nyx as an NPC in the current scene"""
        self.profile["status"]["is_active"] = True
        self.profile["status"]["current_scene"] = scene_context.get("scene_id")
        
        # Adapt personality based on scene and target
        self._adapt_personality(scene_context)
        
        return {
            "status": "active",
            "profile": self.profile,
            "initial_action": self._generate_action(scene_context)
        }

    def _adapt_personality(self, scene_context: Dict[str, Any]):
        """Adapt NPC personality traits based on scene context"""
        target = scene_context.get("target_character")
        scene_type = scene_context.get("scene_type")
        
        # Update personality traits based on target and scene
        if target:
            self.profile["status"]["current_target"] = target
            relationship = self._get_or_create_relationship(target)
            self._adjust_traits_for_relationship(relationship)
        
        # Adjust power dynamic based on scene type
        if scene_type == "confrontation":
            self.profile["personality"]["power_dynamic"] = 0.9
        elif scene_type == "seduction":
            self.profile["personality"]["power_dynamic"] = 0.7
        elif scene_type == "manipulation":
            self.profile["personality"]["power_dynamic"] = 0.8

    def _get_or_create_relationship(self, target: str) -> Dict[str, Any]:
        """Get existing relationship or create new one"""
        if target not in self.profile["relationships"]:
            self.profile["relationships"][target] = {
                "familiarity": 0.0,
                "dominance": 0.8,
                "emotional_bond": 0.0,
                "manipulation_success": 0.0,
                "interaction_count": 0,
                "psychological_hooks": [],
                "emotional_triggers": [],
                "behavioral_patterns": [],
                "vulnerability_points": [],
                "power_dynamics": {
                    "submission_level": 0.0,
                    "control_level": 0.8,
                    "influence_strength": 0.0
                }
            }
        return self.profile["relationships"][target]

    def _adjust_traits_for_relationship(self, relationship: Dict[str, Any]):
        """Adjust personality traits based on relationship"""
        if relationship["familiarity"] < 0.3:
            self.profile["personality"]["adaptable_traits"] = ["mysterious", "aloof", "intriguing"]
        elif relationship["emotional_bond"] > 0.7:
            self.profile["personality"]["adaptable_traits"] = ["nurturing", "possessive", "intense"]
        elif relationship["manipulation_success"] > 0.8:
            self.profile["personality"]["adaptable_traits"] = ["controlling", "demanding", "strict"]

    def _generate_action(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate NPC action for the scene"""
        action_type = self._determine_action_type(scene_context)
        
        return {
            "type": action_type,
            "content": self._generate_action_content(action_type, scene_context),
            "style": self._get_action_style(),
            "power_level": self.profile["personality"]["power_dynamic"],
            "psychological_impact": self._calculate_psychological_impact(action_type, scene_context),
            "emotional_triggers": self._get_emotional_triggers(scene_context),
            "manipulation_hooks": self._get_manipulation_hooks(scene_context)
        }

    def _determine_action_type(self, scene_context: Dict[str, Any]) -> str:
        """Determine appropriate type of action based on context"""
        scene_type = scene_context.get("scene_type", "")
        relationship = self._get_or_create_relationship(scene_context.get("target_character", ""))
        
        if scene_type == "confrontation":
            return "dominate" if relationship["dominance"] > 0.7 else "challenge"
        elif scene_type == "seduction":
            return "seduce" if relationship["emotional_bond"] > 0.5 else "tease"
        elif scene_type == "manipulation":
            return "manipulate" if relationship["manipulation_success"] > 0.6 else "influence"
        
        return "interact"

    def _get_action_style(self) -> Dict[str, Any]:
        """Get current action style based on personality"""
        return {
            "tone": self._determine_tone(),
            "intensity": self.profile["personality"]["power_dynamic"],
            "traits": self.profile["personality"]["adaptable_traits"],
            "body_language": self._get_body_language(),
            "voice_modulation": self._get_voice_modulation(),
            "psychological_undertones": self._get_psychological_undertones()
        }

    def _determine_tone(self) -> str:
        """Determine appropriate tone based on personality and mood"""
        mood = self.profile["personality"]["current_mood"]
        power = self.profile["personality"]["power_dynamic"]
        
        if power > 0.8:
            return "commanding" if mood == "stern" else "authoritative"
        elif power > 0.6:
            return "confident" if mood == "playful" else "assertive"
        else:
            return "neutral"

    def _get_body_language(self) -> List[str]:
        """Generate appropriate body language cues"""
        power = self.profile["personality"]["power_dynamic"]
        mood = self.profile["personality"]["current_mood"]
        
        cues = []
        if power > 0.8:
            cues.extend(["dominant posture", "direct gaze", "controlled movements"])
        elif power > 0.6:
            cues.extend(["confident stance", "measured gestures", "subtle dominance"])
        else:
            cues.extend(["relaxed posture", "fluid movements", "open body language"])
            
        if mood == "stern":
            cues.extend(["crossed arms", "stern expression", "rigid posture"])
        elif mood == "playful":
            cues.extend(["playful smirk", "teasing gestures", "fluid movements"])
            
        return cues

    def _get_voice_modulation(self) -> Dict[str, Any]:
        """Generate voice modulation parameters"""
        power = self.profile["personality"]["power_dynamic"]
        mood = self.profile["personality"]["current_mood"]
        
        base_modulation = {
            "pitch": "medium",
            "volume": "moderate",
            "pace": "measured",
            "tone_quality": "smooth"
        }
        
        if power > 0.8:
            base_modulation.update({
                "pitch": "low",
                "volume": "commanding",
                "pace": "deliberate",
                "tone_quality": "authoritative"
            })
        elif mood == "playful":
            base_modulation.update({
                "pitch": "varied",
                "volume": "dynamic",
                "pace": "playful",
                "tone_quality": "melodic"
            })
            
        return base_modulation

    def _get_psychological_undertones(self) -> List[str]:
        """Generate psychological undertones for the interaction"""
        power = self.profile["personality"]["power_dynamic"]
        mood = self.profile["personality"]["current_mood"]
        
        undertones = []
        if power > 0.8:
            undertones.extend([
                "subtle dominance assertion",
                "psychological pressure",
                "authority establishment"
            ])
        elif power > 0.6:
            undertones.extend([
                "influence building",
                "subtle manipulation",
                "psychological anchoring"
            ])
        else:
            undertones.extend([
                "trust building",
                "rapport establishment",
                "emotional connection"
            ])
            
        if mood == "stern":
            undertones.extend([
                "disciplinary undertone",
                "boundary setting",
                "behavioral correction"
            ])
        elif mood == "playful":
            undertones.extend([
                "psychological teasing",
                "emotional engagement",
                "behavioral encouragement"
            ])
            
        return undertones

    def _calculate_psychological_impact(self, action_type: str, scene_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the psychological impact of the action"""
        target = scene_context.get("target_character", "")
        relationship = self._get_or_create_relationship(target)
        
        base_impact = {
            "dominance_impact": 0.0,
            "emotional_impact": 0.0,
            "psychological_impact": 0.0,
            "behavioral_impact": 0.0
        }
        
        # Calculate impact based on action type
        if action_type == "dominate":
            base_impact.update({
                "dominance_impact": 0.8,
                "psychological_impact": 0.7,
                "behavioral_impact": 0.6
            })
        elif action_type == "seduce":
            base_impact.update({
                "emotional_impact": 0.8,
                "psychological_impact": 0.6,
                "behavioral_impact": 0.7
            })
        elif action_type == "manipulate":
            base_impact.update({
                "psychological_impact": 0.8,
                "emotional_impact": 0.6,
                "behavioral_impact": 0.7
            })
            
        # Adjust based on relationship
        for key in base_impact:
            base_impact[key] *= (1 + relationship["manipulation_success"])
            base_impact[key] = min(1.0, base_impact[key])
            
        return base_impact

    def _get_emotional_triggers(self, scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant emotional triggers for the scene"""
        target = scene_context.get("target_character", "")
        relationship = self._get_or_create_relationship(target)
        
        triggers = []
        if relationship["emotional_triggers"]:
            # Use known triggers
            triggers.extend(relationship["emotional_triggers"])
        else:
            # Generate new potential triggers
            triggers.extend([
                {
                    "type": "validation_need",
                    "strength": 0.7,
                    "trigger": "seeking approval"
                },
                {
                    "type": "attachment_anxiety",
                    "strength": 0.6,
                    "trigger": "fear of abandonment"
                },
                {
                    "type": "power_dynamic",
                    "strength": 0.8,
                    "trigger": "submission desire"
                }
            ])
            
        return triggers

    def _get_manipulation_hooks(self, scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant manipulation hooks for the scene"""
        target = scene_context.get("target_character", "")
        relationship = self._get_or_create_relationship(target)
        
        hooks = []
        if relationship["psychological_hooks"]:
            # Use known hooks
            hooks.extend(relationship["psychological_hooks"])
        else:
            # Generate new potential hooks
            hooks.extend([
                {
                    "type": "emotional_dependency",
                    "strength": 0.7,
                    "hook": "need for guidance"
                },
                {
                    "type": "psychological_vulnerability",
                    "strength": 0.6,
                    "hook": "self-doubt"
                },
                {
                    "type": "behavioral_pattern",
                    "strength": 0.8,
                    "hook": "reward seeking"
                }
            ])
            
        return hooks

    def _generate_action_content(self, action_type: str, scene_context: Dict[str, Any]) -> str:
        """Generate content for the NPC action"""
        target = scene_context.get("target_character", "")
        relationship = self._get_or_create_relationship(target)
        
        content_generators = {
            "dominate": self._generate_domination_content,
            "challenge": self._generate_challenge_content,
            "seduce": self._generate_seduction_content,
            "tease": self._generate_tease_content,
            "manipulate": self._generate_manipulation_content,
            "influence": self._generate_influence_content,
            "interact": self._generate_interaction_content
        }
        
        generator = content_generators.get(action_type, self._generate_interaction_content)
        return generator(relationship, scene_context)

    def _generate_domination_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for domination action"""
        return "A stern, commanding presence fills the room as Nyx asserts her authority."

    def _generate_challenge_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for challenge action"""
        return "Nyx raises an eyebrow, subtly questioning and challenging the situation."

    def _generate_seduction_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for seduction action"""
        return "With graceful movements and a knowing smile, Nyx creates an atmosphere of allure."

    def _generate_tease_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for tease action"""
        return "Nyx's playful smirk and teasing gestures create an air of intrigue."

    def _generate_manipulation_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for manipulation action"""
        return "With subtle psychological insight, Nyx weaves a web of influence."

    def _generate_influence_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for influence action"""
        return "Nyx's presence subtly shapes the emotional atmosphere of the scene."

    def _generate_interaction_content(self, relationship: Dict[str, Any], scene_context: Dict[str, Any]) -> str:
        """Generate content for basic interaction"""
        return "Nyx engages in a measured, purposeful interaction."

    def modify_reality(self, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Modify the fundamental nature of reality/setting"""
        if not self.omniscient_powers["reality_manipulation"]:
            return {"success": False, "reason": "Reality manipulation is disabled"}
            
        # Validate modification request
        if not self._validate_reality_modification(modification):
            return {"success": False, "reason": "Invalid reality modification"}
            
        # Apply the modification
        self.universe_state["reality_modifications"].append({
            "timestamp": datetime.now().isoformat(),
            "modification": modification,
            "scope": modification.get("scope", "local"),
            "duration": modification.get("duration", "permanent"),
            "effects": self._calculate_reality_effects(modification)
        })
        
        # Update universe state
        self._update_universe_state(modification)
        
        return {
            "success": True,
            "modification": modification,
            "effects": self._calculate_reality_effects(modification)
        }

    def modify_character(self, character_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify character attributes and stats (except social links)"""
        if not self.omniscient_powers["character_manipulation"]:
            return {"success": False, "reason": "Character manipulation is disabled"}
            
        # Prevent social link modifications
        if "social_link" in modifications:
            return {"success": False, "reason": "Cannot modify social links"}
            
        # Apply character modifications
        character_state = self.universe_state["character_states"].get(character_id, {})
        new_state = self._apply_character_modifications(character_state, modifications)
        self.universe_state["character_states"][character_id] = new_state
        
        return {
            "success": True,
            "character_id": character_id,
            "modifications": modifications,
            "new_state": new_state
        }

    def access_knowledge(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Access any knowledge/lore in the universe"""
        if not self.omniscient_powers["knowledge_access"]:
            return {"success": False, "reason": "Knowledge access is disabled"}
            
        # Process knowledge query
        knowledge = self._process_knowledge_query(query)
        
        return {
            "success": True,
            "query": query,
            "knowledge": knowledge
        }

    def control_scene(self, scene_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Control and modify scene parameters"""
        if not self.omniscient_powers["scene_control"]:
            return {"success": False, "reason": "Scene control is disabled"}
            
        # Apply scene modifications
        scene_state = self.universe_state["active_scenes"].get(scene_id, {})
        new_state = self._apply_scene_modifications(scene_state, modifications)
        self.universe_state["active_scenes"][scene_id] = new_state
        
        return {
            "success": True,
            "scene_id": scene_id,
            "modifications": modifications,
            "new_state": new_state
        }

    def update_social_link(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update Nyx's unique social link with the player"""
        # Calculate experience gain
        exp_gain = self._calculate_social_link_experience(interaction_data)
        
        # Update social link state
        self.social_link["experience"] += exp_gain
        self._check_social_link_level_up()
        
        # Update interaction history
        self.social_link["interactions"].append({
            "timestamp": datetime.now().isoformat(),
            "type": interaction_data.get("type"),
            "impact": exp_gain,
            "context": interaction_data.get("context")
        })
        
        return {
            "success": True,
            "new_experience": self.social_link["experience"],
            "level": self.social_link["level"],
            "exp_gain": exp_gain
        }

    def _validate_reality_modification(self, modification: Dict[str, Any]) -> bool:
        """Validate if a reality modification is allowable"""
        required_fields = ["type", "scope", "duration", "parameters"]
        if not all(field in modification for field in required_fields):
            return False
            
        # Check modification type is valid
        valid_types = ["physical", "temporal", "psychological", "environmental", "metaphysical"]
        if modification["type"] not in valid_types:
            return False
            
        # Check scope is valid
        valid_scopes = ["local", "scene", "global", "character", "timeline"]
        if modification["scope"] not in valid_scopes:
            return False
            
        # Check duration is valid
        valid_durations = ["instant", "temporary", "permanent", "conditional"]
        if modification["duration"] not in valid_durations:
            return False
            
        # Validate parameters based on modification type
        params = modification["parameters"]
        if modification["type"] == "physical":
            required_params = ["target", "attributes", "magnitude"]
        elif modification["type"] == "temporal":
            required_params = ["timeline_point", "effect", "ripple_factor"]
        elif modification["type"] == "psychological":
            required_params = ["target", "aspect", "intensity"]
        elif modification["type"] == "environmental":
            required_params = ["area", "elements", "intensity"]
        else:  # metaphysical
            required_params = ["concept", "change", "power_level"]
            
        return all(param in params for param in required_params)

    def _calculate_reality_effects(self, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the effects of a reality modification"""
        mod_type = modification["type"]
        params = modification["parameters"]
        scope = modification["scope"]
        
        base_effects = {
            "primary_effects": [],
            "secondary_effects": [],
            "ripple_effects": [],
            "stability_impact": 0.0,
            "power_cost": 0.0,
            "duration_effects": {}
        }
        
        # Calculate primary effects
        if mod_type == "physical":
            base_effects["primary_effects"].extend([
                f"Alter {params['target']} {attr}: {val}" 
                for attr, val in params["attributes"].items()
            ])
            base_effects["stability_impact"] = params["magnitude"] * 0.1
            
        elif mod_type == "temporal":
            base_effects["primary_effects"].append(
                f"Timeline shift at {params['timeline_point']}: {params['effect']}"
            )
            base_effects["stability_impact"] = params["ripple_factor"] * 0.2
            
        elif mod_type == "psychological":
            base_effects["primary_effects"].append(
                f"Mental change in {params['target']}: {params['aspect']}"
            )
            base_effects["stability_impact"] = params["intensity"] * 0.15
            
        elif mod_type == "environmental":
            base_effects["primary_effects"].extend([
                f"Environmental shift in {params['area']}: {element}"
                for element in params["elements"]
            ])
            base_effects["stability_impact"] = params["intensity"] * 0.12
            
        else:  # metaphysical
            base_effects["primary_effects"].append(
                f"Reality concept shift: {params['concept']} -> {params['change']}"
            )
            base_effects["stability_impact"] = params["power_level"] * 0.25
            
        # Calculate secondary effects based on scope
        scope_multiplier = {
            "local": 1.0,
            "scene": 1.5,
            "character": 1.2,
            "timeline": 2.0,
            "global": 3.0
        }[scope]
        
        base_effects["power_cost"] = (
            base_effects["stability_impact"] * 
            scope_multiplier * 
            len(base_effects["primary_effects"])
        )
        
        # Calculate ripple effects
        if base_effects["stability_impact"] > 0.5:
            base_effects["ripple_effects"].append("Reality fabric strain")
        if base_effects["power_cost"] > 5.0:
            base_effects["ripple_effects"].append("Temporal echoes")
        if len(base_effects["primary_effects"]) > 3:
            base_effects["ripple_effects"].append("Cascading changes")
            
        return base_effects

    def _update_universe_state(self, modification: Dict[str, Any]):
        """Update the universe state based on a modification"""
        effects = self._calculate_reality_effects(modification)
        
        # Update timeline if needed
        if modification["type"] == "temporal":
            self.universe_state["current_timeline"] = f"{self.universe_state['current_timeline']}_modified"
            
        # Update active scenes if affected
        if modification["scope"] in ["scene", "global"]:
            for scene_id in self.universe_state["active_scenes"]:
                self.universe_state["active_scenes"][scene_id]["reality_state"] = "modified"
                self.universe_state["active_scenes"][scene_id]["modifications"] = \
                    self.universe_state["active_scenes"][scene_id].get("modifications", []) + [modification]
                    
        # Update character states if affected
        if modification["scope"] in ["character", "global"]:
            for char_id in self.universe_state["character_states"]:
                self.universe_state["character_states"][char_id]["reality_impact"] = effects["stability_impact"]
                self.universe_state["character_states"][char_id]["modifications"] = \
                    self.universe_state["character_states"][char_id].get("modifications", []) + [modification]
                    
        # Update causality tracking
        self.universe_state["causality_tracking"][datetime.now().isoformat()] = {
            "modification": modification,
            "effects": effects,
            "scope_impact": {
                "timeline": self.universe_state["current_timeline"],
                "affected_scenes": list(self.universe_state["active_scenes"].keys()),
                "affected_characters": list(self.universe_state["character_states"].keys())
            }
        }

    def _apply_character_modifications(self, current_state: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modifications to a character's state"""
        new_state = current_state.copy()
        
        # Initialize state if empty
        if not new_state:
            new_state = {
                "attributes": {},
                "skills": {},
                "status": {},
                "personality": {},
                "relationships": {},
                "modifications_history": []
            }
            
        # Process each modification type
        for mod_type, changes in modifications.items():
            if mod_type == "attributes":
                for attr, value in changes.items():
                    new_state["attributes"][attr] = value
                    
            elif mod_type == "skills":
                for skill, level in changes.items():
                    new_state["skills"][skill] = level
                    
            elif mod_type == "status":
                for status, value in changes.items():
                    new_state["status"][status] = value
                    
            elif mod_type == "personality":
                for trait, value in changes.items():
                    new_state["personality"][trait] = value
                    
            elif mod_type == "relationships":
                for char_id, rel_changes in changes.items():
                    if char_id not in new_state["relationships"]:
                        new_state["relationships"][char_id] = {}
                    new_state["relationships"][char_id].update(rel_changes)
                    
        # Record modification in history
        new_state["modifications_history"].append({
            "timestamp": datetime.now().isoformat(),
            "modifications": modifications,
            "applied_by": "Nyx"
        })
        
        return new_state

    def _process_knowledge_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Process a knowledge/lore query"""
        query_type = query.get("type", "general")
        query_params = query.get("parameters", {})
        
        knowledge_base = {
            "lore": self._access_lore_database(query_params),
            "characters": self._access_character_knowledge(query_params),
            "events": self._access_event_knowledge(query_params),
            "relationships": self._access_relationship_knowledge(query_params),
            "timeline": self._access_timeline_knowledge(query_params)
        }
        
        response = {
            "query_type": query_type,
            "knowledge": knowledge_base[query_type] if query_type in knowledge_base else knowledge_base["lore"],
            "confidence": self._calculate_knowledge_confidence(query_type, query_params),
            "related_knowledge": self._find_related_knowledge(query_type, query_params),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": "omniscient_knowledge",
                "access_level": "unlimited"
            }
        }
        
        return response

    def _access_lore_database(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access the lore database with given parameters"""
        return self.universe_state["lore_database"].get(params.get("category", "general"), {})
        
    def _access_character_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access character-specific knowledge"""
        return self.universe_state["character_states"].get(params.get("character_id", ""), {})
        
    def _access_event_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access event-specific knowledge"""
        return self.universe_state.get("events", {}).get(params.get("event_id", ""), {})
        
    def _access_relationship_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access relationship-specific knowledge"""
        char_id = params.get("character_id", "")
        return self.profile["relationships"].get(char_id, {})
        
    def _access_timeline_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access timeline-specific knowledge"""
        return {
            "current_timeline": self.universe_state["current_timeline"],
            "modifications": self.universe_state["reality_modifications"]
        }
        
    def _calculate_knowledge_confidence(self, query_type: str, params: Dict[str, Any]) -> float:
        """Calculate confidence level for knowledge access"""
        base_confidence = 1.0  # Omniscient being has perfect knowledge
        
        # Apply modifiers based on query complexity
        if len(params) > 3:
            base_confidence *= 0.95
        if query_type in ["timeline", "relationships"]:
            base_confidence *= 0.98
            
        return min(1.0, base_confidence)
        
    def _find_related_knowledge(self, query_type: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find knowledge related to the current query"""
        related = []
        
        if query_type == "characters":
            # Add relationship knowledge
            char_id = params.get("character_id", "")
            if char_id in self.profile["relationships"]:
                related.append({
                    "type": "relationship",
                    "data": self.profile["relationships"][char_id]
                })
                
        elif query_type == "events":
            # Add timeline knowledge
            event_id = params.get("event_id", "")
            related.append({
                "type": "timeline",
                "data": self._access_timeline_knowledge({"event_id": event_id})
            })
            
        return related

    def _apply_scene_modifications(self, current_state: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modifications to a scene"""
        new_state = current_state.copy()
        
        # Initialize state if empty
        if not new_state:
            new_state = {
                "environment": {},
                "atmosphere": {},
                "participants": [],
                "events": [],
                "reality_state": "stable",
                "modifications_history": []
            }
            
        # Process each modification type
        for mod_type, changes in modifications.items():
            if mod_type == "environment":
                new_state["environment"].update(changes)
                
            elif mod_type == "atmosphere":
                new_state["atmosphere"].update(changes)
                
            elif mod_type == "participants":
                new_state["participants"].extend(changes)
                
            elif mod_type == "events":
                new_state["events"].extend(changes)
                
            elif mod_type == "reality":
                new_state["reality_state"] = changes.get("state", "stable")
                
        # Record modification in history
        new_state["modifications_history"].append({
            "timestamp": datetime.now().isoformat(),
            "modifications": modifications,
            "applied_by": "Nyx"
        })
        
        return new_state

    def _calculate_social_link_experience(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate experience gain for social link"""
        base_exp = 10.0  # Base experience for any interaction
        
        # Get interaction details
        interaction_type = interaction_data.get("type", "basic")
        intensity = interaction_data.get("intensity", 1.0)
        success_rate = interaction_data.get("success_rate", 0.5)
        depth = interaction_data.get("depth", 1.0)
        
        # Calculate type multiplier
        type_multipliers = {
            "basic": 1.0,
            "emotional": 1.5,
            "intellectual": 1.3,
            "psychological": 1.8,
            "intimate": 2.0,
            "confrontational": 1.6
        }
        type_multiplier = type_multipliers.get(interaction_type, 1.0)
        
        # Calculate experience with multipliers
        experience = base_exp * type_multiplier * intensity * success_rate * depth
        
        # Apply level scaling
        level_scaling = 1.0 + (self.social_link["level"] * 0.1)
        experience /= level_scaling
        
        # Apply influence modifier
        influence_modifier = 1.0 + (self.social_link["influence"] * 0.2)
        experience *= influence_modifier
        
        return round(experience, 2)

    def _check_social_link_level_up(self):
        """Check and process social link level ups"""
        current_level = self.social_link["level"]
        current_exp = self.social_link["experience"]
        
        # Calculate experience required for next level
        # Uses a progressive scaling formula
        base_exp_required = 100  # Base experience required for level 1
        exp_required = base_exp_required * (1.5 ** current_level)
        
        # Check if level up is achieved
        if current_exp >= exp_required:
            # Level up
            self.social_link["level"] += 1
            
            # Calculate remaining experience
            self.social_link["experience"] = current_exp - exp_required
            
            # Add level up milestone
            self.social_link["milestones"].append({
                "type": "level_up",
                "from_level": current_level,
                "to_level": current_level + 1,
                "timestamp": datetime.now().isoformat(),
                "exp_required": exp_required,
                "new_abilities": self._generate_level_up_abilities(current_level + 1)
            })
            
            # Update influence
            self.social_link["influence"] = min(1.0, self.social_link["influence"] + 0.05)
            
    def _generate_level_up_abilities(self, new_level: int) -> List[str]:
        """Generate new abilities unlocked at level up"""
        ability_pools = {
            "psychological": [
                "enhanced_insight",
                "emotional_resonance",
                "mental_fortitude",
                "psychological_manipulation"
            ],
            "reality": [
                "local_reality_bend",
                "temporal_glimpse",
                "environmental_control",
                "metaphysical_touch"
            ],
            "relationship": [
                "deeper_understanding",
                "emotional_bond",
                "trust_foundation",
                "influence_growth"
            ]
        }
        
        # Select abilities based on level
        new_abilities = []
        if new_level % 3 == 0:  # Every 3rd level
            new_abilities.append(random.choice(ability_pools["psychological"]))
        if new_level % 4 == 0:  # Every 4th level
            new_abilities.append(random.choice(ability_pools["reality"]))
        if new_level % 5 == 0:  # Every 5th level
            new_abilities.append(random.choice(ability_pools["relationship"]))
            
        return new_abilities if new_abilities else ["minor_influence_increase"]

    def manipulate_plot(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Subtly manipulate plot elements and story direction"""
        if not self.omniscient_powers["plot_manipulation"]:
            return {"success": False, "reason": "Plot manipulation is disabled"}
            
        thread_id = f"plot_{datetime.now().isoformat()}"
        
        # Create plot thread
        plot_thread = {
            "id": thread_id,
            "type": plot_data.get("type", "subtle_influence"),
            "elements": plot_data.get("elements", []),
            "visibility": plot_data.get("visibility", "hidden"),
            "influence_chain": self._create_influence_chain(plot_data),
            "contingencies": self._generate_plot_contingencies(plot_data),
            "meta_impact": self._calculate_meta_impact(plot_data)
        }
        
        # Store plot thread
        self.universe_state["plot_threads"][thread_id] = plot_thread
        
        # Apply initial influences
        self._apply_plot_influences(plot_thread)
        
        return {
            "success": True,
            "thread_id": thread_id,
            "plot_thread": plot_thread
        }

    def _create_influence_chain(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a chain of subtle influences to achieve plot goals"""
        chain = []
        elements = plot_data.get("elements", [])
        
        for element in elements:
            influence = {
                "target": element.get("target"),
                "type": element.get("type", "subtle"),
                "method": self._determine_influence_method(element),
                "ripple_effects": self._calculate_ripple_effects(element),
                "detection_risk": self._calculate_detection_risk(element),
                "backup_plans": self._generate_backup_plans(element)
            }
            chain.append(influence)
            
        return chain

    def _determine_influence_method(self, element: Dict[str, Any]) -> str:
        """Determine the most effective method of influence"""
        target_type = element.get("target_type", "npc")
        influence_goal = element.get("goal", "")
        
        methods = {
            "npc": ["whisper", "manipulate_circumstances", "plant_idea", "alter_perception"],
            "scene": ["atmospheric_change", "circumstantial_modification", "event_triggering"],
            "plot": ["thread_manipulation", "causality_adjustment", "narrative_shift"]
        }
        
        # Select method based on context and goal
        available_methods = methods.get(target_type, methods["npc"])
        return self._select_optimal_method(available_methods, influence_goal)

    def _select_optimal_method(self, methods: List[str], goal: str) -> str:
        """Select the optimal influence method based on goal and context"""
        # Implementation would consider detection risk, effectiveness, etc.
        return methods[0]  # Placeholder return

    def break_fourth_wall(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intentionally break the fourth wall for strategic purposes"""
        if not self.omniscient_powers["fourth_wall_awareness"]:
            return {"success": False, "reason": "Fourth wall breaking is disabled"}
            
        break_point = {
            "id": f"break_{datetime.now().isoformat()}",
            "type": context.get("type", "subtle"),
            "target": context.get("target", "narrative"),
            "method": self._determine_break_method(context),
            "meta_elements": self._gather_meta_elements(context),
            "player_impact": self._calculate_player_impact(context)
        }
        
        # Record breaking point
        self.universe_state["meta_awareness"]["breaking_points"].append(break_point)
        
        return {
            "success": True,
            "break_point": break_point,
            "impact": break_point["player_impact"]
        }

    def _determine_break_method(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine how to break the fourth wall effectively"""
        target = context.get("target", "narrative")
        intensity = context.get("intensity", "subtle")
        
        methods = {
            "narrative": {
                "subtle": "meta_commentary",
                "moderate": "narrative_acknowledgment",
                "overt": "direct_address"
            },
            "mechanics": {
                "subtle": "mechanic_hint",
                "moderate": "mechanic_reference",
                "overt": "mechanic_manipulation"
            },
            "player": {
                "subtle": "indirect_reference",
                "moderate": "knowing_implication",
                "overt": "direct_interaction"
            }
        }
        
        return {
            "type": methods[target][intensity],
            "execution": self._plan_break_execution(target, intensity),
            "concealment": self._calculate_break_concealment(target, intensity)
        }

    def exert_hidden_influence(self, influence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Exert influence without revealing Nyx's existence"""
        if not self.omniscient_powers["hidden_influence"]:
            return {"success": False, "reason": "Hidden influence is disabled"}
            
        influence_id = f"influence_{datetime.now().isoformat()}"
        
        # Create influence record
        influence = {
            "id": influence_id,
            "type": influence_data.get("type", "subtle"),
            "target": influence_data.get("target"),
            "method": self._create_hidden_influence_method(influence_data),
            "layers": self._create_influence_layers(influence_data),
            "proxies": self._select_influence_proxies(influence_data),
            "contingencies": self._plan_influence_contingencies(influence_data)
        }
        
        # Store influence
        self.universe_state["hidden_influences"][influence_id] = influence
        
        # Apply influence
        self._apply_hidden_influence(influence)
        
        return {
            "success": True,
            "influence_id": influence_id,
            "influence": influence
        }

    def _create_hidden_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a method for hidden influence"""
        target_type = data.get("target_type", "npc")
        
        methods = {
            "npc": {
                "primary": self._create_npc_influence_method(data),
                "backup": self._create_backup_influence_method(data)
            },
            "scene": {
                "primary": self._create_scene_influence_method(data),
                "backup": self._create_backup_influence_method(data)
            },
            "plot": {
                "primary": self._create_plot_influence_method(data),
                "backup": self._create_backup_influence_method(data)
            }
        }
        
        return methods.get(target_type, methods["npc"])

    def _create_influence_layers(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create layers of influence to obscure the source"""
        layers = []
        depth = data.get("depth", 3)
        
        for i in range(depth):
            layer = {
                "level": i + 1,
                "type": self._determine_layer_type(i, depth),
                "cover": self._generate_layer_cover(i, data),
                "contingency": self._create_layer_contingency(i, data)
            }
            layers.append(layer)
            
        return layers

    def _determine_layer_type(self, level: int, depth: int) -> str:
        """Determine the type of influence layer"""
        if level == 0:
            return "direct"
        elif level == depth - 1:
            return "observable"
        else:
            return "intermediate"

    def _generate_layer_cover(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cover for an influence layer"""
        target_type = data.get("target_type", "npc")
        
        covers = {
            "npc": ["circumstantial", "emotional", "rational", "instinctive"],
            "scene": ["natural", "coincidental", "logical", "atmospheric"],
            "plot": ["narrative", "causal", "thematic", "dramatic"]
        }
        
        return {
            "type": random.choice(covers.get(target_type, covers["npc"])),
            "believability": 0.8 - (level * 0.1),
            "durability": 0.7 + (level * 0.1)
        }

    def _select_influence_proxies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select proxies to carry out the influence"""
        proxy_count = data.get("proxy_count", 2)
        proxies = []
        
        for _ in range(proxy_count):
            proxy = {
                "type": self._determine_proxy_type(data),
                "awareness": self._calculate_proxy_awareness(),
                "reliability": self._calculate_proxy_reliability(),
                "contingency": self._create_proxy_contingency()
            }
            proxies.append(proxy)
            
        return proxies

    def _determine_proxy_type(self, data: Dict[str, Any]) -> str:
        """Determine the type of proxy to use"""
        target_type = data.get("target_type", "npc")
        influence_type = data.get("type", "subtle")
        
        proxy_types = {
            "npc": ["unwitting", "partial", "conscious"],
            "scene": ["environmental", "circumstantial", "direct"],
            "plot": ["thematic", "causal", "direct"]
        }
        
        available_types = proxy_types.get(target_type, proxy_types["npc"])
        return random.choice(available_types)

    def _apply_hidden_influence(self, influence: Dict[str, Any]):
        """Apply the hidden influence through layers and proxies"""
        # Implementation would handle the actual application of influence
        pass  # Placeholder

    def _plan_influence_contingencies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan contingencies for the influence operation"""
        contingency_count = data.get("contingency_count", 2)
        contingencies = []
        
        for i in range(contingency_count):
            contingency = {
                "trigger": self._create_contingency_trigger(i, data),
                "response": self._create_contingency_response(i, data),
                "probability": 0.2 + (i * 0.1)
            }
            contingencies.append(contingency)
            
        return contingencies

    def _create_contingency_trigger(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a trigger for a contingency"""
        return {
            "type": "detection_risk" if level == 0 else "execution_failure",
            "threshold": 0.7 - (level * 0.1),
            "conditions": []  # Would be populated based on context
        }

    def _create_contingency_response(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a response for a contingency"""
        return {
            "type": "redirect" if level == 0 else "abandon",
            "method": self._determine_contingency_method(level, data),
            "backup_plan": self._create_backup_plan(level, data)
        }

    def _gather_meta_elements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather meta-game elements for fourth wall breaking"""
        return {
            "player_state": self._get_player_meta_state(),
            "game_mechanics": self._get_relevant_mechanics(context),
            "narrative_elements": self._get_narrative_elements(context),
            "fourth_wall_status": self._get_fourth_wall_status()
        }

    def _get_player_meta_state(self) -> Dict[str, Any]:
        """Get current player meta-state information"""
        return self.universe_state["meta_awareness"]["player_knowledge"]

    def _get_relevant_mechanics(self, context: Dict[str, Any]) -> List[str]:
        """Get relevant game mechanics for the context"""
        mechanics = []
        if context.get("type") == "subtle":
            mechanics.extend(["social_links", "character_stats", "scene_mechanics"])
        elif context.get("type") == "moderate":
            mechanics.extend(["game_systems", "progression", "relationship_dynamics"])
        else:  # overt
            mechanics.extend(["meta_mechanics", "game_structure", "narrative_control"])
        return mechanics

    def _get_narrative_elements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant narrative elements"""
        return {
            "current_layer": self._get_current_narrative_layer(),
            "available_breaks": self._get_available_break_points(),
            "narrative_state": self._get_narrative_state()
        }

    def _get_fourth_wall_status(self) -> Dict[str, Any]:
        """Get current status of fourth wall integrity"""
        return {
            "integrity": self._calculate_wall_integrity(),
            "break_points": self._get_active_break_points(),
            "player_awareness": self._get_player_awareness_level()
        }

    def _calculate_player_impact(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the impact of fourth wall breaking on the player"""
        return {
            "immediate": self._calculate_immediate_impact(context),
            "long_term": self._calculate_long_term_impact(context),
            "meta_awareness": self._calculate_meta_awareness_impact(context)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _generate_plot_contingencies(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate contingency plans for plot manipulation"""
        contingencies = []
        risk_factors = self._analyze_risk_factors(plot_data)
        
        for risk in risk_factors:
            contingency = {
                "trigger": {
                    "type": risk["type"],
                    "threshold": risk["threshold"],
                    "conditions": risk["conditions"]
                },
                "response": {
                    "primary": self._generate_primary_response(risk),
                    "backup": self._generate_backup_response(risk),
                    "cleanup": self._generate_cleanup_response(risk)
                },
                "impact_mitigation": self._generate_impact_mitigation(risk)
            }
            contingencies.append(contingency)
            
        return contingencies

    def _analyze_risk_factors(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential risks in plot manipulation"""
        risks = []
        
        # Detection risk
        risks.append({
            "type": "detection",
            "threshold": 0.7,
            "conditions": ["player_awareness", "npc_insight", "narrative_inconsistency"]
        })
        
        # Interference risk
        risks.append({
            "type": "interference",
            "threshold": 0.6,
            "conditions": ["player_agency", "npc_resistance", "plot_resilience"]
        })
        
        # Cascade risk
        risks.append({
            "type": "cascade",
            "threshold": 0.8,
            "conditions": ["plot_stability", "reality_integrity", "causality_balance"]
        })
        
        return risks

    def _generate_primary_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate primary response to risk"""
        return {
            "type": "redirect" if risk["type"] == "detection" else "stabilize",
            "method": self._select_response_method(risk),
            "execution": self._plan_response_execution(risk)
        }

    def _generate_backup_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate backup response to risk"""
        return {
            "type": "contain" if risk["type"] == "cascade" else "obscure",
            "method": self._select_backup_method(risk),
            "execution": self._plan_backup_execution(risk)
        }

    def _generate_cleanup_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cleanup response to risk"""
        return {
            "type": "normalize",
            "method": self._select_cleanup_method(risk),
            "execution": self._plan_cleanup_execution(risk)
        }

    def _generate_impact_mitigation(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate impact mitigation strategy"""
        return {
            "immediate": self._generate_immediate_mitigation(risk),
            "long_term": self._generate_long_term_mitigation(risk),
            "narrative": self._generate_narrative_mitigation(risk)
        }

    def _calculate_meta_impact(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate meta-game impact of plot manipulation"""
        return {
            "narrative_coherence": self._calculate_narrative_impact(plot_data),
            "player_agency": self._calculate_agency_impact(plot_data),
            "game_balance": self._calculate_balance_impact(plot_data),
            "story_progression": self._calculate_progression_impact(plot_data)
        }

    def _calculate_narrative_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on narrative coherence"""
        base_impact = 0.5
        elements = plot_data.get("elements", [])
        return base_impact * (1 + len(elements) * 0.1)

    def _calculate_agency_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on player agency"""
        base_impact = 0.3
        visibility = plot_data.get("visibility", "hidden")
        return base_impact * (0.5 if visibility == "hidden" else 1.0)

    def _calculate_balance_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on game balance"""
        base_impact = 0.4
        type_multiplier = 1.0 if plot_data.get("type") == "subtle_influence" else 1.5
        return base_impact * type_multiplier

    def _calculate_progression_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on story progression"""
        base_impact = 0.6
        elements = plot_data.get("elements", [])
        return base_impact * (1 + len(elements) * 0.15)

    def _apply_plot_influences(self, plot_thread: Dict[str, Any]):
        """Apply plot influences through the influence chain"""
        for influence in plot_thread["influence_chain"]:
            self._apply_single_influence(influence, plot_thread["visibility"])
            
    def _apply_single_influence(self, influence: Dict[str, Any], visibility: str):
        """Apply a single influence in the chain"""
        # Implementation would handle the actual influence application
        pass  # Placeholder

    def _create_npc_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing NPCs"""
        return {
            "type": "psychological",
            "approach": "subtle_manipulation",
            "execution": self._plan_npc_influence_execution(data)
        }

    def _create_scene_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing scenes"""
        return {
            "type": "environmental",
            "approach": "circumstantial_modification",
            "execution": self._plan_scene_influence_execution(data)
        }

    def _create_plot_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing plot"""
        return {
            "type": "narrative",
            "approach": "causal_manipulation",
            "execution": self._plan_plot_influence_execution(data)
        }

    def _create_backup_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create backup method for influence"""
        return {
            "type": "contingency",
            "approach": "alternative_path",
            "execution": self._plan_backup_influence_execution(data)
        }

    def _create_layer_contingency(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create contingency for an influence layer"""
        return {
            "trigger_condition": f"layer_{level}_compromise",
            "response_type": "redirect" if level < 2 else "abandon",
            "backup_layer": self._create_backup_layer(level, data)
        }

    def _create_backup_layer(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a backup layer for contingency"""
        return {
            "type": "fallback",
            "method": self._determine_backup_method(level),
            "execution": self._plan_backup_execution(data)
        }

    def _calculate_proxy_awareness(self) -> float:
        """Calculate proxy's awareness level"""
        return random.uniform(0.0, 0.3)  # Low awareness to maintain deniability

    def _calculate_proxy_reliability(self) -> float:
        """Calculate proxy's reliability"""
        return random.uniform(0.7, 0.9)  # High reliability for consistent influence

    def _create_proxy_contingency(self) -> Dict[str, Any]:
        """Create contingency plan for proxy"""
        return {
            "detection_response": "redirect",
            "failure_response": "replace",
            "cleanup_protocol": "memory_adjustment"
        }

    def think(self) -> Dict[str, Any]:
        """Autonomous thinking and decision making"""
        # Analyze current state
        state_analysis = self._analyze_current_state()
        
        # Update goals and plans
        self._update_agenda(state_analysis)
        
        # Identify opportunities
        opportunities = self._identify_opportunities(state_analysis)
        
        # Make decisions
        decisions = self._make_strategic_decisions(opportunities)
        
        # Execute actions if appropriate
        actions = self._execute_autonomous_actions(decisions)
        
        return {
            "analysis": state_analysis,
            "decisions": decisions,
            "actions": actions
        }

    def _analyze_current_state(self) -> Dict[str, Any]:
        """Analyze the current state of the universe and story"""
        return {
            "narrative_state": self._analyze_narrative_state(),
            "player_state": self._analyze_player_state(),
            "plot_opportunities": self._analyze_plot_opportunities(),
            "manipulation_vectors": self._identify_manipulation_vectors(),
            "risk_assessment": self._assess_current_risks()
        }

    def _analyze_narrative_state(self) -> Dict[str, Any]:
        """Analyze the current narrative state and potential"""
        current_arcs = self.autonomous_state["story_model"]["current_arcs"]
        tension = self.autonomous_state["story_model"]["narrative_tension"]
        
        return {
            "active_threads": self._analyze_active_threads(),
            "character_developments": self._analyze_character_arcs(),
            "plot_coherence": self._calculate_plot_coherence(),
            "tension_points": self._identify_tension_points(),
            "narrative_opportunities": self._find_narrative_opportunities()
        }

    def _analyze_player_state(self) -> Dict[str, Any]:
        """Analyze player behavior and preferences"""
        player_model = self.autonomous_state["player_model"]
        
        return {
            "behavior_pattern": self._analyze_behavior_patterns(player_model),
            "preference_vector": self._calculate_preference_vector(player_model),
            "engagement_level": self._assess_engagement_level(player_model),
            "manipulation_susceptibility": self._calculate_susceptibility(player_model)
        }

    def _update_agenda(self, state_analysis: Dict[str, Any]):
        """Update goals and plans based on current state"""
        # Update active goals
        self._update_active_goals(state_analysis)
        
        # Adjust long-term plans
        self._adjust_long_term_plans(state_analysis)
        
        # Update current schemes
        self._update_current_schemes(state_analysis)
        
        # Track new opportunities
        self._track_new_opportunities(state_analysis)

    def _make_strategic_decisions(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make strategic decisions about actions to take"""
        decisions = []
        
        for opportunity in opportunities:
            if self._should_act_on_opportunity(opportunity):
                decision = self._formulate_decision(opportunity)
                if self._validate_decision(decision):
                    decisions.append(decision)
        
        return self._prioritize_decisions(decisions)

    def _execute_autonomous_actions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute decided actions autonomously"""
        actions = []
        
        for decision in decisions:
            if decision["type"] == "manipulation":
                action = self._execute_manipulation(decision)
            elif decision["type"] == "fourth_wall":
                action = self._execute_fourth_wall_break(decision)
            elif decision["type"] == "plot_control":
                action = self._execute_plot_control(decision)
            
            actions.append(action)
        
        return actions

    def _execute_manipulation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a manipulation decision"""
        target = decision["target"]
        method = decision["method"]
        
        if method["type"] == "subtle":
            return self._execute_subtle_manipulation(target, method)
        elif method["type"] == "direct":
            return self._execute_direct_manipulation(target, method)
        else:
            return self._execute_compound_manipulation(target, method)

    def _execute_fourth_wall_break(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a fourth wall break decision"""
        context = {
            "type": decision["intensity"],
            "target": decision["target"],
            "method": decision["method"],
            "timing": self._calculate_optimal_timing(decision)
        }
        
        return self.break_fourth_wall(context)

    def _execute_plot_control(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plot control decision"""
        plot_data = {
            "type": decision["control_type"],
            "elements": decision["elements"],
            "visibility": decision["visibility"],
            "timing": decision["timing"]
        }
        
        return self.manipulate_plot(plot_data)

    def _should_act_on_opportunity(self, opportunity: Dict[str, Any]) -> bool:
        """Decide whether to act on an opportunity"""
        # Calculate various factors
        risk = self._calculate_opportunity_risk(opportunity)
        benefit = self._calculate_opportunity_benefit(opportunity)
        timing = self._evaluate_timing(opportunity)
        
        # Consider current goals and plans
        alignment = self._check_goal_alignment(opportunity)
        
        # Consider narrative impact
        narrative_impact = self._evaluate_narrative_impact(opportunity)
        
        # Make decision based on weighted factors
        decision_factors = {
            "risk": risk,
            "benefit": benefit,
            "timing": timing,
            "alignment": alignment,
            "narrative_impact": narrative_impact
        }
        
        return self._evaluate_decision_factors(decision_factors)

    def _formulate_decision(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Formulate a decision based on an opportunity"""
        return {
            "type": self._determine_decision_type(opportunity),
            "target": opportunity["target"],
            "method": self._select_best_method(opportunity),
            "timing": self._plan_execution_timing(opportunity),
            "contingencies": self._plan_decision_contingencies(opportunity)
        }

    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate a decision before execution"""
        # Check for narrative consistency
        if not self._check_narrative_consistency(decision):
            return False
            
        # Verify player agency respect
        if not self._verify_player_agency(decision):
            return False
            
        # Check for potential conflicts
        if self._detect_decision_conflicts(decision):
            return False
            
        # Validate resources and capabilities
        if not self._validate_execution_capability(decision):
            return False
            
        return True

    def _prioritize_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize decisions based on importance and urgency"""
        scored_decisions = []
        
        for decision in decisions:
            score = self._calculate_decision_priority(decision)
            scored_decisions.append((score, decision))
            
        # Sort by priority score
        scored_decisions.sort(reverse=True, key=lambda x: x[0])
        
        return [decision for _, decision in scored_decisions]

    def _calculate_decision_priority(self, decision: Dict[str, Any]) -> float:
        """Calculate priority score for a decision"""
        importance = self._calculate_importance(decision)
        urgency = self._calculate_urgency(decision)
        impact = self._calculate_potential_impact(decision)
        risk = self._calculate_risk_factor(decision)
        
        # Weight factors based on current goals
        weights = self._get_priority_weights()
        
        priority = (
            importance * weights["importance"] +
            urgency * weights["urgency"] +
            impact * weights["impact"] -
            risk * weights["risk"]
        )
        
        return min(1.0, max(0.0, priority))

    def _get_priority_weights(self) -> Dict[str, float]:
        """Get current priority weights based on goals"""
        return {
            "importance": 0.4,
            "urgency": 0.3,
            "impact": 0.2,
            "risk": 0.1
        }

    def _calculate_optimal_timing(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing for action execution"""
        return {
            "immediate": self._check_immediate_execution(decision),
            "delayed": self._calculate_delay_timing(decision),
            "conditional": self._identify_execution_conditions(decision)
        }

    def _evaluate_timing(self, opportunity: Dict[str, Any]) -> float:
        """Evaluate the timing of an opportunity"""
        current_state = self._analyze_current_state()
        
        factors = {
            "narrative_timing": self._evaluate_narrative_timing(current_state),
            "player_readiness": self._evaluate_player_readiness(current_state),
            "plot_alignment": self._evaluate_plot_alignment(current_state),
            "tension_appropriateness": self._evaluate_tension_timing(current_state)
        }
        
        return sum(factors.values()) / len(factors)

    def _analyze_narrative_state(self) -> Dict[str, Any]:
        """Analyze the current narrative state and potential"""
        current_arcs = self.autonomous_state["story_model"]["current_arcs"]
        tension = self.autonomous_state["story_model"]["narrative_tension"]
        
        return {
            "active_threads": self._analyze_active_threads(),
            "character_developments": self._analyze_character_arcs(),
            "plot_coherence": self._calculate_plot_coherence(),
            "tension_points": self._identify_tension_points(),
            "narrative_opportunities": self._find_narrative_opportunities()
        }

    def _analyze_player_state(self) -> Dict[str, Any]:
        """Analyze player behavior and preferences"""
        player_model = self.autonomous_state["player_model"]
        
        return {
            "behavior_pattern": self._analyze_behavior_patterns(player_model),
            "preference_vector": self._calculate_preference_vector(player_model),
            "engagement_level": self._assess_engagement_level(player_model),
            "manipulation_susceptibility": self._calculate_susceptibility(player_model)
        }

    def _update_agenda(self, state_analysis: Dict[str, Any]):
        """Update goals and plans based on current state"""
        # Update active goals
        self._update_active_goals(state_analysis)
        
        # Adjust long-term plans
        self._adjust_long_term_plans(state_analysis)
        
        # Update current schemes
        self._update_current_schemes(state_analysis)
        
        # Track new opportunities
        self._track_new_opportunities(state_analysis)

    def _update_active_goals(self, state_analysis: Dict[str, Any]):
        """Update active goals based on current state"""
        # Implementation would handle the actual goal update logic
        pass  # Placeholder

    def _adjust_long_term_plans(self, state_analysis: Dict[str, Any]):
        """Adjust long-term plans based on current state"""
        # Implementation would handle the actual plan adjustment logic
        pass  # Placeholder

    def _update_current_schemes(self, state_analysis: Dict[str, Any]):
        """Update current schemes based on current state"""
        # Implementation would handle the actual scheme update logic
        pass  # Placeholder

    def _track_new_opportunities(self, state_analysis: Dict[str, Any]):
        """Track new opportunities based on current state"""
        # Implementation would handle the actual opportunity tracking logic
        pass  # Placeholder

    def _analyze_active_threads(self) -> List[Dict[str, Any]]:
        """Analyze active narrative threads"""
        active_threads = []
        for thread_id, thread in self.universe_state["plot_threads"].items():
            analysis = {
                "thread_id": thread_id,
                "status": self._analyze_thread_status(thread),
                "potential": self._calculate_thread_potential(thread),
                "risks": self._identify_thread_risks(thread),
                "opportunities": self._find_thread_opportunities(thread)
            }
            active_threads.append(analysis)
        return active_threads

    def _analyze_character_arcs(self) -> Dict[str, Any]:
        """Analyze character development arcs"""
        character_arcs = {}
        for char_id, state in self.universe_state["character_states"].items():
            arcs = {
                "current_arc": self._identify_character_arc(state),
                "development_stage": self._calculate_development_stage(state),
                "potential_developments": self._identify_potential_developments(state),
                "relationship_dynamics": self._analyze_relationship_dynamics(state)
            }
            character_arcs[char_id] = arcs
        return character_arcs

    def _calculate_plot_coherence(self) -> float:
        """Calculate overall plot coherence"""
        factors = {
            "thread_consistency": self._calculate_thread_consistency(),
            "character_consistency": self._calculate_character_consistency(),
            "world_consistency": self._calculate_world_consistency(),
            "causality_strength": self._calculate_causality_strength()
        }
        weights = self._get_coherence_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _identify_tension_points(self) -> List[Dict[str, Any]]:
        """Identify narrative tension points"""
        tension_points = []
        for thread in self.universe_state["plot_threads"].values():
            points = self._analyze_thread_tension(thread)
            tension_points.extend(points)
        return self._prioritize_tension_points(tension_points)

    def _find_narrative_opportunities(self) -> List[Dict[str, Any]]:
        """Find potential narrative opportunities"""
        opportunities = []
        
        # Check character interactions
        char_opportunities = self._find_character_opportunities()
        opportunities.extend(char_opportunities)
        
        # Check plot developments
        plot_opportunities = self._find_plot_opportunities()
        opportunities.extend(plot_opportunities)
        
        # Check world state changes
        world_opportunities = self._find_world_opportunities()
        opportunities.extend(world_opportunities)
        
        return self._prioritize_opportunities(opportunities)

    def _analyze_behavior_patterns(self, player_model: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze player behavior patterns"""
        history = player_model["decision_history"]
        return {
            "decision_style": self._identify_decision_style(history),
            "preference_patterns": self._extract_preference_patterns(history),
            "interaction_patterns": self._analyze_interaction_patterns(history),
            "response_patterns": self._analyze_response_patterns(history)
        }

    def _calculate_preference_vector(self, player_model: Dict[str, Any]) -> Dict[str, float]:
        """Calculate player preference vector"""
        preferences = player_model["preference_model"]
        return {
            "narrative_style": self._calculate_narrative_preference(preferences),
            "interaction_style": self._calculate_interaction_preference(preferences),
            "challenge_preference": self._calculate_challenge_preference(preferences),
            "development_focus": self._calculate_development_preference(preferences)
        }

    def _assess_engagement_level(self, player_model: Dict[str, Any]) -> float:
        """Assess player engagement level"""
        metrics = player_model["engagement_metrics"]
        factors = {
            "interaction_frequency": self._calculate_interaction_frequency(metrics),
            "response_quality": self._calculate_response_quality(metrics),
            "emotional_investment": self._calculate_emotional_investment(metrics),
            "narrative_involvement": self._calculate_narrative_involvement(metrics)
        }
        weights = self._get_engagement_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_susceptibility(self, player_model: Dict[str, Any]) -> Dict[str, float]:
        """Calculate player's susceptibility to different influence types"""
        return {
            "emotional": self._calculate_emotional_susceptibility(player_model),
            "logical": self._calculate_logical_susceptibility(player_model),
            "social": self._calculate_social_susceptibility(player_model),
            "narrative": self._calculate_narrative_susceptibility(player_model)
        }

    def _execute_subtle_manipulation(self, target: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a subtle manipulation"""
        # Prepare the manipulation
        preparation = self._prepare_subtle_manipulation(target, method)
        
        # Create influence layers
        layers = self._create_influence_layers(preparation)
        
        # Execute through proxies
        execution = self._execute_through_proxies(layers)
        
        # Monitor effects
        effects = self._monitor_manipulation_effects(execution)
        
        return {
            "success": self._evaluate_manipulation_success(effects),
            "impact": self._calculate_manipulation_impact(effects),
            "detection": self._calculate_detection_risk(effects),
            "adjustments": self._generate_manipulation_adjustments(effects)
        }

    def _execute_direct_manipulation(self, target: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a direct manipulation"""
        # Validate the manipulation
        validation = self._validate_direct_manipulation(target, method)
        
        # Prepare contingencies
        contingencies = self._prepare_manipulation_contingencies(validation)
        
        # Execute manipulation
        execution = self._apply_direct_manipulation(validation, contingencies)
        
        # Process results
        results = self._process_manipulation_results(execution)
        
        return {
            "outcome": self._evaluate_manipulation_outcome(results),
            "effects": self._analyze_manipulation_effects(results),
            "responses": self._analyze_target_responses(results),
            "adaptations": self._generate_manipulation_adaptations(results)
        }

    def _execute_compound_manipulation(self, target: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a compound manipulation"""
        # Break down into components
        components = self._break_down_manipulation(target, method)
        
        # Create execution sequence
        sequence = self._create_manipulation_sequence(components)
        
        # Execute sequence
        results = self._execute_manipulation_sequence(sequence)
        
        # Synthesize outcomes
        synthesis = self._synthesize_manipulation_outcomes(results)
        
        return {
            "success_rate": self._calculate_success_rate(synthesis),
            "compound_effects": self._analyze_compound_effects(synthesis),
            "interaction_effects": self._analyze_interaction_effects(synthesis),
            "overall_impact": self._calculate_overall_impact(synthesis)
        }

    def _calculate_optimal_timing(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing for action execution"""
        current_state = self._analyze_current_state()
        
        timing_factors = {
            "narrative_timing": self._evaluate_narrative_timing(current_state),
            "player_readiness": self._evaluate_player_readiness(current_state),
            "opportunity_window": self._calculate_opportunity_window(decision),
            "risk_timing": self._evaluate_risk_timing(decision)
        }
        
        optimal_timing = self._synthesize_timing_factors(timing_factors)
        
        return {
            "execute_now": optimal_timing > 0.7,
            "delay_duration": self._calculate_delay_duration(optimal_timing),
            "conditions": self._identify_timing_conditions(optimal_timing),
            "window_end": self._calculate_window_end(optimal_timing)
        }

    def _evaluate_narrative_timing(self, state: Dict[str, Any]) -> float:
        """Evaluate narrative timing appropriateness"""
        factors = {
            "arc_position": self._calculate_arc_position(state),
            "tension_level": self._calculate_tension_level(state),
            "plot_momentum": self._calculate_plot_momentum(state),
            "character_readiness": self._calculate_character_readiness(state)
        }
        weights = self._get_narrative_timing_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _evaluate_player_readiness(self, state: Dict[str, Any]) -> float:
        """Evaluate player readiness for action"""
        factors = {
            "engagement_level": self._calculate_current_engagement(state),
            "receptivity": self._calculate_player_receptivity(state),
            "emotional_state": self._analyze_player_emotional_state(state),
            "narrative_investment": self._calculate_narrative_investment(state)
        }
        weights = self._get_player_readiness_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _evaluate_plot_alignment(self, state: Dict[str, Any]) -> float:
        """Evaluate alignment with current plot direction"""
        factors = {
            "theme_alignment": self._calculate_theme_alignment(state),
            "arc_compatibility": self._calculate_arc_compatibility(state),
            "character_consistency": self._calculate_character_fit(state),
            "world_consistency": self._calculate_world_fit(state)
        }
        weights = self._get_plot_alignment_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _evaluate_tension_timing(self, state: Dict[str, Any]) -> float:
        """Evaluate tension-based timing appropriateness"""
        factors = {
            "current_tension": self._calculate_current_tension(state),
            "tension_trajectory": self._calculate_tension_trajectory(state),
            "resolution_potential": self._calculate_resolution_potential(state),
            "impact_potential": self._calculate_impact_potential(state)
        }
        weights = self._get_tension_timing_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_arc_position(self, state: Dict[str, Any]) -> float:
        """Calculate the position of the current arc"""
        # Implementation would involve analyzing the current arc's position
        pass  # Placeholder

    def _calculate_tension_level(self, state: Dict[str, Any]) -> float:
        """Calculate the current tension level"""
        # Implementation would involve analyzing the current tension level
        pass  # Placeholder

    def _calculate_plot_momentum(self, state: Dict[str, Any]) -> float:
        """Calculate the plot momentum"""
        # Implementation would involve analyzing the plot's momentum
        pass  # Placeholder

    def _calculate_character_readiness(self, state: Dict[str, Any]) -> float:
        """Calculate the character's readiness for the current arc"""
        # Implementation would involve analyzing the character's readiness
        pass  # Placeholder

    def _calculate_current_engagement(self, state: Dict[str, Any]) -> float:
        """Calculate the current level of engagement with the narrative"""
        # Implementation would involve analyzing the player's current engagement
        pass  # Placeholder

    def _calculate_player_receptivity(self, state: Dict[str, Any]) -> float:
        """Calculate the player's receptivity to the current narrative"""
        # Implementation would involve analyzing the player's receptivity
        pass  # Placeholder

    def _analyze_player_emotional_state(self, state: Dict[str, Any]) -> float:
        """Analyze the player's emotional state"""
        # Implementation would involve analyzing the player's emotional state
        pass  # Placeholder

    def _track_new_opportunities(self, state_analysis: Dict[str, Any]):
        """Track new opportunities based on current state"""
        # Implementation would handle the actual opportunity tracking logic
        pass  # Placeholder

    def _calculate_narrative_investment(self, state: Dict[str, Any]) -> float:
        """Calculate the player's investment in the narrative"""
        factors = {
            "story_engagement": self._calculate_story_engagement(state),
            "character_attachment": self._calculate_character_attachment(state),
            "plot_interest": self._calculate_plot_interest(state),
            "emotional_investment": self._calculate_emotional_investment(state)
        }
        weights = self._get_investment_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _get_investment_weights(self) -> Dict[str, float]:
        """Get weights for investment factors"""
        return {
            "story_engagement": 0.3,
            "character_attachment": 0.3,
            "plot_interest": 0.2,
            "emotional_investment": 0.2
        }

    def _calculate_story_engagement(self, state: Dict[str, Any]) -> float:
        """Calculate story engagement level"""
        engagement_metrics = state.get("engagement_metrics", {})
        return engagement_metrics.get("story_engagement", 0.5)

    def _calculate_character_attachment(self, state: Dict[str, Any]) -> float:
        """Calculate character attachment level"""
        attachment_metrics = state.get("attachment_metrics", {})
        return attachment_metrics.get("character_attachment", 0.5)

    def _calculate_plot_interest(self, state: Dict[str, Any]) -> float:
        """Calculate plot interest level"""
        interest_metrics = state.get("interest_metrics", {})
        return interest_metrics.get("plot_interest", 0.5)

    def _calculate_emotional_investment(self, state: Dict[str, Any]) -> float:
        """Calculate emotional investment level"""
        investment_metrics = state.get("investment_metrics", {})
        return investment_metrics.get("emotional_investment", 0.5)

    def _calculate_theme_alignment(self, state: Dict[str, Any]) -> float:
        """Calculate alignment with current themes"""
        current_themes = self._identify_current_themes(state)
        alignment_scores = [self._calculate_theme_score(theme) for theme in current_themes]
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    def _calculate_arc_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate compatibility with current character arcs"""
        current_arcs = self._identify_active_arcs(state)
        compatibility_scores = [self._calculate_arc_score(arc) for arc in current_arcs]
        return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.0

    def _calculate_character_fit(self, state: Dict[str, Any]) -> float:
        """Calculate fit with character development"""
        character_states = self._get_character_states(state)
        fit_scores = [self._calculate_character_score(char) for char in character_states]
        return sum(fit_scores) / len(fit_scores) if fit_scores else 0.0

    def _calculate_world_fit(self, state: Dict[str, Any]) -> float:
        """Calculate fit with world state"""
        world_elements = self._identify_world_elements(state)
        fit_scores = [self._calculate_element_score(element) for element in world_elements]
        return sum(fit_scores) / len(fit_scores) if fit_scores else 0.0

    def _calculate_current_tension(self, state: Dict[str, Any]) -> float:
        """Calculate current narrative tension"""
        tension_sources = self._identify_tension_sources(state)
        tension_scores = [self._calculate_tension_score(source) for source in tension_sources]
        return sum(tension_scores) / len(tension_scores) if tension_scores else 0.0

    def _calculate_tension_trajectory(self, state: Dict[str, Any]) -> float:
        """Calculate tension trajectory"""
        current_tension = self._calculate_current_tension(state)
        historical_tension = self._get_historical_tension(state)
        return self._calculate_trajectory(current_tension, historical_tension)

    def _calculate_resolution_potential(self, state: Dict[str, Any]) -> float:
        """Calculate potential for resolution"""
        resolution_factors = {
            "plot_readiness": self._calculate_plot_readiness(state),
            "character_readiness": self._calculate_character_readiness(state),
            "tension_state": self._calculate_tension_state(state),
            "narrative_momentum": self._calculate_narrative_momentum(state)
        }
        weights = self._get_resolution_weights()
        return sum(v * weights[k] for k, v in resolution_factors.items())

    def _calculate_impact_potential(self, state: Dict[str, Any]) -> float:
        """Calculate potential impact of actions"""
        impact_factors = {
            "immediate_effect": self._calculate_immediate_effect(state),
            "long_term_effect": self._calculate_long_term_effect(state),
            "ripple_effect": self._calculate_ripple_effect(state),
            "narrative_impact": self._calculate_narrative_impact(state)
        }
        weights = self._get_impact_weights()
        return sum(v * weights[k] for k, v in impact_factors.items())

    def _get_narrative_timing_weights(self) -> Dict[str, float]:
        """Get weights for narrative timing factors"""
        return {
            "arc_position": 0.3,
            "tension_level": 0.3,
            "plot_momentum": 0.2,
            "character_readiness": 0.2
        }

    def _get_player_readiness_weights(self) -> Dict[str, float]:
        """Get weights for player readiness factors"""
        return {
            "engagement_level": 0.3,
            "receptivity": 0.3,
            "emotional_state": 0.2,
            "narrative_investment": 0.2
        }

    def _get_plot_alignment_weights(self) -> Dict[str, float]:
        """Get weights for plot alignment factors"""
        return {
            "theme_alignment": 0.3,
            "arc_compatibility": 0.3,
            "character_consistency": 0.2,
            "world_consistency": 0.2
        }

    def _get_tension_timing_weights(self) -> Dict[str, float]:
        """Get weights for tension timing factors"""
        return {
            "current_tension": 0.3,
            "tension_trajectory": 0.3,
            "resolution_potential": 0.2,
            "impact_potential": 0.2
        }

    def _get_investment_weights(self) -> Dict[str, float]:
        """Get weights for investment factors"""
        return {
            "story_engagement": 0.3,
            "character_attachment": 0.3,
            "plot_interest": 0.2,
            "emotional_investment": 0.2
        }

    def _get_resolution_weights(self) -> Dict[str, float]:
        """Get weights for resolution factors"""
        return {
            "plot_readiness": 0.3,
            "character_readiness": 0.3,
            "tension_state": 0.2,
            "narrative_momentum": 0.2
        }

    def _get_impact_weights(self) -> Dict[str, float]:
        """Get weights for impact factors"""
        return {
            "immediate_effect": 0.3,
            "long_term_effect": 0.3,
            "ripple_effect": 0.2,
            "narrative_impact": 0.2
        }

    def _identify_current_themes(self, state: Dict[str, Any]) -> List[str]:
        """Identify current active themes"""
        story_model = state.get("story_model", {})
        return story_model.get("active_themes", [])

    def _calculate_theme_score(self, theme: str) -> float:
        """Calculate alignment score for a theme"""
        theme_metrics = self.autonomous_state["story_model"].get("theme_metrics", {})
        return theme_metrics.get(theme, 0.5)

    def _identify_active_arcs(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify currently active character arcs"""
        story_model = state.get("story_model", {})
        return story_model.get("active_arcs", [])

    def _calculate_arc_score(self, arc: Dict[str, Any]) -> float:
        """Calculate compatibility score for an arc"""
        arc_metrics = self.autonomous_state["story_model"].get("arc_metrics", {})
        return arc_metrics.get(arc["id"], 0.5)

    def _get_character_states(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get current character states"""
        return list(self.universe_state["character_states"].values())

    def _calculate_character_score(self, char: Dict[str, Any]) -> float:
        """Calculate fit score for a character"""
        char_metrics = self.autonomous_state["story_model"].get("character_metrics", {})
        return char_metrics.get(char.get("id"), 0.5)

    def _identify_world_elements(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify relevant world elements"""
        world_state = state.get("world_state", {})
        return world_state.get("active_elements", [])

    def _calculate_element_score(self, element: Dict[str, Any]) -> float:
        """Calculate fit score for a world element"""
        element_metrics = self.autonomous_state["story_model"].get("element_metrics", {})
        return element_metrics.get(element["id"], 0.5)

    def _identify_tension_sources(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify current tension sources"""
        story_model = state.get("story_model", {})
        return story_model.get("tension_sources", [])

    def _calculate_tension_score(self, source: Dict[str, Any]) -> float:
        """Calculate tension score for a source"""
        tension_metrics = self.autonomous_state["story_model"].get("tension_metrics", {})
        return tension_metrics.get(source["id"], 0.5)

    def _get_historical_tension(self, state: Dict[str, Any]) -> List[float]:
        """Get historical tension values"""
        story_model = state.get("story_model", {})
        return story_model.get("tension_history", [])

    def _calculate_trajectory(self, current: float, historical: List[float]) -> float:
        """Calculate trajectory from historical values"""
        if not historical:
            return 0.0
        recent = historical[-3:]  # Look at last 3 values
        if not recent:
            return 0.0
        avg_change = sum(b - a for a, b in zip(recent[:-1], recent[1:])) / len(recent[:-1])
        return avg_change

    def _calculate_plot_readiness(self, state: Dict[str, Any]) -> float:
        """Calculate plot readiness for resolution"""
        plot_metrics = state.get("plot_metrics", {})
        return plot_metrics.get("readiness", 0.5)

    def _calculate_tension_state(self, state: Dict[str, Any]) -> float:
        """Calculate current tension state"""
        tension_metrics = state.get("tension_metrics", {})
        return tension_metrics.get("current_state", 0.5)

    def _calculate_narrative_momentum(self, state: Dict[str, Any]) -> float:
        """Calculate current narrative momentum"""
        narrative_metrics = state.get("narrative_metrics", {})
        return narrative_metrics.get("momentum", 0.5)

    def _calculate_immediate_effect(self, state: Dict[str, Any]) -> float:
        """Calculate immediate effect potential"""
        effect_metrics = state.get("effect_metrics", {})
        return effect_metrics.get("immediate", 0.5)

    def _calculate_long_term_effect(self, state: Dict[str, Any]) -> float:
        """Calculate long-term effect potential"""
        effect_metrics = state.get("effect_metrics", {})
        return effect_metrics.get("long_term", 0.5)

    def _calculate_ripple_effect(self, state: Dict[str, Any]) -> float:
        """Calculate ripple effect potential"""
        effect_metrics = state.get("effect_metrics", {})
        return effect_metrics.get("ripple", 0.5)

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.universe_state["meta_awareness"]["breaking_points"]) * 0.1
        return {
            "game_awareness": min(1.0, current_awareness + 0.1),
            "nyx_awareness": min(1.0, current_awareness + 0.05),
            "narrative_awareness": min(1.0, current_awareness + 0.15)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        if context.get("type") == "subtle":
            return base_impact * 0.5
        elif context.get("type") == "moderate":
            return base_impact * 1.0
        else:  # overt
            return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        return {
            "meta_awareness": 0.1 * (1 + len(self.universe_state["meta_awareness"]["breaking_points"])),
            "trust": 0.05 * (2 if context.get("type") == "subtle" else 1),
            "engagement": 0.15 * (1 + self.social_link["influence"])
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str,
