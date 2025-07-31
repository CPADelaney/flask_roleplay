# story_agent/story_balance_manager.py

class StoryBalanceManager:
    """Manages the balance between preset story and dynamic generation"""
    
    def __init__(self, flexibility_level: float = 0.7):
        self.flexibility_level = flexibility_level
        
    async def should_enforce_preset(
        self, 
        story_beat: StoryBeat,
        player_actions: List[str],
        current_context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Determine if a preset beat should be enforced"""
        
        # Always enforce non-skippable beats
        if not story_beat.can_skip:
            return True, "Critical story moment"
            
        # Check player resistance
        resistance_score = self.calculate_player_resistance(
            player_actions, 
            story_beat
        )
        
        # Check narrative coherence
        coherence_score = self.calculate_narrative_coherence(
            current_context,
            story_beat
        )
        
        # Make decision based on flexibility
        if resistance_score > self.flexibility_level:
            return False, "Player agency respected"
        elif coherence_score < 0.3:
            return False, "Would break narrative flow"
        else:
            return True, "Story progression needed"
            
    async def adapt_preset_content(
        self,
        preset_content: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt preset content to match player's journey"""
        
        adapted = preset_content.copy()
        
        # Adapt dialogue based on relationships
        if "dialogue" in adapted:
            adapted["dialogue"] = self.personalize_dialogue(
                adapted["dialogue"],
                player_context["relationships"]
            )
            
        # Adjust intensity based on player stats
        if "intensity" in adapted:
            adapted["intensity"] = self.scale_intensity(
                adapted["intensity"],
                player_context["player_stats"]
            )
            
        return adapted
        
    async def handle_player_deviation(self, ctx, deviation_type: str, beat: StoryBeat):
        """Handle when player deviates from preset path"""
        
        if beat.can_skip:
            # Mark as skipped, may trigger later
            await self.queue_for_later_opportunity(beat)
        else:
            # Critical beat - find creative way to guide back
            if deviation_type == "wrong_location":
                # Create NPC motivation to bring player to location
                await self.create_npc_guidance(ctx, beat.required_location)
            elif deviation_type == "avoiding_npc":
                # Have NPC seek out player
                await self.create_npc_encounter(ctx, beat.required_npcs[0])
                
    async def blend_preset_with_dynamic(self, ctx, preset_content, dynamic_events):
        """Seamlessly blend preset and dynamic content"""
        
        # Analyze current dynamic events
        tone = await analyze_narrative_tone(dynamic_events)
        
        # Adjust preset content to match
        adjusted_content = preset_content.copy()
        if tone.get("tension_high"):
            adjusted_content["intensity"] += 1
            
        # Incorporate recent dynamic elements
        if "recent_conflicts" in dynamic_events:
            adjusted_content["acknowledge_conflicts"] = True
            
        return adjusted_content
