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
