# logic/gpt_image_decision.py

import time
import json
import logging
from flask import session
from db.connection import get_db_connection_context

class ImageGenerationDecider:
    """
    Decides whether to generate an image based on scene context, 
    user preferences, and rate limiting.
    """
    
    def __init__(self, user_id, conversation_id):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.user_preferences = None
        self.recent_generations = None
    
    async def initialize(self):
        """Initialize the decider by loading preferences and recent generations."""
        self.user_preferences = await self._load_user_preferences()
        self.recent_generations = self._get_recent_generations()
        return self
    
    async def _load_user_preferences(self):
        """Load user preferences for image generation."""
        preferences = {
            'frequency': 'medium',  # low, medium, high
            'nsfw_level': 'moderate',  # none, mild, moderate, explicit
            'focus_preference': 'balanced',  # character, setting, action, balanced
            'disable_images': False,
            'image_budget': 50  # Number of images per day/session
        }
        
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT key, value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key LIKE 'image_pref%'
                """, self.user_id, self.conversation_id)
                
                # Override defaults with user preferences from DB
                for row in rows:
                    key = row['key'].replace('image_pref_', '')
                    value = row['value']
                    # Convert string "true"/"false" to boolean if needed
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    # Convert numeric strings to integers if needed
                    try:
                        if value.isdigit():
                            value = int(value)
                    except AttributeError:
                        pass
                        
                    preferences[key] = value
        except Exception as e:
            logging.error(f"Error loading user preferences: {e}")
            
        return preferences
    
    def _get_recent_generations(self):
        """Get timestamps of recent image generations."""
        # This could be stored in a database or in session
        return session.get('recent_image_generations', [])
    
    def _update_recent_generations(self, timestamp):
        """Add a new generation timestamp and trim old ones."""
        recent = self.recent_generations
        recent.append(timestamp)
        
        # Keep only last 24 hours of generations
        cutoff = timestamp - (24 * 60 * 60)
        recent = [t for t in recent if t >= cutoff]
        
        session['recent_image_generations'] = recent
        self.recent_generations = recent
    
    def _check_rate_limits(self):
        """Check if we've hit rate limits for image generation."""
        now = time.time()
        recent = self.recent_generations
        
        # Rate limiting rules based on frequency preference
        if self.user_preferences['frequency'] == 'low':
            # Max 1 image per 10 minutes, max 10 per day
            ten_min_ago = now - (10 * 60)
            last_10_min = [t for t in recent if t >= ten_min_ago]
            if len(last_10_min) >= 1:
                return False, "Rate limited: Maximum 1 image per 10 minutes"
            if len(recent) >= 10:
                return False, "Rate limited: Daily image limit reached"
                
        elif self.user_preferences['frequency'] == 'medium':
            # Max 1 image per 5 minutes, max 25 per day
            five_min_ago = now - (5 * 60)
            last_5_min = [t for t in recent if t >= five_min_ago]
            if len(last_5_min) >= 1:
                return False, "Rate limited: Maximum 1 image per 5 minutes"
            if len(recent) >= 25:
                return False, "Rate limited: Daily image limit reached"
                
        elif self.user_preferences['frequency'] == 'high':
            # Max 1 image per 2 minutes, max 50 per day
            two_min_ago = now - (2 * 60)
            last_2_min = [t for t in recent if t >= two_min_ago]
            if len(last_2_min) >= 1:
                return False, "Rate limited: Maximum 1 image per 2 minutes"
            if len(recent) >= 50:
                return False, "Rate limited: Daily image limit reached"
        
        # Check against user's custom image budget
        if len(recent) >= self.user_preferences['image_budget']:
            return False, f"Rate limited: Custom budget of {self.user_preferences['image_budget']} images reached"
        
        return True, None
    
    async def should_generate_image(self, gpt_response):
        """
        Determine if an image should be generated for this response.
        
        Args:
            gpt_response: The JSON response from GPT with scene_data and image_generation
            
        Returns:
            tuple: (should_generate, reason)
        """
        if not self.user_preferences:
            await self.initialize()
            
        # Check if images are disabled entirely
        if self.user_preferences['disable_images']:
            return False, "Images disabled by user preference"
        
        # Parse the GPT response
        try:
            # If it's already parsed into a dictionary
            if isinstance(gpt_response, dict):
                response_data = gpt_response
            else:
                # If it's a JSON string
                response_data = json.loads(gpt_response)
        except (json.JSONDecodeError, TypeError):
            return False, "Invalid response format"
        
        # Check if GPT explicitly requested image generation
        if 'image_generation' in response_data and response_data['image_generation'].get('generate', False):
            explicit_request = True
            priority = response_data['image_generation'].get('priority', 'medium')
        else:
            explicit_request = False
            priority = 'low'
        
        # If not explicit, check scene data for implicit triggers
        if not explicit_request and 'scene_data' in response_data:
            scene_data = response_data['scene_data']
            visibility_triggers = scene_data.get('visibility_triggers', {})
            
            # Evaluate triggers
            trigger_score = 0
            
            if visibility_triggers.get('character_introduction', False):
                trigger_score += 30
            
            if visibility_triggers.get('significant_location', False):
                trigger_score += 20
            
            emotional_intensity = visibility_triggers.get('emotional_intensity', 0)
            trigger_score += min(emotional_intensity // 10, 10)  # Max 10 points
            
            intimacy_level = visibility_triggers.get('intimacy_level', 0)
            trigger_score += min(intimacy_level // 10, 20)  # Max 20 points
            
            if visibility_triggers.get('appearance_change', False):
                trigger_score += 25
            
            # Determine priority based on trigger score
            if trigger_score >= 50:
                priority = 'high'
            elif trigger_score >= 30:
                priority = 'medium'
            else:
                # Not enough implicit triggers
                return False, "Insufficient visual interest in scene"
        
        # Check rate limits
        can_generate, limit_reason = self._check_rate_limits()
        if not can_generate:
            return False, limit_reason
        
        # If we pass all checks and priority is high or explicit, generate
        if priority == 'high' or explicit_request:
            now = time.time()
            self._update_recent_generations(now)
            return True, "High priority scene worthy of visualization"
        
        # For medium priority, apply frequency preferences
        if priority == 'medium':
            if self.user_preferences['frequency'] in ['medium', 'high']:
                now = time.time()
                self._update_recent_generations(now)
                return True, "Medium priority scene with user preference for frequent images"
        
        # Default fallback
        return False, "Scene didn't meet visualization threshold"

async def should_generate_image_for_response(user_id, conversation_id, gpt_response):
    """Convenience function to check if an image should be generated."""
    decider = ImageGenerationDecider(user_id, conversation_id)
    await decider.initialize()
    return await decider.should_generate_image(gpt_response)
