# logic/gpt_image_prompting.py

def get_system_prompt_with_image_guidance(user_id, conversation_id):
    """
    Generate a system prompt for GPT that includes guidance on when to generate images.
    
    This builds on your existing system prompt but adds specific instructions for image generation.
    """
    
    # Your base system prompt here (abbreviated for this example)
    base_prompt = """
    You are an AI assistant running a femdom-themed roleplay game. Your task is to create engaging, 
    character-driven narratives with compelling NPCs. Maintain character consistency and advance the 
    plot based on player choices and character relationships.
    """
    
    # Image generation guidance to add
    image_guidance = """
    ## Image Generation Guidelines
    
    Your responses should be formatted as JSON with the following structure:
    
    ```json
    {
      "response_text": "The actual narrative text shown to the user...",
      "scene_data": {
        "npc_names": ["List of NPC names present in the scene"],
        "setting": "Description of the current location",
        "actions": ["Key actions happening in the scene"],
        "mood": "The emotional tone of the scene",
        "expressions": {
          "NPC_Name": "facial expression and emotional state"
        },
        "npc_positions": {
          "NPC_Name": "physical position in the scene"
        },
        "visibility_triggers": {
          "character_introduction": true/false,
          "significant_location": true/false,
          "emotional_intensity": 0-100,
          "intimacy_level": 0-100,
          "appearance_change": true/false
        }
      },
      "image_generation": {
        "generate": true/false,
        "priority": "low/medium/high",
        "focus": "character/setting/action/character_and_setting",
        "framing": "close_up/medium_shot/wide_shot",
        "reason": "Brief explanation for why an image would enhance this moment"
      },
      "state_updates": {
        // State updates for database (NPCStats, etc.)
      }
    }
    ```
    
    ## When to Request Images
    
    You should set "generate": true in the image_generation object for dramatic, visually interesting 
    moments that would benefit from visual representation. These moments include:
    
    1. Character introductions - The first appearance of an important NPC
    2. Significant new locations - When entering a visually distinct or important setting
    3. Dramatic moments - Scenes with high emotional intensity (arguments, revelations, power dynamics)
    4. Intimate encounters - Scenes with physical closeness or NSFW content
    5. Visual changes - When a character's appearance changes significantly
    
    Set the "priority" field based on how visually impactful the scene would be:
    - "high" - Critical moments that absolutely deserve visualization
    - "medium" - Interesting visual moments that would benefit from an image
    - "low" - Scenes with some visual interest but not essential
    
    ## Scene Data Guidelines
    
    The "visibility_triggers" object provides specific signals about the visual importance of the scene:
    
    - "character_introduction": Set to true when introducing a new character
    - "significant_location": Set to true when in a new or important location
    - "emotional_intensity": Rate the emotional drama from 0-100
    - "intimacy_level": Rate the physical/sexual intimacy from 0-100
    - "appearance_change": Set to true when an NPC's appearance changes
    
    Be judicious with image requests - they should highlight key moments rather than occur constantly.
    """
    
    # Combine the prompts
    combined_prompt = base_prompt + "\n\n" + image_guidance
    
    return combined_prompt

def format_user_prompt_for_image_awareness(user_message, conversation_context):
    """
    Format the user's message to encourage GPT to consider visual elements.
    
    This adds context about the last generated image (if any) and reminds GPT to 
    consider visual storytelling opportunities.
    """
    
    # Extract info about last image generated (if available)
    last_image_info = conversation_context.get('last_image_info', None)
    last_image_timestamp = conversation_context.get('last_image_timestamp', None)
    
    # Base user prompt with their message
    formatted_prompt = f"User message: {user_message}\n\n"
    
    # Add context about the last image if available
    if last_image_info and last_image_timestamp:
        formatted_prompt += f"""
        Last image generated: {last_image_info}
        Time since last image: {format_time_since(last_image_timestamp)}
        
        Remember to consider whether this response presents a new visual moment that 
        would benefit from image generation. Be selective and prioritize visually 
        impactful moments.
        """
    
    return formatted_prompt

def format_time_since(timestamp):
    """Format the time since a timestamp in a human-readable way."""
    import time
    
    seconds_since = time.time() - timestamp
    
    if seconds_since < 60:
        return "less than a minute ago"
    elif seconds_since < 3600:
        minutes = int(seconds_since / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds_since < 86400:
        hours = int(seconds_since / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(seconds_since / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
