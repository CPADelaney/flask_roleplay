# routes/ai_image_generator.py

import os
import json
import hashlib
import requests
import random
from db.connection import get_db_connection

# 🔹 API KEYS
OPENAI_API_KEY = "your_openai_api_key"
STABILITY_API_KEY = "your_stability_ai_key"

# 🔹 API URLs
STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
GPT4O_API_URL = "https://api.openai.com/v1/chat/completions"

# 🔹 Image Caching Directory
CACHE_DIR = "static/images"
os.makedirs(CACHE_DIR, exist_ok=True)


# ================================
# 1️⃣ FETCH ROLEPLAY DATA & NPCS
# ================================
def get_npc_and_roleplay_context(user_id, conversation_id):
    """Fetch up to 5 NPCs and roleplay data for image generation."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT npc_name, physical_description, dominance, cruelty, intensity, 
               personality_traits, likes, dislikes, current_location
        FROM NPCStats WHERE user_id=%s AND conversation_id=%s 
        ORDER BY intensity DESC LIMIT 5
    """, (user_id, conversation_id))
    npc_rows = cursor.fetchall()

    npcs = [{
        "name": row[0],
        "physical_description": row[1],
        "dominance": row[2],
        "cruelty": row[3],
        "intensity": row[4],
        "personality_traits": json.loads(row[5] or "[]"),
        "likes": json.loads(row[6] or "[]"),
        "dislikes": json.loads(row[7] or "[]"),
        "current_location": row[8]
    } for row in npc_rows]

    cursor.execute("""
        SELECT key, value FROM CurrentRoleplay WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    roleplay_data = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.close()
    conn.close()
    return npcs, roleplay_data


# ================================
# 2️⃣ NPC POSITIONING SYSTEM
# ================================
def assign_npc_positions(npcs):
    """Dynamically assigns positions and interactions for multi-character roleplay scenes."""
    positions = []
    dominant_npcs = sorted(npcs, key=lambda x: x['dominance'], reverse=True)
    
    for index, npc in enumerate(dominant_npcs):
        if index == 0:
            role = "Primary Dominant (center of the scene, direct engagement)"
            angle = "close-up or low-angle shot for dominance"
        elif index == 1:
            role = "Secondary Dominant (flanking, supporting actions)"
            angle = "side view or over-the-shoulder perspective"
        elif index == 2:
            role = "Observer NPC (watching the scene unfold, teasing the protagonist)"
            angle = "mid-distance perspective"
        else:
            role = "Peripheral NPC (reacting to the main action, optional involvement)"
            angle = "wide-angle shot"
        
        positions.append({
            "npc": npc["name"],
            "role": role,
            "suggested_camera_angle": angle
        })
    
    return positions


# ================================
# 3️⃣ GPT-4o OPTIMIZED PROMPT
# ================================
def generate_optimized_prompt(npcs, roleplay_data):
    """Uses GPT-4o to refine Stability AI prompts for cinematic perspectives, lighting, and NPC reactions."""
    if not npcs or not roleplay_data:
        return None

    messages = [
        {"role": "system", "content": "You are an expert at crafting highly detailed AI image prompts for Stability AI."},
        {"role": "user", "content": f"""
        Generate a **highly detailed AI image prompt** for an **NSFW roleplay scene** that includes **dynamic camera angles, realistic lighting, motion blur, and proper NPC positioning.**

        **NPCs in Scene (Up to 5):**
        {json.dumps(npcs, indent=2)}

        **Roleplay Context:**
        {json.dumps(roleplay_data, indent=2)}

        📌 **Ensure the prompt includes**:
        - **Character Details** (appearance, expressions, body language)
        - **Actions & Poses** (NPCs interacting dynamically)
        - **Lighting Effects** (Choose dynamically: soft glow, dramatic shadows, neon, candle-lit)
        - **Motion & Blur Effects** (Subtle movement, depth of field, sharp focus on dominant NPC)
        - **Camera Angles** (Choose dynamically: POV, Over-the-shoulder, Low-angle, Side, Close-up, Wide)
        - **NPC Positioning & Reactions** (Ensure proper hierarchy & realistic engagement)

        📌 **Example Output**:
        '[Character 1] with [Character 2], captured in [Camera Angle] with [Lighting Effect] inside [Setting], engaging in [Action], anime-style, highly detailed, motion blur, NSFW.'
        """}
    ]

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"model": "gpt-4o", "messages": messages, "temperature": 0.7}

    response = requests.post(GPT4O_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("Error:", response.text)
        return None


# ================================
# 4️⃣ IMAGE CACHING SYSTEM
# ================================
def get_cached_images(prompt):
    """Check if multiple images exist for a given prompt (different angles & lighting effects)."""
    hashed_prompt = hashlib.md5(prompt.encode()).hexdigest()
    image_paths = [
        os.path.join(CACHE_DIR, f"{hashed_prompt}_variation_{i}.png") for i in range(5)
    ]
    return [path for path in image_paths if os.path.exists(path)]

def save_image_to_cache(image_url, prompt, variation_id):
    """Download and save AI-generated image with a specific effect variation."""
    hashed_prompt = hashlib.md5(prompt.encode()).hexdigest()
    image_path = os.path.join(CACHE_DIR, f"{hashed_prompt}_variation_{variation_id}.png")

    response = requests.get(image_url)
    if response.status_code == 200:
        with open(image_path, "wb") as file:
            file.write(response.content)
        return image_path
    return None


# ================================
# 5️⃣ FULL ROLEPLAY IMAGE PIPELINE
# ================================
def generate_roleplay_image(user_id, conversation_id):
    """Fetches roleplay data, builds a GPT-4o optimized Stability AI prompt, and generates (or retrieves) multiple AI images."""
    npcs, roleplay_data = get_npc_and_roleplay_context(user_id, conversation_id)

    if not npcs or not roleplay_data:
        return {"error": "No NPC or roleplay data found"}

    optimized_prompt = generate_optimized_prompt(npcs, roleplay_data)
    npc_positions = assign_npc_positions(npcs)

    cached_images = get_cached_images(optimized_prompt)
    if cached_images:
        return {"image_urls": cached_images, "npc_positions": npc_positions, "cached": True, "prompt_used": optimized_prompt}

    generated_images = []
    for variation_id in range(5):
        image_url = generate_ai_image(optimized_prompt)
        if image_url:
            cached_path = save_image_to_cache(image_url, optimized_prompt, variation_id)
            generated_images.append(cached_path or image_url)

    return {"image_urls": generated_images, "npc_positions": npc_positions, "cached": False, "prompt_used": optimized_prompt}
