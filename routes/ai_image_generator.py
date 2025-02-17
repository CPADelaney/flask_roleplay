# routes/ai_image_generator.py

import os
import json
import hashlib
import requests
from db.connection import get_db_connection
from flask import Blueprint, request, jsonify, session

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
# 1️⃣ FETCH ROLEPLAY DATA & NPC DETAILS
# ================================
def get_npc_and_roleplay_context(user_id, conversation_id, npc_names):
    """Fetch NPCStats details for the NPCs mentioned in the GPT response."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT npc_name, physical_description, dominance, cruelty, intensity, 
               personality_traits, likes, dislikes, current_location
        FROM NPCStats 
        WHERE user_id=%s AND conversation_id=%s AND npc_name IN %s
    """, (user_id, conversation_id, tuple(npc_names)))

    detailed_npcs = {}
    for row in cursor.fetchall():
        detailed_npcs[row[0]] = {
            "physical_description": row[1],
            "dominance": row[2],
            "cruelty": row[3],
            "intensity": row[4],
            "personality_traits": json.loads(row[5] or "[]"),
            "likes": json.loads(row[6] or "[]"),
            "dislikes": json.loads(row[7] or "[]"),
            "current_location": row[8]
        }

    cursor.close()
    conn.close()
    return detailed_npcs


# ================================
# 2️⃣ PROCESS GPT RESPONSE & MERGE WITH NPCStats
# ================================
def process_gpt_scene_data(gpt_response, user_id, conversation_id):
    """Extracts NPCs, actions, setting from GPT-4o's structured response,
       then enhances it with deeper character details from NPCStats."""
    
    if not gpt_response or "scene_data" not in gpt_response:
        return None

    scene_data = gpt_response["scene_data"]
    npc_names = scene_data["npc_names"]

    # Fetch character details from NPCStats
    detailed_npcs = get_npc_and_roleplay_context(user_id, conversation_id, npc_names)

    # Merge GPT context with database details
    npcs = []
    for name in npc_names:
        npcs.append({
            "name": name,
            "physical_description": detailed_npcs.get(name, {}).get("physical_description", "unknown"),
            "dominance": detailed_npcs.get(name, {}).get("dominance", 50),
            "cruelty": detailed_npcs.get(name, {}).get("cruelty", 50),
            "intensity": detailed_npcs.get(name, {}).get("intensity", 50),
            "expression": scene_data["expressions"].get(name, "neutral"),
            "personality_traits": detailed_npcs.get(name, {}).get("personality_traits", []),
            "likes": detailed_npcs.get(name, {}).get("likes", []),
            "dislikes": detailed_npcs.get(name, {}).get("dislikes", [])
        })

    return {
        "npcs": npcs,
        "actions": scene_data["actions"],
        "setting": scene_data["setting"],
        "npc_positions": scene_data["npc_positions"],
        "mood": scene_data["mood"]
    }


# ================================
# 3️⃣ GENERATE GPT-4o AI IMAGE PROMPT
# ================================
def generate_prompt_from_gpt_and_db(scene_data):
    """Generates an AI image prompt based on merged GPT-4o scene data and NPCStats details."""
    if not scene_data:
        return None

    npc_details = ", ".join([
        f"{npc['name']} ({npc['expression']}), {npc['physical_description']}, "
        f"dominance: {npc['dominance']}/100, cruelty: {npc['cruelty']}/100, intensity: {npc['intensity']}/100"
        for npc in scene_data["npcs"]
    ])

    actions = ", ".join(scene_data["actions"])
    setting = scene_data["setting"]
    mood = scene_data["mood"]

    return f"{npc_details} in {setting}, engaging in {actions}, {mood}, anime-style, highly detailed, NSFW."


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
# 5️⃣ GENERATE ROLEPLAY IMAGE (GPT + NPCStats)
# ================================
def generate_roleplay_image_from_gpt(gpt_response, user_id, conversation_id):
    """Fetches roleplay data from GPT's response, enhances it with NPCStats details, and generates (or retrieves) an AI image."""
    scene_data = process_gpt_scene_data(gpt_response, user_id, conversation_id)
    if not scene_data:
        return {"error": "No valid scene data found in GPT response"}

    optimized_prompt = generate_prompt_from_gpt_and_db(scene_data)

    cached_images = get_cached_images(optimized_prompt)
    if cached_images:
        return {"image_urls": cached_images, "cached": True, "prompt_used": optimized_prompt}

    generated_images = []
    for variation_id in range(5):
        image_url = generate_ai_image(optimized_prompt)
        if image_url:
            cached_path = save_image_to_cache(image_url, optimized_prompt, variation_id)
            generated_images.append(cached_path or image_url)

    return {"image_urls": generated_images, "cached": False, "prompt_used": optimized_prompt}


# ================================
# 6️⃣ PROCESS GPT RESPONSE & TRIGGER IMAGE
# ================================
def process_gpt_response(gpt_response, user_id, conversation_id):
    """Processes GPT's response and generates an image if scene data is present."""
    scene_image_data = generate_roleplay_image_from_gpt(gpt_response, user_id, conversation_id)

    return {
        "response": gpt_response["response"],
        "scene_image_urls": scene_image_data["image_urls"],
        "scene_cached": scene_image_data["cached"]
    }


# ================================
# 7️⃣ API ROUTE: GPT + IMAGE RESPONSE
# ================================
image_bp = Blueprint("image_bp", __name__)

@image_bp.route("/generate_gpt_image", methods=["POST"])
def generate_gpt_image():
    """API route to process a GPT message and generate an AI image based on scene context & NPCStats."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    gpt_response = data.get("gpt_response")
    conversation_id = data.get("conversation_id")

    if not gpt_response or not conversation_id:
        return jsonify({"error": "Missing GPT response or conversation_id"}), 400

    result = process_gpt_response(gpt_response, user_id, conversation_id)
    return jsonify(result)
