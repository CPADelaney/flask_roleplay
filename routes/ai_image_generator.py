# routes/ai_image_generator.py

import os
import json
import hashlib
import logging
import time
import base64
from datetime import datetime
from urllib.parse import urlparse
import requests
from werkzeug.utils import secure_filename
from db.connection import get_db_connection
from flask import Blueprint, request, jsonify, session, current_app
from logic.addiction_system import get_addiction_status
from logic.chatgpt_integration import get_openai_client, safe_json_loads
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API KEYS
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")

# API URLs
STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
GPT4O_API_URL = "https://api.openai.com/v1/chat/completions"

# Image Caching Directory
CACHE_DIR = "static/images"
os.makedirs(CACHE_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# DATABASE SETUP
# ================================
def setup_database():
    """Setup necessary database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # NPCVisualAttributes table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS NPCVisualAttributes (
        id SERIAL PRIMARY KEY,
        npc_id INTEGER,
        user_id INTEGER,
        conversation_id TEXT,
        hair_color TEXT,
        hair_style TEXT,
        eye_color TEXT,
        skin_tone TEXT,
        body_type TEXT,
        height TEXT,
        age_appearance TEXT,
        default_outfit TEXT,
        outfit_variations JSONB,
        makeup_style TEXT,
        accessories JSONB,
        expressions JSONB,
        poses JSONB,
        visual_seed TEXT,
        last_generated_image TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (npc_id) REFERENCES NPCStats(id),
        FOREIGN KEY (user_id) REFERENCES Users(id)
    )
    """)
    
    # ImageFeedback table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ImageFeedback (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        conversation_id TEXT,
        image_path TEXT,
        original_prompt TEXT,
        npc_names JSONB,
        rating INTEGER CHECK (rating BETWEEN 1 AND 5),
        feedback_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES Users(id)
    )
    """)
    
    # NPCVisualEvolution table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS NPCVisualEvolution (
        id SERIAL PRIMARY KEY,
        npc_id INTEGER,
        user_id INTEGER,
        conversation_id TEXT,
        event_type TEXT CHECK (event_type IN ('outfit_change', 'appearance_change', 'location_change', 'mood_change')),
        event_description TEXT,
        previous_state JSONB,
        current_state JSONB,
        scene_context TEXT,
        image_generated TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (npc_id) REFERENCES NPCStats(id),
        FOREIGN KEY (user_id) REFERENCES Users(id)
    )
    """)
    
    # UserVisualPreferences view
    cursor.execute("""
    CREATE OR REPLACE VIEW UserVisualPreferences AS
    SELECT 
        user_id,
        npc_name,
        AVG(rating) as avg_rating,
        COUNT(*) as feedback_count
    FROM 
        ImageFeedback,
        jsonb_array_elements_text(npc_names) as npc_name
    WHERE 
        rating >= 4
    GROUP BY 
        user_id, npc_name
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Database tables setup complete")

# ================================
# 1️⃣ FETCH ROLEPLAY DATA & NPC DETAILS
# ================================
def get_npc_and_roleplay_context(user_id, conversation_id, npc_names, player_name="Chase"):
    """Fetch detailed NPCStats, NPCVisualAttributes, PlayerStats, SocialLinks, and PlayerJournal."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch NPC data
    cursor.execute("""
        SELECT n.id, n.npc_name, n.physical_description, n.dominance, n.cruelty, n.intensity, 
               n.archetype_summary, n.archetype_extras_summary, n.personality_traits, 
               n.likes, n.dislikes, n.current_location, n.visual_seed
        FROM NPCStats n
        WHERE n.user_id=%s AND n.conversation_id=%s AND n.npc_name IN %s
    """, (user_id, conversation_id, tuple(npc_names)))
    
    npc_rows = cursor.fetchall()
    detailed_npcs = {}
    
    for row in npc_rows:
        npc_id = row[0]
        npc_name = row[1]
        
        # Fetch Visual Attributes
        cursor.execute("""
            SELECT hair_color, hair_style, eye_color, skin_tone, body_type, 
                   height, age_appearance, default_outfit, outfit_variations,
                   makeup_style, accessories, expressions, poses, visual_seed,
                   last_generated_image
            FROM NPCVisualAttributes
            WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
        """, (npc_id, user_id, conversation_id))
        visual_attrs = cursor.fetchone()

        # Fetch Previous Images
        cursor.execute("""
            SELECT image_generated
            FROM NPCVisualEvolution
            WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
            ORDER BY timestamp DESC
            LIMIT 5
        """, (npc_id, user_id, conversation_id))
        previous_images = [img[0] for img in cursor.fetchall() if img[0]]

        # Fetch Social Links for this NPC
        cursor.execute("""
            SELECT link_type, link_level, dynamics
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s AND 
                  (entity1_type='npc' AND entity1_id=%s AND entity2_type='player') OR 
                  (entity2_type='npc' AND entity2_id=%s AND entity1_type='player')
            LIMIT 1
        """, (user_id, conversation_id, npc_id, npc_id))
        social_link = cursor.fetchone()

        detailed_npcs[npc_name] = {
            "id": npc_id,
            "physical_description": row[2] or "A figure shaped by her role",
            "dominance": row[3],
            "cruelty": row[4],
            "intensity": row[5],
            "archetype_summary": row[6] or "",
            "archetype_extras_summary": row[7] or "",
            "personality_traits": json.loads(row[8] or "[]"),
            "likes": json.loads(row[9] or "[]"),
            "dislikes": json.loads(row[10] or "[]"),
            "current_location": row[11],
            "visual_seed": row[12] or hashlib.md5(f"{row[1]}{row[6]}".encode()).hexdigest(),
            "previous_images": previous_images,
            "social_link": {
                "link_type": social_link[0] if social_link else None,
                "link_level": social_link[1] if social_link else 0,
                "dynamics": json.loads(social_link[2] or "{}") if social_link else {}
            }
        }
        
        if visual_attrs:
            detailed_npcs[npc_name].update({
                "hair_color": visual_attrs[0],
                "hair_style": visual_attrs[1],
                "eye_color": visual_attrs[2],
                "skin_tone": visual_attrs[3],
                "body_type": visual_attrs[4],
                "height": visual_attrs[5],
                "age_appearance": visual_attrs[6],
                "default_outfit": visual_attrs[7],
                "outfit_variations": json.loads(visual_attrs[8] or "{}"),
                "makeup_style": visual_attrs[9],
                "accessories": json.loads(visual_attrs[10] or "[]"),
                "expressions": json.loads(visual_attrs[11] or "{}"),
                "poses": json.loads(visual_attrs[12] or "[]"),
                "last_generated_image": visual_attrs[14]
            })

    # Fetch PlayerStats
    cursor.execute("""
        SELECT obedience, corruption, willpower, shame, dependency, lust, mental_resilience
        FROM PlayerStats 
        WHERE user_id=%s AND conversation_id=%s AND player_name=%s
        LIMIT 1
    """, (user_id, conversation_id, player_name))
    player_row = cursor.fetchone()
    player_stats = {k: v for k, v in zip(
        ["obedience", "corruption", "willpower", "shame", "dependency", "lust", "mental_resilience"],
        player_row if player_row else [50, 0, 50, 0, 0, 0, 50]
    )}

    # Fetch recent PlayerJournal entries
    cursor.execute("""
        SELECT entry_text, entry_type
        FROM PlayerJournal
        WHERE user_id=%s AND conversation_id=%s
        ORDER BY timestamp DESC
        LIMIT 3
    """, (user_id, conversation_id))
    journal_entries = [{"text": row[0], "type": row[1]} for row in cursor.fetchall()]

    # Fetch UserVisualPreferences
    cursor.execute("""
        SELECT npc_name, avg_rating
        FROM UserVisualPreferences
        WHERE user_id=%s AND npc_name IN %s
    """, (user_id, tuple(npc_names)))
    user_preferences = {row[0]: {"avg_rating": row[1]} for row in cursor.fetchall()}

    cursor.close()
    conn.close()
    return detailed_npcs, player_stats, user_preferences, journal_entries

# ================================
# 2️⃣ PROCESS GPT RESPONSE & MERGE WITH NPCStats
# ================================
def process_gpt_scene_data(gpt_response, user_id, conversation_id):
    """Extract scene data from GPT, enriched with NPCStats, PlayerStats, SocialLinks, and PlayerJournal."""
    if not gpt_response or "scene_data" not in gpt_response:
        return None

    scene_data = gpt_response["scene_data"]
    npc_names = scene_data["npc_names"]
    detailed_npcs, player_stats, user_preferences, journal_entries = get_npc_and_roleplay_context(
        user_id, conversation_id, npc_names
    )
    addiction_status = get_addiction_status(user_id, conversation_id, "Chase")

    npcs = []
    for name in npc_names:
        npc = detailed_npcs.get(name, {})
        npcs.append({
            "id": npc.get("id"),
            "name": name,
            "physical_description": npc.get("physical_description", "unknown"),
            "dominance": npc.get("dominance", 50),
            "cruelty": npc.get("cruelty", 30),
            "intensity": npc.get("intensity", 40),
            "archetype_summary": npc.get("archetype_summary", ""),
            "archetype_extras_summary": npc.get("archetype_extras_summary", ""),
            "expression": scene_data["expressions"].get(name, "neutral"),
            "personality_traits": npc.get("personality_traits", []),
            "likes": npc.get("likes", []),
            "dislikes": npc.get("dislikes", []),
            "visual_seed": npc.get("visual_seed", ""),
            "current_location": npc.get("current_location", "unknown"),
            "hair_color": npc.get("hair_color"),
            "hair_style": npc.get("hair_style"),
            "eye_color": npc.get("eye_color"),
            "skin_tone": npc.get("skin_tone"),
            "body_type": npc.get("body_type"),
            "height": npc.get("height"),
            "age_appearance": npc.get("age_appearance"),
            "default_outfit": npc.get("default_outfit"),
            "outfit_variations": npc.get("outfit_variations", {}),
            "makeup_style": npc.get("makeup_style"),
            "accessories": npc.get("accessories", []),
            "previous_images": npc.get("previous_images", []),
            "last_generated_image": npc.get("last_generated_image"),
            "social_link": npc.get("social_link", {})
        })

    return {
        "npcs": npcs,
        "actions": scene_data["actions"],
        "setting": scene_data["setting"],
        "npc_positions": scene_data["npc_positions"],
        "mood": scene_data["mood"],
        "player_stats": player_stats,
        "addiction_status": addiction_status,
        "user_preferences": user_preferences,
        "journal_entries": journal_entries
    }

# ================================
# 3️⃣ UPDATE VISUAL ATTRIBUTES VIA GPT-4o
# ================================
def update_npc_visual_attributes(user_id, conversation_id, npc_id, prompt_data, image_path=None):
    """Use GPT-4o to extract and update visual attributes from the image prompt."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch current attributes
    cursor.execute("""
        SELECT hair_color, hair_style, eye_color, skin_tone, body_type, 
               height, age_appearance, default_outfit, makeup_style, accessories
        FROM NPCVisualAttributes
        WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
    """, (npc_id, user_id, conversation_id))
    current_attrs = cursor.fetchone()
    current_state = {
        "hair_color": current_attrs[0] if current_attrs else None,
        "hair_style": current_attrs[1] if current_attrs else None,
        "eye_color": current_attrs[2] if current_attrs else None,
        "skin_tone": current_attrs[3] if current_attrs else None,
        "body_type": current_attrs[4] if current_attrs else None,
        "height": current_attrs[5] if current_attrs else None,
        "age_appearance": current_attrs[6] if current_attrs else None,
        "default_outfit": current_attrs[7] if current_attrs else None,
        "makeup_style": current_attrs[8] if current_attrs else None,
        "accessories": json.loads(current_attrs[9] or "[]") if current_attrs else []
    }

    # GPT-4o extraction
    gpt_prompt = f"""
Given this image generation prompt from a femdom visual novel:
'{prompt_data}'

Extract detailed visual attributes for an NPC:
1. Hair Color (e.g., 'raven-black')
2. Hair Style (e.g., 'curly', 'long')
3. Eye Color (e.g., 'emerald')
4. Skin Tone (e.g., 'alabaster')
5. Body Type (e.g., 'voluptuous')
6. Height (e.g., 'tall')
7. Age Appearance (e.g., 'young adult')
8. Default Outfit (e.g., 'Hyper-Silk gown')
9. Makeup Style (e.g., 'bold lipstick')
10. Accessories (e.g., 'starstone choker')

Return a JSON object with these keys, using 'unknown' if not specified."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": gpt_prompt}],
        temperature=0.5,
        response_format={"type": "json_object"}
    )
    new_attrs = safe_json_loads(response.choices[0].message.content) or {
        "hair_color": "unknown", "hair_style": "unknown", "eye_color": "unknown",
        "skin_tone": "unknown", "body_type": "unknown", "height": "unknown",
        "age_appearance": "unknown", "default_outfit": "unknown", "makeup_style": "unknown",
        "accessories": []
    }

    # Update or insert
    if current_attrs:
        cursor.execute("""
            UPDATE NPCVisualAttributes
            SET hair_color=%s, hair_style=%s, eye_color=%s, skin_tone=%s, body_type=%s,
                height=%s, age_appearance=%s, default_outfit=%s, makeup_style=%s, 
                accessories=%s, last_generated_image=%s, updated_at=CURRENT_TIMESTAMP
            WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
        """, (new_attrs["hair_color"], new_attrs["hair_style"], new_attrs["eye_color"],
              new_attrs["skin_tone"], new_attrs["body_type"], new_attrs["height"],
              new_attrs["age_appearance"], new_attrs["default_outfit"], new_attrs["makeup_style"],
              json.dumps(new_attrs["accessories"]), image_path, npc_id, user_id, conversation_id))
    else:
        cursor.execute("""
            INSERT INTO NPCVisualAttributes
            (npc_id, user_id, conversation_id, hair_color, hair_style, eye_color, 
             skin_tone, body_type, height, age_appearance, default_outfit, makeup_style, 
             accessories, visual_seed, last_generated_image)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (npc_id, user_id, conversation_id, new_attrs["hair_color"], new_attrs["hair_style"],
              new_attrs["eye_color"], new_attrs["skin_tone"], new_attrs["body_type"],
              new_attrs["height"], new_attrs["age_appearance"], new_attrs["default_outfit"],
              new_attrs["makeup_style"], json.dumps(new_attrs["accessories"]),
              hashlib.md5(f"{npc_id}".encode()).hexdigest(), image_path))

    conn.commit()
    cursor.close()
    conn.close()
    return new_attrs, current_state

# ================================
# 4️⃣ GENERATE IMAGE-OPTIMIZED PROMPT VIA GPT-4o
# ================================
def generate_image_prompt(scene_data):
    """Use GPT-4o to summarize NPCStats, scene context, SocialLinks, and PlayerJournal into an image-optimized prompt."""
    if not scene_data:
        return None

    # NPC details with visual attributes and SocialLinks
    npc_details = []
    for npc in scene_data["npcs"]:
        visual_attrs = [
            f"Hair: {npc.get('hair_color', 'unknown')} {npc.get('hair_style', '')}",
            f"Eyes: {npc.get('eye_color', 'unknown')}",
            f"Skin: {npc.get('skin_tone', 'unknown')}",
            f"Body: {npc.get('body_type', 'unknown')}",
            f"Outfit: {npc.get('default_outfit', 'unknown')}"
        ]
        social_link = npc["social_link"]
        npc_details.append(
            f"Name: {npc['name']}\n"
            f"Physical Description: {npc['physical_description']}\n"
            f"Visual Attributes: {', '.join(visual_attrs)}\n"
            f"Dominance: {npc['dominance']}\nCruelty: {npc['cruelty']}\nIntensity: {npc['intensity']}\n"
            f"Archetype: {npc['archetype_summary']}\nExtras: {npc['archetype_extras_summary'][:50]}...\n"
            f"Expression: {npc['expression']}\nTraits: {', '.join(npc['personality_traits'])}\n"
            f"Seed: {npc['visual_seed']}\nLocation: {npc['current_location']}\n"
            f"Social Link: Type: {social_link['link_type'] or 'none'}, Level: {social_link['link_level']}, "
            f"Dynamics: {json.dumps(social_link['dynamics'])}\n"
            f"Previous Images: {', '.join(npc['previous_images'][:2]) or 'None'}"
            f"\nUser Rating: {scene_data['user_preferences'].get(npc['name'], {}).get('avg_rating', 'N/A')}"
        )
    npc_details_text = "\n\n".join(npc_details)

    # Scene context with PlayerJournal
    scene_context = (
        f"Setting: {scene_data['setting']}\n"
        f"Actions: {', '.join(scene_data['actions'])}\n"
        f"NPC Positions: {json.dumps(scene_data['npc_positions'])}\n"
        f"Mood: {scene_data['mood']}\n"
        f"Player Stats: Lust {scene_data['player_stats']['lust']}, Dependency {scene_data['player_stats']['dependency']}\n"
        f"Addictions: {json.dumps(scene_data['addiction_status']['addiction_levels'])}\n"
        f"Recent Journal: {'; '.join([f'{entry['type']}: {entry['text'][:50]}...' for entry in scene_data['journal_entries']])}"
    )

    prompt = f"""
Given NPC details and scene context from a femdom visual novel, generate an image-optimized prompt for Stability AI:

=== NPC DETAILS ===
{npc_details_text}

=== SCENE CONTEXT ===
{scene_context}

YOUR TASK:
Create a concise (max 150 words), anime-style CG prompt:
1. Ensure NPC consistency using visual_seed and previous images.
2. Highlight traits from physical_description/visual_attributes (e.g., 'raven-black curls', 'Hyper-Silk gown').
3. Reflect actions (e.g., 'pegging with an 8-inch black dildo') and mood.
4. Scale eroticism with Lust/addictions (tier {min(4, scene_data['player_stats']['lust'] // 20)} if Lust > 60).
5. Use high Dominance/Intensity for commanding poses; SocialLinks for group dynamics (e.g., alliances = close poses).
6. Favor high-rated traits ({json.dumps(scene_data['user_preferences'])}).
7. Incorporate setting and journal hints (e.g., past scenes).
8. Vivid, sensual for NSFW; soft, atmospheric otherwise.

Return JSON with 'image_prompt' and 'negative_prompt' (e.g., 'low quality, blurry')."""
    
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    data = safe_json_loads(response.choices[0].message.content)
    return data if data and "image_prompt" in data else {
        "image_prompt": "Generic anime-style CG of a scene with unknown characters.",
        "negative_prompt": "low quality, blurry, distorted face, extra limbs"
    }

# ================================
# 5️⃣ TRACK VISUAL EVOLUTION
# ================================
def track_visual_evolution(npc_id, user_id, conversation_id, event_type, description, previous, current, scene, image_path):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO NPCVisualEvolution
        (npc_id, user_id, conversation_id, event_type, event_description, 
         previous_state, current_state, scene_context, image_generated)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (npc_id, user_id, conversation_id, event_type, description,
          json.dumps(previous), json.dumps(current), scene, image_path))
    conn.commit()
    cursor.close()
    conn.close()
    return True

# ================================
# 6️⃣ IMAGE CACHING SYSTEM
# ================================
def get_cached_images(prompt):
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cached_files = []
    cache_path = os.path.join(CACHE_DIR, prompt_hash)
    if os.path.exists(cache_path):
        for file in os.listdir(cache_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                cached_files.append(os.path.join(cache_path, file))
    return cached_files

def save_image_to_cache(image_data, prompt, variation_id=0):
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, prompt_hash)
    os.makedirs(cache_path, exist_ok=True)
    if image_data.startswith('data:image') or ';base64,' in image_data:
        base64_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(base64_data)
        file_path = os.path.join(cache_path, f"variation_{variation_id}.png")
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        return file_path
    elif image_data.startswith(('http://', 'https://')):
        response = requests.get(image_data, stream=True)
        if response.status_code == 200:
            file_name = f"variation_{variation_id}.png"
            file_path = os.path.join(cache_path, file_name)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return file_path
    return None

# ================================
# 7️⃣ GENERATE AI IMAGE
# ================================
def generate_ai_image(prompt, negative_prompt=None, seed=None, style="anime"):
    if not negative_prompt:
        negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limbs, blurry, watermark"
    style_presets = {"anime": "anime", "realistic": "photographic", "painting": "digital-art", "sketch": "line-art"}
    style_preset = style_presets.get(style, "anime")
    
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "text_prompts": [
            {"text": prompt, "weight": 1.0},
            {"text": negative_prompt, "weight": -1.0}
        ],
        "cfg_scale": 7.0,
        "height": 768,
        "width": 512,
        "steps": 30,
        "seed": seed,  # Now NPC-specific
        "style_preset": style_preset
    }
    try:
        response = requests.post(STABILITY_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if "artifacts" in data and len(data["artifacts"]) > 0:
            return f"data:image/png;base64,{data['artifacts'][0]['base64']}"
        logger.error(f"No artifacts in Stability API response: {data}")
    except Exception as e:
        logger.error(f"Error generating image: {e}")
    return None

# ================================
# 8️⃣ GENERATE ROLEPLAY IMAGE
# ================================
def generate_roleplay_image_from_gpt(gpt_response, user_id, conversation_id):
    scene_data = process_gpt_scene_data(gpt_response, user_id, conversation_id)
    if not scene_data:
        return {"error": "No valid scene data found in GPT response"}

    prompt_data = generate_image_prompt(scene_data)
    optimized_prompt = prompt_data["image_prompt"]
    negative_prompt = prompt_data.get("negative_prompt", "")
    
    cached_images = get_cached_images(optimized_prompt)
    if cached_images:
        return {"image_urls": cached_images, "cached": True, "prompt_used": optimized_prompt}

    generated_images = []
    updated_visual_attrs = {}
    
    for variation_id in range(3):  # 3 variations for angles/lighting
        seed = int(hashlib.md5(scene_data["npcs"][0]["visual_seed"].encode()).hexdigest(), 16) % (2**32) if scene_data["npcs"] else None
        image_data = generate_ai_image(optimized_prompt, negative_prompt, seed)
        if image_data:
            cached_path = save_image_to_cache(image_data, optimized_prompt, variation_id)
            generated_images.append(cached_path or image_data)
            
            # Update visual attributes for each NPC
            for npc in scene_data["npcs"]:
                if "id" in npc and npc["id"]:
                    new_attrs, current_state = update_npc_visual_attributes(
                        user_id, conversation_id, npc["id"], optimized_prompt, cached_path
                    )
                    if any(current_state.get(k) != new_attrs.get(k) for k in new_attrs if new_attrs.get(k)):
                        track_visual_evolution(
                            npc["id"], user_id, conversation_id, "appearance_change",
                            f"Updated visual appearance in scene", current_state, new_attrs,
                            optimized_prompt, cached_path
                        )
                    updated_visual_attrs[npc["name"]] = new_attrs

    return {
        "image_urls": generated_images,
        "cached": False,
        "prompt_used": optimized_prompt,
        "negative_prompt": negative_prompt,
        "updated_visual_attrs": updated_visual_attrs
    }

# ================================
# 9️⃣ API ROUTES
# ================================
image_bp = Blueprint("image_bp", __name__)

@image_bp.route("/generate_gpt_image", methods=["POST"])
def generate_gpt_image():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    gpt_response = data.get("gpt_response")
    conversation_id = data.get("conversation_id")
    if not gpt_response or not conversation_id:
        return jsonify({"error": "Missing GPT response or conversation_id"}), 400
    result = generate_roleplay_image_from_gpt(gpt_response, user_id, conversation_id)
    return jsonify(result)

# [Unchanged routes: /image_feedback, /npc_images/<npc_id>]

# ================================
# 10️⃣ INITIALIZATION
# ================================
def init_app(app):
    with app.app_context():
        setup_database()
    app.register_blueprint(image_bp, url_prefix="/api/image")
    logger.info("AI Image Generator initialized")
