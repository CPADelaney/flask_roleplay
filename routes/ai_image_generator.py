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
from db.connection import get_db_connection_context
from quart import Blueprint, request, jsonify, session, current_app
from logic.addiction_system_sdk import check_addiction_status
from logic.chatgpt_integration import get_openai_client, safe_json_loads
from dotenv import load_dotenv
from lore.core.lore_system import LoreSystem
from lore.core import canon

# Import the new dynamic relationships system
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    RelationshipState,
    RelationshipDimensions
)

# Load environment variables
load_dotenv()

# API KEYS
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SINKIN_ACCESS_TOKEN = os.environ.get("SD_Access")  # from your GitHub secret

# API URLs
SINKIN_API_URL = "https://sinkin.ai/api/inference"
GPT4O_API_URL = "https://api.openai.com/v1/chat/completions"

# Image Caching Directory
CACHE_DIR = "static/images"
os.makedirs(CACHE_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================
# 1️⃣ FETCH ROLEPLAY DATA & NPC DETAILS
# ======================================================
async def get_npc_and_roleplay_context(user_id, conversation_id, npc_names, player_name="Chase"):
    """
    Fetch detailed NPCStats, NPCVisualAttributes, PlayerStats, Dynamic Relationships, and PlayerJournal.
    Safely handles empty npc_names to avoid an 'IN ()' syntax error and omits `visual_seed`
    if it's not actually a column in the NPCStats table.
    """
    # 1. If npc_names is empty, immediately return empty results
    if not npc_names:
        # Provide some default player stats in case there's no NPC yet
        default_player_stats = {
            "obedience": 50,
            "corruption": 0,
            "willpower": 50,
            "shame": 0,
            "dependency": 0,
            "lust": 0,
            "mental_resilience": 50
        }
        return {}, default_player_stats, {}, []

    # Initialize relationship manager for dynamic relationships
    rel_manager = OptimizedRelationshipManager(user_id, conversation_id)

    async with get_db_connection_context() as conn:
        # 2. Fetch NPC data
        # Omit the visual_seed column if it doesn't exist in your table:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT n.id, 
                       n.npc_name,
                       n.physical_description, 
                       n.dominance, 
                       n.cruelty, 
                       n.intensity,
                       n.archetype_summary, 
                       n.archetype_extras_summary, 
                       n.personality_traits, 
                       n.likes,
                       n.dislikes, 
                       n.current_location
                FROM NPCStats n
                WHERE n.user_id=%s 
                  AND n.conversation_id=%s 
                  AND n.npc_name IN %s
            """, (user_id, conversation_id, tuple(npc_names)))

            npc_rows = await cursor.fetchall()
            detailed_npcs = {}

            for row in npc_rows:
                npc_id = row[0]
                npc_name = row[1]

                # 3. Fetch Visual Attributes
                await cursor.execute("""
                    SELECT hair_color, 
                           hair_style, 
                           eye_color, 
                           skin_tone, 
                           body_type, 
                           height, 
                           age_appearance, 
                           default_outfit, 
                           outfit_variations,
                           makeup_style, 
                           accessories, 
                           expressions, 
                           poses, 
                           visual_seed,
                           last_generated_image
                    FROM NPCVisualAttributes
                    WHERE npc_id=%s 
                      AND user_id=%s 
                      AND conversation_id=%s
                """, (npc_id, user_id, conversation_id))
                visual_attrs = await cursor.fetchone()

                # 4. Fetch Previous Images
                await cursor.execute("""
                    SELECT image_generated
                    FROM NPCVisualEvolution
                    WHERE npc_id=%s 
                      AND user_id=%s 
                      AND conversation_id=%s
                    ORDER BY timestamp DESC
                    LIMIT 5
                """, (npc_id, user_id, conversation_id))
                visual_evol_rows = await cursor.fetchall()
                previous_images = [img[0] for img in visual_evol_rows if img[0]]

                # 5. Fetch Dynamic Relationship for this NPC with player
                # Get player ID (assuming player_id = 1 or fetch from PlayerStats)
                await cursor.execute("""
                    SELECT id FROM PlayerStats 
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                    LIMIT 1
                """, (user_id, conversation_id, player_name))
                player_row = await cursor.fetchone()
                player_id = player_row[0] if player_row else 1

                # Get relationship state using the new system
                rel_state = await rel_manager.get_relationship_state(
                    entity1_type="player",
                    entity1_id=player_id,
                    entity2_type="npc",
                    entity2_id=npc_id
                )

                # 6. Build the NPC dictionary
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
                    "previous_images": previous_images,
                    "relationship": {
                        "dimensions": rel_state.dimensions.to_dict(),
                        "patterns": list(rel_state.history.active_patterns),
                        "archetypes": list(rel_state.active_archetypes),
                        "momentum": rel_state.momentum.get_magnitude(),
                        "duration_days": rel_state.get_duration_days()
                    }
                }

                # Merge Visual Attributes if present
                if visual_attrs:
                    # notice we keep visual_seed from NPCVisualAttributes, if that's valid
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
                        "visual_seed": visual_attrs[13],  # Or remove if not needed
                        "last_generated_image": visual_attrs[14]
                    })

        # 7. Fetch PlayerStats
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT obedience, corruption, willpower, shame, dependency, lust, mental_resilience
                FROM PlayerStats 
                WHERE user_id=%s 
                  AND conversation_id=%s 
                  AND player_name=%s
                LIMIT 1
            """, (user_id, conversation_id, player_name))
            player_row = await cursor.fetchone()
            
        if player_row:
            player_stats = dict(zip(
                ["obedience", "corruption", "willpower", "shame", "dependency", "lust", "mental_resilience"],
                player_row
            ))
        else:
            # Default stats if none found
            player_stats = {
                "obedience": 50,
                "corruption": 0,
                "willpower": 50,
                "shame": 0,
                "dependency": 0,
                "lust": 0,
                "mental_resilience": 50
            }

        # 8. Fetch recent PlayerJournal entries
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT entry_text, entry_type
                FROM PlayerJournal
                WHERE user_id=%s 
                  AND conversation_id=%s
                ORDER BY timestamp DESC
                LIMIT 3
            """, (user_id, conversation_id))
            journal_rows = await cursor.fetchall()
            
        journal_entries = [
            {"text": row[0], "type": row[1]}
            for row in journal_rows
        ]

        # 9. Fetch UserVisualPreferences
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT npc_name, avg_rating
                FROM UserVisualPreferences
                WHERE user_id=%s 
                  AND npc_name IN %s
            """, (user_id, tuple(npc_names)))
            pref_rows = await cursor.fetchall()
            
        user_preferences = {
            row[0]: {"avg_rating": row[1]}
            for row in pref_rows
        }

    return detailed_npcs, player_stats, user_preferences, journal_entries


# ======================================================
# 2️⃣ PROCESS GPT RESPONSE & MERGE WITH NPCStats
# ======================================================
async def process_gpt_scene_data(gpt_response, user_id, conversation_id):
    """Extract scene data from GPT, enriched with NPCStats, PlayerStats, Dynamic Relationships, and PlayerJournal."""
    if not gpt_response or "scene_data" not in gpt_response:
        return None

    scene_data = gpt_response["scene_data"]
    npc_names = scene_data["npc_names"]
    detailed_npcs, player_stats, user_preferences, journal_entries = await get_npc_and_roleplay_context(
        user_id, conversation_id, npc_names
    )
    addiction_status = await check_addiction_status(user_id, conversation_id, "Chase")

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
            "relationship": npc.get("relationship", {})
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


# ======================================================
# 3️⃣ UPDATE VISUAL ATTRIBUTES VIA gpt-5-nano
# ======================================================
async def update_npc_visual_attributes(user_id, conversation_id, npc_id, prompt_data, image_path=None):
    """Use gpt-5-nano to extract and update visual attributes from the image prompt."""
    
    # Create context
    class ImageContext:
        def __init__(self, user_id, conversation_id):
            self.user_id = user_id
            self.conversation_id = conversation_id
    
    ctx = ImageContext(user_id, conversation_id)
    
    async with get_db_connection_context() as conn:
        # Fetch current attributes (read is fine)
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT hair_color, hair_style, eye_color, skin_tone, body_type, 
                       height, age_appearance, default_outfit, makeup_style, accessories
                FROM NPCVisualAttributes
                WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
            """, (npc_id, user_id, conversation_id))
            current_attrs = await cursor.fetchone()
            
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

        # gpt-5-nano extraction
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
            model="gpt-5-nano",
            messages=[{"role": "system", "content": gpt_prompt}],
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        lore_system = await LoreSystem.get_instance(user_id, conversation_id)
        new_attrs = safe_json_loads(response.choices[0].message.content)

        # Update visual attributes through LoreSystem
        if current_attrs:
            # Update existing
            result = await lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="NPCVisualAttributes",
                entity_identifier={"npc_id": npc_id},
                updates={
                    "hair_color": new_attrs["hair_color"],
                    "hair_style": new_attrs["hair_style"],
                    "eye_color": new_attrs["eye_color"],
                    "skin_tone": new_attrs["skin_tone"],
                    "body_type": new_attrs["body_type"],
                    "height": new_attrs["height"],
                    "age_appearance": new_attrs["age_appearance"],
                    "default_outfit": new_attrs["default_outfit"],
                    "makeup_style": new_attrs["makeup_style"],
                    "accessories": json.dumps(new_attrs["accessories"]),
                    "last_generated_image": image_path,
                    "updated_at": "CURRENT_TIMESTAMP"
                },
                reason=f"Visual attributes updated from generated image"
            )
        else:
            # For new records, we might need a canon function or direct insert
            # Since NPCVisualAttributes might not be a "core" lore table
            # Keep the insert for now or create a canon function
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO NPCVisualAttributes (
                        npc_id, user_id, conversation_id, hair_color, hair_style, eye_color,
                        skin_tone, body_type, height, age_appearance, default_outfit, makeup_style,
                        accessories, visual_seed, last_generated_image
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        npc_id,
                        user_id,
                        conversation_id,
                        new_attrs["hair_color"],
                        new_attrs["hair_style"],
                        new_attrs["eye_color"],
                        new_attrs["skin_tone"],
                        new_attrs["body_type"],
                        new_attrs["height"],
                        new_attrs["age_appearance"],
                        new_attrs["default_outfit"],
                        new_attrs["makeup_style"],
                        json.dumps(new_attrs["accessories"]),
                        hashlib.md5(f"{npc_id}".encode()).hexdigest(),
                        image_path,
                    ),
                )

            await canon.log_canonical_event(
                ctx,
                conn,
                f"Initialized visual attributes for NPC {npc_id}",
                tags=["visual", "creation"],
                significance=5,
            )
    
    return new_attrs, current_state


# ======================================================
# 4️⃣ GENERATE IMAGE-OPTIMIZED PROMPT VIA gpt-5-nano
# ======================================================
def generate_image_prompt(scene_data):
    """Use gpt-5-nano to summarize NPCStats, scene context, Dynamic Relationships, and PlayerJournal into an image-optimized prompt."""
    if not scene_data:
        return None

    # NPC details with visual attributes and Dynamic Relationships
    npc_details = []
    for npc in scene_data["npcs"]:
        visual_attrs = [
            f"Hair: {npc.get('hair_color', 'unknown')} {npc.get('hair_style', '')}",
            f"Eyes: {npc.get('eye_color', 'unknown')}",
            f"Skin: {npc.get('skin_tone', 'unknown')}",
            f"Body: {npc.get('body_type', 'unknown')}",
            f"Outfit: {npc.get('default_outfit', 'unknown')}"
        ]
        relationship = npc["relationship"]
        dims = relationship.get("dimensions", {})
        
        # Format relationship info
        rel_summary = []
        if dims.get("trust", 0) > 70:
            rel_summary.append("deeply trusted")
        elif dims.get("trust", 0) < -30:
            rel_summary.append("distrusted")
        
        if dims.get("affection", 0) > 70:
            rel_summary.append("beloved")
        elif dims.get("affection", 0) < -30:
            rel_summary.append("despised")
        
        if dims.get("influence", 0) > 50:
            rel_summary.append("dominant over player")
        elif dims.get("influence", 0) < -50:
            rel_summary.append("submissive to player")
        
        patterns = relationship.get("patterns", [])
        archetypes = relationship.get("archetypes", [])
        
        npc_details.append(
            f"Name: {npc['name']}\n"
            f"Physical Description: {npc['physical_description']}\n"
            f"Visual Attributes: {', '.join(visual_attrs)}\n"
            f"Dominance: {npc['dominance']}\n"
            f"Cruelty: {npc['cruelty']}\n"
            f"Intensity: {npc['intensity']}\n"
            f"Archetype: {npc['archetype_summary']}\n"
            f"Extras: {npc['archetype_extras_summary'][:50]}...\n"
            f"Expression: {npc['expression']}\n"
            f"Traits: {', '.join(npc['personality_traits'])}\n"
            f"Seed: {npc['visual_seed']}\n"
            f"Location: {npc['current_location']}\n"
            f"Relationship: {', '.join(rel_summary) or 'neutral'}\n"
            f"Patterns: {', '.join(patterns) or 'none'}\n"
            f"Archetypes: {', '.join(archetypes) or 'none'}\n"
            f"Trust: {dims.get('trust', 0)}, Affection: {dims.get('affection', 0)}, "
            f"Influence: {dims.get('influence', 0)}, Intimacy: {dims.get('intimacy', 0)}\n"
            f"Previous Images: {', '.join(npc['previous_images'][:2]) or 'None'}\n"
            f"User Rating: {scene_data['user_preferences'].get(npc['name'], {}).get('avg_rating', 'N/A')}"
        )
    npc_details_text = "\n\n".join(npc_details)

    # Scene context with PlayerJournal
    scene_context = (
        f"Setting: {scene_data['setting']}\n"
        f"Actions: {', '.join(scene_data['actions'])}\n"
        f"NPC Positions: {json.dumps(scene_data['npc_positions'])}\n"
        f"Mood: {scene_data['mood']}\n"
        f"Player Stats: Lust {scene_data['player_stats']['lust']}, "
        f"Dependency {scene_data['player_stats']['dependency']}\n"
        f"Addictions: {json.dumps(scene_data['addiction_status']['addiction_levels'])}\n"
        "Recent Journal: "
        + "; ".join([
            f"{entry['type']}: {entry['text'][:50]}..."
            for entry in scene_data['journal_entries']
        ])
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
5. Use high Dominance/Intensity for commanding poses; Relationship dynamics for positioning:
   - High trust/affection = intimate positioning, close contact
   - High influence = NPC in dominant position over player
   - Negative affection = distant or confrontational poses
   - "Toxic_bond" archetype = intense but unstable positioning
6. Favor high-rated traits ({json.dumps(scene_data['user_preferences'])}).
7. Incorporate setting and journal hints (e.g., past scenes).
8. Vivid, sensual for NSFW; soft, atmospheric otherwise.
9. Ensure the player is featured (generic, nondescript, faceless brown-haired male, like how VNs tend to do it)
10. Consider relationship patterns: 
    - "push_pull" = mixed signals in positioning
    - "explosive_chemistry" = intense physical closeness
    - "growing_distance" = physical separation in scene

Return JSON with 'image_prompt' and 'negative_prompt' (e.g., 'low quality, blurry')."""
    
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    data = safe_json_loads(response.choices[0].message.content)
    return data if data and "image_prompt" in data else {
        "image_prompt": "Generic anime-style CG of a scene with unknown characters.",
        "negative_prompt": "low quality, blurry, distorted face, extra limbs"
    }


# ======================================================
# 5️⃣ TRACK VISUAL EVOLUTION
# ======================================================
async def track_visual_evolution(npc_id, user_id, conversation_id, event_type, description, previous, current, scene, image_path):
    # Create context
    class EvolutionContext:
        def __init__(self, user_id, conversation_id):
            self.user_id = user_id
            self.conversation_id = conversation_id
    
    ctx = EvolutionContext(user_id, conversation_id)
    
    async with get_db_connection_context() as conn:
        # Log this as a canonical event
        await canon.log_canonical_event(
            ctx, conn,
            f"NPC {npc_id} visual evolution: {event_type} - {description}",
            tags=["visual", "evolution", event_type],
            significance=5
        )
        
        # The actual NPCVisualEvolution table insert can remain if it's not a core table
        async with conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO NPCVisualEvolution
                (npc_id, user_id, conversation_id, event_type, event_description, 
                 previous_state, current_state, scene_context, image_generated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                npc_id, user_id, conversation_id, event_type, description,
                json.dumps(previous), json.dumps(current), scene, image_path
            ))
    
    return True


# ======================================================
# 6️⃣ IMAGE CACHING SYSTEM
# ======================================================
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


# ======================================================
# 7️⃣ GENERATE AI IMAGE
# ======================================================
def generate_ai_image(
    prompt,
    negative_prompt="low quality, blurry, distorted face, bad anatomy, watermark",
    model_id="nGyN44N", 
    width=512,
    height=768,
    steps=30,
    scale=7.5,
    seed=-1,
    num_images=1,
    scheduler="DPMSolverMultistep",
    use_default_neg="true",
    lcm="false"
):
    """
    Calls SinkIn's text2img using the given prompt and optional parameters.
    Returns a list of image URLs or None on failure.
    """

    # Prepare the data payload (multipart/form-data).
    # Note: We do NOT need to pass JSON as the body; we'll rely on 'data' and 'files'.
    payload = {
        "access_token": SINKIN_ACCESS_TOKEN,
        "model_id": model_id,
        "prompt": prompt,
        "width": str(width),             # must be strings if passing via 'data'
        "height": str(height),
        "steps": str(steps),
        "scale": str(scale),
        "num_images": str(num_images),
        "scheduler": scheduler,
        "seed": str(seed),
        "negative_prompt": negative_prompt,
        "use_default_neg": use_default_neg,
        "lcm": lcm,
    }

    # If you do NOT need img2img, you won't pass any files.
    # If you *do* want img2img, see below for an example.

    try:
        response = requests.post(
            SINKIN_API_URL, 
            data=payload,
            # no files here for text2img
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error_code") == 0:
            # Return the image URLs
            return data.get("images", [])
        else:
            logger.error(f"SinkIn error: {data.get('message')}")
    except Exception as e:
        logger.error(f"SinkIn request failed: {e}")

    return None

def generate_nyx_fallback_image():
    """
    Returns a stable fallback image prompt for Nyx,
    referencing her 'cruel goth mommy domme' persona.
    """
    # A simple prompt referencing the system description. 
    # Tailor this to your style/NSFW needs.
    fallback_prompt = (
        "A dark gothic chamber where Nyx, a cruel goth mommy domme, stands "
        "radiating sadistic charm and ironclad confidence. Pale skin, black lipstick, "
        "tattoos, piercings, high-heeled boots, exuding an aura of twisted power. "
        "Shot in a stylized anime aesthetic, high contrast and sultry shadows."
    )
    negative_prompt = (
        "low quality, blurry, disfigured, poorly drawn, bad anatomy, extra limbs, watermark"
    )
    return {
        "image_prompt": fallback_prompt,
        "negative_prompt": negative_prompt
    }



# ======================================================
# 8️⃣ GENERATE ROLEPLAY IMAGE
# ======================================================
async def generate_roleplay_image_from_gpt(gpt_response, user_id, conversation_id):
    scene_data = await process_gpt_scene_data(gpt_response, user_id, conversation_id)
    
    # If there's no scene_data or no NPCs, fallback to generating an image of Nyx
    if not scene_data or not scene_data.get("npcs"):
        logger.warning("No NPCs found—using Nyx fallback image prompt.")
        
        # fallback block
        logger.warning("No NPCs found—using Nyx fallback image prompt.")
        
        fallback_data = generate_nyx_fallback_image()
        fallback_prompt = fallback_data["image_prompt"]
        fallback_negative = fallback_data["negative_prompt"]
        
        cached_images = get_cached_images(fallback_prompt)
        if cached_images:
            return {
                "image_urls": cached_images,
                "cached": True,
                "prompt_used": fallback_prompt
            }
        
        # Call generate_ai_image -> returns a list of URLs or None
        fallback_urls = generate_ai_image(fallback_prompt, fallback_negative, num_images=1)
        
        if not fallback_urls or len(fallback_urls) == 0:
            return {"error": "Failed to generate fallback Nyx image"}
        
        # Save each URL
        saved_image_paths = []
        for idx, url in enumerate(fallback_urls):
            path = save_image_to_cache(url, fallback_prompt, idx)
            saved_image_paths.append(path or url)
        
        return {
            "image_urls": saved_image_paths,
            "cached": False,
            "prompt_used": fallback_prompt,
            "negative_prompt": fallback_negative,
            "updated_visual_attrs": {}
        }

    # Otherwise, we proceed with your usual logic
    prompt_data = generate_image_prompt(scene_data)
    optimized_prompt = prompt_data["image_prompt"]
    negative_prompt = prompt_data.get("negative_prompt", "")
    
    cached_images = get_cached_images(optimized_prompt)
    if cached_images:
        return {"image_urls": cached_images, "cached": True, "prompt_used": optimized_prompt}
    
    generated_images = []
    updated_visual_attrs = {}
    
    for variation_id in range(3):
        # optional seed calculation
        seed = None
        if scene_data["npcs"]:
            npc_seed_text = scene_data["npcs"][0]["visual_seed"]
            seed = int(hashlib.md5(npc_seed_text.encode()).hexdigest(), 16) % (2**32)
    
        # generate_ai_image will return a LIST of URLs
        image_urls = generate_ai_image(
            optimized_prompt, 
            negative_prompt=negative_prompt, 
            seed=seed, 
            num_images=1  # or 3 if you want 3 at once
        )
    
        if not image_urls:
            continue  # skip if no images returned
    
        # For each returned URL (likely just 1 if num_images=1)
        for single_url in image_urls:
            cached_path = save_image_to_cache(single_url, optimized_prompt, variation_id)
            generated_images.append(cached_path or single_url)
    
            # Now update NPC attributes for each NPC
            for npc in scene_data["npcs"]:
                npc_id = npc.get("id")
                if npc_id:
                    new_attrs, current_state = await update_npc_visual_attributes(
                        user_id, conversation_id, npc_id, 
                        prompt_data=optimized_prompt, 
                        image_path=cached_path
                    )
    
                    # track changes
                    if any(
                        current_state.get(k) != new_attrs.get(k)
                        for k in new_attrs
                    ):
                        await track_visual_evolution(
                            npc_id, user_id, conversation_id,
                            "appearance_change",
                            "Updated visual appearance in scene",
                            previous=current_state,
                            current=new_attrs,
                            scene=optimized_prompt,
                            image_path=cached_path
                        )
                    updated_visual_attrs[npc["name"]] = new_attrs
    
    return {
        "image_urls": generated_images,
        "cached": False,
        "prompt_used": optimized_prompt,
        "negative_prompt": negative_prompt,
        "updated_visual_attrs": updated_visual_attrs
    }

# ======================================================
# 9️⃣ API ROUTES
# ======================================================
image_bp = Blueprint("image_bp", __name__)

@image_bp.route("/generate_gpt_image", methods=["POST"])
async def generate_gpt_image():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    data = await request.get_json()
    gpt_response = data.get("gpt_response")
    conversation_id = data.get("conversation_id")
    if not gpt_response or not conversation_id:
        return jsonify({"error": "Missing GPT response or conversation_id"}), 400
    result = await generate_roleplay_image_from_gpt(gpt_response, user_id, conversation_id)
    return jsonify(result)


# ======================================================
# 10️⃣ INITIALIZATION
# ======================================================
def init_app(app):
    """
    Initialize this blueprint. 
    The main table creation now happens elsewhere, so we do not create or alter tables here.
    """
    app.register_blueprint(image_bp, url_prefix="/api/image")
    logger.info("AI Image Generator initialized")
