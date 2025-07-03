# routes/settings_routes.py

import os
import json
import logging
import collections
import random
from quart import Blueprint, request, jsonify, session
from db.connection import get_db_connection_context
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client, build_message_history
from lore.core import canon
from lore.lore_system import LoreSystem

settings_bp = Blueprint('settings_bp', __name__)

def _coerce_mods(raw: dict[str, Any]) -> dict[str, float]:
    """
    Keep only keys that look like stats and values that can become floats.
    Invalid items are silently skipped (but logged once).
    """
    clean: dict[str, float] = {}
    for k, v in (raw or {}).items():
        try:
            clean[k.lower()] = float(v)
        except (TypeError, ValueError):
            logging.warning("Skipping non-numeric stat modifier %s=%s", k, v)
    return clean

async def insert_missing_settings():
    logging.info("[insert_missing_settings] Starting...")

    # Create context for system operations
    class SettingsContext:
        user_id = 0  # System settings
        conversation_id = 0
    
    ctx = SettingsContext()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    settings_json_path = os.path.join(current_dir, "..", "data", "settings_data.json")
    settings_json_path = os.path.normpath(settings_json_path)

    try:
        with open(settings_json_path, "r", encoding="utf-8") as f:
            settings_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Settings file not found at {settings_json_path}!")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {settings_json_path}: {e}")
        return

    async with get_db_connection_context() as conn:
        async with conn.transaction():
            rows = await conn.fetch("SELECT name FROM Settings")
            existing = {row["name"] for row in rows}

            inserted_count = 0
            for s in settings_data:
                sname = s["name"]
                if sname in existing:
                    continue

                ef = json.dumps(s["enhanced_features"], ensure_ascii=False)
                sm = json.dumps(_coerce_mods(s["stat_modifiers"]), ensure_ascii=False)
                ae = json.dumps(s["activity_examples"],   ensure_ascii=False)

                await conn.execute(
                    """
                    INSERT INTO Settings (name, mood_tone, enhanced_features,
                                          stat_modifiers, activity_examples)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    sname, s["mood_tone"], ef, sm, ae
                )
                logging.info("Inserted new setting: %s", sname)
                inserted_count += 1

            if inserted_count > 0:
                # Log the settings update as a canonical event
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Game settings updated: {inserted_count} new settings added",
                    tags=["system", "settings", "configuration"],
                    significance=6
                )

    logging.info(f"[insert_missing_settings] Done. Inserted {inserted_count} new settings.")


@settings_bp.route('/insert_settings', methods=['POST'])
async def insert_settings_route():
    try:
        await insert_missing_settings()
        return jsonify({"message": "Settings inserted/updated successfully"}), 200
    except Exception as e:
        logging.exception("Error inserting settings.")
        return jsonify({"error": str(e)}), 500


async def generate_mega_setting_logic() -> dict[str, Any]:
    async with get_db_connection_context() as conn:
        rows = await conn.fetch("""
            SELECT id, name, mood_tone,
                   enhanced_features, stat_modifiers, activity_examples
            FROM Settings
        """)

    if not rows:
        return {           # unchanged fallback
            "error": "No settings found in DB.",
            "mega_name": "Empty Settings Table",
            "mega_description": "No environment generated",
            "enhanced_features": [],
            "stat_modifiers": {},
            "activity_examples": []
        }

    # ── normalise each row ──────────────────────────────────────────────────
    all_settings: list[dict[str, Any]] = []
    for row in rows:
        ef_list = json.loads(row["enhanced_features"])
        sm_dict = json.loads(row["stat_modifiers"])
        ae_list = json.loads(row["activity_examples"])

        all_settings.append({
            "id": row["id"],
            "name": row["name"],
            "mood_tone": row["mood_tone"],
            "enhanced_features": ef_list,
            "stat_modifiers": {k.lower(): float(v) for k, v in sm_dict.items()},
            "activity_examples": ae_list,
        })

    # ── pick & blend ────────────────────────────────────────────────────────
    num_settings = random.choice([3, 4, 5])
    selected = random.sample(all_settings, min(num_settings, len(all_settings)))
    picked_names = [s["name"] for s in selected]

    # numeric merge  ➜  additive; change to mean/max if you prefer
    combined_modifiers: dict[str, float] = collections.defaultdict(float)
    for s in selected:
        for k, v in s["stat_modifiers"].items():
            combined_modifiers[k] += v

    combined_features: list[str] = []
    combined_activity_examples: list[str] = []
    for s in selected:
        combined_features.extend(s["enhanced_features"])
        combined_activity_examples.extend(s["activity_examples"])

    fusion_prompt = (
        'You are a creative writer tasked with creating a single, immersive world description that blends together a variety of environment elements. '
        'Do not list or name each element separately. Instead, incorporate all of the following details into one cohesive narrative that reads as if it were describing one unified setting. '
        'Focus on the overall atmosphere, mood, and aesthetic, ensuring that the final description feels like a single, organically integrated environment. '
        'Eg., if you have "circus" and "golden age of piracy," the circus can be a boat of performers, an island with a circus, or even a circus run by pirates. Get creative! '
        'Your output should be a single paragraph of 3-5 sentences without bullet points, numbering, or obvious transitions between separate settings.\n\n'
        'Below are the details from the selected settings:\n'
    )
    # For each selected setting, format its details on one line (but the instruction is to blend them)
    for s in selected:
        ef = ", ".join(s["enhanced_features"]) or "None"
        sm = ", ".join(f"{k}: {v:+}" for k, v in s["stat_modifiers"].items()) or "None"
        ae = ", ".join(s["activity_examples"])  or "None"
        # Instead of using bullet points, you could simply separate them with a semicolon:
        fusion_prompt += f"- {s['name']} (Mood: {s['mood_tone']}; Features: {ef}; " \
                         f"Stat modifiers: {sm}; Examples: {ae})\n"
    
    fusion_prompt += (
        "\n\nNow, synthesize all of these details into one unified, creative world description that reads naturally as one setting. "
        "Output only the final descriptive paragraph."
    )
        
    # Log the fusion prompt for debugging.
    logging.info("Fusion prompt for mega setting: %s", fusion_prompt)

    # Call GPT to generate the unified environment description.
    gpt_client = get_openai_client()
    messages = [{"role": "system", "content": fusion_prompt}]
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        unified_description = response.choices[0].message.content.strip()
        logging.info("Unified mega setting description generated: %s", unified_description)
    except Exception as e:
        logging.error("Error generating unified mega setting description: %s", e, exc_info=True)
        unified_description = (
            "The environment is an intricate tapestry woven from diverse settings, "
            "each contributing its own mood, technological marvels, and cultural quirks into one grand, unified world."
        )

    # Also combine additional details for logging or later use.
    combined_enhanced_features = []
    combined_stat_modifiers = {}
    combined_activity_examples = []
    for s_obj in selected:
        combined_enhanced_features.extend(s_obj["enhanced_features"])
        for key, val in s_obj["stat_modifiers"].items():
            if key not in combined_stat_modifiers:
                combined_stat_modifiers[key] = val
            else:
                combined_stat_modifiers[key] = f"{combined_stat_modifiers[key]}, {val}"
        combined_activity_examples.extend(s_obj["activity_examples"])

    return {
        "selected_settings": picked_names,
        "mega_name": " + ".join(picked_names),
        "mega_description": unified_description,
        "enhanced_features": combined_features,
        "stat_modifiers": dict(combined_modifiers),      # ← numbers, not strings
        "activity_examples": combined_activity_examples,
        "message": "Mega setting generated successfully."
    }


@settings_bp.route('/generate_mega_setting', methods=['POST'])
async def generate_mega_setting_route():
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = await request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "No conversation_id provided"}), 400

        # Now calling the async version
        result = await generate_mega_setting_logic()
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result), 200
    except Exception as e:
        logging.exception("Error generating mega setting.")
        return jsonify({"error": str(e)}), 500
