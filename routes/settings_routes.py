# routes/settings_routes.py
import os
import json
import logging
import random
from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection

settings_bp = Blueprint('settings_bp', __name__)

def insert_missing_settings():
    logging.info("[insert_missing_settings] Starting...")

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

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM Settings")
    existing = {row[0] for row in cursor.fetchall()}

    inserted_count = 0
    for s in settings_data:
        sname = s["name"]
        if sname not in existing:
            ef = json.dumps(s["enhanced_features"])
            sm = json.dumps(s["stat_modifiers"])
            ae = json.dumps(s["activity_examples"])

            cursor.execute("""
                INSERT INTO Settings (name, mood_tone, enhanced_features, stat_modifiers, activity_examples)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                sname,
                s["mood_tone"],
                ef,
                sm,
                ae
            ))
            logging.info(f"Inserted new setting: {sname}")
            inserted_count += 1
        else:
            logging.debug(f"Skipped existing setting: {sname}")

    conn.commit()
    conn.close()
    logging.info(f"[insert_missing_settings] Done. Inserted {inserted_count} new settings.")


@settings_bp.route('/insert_settings', methods=['POST'])
def insert_settings_route():
    try:
        insert_missing_settings()
        return jsonify({"message": "Settings inserted/updated successfully"}), 200
    except Exception as e:
        logging.exception("Error inserting settings.")
        return jsonify({"error": str(e)}), 500


def generate_mega_setting_logic():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, mood_tone, enhanced_features, stat_modifiers, activity_examples
        FROM Settings
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "error": "No settings found in DB.",
            "mega_name": "Empty Settings Table",
            "mega_description": "No environment generated",
            "enhanced_features": [],
            "stat_modifiers": {},
            "activity_examples": []
        }

    all_settings = []
    for row_id, row_name, row_mood, row_ef, row_sm, row_ae in rows:
        # Ensure proper type conversion
        ef_list = row_ef if isinstance(row_ef, list) else json.loads(row_ef)
        sm_dict = row_sm if isinstance(row_sm, dict) else json.loads(row_sm)
        ae_list = row_ae if isinstance(row_ae, list) else json.loads(row_ae)
        all_settings.append({
            "id": row_id,
            "name": row_name,
            "mood_tone": row_mood,
            "enhanced_features": ef_list,
            "stat_modifiers": sm_dict,
            "activity_examples": ae_list
        })

    # Choose a random number of settings (3-5) to blend.
    num_settings = random.choice([3, 4, 5])
    selected = random.sample(all_settings, min(num_settings, len(all_settings)))
    picked_names = [s["name"] for s in selected]

    # Create a preliminary mega name by joining the names.
    mega_name = " + ".join(picked_names)

    # Build a fusion prompt that instructs GPT to blend these settings into one cohesive narrative.
    fusion_prompt = (
        "You are a creative writer tasked with blending together the following settings into a single cohesive world description. "
        "Each setting is described by its name, mood tone, enhanced features, stat modifiers, and activity examples. "
        "Embrace contradictions, get creative, and understand that the implementation of these doesn't need to be 'literal.' (eg., 'prison' can be a metaphorical prison)"
        "The final description should be a unified, evocative narrative that seamlessly fuses these diverse elements into one world.\n\n"
        "Settings:\n"
    )
    for s in selected:
        # For clarity, join the lists; for stat modifiers, you can format them as key: value pairs.
        ef = ", ".join(s["enhanced_features"]) if s["enhanced_features"] else "None"
        ae = ", ".join(s["activity_examples"]) if s["activity_examples"] else "None"
        sm = ", ".join([f"{k}: {v}" for k, v in s["stat_modifiers"].items()]) if s["stat_modifiers"] else "None"
        fusion_prompt += (
            f"- {s['name']}: Mood tone: {s['mood_tone']}; "
            f"Enhanced features: {ef}; Stat modifiers: {sm}; "
            f"Activity examples: {ae}\n"
        )
    fusion_prompt += (
        "\nPlease produce a single, creative, and cohesive description (3-5 sentences) that blends these settings together into one unified environment. "
        "Output only the final description text without any extra commentary or formatting."
    )
    
    # Log the fusion prompt for debugging.
    logging.info("Fusion prompt for mega setting: %s", fusion_prompt)

    # Call GPT to generate the unified environment description.
    gpt_client = get_openai_client()
    messages = [{"role": "system", "content": fusion_prompt}]
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o",
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
        "mega_name": mega_name,
        "mega_description": unified_description,
        "enhanced_features": combined_enhanced_features,
        "stat_modifiers": combined_stat_modifiers,
        "activity_examples": combined_activity_examples,
        "message": "Mega setting generated successfully."
    }


@settings_bp.route('/generate_mega_setting', methods=['POST'])
def generate_mega_setting_route():
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "No conversation_id provided"}), 400

        result = generate_mega_setting_logic()  # ignoring user_id, conv_id if truly global
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result), 200
    except Exception as e:
        logging.exception("Error generating mega setting.")
        return jsonify({"error": str(e)}), 500
