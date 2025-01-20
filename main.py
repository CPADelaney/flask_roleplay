import random
from flask import Flask, request, jsonify
import psycopg2
import os

app = Flask(__name__)

def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise EnvironmentError("DATABASE_URL is not set.")
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    return conn

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Existing table creation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Settings (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            mood_tone TEXT NOT NULL,
            enhanced_features JSONB NOT NULL,
            stat_modifiers JSONB NOT NULL,
            activity_examples JSONB NOT NULL
        );
    ''')
    # New or existing table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CurrentRoleplay (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()

@app.route('/test_db_connection', methods=['GET'])
def test_db_connection():
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({"message": "Connected to the database successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_mega_setting', methods=['POST'])
def generate_mega_setting():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, name, mood_tone, enhanced_features, stat_modifiers, activity_examples 
        FROM Settings
    ''')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return jsonify({"error": "No settings found in the database."}), 500

    num_settings = random.choice([3,4,5])
    selected = random.sample(rows, min(num_settings, len(rows)))

    # Merge name, mood tone
    mega_name = " + ".join([s[1] for s in selected])
    mood_tones = [s[2] for s in selected]
    mega_description = (
        f"The settings intertwine: {', '.join(mood_tones[:-1])}, and finally, {mood_tones[-1]}. "
        "Together, they form a grand vision, unexpected and brilliant."
    )

    # Merge JSON data
    combined_enhanced_features = []
    combined_stat_modifiers = {}
    combined_activity_examples = []

    for s in selected:
        ef = s[3]  # enhanced_features
        sm = s[4]  # stat_modifiers
        ae = s[5]  # activity_examples

        # Extend combined_enhanced_features if it's a list
        combined_enhanced_features.extend(ef)

        # Merge stat_modifiers if it's a dict
        for key, val in sm.items():
            if key not in combined_stat_modifiers:
                combined_stat_modifiers[key] = val
            else:
                combined_stat_modifiers[key] = f"{combined_stat_modifiers[key]}, {val}"

        # Extend combined_activity_examples if it's a list
        combined_activity_examples.extend(ae)

    return jsonify({
        "mega_name": mega_name,
        "mega_description": mega_description,
        "enhanced_features": combined_enhanced_features,
        "stat_modifiers": combined_stat_modifiers,
        "activity_examples": combined_activity_examples,
        "message": "Mega setting generated and stored successfully."
    })

#
# NEW ENDPOINTS HERE
#

@app.route('/get_current_roleplay', methods=['GET'])
def get_current_roleplay():
    """
    Returns an array of {key, value} objects from the CurrentRoleplay table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM CurrentRoleplay")
    rows = cursor.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({"key": r[0], "value": r[1]})
    
    return jsonify(data), 200

@app.route('/store_roleplay_segment', methods=['POST'])
def store_roleplay_segment():
    """
    Expects JSON: {"key": "...", "value": "..."}
    Overwrites any existing entry for that key.
    """
    payload = request.get_json()
    segment_key = payload.get("key")
    segment_value = payload.get("value")

    if not segment_key:
        return jsonify({"error": "Missing 'key'"}), 400
    if segment_value is None:
        return jsonify({"error": "Missing 'value'"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    # delete old entry if exists
    cursor.execute("DELETE FROM CurrentRoleplay WHERE key = %s", (segment_key,))
    # insert the new one
    cursor.execute("INSERT INTO CurrentRoleplay (key, value) VALUES (%s, %s)", (segment_key, segment_value))
    conn.commit()
    conn.close()

    return jsonify({"message": "Stored successfully"}), 200

# Initialize & run (only if running locally—on Railway, it might auto-run via Gunicorn)
initialize_database()
app.run(host='0.0.0.0', port=5000)
