import random
from flask import Flask, request, jsonify
import psycopg2
import os

app = Flask(__name__)

# Connect to PostgreSQL using the DATABASE_URL environment variable
def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")  # Fetch the database URL from Railway
    if not DATABASE_URL:
        raise EnvironmentError("DATABASE_URL is not set in the environment.")
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    return conn

# Create tables if they don't already exist
def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CurrentRoleplay (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()

# Route to test database connection
@app.route('/test_db_connection', methods=['GET'])
def test_db_connection():
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({"message": "Connected to the database successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to generate a mega setting
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
                # If numeric or string, handle how you want to combine them
                combined_stat_modifiers[key] = f"{combined_stat_modifiers[key]}, {val}"

        # Extend combined_activity_examples if it's a list
        combined_activity_examples.extend(ae)

    # Return everything in one response
    return jsonify({
        "mega_name": mega_name,
        "mega_description": mega_description,
        "enhanced_features": combined_enhanced_features,
        "stat_modifiers": combined_stat_modifiers,
        "activity_examples": combined_activity_examples,
        "message": "Mega setting generated and stored successfully."
    })

# Initialize the database on startup
initialize_database()  # Just call it immediately
app.run(host='0.0.0.0', port=5000)
