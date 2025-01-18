import random
from flask import Flask, request, jsonify
import psycopg2
import os

app = Flask(__name__)

# Connect to PostgreSQL using the DATABASE_URL environment variable
def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")  # Fetch the database URL from Railway
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
            description TEXT NOT NULL
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

    # Fetch settings from the "Settings" table
    cursor.execute('SELECT name, description FROM settings_alt')
    settings = cursor.fetchall()

    if not settings:
        conn.close()
        return jsonify({"error": "No settings found in the database."}), 500

    # Randomly select 3, 4, or 5 rows
    num_settings = random.choice([3, 4, 5])
    selected_settings = random.sample(settings, min(num_settings, len(settings)))

    # Combine Names and Descriptions
    mega_name = " + ".join([row[0] for row in selected_settings])
    descriptions = [row[1] for row in selected_settings]
    mega_description = (
        f"The settings intertwine: {', '.join(descriptions[:-1])}, and finally, {descriptions[-1]}. "
        "Together, they form a grand vision, unexpected and brilliant."
    )

    # Store results in the "CurrentRoleplay" table
    cursor.execute('DELETE FROM CurrentRoleplay')  # Clear previous data
    cursor.execute('INSERT INTO CurrentRoleplay (key, value) VALUES (%s, %s)', ("MegaSetting", mega_name))
    cursor.execute('INSERT INTO CurrentRoleplay (key, value) VALUES (%s, %s)', ("MegaDescription", mega_description))
    conn.commit()
    conn.close()

    # Debug logs for clarity
    print("Mega Setting Name:", mega_name)
    print("Mega Setting Description:", mega_description)

    return jsonify({
        "mega_name": mega_name,
        "mega_description": mega_description,
        "message": "Mega setting generated and stored successfully."
    })


# Initialize the database on startup
if __name__ == '__main__':
    initialize_database()
    app.run(host='0.0.0.0', port=5000)
