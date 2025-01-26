# logic/npc_creation.py

import random
import json
from db.connection import get_db_connection
# from logic.meltdown_logic import meltdown_dialog_gpt  # If needed
from routes.archetypes import assign_archetypes_to_npc

logging.basicConfig(level=logging.DEBUG)  # or configure differently

def create_npc_logic(npc_name=None, introduced=False):
    """
    Core function that creates a new NPC in NPCStats.
    """
    if not npc_name:
        npc_name = f"NPC_{random.randint(1000,9999)}"

    logging.debug(f"[create_npc_logic] Starting with npc_name={npc_name}, introduced={introduced}")

    dominance = random.randint(10, 40)
    cruelty = random.randint(10, 40)
    closeness = random.randint(0, 30)
    trust = random.randint(-20, 20)
    respect = random.randint(-20, 20)
    intensity = random.randint(0, 40)

    logging.debug(f"[create_npc_logic] Random stats => dom={dominance}, cru={cruelty}, clos={closeness}, "
                  f"trust={trust}, respect={respect}, intensity={intensity}")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO NPCStats (
                npc_name,
                dominance, cruelty, closeness, trust, respect, intensity,
                memory, monica_level, monica_games_left, introduced
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING npc_id
        """, (
            npc_name,
            dominance, cruelty, closeness, trust, respect, intensity,
            "",  # memory
            0,   # monica_level
            0,   # monica_games_left
            introduced
        ))
        new_npc_id = cursor.fetchone()[0]
        conn.commit()

        logging.debug(f"[create_npc_logic] Inserted new NPC: npc_id={new_npc_id}")

        # Assign random flavor
        assign_npc_flavor(new_npc_id)
        logging.debug(f"[create_npc_logic] Flavor assigned for npc_id={new_npc_id}")

        return new_npc_id

    except Exception as e:
        conn.rollback()
        logging.error(f"[create_npc_logic] ERROR: {e}", exc_info=True)
        raise  # re-raise so calling code can handle the exception
    finally:
        conn.close()

def assign_npc_flavor(npc_id: int):
    """
    Assigns random occupation, hobbies, personality traits, 
    likes, and dislikes to the given NPC.
    
    Stores them in NPCStats as new columns:
      - occupation (TEXT)
      - hobbies (JSONB array)
      - personality_traits (JSONB array)
      - likes (JSONB array)
      - dislikes (JSONB array)
    """

    # 1) Prepare lists to pick from
    occupations = [
        "College Student", "Bartender", "Software Engineer", "Nurse",
        "Retail Clerk", "CEO of a Startup", "Freelance Artist",
        "Research Scientist", "Fitness Trainer", "Police Officer",
        "Influencer/Streamer", "Musician", "Gothic Fashion Designer"
    ]
    hobbies_pool = [
        "Reading", "Guitar Playing", "Jogging", "Dancing", "Video Gaming",
        "Knitting", "Cosplay", "Painting", "Tarot Reading", "Yoga",
        "Cooking", "Rock Climbing", "Embroidery", "Archery", "Hacking",
        "Street Racing", "Chess", "Gardening", "Swimming", "Singing",
        "Horror Movie Marathons", "Tabletop RPGs", "Fashion Blogging"
    ]
    personality_pool = [
        "Sarcastic", "Gentle", "Impatient", "Bubbly", "Brooding",
        "Obsessive", "Analytical", "Cruel", "Protective", "Lazy",
        "Ambitious", "Playful", "Flirtatious", "Passive-Aggressive",
        "Hotheaded", "Stoic", "Whimsical", "Needy", "Sultry", "Dominant",
        "Energetic", "Proud", "Compassionate", "Sadistic", "Meticulous"
    ]
    likes_pool = [
        "Chocolate", "Rainy Days", "Silly Memes", "Spicy Food", 
        "Bubble Baths", "Hard Rock Music", "Romance Novels",
        "Collecting Figurines", "Anime", "Sci-Fi Movies",
        "Cats", "Lavender Scent", "Camping", "Freaky Experiments",
        "Piercings & Tattoos"
    ]
    dislikes_pool = [
        "Crowded Places", "Dishonesty", "Boredom", "Loud Noises",
        "Being Ignored", "Incompetence", "Early Mornings", "Weak Coffee",
        "Spiders", "Obnoxious Laughter", "Long Meetings", "Cheap Perfume",
        "Paperwork", "Reality TV", "Cold Food"
    ]

    # 2) Randomly pick single occupation
    occupation = random.choice(occupations)

    # 3) Randomly pick 3 distinct hobbies
    hobbies = random.sample(hobbies_pool, k=3)

    # 4) Randomly pick 5 personality traits
    personality_traits = random.sample(personality_pool, k=5)

    # 5) Randomly pick 3 likes, 3 dislikes
    likes = random.sample(likes_pool, k=3)
    dislikes = random.sample(dislikes_pool, k=3)

    # 6) Store in DB
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE NPCStats
            SET occupation = %s,
                hobbies = %s,
                personality_traits = %s,
                likes = %s,
                dislikes = %s
            WHERE npc_id = %s
        """, (
            occupation,
            json.dumps(hobbies),               # store as JSONB
            json.dumps(personality_traits),    # store as JSONB
            json.dumps(likes),
            json.dumps(dislikes),
            npc_id
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def introduce_random_npc():
    """
    Finds an unintroduced NPC in NPCStats (introduced=FALSE), 
    flips introduced=TRUE, and returns the npc_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_id
        FROM NPCStats
        WHERE introduced = FALSE
        LIMIT 1
    """)
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    npc_id = row[0]
    cursor.execute("""
        UPDATE NPCStats
        SET introduced = TRUE
        WHERE npc_id = %s
    """, (npc_id,))
    conn.commit()
    conn.close()
    return npc_id
