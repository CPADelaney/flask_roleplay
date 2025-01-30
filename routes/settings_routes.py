# routes/settings_routes.py

import random, json
import logging
from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection

settings_bp = Blueprint('settings_bp', __name__)

def generate_mega_setting_logic(user_id, conversation_id):
    """
    This function merges random 'Settings' rows for (user_id, conversation_id)
    into a single "mega setting". Returns a Python dict with results.

    If you want them to be truly global, you could remove user_id/conversation_id scoping.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Fetch all Settings for this user + conversation
    cursor.execute("""
        SELECT id, name, mood_tone, enhanced_features, stat_modifiers, activity_examples
        FROM Settings
        WHERE user_id=%s
          AND conversation_id=%s
    """, (user_id, conversation_id))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        # Return a dict with an "error" or fallback
        return {
            "error": "No settings found in DB for this user & conversation.",
            "mega_name": "Empty Settings Table",
            "mega_description": "No environment generated",
            "enhanced_features": [],
            "stat_modifiers": {},
            "activity_examples": []
        }

    # 2) Convert rows to a list of dicts
    all_settings = []
    for row_id, row_name, row_mood, row_ef, row_sm, row_ae in rows:
        # row_ef, row_sm, row_ae might be stored as JSON in the DB. If so, parse them:
        # (If they're already JSON, remove the loads part.)
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

    # 3) Randomly pick 3–5 of them
    num_settings = random.choice([3, 4, 5])
    selected = random.sample(all_settings, min(num_settings, len(all_settings)))
    picked_names = [s["name"] for s in selected]

    # 4) Build mega_name
    mega_name = " + ".join(picked_names)

    # 5) Merge mood_tones into a single descriptive line
    all_mood_tones = [s["mood_tone"] for s in selected]
    if len(all_mood_tones) == 1:
        mega_description = f"The setting is just {all_mood_tones[0]}, forming a single thematic environment."
    else:
        mega_description = (
            f"The settings intertwine: {', '.join(all_mood_tones[:-1])}, "
            f"and finally, {all_mood_tones[-1]}. "
            "Together, they form a grand vision, unexpected and brilliant."
        )

    # 6) Merge features, stat_modifiers, activities
    combined_enhanced_features = []
    combined_stat_modifiers = {}
    combined_activity_examples = []

    for s_obj in selected:
        ef_list = s_obj["enhanced_features"]
        sm_dict = s_obj["stat_modifiers"]
        ae_list = s_obj["activity_examples"]

        # Merge features
        combined_enhanced_features.extend(ef_list)

        # Merge stat_modifiers
        for key, val in sm_dict.items():
            if key not in combined_stat_modifiers:
                combined_stat_modifiers[key] = val
            else:
                # If you want to chain them, or sum them, or override them, etc.
                combined_stat_modifiers[key] = f"{combined_stat_modifiers[key]}, {val}"

        # Merge activities
        combined_activity_examples.extend(ae_list)

    # 7) Return final dictionary
    return {
        "selected_settings": picked_names,
        "mega_name": mega_name,
        "mega_description": mega_description,
        "enhanced_features": combined_enhanced_features,
        "stat_modifiers": combined_stat_modifiers,
        "activity_examples": combined_activity_examples,
        "message": "Mega setting generated successfully via logic function."
    }

@settings_bp.route('/generate_mega_setting', methods=['POST'])
def generate_mega_setting_route():
    """
    A route that calls the logic function, scoping by user_id + conversation_id.
    Expects { "conversation_id": XXX } in JSON or something similar.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error":"No conversation_id provided"}), 400

        result = generate_mega_setting_logic(user_id, conversation_id)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def insert_missing_settings(user_id, conversation_id):
    """
    Inserts the default 1–30 settings if they do not exist for this user & conversation.
    Each setting row is associated with (user_id, conversation_id, name).
    """
    # Full set of 30 (truncated excerpt). 
    # Here, each setting is basically the same structure, but we also store them with user_id/conversation_id.
    settings_data = [
        {
            "name": "All-Girls College",  # #1
            "mood_tone": "A socially charged, cliquish environment where gossip is weaponized, and every interaction reinforces your inferiority.",
            "enhanced_features": [
                "NPCs exploit social hierarchies, ensuring public shame at every opportunity.",
                "Dormitories act as cages of judgment, with NPCs infiltrating your personal space to amplify humiliation. Gossip flows relentlessly, ensuring no failure is private."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs invade every aspect of your life. Shame increases with every public task, becoming a permanent modifier.",
                "dependency": "grows as NPCs offer 'friendship' in exchange for degrading tasks."
            },
            "activity_examples": [
                "Forced to perform demeaning tasks in front of classmates during group projects.",
                "Publicly confessing imagined 'infractions' during assemblies, ensuring your embarrassment becomes a spectacle.",
                "Participating in 'initiation ceremonies,' such as being paraded blindfolded through campus while peers mock your compliance.",
                "Professors assigning tasks like scrubbing classroom floors while students watch and critique."
            ]
        },
        {
            "name": "Corporate Office",  # #2
            "mood_tone": "A cold, hierarchical battlefield where power dynamics thrive, and mistakes are met with merciless punishment.",
            "enhanced_features": [
                "Surveillance systems capture every failure, broadcasting them to colleagues.",
                "Promotions turn into punishments, placing you under the cruelest NPCs.",
                "Professional authority is wielded to coerce submission, tying humiliation to job security."
            ],
            "stat_modifiers": {
                "dominance": "spikes due to managerial control.",
                "obedience": "locks at higher levels, making defiance impossible.",
                "shame": "permanently rises with public performance reviews."
            },
            "activity_examples": [
                "Crawling under conference tables while balancing items on your back during 'team-building exercises.'",
                "Publicly reciting demeaning apologies during performance reviews, amplifying your Shame.",
                "Wearing a placard listing your 'shortcomings' during work hours."
            ]
        },
        {
            "name": "Urban Life",  # #3
            "mood_tone": "A bustling, dynamic world where anonymity and exposure coexist, creating endless opportunities for humiliation.",
            "enhanced_features": [
                "Public spaces become traps, with NPCs manipulating crowds to isolate and shame you.",
                "NPCs manipulate casual encounters to trap you in degrading scenarios."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs exploit casual encounters to assert dominance.",
                "shame": "spikes during public failures, ensuring no dignity remains."
            },
            "activity_examples": [
                "Singing apologies at a crowded park, with NPCs recording and sharing the footage.",
                "Carrying absurd items through busy streets as passersby stare and mock."
            ]
        },
        {
            "name": "Post-Apocalypse",  # #4
            "mood_tone": "Brutal and desperate, where survival depends on submission to those who control resources.",
            "enhanced_features": [
                "Scarcity of resources forces submission, with NPCs fabricating need to deepen control.",
                "Lawlessness amplifies physical dominance, with punishments designed to leave lasting scars."
            ],
            "stat_modifiers": {
                "physical_endurance": "is tested through grueling tasks.",
                "cruelty": "of NPCs escalates in response to resistance."
            },
            "activity_examples": [
                "Digging ditches in the blistering sun while NPCs mock your weakness.",
                "Publicly bartering your dignity for food or water, enduring ridicule as payment."
            ]
        },
        {
            "name": "Traditional Horror",  # #5
            "mood_tone": "Dark and foreboding, where fear and submission intertwine in a supernatural nightmare.",
            "enhanced_features": [
                "NPCs use the environment’s eerie elements to isolate and disorient you.",
                "Shadows shift and objects move, creating an inescapable sense of vulnerability.",
                "Paranormal forces manipulate your senses, deepening your vulnerability."
            ],
            "stat_modifiers": {
                "intensity": "rises during fear-inducing tasks.",
                "corruption": "increases as you confront otherworldly influences."
            },
            "activity_examples": [
                "Crawling through haunted corridors while NPCs taunt you from unseen vantage points.",
                "Performing rituals meant to 'protect' you but designed to reinforce your dependence on NPCs.",
                "Being locked in a room where whispers mock your failures, forcing you to beg for release."
            ]
        },
        {
            "name": "Occult Ritual",  # #6
            "mood_tone": "Mystical, eerie, and inescapable, where submission feels inevitable under the weight of the supernatural.",
            "enhanced_features": [
                "Rituals are designed to strip you of autonomy, reframing your obedience as a 'sacred duty.'",
                "Ritual sites warp reality, amplifying NPC power and deepening your helplessness.",
                "Supernatural forces punish defiance with physical and psychological torment."
            ],
            "stat_modifiers": {
                "corruption": "escalates with every completed ritual, permanently altering your dialogue and choices.",
                "obedience": "locks through repeated indoctrination tasks."
            },
            "activity_examples": [
                "Kneeling in a summoning circle, chanting humiliating phrases dictated by NPCs.",
                "Serving as a living altar during sacrifices, with NPCs mocking your trembling.",
                "Reciting chants that degrade your sense of self while NPCs inscribe magical symbols onto your body."
            ]
        },
        {
            "name": "Vampire Society/Clan",  # #7
            "mood_tone": "Alluring yet predatory, with dominance wrapped in sensual manipulation and ancient traditions.",
            "enhanced_features": [
                "NPCs seduce and intimidate, using your desires and fears to enforce submission.",
                "Mansions become labyrinths of sensual dominance, where every room tests your loyalty.",
                "Vampire hierarchies demand constant displays of loyalty through degrading acts."
            ],
            "stat_modifiers": {
                "lust": "spikes during intimate tasks.",
                "corruption": "rises as submission feels increasingly seductive."
            },
            "activity_examples": [
                "Serving as a blood bearer during ceremonies, kneeling to offer yourself as a living chalice.",
                "Performing tasks that test your loyalty, such as reciting vows of eternal servitude."
            ]
        },
        {
            "name": "Femdom Empire",  # #8
            "mood_tone": "Grand and unyielding, where matriarchal dominance is codified into law.",
            "enhanced_features": [
                "Laws demand constant displays of submission, ensuring every act reinforces systemic power.",
                "Laws, customs, and societal norms ensure no act of rebellion goes unnoticed or unpunished.",
                "Public punishments are commonplace, drawing crowds to witness your humiliation."
            ],
            "stat_modifiers": {
                "dominance": "remains perpetually high due to the systemic power imbalance.",
                "trust": "becomes nearly impossible to gain without total obedience.",
                "respect": "is nearly impossible to earn without extreme acts of devotion."
            },
            "activity_examples": [
                "Kneeling during public ceremonies while NPCs list your 'offenses.'",
                "Paying exaggerated tributes, ensuring your financial ruin and deeper dependence."
            ]
        },
        {
            "name": "A Palace",  # #9
            "mood_tone": "Lavish and opulent, but suffocatingly hierarchical, where every misstep becomes a public spectacle.",
            "enhanced_features": [
                "NPCs wield social influence to create tasks that showcase your inferiority.",
                "The opulence of the palace becomes suffocating, with every room designed to highlight your inferiority.",
                "Ceremonial authority ensures punishments are grand and theatrical."
            ],
            "stat_modifiers": {
                "respect": "must be earned through acts of extreme servitude.",
                "closeness": "rises with court advisors and servants who oversee your tasks.",
                "shame": "rises with public punishments during court functions."
            },
            "activity_examples": [
                "Cleaning golden staircases while nobles step over you.",
                "Performing as a servant during court functions, wearing attire designed to humiliate."
            ]
        },
        {
            "name": "Matriarchy Kingdom",  # #10
            "mood_tone": "Structured and suffocating, where female dominance is woven into every aspect of society.",
            "enhanced_features": [
                "NPCs enforce submission through public rituals and strict laws.",
                "Towns and villages are structured to enforce public displays of submission.",
                "Social norms demand visible acts of obedience, ensuring your degradation is always on display."
            ],
            "stat_modifiers": {
                "respect": "is rarely granted, with men seen as inherently inferior.",
                "trust": "grows only through unflinching compliance.",
                "dependency": "grows as laws strip away autonomy."
            },
            "activity_examples": [
                "Participating in town square punishments, with NPCs ensuring large crowds witness your shame.",
                "Competing in degrading games during festivals, where the 'losers' face harsher public punishments."
            ]
        },
        {
            "name": "Monster Girl Alternate World",  # #11
            "mood_tone": "Whimsical yet terrifying, where non-human NPCs exploit their physical, magical, and primal advantages to dominate you.",
            "enhanced_features": [
                "The world itself feels alive, bending to the will of monstrous NPCs.",
                "Primal instincts and unfamiliar customs leave you constantly at a disadvantage."
            ],
            "stat_modifiers": {
                "intensity": "escalates rapidly as the setting highlights the power gap.",
                "corruption": "rises as you adapt to alien norms that redefine submission."
            },
            "activity_examples": [
                "Forced to serve as a perch for a winged NPC, enduring her weight as she 'rests.'",
                "Participating in rituals designed to bind you to a specific species, amplifying your humiliation with physical marks."
            ]
        },
        {
            "name": "Space",  # #12
            "mood_tone": "Isolated and vast, where survival depends on the benevolence of those in control.",
            "enhanced_features": [
                "The vacuum of space heightens your dependence on NPCs for air, food, and protection.",
                "Every system and structure is designed to magnify your vulnerability."
            ],
            "stat_modifiers": {
                "dominance": "rises as NPCs control access to essential resources.",
                "closeness": "grows as confined quarters force frequent interactions."
            },
            "activity_examples": [
                "Cleaning exterior ship surfaces while tethered by a thin cord, with NPCs watching and mocking your fear of floating away.",
                "Completing degrading tasks in cramped spaces, such as crawling through ventilation ducts to repair systems."
            ]
        },
        {
            "name": "Space Station or Alien Society",  # #13
            "mood_tone": "Cold, detached, and authoritarian, with advanced technology ensuring no resistance goes unnoticed.",
            "enhanced_features": [
                "Alien AI monitors every action, issuing corrections or punishments instantly.",
                "NPC customs and laws force compliance, framing your submission as a cultural necessity."
            ],
            "stat_modifiers": {
                "dominance": "is amplified by technological superiority.",
                "intensity": "escalates in the confined, sterile setting."
            },
            "activity_examples": [
                "Reciting alien creeds while restrained in a stasis field, with NPCs grading your performance.",
                "Serving as a test subject for 'research,' enduring physical or psychological experiments."
            ]
        },
        {
            "name": "Cyberpunk Future",  # #14
            "mood_tone": "A high-tech dystopia where surveillance and corporate control enforce submission at every level.",
            "enhanced_features": [
                "AI tracks your every move, ensuring no defiance escapes punishment.",
                "NPCs use leaked data to blackmail you, amplifying your reliance on their protection."
            ],
            "stat_modifiers": {
                "dominance": "rises through NPCs' technological control.",
                "closeness": "increases as NPCs manipulate digital connections to draw you closer."
            },
            "activity_examples": [
                "Publicly confessing imagined crimes during holographic broadcasts, with your face projected citywide.",
                "Completing degrading tasks for NPC-controlled AI, such as cleaning cybernetic implants in crowded marketplaces."
            ]
        },
        {
            "name": "Matriarchy/Gynarchy Future",  # #15
            "mood_tone": "A futuristic society where female dominance is institutionalized through technology and culture.",
            "enhanced_features": [
                "Holographic displays broadcast NPC commands, ensuring public compliance.",
                "Smart devices monitor and enforce obedience, issuing reminders or punishments remotely."
            ],
            "stat_modifiers": {
                "dominance": "remains high due to systemic enforcement.",
                "corruption": "escalates as you adapt to a world of unyielding female superiority."
            },
            "activity_examples": [
                "Performing tasks dictated by AI-controlled mistresses, such as fetching items or cleaning devices.",
                "Submitting to public corrections broadcast through holographic screens, ensuring widespread humiliation."
            ]
        },
        {
            "name": "Forgotten Realms",  # #16
            "mood_tone": "Mystical and grandiose, where magic and nobility combine to enforce your submission.",
            "enhanced_features": [
                "Magical laws bind you to NPCs, ensuring rebellion carries swift and supernatural consequences.",
                "The setting’s grandeur magnifies your insignificance."
            ],
            "stat_modifiers": {
                "respect": "is earned only through grand acts of loyalty.",
                "closeness": "rises with rulers and magical mentors who control your fate."
            },
            "activity_examples": [
                "Binding magical oaths that compel obedience, enforced through physical or emotional pain.",
                "Serving as a ceremonial attendant during royal functions, kneeling for hours while nobles mock your servitude."
            ]
        },
        {
            "name": "Final Fantasy (Generalized)",  # #17
            "mood_tone": "Magical and heroic on the surface, but rife with dark undercurrents of power imbalance.",
            "enhanced_features": [
                "NPCs wield magical powers to trap or punish you, framing your submission as a 'heroic sacrifice.'",
                "Quests are manipulated into tasks that degrade rather than empower."
            ],
            "stat_modifiers": {
                "respect": "rises only through extreme acts of devotion.",
                "closeness": "increases with magical rulers who exploit your 'destiny.'"
            },
            "activity_examples": [
                "Cleaning a powerful NPC’s magical artifacts, with mistakes resulting in immediate punishment.",
                "Wearing demeaning 'heroic' attire during public quests, ensuring your humiliation is visible to all."
            ]
        },
        {
            "name": "High Society",  # #18
            "mood_tone": "Elegant yet oppressive, where social standing dictates power, and humiliation is delivered with sophistication.",
            "enhanced_features": [
                "Gossip networks amplify every mistake, ensuring no act of defiance remains private.",
                "NPCs orchestrate events that subtly trap you into public embarrassment."
            ],
            "stat_modifiers": {
                "respect": "is hard to gain and easily lost.",
                "closeness": "rises as NPCs maintain appearances by engaging frequently."
            },
            "activity_examples": [
                "Serving drinks during formal events, wearing attire chosen specifically to humiliate you.",
                "Publicly apologizing for 'disrespecting' an NPC, with your words scripted to maximize your embarrassment."
            ]
        },
        {
            "name": "Manor",  # #19
            "mood_tone": "Intimate and suffocating, where NPCs monitor your every move within the isolated setting.",
            "enhanced_features": [
                "Every room is designed to trap or disorient you, ensuring escape is impossible.",
                "NPCs control access to comfort, food, and freedom."
            ],
            "stat_modifiers": {
                "closeness": "intensifies as NPCs oversee your tasks personally.",
                "trust": "is hard to build due to constant surveillance."
            },
            "activity_examples": [
                "Completing degrading chores while NPCs critique your performance.",
                "Being locked in rooms as part of 'correction' rituals, ensuring psychological dominance."
            ]
        },
        {
            "name": "Gothic Carnival/Festival",  # #20
            "mood_tone": "Whimsical and chaotic, with a dark undercurrent of manipulation and control.",
            "enhanced_features": [
                "NPCs use the carnival’s surreal elements to isolate and confuse you, ensuring submission feels inescapable.",
                "Social norms are skewed, making resistance seem irrational."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs engage frequently in the festival’s chaotic events.",
                "corruption": "escalates as the setting’s surreal elements warp your perspective."
            },
            "activity_examples": [
                "Participating in rigged games designed to ensure your public failure.",
                "Being tied to carnival attractions, such as a spinning wheel, while NPCs mock and taunt you."
            ]
        },
        {
            "name": "Giantess Colony",  # #21
            "mood_tone": "Surreal and overwhelming, where physical dominance is unavoidable, and every interaction reinforces your insignificance.",
            "enhanced_features": [
                "The colony itself is designed to highlight your smallness, with towering structures and colossal NPCs reminding you of your inferiority."
            ],
            "stat_modifiers": {
                "dominance": "is amplified by the NPCs' sheer size and power.",
                "intensity": "escalates as physical tasks push you to your limits."
            },
            "activity_examples": [
                "Acting as a human footstool during communal gatherings, enduring hours of physical strain and mockery.",
                "Carrying oversized objects far beyond your strength, collapsing under NPC taunts."
            ]
        },
        {
            "name": "Ruined/Decayed Setting",  # #22
            "mood_tone": "Oppressive and desolate, where survival feels like a form of submission to the world itself.",
            "enhanced_features": [
                "The decayed surroundings create constant danger, forcing reliance on NPCs for protection and resources."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs exploit your dependence.",
                "dominance": "is amplified by the environment’s inherent power imbalance."
            },
            "activity_examples": [
                "Scavenging dangerous areas for resources under the threat of punishment for failure.",
                "Ritualistic punishments for disobedience, such as being left to fend off environmental dangers alone."
            ]
        },
        {
            "name": "Underwater/Flooded World",  # #23
            "mood_tone": "Submerged, eerie, and suffocating, where survival depends on obedience to those who control essential resources.",
            "enhanced_features": [
                "The water itself feels oppressive, limiting your mobility and visibility."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs enforce dependency on oxygen and shelter.",
                "corruption": "increases as you adapt to the setting’s unnatural, oppressive atmosphere."
            },
            "activity_examples": [
                "Completing underwater tasks under constant threat of drowning, with NPCs controlling your oxygen supply.",
                "Participating in aquatic rituals designed to strip away your sense of autonomy."
            ]
        },
        {
            "name": "Prison/Detention Facility",  # #24
            "mood_tone": "Harsh and authoritarian, where every aspect of life is controlled, and resistance is punished severely.",
            "enhanced_features": [
                "Every movement is monitored, with guards using the system to amplify your humiliation."
            ],
            "stat_modifiers": {
                "dominance": "remains perpetually high due to systemic control.",
                "closeness": "intensifies with NPCs who oversee your punishments."
            },
            "activity_examples": [
                "Begging for food during roll call, groveling at guards' feet.",
                "Wearing chains that symbolize your failures, ensuring no one forgets your place."
            ]
        },
        {
            "name": "Circus Freak Show",  # #25
            "mood_tone": "Chaotic and grotesque, where you are the star attraction in a surreal and humiliating spectacle.",
            "enhanced_features": [
                "The circus becomes a stage for your degradation, with every act designed to draw laughter and scorn."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs involve you in increasingly public performances.",
                "corruption": "escalates as you adapt to the degrading role."
            },
            "activity_examples": [
                "Performing 'tricks' for the crowd, such as balancing objects while crawling.",
                "Serving as a human target for audience members to throw soft projectiles at, ensuring constant public ridicule."
            ]
        },
        {
            "name": "Desert Wasteland",  # #26
            "mood_tone": "Brutal and unforgiving, where survival depends entirely on submission to those who control resources.",
            "enhanced_features": [
                "The heat and lack of shelter create constant vulnerability, with NPCs leveraging the environment to enforce control."
            ],
            "stat_modifiers": {
                "intensity": "remains high due to the brutal conditions.",
                "trust": "is hard to gain, as NPCs value their resources above all else."
            },
            "activity_examples": [
                "Fetching water from distant wells, collapsing under the harsh sun as NPCs mock your weakness.",
                "Enduring public punishments for failure, such as being tied to a post and left exposed to the elements."
            ]
        },
        {
            "name": "Cult Compound",  # #27
            "mood_tone": "Isolated and suffocating, where submission is reframed as devotion and individuality is erased.",
            "enhanced_features": [
                "Every aspect of life is controlled, from meals to social interactions, ensuring no escape from NPC influence."
            ],
            "stat_modifiers": {
                "corruption": "rises with exposure to indoctrination.",
                "dependency": "grows as NPCs enforce communal rituals."
            },
            "activity_examples": [
                "Publicly confessing imagined sins, groveling for forgiveness at the feet of NPCs.",
                "Participating in humiliating rituals designed to 'purify' you of rebellion."
            ]
        },
        {
            "name": "Medieval Dungeon",  # #28
            "mood_tone": "Dark, oppressive, and brutal, where fear and physical punishment enforce obedience.",
            "enhanced_features": [
                "The dungeon itself becomes a tool of control, with chains, restraints, and cold stone amplifying your helplessness."
            ],
            "stat_modifiers": {
                "intensity": "spikes due to the setting’s inherent harshness.",
                "closeness": "increases as NPCs personally oversee your punishments."
            },
            "activity_examples": [
                "Enduring public whippings in the dungeon square, with crowds gathered to watch your humiliation.",
                "Completing degrading tasks, such as scrubbing floors with your bare hands, for scraps of food."
            ]
        },
        {
            "name": "Floating Sky City",  # #29
            "mood_tone": "Awe-inspiring yet oppressive, where the grandeur of the city magnifies your insignificance.",
            "enhanced_features": [
                "The city’s height and isolation ensure no escape, with every act of defiance punished publicly."
            ],
            "stat_modifiers": {
                "respect": "is granted only through extreme acts of loyalty.",
                "dominance": "remains high due to the rulers’ elevated status."
            },
            "activity_examples": [
                "Cleaning ornate statues while suspended above the city, amplifying both fear and humiliation.",
                "Participating in public ceremonies where you’re paraded as an example of loyalty and submission."
            ]
        },
        {
            "name": "Surprise Me with Your Own Custom Creation",  # #30
            "mood_tone": "Tailored to your current narrative, designed to exploit your specific weaknesses and stats.",
            "enhanced_features": [
                "The setting shifts dynamically based on NPC dominance and your own vulnerability."
            ],
            "stat_modifiers": {
                "depends": "on the NPCs and tasks involved."
            },
            "activity_examples": [
                "Constructed on the spot to fit the narrative’s flow, ensuring maximum humiliation and submission."
            ]
        }
    ]

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Retrieve existing setting names for (user_id, conversation_id)
    cursor.execute("""
        SELECT name
        FROM Settings
        WHERE user_id=%s
          AND conversation_id=%s
    """, (user_id, conversation_id))
    existing = {row[0] for row in cursor.fetchall()}

    # 2) Insert if not existing
    for s in settings_data:
        if s["name"] not in existing:
            cursor.execute("""
                INSERT INTO Settings (
                  user_id, conversation_id,
                  name, mood_tone,
                  enhanced_features, stat_modifiers, activity_examples
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, conversation_id,
                s["name"],
                s["mood_tone"],
                json.dumps(s["enhanced_features"]),
                json.dumps(s["stat_modifiers"]),
                json.dumps(s["activity_examples"])
            ))
            print(f"Inserted new setting for user={user_id}, conv={conversation_id}, name={s['name']}")
        else:
            print(f"Skipped existing setting: {s['name']}")

    conn.commit()
    conn.close()
    print("All settings processed or skipped (already existed).")
