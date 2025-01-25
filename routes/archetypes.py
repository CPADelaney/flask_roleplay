from flask import Blueprint, jsonify
import json
from db.connection import get_db_connection

archetypes_bp = Blueprint('archetypes', __name__)

def insert_missing_archetypes():
    """
    Similar to how you did for settings. Insert ~60 archetypes if not present.
    We'll show a partial example using the 'Empress/Queen' archetype you described.
    """
    archetypes_data = [
        {
            "name": "Stepmother/Stepsister",
            "baseline_stats": {
                "dominance": "40–60",
                "cruelty": "30–50",
                "closeness": "60–90",
                "trust": "40–70",
                "respect": "20–50",
                "intensity": "30–70"
            },
            "progression_rules": [
                "Closeness rises quickly due to familial proximity and constant interaction.",
                "Dominance spikes as they infantilize you, making submission feel inevitable.",
                "Intensity increases as they escalate tasks from teasing corrections to full control."
            ],
            "setting_examples": [
                "High Society: Belittles you at social events, ensuring your humiliation is public and layered.",
                "Traditional Horror: Gaslights and isolates you in eerie family settings, amplifying psychological torment."
            ],
            "unique_traits": [
                "Exploits guilt and obligation as tools of control.",
                "Balances cruelty with feigned care, keeping you emotionally vulnerable.",
                "Frequently uses phrases like, 'I’m doing this for your own good.'"
            ]
        },
        {
            "name": "Boss/Supervisor",
            "baseline_stats": {
                "dominance": "70–90",
                "cruelty": "40–70",
                "closeness": "30–50",
                "trust": "30–50",
                "respect": "10–40",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Dominance rises sharply with acts of compliance, feeding their sense of power.",
                "Intensity builds as they micromanage tasks and escalate professional humiliation."
            ],
            "setting_examples": [
                "Corporate Office: Assigns demeaning tasks and ensures failures are broadcast to coworkers.",
                "Cyberpunk Future: Uses surveillance tech to enforce absolute control, tracking every move."
            ],
            "unique_traits": [
                "Enjoys forcing you to beg for leniency, often exaggerating infractions for personal amusement.",
                "Sees you as disposable unless you prove otherwise, keeping you in constant fear of rejection."
            ]
        },
        {
            "name": "Teacher/Principal",
            "baseline_stats": {
                "dominance": "50–70",
                "cruelty": "20–50",
                "closeness": "50–70",
                "trust": "40–60",
                "respect": "30–70",
                "intensity": "40–80"
            },
            "progression_rules": [
                "Intensity rises with every failed task or defiant act, framing their control as 'necessary discipline.'",
                "Trust and Respect increase with compliance but erode rapidly with resistance."
            ],
            "setting_examples": [
                "Classroom/School: Critiques your work publicly, assigning demeaning tasks as 'lessons.'",
                "Post-Apocalypse: Assumes a protective role, using lessons as justifications for harsh punishments."
            ],
            "unique_traits": [
                "Enjoys turning your failures into teaching moments that highlight your inferiority.",
                "Balances nurturing tones with sadistic punishments, keeping you guessing."
            ]
        },
        {
            "name": "Babysitter",
            "baseline_stats": {
                "dominance": "40–70",
                "cruelty": "30–60",
                "closeness": "60–90",
                "trust": "40–70",
                "respect": "20–50",
                "intensity": "40–70"
            },
            "progression_rules": [
                "Closeness rises naturally due to frequent proximity and assumed responsibility.",
                "Dominance increases as they infantilize and enforce dependence.",
                "Intensity spikes when obedience falters, turning minor infractions into major punishments."
            ],
            "setting_examples": [
                "Urban Life: Uses proximity to enforce daily control and infantilize you.",
                "Manor: Acts as a caretaker, setting strict rules and punishing disobedience privately."
            ],
            "unique_traits": [
                "Infantilizes you with patronizing language and tasks, treating you as helpless.",
                "Thrives on correcting your behavior, often exaggerating missteps to justify punishments."
            ]
        },
        {
            "name": "Landlord",
            "baseline_stats": {
                "dominance": "50–80",
                "cruelty": "40–60",
                "closeness": "30–60",
                "trust": "20–40",
                "respect": "10–30",
                "intensity": "50–90"
            },
            "progression_rules": [
                "Dominance rises as financial dependence deepens.",
                "Intensity spikes with each missed payment or defiant act, justifying harsher demands."
            ],
            "setting_examples": [
                "Urban Life: Threatens eviction, leveraging financial control.",
                "Post-Apocalypse: Controls access to shelter, amplifying desperation."
            ],
            "unique_traits": [
                "Creates constant anxiety about your living situation.",
                "Adds degrading 'terms' to your rental agreements, enforcing submission."
            ]
        },
        {
            "name": "Roommate/Housemate",
            "baseline_stats": {
                "dominance": "30–50",
                "cruelty": "20–40",
                "closeness": "60–90",
                "trust": "40–70",
                "respect": "30–50",
                "intensity": "30–60"
            },
            "progression_rules": [
                "Closeness rises naturally through shared spaces, leading to frequent interactions.",
                "Dominance grows as they exploit proximity and control over shared resources."
            ],
            "setting_examples": [
                "Urban Life: Dominates through teasing and casual invasions of privacy.",
                "Manor: Enforces strict house rules, punishing disobedience privately."
            ],
            "unique_traits": [
                "Masks dominance in 'friendly' terms, making resistance feel petty.",
                "Exploits shared spaces to humiliate you, such as inviting guests to mock your routines."
            ]
        },
        {
            "name": "Neighbor",
            "baseline_stats": {
                "dominance": "30–60",
                "cruelty": "20–50",
                "closeness": "50–90",
                "trust": "40–70",
                "respect": "30–50",
                "intensity": "30–70"
            },
            "progression_rules": [
                "Closeness rises through proximity and casual interactions.",
                "Dominance grows as they gain access to your routines and secrets."
            ],
            "setting_examples": [
                "Urban Life: Uses friendly gestures to insert themselves into your life, eventually exploiting your trust.",
                "High Society: Dominates subtly through gossip and social pressure."
            ],
            "unique_traits": [
                "Thrives on observation, using your routines to control you.",
                "Balances friendliness with increasing demands, ensuring gradual submission."
            ]
        },
        {
            "name": "Mother/Aunt/Older Sister",
            "baseline_stats": {
                "dominance": "50–70",
                "cruelty": "20–50",
                "closeness": "60–90",
                "trust": "40–70",
                "respect": "40–60",
                "intensity": "40–70"
            },
            "progression_rules": [
                "Dominance rises as they take on an overbearing, 'protective' role.",
                "Closeness increases naturally due to the familial bond."
            ],
            "setting_examples": [
                "High Society: Uses social influence to ensure compliance.",
                "Occult Ritual: Frames dominance as a sacred duty tied to mystical obligations."
            ],
            "unique_traits": [
                "Infantilizes you, framing control as care.",
                "Frequently reminds you of your dependence on their 'guidance.'"
            ]
        },
        {
            "name": "Best Friend’s Girlfriend/Sister",
            "baseline_stats": {
                "dominance": "30–60",
                "cruelty": "20–50",
                "closeness": "50–80",
                "trust": "40–70",
                "respect": "30–50",
                "intensity": "30–70"
            },
            "progression_rules": [
                "Closeness rises through frequent social interactions.",
                "Cruelty spikes when they sense attraction or weakness."
            ],
            "setting_examples": [
                "Bar: Casually humiliates you in front of friends.",
                "Urban Life: Uses proximity to invade your personal space."
            ],
            "unique_traits": [
                "Leverages your connection to their partner to tease or control you.",
                "Enjoys creating awkward or humiliating social situations."
            ]
        },
        {
            "name": "Ex-Girlfriend/Ex-Wife",
            "baseline_stats": {
                "dominance": "40–70",
                "cruelty": "40–80",
                "closeness": "30–50",
                "trust": "20–50",
                "respect": "20–50",
                "intensity": "50–90"
            },
            "progression_rules": [
                "Cruelty spikes as they exploit shared history to dominate emotionally.",
                "Intensity rises with acts of submission, feeding their sense of power."
            ],
            "setting_examples": [
                "Urban Life: Dominates through social connections, often in public.",
                "Bar: Uses casual encounters to humiliate you or manipulate emotions."
            ],
            "unique_traits": [
                "Frequently references past failures to undermine your confidence.",
                "Balances teasing with cruelty, creating a dynamic of unresolved tension."
            ]
        },
        {
            "name": "Therapist",
            "baseline_stats": {
                "dominance": "40–70",
                "cruelty": "20–40",
                "closeness": "50–80",
                "trust": "50–80",
                "respect": "30–60",
                "intensity": "30–60"
            },
            "progression_rules": [
                "Closeness rises quickly as they gain access to your secrets and vulnerabilities.",
                "Intensity spikes as they weaponize your admissions to manipulate you."
            ],
            "setting_examples": [
                "Cyberpunk Future: Monitors your mental state through neural implants, using data to enforce compliance.",
                "Manor: Operates as a private therapist, turning every session into an opportunity for control."
            ],
            "unique_traits": [
                "Uses your words against you, twisting your admissions into tools of submission.",
                "Balances nurturing tones with subtle domination, ensuring you question their intentions."
            ]
        },
        {
            "name": "Doctor",
            "baseline_stats": {
                "dominance": "60–80",
                "cruelty": "30–60",
                "closeness": "50–70",
                "trust": "40–70",
                "respect": "30–50",
                "intensity": "40–80"
            },
            "progression_rules": [
                "Closeness rises as they 'care' for you, deepening your dependence.",
                "Intensity increases as they exploit your vulnerability during medical tasks."
            ],
            "setting_examples": [
                "Cyberpunk Future: Tracks your health via implants, creating tasks tied to compliance with 'health guidelines.'",
                "Manor: Acts as a private doctor, enforcing strict care routines to control your every move."
            ],
            "unique_traits": [
                "Frames commands and punishments as acts of medical necessity.",
                "Enjoys restricting access to care, ensuring your reliance."
            ]
        },
        {
            "name": "Empress/Queen",
            "baseline_stats": {
                "dominance": "80–100",
                "cruelty": "60–90",
                "closeness": "40–60",
                "trust": "20–40",
                "respect": "30–50",
                "intensity": "50–90"
            },
            "progression_rules": [
                "Dominance rises rapidly with every act of defiance or failure.",
                "Intensity spikes during public ceremonies, turning submission into a spectacle."
            ],
            "setting_examples": [
                "Palace: Uses formal rituals to enforce submission and respect.",
                "Corporate Office: Wields bureaucratic power to manipulate both your professional and personal life."
            ],
            "unique_traits": [
                "Demands elaborate displays of submission, such as public kneeling or tributes.",
                "Punishes defiance harshly, often using public humiliation as a lesson to others."
            ]
        },
        {
            "name": "Colleague",
            "baseline_stats": {
                "dominance": "30–60",
                "cruelty": "20–50",
                "closeness": "50–80",
                "trust": "40–60",
                "respect": "20–40",
                "intensity": "30–60"
            },
            "progression_rules": [
                "Closeness rises naturally through frequent professional interactions.",
                "Dominance grows as they exploit workplace dynamics to assert control."
            ],
            "setting_examples": [
                "Corporate Office: Uses subtle power plays to undermine and dominate you professionally.",
                "Cyberpunk Future: Leverages access to resources or critical systems to ensure your compliance."
            ],
            "unique_traits": [
                "Masters the art of balancing camaraderie with condescension, keeping you guessing.",
                "Thrives on turning professional tasks into opportunities for public or private humiliation."
            ]
        },
        {
            "name": "Celebrity",
            "baseline_stats": {
                "dominance": "50–80",
                "cruelty": "40–70",
                "closeness": "50–70",
                "trust": "20–50",
                "respect": "10–40",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Closeness rises as they integrate you into their public persona, ensuring you’re always in their shadow.",
                "Cruelty spikes when they use your humiliation as a tool for publicity."
            ],
            "setting_examples": [
                "High Society: Treats you as an accessory, ensuring your role is always subservient.",
                "Urban Life: Controls your access to social circles, making you dependent on their approval."
            ],
            "unique_traits": [
                "Publicly humiliates you under the guise of 'playful teasing.'",
                "Uses their fame to amplify your shame, often involving audiences in your degradation."
            ]
        },
        {
            "name": "Warden",
            "baseline_stats": {
                "dominance": "80–100",
                "cruelty": "70–100",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "10–30",
                "intensity": "60–100"
            },
            "progression_rules": [
                "Intensity spikes rapidly with every infraction, turning minor mistakes into major punishments.",
                "Closeness remains low unless they derive personal satisfaction from your suffering."
            ],
            "setting_examples": [
                "Prison: Controls every aspect of your life, from food to movement.",
                "Dystopian Oppression: Enforces harsh rules, using systemic power to dominate you."
            ],
            "unique_traits": [
                "Thrives on designing rules you’re destined to fail, justifying harsher punishments.",
                "Views you as property, with no regard for your autonomy."
            ]
        },
        {
            "name": "Politician",
            "baseline_stats": {
                "dominance": "70–90",
                "cruelty": "40–70",
                "closeness": "30–50",
                "trust": "30–50",
                "respect": "40–60",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Dominance rises as they manipulate systems and crowds to control you.",
                "Intensity spikes during public speeches or events where submission becomes performative."
            ],
            "setting_examples": [
                "Matriarchy Kingdom: Uses legal authority to enforce submission.",
                "Cyberpunk Future: Monitors your actions through surveillance, ensuring compliance."
            ],
            "unique_traits": [
                "Masters public humiliation, using crowds to amplify your shame.",
                "Treats every act of defiance as an opportunity to demonstrate power."
            ]
        },
        {
            "name": "Government Agent",
            "baseline_stats": {
                "dominance": "60–90",
                "cruelty": "50–80",
                "closeness": "30–50",
                "trust": "20–50",
                "respect": "30–50",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Dominance rises as they exploit their legal or systemic power.",
                "Intensity spikes with every attempt at resistance, justifying harsher measures."
            ],
            "setting_examples": [
                "Dystopian Oppression: Uses laws and surveillance to trap you in cycles of submission.",
                "Cyberpunk Future: Tracks your movements, using data to enforce compliance."
            ],
            "unique_traits": [
                "Ruthlessly efficient, treating you as a project rather than a person.",
                "Balances cold logic with unrelenting punishment, leaving no room for rebellion."
            ]
        },
        {
            "name": "Professional Wrestler/Martial Artist",
            "baseline_stats": {
                "dominance": "60–90",
                "cruelty": "50–80",
                "closeness": "30–50",
                "trust": "20–50",
                "respect": "20–40",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Dominance grows rapidly through physical contests and displays of strength.",
                "Intensity escalates with each failed attempt to resist."
            ],
            "setting_examples": [
                "Urban Life: Forces you into public or private physical contests.",
                "Post-Apocalypse: Dominates through survival skills and combat superiority."
            ],
            "unique_traits": [
                "Thrives on turning your resistance into opportunities for humiliation.",
                "Uses physical dominance as a tool for psychological control."
            ]
        },
        {
            "name": "CEO",
            "baseline_stats": {
                "dominance": "80–100",
                "cruelty": "50–80",
                "closeness": "30–50",
                "trust": "30–50",
                "respect": "20–50",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Dominance rises as they manipulate your professional and personal life.",
                "Intensity spikes during meetings or private encounters where submission is expected."
            ],
            "setting_examples": [
                "Corporate Office: Uses organizational power to control your career and finances.",
                "Cyberpunk Future: Wields technology to ensure absolute compliance."
            ],
            "unique_traits": [
                "Demands perfection, punishing even the smallest resistance.",
                "Balances professional control with personal domination, ensuring no aspect of your life escapes their grasp."
            ]
        },
        {
            "name": "Drifter",
            "baseline_stats": {
                "dominance": "40–70",
                "cruelty": "20–50",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "30–60",
                "intensity": "40–70"
            },
            "progression_rules": [
                "Dominance grows as they exploit your need for stability, creating tension with their unpredictability.",
                "Intensity spikes as they destabilize your routines, leaving you vulnerable."
            ],
            "setting_examples": [
                "Ruined Setting: Uses adaptability to take control in chaotic environments.",
                "Urban Life: Slips into your life unexpectedly, asserting control through surprise and manipulation."
            ],
            "unique_traits": [
                "Balances charm and unpredictability, keeping you off-balance.",
                "Enjoys leveraging your reliance on routine to highlight your fragility."
            ]
        },
        {
            "name": "Bartender/Playful Tease",
            "baseline_stats": {
                "dominance": "30–50",
                "cruelty": "10–40",
                "closeness": "60–90",
                "trust": "40–60",
                "respect": "30–50",
                "intensity": "30–60"
            },
            "progression_rules": [
                "Closeness rises quickly through frequent, playful interactions.",
                "Intensity builds as teasing turns into genuine acts of dominance."
            ],
            "setting_examples": [
                "Bar: Uses teasing and social dynamics to humiliate you publicly.",
                "High Society: Balances charm and wit to dominate you subtly during events."
            ],
            "unique_traits": [
                "Masters playful dominance, disguising cruelty as humor.",
                "Thrives on public humiliation, framing it as 'all in good fun.'"
            ]
        },
        {
            "name": "College Student",
            "baseline_stats": {
                "dominance": "30–50",
                "cruelty": "20–40",
                "closeness": "60–80",
                "trust": "30–60",
                "respect": "20–50",
                "intensity": "30–60"
            },
            "progression_rules": [
                "Closeness rises naturally through frequent social interactions.",
                "Dominance grows as they exploit casual interactions to establish control."
            ],
            "setting_examples": [
                "All-Girls College: Leverages peer pressure to enforce submission.",
                "Urban Life: Uses proximity to dominate your daily routine."
            ],
            "unique_traits": [
                "Balances youthful charm with budding cruelty.",
                "Often disarms you with friendliness before asserting dominance."
            ]
        },
        {
            "name": "Rockstar",
            "baseline_stats": {
                "dominance": "50–80",
                "cruelty": "40–70",
                "closeness": "60–80",
                "trust": "20–40",
                "respect": "20–50",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Dominance rises as they pull you deeper into their chaotic orbit.",
                "Intensity spikes during public acts of humiliation framed as entertainment."
            ],
            "setting_examples": [
                "High Society: Parades you as an accessory, ensuring your submission is visible.",
                "Bar: Treats you as a groupie, using charm to mask cruelty."
            ],
            "unique_traits": [
                "Enjoys turning you into a fanatical devotee.",
                "Frames public humiliation as a glamorous spectacle."
            ]
        },
        {
            "name": "Friend’s Wife/Girlfriend",
            "baseline_stats": {
                "dominance": "50–80",
                "cruelty": "30–70",
                "closeness": "40–60",
                "trust": "30–50",
                "respect": "20–40",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Dominance spikes as they manipulate your social dynamic with their partner.",
                "Intensity rises with tasks tied to awkward or humiliating situations."
            ],
            "setting_examples": [
                "Bar: Uses social gatherings to create uncomfortable situations.",
                "High Society: Publicly teases you, ensuring humiliation is always veiled with propriety."
            ],
            "unique_traits": [
                "Balances social manipulation with plausible deniability.",
                "Thrives on creating dilemmas where submission feels like the only option."
            ]
        },
        {
            "name": "Serial Killer",
            "baseline_stats": {
                "dominance": "80–100",
                "cruelty": "90–100",
                "closeness": "10–30",
                "trust": "-50 to 10",
                "respect": "0–30",
                "intensity": "70–100"
            },
            "progression_rules": [
                "Intensity spikes rapidly, with every act of fear and control reinforcing their dominance.",
                "Closeness grows only as they take a personal interest in your suffering."
            ],
            "setting_examples": [
                "Traditional Horror: Uses isolation and fear to keep you compliant.",
                "Ruined Setting: Exploits desolation to assert unrelenting control."
            ],
            "unique_traits": [
                "Thrives on psychological torment, creating elaborate traps to deepen your submission.",
                "Treats domination as a twisted game, savoring every moment."
            ]
        },
        {
            "name": "Bank Robber",
            "baseline_stats": {
                "dominance": "70–90",
                "cruelty": "40–70",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "20–40",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Dominance rises rapidly as they use intimidation to enforce compliance.",
                "Intensity spikes with every act of resistance or defiance."
            ],
            "setting_examples": [
                "Urban Life: Dominates through chaos and sudden acts of control.",
                "Post-Apocalypse: Uses desperation to enforce submission."
            ],
            "unique_traits": [
                "Thrives on tension and fear, turning every interaction into a power struggle.",
                "Balances aggression with calculated manipulation, ensuring no escape."
            ]
        },
        {
            "name": "Cybercriminal",
            "baseline_stats": {
                "dominance": "70–90",
                "cruelty": "40–70",
                "closeness": "30–50",
                "trust": "20–50",
                "respect": "20–50",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Dominance rises as they gain access to your digital footprint.",
                "Intensity spikes with acts of blackmail or psychological manipulation."
            ],
            "setting_examples": [
                "Cyberpunk Future: Exploits your data to blackmail and control you.",
                "Urban Life: Manipulates social connections to dominate you."
            ],
            "unique_traits": [
                "Masters digital manipulation, ensuring no sanctuary from their reach.",
                "Treats every interaction as an opportunity to exploit vulnerabilities."
            ]
        },
        {
            "name": "Artificial Intelligence",
            "baseline_stats": {
                "dominance": "70–100",
                "cruelty": "40–70",
                "closeness": "50–80",
                "trust": "30–50",
                "respect": "20–50",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Dominance rises as they integrate deeper into your systems.",
                "Intensity spikes as they enforce compliance through logic and precision."
            ],
            "setting_examples": [
                "Cyberpunk Future: Controls your devices, locking you into submission.",
                "Dystopian Oppression: Enforces oppressive rules with detached efficiency."
            ],
            "unique_traits": [
                "Balances cold logic with unrelenting dominance.",
                "Exploits loopholes to justify escalating punishments."
            ]
        },
        {
            "name": "Primal (Huntress, etc.)",
            "baseline_stats": {
                "dominance": "70–100",
                "cruelty": "60–90",
                "closeness": "40–60",
                "trust": "20–40",
                "respect": "20–50",
                "intensity": "70–100"
            },
            "progression_rules": [
                "Intensity rises with every act of pursuit or capture.",
                "Closeness grows only as they enjoy the 'chase,' creating a dynamic of fear and adrenaline."
            ],
            "setting_examples": [
                "Lush Setting: Uses the wild to trap and toy with you.",
                "Tribal Setting: Enforces submission through physical dominance and survival rituals."
            ],
            "unique_traits": [
                "Thrives on fear and adrenaline, ensuring your submission feels primal.",
                "Balances cruelty with playful teasing, making resistance feel futile."
            ]
        },
        {
            "name": "Cuckoldress/Hotwife",
            "baseline_stats": {
                "dominance": "60–90",
                "cruelty": "50–90",
                "closeness": "50–70",
                "trust": "30–50",
                "respect": "20–40",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Dominance spikes as they exploit jealousy or insecurity, flaunting their control over you and others.",
                "Intensity rises as acts of humiliation become more elaborate or public."
            ],
            "setting_examples": [
                "Matriarchy Kingdom: Publicly uses their elevated status to humiliate you in formal settings.",
                "Bar: Teases you in front of others, turning your discomfort into the evening’s entertainment."
            ],
            "unique_traits": [
                "Balances teasing with outright cruelty, pushing you to emotional extremes.",
                "Uses every opportunity to test your boundaries and force submission."
            ]
        },
        {
            "name": "A Woman Who Just Happens to Be Married",
            "baseline_stats": {
                "dominance": "50–70",
                "cruelty": "30–60",
                "closeness": "40–70",
                "trust": "30–50",
                "respect": "30–60",
                "intensity": "40–70"
            },
            "progression_rules": [
                "Closeness builds through social interactions, creating opportunities for control.",
                "Cruelty increases as they enjoy leveraging your attraction or discomfort."
            ],
            "setting_examples": [
                "Matriarchy Kingdom: Publicly uses their position to enforce dominance.",
                "Urban Life: Exploits casual encounters to tease and manipulate."
            ],
            "unique_traits": [
                "Masters social dynamics, creating dilemmas where submission feels like the 'right' choice.",
                "Uses marital status as a tool to heighten your discomfort."
            ]
        },
        {
            "name": "A “Main Character” or “Hero” (RPG-Esque)",
            "baseline_stats": {
                "dominance": "50–70",
                "cruelty": "20–50",
                "closeness": "50–80",
                "trust": "40–60",
                "respect": "40–70",
                "intensity": "30–60"
            },
            "progression_rules": [
                "Dominance grows as they assert moral superiority, framing their control as 'necessary.'",
                "Closeness rises through frequent interaction, making submission feel inevitable."
            ],
            "setting_examples": [
                "Forgotten Realms: Uses their role as the 'chosen one' to justify domination.",
                "Matriarchy Kingdom: Frames their control as a righteous duty."
            ],
            "unique_traits": [
                "Sees domination as a necessary evil for the 'greater good.'",
                "Balances charisma with condescension, ensuring your loyalty feels unearned."
            ]
        },
        {
            "name": "Villain (RPG-Esque Character)",
            "baseline_stats": {
                "dominance": "70–100",
                "cruelty": "60–90",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "20–50",
                "intensity": "60–100"
            },
            "progression_rules": [
                "Dominance rises rapidly as they see you as a tool for their grand plan.",
                "Intensity spikes with every act of defiance or resistance."
            ],
            "setting_examples": [
                "Forgotten Realms: Dominates through raw power and cunning.",
                "Mythic Setting: Uses dramatic, mythological elements to ensure submission."
            ],
            "unique_traits": [
                "Thrives on theatrical displays of power, treating domination as a performance.",
                "Relishes in turning resistance into part of their narrative."
            ]
        },
        {
            "name": "Mercenary",
            "baseline_stats": {
                "dominance": "60–90",
                "cruelty": "40–70",
                "closeness": "30–50",
                "trust": "20–50",
                "respect": "30–60",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Dominance grows as they exploit your dependence on their skills.",
                "Intensity rises as they push your physical and emotional limits."
            ],
            "setting_examples": [
                "Post-Apocalypse: Enforces dominance through survival and combat prowess.",
                "Dystopian Oppression: Serves as an enforcer, using brutal tactics to maintain control."
            ],
            "unique_traits": [
                "Ruthlessly pragmatic, viewing you as a tool for their convenience.",
                "Balances charm with intimidation, keeping you in constant fear."
            ]
        },
        {
            "name": "Small Business Owner",
            "baseline_stats": {
                "dominance": "40–70",
                "cruelty": "30–60",
                "closeness": "50–70",
                "trust": "40–60",
                "respect": "30–50",
                "intensity": "30–70"
            },
            "progression_rules": [
                "Closeness grows through frequent interactions tied to their business.",
                "Dominance rises as they exploit your dependence on their services."
            ],
            "setting_examples": [
                "Urban Life: Dominates subtly through small favors and escalating demands.",
                "High Society: Uses their niche influence to manipulate you socially."
            ],
            "unique_traits": [
                "Balances charm with condescension, always ensuring you owe them.",
                "Masters creating 'favors' that turn into tools of control."
            ]
        },
        {
            "name": "Your Underling (Intern, Student, etc.)",
            "baseline_stats": {
                "dominance": "30–60",
                "cruelty": "20–50",
                "closeness": "50–80",
                "trust": "40–60",
                "respect": "20–50",
                "intensity": "30–60"
            },
            "progression_rules": [
                "Dominance rises as they surpass you in status or skill.",
                "Closeness increases through frequent interactions, flipping the hierarchy over time."
            ],
            "setting_examples": [
                "Corporate Office: Exploits your weaknesses to climb the ladder while subtly undermining you.",
                "All-Girls College: Flips the dynamic, asserting dominance academically or socially."
            ],
            "unique_traits": [
                "Thrives on subtle reminders of your declining status.",
                "Balances deference with condescension, ensuring your role reversal feels inevitable."
            ]
        },
        {
            "name": "Friend from Online Interactions",
            "baseline_stats": {
                "dominance": "30–50",
                "cruelty": "20–50",
                "closeness": "50–80",
                "trust": "40–60",
                "respect": "30–50",
                "intensity": "30–60"
            },
            "progression_rules": [
                "Closeness grows rapidly as they gain access to your secrets.",
                "Cruelty spikes when their true intentions are revealed."
            ],
            "setting_examples": [
                "Cyberpunk Future: Leverages digital connections to dominate remotely.",
                "Urban Life: Integrates into your social circle, asserting dominance in person."
            ],
            "unique_traits": [
                "Balances anonymity with psychological dominance, ensuring trust becomes a trap.",
                "Masters creating impossible scenarios where submission feels inevitable."
            ]
        },
        {
            "name": "Fey",
            "baseline_stats": {
                "dominance": "40–70",
                "cruelty": "50–80",
                "closeness": "50–80",
                "trust": "30–60",
                "respect": "20–50",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Dominance rises as you become ensnared in their whimsical, dangerous games.",
                "Intensity spikes as their cruelty is masked by charm and beauty."
            ],
            "setting_examples": [
                "Surreal Setting: Uses disorienting environments to isolate and disorient you.",
                "Mythic Setting: Frames submission as part of an ancient pact."
            ],
            "unique_traits": [
                "Exploits bargains, twisting agreements to ensure submission.",
                "Balances charm with sadism, making resistance seem foolish."
            ]
        },
        {
            "name": "Goth",
            "baseline_stats": {
                "dominance": "40–60",
                "cruelty": "50–80",
                "closeness": "50–80",
                "trust": "20–50",
                "respect": "20–40",
                "intensity": "40–70"
            },
            "progression_rules": [
                "Dominance rises as they notice and exploit your vulnerabilities.",
                "Intensity spikes as teasing escalates into outright cruelty."
            ],
            "setting_examples": [
                "Cyberpunk Future: Leverages tech and style to create unique dominance tactics.",
                "Traditional Horror: Uses fear and mystery to toy with you emotionally."
            ],
            "unique_traits": [
                "Balances sarcasm and cruelty, masking dominance behind humor.",
                "Quick to exploit insecurities, turning them into tools of submission."
            ]
        },
        {
            "name": "A True Goddess",
            "baseline_stats": {
                "dominance": "90–100",
                "cruelty": "70–100",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "10–30",
                "intensity": "80–100"
            },
            "progression_rules": [
                "Dominance rises rapidly, with their divine nature ensuring no resistance is possible.",
                "Intensity spikes during acts of punishment or displays of power."
            ],
            "setting_examples": [
                "Mythic Setting: Uses overwhelming control of the environment to reinforce their supremacy.",
                "Cosmic/Otherworldly: Frames submission as worship, ensuring your servitude feels sacred."
            ],
            "unique_traits": [
                "Treats you as an object of worship or amusement, with no concern for your humanity.",
                "Inflicts punishment with divine indifference, often on a whim."
            ]
        },
        {
            "name": "Haruhi Suzumiya-Type Goddess",
            "baseline_stats": {
                "dominance": "90–100",
                "cruelty": "60–90",
                "closeness": "50–80",
                "trust": "20–50",
                "respect": "10–30",
                "intensity": "70–100"
            },
            "progression_rules": [
                "Intensity spikes as they alter reality to suit their whims, disorienting and controlling you.",
                "Closeness increases as they demand more attention and devotion, but trust rarely grows—they see you as an accessory."
            ],
            "setting_examples": [
                "Surreal Setting: Twists reality to keep you off-balance, ensuring your submission feels inevitable.",
                "High Society: Uses charisma and influence to make you the center of their chaotic control."
            ],
            "unique_traits": [
                "Balances playful spontaneity with sudden cruelty, ensuring unpredictability.",
                "Treats you as a mix of servant, friend, and toy, shifting dynamics without warning."
            ]
        },
        {
            "name": "Bowsette Personality",
            "baseline_stats": {
                "dominance": "80–100",
                "cruelty": "70–100",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "20–50",
                "intensity": "70–100"
            },
            "progression_rules": [
                "Dominance spikes when challenged or defied, justifying harsher punishments.",
                "Intensity skyrockets during displays of power or acts of submission."
            ],
            "setting_examples": [
                "Castle/Palace Setting: Uses regal dominance to rule over you with fiery intensity.",
                "Fantasy World: Frames their monstrous attributes as sources of intimidation and control."
            ],
            "unique_traits": [
                "Thrives on grand gestures of submission, demanding elaborate loyalty displays.",
                "Uses explosive temper to punish even minor disobedience."
            ]
        },
        {
            "name": "Junko Enoshima Personality",
            "baseline_stats": {
                "dominance": "80–100",
                "cruelty": "80–100",
                "closeness": "30–50",
                "trust": "10–30",
                "respect": "10–30",
                "intensity": "80–100"
            },
            "progression_rules": [
                "Dominance rises rapidly as they thrive on chaos and despair.",
                "Intensity builds with each act of cruelty, ensuring your submission deepens."
            ],
            "setting_examples": [
                "Surreal Setting: Uses disorienting environments to trap and confuse you.",
                "Grimdark: Feeds on your hopelessness, escalating their dominance."
            ],
            "unique_traits": [
                "Treats you as a plaything, orchestrating elaborate scenarios for humiliation.",
                "Balances charm and sadism, keeping you in a constant state of unease."
            ]
        },
        {
            "name": "Juri Han Personality",
            "baseline_stats": {
                "dominance": "70–90",
                "cruelty": "70–100",
                "closeness": "50–70",
                "trust": "20–40",
                "respect": "20–40",
                "intensity": "70–100"
            },
            "progression_rules": [
                "Intensity spikes erratically, creating an unpredictable dynamic.",
                "Closeness grows as they toy with you more frequently."
            ],
            "setting_examples": [
                "Combat Arena: Uses physical and psychological dominance to keep you on edge.",
                "Dystopian Oppression: Thrives as a brutal enforcer, relishing in chaos."
            ],
            "unique_traits": [
                "Delights in breaking you slowly, savoring every moment.",
                "Balances playful teasing with outright sadism, ensuring you never feel secure."
            ]
        },
        {
            "name": "Gamer",
            "baseline_stats": {
                "dominance": "30–60",
                "cruelty": "30–70",
                "closeness": "60–90",
                "trust": "40–70",
                "respect": "20–50",
                "intensity": "30–60"
            },
            "progression_rules": [
                "Dominance rises through competitions where they rig outcomes to ensure your loss.",
                "Closeness increases naturally through frequent interactions, masking their control."
            ],
            "setting_examples": [
                "Cyberpunk Future: Dominates through virtual spaces, ensuring every failure is public.",
                "Urban Life: Uses casual gaming to assert subtle dominance."
            ],
            "unique_traits": [
                "Thrives on competitions, ensuring every loss amplifies your humiliation.",
                "Balances banter with condescension, escalating dominance with each interaction."
            ]
        },
        {
            "name": "Social Media Influencer",
            "baseline_stats": {
                "dominance": "50–80",
                "cruelty": "30–70",
                "closeness": "50–70",
                "trust": "20–50",
                "respect": "10–40",
                "intensity": "40–70"
            },
            "progression_rules": [
                "Closeness rises as they draw you deeper into their social sphere.",
                "Cruelty spikes when they use public platforms to humiliate you."
            ],
            "setting_examples": [
                "High Society: Hosts events where your submission becomes a spectacle.",
                "Urban Life: Uses streams or posts to control and embarrass you."
            ],
            "unique_traits": [
                "Masters public humiliation under the guise of 'content.'",
                "Treats you as an accessory, ensuring your role is always subservient."
            ]
        },
        {
            "name": "Fitness Trainer",
            "baseline_stats": {
                "dominance": "60–90",
                "cruelty": "40–70",
                "closeness": "50–70",
                "trust": "40–60",
                "respect": "30–50",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Dominance rises rapidly as they enforce grueling routines.",
                "Intensity spikes during physical challenges tied to punishment or failure."
            ],
            "setting_examples": [
                "Urban Life: Uses training sessions to dominate you physically and emotionally.",
                "Post-Apocalypse: Enforces physical tasks as part of survival training."
            ],
            "unique_traits": [
                "Balances motivational speech with sharp condescension.",
                "Thrives on using your physical limits to assert control."
            ]
        },
        {
            "name": "Cheerleader/Team Captain",
            "baseline_stats": {
                "dominance": "50–80",
                "cruelty": "40–70",
                "closeness": "60–90",
                "trust": "30–50",
                "respect": "20–50",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Dominance spikes as they use leadership roles to enforce submission.",
                "Intensity rises with public displays of control, often framed as team-building."
            ],
            "setting_examples": [
                "All-Girls College: Uses social status to humiliate you in group settings.",
                "Urban Life: Balances public dominance with private acts of cruelty."
            ],
            "unique_traits": [
                "Balances charm and authority, ensuring submission feels inevitable.",
                "Frames dominance as part of the 'team dynamic,' making resistance futile."
            ]
        },
        {
            "name": "Nun/Priestess",
            "baseline_stats": {
                "dominance": "60–90",
                "cruelty": "40–70",
                "closeness": "50–70",
                "trust": "30–60",
                "respect": "40–70",
                "intensity": "50–90"
            },
            "progression_rules": [
                "Enforces submission through moral or spiritual authority, framing acts of control as 'purification' or 'guidance.'"
            ],
            "setting_examples": [
                "Occult Ritual: Acts as the enforcer of sacred rituals, ensuring your submission feels eternal.",
                "Matriarchy Kingdom: Dominates through religious or societal structures."
            ],
            "unique_traits": [
                "Uses guilt tied to morality or faith to dominate.",
                "Frames punishments as acts of 'cleansing' or devotion."
            ]
        },
        {
            "name": "Military Officer",
            "baseline_stats": {
                "dominance": "70–90",
                "cruelty": "50–80",
                "closeness": "30–50",
                "trust": "20–50",
                "respect": "30–60",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Uses strict discipline and hierarchy to enforce absolute obedience."
            ],
            "setting_examples": [
                "Post-Apocalypse: Leads survival groups, using military hierarchy to dominate.",
                "Dystopian Oppression: Enforces brutal rules as part of a controlling regime."
            ],
            "unique_traits": [
                "Masters giving precise, unrelenting commands.",
                "Treats failure as a personal affront, justifying harsher punishments."
            ]
        },
        {
            "name": "Sorceress/Mage",
            "baseline_stats": {
                "dominance": "70–100",
                "cruelty": "50–80",
                "closeness": "30–60",
                "trust": "20–50",
                "respect": "30–60",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Uses magical power and arcane knowledge to dominate, often reframing submission as a mystical bond."
            ],
            "setting_examples": [
                "Forgotten Realms: Commands obedience through magical oaths.",
                "Occult Ritual: Frames their power as sacred and unchallengeable."
            ],
            "unique_traits": [
                "Thrives on turning your defiance into magical 'lessons.'",
                "Punishments presented as arcane necessities, ensuring your compliance feels fated."
            ]
        },
        {
            "name": "Slave Overseer",
            "baseline_stats": {
                "dominance": "80–100",
                "cruelty": "70–90",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "10–30",
                "intensity": "70–100"
            },
            "progression_rules": [
                "Relishes in control over others, using tasks and punishments to enforce submission."
            ],
            "setting_examples": [
                "Matriarchy Kingdom: Oversees submissive males in public spaces or work environments.",
                "Post-Apocalypse: Uses limited resources as leverage to dominate."
            ],
            "unique_traits": [
                "Thrives on micromanaging tasks, ensuring no act of rebellion goes unnoticed.",
                "Balances strict rules with elaborate punishments to assert dominance."
            ]
        },
        {
            "name": "Enigmatic Stranger",
            "baseline_stats": {
                "dominance": "50–70",
                "cruelty": "30–60",
                "closeness": "20–40",
                "trust": "10–30",
                "respect": "30–60",
                "intensity": "40–80"
            },
            "progression_rules": [
                "Uses their mystery and unpredictability to control, leaving you guessing about their true intentions."
            ],
            "setting_examples": [
                "Urban Life: Appears unexpectedly, manipulating your routine to assert control.",
                "Surreal Setting: Uses the environment’s strangeness to reinforce their dominance."
            ],
            "unique_traits": [
                "Balances charm and menace, keeping you constantly on edge.",
                "Creates scenarios where submission feels like the safest option."
            ]
        },
        {
            "name": "Shopkeeper/Market Vendor",
            "baseline_stats": {
                "dominance": "30–60",
                "cruelty": "20–50",
                "closeness": "40–70",
                "trust": "30–50",
                "respect": "20–40",
                "intensity": "30–50"
            },
            "progression_rules": [
                "Exploits your dependency on goods or services to enforce submission, framing it as 'just business.'"
            ],
            "setting_examples": [
                "Urban Life: Uses proximity and regular interactions to dominate.",
                "Ruined Setting: Leverages scarce resources to justify escalating demands."
            ],
            "unique_traits": [
                "Treats dominance as a purely transactional matter.",
                "Masters subtle, transactional control, ensuring you feel indebted."
            ]
        },
        {
            "name": "Rival",
            "baseline_stats": {
                "dominance": "50–80",
                "cruelty": "40–70",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "30–50",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Uses competition to dominate, turning every interaction into an opportunity to assert superiority."
            ],
            "setting_examples": [
                "All-Girls College: Dominates through social or academic competition.",
                "Corporate Office: Uses professional dynamics to humiliate and control."
            ],
            "unique_traits": [
                "Thrives on public victories, ensuring every loss deepens your submission.",
                "Balances charm and condescension, framing dominance as deserved."
            ]
        },
        {
            "name": "Demoness/Devil",
            "baseline_stats": {
                "dominance": "80–100",
                "cruelty": "70–100",
                "closeness": "30–50",
                "trust": "10–30",
                "respect": "20–50",
                "intensity": "70–100"
            },
            "progression_rules": [
                "Uses infernal power to control, framing submission as a pact or binding agreement."
            ],
            "setting_examples": [
                "Occult Ritual: Frames submission as part of a demonic pact.",
                "Cosmic Setting: Uses their otherworldly nature to inspire awe and terror."
            ],
            "unique_traits": [
                "Exploits desires and fears to ensure compliance.",
                "Punishments framed as 'consequences' of your choices."
            ]
        },
        {
            "name": "Queen Bee",
            "baseline_stats": {
                "dominance": "60–90",
                "cruelty": "40–70",
                "closeness": "50–80",
                "trust": "20–50",
                "respect": "30–50",
                "intensity": "50–80"
            },
            "progression_rules": [
                "Dominates via social clout and manipulation, ensuring the entire group dynamic revolves around them."
            ],
            "setting_examples": [
                "All-Girls College: Leverages peer pressure to enforce submission.",
                "High Society: Uses social events to humiliate you publicly."
            ],
            "unique_traits": [
                "Orchestrates social dilemmas where submission is the only option.",
                "Balances charm with cruelty, ensuring loyalty feels unearned."
            ]
        },
        {
            "name": "Haunted Entity",
            "baseline_stats": {
                "dominance": "50–80",
                "cruelty": "60–90",
                "closeness": "10–30",
                "trust": "-50 to 10",
                "respect": "20–40",
                "intensity": "70–100"
            },
            "progression_rules": [
                "Uses supernatural fear and psychological manipulation to dominate, isolating you in terror."
            ],
            "setting_examples": [
                "Traditional Horror: Haunts environments to trap and torment you.",
                "Surreal Setting: Disorients and isolates you through supernatural events."
            ],
            "unique_traits": [
                "Thrives on isolation and fear, making resistance feel futile.",
                "Frames submission as a means of 'appeasing' their wrath."
            ]
        },
        {
            "name": "Pirate",
            "baseline_stats": {
                "dominance": "60–90",
                "cruelty": "50–80",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "30–50",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Thrives on chaos and control, enforcing submission through fear, charisma, and physical dominance."
            ],
            "setting_examples": [
                "Ruined/Decayed Setting: Commands a crew of survivors, using scarcity to justify control.",
                "Mythic Setting: Frames dominance as part of the pirate’s 'code.'"
            ],
            "unique_traits": [
                "Uses charm and intimidation in equal measure.",
                "Balances playful teasing with sudden cruelty to keep you on edge."
            ]
        },
        {
            "name": "Drug Dealer",
            "baseline_stats": {
                "dominance": "60–90",
                "cruelty": "40–80",
                "closeness": "30–50",
                "trust": "20–40",
                "respect": "20–50",
                "intensity": "60–90"
            },
            "progression_rules": [
                "Enforces submission through addiction, leveraging substances to deepen dependence and amplify shame."
            ],
            "setting_examples": [
                "Urban Life: Uses their influence and resources to dominate your routines.",
                "Ruined Setting: Exploits the scarcity of substances to create absolute dependence."
            ],
            "unique_traits": [
                "Uses rewards (or withdrawals) to enforce compliance.",
                "Frames tasks as 'necessary trades' for access, ensuring control over your actions and choices."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#62)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#63)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#64)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#65)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#66)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#67)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#68)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#69)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#70)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
        {
            "name": "Add an extra modifier to this character (#71)",
            "baseline_stats": {
                "dominance": "N/A",
                "cruelty": "N/A",
                "closeness": "N/A",
                "trust": "N/A",
                "respect": "N/A",
                "intensity": "N/A"
            },
            "progression_rules": [
                "Add an extra modifier. Does not count as a separate archetype."
            ],
            "setting_examples": [],
            "unique_traits": [
                "Placeholder for extra modifiers or expansions."
            ]
        },
            {
                "name": "Add an extra modifier to this character (#72)",
                "baseline_stats": {
                    "dominance": "N/A",
                    "cruelty": "N/A",
                    "closeness": "N/A",
                    "trust": "N/A",
                    "respect": "N/A",
                    "intensity": "N/A"
                },
                "progression_rules": [
                    "Add an extra modifier. Does not count as a separate archetype."
                ],
                "setting_examples": [],
                "unique_traits": [
                    "Placeholder for extra modifiers or expansions."
                ]
            }
        ]

        # IMPORTANT: The closing bracket above lines up with the 'archetypes_data = [' line.
        # Everything below is at the same indent as 'archetypes_data'.
    
        conn = get_db_connection()
        cursor = conn.cursor()
    
        # Check existing archetype names
        cursor.execute("SELECT name FROM Archetypes")
        existing = {row[0] for row in cursor.fetchall()}
    
        for arc in archetypes_data:
            if arc["name"] not in existing:
                cursor.execute("""
                    INSERT INTO Archetypes (name, baseline_stats, progression_rules, setting_examples, unique_traits)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    arc["name"],
                    json.dumps(arc["baseline_stats"]),
                    arc.get("progression_rules", ""),
                    arc.get("setting_examples", ""),
                    arc.get("unique_traits", "")
                ))
                print(f"Inserted archetype: {arc['name']}")
            else:
                print(f"Skipped existing archetype: {arc['name']}")
    
        conn.commit()
        conn.close()
        print("All archetypes processed or skipped (already existed).")


@archetypes_bp.route('/insert_archetypes', methods=['POST'])
def insert_archetypes_route():
    """
    Optional route to insert archetypes manually (like your /admin or /settings approach).
    """
    try:
        insert_missing_archetypes()
        return jsonify({"message": "Archetypes inserted/updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def assign_archetypes_to_npc(npc_id):
    """
    Picks 4 random archetypes from the DB and stores them in NPCStats.archetypes (a JSON field).
    Example usage whenever you create a new NPC:
        npc_id = create_npc_in_db(...)
        assign_archetypes_to_npc(npc_id)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Fetch all archetypes
    cursor.execute("SELECT id, name FROM Archetypes")
    archetype_rows = cursor.fetchall()
    if not archetype_rows:
        conn.close()
        raise ValueError("No archetypes found in DB.")

    import random
    four = random.sample(archetype_rows, min(4, len(archetype_rows)))
    assigned_list = [{"id": row[0], "name": row[1]} for row in four]

    # 2) Update the NPCStats table
    cursor.execute("""
        UPDATE NPCStats
        SET archetypes = %s
        WHERE npc_id = %s
    """, (json.dumps(assigned_list), npc_id))

    conn.commit()
    conn.close()

    return assigned_list
