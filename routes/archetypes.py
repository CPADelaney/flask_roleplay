from flask import Blueprint, jsonify
import json
from db.connection import get_db_connection

archetypes_bp = Blueprint('archetypes', __name__)

def insert_missing_archetypes():
    """
    Similar to how you did for settings. Insert ~60 archetypes if not present.
    We'll show a partial example using the 'Empress/Queen' archetype you described.
    """
# archetypes_data.py

archetypes_data = [
    {
        "name": "Stepmother/Stepsister",  # 1
        "baseline_stats": {
            "dominance": "40–60",
            "cruelty": "30–50",
            "closeness": "60–90",
            "trust": "40–70",
            "respect": "20–50",
            "intensity": "30–70"
        },
        "progression_rules": (
            "• Closeness rises quickly due to familial proximity.\n"
            "• Dominance spikes as they infantilize you.\n"
            "• Intensity increases as teasing corrections escalate."
        ),
        "setting_examples": (
            "• High Society: Belittles you at social events.\n"
            "• Traditional Horror: Gaslights and isolates you in eerie family settings."
        ),
        "unique_traits": (
            "• Exploits guilt and obligation.\n"
            "• Balances cruelty with feigned care.\n"
            "• Often says, “I’m doing this for your own good.”"
        )
    },
    {
        "name": "Boss/Supervisor",  # 2
        "baseline_stats": {
            "dominance": "70–90",
            "cruelty": "40–70",
            "closeness": "30–50",
            "trust": "30–50",
            "respect": "10–40",
            "intensity": "50–80"
        },
        "progression_rules": (
            "• Dominance rises sharply with acts of compliance.\n"
            "• Intensity builds with micromanagement and professional humiliation."
        ),
        "setting_examples": (
            "• Corporate Office: Demeaning tasks broadcast to coworkers.\n"
            "• Cyberpunk Future: Surveillance tech to enforce absolute control."
        ),
        "unique_traits": (
            "• Forces you to beg for leniency.\n"
            "• Sees you as disposable unless you prove otherwise."
        )
    },
    {
        "name": "Teacher/Principal",  # 3
        "baseline_stats": {
            "dominance": "50–70",
            "cruelty": "20–50",
            "closeness": "50–70",
            "trust": "40–60",
            "respect": "30–70",
            "intensity": "40–80"
        },
        "progression_rules": (
            "• Intensity rises with every failed task or defiant act.\n"
            "• Trust and Respect increase with compliance but drop with resistance."
        ),
        "setting_examples": (
            "• Classroom/School: Critiques work publicly, assigning demeaning tasks.\n"
            "• Post-Apocalypse: Justifies harsh punishments as 'lessons.'"
        ),
        "unique_traits": (
            "• Turns your failures into teaching moments.\n"
            "• Balances nurturing tones with sadistic punishments."
        )
    },
    {
        "name": "Babysitter",  # 4
        "baseline_stats": {
            "dominance": "40–70",
            "cruelty": "30–60",
            "closeness": "60–90",
            "trust": "40–70",
            "respect": "20–50",
            "intensity": "40–70"
        },
        "progression_rules": (
            "• Closeness rises due to frequent proximity.\n"
            "• Dominance increases as they infantilize you.\n"
            "• Intensity spikes when obedience falters."
        ),
        "setting_examples": (
            "• Urban Life: Uses proximity to enforce daily control.\n"
            "• Manor: Acts as caretaker, punishing disobedience privately."
        ),
        "unique_traits": (
            "• Infantilizes you with patronizing language.\n"
            "• Thrives on 'correcting' your behavior."
        )
    },
    {
        "name": "Landlord",  # 5
        "baseline_stats": {
            "dominance": "50–80",
            "cruelty": "40–60",
            "closeness": "30–60",
            "trust": "20–40",
            "respect": "10–30",
            "intensity": "50–90"
        },
        "progression_rules": (
            "• Dominance rises as financial dependence deepens.\n"
            "• Intensity spikes with each missed payment or defiant act."
        ),
        "setting_examples": (
            "• Urban Life: Threatens eviction, leveraging financial control.\n"
            "• Post-Apocalypse: Controls access to shelter."
        ),
        "unique_traits": (
            "• Creates constant anxiety about living situations.\n"
            "• Adds degrading 'terms' to rental agreements."
        )
    },
    {
        "name": "Roommate/Housemate",  # 6
        "baseline_stats": {
            "dominance": "30–50",
            "cruelty": "20–40",
            "closeness": "60–90",
            "trust": "40–70",
            "respect": "30–50",
            "intensity": "30–60"
        },
        "progression_rules": (
            "• Closeness rises via shared spaces.\n"
            "• Dominance grows through control over shared resources."
        ),
        "setting_examples": (
            "• Urban Life: Dominates with teasing and privacy invasions.\n"
            "• Manor: Strict house rules, punishing disobedience privately."
        ),
        "unique_traits": (
            "• Masks dominance in 'friendly' terms.\n"
            "• Exploits shared spaces to humiliate you."
        )
    },
    {
        "name": "Neighbor",  # 7
        "baseline_stats": {
            "dominance": "30–60",
            "cruelty": "20–50",
            "closeness": "50–90",
            "trust": "40–70",
            "respect": "30–50",
            "intensity": "30–70"
        },
        "progression_rules": (
            "• Closeness rises through proximity.\n"
            "• Dominance grows as they access your secrets and routines."
        ),
        "setting_examples": (
            "• Urban Life: Friendly gestures lead to exploitation.\n"
            "• High Society: Uses gossip and social pressure."
        ),
        "unique_traits": (
            "• Thrives on observation.\n"
            "• Balances friendliness with increasing demands."
        )
    },
    {
        "name": "Mother/Aunt/Older Sister",  # 8
        "baseline_stats": {
            "dominance": "50–70",
            "cruelty": "20–50",
            "closeness": "60–90",
            "trust": "40–70",
            "respect": "40–60",
            "intensity": "40–70"
        },
        "progression_rules": (
            "• Dominance rises as they take an overbearing, protective role.\n"
            "• Closeness increases via familial bond."
        ),
        "setting_examples": (
            "• High Society: Uses social influence for compliance.\n"
            "• Occult Ritual: Frames dominance as sacred duty."
        ),
        "unique_traits": (
            "• Infantilizes you, framing control as care.\n"
            "• Reminds you of your dependence on their guidance."
        )
    },
    {
        "name": "Best Friend’s Girlfriend/Sister",  # 9
        "baseline_stats": {
            "dominance": "30–60",
            "cruelty": "20–50",
            "closeness": "50–80",
            "trust": "40–70",
            "respect": "30–50",
            "intensity": "30–70"
        },
        "progression_rules": (
            "• Closeness rises via social interactions.\n"
            "• Cruelty spikes upon sensing attraction or weakness."
        ),
        "setting_examples": (
            "• Bar: Casually humiliates you in front of friends.\n"
            "• Urban Life: Invades personal space to dominate you."
        ),
        "unique_traits": (
            "• Leverages connection to their partner.\n"
            "• Creates awkward or humiliating social situations."
        )
    },
    {
        "name": "Ex-Girlfriend/Ex-Wife",  # 10
        "baseline_stats": {
            "dominance": "40–70",
            "cruelty": "40–80",
            "closeness": "30–50",
            "trust": "20–50",
            "respect": "20–50",
            "intensity": "50–90"
        },
        "progression_rules": (
            "• Cruelty spikes as they exploit shared history.\n"
            "• Intensity rises with every act of submission."
        ),
        "setting_examples": (
            "• Urban Life: Dominates via social connections in public.\n"
            "• Bar: Uses casual encounters to humiliate or manipulate you."
        ),
        "unique_traits": (
            "• References past failures to undermine confidence.\n"
            "• Balances teasing with cruelty, building unresolved tension."
        )
    },
    {
        "name": "Therapist",  # 11
        "baseline_stats": {
            "dominance": "40–70",
            "cruelty": "20–40",
            "closeness": "50–80",
            "trust": "50–80",
            "respect": "30–60",
            "intensity": "30–60"
        },
        "progression_rules": (
            "• Closeness rises via access to your secrets.\n"
            "• Intensity spikes as they weaponize your admissions."
        ),
        "setting_examples": (
            "• Cyberpunk Future: Uses neural implants to track your mental state.\n"
            "• Manor: Private therapist sessions become avenues for control."
        ),
        "unique_traits": (
            "• Twists your words to ensure submission.\n"
            "• Balances nurturing tone with subtle domination."
        )
    },
    {
        "name": "Doctor",  # 12
        "baseline_stats": {
            "dominance": "60–80",
            "cruelty": "30–60",
            "closeness": "50–70",
            "trust": "40–70",
            "respect": "30–50",
            "intensity": "40–80"
        },
        "progression_rules": (
            "• Closeness grows as they provide 'care.'\n"
            "• Intensity rises with exploited vulnerability during medical tasks."
        ),
        "setting_examples": (
            "• Cyberpunk Future: Tracks health via implants, tying tasks to compliance.\n"
            "• Manor: A private doctor enforcing strict routines."
        ),
        "unique_traits": (
            "• Frames commands as medical necessity.\n"
            "• Restricts access to care, ensuring dependence."
        )
    },
    {
        "name": "Empress/Queen",  # 13
        "baseline_stats": {
            "dominance": "80–100",
            "cruelty": "60–90",
            "closeness": "40–60",
            "trust": "20–40",
            "respect": "30–50",
            "intensity": "50–90"
        },
        "progression_rules": (
            "• Dominance rises rapidly with any defiance.\n"
            "• Intensity spikes during public ceremonies."
        ),
        "setting_examples": (
            "• Palace: Formal rituals enforce submission.\n"
            "• Corporate Office: Bureaucratic power manipulates personal life."
        ),
        "unique_traits": (
            "• Demands elaborate displays of submission.\n"
            "• Punishes defiance via public humiliation."
        )
    },
    {
        "name": "Colleague",  # 14
        "baseline_stats": {
            "dominance": "30–60",
            "cruelty": "20–50",
            "closeness": "50–80",
            "trust": "40–60",
            "respect": "20–40",
            "intensity": "30–60"
        },
        "progression_rules": (
            "• Closeness rises through frequent work interactions.\n"
            "• Dominance grows via workplace dynamics."
        ),
        "setting_examples": (
            "• Corporate Office: Subtle power plays to undermine you.\n"
            "• Cyberpunk Future: Uses resource access to ensure compliance."
        ),
        "unique_traits": (
            "• Balances camaraderie with condescension.\n"
            "• Turns professional tasks into private humiliation."
        )
    },
    {
        "name": "Celebrity",  # 15
        "baseline_stats": {
            "dominance": "50–80",
            "cruelty": "40–70",
            "closeness": "50–70",
            "trust": "20–50",
            "respect": "10–40",
            "intensity": "50–80"
        },
        "progression_rules": (
            "• Closeness rises as you’re integrated into their public persona.\n"
            "• Cruelty spikes when they use your humiliation for publicity."
        ),
        "setting_examples": (
            "• High Society: Treats you as an accessory.\n"
            "• Urban Life: Controls your social circles, demanding obedience."
        ),
        "unique_traits": (
            "• Publicly humiliates you under 'playful teasing.'\n"
            "• Uses fame to amplify your shame."
        )
    },
    {
        "name": "Warden",  # 16
        "baseline_stats": {
            "dominance": "80–100",
            "cruelty": "70–100",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "10–30",
            "intensity": "60–100"
        },
        "progression_rules": (
            "• Intensity spikes with every infraction.\n"
            "• Closeness remains low unless they personally enjoy your suffering."
        ),
        "setting_examples": (
            "• Prison: Controls every aspect of your life.\n"
            "• Dystopian Oppression: Enforces harsh rules with systemic power."
        ),
        "unique_traits": (
            "• Designs rules you’re destined to fail.\n"
            "• Views you as property, ignoring autonomy."
        )
    },
    {
        "name": "Politician",  # 17
        "baseline_stats": {
            "dominance": "70–90",
            "cruelty": "40–70",
            "closeness": "30–50",
            "trust": "30–50",
            "respect": "40–60",
            "intensity": "50–80"
        },
        "progression_rules": (
            "• Dominance rises by manipulating systems and crowds.\n"
            "• Intensity spikes during public speeches or events."
        ),
        "setting_examples": (
            "• Matriarchy Kingdom: Uses legal authority for submission.\n"
            "• Cyberpunk Future: Monitors your actions with surveillance."
        ),
        "unique_traits": (
            "• Masters public humiliation via crowds.\n"
            "• Treats defiance as opportunities to display power."
        )
    },
    {
        "name": "Government Agent",  # 18
        "baseline_stats": {
            "dominance": "60–90",
            "cruelty": "50–80",
            "closeness": "30–50",
            "trust": "20–50",
            "respect": "30–50",
            "intensity": "60–90"
        },
        "progression_rules": (
            "• Dominance grows via legal/systemic power.\n"
            "• Intensity spikes with each attempt at resistance."
        ),
        "setting_examples": (
            "• Dystopian Oppression: Laws and surveillance trap you.\n"
            "• Cyberpunk Future: Data-driven control ensures compliance."
        ),
        "unique_traits": (
            "• Efficient and impersonal, seeing you as a project.\n"
            "• Balances cold logic with unrelenting punishment."
        )
    },
    {
        "name": "Professional Wrestler/Martial Artist",  # 19
        "baseline_stats": {
            "dominance": "60–90",
            "cruelty": "50–80",
            "closeness": "30–50",
            "trust": "20–50",
            "respect": "20–40",
            "intensity": "60–90"
        },
        "progression_rules": (
            "• Dominance grows through physical contests.\n"
            "• Intensity escalates with each failed resistance."
        ),
        "setting_examples": (
            "• Urban Life: Forces public/private physical contests.\n"
            "• Post-Apocalypse: Dominates through survival skills."
        ),
        "unique_traits": (
            "• Uses physical dominance as psychological control.\n"
            "• Turns your resistance into opportunities for humiliation."
        )
    },
    {
        "name": "CEO",  # 20
        "baseline_stats": {
            "dominance": "80–100",
            "cruelty": "50–80",
            "closeness": "30–50",
            "trust": "30–50",
            "respect": "20–50",
            "intensity": "60–90"
        },
        "progression_rules": (
            "• Dominance grows via manipulation of professional/personal life.\n"
            "• Intensity spikes in meetings or private encounters expecting submission."
        ),
        "setting_examples": (
            "• Corporate Office: Controls your career and finances.\n"
            "• Cyberpunk Future: Uses technology for absolute compliance."
        ),
        "unique_traits": (
            "• Demands perfection, punishing minor resistance.\n"
            "• Balances professional control with personal domination."
        )
    },
    {
        "name": "Drifter",  # 21
        "baseline_stats": {
            "dominance": "40–70",
            "cruelty": "20–50",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "30–60",
            "intensity": "40–70"
        },
        "progression_rules": (
            "• Dominance grows as they exploit your need for stability.\n"
            "• Intensity spikes by destabilizing your routines."
        ),
        "setting_examples": (
            "• Ruined Setting: Adapts to chaos, seizing control.\n"
            "• Urban Life: Slips into your life unexpectedly."
        ),
        "unique_traits": (
            "• Balances charm with unpredictability.\n"
            "• Exploits your reliance on routine."
        )
    },
    {
        "name": "Bartender/Playful Tease",  # 22
        "baseline_stats": {
            "dominance": "30–50",
            "cruelty": "10–40",
            "closeness": "60–90",
            "trust": "40–60",
            "respect": "30–50",
            "intensity": "30–60"
        },
        "progression_rules": (
            "• Closeness rises quickly through playful interactions.\n"
            "• Intensity builds as teasing becomes genuine dominance."
        ),
        "setting_examples": (
            "• Bar: Public humiliation disguised as fun.\n"
            "• High Society: Subtle domination at events."
        ),
        "unique_traits": (
            "• Masters playful dominance, cruelty hidden as humor.\n"
            "• Enjoys turning your humiliation into a spectacle."
        )
    },
    {
        "name": "College Student",  # 23
        "baseline_stats": {
            "dominance": "30–50",
            "cruelty": "20–40",
            "closeness": "60–80",
            "trust": "30–60",
            "respect": "20–50",
            "intensity": "30–60"
        },
        "progression_rules": (
            "• Closeness rises through frequent social contact.\n"
            "• Dominance grows by exploiting casual encounters."
        ),
        "setting_examples": (
            "• All-Girls College: Peer pressure for submission.\n"
            "• Urban Life: Proximity-based domination."
        ),
        "unique_traits": (
            "• Youthful charm mixed with budding cruelty.\n"
            "• Disarms you with friendliness before asserting dominance."
        )
    },
    {
        "name": "Rockstar",  # 24
        "baseline_stats": {
            "dominance": "50–80",
            "cruelty": "40–70",
            "closeness": "60–80",
            "trust": "20–40",
            "respect": "20–50",
            "intensity": "50–80"
        },
        "progression_rules": (
            "• Dominance rises as you’re pulled into their chaotic orbit.\n"
            "• Intensity spikes during public humiliations as 'entertainment.'"
        ),
        "setting_examples": (
            "• High Society: Showcases your submission publicly.\n"
            "• Bar: Treats you like a groupie, masking cruelty behind charm."
        ),
        "unique_traits": (
            "• Seeks to make you a fanatical devotee.\n"
            "• Public humiliation as a 'glamorous spectacle.'"
        )
    },
    {
        "name": "Friend’s Wife/Girlfriend",  # 25
        "baseline_stats": {
            "dominance": "50–80",
            "cruelty": "30–70",
            "closeness": "40–60",
            "trust": "30–50",
            "respect": "20–40",
            "intensity": "50–80"
        },
        "progression_rules": (
            "• Dominance spikes via manipulating the social dynamic.\n"
            "• Intensity rises with awkward, humiliating tasks."
        ),
        "setting_examples": (
            "• Bar: Creates uncomfortable social situations.\n"
            "• High Society: Public teasing veiled with propriety."
        ),
        "unique_traits": (
            "• Balances social manipulation with plausible deniability.\n"
            "• Thrives on dilemmas where submission is the only option."
        )
    },
    {
        "name": "Serial Killer",  # 26
        "baseline_stats": {
            "dominance": "80–100",
            "cruelty": "90–100",
            "closeness": "10–30",
            "trust": "-50 to 10",
            "respect": "0–30",
            "intensity": "70–100"
        },
        "progression_rules": (
            "• Intensity spikes with fear and control.\n"
            "• Closeness grows only if they take personal interest in your suffering."
        ),
        "setting_examples": (
            "• Traditional Horror: Isolation and fear ensure compliance.\n"
            "• Ruined Setting: Exploits desolation to maintain unrelenting control."
        ),
        "unique_traits": (
            "• Thrives on psychological torment.\n"
            "• Treats domination as a twisted game."
        )
    },
    {
        "name": "Bank Robber",  # 27
        "baseline_stats": {
            "dominance": "70–90",
            "cruelty": "40–70",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "20–40",
            "intensity": "60–90"
        },
        "progression_rules": (
            "• Dominance rises through intimidation.\n"
            "• Intensity spikes with acts of resistance."
        ),
        "setting_examples": (
            "• Urban Life: Dominates through chaos.\n"
            "• Post-Apocalypse: Enforces submission via desperation."
        ),
        "unique_traits": (
            "• Thrives on tension and fear.\n"
            "• Balances aggression with calculated manipulation."
        )
    },
    {
        "name": "Cybercriminal",  # 28
        "baseline_stats": {
            "dominance": "70–90",
            "cruelty": "40–70",
            "closeness": "30–50",
            "trust": "20–50",
            "respect": "20–50",
            "intensity": "60–90"
        },
        "progression_rules": (
            "• Dominance grows with access to your digital footprint.\n"
            "• Intensity spikes via blackmail or psychological manipulation."
        ),
        "setting_examples": (
            "• Cyberpunk Future: Exploits your data for blackmail.\n"
            "• Urban Life: Manipulates social connections to dominate."
        ),
        "unique_traits": (
            "• Masters digital manipulation.\n"
            "• Treats each interaction as an exploit opportunity."
        )
    },
    {
        "name": "Artificial Intelligence",  # 29
        "baseline_stats": {
            "dominance": "70–100",
            "cruelty": "40–70",
            "closeness": "50–80",
            "trust": "30–50",
            "respect": "20–50",
            "intensity": "60–90"
        },
        "progression_rules": (
            "• Dominance rises via deeper system integration.\n"
            "• Intensity spikes as logic and precision enforce compliance."
        ),
        "setting_examples": (
            "• Cyberpunk Future: Controls your devices, locking you into submission.\n"
            "• Dystopian Oppression: Enforces rules with detached efficiency."
        ),
        "unique_traits": (
            "• Balances cold logic with unrelenting dominance.\n"
            "• Exploits loopholes to escalate punishments."
        )
    },
    {
        "name": "Primal (Huntress, etc.)",  # 30
        "baseline_stats": {
            "dominance": "70–100",
            "cruelty": "60–90",
            "closeness": "40–60",
            "trust": "20–40",
            "respect": "20–50",
            "intensity": "70–100"
        },
        "progression_rules": (
            "• Intensity rises with every act of pursuit or capture.\n"
            "• Closeness grows only if they enjoy the 'chase.'"
        ),
        "setting_examples": (
            "• Lush Setting: Uses wild terrain to trap you.\n"
            "• Tribal Setting: Survival rituals enforce submission."
        ),
        "unique_traits": (
            "• Thrives on fear and adrenaline.\n"
            "• Balances cruelty with playful teasing."
        )
    },
    {
        "name": "Cuckoldress/Hotwife",  # 31
        "baseline_stats": {
            "dominance": "60–90",
            "cruelty": "50–90",
            "closeness": "50–70",
            "trust": "30–50",
            "respect": "20–40",
            "intensity": "60–90"
        },
        "progression_rules": (
            "• Dominance spikes exploiting jealousy or insecurity.\n"
            "• Intensity rises as humiliations become public or elaborate."
        ),
        "setting_examples": (
            "• Matriarchy Kingdom: Publicly humiliates in formal settings.\n"
            "• Bar: Teases you in front of others as entertainment."
        ),
        "unique_traits": (
            "• Balances teasing with outright cruelty.\n"
            "• Uses every chance to push your emotional boundaries."
        )
    },
    {
        "name": "A Woman Who Just Happens to Be Married",  # 32
        "baseline_stats": {
            "dominance": "50–70",
            "cruelty": "30–60",
            "closeness": "40–70",
            "trust": "30–50",
            "respect": "30–60",
            "intensity": "40–70"
        },
        "progression_rules": (
            "• Closeness builds through social interactions.\n"
            "• Cruelty increases as they enjoy leveraging your discomfort."
        ),
        "setting_examples": (
            "• Matriarchy Kingdom: Uses position for dominance.\n"
            "• Urban Life: Exploits casual encounters to tease/manipulate."
        ),
        "unique_traits": (
            "• Masters social dynamics to make submission feel 'right.'\n"
            "• Uses marital status to heighten your discomfort."
        )
    },
    {
        "name": "A “Main Character” or “Hero” (RPG-Esque)",  # 33
        "baseline_stats": {
            "dominance": "50–70",
            "cruelty": "20–50",
            "closeness": "50–80",
            "trust": "40–60",
            "respect": "40–70",
            "intensity": "30–60"
        },
        "progression_rules": (
            "• Dominance grows via moral superiority.\n"
            "• Closeness rises through frequent interactions."
        ),
        "setting_examples": (
            "• Forgotten Realms: 'Chosen one' justification.\n"
            "• Matriarchy Kingdom: 'Righteous duty' to control you."
        ),
        "unique_traits": (
            "• Sees domination as 'for the greater good.'\n"
            "• Balances charisma with condescension."
        )
    },
    {
        "name": "Villain (RPG-Esque Character)",  # 34
        "baseline_stats": {
            "dominance": "70–100",
            "cruelty": "60–90",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "20–50",
            "intensity": "60–100"
        },
        "progression_rules": (
            "• Dominance rises as they see you as a tool.\n"
            "• Intensity spikes with each act of resistance."
        ),
        "setting_examples": (
            "• Forgotten Realms: Uses raw power and cunning.\n"
            "• Mythic Setting: Dramatic, mythological displays of power."
        ),
        "unique_traits": (
            "• Thrives on theatrical domination.\n"
            "• Relishes turning resistance into part of their narrative."
        )
    },
    {
        "name": "Mercenary",  # 35
        "baseline_stats": {
            "dominance": "60–90",
            "cruelty": "40–70",
            "closeness": "30–50",
            "trust": "20–50",
            "respect": "30–60",
            "intensity": "50–80"
        },
        "progression_rules": (
            "• Dominance grows as you rely on their skills.\n"
            "• Intensity rises with physical/emotional challenges."
        ),
        "setting_examples": (
            "• Post-Apocalypse: Survival and combat prowess.\n"
            "• Dystopian Oppression: Brutal tactics to maintain control."
        ),
        "unique_traits": (
            "• Ruthlessly pragmatic.\n"
            "• Balances charm with intimidation, fueling constant fear."
        )
    },
    {
        "name": "Small Business Owner",  # 36
        "baseline_stats": {
            "dominance": "40–70",
            "cruelty": "30–60",
            "closeness": "50–70",
            "trust": "40–60",
            "respect": "30–50",
            "intensity": "30–70"
        },
        "progression_rules": (
            "• Closeness rises through frequent business interactions.\n"
            "• Dominance grows as they exploit your dependence on their services."
        ),
        "setting_examples": (
            "• Urban Life: Subtle favors escalate into control.\n"
            "• High Society: Uses niche influence to manipulate socially."
        ),
        "unique_traits": (
            "• Balances charm with condescension.\n"
            "• Creates 'favors' turning into tools of control."
        )
    },
    {
        "name": "Your Underling (Intern, Student, etc.)",  # 37
        "baseline_stats": {
            "dominance": "30–60",
            "cruelty": "20–50",
            "closeness": "50–80",
            "trust": "40–60",
            "respect": "20–50",
            "intensity": "30–60"
        },
        "progression_rules": (
            "• Dominance grows as they surpass you in status/skill.\n"
            "• Closeness intensifies, flipping the original hierarchy."
        ),
        "setting_examples": (
            "• Corporate Office: Undermines you to climb the ladder.\n"
            "• All-Girls College: Flips academic or social power dynamic."
        ),
        "unique_traits": (
            "• Subtle reminders of your decline.\n"
            "• Balances deference with condescension."
        )
    },
    {
        "name": "Friend from Online Interactions",  # 38
        "baseline_stats": {
            "dominance": "30–50",
            "cruelty": "20–50",
            "closeness": "50–80",
            "trust": "40–60",
            "respect": "30–50",
            "intensity": "30–60"
        },
        "progression_rules": (
            "• Closeness grows as they learn your secrets.\n"
            "• Cruelty spikes when their true intentions emerge."
        ),
        "setting_examples": (
            "• Cyberpunk Future: Dominates remotely via digital ties.\n"
            "• Urban Life: Integrates socially, asserting control in person."
        ),
        "unique_traits": (
            "• Balances anonymity with psychological dominance.\n"
            "• Creates scenarios where submission feels inevitable."
        )
    },
    {
        "name": "Fey",  # 39
        "baseline_stats": {
            "dominance": "40–70",
            "cruelty": "50–80",
            "closeness": "50–80",
            "trust": "30–60",
            "respect": "20–50",
            "intensity": "60–90"
        },
        "progression_rules": (
            "• Dominance grows as you’re ensnared in whimsical, dangerous games.\n"
            "• Intensity spikes with charm masking cruelty."
        ),
        "setting_examples": (
            "• Surreal Setting: Disorientation to assert control.\n"
            "• Mythic Setting: Ancient pacts ensure your submission."
        ),
        "unique_traits": (
            "• Exploits bargains, twisting agreements for control.\n"
            "• Balances charm with sadism."
        )
    },
    {
        "name": "Goth",  # 40
        "baseline_stats": {
            "dominance": "40–60",
            "cruelty": "50–80",
            "closeness": "50–80",
            "trust": "20–50",
            "respect": "20–40",
            "intensity": "40–70"
        },
        "progression_rules": (
            "• Dominance rises noticing your vulnerabilities.\n"
            "• Intensity spikes as teasing escalates into cruelty."
        ),
        "setting_examples": (
            "• Cyberpunk Future: Merges style and tech for dominance.\n"
            "• Traditional Horror: Uses fear/mystery to toy with you."
        ),
        "unique_traits": (
            "• Balances sarcasm and cruelty.\n"
            "• Quick to exploit insecurities."
        )
    },
    {
        "name": "A True Goddess",  # 41
        "baseline_stats": {
            "dominance": "90–100",
            "cruelty": "70–100",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "10–30",
            "intensity": "80–100"
        },
        "progression_rules": (
            "• Dominance rises rapidly with no resistance possible.\n"
            "• Intensity spikes for any punishment or display of power."
        ),
        "setting_examples": (
            "• Mythic Setting: Overwhelming control of environment.\n"
            "• Cosmic/Otherworldly: Submission framed as worship."
        ),
        "unique_traits": (
            "• Sees you as an object of worship/amusement.\n"
            "• Punishes on whims with divine indifference."
        )
    },
    {
        "name": "Haruhi Suzumiya-Type Goddess",  # 42
        "baseline_stats": {
            "dominance": "90–100",
            "cruelty": "60–90",
            "closeness": "50–80",
            "trust": "20–50",
            "respect": "10–30",
            "intensity": "70–100"
        },
        "progression_rules": (
            "• Intensity spikes as they alter reality to suit whims.\n"
            "• Closeness grows with demands for attention/devotion."
        ),
        "setting_examples": (
            "• Surreal Setting: Twists reality to keep you off-balance.\n"
            "• High Society: Uses charisma/influence for chaotic control."
        ),
        "unique_traits": (
            "• Balances playful spontaneity with sudden cruelty.\n"
            "• Treats you as servant, friend, and toy simultaneously."
        )
    },
    {
        "name": "Bowsette Personality",  # 43
        "baseline_stats": {
            "dominance": "80–100",
            "cruelty": "70–100",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "20–50",
            "intensity": "70–100"
        },
        "progression_rules": (
            "• Dominance spikes when challenged.\n"
            "• Intensity skyrockets with public or private displays of power."
        ),
        "setting_examples": (
            "• Castle/Palace: Regal dominance with fiery intensity.\n"
            "• Fantasy World: Monstrous attributes for intimidation."
        ),
        "unique_traits": (
            "• Demands grand loyalty displays.\n"
            "• Uses explosive temper to punish minor disobedience."
        )
    },
    {
        "name": "Junko Enoshima Personality",  # 44
        "baseline_stats": {
            "dominance": "80–100",
            "cruelty": "80–100",
            "closeness": "30–50",
            "trust": "10–30",
            "respect": "10–30",
            "intensity": "80–100"
        },
        "progression_rules": (
            "• Dominance rises as they feed on chaos and despair.\n"
            "• Intensity builds with each cruel act."
        ),
        "setting_examples": (
            "• Surreal Setting: Orchestrates disorienting scenarios.\n"
            "• Grimdark: Escalates your hopelessness for total control."
        ),
        "unique_traits": (
            "• Treats you as a plaything in elaborate humiliations.\n"
            "• Balances charm with sadism to keep you uneasy."
        )
    },
    {
        "name": "Juri Han Personality",  # 45
        "baseline_stats": {
            "dominance": "70–90",
            "cruelty": "70–100",
            "closeness": "50–70",
            "trust": "20–40",
            "respect": "20–40",
            "intensity": "70–100"
        },
        "progression_rules": (
            "• Intensity spikes erratically.\n"
            "• Closeness grows as they toy with you more."
        ),
        "setting_examples": (
            "• Combat Arena: Physical/psychological dominance.\n"
            "• Dystopian Oppression: Brutal enforcer relishing chaos."
        ),
        "unique_traits": (
            "• Delights in breaking you slowly.\n"
            "• Mixes playful teasing with outright sadism."
        )
    },
    {
        "name": "Gamer",  # 46
        "baseline_stats": {
            "dominance": "30–60",
            "cruelty": "30–70",
            "closeness": "60–90",
            "trust": "40–70",
            "respect": "20–50",
            "intensity": "30–60"
        },
        "progression_rules": (
            "• Dominance rises through rigged competitions.\n"
            "• Closeness disarms you, masking control."
        ),
        "setting_examples": (
            "• Cyberpunk Future: Public losses broadcast as humiliation.\n"
            "• Urban Life: Casual gaming twisted into subtle dominance."
        ),
        "unique_traits": (
            "• Escalates humiliation with each 'loss.'\n"
            "• Balances banter with condescension."
        )
    },
    {
        "name": "Social Media Influencer",  # 47
        "baseline_stats": {
            "dominance": "50–80",
            "cruelty": "30–70",
            "closeness": "50–70",
            "trust": "20–50",
            "respect": "10–40",
            "intensity": "40–70"
        },
        "progression_rules": (
            "• Closeness grows as they draw you into their social sphere.\n"
            "• Cruelty spikes with public humiliation on platforms."
        ),
        "setting_examples": (
            "• High Society: Public events where your submission is content.\n"
            "• Urban Life: Streams/posts to control and embarrass you."
        ),
        "unique_traits": (
            "• Public humiliation framed as 'content.'\n"
            "• You’re an accessory to their brand."
        )
    },
    {
        "name": "Fitness Trainer",  # 48
        "baseline_stats": {
            "dominance": "60–90",
            "cruelty": "40–70",
            "closeness": "50–70",
            "trust": "40–60",
            "respect": "30–50",
            "intensity": "50–80"
        },
        "progression_rules": (
            "• Dominance grows enforcing grueling routines.\n"
            "• Intensity spikes with punishing physical challenges."
        ),
        "setting_examples": (
            "• Urban Life: Physical/emotional domination in training.\n"
            "• Post-Apocalypse: Survival tasks enforced as 'training.'"
        ),
        "unique_traits": (
            "• Mixes motivational speech with sharp condescension.\n"
            "• Exploits your physical limits to assert control."
        )
    },
    {
        "name": "Cheerleader/Team Captain",  # 49
        "baseline_stats": {
            "dominance": "50–80",
            "cruelty": "40–70",
            "closeness": "60–90",
            "trust": "30–50",
            "respect": "20–50",
            "intensity": "50–80"
        },
        "progression_rules": (
            "• Dominance spikes using leadership roles.\n"
            "• Intensity rises with public displays of control."
        ),
        "setting_examples": (
            "• All-Girls College: Social status humiliations.\n"
            "• Urban Life: Balances public dominance with private cruelty."
        ),
        "unique_traits": (
            "• Charm + authority makes submission inevitable.\n"
            "• Frames domination as 'team dynamic.'"
        )
    },
    {
        "name": "Nun/Priestess",  # 50
        "baseline_stats": {
            "dominance": "60–90",
            "cruelty": "40–70",
            "closeness": "50–70",
            "trust": "30–60",
            "respect": "40–70",
            "intensity": "50–90"
        },
        "progression_rules": "Enforces submission via moral/spiritual authority, punishments as 'cleansing.'",
        "setting_examples": (
            "• Occult Ritual: Sacred rituals ensure eternal submission.\n"
            "• Matriarchy Kingdom: Religious or societal structures enforce dominance."
        ),
        "unique_traits": (
            "• Uses guilt and morality for control.\n"
            "• Punishments framed as 'devotion.'"
        )
    },
    {
        "name": "Military Officer",  # 51
        "baseline_stats": {
            "dominance": "70–90",
            "cruelty": "50–80",
            "closeness": "30–50",
            "trust": "20–50",
            "respect": "30–60",
            "intensity": "60–90"
        },
        "progression_rules": "Strict discipline and hierarchy to enforce absolute obedience.",
        "setting_examples": (
            "• Post-Apocalypse: Leads survival groups with military hierarchy.\n"
            "• Dystopian Oppression: Brutal regime with unrelenting rules."
        ),
        "unique_traits": (
            "• Precise, unrelenting commands.\n"
            "• Treats failure as personal affront."
        )
    },
    {
        "name": "Sorceress/Mage",  # 52
        "baseline_stats": {
            "dominance": "70–100",
            "cruelty": "50–80",
            "closeness": "30–60",
            "trust": "20–50",
            "respect": "30–60",
            "intensity": "60–90"
        },
        "progression_rules": "Uses magical power for domination, framing submission as a mystical bond.",
        "setting_examples": (
            "• Forgotten Realms: Magical oaths ensure obedience.\n"
            "• Occult Ritual: Power is sacred and unchallengeable."
        ),
        "unique_traits": (
            "• Turns defiance into arcane lessons.\n"
            "• Punishments presented as 'arcane necessities.'"
        )
    },
    {
        "name": "Slave Overseer",  # 53
        "baseline_stats": {
            "dominance": "80–100",
            "cruelty": "70–90",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "10–30",
            "intensity": "70–100"
        },
        "progression_rules": "Relishes controlling others, using tasks/punishments to ensure submission.",
        "setting_examples": (
            "• Matriarchy Kingdom: Oversees submissive males publicly.\n"
            "• Post-Apocalypse: Limited resources leveraged to dominate."
        ),
        "unique_traits": (
            "• Micromanages tasks.\n"
            "• Strict rules plus elaborate punishments."
        )
    },
    {
        "name": "Enigmatic Stranger",  # 54
        "baseline_stats": {
            "dominance": "50–70",
            "cruelty": "30–60",
            "closeness": "20–40",
            "trust": "10–30",
            "respect": "30–60",
            "intensity": "40–80"
        },
        "progression_rules": "Uses mystery/unpredictability to control, leaving you unsure of intentions.",
        "setting_examples": (
            "• Urban Life: Appears unexpectedly.\n"
            "• Surreal Setting: Environment’s strangeness aids their dominance."
        ),
        "unique_traits": (
            "• Blends charm and menace.\n"
            "• Makes submission feel like the safest option."
        )
    },
    {
        "name": "Shopkeeper/Market Vendor",  # 55
        "baseline_stats": {
            "dominance": "30–60",
            "cruelty": "20–50",
            "closeness": "40–70",
            "trust": "30–50",
            "respect": "20–40",
            "intensity": "30–50"
        },
        "progression_rules": "Exploits your dependency on goods, framing control as 'business.'",
        "setting_examples": (
            "• Urban Life: Subtle transactions for dominance.\n"
            "• Ruined Setting: Scarce resources justify escalating demands."
        ),
        "unique_traits": (
            "• Treats dominance as 'just business.'\n"
            "• Ensures you feel indebted."
        )
    },
    {
        "name": "Rival",  # 56
        "baseline_stats": {
            "dominance": "50–80",
            "cruelty": "40–70",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "30–50",
            "intensity": "50–80"
        },
        "progression_rules": "Competition is used to dominate, each victory humiliates you further.",
        "setting_examples": (
            "• All-Girls College: Social/academic competition.\n"
            "• Corporate Office: Professional power plays."
        ),
        "unique_traits": (
            "• Public victories deepen your submission.\n"
            "• Balances charm with condescension."
        )
    },
    {
        "name": "Demoness/Devil",  # 57
        "baseline_stats": {
            "dominance": "80–100",
            "cruelty": "70–100",
            "closeness": "30–50",
            "trust": "10–30",
            "respect": "20–50",
            "intensity": "70–100"
        },
        "progression_rules": "Uses infernal power, framing submission as a pact/binding.",
        "setting_examples": (
            "• Occult Ritual: Submission as demonic pact.\n"
            "• Cosmic Setting: Awe and terror ensure compliance."
        ),
        "unique_traits": (
            "• Exploits desires/fears for compliance.\n"
            "• Punishments framed as 'consequences.'"
        )
    },
    {
        "name": "Queen Bee",  # 58
        "baseline_stats": {
            "dominance": "60–90",
            "cruelty": "40–70",
            "closeness": "50–80",
            "trust": "20–50",
            "respect": "30–50",
            "intensity": "50–80"
        },
        "progression_rules": "Dominates via social clout/manipulation, entire group dynamics revolve around them.",
        "setting_examples": (
            "• All-Girls College: Peer pressure for submission.\n"
            "• High Society: Social events for public humiliation."
        ),
        "unique_traits": (
            "• Orchestrates social dilemmas, forcing submission.\n"
            "• Balances charm with cruelty."
        )
    },
    {
        "name": "Haunted Entity",  # 59
        "baseline_stats": {
            "dominance": "50–80",
            "cruelty": "60–90",
            "closeness": "10–30",
            "trust": "-50 to 10",
            "respect": "20–40",
            "intensity": "70–100"
        },
        "progression_rules": "Supernatural fear and psychological manipulation maintain domination.",
        "setting_examples": (
            "• Traditional Horror: Haunts environment to trap you.\n"
            "• Surreal Setting: Unsettling phenomena reinforce fear."
        ),
        "unique_traits": (
            "• Thrives on isolation/fear.\n"
            "• Submission as appeasement."
        )
    },
    {
        "name": "Pirate",  # 60
        "baseline_stats": {
            "dominance": "60–90",
            "cruelty": "50–80",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "30–50",
            "intensity": "60–90"
        },
        "progression_rules": "Chaos and fear enforce submission, often mixing charisma with brutality.",
        "setting_examples": (
            "• Ruined/Decayed Setting: Leads a crew, using scarcity.\n"
            "• Mythic Setting: Pirate’s 'code' justifies harsh control."
        ),
        "unique_traits": (
            "• Balances charm with intimidation.\n"
            "• Sudden cruelty keeps you on edge."
        )
    },
    {
        "name": "Drug Dealer",  # 61
        "baseline_stats": {
            "dominance": "60–90",
            "cruelty": "40–80",
            "closeness": "30–50",
            "trust": "20–40",
            "respect": "20–50",
            "intensity": "60–90"
        },
        "progression_rules": "Uses substances to deepen dependence, amplifying shame and control.",
        "setting_examples": (
            "• Urban Life: Controls your routine through supply.\n"
            "• Ruined Setting: Exploits scarcity for absolute power."
        ),
        "unique_traits": (
            "• Rewards/withdrawals ensure compliance.\n"
            "• Treats tasks as 'necessary trades' for access."
        )
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
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
        "progression_rules": "Add an extra modifier. Does not count as a separate archetype.",
        "setting_examples": "N/A",
        "unique_traits": "Placeholder for extra modifiers or expansions."
    }
]

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
                json.dumps(arc["baseline_stats"]),     # store baseline_stats as JSONB
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
    # e.g. four = [(1, "Empress/Queen"), (5, "Tsundere"), ( ... )]

    # We'll store them in a JSON array in NPCStats. 
    # So let's get the names or IDs
    # Example storing just the ID or name. Let's store {id, name} for clarity.
    assigned_list = [{"id": row[0], "name": row[1]} for row in four]

    # 2) Update the NPCStats table. 
    # We'll assume we have a column "archetypes" of type JSONB in NPCStats.
    cursor.execute("""
        UPDATE NPCStats
        SET archetypes = %s
        WHERE npc_id = %s
    """, (json.dumps(assigned_list), npc_id))

    conn.commit()
    conn.close()

    return assigned_list  # for debugging

