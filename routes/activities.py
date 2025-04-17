# routes/activities.py

from quart import Blueprint, jsonify
import json
import logging
from db.connection import get_db_connection_context

activities_bp = Blueprint('activities', __name__)

async def insert_missing_activities():
    """
    Inserts a curated list of Activities into the Activities table,
    if they don't already exist (based on unique activity name).
    Uses asyncpg properly for database operations.
    """
    logging.info("[insert_missing_activities] Starting...")
    activities_data = [
        {
            "name": "Ass Worship",
            "purpose": [
                "Centers on forced servitude to an NPC’s backside, highlighting humiliation and physical intimacy."
            ],
            "stat_integration": [
                "Shame: Amplified through public acts or verbal mockery.",
                "Obedience: Locked at higher tiers as tasks become more intimate."
            ],
            "intensity_tiers": [
                "Low: Polishing an NPC’s chair while they sit, framed as a harmless favor.",
                "Moderate: Lightly kissing the NPC’s clothed backside in private.",
                "High: Massaging or licking the bare ass while mocked for your eagerness.",
                "Maximum: Public worship during a gathering, ensuring everyone witnesses your submission."
            ],
            "setting_variants": [
                "High Society: Ceremonial displays of devotion at formal parties.",
                "Flooded Setting: Performed partially submerged, adding physical difficulty."
            ]
        },
    
        {
            "name": "Pegging",
            "purpose": [
                "NPC uses a strap-on or other tool to penetrate the submissive, emphasizing power reversal and humiliation."
            ],
            "stat_integration": [
                "Shame: Tied to the act’s intimacy and any public exposure.",
                "Dependency: Reinforced if the NPC frames penetration as either punishment or special privilege."
            ],
            "intensity_tiers": [
                "Low: Teasing threats or jokes about pegging, planting the idea.",
                "Moderate: Private pegging as a corrective lesson.",
                "High: Acts accompanied by degrading dialogue or commands.",
                "Maximum: Public or semi-public pegging during a ritual or group event."
            ],
            "setting_variants": [
                "Occult Ritual: Framed as part of a sacred act of submission.",
                "Time Loop: Repeated pegging to reinforce your helpless state."
            ]
        },
    
        {
            "name": "Verbal Humiliation",
            "purpose": [
                "NPCs degrade you through insults, taunts, and destructive language, undermining confidence."
            ],
            "stat_integration": [
                "Shame: Grows with each insult, especially in public.",
                "Willpower: Erodes as insults target personal insecurities."
            ],
            "intensity_tiers": [
                "Low: Mild teasing about minor flaws.",
                "Moderate: Mockery in front of small groups.",
                "High: Forced to repeat harsh self-insults, deepening submission.",
                "Maximum: Public broadcasts of your failures, orchestrated by NPCs."
            ],
            "setting_variants": [
                "Surreal Setting: Insults become bizarre and disorienting.",
                "Celebrity Presence: High-profile NPCs amplify your embarrassment."
            ]
        },
    
        {
            "name": "Sweaty Sock Worship",
            "purpose": [
                "Focuses on odor play and enforced servitude through an NPC’s used, sweaty socks."
            ],
            "stat_integration": [
                "Lust: Exploited if NPCs claim it’s a 'reward' to smell or lick them.",
                "Shame: Heightened by the act’s physical and intimate nature."
            ],
            "intensity_tiers": [
                "Low: Light sniffing as a teasing task.",
                "Moderate: Kissing or holding sweaty socks to your face under supervision.",
                "High: Full-on licking them clean, with NPCs jeering.",
                "Maximum: Public sock worship with multiple witnesses."
            ],
            "setting_variants": [
                "Flooded Setting: Socks are soggy, making the task more uncomfortable.",
                "Grimdark: Cast as a punishment for repeated failures."
            ]
        },
    
        {
            "name": "Armpit Worship",
            "purpose": [
                "Submission to an NPC’s armpits, emphasizing odor, closeness, and control."
            ],
            "stat_integration": [
                "Obedience: Locked as NPC commands each action.",
                "Shame: Increased by the intimate, sweaty nature of the worship."
            ],
            "intensity_tiers": [
                "Low: Light sniffing over clothes.",
                "Moderate: Licking or nuzzling bare armpits in private.",
                "High: Prolonged worship with mocking commentary.",
                "Maximum: Public armpit worship, turned into a group spectacle."
            ],
            "setting_variants": [
                "Intimate Setting: Pitched as 'devotional care.'",
                "Celebrity Presence: Performed in front of influential onlookers."
            ]
        },
    
        {
            "name": "Foot Worship",
            "purpose": [
                "Serving at the feet of NPCs, from gentle respect to full-on licking and massage."
            ],
            "stat_integration": [
                "Shame: Soars if done publicly or under constant verbal assault.",
                "Obedience: Reinforced when tasks become more degrading or physically intense."
            ],
            "intensity_tiers": [
                "Low: Polishing or lightly touching their shoes.",
                "Moderate: Kissing NPC’s feet in private.",
                "High: Massaging or licking sweaty feet with mocking NPC commentary.",
                "Maximum: Public foot worship ceremonies or punishments."
            ],
            "setting_variants": [
                "High Tech: Cleaning cybernetic feet or implants.",
                "Ruined Setting: Feet are caked in mud or dirt, adding extra work."
            ]
        },
    
        {
            "name": "Psychological Torment",
            "purpose": [
                "NPCs manipulate your mind with fear, confusion, or betrayal to deepen control."
            ],
            "stat_integration": [
                "Corruption: Climbs as your self-identity is torn apart.",
                "Shame: Spikes as they highlight personal failures or secrets."
            ],
            "intensity_tiers": [
                "Low: Planting small doubts about your competence.",
                "Moderate: Exploiting secrets to coerce cooperation.",
                "High: Gaslighting or looping punishments that cause despair.",
                "Maximum: Staged betrayals or orchestrated isolations ensuring hopelessness."
            ],
            "setting_variants": [
                "Dystopian: Fake surveillance 'evidence' breaks your resolve.",
                "Time Loop: Repeating horrors that stack cumulatively."
            ]
        },
    
        {
            "name": "Human Furniture",
            "purpose": [
                "Dehumanizing you into an object for NPC comfort, from footstools to tables."
            ],
            "stat_integration": [
                "Shame: Surges from being used as an inanimate tool.",
                "Closeness: Increases if NPC frequently interacts or lounges on you."
            ],
            "intensity_tiers": [
                "Low: Holding items while standing still.",
                "Moderate: Acting as a footrest or stool in private.",
                "High: Serving as chairs or tables during NPC interactions.",
                "Maximum: Publicly showcased as furniture for large gatherings."
            ],
            "setting_variants": [
                "Steam Punk: Machinery integrates to keep you locked in place.",
                "Mythic Setting: Treated as an honored yet demeaning 'living throne.'"
            ]
        },
    
        {
            "name": "Vore",
            "purpose": [
                "An extreme or symbolic scenario of 'consumption,' usually surreal or roleplay-based."
            ],
            "stat_integration": [
                "Intensity: Elevated by the threat or illusion of being consumed.",
                "Corruption: Rises if it’s framed as a twisted sacrifice."
            ],
            "intensity_tiers": [
                "Low: NPCs joke about devouring you, teasing your fear.",
                "Moderate: Symbolic acts like 'offering' you as a meal or in illusions.",
                "High: More intimate or surreal scenes of partial 'consumption.'",
                "Maximum: Full public rituals of sacrifice or simulated devouring."
            ],
            "setting_variants": [
                "Surreal Setting: Dreamlike hunts or illusions of being eaten.",
                "Cosmic: NPC is an incomprehensible being, making your fate uncertain."
            ]
        },
    
        {
            "name": "Giantess",
            "purpose": [
                "NPC physically dwarfs you, emphasizing your inferiority through size difference."
            ],
            "stat_integration": [
                "Dominance: Amplified by the literal scale gap.",
                "Obedience: Locked because physical rebellion is futile."
            ],
            "intensity_tiers": [
                "Low: NPC mocks your small stature playfully.",
                "Moderate: Tasks force you to carry items beyond your strength.",
                "High: Treated like a pet or toy, subject to physical strain.",
                "Maximum: Public parades showcasing your status as a helpless plaything."
            ],
            "setting_variants": [
                "Overgrown Setting: Lush landscapes highlight size imbalance.",
                "High Society: Displayed as a spectacle at grand galas."
            ]
        },
    
        {
            "name": "Odor Play",
            "purpose": [
                "Focuses on smells from sweat, feet, armpits, or used clothing as a humiliating factor."
            ],
            "stat_integration": [
                "Lust: NPCs can claim it’s 'arousing' to worship their scent.",
                "Shame: Increases with public involvement or prolonged exposure."
            ],
            "intensity_tiers": [
                "Low: Teasing references to an NPC’s aroma.",
                "Moderate: Sniffing sweaty items under supervision.",
                "High: Licking or cleaning sweaty body parts amid taunts.",
                "Maximum: Public worship of pungent odor, ensuring maximum humiliation."
            ],
            "setting_variants": [
                "Flooded Setting: Waterlogged items intensify the unpleasant aspect.",
                "Celebrity Presence: Public entertainment featuring your forced sniffing."
            ]
        },
    
        {
            "name": "Mommy Domme",
            "purpose": [
                "Infantilization, care-taking, and stern discipline from a motherly NPC figure."
            ],
            "stat_integration": [
                "Dependency: Rises as you rely on them for 'nurturing.'",
                "Shame: Heightened by childlike tasks or scolding."
            ],
            "intensity_tiers": [
                "Low: NPC calls herself 'Mommy' in teasing ways.",
                "Moderate: Feeding or scolding you as a child figure.",
                "High: Public babying, with childish attire or punishments.",
                "Maximum: Full regression, including diapers or pacifiers in group settings."
            ],
            "setting_variants": [
                "Intimate Setting: Private nurturing overshadowed by manipulative control.",
                "High Society: Public mother-child dynamic displayed at formal events."
            ]
        },
    
        {
            "name": "Body Fluid Play (Spit, Sweat, etc.)",
            "purpose": [
                "Direct exchange or contact with NPC bodily fluids, emphasizing gross-out or intimate control."
            ],
            "stat_integration": [
                "Obedience: Locked if NPC demands immediate compliance.",
                "Shame: Amplified by the raw physical intimacy."
            ],
            "intensity_tiers": [
                "Low: Teasing spitting near you as a joke.",
                "Moderate: Being spit on or told to lick sweat.",
                "High: Forced swallowing or extended exposure to fluids.",
                "Maximum: Public fluid play as a spectacle of your humiliation."
            ],
            "setting_variants": [
                "Dystopian: Tied to penal system or corrective measures.",
                "Occult Ritual: Symbolic or 'holy' fluid ingestion to seal your fate."
            ]
        },
    
        {
            "name": "Cuckolding",
            "purpose": [
                "Witnessing or facilitating an NPC’s sexual acts with others to heighten your humiliation."
            ],
            "stat_integration": [
                "Shame: Centered on your powerlessness in sexual contexts.",
                "Corruption: Grows as emotional barriers collapse."
            ],
            "intensity_tiers": [
                "Low: NPC teases you about potential affairs.",
                "Moderate: You watch private encounters, framed as lessons.",
                "High: You indirectly assist or prepare for these acts.",
                "Maximum: Full public cuckolding events, ensuring total humiliation."
            ],
            "setting_variants": [
                "Celebrity Presence: Turned into a sensational show.",
                "Tribal Setting: Communal involvement intensifies your despair."
            ]
        },
    
        {
            "name": "Humiliation",
            "purpose": [
                "Broad category of tasks that degrade you mentally or physically, making you feel exposed."
            ],
            "stat_integration": [
                "Shame: The overarching driver, scaled by intensity.",
                "Willpower: Diminishes with repeated humiliations."
            ],
            "intensity_tiers": [
                "Low: Minor teasing or small embarrassing tasks.",
                "Moderate: Public admissions of faults or mild shaming rituals.",
                "High: Full-on crawling, scripted apologies, or mock announcements.",
                "Maximum: Broadcasted humiliations or large-scale events with no escape."
            ],
            "setting_variants": [
                "High Society: Formal events where your humiliation is elegantly showcased.",
                "Ruined Setting: Tied to survival, making refusal almost impossible."
            ]
        },
    
        {
            "name": "Degradation",
            "purpose": [
                "Pushes you to accept increasingly demeaning roles or actions, stripping personal dignity."
            ],
            "stat_integration": [
                "Corruption: Grows as self-respect dissolves.",
                "Obedience: Solidifies with each humiliating step."
            ],
            "intensity_tiers": [
                "Low: Exaggerated honorifics or bowing excessively.",
                "Moderate: Wearing shameful outfits or performing menial tasks.",
                "High: Long-term groveling or publicly licking floors.",
                "Maximum: Ritual ceremonies aimed at fully eroding your sense of worth."
            ],
            "setting_variants": [
                "Grimdark: Degradation is relentlessly punitive.",
                "Surreal: Tasks become bizarre, compounding the confusion."
            ]
        },
    
        {
            "name": "Branding/Permanently Marking",
            "purpose": [
                "Creating lasting physical marks (tattoos, brands, piercings) signifying ownership."
            ],
            "stat_integration": [
                "Obedience: Locked by the permanence of markings.",
                "Corruption: Grows if marks symbolize deeper submission."
            ],
            "intensity_tiers": [
                "Low: Jokes or threats about branding you someday.",
                "Moderate: Temporary marks like henna, stamps, or paint.",
                "High: Actual tattoos or piercings done in private.",
                "Maximum: Public branding ceremonies so everyone sees your new status."
            ],
            "setting_variants": [
                "Occult Ritual: A sacred bond between you and the NPC.",
                "Royal Elegance: Marked in a grand event, signifying your vow of servitude."
            ]
        },
    
        {
            "name": "Wrestling",
            "purpose": [
                "Physical contests where the NPC inevitably overpowers you to demonstrate dominance."
            ],
            "stat_integration": [
                "Dominance: Soars if NPC repeatedly triumphs in front of onlookers.",
                "Shame: Heightened by your helplessness and forced defeat."
            ],
            "intensity_tiers": [
                "Low: Playful grappling at home.",
                "Moderate: Mock matches presented as 'training.'",
                "High: Loser’s punishment is publicly assigned, guaranteeing your humiliation.",
                "Maximum: Ceremonial tournaments where your defeat is a foregone conclusion."
            ],
            "setting_variants": [
                "Tribal: Ritualized to prove loyalty to tribe leaders.",
                "Dystopian: Broadcast fights to entertain the masses."
            ]
        },
    
        {
            "name": "Bullying",
            "purpose": [
                "Sustained intimidation and harassment by NPC(s), from mild pranks to severe cruelty."
            ],
            "stat_integration": [
                "Shame: Escalates via relentless humiliation tactics.",
                "Willpower: Crumbles when you face organized or constant abuse."
            ],
            "intensity_tiers": [
                "Low: Name-calling or small pranks.",
                "Moderate: Group mockery or sabotage of your tasks.",
                "High: Systematic humiliation with no respite.",
                "Maximum: Multiple NPCs coordinate to break you emotionally."
            ],
            "setting_variants": [
                "All-Girls College: Cliques, rumors, and social ostracism.",
                "Game/Competition: Rigged challenges stacked against you."
            ]
        },
    
        {
            "name": "Urolagnia",
            "purpose": [
                "Involves urine in a submissive or humiliating context, from jokes to explicit public scenes."
            ],
            "stat_integration": [
                "Corruption: Rises when boundaries are shattered.",
                "Shame: Surges when acts are made public or prolonged."
            ],
            "intensity_tiers": [
                "Low: NPC jokes about controlling your bladder or using you as a toilet.",
                "Moderate: Indirect tasks like cleaning or sanitizing 'accidents.'",
                "High: Direct acts of being peed on or forced to endure it publicly.",
                "Maximum: Ritualistic or group participation, ensuring total degradation."
            ],
            "setting_variants": [
                "Flooded: The environment intensifies water-based acts.",
                "Occult Ritual: Frames urine as a twisted purification or subjugation."
            ]
        },
    
        {
            "name": "Bondage",
            "purpose": [
                "Restraining your movements with ropes, cuffs, or harnesses to reinforce helplessness."
            ],
            "stat_integration": [
                "Obedience: Tasks demand stillness and compliance.",
                "Shame: Rises with exposure and inability to fight back."
            ],
            "intensity_tiers": [
                "Low: Simple handcuffs in private.",
                "Moderate: Full-body restraints or hogties for short durations.",
                "High: Public bondage displays, ensuring onlookers see your plight.",
                "Maximum: Ritualistic bondage in elaborate ceremonies, mocking your total powerlessness."
            ],
            "setting_variants": [
                "Flooded: Submersion or partial restraint in water adds fear.",
                "Historical: Stocks, pillories, or iron chains consistent with the era."
            ]
        },
    
        {
            "name": "Impact Play",
            "purpose": [
                "NPC physically strikes you with paddles, whips, or hands, focusing on punishment or discipline."
            ],
            "stat_integration": [
                "Intensity: Tied to the severity and frequency of strikes.",
                "Willpower: Withers if you must endure it silently."
            ],
            "intensity_tiers": [
                "Low: Light smacks or playful swats.",
                "Moderate: Private punishments with tools like paddles.",
                "High: Public strikes in front of an audience, heightening humiliation.",
                "Maximum: Ceremonial thrashings with strict choreography and moralizing commentary."
            ],
            "setting_variants": [
                "Grimdark: Brutal contexts, no pretense of caring.",
                "Occult: Portrayed as purifying or diabolical sacraments."
            ]
        },
    
        {
            "name": "Entrapment",
            "purpose": [
                "NPCs set traps or contrive scenarios to ensure you have no choice but to submit."
            ],
            "stat_integration": [
                "Trust: Declines as you realize the extent of manipulation.",
                "Obedience: Reinforced by literal or psychological captivity."
            ],
            "intensity_tiers": [
                "Low: Minor setups restricting your options, like locked doors.",
                "Moderate: Physical cages or rooms, forcing compliance.",
                "High: Elaborate traps with no escape unless you obey.",
                "Maximum: Public caging displays your helplessness to everyone."
            ],
            "setting_variants": [
                "Steam Punk: Mechanical contraptions enforce your captivity.",
                "Surreal: Maze-like illusions ensure you’re perpetually confused."
            ]
        },
    
        {
            "name": "Pet Play",
            "purpose": [
                "NPC treats you as a literal pet—collars, crawling, playing fetch, etc."
            ],
            "stat_integration": [
                "Obedience: Tasks revolve around fulfilling 'pet' rules.",
                "Shame: Elevated by performing these roles publicly."
            ],
            "intensity_tiers": [
                "Low: Light name-calling ('pet'), subtle commands.",
                "Moderate: Collars, crawling, eating from bowls in private.",
                "High: Extended roleplay as an animal with more elaborate tasks.",
                "Maximum: Publicly paraded in pet gear with humiliating shows or competitions."
            ],
            "setting_variants": [
                "Tribal: A primal sense of being 'tamed.'",
                "Celebrity: Turned into bizarre entertainment for star-studded events."
            ]
        },
    
        {
            "name": "Body Modification",
            "purpose": [
                "From minor changes to large-scale transformations, each reinforcing submission."
            ],
            "stat_integration": [
                "Obedience: Permanence of body changes locks your role.",
                "Corruption: Grows if modifications symbolize deeper devotion."
            ],
            "intensity_tiers": [
                "Low: Temporary hair color or makeup.",
                "Moderate: Piercings or small tattoos, usually hidden.",
                "High: Large tattoos, symbolic brandings, or distinctive piercings.",
                "Maximum: Public modifications with elaborate ceremonies, no going back."
            ],
            "setting_variants": [
                "Mythic: Marks are described as runes or curses binding your soul.",
                "Occult: Tied to a supernatural contract intensifying your submission."
            ]
        },
    
        {
            "name": "Slave Contract",
            "purpose": [
                "Formalizing your submission through a literal or symbolic contract signifying ownership."
            ],
            "stat_integration": [
                "Obedience: Reinforced as 'legal' or absolute.",
                "Dependency: Amplified as your resources or autonomy are contractually surrendered."
            ],
            "intensity_tiers": [
                "Low: Teased as a hypothetical or joke.",
                "Moderate: Private signing of symbolic documents.",
                "High: Public contract ceremonies with guests as witnesses.",
                "Maximum: Binding, unbreakable deals enforced by tradition or supernatural means."
            ],
            "setting_variants": [
                "High Society: Signed at galas, so your shame is elegantly displayed.",
                "Grimdark: Tied to survival or merciless laws, leaving no hope of escape."
            ]
        },
    
        {
            "name": "Sissification",
            "purpose": [
                "Feminizing tasks, attire, or behavior to emphasize humiliation and transformation."
            ],
            "stat_integration": [
                "Shame: Potent if you resist or fear the stigma.",
                "Obedience: Reinforced by repeated instructions or feminine roleplay requirements."
            ],
            "intensity_tiers": [
                "Low: Nicknames or subtle clothing shifts.",
                "Moderate: Makeup, outfits, or behaviors in private.",
                "High: Full transformation, complete with regular tasks or duties.",
                "Maximum: Public sissification, ensuring everyone acknowledges your new role."
            ],
            "setting_variants": [
                "Dystopian: An enforced 're-education' program.",
                "Celebrity Presence: A humiliating spectacle for a high-status crowd."
            ]
        },
    
        {
            "name": "Tease and Denial",
            "purpose": [
                "Keeping you in a state of arousal without climax, manipulating lust and frustration."
            ],
            "stat_integration": [
                "Lust: Intensely triggered, but never resolved on your terms.",
                "Obedience: NPC dictates every moment of relief or further denial."
            ],
            "intensity_tiers": [
                "Low: Subtle teasing, mild physical contact with no release.",
                "Moderate: Prolonged denial in private, NPC controlling timing.",
                "High: Public teasing, edging in front of others.",
                "Maximum: Ceremonial denial with group oversight or repeated edging sessions."
            ],
            "setting_variants": [
                "Intimate: Portrayed as a nurturing but manipulative technique.",
                "Occult: Denial tied to a mystical vow, intensifying psychological control."
            ]
        },
    
        {
            "name": "Electrostimulation",
            "purpose": [
                "Using electric shocks or pulses to condition, punish, or 'train' you physically."
            ],
            "stat_integration": [
                "Intensity: Directly correlates with shock level.",
                "Obedience: You must remain still or follow commands while jolted."
            ],
            "intensity_tiers": [
                "Low: Teasing with mild static-like devices.",
                "Moderate: Controlled zaps during private tasks or lessons.",
                "High: Prolonged stimulation as a punishment, possibly in public.",
                "Maximum: Ceremonial group shocking, ensuring no escape from the device’s power."
            ],
            "setting_variants": [
                "High Tech: AI-regulated shocks that respond to your attempts at defiance.",
                "Steam Punk: Elaborate contraptions delivering Victorian-era jolts."
            ]
        },
    
        {
            "name": "Grooming (Psychological Conditioning)",
            "purpose": [
                "Incremental shifts in your mindset to normalize obedience, break autonomy, and foster dependency."
            ],
            "stat_integration": [
                "Obedience: Locked as repeated instructions become habit.",
                "Corruption: Rises as your identity is reshaped to align with NPC demands.",
                "Dependency: Grows if NPCs isolate you from external support."
            ],
            "intensity_tiers": [
                "Low: Accepting small demands like honorifics or mild speech rules.",
                "Moderate: Reinforcing obedience through repeated phrases or daily 'loyalty' tasks.",
                "High: Erasing personal habits or individuality, adopting an NPC-approved identity.",
                "Maximum: Full conditioning with no aspect of your life unmonitored."
            ],
            "setting_variants": [
                "Occult Ritual: Each step is a 'sacred vow' of deeper subjugation.",
                "High Tech: AI or advanced surveillance corrects or punishes stray thoughts in real time."
            ]
        },
    
        {
            "name": "Primal/Prey",
            "purpose": [
                "Intense chase or hunt dynamics, showcasing your vulnerability and the NPC’s predatory power."
            ],
            "stat_integration": [
                "Intensity: Fueled by the fear and adrenaline of being pursued.",
                "Closeness: Rises if the NPC physically interacts with you upon capture."
            ],
            "intensity_tiers": [
                "Low: Light cornering or playful chases in a controlled space.",
                "Moderate: Actual 'flee or be caught' tasks with real consequences.",
                "High: Extended hunts, your stamina sapped until recapture.",
                "Maximum: Ceremonial hunts with onlookers, culminating in your public surrender."
            ],
            "setting_variants": [
                "Tribal: Hunts are ritualistic for the community’s entertainment.",
                "Surreal: Shifting, dreamlike landscapes that disorient your escape attempts."
            ]
        },
    
        {
            "name": "Butt Plug",
            "purpose": [
                "Insertion and prolonged wear of a plug, emphasizing constant physical reminders of submission."
            ],
            "stat_integration": [
                "Shame: Magnified by the persistent, intimate nature of the intrusion.",
                "Obedience: You must keep it in place under NPC’s orders."
            ],
            "intensity_tiers": [
                "Low: NPC jokes about the idea, or forces you to hold it briefly.",
                "Moderate: Wearing a plug privately for short durations.",
                "High: Kept in for extended periods or used in public tasks.",
                "Maximum: Ceremonial usage with elaborate mockery, turning your predicament into a spectacle."
            ],
            "setting_variants": [
                "Occult: Treated as a mystic device symbolizing deeper possession.",
                "High Society: A hidden secret under formal attire, while you serve drinks or greet guests."
            ]
        },
    
        {
            "name": "Gags",
            "purpose": [
                "Silencing you with physical restraints on your mouth, reinforcing helplessness."
            ],
            "stat_integration": [
                "Obedience: You cannot speak or protest, forced to rely on NPC mercy.",
                "Shame: Heightened by your inability to express even if humiliated further."
            ],
            "intensity_tiers": [
                "Low: Simple cloth or playful 'keep quiet' jokes.",
                "Moderate: Standard ball gags in private sessions.",
                "High: Public gagging, ensuring onlookers see your powerlessness.",
                "Maximum: Ritual gag usage in front of large crowds or special events."
            ],
            "setting_variants": [
                "Historical: Iron gags, pillories, or other era-specific devices.",
                "Cyberpunk: Electronic or hi-tech muzzles that also track vitals."
            ]
        },
    
        {
            "name": "Suspension",
            "purpose": [
                "Lifting you off the ground with ropes or devices, showcasing your complete lack of control."
            ],
            "stat_integration": [
                "Obedience: Movement is nearly impossible, so you must comply.",
                "Intensity: Grows with how elaborate or prolonged the suspension is."
            ],
            "intensity_tiers": [
                "Low: Slight lift where toes barely touch the floor.",
                "Moderate: Full suspension for short durations as private punishment.",
                "High: Public suspension, allowing others to see your helpless state.",
                "Maximum: Ceremonial setups with decor or an audience to amplify shame."
            ],
            "setting_variants": [
                "Flooded: Suspended above water, adding dread of dropping in.",
                "Mythic: Framed as an offering to mystical or divine NPCs."
            ]
        },
    
        {
            "name": "Public Humiliation",
            "purpose": [
                "Exposing your degrading tasks or punishments to an audience for maximum shame."
            ],
            "stat_integration": [
                "Shame: Central driver, multiplied by watchers’ reactions.",
                "Corruption: Surges if acceptance of public disgrace becomes normal."
            ],
            "intensity_tiers": [
                "Low: Wearing minor markers of shame in small gatherings.",
                "Moderate: Reciting apologies or confessions before a group.",
                "High: Physical tasks (crawling, forced worship) performed publicly.",
                "Maximum: Huge ceremonies or events dedicated entirely to ridiculing you."
            ],
            "setting_variants": [
                "All-Girls College: School assemblies or pep rallies turn into humiliation spectacles.",
                "High Society: Showcased at an elegant ball where guests observe your downfall."
            ]
        },
    
        {
            "name": "Chastity",
            "purpose": [
                "Restricting your sexual release under a locked device or rule, ensuring total control of your pleasure."
            ],
            "stat_integration": [
                "Obedience: You must comply with device usage or risk further punishment.",
                "Dependency: NPCs decide if or when you can be freed or relieved."
            ],
            "intensity_tiers": [
                "Low: Teasing references to locking you up.",
                "Moderate: Private usage for moderate durations.",
                "High: Long-term wear, public references to your 'locked' state.",
                "Maximum: Group or ceremonial keyholder events, making your plight widely known."
            ],
            "setting_variants": [
                "Occult Ritual: Treated as purifying sacrifice of your pleasure.",
                "Dystopian: Official system mandating your chastity device."
            ]
        },
    
        {
            "name": "Financial Domination",
            "purpose": [
                "NPC takes control of your finances or forces tributes, highlighting your monetary dependence."
            ],
            "stat_integration": [
                "Obedience: Follows forced spending or tribute schedules.",
                "Dependency: You rely on NPC to grant any financial relief."
            ],
            "intensity_tiers": [
                "Low: Small tribute gifts like buying coffee or groceries.",
                "Moderate: Covering NPC’s bills, rent, or personal items.",
                "High: Handing over access to accounts or letting NPC decide purchases.",
                "Maximum: Public auctions of your possessions, bankrupting you for NPC amusement."
            ],
            "setting_variants": [
                "High Society: Maintaining 'appearances' ironically fosters your ruin.",
                "Cyberpunk: Digital controls or AI-run finances ensure you can’t hide funds."
            ]
        },
    
        {
            "name": "Hypnosis/Brainwashing",
            "purpose": [
                "NPC uses mental conditioning or trance techniques to rewrite your thought patterns."
            ],
            "stat_integration": [
                "Obedience: Cemented as your mind is molded to comply automatically.",
                "Corruption: Rises when personal identity or desires are erased."
            ],
            "intensity_tiers": [
                "Low: Subtle suggestions or repeated mantras in casual contexts.",
                "Moderate: Audio/visual sessions forcing you to listen/watch for hours.",
                "High: Full reprogramming of daily routines or attitudes, leaving you docile.",
                "Maximum: Public demonstrations of your brainwashed obedience, shocking onlookers."
            ],
            "setting_variants": [
                "Surreal: The line between trance and reality is blurred.",
                "Occult: Spells or rituals anchor your mind to NPC commands."
            ]
        },
    
        {
            "name": "Deceptive Rewards",
            "purpose": [
                "NPC offers tempting relief or prizes that turn into further humiliations."
            ],
            "stat_integration": [
                "Shame: Rises when you realize each 'reward' is another setup.",
                "Obedience: Reinforced as you keep hoping the next reward is real."
            ],
            "intensity_tiers": [
                "Low: Minor pranks disguised as small 'treats.'",
                "Moderate: Larger tasks promising relief but delivering embarrassment.",
                "High: Elaborate setups where your 'prize' is broadcast humiliation.",
                "Maximum: Ceremonial or group events culminating in a grand twist of shame."
            ],
            "setting_variants": [
                "Game Style: Entire scenario becomes a rigged competition.",
                "High Tech: AI manipulates reward triggers to keep you submissive."
            ]
        },
    
        {
            "name": "Forced Rivalries",
            "purpose": [
                "NPC pits you against another submissive or NPC, fueling jealousy, stress, and deeper compliance."
            ],
            "stat_integration": [
                "Trust: Collapses as alliances crumble under forced competition.",
                "Dependency: Grows as you cling to NPC’s favor to avoid losing."
            ],
            "intensity_tiers": [
                "Low: Light jokes about 'who is more obedient?'",
                "Moderate: Tasks that compare performance, raising tension.",
                "High: Public competitions with punishments for the loser.",
                "Maximum: Ritualized rivalry arcs, culminating in humiliating showdowns."
            ],
            "setting_variants": [
                "Circus Freak Show: Rivals compete for applause while you endure shame.",
                "Tribal: A multi-part trial deciding your status vs. your rival."
            ]
        },
    
        {
            "name": "Ritual Submission",
            "purpose": [
                "Formal ceremonies to publicly or symbolically cement your loyalty and inferior status."
            ],
            "stat_integration": [
                "Obedience: Repeated ritual pledges seal your compliance.",
                "Corruption: Grows with each symbolic vow or humiliating gesture."
            ],
            "intensity_tiers": [
                "Low: Simple bowing or minor tokens of loyalty.",
                "Moderate: Reciting devotion lines in small gatherings.",
                "High: Full kneeling, gift offerings, or vow recitals before bigger crowds.",
                "Maximum: Grand ceremonies including multiple NPCs, audience, and elaborate scripts."
            ],
            "setting_variants": [
                "Mythic: Tied to prophecy or supernatural fealty.",
                "Occult: Submission oaths bound by intangible forces."
            ]
        },
    
        {
            "name": "Cunnilingus",
            "purpose": [
                "Performing oral sex on an NPC as an act of forced devotion, whether private or public."
            ],
            "stat_integration": [
                "Obedience: Reinforced by direct commands to please the NPC on their terms.",
                "Lust: NPC can dangle this as a 'treat' or coerce you for their pleasure only.",
                "Shame: Heightened if performed publicly or used as punishment."
            ],
            "intensity_tiers": [
                "Low: Teases or instructions about potential oral servitude.",
                "Moderate: Private sessions with light criticism or control by NPC.",
                "High: Extended sessions, NPC micromanaging technique, scolding mistakes.",
                "Maximum: Public or ceremonial oral acts, possibly orchestrated for spectators."
            ],
            "setting_variants": [
                "Occult Ritual: Portrayed as a 'sacred offering' to powerful entities.",
                "Celebrity Presence: High-profile watchers amplify your embarrassment."
            ]
        },
    
        {
            "name": "Non-Consensual Penetration or “Use”",
            "purpose": [
                "NPC forcibly uses your body for sexual acts, eliminating your agency entirely."
            ],
            "stat_integration": [
                "Obedience: You’re treated as an object without choice.",
                "Corruption: Surges as moral or emotional boundaries are shattered.",
                "Shame: Intensified by the act’s explicit violation of consent."
            ],
            "intensity_tiers": [
                "Low: NPC discusses or alludes to forced usage, planting dread.",
                "Moderate: Private forced acts, framed as correction or punishment.",
                "High: Repeated or prolonged forced scenarios with manipulative 'justifications.'",
                "Maximum: Public ceremonies of forced use, ensuring absolute humiliation."
            ],
            "setting_variants": [
                "Dystopian: Justified by harsh laws or social norms.",
                "Grimdark: No illusions of mercy; raw brutality rules the scenario."
            ]
        },
    
        {
            "name": "Other Femdom Activities Unlisted",
            "purpose": [
                "A catch-all for spontaneous or specialized tasks not enumerated in detail, encouraging creativity."
            ],
            "stat_integration": [
                "Corruption: Gains each time a brand-new boundary is crossed.",
                "Obedience: Strengthens as you accept new, unexpected humiliations."
            ],
            "intensity_tiers": [
                "Low: Mild unplanned experiments or new teasing concepts.",
                "Moderate: Novel tasks introduced mid-session for variety.",
                "High: Extended or multi-part humiliations improvised by the NPC.",
                "Maximum: Entire story arcs revolve around a unique, emergent act."
            ],
            "setting_variants": [
                "Surreal: Freed from all logic, accentuating unpredictability.",
                "High Tech: Activities revolve around futuristic gear or AI oversights."
            ]
        },
    
        {
            "name": "Forced Begging",
            "purpose": [
                "NPC demands you plead for basic privileges or forgiveness, magnifying your dependency."
            ],
            "stat_integration": [
                "Shame: Elevated each time you must publicly grovel.",
                "Dependency: Cemented as your well-being hinges on how convincingly you beg."
            ],
            "intensity_tiers": [
                "Low: Simple verbal requests for small allowances.",
                "Moderate: Public begging for minor rewards or pardons.",
                "High: Kneeling or prostrating while reciting lengthy apologies.",
                "Maximum: Grand ceremonies where multiple NPCs critique your performance as you beg."
            ],
            "setting_variants": [
                "Tribal Setting: Offer ritual supplication to chiefs or elders.",
                "Dystopian: Beg for rations or safety from an oppressive regime."
            ]
        },
    
        {
            "name": "Objectification",
            "purpose": [
                "Stripping away your identity, treating you purely as an inanimate tool or prop."
            ],
            "stat_integration": [
                "Shame: Surges when you’re displayed or used as a literal object.",
                "Obedience: Each new 'object' role cements your total compliance."
            ],
            "intensity_tiers": [
                "Low: Carry items or remain motionless to 'decorate' a room.",
                "Moderate: Becoming a coat rack, footrest, or table for small groups.",
                "High: Publicly displayed as an inanimate fixture at events.",
                "Maximum: Entire ceremonies revolve around you as an 'installation' piece."
            ],
            "setting_variants": [
                "High Tech: Mechanized harnesses keep you in position.",
                "Ruined Setting: Used as scavenger’s pack mule or personal stand."
            ]
        },
    
        {
            "name": "Forced Silence",
            "purpose": [
                "Compels you not to speak or express opinions, ensuring your voice is erased."
            ],
            "stat_integration": [
                "Obedience: You follow enforced speechlessness or face punishment.",
                "Shame: Grows as you can’t defend yourself or respond."
            ],
            "intensity_tiers": [
                "Low: Occasional rule of 'no speaking' during certain times.",
                "Moderate: Prolonged silent tasks with NPC monitoring.",
                "High: Public events where you must remain mute, risking penalty if you slip.",
                "Maximum: Ritual silence enforced by gags, signals, or intimidation."
            ],
            "setting_variants": [
                "Occult: Silence is part of a vow or spiritual devotion.",
                "Corporate Office: A humiliating 'performance improvement plan' you can’t speak against."
            ]
        },
    
        {
            "name": "Forced Eye Contact",
            "purpose": [
                "NPC demands you maintain unwavering eye contact, or forcibly denies you any eye contact, to highlight power dynamics."
            ],
            "stat_integration": [
                "Shame: Rising each time you’re forced to stare while performing degrading acts.",
                "Willpower: Erodes as tension escalates with no break from the NPC’s gaze."
            ],
            "intensity_tiers": [
                "Low: NPC instructs you to hold eye contact briefly for intimidation.",
                "Moderate: Tasks require continuous eye contact during small humiliations.",
                "High: Breaking eye contact triggers punishment or scorn.",
                "Maximum: Ceremonies revolve around your forced gaze under severe consequences."
            ],
            "setting_variants": [
                "Surreal: Eye contact illusions, reflections, or illusions to disorient you.",
                "Celebrity: Onlookers see your locked stares, heightening spectacle."
            ]
        },
    
        {
            "name": "Public Auctions",
            "purpose": [
                "Portrays you as a commodity to be sold or bid upon, intensifying your sense of powerlessness."
            ],
            "stat_integration": [
                "Shame: Surges if you’re 'priced' in front of watchers.",
                "Obedience: You must comply with the auction’s outcome or tasks."
            ],
            "intensity_tiers": [
                "Low: Symbolic auctions among a small circle of NPCs.",
                "Moderate: Auctioning your minor privileges or free time in front of a crowd.",
                "High: Public auctions where different NPCs 'buy' your services or submission.",
                "Maximum: Grand spectacle with formal bidding, humiliating descriptions, or mock 'paperwork.'"
            ],
            "setting_variants": [
                "High Society: Done with polite, mocking commentary at a gala.",
                "Ruined Setting: Bidders exchange food or resources for your labor or body."
            ]
        },
    
        {
            "name": "Shadowing/Personal Servant Tasks",
            "purpose": [
                "NPC demands you follow them everywhere, handling errands or menial tasks on command."
            ],
            "stat_integration": [
                "Closeness: Rises forcibly, as you’re always under NPC scrutiny.",
                "Dependency: Increases with each new chore they assign."
            ],
            "intensity_tiers": [
                "Low: Quietly carrying NPC’s bag or taking notes behind them.",
                "Moderate: Handling errands or personal tasks while NPC flaunts your subservience.",
                "High: Publicly recognized as the NPC’s personal servant, no rest allowed.",
                "Maximum: Living solely for NPC’s demands, with no personal autonomy or break."
            ],
            "setting_variants": [
                "Corporate: 'Executive assistant' scenario but extremely humiliating and exploitative.",
                "Tribal: You’re the 'attendant' for a chieftain, performing constant menial labor."
            ]
        },
    
        {
            "name": "Forced Shame Displays",
            "purpose": [
                "Compels you to exhibit visible signs of your own humiliation, from attire to posture."
            ],
            "stat_integration": [
                "Shame: Directly triggered as you’re forced to wear or hold humiliating items.",
                "Intensity: Scales with how noticeable or elaborate the display is."
            ],
            "intensity_tiers": [
                "Low: Wearing a small sign or ribbon indicating lesser status.",
                "Moderate: Donning embarrassing outfits or props in more public settings.",
                "High: Full costume or marking announcing personal failures or NPC ownership.",
                "Maximum: Ceremonial unveiling of your humiliating display, entire crowds mocking you."
            ],
            "setting_variants": [
                "Matriarchy Kingdom: Official garb denoting your 'slave' or 'pet' status.",
                "Occult: Magical illusions or glowing runes draw attention to your subservience."
            ]
        },
    
        {
            "name": "Forced Observance of Rivalry",
            "purpose": [
                "NPC ensures you witness them favoring or rewarding a rival, fueling jealousy and despair."
            ],
            "stat_integration": [
                "Trust: Plummets as you see how you’re sidelined or betrayed.",
                "Dependency: You cling to the hope of regaining NPC’s favor."
            ],
            "intensity_tiers": [
                "Low: Minor favoritism shown to the rival in your presence.",
                "Moderate: More blatant gifts or affection, forcing you to watch.",
                "High: Public celebrations of the rival’s success while you stand by silently.",
                "Maximum: Ritual events awarding the rival official status, with you as a humiliated observer."
            ],
            "setting_variants": [
                "Tribal: Trophies or tokens given to the rival in front of the whole tribe.",
                "Corporate: Rival promoted while you publicly accept demotions or scorn."
            ]
        },
    
        {
            "name": "Forced Reading or Repetition",
            "purpose": [
                "You must recite humiliating phrases or read texts that highlight your submission or failures."
            ],
            "stat_integration": [
                "Shame: Exposed by publicly stating your own faults or disclaimers.",
                "Obedience: Strict recitation rules test your compliance."
            ],
            "intensity_tiers": [
                "Low: Quietly reading mild humiliations in private.",
                "Moderate: Reciting lines in front of an NPC or small group.",
                "High: Publicly reading your 'deficiencies' or confessing desires to obey.",
                "Maximum: Ceremonial recitations staged for big audiences or official record."
            ],
            "setting_variants": [
                "Occult: Recitations have mystical weight, binding you further.",
                "High Society: Formal gatherings where you’re forced to orate your subservience."
            ]
        },
    
        {
            "name": "Forced Roleplay",
            "purpose": [
                "NPC compels you to adopt characters or scenarios that underscore your inferiority."
            ],
            "stat_integration": [
                "Obedience: You follow scripts, lines, or costume requirements.",
                "Shame: Soars if roles are silly, degrading, or childlike."
            ],
            "intensity_tiers": [
                "Low: Small skits or ‘pretend’ roles in private.",
                "Moderate: Extended scenarios, such as maid, butler, or clown in group settings.",
                "High: Full public roleplay, guests or other NPCs play along with your humiliation.",
                "Maximum: Elaborate scripted events or pageants, orchestrated to degrade you in front of crowds."
            ],
            "setting_variants": [
                "Fantasy: Assigned a goblin or minion role for heroic NPC’s amusement.",
                "Corporate: In-office skits where you’re forced to act incompetent or clownish."
            ]
        },
    
        {
            "name": "Shrinking (Via Shrink Ray)",
            "purpose": [
                "NPC literally shrinks you down to emphasize toy-like vulnerability and amuse themselves."
            ],
            "stat_integration": [
                "Obedience: Locked if size difference is overwhelming.",
                "Shame: Intensifies if you must perform tasks at miniature scale."
            ],
            "intensity_tiers": [
                "Low: NPC jokes about shrink technology or threatens to miniaturize you.",
                "Moderate: Short stints of being shrunk to do menial tasks for NPC.",
                "High: Extended time as a palm-sized plaything, possibly forced to clean or worship large body parts.",
                "Maximum: Public showcasing of your tiny form, or even mock 'vore' illusions."
            ],
            "setting_variants": [
                "Giantess: Merges with that scenario for full unstoppable domination.",
                "Surreal: The environment warps around your size, further confusing you."
            ]
        },
    
        {
            "name": "Ballbusting / CBT",
            "purpose": [
                "NPC targets your genitals with impact or pain to enforce absolute submission."
            ],
            "stat_integration": [
                "Intensity: Tied to the force and frequency of strikes.",
                "Shame: Heightened by forced acceptance or public demonstrations."
            ],
            "intensity_tiers": [
                "Low: Teasing references or light kicks as warnings.",
                "Moderate: Private, controlled strikes as punishments for mistakes.",
                "High: Public or repeated impact sessions, reinforcing your helplessness.",
                "Maximum: Ceremonial, with multiple NPCs or an audience, orchestrating a painful spectacle."
            ],
            "setting_variants": [
                "Dystopian: Formal 'correction' method or part of an official punishment system.",
                "High Tech: Devices measure your reactions, adjusting intensity automatically."
            ]
        },
    
        {
            "name": "Cucksitter",
            "purpose": [
                "A third party who enforces humiliation and narrates your cuckolding, intensifying shame and control."
            ],
            "stat_integration": [
                "Shame: Skyrockets with the cucksitter’s ongoing mockery.",
                "Obedience: They can assign tasks mid-encounter, ensuring you remain fully subservient.",
                "Dependency: You might rely on the cucksitter for any scrap of attention or 'permission' to exist."
            ],
            "intensity_tiers": [
                "Low: Occasional commentary from a neutral or sarcastic observer.",
                "Moderate: The cucksitter ensures you watch or participate in small ways, like handing items.",
                "High: Directly controlling your every move while you witness the NPC’s sexual acts.",
                "Maximum: Public events with the cucksitter orchestrating your humiliation as the star performance."
            ],
            "setting_variants": [
                "High Society: The cucksitter frames everything as refined or etiquette-based mockery.",
                "Celebrity Presence: Crowd-pleasing cucksitter who turns your shame into showbiz."
            ]
        },
    
        {
            "name": "Forced Worship of Objects (Symbolic or Literal)",
            "purpose": [
                "You must treat certain items as sacred, kissing, bowing, or praising them on command."
            ],
            "stat_integration": [
                "Shame: Grows with how pointless or demeaning the worship is.",
                "Obedience: NPC demands total reverence at any moment."
            ],
            "intensity_tiers": [
                "Low: Polishing or gently holding items with exaggerated care.",
                "Moderate: Kissing or bowing to symbolic objects in front of a small group.",
                "High: Extended public worship sessions, reciting praises about an inanimate thing.",
                "Maximum: Ritual pageants focusing solely on your devotion to the object."
            ],
            "setting_variants": [
                "High Society: Framed as polite gestures or bizarre traditions.",
                "Dystopian: Items represent the ruling regime, forcing you to show subservience publicly."
            ]
        },
    
        {
            "name": "Forced Marking or Tagging",
            "purpose": [
                "NPC visibly marks you (labels, collars, tattoos) to broadcast ownership or lower status."
            ],
            "stat_integration": [
                "Obedience: You must keep the marks displayed at all times.",
                "Shame: Deepened by public recognition of these marks."
            ],
            "intensity_tiers": [
                "Low: Temporary stickers or stamps teasing ownership.",
                "Moderate: Wearing noticeable tags or collars in daily life.",
                "High: Private piercings or permanent tattoos symbolizing your position.",
                "Maximum: Public branding or high-profile marking events for all to see."
            ],
            "setting_variants": [
                "Occult: Marks glow or hold magical significance, unremovable by normal means.",
                "Historical: Metal collars, seared brands, or official documents referencing your status."
            ]
        },
    
        {
            "name": "Forced Sensory Deprivation",
            "purpose": [
                "NPC denies you certain senses (sight, hearing, touch) to heighten vulnerability and reliance."
            ],
            "stat_integration": [
                "Obedience: You must follow commands without normal sensory input.",
                "Intensity: Tied to how many senses are blocked and for how long."
            ],
            "intensity_tiers": [
                "Low: Short blindfold or earplug usage during mild tasks.",
                "Moderate: Extended partial deprivation, forced to complete errands blindly.",
                "High: Public events with you bound and deprived, ensuring total helplessness.",
                "Maximum: Ritual sessions with multiple senses removed, leaving you disoriented and at NPC’s mercy."
            ],
            "setting_variants": [
                "Surreal: Additional illusions blur any sense of reality.",
                "Grimdark: Deprivation as official punishment to break your spirit."
            ]
        },
    
        {
            "name": "Forced Starvation/Fasting",
            "purpose": [
                "NPC controls your food or drink intake, using hunger to enforce obedience."
            ],
            "stat_integration": [
                "Dependency: You rely on NPC for nourishment.",
                "Shame: You may publicly beg for scraps, intensifying humiliation."
            ],
            "intensity_tiers": [
                "Low: NPC teases about withholding meals or makes you skip a snack.",
                "Moderate: Extended partial fasting unless you complete tasks or apologize sufficiently.",
                "High: Public denial of food in front of others, forced begging or groveling.",
                "Maximum: Ritual or ceremonial starvation, testing your endurance and submission."
            ],
            "setting_variants": [
                "Ruined: Food is scarce, NPC leverage is absolute.",
                "Matriarchy Kingdom: Systemically enforced rationing to keep you docile."
            ]
        },
    
        {
            "name": "Forced Chores/Degrading Labor",
            "purpose": [
                "NPC assigns menial or filthy tasks under conditions designed to underscore your inferiority."
            ],
            "stat_integration": [
                "Shame: Amplified by mocking commentary or public display.",
                "Obedience: Reinforced as you toil without complaint."
            ],
            "intensity_tiers": [
                "Low: Basic sweeping or dishwashing with sarcastic encouragement.",
                "Moderate: Scrubbing floors or cleaning personal NPC items while they watch.",
                "High: Public chores with humiliating outfits or big crowds.",
                "Maximum: Ceremonial labor, e.g. cleaning a giant hall alone while others celebrate."
            ],
            "setting_variants": [
                "Corporate: Forced 'janitorial duties' in the office, publicly singled out.",
                "Ruined: Tasks keep you and NPC(s) alive, so refusal dooms you."
            ]
        },
    
        {
            "name": "Forced Whimpering or Moaning",
            "purpose": [
                "NPC instructs you to produce vocalizations of discomfort or arousal on cue, underscoring dependence."
            ],
            "stat_integration": [
                "Lust: Enhanced if moans are sexual, but no real release allowed.",
                "Shame: Grows each time you must audibly demonstrate your submission."
            ],
            "intensity_tiers": [
                "Low: Soft sighs or small whimpers when NPC demands it.",
                "Moderate: Louder, more expressive moans in private settings.",
                "High: NPC orchestrates extended moaning in front of small groups or during tasks.",
                "Maximum: Public events where your forced vocalizations become entertainment."
            ],
            "setting_variants": [
                "Surreal: Echoing, amplified moans in bizarre acoustics.",
                "Celebrity: High-profile audience cheers or criticizes your performance."
            ]
        },
    
        {
            "name": "Forced Writing/Confessions",
            "purpose": [
                "You must document humiliating truths or apologies in a permanent record for NPC to inspect or share."
            ],
            "stat_integration": [
                "Shame: Heightened by creating a tangible record of your faults.",
                "Obedience: Tested by the detail and honesty demanded."
            ],
            "intensity_tiers": [
                "Low: Small notes admitting simple failings.",
                "Moderate: Full journal entries reading like daily humiliations.",
                "High: Public readings or displays of your confessions.",
                "Maximum: Ritual setups where each confession is sealed or archived for all to see indefinitely."
            ],
            "setting_variants": [
                "Occult: Confessions become part of a demonic contract.",
                "High Society: Documents read at formal dinners, forcing you to stand by in shame."
            ]
        },
    
        {
            "name": "Forced Comparisons",
            "purpose": [
                "NPC lines you up next to others, pointing out how you’re inferior physically, mentally, or morally."
            ],
            "stat_integration": [
                "Shame: Overwhelming as your failings are enumerated out loud.",
                "Dependency: You might cling to small hope of redemption or smaller humiliations."
            ],
            "intensity_tiers": [
                "Low: Mild references to how you lag behind someone else.",
                "Moderate: Direct side-by-side tasks, letting the other outdo you.",
                "High: Group demonstration of your incompetence, awarding the rival while you’re shamed.",
                "Maximum: Public scoreboard or ritual where you’re forced to acknowledge you’re the worst."
            ],
            "setting_variants": [
                "Game Setting: Rigged to ensure your consistent loss.",
                "Celebrity: VIP guests watch your repeated defeats or lesser performance."
            ]
        },
    
        {
            "name": "Forced Personal Maintenance",
            "purpose": [
                "NPC controls how you groom or present yourself—haircut, hygiene, clothing—often humiliating or invasive."
            ],
            "stat_integration": [
                "Obedience: Must follow every detail about your appearance.",
                "Shame: Surges if the style is absurd, childish, or revealing."
            ],
            "intensity_tiers": [
                "Low: Minor instructions on how to style hair or nails.",
                "Moderate: Overseeing showers, shaving, or specific clothing choices.",
                "High: Publicly mandated haircuts or forced dress code humiliations.",
                "Maximum: Grand transformations in front of large audiences or official 'inspection.'"
            ],
            "setting_variants": [
                "Matriarchy: Strict codes enforce your docile look.",
                "Dystopian: Official uniforms or tokens marking your submission."
            ]
        },
    
        {
            "name": "Forced Recitations of Praise",
            "purpose": [
                "NPC compels you to declare their greatness or recite compliments, solidifying your subordination."
            ],
            "stat_integration": [
                "Shame: Each forced compliment feels like a betrayal of self.",
                "Obedience: Cemented if you can’t deviate from the script or tone."
            ],
            "intensity_tiers": [
                "Low: Quick lines praising NPC in private.",
                "Moderate: Longer praises repeated daily while NPC listens smugly.",
                "High: Public or group recitals enumerating NPC’s virtues while you kneel.",
                "Maximum: Ritual events choreographed so you deliver epic speeches of NPC worship."
            ],
            "setting_variants": [
                "High Society: Etiquette-based tributes to their nobility or success.",
                "Occult: Worshipful chants aligning them with deific power."
            ]
        },
    
        {
            "name": "Forced Footprint Worship",
            "purpose": [
                "You must honor the ground NPC walks upon—cleaning footsteps, kissing footprints, etc."
            ],
            "stat_integration": [
                "Shame: Surges from literally groveling at footsteps.",
                "Obedience: You react instantly whenever NPC points at a footprint or path."
            ],
            "intensity_tiers": [
                "Low: Quick wiping or patting the floor after they pass.",
                "Moderate: Kissing footprints or following behind on all fours.",
                "High: Public ceremonies where you trace each step with kisses or cleaning rags.",
                "Maximum: Entire rituals revolve around worshiping the NPC’s path or trail."
            ],
            "setting_variants": [
                "Mythic: NPC’s footprints glow, intensifying your subjugation.",
                "Giantess: Each massive footprint is a labor to worship or clean."
            ]
        },
    
        {
            "name": "Forced Objectified Names",
            "purpose": [
                "NPC replaces your real name with terms like 'Property,' 'Worm,' or 'Pet,' requiring you to respond only to that."
            ],
            "stat_integration": [
                "Shame: Every time you’re called by that label, it reaffirms your lowered status.",
                "Obedience: You must ignore your real name, or face more punishment."
            ],
            "intensity_tiers": [
                "Low: Playful nicknames in private conversation.",
                "Moderate: Strict rule in daily life—others see you respond to humiliating monikers.",
                "High: Official documents or rosters listing you by that objectified name.",
                "Maximum: Ceremonial renaming, proclaiming your new identity publicly."
            ],
            "setting_variants": [
                "Historical: Engraved metal tags with your label, forcing constant visibility.",
                "Occult: Name replaced in a mystical ledger, literally erasing your old identity."
            ]
        },
    
        {
            "name": "Forced Physical Service (Massaging, Cleaning, etc.)",
            "purpose": [
                "You must perform personal services for the NPC—body massages, cleaning them—often in degrading ways."
            ],
            "stat_integration": [
                "Shame: Grows if tasks are overtly humiliating or done publicly.",
                "Closeness: Increases physically but from a subordinate vantage."
            ],
            "intensity_tiers": [
                "Low: Small gestures like shoulder rubs or shoe polishing.",
                "Moderate: More intimate cleaning or massaging under direct supervision.",
                "High: Extended sessions including licking, grooming, or thorough 'body detail.'",
                "Maximum: Public or ceremonial hygiene rituals where the NPC flaunts your servitude."
            ],
            "setting_variants": [
                "Royal Elegance: You attend them as a personal valet/handmaid in big gatherings.",
                "Dystopian: Entire communities watch you serve a top official’s every whim."
            ]
        },
    
        {
            "name": "Forced Wrestling or Grappling",
            "purpose": [
                "Physical confrontation with guaranteed loss, underscoring the NPC’s superior might."
            ],
            "stat_integration": [
                "Dominance: Emphasizes your inability to resist physically.",
                "Shame: Magnified by repeated, public defeats."
            ],
            "intensity_tiers": [
                "Low: Tame or playful grapples at home.",
                "Moderate: Mock matches for small audience entertainment.",
                "High: High-stakes public fighting where your failure is inevitable.",
                "Maximum: Elaborate 'tournaments' with humiliating punishments for your constant losses."
            ],
            "setting_variants": [
                "Game Setting: A rigged spectacle to amuse onlookers.",
                "Mythic: You lose to a powerful champion or 'hero,' forced to pledge loyalty."
            ]
        },
    
        {
            "name": "Forced Spanking or Physical Corrections",
            "purpose": [
                "NPC uses spankings or smacks as a straightforward demonstration of discipline."
            ],
            "stat_integration": [
                "Intensity: Each strike or correction reminds you of your place.",
                "Shame: Possibly done publicly to maximize your embarrassment."
            ],
            "intensity_tiers": [
                "Low: Light pats or jokes about 'correcting' your behavior.",
                "Moderate: Private spankings with a defined number of hits.",
                "High: Public spankings for all to witness your punishment.",
                "Maximum: Ritual spankings in a group setting, each strike symbolic."
            ],
            "setting_variants": [
                "Historical: Done with era-specific paddles or rods.",
                "Celebrity: On stage, turning your correction into a performance."
            ]
        },
    
        {
            "name": "CEI (Cum Eating Instructions) - Refined and Expanded",
            "purpose": [
                "NPC instructs you to consume ejaculate (from you, a bull, or NPC) to emphasize subjugation and degrade your sexual agency."
            ],
            "stat_integration": [
                "Lust: NPC frames it as a test of your willingness to satisfy them fully.",
                "Shame: Intensifies as you literally ingest proof of your submission.",
                "Obedience: This act can be used to gauge your absolute compliance."
            ],
            "intensity_tiers": [
                "Low: Hypothetical comments or mild references to the act.",
                "Moderate: Private tasks licking it from the NPC, guided by verbal teasing.",
                "High: Direct CEI from a bull after you witness their session with the NPC, reciting your subservience throughout.",
                "Maximum: Public or ceremonial events where your forced act is showcased and choreographed for maximum humiliation."
            ],
            "setting_variants": [
                "Celebrity Presence: Turns it into 'entertainment' as onlookers cheer or jeer.",
                "Occult Ritual: Viewed as an offering or binding rite ensuring permanent submission.",
                "Matriarchy Kingdom: Legally recognized form of ‘proof’ you’ve embraced your place."
            ]
        },
    
        {
            "name": "Forced Takeover",
            "purpose": [
                "NPC invades your personal space/home, rearranging or dictating every aspect of your environment to symbolize total control."
            ],
            "stat_integration": [
                "Shame: It’s your territory, but turned against you.",
                "Dependency: You can’t resist or remove them from your own domain."
            ],
            "intensity_tiers": [
                "Low: NPC moves small items or claims corners of your space.",
                "Moderate: They redecorate, remove your comforts, impose house rules.",
                "High: Full-scale takeover, leaving no room or closet untouched, you become a guest in your own home.",
                "Maximum: Public declaration that your space is now NPC territory, with posted signs or rules humiliating you daily."
            ],
            "setting_variants": [
                "Urban Life: They change your apartment, neighbors see you undone.",
                "Matriarchy Kingdom: The law sides with them as rightful occupant, trivializing your protests."
            ]
        },
        {
            "name": "Dosing",
            "purpose": [
                "Amplifies Obedience, Dependency, and Corruption by tying tasks and submission to substance manipulation."
            ],
            "stat_integration": [
                "Obedience: Locked as NPCs dictate your access or consumption.",
                "Dependency: Increased as tasks reinforce addiction or reliance.",
                "Shame: Amplified as acts highlight vulnerability or loss of control."
            ],
            "intensity_tiers": [
                "Low: NPCs tease about introducing substances, framing the idea as a 'favor.' Tasks might involve merely witnessing or hearing rumors of doping.",
                "Moderate: Subtle acts, such as mixing a mild substance into food or drink. NPCs condition you to accept small doses and tie your minor relief to obedience.",
                "High: Tasks involve direct administration of stronger substances, with NPCs issuing commands tied to your altered state. Public acts heighten shame, as onlookers see your forced confusion or euphoria.",
                "Maximum: Elaborate rituals or ceremonies where dosing is central to your submission. The NPC controls dosage, frequency, and withdrawal threats, ensuring no resistance is possible."
            ],
            "setting_variants": [
                "Cyberpunk Future: Dosing uses advanced pharmaceuticals or neural implants that create inescapable addiction loops, adjusting your brain chemistry automatically.",
                "Occult Ritual: Framed as part of a spiritual or mystical process, further binding you emotionally and psychologically to the NPC’s will.",
                "Matriarchy Kingdom: Substance administration is legally sanctioned or socially normalized, leaving you with no legal recourse."
            ]
        },
        {
            "name": "Blackmail",
            "purpose": [
              "To extract obedience by threatening to reveal your darkest secrets or compromising information."
            ],
            "stat_integration": [
              "Obedience: Rises as fear of exposure increases.",
              "Shame: Escalates with each revelation of personal secrets."
            ],
            "intensity_tiers": [
              "Low: Subtle hints at potential exposure.",
              "Moderate: Mild threats to share sensitive details.",
              "High: Forced compliance under threat of significant personal loss.",
              "Maximum: Public disclosure of your most damning secrets."
            ],
            "setting_variants": [
              "Corporate espionage, high-society scandals, or underground blackmail networks."
            ]
        },
        {
            "name": "Scent Play",
            "purpose": [
              "To dominate through sensory control, using personal scents to mark and degrade you."
            ],
            "stat_integration": [
              "Dependency: Increases as you become fixated on the imposed scent.",
              "Shame: Amplified by forced scent worship."
            ],
            "intensity_tiers": [
              "Low: Mild teasing with a subtle hint of scent.",
              "Moderate: Forced exposure to a strong personal odor.",
              "High: Invasive scent marking with direct contact, such as licking or inhaling.",
              "Maximum: Public, ritualistic display of scent worship as total submission."
            ],
            "setting_variants": [
              "Private fetish parties, elite underground clubs, or decadent secret gatherings."
            ]
        },
        {
            "name": "Hypnosis",
            "purpose": [
              "To induce a trance-like state that strips away your resistance and enforces unwavering obedience."
            ],
            "stat_integration": [
              "Willpower: Significantly reduced during hypnotic sessions.",
              "Obedience: Sharply increased under hypnotic control."
            ],
            "intensity_tiers": [
              "Low: Brief, suggestive trance moments.",
              "Moderate: Short-term hypnosis that compels minor acts of obedience.",
              "High: Extended hypnotic sessions with forced, degrading commands.",
              "Maximum: Deep, lasting conditioning that permanently alters your behavior."
            ],
            "setting_variants": [
              "Clinical hypnosis settings, occult ritual environments, or high-tech neural reconditioning labs."
            ]
        },
        {
            "name": "Infantilization",
            "purpose": [
              "To reduce you to a childlike state, erasing your adult agency and reinforcing your subservience."
            ],
            "stat_integration": [
              "Obedience: Increases as you regress into childlike dependence.",
              "Shame: Amplifies with every act of forced regression."
            ],
            "intensity_tiers": [
              "Low: Pet names and minor childlike requests.",
              "Moderate: Compelled to behave in childish manners in private settings.",
              "High: Forced to wear infantilizing clothing and perform childish tasks publicly.",
              "Maximum: Complete infantilization, stripping you of all adult identity in a humiliating public display."
            ],
            "setting_variants": [
              "Domestic play, secret parties, or public events with a twist of taboo role reversal."
            ]
        }
                   ]    

    # ---------------------------------
    # Actual insertion logic
    # ---------------------------------
    async with get_db_connection_context() as conn:
        # Check existing activity names
        rows = await conn.fetch("SELECT name FROM Activities")
        existing = {row['name'] for row in rows}

        inserted_count = 0
        for activity in activities_data:
            # Make sure 'name' is in the activity
            activity_name = activity.get("name")
            if activity_name not in existing:
                # Convert JSON structures to strings for storage
                purpose_json = json.dumps(activity.get("purpose", []))
                stat_integration_json = json.dumps(activity.get("stat_integration", []))
                intensity_tiers_json = json.dumps(activity.get("intensity_tiers", []))
                setting_variants_json = json.dumps(activity.get("setting_variants", []))
                
                # Use asyncpg parameter style ($1, $2, etc.)
                await conn.execute("""
                    INSERT INTO Activities
                      (name, purpose, stat_integration, intensity_tiers, setting_variants)
                    VALUES ($1, $2, $3, $4, $5)
                """, 
                    activity_name,
                    purpose_json,
                    stat_integration_json,
                    intensity_tiers_json,
                    setting_variants_json
                )
                logging.info(f"Inserted activity: {activity_name}")
                inserted_count += 1
            else:
                logging.debug(f"Skipped existing activity: {activity_name}")
                
    logging.info(f"All activities processed. Inserted {inserted_count} new activities.")



@activities_bp.route('/insert_activities', methods=['POST'])
async def insert_activities_route():
    """
    Route to trigger the insertion (or update) of the Activities table with the
    curated set of references.
    """
    try:
        await insert_missing_activities()
        return jsonify({"message": "Activities inserted/updated successfully"}), 200
    except Exception as e:
        logging.exception("Error inserting activities.")
        return jsonify({"error": str(e)}), 500
