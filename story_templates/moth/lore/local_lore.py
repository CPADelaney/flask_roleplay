# story_templates/moth/lore/local_lore.py
"""
Local lore - myths, histories, landmarks, and cultural elements for SF Bay Area
"""

from typing import Dict, Any, List

class SFLocalLore:
    """Local myths, histories, and cultural elements for SF Bay Area"""
    
    @staticmethod
    def get_specific_locations() -> List[Dict[str, Any]]:
        """Get story-specific detailed locations"""
        return [
            {
                "name": "The Velvet Sanctum",
                "address": "Valencia Street Basement (Unmarked)",
                "type": "bdsm_temple",
                "description": (
                    "Descend through the vintage boutique 'Forgotten Seasons,' past mannequins "
                    "dressed in another era's dreams. A hidden door behind the Victorian mourning "
                    "dress display. Down, down, down - each level darker than the last. The main "
                    "chamber opens like a cavern dressed in blood-red velvet. Candles flicker in "
                    "iron sconces, casting dancing shadows on devotees. At the center, an obsidian "
                    "throne where She holds court. The air tastes of leather, incense, and promises."
                ),
                "levels": {
                    "Level -1": "Reception and Evaluation - Where pilgrims are judged",
                    "Level -2": "The Main Chamber - Public worship space with throne",
                    "Level -3": "Private Rooms - For intimate surrenders",
                    "Level -4": "The True Dungeon - Where limits dissolve",
                    "Level -5": "Her Sanctuary - Private apartment, forbidden to most"
                },
                "schedule": {
                    "Monday": "8 PM - 3 AM (Devotional Monday)",
                    "Tuesday": "Private clients only",
                    "Wednesday": "8 PM - 4 AM (Performance Night)",
                    "Thursday": "8 PM - 3 AM (Training Night)",
                    "Friday": "8 PM - 5 AM (The Queen's Court)",
                    "Saturday": "8 PM - 5 AM (Grand Gathering)",
                    "Sunday": "Closed (Her day of rest)"
                },
                "staff": {
                    "Door Guardian": "Dmitri - Former Spetsnaz, owes her his life",
                    "House Mistress": "Valentina - Trains new submissives",
                    "Safety Master": "Jorge - Ensures all limits are respected"
                }
            },
            {
                "name": "The Butterfly House",
                "address": "Marina Boulevard (Appears as private residence)",
                "type": "primary_safehouse",
                "description": (
                    "A Queen Anne Victorian painted soft blue with white trim. To neighbors, it's "
                    "home to a reclusive tech heiress. In truth, it's where broken wings learn to "
                    "fly again. Seven bedrooms, each named for a butterfly. Medical facilities in "
                    "the basement. New identities crafted in the attic office. The garden grows "
                    "herbs that help with trauma. Guards who look like gardeners, security that "
                    "seems like art installations. Hope lives here."
                ),
                "rooms": {
                    "Monarch Suite": "For those ready to transform",
                    "Luna Room": "Night terrors and healing",
                    "Swallowtail": "Medical recovery",
                    "Blue Morpho": "Identity reconstruction lab",
                    "Garden Room": "Therapy and grounding"
                },
                "security": {
                    "External": "Cameras, motion sensors, silent alarms",
                    "Internal": "Panic rooms, escape tunnel to marina",
                    "Personnel": "Rotating guards, all trafficking survivors"
                }
            },
            {
                "name": "The Bone Garden",
                "address": "16th and Mission (Basement of El Corazón Negro)",
                "type": "goth_club_entrance",
                "description": (
                    "A goth club that serves as cover for one of the safehouse entrances. "
                    "Victorian mourning aesthetic meets Mexican Day of the Dead. Those who "
                    "know the phrase 'the moths seek light' find more than dancing here."
                ),
                "features": [
                    "Main floor - Gothic dance club",
                    "VIP area - Information exchange",
                    "Office - Hidden tunnel entrance",
                    "Emergency exit - Leads to Mission tunnels"
                ]
            }
        ]
    
    @staticmethod
    def get_urban_myths() -> List[Dict[str, Any]]:
        """Get SF-specific urban myths for the story"""
        return [
            {
                "name": "The Fog Mother",
                "type": "protector_legend",
                "description": (
                    "When the fog rolls thick through the Tenderloin, they say She walks within it. "
                    "The Moth Queen becomes one with the mist, spiriting away those about to be taken. "
                    "Traffickers speak of girls vanishing into fog that moves against the wind. Some "
                    "say she commands the fog itself, that Karl the Fog is her lover and conspirator. "
                    "Leave a moth drawn in window condensation if you need her protection."
                ),
                "origin_location": "Tenderloin",
                "spread_regions": ["Mission", "SoMa", "Financial District"],
                "believability": 7,
                "truth_level": "Metaphorically true - she uses fog as cover"
            },
            {
                "name": "The Pyramid's Dark Heart",
                "type": "location_legend",
                "description": (
                    "The Transamerica Pyramid isn't just an office building. Deep beneath, where "
                    "foundations meet old Shanghai tunnels, the ultra-wealthy built a temple to their "
                    "darkest desires. Entry requires a gold moth pin and a terrible secret. The Moth "
                    "Queen has infiltrated it three times, each visit followed by a CEO's mysterious "
                    "downfall. They say she has videos that could topple the city's power structure."
                ),
                "origin_location": "Financial District",
                "spread_regions": ["SoMa", "Pacific Heights"],
                "believability": 5,
                "truth_level": "Partially true - there is an exclusive underground club"
            },
            {
                "name": "The Valencia Street Vanishings",
                "type": "historical_legend",
                "description": (
                    "Every year, on the anniversary of the Great Quake, someone disappears from "
                    "Valencia Street between midnight and dawn. But unlike other vanishings, these "
                    "people reappear months later in other cities, with new names and mysterious "
                    "benefactors. They never speak of what happened, but they all have one thing "
                    "in common - a small moth tattoo. The saved marking the saved."
                ),
                "origin_location": "Mission District",
                "spread_regions": ["Castro", "Noe Valley"],
                "believability": 8,
                "truth_level": "True - it's the annual rescue operation"
            },
            {
                "name": "The Golden Gate Jumper Who Flies",
                "type": "transformation_legend",
                "description": (
                    "Bridge workers whisper about the woman who jumped but grew wings. They say "
                    "she was trafficked, escaped, but had nowhere to go. As she fell, she transformed "
                    "into a massive moth and flew back to the city. Now she saves others from the same "
                    "fate. On foggy nights, maintenance crews report seeing giant wings in their lights."
                ),
                "origin_location": "Golden Gate Bridge",
                "spread_regions": ["Entire Bay Area"],
                "believability": 3,
                "truth_level": "Mythological transformation of real rescue"
            },
            {
                "name": "The Tech Bro Harvest",
                "type": "cautionary_legend",
                "description": (
                    "Young tech workers with too much money and too little sense vanish from SoMa "
                    "clubs. They say the Queen selects the worst - those who treat people as objects "
                    "to be optimized. They're found days later, accounts drained, dignity shattered, "
                    "with footage of their depravities sent to their boards. A moth calling card left "
                    "behind. Silicon Valley's #MeToo has an avenging angel."
                ),
                "origin_location": "SoMa",
                "spread_regions": ["Financial District", "Palo Alto"],
                "believability": 6,
                "truth_level": "Some incidents are real"
            }
        ]
    
    @staticmethod
    def get_historical_events() -> List[Dict[str, Any]]:
        """Get historical events that shape the story world"""
        return [
            {
                "event_name": "The Great Quake of 1906 - Hidden Truth",
                "date": "April 18, 1906",
                "location": "San Francisco",
                "description": (
                    "Official records speak of earthquake and fire. Hidden archives tell of the "
                    "trafficking rings destroyed in the chaos, the brothels that burned with their "
                    "prisoners inside. The first Moth Queen arose from these ashes - a madam who "
                    "chose to save rather than sell. The tunnel systems dug for escape still exist."
                ),
                "significance": 9,
                "impact": "Created the underground tunnel network still used today",
                "legacy": "The Moth Queen tradition began here"
            },
            {
                "event_name": "The Summer of Love's Dark Side",
                "date": "Summer 1967",
                "location": "Haight-Ashbury spreading to Mission",
                "description": (
                    "While hippies preached free love, predators hunted the runaways flooding the city. "
                    "Seventeen flower children vanished, their disappearances dismissed as 'joining "
                    "communes.' The second Moth Queen emerged then - a dominatrix who used her dungeon "
                    "as a safehouse. She saved thirty-three that summer. Her methods inspired the current "
                    "system of hiding salvation in spaces of submission."
                ),
                "significance": 8,
                "impact": "Established the dual-purpose venue model",
                "legacy": "The Queen's two faces - domination and salvation"
            },
            {
                "event_name": "The Folsom Street Riots of 1990",
                "date": "September 30, 1990",
                "location": "SoMa District",
                "description": (
                    "When police raided leather bars searching for 'perverts,' the community fought back. "
                    "In the chaos, a trafficking ring's warehouse burned. Witnesses saw a woman in a "
                    "leather mask leading captives to safety through the smoke. The riots ended with an "
                    "unspoken truce: the underground polices itself. The Moth Queen was recognized as "
                    "legitimate authority in the shadow realm."
                ),
                "significance": 7,
                "impact": "Established underground self-governance",
                "legacy": "Police avoid the deep underground if it stays quiet"
            },
            {
                "event_name": "The Dot-Com Boom's Hidden Victims",
                "date": "1999-2001",
                "location": "City-wide",
                "description": (
                    "As tech money flooded in, so did demand for 'exclusive experiences.' High-end "
                    "trafficking rings emerged, catering to new millionaires. The third Moth Queen "
                    "infiltrated their ranks, gathering evidence while saving victims. When the bubble "
                    "burst, so did the rings - exposed by leaked videos. Twenty executives fled the "
                    "country. She used their abandoned assets to expand the safehouse network."
                ),
                "significance": 8,
                "impact": "Safehouse network expanded with tech money",
                "legacy": "Tech elite's fear and funding of the Queen"
            },
            {
                "event_name": "The Marina Massacre Prevention",
                "date": "June 15, 2018",
                "location": "Marina District",
                "description": (
                    "Intelligence suggested a trafficking ship would dock with fifty victims for a "
                    "buyer's convention. The current Moth Queen coordinated with sympathetic port "
                    "workers, underground allies, and even some cops who'd 'seen enough.' The ship "
                    "arrived to find its buyers arrested or vanished, victims already spirited away. "
                    "The only trace: moth wings painted on the empty containers."
                ),
                "significance": 9,
                "impact": "Largest single rescue in Bay Area history",
                "legacy": "Proved the underground network's power"
            }
        ]
    
    @staticmethod
    def get_cultural_elements() -> List[Dict[str, Any]]:
        """Get unique cultural elements of the SF underground"""
        return [
            {
                "name": "Fog Night Sanctuary",
                "type": "protective_tradition",
                "description": (
                    "On nights when the fog is thickest, all underground venues become sanctuary "
                    "spaces. No questions asked, no payment required. Anyone fleeing abuse or "
                    "trafficking can claim fog sanctuary. Even the darkest venues honor this."
                ),
                "frequency": "Whenever fog visibility < 100 feet",
                "participants": "All underground venues and workers"
            },
            {
                "name": "The Moth Migration",
                "type": "annual_ceremony",
                "description": (
                    "Every April 18th (earthquake anniversary), those saved by the network gather "
                    "in secret locations across the city. They release live moths at midnight, each "
                    "one representing a life saved. The Moth Queen appears at one location, never "
                    "announced in advance."
                ),
                "frequency": "Annual",
                "significance": "Celebrates survival and transformation"
            },
            {
                "name": "Confession Booths",
                "type": "underground_service",
                "description": (
                    "Modified photo booths in underground venues where people can confess desires, "
                    "crimes, or needs. Some confessions lead to scene negotiations, others to "
                    "safehouse rescues. The Moth Queen reviews dangerous ones personally."
                ),
                "locations": "Most underground venues",
                "protocol": "Anonymous, encrypted, selectively acted upon"
            },
            {
                "name": "The Folsom Truce",
                "type": "annual_agreement",
                "description": (
                    "During Folsom Street Fair, all factional conflicts pause. The celebration of "
                    "consensual kink overshadows darker business. Even traffickers avoid the area, "
                    "knowing the concentrated underground presence."
                ),
                "frequency": "Last Sunday in September",
                "effect": "24-hour ceasefire in underground conflicts"
            },
            {
                "name": "Tech Confessional",
                "type": "blackmail_tradition",
                "description": (
                    "Silicon Valley executives can 'confess' their crimes to the Velvet Court "
                    "and receive absolution - for a price. Usually involves massive donations to "
                    "victim services. Refusal means exposure. Many prefer paying to prison."
                ),
                "frequency": "Monthly closed sessions",
                "revenue": "Funds 40% of safehouse operations"
            }
        ]
    
    @staticmethod
    def get_landmarks() -> List[Dict[str, Any]]:
        """Get significant landmarks for the story"""
        return [
            {
                "name": "The Sutro Baths Ruins",
                "type": "ceremonial_site",
                "description": (
                    "The ruins of the old bathhouse serve as a meeting place for the underground. "
                    "Major decisions are made here, with the Pacific as witness. The Moth Queen "
                    "was crowned here, in a ceremony involving all faction leaders."
                ),
                "location": "Lands End",
                "significance": "Neutral ground for all factions",
                "ceremonies": ["Leadership changes", "Major truces", "Territory disputes"]
            },
            {
                "name": "The Castro Theatre",
                "type": "information_hub",
                "description": (
                    "More than a movie palace. The projection booth serves as a secure meeting "
                    "room. Classic films provide cover for information exchanges. The organist "
                    "plays coded messages in the pre-show music."
                ),
                "location": "Castro District",
                "significance": "Communications center for the underground"
            },
            {
                "name": "Grace Cathedral Labyrinth",
                "type": "sanctuary_marker",
                "description": (
                    "Walking the labyrinth at night signals need for sanctuary. The cathedral's "
                    "progressive leadership provides no-questions-asked shelter. Moths sometimes "
                    "appear in the stained glass reflections."
                ),
                "location": "Nob Hill",
                "significance": "Last resort sanctuary, even from the Queen herself"
            }
        ]
    
    @staticmethod
    def get_seasonal_events() -> List[Dict[str, Any]]:
        """Get regular events beyond daily operations"""
        return [
            {
                "name": "The Harvest Moon Hunt",
                "event_type": "protective_action",
                "description": (
                    "Every harvest moon, the Broken Dolls and allies sweep trafficking "
                    "locations. A night of coordinated raids. The Queen provides intel, "
                    "the Dolls provide violence. By dawn, victims are freed."
                ),
                "frequency": "Annual - Harvest moon",
                "participants": ["Broken Dolls", "Moth Queen network", "Sympathetic cops"],
                "traditions": [
                    "Silent coordination",
                    "No killing unless necessary",
                    "Every saved victim gets a moth pin"
                ],
                "typical_outcomes": ["5-20 rescued", "1-3 operations shut down"],
                "community_impact": "Reminds traffickers they're hunted"
            },
            {
                "name": "Leather Week Pilgrimage",
                "event_type": "religious_gathering",
                "description": (
                    "Annual gathering of the faithful. Public events hide private ceremonies. "
                    "The Queen holds special court, blessing unions and hearing petitions. "
                    "New dominants are recognized, new submissives welcomed."
                ),
                "frequency": "Annual - September",
                "participants": ["Entire BDSM community", "Tourists", "Curious vanillas"],
                "traditions": [
                    "Blessing of the leathers",
                    "Collar ceremonies",
                    "Public demonstrations",
                    "Private initiations"
                ],
                "typical_outcomes": ["Community bonding", "New members initiated"],
                "community_impact": "Strengthens underground unity"
            },
            {
                "name": "Night of Broken Masks",
                "event_type": "cathartic_ritual",
                "description": (
                    "One night when everyone drops their masks - dominants show vulnerability, "
                    "submissives show strength. The Queen traditionally removes all masks "
                    "this night. Role reversals allowed. Emotional breakthrough common."
                ),
                "frequency": "Quarterly - Solstices and equinoxes",
                "participants": ["Sanctum members only"],
                "traditions": [
                    "Mask burning ceremony",
                    "Truth circles",
                    "Role reversal scenes",
                    "Group aftercare"
                ],
                "typical_outcomes": ["Emotional release", "Deeper connections"],
                "community_impact": "Prevents burnout, builds trust"
            }
        ]
    
    @staticmethod
    def get_specialized_locations() -> List[Dict[str, Any]]:
        """Get additional specialized underground locations"""
        return [
            {
                "name": "The Chrysalis Medical Center",
                "location_type": "underground_hospital",
                "description": (
                    "Hidden medical facility for those who can't use regular hospitals. "
                    "Treats trafficking victims, scene injuries, and those needing discretion. "
                    "Dr. Martinez runs it with volunteer staff who don't ask questions."
                ),
                "services": [
                    "Trauma surgery",
                    "STI treatment",
                    "Injury care",
                    "Mental health support",
                    "Hormone therapy"
                ],
                "location": "Tenderloin basement",
                "access": "Referral only",
                "security": "Biometric locks, panic buttons, escape routes"
            },
            {
                "name": "The Identity Forge",
                "location_type": "document_center",
                "description": (
                    "Where new lives are created. Master forgers craft identities that "
                    "pass any scrutiny. Birth certificates, passports, entire histories. "
                    "Run by someone known only as 'The Scribe'."
                ),
                "services": [
                    "Complete identity packages",
                    "Supporting documentation",
                    "Digital footprint creation",
                    "Backstory coaching"
                ],
                "location": "Moves monthly",
                "access": "Moth Queen referral only",
                "security": "If you find it uninvited, it's already gone"
            },
            {
                "name": "The Memory Garden",
                "location_type": "memorial",
                "description": (
                    "Hidden rooftop garden where the lost are remembered. Each plant "
                    "represents someone who didn't make it out. The Queen tends it "
                    "personally. Moths breed here naturally."
                ),
                "features": [
                    "Memorial plants",
                    "Meditation space",
                    "Memory wall",
                    "Moth sanctuary"
                ],
                "location": "Above the Marina safehouse",
                "access": "Survivors and family only",
                "security": "Hidden from street view"
            }
        ]
    
    @staticmethod
    def get_communication_networks() -> List[Dict[str, Any]]:
        """Get how the underground communicates"""
        return [
            {
                "name": "The Moth Signal",
                "network_type": "emergency",
                "description": (
                    "Graffiti moths appear when danger threatens. Different wing positions "
                    "mean different warnings. A moth with spread wings means safe passage, "
                    "closed wings mean danger, burning moth means run."
                ),
                "coverage": "Citywide",
                "users": ["Anyone who knows the code"],
                "security": "Hidden in plain sight",
                "examples": [
                    "Moth on safehouse wall = operating",
                    "Moth with crown = Queen's protection",
                    "Dead moth = location compromised"
                ]
            },
            {
                "name": "The Velvet Wire",
                "network_type": "information",
                "description": (
                    "Encrypted app that looks like a dating platform but connects the "
                    "underground. Messages self-destruct, locations are approximate, "
                    "identities verified through web of trust."
                ),
                "coverage": "Digital",
                "users": ["Verified underground members"],
                "security": "End-to-end encryption, onion routing",
                "examples": [
                    "Event announcements",
                    "Safety warnings",
                    "Resource sharing"
                ]
            },
            {
                "name": "The Whisper Chain",
                "network_type": "human",
                "description": (
                    "Old-school human network. Messages passed person to person, "
                    "modified slightly each time for security. By the time cops "
                    "hear it, it's unrecognizable from the original."
                ),
                "coverage": "Underground venues and streets",
                "users": ["Everyone in the scene"],
                "security": "Plausible deniability",
                "examples": [
                    "Raid warnings",
                    "New arrival alerts",
                    "Help requests"
                ]
            }
        ]
    
    @staticmethod
    def get_underground_economies() -> List[Dict[str, Any]]:
        """Get economic systems within the underground"""
        return [
            {
                "name": "The Tribute System",
                "economy_type": "gift_economy",
                "description": (
                    "Submissives offer tribute to dominants - not payment but devotion "
                    "made tangible. Money, service, gifts, all freely given. The Queen's "
                    "tributes fund safehouses, though tributors don't know this."
                ),
                "currency": "Devotion acts",
                "major_trades": ["Service for attention", "Gifts for time", "Pain for pleasure"],
                "regulation": "Protocol and tradition",
                "key_players": ["Dominants", "Devoted submissives", "The Queen"],
                "estimated_value": "$2-3 million annually"
            },
            {
                "name": "The Safehouse Fund",
                "economy_type": "shadow_charity",
                "description": (
                    "Complex network of donations, blackmail payments, and guilt money "
                    "that funds rescue operations. Officially doesn't exist. Actually "
                    "saves hundreds of lives yearly."
                ),
                "currency": "Cryptocurrency and cash",
                "major_trades": ["Silence for funding", "Rescue for donors"],
                "regulation": "The Moth Queen alone",
                "key_players": ["Anonymous donors", "Keeper network", "Rescued victims"],
                "estimated_value": "$10+ million annually"
            },
            {
                "name": "The Skills Exchange",
                "economy_type": "barter_system",
                "description": (
                    "Underground members trade expertise. Rope skills for impact play "
                    "training, medical knowledge for legal advice. Creates tight bonds "
                    "and ensures skill preservation."
                ),
                "currency": "Time and expertise",
                "major_trades": ["Training for training", "Services for skills"],
                "regulation": "Community reputation",
                "key_players": ["Skilled practitioners", "Eager learners"],
                "estimated_value": "Invaluable"
            }
        ]
    
    @staticmethod
    def get_district_cultural_norms() -> List[Dict[str, Any]]:
        """Get detailed cultural norms for each SF district"""
        return [
            # Mission Underground Norms
            {
                "nation_id": "mission_underground",
                "category": "greeting",
                "description": (
                    "Greetings involve subtle moth gestures - touching a moth pin, "
                    "tracing wings on the palm. Never grab someone unexpectedly. "
                    "Eye contact is earned, not taken. The Queen is greeted with "
                    "lowered eyes unless she lifts your chin."
                ),
                "formality_level": "varies",
                "gender_specific": True,
                "female_variation": "Women may initiate eye contact, offer hand first",
                "male_variation": "Men wait for permission, keep eyes down initially",
                "taboo_level": 7,
                "consequence": "Grabbing without consent marks you as predator",
                "regional_variations": {
                    "Velvet Sanctum": "Kneel to greet dominants",
                    "Street level": "Casual nod suffices"
                }
            },
            {
                "nation_id": "mission_underground",
                "category": "dining",
                "description": (
                    "Sharing food is intimate. The Queen eats first at formal gatherings. "
                    "Feeding someone by hand shows dominance or deep care. Refusing "
                    "offered food is severe insult unless you cite dietary restrictions."
                ),
                "formality_level": "formal",
                "gender_specific": True,
                "female_variation": "Women may feed others, choose serving order",
                "male_variation": "Men serve, wait to eat until permitted",
                "taboo_level": 5,
                "consequence": "Refusing food means rejecting hospitality",
                "regional_variations": {
                    "Safe houses": "Communal eating, no hierarchy"
                }
            },
            {
                "nation_id": "mission_underground",
                "category": "authority",
                "description": (
                    "Authority flows from the Queen downward through the Velvet Court. "
                    "Challenge is allowed through proper channels - kneel, state grievance, "
                    "accept judgment. Public defiance brings swift retribution. Female "
                    "authority is absolute in the underground."
                ),
                "formality_level": "ceremonial",
                "gender_specific": True,
                "female_variation": "Women hold positions of power, make decisions",
                "male_variation": "Men influence through service and loyalty",
                "taboo_level": 9,
                "consequence": "Defying the Queen means exile or worse",
                "regional_variations": {}
            },
            {
                "nation_id": "mission_underground",
                "category": "gift_giving",
                "description": (
                    "Gifts are offerings, not transactions. Moths (jewelry, art) show "
                    "allegiance. Practical gifts (clothes, money) imply someone needs help. "
                    "The Queen's gifts are kept forever - discarding them is betrayal. "
                    "Anonymous gifts to safe houses are highest honor."
                ),
                "formality_level": "formal",
                "gender_specific": False,
                "taboo_level": 6,
                "consequence": "Discarding a dominant's gift shows disrespect",
                "regional_variations": {
                    "Wealthy areas": "Expensive tributes expected",
                    "Safe houses": "Practical gifts preferred"
                }
            },
            
            # Financial District After Hours Norms
            {
                "nation_id": "financial_district",
                "category": "business_conduct",
                "description": (
                    "Daytime personas never acknowledged at night. What happens in "
                    "private clubs stays buried. Blackmail is handled by the Court. "
                    "Money talks but submission speaks louder. CEOs kneel to no one "
                    "except those who hold their secrets."
                ),
                "formality_level": "formal",
                "gender_specific": False,
                "taboo_level": 10,
                "consequence": "Outing someone means total destruction",
                "regional_variations": {
                    "Apex Club": "Ultimate discretion required",
                    "Street level": "Don't acknowledge unless acknowledged"
                }
            },
            {
                "nation_id": "financial_district",
                "category": "gender_relations",
                "description": (
                    "Power dynamics flip after dark. Female dominants command men who "
                    "rule companies by day. This inversion is never discussed in daylight. "
                    "Submission is currency more valuable than stock options."
                ),
                "formality_level": "ceremonial",
                "gender_specific": True,
                "female_variation": "Women wield absolute power in these spaces",
                "male_variation": "Men find freedom in temporary powerlessness",
                "taboo_level": 8,
                "consequence": "Breaking the illusion ruins the escape",
                "regional_variations": {}
            }
        ]
    
    @staticmethod
    def get_underground_etiquette() -> List[Dict[str, Any]]:
        """Get etiquette rules for different underground contexts"""
        return [
            {
                "nation_id": "sf_underground",
                "context": "sanctum_court",
                "title_system": (
                    "The Queen (absolute), Court Members (by venue name), "
                    "Established Dominants (Sir/Ma'am/Mx), Submissives (by earned names), "
                    "Newcomers (no title until earned)"
                ),
                "greeting_ritual": (
                    "Submissives kneel or bow to dominants. Dominants nod to equals. "
                    "The Queen is greeted in whatever manner she demands that night. "
                    "Touch is privilege, not right. Wait for permission."
                ),
                "body_language": (
                    "Submissives: eyes down, hands visible, knees together or spread "
                    "as ordered. Dominants: straight posture, deliberate movements, "
                    "commanding presence. Never turn back on the Queen."
                ),
                "eye_contact": (
                    "Earned through service or granted as gift. Staring is aggressive. "
                    "The Queen's gaze is blessing or curse. Looking away first shows "
                    "submission. Holding her gaze requires permission."
                ),
                "distance_norms": (
                    "Three feet minimum unless invited closer. The Queen's space is "
                    "sacred - approach only when summoned. In scene, distance set by "
                    "dominant. Crowding is threat behavior."
                ),
                "gift_giving": (
                    "Tributes presented on knees. Never hand directly - place before "
                    "them. Gifts to Queen left at throne. Anonymous gifts respected. "
                    "Rejecting gifts means rejecting the giver."
                ),
                "dining_etiquette": (
                    "Queen eats first. Dominants served by their submissives. "
                    "Hand feeding shows intimacy or control. Never reach across "
                    "someone. Sharing drink is sharing essence."
                ),
                "power_display": (
                    "Dominants command space, submissives compress. Voice volume "
                    "shows rank - Queen speaks softly because all lean in to hear. "
                    "Clothing is armor or vulnerability by choice."
                ),
                "respect_indicators": (
                    "Kneeling, lowered eyes, offered hands, silent waiting, "
                    "anticipating needs, accepting pain, wearing their marks, "
                    "returning despite fear, keeping secrets."
                ),
                "gender_distinctions": (
                    "Female dominants addressed as Ma'am/Mistress/Goddess. "
                    "Male dominants as Sir/Master. Non-binary as Mx/Their chosen title. "
                    "The Queen is simply 'My Queen' or as she demands."
                ),
                "taboos": [
                    "Touching without consent",
                    "Speaking over the Queen",
                    "Breaking scene protocol",
                    "Revealing identities",
                    "Phone use during scenes",
                    "Vanilla-shaming",
                    "Comparing dominants"
                ]
            },
            {
                "nation_id": "sf_underground",
                "context": "safehouse",
                "title_system": (
                    "Keepers (house runners), Protectors (security), "
                    "Guests (refugees), Healers (medical/psychological)"
                ),
                "greeting_ritual": (
                    "Gentle, no sudden movements. Verbal consent before touch. "
                    "Names optional - safety first. Moth pins identify allies."
                ),
                "body_language": (
                    "Open palms, slow movements, respectful distance. "
                    "Never block exits. Sit below standing trauma victims."
                ),
                "eye_contact": (
                    "Optional - many avoid it. Follow their lead. "
                    "Direct stare can trigger. Soft focus preferred."
                ),
                "distance_norms": (
                    "Let them set distance. Back away if they flinch. "
                    "Announce movement. Never approach from behind."
                ),
                "gift_giving": (
                    "Practical items welcome. No strings attached. "
                    "Anonymity respected. Clothes, toiletries, phones valued."
                ),
                "dining_etiquette": (
                    "Communal, no hierarchy. Let them serve themselves. "
                    "Some won't eat while watched. Patience required."
                ),
                "power_display": (
                    "Minimize it. Protectors stay background. "
                    "Power used only for their safety."
                ),
                "respect_indicators": (
                    "Believing their story, not pushing for details, "
                    "respecting chosen names, maintaining boundaries"
                ),
                "gender_distinctions": (
                    "Use their chosen pronouns. No assumptions. "
                    "Some fear specific genders - accommodate."
                ),
                "taboos": [
                    "Asking real names",
                    "Demanding their story",
                    "Taking photos",
                    "Surprise touches",
                    "Loud noises",
                    "Blocking exits",
                    "Breaking confidentiality"
                ]
            }
        ]
    
    @staticmethod
    def get_mystical_phenomena() -> List[Dict[str, Any]]:
        """Get supernatural/psychological phenomena specific to setting"""
        return [
            {
                "name": "Subspace Visions",
                "phenomenon_type": "psychological",
                "description": (
                    "Deep submissives report shared visions during intense scenes. Always "
                    "featuring moths, flames, and a woman with three faces. Skeptics say "
                    "it's endorphins, believers say it's divine contact."
                ),
                "occurrence_rate": "1 in 100 deep scenes",
                "affected_groups": ["Experienced submissives", "Natural pain sluts"],
                "documented_effects": [
                    "Prophetic dreams",
                    "Sensing danger to the community",
                    "Knowing when someone needs help"
                ],
                "scholarly_theories": [
                    "Collective unconscious activation",
                    "Moth Queen psychic network",
                    "Brain chemistry creating shared hallucinations"
                ]
            },
            {
                "name": "The Vanishing",
                "phenomenon_type": "protective",
                "description": (
                    "People fleeing trafficking sometimes simply vanish from pursuit. One "
                    "moment they're cornered, the next gone. Always near moth graffiti. "
                    "The saved claim a woman in a mask led them through walls."
                ),
                "occurrence_rate": "Dozen times per year",
                "affected_groups": ["Trafficking victims", "Those who help them"],
                "documented_effects": [
                    "Spatial displacement",
                    "Pursuer memory gaps",
                    "Moth scales at scene"
                ],
                "scholarly_theories": [
                    "Underground tunnel expertise",
                    "Mass hypnosis",
                    "The Queen has supernatural powers"
                ]
            },
            {
                "name": "Dominant's Intuition",
                "phenomenon_type": "psychic",
                "description": (
                    "Experienced dominants develop uncanny ability to sense limits, needs, "
                    "and dangers. They know when to push and when to comfort. The Queen's "
                    "intuition is legendary - she knows your needs before you do."
                ),
                "occurrence_rate": "Develops after 5+ years",
                "affected_groups": ["Serious dominants", "Professional mistresses"],
                "documented_effects": [
                    "Knowing safe words before spoken",
                    "Sensing medical issues",
                    "Predicting emotional breakthroughs"
                ],
                "scholarly_theories": [
                    "Hypervigilance refinement",
                    "Pheromone sensitivity",
                    "Energy work is real"
                ]
            }
        ]
    
    @staticmethod
    def get_expanded_seasonal_events() -> List[Dict[str, Any]]:
        """More seasonal/cyclical events"""
        return [
            {
                "name": "Summer Fog Festival",
                "frequency": "Annual - July when fog is thickest",
                "description": (
                    "Three nights when the fog rolls in so thick you can't see five feet. "
                    "The underground celebrates their protector. Every venue becomes a "
                    "safehouse, every predator knows to stay hidden."
                ),
                "traditions": [
                    "Fog walks - groups move through the city helping the lost",
                    "Moth releases in the fog",
                    "Stories of salvation shared in the mist"
                ],
                "special_rule": "No enforcement during fog nights - unofficial truce"
            },
            {
                "name": "The Leather Equinox",
                "frequency": "Biannual - Spring and Fall equinoxes",
                "description": (
                    "When day equals night, dominants and submissives exchange roles. "
                    "The Queen serves, the servants rule. Power structures flip for 24 hours."
                ),
                "significance": "Prevents power stagnation, builds empathy",
                "taboos": [
                    "Refusing to switch",
                    "Taking advantage of temporary power",
                    "Breaking scene during the switch"
                ]
            },
            {
                "name": "Monthly Safehouse Rotation",
                "frequency": "Every new moon",
                "description": (
                    "Safehouses change entry codes and locations. The underground "
                    "railroad shifts its tracks. What was safe last month may not be now."
                ),
                "logistics": [
                    "Coded messages in Moth Tongue announce changes",
                    "Guides stationed at old locations for one night",
                    "Emergency protocols for those who miss the transition"
                ]
            }
        ]
    
    @staticmethod
    def get_myth_evolution_scenarios() -> List[Dict[str, Any]]:
        """How urban myths evolve over time in the SF underground"""
        return [
            {
                "myth_name": "The Fog Mother",
                "evolution_type": "spreading",
                "change_description": (
                    "The myth spreads beyond the Tenderloin. Now Mission techies claim to see her "
                    "in their Uber rides, and Marina mothers whisper her name when daughters "
                    "stay out too late."
                ),
                "new_variations": [
                    "She appears in AR/VR spaces, pulling victims from virtual trafficking",
                    "The fog now carries her voice - a mother's lullaby for the lost",
                    "Climate change means less fog, weakening her power"
                ],
                "believability_change": +2,
                "spread_rate_change": +3
            },
            {
                "myth_name": "The Tech Bro Harvest",
                "evolution_type": "intensifying",
                "change_description": (
                    "After three CEOs fled the country, the myth gains teeth. Now venture "
                    "capitalists hire bodyguards specifically trained to spot moth symbols."
                ),
                "new_elements": [
                    "Crypto wallets drained to fund safehouses",
                    "Dating apps used as hunting grounds",
                    "AI-generated blackmail from recorded sessions"
                ],
                "corporate_response": "Tech companies now screen for 'Queen exposure risk'"
            }
        ]
    
    @staticmethod
    def get_cultural_evolution_scenarios() -> List[Dict[str, Any]]:
        """How underground cultural elements develop"""
        return [
            {
                "element_name": "Confession Night",
                "evolution_type": "formalization",
                "change_description": (
                    "What started as impromptu gatherings now has strict protocols. "
                    "Professional 'Confessors' trained in both BDSM and therapy emerge."
                ),
                "new_practices": [
                    "Confession licenses issued by Velvet Court",
                    "Digital confession booths with voice modulation",
                    "Confession insurance for particularly dangerous admissions"
                ],
                "significance_change": +2
            },
            {
                "element_name": "The Moth Migration",
                "evolution_type": "digitization",
                "change_description": (
                    "Pandemic forced the ceremony online. Now global survivors release "
                    "digital moths in a synchronized AR experience. Physical and virtual "
                    "merge as participants share coordinates of their salvation."
                ),
                "technological_integration": [
                    "Blockchain verification of survivor stories",
                    "NFT moths that fund safehouse operations",
                    "VR support groups for those still healing"
                ]
            }
        ]
    
    @staticmethod
    def get_communication_evolution() -> List[Dict[str, Any]]:
        """How communication networks adapt"""
        return [
            {
                "network": "The Moth Signal",
                "evolution_stage": "augmented_reality",
                "changes": {
                    "old_method": "Physical graffiti moths",
                    "new_method": "AR moths visible through encrypted apps",
                    "advantages": "No vandalism charges, dynamic updates, precise GPS",
                    "preservation": "Physical moths still used for those without tech"
                },
                "new_features": [
                    "Moths animate to show threat direction",
                    "Color changes indicate threat level",
                    "Swarm patterns encode complex messages"
                ]
            },
            {
                "network": "The Whisper Chain",
                "evolution_stage": "AI_enhancement",
                "changes": {
                    "innovation": "AI scrambles messages uniquely for each link",
                    "benefit": "Even if intercepted, message is gibberish",
                    "risk": "AI could be compromised or pattern-matched"
                }
            }
        ]
    
    @staticmethod
    def get_underground_economy_evolution() -> List[Dict[str, Any]]:
        """How the shadow economy adapts"""
        return [
            {
                "economy_name": "The Tribute System",
                "evolution_type": "cryptocurrency_integration",
                "changes": {
                    "old_system": "Cash in envelopes, physical gifts",
                    "new_system": "Monero donations, NFT tributes, DeFi protocols",
                    "benefits": "Harder to trace, easier to redistribute to safehouses",
                    "risks": "Digital trail, volatility affecting safehouse funding"
                },
                "new_players": [
                    "Crypto Dommes teaching blockchain",
                    "DeFi developers creating 'TributeDAO'",
                    "Digital artists creating tribute NFTs"
                ]
            },
            {
                "economy_name": "Skills Exchange",
                "evolution_type": "professionalization",
                "changes": {
                    "certification_system": "Velvet Court now issues skill certifications",
                    "online_platform": "Encrypted skill-sharing platform launches",
                    "quality_control": "Peer review system for practitioners",
                    "expansion": "Legal, medical, tech skills added to traditional BDSM"
                }
            }
        ]
