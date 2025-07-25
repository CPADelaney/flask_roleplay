# story_templates/moth/lore/sf_bay_lore_preset.py
"""
San Francisco Bay Area preset lore for The Moth and Flame story
Predefined locations, myths, histories, and factions
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class SFBayMothFlamePreset:
    """Complete preset lore for The Moth and Flame set in SF Bay Area"""
    
    @staticmethod
    def get_world_foundation() -> Dict[str, Any]:
        """Get the base world setting"""
        return {
            "world_setting": {
                "name": "San Francisco Bay Metropolitan Area",
                "type": "modern_gothic_city",
                "population": "7.7 million",
                "established": "1776 (Mission), 1835 (Yerba Buena)",
                "description": (
                    "A city of stark contrasts where tech billionaires step over the homeless, "
                    "where Victorian painted ladies cast shadows on underground dungeons. The fog "
                    "rolls in each evening like a living thing, obscuring sins and swallowing screams. "
                    "Beneath the progressive facade thrives an economy of desire, desperation, and "
                    "dark salvation. They say the city's heart beats strongest after midnight."
                ),
                "atmosphere": {
                    "physical": "Fog-shrouded hills, cold ocean wind, Victorian shadows",
                    "social": "Tech wealth, artistic rebellion, hidden desperation",
                    "spiritual": "Lost souls seeking meaning in sensation"
                }
            }
        }
    
    @staticmethod
    def get_districts() -> List[Dict[str, Any]]:
        """Get the main districts/locations"""
        return [
            {
                "name": "The Mission Underground",
                "type": "entertainment_district",
                "description": (
                    "Beneath the Mission District's trendy surface lies the true underground. "
                    "Former speakeasies and bomb shelters converted to temples of desire. The "
                    "Velvet Sanctum reigns here, three levels below a vintage clothing boutique "
                    "on Valencia Street. The district's Latino heritage mixes with gothic subculture, "
                    "creating a unique aesthetic of Día de los Muertos meets dark romance."
                ),
                "notable_features": [
                    "The Velvet Sanctum - Lilith's domain",
                    "The Bone Garden - Goth club and safehouse entry",
                    "Dolores Park After Dark - Where deals are made",
                    "The Mission tunnels - Escape routes and hideaways"
                ],
                "areas": [
                    "Valencia Street (Surface) - Hip boutiques hiding secrets",
                    "The Catacombs - Underground club network",
                    "16th Street Station - Transit hub and trafficking danger zone",
                    "The Old Mission - Sanctuary that doesn't ask questions"
                ],
                "demographics": "Artists, rebels, night workers, lost youth",
                "danger_level": "High after 2 AM"
            },
            {
                "name": "SoMa Shadow District",
                "type": "industrial_nightlife",
                "description": (
                    "South of Market transforms after dark. Warehouses that house startups by day "
                    "become dungeons and underground fight clubs by night. The leather culture that "
                    "once defined Folsom Street evolved into something darker. Here, power is "
                    "exchanged like cryptocurrency, and consent is the only law that matters."
                ),
                "notable_features": [
                    "The Power Exchange - Infamous multi-level dungeon",
                    "Folsom Street Fair grounds - Annual celebration of kink",
                    "Abandoned Twitter building - Now a haven for the displaced",
                    "The Loading Dock - Where 'shipments' arrive"
                ],
                "demographics": "Tech workers' dark sides, leather community, sex workers"
            },
            {
                "name": "Tenderloin Shadows",
                "type": "red_light_district",
                "description": (
                    "The city's most honest district - it doesn't hide what it is. Here, "
                    "desperation and survival create their own economy. The Moth Queen's "
                    "safehouse network has three hidden entrances here, disguised as "
                    "massage parlors and SRO hotels. Every corner has eyes, every alley "
                    "holds a story of someone who didn't make it out."
                ),
                "notable_features": [
                    "The Phoenix Hotel - Safehouse disguised as SRO",
                    "Aunt Charlie's - Historic gay bar and info hub",
                    "The Screening Room - Where girls are 'evaluated'",
                    "St. Anthony's - Soup kitchen that shelters more than the hungry"
                ],
                "demographics": "The desperate, the hunted, the hunters"
            },
            {
                "name": "Financial District After Hours",
                "type": "hidden_playground",
                "description": (
                    "When the markets close, another economy opens. Private clubs in penthouses, "
                    "CEOs who kneel in Armani suits, venture capitalists funding their darkest "
                    "fantasies. The Moth Queen has clients here who pay in cryptocurrency and "
                    "guilt. Some fund her safehouses to ease their consciences."
                ),
                "notable_features": [
                    "The Apex Club - 40th floor, members only",
                    "Montgomery Station - Where worlds collide",
                    "The Pyramid's Shadow - Urban legend says there's a dungeon beneath"
                ],
                "demographics": "The powerful seeking to submit"
            },
            {
                "name": "The Marina Safehouse",
                "type": "rescue_network",
                "description": (
                    "Behind the wealthy facade of the Marina District, the Moth Queen "
                    "maintains her most secure safehouse. A mansion overlooking the bay, "
                    "donated by a tech heiress who survived trafficking. New identities "
                    "are crafted here, escape routes to new lives begin at this dock."
                ),
                "notable_features": [
                    "The Butterfly House - Main safehouse",
                    "Marina Green - Dead drop location",
                    "The Yacht Club - Unexpected ally",
                    "Palace of Fine Arts - Meeting spot"
                ],
                "demographics": "The saved, the saviors, the reborn"
            }
        ]
    
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
    def get_factions() -> List[Dict[str, Any]]:
        """Get the power factions in the story"""
        return [
            {
                "name": "The Velvet Court",
                "type": "underground_authority",
                "description": (
                    "The alliance of underground venue owners who maintain order in SF's shadow economy. "
                    "Led by the Moth Queen and six other venue owners. They ensure consent, safety, and "
                    "silence. Cross them, and you'll never work in the city's darkness again. They have "
                    "judges who visit, cops who look away, and tech executives who fund them out of guilt."
                ),
                "territory": "Mission Underground, SoMa Shadow District",
                "leadership": {
                    "The Moth Queen": "Velvet Sanctum",
                    "Master Chen": "The Jade Chamber",
                    "Mistress Raven": "Corvus House",
                    "The Twins": "Gemini Gardens",
                    "Iron Hans": "The Foundry",
                    "Lady Midnight": "The Twilight Room",
                    "The Architect": "The Blueprint"
                },
                "resources": [
                    "Underground venues",
                    "Information network", 
                    "Safehouse system",
                    "Blackmail archives",
                    "Private security force"
                ],
                "goals": [
                    "Protect consensual adult activities",
                    "Eliminate trafficking",
                    "Maintain the underground's independence"
                ],
                "enemies": ["Trafficking rings", "Moral crusaders", "Corrupt cops"]
            },
            {
                "name": "The Pacific Ring",
                "type": "trafficking_syndicate",
                "description": (
                    "International trafficking organization using SF's ports. Run by Viktor Kozlov and "
                    "his lieutenants. They see the city as a gateway to profit, people as products. "
                    "Connected to corrupt port officials and some tech companies needing 'clean' staff. "
                    "At war with the Moth Queen since she cost them millions."
                ),
                "territory": "Port of SF, Parts of Tenderloin, Hidden warehouses",
                "leadership": {
                    "Viktor 'The Collector' Kozlov": "Russian connection",
                    "Jimmy Tran": "Vietnamese pipeline", 
                    "Maria Santos": "Latin American routes",
                    "The Broker": "Tech industry connections"
                },
                "resources": [
                    "Shipping routes",
                    "Corrupt officials",
                    "International connections",
                    "Venture capital fronts"
                ],
                "enemies": ["The Moth Queen", "The Velvet Court", "FBI Task Force"]
            },
            {
                "name": "The Fog Walkers",
                "type": "rescue_network",
                "description": (
                    "The Moth Queen's extended network of protectors and allies. Former victims who've "
                    "become saviors, sympathetic officials, underground railroad operators. They use "
                    "the city's fog as cover and metaphor. Identified by moth tattoos in hidden places."
                ),
                "territory": "City-wide, strongest in Mission and Tenderloin",
                "membership": [
                    "Trafficking survivors",
                    "Safehouse operators",
                    "Sympathetic medical staff",
                    "Document forgers",
                    "Underground railroad conductors"
                ],
                "resources": [
                    "Safehouses",
                    "Escape routes",
                    "New identity creators",
                    "Emergency funds",
                    "Information network"
                ],
                "symbols": ["Moth tattoos", "Fog metaphors", "Light signals"]
            },
            {
                "name": "The Silicon Shadows",
                "type": "tech_elite_cabal",
                "description": (
                    "Tech executives and VCs who partake in the underground's offerings. Some are "
                    "genuine enthusiasts of BDSM culture, others are predators with money. The Moth "
                    "Queen cultivates the former and blackmails the latter. Their guilt-money funds "
                    "many safehouses, though they don't know it."
                ),
                "territory": "Financial District, Pacific Heights, Palo Alto",
                "notable_members": [
                    "Anonymous CEO who donated the Marina safehouse",
                    "The venture capitalist who funds new identities",
                    "The security executive who provides intel",
                    "The judge who ensures certain cases disappear"
                ],
                "resources": [
                    "Vast wealth",
                    "Political influence",
                    "Technology access",
                    "Legal protection"
                ]
            },
            {
                "name": "SFPD Vice Division - The Pragmatists",
                "type": "law_enforcement",
                "description": (
                    "The cops who've learned to work with the underground's rules. They know the "
                    "Moth Queen does their job better than they can. An unspoken agreement: she "
                    "keeps the real criminals out, they don't dig too deep into consensual activities."
                ),
                "leadership": "Captain Maria Rodriguez - 20-year veteran who's 'seen enough'",
                "factions": {
                    "The Blind": "Look the other way entirely",
                    "The Watchers": "Monitor but don't interfere",
                    "The Crusaders": "Still trying to 'clean up' the city"
                }
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
    def get_conflicts() -> List[Dict[str, Any]]:
        """Get ongoing conflicts in the story world"""
        return [
            {
                "name": "The Shadow War",
                "type": "underground_conflict",
                "description": (
                    "The ongoing battle between the Moth Queen's network and the Pacific Ring. "
                    "Fought in darkness with rescues, raids, and retributions. Bodies are rarely "
                    "found, but moth wings appear on trafficking sites before they burn."
                ),
                "factions": ["The Velvet Court + Fog Walkers", "The Pacific Ring"],
                "stakes": "Control of human trafficking in the Bay Area",
                "current_state": "Active warfare, Queen winning but at great cost"
            },
            {
                "name": "The Tech Reckoning",
                "type": "blackmail_war",
                "description": (
                    "The Moth Queen's systematic exposure of tech predators. Using their own "
                    "surveillance technology against them. Three CEOs have fled the country, "
                    "two have 'committed suicide,' five fund safehouses to avoid exposure."
                ),
                "factions": ["The Velvet Court", "Silicon Shadows predator faction"],
                "stakes": "Cleanup of tech industry abuse",
                "current_state": "Queen holds devastating evidence"
            },
            {
                "name": "The Sanctuary Debate",
                "type": "political_conflict",
                "description": (
                    "City government debates legalizing the underground venues. The Velvet Court "
                    "opposes it - legitimacy means regulation, visibility, vulnerability. Better "
                    "to rule in shadow than serve in light."
                ),
                "factions": ["Progressive politicians", "The Velvet Court", "Conservative groups"],
                "stakes": "Future of the underground's independence",
                "current_state": "Heated debate, Queen lobbying against"
            }
        ]
    
    @staticmethod
    def get_notable_figures() -> List[Dict[str, Any]]:
        """Get important NPCs beyond the main cast"""
        return [
            {
                "name": "Captain Maria Rodriguez",
                "role": "SFPD Vice - The Pragmatist",
                "description": (
                    "20-year veteran who learned the hard way that some wars can't be won with "
                    "badges and guns. Has an understanding with the Moth Queen - she handles the "
                    "real monsters, Maria handles the paperwork. Lost her partner to traffickers."
                ),
                "relationship_to_queen": "Uneasy alliance",
                "influence": "Keeps cops away from Queen's operations"
            },
            {
                "name": "Chen Wei",
                "role": "Master of the Jade Chamber",
                "description": (
                    "Lilith's closest ally in the Velvet Court. Runs high-end BDSM experiences "
                    "for the Pacific Heights crowd. Former Triad, went legitimate. His venue "
                    "connects to three safehouse tunnels."
                ),
                "relationship_to_queen": "Trusted ally, occasional lover",
                "influence": "Controls Chinatown underground"
            },
            {
                "name": "Dr. Sarah Martinez",
                "role": "Underground Medic",
                "description": (
                    "ER doctor at SF General who moonlights patching up trafficking victims. "
                    "No questions asked, no records kept. The Queen saved her sister in 2019. "
                    "Now she saves others."
                ),
                "relationship_to_queen": "Owes life debt",
                "influence": "Medical care for victims"
            },
            {
                "name": "The Architect",
                "role": "Mystery Power Broker",
                "description": (
                    "Unknown member of the Velvet Court who designs escape routes and safehouses. "
                    "Some say it's a tech billionaire, others a city planner. Only the Queen knows "
                    "their true identity."
                ),
                "relationship_to_queen": "Inner circle",
                "influence": "Infrastructure of salvation"
            },
            {
                "name": "Sister Catherine Ng",
                "role": "Progressive Nun",
                "description": (
                    "Runs a shelter in the Tenderloin that doesn't check IDs or immigration status. "
                    "Provides cover stories for rescued victims. Thinks the Queen does God's work, "
                    "even if the methods are unorthodox."
                ),
                "relationship_to_queen": "Spiritual advisor of sorts",
                "influence": "Religious sanctuary network"
            }
        ]
    
    @staticmethod
    def get_educational_systems() -> List[Dict[str, Any]]:
        """Get educational institutions for SF Bay Area"""
        return [
            {
                "name": "The Conservatory of Shadows",
                "system_type": "alternative_education",
                "description": (
                    "Hidden beneath the California Academy of Sciences, this underground school "
                    "teaches the arts of dominance, submission, and power exchange. Run by former "
                    "Velvet Court members, it maintains the old traditions while preparing the next "
                    "generation. The Moth Queen occasionally guest lectures on consent and protection."
                ),
                "target_demographics": ["Adult practitioners", "Underground community", "Those seeking mastery"],
                "controlled_by": "The Velvet Court Educational Trust",
                "core_teachings": [
                    "Consent as sacred law",
                    "Power exchange dynamics",
                    "Safety and risk awareness",
                    "Underground history and traditions",
                    "Protection of the vulnerable"
                ],
                "teaching_methods": ["Mentorship", "Practical workshops", "Ritualized lessons"],
                "coming_of_age_rituals": "The First Mask Ceremony - earning your place in the underground",
                "knowledge_restrictions": "Outsiders forbidden, law enforcement especially",
                "female_leadership_roles": ["Headmistress", "Senior Dominatrixes", "Safety Wardens"],
                "male_roles": ["Assistant instructors", "Demonstration subjects", "Security"],
                "gender_specific_teachings": {
                    "female": ["Dominance arts", "Protective leadership", "Underground governance"],
                    "male": ["Service protocols", "Protective submission", "Support roles"]
                },
                "taboo_subjects": ["Trafficking methods", "Breaking consent", "Exposing the network"],
                "censorship_level": 8,
                "censorship_enforcement": "Immediate expulsion and blacklisting"
            },
            {
                "name": "St. Dominic's Reform School",
                "system_type": "religious_education",
                "description": (
                    "Catholic school in the Sunset District with a dark reputation. Officially for "
                    "'troubled' youth, it's known for producing either broken spirits or fierce rebels. "
                    "Several of the Moth Queen's rescued victims attended here. The rigid discipline "
                    "and shame-based teaching create perfect future submissives - or revolutionaries."
                ),
                "target_demographics": ["Troubled teens", "Court-mandated youth", "Parents' last resort"],
                "controlled_by": "Archdiocese of San Francisco",
                "core_teachings": [
                    "Strict Catholic doctrine",
                    "Corporal discipline",
                    "Shame and redemption",
                    "Gender role enforcement"
                ],
                "teaching_methods": ["Rote learning", "Physical punishment", "Isolation", "Prayer"],
                "coming_of_age_rituals": "Confirmation under duress",
                "knowledge_restrictions": "No questioning doctrine, no outside media",
                "connections_to_story": "Source of trauma for many underground members"
            },
            {
                "name": "Berkeley Underground Studies",
                "system_type": "informal_education",
                "description": (
                    "Anarchist collective near UC Berkeley teaching survival skills, legal rights, "
                    "and resistance tactics. Secretly funded by the Moth Queen, it helps at-risk "
                    "youth avoid trafficking. Classes held in rotating locations, communicated "
                    "through encrypted channels."
                ),
                "target_demographics": ["Runaways", "At-risk youth", "Sex workers", "Activists"],
                "controlled_by": "The Collective (anarchist structure)",
                "core_teachings": [
                    "Know your rights",
                    "Self-defense (physical and legal)",
                    "Recognizing predators",
                    "Safe communication",
                    "Mutual aid networks"
                ],
                "teaching_methods": ["Peer education", "Street workshops", "Practical exercises"],
                "connections_to_underground": "Direct pipeline to safehouse network"
            }
        ]

    @staticmethod
    def get_religious_institutions() -> List[Dict[str, Any]]:
        """Get religious organizations in SF Bay Area"""
        return [
            {
                "name": "Church of Sacred Flesh",
                "type": "alternative_spirituality",
                "founded": "2015",
                "description": (
                    "Neo-pagan church in the Mission that sanctifies BDSM as spiritual practice. "
                    "Founded by ex-Catholic dommes who blend ritual and rope. They provide "
                    "spiritual counseling for sex workers and host 'confession circles' where "
                    "shame transforms to power. The Moth Queen is rumored to be a silent patron."
                ),
                "leadership": "High Priestess Magdalena - former nun turned dominatrix",
                "practices": [
                    "Rope bondage as meditation",
                    "Pain as transcendence",
                    "Confession without judgment",
                    "Sacred sexuality rituals"
                ],
                "location": "Converted church on 20th Street",
                "secrets": "Underground railroad meeting point, hidden basement temple",
                "connections": ["Velvet Court", "Sex worker unions", "Pagan community"]
            },
            {
                "name": "Grace Cathedral",
                "type": "episcopal_church",
                "founded": "1849",
                "description": (
                    "Progressive Episcopal cathedral on Nob Hill. The labyrinth provides sanctuary "
                    "to anyone, no questions asked. Bishop Williams knows about the underground "
                    "but maintains plausible deniability. Late night 'prayer services' sometimes "
                    "shelter trafficking victims."
                ),
                "leadership": "Bishop Sarah Williams - pragmatic progressive",
                "practices": [
                    "Labyrinth sanctuary walks",
                    "No-questions-asked shelter",
                    "Progressive theology",
                    "LGBTQ+ affirmation"
                ],
                "location": "Nob Hill",
                "secrets": "Emergency beacon for safehouse network in bell tower",
                "connections": ["City government", "Progressive coalitions", "Underground (unofficial)"]
            },
            {
                "name": "Temple of Inanna",
                "type": "reconstructionist_pagan",
                "founded": "2008",
                "description": (
                    "Sacred sexuality temple in Oakland honoring the Mesopotamian goddess of "
                    "love and war. Provides ritual space for ethical power exchange and trains "
                    "sacred intimates. Some graduates work in the underground, using their "
                    "skills to heal trauma through conscious kink."
                ),
                "leadership": "Temple Keeper Ashara - trained in tantra and therapy",
                "practices": [
                    "Sacred sexuality training",
                    "Trauma-informed intimacy",
                    "Ritual power exchange",
                    "Monthly public rituals"
                ],
                "location": "Oakland warehouse district",
                "connections": ["Alternative healing community", "Ethical kink educators"]
            }
        ]

    @staticmethod
    def get_knowledge_traditions() -> List[Dict[str, Any]]:
        """Get knowledge transmission traditions"""
        return [
            {
                "name": "The Velvet Protocols",
                "tradition_type": "oral_tradition",
                "description": (
                    "The underground's unwritten rules passed down through mentorship. "
                    "Covers everything from scene negotiation to recognizing trafficking "
                    "signs. Each venue has a Protocol Keeper who trains newcomers. "
                    "Breaking protocol means exile from all underground spaces."
                ),
                "knowledge_domain": "Underground safety and ethics",
                "preservation_method": "Mentorship chains and practical demonstration",
                "access_requirements": "Vouching by established member, proven trustworthiness",
                "associated_group": "The Velvet Court",
                "examples": [
                    "The Three Taps (emergency signal)",
                    "Moth Sign (protection needed)",
                    "Red Night Rules (when cops raid)"
                ],
                "female_gatekeepers": True,
                "gendered_access": {
                    "female": "Full protocol training including leadership",
                    "male": "Service protocols and protection duties"
                },
                "matriarchal_reinforcement": "Female dominants as protocol authorities"
            },
            {
                "name": "Trauma Mapping",
                "tradition_type": "healing_practice",
                "description": (
                    "Body-based healing tradition developed by trafficking survivors. "
                    "Uses touch, movement, and power exchange to reclaim agency. The "
                    "Moth Queen learned this from the previous Queen. Practitioners "
                    "can 'read' trauma in body language and help release it."
                ),
                "knowledge_domain": "Somatic healing through kink",
                "preservation_method": "Direct body-to-body teaching",
                "access_requirements": "Must be survivor or proven ally",
                "associated_group": "The Fog Walkers",
                "female_gatekeepers": True,
                "matriarchal_reinforcement": "Female survivors as wisdom keepers"
            },
            {
                "name": "The Mask Makers' Art",
                "tradition_type": "craft_tradition",
                "description": (
                    "Creating personas and masks - both physical and psychological - "
                    "for protection and power. Master mask makers teach selected "
                    "students how to craft identities that can withstand scrutiny. "
                    "Used for both underground personas and new identities for rescued."
                ),
                "knowledge_domain": "Identity crafting and protection",
                "preservation_method": "Apprenticeship system",
                "access_requirements": "Artistic skill and absolute discretion",
                "examples": [
                    "Legal identity creation",
                    "Persona development",
                    "Physical mask crafting",
                    "Digital identity management"
                ]
            }
        ]

    @staticmethod
    def get_quest_hooks() -> List[Dict[str, Any]]:
        """Get story quests and missions"""
        return [
            {
                "quest_name": "The Missing Moths",
                "quest_giver": "Sarah Chen at the Butterfly House",
                "location": "Marina Safehouse",
                "description": (
                    "Three rescued victims have vanished from the safehouse network. "
                    "Were they recaptured, or is there a mole in the organization? "
                    "The Moth Queen needs someone she trusts to investigate quietly."
                ),
                "objectives": [
                    "Interview safehouse staff",
                    "Check underground contacts",
                    "Follow the money trail",
                    "Identify the leak"
                ],
                "rewards": ["Deeper trust with Moth Queen", "Access to restricted areas"],
                "difficulty": 7,
                "lore_significance": "Reveals corruption within the protection network"
            },
            {
                "quest_name": "The Gilded Trap",
                "quest_giver": "Anonymous tip at Velvet Sanctum",
                "location": "Financial District",
                "description": (
                    "A tech CEO is hosting a 'private party' that's actually an auction "
                    "for trafficked victims. The Moth Queen needs evidence and a way "
                    "to save the victims without revealing her identity."
                ),
                "objectives": [
                    "Infiltrate the CEO's circle",
                    "Gather video evidence",
                    "Identify the victims",
                    "Coordinate the rescue"
                ],
                "rewards": ["Major blow to trafficking ring", "Blackmail material"],
                "difficulty": 9,
                "lore_significance": "Exposes Silicon Valley's dark connections"
            },
            {
                "quest_name": "The Confession Booth Conspiracy",
                "quest_giver": "Mistress Raven",
                "location": "Multiple underground venues",
                "description": (
                    "Someone is using the confession booths to gather blackmail on "
                    "vulnerable people. The sanctity of confession has been violated, "
                    "and the underground's trust system is at risk."
                ),
                "objectives": [
                    "Investigate compromised booths",
                    "Trace the surveillance equipment",
                    "Identify the blackmailer",
                    "Restore trust in the system"
                ],
                "rewards": ["Velvet Court gratitude", "Enhanced reputation"],
                "difficulty": 6,
                "lore_significance": "Threatens the underground's core traditions"
            }
        ]

    @staticmethod
    def get_domestic_issues() -> List[Dict[str, Any]]:
        """Get local political conflicts"""
        return [
            {
                "name": "The Sanctuary City Debate",
                "issue_type": "political",
                "description": (
                    "Mayor Chen pushes to legalize and regulate underground venues, "
                    "claiming it would improve safety. The Velvet Court opposes this - "
                    "legitimacy means visibility, regulation means vulnerability. The "
                    "community is split between safety and autonomy."
                ),
                "severity": 8,
                "status": "escalating",
                "supporting_factions": ["Progressive politicians", "Some sex workers", "Health advocates"],
                "opposing_factions": ["The Velvet Court", "Privacy advocates", "Old guard underground"],
                "neutral_factions": ["SFPD (officially)", "Business community"],
                "public_opinion": {
                    "general_public": "45% support, 30% oppose, 25% unaware",
                    "underground_community": "20% support, 70% oppose, 10% undecided"
                },
                "government_response": "Committee hearings scheduled",
                "potential_resolution": "Compromise on health/safety regulations only"
            },
            {
                "name": "The Port Authority Corruption Scandal",
                "issue_type": "corruption",
                "description": (
                    "Leaked documents suggest Port Commissioner Huang's reforms are "
                    "a cover - she's redirecting trafficking through 'clean' channels. "
                    "The Moth Queen has evidence but revealing it exposes her network."
                ),
                "severity": 9,
                "status": "active",
                "stakes": "Control of trafficking routes through the port",
                "dilemma": "Expose corruption vs protect underground network"
            },
            {
                "name": "The Mission Gentrification War",
                "issue_type": "social",
                "description": (
                    "Tech money pushes into the Mission, threatening underground venues "
                    "with rising rents. The Velvet Sanctum's lease is up for renewal. "
                    "Developer Magnus Thornwood wants the building - and knows what's beneath."
                ),
                "severity": 7,
                "status": "active",
                "supporting_factions": ["Tech companies", "Developers", "New residents"],
                "opposing_factions": ["Underground venues", "Latino community", "Artists"],
                "potential_resolution": "The Moth Queen has dirt on Thornwood"
            }
        ]

    @staticmethod
    def get_pantheons() -> List[Dict[str, Any]]:
        """Get belief systems that function as 'religions' in modern setting"""
        return [
            {
                "name": "The Doctrine of Consensual Exchange",
                "description": (
                    "The underground's sacred philosophy that elevates BDSM beyond mere kink "
                    "into spiritual practice. Power exchange becomes prayer, pain becomes "
                    "transcendence, and consent becomes the highest sacrament."
                ),
                "origin_story": (
                    "Born from the leather culture of 1970s SoMa, evolved through feminist "
                    "sex-positive movements and the need to sanctify what society condemns."
                ),
                "creation_myth": (
                    "In the beginning was the Word, and the Word was 'Yes.' From this first "
                    "consent, all power dynamics flowed. The Goddess of Pain blessed those who "
                    "could transform suffering into ecstasy."
                ),
                "afterlife_beliefs": (
                    "Subspace is a glimpse of the eternal. Those who master the exchange of "
                    "power transcend flesh and touch divinity. The ultimate submission is ego death."
                ),
                "cosmic_structure": (
                    "The universe is built on tension - dominance and submission, pain and "
                    "pleasure, control and surrender. Balance comes not from equality but from "
                    "conscious exchange."
                ),
                "matriarchal_elements": (
                    "Female dominants are High Priestesses. The feminine holds ultimate power "
                    "to give or deny consent, to nurture or destroy. Male submission is holy."
                ),
                "major_holy_days": [
                    "Folsom Street Fair - The High Holy Day",
                    "International Women's Day - Celebration of Female Power",
                    "Leather Pride Week - Sacred Remembrance"
                ],
                "taboos": [
                    "Breaking consent",
                    "Outing someone without permission",
                    "Vanilla-shaming",
                    "Touching someone's collar without permission"
                ],
                "geographical_spread": ["SoMa", "Mission", "Castro"],
                "dominant_nations": ["Underground SF"],
                "primary_worshippers": ["BDSM practitioners", "Power exchange couples", "The searching"]
            },
            {
                "name": "The Church of the Wounded Healer",
                "description": (
                    "A survivor-founded spiritual movement that sees trauma as a path to "
                    "transcendence. Scars are scripture, survival is sainthood, and helping "
                    "others escape is the highest calling."
                ),
                "origin_story": (
                    "Founded by trafficking survivors who found meaning in their pain. They "
                    "believe their suffering gave them power to save others. The Moth Queen "
                    "is considered a living saint, though she rejects the title."
                ),
                "creation_myth": (
                    "Every soul chooses its trials before birth. The wounded are the strongest "
                    "because they chose the hardest paths. Moths are drawn to flames because "
                    "they remember being stars."
                ),
                "afterlife_beliefs": (
                    "Those who save others are reborn as protectors. Those who prey on the "
                    "innocent become the hunted in the next life. Karma is actively enforced."
                ),
                "cosmic_structure": (
                    "The universe is a healing journey. Every trauma is a lesson, every scar "
                    "a medal. The broken places let the light in."
                ),
                "matriarchal_elements": (
                    "Women who survive become mothers to all. Female rage is holy justice. "
                    "The Divine Feminine protects her children with terrible fury."
                ),
                "major_holy_days": [
                    "Night of Escaped Moths - Anniversary of major rescue",
                    "Day of Empty Cages - Celebrating freedom",
                    "Feast of Broken Chains - Honoring liberators"
                ],
                "taboos": [
                    "Victim blaming",
                    "Revealing safehouse locations",
                    "Abandoning someone in need",
                    "Profiting from others' pain"
                ],
                "geographical_spread": ["Tenderloin", "Mission", "Scattered safehouses"],
                "dominant_nations": ["Survivor networks"],
                "primary_worshippers": ["Survivors", "Protectors", "Social workers", "The saved"]
            }
        ]
    
    @staticmethod
    def get_deities() -> List[Dict[str, Any]]:
        """Get deity-like figures in the modern pantheon"""
        return [
            {
                "name": "The Moth Queen",
                "gender": "female",
                "domain": ["Protection", "Transformation", "Dark Salvation", "Power Exchange"],
                "description": (
                    "The living goddess of the underground. By day a myth, by night a reality. "
                    "She rules through devotion, saves through destruction, loves through control. "
                    "They say she has died and been reborn three times."
                ),
                "iconography": "Moth wings, porcelain masks, red velvet, thorns and roses",
                "holy_symbol": "A moth with human eyes",
                "sacred_animals": ["Moths", "Black cats", "Ravens"],
                "sacred_colors": ["Crimson", "Black", "Silver"],
                "relationships": [
                    {"deity_name": "The First Queen", "relationship_type": "predecessor"},
                    {"deity_name": "The Collector", "relationship_type": "nemesis"}
                ],
                "rank": 10,
                "worshippers": ["Submissives", "Survivors", "Protectors", "The lost"],
                "pantheon_id": 1  # Doctrine of Consensual Exchange
            },
            {
                "name": "The Collector",
                "gender": "male",
                "domain": ["Predation", "Greed", "Corruption", "False Power"],
                "description": (
                    "The anti-deity, the devourer. Viktor Kozlov elevated to mythic status by "
                    "those who fear him. He represents all who see people as products. The "
                    "darkness the Moth Queen fights."
                ),
                "iconography": "Chains, cages, hundred dollar bills, empty eyes",
                "holy_symbol": "A cage with no door",
                "sacred_animals": ["Vultures", "Hyenas", "Parasites"],
                "sacred_colors": ["Gold", "Rust", "Bruise purple"],
                "relationships": [
                    {"deity_name": "The Moth Queen", "relationship_type": "eternal enemy"},
                    {"deity_name": "The Broken Judge", "relationship_type": "corrupted"}
                ],
                "rank": 8,
                "worshippers": ["Traffickers", "Corrupt officials", "The greedy"],
                "pantheon_id": None  # Represents anti-faith
            },
            {
                "name": "The First Queen",
                "gender": "female",
                "domain": ["Origin", "Sacrifice", "Foundation", "Memory"],
                "description": (
                    "Liberty Chen, the original protector. She who first turned pain into power, "
                    "victimhood into victory. Her disappearance in 1978 only made her stronger. "
                    "Some say she still guides from shadows."
                ),
                "iconography": "Broken chains, phoenix, old photographs, hidden doors",
                "holy_symbol": "A key that opens any lock",
                "sacred_animals": ["Phoenix", "Butterflies emerging from cocoons"],
                "sacred_colors": ["Sepia", "Gold", "Dawn colors"],
                "relationships": [
                    {"deity_name": "The Moth Queen", "relationship_type": "spiritual heir"},
                    {"deity_name": "Sister Mary Catherine", "relationship_type": "ally"}
                ],
                "rank": 9,
                "worshippers": ["The old guard", "Historians of the underground", "Seekers"],
                "pantheon_id": 2  # Church of the Wounded Healer
            }
        ]
    
    @staticmethod
    def get_religious_practices() -> List[Dict[str, Any]]:
        """Get ritual practices of the underground faiths"""
        return [
            {
                "name": "The Collar Ceremony",
                "practice_type": "initiation",
                "description": (
                    "The sacred ritual of accepting ownership. More binding than marriage, "
                    "deeper than blood oaths. The collar is blessed with tears and sealed "
                    "with a kiss. Breaking this bond is the ultimate betrayal."
                ),
                "purpose": "Establishing permanent D/s relationship",
                "frequency": "Once per lifetime per dominant",
                "required_elements": ["Collar", "Witnesses", "Vows", "Kneeling"],
                "performed_by": ["Experienced dominants", "High Priestesses"],
                "restricted_to": ["Those who've proven worthy"],
                "deity_id": 1,  # Moth Queen
                "pantheon_id": 1  # Doctrine of Consensual Exchange
            },
            {
                "name": "Confession in Darkness",
                "practice_type": "cathartic",
                "description": (
                    "Monthly ritual where people confess desires and sins to masked strangers. "
                    "What's spoken in darkness stays there. The Queen herself sometimes listens, "
                    "choosing who needs salvation and who needs punishment."
                ),
                "purpose": "Release and absolution",
                "frequency": "First Friday monthly",
                "required_elements": ["Darkness", "Masks", "Whispered truths"],
                "performed_by": ["Anyone", "Confessors are volunteers"],
                "restricted_to": ["Adults only"],
                "deity_id": 1,
                "pantheon_id": 1
            },
            {
                "name": "The Moth Release",
                "practice_type": "memorial",
                "description": (
                    "Annual ceremony where survivors release live moths for those who didn't "
                    "make it. Each moth carries a name, a prayer, a promise to remember. The "
                    "Queen always releases the first moth in silence."
                ),
                "purpose": "Honoring the lost",
                "frequency": "Annual - April 18th",
                "required_elements": ["Live moths", "Names of the lost", "Candles"],
                "performed_by": ["Survivors", "Families", "Protectors"],
                "restricted_to": ["None - all welcome"],
                "deity_id": 3,  # The First Queen
                "pantheon_id": 2  # Church of the Wounded Healer
            },
            {
                "name": "Subspace Meditation",
                "practice_type": "transcendent",
                "description": (
                    "Using controlled pain and breathwork to achieve altered consciousness. "
                    "Practitioners report visions, out-of-body experiences, and profound "
                    "spiritual insights. The deeper the submission, the higher the flight."
                ),
                "purpose": "Spiritual transcendence through BDSM",
                "frequency": "As needed",
                "required_elements": ["Experienced guide", "Safety protocols", "Aftercare"],
                "performed_by": ["Trained dominants", "Spiritual guides"],
                "restricted_to": ["Experienced practitioners"],
                "deity_id": 1,
                "pantheon_id": 1
            }
        ]
    
    @staticmethod
    def get_holy_sites() -> List[Dict[str, Any]]:
        """Get sacred locations beyond the main venues"""
        return [
            {
                "name": "The Sutro Bath Ruins",
                "site_type": "ceremonial_ground",
                "description": (
                    "The ruins by the sea where major underground decisions are made. The "
                    "crashing waves mask conversations, the ruins remind all of impermanence. "
                    "The Moth Queen was crowned here in a ceremony involving all factions."
                ),
                "clergy_type": "None - neutral ground",
                "location_description": "Lands End, where city meets ocean",
                "deity_id": None,  # Sacred to all
                "pantheon_id": None,  # Neutral
                "clergy_hierarchy": ["Gathering of equals"],
                "pilgrimage_info": "Leaders come here to settle disputes",
                "miracles_reported": [
                    "Enemies making peace",
                    "The fog parting at crucial moments",
                    "Visions in the pools"
                ],
                "restrictions": ["No violence", "No recordings", "Truth only"],
                "architectural_features": "Natural amphitheater, tidal pools, concrete ruins"
            },
            {
                "name": "The First Safehouse",
                "site_type": "shrine",
                "description": (
                    "The original safehouse Liberty Chen established, now a secret shrine. "
                    "Hidden in Chinatown, still operational. Those who know can light "
                    "incense for protection. The walls hold forty years of grateful prayers."
                ),
                "clergy_type": "Keepers of Memory",
                "location_description": "Chinatown basement, unmarked door",
                "deity_id": 3,  # The First Queen
                "pantheon_id": 2,  # Church of the Wounded Healer
                "clergy_hierarchy": ["Elder Keeper", "Memory Holders", "Door Watchers"],
                "pilgrimage_info": "Survivors come to give thanks",
                "miracles_reported": [
                    "Healing touch",
                    "Prophetic dreams after visiting",
                    "Protection from pursuers"
                ],
                "restrictions": ["Must be brought by one who knows", "No photographs"],
                "architectural_features": "Hidden altar, wall of photographs, escape tunnel"
            },
            {
                "name": "The Folsom Sanctuary",
                "site_type": "temple",
                "description": (
                    "A former church converted to BDSM temple. Gothic architecture meets "
                    "dungeon aesthetic. Stained glass windows depict scenes of consensual "
                    "power exchange. The altar is now a stage for sacred scenes."
                ),
                "clergy_type": "Leather Priests/Priestesses",
                "location_description": "Folsom Street, looks abandoned from outside",
                "deity_id": 1,  # The Moth Queen
                "pantheon_id": 1,  # Doctrine of Consensual Exchange
                "clergy_hierarchy": [
                    "High Priestess of Pain",
                    "Masters and Mistresses",
                    "Acolytes in Training"
                ],
                "pilgrimage_info": "Serious practitioners come for blessing",
                "miracles_reported": [
                    "Instant subspace",
                    "Healing through scene work",
                    "Finding one's true nature"
                ],
                "restrictions": ["Strict protocols", "Membership required", "No tourists"],
                "architectural_features": "Dungeon equipment as religious fixtures"
            }
        ]
    
    @staticmethod
    def get_religious_texts() -> List[Dict[str, Any]]:
        """Get sacred writings and teachings"""
        return [
            {
                "name": "The Velvet Codex",
                "text_type": "scripture",
                "description": (
                    "The underground's bible, written collectively over decades. Part safety "
                    "manual, part philosophy, part poetry. Contains the sacred protocols, "
                    "consent frameworks, and wisdom of the community."
                ),
                "authorship": "Collective - each generation adds",
                "key_teachings": [
                    "Consent is continuous and revocable",
                    "Pain with purpose transcends suffering",
                    "The dominant serves the submissive's needs",
                    "Safe, sane, and consensual above all"
                ],
                "restricted_to": ["Initiated members"],
                "deity_id": 1,
                "pantheon_id": 1,
                "notable_passages": [
                    "Chapter of Knots: rope work as meditation",
                    "Book of Limits: establishing boundaries",
                    "Songs of Subspace: ecstatic poetry"
                ],
                "age_description": "Begun in 1978, still growing"
            },
            {
                "name": "Letters Never Sent",
                "text_type": "testimonial",
                "description": (
                    "A collection of letters written by the saved to those who saved them. "
                    "Most are addressed to the Moth Queen, though she claims not to read them. "
                    "Kept in the safehouse archives as proof that salvation is possible."
                ),
                "authorship": "Anonymous survivors",
                "key_teachings": [
                    "Gratitude for life restored",
                    "The power of being believed",
                    "How one person can change everything",
                    "Hope in darkness"
                ],
                "restricted_to": ["Protectors and survivors"],
                "deity_id": 3,
                "pantheon_id": 2,
                "notable_passages": [
                    "To the woman who carried me from hell",
                    "Three years free today",
                    "Teaching others what you taught me"
                ],
                "age_description": "Ongoing collection"
            },
            {
                "name": "The Moth Queen's Journals",
                "text_type": "apocrypha",
                "description": (
                    "Rumored to exist but never confirmed. Said to contain her true thoughts, "
                    "the names on her lists, and those three words she cannot speak. Some "
                    "claim to have seen pages, but none can prove it."
                ),
                "authorship": "The Moth Queen (allegedly)",
                "key_teachings": [
                    "Unknown - speculation only",
                    "The weight of saving others",
                    "The cost of power",
                    "What lies beneath masks"
                ],
                "restricted_to": ["No one - if they exist"],
                "deity_id": 1,
                "pantheon_id": None,
                "notable_passages": [
                    "They all promise not to disappear",
                    "Saved another tonight, lost myself a little more",
                    "(The three words, scratched out)"
                ],
                "age_description": "Current era"
            }
        ]
    
    @staticmethod
    def get_religious_orders() -> List[Dict[str, Any]]:
        """Get organized groups within the faiths"""
        return [
            {
                "name": "The Order of Broken Dolls",
                "order_type": "militant",
                "description": (
                    "Former victims who've become hunters. They track traffickers, gather "
                    "evidence, and sometimes take direct action. Named for how they were "
                    "seen (dolls) and what they became (broken but dangerous)."
                ),
                "gender_composition": "female_led",
                "founding_story": (
                    "Three survivors met in a safehouse and swore to ensure no one else "
                    "suffered as they had. They learned to fight, to track, to disappear. "
                    "Now they are legend."
                ),
                "headquarters": "Mobile - they don't stay still",
                "hierarchy_structure": [
                    "The Untouchable Three (founders)",
                    "Hunters (field operatives)",
                    "Watchers (intelligence)",
                    "Menders (safehouse staff)"
                ],
                "vows": [
                    "Never forget where we came from",
                    "Save who we can, avenge who we couldn't",
                    "Our pain becomes their justice"
                ],
                "practices": [
                    "Combat training",
                    "Surveillance",
                    "Extraction operations",
                    "Trauma counseling"
                ],
                "deity_id": 3,  # The First Queen
                "pantheon_id": 2,
                "special_abilities": [
                    "Underground railroad expertise",
                    "Combat training",
                    "Disguise and infiltration"
                ],
                "notable_members": ["Unknown - they stay anonymous"]
            },
            {
                "name": "The Sanctuary Keepers",
                "order_type": "protective",
                "description": (
                    "Those who maintain and protect the safehouses. They live normal lives "
                    "as cover - shopkeepers, teachers, nurses - but their true calling is "
                    "providing sanctuary. They are the infrastructure of salvation."
                ),
                "gender_composition": "mixed",
                "founding_story": (
                    "When the first safehouse proved insufficient, the Queen recruited "
                    "civilians willing to risk everything. Each keeper maintains a safe "
                    "space and asks no questions."
                ),
                "headquarters": "Distributed - each safehouse is autonomous",
                "hierarchy_structure": [
                    "The Moth Queen (spiritual leader)",
                    "Regional Coordinators",
                    "House Keepers",
                    "Runners (transport between houses)"
                ],
                "vows": [
                    "This house is sanctuary",
                    "No questions, only safety",
                    "We are links in an unbreakable chain"
                ],
                "practices": [
                    "Maintaining cover identities",
                    "First aid and trauma care",
                    "Document forgery",
                    "Quick extraction protocols"
                ],
                "deity_id": 1,
                "pantheon_id": 2,
                "special_abilities": [
                    "Safe house management",
                    "Identity creation",
                    "Medical care",
                    "Absolute discretion"
                ],
                "notable_members": ["Known only by first names"]
            },
            {
                "name": "The Crimson Guard",
                "order_type": "ceremonial",
                "description": (
                    "Elite dominants who've achieved mastery and now teach others. They "
                    "guard the sacred practices and ensure the old ways aren't lost to "
                    "commercialization. The Queen's inner circle."
                ),
                "gender_composition": "female_led",
                "founding_story": (
                    "As BDSM became mainstream, the Queen gathered the most skilled to "
                    "preserve the spiritual aspects. They are living libraries of technique "
                    "and protocol."
                ),
                "headquarters": "The Velvet Sanctum's hidden floor",
                "hierarchy_structure": [
                    "The Queen (Grand Mistress)",
                    "Cardinals of Pain (master teachers)",
                    "Crimson Knights (guardians)",
                    "Scarlet Novices (students)"
                ],
                "vows": [
                    "Preserve the sacred in the profane",
                    "Technique serves connection",
                    "We are servants to the power we wield"
                ],
                "practices": [
                    "Advanced technique training",
                    "Spiritual counseling",
                    "Protocol preservation",
                    "Ceremonial scenes"
                ],
                "deity_id": 1,
                "pantheon_id": 1,
                "special_abilities": [
                    "Master-level BDSM skills",
                    "Energy manipulation",
                    "Psychological insight",
                    "Ceremonial magic"
                ],
                "notable_members": [
                    "Master Chen - The Queen's right hand",
                    "Mistress Scarlet - Keeper of Protocols"
                ]
            }
        ]
    
    @staticmethod
    def get_religious_conflicts() -> List[Dict[str, Any]]:
        """Get theological and practical disputes"""
        return [
            {
                "name": "The Monetization Schism",
                "conflict_type": "theological",
                "description": (
                    "Dispute between those who see BDSM as sacred practice and those who've "
                    "commercialized it. OnlyFans dominatrixes vs old guard. The Queen stays "
                    "neutral publicly but enforces standards in her domain."
                ),
                "parties_involved": [
                    "Traditional Leather Community",
                    "Commercial Sex Workers",
                    "The Velvet Court (mediating)"
                ],
                "core_disagreement": "Can the sacred be sold?",
                "beginning_date": "2020 - Pandemic lockdowns",
                "resolution_date": None,
                "status": "ongoing",
                "casualties": "Community cohesion",
                "historical_impact": "Fractured the underground"
            },
            {
                "name": "The Consent Wars",
                "conflict_type": "doctrinal",
                "description": (
                    "Ongoing debate about edge play, consensual non-consent, and where "
                    "boundaries lie. Some push for absolute safety, others for absolute "
                    "freedom. Blood has been spilled over protocol violations."
                ),
                "parties_involved": [
                    "Risk-Aware Consensual Kink faction",
                    "Safe, Sane, Consensual traditionalists",
                    "Edge Players"
                ],
                "core_disagreement": "How much risk is acceptable?",
                "beginning_date": "2010",
                "resolution_date": None,
                "status": "uneasy truce",
                "casualties": "Several excommunications",
                "historical_impact": "Created parallel communities"
            },
            {
                "name": "The Succession Question",
                "conflict_type": "leadership",
                "description": (
                    "Who will inherit the Moth Queen's crown? She has no named heir, and "
                    "various factions position themselves. Some say she cannot die, others "
                    "prepare for the inevitable. The underground watches nervously."
                ),
                "parties_involved": [
                    "The Crimson Guard",
                    "The Broken Dolls",
                    "Various would-be successors"
                ],
                "core_disagreement": "Can the Queen be replaced?",
                "beginning_date": "2023 - After a close call",
                "resolution_date": None,
                "status": "simmering",
                "casualties": "None yet",
                "historical_impact": "Increasing tension"
            }
        ]
    
    @staticmethod
    def get_faction_relationships() -> List[Dict[str, Any]]:
        """Get detailed faction relationships beyond basic conflicts"""
        return [
            {
                "faction1": "The Velvet Court",
                "faction2": "SFPD Vice Division",
                "relationship_type": "uneasy_truce",
                "description": (
                    "They pretend we don't exist, we pretend to follow laws. Captain Rodriguez "
                    "and the Queen have an understanding - she keeps the real criminals out, "
                    "they don't raid consensual venues."
                ),
                "trade_agreements": ["Information for blindness"],
                "tension_points": ["New cops who don't know the rules"],
                "collaboration_areas": ["Anti-trafficking operations"]
            },
            {
                "faction1": "The Broken Dolls",
                "faction2": "The Pacific Ring",
                "relationship_type": "blood_war",
                "description": (
                    "No quarter given. The Dolls hunt Ring members, the Ring puts bounties "
                    "on Dolls. Bodies turn up in the Bay. This war has rules - no civilians, "
                    "no families - but otherwise, anything goes."
                ),
                "trade_agreements": ["None"],
                "tension_points": ["Everything"],
                "collaboration_areas": ["None - kill on sight"]
            },
            {
                "faction1": "Silicon Shadows",
                "faction2": "The Velvet Court",
                "relationship_type": "parasitic_symbiosis",
                "description": (
                    "Tech money funds the underground, the underground keeps tech's secrets. "
                    "The Queen has enough blackmail to crash the NASDAQ. They know it, she "
                    "knows it, everyone pretends it's voluntary patronage."
                ),
                "trade_agreements": [
                    "Money for silence",
                    "Protection for funding",
                    "Exclusive access for donations"
                ],
                "tension_points": ["New money not understanding protocol"],
                "collaboration_areas": ["Safehouse funding", "Identity creation tech"]
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
    async def initialize_complete_sf_preset(ctx, user_id: int, conversation_id: int):
        """Initialize ALL lore components for SF Bay preset"""
        logger.info("Initializing complete SF Bay Area preset for The Moth and Flame")
        
        from lore.core.lore_system import LoreSystem
        lore_system = await LoreSystem.get_instance(user_id, conversation_id)
        
        # Get all preset data
        preset_data = {
            "world": SFBayMothFlamePreset.get_world_foundation(),
            "districts": SFBayMothFlamePreset.get_districts(),
            "locations": SFBayMothFlamePreset.get_specific_locations(),
            "myths": SFBayMothFlamePreset.get_urban_myths(),
            "history": SFBayMothFlamePreset.get_historical_events(),
            "factions": SFBayMothFlamePreset.get_factions(),
            "culture": SFBayMothFlamePreset.get_cultural_elements(),
            "landmarks": SFBayMothFlamePreset.get_landmarks(),
            "conflicts": SFBayMothFlamePreset.get_conflicts(),
            "figures": SFBayMothFlamePreset.get_notable_figures(),
            # New additions
            "education": SFBayMothFlamePreset.get_educational_systems(),
            "religion": SFBayMothFlamePreset.get_religious_institutions(),
            "knowledge": SFBayMothFlamePreset.get_knowledge_traditions(),
            "quests": SFBayMothFlamePreset.get_quest_hooks(),
            "domestic_issues": SFBayMothFlamePreset.get_domestic_issues()
        }
        
        # Initialize through proper managers
        # Education
        from lore.managers.education import get_education_manager
        edu_mgr = await get_education_manager(user_id, conversation_id)
        for edu_system in preset_data['education']:
            await edu_mgr.add_educational_system(ctx, **edu_system)
        
        for tradition in preset_data['knowledge']:
            await edu_mgr.add_knowledge_tradition(ctx, **tradition)
        
        # Politics
        from lore.managers.politics import WorldPoliticsManager
        politics_mgr = WorldPoliticsManager(user_id, conversation_id)
        await politics_mgr.ensure_initialized()
        
        # Create SF as a "nation" for the politics system
        sf_nation_id = await politics_mgr.add_nation(
            ctx,
            name="San Francisco Bay Area",
            government_type="Municipal Democracy",
            description="Progressive city with dark underbelly",
            relative_power=8,
            matriarchy_level=6,  # Underground is matriarchal
            population_scale="7.7 million",
            major_resources=["Tech industry", "Port trade", "Tourism"],
            major_cities=["San Francisco", "Oakland", "San Jose"],
            cultural_traits=["Progressive", "Tech-focused", "Diverse", "Underground culture"]
        )
        
        # Add domestic issues
        for issue in preset_data['domestic_issues']:
            await politics_mgr.generate_domestic_issues(ctx, sf_nation_id, issue)
        
        logger.info("Complete SF Bay preset initialized")
        return preset_data

# Enhanced story initializer that uses the preset
class EnhancedMothFlameInitializer:
    """Enhanced initializer that loads all preset lore"""
    
    @staticmethod
    async def initialize_with_sf_preset(ctx, user_id: int, conversation_id: int):
        """Initialize the story with full SF Bay Area preset"""
        
        # First, get all the preset data
        preset_data = await SFBayMothFlamePreset.initialize_complete_sf_preset(
            ctx, user_id, conversation_id
        )
        
        # Get manager instances
        from lore.managers.local_lore import (
            add_urban_myth, add_local_history, add_landmark,
            generate_location_lore, LocationDataInput,
            MythCreationInput, HistoryCreationInput, LandmarkCreationInput
        )
        from lore.managers.politics import WorldPoliticsManager
        
        # Initialize districts as locations
        for district in preset_data['districts']:
            location_input = LocationDataInput(
                id=0,  # Will be assigned
                location_name=district['name'],
                location_type=district['type'],
                description=district['description']
            )
            
            # Generate location
            await generate_location_lore(ctx, location_input)
        
        # Add all urban myths
        for myth in preset_data['myths']:
            myth_input = MythCreationInput(
                name=myth['name'],
                description=myth['description'],
                origin_location=myth['origin_location'],
                believability=myth['believability'],
                spread_rate=8,  # High spread for well-known myths
                regions_known=myth['spread_regions'],
                themes=['gothic', 'protection', 'transformation'],
                matriarchal_elements=['female_savior', 'moth_queen']
            )
            
            await add_urban_myth(ctx, myth_input)
        
        # Add historical events
        for event in preset_data['history']:
            # First ensure location exists
            location_id = await _ensure_location_exists(ctx, event['location'])
            
            history_input = HistoryCreationInput(
                location_id=location_id,
                event_name=event['event_name'],
                description=event['description'],
                date_description=event['date'],
                significance=event['significance'],
                impact_type='transformative',
                notable_figures=['The Moth Queen lineage'],
                current_relevance=event.get('legacy', '')
            )
            
            await add_local_history(ctx, history_input)
        
        # Add specific landmarks
        for location in preset_data['locations']:
            # Get district ID
            district_name = _determine_district(location['name'])
            location_id = await _ensure_location_exists(ctx, district_name)
            
            landmark_input = LandmarkCreationInput(
                name=location['name'],
                location_id=location_id,
                landmark_type=location['type'],
                description=location['description'],
                current_use=location.get('schedule', {}).get('Monday', 'Active'),
                controlled_by='The Velvet Court',
                legends=[m['name'] for m in preset_data['myths'] if location['name'] in m.get('description', '')],
                matriarchal_significance='high'
            )
            
            await add_landmark(ctx, landmark_input)
        
        # Initialize political factions
        politics_mgr = WorldPoliticsManager(user_id, conversation_id)
        await politics_mgr.ensure_initialized()
        
        # Create factions
        for faction in preset_data['factions']:
            # Store faction data
            pass  # Implementation depends on your faction system
        
        logger.info("SF Bay Area preset lore initialized successfully")
        
        return preset_data


# Helper functions
async def _ensure_location_exists(ctx, location_name: str) -> int:
    """Ensure a location exists and return its ID"""
    from lore.core import canon
    from db.connection import get_db_connection_context
    
    async with get_db_connection_context() as conn:
        location_id = await canon.find_or_create_location(
            ctx, conn, location_name
        )
        
    return location_id


def _determine_district(location_name: str) -> str:
    """Determine which district a location belongs to"""
    location_lower = location_name.lower()
    
    if 'mission' in location_lower or 'valencia' in location_lower:
        return "The Mission Underground"
    elif 'soma' in location_lower or 'folsom' in location_lower:
        return "SoMa Shadow District"
    elif 'tenderloin' in location_lower:
        return "Tenderloin Shadows"
    elif 'financial' in location_lower or 'montgomery' in location_lower:
        return "Financial District After Hours"
    elif 'marina' in location_lower:
        return "The Marina Safehouse"
    else:
        return "The Mission Underground"  # Default
