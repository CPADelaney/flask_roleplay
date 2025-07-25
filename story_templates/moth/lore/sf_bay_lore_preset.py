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
    async def initialize_preset_lore(ctx, user_id: int, conversation_id: int):
        """Initialize all the preset lore for the story"""
        logger.info("Initializing SF Bay Area preset lore for The Moth and Flame")
        
        results = {
            "world": SFBayMothFlamePreset.get_world_foundation(),
            "districts": SFBayMothFlamePreset.get_districts(),
            "locations": SFBayMothFlamePreset.get_specific_locations(),
            "myths": SFBayMothFlamePreset.get_urban_myths(),
            "history": SFBayMothFlamePreset.get_historical_events(),
            "factions": SFBayMothFlamePreset.get_factions(),
            "culture": SFBayMothFlamePreset.get_cultural_elements(),
            "landmarks": SFBayMothFlamePreset.get_landmarks(),
            "conflicts": SFBayMothFlamePreset.get_conflicts(),
            "figures": SFBayMothFlamePreset.get_notable_figures()
        }
        
        return results


# Enhanced story initializer that uses the preset
class EnhancedMothFlameInitializer:
    """Enhanced initializer that loads all preset lore"""
    
    @staticmethod
    async def initialize_with_sf_preset(ctx, user_id: int, conversation_id: int):
        """Initialize the story with full SF Bay Area preset"""
        
        # First, get all the preset data
        preset_data = await SFBayMothFlamePreset.initialize_preset_lore(
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
