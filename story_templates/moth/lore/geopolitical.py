# story_templates/moth/lore/geopolitical.py
"""
Geopolitical structures - districts, factions, and power dynamics for SF Bay Area
"""

from typing import Dict, Any, List

class SFGeopoliticalLore:
    """Geopolitical and factional lore for SF Bay Area"""
    
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
    def get_geopolitical_shift_scenarios() -> List[Dict[str, Any]]:
        """Power dynamics evolution in the Bay Area underground"""
        return [
            {
                "shift_type": "velvet_court_expansion",
                "description": (
                    "The Velvet Court's influence spreads to Oakland and San Jose. "
                    "New venue owners petition for membership, creating a Bay Area "
                    "underground federation."
                ),
                "power_changes": {
                    "The Moth Queen": "Becomes first among equals rather than absolute ruler",
                    "Regional Courts": "Oakland Leather Assembly, South Bay Shadows",
                    "New Tensions": "Traditional SF venues vs suburban expansions"
                },
                "trigger_event": "Major trafficking bust creates power vacuum"
            },
            {
                "shift_type": "tech_industry_infiltration",
                "description": (
                    "Silicon Shadows members begin placing allies in key tech positions. "
                    "The underground gains access to surveillance tech, turning the "
                    "panopticon against itself."
                ),
                "new_capabilities": [
                    "Facial recognition to identify trafficking victims",
                    "Encrypted communication networks",
                    "Predictive algorithms for tracking predators"
                ],
                "resistance": "Privacy advocates worry about underground surveillance state"
            }
        ]
    
    @staticmethod
    def get_world_evolution_scenarios() -> List[Dict[str, Any]]:
        """Major evolutionary scenarios for the entire setting"""
        return [
            {
                "scenario": "The Great Exposure",
                "description": "Major news outlet publishes underground expose",
                "steps": [
                    {
                        "phase": "Media Frenzy",
                        "duration": "1 week",
                        "effects": [
                            "Tourist influx to underground venues",
                            "Police pressure increases",
                            "Vanilla clients flee"
                        ]
                    },
                    {
                        "phase": "Underground Lockdown",
                        "duration": "1 month",
                        "effects": [
                            "Venues close or go deeper",
                            "Trust networks tighten",
                            "New verification protocols"
                        ]
                    },
                    {
                        "phase": "Evolution",
                        "duration": "6 months",
                        "effects": [
                            "New venues replace compromised ones",
                            "Stronger encryption adopted",
                            "Public face vs true underground splits"
                        ]
                    }
                ]
            },
            {
                "scenario": "The Tech Disruption",
                "description": "Silicon Valley fully embraces underground culture",
                "steps": [
                    {
                        "phase": "Corporate Adoption",
                        "effects": [
                            "HR departments add BDSM policies",
                            "Power exchange workshops at tech campuses",
                            "Sanitized version emerges"
                        ]
                    },
                    {
                        "phase": "Underground Resistance", 
                        "effects": [
                            "True underground goes deeper",
                            "Authentication becomes crucial",
                            "Commercial vs authentic split"
                        ]
                    },
                    {
                        "phase": "New Equilibrium",
                        "effects": [
                            "Surface-level mainstream version",
                            "Deep underground preserves truth",
                            "Middle layer translates between"
                        ]
                    }
                ]
            }
        ]
    
    @staticmethod
    def get_figure_evolution_scenarios() -> List[Dict[str, Any]]:
        """How key figures' stories evolve"""
        return [
            {
                "figure": "Lilith/The Moth Queen",
                "evolution_path": "legend_crystallization",
                "changes": [
                    {
                        "stage": "Local Mystery",
                        "reputation": 60,
                        "description": "Known only in the Mission underground"
                    },
                    {
                        "stage": "Underground Celebrity", 
                        "reputation": 75,
                        "description": "Every venue knows her name, not all believe"
                    },
                    {
                        "stage": "Urban Legend",
                        "reputation": 85,
                        "description": "Mainstream media mentions 'alleged vigilante'"
                    },
                    {
                        "stage": "Living Myth",
                        "reputation": 95,
                        "description": "Multiple people claim to be her, diluting/protecting truth"
                    }
                ]
            },
            {
                "figure": "Captain Maria Rodriguez",
                "evolution_path": "pragmatist_to_ally",
                "changes": [
                    {
                        "stage": "Skeptical Cop",
                        "trust_with_underground": 20,
                        "description": "Thinks Moth Queen is criminal playing hero"
                    },
                    {
                        "stage": "Reluctant Pragmatist",
                        "trust_with_underground": 50,
                        "description": "Realizes Moth Queen solves cases she can't"
                    },
                    {
                        "stage": "Unofficial Ally",
                        "trust_with_underground": 80,
                        "description": "Actively diverts resources from Queen investigations"
                    }
                ]
            }
        ]
