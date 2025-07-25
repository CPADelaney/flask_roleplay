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
                "name": "The Mission District",
                "type": "cultural_hub",
                "description": (
                    "Historic Latino neighborhood fighting gentrification. Murals cover every "
                    "available wall, telling stories of struggle and celebration. Tech workers "
                    "in $4000/month studios live next to families who've been here for generations. "
                    "Valencia Street hosts trendy restaurants and vintage shops. But certain "
                    "boutiques have back rooms that only special clients know about."
                ),
                "notable_features": [
                    "Dolores Park - Weekend gatherings and hidden meetings",
                    "Valencia Street - Boutiques with secret purposes",
                    "Mission murals - Some contain coded messages",
                    "BART stations - Where different worlds collide"
                ],
                "demographics": "Latino families, artists, tech workers, activists",
                "hidden_element": "Certain shops cater to... specialized clientele"
            },
            {
                "name": "South of Market (SoMa)",
                "type": "tech_industrial",
                "description": (
                    "Former warehouses converted to tech offices and luxury lofts. By day, "
                    "it's all startups and artisanal coffee. The Giants' stadium brings crowds "
                    "on game days. But industrial spaces hide more than server farms. Some "
                    "CEOs have interests beyond quarterly earnings."
                ),
                "notable_features": [
                    "Oracle Park - Giants baseball and corporate boxes",
                    "Salesforce Tower - Tallest building, highest secrets",
                    "Museum of Modern Art - Public face of culture",
                    "Converted warehouses - Not all are offices"
                ],
                "demographics": "Tech workers, urban professionals, artists",
                "hidden_element": "Private clubs in unmarked buildings"
            },
            {
                "name": "Tenderloin",
                "type": "urban_struggle",
                "description": (
                    "The city's most honest neighborhood - poverty, addiction, and survival "
                    "in plain sight. Social services struggle to help. Vietnamese restaurants "
                    "serve authentic pho. Some people disappear here, but others find unexpected "
                    "protection. Certain apartments are safer than they appear."
                ),
                "notable_features": [
                    "Glide Memorial Church - Feeds body and soul",
                    "Little Saigon - Authentic food, watchful eyes",
                    "SRO hotels - More than just cheap housing",
                    "Street corners - Information exchanges"
                ],
                "demographics": "Low-income residents, immigrants, social workers",
                "hidden_element": "Underground protection networks"
            },
            {
                "name": "Financial District",
                "type": "corporate_center",
                "description": (
                    "Skyscrapers housing banks, law firms, and tech headquarters. Power lunches "
                    "at $200/plate restaurants. After the markets close, different transactions "
                    "begin. Some executives seek experiences their boards would never approve."
                ),
                "notable_features": [
                    "Transamerica Pyramid - Iconic architecture",
                    "Embarcadero Center - Shopping and secrets",
                    "Union Square - Luxury retail therapy",
                    "Private clubs - Members only"
                ],
                "demographics": "Executives, lawyers, financial professionals",
                "hidden_element": "Power dynamics that flip after dark"
            },
            {
                "name": "Marina District",
                "type": "wealthy_residential",
                "description": (
                    "Pastel houses with bay views, young professionals jogging past. "
                    "Wine bars and yoga studios on every corner. Perfect facades hide "
                    "imperfect lives. Some of the most expensive real estate shelters "
                    "unexpected residents."
                ),
                "notable_features": [
                    "Marina Green - Joggers and dog walkers",
                    "Chestnut Street - Shopping and dining",
                    "Palace of Fine Arts - Beautiful ruins",
                    "Private residences - Not all they seem"
                ],
                "demographics": "Wealthy professionals, trust fund kids",
                "hidden_element": "Safe houses in plain sight"
            },
            {
                "name": "Oakland",
                "type": "diverse_urban",
                "description": (
                    "Across the bay, more affordable but rapidly gentrifying. Strong African "
                    "American and artist communities. Lake Merritt provides urban nature. "
                    "The port brings international connections - not all legitimate."
                ),
                "notable_features": [
                    "Lake Merritt - Urban oasis with secrets",
                    "Jack London Square - Waterfront dining and more",
                    "Fruitvale - Latino community strength",
                    "Port of Oakland - Global connections"
                ],
                "demographics": "Diverse working class, artists, activists",
                "hidden_element": "Alternative power structures"
            }
        ]
    
    @staticmethod
    def get_factions() -> List[Dict[str, Any]]:
        """Get the power factions in the story"""
        return [
            {
                "name": "Tech Elite Network",
                "type": "corporate_power",
                "description": (
                    "CEOs, VCs, and founders who shape the Bay Area's economy. They disrupt "
                    "industries by day and seek other disruptions by night. Some fund causes "
                    "their boards would never approve. Guilt and desire make strange bedfellows."
                ),
                "territory": "Financial District, Palo Alto, Marina",
                "key_figures": {
                    "Public faces": "Featured in TechCrunch",
                    "Private interests": "Known to very few"
                },
                "resources": [
                    "Vast wealth",
                    "Political connections",
                    "Media influence",
                    "Private security"
                ],
                "goals": [
                    "Maintain public image",
                    "Pursue private interests",
                    "Avoid scandal"
                ],
                "hidden_nature": "Some seek experiences that would tank stock prices"
            },
            {
                "name": "The Thorn Garden",
                "type": "shadow_network",
                "description": (
                    "An invitation-only society of powerful women who've learned that true "
                    "power isn't always visible. They meet in book clubs, wine tastings, "
                    "charity boards - but these are just the surface. The Queen of Thorns "
                    "coordinates their deeper purposes."
                ),
                "territory": "Hidden throughout the Bay Area",
                "leadership": {
                    "The Queen of Thorns": "Identity unknown",
                    "The Roses": "Her inner circle",
                    "Gardeners": "Those who tend to problems"
                },
                "resources": [
                    "Information networks",
                    "Legitimate businesses as fronts",
                    "Loyalty beyond question",
                    "Ways to apply pressure"
                ],
                "goals": [
                    "Protect the vulnerable",
                    "Punish the predatory",
                    "Maintain the masquerade"
                ],
                "hidden_nature": "What looks like networking is actually power-broking"
            },
            {
                "name": "Bay Area Housing Coalition",
                "type": "activist_network",
                "description": (
                    "Fighting the housing crisis through protests, policy, and direct action. "
                    "They occupy empty buildings, protect tent cities, and expose slumlords. "
                    "Some members have unexpected allies in high places."
                ),
                "territory": "Wherever housing justice is needed",
                "leadership": "Rotating collective leadership",
                "resources": [
                    "Grassroots organizing",
                    "Legal support",
                    "Media connections",
                    "Hidden donors"
                ],
                "goals": [
                    "Affordable housing for all",
                    "Tenant protections",
                    "Community over profit"
                ]
            },
            {
                "name": "Old Money Families",
                "type": "traditional_power",
                "description": (
                    "Families who built San Francisco - railroad, shipping, banking dynasties. "
                    "They watch tech money with amusement and disdain. Their charity galas "
                    "and private clubs maintain older traditions. Some matriarchs remember "
                    "when women's power had to be entirely hidden."
                ),
                "territory": "Pacific Heights, Nob Hill, Atherton",
                "key_families": [
                    "Names on hospital wings",
                    "Museum benefactors",
                    "Silent partners"
                ],
                "resources": [
                    "Generational wealth",
                    "Political connections",
                    "Cultural institutions",
                    "Family secrets"
                ]
            },
            {
                "name": "International Trade Syndicate",
                "type": "criminal_network",
                "description": (
                    "Using the ports of Oakland and SF for smuggling. Everything from "
                    "counterfeit goods to worse. They see the Bay as a gateway to profit. "
                    "But certain shipments have been mysteriously disrupted lately."
                ),
                "territory": "Port areas, warehouses, distribution networks",
                "operations": [
                    "Import/export fronts",
                    "Warehouse networks",
                    "Transportation companies",
                    "Money laundering"
                ],
                "enemies": ["Law enforcement", "The Thorn Garden", "Rival syndicates"]
            },
            {
                "name": "SFPD and Law Enforcement",
                "type": "official_power",
                "description": (
                    "Trying to maintain order in a city of extremes. Some cops are corrupt, "
                    "some are crusaders, most just want to survive their shift. A few have "
                    "learned that some problems solve themselves if you look the other way."
                ),
                "territory": "Citywide jurisdiction",
                "factions": {
                    "By the book": "Follow procedure",
                    "Pragmatists": "Choose their battles",
                    "Corrupted": "On various payrolls"
                }
            }
        ]
    
    @staticmethod
    def get_faction_relationships() -> List[Dict[str, Any]]:
        """Get detailed faction relationships"""
        return [
            {
                "faction1": "Tech Elite Network",
                "faction2": "The Thorn Garden",
                "relationship_type": "complex_interdependence",
                "description": (
                    "Some tech elites unknowingly fund Thorn Garden operations through "
                    "charitable donations. Others are more directly involved, seeking "
                    "experiences they can't find elsewhere. The Garden keeps their secrets "
                    "- for a price."
                ),
                "public_face": "Charity partnerships",
                "hidden_reality": "Power games and mutual benefit"
            },
            {
                "faction1": "The Thorn Garden",
                "faction2": "International Trade Syndicate",
                "relationship_type": "shadow_war",
                "description": (
                    "The Garden disrupts trafficking operations without revealing themselves. "
                    "Shipments go missing, key players vanish, plans leak to authorities. "
                    "The Syndicate suspects but can't prove the Garden's involvement."
                ),
                "public_face": "No known connection",
                "hidden_reality": "The Queen of Thorns hunts predators"
            },
            {
                "faction1": "Old Money Families",
                "faction2": "Tech Elite Network",
                "relationship_type": "generational_tension",
                "description": (
                    "Old money sees new money as crass and destructive. Tech elite see "
                    "old families as outdated obstacles. But in certain private clubs, "
                    "they find common ground in shared... interests."
                ),
                "public_face": "Polite disdain",
                "hidden_reality": "More connections than either admits"
            }
        ]
    
    @staticmethod
    def get_conflicts() -> List[Dict[str, Any]]:
        """Get ongoing conflicts in the story world"""
        return [
            {
                "name": "The Housing Wars",
                "type": "socioeconomic",
                "description": (
                    "Tech money drives out long-time residents. Homeless camps cleared "
                    "for development. Activists fight back with protests and occupations. "
                    "Some tech workers secretly fund resistance. The city bleeds culture "
                    "for profit."
                ),
                "factions": ["Tech companies", "Real estate developers", "Housing activists", "Residents"],
                "stakes": "The soul of San Francisco",
                "current_state": "Escalating tensions"
            },
            {
                "name": "The Shadow Economy",
                "type": "hidden_conflict",
                "description": (
                    "Beneath legitimate business, power flows through hidden channels. "
                    "The Thorn Garden protects some, punishes others. Corporate executives "
                    "with dark interests. A careful balance that could explode if exposed."
                ),
                "factions": ["The Thorn Garden", "Hidden predators", "Unknowing public"],
                "stakes": "Maintaining the masquerade",
                "current_state": "Delicate balance"
            },
            {
                "name": "Port Control",
                "type": "criminal",
                "description": (
                    "International syndicates fight for smuggling routes. Corruption reaches "
                    "into port authority and city hall. The Thorn Garden disrupts the worst "
                    "operations while authorities play catch-up."
                ),
                "factions": ["Trade syndicates", "Law enforcement", "The Thorn Garden"],
                "stakes": "Control of trafficking routes",
                "current_state": "Three-way chess game"
            }
        ]

    @staticmethod
    def get_notable_figures() -> List[Dict[str, Any]]:
        """Get important NPCs beyond the main cast"""
        return [
            {
                "name": "Victoria Chen",
                "role": "Tech CEO / Hidden Thorn",
                "public_persona": (
                    "Founded a successful fintech startup. Regular speaker at women in tech "
                    "events. Philanthropist focused on girls' education. Soft-spoken, professional, "
                    "always perfectly put together."
                ),
                "hidden_nature": (
                    "High-ranking member of the Thorn Garden. Uses her company's resources "
                    "to track financial crimes. Has a private floor in her building that "
                    "employees can't access. Those who know her true nature don't speak of it."
                ),
                "influence": "Controls information and money flows"
            },
            {
                "name": "Detective Maria Rodriguez",
                "role": "SFPD - Special Victims Unit",
                "public_persona": (
                    "20-year veteran known for being tough but fair. Solved several high-profile "
                    "cases. Divorced, two kids in college. Drinks too much coffee, works too late."
                ),
                "hidden_nature": (
                    "Has learned some cases solve themselves if she waits. Occasionally receives "
                    "anonymous tips that pan out perfectly. Suspects but doesn't investigate "
                    "certain patterns. Pragmatist who chooses her battles."
                ),
                "influence": "Can make problems disappear or appear"
            },
            {
                "name": "Evelyn Thornfield",
                "role": "Society Matron / The Queen's Voice",
                "public_persona": (
                    "Old money philanthropist in her 60s. Sits on museum boards, hosts charity "
                    "galas. Married to a federal judge. Known for her rose garden in Pacific Heights."
                ),
                "hidden_nature": (
                    "One of the Queen of Thorns' most trusted advisors. Her charity events "
                    "are recruiting grounds. Her 'book club' makes decisions that affect "
                    "the entire city. Even her husband doesn't know."
                ),
                "influence": "Connects old power to new purposes"
            },
            {
                "name": "James Morrison",
                "role": "Venture Capitalist / Useful Fool",
                "public_persona": (
                    "Partner at top Sand Hill Road firm. Funded three unicorns. Married "
                    "with kids in private school. Weekend warrior cyclist. Ted Talks about "
                    "disruption."
                ),
                "hidden_nature": (
                    "Seeks experiences his wife would never understand. Thinks he's found "
                    "discrete services. Doesn't realize he's being managed, his desires "
                    "channeled into useful directions. His 'donations' fund many things."
                ),
                "influence": "Wealth directed by others"
            },
            {
                "name": "Dr. Sarah Kim",
                "role": "ER Doctor / Underground Healer",
                "public_persona": (
                    "Trauma surgeon at SF General. Published researcher. Volunteers at "
                    "free clinics. Professional, compassionate, always exhausted."
                ),
                "hidden_nature": (
                    "Treats injuries no questions asked. Has a network that brings her "
                    "special cases. Owes the Queen of Thorns a debt that can never be "
                    "repaid. Her oath includes more than 'do no harm.'"
                ),
                "influence": "Heals what shouldn't be seen"
            }
        ]
    
    @staticmethod
    def get_world_foundation() -> Dict[str, Any]]:
        """Get the base world setting"""
        return {
            "world_setting": {
                "name": "San Francisco Bay Area 2025",
                "type": "modern_metropolis",
                "population": "7.7 million",
                "description": (
                    "A region of stunning beauty and stark contrasts. Tech wealth creates "
                    "gleaming towers while tent cities grow in their shadows. Progressive "
                    "ideals clash with capitalist reality. The fog rolls in each evening, "
                    "blurring the lines between what's seen and what's hidden. Beneath "
                    "the surface of artisanal coffee shops and billion-dollar valuations, "
                    "older powers flow through new channels."
                ),
                "atmosphere": {
                    "physical": "Hills, bay views, Karl the Fog, earthquake country",
                    "social": "Innovation and inequality, diversity and displacement",
                    "hidden": "Power games played behind progressive facades"
                }
            }
        }
