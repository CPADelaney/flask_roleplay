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
                "type": "cultural_battleground",
                "description": (
                    "Where San Francisco's soul fights for survival. Murals depicting Aztec "
                    "warriors share walls with 'Apartment Available' signs nobody can afford. "
                    "Valencia Street hosts $20 cocktail bars next to panaderías older than "
                    "Silicon Valley. The underground BART stations pulse with more than trains - "
                    "they're crossroads where different worlds negotiate. Certain vintage boutiques "
                    "have fitting rooms that fit more than clothes. The back room at Therapy "
                    "restaurant isn't about food."
                ),
                "notable_features": [
                    "The Women's Building - Hosts 'workshops' on empowerment",
                    "Dolores Park - Weekend gatherings with hidden hierarchies",
                    "Valencia Street boutiques - Some cater to specific tastes",
                    "Mission Cultural Center - Traditional dance, modern power",
                    "El Rio - Queer bar where certain Sundays are invitation-only",
                    "Secret alley galleries - Art that depicts interesting dynamics"
                ],
                "areas": [
                    "24th Street - Latino heart with underground pulse",
                    "Valencia Corridor - Gentrification and resistance",
                    "The warehouses off Cesar Chavez - Not all are storage",
                    "Clarion Alley - Murals hide messages for those who read"
                ],
                "demographics": "Latino families, artists, tech colonizers, night wanderers",
                "danger_level": "Depends who you are and who you cross",
                "hidden_elements": (
                    "The Rose & Thorn bookstore's reading groups discuss more than literature. "
                    "Certain taquerías have special late-night menus. The dance studio above "
                    "the hardware store teaches movements not found in any manual."
                )
            },
            {
                "name": "South of Market (SoMa)",
                "type": "transformed_industrial",
                "description": (
                    "Leather bars gave way to leather seats in Teslas, but the infrastructure "
                    "of desire adapted rather than died. Glass towers shade warehouses where "
                    "different kinds of unicorns are born. The Folsom Street Fair maintains "
                    "traditions tech bros Instagram without understanding. Beneath startup offices, "
                    "basement spaces serve needs no app can disrupt. The power exchange here "
                    "involves more than equity."
                ),
                "notable_features": [
                    "The Armory - Former porn studio, now 'event space' with history",
                    "Folsom Street - Leather legacy lives in certain establishments",
                    "The End Up - After-hours institution with coded nights",
                    "DNA Lounge - All-ages by day, specific ages by night",
                    "Slim's ghost - Closed venue, but the basement still operates",
                    "Private clubs in unmarked buildings - Members know the doors"
                ],
                "areas": [
                    "11th Street corridor - Nightlife with deeper purposes",
                    "Folsom between 7th and 11th - The leather trail",
                    "Former Twitter building - Certain floors were never Twitter's",
                    "The Mint - Karaoke bar where some performances are rituals"
                ],
                "demographics": "Tech workers, artists, leather community veterans, seekers",
                "hidden_elements": (
                    "The Executive Network meets monthly in different penthouses. Entry "
                    "requires more than money. The old bathhouse tunnels still connect "
                    "buildings - now they transport different cargo."
                )
            },
            {
                "name": "Tenderloin",
                "type": "unvarnished_truth",
                "description": (
                    "The city's id laid bare - desperation, survival, and unexpected sanctuaries. "
                    "While City Hall gleams across the plaza, these streets tell different stories. "
                    "But between the dealer corners and SRO hotels, protection flows through "
                    "unexpected channels. The woman feeding the homeless might be feeding "
                    "information upward. Certain massage parlors offer more protection than "
                    "pleasure. The Thorn Garden grows wild here."
                ),
                "notable_features": [
                    "Glide Memorial - Sanctuary in multiple senses",
                    "The Phoenix Hotel - Rock & roll history, current mysteries",
                    "Aunt Charlie's - Gay institution with protective networks",
                    "Boeddeker Park - Renovated surface, unchanged depths",
                    "Salon Meritage - 'Wine bar' with interesting tastings",
                    "The Cadillac Hotel - SRO with safer rooms than others"
                ],
                "areas": [
                    "Little Saigon - Pho shops and protective networks",
                    "Turk Street corridor - Information flows with other trades",
                    "Hyde Street - Where visibility becomes invisibility",
                    "UN Plaza - Farmers market by day, different harvest by night"
                ],
                "demographics": "Survivors, social workers, hidden protectors, the desperate",
                "danger_level": "High but navigable with the right connections",
                "hidden_elements": (
                    "Room 237 at the Phoenix has a guest list that would shock. The Vietnamese "
                    "grandmothers who play cards see everything and forget nothing. Certain "
                    "social workers provide more than county services."
                )
            },
            {
                "name": "Financial District (FiDi)",
                "type": "daytime_masquerade",
                "description": (
                    "Fifty floors of spreadsheets hiding five basements of secrets. Power "
                    "lunches at $500/plate restaurants where the real negotiations happen "
                    "in private dining rooms. After the markets close, different trades begin. "
                    "Executive assistants who manage more than calendars. Board rooms that "
                    "transform after hours. The higher the floor, the deeper the surrender."
                ),
                "notable_features": [
                    "Transamerica Pyramid - Urban legends about the sub-basements",
                    "555 California - Former BofA building with executive secrets",
                    "Private clubs - Pacific Union, Metropolitan, others unnamed",
                    "Tadich Grill - Old money meets new proclivities",
                    "The Punch Line - Comedy club where some jokes aren't",
                    "One Montgomery - Penthouse 'conferences' by invitation"
                ],
                "areas": [
                    "Montgomery Street - The traditional power corridor",
                    "Battery Street - Hedge funds and hedged bets",
                    "Jackson Square - Antiques and antiquated power structures",
                    "The Embarcadero - Where international connections dock"
                ],
                "demographics": "Executives, lawyers, international traders, expensive secrets",
                "hidden_elements": (
                    "The Executive Accountability Group meets in different offices. "
                    "Membership is involuntary once certain information surfaces. "
                    "The concierge at certain buildings provides unique services."
                )
            },
            {
                "name": "Marina District",
                "type": "pastel_paradise",
                "description": (
                    "Lululemon and lies, yoga mats and hidden masters. The morning joggers "
                    "passing million-dollar views don't see the midnight transformations. "
                    "Wine bars where the pairing notes include power dynamics. Pilates studios "
                    "teaching core strength and control. Behind perfect facades, marriages have "
                    "interesting agreements. The marina itself hosts more than yachts - certain "
                    "boats never sail but always have visitors."
                ),
                "notable_features": [
                    "Marina Green - Morning yoga, midnight meetings",
                    "The Palace of Fine Arts - Beautiful ruins hiding modern secrets",
                    "Crissy Field - Dog walking and coded conversations",
                    "Fort Mason - Art galleries and private showings",
                    "The Wave Organ - Acoustic privacy for sensitive talks",
                    "Marina Safehouse - Disguised as wellness retreat"
                ],
                "areas": [
                    "Chestnut Street - Shopping for more than fashion",
                    "Yacht Harbor - Floating private venues",
                    "The residential blocks - Perfect facades, complex interiors",
                    "Fillmore connection - Where different worlds meet"
                ],
                "demographics": "Trust fund babies, young professionals, hidden complexities",
                "hidden_elements": (
                    "The Marina Wellness Center's 'executive program' includes unique therapies. "
                    "Certain book clubs read between very different lines. The sailing club "
                    "has members who never sail but always pay dues."
                )
            },
            {
                "name": "Castro District",
                "type": "liberated_territory",
                "description": (
                    "Where America learned queer power, now teaching other lessons. The rainbow "
                    "flags fly over deeper revolutions. Harvey Milk's legacy includes structures "
                    "he never imagined. Leather bars that survived gentrification guard older "
                    "protocols. The Sisters of Perpetual Indulgence know more than they absolve. "
                    "In the birthplace of visible pride, invisible hierarchies flourish."
                ),
                "notable_features": [
                    "The Edge - Leather bar maintaining traditions",
                    "440 Castro - Bar with particularly interesting theme nights",
                    "Castro Theatre - Midnight shows aren't all on screen",
                    "Cliff's Variety - Hardware store with specialized inventory",
                    "Hot Cookie - Late night bakery and information exchange",
                    "The Mix - Bar where power dynamics are explicit"
                ],
                "demographics": "LGBTQ+ community, allies, leather culture, explorers",
                "hidden_elements": (
                    "The Harvey Milk Plaza meetings that happen after midnight. Certain "
                    "Victorian houses host salons discussing more than politics. The back "
                    "bar at The Edge requires more than ID for entry."
                )
            },
            {
                "name": "Pacific Heights",
                "type": "mansion_mysteries",
                "description": (
                    "Old money built these hills, new money can't buy what happens behind "
                    "the gates. Consulate rows where diplomatic immunity covers interesting "
                    "activities. Private gardens where the Rose Society meets monthly. Views "
                    "worth millions hiding dynamics worth more. The help knows everything and "
                    "says nothing - until they're asked by the right person."
                ),
                "notable_features": [
                    "The Flood Mansion - Private events with specific guest lists",
                    "Lafayette Park - Dog walking and power broking",
                    "Fillmore shopping - Boutiques with back rooms",
                    "Private clubs - Names known only to members",
                    "Consulate events - International power mingles"
                ],
                "demographics": "Old money, new billionaires, diplomats, those who serve",
                "hidden_elements": (
                    "The Monday Afternoon Club isn't about bridge. Certain estates have "
                    "dungeons older than tech. The society photographers know which "
                    "events not to document."
                )
            },
            {
                "name": "Chinatown",
                "type": "ancient_adaptations",
                "description": (
                    "America's oldest Chinatown keeps secrets in languages startup founders "
                    "can't Google Translate. Family associations that predate California "
                    "understand power structures Silicon Valley disrupts. Herb shops dispense "
                    "more than traditional medicine. Mah-jong parlors where the real game "
                    "isn't tiles. The matriarchs here taught the Queen of Thorns about "
                    "invisible authority."
                ),
                "notable_features": [
                    "The associations - Benevolent on surface, complex beneath",
                    "Underground tunnels - Tourist myth, practical reality",
                    "Herb shops - Traditional medicine, modern chemistry",
                    "Tea rooms - Negotiations in Mandarin and silence",
                    "The fortune tellers - Some predictions are plans"
                ],
                "demographics": "Multi-generational families, tourists, initiates",
                "hidden_elements": (
                    "The Empress of China's top floor was more than a restaurant. "
                    "Certain import shops handle special orders. The women's mah-jong "
                    "groups make decisions that ripple through the city."
                )
            }
        ]
    
    @staticmethod
    def get_factions() -> List[Dict[str, Any]]:
        """Get the power factions in the story"""
        return [
            {
                "name": "The Shadow Network (outsiders call it 'The Rose & Thorn Society')",
                "type": "shadow_matriarchy",
                "description": (
                    "What began as garden clubs and professional women's associations evolved "
                    "into something without a name. Outsiders call it The Rose & Thorn Society, "
                    "The Garden, or The Thorn Network, but insiders know it has no formal title. "
                    "They meet at charity galas and wine tastings, but their real work happens "
                    "in private rooms and encrypted channels. The Queen of Thorns - the only "
                    "certainty - coordinates through cut flowers and careful words..."
                ),
                "territory": "Embedded throughout Bay Area society",
                "leadership": {
                    "The Queen of Thorns": "Identity unknown - possibly multiple women",
                    "The Rose Council": "Seven women who run major institutions",
                    "Thorn Bearers": "Enforcement and protection specialists",
                    "Gardeners": "Those who tend problems and cultivate assets",
                    "Seedlings": "Initiates learning to bloom"
                },
                "resources": [
                    "Information networks spanning tech to judiciary",
                    "Kompromat on half of Silicon Valley",
                    "Alternative justice enforcement",
                    "Safe houses disguised as spas and wellness centers",
                    "Funding through 'consulting fees' and 'donations'"
                ],
                "goals": [
                    "Protect women from predators",
                    "Redistribute power through careful pressure",
                    "Maintain the secret matriarchy",
                    "Transform surrender into strength"
                ],
                "methods": [
                    "Blackmail disguised as business leverage",
                    "Protection through preemptive strikes",
                    "Behavioral modification via targeted pressure",
                    "Alternative justice outside legal systems"
                ],
                "recognition_signs": [
                    "Rose jewelry, especially antique pieces",
                    "Attendance at specific charity events",
                    "Language involving gardening metaphors",
                    "Positions of quiet influence"
                ]
            },
            {
                "name": "Tech Titans & Fallen Angels",
                "type": "economic_aristocracy",
                "description": (
                    "The visible gods of Silicon Valley and their hidden hungers. Disrupting "
                    "industries by day, seeking disruption by night. Some stumble into the "
                    "Thorn Garden's web through arrogance, others crawl in through need. "
                    "Their clean public images hide private stains that become leverage. "
                    "The smartest ones learn to submit before they're made to."
                ),
                "territory": "Sand Hill Road to Salesforce Tower",
                "key_players": {
                    "The Untouchables": "Too big to fall, too careful to catch",
                    "The Compromised": "Paying for silence and protection",
                    "The Converted": "Found peace in surrender",
                    "The Hunted": "Still think they can win"
                },
                "resources": [
                    "Billions in venture capital",
                    "Political influence through lobbying",
                    "Media control through investment",
                    "Technology for surveillance and control"
                ],
                "vulnerabilities": [
                    "Hubris leading to exploitable mistakes",
                    "Digital trails they think are hidden",
                    "Employees who know too much",
                    "Desires that compromise their images"
                ],
                "relationship_to_thorns": (
                    "Complex interdependence - some fund unknowingly, some fund "
                    "knowingly, some ARE the funding after special negotiations"
                )
            },
            {
                "name": "The Underground Railroad Redux",
                "type": "protection_network",
                "description": (
                    "Modern abolitionists running digital and physical routes for those "
                    "escaping trafficking, abuse, or worse. They learned from history and "
                    "adapted - safe houses look like Airbnbs, conductors drive for rideshares, "
                    "the network runs on encrypted apps designed by sympathetic coders. The "
                    "Queen of Thorns ensures no one travels alone."
                ),
                "territory": "The spaces between - routes, not regions",
                "structure": {
                    "Station Masters": "Safe house operators",
                    "Conductors": "Those who move people",
                    "Scouts": "Identify those needing help",
                    "Shepherds": "Recovery and integration specialists",
                    "The Underground": "Everyone who helps, no questions asked"
                },
                "resources": [
                    "Network of safe locations",
                    "Identity creation capabilities",
                    "Medical care off the books",
                    "Legal assistance that bends rules",
                    "Funding from converted executives"
                ],
                "allies": ["The Rose & Thorn Society", "Progressive churches", "Certain cops"],
                "enemies": ["Traffickers", "Abusive partners", "Those who profit from pain"]
            },
            {
                "name": "The Velvet Rope",
                "type": "pleasure_merchants",
                "description": (
                    "The evolution of SF's leather and kink community into something more "
                    "structured. They run dungeons disguised as yoga studios, clubs that "
                    "look like restaurants, services that advertise wellness but deliver "
                    "submission. The old guard maintains protocols while new money learns "
                    "positions. Not all are aligned with the Thorns, creating interesting tensions."
                ),
                "territory": "SoMa primarily, satellites throughout",
                "hierarchy": {
                    "Masters & Mistresses": "Those who've earned titles",
                    "Professionals": "Those who make it a living",
                    "Devotees": "Lifestyle participants",
                    "Tourists": "Weekend warriors and curious"
                },
                "venues": [
                    "Private dungeons in industrial spaces",
                    "Members-only clubs",
                    "Professional houses of discipline",
                    "Training academies for dominants and submissives"
                ],
                "code": "Safe, Sane, Consensual - until it needs not to be",
                "relationship_to_power": "They understand it intimately"
            },
            {
                "name": "Old San Francisco Matriarchy",
                "type": "traditional_power",
                "description": (
                    "The women who ran the city when their husbands thought they did. "
                    "Pacific Heights dynasties, Nob Hill nobility, Marina mavens who "
                    "inherited more than money. They've always known how to wield power "
                    "from behind thrones. The Rose & Thorn Society learned much from them. "
                    "Their charity boards are war councils, their garden parties are parliaments."
                ),
                "territory": "The traditional heights of power",
                "institutions": [
                    "Museum boards and cultural committees",
                    "Private schools and their parent associations",
                    "Charity organizations with nine-figure endowments",
                    "Country clubs that don't advertise membership"
                ],
                "methods": [
                    "Social exclusion as execution",
                    "Philanthropy as power projection",
                    "Marriage as merger and acquisition",
                    "Gossip as intelligence network"
                ],
                "evolution": "Teaching new money old ways, learning new tricks themselves"
            },
            {
                "name": "International Shadows",
                "type": "criminal_syndicate",
                "description": (
                    "Global trafficking networks using SF's ports and tech connections. "
                    "They run crypto, code, and cargo - not all of it willing. But the "
                    "city has antibodies they didn't expect. Their shipments disappear, "
                    "their codes get leaked, their cargo finds freedom. They're learning "
                    "that SF's underground has thorns."
                ),
                "operations": [
                    "Port of Oakland shipping",
                    "Tech visa manipulation",
                    "Cryptocurrency laundering",
                    "Dark web marketplaces"
                ],
                "opposition": "The Rose & Thorn Society wages quiet war",
                "losses": "Mounting, unexplained, concerning their bosses"
            },
            {
                "name": "The Badge & The Burden",
                "type": "law_enforcement",
                "description": (
                    "SFPD, FBI, DEA - all trying to police a city that polices itself in "
                    "shadows. Some cops learn to work with the underground rivers, others "
                    "try to dam them. The smart ones realize certain cases solve themselves "
                    "if they wait. The Queen of Thorns has files on judges, DAs, and chiefs "
                    "- but prefers cooperation to coercion."
                ),
                "factions": {
                    "The Pragmatists": "Work with the shadows",
                    "The Crusaders": "Try to bring light everywhere",
                    "The Corrupted": "Serve various masters",
                    "The Converted": "Understand alternative justice"
                },
                "challenge": "Balancing official duty with street reality",
                "accommodation": "Some crimes aren't reported, some solutions aren't official"
            },
            {
                "name": "The Service Submissives",
                "type": "hidden_infrastructure",
                "description": (
                    "They clean houses and secrets, serve coffee and surveillance, manage "
                    "schedules and souls. Administrative assistants who administer more than "
                    "calendars. House cleaners who clean more than houses. Personal trainers "
                    "making bodies and minds submit. They see everything, serve faithfully, "
                    "and report to thornier authorities than their employers know."
                ),
                "positions": [
                    "Executive assistants with executive power",
                    "Household staff who manage more than homes",
                    "Personal services that get very personal",
                    "Drivers who know where bodies are buried"
                ],
                "network": "Informal but unbreakable",
                "power": "Access plus invisibility equals control"
            }
        ]
    
    @staticmethod
    def get_faction_relationships() -> List[Dict[str, Any]]:
        """Get detailed faction relationships beyond basic conflicts"""
        return [
            {
                "faction1": "The Rose & Thorn Society",
                "faction2": "Tech Titans & Fallen Angels",
                "relationship_type": "predator_and_prey_and_partner",
                "description": (
                    "The Thorns hunt those who abuse power while cultivating those who "
                    "surrender it properly. Every tech scandal that doesn't break was "
                    "bought with submission. Some executives pay willingly for silence, "
                    "others discover the price of defiance. The smartest ones join the "
                    "garden voluntarily."
                ),
                "public_interface": "Women in tech initiatives, executive coaching",
                "private_reality": "Behavioral modification through pressure and pleasure",
                "power_dynamic": "The Thorns hold leashes disguised as consulting contracts",
                "notable_conversions": "Three unicorn founders now fund safe houses",
                "ongoing_operations": "Quarterly 'executive retreats' that transform leaders"
            },
            {
                "faction1": "The Underground Railroad Redux",
                "faction2": "International Shadows",
                "relationship_type": "shadow_war",
                "description": (
                    "Every trafficking route has a parallel extraction route. Every "
                    "shipment tracked, every victim marked for rescue. The Railroad "
                    "uses the Shadows' own infrastructure against them. Containers "
                    "arrive empty, safe houses appear where brothels were planned. "
                    "It's underground warfare with no recognition, only results."
                ),
                "tactics": [
                    "Information warfare through tech allies",
                    "Physical extraction using Shadow transport",
                    "Financial strangulation via converted executives",
                    "Legal pressure through placed prosecutors"
                ],
                "body_count": "Low - disappearances, not deaths",
                "success_metric": "Lives saved versus cargo lost"
            },
            {
                "faction1": "The Velvet Rope",
                "faction2": "The Rose & Thorn Society",
                "relationship_type": "allied_competition",
                "description": (
                    "Both understand power exchange, but their purposes diverge. The "
                    "Rope serves pleasure, the Thorns serve justice - though both blur. "
                    "They share members, venues, and victims of different sorts. The "
                    "Queen of Thorns was trained in Velvet traditions before she "
                    "transcended them."
                ),
                "cooperation": "Shared safe words, protocols, and protection",
                "competition": "Recruiting dominants, claiming submissives",
                "neutral_ground": "Folsom Street Fair and similar events",
                "cross_pollination": "Techniques, personnel, and purposes"
            },
            {
                "faction1": "Old San Francisco Matriarchy",
                "faction2": "The Rose & Thorn Society",
                "relationship_type": "mentor_and_evolution",
                "description": (
                    "The old guard taught subtlety, the new guard teaches action. "
                    "Pacific Heights matrons who ruled through tradition watch their "
                    "daughters rule through transformation. Some join the Thorns, "
                    "others resist the crude new methods. But all acknowledge the power."
                ),
                "generational_shift": "From implicit to explicit feminine authority",
                "shared_resources": "Charity networks, social capital, safe spaces",
                "philosophical_differences": "Patience versus urgency, tradition versus innovation",
                "bridge_figures": "Women who speak both languages of power"
            },
            {
                "faction1": "The Badge & The Burden",
                "faction2": "The Rose & Thorn Society",
                "relationship_type": "official_ignorance",
                "description": (
                    "The best cops know when not to investigate. Certain missing persons "
                    "aren't missing, they're transformed. Certain crimes solve themselves "
                    "with poetic justice. Detective Rodriguez has a rose on her desk she "
                    "never explains. The Chief attends charity galas where thorns are "
                    "discussed metaphorically."
                ),
                "unwritten_rules": [
                    "No civilians harmed, no questions asked",
                    "Trafficking trumps all other crimes",
                    "Alternative justice for special cases",
                    "Information flows both ways when needed"
                ],
                "breaking_points": "When bodies appear or media notices",
                "mutual_benefit": "Thorns solve cases cops can't touch legally"
            },
            {
                "faction1": "The Service Submissives",
                "faction2": "All Power Factions",
                "relationship_type": "invisible_omnipresence",
                "description": (
                    "They serve everyone and belong to the shadows. Every faction thinks "
                    "they own these servants, but the service network owns information. "
                    "They're the nervous system of the city's hidden body, carrying "
                    "messages, warnings, and occasionally, revenge."
                ),
                "dual_loyalties": "Paid by employers, loyal to something deeper",
                "information_flow": "Everything seen reported somewhere",
                "protection_mechanism": "Too useful to eliminate, too numerous to control",
                "ultimate_allegiance": "To each other and those who protect them"
            }
        ]
    
    @staticmethod
    def get_conflicts() -> List[Dict[str, Any]]:
        """Get ongoing conflicts in the story world"""
        return [
            {
                "name": "The Transformation Wars",
                "type": "shadow_conflict",
                "description": (
                    "Not everyone taken by the Thorns wants transformation. Some tech bros "
                    "resist their reduction, some traffickers refuse reeducation. The city "
                    "becomes a battlefield of wills where submission is victory and resistance "
                    "is slowly ground down. Bodies are rarely found, but personalities disappear "
                    "regularly."
                ),
                "battlegrounds": [
                    "Corporate boardrooms where CEOs suddenly resign",
                    "Exclusive clubs where members vanish from rolls",
                    "Private dungeons where resistance meets inevitability",
                    "Therapists' offices practicing unorthodox methods"
                ],
                "weapons": [
                    "Kompromat and coercion",
                    "Psychological pressure and physical persuasion",
                    "Financial strangulation and social isolation",
                    "Carrot and stick, pleasure and pain"
                ],
                "casualties": "Egos destroyed, personalities rebuilt",
                "victory_conditions": "Submission or disappearance"
            },
            {
                "name": "The Great Revelation Threat",
                "type": "information_warfare",
                "description": (
                    "Investigative journalists, federal agents, and tech whistleblowers "
                    "circle the truth. The shadow matriarchy faces exposure. But every "
                    "investigation hits walls of lawyers, losses of evidence, witnesses "
                    "who recant or relocate. The Thorns protect their garden viciously."
                ),
                "threat_vectors": [
                    "Tech workers discovering executive blackmail",
                    "Journalists connecting disappearances",
                    "Federal investigation into trafficking",
                    "Social media activists noticing patterns"
                ],
                "defense_mechanisms": [
                    "Discrediting through manufactured scandals",
                    "Recruiting through conversion",
                    "Legal labyrinth via allied lawyers",
                    "Alternative facts flooding information space"
                ],
                "close_calls": "The Chronicle series that never ran, the FBI task force that dissolved",
                "current_status": "Multiple threats requiring constant gardening"
            },
            {
                "name": "The Succession Question",
                "type": "internal_struggle",
                "description": (
                    "Who becomes the next Queen of Thorns? The current Queen (or Queens) "
                    "age, and ambitious Roses vie for position. Some push for expansion, "
                    "others for deeper secrecy. The struggle plays out in charity board "
                    "elections and hostile business takeovers."
                ),
                "factions_within": [
                    "Expansionists wanting open matriarchy",
                    "Traditionalists maintaining shadows",
                    "Revolutionaries proposing new methods",
                    "Pragmatists balancing all approaches"
                ],
                "proxy_battles": "Control of safe houses, loyalty of key submissives",
                "potential_outcomes": "Evolution or civil war"
            },
            {
                "name": "The Port Wars",
                "type": "territorial_conflict",
                "description": (
                    "International trafficking syndicates fight the Underground Railroad "
                    "for control of shipping routes. Containers arrive empty, crews report "
                    "ghost boardings, manifests alter themselves. The FBI investigates "
                    "everyone while understanding nothing."
                ),
                "physical_battles": "Warehouse raids, dock ambushes, sea extractions",
                "digital_warfare": "Shipping databases, cryptocurrency traces, dark web markets",
                "body_count": "Rising on the shadow side",
                "public_impact": "Unexplained port delays, missing cargo reports"
            },
            {
                "name": "The Consent Wars",
                "type": "philosophical_conflict",
                "description": (
                    "Within the kink community, debates rage about consensual non-consent, "
                    "financial domination, and behavioral modification. The Thorns practice "
                    "edge play with entire lives. The Velvet Rope maintains strict protocols. "
                    "Where's the line between justice and abuse?"
                ),
                "ethical_questions": [
                    "Can someone consent to transformation?",
                    "Is blackmail acceptable for protection?",
                    "Where does justice end and revenge begin?",
                    "Who decides who needs saving?"
                ],
                "practical_impacts": "Schisms in dungeons, competing protocols",
                "philosophical_stakes": "The soul of the shadow community"
            }
        ]

    @staticmethod
    def get_world_foundation() -> Dict[str, Any]:
        """Get the base world setting"""
        return {
            "world_setting": {
                "name": "San Francisco Bay Area 2025",
                "type": "modern_metropolis_with_shadows",
                "population": "7.7 million (documented)",
                "description": (
                    "A peninsula of paradoxes where trillion-dollar companies rise from "
                    "former graveyards, where progressive politics mask ancient power games. "
                    "The City by the Bay gleams with glass towers while tent cities multiply "
                    "in their shadows. Tech disruption meets traditions that refuse to be "
                    "disrupted. The famous fog isn't the only thing obscuring truth here. "
                    "Beneath the hackathons and protests, the fundraisers and festivals, "
                    "an older economy persists - one that trades in power, pleasure, and "
                    "transformation."
                ),
                "atmosphere": {
                    "physical": "Fog and microclimates, hills and hidden valleys, bridges between worlds",
                    "social": "Radical wealth disparity, performative progressivism, genuine revolution",
                    "psychological": "Ambition and emptiness, seeking and surrendering",
                    "hidden": "Power flows through unexpected channels, submission hides in success"
                },
                "temporal_layers": {
                    "Gold Rush echoes": "Fortune seekers still arrive seeking transformation",
                    "Sixties liberation": "Free love evolved into structured power exchange",
                    "Tech boom waves": "Each iteration brings new money to old games",
                    "Current crisis": "Inequality creating opportunities for alternative authority"
                }
            }
        }
    
    @staticmethod
    def get_notable_figures() -> List[Dict[str, Any]]:
        """Get important NPCs beyond the main cast"""
        return [
            {
                "name": "Victoria Chen",
                "role": "VC Partner / Rose Council Member",
                "public_persona": (
                    "Sequoia Capital's youngest female partner. MIT graduate, Forbes 30 Under 30. "
                    "Specializes in B2B SaaS and enterprise software. Known for her 'founder-friendly' "
                    "approach and uncanny ability to predict which CEOs will flame out. Drives a "
                    "sensible Tesla, lives in a modest Noe Valley Victorian."
                ),
                "hidden_nature": (
                    "Her 'founder coaching' includes very specific behavioral modifications. The "
                    "basement of her Victorian contains equipment not found at CrossFit. Three "
                    "portfolio company CEOs wear her collar under their hoodies. She identifies "
                    "predators in pitch meetings and ensures they never raise again."
                ),
                "power_indicators": [
                    "Unexplained influence over term sheets",
                    "Founders who radically change behavior after 'mentorship'",
                    "Her rose garden wins prizes no one remembers entering"
                ],
                "influence": "Can make or break unicorns with whispers"
            },
            {
                "name": "Judge Elizabeth Thornfield",
                "role": "Federal Judge / The Queen's Left Hand",
                "public_persona": (
                    "Appointed to the Northern District of California under Obama. Harvard Law, "
                    "former ACLU lawyer. Tough but fair reputation. Known for creative sentencing "
                    "in white collar crime. Married to a Stanford professor, two kids at Town School."
                ),
                "hidden_nature": (
                    "Certain cases find their way to her docket through careful manipulation. Her "
                    "'alternative sentencing' programs include options not in any legal textbook. "
                    "She maintains the legal infrastructure that keeps the Thorns' garden growing. "
                    "Her Thursday 'book club' shapes jurisprudence."
                ),
                "judicial_innovations": [
                    "Community service at specific nonprofits",
                    "Behavioral modification as rehabilitation",
                    "Sealed settlements with interesting terms"
                ],
                "network_role": "Legal protection and strategic prosecution"
            },
            {
                "name": "Isabella Montenegro",
                "role": "Gallery Owner / Thorn Bearer",
                "public_persona": (
                    "Runs Montenegro Modern in Jackson Square, specializing in contemporary "
                    "feminist art. Cuban exile family money. Hosts the hottest art openings "
                    "where tech meets culture. Patron of emerging artists, especially women. "
                    "Always photographed with interesting jewelry."
                ),
                "hidden_nature": (
                    "Her gallery's back room shows art that transforms viewers. She identifies "
                    "submissive tendencies in tech bros who buy aggressively masculine pieces. "
                    "Her opening night after-parties have waiting lists that money can't crack. "
                    "The Queen of Thorns may have been her protégé - or vice versa."
                ),
                "specialties": [
                    "Reading desire through aesthetic choices",
                    "Creating situations for revelation",
                    "Turning collectors into collections"
                ],
                "signature": "Gifts rose-themed art to those marked for transformation"
            },
            {
                "name": "Marcus Sterling",
                "role": "Former Tech CEO / Willing Servant",
                "public_persona": (
                    "Founded three successful startups, exited for nine figures total. Now "
                    "angel investor and 'executive coach.' Speaks at conferences about servant "
                    "leadership and conscious capitalism. Married with kids in Marin. Cycles "
                    "competitively, meditates publicly."
                ),
                "hidden_transformation": (
                    "After his MeToo near-miss, he encountered the Thorns. Now he serves "
                    "enthusiastically, funding what he once would have exploited. Wears his "
                    "submission like his Patagonia vest - carefully hidden but always present. "
                    "His coaching transforms predators into protectors."
                ),
                "current_service": [
                    "Funding safe houses through 'investment properties'",
                    "Recruiting other executives for transformation",
                    "Teaching power through powerlessness"
                ],
                "psychological_state": "Found peace in surrender, evangelizes subtly"
            },
            {
                "name": "Dr. Amara Johnson",
                "role": "Psychiatrist / Mind Gardener",
                "public_persona": (
                    "UCSF faculty, private practice in Pacific Heights. Specializes in executive "
                    "burnout and relationship counseling. Published researcher on power dynamics "
                    "in corporate settings. Speaks carefully, dresses impeccably, never rushes."
                ),
                "hidden_practice": (
                    "Her therapy includes methods not approved by the APA. She helps willing "
                    "clients discover their need to submit, guides unwilling ones toward "
                    "acceptance. The pharmaceutical combinations she prescribes alter more "
                    "than brain chemistry. Some say she trained the Queen of Thorns in "
                    "psychological control."
                ),
                "techniques": [
                    "Hypnotherapy that installs triggers",
                    "Medication regimens that increase suggestibility",
                    "Couples counseling that redistributes power"
                ],
                "client_base": "Those referred by roses, those who resist thorns"
            },
            {
                "name": "Supervisor Lisa Wong",
                "role": "City Government / Hidden Protector",
                "public_persona": (
                    "Second-generation San Franciscan representing the Sunset district. "
                    "Champion of affordable housing and small business. Former teacher, "
                    "union supporter. Known for asking tough questions at Board meetings. "
                    "Single mother, daughter at Lowell High."
                ),
                "underground_connection": (
                    "Her questions derail projects that would displace protected communities. "
                    "She knows which developments hide trafficking, which businesses launder "
                    "pain. The women's shelter in her district has a basement the city "
                    "doesn't know about. She's never met the Queen but follows her guidance."
                ),
                "political_maneuvers": [
                    "Strategic delays through environmental review",
                    "Funding streams redirected to protective organizations",
                    "Zoning changes that preserve underground infrastructure"
                ],
                "motivation": "Her sister disappeared at 16, returned transformed at 18"
            },
            {
                "name": "James Morrison III",
                "role": "Old Money Scion / Useful Fool",
                "public_persona": (
                    "Third generation Pacific Heights, inherited Grandfather's shipping fortune. "
                    "Sits on museum boards, funds the Opera. Known for his patronage of arts "
                    "and his terrible taste in women. Drinks at the Olympic Club, sails at "
                    "St. Francis Yacht Club."
                ),
                "exploitation": (
                    "Thinks he's found discrete dominatrixes, doesn't realize he's funding "
                    "a revolution. His need for maternal discipline makes him endlessly "
                    "manipulable. Every 'tribute' he pays frees another woman. His shipping "
                    "contacts unknowingly facilitate the Underground Railroad."
                ),
                "psychological_profile": "Desperate for approval from powerful women",
                "current_status": "Blissfully unaware of his true role"
            },
            {
                "name": "Captain Maria Rodriguez",
                "role": "SFPD Special Victims / Pragmatic Ally",
                "public_persona": (
                    "Twenty-year veteran, first Latina captain in SVU. Tough reputation, "
                    "high clearance rate. Divorced, kids grown. Known for working late, "
                    "taking cases personally. Speaks at community meetings about safety."
                ),
                "understanding": (
                    "She's seen enough to know some justice happens outside courtrooms. "
                    "When trafficking victims disappear from witness protection into better "
                    "protection, she doesn't investigate too hard. The rose on her desk was "
                    "a gift from a grateful mother whose daughter she 'couldn't find.'"
                ),
                "operational_philosophy": [
                    "Some cases solve themselves",
                    "Not all missing persons want to be found",
                    "Alternative justice beats no justice"
                ],
                "boundaries": "Won't ignore bodies, won't protect predators"
            },
            {
                "name": "Lily Chen (The Gardener)",
                "role": "Café Owner / Network Hub",
                "public_persona": (
                    "Runs The Rose Garden Café on Valencia. Berkeley grad, former Google UX "
                    "designer who 'followed her passion.' Makes legendary lavender lattes. "
                    "Knows everyone's name and coffee order. Hosts poetry nights and book clubs."
                ),
                "true_function": (
                    "Evaluates potential roses and identifies thorns needing attention. Her "
                    "café is a soft entry point to the harder truths. The book clubs read "
                    "between different lines. She's recruited more dominants than any other "
                    "gardener. May be the Queen herself, or her daughter, or her camouflage."
                ),
                "recruitment_methods": [
                    "Observing power dynamics in casual interactions",
                    "Coffee conversations that probe deeper",
                    "Invitations to increasingly intimate gatherings"
                ],
                "specialty": "Identifying latent dominance in unassuming women"
            }
        ]
    
    @staticmethod
    def get_underground_infrastructure() -> List[Dict[str, Any]]:
        """The hidden systems that make the shadow world function"""
        return [
            {
                "name": "The Transformation Pipeline",
                "type": "behavioral_modification_system",
                "description": (
                    "From identification to transformation, a careful process. Tech bros "
                    "who harass become advocates for women in tech. Traffickers become "
                    "funding sources for safe houses. The unwilling become willing through "
                    "methods psychiatry doesn't acknowledge."
                ),
                "stages": [
                    "Identification via network intelligence",
                    "Approach through seemingly random encounters",
                    "Pressure via revealed vulnerabilities",
                    "Breaking through targeted experiences",
                    "Rebuilding with new parameters",
                    "Release with permanent modifications"
                ],
                "infrastructure": [
                    "Safe houses doubling as dungeons",
                    "Therapists with flexible ethics",
                    "Lawyers who craft special agreements",
                    "Doctors who don't document everything"
                ],
                "success_rate": "87% permanent transformation",
                "failures": "Disappear to other cities with warnings"
            },
            {
                "name": "The Financial Web",
                "type": "economic_control_system",
                "description": (
                    "Money flows through channels the IRS doesn't track. Consulting fees "
                    "that buy silence, venture investments that ensure cooperation, nonprofit "
                    "donations that fund underground operations. Cryptocurrency meets "
                    "old-fashioned blackmail."
                ),
                "mechanisms": [
                    "Shell companies registered to roses",
                    "Cryptocurrency wallets with poetic passwords",
                    "Art sales that aren't about art",
                    "Consulting contracts with unique deliverables"
                ],
                "scale": "$50-100 million annually",
                "uses": [
                    "Safe house operations",
                    "Identity creation for the rescued",
                    "Legal defense funds",
                    "Transformation infrastructure"
                ]
            },
            {
                "name": "The Information Garden",
                "type": "intelligence_network",
                "description": (
                    "Every rose has eyes, every thorn has ears. Administrative assistants "
                    "who read emails, house cleaners who photograph documents, bartenders "
                    "who remember confessions. The network sees all and forgets nothing."
                ),
                "collection_methods": [
                    "Service workers with access",
                    "Converted executives sharing intel",
                    "Therapy session revelations",
                    "Pillow talk recordings"
                ],
                "storage": "Distributed, encrypted, backed up offshore",
                "usage": "Leverage, protection, preemptive strikes",
                "cardinal_rule": "Information is power, but timing is everything"
            }
        ]
