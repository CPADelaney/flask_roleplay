# story_templates/moth/lore/religion.py
"""
Religious and spiritual systems for SF Bay Area underground
"""

from typing import Dict, Any, List

class SFReligionLore:
    """Religious and spiritual lore for SF Bay Area"""
    
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
    def get_district_religious_distribution() -> List[Dict[str, Any]]:
        """Get religious distribution by district"""
        return [
            {
                "nation_id": "mission_underground",
                "state_religion": False,
                "primary_pantheon_id": "doctrine_consensual_exchange",
                "pantheon_distribution": {
                    "doctrine_consensual_exchange": 60,
                    "church_wounded_healer": 30,
                    "traditional_catholic": 5,
                    "other": 5
                },
                "religiosity_level": 8,
                "religious_tolerance": 9,
                "religious_leadership": "The Moth Queen as High Priestess (unofficial)",
                "religious_laws": {
                    "consent_absolute": "Breaking consent is highest sin",
                    "sanctuary_sacred": "Safe houses are holy ground",
                    "moth_protection": "Those marked by moths are protected"
                },
                "religious_holidays": [
                    "Folsom Street Fair - High Holy Days",
                    "Night of Moths - April 18",
                    "Longest Night - Winter Solstice ceremonies"
                ],
                "religious_conflicts": [
                    "Old Guard vs New School practices",
                    "Sacred vs commercial sexuality"
                ],
                "religious_minorities": [
                    "Catholics who blend traditions",
                    "Tech pagans seeking transcendence"
                ]
            },
            {
                "nation_id": "financial_district",
                "state_religion": False,
                "primary_pantheon_id": None,
                "pantheon_distribution": {
                    "secular": 40,
                    "church_personal_revelation": 30,
                    "doctrine_consensual_exchange": 20,
                    "traditional": 10
                },
                "religiosity_level": 4,
                "religious_tolerance": 6,
                "religious_leadership": "Money itself as deity, pleasure as prayer",
                "religious_laws": {
                    "discretion_absolute": "Privacy is sacred",
                    "power_inversion": "Night inverts day hierarchies"
                },
                "religious_holidays": [
                    "Quarterly earnings (joke but not)",
                    "Private festival dates"
                ],
                "religious_conflicts": [
                    "Guilt vs liberation",
                    "Public morality vs private desire"
                ],
                "religious_minorities": [
                    "True believers seeking meaning in sensation"
                ]
            }
        ]
    
    @staticmethod
    def get_regional_religious_practices() -> List[Dict[str, Any]]:
        """Get regional variations of religious practices"""
        return [
            {
                "nation_id": "mission_underground",
                "practice_id": "collar_ceremony",
                "regional_variation": (
                    "Mission collar ceremonies include Day of Dead imagery. "
                    "Collars often feature marigolds or sugar skulls alongside "
                    "traditional moths. The Queen blesses with copal incense."
                ),
                "importance": 10,
                "frequency": "Monthly on new moons",
                "local_additions": "Mezcal toast, ancestor acknowledgment",
                "gender_differences": "Women collar anyone, men only collar men"
            },
            {
                "nation_id": "soma_shadow",
                "practice_id": "collar_ceremony",
                "regional_variation": (
                    "SoMa ceremonies are more theatrical, often public. "
                    "Industrial aesthetic - metal collars, chain rituals. "
                    "The Queen appears via video screen if not present."
                ),
                "importance": 8,
                "frequency": "During major scene nights",
                "local_additions": "Leather family witnesses, bootblacking ritual",
                "gender_differences": "More egalitarian, focus on protocol over gender"
            },
            {
                "nation_id": "marina_safehouse",
                "practice_id": "moth_release",
                "regional_variation": (
                    "Releases happen at Marina Green overlooking the bay. "
                    "Each moth carries a small LED so they glow in the dark. "
                    "Survivors choose moth colors based on their journey."
                ),
                "importance": 10,
                "frequency": "Quarterly, plus special occasions",
                "local_additions": "Boat ceremony for those lost at sea",
                "gender_differences": "None - grief transcends all"
            }
        ]
