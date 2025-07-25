# story_templates/moth/lore/religion.py
"""
Religious and spiritual systems for SF Bay Area - surface normalcy with hidden depths
"""

from typing import Dict, Any, List

class SFReligionLore:
    """Religious and spiritual lore for SF Bay Area"""
    
    @staticmethod
    def get_religious_institutions() -> List[Dict[str, Any]]:
        """Get religious organizations in SF Bay Area"""
        return [
            {
                "name": "Grace Cathedral",
                "type": "episcopal_church",
                "founded": "1928 (current building)",
                "description": (
                    "Neo-Gothic cathedral atop Nob Hill, famous for its labyrinths, "
                    "AIDS memorial chapel, and progressive theology. Tourists come for "
                    "the architecture and evening concerts. The outdoor labyrinth is "
                    "walked by seekers at all hours. But certain evening services aren't "
                    "listed in the bulletin. The indoor labyrinth has patterns within "
                    "patterns for those who know the deeper walk."
                ),
                "leadership": "Bishop Sarah Williams - Oxford-educated, quietly radical",
                "public_practices": [
                    "Sunday Eucharist",
                    "Yoga on the labyrinth",
                    "Interfaith dialogue",
                    "Social justice advocacy"
                ],
                "hidden_practices": [
                    "Thursday midnight 'Contemplative Services' - by invitation",
                    "Confession booths that hear more than sins",
                    "Sanctuary that extends beyond spiritual",
                    "Certain clergy understand power exchange as sacred"
                ],
                "location": "1100 California Street, Nob Hill",
                "connections": ["City elite", "Progressive coalitions", "Hidden sanctuaries"],
                "symbols": ["Labyrinth walked in specific patterns", "Rose windows with thorns"],
                "secret_history": (
                    "During the 1989 earthquake, became organizing center for women's "
                    "protection networks. The Bishop then, Margaret Thorn, established "
                    "protocols still followed today."
                )
            },
            {
                "name": "The Zen Center San Francisco",
                "type": "soto_zen_temple",
                "founded": "1962",
                "description": (
                    "Page Street temple in Hayes Valley, offering meditation, retreats, "
                    "and famously good vegetarian food at Greens Restaurant. Tech workers "
                    "seek mindfulness between funding rounds. But the Teacher Training "
                    "Program includes lessons not found in any sutra. Advanced practitioners "
                    "learn that surrender and dominance are both paths to enlightenment."
                ),
                "leadership": "Abbess Kiku Thornfield - coincidental name, she claims",
                "public_practices": [
                    "Daily zazen meditation",
                    "Beginner's instruction",
                    "Work practice",
                    "Sesshin retreats"
                ],
                "hidden_practices": [
                    "Power exchange as spiritual practice",
                    "Domination and submission as teaching tools",
                    "The 'Thorn Lineage' - unofficial transmission",
                    "Silent retreats where more than speech is surrendered"
                ],
                "location": "300 Page Street",
                "special_programs": {
                    "Public": "Introduction to Zen Practice",
                    "Selective": "Advanced Koan Study (psychological domination)",
                    "Secret": "The Rose Sangha - female practitioners only"
                },
                "connections": ["Tech elite seeking meaning", "Artists and intellectuals"],
                "transformation_offered": "Ego death through various means"
            },
            {
                "name": "Glide Memorial Church",
                "type": "united_methodist",
                "founded": "1929",
                "description": (
                    "Radical inclusion church in the Tenderloin, famous for helping "
                    "the homeless, fighting injustice, and Sunday celebrations that "
                    "feel more like concerts. No questions asked, ever. This policy "
                    "extends further than most realize. Some who enter broken leave "
                    "whole - or different. Certain volunteers specialize in specific "
                    "types of healing."
                ),
                "leadership": "Rev. Dr. Marvin White - poet, prophet, protector",
                "public_practices": [
                    "Sunday Celebrations",
                    "Daily free meals", 
                    "Recovery programs",
                    "Social services"
                ],
                "hidden_practices": [
                    "Underground railroad for those escaping abuse",
                    "Identity reconstruction services",
                    "Alternative justice mediation",
                    "Healing through power reclamation"
                ],
                "location": "330 Ellis Street",
                "special_services": {
                    "Walk-in": "Anyone welcome for meals and services",
                    "Referred": "Deeper healing for those who need it",
                    "Protected": "Complete transformation possible"
                },
                "volunteer_network": "Some volunteers have very specific skills",
                "motto": "'Unconditionally Loving' has many interpretations"
            },
            {
                "name": "St. Dominic's Catholic Church",
                "type": "roman_catholic",
                "founded": "1873",
                "description": (
                    "Traditional Gothic church in Pacific Heights, serving old San Francisco "
                    "families. Beautiful rose windows, excellent choir, conservative theology "
                    "from the pulpit. But certain confessors hear with different ears. The "
                    "Rosary Society does more than pray. Some penances granted are quite... "
                    "specific. The basement hosts 'prayer groups' with interesting dynamics."
                ),
                "leadership": "Msgr. Xavier Bancroft - knows more than he absolves",
                "public_practices": [
                    "Traditional Latin Mass",
                    "Daily confessions",
                    "Rosary groups",
                    "Catholic school"
                ],
                "hidden_elements": [
                    "Confessors who understand power and submission",
                    "Penances that involve specific submissions",
                    "Rosary Society with thorns beneath petals",
                    "School discipline that shapes particular appetites"
                ],
                "location": "2390 Bush Street",
                "secret_orders": {
                    "Sisters of Perpetual Suffering": "Not what it seems",
                    "Brothers of Sacred Submission": "Serve in specific ways"
                },
                "historic_note": "The '60s reforms never quite took here",
                "connections": ["Old money families", "Certain judges", "Private schools"]
            },
            {
                "name": "Women's Sacred Circle",
                "type": "goddess_spirituality",
                "founded": "1979",
                "description": (
                    "Feminist spirituality group meeting in various locations - bookstores, "
                    "private homes, outdoor spaces. Public moon circles, seasonal celebrations, "
                    "goddess workshops. Reclaiming women's power through ritual. But some "
                    "circles go deeper than candles and crystals. Ancient practices of "
                    "domination as divine feminine right. The Queen of Thorns is honored "
                    "in ways outsiders wouldn't recognize."
                ),
                "leadership": "Rotating priestesses - some more influential than others",
                "public_practices": [
                    "New moon gatherings",
                    "Seasonal celebrations",
                    "Goddess meditation",
                    "Feminine empowerment workshops"
                ],
                "deeper_practices": [
                    "Power reclamation rituals",
                    "Sacred domination ceremonies",
                    "Thorn magic for protection",
                    "Binding and loosing that's quite literal"
                ],
                "meeting_locations": [
                    "Book Passage Marin - public circles",
                    "Private Noe Valley homes - invitation only",
                    "Mount Tam full moons - advanced practitioners",
                    "Secret rose garden locations - innermost mysteries"
                ],
                "initiation_levels": {
                    "Seeker": "Public workshops and circles",
                    "Initiate": "Private rituals begin",
                    "Priestess": "Learn to wield divine feminine power",
                    "Thorn Bearer": "Serve the Queen directly"
                },
                "tools_and_symbols": [
                    "Rose quartz and obsidian",
                    "Thorned roses on altars",
                    "Specific knots in ritual cords",
                    "Dominance mudras during meditation"
                ]
            },
            {
                "name": "First Unitarian Universalist",
                "type": "unitarian_universalist",
                "founded": "1850",
                "description": (
                    "Progressive congregation in downtown SF. Social justice, LGBTQ+ "
                    "affirming, interfaith dialogue. Hosts everything from Buddhist "
                    "meditation to pagan rituals. This radical openness provides perfect "
                    "cover. Certain study groups explore power dynamics as spiritual practice. "
                    "The Social Action Committee takes very direct action sometimes."
                ),
                "leadership": "Rev. Dr. Vanessa Thornwood - 'coincidentally' named",
                "public_practices": [
                    "Sunday services with diverse themes",
                    "Social justice work",
                    "Adult religious education",
                    "Community organizing"
                ],
                "specialized_groups": [
                    "Women and Power study group - goes beyond theory",
                    "Sacred Sexuality circle - very specific practices",
                    "Alternative Relationships support - hierarchy friendly",
                    "Direct Action Committee - some actions aren't public"
                ],
                "location": "1187 Franklin Street",
                "connections": ["Activist networks", "Alternative communities", "Hidden power brokers"],
                "library_secrets": "Certain books only available by request"
            },
            {
                "name": "Temple Shalom Rose",
                "type": "reform_judaism",
                "founded": "1953",
                "description": (
                    "Reform synagogue in Presidio Heights. Known for progressive values, "
                    "female rabbis, and beautiful rose garden (unusual for a synagogue). "
                    "The Sisterhood runs more than bake sales. Kabbalah study includes "
                    "mysteries of divine feminine power. Some prayers bind more than souls."
                ),
                "leadership": "Rabbi Sarah Rosenthorn - 'rabbinical family going way back'",
                "public_practices": [
                    "Shabbat services",
                    "Torah study",
                    "Tikkun olam (social justice)",
                    "Lifecycle events"
                ],
                "mystical_practices": [
                    "Women's Kabbalah circle - Shekinah as dominatrix",
                    "Lilith reclamation rituals",
                    "Binding prayers that literally bind",
                    "Rose garden meditations with specific intent"
                ],
                "location": "3200 California Street",
                "sisterhood_activities": {
                    "Public": "Fundraising and social events",
                    "Private": "Power cultivation techniques",
                    "Secret": "Enforcement of community standards"
                },
                "historic_note": "The rose garden was planted in 1989 by survivors",
                "connections": ["Legal community", "Medical professionals", "Old SF families"]
            },
            {
                "name": "Sacred BDSM Temple",
                "type": "alternative_spirituality",
                "founded": "2012",
                "description": (
                    "Officially registered church dedicated to BDSM as spiritual practice. "
                    "Meets in SoMa dungeon space. Completely legal, surprisingly open. But "
                    "this openness hides deeper currents. Public classes teach technique; "
                    "private ceremonies transform souls. The High Priestesses serve powers "
                    "most members don't know exist."
                ),
                "leadership": "High Priestess Lilith Rose - 'Obviously a scene name'",
                "public_practices": [
                    "Consent as sacrament workshops",
                    "Rope bondage as meditation",
                    "Power exchange rituals",
                    "Sacred sexuality ceremonies"
                ],
                "hidden_hierarchy": [
                    "Public Temple - Safe, Sane, Consensual",
                    "Inner Temple - Risk Aware Consensual",
                    "Hidden Temple - Where consent gets complex",
                    "Thorn Chamber - Service to the Queen herself"
                ],
                "location": "Folsom Street location - members only",
                "initiation_path": {
                    "Novice": "Public workshops and scenes",
                    "Dedicant": "Personal power dynamics explored",
                    "Ordained": "Can lead others in power exchange",
                    "Thorn Priestess": "Direct service to hidden powers"
                },
                "connections": ["Leather community", "Sex-positive activists", "The deeper network"],
                "theological_position": "Dominance and submission as paths to transcendence",
                "practices_taught": [
                    "Psychological domination techniques",
                    "Energy manipulation through pain",
                    "Submission as strength",
                    "Transformation through power exchange"
                ]
            }
        ]
    
    @staticmethod
    def get_spiritual_movements() -> List[Dict[str, Any]]:
        """Get spiritual movements and philosophies"""
        return [
            {
                "name": "Bay Area Sacred Sexuality",
                "movement_type": "neo_tantra",
                "description": (
                    "Fusion of tantra, BDSM, and New Age practices. Weekend workshops "
                    "in Marin County mansions. Teaches energy exchange through intimate "
                    "connection. But advanced workshops explore power dynamics that would "
                    "shock vanilla tantrikas. The teachers have interesting day jobs."
                ),
                "public_teaching": "Conscious sexuality and energy cultivation",
                "deeper_teaching": "Power exchange as energy circulation",
                "secret_teaching": "Domination as divine feminine birthright",
                "key_figures": [
                    "Dakini Rose - tantra teacher/corporate consultant",
                    "Shiva Michael - submissive teacher of male surrender",
                    "The Marin Circle - invitation only advanced group"
                ],
                "workshop_levels": {
                    "Introduction": "$500 weekend - basic energy work",
                    "Intermediate": "$2000 week - power dynamics introduced",
                    "Advanced": "Price negotiable - life transformation",
                    "Inner Circle": "Service, not payment"
                }
            },
            {
                "name": "Executive Mindfulness",
                "movement_type": "corporate_spirituality",
                "description": (
                    "Meditation and presence training for business leaders. Google, "
                    "Facebook, and Salesforce all have programs. Reduce stress, increase "
                    "focus, lead mindfully. But certain teachers include lessons on "
                    "psychological dominance as 'executive presence.' Some CEOs learn "
                    "to kneel as well as command."
                ),
                "public_benefits": "Stress reduction, leadership skills",
                "hidden_curriculum": "Power dynamics, psychological control",
                "secret_outcomes": "Executives who understand submission",
                "notable_teachers": [
                    "Victoria Chen - mindfulness coach with sharp thorns",
                    "Dr. Sarah Kim - neuroscience of power exchange",
                    "Anonymous facilitators for 'special needs'"
                ]
            },
            {
                "name": "Goddess Rising Movement",
                "movement_type": "feminist_spirituality",
                "description": (
                    "Reclaiming divine feminine power. Started in Berkeley, spread "
                    "throughout Bay Area. Public marches, private rituals. Empowerment "
                    "through embodiment. But some circles define power very specifically. "
                    "The Goddess they serve has thorns."
                ),
                "public_face": "Women's empowerment and spirituality",
                "private_practice": "Cultivation of dominant feminine energy",
                "secret_core": "Service to the Queen of Thorns as avatar",
                "practices": [
                    "Rose meditation (thorns included)",
                    "Power breath work",
                    "Domination as devotion",
                    "Submission training for allies"
                ]
            }
        ]
    
    @staticmethod
    def get_religious_practices() -> List[Dict[str, Any]]:
        """Get ritual practices across traditions"""
        return [
            {
                "name": "The Rose Meditation",
                "practice_type": "contemplative",
                "tradition": "Multiple (Zen, Christian, Goddess)",
                "public_description": (
                    "Meditation on the beauty and thorns of roses. Used in various "
                    "traditions as symbol of love's complexity."
                ),
                "deeper_practice": (
                    "Visualization of thorns as protective power. Advanced practitioners "
                    "feel phantom thorns growing from their aura. Some report ability "
                    "to psychically 'prick' those who threaten them."
                ),
                "secret_level": (
                    "Direct communion with the Queen of Thorns. Practitioners report "
                    "receiving guidance, protection, and assignments. Thorns manifest "
                    "as actual power to influence and control."
                ),
                "locations_taught": [
                    "Zen Center - 'Flower Sermon' workshops",
                    "Women's Sacred Circle - central practice",
                    "Even Grace Cathedral mentions roses often"
                ]
            },
            {
                "name": "Confession and Penance",
                "practice_type": "sacramental",
                "tradition": "Catholic (with variations)",
                "public_description": (
                    "Traditional sacrament of reconciliation. Confess sins, receive "
                    "absolution, perform penance."
                ),
                "modified_practice": (
                    "Certain confessors assign penances involving submission to "
                    "wronged parties. Some booths hear confessions of dominance "
                    "desires. Absolution comes through power exchange."
                ),
                "underground_version": (
                    "Mobile confessionals at BDSM events. Psychological domination "
                    "as penance. Some priests/priestesses serve the Queen directly, "
                    "gathering intelligence through confession."
                ),
                "known_practitioners": [
                    "Father Marcus at St. Dominic's",
                    "The touring 'Confession Truck' at Folsom",
                    "Anonymous online confession with real consequences"
                ]
            },
            {
                "name": "Labyrinth Walking",
                "practice_type": "meditative_movement",
                "tradition": "Interfaith",
                "public_description": (
                    "Walking meditation on circular path. Grace Cathedral famous "
                    "for indoor and outdoor labyrinths. Releases stress, provides clarity."
                ),
                "advanced_practice": (
                    "Specific patterns within the walk create different effects. "
                    "Clockwise for building power, counter for release. Certain "
                    "walkers seem to disappear briefly at the center."
                ),
                "initiated_practice": (
                    "The labyrinth as power mandala. Walking while holding dominant "
                    "or submissive energy. Meeting others at center for energy exchange. "
                    "Night walks when the veils are thin."
                ),
                "secret_knowledge": "The Grace Cathedral labyrinth has 13 hidden thorns"
            },
            {
                "name": "Sacred Rope Ceremony",
                "practice_type": "embodied_ritual",
                "tradition": "BDSM Temple/Japanese influenced",
                "public_description": (
                    "Rope bondage as meditation and art. Public classes teach safety "
                    "and basic ties. Beautiful, consensual, athletic."
                ),
                "spiritual_dimension": (
                    "Rope as energetic binding. The ties create sacred geometry on "
                    "the body. Surrender and control as prayer. Advanced practitioners "
                    "report out-of-body experiences."
                ),
                "mystery_teaching": (
                    "Certain patterns known only to Thorn initiates. These ties "
                    "don't just bind body but will. Used in initiation ceremonies "
                    "and power transfer rituals. The Queen knows ties that bind fate."
                ),
                "learning_path": [
                    "Public classes at Twisted Monk",
                    "Private teaching with selected riggers",
                    "Invitation to Temple ceremonies",
                    "Initiation into Thorn patterns"
                ]
            },
            {
                "name": "Power Exchange Communion",
                "practice_type": "sacramental",
                "tradition": "BDSM Temple",
                "public_description": (
                    "Ritual sharing of power between dominant and submissive. "
                    "Witnessed by community. Similar to handfasting but for power dynamics."
                ),
                "ritual_elements": [
                    "Verbal declaration of dynamic",
                    "Symbolic binding (collar, rope, marks)",
                    "Energy exchange through touch/pain",
                    "Community blessing"
                ],
                "advanced_form": (
                    "Permanent power transfer ceremonies. The dominant gains actual "
                    "influence over submissive's life. Sometimes careers advance or "
                    "fail based on these bonds. The Queen presides over the deepest ones."
                ),
                "theological_meaning": "Power freely given is power multiplied"
            },
            {
                "name": "Thorn Initiation",
                "practice_type": "mystery_initiation",
                "tradition": "Hidden/multiple",
                "public_knowledge": "Rumors only - no public form",
                "actual_practice": (
                    "Multi-stage initiation into service of the Queen of Thorns. "
                    "Begins with rose meditation, progresses through tests of "
                    "dominance or submission. Final stage involves direct contact "
                    "with Queen or her avatars."
                ),
                "stages": [
                    "Recognition - seeing power dynamics clearly",
                    "Cultivation - developing personal power/submission",
                    "Testing - proving commitment through action",
                    "Binding - permanent connection to the network",
                    "Service - active role in the hidden structure"
                ],
                "outcomes": "Initiates gain protection, purpose, and terrible knowledge"
            }
        ]
    
    @staticmethod
    def get_sacred_texts() -> List[Dict[str, Any]]:
        """Get religious and spiritual writings"""
        return [
            {
                "name": "The Rose Gospel",
                "text_type": "modern_scripture",
                "authorship": "Anonymous - 'channeled by many'",
                "description": (
                    "Circulated in handwritten copies and encrypted PDFs. Appears to be "
                    "feminist theology about reclaiming power. Actually detailed manual "
                    "for psychological and energetic domination. Each chapter deepens "
                    "understanding of power exchange as spiritual path."
                ),
                "key_teachings": [
                    "The Divine Feminine has thorns for a reason",
                    "Power given freely multiplies; power taken withers",
                    "Submission to the worthy is strength",
                    "The Queen tends all gardens"
                ],
                "circulation": "Passed hand to hand, never sold",
                "danger_level": "Changes readers fundamentally",
                "known_copies": [
                    "Grace Cathedral rare books room",
                    "Women's Sacred Circle library",
                    "Private collections"
                ],
                "sample_passage": (
                    "'She who would wear the crown of thorns must first learn to "
                    "draw blood with kindness, to bind with silk, to command with "
                    "silence. Power is not taken but recognized, not imposed but invoked.'"
                )
            },
            {
                "name": "Meditations on Submission",
                "text_type": "contemplative_guide",
                "authorship": "Various submissives, compiled by 'Sister Thorn'",
                "description": (
                    "Looks like standard meditation guide. Actually profound exploration "
                    "of submission as spiritual practice. Used in convents, zen centers, "
                    "and dungeons. Each meditation deepens capacity for surrender."
                ),
                "structure": [
                    "Morning Offerings - preparing for day's submissions",
                    "Noon Reflections - finding divine in service",
                    "Evening Gratitudes - honoring dominant forces",
                    "Night Prayers - submitting even in sleep"
                ],
                "hidden_chapters": "Only revealed to proven submissives",
                "influence": "Creates deeply devoted servants"
            },
            {
                "name": "The Thorn Codex",
                "text_type": "secret_manual",
                "authorship": "The Queens of Thorns through generations",
                "description": (
                    "Master manual of feminine domination disguised as gardening guide. "
                    "Details psychological techniques, energy manipulation, network building. "
                    "Each Queen adds chapters. Current edition includes digital age updates."
                ),
                "access": "Only confirmed Thorn Bearers",
                "contents": [
                    "Recognizing natural submissives",
                    "Cultivation techniques for power",
                    "Network security protocols",
                    "Advanced psychological control",
                    "Intergenerational power transfer"
                ],
                "format": "Living document, constantly updated",
                "security": "Self-destructs if unauthorized access attempted"
            },
            {
                "name": "Corporate Tantras",
                "text_type": "business_spirituality",
                "authorship": "Victoria Chen (with anonymous contributors)",
                "description": (
                    "Bestselling business book about 'mindful leadership.' Actually "
                    "teaches domination techniques disguised as management. Used in "
                    "MBA programs and executive coaching. Readers report dramatic "
                    "increases in influence."
                ),
                "public_chapters": [
                    "Presence as Power",
                    "The Art of Beneficial Surrender",
                    "Creating Willing Compliance",
                    "Leadership Through Liberation"
                ],
                "hidden_meanings": "Initiates read completely different book",
                "distribution": "Amazon bestseller with secret supplementary materials"
            }
        ]
    
    @staticmethod
    def get_religious_orders() -> List[Dict[str, Any]]:
        """Get organized religious and spiritual groups"""
        return [
            {
                "name": "Sisters of Sacred Submission",
                "order_type": "contemplative",
                "tradition": "Catholic-influenced",
                "public_description": (
                    "Small order of religious women dedicated to service. Run halfway "
                    "houses and counseling centers. Known for taking difficult cases."
                ),
                "true_nature": (
                    "Train submissives in sacred service while cultivating dominant "
                    "nuns who understand power profoundly. Mother Superior answers "
                    "to authorities beyond the Vatican."
                ),
                "location": "Convent in Noe Valley",
                "practices": [
                    "Liturgy of the Hours (modified)",
                    "Confession as intelligence gathering",
                    "Penance as power exchange",
                    "Obedience with deeper meanings"
                ],
                "hierarchy": [
                    "Postulants - learning submission",
                    "Novices - choosing their path",
                    "Sisters - serving specific roles",
                    "Mother Superior - wielding accumulated power"
                ],
                "services_provided": [
                    "Counseling for 'difficult' women",
                    "Halfway house with strict rules",
                    "Discrete problem solving"
                ],
                "connections": "Report to the Thorn Garden"
            },
            {
                "name": "Order of the Eastern Rose",
                "order_type": "martial_contemplative",
                "tradition": "Buddhist-influenced",
                "public_description": (
                    "Zen practitioners who include martial arts in their practice. "
                    "Teach self-defense to women, meditation to all."
                ),
                "deeper_reality": (
                    "Warrior monks/nuns who protect the network physically. Trained "
                    "in combat, intimidation, and extraction. Meditation includes "
                    "visualization of dominating opponents."
                ),
                "training_location": "Dojo in Mission District",
                "ranks": [
                    "White Thorn - beginners",
                    "Red Thorn - proven fighters",
                    "Black Thorn - teachers and protectors",
                    "Golden Thorn - strategic command"
                ],
                "special_skills": [
                    "Pressure points for compliance",
                    "Psychological warfare",
                    "Protective surveillance",
                    "Silent elimination"
                ],
                "motto": "Soft as petals, sharp as thorns"
            },
            {
                "name": "The Rose Gardeners",
                "order_type": "secular_religious",
                "tradition": "Interfaith",
                "public_description": (
                    "Volunteer group maintaining rose gardens throughout Bay Area. "
                    "Mostly retired women with gardening passion."
                ),
                "actual_function": (
                    "Intelligence network and communication system. Each garden "
                    "is a node, each gardener a watcher. They see everything, "
                    "prune what needs pruning."
                ),
                "organization": [
                    "Master Gardeners - regional coordinators",
                    "Senior Gardeners - neighborhood watchers",  
                    "Gardeners - eyes and ears",
                    "Apprentices - being evaluated"
                ],
                "communication_methods": [
                    "Which roses are pruned/planted",
                    "Garden layout changes",
                    "Color combinations",
                    "Timing of maintenance"
                ],
                "special_activities": "Sometimes bodies fertilize roses"
            },
            {
                "name": "Brothers of Blessed Servitude",
                "order_type": "service_order",
                "tradition": "Mystical Christian",
                "public_description": (
                    "Small order of men dedicated to serving others. Run soup "
                    "kitchens, homeless services. Very humble."
                ),
                "hidden_reality": (
                    "Male submissives trained to serve dominant women while "
                    "appearing simply charitable. Provide support services to "
                    "the network. Their vow of obedience has specific meaning."
                ),
                "location": "House in Tenderloin",
                "vows": [
                    "Poverty (dependence on dominants)",
                    "Chastity (controlled sexuality)",
                    "Obedience (to female authority)",
                    "Service (in all forms required)"
                ],
                "brothers_include": [
                    "Former executives learning humility",
                    "Natural submissives finding purpose",
                    "Men paying penance",
                    "True believers in female divinity"
                ]
            },
            {
                "name": "Temple Guardians",
                "order_type": "protective_service",
                "tradition": "BDSM Temple",
                "public_description": (
                    "Volunteer security for BDSM events. Check IDs, maintain "
                    "safe space, handle problems."
                ),
                "deeper_function": (
                    "Elite guards who protect high-level gatherings. Trained in "
                    "combat, discretion, and reading power dynamics. Know when "
                    "to intervene and when to allow. Report to Thorn leadership."
                ),
                "selection_criteria": [
                    "Physical capability",
                    "Psychological stability",
                    "Power dynamic fluency",
                    "Absolute loyalty"
                ],
                "training_includes": [
                    "Threat assessment",
                    "Discrete intervention",
                    "Information gathering",
                    "Emergency protocols"
                ],
                "chain_of_command": "Direct to Queen's security chief"
            }
        ]
    
    @staticmethod
    def get_pilgrimage_sites() -> List[Dict[str, Any]]:
        """Get places of spiritual significance"""
        return [
            {
                "name": "The Original Rose Garden",
                "site_type": "historical_spiritual",
                "location": "Golden Gate Park (specific section known to initiates)",
                "public_significance": "Beautiful public garden",
                "hidden_significance": (
                    "Where the first Queen of Thorns held court in 1960s. Certain "
                    "bushes descended from her original plantings. Initiates come "
                    "at specific times to meditate and receive guidance."
                ),
                "pilgrimage_practices": [
                    "Rose meditation at dawn",
                    "Leaving offerings at unmarked spots",
                    "Walking patterns among the beds",
                    "Silent vigils on significant dates"
                ],
                "reported_experiences": [
                    "Visions of past Queens",
                    "Spontaneous understanding of role",
                    "Thorns that don't wound the worthy",
                    "Roses blooming out of season for chosen"
                ]
            },
            {
                "name": "The Basement Temple",
                "site_type": "active_sacred",
                "location": "Unmarked building, SoMa district",
                "public_knowledge": "Urban legend only",
                "reality": (
                    "Deepest BDSM temple where serious initiations occur. Original "
                    "dungeon furniture from pre-gentrification leather scene. Walls "
                    "hold decades of power exchange energy."
                ),
                "access": "Invitation only, usually for initiation",
                "what_happens": [
                    "Deepest power exchanges",
                    "Initiation ceremonies",
                    "Direct Queen appearances",
                    "Transformational scenes"
                ],
                "energy_description": "Thick with accumulated surrender and dominance"
            },
            {
                "name": "Mount Tam Moon Circle",
                "site_type": "natural_sacred",
                "location": "Specific grove on Mount Tamalpais",
                "public_use": "Popular hiking area",
                "sacred_use": (
                    "Full moon gatherings of women's mystery groups. Where "
                    "dominant feminine energy is raised and directed. Natural "
                    "amphitheater hidden from trails."
                ),
                "practices": [
                    "Skyclad rituals (weather permitting)",
                    "Power raising through dance",
                    "Thorn magic under stars",
                    "Initiation of new Priestesses"
                ],
                "best_times": "Full moons, especially in spring",
                "guardian_spirit": "The Mountain herself, they say"
            },
            {
                "name": "The Switchback Stairs",
                "site_type": "urban_pilgrimage",
                "location": "Filbert Street Steps",
                "public_face": "Tourist attraction with gardens",
                "pilgrimage_meaning": (
                    "Ascending represents submission, descending domination. "
                    "Different staircases for different workings. Roses planted "
                    "at significant points hold messages."
                ),
                "practice": "Walk with intention, leave offerings in gardens",
                "reported_effects": "Clarity about one's true nature"
            }
        ]
    
    @staticmethod
    def get_theological_concepts() -> List[Dict[str, Any]]:
        """Get religious/spiritual concepts unique to this setting"""
        return [
            {
                "concept": "Sacred Feminine Dominance",
                "traditions": ["Goddess spirituality", "Modified Christianity", "BDSM Temple"],
                "description": (
                    "The Divine Feminine isn't just nurturing mother but terrible "
                    "queen. Power to create includes power to destroy. True female "
                    "divinity includes dominance as birthright."
                ),
                "scriptural_support": [
                    "Kali dancing on Shiva",
                    "Lilith refusing submission",
                    "Mary's Magnificat as power anthem",
                    "Inanna ruling the underworld"
                ],
                "practical_application": (
                    "Women encouraged to embrace dominant aspects. Submission to "
                    "feminine considered holy. Male dominance seen as aberration."
                ),
                "levels_of_understanding": [
                    "Feminist empowerment (public)",
                    "Female supremacy (private)",
                    "Literal divine mandate (initiated)"
                ]
            },
            {
                "concept": "Submission as Strength",
                "traditions": ["Zen Buddhism", "Mystical Christianity", "BDSM Temple"],
                "description": (
                    "True strength comes from conscious surrender to worthy authority. "
                    "Ego death through submission leads to rebirth. Power flows to "
                    "those who can kneel."
                ),
                "paradox": "The submissive controls through surrender",
                "practices": [
                    "Meditation on knees",
                    "Service as prayer",
                    "Pain as teacher",
                    "Obedience as liberation"
                ],
                "transformation_offered": "Ego dissolution and reconstruction"
            },
            {
                "concept": "The Thorn Path",
                "traditions": ["Unique to Bay Area synthesis"],
                "description": (
                    "Spiritual evolution through embracing both beauty and pain, "
                    "dominance and submission. The rose has thorns for protection "
                    "and selection. Not all can grasp the stem."
                ),
                "stages": [
                    "Pricking - first awareness of power dynamics",
                    "Bleeding - sacrifice and transformation",
                    "Scarring - permanent change",
                    "Blooming - full power realized"
                ],
                "ultimate_goal": "Service to the Queen/becoming a Queen"
            },
            {
                "concept": "Power Exchange as Sacrament",
                "traditions": ["BDSM Temple", "Modified across traditions"],
                "description": (
                    "The giving and taking of power as holy act. More sacred than "
                    "communion, more binding than marriage. Creates actual energetic "
                    "bonds between participants."
                ),
                "theology": [
                    "Power is divine energy",
                    "Exchange creates circulation",
                    "Stagnant power corrupts",
                    "Flowing power transforms"
                ],
                "ritual_requirements": [
                    "Full consent",
                    "Clear intention", 
                    "Witnessed by community",
                    "Sealed with pain/pleasure"
                ]
            }
        ]
    
    @staticmethod
    def get_religious_conflicts() -> List[Dict[str, Any]]:
        """Get theological and organizational disputes"""
        return [
            {
                "conflict": "The Visibility Question",
                "parties": ["Public temples", "Hidden practitioners"],
                "description": (
                    "Should sacred BDSM practices be public or hidden? Some say "
                    "visibility brings acceptance and safety. Others insist mystery "
                    "and secrecy preserve power. The Queen remains silent."
                ),
                "arguments": {
                    "Public faction": "Legitimacy protects practitioners",
                    "Hidden faction": "Exposure dilutes power",
                    "Middle path": "Public face, private depths"
                },
                "current_state": "Uneasy coexistence",
                "stakes": "Future of the movement"
            },
            {
                "conflict": "Male Dominants Question",
                "parties": ["Female supremacists", "Gender egalitarians"],
                "description": (
                    "Can men be truly dominant in sacred context? Hardliners say no - "
                    "male dominance is always profane. Moderates accept male dominants "
                    "who acknowledge female spiritual superiority."
                ),
                "theological_positions": [
                    "Female only - divine feminine monopoly",
                    "Female primary - men serve even as dominants",
                    "Equal but different - separate spheres",
                    "Individual calling - gender irrelevant"
                ],
                "Queen's_position": "Strategically ambiguous",
                "practical_result": "Male dominants marginalized in Temple"
            },
            {
                "conflict": "The Succession Crisis",
                "parties": ["Various factions anticipating Queen's retirement"],
                "description": (
                    "Who becomes next Queen of Thorns? Is it inherited, earned, "
                    "or divinely appointed? Different groups prepare candidates. "
                    "Some say the Queen is immortal, others that she's already "
                    "been replaced multiple times."
                ),
                "proposed_solutions": [
                    "Democratic election by initiated",
                    "Trial by ordeal",
                    "Divine selection through signs",
                    "Current Queen chooses",
                    "There have always been multiple Queens"
                ],
                "underground_tension": "Increasing as Queen ages (or doesn't)"
            },
            {
                "conflict": "Sacred vs Commercial",
                "parties": ["Pro dommes", "Lifestyle practitioners"],
                "description": (
                    "Is charging for domination services sacred or profane? Some "
                    "see it as ministry deserving support. Others insist money "
                    "corrupts the power exchange. Temple remains neutral officially."
                ),
                "practical_divisions": [
                    "Pro dommes fund many Temple activities",
                    "Lifestyle dommes hold spiritual authority",
                    "Clients vs congregants confusion",
                    "Money as energy exchange debate"
                ],
                "Queen's_network": "Uses both but keeps separate"
            }
        ]
    
    @staticmethod
    def get_mystical_experiences() -> List[Dict[str, Any]]:
        """Get reported spiritual phenomena"""
        return [
            {
                "phenomenon": "Thorn Stigmata",
                "description": (
                    "Devoted practitioners develop thorn-like marks on palms, "
                    "feet, or forehead. Appear during intense power exchange "
                    "or deep meditation. Medical explanation lacking."
                ),
                "occurrence_rate": "Rare but documented",
                "affected_groups": ["Deeply committed dominants", "Sacred submissives"],
                "interpretations": [
                    "Mark of divine favor",
                    "Psychosomatic manifestation",
                    "Energy work made physical",
                    "Queen's blessing/curse"
                ],
                "effects": "Increased sensitivity to power dynamics"
            },
            {
                "phenomenon": "Shared Subspace",
                "description": (
                    "During intense scenes, multiple participants report shared "
                    "consciousness. See through each other's eyes, feel merged "
                    "sensations. Sometimes extends to non-participants nearby."
                ),
                "triggers": [
                    "Group scenes with strong energy",
                    "Initiation ceremonies",
                    "Deep power exchange",
                    "Presence of advanced practitioners"
                ],
                "duration": "Minutes to hours",
                "after_effects": "Permanent psychic connection reported"
            },
            {
                "phenomenon": "The Queen's Dreams",
                "description": (
                    "Initiates report shared dreams featuring a woman with "
                    "changing faces but consistent rose/thorn imagery. Often "
                    "receive instructions or warnings. Dreams increase near "
                    "significant events."
                ),
                "common_elements": [
                    "Garden settings",
                    "Thorns that guide or restrict",
                    "Multiple female faces/forms",
                    "Specific instructions"
                ],
                "verification": "Instructions often prove prescient",
                "theory": "Collective unconscious or actual communication?"
            },
            {
                "phenomenon": "Power Drunk",
                "description": (
                    "Dominants report intoxication-like effects from intense "
                    "power exchange. Enhanced perception, time dilation, "
                    "synesthesia. Some become literally addicted."
                ),
                "physiological_markers": [
                    "Dilated pupils",
                    "Elevated heart rate",
                    "Endorphin surge",
                    "Sometimes literal pheromone production"
                ],
                "danger": "Can lead to abuse without proper training",
                "management": "Temple teaches moderation and grounding"
            }
        ]
