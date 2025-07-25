# story_templates/moth/lore/city.py
"""
Pre-generated lore for a modern Earth city setting
Creates a rich, believable urban environment with depth
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class ModernCityLorePresets:
    """Pre-generated lore content for modern city setting"""
    
    @staticmethod
    def get_city_foundation_lore() -> Dict[str, Any]:
        """Get foundation lore for the modern city"""
        return {
            "world_setting": {
                "name": "Shadow Bay Metropolitan Area",
                "type": "modern_earth_city",
                "population": "4.2 million",
                "established": "1847",
                "description": (
                    "A major port city on the eastern seaboard, Shadow Bay grew from a "
                    "trading post to a sprawling metropolis. Known for its stark contrasts - "
                    "gleaming financial districts shadow forgotten neighborhoods, and beneath "
                    "the legitimate city thrives an underground economy of desire and desperation."
                )
            },
            "districts": [
                {
                    "name": "The Financial District",
                    "type": "commercial",
                    "description": "Glass towers pierce the sky, home to banks and corporations",
                    "notable_features": ["Stock Exchange", "Corporate Plaza", "Luxury Hotels"],
                    "demographics": "Wealthy professionals, some who seek darker pleasures after hours"
                },
                {
                    "name": "Old Harbor",
                    "type": "historical",
                    "description": "The city's original port, now gentrifying but retaining character",
                    "notable_features": ["Fish Market", "Historic Warehouses", "Immigrant Communities"],
                    "demographics": "Working class, artists, new money trying to push out the old"
                },
                {
                    "name": "The Underground District",
                    "type": "entertainment",
                    "description": "Where the city's shadow economy thrives after midnight",
                    "notable_features": ["Hidden Clubs", "Black Markets", "Safe Houses"],
                    "demographics": "Night workers, thrill seekers, those who profit from desire"
                },
                {
                    "name": "University Heights",
                    "type": "educational",
                    "description": "Home to three major universities and countless students",
                    "notable_features": ["Shadow Bay University", "Medical School", "Student Quarter"],
                    "demographics": "Students, professors, idealists not yet broken by the city"
                },
                {
                    "name": "Little Odessa",
                    "type": "ethnic_enclave",
                    "description": "Eastern European immigrant community with its own rules",
                    "notable_features": ["Orthodox Churches", "Ethnic Markets", "Social Clubs"],
                    "demographics": "Recent immigrants, established families, organized crime"
                },
                {
                    "name": "The Forgotten Quarter",
                    "type": "abandoned",
                    "description": "Failed urban renewal left these blocks to decay",
                    "notable_features": ["Abandoned Factories", "Squatter Communities", "Hidden Networks"],
                    "demographics": "The displaced, the hiding, those who've fallen through cracks"
                }
            ]
        }
    
    @staticmethod
    def get_educational_systems() -> List[Dict[str, Any]]:
        """Get educational institutions for the city"""
        return [
            {
                "name": "Shadow Bay University",
                "type": "private_university",
                "founded": "1891",
                "description": (
                    "Elite private university known for business and law programs. Many students "
                    "from wealthy families experiment with the city's darker pleasures, some "
                    "getting in too deep. The university quietly handles 'incidents' to protect "
                    "its reputation."
                ),
                "notable_programs": ["Business School", "Law School", "Psychology Department"],
                "secrets": "Several professors frequent the Underground District",
                "connections": ["Financial elite", "Political establishment", "Old money families"]
            },
            {
                "name": "St. Catherine's Academy",
                "type": "private_high_school",
                "founded": "1923",
                "description": (
                    "Catholic girls' school with a pristine reputation hiding troubling secrets. "
                    "Some graduates end up in the Underground District, rebellion taken to extremes. "
                    "The school's counseling center has unusually high usage rates."
                ),
                "notable_programs": ["Classical Education", "Arts Program", "Strict Discipline"],
                "secrets": "Cover-ups of student trauma, connections to trafficking prevention",
                "connections": ["Catholic Diocese", "Wealthy Families", "Charity Networks"]
            },
            {
                "name": "Shadow Bay Community College",
                "type": "public_college",
                "founded": "1967",
                "description": (
                    "Underfunded but vital institution serving working-class students. Night "
                    "school programs mean students and staff often travel through dangerous "
                    "areas. Some students work in the Underground to pay tuition."
                ),
                "notable_programs": ["Nursing", "Criminal Justice", "Night School"],
                "secrets": "Underground recruiting happens near campus",
                "connections": ["City Government", "Labor Unions", "Social Services"]
            },
            {
                "name": "The Chrysalis Center",
                "type": "alternative_education",
                "founded": "2018",
                "description": (
                    "Privately funded education center for trafficking survivors and at-risk "
                    "youth. Teaches practical skills and emotional healing. Connected to the "
                    "underground railroad of safe houses. Officially just a 'community center.'"
                ),
                "notable_programs": ["Trauma Recovery", "Job Skills", "Self-Defense"],
                "secrets": "Funded partially by guilty conscience money from the elite",
                "connections": ["Safehouse Network", "Anonymous Donors", "Underground Protectors"]
            }
        ]
    
    @staticmethod
    def get_religious_institutions() -> List[Dict[str, Any]]:
        """Get religious organizations for the city"""
        return [
            {
                "name": "Cathedral of St. Sebastian",
                "type": "catholic_church",
                "founded": "1852",
                "description": (
                    "Gothic cathedral in the heart of the city. Provides sanctuary in more "
                    "ways than one - some priests don't ask questions about those seeking "
                    "help at odd hours. The confessionals have heard the city's darkest secrets."
                ),
                "leadership": "Archbishop Michael O'Brien - pragmatic, protective of his flock",
                "practices": ["Traditional Mass", "Confession", "Sanctuary Services"],
                "secrets": "Some clergy are part of the underground railroad for victims",
                "connections": ["City Elite", "Immigrant Communities", "Social Services"]
            },
            {
                "name": "Temple Beth Shalom",
                "type": "reform_synagogue",
                "founded": "1896",
                "description": (
                    "Progressive congregation with strong social justice focus. Runs legitimate "
                    "programs for vulnerable women, some of whom are fleeing the sex trade. "
                    "The Rabbi asks no questions about past, only helps with future."
                ),
                "leadership": "Rabbi Sarah Goldstein - fierce advocate for the vulnerable",
                "practices": ["Reform Services", "Social Justice Work", "Women's Shelter"],
                "secrets": "Provides new identities for those needing to disappear",
                "connections": ["Legal Aid", "Women's Groups", "Progressive Networks"]
            },
            {
                "name": "St. Nicholas Orthodox Church",
                "type": "orthodox_church",
                "founded": "1924",
                "description": (
                    "Heart of the Eastern European community. Beautiful icons hide meeting "
                    "rooms where community justice is dispensed. Some congregants have ties "
                    "to organized crime, others fight against it. Complex loyalties."
                ),
                "leadership": "Father Dmitri Volkov - maintains delicate balance",
                "practices": ["Orthodox Liturgy", "Community Courts", "Cultural Preservation"],
                "secrets": "Neutral ground between criminal factions and their victims",
                "connections": ["Eastern European Crime", "Immigrant Networks", "Old Country"]
            },
            {
                "name": "The Church of Personal Revelation",
                "type": "new_religious_movement",
                "founded": "2010",
                "description": (
                    "New age church preaching enlightenment through experience, including "
                    "controlled pain and pleasure. Some members blur lines between spiritual "
                    "and sexual. Popular with those seeking meaning in sensation."
                ),
                "leadership": "Prophetess Luna - charismatic former dominatrix",
                "practices": ["Ecstatic Meditation", "Sensation Rituals", "Confession Circles"],
                "secrets": "Recruiting ground for high-end sex work, but also genuine healing",
                "connections": ["Underground District", "Wealthy Seekers", "Alternative Scene"]
            },
            {
                "name": "Shadow Bay Islamic Center",
                "type": "mosque",
                "founded": "1987",
                "description": (
                    "Moderate mosque serving diverse Muslim community. Runs respected women's "
                    "programs. Some conservative members clash with the city's permissive "
                    "culture. The Imam walks a careful line between tradition and adaptation."
                ),
                "leadership": "Imam Ahmad Hassan - scholarly, protective",
                "practices": ["Daily Prayers", "Community Education", "Women's Support"],
                "secrets": "Helps Muslim women escape honor-based violence",
                "connections": ["Immigrant Communities", "Interfaith Council", "Social Services"]
            }
        ]
    
    @staticmethod
    def get_local_histories() -> List[Dict[str, Any]]:
        """Get historical events and local lore"""
        return [
            {
                "event_name": "The Great Fire of 1889",
                "date": "October 13, 1889",
                "description": (
                    "Fire destroyed much of the original harbor district. Officially started "
                    "in a warehouse, but rumors persist it was set to destroy evidence of "
                    "a trafficking ring. The rebuild created the first Underground tunnels."
                ),
                "location": "Old Harbor",
                "significance": "Created the physical underground that would later host the shadow economy",
                "mysteries": ["True cause", "Missing persons never found", "Hidden tunnels"]
            },
            {
                "event_name": "The Vanishing of 1967",
                "date": "Summer 1967",
                "description": (
                    "Seventeen young women disappeared over three months. Police claimed they "
                    "were runaways, but patterns suggested organized abduction. Case went cold. "
                    "Some say this was when the first 'Moth Queens' began protecting the vulnerable."
                ),
                "location": "Citywide",
                "significance": "Sparked creation of underground protection networks",
                "mysteries": ["True perpetrators", "Why investigation stopped", "The first protectors"]
            },
            {
                "event_name": "The Millennium Riots",
                "date": "December 31, 1999",
                "description": (
                    "Y2K celebration turned violent when police raided Underground District clubs. "
                    "Three days of riots followed. Afterwards, an uneasy truce: the Underground "
                    "could operate if it self-policed. Birth of the modern shadow economy."
                ),
                "location": "Underground District",
                "significance": "Established current relationship between law and shadow economy",
                "mysteries": ["Who ordered the raid", "The Peace Broker identity", "Hidden agreements"]
            },
            {
                "event_name": "The Harbor Bridge Incident",
                "date": "March 15, 2015",
                "description": (
                    "Body of known trafficker found hanging from Harbor Bridge with moth wings "
                    "pinned to his back. Message: 'No more.' Police investigation went nowhere. "
                    "Underground whispers of a vigilante protector. Similar incidents followed."
                ),
                "location": "Harbor Bridge",
                "significance": "Announced the Moth Queen's presence to the criminal underworld",
                "mysteries": ["Moth Queen identity", "How many protectors exist", "Police complicity"]
            }
        ]
    
    @staticmethod
    def get_urban_myths() -> List[Dict[str, Any]]:
        """Get urban legends and myths"""
        return [
            {
                "name": "The Moth Queen",
                "type": "protector_legend",
                "description": (
                    "They say she rules the Underground by night, but uses her power to save "
                    "the lost. Beautiful and terrible, she marks those under her protection "
                    "with moth tattoos. Cross her, and you'll be found with wings pinned to "
                    "your corpse. Aid her, and you'll have sanctuary forever."
                ),
                "locations": ["Underground District", "Velvet Sanctum"],
                "truth_level": "Based on real person",
                "variations": ["Single protector", "Network of protectors", "Supernatural being"]
            },
            {
                "name": "The Vanishing Floors",
                "type": "location_legend",
                "description": (
                    "Certain elevators in old Financial District buildings have floors that "
                    "only appear after midnight. Press the right combination, and doors open "
                    "to impossible clubs where anything can be bought. Some who enter never "
                    "leave. Others return changed."
                ),
                "locations": ["Financial District", "Old office buildings"],
                "truth_level": "Partially true - hidden clubs exist",
                "variations": ["Time distortion", "Portal to desires", "Test of character"]
            },
            {
                "name": "The Collectors",
                "type": "threat_legend",
                "description": (
                    "Men in expensive suits who 'collect' people like art. They frequent "
                    "university areas and struggling neighborhoods, looking for beauty in "
                    "desperation. Those collected are never seen again, or return as empty "
                    "shells. Some say they work for foreign buyers."
                ),
                "locations": ["University Heights", "Poor neighborhoods"],
                "truth_level": "Based on real trafficking rings",
                "variations": ["Organ harvesting", "Sex trafficking", "Occult purposes"]
            },
            {
                "name": "The Underground Railroad",
                "type": "hope_legend",
                "description": (
                    "A network of safe houses and protectors that can make people disappear "
                    "- but in a good way. New identities, new cities, new lives for those "
                    "escaping abuse or trafficking. You need to know the right words, find "
                    "the right doors. Moths mark the way."
                ),
                "locations": ["Throughout city", "Moth symbols"],
                "truth_level": "Completely true",
                "variations": ["Run by one person", "Secret society", "Divine intervention"]
            },
            {
                "name": "The Confession Booth",
                "type": "mystery_legend",
                "description": (
                    "An old photo booth in the abandoned subway station. Leave your darkest "
                    "secret and a offering, and sometimes you get absolution. Sometimes you "
                    "get revenge on those who wronged you. Sometimes you get a moth pin and "
                    "an address where you'll be safe."
                ),
                "locations": ["Abandoned subway", "Underground District"],
                "truth_level": "Possibly true",
                "variations": ["AI listening", "The Moth Queen's recruitment", "Collective judgment"]
            }
        ]
    
    @staticmethod
    def get_criminal_factions() -> List[Dict[str, Any]]:
        """Get criminal organizations"""
        return [
            {
                "name": "The Kozlov Syndicate",
                "type": "organized_crime",
                "description": (
                    "Eastern European crime family controlling much of the trafficking in "
                    "the city. Run by Viktor Kozlov, they see people as commodities. In "
                    "constant conflict with the Moth Queen's protective network. Brutal "
                    "but business-minded."
                ),
                "territory": "Little Odessa, Docks, Parts of Underground",
                "activities": ["Human trafficking", "Protection rackets", "Underground casinos"],
                "leadership": "Viktor Kozlov - Cold, calculating, violent",
                "enemies": ["Moth Queen network", "FBI task force", "Rival families"],
                "resources": ["Corrupt cops", "International connections", "Safe houses"]
            },
            {
                "name": "The Velvet Court",
                "type": "underground_alliance",
                "description": (
                    "Loose alliance of Underground District venue owners who maintain order "
                    "in the shadow economy. They ensure consent and safety in their "
                    "establishments, and don't tolerate trafficking. The Moth Queen is "
                    "rumored to be a founding member."
                ),
                "territory": "Underground District",
                "activities": ["Sex work regulation", "Security services", "Information brokering"],
                "leadership": "The Council of Seven - Anonymous venue owners",
                "enemies": ["Traffickers", "Vice squad hardliners", "Moral crusaders"],
                "resources": ["Venues", "Private security", "Information network"]
            },
            {
                "name": "The Lost Angels",
                "type": "street_gang",
                "description": (
                    "All-female gang formed by trafficking survivors. They protect working "
                    "girls and hunt predators. Violent when needed, but focused on "
                    "protection over profit. Many members bear moth tattoos. Direct allies "
                    "of the Moth Queen."
                ),
                "territory": "Streets around Underground, Safe zones",
                "activities": ["Protection services", "Vigilante justice", "Rescue operations"],
                "leadership": "Rotating leadership of survivors",
                "enemies": ["Pimps", "Traffickers", "Corrupt cops"],
                "resources": ["Street knowledge", "Survivor network", "Hidden weapons"]
            },
            {
                "name": "The Gilded Circle",
                "type": "elite_conspiracy",
                "description": (
                    "Wealthy elites who partake in the darkest pleasures the city offers. "
                    "They protect each other and their secrets. Some are genuine sadists, "
                    "others trapped by blackmail. They fear the Moth Queen, who knows their "
                    "secrets."
                ),
                "territory": "Financial District, Private clubs",
                "activities": ["High-end vice", "Blackmail", "Cover-ups"],
                "leadership": "Unknown - possibly Judge Reynolds",
                "enemies": ["Press", "Moth Queen", "Their own conscience"],
                "resources": ["Money", "Political influence", "Legal protection"]
            }
        ]
    
    @staticmethod
    def get_political_entities() -> List[Dict[str, Any]]:
        """Get political structures and entities"""
        return [
            {
                "name": "Shadow Bay City Council",
                "type": "local_government",
                "description": (
                    "Elected body governing the city. Split between reformers wanting to "
                    "clean up the Underground and pragmatists who prefer the current "
                    "détente. Several members are compromised by either bribes or blackmail."
                ),
                "leadership": "Mayor Patricia Chen - Reform-minded but pragmatic",
                "factions": ["Reformers", "Pragmatists", "Compromised"],
                "key_issues": ["Underground regulation", "Police corruption", "Trafficking"],
                "secrets": "Several members visit the Underground privately"
            },
            {
                "name": "SBPD Vice Division",
                "type": "law_enforcement",
                "description": (
                    "Police unit responsible for the Underground District. Some genuinely "
                    "fight trafficking, others take bribes to look away. The good cops "
                    "sometimes work with the Moth Queen's network, though never officially."
                ),
                "leadership": "Captain James Morrison - Trying to clean house",
                "factions": ["Clean cops", "Corrupt cops", "Pragmatists"],
                "key_issues": ["Corruption", "Trafficking", "Jurisdiction disputes"],
                "secrets": "Unofficial cooperation with Underground protectors"
            },
            {
                "name": "The Harbor Commission",
                "type": "port_authority",
                "description": (
                    "Controls the docks where much trafficking enters the city. Historically "
                    "corrupt but recent leadership trying to reform. Constant battlefield "
                    "between criminal influence and law enforcement."
                ),
                "leadership": "Commissioner Rita Huang - Former prosecutor",
                "factions": ["Reformers", "Old guard", "Union representatives"],
                "key_issues": ["Smuggling", "Union corruption", "Federal oversight"],
                "secrets": "Hidden tunnels from Prohibition era still in use"
            },
            {
                "name": "Shadow Bay FBI Task Force",
                "type": "federal_law",
                "description": (
                    "Federal team investigating interstate trafficking. They know about the "
                    "Moth Queen but consider her a lower priority than the major trafficking "
                    "rings. Some agents sympathize with her goals."
                ),
                "leadership": "Special Agent Diana Foster - By the book but practical",
                "factions": ["Hard-liners", "Pragmatists", "Sympathizers"],
                "key_issues": ["Jurisdiction", "Local corruption", "Resource allocation"],
                "secrets": "Unofficial policy to ignore Moth Queen activities"
            }
        ]
    
    @staticmethod
    def get_cultural_elements() -> List[Dict[str, Any]]:
        """Get cultural traditions and practices"""
        return [
            {
                "name": "The Midnight Markets",
                "type": "underground_tradition",
                "description": (
                    "Every new moon, hidden markets appear in the Underground District. "
                    "Anything can be bought or sold except people - that's the one rule "
                    "everyone follows. Break it and face the Moth Queen's justice."
                ),
                "participants": ["Underground residents", "Thrill seekers", "Criminals"],
                "practices": ["Bartering", "Information trading", "Masked attendance"],
                "significance": "Maintains Underground economy and rules"
            },
            {
                "name": "Confession Night",
                "type": "cathartic_ritual",
                "description": (
                    "First Friday of each month at certain Underground venues. People "
                    "confess their darkest desires to masked strangers. Some seek "
                    "absolution, others find someone who shares their needs. All "
                    "confessions are sacred and secret."
                ),
                "participants": ["The guilty", "The curious", "The desperate"],
                "practices": ["Anonymous confession", "Ritual absolution", "Connection making"],
                "significance": "Provides release valve for the city's pressure"
            },
            {
                "name": "The Moths' Ball",
                "type": "protective_celebration",
                "description": (
                    "Annual masquerade celebrating those who protect the vulnerable. "
                    "Attended by survivors, protectors, and allies. The Moth Queen "
                    "traditionally appears at midnight. Raises funds for safe houses."
                ),
                "participants": ["Survivors", "Protectors", "Wealthy allies"],
                "practices": ["Masquerade", "Testimony", "Fundraising"],
                "significance": "Builds community and resources for protection network"
            },
            {
                "name": "Harbor Day Festival",
                "type": "city_celebration",
                "description": (
                    "Official city celebration of its founding. By day, family friendly "
                    "fun. By night, the Underground hosts its own darker festivities. "
                    "Traditional time for truces between rival factions."
                ),
                "participants": ["All city residents"],
                "practices": ["Parades", "Fireworks", "Underground parties"],
                "significance": "One day when all parts of the city coexist"
            },
            {
                "name": "The Red Light Vigil",
                "type": "memorial_tradition",
                "description": (
                    "Monthly vigil for those lost to violence and trafficking. Red "
                    "candles are lit in windows throughout the Underground. Names "
                    "are read. The missing are remembered. Sometimes, the lost return."
                ),
                "participants": ["Sex workers", "Families", "Advocates"],
                "practices": ["Candle lighting", "Name reading", "Silent march"],
                "significance": "Remembers the lost and builds solidarity"
            }
        ]
    
    @staticmethod
    def get_notable_figures() -> List[Dict[str, Any]]:
        """Get important historical and current figures"""
        return [
            {
                "name": "Elizabeth 'Liberty' Chen",
                "era": "1960s-1970s",
                "role": "The First Protector",
                "description": (
                    "Former trafficking victim who became the first to fight back systematically. "
                    "Founded the original safe house network. Disappeared in 1978, but her "
                    "methods inspired others. Some say the Moth Queen follows her playbook."
                ),
                "legacy": "Created the protection network model",
                "mysteries": ["Final fate", "Hidden safe houses", "Successor identity"]
            },
            {
                "name": "Judge Marcus Reynolds",
                "era": "1990s-present",
                "role": "The Compromised Idealist",
                "description": (
                    "Respected judge who tried to reform the system from within. Made deals "
                    "with devils to protect the innocent. Now trapped by his compromises. "
                    "Rumored to be the Gilded Circle's puppet or leader."
                ),
                "legacy": "Established legal precedents protecting sex workers",
                "mysteries": ["True allegiances", "Blackmail material", "Redemption possibility"]
            },
            {
                "name": "Sister Mary Catherine",
                "era": "1980s-2010s",
                "role": "The Sanctuary Keeper",
                "description": (
                    "Nun who turned St. Sebastian's into a true sanctuary. Never asked "
                    "questions, just offered help. Died peacefully in 2010, but her "
                    "policies continue. Said to have hidden records of all she helped."
                ),
                "legacy": "Made religious sanctuary practical reality",
                "mysteries": ["Hidden records", "True identity", "Miraculous escapes"]
            },
            {
                "name": "Viktor 'The Collector' Kozlov",
                "era": "2000s-present",
                "role": "The Modern Monster",
                "description": (
                    "Current head of the most brutal trafficking ring. Sees people as "
                    "products. Has survived multiple assassination attempts. They say "
                    "he keeps trophies from those he's sold. Prime target of the Moth Queen."
                ),
                "legacy": "Industrialized human trafficking",
                "mysteries": ["True base of operations", "Political protection", "Weaknesses"]
            },
            {
                "name": "The Moth Queen",
                "era": "2010s-present",
                "role": "The Dark Protector",
                "description": (
                    "Current protector of the vulnerable. Rules the Velvet Sanctum by "
                    "night, saves the trafficked by later night. Identity unknown but "
                    "impact undeniable. Has built most extensive protection network yet."
                ),
                "legacy": "Revolutionized underground protection",
                "mysteries": ["True identity", "Origin story", "Future plans"]
            }
        ]

    @staticmethod
    async def generate_complete_city_lore(ctx, user_id: int, conversation_id: int):
        """Generate and store complete city lore"""
        from lore.core.lore_system import LoreSystem
        
        logger.info("Generating complete modern city lore preset")
        
        # Get lore system instance
        lore_system = await LoreSystem.get_instance(user_id, conversation_id)
        await lore_system.ensure_initialized()
        
        # Generate foundation
        foundation = ModernCityLorePresets.get_city_foundation_lore()
        
        # Generate all lore categories
        results = {
            "foundation": foundation,
            "education": ModernCityLorePresets.get_educational_systems(),
            "religion": ModernCityLorePresets.get_religious_institutions(),
            "history": ModernCityLorePresets.get_local_histories(),
            "myths": ModernCityLorePresets.get_urban_myths(),
            "criminal_factions": ModernCityLorePresets.get_criminal_factions(),
            "political": ModernCityLorePresets.get_political_entities(),
            "culture": ModernCityLorePresets.get_cultural_elements(),
            "figures": ModernCityLorePresets.get_notable_figures()
        }
        
        # Store in database through the lore system
        # This would need implementation in the actual system
        
        logger.info("Modern city lore generation complete")
        return results
