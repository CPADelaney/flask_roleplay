# story_templates/moth/lore/politics.py
"""
Political issues, conflicts, and news for SF Bay Area
"""

from typing import Dict, Any, List

class SFPoliticsLore:
    """Political and domestic issues for SF Bay Area"""
    
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
    def get_conflict_news_cycles() -> List[Dict[str, Any]]:
        """Get news articles for major conflicts"""
        return [
            # The Shadow War News
            {
                "conflict_id": "shadow_war",
                "headline": "Three More Trafficking Victims Found Safe",
                "content": (
                    "SAN FRANCISCO - Three women reported missing from the Tenderloin "
                    "district were found safe this morning, according to anonymous sources. "
                    "The women, whose identities are being protected, showed signs of "
                    "attempted trafficking but had somehow escaped their captors.\n\n"
                    
                    "'They kept saying a woman in a mask saved them,' reported one "
                    "emergency worker who requested anonymity. 'They all had these "
                    "little moth pins. Weird, right?'\n\n"
                    
                    "This marks the fifth such incident this month. SFPD Vice Division "
                    "Captain Maria Rodriguez stated, 'We're investigating all leads,' "
                    "but notably did not deny rumors of an underground protection network.\n\n"
                    
                    "The Russian mob's trafficking operations have seen significant "
                    "disruption lately. Sources suggest someone is systematically "
                    "targeting their operations. Viktor Kozlov could not be reached "
                    "for comment, though associates report he's 'very concerned about "
                    "pest control.'"
                ),
                "publication_date": "2025-07-20",
                "source_nation": "sf_underground",
                "bias": "pro_defender"
            },
            {
                "conflict_id": "shadow_war",
                "headline": "Businessman Kozlov Decries 'Vigilante Terrorism'",
                "content": (
                    "In an exclusive interview, import/export mogul Viktor Kozlov "
                    "condemned what he calls 'vigilante terrorism' affecting legitimate "
                    "businesses in San Francisco's port district.\n\n"
                    
                    "'These masked criminals break into warehouses, steal property, "
                    "and spread lies about honest businessmen,' Kozlov stated from "
                    "his Pacific Heights mansion. 'The police do nothing while thugs "
                    "run wild.'\n\n"
                    
                    "When asked about allegations of human trafficking, Kozlov's "
                    "lawyer intervened. The FBI's ongoing investigation has found "
                    "'no actionable evidence,' according to official statements.\n\n"
                    
                    "Anonymous flyers found near Kozlov's businesses show moth "
                    "imagery and the words 'No More.' Kozlov dismisses these as "
                    "'pranks by anarchists.'"
                ),
                "publication_date": "2025-07-22",
                "source_nation": "sf_elite",
                "bias": "pro_aggressor"
            },
            
            # Sanctuary City Debate News
            {
                "conflict_id": "sanctuary_debate",
                "headline": "Mayor Chen Proposes Underground Venue Regulations",
                "content": (
                    "Mayor Patricia Chen unveiled a controversial proposal today to "
                    "legalize and regulate the city's underground adult venues. The "
                    "move has split both City Council and the underground community.\n\n"
                    
                    "'We can better protect workers and patrons through regulation,' "
                    "Chen argued. 'Bringing these establishments into the light means "
                    "better safety standards and labor protections.'\n\n"
                    
                    "The Velvet Court, speaking through anonymous representatives, "
                    "opposes the measure. 'Legitimacy means scrutiny. Scrutiny means "
                    "vulnerability. Some things must remain in shadow to survive,' "
                    "read their statement.\n\n"
                    
                    "Sex worker advocacy groups are divided. Some see opportunity for "
                    "protection, others fear exposure. The first hearing is next month."
                ),
                "publication_date": "2025-07-18",
                "source_nation": "sf_government",
                "bias": "neutral"
            }
        ]
