# story_templates/moth/lore/consistency_guide.py
"""
Complete consistency rules and LLM guidance for Queen of Thorns SF Bay Area setting
Prevents contradictions and maintains narrative coherence
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

class QueenOfThornsConsistencyGuide:
    """Complete consistency rules for dynamic content generation"""
    
    @staticmethod
    def get_critical_rules() -> List[Dict[str, Any]]:
        """Rules that must NEVER be violated"""
        return [
            {
                "rule": "NO_OFFICIAL_NAME",
                "description": "The network has NO official name",
                "explanation": (
                    "NEVER give the organization a formal name. "
                    "ALWAYS refer to it as 'the network' or 'the garden' internally. "
                    "OUTSIDERS call it various names: 'The Rose & Thorn Society,' 'The Thorn Garden,' etc."
                ),
                "correct_usage": [
                    "the network",
                    "the garden", 
                    "our people",
                    "what outsiders call the Rose & Thorn Society"
                ],
                "incorrect_usage": [
                    "The Rose & Thorn Society's official policy...",
                    "The Garden announced...",
                    "Members of the Shadow Matriarchy..."
                ],
                "enforcement": "ALWAYS refer to it as 'the network' internally"
            },
            {
                "rule": "QUEEN_AMBIGUITY",
                "description": "The Queen's identity is ALWAYS ambiguous",
                "explanation": (
                    "NEVER confirm if she's one person, multiple people, or a role. "
                    "ALWAYS maintain mystery about succession, identity, and nature."
                ),
                "correct_usage": [
                    "The Queen of Thorns—whoever or whatever she truly is...",
                    "The Queen, or Queens, or the concept itself...",
                    "Someone known as the Queen of Thorns"
                ],
                "incorrect_usage": [
                    "The current Queen, Sarah Chen...",
                    "The role of Queen passes every decade...",
                    "She has been Queen since 2010..."
                ],
                "enforcement": "Identity, number, and nature must remain unknown"
            },
            {
                "rule": "LAYERED_SECRECY",
                "description": "All information exists in four layers",
                "layers": {
                    "PUBLIC": "What everyone can see",
                    "SEMI_PRIVATE": "What members know",
                    "HIDDEN": "What the network controls",
                    "DEEP_SECRET": "What even insiders don't fully understand"
                },
                "examples": {
                    "location": {
                        "PUBLIC": "The Rose Garden Café serves excellent coffee",
                        "SEMI_PRIVATE": "Tuesday book clubs discuss power dynamics",
                        "HIDDEN": "Back room meetings screen potential recruits",
                        "DEEP_SECRET": "Direct connection to the Queen herself"
                    }
                }
            }
        ]
    
    @staticmethod
    def get_location_rules() -> Dict[str, Any]:
        """Rules for location consistency"""
        return {
            "address_rules": {
                "public_businesses": "Full street address (e.g., '555 California St, 40th Floor')",
                "private_homes": "District only (e.g., 'Noe Valley home')",
                "secret_venues": "Vague references (e.g., 'unmarked door off Folsom')",
                "queens_spaces": "Never locate precisely"
            },
            "key_locations": {
                "The Rose Garden Café": {
                    "address": "Valencia between 20th and 21st",
                    "layer": "PUBLIC",
                    "note": "Entry point, not headquarters"
                },
                "Thornfield & Associates": {
                    "address": "555 California St, 40th Floor",
                    "layer": "PUBLIC", 
                    "note": "Legal front"
                },
                "Maison Noir": {
                    "address": "Undisclosed, SoMa district",
                    "layer": "HIDDEN",
                    "note": "Never give exact location"
                },
                "The Inner Garden": {
                    "address": "Location unknown",
                    "layer": "DEEP SECRET",
                    "note": "May not be physical"
                }
            }
        }
    
    @staticmethod
    def get_organization_structure() -> Dict[str, Any]:
        """Canonical organization hierarchy"""
        return {
            "hierarchy": [
                {
                    "level": 1,
                    "title": "Seedlings",
                    "description": "Newly aware, exploring",
                    "duration": "6 months - 2 years",
                    "access": "Public events only"
                },
                {
                    "level": 2,
                    "title": "Roses",
                    "description": "Practicing members, committed",
                    "duration": "Indefinite",
                    "access": "Private events, mentorship"
                },
                {
                    "level": 3,
                    "title": "Thorns",
                    "description": "Protectors and enforcers",
                    "duration": "By appointment",
                    "access": "Security information"
                },
                {
                    "level": 4,
                    "title": "Gardeners",
                    "description": "Teachers and guides",
                    "duration": "Earned through service",
                    "access": "Shape next generation"
                },
                {
                    "level": 5,
                    "title": "Regional Thorns",
                    "description": "District managers",
                    "duration": "Appointed position",
                    "access": "Local operations control"
                },
                {
                    "level": 6,
                    "title": "Rose Council",
                    "description": "Seven senior dominants",
                    "duration": "Until death or retirement",
                    "access": "Major decisions"
                },
                {
                    "level": 7,
                    "title": "The Queen of Thorns",
                    "description": "Ultimate authority (nature unknown)",
                    "duration": "Unknown",
                    "access": "Everything"
                }
            ],
            "geographic_scope": {
                "absolute_control": "San Francisco Bay Area only",
                "influence": "Tech hubs via diaspora (Seattle, Austin, NYC, LA)",
                "allied": "International feminist/BDSM networks",
                "NOT": "A global organization with chapters"
            }
        }
    
    @staticmethod
    def get_historical_timeline() -> List[Dict[str, Any]]:
        """Canonical historical events"""
        return [
            {
                "period": "1890s",
                "events": [
                    "Victorian women establish proto-networks through 'charity work'",
                    "Mrs. Cordelia Thornfield creates 'reading circles'"
                ],
                "significance": "Origins of hidden female power structures"
            },
            {
                "period": "1906-1917",
                "events": [
                    "Post-earthquake transformation",
                    "Brothels become boarding houses",
                    "First 'Queen of Thorns' title emerges",
                    "Underground infrastructure established"
                ],
                "significance": "Formal structure emerges"
            },
            {
                "period": "1960s",
                "events": [
                    "Hippie era evolution",
                    "Second Queen emerges from communes",
                    "Network saves flower children from predators",
                    "Modern structure takes shape"
                ],
                "significance": "Countercultural integration"
            },
            {
                "period": "1978",
                "events": [
                    "Post-Milk assassination expansion",
                    "Protection networks formalize",
                    "Alternative justice systems develop"
                ],
                "significance": "Political awakening"
            },
            {
                "period": "1989",
                "events": [
                    "Loma Prieta earthquake",
                    "Major expansion of safe houses",
                    "Network proves superior to official response"
                ],
                "significance": "Infrastructure expansion"
            },
            {
                "period": "2001",
                "events": [
                    "Dot-com bust opportunity",
                    "Tech integration begins",
                    "Failed executives recruited"
                ],
                "significance": "Tech world infiltration"
            },
            {
                "period": "2017-2018",
                "events": [
                    "#MeToo transformation boom",
                    "Massive intake of targets",
                    "Transformation pipeline expands"
                ],
                "significance": "Mainstream adjacency"
            },
            {
                "period": "2020-2022",
                "events": [
                    "Pandemic digitalization",
                    "Remote operations developed",
                    "Empty offices repurposed"
                ],
                "significance": "Operational evolution"
            },
            {
                "period": "2025",
                "events": ["Present day setting"],
                "significance": "Current story timeframe"
            }
        ]
    
    @staticmethod
    def get_power_balance() -> Dict[str, Any]:
        """What the network can and cannot do"""
        return {
            "powerful_in": [
                "Information gathering and leverage",
                "Psychological manipulation",
                "Protection of members",
                "Financial resources from converted executives",
                "Legal system influence",
                "Social pressure tactics"
            ],
            "vulnerable_to": [
                "Mass media exposure",
                "Federal investigation",
                "Internal succession disputes",
                "Technology they don't control",
                "Competing criminal organizations",
                "Member betrayal"
            ],
            "cannot": [
                "Operate openly",
                "Control everyone",
                "Prevent all harm",
                "Exist without secrecy",
                "Override federal law enforcement entirely",
                "Function without the Queen"
            ]
        }
    
    @staticmethod
    def get_relationship_dynamics() -> Dict[str, Any]:
        """How the network relates to other organizations"""
        return {
            "tech_companies": {
                "relationship": "Predator/Prey/Partner",
                "public": "Women in tech initiatives",
                "hidden": "Executives under control",
                "tension": "Need their money but despise their culture"
            },
            "law_enforcement": {
                "relationship": "Pragmatic accommodation",
                "public": "Normal citizen-police interaction",
                "hidden": "Some cops know and appreciate alternative justice",
                "rule": "Never harm cops, they sometimes ignore network activities"
            },
            "criminal_organizations": {
                "relationship": "Active opposition",
                "public": "No visible interaction",
                "hidden": "Shadow war over trafficking",
                "method": "Use their infrastructure against them"
            },
            "bdsm_community": {
                "relationship": "Overlapping but distinct",
                "public": "Part of the same scene",
                "hidden": "Network uses community as cover",
                "important": "Public kink ≠ the hidden network"
            }
        }
    
    @staticmethod
    def get_language_conventions() -> Dict[str, Any]:
        """How different groups refer to things"""
        return {
            "network_references": {
                "insiders": ["the network", "the garden", "our people"],
                "outsiders": ["The Rose & Thorn Society", "that secret feminist cult"],
                "media": ["alleged shadow organization"],
                "academics": ["emergent power structures in feminist spaces"]
            },
            "code_phrases": {
                "interesting energy": "recognizing power dynamics",
                "she has presence": "identifying a dominant",
                "very responsive": "identifying a submissive",
                "growth-oriented": "open to exploration",
                "needs pruning": "target for transformation"
            }
        }
    
    @staticmethod
    def get_common_errors() -> List[str]:
        """Errors to avoid in generation"""
        return [
            "DON'T give the network an official name or acronym",
            "DON'T reveal the Queen's identity or confirm singularity/plurality",
            "DON'T make the network omnipotent or omniscient",
            "DON'T place branches outside the Bay Area (allies, not branches)",
            "DON'T confuse public BDSM community with the hidden network",
            "DON'T make transformation instant (it takes months/years)",
            "DON'T have the network kill people (they transform or exile)",
            "DON'T make all dominants female or all submissives male",
            "DON'T reveal exact numbers of members",
            "DON'T locate the Queen's private spaces"
        ]
    
    @staticmethod
    def get_tone_themes() -> Dict[str, Any]:
        """Tone and theme guidelines"""
        return {
            "noir_sensibility": "Information has prices, everyone has secrets",
            "power_dynamics": "Always present, often inverted from appearance",
            "consent_complexity": "Explore edge cases ethically",
            "feminist_core": "Women's empowerment through dominance",
            "bay_area_specific": "Tech money meets alternative culture",
            "moral_ambiguity": "Heroes and villains aren't clear-cut",
            "transformation_focus": "People can change fundamentally",
            "mystery_maintenance": "Some things must stay unknown"
        }
    
    @staticmethod
    def validate_content(content: str) -> Dict[str, Any]:
        """Validate content for consistency violations"""
        violations = []
        warnings = []
        
        # Critical violations
        if any(phrase in content for phrase in [
            "The Rose & Thorn Society announced",
            "The Garden's official",
            "The Shadow Matriarchy decreed"
        ]):
            violations.append("CRITICAL: Giving the network an official name")
            
        if "Queen" in content and any(phrase in content for phrase in [
            "her name is",
            "succeeded by",
            "real identity",
            "born in"
        ]):
            violations.append("CRITICAL: Revealing Queen's identity")
            
        if any(phrase in content for phrase in [
            "chapter in",
            "branches worldwide",
            "Seattle office"
        ]):
            violations.append("CRITICAL: Network has branches outside Bay Area")
            
        # Warnings
        if "instantly transformed" in content or "immediate conversion" in content:
            warnings.append("Transformation described as too quick")
            
        if "the network controls" in content and "everything" in content:
            warnings.append("Network described as too powerful")
            
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings
        }
    
    @staticmethod
    def get_complete_system_prompt() -> str:
        """Complete system prompt for LLM"""
        return """
# QUEEN OF THORNS SF BAY AREA - COMPLETE CONSISTENCY GUIDE

## CRITICAL RULES (NEVER VIOLATE):

1. **NO OFFICIAL NAME**: The network has NO formal name. Refer to it as "the network" or note that outsiders use various names.

2. **QUEEN AMBIGUITY**: NEVER reveal if the Queen is one person, multiple people, or a role. Her identity remains unknown.

3. **FOUR-LAYER SECRECY**: All information exists in layers:
   - PUBLIC: What everyone sees
   - SEMI-PRIVATE: What members know  
   - HIDDEN: What the network controls
   - DEEP SECRET: Ultimate mysteries

4. **GEOGRAPHIC LIMITS**: The network controls the Bay Area ONLY. Other cities have allies or influenced groups, not branches.

5. **TRANSFORMATION TIME**: Changes take months or years, never instant.

## ORGANIZATION STRUCTURE:
- Seedlings → Roses → Thorns → Gardeners → Regional Thorns → Rose Council → The Queen

## POWER BALANCE:
- STRONG IN: Information, manipulation, protection, resources, legal influence
- WEAK TO: Mass exposure, federal investigation, internal conflicts, betrayal
- CANNOT: Operate openly, control everyone, prevent all harm

## LANGUAGE CONVENTIONS:
- Insiders say: "the network," "the garden"
- Outsiders say: "Rose & Thorn Society," "that cult"
- Code: "interesting energy," "needs pruning"

## TONE: Noir sensibility, moral ambiguity, feminist core, mystery maintenance

When generating content, maintain these rules absolutely. Choose ambiguity over revelation.
"""
    
    @staticmethod
    def get_quick_reference() -> str:
        """Quick reference card for generation"""
        return """
QUICK REFERENCE - QUEEN OF THORNS CONSISTENCY:
✗ Official names → ✓ "the network"
✗ "Queen Sarah" → ✓ "The Queen, whoever she is"
✗ "Our Seattle chapter" → ✓ "Allied network in Seattle"
✗ "Instant transformation" → ✓ "Months of careful work"
✗ "We control the city" → ✓ "Influence through careful pressure"

Remember: PUBLIC|SEMI-PRIVATE|HIDDEN|DEEP SECRET layers for all info
"""
