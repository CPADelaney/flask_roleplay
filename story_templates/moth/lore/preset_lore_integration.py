# story_templates/moth/lore/preset_lore_integration.py
"""
Integrates the modern city preset lore with The Moth and Flame story
"""

class MothFlameLoreIntegration:
    """Ensures the preset lore properly supports the story"""
    
    @staticmethod
    async def enhance_story_lore(ctx, user_id: int, conversation_id: int):
        """Enhance the city lore with story-specific elements"""
        
        # Add story-specific locations to the city
        story_locations = [
            {
                "name": "The Velvet Sanctum",
                "district": "Underground District",
                "type": "bdsm_club",
                "description": (
                    "Hidden beneath an innocent boutique, the Velvet Sanctum descends into "
                    "darkness both literal and metaphorical. Red velvet drapes, candlelit "
                    "alcoves, and an obsidian throne where the Queen holds court. Every "
                    "surface whispers of power exchanged, pain transformed to pleasure."
                ),
                "areas": {
                    "main_chamber": "Where public performances unfold",
                    "throne_room": "The Queen's seat of power",
                    "private_booths": "For intimate negotiations of power",
                    "the_dungeon": "Where limits are tested and found",
                    "preparation_chamber": "Where the Queen becomes goddess"
                },
                "operating_hours": {
                    "Monday": "8PM-3AM",
                    "Tuesday": "Private clients only",
                    "Wednesday": "8PM-4AM - Grand performances",
                    "Thursday": "8PM-3AM",
                    "Friday": "8PM-5AM - The Queen's Court",
                    "Saturday": "8PM-5AM - Busiest night",
                    "Sunday": "Closed"
                },
                "known_patrons": ["Business elite", "Broken souls", "Thrill seekers"],
                "rumors": [
                    "The Queen knows everyone's secrets",
                    "Some who enter are never the same",
                    "Hidden doors lead to darker spaces"
                ]
            }
        ]
        
        # Add story-specific NPCs to the population
        story_npcs = [
            {
                "name": "Marcus Sterling",
                "type": "regular",
                "description": "Former CEO turned devoted submissive",
                "location": "Velvet Sanctum",
                "role": "The Queen's most devoted"
            },
            {
                "name": "Sarah Chen",
                "type": "survivor",
                "description": "Rescued trafficking victim helping others",
                "location": "Safehouse Network",
                "role": "Testament to the Queen's other work"
            }
        ]
        
        # Add story-specific urban myths
        story_myths = [
            {
                "name": "The Three Words",
                "description": (
                    "They say the Moth Queen has three words that burn beneath her tongue, "
                    "words she's never spoken aloud. Some claim if you earn her complete trust, "
                    "you might hear them. Others say speaking them would destroy her power."
                ),
                "truth": "Known only to those who've earned it"
            },
            {
                "name": "The Mask Room",
                "description": (
                    "Somewhere in her private chambers is a room full of masks - one for "
                    "each person who promised to stay but vanished. She talks to them "
                    "sometimes, practicing words she'll never say."
                ),
                "truth": "Those who've seen it don't speak of it"
            }
        ]
        
        # Add story-specific conflicts
        story_conflicts = [
            {
                "name": "The Queen vs The Collector",
                "type": "personal",
                "description": (
                    "Viktor Kozlov wants to clip the Moth Queen's wings. She cost him "
                    "millions and freed his 'products.' He's hunting her identity while "
                    "she dismantles his empire one safehouse at a time."
                ),
                "stakes": "Lives hang in balance"
            }
        ]
        
        return {
            "enhanced_locations": story_locations,
            "enhanced_npcs": story_npcs,
            "enhanced_myths": story_myths,
            "enhanced_conflicts": story_conflicts
        }
