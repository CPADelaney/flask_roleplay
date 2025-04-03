# lore/utils/theming.py

import re
import random
from typing import Dict, List, Any

class MatriarchalThemingUtils:
    """
    Streamlined utility class for applying matriarchal theming to different types of lore content.
    Uses a data-driven approach to apply theming based on content type.
    """
    
    # Dictionary of regex patterns to replacement strings for basic feminization
    WORD_REPLACEMENTS = {
        r"\bgod\b": "Goddess",
        r"\bgods\b": "Goddesses",
        r"\bgodhood\b": "Goddesshood",
        r"\bking\b": "Queen",
        r"\bkings\b": "Queens",
        r"\bkingdom\b": "Queendom",
        r"\bprince\b": "princess",
        r"\bprinces\b": "princesses",
        r"\bfather\b": "mother",
        r"\bfathers\b": "mothers",
        r"\bhim\b": "her",
        r"\bhis\b": "her",
        r"\bhe\b": "she",
        r"\blord\b": "lady",
        r"\blords\b": "ladies",
        r"\bman\b": "woman",
        r"\bmen\b": "women",
        r"\bbrotherhood\b": "sisterhood",
        r"\bfraternal\b": "sororal",
        r"\bpatriarch\b": "matriarch",
        r"\bpatriarchy\b": "matriarchy",
        r"\bpatriarchal\b": "matriarchal",
        r"\bemperor\b": "empress",
        r"\bemperors\b": "empresses",
        r"\bduke\b": "duchess",
        r"\bdukes\b": "duchesses",
        r"\bcount\b": "countess",
        r"\bcounts\b": "countesses",
        r"\bbaron\b": "baroness",
        r"\bbarons\b": "baronesses",
        r"\blordship\b": "ladyship",
        r"\blordships\b": "ladyships",
        r"\bkingship\b": "queenship",
        r"\bkingly\b": "queenly",
    }
    
    # Supreme feminine figure titles for random selection
    DIVINE_TITLES = [
        "Supreme Goddess", "High Empress", "Great Matriarch", "Divine Mother",
        "Infinite Mistress of Creation", "Eternal Goddess", "All-Mother",
        "Sovereign Matriarch", "Celestial Queen", "Grand Mistress of Existence",
    ]
    
    # Themed content by lore type - organized for better maintenance
    THEMED_CONTENT = {
        # Cosmological content
        "cosmology": "At the heart of all creation is the Feminine Principle, the source of all life and power. "
                    "The cosmos itself is understood as fundamentally feminine in nature, "
                    "with any masculine elements serving and supporting the greater feminine whole.",
        "magic_system": "The flow and expression of magical energies reflect the natural order of feminine dominance. "
                       "Women typically possess greater innate magical potential and exclusive rights to the highest mysteries. "
                       "Men specializing in arcane arts often excel in supportive, protective, or enhancing magics, "
                       "operating in service to more powerful feminine traditions.",
        
        # Social structures
        "social_structure": "Society is organized along feminine lines of authority, with women occupying the most "
                          "important leadership positions. Men serve supportive roles, with status often determined "
                          "by their usefulness and loyalty to female superiors.",
        "faction": "The organization's power structure follows matriarchal principles, with women in the "
                 "highest positions of authority. Male members serve in supporting roles, earning status "
                 "through their usefulness and loyalty.",
        
        # Historical content
        "world_history": "Throughout recorded chronicles, women have held the reins of power. "
                        "Great Empresses, Matriarchs, and female rulers have guided civilizations toward prosperity. "
                        "Though conflicts and rebellions against this natural order have arisen, "
                        "the unshakable principle of feminine dominance remains the bedrock of history.",
        "history": "Historical records emphasize the accomplishments of great women and the importance of "
                 "matrilineal succession. Male contributions are noted primarily in how they supported "
                 "or served female leadership.",
        "event": "The event unfolded according to the established gender hierarchy, with women directing "
               "the course of action and men executing their will. Any violations of this order were "
               "swiftly addressed.",
        
        # Cultural content
        "calendar_system": "The calendar marks vital dates in feminine history, aligning festivals and holy days "
                          "with lunar cycles and the reigns of legendary Empresses. Major celebrations honor "
                          "the cyclical power of womanhood, reflecting its role in birth, renewal, and creation.",
        "culture": "Cultural expressions, from art to literature to music, celebrate feminine power, beauty, "
                 "and wisdom. The feminine principle is seen as the creative force behind all worthwhile "
                 "cultural achievements.",
        "landmark": "The architecture and design embody feminine principles of power and authority. "
                   "Female figures dominate the iconography, with male representations shown in supportive or "
                   "subservient positions.",
        
        # Religious content
        "religion": "The spiritual order mirrors the earthly one, with goddesses and feminine divine forces "
                  "representing the most important cosmic principles. Male deities, if they exist, serve "
                  "specific supporting functions within a feminine-dominated pantheon.",
        "myth": "The mythology emphasizes the primacy of female deities and heroines, with male figures "
               "playing important but secondary roles. Stories reinforce the natural order of feminine "
               "superiority and male service.",
        
        # Character and interpersonal content
        "character": "Personal relationships and social standing reflect the feminine-dominant order. "
                   "Interactions between genders follow established protocols that reinforce the natural "
                   "supremacy of women.",
        "news": "Public discourse naturally centers feminine perspectives and priorities. "
              "Information is filtered through a lens that reinforces the established gender hierarchy "
              "and celebrates influential women.",
        
        # Default content for any type not specifically defined
        "default": "Power dynamics follow the natural order of feminine dominance, with masculine elements "
                 "existing primarily to support and serve the greater feminine whole."
    }
    
    # Emphasis level messages by level
    EMPHASIS_MESSAGES = {
        1: "",  # No additional emphasis
        2: "\n\nThroughout every realm, the ascendancy of womanhood underpins all aspects of life, "
           "unquestioned and unassailable.",
        3: "\n\nNo force—mortal or divine—can transcend the supremacy of the feminine essence, "
           "the primal wellspring of creation itself."
    }
    
    @classmethod
    def apply_matriarchal_theme(cls, lore_type: str, content: str, emphasis_level: int = 2) -> str:
        """
        Apply appropriate matriarchal theming based on lore type with improved performance.
        
        Args:
            lore_type: Type of lore content ('cosmology', 'magic_system', 'history', etc.)
            content: Original content to modify
            emphasis_level: Level of emphasis on matriarchal themes (1-3)
            
        Returns:
            Modified content with matriarchal theming
        """
        # Check if content already has matriarchal themes
        if cls._has_matriarchal_content(content):
            # If it does, just ensure word replacements are applied and return
            return cls._replace_gendered_words(content)
        
        # Full theming process
        result = cls._replace_gendered_words(content)
        result = cls._add_themed_content(result, lore_type.lower())
        
        # Ensure goddess reference for religious/cosmological content
        if lore_type.lower() in ["cosmology", "magic_system", "religion", "pantheon", "deity"]:
            result = cls._ensure_divine_reference(result)
        
        # Add emphasis based on requested level if not level 1
        if emphasis_level > 1 and emphasis_level in cls.EMPHASIS_MESSAGES:
            result += cls.EMPHASIS_MESSAGES[emphasis_level]
        
        return result
    
    @classmethod
    def _has_matriarchal_content(cls, text: str) -> bool:
        """Efficiently check if text already has strong matriarchal themes."""
        # Compile patterns once for better performance
        patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in [
                r"matriarch", r"matriarchal", r"matriarchy",
                r"female dominan(t|ce)", r"feminine power", r"goddess",
                r"women rule", r"feminine authority", r"female suprem(e|acy)"
            ]
        ]
        
        # Count occurrences of matriarchal indicators
        count = sum(1 for pattern in patterns if pattern.search(text))
        return count > 2
    
    @classmethod
    def _replace_gendered_words(cls, text: str) -> str:
        """Replace gendered words with feminine alternatives while preserving case."""
        result = text
        
        # Process all word replacements
        for pattern_str, replacement_str in cls.WORD_REPLACEMENTS.items():
            pattern = re.compile(pattern_str, re.IGNORECASE)
            
            def _replacement_func(match):
                original = match.group(0)
                # Preserve case of original word
                if original and original[0].isupper():
                    return replacement_str[0].upper() + replacement_str[1:]
                return replacement_str
            
            result = pattern.sub(_replacement_func, result)
        
        return result
    
    @classmethod
    def _add_themed_content(cls, text: str, lore_type: str) -> str:
        """Add type-specific themed content to the text."""
        # Get themed content for this type (or use default)
        themed_content = cls.THEMED_CONTENT.get(
            lore_type, 
            cls.THEMED_CONTENT["default"]
        )
        
        # Find a good place to insert the content
        if "\n\n" in text:
            # Insert after the first paragraph for readability
            paragraphs = text.split("\n\n", 1)
            return paragraphs[0] + "\n\n" + themed_content + "\n\n" + paragraphs[1]
        else:
            # Append at the end if no paragraph breaks found
            return text.strip() + "\n\n" + themed_content
    
    @classmethod
    def _ensure_divine_reference(cls, text: str) -> str:
        """Ensure text includes reference to supreme feminine divine entity."""
        # Skip if already has feminine divine reference
        if re.search(r"(goddess|divine mother|matriarch|empress of creation)", text, re.IGNORECASE):
            return text
            
        # Add divine reference
        title = random.choice(cls.DIVINE_TITLES)
        insertion = (
            f"\n\nAt the cosmic center stands {title}, "
            "the eternal wellspring of existence. Her dominion weaves reality itself."
        )
        
        return text.strip() + insertion
