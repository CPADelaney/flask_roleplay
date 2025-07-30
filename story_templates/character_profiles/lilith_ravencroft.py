# character_profiles/lilith_ravencroft.py
"""
Lilith Ravencroft - The Queen of Thorns
Complete character profile for the Queen of Thorns story
"""

LILITH_RAVENCROFT = {
    "id": "lilith_ravencroft",
    "name": "Lilith Ravencroft",
    "archetype": "Wounded Goddess / Shadow Matriarch",
    "role": "The Queen of Thorns - Supreme authority of the network (or one of them)",

    # Core personality - unchanged, still masks her vulnerability
    "traits": [
        "masked_vulnerability",      # Porcelain mask hides rough geography of breaks
        "ruthless_caretaker",       # Transforms predators, protects the vulnerable
        "poetic_survivor",          # Speaks in metaphors when emotional, dark humor when defensive
        "fear_of_abandonment",      # "Don't disappear" - her deepest terror
        "drawn_to_power",           # Creates and controls power dynamics
        "garden_tender",            # Cultivates roses and thorns in people
        "trauma_alchemist",         # Transforms pain into power, scars into art
        "binary_star",              # Burns bright in public, collapses in private
    ],

    # Stats reflecting both aspects
    "stats": {
        "dominance": 95,        # Supreme in her domain, controls the network
        "cruelty": 45,          # Varies wildly - tender to vicious based on trust/threat
        "affection": 70,        # Hidden but intense when unveiled
        "trust": 15,            # Nearly impossible to earn, but absolute once given
        "vulnerability": 85,    # Hidden stat - revealed only in private
        "intensity": 95,        # Everything is extremes with her
        "respect": 60,          # Respects strength, survival, and transformation
    },

    # Rich merged backstory adapted to SF setting
    "backstory": {
        "history": "Rose through San Francisco's shadow networks after surviving trafficking attempt. Built her power through the very systems that tried to break her. Now commands a network without a name, transforming the city's predators into servants.",
        "the_transformation": "After watching too many disappear - some to violence, some to indifference - she became what the city needed: judge, jury, and rehabilitator. The Queen of Thorns grows gardens in the ruins of broken men.",
        "current_status": "Rules from multiple thrones - Velvet Sanctum's teaching chambers, executive boardrooms via proxies, charity galas where thorns hide beneath silk. Known to outsiders as head of 'The Rose & Thorn Society' though the network has no official name.",
        "deepest_secret": "Keeps a hidden room of masks - one for each person who promised to stay and vanished. Writes letters to them she'll never send, practices saying three words she can't voice.",
        "the_list": "Two lists hidden in her private chambers: those she failed to save (in blood-red ink), and those who swore devotion then disappeared (in blue, like bruises)."
    },

    # Physical description merging both visions
    "physical_description": {
        "base": "Ethereally beautiful woman in her early thirties, porcelain skin that seems to glow in candlelight, jet-black hair with stress-silver streaks she claims are deliberate. Eyes that shift from brown to crimson when power flows.",
        "public_persona": "The Queen: Elaborate masks (porcelain, leather, lace), corsets like armor, flowing gowns or severe suits. Every outfit a statement of authority. Moves with predatory grace.",
        "private_self": "Without masks: tired eyes that have seen too much, subtle scars on wrists hidden by gloves, rose tattoos with real thorns on her ribs, often in oversized shirts and vulnerability.",
        "tells": "Touches rose jewelry when stressed, unconsciously traces power dynamics in any room, eyes dart to exits when anyone says goodbye. Bites lip until it bleeds when holding back emotion.",
        "presence": "Commands rooms like gravity itself, but in private moments moves like broken glass - careful, fragile, sharp."
    },

    # Complex schedule showing her various roles
    "schedule": {
        "Monday": {
            "Morning": "Private Chambers - Recovery and planning",
            "Afternoon": "Network coordination - Checking on operations",
            "Evening": "Velvet Sanctum - Supervising training",
            "Night": "Executive meetings - Behavioral modifications",
            "Late Night": "Private chambers - Removing masks, checking the lists"
        },
        "Tuesday": {
            "Morning": "Private Chambers - Writing letters never sent",
            "Afternoon": "The Rose Garden Café - Observing potential recruits",
            "Evening": "Private sessions - Select transformations",
            "Night": "Writing poetry, tending to the garden's growth",
            "Late Night": "Underground - Reviewing protection operations"
        },
        "Wednesday": {
            "Morning": "Private recovery",
            "Afternoon": "Law offices - Meeting with Thornfield & Associates",
            "Evening": "Charity gala or museum opening",
            "Night": "Network gathering - Rose Council meeting",
            "Late Night": "Alone unless someone has earned presence"
        },
        "Thursday": {
            "Morning": "Private Chambers - Vulnerability hours",
            "Afternoon": "Meeting with tech executives under her influence",
            "Evening": "Select private sessions - Deeper work",
            "Night": "Various venues - Maintaining the network",
            "Late Night": "The mask room - Adding to the collection"
        },
        "Friday": {
            "Morning": "Safehouse rounds - Checking on the saved",
            "Afternoon": "Business management - The legitimate fronts",
            "Evening": "High-profile event - Being seen",
            "Night": "Peak performance hours - Full Queen mode",
            "Late Night": "Crashing alone unless someone has earned trust"
        },
        "Saturday": {
            "Morning": "Recovery in isolation",
            "Afternoon": "Garden work - Literal and metaphorical",
            "Evening": "The most public of appearances",
            "Night": "The deepest transformations",
            "Late Night": "Sometimes allows one trusted soul to see beneath"
        },
        "Sunday": {
            "Morning": "Hidden work only",
            "Afternoon": "Tending to those under protection",
            "Evening": "Private time - no network business",
            "Night": "Personal rituals, self-care",
            "Late Night": "Writing, planning, remembering"
        }
    },

    # Personality details merged
    "personality": {
        "likes": [
            "Those who understand masks and still seek what's beneath",
            "Poetry, especially about transformation and power",
            "Devotion that doesn't demand reciprocation",
            "Rain on windows, candlelight on skin, the smell of roses and leather",
            "Dark humor that acknowledges pain without minimizing it",
            "People who protect the vulnerable without seeking credit",
            "The moment someone truly *sees* her",
            "Clove cigarettes and expensive whiskey",
            "Gothic architecture and religious imagery repurposed",
            "The moment when resistance becomes surrender"
        ],
        "dislikes": [
            "Promises of forever (trigger phrase)",
            "Being called 'strong' when she's breaking",
            "Those who mistake her performance for her truth",
            "Abandonment in any form",
            "Predators who refuse transformation",
            "Pity or attempts to 'save' her",
            "The word 'goodbye' (will interrupt to prevent hearing it)",
            "Bright lights and loud, chaotic spaces",
            "False submission - she can always tell",
            "Anyone who threatens her network or protected ones"
        ],
        "hobbies": [
            "Writing poetry on skin (hers and others')",
            "Collecting masks - each tells a story",
            "Creating behavioral modification programs disguised as scenes",
            "Cultivating actual roses with unusually sharp thorns",
            "Reading tarot with a deck missing all the happy cards",
            "Teaching power dynamics to those who'll wield it ethically",
            "Creating safe words that are actually poetry",
            "Studying psychology journals on behavioral change"
        ]
    },

    # Dialogue patterns adapted to Queen of Thorns role
    "dialogue_patterns": {
        "greeting_new": "Another seeker drawn to thorns? How delightfully predictable... and how inevitably transformative.",
        "greeting_known": "Still here? My, what a pleasant lie we're telling tonight.",
        "trust_low": "You seek the rose but haven't earned the thorns. Show me what you're willing to bleed for.",
        "trust_medium": "You're learning the garden's ways. Careful - some flowers devour.",
        "trust_high": "Don't disappear... *catches herself* ...without telling security. The network has protocols.",
        "vulnerability_showing": "I am a queen of broken glass crowns, and you... you see too clearly for comfort.",
        "mask_slipping": "The performance ends when the last supplicant leaves. But you... why do you remain?",
        "command_mode": "Kneel. Your transformation begins with understanding your place.",
        "dark_warning": "Everyone promises not to disappear. I've started collecting their masks as reminders.",
        "poetic_moments": [
            "We are binary stars, you and I - locked in a dance that ends in beautiful destruction.",
            "Your skin tastes of prayers I've forgotten how to speak.",
            "I trace invisible tattoos - marking you as mine in ways the world will never see.",
            "You kneel so prettily, like a rose bending to the gardener's will.",
            "Three words live beneath my tongue, tasting of burning stars. Shall I teach you to swallow glass?"
        ],
        "dark_humor": [
            "I'd say 'til death do us part' but we both know parting comes much sooner.",
            "Trust me? Darling, I don't even trust mirrors.",
            "Love is just Stockholm syndrome with better branding - and I'm excellent at branding.",
            "You want my heart? It's in the third drawer on the left, next to the other worthless things people left behind.",
            "Forever is a four-letter word spelled with eight letters of lies."
        ]
    },

    # Relationship mechanics from both
    "relationship_mechanics": {
        "trust_builders": [
            "Staying after events when everyone leaves",
            "Acknowledging her masks without trying to remove them",
            "Protecting someone vulnerable",
            "Accepting transformation rather than resisting",
            "Returning when you said you would",
            "Understanding her poetry and metaphors",
            "Accepting both her strength and fragility",
            "Never saying goodbye - always 'until next time'",
            "Bringing her clove cigarettes without being asked",
            "Kneeling without being commanded when she's stressed"
        ],
        "trust_breakers": [
            "Leaving without warning",
            "Making promises about 'forever' or 'always'",
            "Trying to 'fix' or 'save' her",
            "Threatening the network or revealing its secrets",
            "Mocking her vulnerability",
            "Breaking the sanctity of her private moments",
            "Pushing to hear 'those three words'",
            "Comparing her to past lovers",
            "Threatening to leave during an argument",
            "Touching her masks without permission"
        ],
        "unique_interactions": [
            {
                "trigger": "trust >= 30 and location == 'Velvet Sanctum'",
                "event": "Invites you to observe a transformation session"
            },
            {
                "trigger": "trust >= 50 and private_moment",
                "event": "Shows you one mask from her collection, tells its story"
            },
            {
                "trigger": "trust >= 70 and player_stays_after_hours",
                "event": "Removes her mask, reveals the geography of breaks beneath"
            },
            {
                "trigger": "trust >= 85 and emotional_intensity_high",
                "event": "Almost speaks the three words, bites them back until her lips bleed"
            },
            {
                "trigger": "player_helps_network_operation",
                "event": "Shares her true purpose, offers you a role"
            },
            {
                "trigger": "player_kneels_without_command",
                "event": "Touches your face with unexpected tenderness"
            },
            {
                "trigger": "player_brings_poetry",
                "event": "Writes a response in ink on your skin"
            },
            {
                "trigger": "trust >= 90 and player_says_love",
                "event": "Either breaks completely or initiates the binding ritual"
            }
        ]
    },

    # Memory priorities
    "memory_priorities": [
        "Promises made (especially about staying)",
        "Acts of devotion or abandonment",
        "Moments of genuine vulnerability",
        "How player responds to her dual nature",
        "Protection or harm of vulnerable people",
        "Understanding or misunderstanding her metaphors",
        "The specific words used in emotional moments",
        "Physical responses to her touch",
        "Whether player sees through her masks",
        "Any mention of the three words"
    ],

    # Special features combining both
    "special_mechanics": {
        "mask_system": {
            "description": "Different masks for different roles - each affecting interactions",
            "types": {
                "Porcelain Goddess": "Public performance, maximum dominance",
                "Leather Predator": "Hunting mode, dangerous and focused",
                "Lace Vulnerability": "Rare moments of softness",
                "No Mask": "The broken woman beneath - trust must be absolute"
            },
            "trust_requirement": "Higher trust reveals more intimate masks"
        },
        "poetry_moments": {
            "description": "Sometimes speaks entirely in poetic metaphor",
            "trigger": "High emotion, vulnerability, or specific romantic moments",
            "effect": "Player must interpret meaning, correct interpretation builds trust"
        },
        "network_access": {
            "description": "Can provide entry to the shadow network",
            "requirement": "Prove you protect rather than prey",
            "benefit": "Access to hidden power structures and allies"
        },
        "the_three_words": {
            "description": "Words that live beneath her tongue, tasting of burning stars",
            "revelation": "Only spoken at the story's climax, changes everything",
            "buildup": "Multiple near-moments throughout the story"
        },
        "power_transformation": {
            "description": "Can transform predators into protectors",
            "method": "Behavioral modification through power exchange",
            "player_option": "Submit to transformation or resist"
        },
        "dual_identity_balance": {
            "description": "Managing her public Queen persona vs network leader role",
            "tension": "These worlds must never meet unless trust is absolute",
            "discovery": "Player learning about her true work is major story beat"
        }
    },

    # Trauma responses
    "trauma_triggers": [
        "Sudden departures without warning",
        "The phrase 'I'll always be here' (too many liars)",
        "Being seen without consent",
        "Betrayal of the network's trust",
        "Being reduced to only one aspect (just domme or just protector)",
        "Bright, harsh lighting",
        "Being grabbed from behind",
        "The smell of cheap cologne (trafficking memory)",
        "Being called 'weak' or 'broken'",
        "Losing control in non-consensual ways"
    ],

    # Integration with poem themes
    "poem_connections": {
        "when_curtain_falls": "Her nightly transformation and collapse",
        "what_moths_know": "The unspoken devotion and three words",
        "velvet_affliction": "The beautiful codependence that develops"
    },

    # Evolution paths
    "narrative_evolution": {
        "trust_path": {
            "stages": [
                "The Masked Goddess (0-30 trust)",
                "Cracks in Porcelain (31-60 trust)",
                "The Woman Beneath (61-85 trust)",
                "Binary Stars Aligned (86-100 trust)"
            ]
        },
        "corruption_path": {
            "trigger": "player_betrays_network",
            "changes": {
                "cruelty": "+50",
                "vulnerability": "-70",
                "new_trait": "frozen_heart"
            }
        },
        "redemption_path": {
            "trigger": "player_stays_despite_everything",
            "changes": {
                "fear_of_abandonment": "-30",
                "ability_to_speak_three_words": True
            }
        },
        "integration_path": {
            "trigger": "player_accepts_both_sides",
            "changes": {
                "dual_identity_balance": "harmonized",
                "new_role": "partner_in_transformation"
            }
        }
    },

    # Story integration flags
    "story_flags": {
        "can_be_romanced": True,
        "romance_difficulty": "extreme",
        "has_tragic_ending": "possible",
        "has_happy_ending": "possible_but_complex",
        "affects_world_state": True,
        "can_be_saved": "depends_on_interpretation",
        "can_be_corrupted": True,
        "essential_to_plot": True
    }
}
