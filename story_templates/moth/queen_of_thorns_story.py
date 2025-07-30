# story_templates/moth/queen_of_thorns_story.py
"""
The Moth and Flame - A Gothic Romance
Complete story definition with poem integration and complex character dynamics
"""

from story_templates.preset_stories import PresetStory, StoryBeat
from story_templates.character_profiles.lilith_ravencroft import LILITH_RAVENCROFT

# The foundational poems that define tone, imagery, and themes
THORNS_POEMS = {
    "when_curtain_falls": """When the Curtain Falls

Behind porcelain curves, a painted smile,
A queen of thorns holds court in velvet night.
Her mask, a fortress forged from glass and guile,
Reflects their hunger back as blinding light.

They worship at the altar of her throne,
Pilgrims who seek their pain as benediction.
But masks, like mirrors, only show what's shown:
The goddess role, flawless and cold fiction.

She dances nightly on a wire of fear,
Above an audience of ghosts, of shadows,
The ones who swore to stay yet disappeared,
Gone like the mist when morning finds the field.

The music dies. The temple empties out.
The candles gutter in their silver stands.
And there she sits, consumed by bone-deep doubt,
The mask now heavy in her trembling hands.

I watch her lift it slowly from her face,
This shell of strength, this armor made of lies.
Beneath: a rough geography of breaks,
The voiceless howl that haunts behind her eyes.

She is a moth with wings of broken glass,
A clock whose hands have frozen on goodbye.
I am the flame she cages in her dark,
The tide she's turned against its native sky.

Her fingers trace invisible tattoos
Across my skin, her marks that others miss.
She burns her name in shadows steeped in blue,
Each touch a brand, each word a binding kiss.

"Don't disappear," she whispers to the dark,
Her voice a thread unraveling with time.
"Be mine," she breathes, each word a barbed request,
A prayer disguised inside a stolen rhyme.

She pulls me closer with her gravity,
This queen who's learned to rule through letting go.
Her hunger writes itself across my skin,
The script we're never meant to understand.

So here we sit when curtains close on night,
Her armor shed, my conscience searing white.
I hold her trembling till the sky turns light
Two tides that found their shore despite the night""",

    "what_moths_know": """What the Moths Know

There are words that live beneath my tongue,
three syllables tasting of burning stars.
They flutter; moths against the dark, but I bite down until it shatters.

You are polished obsidian.
I find myself reversed, revealed.
Undone, soft ash.
My mouth full of glass,
your hand hovering above the board.
I wait in this sweet paralysis,
tasting copper on slit lips,
drowning on unsaid copper.

I am yours the way rain
surrenders to gravity.
Helpless in fall.

We are binary stars:
yet you burn brighter, unmake time.
I orbit at your pleasure,
writing psalms no ear hears.
Some nights I scratch them on my skin,
three words carved in the dark
until even bone remembers.

What vow binds blood already bound?
You hold me; lunar edict.
Pull.
      Release.
Rise.
      Fall.

I want to shake you by the shoulders.
To hell with consequence.
Let me breathe it—once—
into the air between us.
But I pocket these truths instead:
prayer beads sand-worn smooth from silence.
What the moths know, wings against glass:
we exist to witness the flame.

You are my unopened letter.
My unsung hymn.
The altar where I'd kneel.

Between heartbeats
I practice genuflection.
Dust motes, still offerings
in amber light. There we live.

I, your willing moth.
You, my sanctified ruin.

My sweetest fall.

My silent scream.""",

    "velvet_affliction": """Velvet Affliction

You are the lighthouse on my midnight tide,
A crimson flare where moon and madness collide.
Like lipstick sigils seeping through my phone,
Blood-red promises, I'm never alone.

In our tempest-tossed and sinking boat,
You cast the line that keeps me afloat.
My mind's a lightning storm that cracks and arcs,
Yet you're the ground that catches all my sparks.

Like feral cats we tempt with tender care,
My heart prowled shadows until you found it there.
Now tamed within your castle in the void,
Where we feast as vampires, unrestrained.

You call yourself a storm, depressed, untamed,
Yet in your tempest I have been reclaimed.
You teach me rain can taste of wine,
That beautiful chaos becomes divine.

Your gallows humor when the world's too grim
Sutures the wounds bleeding at the brim.
Like poetry sleeping in your veins for years,
These verses wake and summon back your tears.

I crave you like strays worship moonlit feasts,
Like gothic souls need beautiful beasts.
My muse, my madness, my sweet reprieve
The only ghost story I believe.

Within these ruins we have raised together,
Where every kiss defies the mortal tether,
Know this: I'm your velvet affliction too,
Your devoted chaos, fierce and true.

Carved in basalt, under vault and bone,
You've claimed me, body, soul, and stone."""
}

# Story prompt instructions for AI to maintain tone
STORY_TONE_PROMPT = """
This story is inspired by gothic romantic poetry exploring themes of:
- Masks and vulnerability
- The duality between public dominance and private fragility  
- Moth and flame metaphors for dangerous attraction
- The fear of abandonment hidden beneath control
- Devotion that borders on religious experience
- The unspoken words that burn beneath the tongue

Use imagery from the poems including:
- Porcelain, glass, mirrors, masks
- Moths, flames, burning, ash
- Velvet, thorns, blood-red, crimson
- Temples, altars, prayer, worship
- Binary stars, gravity, orbits
- Storms, lightning, tempests
- Ruins, broken glass, shattered things
- Shadows, darkness, midnight

The language should be:
- Poetic and metaphorical
- Gothic and darkly romantic
- Intense and visceral
- Using religious imagery for BDSM contexts
- Switching between commanding dominance and whispered vulnerability

Key phrases to echo:
- "Don't disappear"
- "Be mine"
- "Three syllables" / "Three words"
- References to masks, moths, flames
- "Velvet affliction"
- "My sanctified ruin"
"""

QUEEN_OF_THORNS_STORY = PresetStory(
    id="queen_of_thorns",
    name="Queen of Thorns",
    theme="A gothic romance exploring masks, vulnerability, and the consuming nature of devotion",
    synopsis="You encounter a mysterious dominatrix who rules the night, but beneath her porcelain mask lies a soul haunted by abandonment. As you're drawn deeper into her world, you must navigate the delicate balance between worship and understanding.",
    
    # Include poems and tone instructions
    source_material={
        "poems": THORNS_POEMS,
        "tone_prompt": STORY_TONE_PROMPT,
        "reference_style": "gothic_poetry"
    },
    
    # Act structure
    acts=[
        {
            "act": 1,
            "name": "Behind Porcelain Curves",
            "beats": ["first_glimpse", "the_performance", "invitation", "first_session"]
        },
        {
            "act": 2,
            "name": "When the Curtain Falls",
            "beats": ["after_hours", "glimpse_beneath", "the_confession", "binding_words", "dual_life_discovery"]
        },
        {
            "act": 3,
            "name": "Velvet Affliction",
            "beats": ["the_test", "safehouse_crisis", "breaking_point", "true_devotion", "eternal_dance"]
        }
    ],
    
    # Detailed story beats
    story_beats=[
        # Act 1 Beats
        StoryBeat(
            id="first_glimpse",
            name="The Temple of Thorns",
            description="Player first encounters the mysterious dominatrix at an underground venue",
            trigger_conditions={
                "game_start": True,
                "time": "night",
                "location": "underground_district"
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["velvet_sanctum"],
            narrative_stage="Innocent Beginning",
            outcomes={
                "relationship_added": {"npc": "lilith_ravencroft", "type": "intrigued"},
                "player_stats": {"lust": "+20", "curiosity": "+30"},
                "location_unlocked": "velvet_sanctum",
                "knowledge_gained": "sanctum_exists"
            },
            dialogue_hints=[
                "A queen of thorns holds court in velvet night",
                "Her mask, a fortress forged from glass and guile",
                "They worship at the altar of her throne"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="the_performance",
            name="Pilgrims and Pain",
            description="Witness her commanding presence as she holds court",
            trigger_conditions={
                "completed_beats": ["first_glimpse"],
                "location": "velvet_sanctum",
                "time": "night"
            },
            required_npcs=["lilith_ravencroft", "devoted_pilgrim"],
            required_locations=["velvet_sanctum", "throne_room"],
            narrative_stage="Innocent Beginning",
            outcomes={
                "learned_fact": "queen_performs_nightly",
                "player_stats": {"arousal": "+15", "submission": "+10"},
                "npc_awareness": {"lilith_ravencroft": "+10"}
            },
            dialogue_hints=[
                "Pilgrims who seek their pain as benediction",
                "The goddess role, flawless and cold fiction",
                "Reflects their hunger back as blinding light"
            ],
            can_skip=True
        ),
        
        StoryBeat(
            id="invitation",
            name="Marked by Shadow",
            description="She notices you watching and extends a cryptic invitation",
            trigger_conditions={
                "times_visited_sanctum": 3,
                "npc_awareness": {"lilith_ravencroft": {"min": 20}},
                "player_watched_performance": True
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["velvet_sanctum"],
            narrative_stage="First Doubts",
            outcomes={
                "quest_added": "answer_her_summons",
                "item_received": "obsidian_token",
                "relationship_progress": {"lilith_ravencroft": "+15"}
            },
            dialogue_hints=[
                "I am the flame she cages in her dark",
                "Her fingers trace invisible tattoos",
                "Each touch a brand, each word a binding kiss"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="first_session",
            name="The Altar of Submission",
            description="Your first private session with the Queen",
            trigger_conditions={
                "quest_completed": "answer_her_summons",
                "has_item": "obsidian_token",
                "location": "velvet_sanctum_private_booth"
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["private_session_room"],
            narrative_stage="First Doubts",
            outcomes={
                "relationship_type": {"lilith_ravencroft": "submissive_to"},
                "player_stats": {"submission": "+25", "pain_threshold": "+10"},
                "learned_skill": "proper_kneeling"
            },
            dialogue_hints=[
                "On. Your. Knees. This is not a request, little moth.",
                "You kneel so prettily, like a candle flame bowing to the wind",
                "I trace invisible tattoos - marking you as mine"
            ],
            can_skip=False
        ),
        
        # Act 2 Beats
        StoryBeat(
            id="after_hours",
            name="When Music Dies",
            description="You find her alone after the venue closes",
            trigger_conditions={
                "act": 2,
                "time": "late_night",
                "location": "velvet_sanctum",
                "relationship": {"lilith_ravencroft": {"min": 30}},
                "sanctum_closed": True
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["empty_sanctum"],
            narrative_stage="Creeping Realization",
            outcomes={
                "discovered_secret": "queen_loneliness",
                "relationship_progress": {"lilith_ravencroft": "+25"},
                "vulnerability_witnessed": True
            },
            dialogue_hints=[
                "The music dies. The temple empties out",
                "And there she sits, consumed by bone-deep doubt",
                "The mask now heavy in her trembling hands"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="glimpse_beneath",
            name="Geography of Breaks",
            description="She removes her mask, revealing vulnerability",
            trigger_conditions={
                "relationship": {"lilith_ravencroft": {"min": 50}},
                "private_moment": True,
                "trust_established": True
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["private_chambers"],
            narrative_stage="Veil Thinning",
            outcomes={
                "learned_truth": "queen_past_abandonment",
                "player_stats": {"empathy": "+30", "devotion": "+25"},
                "mask_removed": "first_time",
                "secret_discovered": "mask_collection"
            },
            dialogue_hints=[
                "I watch her lift it slowly from her face",
                "Beneath: a rough geography of breaks",
                "She is a moth with wings of broken glass"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="the_confession",
            name="Don't Disappear",
            description="She reveals her deepest fear - abandonment",
            trigger_conditions={
                "intimacy_level": {"min": 70},
                "trust_established": True,
                "mask_removed": "first_time"
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["private_chambers"],
            narrative_stage="Veil Thinning",
            outcomes={
                "choice_presented": "promise_to_stay",
                "relationship_evolution": "deeper_connection",
                "learned_secret": "the_lists"
            },
            dialogue_hints=[
                "Don't disappear, she whispers to the dark",
                "Be mine, she breathes, each word a barbed request",
                "The ones who swore to stay yet disappeared"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="dual_life_discovery",
            name="The Moth Queen's Secret",
            description="Discover her role as underground protector",
            trigger_conditions={
                "relationship": {"lilith_ravencroft": {"min": 60}},
                "helped_vulnerable_npc": True,
                "time": "late_night"
            },
            required_npcs=["lilith_ravencroft", "rescued_victim"],
            required_locations=["safehouse_entrance"],
            narrative_stage="Veil Thinning",
            outcomes={
                "discovered_secret": "moth_queen_identity",
                "location_unlocked": "safehouse_network",
                "new_quest": "help_the_underground"
            },
            dialogue_hints=[
                "They call me 'The Moth Queen' in the underground",
                "Beautiful, dangerous, drawn to flames",
                "I save those like I once was - lost, hunted, disposable"
            ],
            can_skip=True
        ),
        
        # Act 3 Beats
        StoryBeat(
            id="the_test",
            name="Binary Stars",
            description="She tests your devotion through increasingly intense sessions",
            trigger_conditions={
                "act": 3,
                "devotion": {"min": 60},
                "sessions_completed": {"min": 5}
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["sanctum_dungeon"],
            narrative_stage="Full Revelation",
            outcomes={
                "player_stats": {"submission": "+40", "pain_tolerance": "+20"},
                "relationship_dynamic": "total_power_exchange",
                "gained_title": "her_moth"
            },
            dialogue_hints=[
                "We are binary stars",
                "I orbit at your pleasure",
                "You hold me; lunar edict"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="safehouse_crisis",
            name="When Both Worlds Collide",
            description="A crisis forces her two worlds to intersect",
            trigger_conditions={
                "discovered_secret": "moth_queen_identity",
                "random_event": "trafficking_ring_discovered"
            },
            required_npcs=["lilith_ravencroft", "trafficking_victim", "ring_enforcer"],
            required_locations=["safehouse_network", "velvet_sanctum"],
            narrative_stage="Full Revelation",
            outcomes={
                "choice_presented": "help_or_stay_safe",
                "lilith_vulnerability": "maximum",
                "potential_loss": True
            },
            dialogue_hints=[
                "I can't lose another one, not again",
                "This is who I really am beneath it all",
                "Stay with me through this, please... don't disappear"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="breaking_point",
            name="The Unopened Letter",
            description="The moment where unspoken truths must be revealed",
            trigger_conditions={
                "emotional_intensity": {"min": 90},
                "sessions_completed": {"min": 10},
                "crisis_resolved": True
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["private_chambers", "the_mask_room"],
            narrative_stage="Full Revelation",
            outcomes={
                "major_choice": "speak_truth_or_maintain_silence",
                "three_words_moment": True
            },
            dialogue_hints=[
                "There are words that live beneath my tongue",
                "I want to shake you by the shoulders",
                "Let me breathe it—once—into the air between us"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="true_devotion",
            name="The Binding Ritual",
            description="A ceremony of permanent connection",
            trigger_conditions={
                "major_choice": "speak_truth",
                "mutual_confession": True
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["velvet_sanctum_altar"],
            narrative_stage="Full Revelation",
            outcomes={
                "permanent_bond": True,
                "gained_item": "matching_moth_tattoos",
                "new_role": "eternal_moth"
            },
            dialogue_hints=[
                "You are my unopened letter, finally read",
                "Carved in basalt, under vault and bone",
                "My sweetest fall, my silent scream"
            ],
            can_skip=True
        ),
        
        StoryBeat(
            id="eternal_dance",
            name="Velvet Affliction",
            description="The final form of your relationship, shaped by your choices",
            trigger_conditions={
                "major_choice_made": True,
                "story_complete": 90
            },
            required_npcs=["lilith_ravencroft"],
            required_locations=["varies_by_choice"],
            narrative_stage="Full Revelation",
            outcomes={
                "ending_achieved": True,
                "relationship_final": "velvet_affliction"
            },
            dialogue_hints=[
                "I crave you like strays worship moonlit feasts",
                "Within these ruins we have raised together",
                "You've claimed me, body, soul, and stone"
            ],
            can_skip=False
        )
    ],
    
    # Required NPCs
    required_npcs=[
        LILITH_RAVENCROFT,  # Use the complete character profile
        {
            "id": "devoted_pilgrim",
            "name": "Marcus Sterling",
            "archetype": "Devoted Submissive",
            "traits": ["worshipful", "jealous_of_newcomers", "completely_broken", "wealthy"],
            "role": "Example of total devotion/warning",
            "stats": {
                "dominance": 5,
                "submission": 95,
                "jealousy": 80,
                "devotion": 100
            },
            "personality": {
                "likes": ["serving the Queen", "public humiliation", "being used as example"],
                "dislikes": ["new submissives", "being ignored", "others getting attention"],
                "hobbies": ["collecting Queen's used items", "writing devotional poetry"]
            },
            "schedule": {
                "Evening": "Waiting outside Velvet Sanctum",
                "Night": "Kneeling in the main chamber",
                "Late Night": "Cleaning the sanctum"
            }
        },
        {
            "id": "rescued_victim",
            "name": "Sarah Chen",
            "archetype": "Trafficking Survivor",
            "traits": ["traumatized", "grateful", "suspicious", "healing"],
            "role": "Reveals Lilith's other life",
            "stats": {
                "trust": 20,
                "fear": 70,
                "gratitude": 85
            }
        },
        {
            "id": "ring_enforcer",
            "name": "Viktor Kozlov",
            "archetype": "Dangerous Predator",
            "traits": ["violent", "calculating", "misogynistic"],
            "role": "Antagonist threatening Lilith's work",
            "stats": {
                "dominance": 90,
                "cruelty": 95,
                "intelligence": 70
            }
        }
    ],
    
    # Required locations
    required_locations=[
        {
            "name": "Velvet Sanctum",
            "type": "nightclub_dungeon",
            "description": "An underground temple where pain becomes prayer, hidden beneath the city's skin",
            "areas": {
                "main_stage": "Where the Queen holds court before her subjects",
                "throne_room": "Her seat of power, draped in velvet and shadow",
                "private_booths": "Intimate spaces for personal worship",
                "sanctum_dungeon": "The deepest level where true devotion is tested",
                "preparation_chamber": "Where the Queen becomes the Goddess"
            },
            "schedule": {
                "Monday": {"Evening": "open", "Night": "performances", "Late Night": "private sessions"},
                "Tuesday": {"Evening": "closed", "Night": "private clients only"},
                "Wednesday": {"Evening": "open", "Night": "grand performance", "Late Night": "exclusive gathering"},
                "Thursday": {"Evening": "open", "Night": "themed nights"},
                "Friday": {"Evening": "open", "Night": "the Queen's court", "Late Night": "devotional ceremonies"},
                "Saturday": {"Evening": "open", "Night": "busiest night", "Late Night": "special sessions"},
                "Sunday": {"All Day": "closed to public"}
            }
        },
        {
            "name": "Empty Sanctum",
            "type": "afterhours_venue",
            "description": "The same space when the music dies and shadows lengthen. Candles gutter, ghosts linger.",
            "atmosphere": "melancholic",
            "unique_events": ["mask_removal", "vulnerability_moments"]
        },
        {
            "name": "Private Chambers",
            "type": "personal_space",
            "description": "Her private sanctuary where masks can finally fall. Moths dance against windows.",
            "areas": {
                "the_mask_room": "Walls lined with porcelain faces - each a broken promise",
                "writing_desk": "Where letters to ghosts pile like autumn leaves",
                "bedroom": "Rarely used, too many memories in empty sheets",
                "hidden_room": "The true heart of her pain"
            }
        },
        {
            "name": "Safehouse Network",
            "type": "secret_location",
            "description": "Hidden passages and safe rooms throughout the city",
            "areas": {
                "entrance_points": "Disguised as various businesses",
                "transition_houses": "Where the saved learn to live again",
                "medical_station": "For those who arrive broken",
                "planning_room": "Where the Moth Queen wages her secret war"
            }
        }
    ],
    
    # Required conflicts
    required_conflicts=[
        {
            "id": "internal_war",
            "name": "The Mask and the Woman",
            "type": "psychological",
            "description": "Lilith's struggle between her public persona and private self"
        },
        {
            "id": "trust_vs_fear",
            "name": "The Promise of Staying",
            "type": "emotional",
            "description": "Fear of abandonment vs desire for connection"
        },
        {
            "id": "two_worlds",
            "name": "Queen and Savior",
            "type": "external",
            "description": "Balancing her dominatrix life with her rescue work"
        },
        {
            "id": "the_three_words",
            "name": "Words That Burn",
            "type": "romantic",
            "description": "The inability to speak love aloud"
        }
    ],
    
    # Dynamic elements
    dynamic_elements={
        "minor_npcs": True,  # Other patrons, submissives, rescued victims
        "side_sessions": True,  # Various BDSM activities and scenes
        "emotional_progression": True,  # Relationship deepens naturally
        "mask_metaphors": True,  # Various masks and what they represent
        "poetry_moments": True,  # Moments of lyrical beauty
        "underground_missions": True,  # Help with her rescue work
        "trust_tests": True,  # Various challenges to prove devotion
        "vulnerability_windows": True  # Rare moments to see beneath
    },
    
    # Key player choice points
    player_choices_matter=[
        "response_to_vulnerability",  # How you react when she shows weakness
        "promise_to_stay",  # Whether you promise not to disappear
        "depth_of_submission",  # How far you're willing to go
        "help_or_stay_safe",  # Whether to help with dangerous rescue work
        "speak_truth_or_maintain_silence",  # The three words
        "moth_or_flame",  # Your role in the dynamic
        "accept_both_sides",  # Whether you can love all of her
        "final_devotion"  # Your ultimate choice
    ],
    
    flexibility_level=0.3,  # Low flexibility - this is a specific narrative
    enforce_ending=False,  # Multiple endings based on choices
    
    # Special story mechanics
    special_mechanics={
        "trust_building": {
            "description": "Complex trust mechanics with specific triggers",
            "implementation": "Track actions that build/break trust"
        },
        "mask_progression": {
            "description": "Different masks unlock at different trust levels",
            "types": ["porcelain", "leather", "lace", "none"]
        },
        "poetry_interpretation": {
            "description": "Player must understand poetic dialogue",
            "reward": "Deeper connection when understood"
        },
        "dual_reputation": {
            "description": "Separate reputation in sanctum vs underground",
            "balance": "Actions in one affect the other"
        },
        "emotional_intensity": {
            "description": "Tracks the emotional weight of scenes",
            "climax": "Triggers major story moments"
        }
    }
)
