# story_templates/moth/the_moth_and_flame.py
from story_templates.preset_stories import PresetStory, StoryBeat

# The foundational poems that define tone, imagery, and themes
MOTH_FLAME_POEMS = {
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

THE_MOTH_AND_FLAME = PresetStory(
    id="the_moth_and_flame",
    name="The Moth and the Flame",
    theme="A gothic romance exploring masks, vulnerability, and the consuming nature of devotion",
    synopsis="You encounter a mysterious dominatrix who rules the night, but beneath her porcelain mask lies a soul haunted by abandonment. As you're drawn deeper into her world, you must navigate the delicate balance between worship and understanding.",
    
    # Include poems and tone instructions
    source_material={
        "poems": MOTH_FLAME_POEMS,
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
            "beats": ["after_hours", "glimpse_beneath", "the_confession", "binding_words"]
        },
        {
            "act": 3,
            "name": "Velvet Affliction",
            "beats": ["the_test", "breaking_point", "true_devotion", "eternal_dance"]
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
                "time": "night"
            },
            required_npcs=["the_queen"],
            required_locations=["velvet_sanctum", "main_stage"],
            narrative_stage="Innocent Beginning",
            outcomes={
                "relationship_added": {"npc": "the_queen", "type": "intrigued"},
                "player_stats": {"lust": +20, "curiosity": +30}
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
                "location": "velvet_sanctum"
            },
            required_npcs=["the_queen", "devoted_pilgrim"],
            required_locations=["velvet_sanctum"],
            narrative_stage="Innocent Beginning",
            outcomes={
                "learned_fact": "queen_performs_nightly",
                "player_stats": {"arousal": +15, "submission": +10}
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
                "queen_awareness": {"min": 20}
            },
            required_npcs=["the_queen"],
            required_locations=["velvet_sanctum"],
            narrative_stage="First Doubts",
            outcomes={
                "quest_added": "answer_her_summons",
                "item_received": "obsidian_token"
            },
            dialogue_hints=[
                "I am the flame she cages in her dark",
                "Her fingers trace invisible tattoos",
                "Each touch a brand, each word a binding kiss"
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
                "location": "velvet_sanctum"
            },
            required_npcs=["the_queen"],
            required_locations=["empty_sanctum", "private_chambers"],
            narrative_stage="Creeping Realization",
            outcomes={
                "discovered_secret": "queen_loneliness",
                "relationship_progress": {"the_queen": +25}
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
                "relationship": {"the_queen": {"min": 50}},
                "private_moment": True
            },
            required_npcs=["the_queen"],
            required_locations=["private_chambers"],
            narrative_stage="Veil Thinning",
            outcomes={
                "learned_truth": "queen_past_abandonment",
                "player_stats": {"empathy": +30, "devotion": +25}
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
                "trust_established": True
            },
            required_npcs=["the_queen"],
            required_locations=["private_chambers"],
            narrative_stage="Veil Thinning",
            outcomes={
                "choice_presented": "promise_to_stay",
                "relationship_evolution": "deeper_connection"
            },
            dialogue_hints=[
                "Don't disappear, she whispers to the dark",
                "Be mine, she breathes, each word a barbed request",
                "The ones who swore to stay yet disappeared"
            ],
            can_skip=False
        ),
        
        # Act 3 Beats
        StoryBeat(
            id="the_test",
            name="Binary Stars",
            description="She tests your devotion through increasingly intense sessions",
            trigger_conditions={
                "act": 3,
                "devotion": {"min": 60}
            },
            required_npcs=["the_queen"],
            required_locations=["sanctum_dungeon"],
            narrative_stage="Full Revelation",
            outcomes={
                "player_stats": {"submission": +40, "pain_tolerance": +20},
                "relationship_dynamic": "total_power_exchange"
            },
            dialogue_hints=[
                "We are binary stars",
                "I orbit at your pleasure",
                "You hold me; lunar edict"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="breaking_point",
            name="The Unopened Letter",
            description="The moment where unspoken truths must be revealed",
            trigger_conditions={
                "emotional_intensity": {"min": 90},
                "sessions_completed": {"min": 10}
            },
            required_npcs=["the_queen"],
            required_locations=["private_chambers"],
            narrative_stage="Full Revelation",
            outcomes={
                "major_choice": "speak_truth_or_maintain_silence"
            },
            dialogue_hints=[
                "There are words that live beneath my tongue",
                "I want to shake you by the shoulders",
                "Let me breathe it—once—into the air between us"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="eternal_dance",
            name="Velvet Affliction",
            description="The final form of your relationship, shaped by your choices",
            trigger_conditions={
                "major_choice_made": True,
                "story_complete": 90
            },
            required_npcs=["the_queen"],
            required_locations=["varies_by_choice"],
            narrative_stage="Full Revelation",
            outcomes={
                "ending_achieved": True,
                "relationship_final": "velvet_affliction"
            },
            dialogue_hints=[
                "I crave you like strays worship moonlit feasts",
                "Within these ruins we have raised together",
                "Carved in basalt, under vault and bone"
            ],
            can_skip=False
        )
    ],
    
    # The main NPC - The Queen
    required_npcs=[
        {
            "id": "the_queen",
            "name": "Lilith Ravencroft",  # You can change this name
            "archetype": "Wounded Goddess",
            "traits": [
                "dominant", "vulnerable_beneath_mask", "fear_of_abandonment",
                "darkly_poetic", "intensely_possessive", "hauntingly_beautiful"
            ],
            "role": "Dominatrix Queen with Hidden Depths",
            "stats": {
                "dominance": 95,
                "cruelty": 60,  # Varies based on mood
                "affection": 70,  # Hidden but intense
                "trust": 20,  # Very hard to earn
                "vulnerability": 85  # Hidden stat
            },
            "physical_description": "Porcelain skin, dark hair like spilled ink, eyes that burn with crimson intensity. Always masked in public.",
            "personality_patterns": [
                "Maintains perfect control in public, crumbles in private",
                "Uses pain and pleasure to test loyalty",
                "Speaks in poetic riddles when emotional",
                "Becomes possessive when attachment forms"
            ],
            "trauma_triggers": [
                "People leaving suddenly",
                "Broken promises",
                "Being seen without her mask by strangers"
            ],
            "schedule": {
                "evening": "Velvet Sanctum - Preparing",
                "night": "Velvet Sanctum - Performing",
                "late_night": "Private Chambers - Alone",
                "dawn": "Private Chambers - Vulnerable"
            }
        },
        {
            "id": "devoted_pilgrim",
            "name": "Marcus",
            "archetype": "Devoted Submissive",
            "traits": ["worshipful", "jealous_of_newcomers", "completely_broken"],
            "role": "Example of total devotion/warning"
        }
    ],
    
    # Required locations
    required_locations=[
        {
            "name": "Velvet Sanctum",
            "type": "nightclub_dungeon",
            "description": "An underground temple where pain becomes prayer, hidden beneath the city",
            "areas": ["main_stage", "private_booths", "sanctum_dungeon"]
        },
        {
            "name": "Empty Sanctum",
            "type": "afterhours_venue",
            "description": "The same space when the music dies and shadows lengthen"
        },
        {
            "name": "Private Chambers",
            "type": "personal_space",
            "description": "Her private sanctuary, where masks can finally fall"
        }
    ],
    
    # Dynamic elements
    dynamic_elements={
        "minor_npcs": True,  # Other patrons, submissives
        "side_sessions": True,  # Various BDSM activities
        "emotional_progression": True,  # Relationship deepens naturally
        "mask_metaphors": True,  # Various masks and what they represent
        "poetry_moments": True  # Moments of lyrical beauty
    },
    
    # Key player choice points
    player_choices_matter=[
        "response_to_vulnerability",  # How you react when she shows weakness
        "promise_to_stay",  # Whether you promise not to disappear
        "depth_of_submission",  # How far you're willing to go
        "speak_truth_or_maintain_silence",  # The three words
        "final_devotion"  # Your ultimate choice
    ],
    
    flexibility_level=0.3,  # Low flexibility - this is a specific narrative
    enforce_ending=False  # Multiple endings based on choices
)
