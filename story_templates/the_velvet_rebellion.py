# story_templates/the_velvet_rebellion.py
from story_templates.preset_stories import PresetStory, StoryBeat

THE_VELVET_REBELLION = PresetStory(
    id="the_velvet_rebellion",
    name="The Velvet Rebellion",
    theme="A town's hidden matriarchy slowly reveals itself as you try to establish independence",
    synopsis="You arrive in Havenbrook seeking to start your own business, but discover the town's economy is controlled by an elegant conspiracy of powerful women.",
    
    # Act structure
    acts=[
        {
            "act": 1,
            "name": "Fresh Ambitions",
            "beats": ["arrival", "business_permit", "first_obstacle", "helpful_mentor"]
        },
        {
            "act": 2,
            "name": "Silk Strings",
            "beats": ["supply_issues", "competitor_pressure", "council_meeting", "the_offer"]
        },
        {
            "act": 3,
            "name": "The Price of Success",
            "beats": ["final_terms", "rebellion_attempt", "consequences", "new_equilibrium"]
        }
    ],
    
    # Detailed story beats
    story_beats=[
        # Act 1 Beats
        StoryBeat(
            id="arrival",
            name="Welcome to Havenbrook",
            description="Player arrives with plans to open an independent business",
            trigger_conditions={
                "game_start": True
            },
            required_npcs=["mayor_elena", "assistant_clara"],
            required_locations=["town_square", "mayors_office"],
            narrative_stage="Innocent Beginning",
            outcomes={
                "player_goal": "open_business",
                "town_reputation": 50,
                "independence_score": 100
            },
            dialogue_hints=[
                "Welcome to our thriving community",
                "We're always excited about new entrepreneurs",
                "Just need to follow proper procedures"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="business_permit",
            name="The Permit Process",
            description="Player discovers getting permits requires 'recommendations'",
            trigger_conditions={
                "completed_beats": ["arrival"],
                "location": "mayors_office"
            },
            required_npcs=["mayor_elena", "assistant_clara"],
            required_locations=["mayors_office"],
            narrative_stage="Innocent Beginning",
            outcomes={
                "quest_added": "get_three_recommendations",
                "independence_score": -10
            },
            dialogue_hints=[
                "Three established businesswomen must vouch for you",
                "It's for the community's protection",
                "I'm sure they'll be reasonable"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="first_obstacle",
            name="Supplier Difficulties",
            description="Player finds all suppliers require 'special arrangements'",
            trigger_conditions={
                "quest_active": "get_three_recommendations",
                "days_passed": 2
            },
            required_npcs=["supplier_victoria"],
            required_locations=["warehouse_district"],
            narrative_stage="First Doubts",
            outcomes={
                "player_stats": {"confidence": -10, "independence": -10},
                "learned_fact": "supplier_network_controlled"
            },
            dialogue_hints=[
                "I'd love to help, but without a council seal...",
                "Perhaps we could work out an... alternative arrangement",
                "You scratch my back, I'll scratch yours"
            ],
            can_skip=True
        ),
        
        StoryBeat(
            id="helpful_mentor",
            name="An Understanding Friend",
            description="A successful businesswoman offers to mentor the player",
            trigger_conditions={
                "independence_score": {"max": 80},
                "location": ["cafe", "town_square"]
            },
            required_npcs=["mentor_isabella"],
            required_locations=["belladonna_cafe"],
            narrative_stage="First Doubts",
            outcomes={
                "relationship_added": {"npc": "mentor_isabella", "type": "mentor"},
                "quest_progress": {"get_three_recommendations": 1}
            },
            dialogue_hints=[
                "You remind me of myself when I first arrived",
                "Let me show you how things really work here",
                "Success requires... flexibility"
            ],
            can_skip=False
        ),
        
        # Act 2 Beats
        StoryBeat(
            id="supply_issues",
            name="The Supply Chain Squeeze",
            description="Player's supplies mysteriously become unavailable",
            trigger_conditions={
                "act": 2,
                "business_started": True
            },
            required_npcs=["supplier_victoria", "competitor_sarah"],
            required_locations=["warehouse_district", "player_business"],
            narrative_stage="Creeping Realization",
            outcomes={
                "business_health": -30,
                "player_stats": {"willpower": -15, "dependency": +20}
            },
            dialogue_hints=[
                "So sorry, but priority customers come first",
                "Perhaps if you joined the Merchant's Circle...",
                "Ms. Winters mentioned you haven't been... cooperative"
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="council_meeting",
            name="Called Before the Council",
            description="The player is summoned to a private council meeting",
            trigger_conditions={
                "business_health": {"max": 70},
                "dependency": {"min": 30}
            },
            required_npcs=["mayor_elena", "council_member_1", "council_member_2"],
            required_locations=["council_chambers"],
            narrative_stage="Creeping Realization",
            outcomes={
                "revealed_info": "council_true_nature",
                "player_stats": {"corruption": +20}
            },
            dialogue_hints=[
                "We've been watching your struggles with interest",
                "Havenbrook thrives because we work together",
                "Stop fighting the current and let us guide you to success"
            ],
            can_skip=False
        ),
        
        # Act 3 Beats
        StoryBeat(
            id="final_terms",
            name="The Velvet Contract",
            description="The council presents their final offer",
            trigger_conditions={
                "act": 3,
                "council_relationship": {"min": -20}
            },
            required_npcs=["mayor_elena", "mentor_isabella", "supplier_victoria"],
            required_locations=["council_chambers"],
            narrative_stage="Veil Thinning",
            outcomes={
                "choice_presented": "accept_or_rebel"
            },
            dialogue_hints=[
                "Sign this, and your business will thrive beyond imagination",
                "You'll have a seat at our table... eventually",
                "Or continue struggling alone. Your choice."
            ],
            can_skip=False
        ),
        
        StoryBeat(
            id="new_equilibrium",
            name="Your Place in Havenbrook",
            description="The final state based on player choices",
            trigger_conditions={
                "choice_made": ["accept_or_rebel"]
            },
            required_npcs=["mayor_elena"],
            required_locations=["town_square"],
            narrative_stage="Full Revelation",
            outcomes={
                "ending_achieved": True
            },
            dialogue_hints=[
                "Welcome to your new life in Havenbrook",
                "You've found your place in our order",
                "Some chains are silk, but chains nonetheless"
            ],
            can_skip=False
        )
    ],
    
    # Required NPCs with detailed specifications
    required_npcs=[
        {
            "id": "mayor_elena",
            "name": "Elena Winters",
            "archetype": "Shadow Ruler",
            "traits": ["charismatic", "calculating", "maternally_dominant"],
            "role": "Mayor and Secret Council Leader",
            "stats": {
                "dominance": 85,
                "cruelty": 30,
                "affection": 40
            },
            "schedule": {
                "morning": "Mayor's Office",
                "afternoon": "Town Square or Council Chambers",
                "evening": "Belladonna Cafe",
                "night": "Private Estate"
            }
        },
        {
            "id": "assistant_clara",
            "name": "Clara Dutton",
            "archetype": "Helpful Manipulator",
            "traits": ["sweet", "organized", "subtly_controlling"],
            "role": "Mayor's Assistant",
            "stats": {
                "dominance": 60,
                "cruelty": 10,
                "affection": 70
            }
        },
        {
            "id": "mentor_isabella",
            "name": "Isabella Rosewood",
            "archetype": "Seductive Mentor",
            "traits": ["successful", "alluring", "strategically_helpful"],
            "role": "Business Mentor and Council Member",
            "stats": {
                "dominance": 75,
                "cruelty": 20,
                "affection": 60
            }
        },
        {
            "id": "supplier_victoria",
            "name": "Victoria Sterling",
            "archetype": "Economic Gatekeeper",
            "traits": ["businesslike", "uncompromising", "reward_focused"],
            "role": "Controls Supply Networks",
            "stats": {
                "dominance": 70,
                "cruelty": 40,
                "affection": 30
            }
        },
        {
            "id": "competitor_sarah",
            "name": "Sarah Chen",
            "archetype": "Rival",
            "traits": ["competitive", "cunning", "secretly_sympathetic"],
            "role": "Established Business Owner",
            "stats": {
                "dominance": 65,
                "cruelty": 35,
                "affection": 45
            }
        }
    ],
    
    # Required locations
    required_locations=[
        {
            "name": "Town Square",
            "type": "plaza",
            "description": "The bustling heart of Havenbrook, where paths cross and deals are made"
        },
        {
            "name": "Mayor's Office",
            "type": "government",
            "description": "An elegant office that radiates subtle power"
        },
        {
            "name": "Council Chambers",
            "type": "government", 
            "description": "A private meeting room with no windows and thick walls"
        },
        {
            "name": "Warehouse District",
            "type": "commercial",
            "description": "Where Havenbrook's goods flow through carefully controlled channels"
        },
        {
            "name": "Belladonna Cafe",
            "type": "business",
            "description": "An upscale cafe where influential women gather"
        }
    ],
    
    # Preset conflicts
    required_conflicts=[
        {
            "name": "The Permit Struggle",
            "type": "economic",
            "factions": ["Independent Business", "Council Control"],
            "starting_phase": "brewing"
        }
    ],
    
    # What can be dynamically generated
    dynamic_elements={
        "minor_npcs": True,  # Can add other townsfolk
        "side_quests": True,  # Can add minor quests
        "daily_activities": True,  # Normal life continues
        "relationship_dynamics": True,  # Relationships can evolve naturally
        "business_details": True  # Player chooses business type
    },
    
    # Key player choice points
    player_choices_matter=[
        "business_type",  # What business to start
        "recommendation_approach",  # How to get recommendations
        "council_response",  # How to respond to pressure
        "final_choice"  # Accept integration or maintain independence
    ],
    
    flexibility_level=0.5,  # Medium flexibility - key beats must happen
    enforce_ending=False  # Multiple endings possible
)
