# lore/lore_agents.py
from agents import Agent, ModelSettings
from lore.lore_schemas import FoundationLoreOutput, FactionsOutput, CulturalElementsOutput, HistoricalEventsOutput, LocationsOutput, QuestsOutput

foundation_lore_agent = Agent(
    name="FoundationLoreAgent",
    instructions=(
        "You produce foundational world lore for a fantasy environment. "
        "Return valid JSON that matches FoundationLoreOutput: {cosmology, magic_system, "
        "world_history, calendar_system, social_structure}."
    ),
    output_type=FoundationLoreOutput,
    model_settings=ModelSettings(temperature=0.4)  # or whichever
)

factions_agent = Agent(
    name="FactionsAgent",
    instructions=(
        "You generate 3-5 distinct factions for a given setting. Return valid JSON as an array of objects, "
        "matching FactionsOutput. Each faction has: name, type, description, values, goals, etc."
    ),
    output_type=FactionsOutput,
    model_settings=ModelSettings(temperature=0.7)
)

cultural_agent = Agent(
    name="CulturalAgent",
    instructions=(
        "You create cultural elements like traditions, customs, rituals. Return JSON matching CulturalElementsOutput, "
        "an array of objects with name, type, description, practiced_by, significance, historical_origin."
    ),
    output_type=CulturalElementsOutput,
    model_settings=ModelSettings(temperature=0.5)
)

history_agent = Agent(
    name="HistoryAgent",
    instructions=(
        "You create major historical events. Return JSON matching HistoricalEventsOutput, an array of objects with "
        "name, date_description, description, participating_factions, consequences, significance."
    ),
    output_type=HistoricalEventsOutput
)

locations_agent = Agent(
    name="LocationsAgent",
    instructions=(
        "You generate 5-8 significant locations. Return JSON matching LocationsOutput, an array with each location's "
        "name, description, type, controlling_faction, notable_features, hidden_secrets, strategic_importance."
    ),
    output_type=LocationsOutput
)

quests_agent = Agent(
    name="QuestsAgent",
    instructions=(
        "You create 5-7 quest hooks. Return JSON matching QuestsOutput, an array of objects with quest_name, quest_giver, "
        "location, description, objectives, rewards, difficulty, and lore_significance."
    ),
    output_type=QuestsOutput
)
