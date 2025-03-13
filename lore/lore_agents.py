# file: lore_agents.py

from typing import Any, Dict, List

# Agents SDK imports
from agents import Agent, ModelSettings
from agents.models.openai_responses import OpenAIResponsesModel

# Pydantic schemas for your outputs
from lore.lore_schemas import (
    FoundationLoreOutput,
    FactionsOutput,
    CulturalElementsOutput,
    HistoricalEventsOutput,
    LocationsOutput,
    QuestsOutput
)

###############################################################################
# Foundation lore agent
###############################################################################
foundation_lore_agent = Agent(
    name="FoundationLoreAgent",
    instructions=(
        "You produce foundational world lore for a fantasy environment. "
        "Return valid JSON that matches FoundationLoreOutput, which has keys: "
        "[cosmology, magic_system, world_history, calendar_system, social_structure]. "
        "Do NOT include any extra text outside the JSON."
    ),
    # Use whichever model you want (the new 'Responses' API or normal Chat completions)
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.4),
    output_type=FoundationLoreOutput,
)

###############################################################################
# Factions agent
###############################################################################
factions_agent = Agent(
    name="FactionsAgent",
    instructions=(
        "You generate 3-5 distinct factions for a given setting. "
        "Return valid JSON as an array of objects, matching FactionsOutput. "
        "Each faction object has: name, type, description, values, goals, "
        "headquarters, rivals, allies, hierarchy_type, etc. "
        "No extra text outside the JSON."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    output_type=FactionsOutput,
)

###############################################################################
# Cultural elements agent
###############################################################################
cultural_agent = Agent(
    name="CulturalAgent",
    instructions=(
        "You create cultural elements like traditions, customs, rituals. "
        "Return JSON matching CulturalElementsOutput: an array of objects. "
        "Fields include: name, type, description, practiced_by, significance, "
        "historical_origin. No extra text outside the JSON."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.5),
    output_type=CulturalElementsOutput,
)

###############################################################################
# Historical events agent
###############################################################################
history_agent = Agent(
    name="HistoryAgent",
    instructions=(
        "You create major historical events. Return JSON matching "
        "HistoricalEventsOutput: an array with fields name, date_description, "
        "description, participating_factions, consequences, significance. "
        "No extra text outside the JSON."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.6),
    output_type=HistoricalEventsOutput,
)

###############################################################################
# Locations agent
###############################################################################
locations_agent = Agent(
    name="LocationsAgent",
    instructions=(
        "You generate 5-8 significant locations. Return JSON matching "
        "LocationsOutput: an array of objects with fields name, description, "
        "type, controlling_faction, notable_features, hidden_secrets, "
        "strategic_importance. No extra text outside the JSON."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    output_type=LocationsOutput,
)

###############################################################################
# Quests agent
###############################################################################
quests_agent = Agent(
    name="QuestsAgent",
    instructions=(
        "You create 5-7 quest hooks. Return JSON matching QuestsOutput: an "
        "array of objects with quest_name, quest_giver, location, description, "
        "objectives, rewards, difficulty, lore_significance. "
        "No extra text outside the JSON."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    output_type=QuestsOutput,
)

###############################################################################
# Optional: A "SettingAnalysisAgent" if you want to move old GPT-based logic 
# for analyzing NPC data / environment into a single specialized agent.
###############################################################################
setting_analysis_agent = Agent(
    name="SettingAnalysisAgent",
    instructions=(
        "You analyze the game setting and NPC data to propose relevant "
        "organizations or factions. Return JSON with categories like "
        "academic, athletic, social, professional, cultural, political, other. "
        "Each category is an array of objects with fields like name, type, "
        "description, membership_basis, hierarchy, gathering_location, etc. "
        "No extra text outside the JSON."
    ),
    # If you want to keep it simpler, you can also provide 
    # a function_tool for 'generate_organizations_for_setting' 
    # but that depends on your architecture.
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    # If you want a strictly typed output, create a Pydantic schema for it. 
    # Or use plain str if you just parse the JSON yourself.
    output_type=None,  # or some pydantic schema if you have it
)

#
# End of file
#
