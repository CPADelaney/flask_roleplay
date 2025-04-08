# lore/unified_schemas.py

"""
Unified schema definitions for lore generation components.
This module provides Pydantic models for structuring and validating lore data.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, root_validator
from datetime import datetime


class FoundationLoreOutput(BaseModel):
    """Foundation lore schema including cosmology, magic system, etc."""
    cosmology: str = Field(..., description="The cosmic structure and origin of the world")
    magic_system: str = Field(..., description="Rules and nature of magic in the world")
    world_history: str = Field(..., description="Brief overview of significant historical periods")
    calendar_system: str = Field(..., description="Time measurement and significant dates")
    social_structure: str = Field(..., description="Major social structures and hierarchies")


class FactionModel(BaseModel):
    """Individual faction schema."""
    name: str = Field(..., description="Name of the faction")
    type: str = Field(..., description="Type of faction (political, religious, criminal, etc.)")
    description: str = Field(..., description="Detailed description of the faction")
    values: List[str] = Field(..., description="Core values and beliefs")
    goals: List[str] = Field(..., description="Current objectives and long-term aims")
    headquarters: Optional[str] = Field(None, description="Main base of operations")
    rivals: Optional[List[str]] = Field(None, description="Names of rival factions")
    allies: Optional[List[str]] = Field(None, description="Names of allied factions")
    hierarchy_type: Optional[str] = Field(None, description="Structure of leadership")
    territory: Optional[str] = Field(None, description="Areas controlled or influenced")
    resources: Optional[List[str]] = Field(None, description="Key resources controlled")
    secret_knowledge: Optional[str] = Field(None, description="Hidden knowledge or secrets")
    public_reputation: Optional[str] = Field(None, description="How they're perceived")
    color_scheme: Optional[str] = Field(None, description="Associated colors")
    symbol_description: Optional[str] = Field(None, description="Description of faction symbol")


class FactionsOutput(BaseModel):
    """Container for multiple factions."""
    __root__: List[FactionModel]


class CulturalElementModel(BaseModel):
    """Individual cultural element schema."""
    name: str = Field(..., description="Name of the cultural element")
    type: str = Field(..., description="Type (tradition, custom, taboo, holiday, etc.)")
    description: str = Field(..., description="Detailed description")
    practiced_by: List[str] = Field(..., description="Groups who observe this element")
    significance: int = Field(..., description="Importance (1-10)", ge=1, le=10)
    historical_origin: Optional[str] = Field(None, description="Origin story or history")
    related_elements: Optional[List[str]] = Field(None, description="Connected cultural elements")


class CulturalElementsOutput(BaseModel):
    """Container for multiple cultural elements."""
    __root__: List[CulturalElementModel]


class HistoricalEventModel(BaseModel):
    """Individual historical event schema."""
    name: str = Field(..., description="Name of the historical event")
    date_description: str = Field(..., description="When it occurred (e.g., '200 years ago')")
    description: str = Field(..., description="Detailed description of the event")
    participating_factions: Optional[List[str]] = Field(None, description="Factions involved")
    consequences: List[str] = Field(..., description="Effects and aftermath of the event")
    significance: int = Field(..., description="Historical importance (1-10)", ge=1, le=10)
    affected_locations: Optional[List[str]] = Field(None, description="Places affected")
    historical_figures: Optional[List[str]] = Field(None, description="Key people involved")
    commemorated_by: Optional[str] = Field(None, description="How it's remembered")


class HistoricalEventsOutput(BaseModel):
    """Container for multiple historical events."""
    __root__: List[HistoricalEventModel]


class LocationModel(BaseModel):
    """Individual location schema."""
    name: str = Field(..., description="Name of the location")
    description: str = Field(..., description="Detailed description")
    type: str = Field(..., description="Type (city, dungeon, landmark, etc.)")
    controlling_faction: Optional[str] = Field(None, description="Faction in control")
    notable_features: List[str] = Field(..., description="Distinctive aspects")
    hidden_secrets: Optional[List[str]] = Field(None, description="Hidden elements")
    strategic_importance: Optional[str] = Field(None, description="Tactical or resource value")


class LocationsOutput(BaseModel):
    """Container for multiple locations."""
    __root__: List[LocationModel]


class QuestModel(BaseModel):
    """Individual quest schema."""
    quest_name: str = Field(..., description="Name of the quest")
    quest_giver: str = Field(..., description="Who offers the quest")
    location: str = Field(..., description="Where the quest takes place")
    description: str = Field(..., description="Detailed description")
    objectives: List[str] = Field(..., description="Goals to complete")
    rewards: Optional[List[str]] = Field(None, description="What is gained upon completion")
    difficulty: int = Field(..., description="Challenge level (1-10)", ge=1, le=10)
    lore_significance: Optional[str] = Field(None, description="Connection to world lore")


class QuestsOutput(BaseModel):
    """Container for multiple quests."""
    __root__: List[QuestModel]
