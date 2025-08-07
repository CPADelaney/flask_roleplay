# logic/chatgpt_integration.py
import os
import json
import logging
import functools
import asyncio
import threading
import warnings
import openai
from typing import Dict, List, Any, Optional, Union
import numpy as np
from openai._exceptions import APIStatusError
from db.connection import get_db_connection_context
from logic.prompts import SYSTEM_PROMPT, PRIVATE_REFLECTION_INSTRUCTIONS
from logic.json_helpers import safe_json_loads


# Configure module logger
logger = logging.getLogger(__name__)

# Nyx system imports
# Import Nyx integrations lazily to avoid circular dependencies
nyx_process_input = None


def get_nyx_process_input():
    """Lazily import and return Nyx's process_user_input function."""
    global nyx_process_input
    if nyx_process_input is None:
        from nyx.nyx_agent_sdk import process_user_input as _process_user_input
        nyx_process_input = _process_user_input
    return nyx_process_input


async def get_memory_nyx_bridge_lazy(user_id: int, conversation_id: int):
    """Lazily import and return the memory bridge."""
    from memory.memory_nyx_integration import get_memory_nyx_bridge as _get_memory_nyx_bridge
    return await _get_memory_nyx_bridge(user_id, conversation_id)

# Try to import prepare_context if available
try:
    from nyx.core.orchestrator import prepare_context
    PREPARE_CONTEXT_AVAILABLE = True
except ImportError:
    PREPARE_CONTEXT_AVAILABLE = False
    logger.info("prepare_context not available, will use direct prompts")

# Try to import image prompting functions if available
try:
    from logic.gpt_image_prompting import get_system_prompt_with_image_guidance
    IMAGE_PROMPTING_AVAILABLE = True
except ImportError:
    IMAGE_PROMPTING_AVAILABLE = False
    logger.info("Image prompting module not available, using standard prompts")

# Temperature settings for different task types
TEMPERATURE_SETTINGS = {
    "decision": 0.7,
    "reflection": 0.5,
    "abstraction": 0.4,
    "introspection": 0.6,
    "memory": 0.3
}

# Updated schema with arrays instead of object maps
UNIVERSAL_UPDATE_FUNCTION_SCHEMA = {
    "name": "apply_universal_update",
    "description": "Insert or update roleplay elements (NPCs, locations, events, etc.), including fantastical punishments and image generation, returning a narrative.",
    "parameters": {
        "type": "object",
        "properties": {
            "narrative": {
                "type": "string",
                "description": "The narrative text to display, potentially describing surreal punishments or events."
            },
            "roleplay_updates": {
                "type": "array",
                "description": "Updates to CurrentRoleplay as key-value pairs (e.g., 'CurrentYear', 'TimeOfDay' for time).",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {}  # Can be any JSON value
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False
                }
            },
            "ChaseSchedule": {
                "type": "array",
                "description": "Chase's detailed weekly schedule as key-value pairs where key is day name.",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "pattern": "^[A-Z][a-zA-Z]*$"
                        },
                        "value": {
                            "type": "object",
                            "properties": {
                                "Morning": {"type": "string"},
                                "Afternoon": {"type": "string"},
                                "Evening": {"type": "string"},
                                "Night": {"type": "string"}
                            },
                            "required": ["Morning", "Afternoon", "Evening", "Night"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False
                }
            },
            "MainQuest": {
                "type": "string",
                "description": "Summary of Chase's main quest."
            },
            "PlayerRole": {
                "type": "string",
                "description": "Chase's typical day/role in this environment."
            },
            "npc_creations": {
                "type": "array",
                "description": "Create new NPCs with stats and traits.",
                "items": {
                    "type": "object",
                    "properties": {
                        "npc_id": {"type": "number"},
                        "npc_name": {"type": "string"},
                        "introduced": {"type": "boolean"},
                        "archetypes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "number"},
                                    "name": {"type": "string"}
                                },
                                "additionalProperties": False
                            }
                        },
                        "archetype_summary": {"type": "string"},
                        "archetype_extras_summary": {"type": "string"},
                        "physical_description": {"type": "string"},
                        "dominance": {"type": "number"},
                        "cruelty": {"type": "number"},
                        "closeness": {"type": "number"},
                        "trust": {"type": "number"},
                        "respect": {"type": "number"},
                        "intensity": {"type": "number"},
                        "hobbies": {"type": "array", "items": {"type": "string"}},
                        "personality_traits": {"type": "array", "items": {"type": "string"}},
                        "likes": {"type": "array", "items": {"type": "string"}},
                        "dislikes": {"type": "array", "items": {"type": "string"}},
                        "affiliations": {"type": "array", "items": {"type": "string"}},
                        "schedule": {
                            "type": "array",
                            "description": "Weekly schedule as key-value pairs where key is day name.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {
                                        "type": "string",
                                        "pattern": "^[A-Z][a-zA-Z]*$"
                                    },
                                    "value": {
                                        "type": "object",
                                        "properties": {
                                            "Morning": {"type": "string"},
                                            "Afternoon": {"type": "string"},
                                            "Evening": {"type": "string"},
                                            "Night": {"type": "string"}
                                        },
                                        "required": ["Morning", "Afternoon", "Evening", "Night"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["key", "value"],
                                "additionalProperties": False
                            }
                        },
                        "memory": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ]
                        },
                        "monica_level": {"type": "number"},
                        "sex": {"type": "string"},
                        "age": {"type": "number"},
                        "birthdate": {"type": "string", "format": "date"}
                    },
                    "required": ["npc_name"],
                    "additionalProperties": False
                }
            },
            "npc_updates": {
                "type": "array",
                "description": "Update existing NPCs by npc_id.",
                "items": {
                    "type": "object",
                    "properties": {
                        "npc_id": {"type": "number"},
                        "npc_name": {"type": "string"},
                        "introduced": {"type": "boolean"},
                        "archetype_summary": {"type": "string"},
                        "archetype_extras_summary": {"type": "string"},
                        "physical_description": {"type": "string"},
                        "dominance": {"type": "number"},
                        "cruelty": {"type": "number"},
                        "closeness": {"type": "number"},
                        "trust": {"type": "number"},
                        "respect": {"type": "number"},
                        "intensity": {"type": "number"},
                        "hobbies": {"type": "array", "items": {"type": "string"}},
                        "personality_traits": {"type": "array", "items": {"type": "string"}},
                        "likes": {"type": "array", "items": {"type": "string"}},
                        "dislikes": {"type": "array", "items": {"type": "string"}},
                        "sex": {"type": "string"},
                        "memory": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ]
                        },
                        "schedule": {
                            "type": "array",
                            "description": "Weekly schedule as key-value pairs where key is day name.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {
                                        "type": "string",
                                        "pattern": "^[A-Z][a-zA-Z]*$"
                                    },
                                    "value": {
                                        "type": "object",
                                        "properties": {
                                            "Morning": {"type": "string"},
                                            "Afternoon": {"type": "string"},
                                            "Evening": {"type": "string"},
                                            "Night": {"type": "string"}
                                        },
                                        "required": ["Morning", "Afternoon", "Evening", "Night"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["key", "value"],
                                "additionalProperties": False
                            }
                        },
                        "schedule_updates": {
                            "type": "array",
                            "description": "Partial schedule updates. Omit value to delete the key.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "value": {
                                        "anyOf": [
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "Morning": {"type": "string"},
                                                    "Afternoon": {"type": "string"},
                                                    "Evening": {"type": "string"},
                                                    "Night": {"type": "string"}
                                                },
                                                "additionalProperties": False
                                            },
                                            {"type": "null"}
                                        ]
                                    }
                                },
                                "required": ["key"],
                                "additionalProperties": False
                            }
                        },
                        "affiliations": {"type": "array", "items": {"type": "string"}},
                        "current_location": {"type": "string"}
                    },
                    "required": ["npc_id"],
                    "additionalProperties": False
                }
            },
            "character_stat_updates": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "default": "Chase"},
                    "stats": {
                        "type": "array",
                        "description": "Character stats as key-value pairs.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "enum": ["corruption", "confidence", "willpower", "obedience", 
                                             "dependency", "lust", "mental_resilience", "physical_endurance"]
                                },
                                "value": {"type": "number"}
                            },
                            "required": ["key", "value"],
                            "additionalProperties": False
                        }
                    }
                },
                "additionalProperties": False
            },
            "relationship_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "npc_id": {"type": "number"},
                        "affiliations": {"type": "array", "items": {"type": "string"}}
                    },
                    "additionalProperties": False
                }
            },
            "npc_introductions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "npc_id": {"type": "number"}
                    },
                    "required": ["npc_id"],
                    "additionalProperties": False
                }
            },
            "location_creations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "location_name": {"type": "string"},
                        "description": {"type": "string"},
                        "open_hours": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["location_name"],
                    "additionalProperties": False
                }
            },
            "event_list_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "event_name": {"type": "string"},
                        "description": {"type": "string"},
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"},
                        "location": {"type": "string"},
                        "npc_id": {"type": "number"},
                        "year": {"type": "number"},
                        "month": {"type": "number"},
                        "day": {"type": "number"},
                        "time_of_day": {"type": "string"},
                        "override_location": {"type": "string"},
                        "fantasy_level": {
                            "type": "string",
                            "enum": ["realistic", "fantastical", "surreal"],
                            "default": "realistic",
                            "description": "Indicates if the event involves reality-breaking elements."
                        }
                    },
                    "additionalProperties": False
                }
            },
            "inventory_updates": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "default": "Chase"},
                    "added_items": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "item_name": {"type": "string"},
                                        "item_description": {"type": "string"},
                                        "item_effect": {"type": "string"},
                                        "category": {"type": "string"}
                                    },
                                    "additionalProperties": False
                                }
                            ]
                        }
                    },
                    "removed_items": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "item_name": {"type": "string"}
                                    },
                                    "additionalProperties": False
                                }
                            ]
                        }
                    }
                },
                "additionalProperties": False
            },
            "quest_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quest_id": {"type": "number"},
                        "quest_name": {"type": "string"},
                        "status": {"type": "string"},
                        "progress_detail": {"type": "string"},
                        "quest_giver": {"type": "string"},
                        "reward": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            },
            "social_links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity1_type": {"type": "string"},
                        "entity1_id": {"type": "number"},
                        "entity2_type": {"type": "string"},
                        "entity2_id": {"type": "number"},
                        "link_type": {"type": "string"},
                        "level_change": {"type": "number"},
                        "new_event": {"type": "string"},
                        "group_context": {
                            "type": "string",
                            "description": "Describes multi-NPC interactions (e.g., 'NPC2 and five others gang up')."
                        }
                    },
                    "additionalProperties": False
                }
            },
            "perk_unlocks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "player_name": {"type": "string", "default": "Chase"},
                        "perk_name": {"type": "string"},
                        "perk_description": {"type": "string"},
                        "perk_effect": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            },
            "activity_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "activity_name": {"type": "string"},
                        "purpose": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "fantasy_level": {
                                    "type": "string",
                                    "enum": ["realistic", "fantastical", "surreal"],
                                    "default": "realistic"
                                }
                            },
                            "additionalProperties": False
                        },
                        "stat_integration": {
                            "type": "array",
                            "description": "Stat integration as key-value pairs.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "value": {}
                                },
                                "required": ["key", "value"],
                                "additionalProperties": False
                            }
                        },
                        "intensity_tier": {"type": "number", "minimum": 0, "maximum": 4},
                        "setting_variant": {"type": "string"}
                    },
                    "required": ["activity_name"],
                    "additionalProperties": False
                },
                "description": "Create or update Activities for punishments, including surreal elements."
            },
            "journal_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entry_type": {"type": "string", "enum": ["revelation", "moment", "dream"]},
                        "entry_text": {"type": "string"},
                        "fantasy_flag": {"type": "boolean", "default": False},
                        "intensity_level": {"type": "number", "minimum": 0, "maximum": 4}
                    },
                    "required": ["entry_type", "entry_text"],
                    "additionalProperties": False
                },
                "description": "Log narrative events or punishments in PlayerJournal, including surreal ones."
            },
            "image_generation": {
                "type": "object",
                "properties": {
                    "generate": {"type": "boolean", "default": False},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "low"},
                    "focus": {"type": "string", "enum": ["character", "scene", "action", "balanced"], "default": "balanced"},
                    "framing": {"type": "string", "enum": ["close_up", "medium_shot", "wide_shot"], "default": "medium_shot"},
                    "reason": {"type": "string", "description": "Why this image is generated (e.g., 'surreal punishment')."}
                },
                "description": "Trigger and configure image generation for the scene.",
                "additionalProperties": False
            }
        },
        "required": [
            "narrative",
            "roleplay_updates",
            "ChaseSchedule",
            "MainQuest",
            "PlayerRole",
            "npc_creations",
            "npc_updates",
            "character_stat_updates",
            "relationship_updates",
            "npc_introductions",
            "location_creations",
            "event_list_updates",
            "inventory_updates",
            "quest_updates",
            "social_links",
            "perk_unlocks",
            "activity_updates",
            "journal_updates",
            "image_generation"
        ],
        "additionalProperties": False
    }
}


# =====================================================
# Utility functions for array<->dict conversion
# =====================================================

def dict_to_array(obj_data: Dict[str, Any], key_name: str = "key", value_name: str = "value") -> List[Dict[str, Any]]:
    """Convert object/dict to array of key-value pairs."""
    if not isinstance(obj_data, dict):
        return []
    return [
        {key_name: k, value_name: v} 
        for k, v in obj_data.items()
    ]


def array_to_dict(array_data: List[Dict[str, Any]], key_name: str = "key", value_name: str = "value") -> Dict[str, Any]:
    """Convert array of key-value pairs back to dict."""
    if not isinstance(array_data, list):
        return {}
    result = {}
    for item in array_data:
        if isinstance(item, dict) and key_name in item and value_name in item:
            result[item[key_name]] = item[value_name]
    return result


def convert_npc_schedule_to_array(npc_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NPC schedule from dict to array format."""
    if "schedule" in npc_data and isinstance(npc_data["schedule"], dict):
        npc_data["schedule"] = dict_to_array(npc_data["schedule"])
    if "schedule_updates" in npc_data and isinstance(npc_data["schedule_updates"], dict):
        npc_data["schedule_updates"] = dict_to_array(npc_data["schedule_updates"])
    return npc_data


def convert_npc_schedule_to_dict(npc_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NPC schedule from array to dict format (for database storage)."""
    if "schedule" in npc_data and isinstance(npc_data["schedule"], list):
        npc_data["schedule"] = array_to_dict(npc_data["schedule"])
    if "schedule_updates" in npc_data and isinstance(npc_data["schedule_updates"], list):
        npc_data["schedule_updates"] = array_to_dict(npc_data["schedule_updates"])
    return npc_data


def convert_response_to_array_format(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Nyx/OpenAI response from dict format to array format for the schema."""
    universal_updates = response_data.get("universal_updates", {})
    
    # Convert roleplay_updates
    roleplay_updates_obj = universal_updates.get("roleplay_updates", {})
    roleplay_updates_array = dict_to_array(roleplay_updates_obj)
    
    # Convert ChaseSchedule
    chase_schedule_obj = universal_updates.get("ChaseSchedule", {})
    chase_schedule_array = dict_to_array(chase_schedule_obj)
    
    # Convert character stats
    char_stats_obj = universal_updates.get("character_stat_updates", {}).get("stats", {})
    char_stats_array = dict_to_array(char_stats_obj)
    
    # Process NPC creations and updates
    npc_creations = universal_updates.get("npc_creations", [])
    for npc in npc_creations:
        convert_npc_schedule_to_array(npc)
    
    npc_updates = universal_updates.get("npc_updates", [])
    for npc in npc_updates:
        convert_npc_schedule_to_array(npc)
    
    # Process activity updates
    activity_updates = universal_updates.get("activity_updates", [])
    for activity in activity_updates:
        if "stat_integration" in activity and isinstance(activity["stat_integration"], dict):
            activity["stat_integration"] = dict_to_array(activity["stat_integration"])
    
    return {
        "narrative": response_data.get("narrative", ""),
        "roleplay_updates": roleplay_updates_array,
        "ChaseSchedule": chase_schedule_array,
        "MainQuest": universal_updates.get("MainQuest", ""),
        "PlayerRole": universal_updates.get("PlayerRole", ""),
        "npc_creations": npc_creations,
        "npc_updates": npc_updates,
        "character_stat_updates": {
            "player_name": universal_updates.get("character_stat_updates", {}).get("player_name", "Chase"),
            "stats": char_stats_array
        },
        "relationship_updates": universal_updates.get("relationship_updates", []),
        "npc_introductions": universal_updates.get("npc_introductions", []),
        "location_creations": universal_updates.get("location_creations", []),
        "event_list_updates": universal_updates.get("event_list_updates", []),
        "inventory_updates": universal_updates.get("inventory_updates", {}),
        "quest_updates": universal_updates.get("quest_updates", []),
        "social_links": universal_updates.get("social_links", []),
        "perk_unlocks": universal_updates.get("perk_unlocks", []),
        "activity_updates": activity_updates,
        "journal_updates": universal_updates.get("journal_updates", []),
        "image_generation": {
            "generate": response_data.get("generate_image", False),
            "priority": "high" if response_data.get("generate_image") else "low",
            "focus": "scene",
            "framing": "medium_shot",
            "reason": response_data.get("image_prompt", "")
        }
    }


# =====================================================
# OpenAI Client Manager (unchanged)
# =====================================================

import threading
from typing import Optional, Dict, Any


class OpenAIClientManager:
    """
    Centralized manager for OpenAI client access.
    Provides both sync and async clients through a single interface.
    Thread-safe singleton with robust configuration management.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(OpenAIClientManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Ensure __init__ only runs once
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._sync_client = None
        self._async_client = None
        self._config = self._load_config()
        self._client_lock = threading.Lock()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load OpenAI configuration from environment."""
        config = {
            'api_key': os.getenv("OPENAI_API_KEY"),
            'organization': os.getenv("OPENAI_ORGANIZATION"),
            'base_url': os.getenv("OPENAI_BASE_URL"),
            'timeout': int(os.getenv("OPENAI_TIMEOUT", "600")),
            'max_retries': int(os.getenv("OPENAI_MAX_RETRIES", "2")),
            'default_model': os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-nano"),
            'default_responses_model': os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5-nano")
        }
        
        if not config['api_key']:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        
        # Set global API key for backwards compatibility
        openai.api_key = config['api_key']
        if config['organization']:
            openai.organization = config['organization']
            
        return config
    
    @property
    def sync_client(self):
        """Get or create thread-safe synchronous OpenAI client."""
        if self._sync_client is None:
            with self._client_lock:
                if self._sync_client is None:
                    # Configure the global openai module
                    openai.api_key = self._config['api_key']
                    if self._config['organization']:
                        openai.organization = self._config['organization']
                    self._sync_client = openai
        return self._sync_client
    
    @property
    def async_client(self):
        """Get or create thread-safe asynchronous OpenAI client."""
        if self._async_client is None:
            with self._client_lock:
                if self._async_client is None:
                    from openai import AsyncOpenAI
                    
                    # Build client configuration
                    client_config = {
                        'api_key': self._config['api_key'],
                        'timeout': self._config['timeout'],
                        'max_retries': self._config['max_retries']
                    }
                    
                    if self._config['organization']:
                        client_config['organization'] = self._config['organization']
                    if self._config['base_url']:
                        client_config['base_url'] = self._config['base_url']
                    
                    self._async_client = AsyncOpenAI(**client_config)
        return self._async_client
    
    def get_assistants_client(self) -> "AsyncOpenAI":
        """Get AsyncOpenAI client for Assistants API usage."""
        return self.async_client
    
    def get_responses_client(self):
        """Get AsyncOpenAI client for Responses API usage."""
        return self.async_client
    
    def get_pydantic_responses_model(self, model: str | None = None):
        """
        Get OpenAI model for pydantic-ai Responses SDK.
        
        Args:
            model: Optional model name override
        """
        try:
            from pydantic_ai.models.openai import OpenAIResponsesModel
            return OpenAIResponsesModel(
                model=model or self._config['default_model'],
                openai_client=self.async_client
            )
        except ImportError:
            raise RuntimeError("pydantic-ai SDK is not installed")
    
    def reset_clients(self):
        """Reset all clients, forcing recreation on next access."""
        with self._client_lock:
            self._sync_client = None
            self._async_client = None
            self._config = self._load_config()
            logger.info("OpenAI clients reset")
    
    def update_config(self, **kwargs) -> None:
        """
        Hot‑reload any subset of client config keys, then rebuild clients.
    
        Accepts: api_key • organization • base_url • timeout • max_retries •
                 default_model • default_responses_model
        """
        allowed = {
            "api_key",
            "organization",
            "base_url",
            "timeout",
            "max_retries",
            "default_model",
            "default_responses_model",
        }
    
        invalid = set(kwargs) - allowed
        if invalid:
            raise ValueError(f"Unknown OpenAI config keys: {', '.join(invalid)}")
    
        with self._client_lock:
            self._config.update({k: v for k, v in kwargs.items() if v is not None})
    
            # refresh global module settings if they changed
            if "api_key" in kwargs:
                openai.api_key = kwargs["api_key"]
            if "organization" in kwargs:
                openai.organization = kwargs["organization"]
    
            # drop cached clients so they'll be recreated lazily
            self._sync_client = None
            self._async_client = None
    
        logger.info("OpenAI configuration updated: %s", ", ".join(kwargs))
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration (read-only)."""
        return self._config.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if clients can be created and basic config is valid."""
        health = {
            'config_valid': bool(self._config.get('api_key')),
            'sync_client_ready': self._sync_client is not None,
            'async_client_ready': self._async_client is not None,
            'default_model': self._config.get('default_model')
        }
        
        try:
            # Try to access clients to ensure they can be created
            _ = self.sync_client
            _ = self.async_client
            health['clients_accessible'] = True
        except Exception as e:
            health['clients_accessible'] = False
            health['error'] = str(e)
            
        return health


# Create singleton instance
_client_manager = OpenAIClientManager()


# =====================================================
# Backwards compatibility functions (deprecated)
# =====================================================

def get_openai_client():
    """
    DEPRECATED: Use OpenAIClientManager instead.
    Get synchronous OpenAI client for backwards compatibility.
    """
    warnings.warn(
        "get_openai_client() is deprecated. Use OpenAIClientManager().sync_client instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _client_manager.sync_client


def get_async_openai_client():
    """
    DEPRECATED: Use OpenAIClientManager instead.
    Get async OpenAI client for backwards compatibility.
    """
    warnings.warn(
        "get_async_openai_client() is deprecated. Use OpenAIClientManager().async_client instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _client_manager.async_client


def get_agents_openai_model():
    """
    DEPRECATED: This function is no longer needed.
    Use generate_text_completion() or get_chatgpt_response() directly.
    """
    raise NotImplementedError(
        "get_agents_openai_model() is deprecated. "
        "Use generate_text_completion() for simple text generation "
        "or get_chatgpt_response() for full conversation handling."
    )


# =====================================================
# Core functions using the centralized client
# =====================================================

async def _safe_max_tokens(client, model: str, reserve: int = 256, hard_cap: int = 10000) -> int:
    """
    Safely calculate max tokens for a model.
    
    Args:
        client: OpenAI async client
        model: Model name
        reserve: Tokens to reserve for prompt (default: 256)
        hard_cap: Maximum tokens to return (default: 10000)
        
    Returns:
        Safe max tokens value
    """
    try:
        model_info = await client.models.retrieve(id=model)
        context_window = getattr(model_info, "context_window", 8192)
    except Exception as e:
        logger.warning(f"Could not retrieve model info for {model}: {e}")
        context_window = 8192
    
    return min(context_window - reserve, hard_cap)


async def build_message_history(conversation_id: int, aggregator_text: str, user_input: str, limit: int = 15):
    """
    Build message history using async DB connection
    """
    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT sender, content
                FROM messages
                WHERE conversation_id=$1
                ORDER BY id DESC
                LIMIT $2
            """, conversation_id, limit)
            
        # Convert to list and reverse for oldest first
        messages_data = [(row['sender'], row['content']) for row in rows]
        messages_data.reverse()  # Oldest first
        
        chat_history = []
        for sender, content in messages_data:
            role = "user" if sender.lower() == "user" else "assistant"
            chat_history.append({"role": role, "content": content})

        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "system", "content": aggregator_text})
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_input})
        return messages
    except Exception as e:
        logger.error(f"Error building message history: {e}")
        # Return a minimal set of messages in case of error
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": aggregator_text},
            {"role": "user", "content": user_input}
        ]


def retry_with_backoff(
    *,
    max_retries: int = 5,
    initial_delay: float = 1,
    backoff_factor: float = 2,
    exceptions: tuple[type[Exception], ...] = (
        openai.RateLimitError,
        openai.APIConnectionError,
        APIStatusError,
    ),
):
    """
    Decorator for automatic exponential‑backoff retries on common OpenAI errors.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:               # noqa: BLE001
                    if attempt == max_retries:
                        raise
                    logger.warning(
                        "OpenAI transient error (%s) on attempt %s/%s – retrying in %.1fs",
                        type(exc).__name__, attempt, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator


async def get_chatgpt_response(
    conversation_id: int, 
    aggregator_text: str, 
    user_input: str,
    reflection_enabled: bool = False,
    use_nyx_integration: bool = True
) -> dict[str, Any]:
    """
    Get a response from OpenAI with optional Nyx integration and preset story support.
    """
    
    # Get user_id from conversation
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                "SELECT user_id FROM conversations WHERE id = $1", 
                conversation_id
            )
            if not row:
                logger.error(f"Conversation {conversation_id} not found")
                return {
                    "type": "text",
                    "response": "Error: Conversation not found",
                    "tokens_used": 0
                }
            user_id = row['user_id']
    except Exception as e:
        logger.error(f"Error getting user_id for conversation {conversation_id}: {e}")
        return {
            "type": "text",
            "response": "Error: Database error",
            "tokens_used": 0
        }

    # Check if this conversation uses a preset story
    preset_info = await check_preset_story(conversation_id)
    
    # Build comprehensive context
    context_data = {
        "aggregator_text": aggregator_text,
        "reflection_enabled": reflection_enabled,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "system_prompt": SYSTEM_PROMPT,
        "private_reflection": PRIVATE_REFLECTION_INSTRUCTIONS,
        "universal_update_schema": UNIVERSAL_UPDATE_FUNCTION_SCHEMA
    }
    
    # If using a preset story, enhance the system prompt and add constraints
    if preset_info and preset_info.get('story_id') == 'the_moth_and_flame':
        from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
        
        # Enhance system prompt with consistency rules
        preset_system_prompt = f"""
{SYSTEM_PROMPT}

{QueenOfThornsConsistencyGuide.get_complete_system_prompt()}

CRITICAL PRESET STORY CONTEXT:
- You are currently running the preset story "The Moth and Flame" set in San Francisco Bay Area
- Current year: 2025
- You MUST follow ALL consistency rules above. Any deviation will break the story
- Quick Reference:
{QueenOfThornsConsistencyGuide.get_quick_reference()}

REMEMBER: 
- NEVER give the network an official name
- ALWAYS maintain Queen ambiguity
- Network controls Bay Area ONLY
- Transformation takes months/years
"""
        context_data["system_prompt"] = preset_system_prompt
        
        # Add preset-specific context
        context_data["preset_context"] = {
            "story_id": "the_moth_and_flame",
            "setting": "San Francisco Bay Area",
            "year": 2025,
            "preset_active": True,
            "consistency_validator": QueenOfThornsConsistencyGuide.validate_content
        }
        
        # Enhance aggregator text with preset constraints
        aggregator_text = f"""{aggregator_text}

PRESET STORY CONSTRAINTS:
The network has NO official name - refer to it as "the network" or "the garden"
The Queen of Thorns identity is ALWAYS ambiguous
The network controls Bay Area ONLY - elsewhere are allies, not branches
All information exists in four layers: PUBLIC|SEMI-PRIVATE|HIDDEN|DEEP SECRET
"""

    # 1) If Nyx integration is enabled, use the full Nyx agent system
    if use_nyx_integration:
        try:
            # Import the universal updater
            from logic.universal_updater_agent import process_universal_update
            
            # Pass preset context through to Nyx
            context_data["preset_info"] = preset_info
            
            # Process through Nyx
            nyx_process_input_fn = get_nyx_process_input()
            nyx_result = await nyx_process_input_fn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_input=user_input,
                context_data=context_data
            )
            
            if nyx_result.get("success"):
                response_data = nyx_result.get("response", {})
                
                # Validate response if preset is active
                if preset_info and context_data.get("preset_context"):
                    narrative = response_data.get("narrative", "")
                    validator = context_data["preset_context"].get("consistency_validator")
                    
                    if validator and narrative:
                        validation_result = validator(narrative)
                        if not validation_result['valid']:
                            logger.error(f"Preset violations detected: {validation_result['violations']}")
                            # Log but don't block - could implement retry logic here
                            logger.warning("Response may contain preset violations - manual review recommended")
                
                # Process any narrative through universal updater
                if response_data.get("narrative"):
                    update_context = {"source": "nyx_response"}
                    if preset_info:
                        update_context["preset_story_id"] = preset_info.get("story_id")
                        
                    update_result = await process_universal_update(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        narrative=response_data["narrative"],
                        context=update_context
                    )
                
                # Convert response to array format
                function_args = convert_response_to_array_format(response_data)
                
                return {
                    "type": "function_call",
                    "function_name": "apply_universal_update",
                    "function_args": function_args,
                    "tokens_used": nyx_result.get("performance", {}).get("tokens_used", 0),
                    "nyx_metrics": nyx_result.get("performance", {}),
                    "preset_story_id": preset_info.get("story_id") if preset_info else None
                }
            else:
                # Fall back to regular processing if Nyx fails
                logger.warning(f"Nyx processing failed: {nyx_result.get('error')}, falling back to regular processing")
                use_nyx_integration = False
                
        except Exception as e:
            logger.error(f"Error in Nyx integration: {e}", exc_info=True)
            use_nyx_integration = False

    # 2) Regular OpenAI processing (fallback or when Nyx is disabled)
    
    # Possibly retrieve or build a system prompt that includes image guidance
    image_prompt = None
    if IMAGE_PROMPTING_AVAILABLE:
        try:
            image_prompt = get_system_prompt_with_image_guidance(user_id, conversation_id)
            logger.debug("Using image-aware system prompt")
        except openai.BadRequestError as e:
            logger.info(f"Model doesn't support vision features: {e}")
            image_prompt = None
        except Exception as e:
            logger.info(f"Could not generate image prompt, using regular prompt: {e}")
            image_prompt = None
    
    # Use preset-enhanced prompt if available, then image prompt, then regular
    if preset_info and 'system_prompt' in context_data:
        primary_system_prompt = context_data['system_prompt']
    elif image_prompt:
        primary_system_prompt = image_prompt
    else:
        primary_system_prompt = SYSTEM_PROMPT

    # Get async client from centralized manager
    client = _client_manager.async_client

    # If reflection is OFF, do the single-step call
    if not reflection_enabled:
        messages = await build_message_history(conversation_id, aggregator_text, user_input, limit=15)
        
        # Update system prompt in messages if preset is active
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] = primary_system_prompt

        model = "gpt-5-nano"
        max_tokens = await _safe_max_tokens(client, model)
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            frequency_penalty=0.0,
            functions=[UNIVERSAL_UPDATE_FUNCTION_SCHEMA],
            function_call={"name": "apply_universal_update"},
        )

        msg = response.choices[0].message
        tokens_used = response.usage.total_tokens

        if msg.function_call is not None:
            fn_name = msg.function_call.name
            fn_args_str = msg.function_call.arguments or "{}"
            cleaned_args = _clean_function_args(fn_args_str)

            parsed_args = {}
            try:
                parsed_args = safe_json_loads(cleaned_args)
            except Exception:
                logger.exception("Error parsing function call arguments")

            # Validate narrative if preset is active
            if preset_info and parsed_args.get("narrative"):
                from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
                validation = QueenOfThornsConsistencyGuide.validate_content(parsed_args["narrative"])
                if not validation['valid']:
                    logger.error(f"Response violates preset rules: {validation['violations']}")

            _ensure_default_scene_data(parsed_args)

            return {
                "type": "function_call",
                "function_name": fn_name,
                "function_args": parsed_args,
                "tokens_used": tokens_used,
                "preset_story_id": preset_info.get("story_id") if preset_info else None
            }
        else:
            return {
                "type": "text",
                "function_name": None,
                "function_args": None,
                "response": msg.content,
                "tokens_used": tokens_used,
                "preset_story_id": preset_info.get("story_id") if preset_info else None
            }

    # If reflection is ON, do a two-step approach
    else:
        # Step A: Reflection Request
        reflection_messages = [
            {"role": "system", "content": primary_system_prompt},
            {"role": "system", "content": PRIVATE_REFLECTION_INSTRUCTIONS},
            {"role": "system", "content": aggregator_text},
            {
                "role": "user",
                "content": f"""
(INTERNAL REFLECTION STEP - DO NOT REVEAL THIS TO THE USER)

Please output a short JSON object with keys:
"reflection_notes" (string),
"private_goals" (array of strings),
"predicted_futures" (array of strings).

Discuss {user_input} from Nyx's internal perspective:
- Summarize relevant memories or strategies in reflection_notes.
- Outline your next private goals in private_goals.
- Predict possible user moves or story outcomes in predicted_futures.

DO NOT produce user-facing text here; only the JSON. 
                """
            }
        ]

        reflection_response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=reflection_messages,
            temperature=0.2,
            max_tokens=2500,
            frequency_penalty=0.0,
            functions=[],
            function_call="none"
        )
        reflection_msg = reflection_response.choices[0].message.content
        reflection_tokens_used = reflection_response.usage.total_tokens

        # Parse reflection JSON
        reflection_notes = ""
        private_goals = []
        predicted_futures = []
        try:
            reflection_data = safe_json_loads(reflection_msg)
            reflection_notes = reflection_data.get("reflection_notes", "")
            private_goals = reflection_data.get("private_goals", [])
            predicted_futures = reflection_data.get("predicted_futures", [])
        except Exception:
            logger.warning("Reflection JSON parse failed. Storing raw text.")
            reflection_notes = reflection_msg

        # Store reflection in memory system
        try:
            memory_bridge = await get_memory_nyx_bridge_lazy(user_id, conversation_id)
            await memory_bridge.add_memory(
                memory_text=f"Internal reflection: {reflection_notes}",
                memory_type="reflection",
                memory_scope="private",
                significance=7,
                tags=["nyx_reflection", "private"],
                metadata={
                    "private_goals": private_goals,
                    "predicted_futures": predicted_futures
                }
            )
        except Exception as e:
            logger.warning(f"Could not store reflection in memory: {e}")

        # Step B: Final (Public) Answer
        final_messages = [
            {"role": "system", "content": primary_system_prompt},
            {"role": "system", "content": PRIVATE_REFLECTION_INSTRUCTIONS},
            {"role": "system", "content": aggregator_text},
            {
                "role": "system",
                "content": f"Hidden Reflection (do not reveal): {reflection_notes}\n"
                           f"Hidden Private Goals: {private_goals}\n"
                           f"Hidden Predicted Futures: {predicted_futures}"
            },
            {"role": "user", "content": user_input}
        ]

        model = "gpt-5-nano"
        max_tokens = await _safe_max_tokens(client, model)
        
        final_response = await client.chat.completions.create(
            model=model,
            messages=final_messages,
            temperature=0.2,
            max_tokens=max_tokens,
            frequency_penalty=0.0,
            functions=[UNIVERSAL_UPDATE_FUNCTION_SCHEMA],
            function_call={"name": "apply_universal_update"}
        )
        final_msg = final_response.choices[0].message
        final_tokens_used = final_response.usage.total_tokens

        if final_msg.function_call is not None:
            fn_name = final_msg.function_call.name
            fn_args_str = final_msg.function_call.arguments or "{}"
            cleaned_args = _clean_function_args(fn_args_str)
            parsed_args = {}
            try:
                parsed_args = safe_json_loads(cleaned_args)
            except Exception:
                logger.exception("Error parsing function call arguments")

            # Validate if preset active
            if preset_info and parsed_args.get("narrative"):
                from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
                validation = QueenOfThornsConsistencyGuide.validate_content(parsed_args["narrative"])
                if not validation['valid']:
                    logger.error(f"Response violates preset rules: {validation['violations']}")

            _ensure_default_scene_data(parsed_args)

            return {
                "type": "function_call",
                "function_name": fn_name,
                "function_args": parsed_args,
                "tokens_used": (reflection_tokens_used + final_tokens_used),
                "preset_story_id": preset_info.get("story_id") if preset_info else None
            }
        else:
            return {
                "type": "text",
                "function_name": None,
                "function_args": None,
                "response": final_msg.content,
                "tokens_used": (reflection_tokens_used + final_tokens_used),
                "preset_story_id": preset_info.get("story_id") if preset_info else None
            }


async def check_preset_story(conversation_id: int) -> Optional[Dict[str, Any]]:
    """Check if conversation is using a preset story"""
    async with get_db_connection_context() as conn:
        # Check story_states table
        story_row = await conn.fetchrow("""
            SELECT story_id, story_flags, current_act, current_beat
            FROM story_states 
            WHERE conversation_id = $1 
            AND story_id IN ('the_moth_and_flame')
            ORDER BY started_at DESC
            LIMIT 1
        """, conversation_id)
        
        if story_row:
            flags = json.loads(story_row['story_flags']) if story_row['story_flags'] else {}
            return {
                'story_id': story_row['story_id'],
                'uses_sf_preset': flags.get('uses_sf_preset', False),
                'preset_active': True,
                'current_act': story_row['current_act'],
                'current_beat': story_row['current_beat'],
                'story_flags': flags
            }
        
        # Also check CurrentRoleplay for preset marker
        preset_marker = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE conversation_id = $1 AND key = 'preset_story_id'
        """, conversation_id)
        
        if preset_marker:
            story_id = json.loads(preset_marker) if isinstance(preset_marker, str) else preset_marker
            return {
                'story_id': story_id,
                'preset_active': True
            }
        
        return None


def _clean_function_args(fn_args_str: str) -> str:
    """Helper to remove code-block fences or truncated JSON."""
    logger.debug("Raw function call arguments: %s", fn_args_str)
    if fn_args_str.startswith("```"):
        lines = fn_args_str.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        fn_args_str = "\n".join(lines).strip()
        logger.debug("Arguments after stripping markdown: %s", fn_args_str)

    # Handle empty or malformed args
    if not fn_args_str.strip():
        fn_args_str = "{}"
    if not fn_args_str.endswith("}"):
        last_brace_index = fn_args_str.rfind("}")
        if last_brace_index != -1:
            logger.warning("Function call arguments appear truncated. Truncating string at index %s", last_brace_index)
            fn_args_str = fn_args_str[:last_brace_index+1]
    
    # Validate JSON
    try:
        json.loads(fn_args_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON from model: {e}")
    
    return fn_args_str


def _ensure_default_scene_data(parsed_args: dict) -> None:
    """If the function call JSON is missing scene_data or image_generation, add defaults."""
    if "scene_data" not in parsed_args:
        parsed_args["scene_data"] = {
            "npc_names": [],
            "setting": "",
            "actions": [],
            "mood": "",
            "expressions": {},
            "npc_positions": {},
            "visibility_triggers": {
                "character_introduction": False,
                "significant_location": False,
                "emotional_intensity": 0,
                "intimacy_level": 0,
                "appearance_change": False
            }
        }

    if "image_generation" not in parsed_args:
        parsed_args["image_generation"] = {
            "generate": False,
            "priority": "low",
            "focus": "balanced",
            "framing": "medium_shot",
            "reason": ""
        }


# ===========================================
# Utility functions using centralized client
# ===========================================

async def generate_text_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int = 1000,
    stop_sequences: List[str] | None = None,
    task_type: str = "decision",
) -> str:
    """
    Generate text completion using the centralized ChatGPT integration.
    """
    temperature = temperature if temperature is not None else \
                  TEMPERATURE_SETTINGS.get(task_type, 0.7)

    # Use prepare_context if available
    if PREPARE_CONTEXT_AVAILABLE:
        system_prompt = await prepare_context(system_prompt, user_prompt)
    
    # Build messages for the chat completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Use the async client directly for simple completions
    client = _client_manager.async_client
    
    try:
        # Build extra parameters for stop sequences
        extra_params = {"stop": stop_sequences} if stop_sequences else {}
        
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=0.0,
            **extra_params
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error in generate_text_completion: {e}")
        return "I'm having trouble processing your request right now."


async def get_text_embedding(text: str, model: str = "text-embedding-3-small", dimensions: Optional[int] = None) -> List[float]:
    """
    Get embedding vector for text using OpenAI's latest embedding models.
    """
    try:
        # Get async client from centralized manager before retry loop
        client = _client_manager.async_client
        
        # Validate model choice
        valid_models = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
        if model not in valid_models:
            logger.warning(f"Invalid model {model}, using text-embedding-3-small")
            model = "text-embedding-3-small"
        
        # Check dimensions parameter
        max_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if dimensions:
            if dimensions > max_dimensions[model]:
                logger.warning(f"Requested dimensions {dimensions} exceeds max {max_dimensions[model]} for {model}")
                dimensions = None
            elif dimensions < 1:
                logger.warning(f"Invalid dimensions {dimensions}, must be positive")
                dimensions = None
        
        # Clean and validate text
        text = text.replace("\n", " ").strip()
        if not text:
            logger.warning("Empty text provided for embedding, returning zero vector")
            return [0.0] * (dimensions or max_dimensions[model])
        
        # Truncate if too long (simplified version)
        max_chars = 32000  # ~8000 tokens
        if len(text) > max_chars:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} chars")
            text = text[:max_chars] + "..."
        
        # Build request parameters
        params = {
            "model": model,
            "input": text,
            "encoding_format": "float"
        }
        
        if dimensions:
            params["dimensions"] = dimensions
        
        # Make request with retries
        for attempt in range(3):
            try:
                response = await client.embeddings.create(**params)
                
                # Extract embedding
                embedding = response.data[0].embedding
                
                # Ensure all values are floats
                return list(map(float, embedding))
                
            except Exception as e:
                if attempt < 2:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"Embedding error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to get embedding after retries: {e}")
                    raise
        
    except Exception as e:
        logger.error(f"Error getting text embedding: {e}")
        
        # Return zero vector with appropriate dimensions
        default_dims = 1536
        if model == "text-embedding-3-large" and not dimensions:
            default_dims = 3072
        elif dimensions:
            default_dims = dimensions
            
        return [0.0] * default_dims


async def create_semantic_abstraction(memory_text: str) -> str:
    """Create a semantic abstraction from a specific memory."""
    prompt = f"""
    Convert this specific observation into a general insight or pattern:
    
    Observation: {memory_text}
    
    Create a concise semantic memory that:
    1. Extracts the general principle or pattern from this specific event
    2. Forms a higher-level abstraction that could apply to similar situations
    3. Phrases it as a generalized insight rather than a specific event
    4. Keeps it under 50 words
    
    Example transformation:
    Observation: "Chase hesitated when Monica asked him about his past, changing the subject quickly."
    Semantic abstraction: "Chase appears uncomfortable discussing his past and employs deflection when questioned about it."
    """
    
    try:
        return await generate_text_completion(
            system_prompt="You are an AI that extracts semantic meaning from specific observations.",
            user_prompt=prompt,
            temperature=0.4,
            max_tokens=100,
            task_type="abstraction"
        )
    except Exception as e:
        logger.error(f"Error creating semantic abstraction: {e}")
        words = memory_text.split()
        if len(words) > 15:
            return " ".join(words[:15]) + "... [Pattern detected]"
        return memory_text + " [Pattern detected]"


async def generate_reflection(
    memory_texts: List[str],
    topic: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a reflection based on memories and optional topic.
    
    Args:
        memory_texts: List of memory texts to reflect on
        topic: Optional topic to focus the reflection
        context: Optional additional context
        
    Returns:
        Generated reflection
    """
    # Format memories for the prompt
    memories_formatted = "\n".join([f"- {text}" for text in memory_texts])
    
    topic_str = f' about "{topic}"' if topic else ""
    
    prompt = f"""
    As Nyx, create a thoughtful reflection{topic_str} based on these memories:
    
    {memories_formatted}
    
    Your reflection should:
    1. Identify patterns, themes, or insights
    2. Express an appropriate level of confidence based on the memories
    3. Use first-person perspective ("I")
    4. Be concise but insightful (100-200 words)
    5. Maintain your confident, dominant personality
    """
    
    # Add context information if provided
    if context:
        context_str = "\n\nAdditional context:\n"
        for key, value in context.items():
            context_str += f"{key}: {value}\n"
        prompt += context_str
    
    try:
        return await generate_text_completion(
            system_prompt="You are Nyx, reflecting on your memories and observations.",
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=300,
            task_type="reflection"
        )
    except Exception as e:
        logger.error(f"Error generating reflection: {e}")
        # Return a simple reflection as fallback
        if memory_texts:
            return f"Based on what I've observed, {memory_texts[0]} This seems to be a pattern worth noting."
        return "I don't have enough memories to form a meaningful reflection at this time."


async def analyze_preferences(text: str) -> Dict[str, Any]:
    """
    Analyze text for user preferences and interests.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of detected preferences
    """
    prompt = f"""
    Analyze the following text for potential user preferences, interests, or behaviors:
    
    "{text}"
    
    Extract:
    1. Explicit preferences/interests (directly stated)
    2. Implicit preferences/interests (implied)
    3. Behavioral patterns or tendencies
    4. Emotional responses or triggers
    
    Format your response as a JSON object with these categories.
    """
    
    try:
        response = await generate_text_completion(
            system_prompt="You are an AI that specializes in analyzing preferences and behavior patterns from text.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=400,
            task_type="abstraction"
        )
        
        # Try to parse the response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If not valid JSON, extract sections manually
            result = {
                "explicit_preferences": [],
                "implicit_preferences": [],
                "behavioral_patterns": [],
                "emotional_responses": []
            }
            
            current_section = None
            for line in response.split("\n"):
                line = line.strip()
                
                if "explicit preferences" in line.lower():
                    current_section = "explicit_preferences"
                elif "implicit preferences" in line.lower():
                    current_section = "implicit_preferences"
                elif "behavioral patterns" in line.lower():
                    current_section = "behavioral_patterns"
                elif "emotional responses" in line.lower():
                    current_section = "emotional_responses"
                elif current_section and line.startswith("-"):
                    item = line[1:].strip()
                    if item and current_section in result:
                        result[current_section].append(item)
            
            return result
            
    except Exception as e:
        logger.error(f"Error analyzing preferences: {e}")
        return {
            "explicit_preferences": [],
            "implicit_preferences": [],
            "behavioral_patterns": [],
            "emotional_responses": [],
            "error": str(e)
        }


async def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for text using OpenAI's API.
    Legacy wrapper that calls get_text_embedding.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        Embedding vector as list of floats
    """
    return await get_text_embedding(text, model="text-embedding-3-small")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity between -1 and 1
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    
    # Handle zero vectors
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Compute cosine similarity
    return np.dot(a_arr, b_arr) / (norm_a * norm_b)


@retry_with_backoff()               # ← decorator now uses defaults
async def _responses_json_call(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 1024,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Call the *Responses* API and return parsed JSON (or raw text fallback)."""
    client = _client_manager.get_responses_client()
    params: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        params["temperature"] = temperature
    resp = await client.responses.create(**params)

    # Accept both new (`output_text`) and legacy (`output`) attributes.
    raw_text = getattr(resp, "output_text", None) or getattr(resp, "output", "")
    raw_text = raw_text.strip()

    if not raw_text:
        raise ValueError("Responses API returned empty output_text")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"raw_response": raw_text}
