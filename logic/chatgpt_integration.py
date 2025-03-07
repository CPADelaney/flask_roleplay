# logic/chatgpt_integration.py
import os
import json
import logging
import functools
import time
import openai
from db.connection import get_db_connection
from logic.prompts import SYSTEM_PROMPT
from logic.json_helpers import safe_json_loads

# Use your full schema, but add a "narrative" field at the top.
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
                "type": "object",
                "description": "Updates to CurrentRoleplay with arbitrary key-value pairs (e.g., 'CurrentYear', 'TimeOfDay' for time).",
                "properties": {
                    "CurrentYear": {"type": "number"},
                    "CurrentMonth": {"type": "number"},
                    "CurrentDay": {"type": "number"},
                    "TimeOfDay": {"type": "string"}
                },
                "additionalProperties": True
            },
            "ChaseSchedule": {
                "type": "object",
                "patternProperties": {
                    "^[A-Z][a-zA-Z]*$": {
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
                "additionalProperties": False,
                "description": "Chase’s detailed weekly schedule."
            },
            "MainQuest": {
                "type": "string",
                "description": "Summary of Chase’s main quest."
            },
            "PlayerRole": {
                "type": "string",
                "description": "Chase’s typical day/role in this environment."
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
                                }
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
                            "type": "object",
                            "patternProperties": {
                                "^[A-Z][a-zA-Z]*$": {
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
                            "additionalProperties": False
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
                    "required": ["npc_name"]
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
                            "type": "object",
                            "patternProperties": {
                                "^[A-Z][a-zA-Z]*$": {
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
                            "additionalProperties": False
                        },
                        "schedule_updates": {"type": "object"},
                        "affiliations": {"type": "array", "items": {"type": "string"}},
                        "current_location": {"type": "string"}
                    },
                    "required": ["npc_id"]
                }
            },
            "character_stat_updates": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "default": "Chase"},
                    "stats": {
                        "type": "object",
                        "properties": {
                            "corruption": {"type": "number"},
                            "confidence": {"type": "number"},
                            "willpower": {"type": "number"},
                            "obedience": {"type": "number"},
                            "dependency": {"type": "number"},
                            "lust": {"type": "number"},
                            "mental_resilience": {"type": "number"},
                            "physical_endurance": {"type": "number"}
                        }
                    }
                }
            },
            "relationship_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "npc_id": {"type": "number"},
                        "affiliations": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "npc_introductions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "npc_id": {"type": "number"}
                    },
                    "required": ["npc_id"]
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
                    "required": ["location_name"]
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
                        "fantasy_level": {  # Added for surreal events
                            "type": "string",
                            "enum": ["realistic", "fantastical", "surreal"],
                            "default": "realistic",
                            "description": "Indicates if the event involves reality-breaking elements."
                        }
                    }
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
                                    }
                                }
                            ]
                        }
                    },
                    "removed_items": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "object", "properties": {"item_name": {"type": "string"}}}
                            ]
                        }
                    }
                }
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
                    }
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
                        "group_context": {  # Added for group dynamics
                            "type": "string",
                            "description": "Describes multi-NPC interactions (e.g., 'NPC2 and five others gang up')."
                        }
                    }
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
                    }
                }
            },
            "activity_updates": {  # Added for punishment tracking
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
                            }
                        },
                        "stat_integration": {"type": "object"},
                        "intensity_tier": {"type": "number", "minimum": 0, "maximum": 4},
                        "setting_variant": {"type": "string"}
                    },
                    "required": ["activity_name"]
                },
                "description": "Create or update Activities for punishments, including surreal elements."
            },
            "journal_updates": {  # Added for PlayerJournal
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entry_type": {"type": "string", "enum": ["revelation", "moment", "dream"]},
                        "entry_text": {"type": "string"},
                        "fantasy_flag": {"type": "boolean", "default": False},
                        "intensity_level": {"type": "number", "minimum": 0, "maximum": 4}
                    },
                    "required": ["entry_type", "entry_text"]
                },
                "description": "Log narrative events or punishments in PlayerJournal, including surreal ones."
            },
            "image_generation": {  # Added for explicit image triggering
                "type": "object",
                "properties": {
                    "generate": {"type": "boolean", "default": False},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "low"},
                    "focus": {"type": "string", "enum": ["character", "scene", "action", "balanced"], "default": "balanced"},
                    "framing": {"type": "string", "enum": ["close_up", "medium_shot", "wide_shot"], "default": "medium_shot"},
                    "reason": {"type": "string", "description": "Why this image is generated (e.g., 'surreal punishment')."}
                },
                "description": "Trigger and configure image generation for the scene."
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
        ]
    }
}





def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")
    openai.api_key = api_key
    return openai

def build_message_history(conversation_id: int, aggregator_text: str, user_input: str, limit: int = 15):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT sender, content
        FROM messages
        WHERE conversation_id=%s
        ORDER BY id DESC
        LIMIT %s
    """, (conversation_id, limit))
    rows = cur.fetchall()
    conn.close()

    rows.reverse()  # Oldest first
    chat_history = []
    for sender, content in rows:
        role = "user" if sender.lower() == "user" else "assistant"
        chat_history.append({"role": role, "content": content})

    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "system", "content": aggregator_text})
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_input})
    return messages

def retry_with_backoff(max_retries=5, initial_delay=1, backoff_factor=2, exceptions=(openai.RateLimitError,)):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logging.warning(f"Rate limit hit on attempt {attempt+1}/{max_retries}: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
            raise Exception("Max retries exceeded")
        return wrapper_retry
    return decorator_retry

@retry_with_backoff(max_retries=5, initial_delay=1, backoff_factor=2, exceptions=(openai.RateLimitError,))
def get_chatgpt_response(conversation_id: int, aggregator_text: str, user_input: str) -> dict:
    """Get a response from OpenAI with image-aware prompting."""
    # First, get the user_id from the conversation_id
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conversation_id,))
            result = cur.fetchone()
            if not result:
                logging.error(f"Conversation {conversation_id} not found")
                return {"type": "text", "response": "Error: Conversation not found", "tokens_used": 0}
            user_id = result[0]
    
    # Get image-aware system prompt
    from logic.gpt_image_prompting import get_system_prompt_with_image_guidance
    system_prompt = get_system_prompt_with_image_guidance(user_id, conversation_id)
    
    # Create OpenAI client
    client = get_openai_client()
    
    # Build message history, but prepend the system prompt
    raw_messages = build_message_history(conversation_id, aggregator_text, user_input, limit=15)
    
    # Add the system prompt as the first message
    messages = [   
        {
            "role": "system",
            "content": PUBLIC_SYSTEM_PROMPT
        },
        {
            "role": "system",
            "content": PRIVATE_REFLECTION_INSTRUCTIONS
        },
        {
            "role": "user",
            "content": user_input
        }
    ]
    messages.extend(raw_messages)
    
    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        max_tokens=10000,
        frequency_penalty=0.0,
        functions=[UNIVERSAL_UPDATE_FUNCTION_SCHEMA],
        function_call={"name": "apply_universal_update"}
    )
    
    # Process the response
    msg = response.choices[0].message
    tokens_used = response.usage.total_tokens
    
    if msg.function_call is not None:
        fn_name = msg.function_call.name
        fn_args_str = msg.function_call.arguments or "{}"
        
        # Clean up markdown in args if present
        logging.debug("Raw function call arguments: %s", fn_args_str)
        if fn_args_str.startswith("```"):
            lines = fn_args_str.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            fn_args_str = "\n".join(lines).strip()
            logging.debug("Arguments after stripping markdown: %s", fn_args_str)
        
        # Handle empty or malformed args
        if not fn_args_str.strip():
            fn_args_str = "{}"
        if not fn_args_str.endswith("}"):
            last_brace_index = fn_args_str.rfind("}")
            if last_brace_index != -1:
                logging.warning("Function call arguments appear truncated. Truncating string at index %s", last_brace_index)
                fn_args_str = fn_args_str[:last_brace_index+1]
        
        try:
            parsed_args = safe_json_loads(fn_args_str)
            
            # Ensure scene_data is included for image generation
            if "scene_data" not in parsed_args:
                # Try to construct basic scene data
                parsed_args["scene_data"] = {
                    "npc_names": [],  # Extract from narrative if possible
                    "setting": "",     # Extract from narrative if possible
                    "actions": [],     # Extract from narrative if possible
                    "mood": "",        # Extract from narrative if possible
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
                
                # Optionally add explicit image generation field
                parsed_args["image_generation"] = {
                    "generate": False,  # Default to false
                    "priority": "low",
                    "focus": "balanced",
                    "framing": "medium_shot",
                    "reason": ""
                }
            
        except Exception:
            logging.exception("Error parsing function call arguments")
            parsed_args = {}
        
        return {
            "type": "function_call",
            "function_name": fn_name,
            "function_args": parsed_args,
            "tokens_used": tokens_used
        }
    else:
        return {
            "type": "text",
            "function_name": None,
            "function_args": None,
            "response": msg.content,
            "tokens_used": tokens_used
        }
