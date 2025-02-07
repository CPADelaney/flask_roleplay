# logic/chatgpt_integration.py
import os
import json
import logging
import openai
from db.connection import get_db_connection
from logic.prompts import SYSTEM_PROMPT

UNIVERSAL_UPDATE_FUNCTION_SCHEMA = {
    "name": "apply_universal_update",
    "description": "Insert or update various roleplay elements in the database (NPCs, locations, events, etc.).",
    "parameters": {
        "type": "object",
        "properties": {
            "roleplay_updates": {
                "type": "object",
                "description": "Updates to the CurrentRoleplay table, arbitrary key-value pairs."
            },
            "ChaseSchedule": {
                "type": "object",
                "properties": {
                    "Monday": {
                        "type": "object",
                        "properties": {
                            "Morning": {"type": "string"},
                            "Afternoon": {"type": "string"},
                            "Evening": {"type": "string"},
                            "Night": {"type": "string"}
                        },
                        "required": ["Morning", "Afternoon", "Evening", "Night"],
                        "additionalProperties": False
                    },
                    "Tuesday": {
                        "type": "object",
                        "properties": {
                            "Morning": {"type": "string"},
                            "Afternoon": {"type": "string"},
                            "Evening": {"type": "string"},
                            "Night": {"type": "string"}
                        },
                        "required": ["Morning", "Afternoon", "Evening", "Night"],
                        "additionalProperties": False
                    },
                    "Wednesday": {
                        "type": "object",
                        "properties": {
                            "Morning": {"type": "string"},
                            "Afternoon": {"type": "string"},
                            "Evening": {"type": "string"},
                            "Night": {"type": "string"}
                        },
                        "required": ["Morning", "Afternoon", "Evening", "Night"],
                        "additionalProperties": False
                    },
                    "Thursday": {
                        "type": "object",
                        "properties": {
                            "Morning": {"type": "string"},
                            "Afternoon": {"type": "string"},
                            "Evening": {"type": "string"},
                            "Night": {"type": "string"}
                        },
                        "required": ["Morning", "Afternoon", "Evening", "Night"],
                        "additionalProperties": False
                    },
                    "Friday": {
                        "type": "object",
                        "properties": {
                            "Morning": {"type": "string"},
                            "Afternoon": {"type": "string"},
                            "Evening": {"type": "string"},
                            "Night": {"type": "string"}
                        },
                        "required": ["Morning", "Afternoon", "Evening", "Night"],
                        "additionalProperties": False
                    },
                    "Saturday": {
                        "type": "object",
                        "properties": {
                            "Morning": {"type": "string"},
                            "Afternoon": {"type": "string"},
                            "Evening": {"type": "string"},
                            "Night": {"type": "string"}
                        },
                        "required": ["Morning", "Afternoon", "Evening", "Night"],
                        "additionalProperties": False
                    },
                    "Sunday": {
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
                "required": [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday"
                ],
                "additionalProperties": False,
                "description": "The detailed weekly schedule for Chase, with keys for each day of the week and nested keys for 'Morning', 'Afternoon', 'Evening', and 'Night'."
            },
            "MainQuest": {
                "type": "string",
                "description": "A short, intriguing summary of the main quest that Chase is about to undertake."
            },
            "PlayerRole": {
                "type": "string",
                "description": "A concise description of Chase's typical day (his role, career, and daily routine) in this environment."
            },
            "npc_creations": {
                "type": "array",
                "description": "Create new NPCs with stats/hobbies/affiliations, etc.",
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
                            "properties": {
                                "Monday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Tuesday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Wednesday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Thursday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Friday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Saturday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Sunday": {
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
                            "required": [
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday",
                                "Sunday"
                            ],
                            "additionalProperties": False,
                            "description": "The detailed weekly schedule for this NPC, formatted as a JSON object with keys for each day and nested keys for 'Morning', 'Afternoon', 'Evening', and 'Night'."
                        },
                        "memory": {
                            "description": "NPC memory can be a string or an array of strings.",
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ]
                        },
                        "monica_level": {"type": "number"},
                        "sex": {"type": "string"}
                    },
                    "required": ["npc_name"]
                },
                "description": "Create new NPCs with stats/hobbies/affiliations, etc."
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
                            "description": "NPC memory can be a string or an array of strings.",
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ]
                        },
                        "schedule": {
                            "type": "object",
                            "properties": {
                                "Monday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Tuesday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Wednesday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Thursday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Friday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Saturday": {
                                    "type": "object",
                                    "properties": {
                                        "Morning": {"type": "string"},
                                        "Afternoon": {"type": "string"},
                                        "Evening": {"type": "string"},
                                        "Night": {"type": "string"}
                                    },
                                    "required": ["Morning", "Afternoon", "Evening", "Night"],
                                    "additionalProperties": False
                                },
                                "Sunday": {
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
                            "required": [
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday",
                                "Sunday"
                            ],
                            "additionalProperties": False,
                            "description": "The detailed weekly schedule for this NPC."
                        },
                        "schedule_updates": {"type": "object"},
                        "affiliations": {"type": "array", "items": {"type": "string"}},
                        "current_location": {"type": "string"}
                    },
                    "required": ["npc_id"]
                },
                "description": "Update existing NPCs by npc_id."
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
                },
                "description": "Update the player's stats in PlayerStats."
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
                        "open_hours": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["location_name"]
                },
                "description": "Create new locations in the world."
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
                        "day": {"type": "number"},
                        "time_of_day": {"type": "string"},
                        "override_location": {"type": "string"}
                    }
                },
                "description": "Create or update normal Events or PlannedEvents."
            },
            "inventory_updates": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string"},
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
                                {
                                    "type": "object",
                                    "properties": {
                                        "item_name": {"type": "string"}
                                    }
                                }
                            ]
                        }
                    }
                },
                "description": "Add or remove items from the player's inventory."
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
                        "new_event": {"type": "string"}
                    }
                },
                "description": "NPC<->NPC or Player<->NPC relationships."
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
            }
        },
        "required": []
    }
}



def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")
    # You might just use openai, but if you want to keep a custom "OpenAI" usage, do so:
    openai.api_key = api_key
    return openai  # the standard openai module

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

    # Oldest first
    rows.reverse()

    chat_history = []
    for sender, content in rows:
        role = "user" if sender.lower() == "user" else "assistant"
        chat_history.append({"role": role, "content": content})

    # Start building final messages
    messages = []
    # system instructions
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    # aggregator text as developer or system
    messages.append({"role": "system", "content": aggregator_text})

    # existing conversation
    messages.extend(chat_history)

    # user message
    messages.append({"role": "user", "content": user_input})
    return messages

def get_chatgpt_response(conversation_id: int, aggregator_text: str, user_input: str) -> dict:
    client = get_openai_client()

    # Build the conversation history as a list of message dicts
    messages = build_message_history(conversation_id, aggregator_text, user_input, limit=15)

    # Pass the messages along with functions and function_call parameters
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
        frequency_penalty=0.0,
        functions=[UNIVERSAL_UPDATE_FUNCTION_SCHEMA],
        function_call="auto"
    )

    msg = response.choices[0].message
    tokens_used = response.usage.total_tokens

    # Check if GPT called the function
    if msg.function_call is not None:
        fn_name = msg.function_call.name
        fn_args_str = msg.function_call.arguments or "{}"
        try:
            parsed_args = json.loads(fn_args_str)
        except Exception:
            logging.exception("Error parsing function call arguments")
            parsed_args = {}

        return {
            "type": "function_call",
            "function_name": fn_name,
            "function_args": parsed_args,
            "response": None,
            "tokens_used": tokens_used
        }
    else:
        # Return a normal text response
        return {
            "type": "text",
            "function_name": None,
            "function_args": None,
            "response": msg.content,
            "tokens_used": tokens_used
        }

