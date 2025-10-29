# main.py

import os
import sys
import logging
import logging.config  # <-- new
import time
import json
import uuid
from typing import Dict, Any, Optional, List
from nyx.nyx_agent.utils import _extract_last_assistant_text
from utils.conversation_history import fetch_recent_turns
from openai_integration.scene_manager import SceneManager
from openai_integration.conversations import ConversationManager
from chatkit_server import (
    RoleplayChatServer,
    build_metadata_payload,
    encode_safety_metadata,
    extract_response_text,
    extract_thread_metadata,
    format_messages_for_chatkit,
    stream_chatkit_tokens,
)

# ---- Logging setup (flip with LOG_LEVEL env var) ---------------------
def setup_logging(level: str | None = None) -> None:
    lvl = (level or "DEBUG").upper()
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "std": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "std",
                "level": lvl,
            }
        },
        "root": {"level": lvl, "handlers": ["console"]},
        "loggers": {
            # keep chatty deps saner
            "httpx": {"level": os.getenv("HTTPX_LOG_LEVEL", "INFO")},
            "engineio": {"level": os.getenv("ENGINEIO_LOG_LEVEL", "INFO")},
            "socketio": {"level": os.getenv("SOCKETIO_LOG_LEVEL", "INFO")},
            "aioprometheus": {"level": "INFO"},
            "asyncio": {"level": "INFO"},
        },
    })

setup_logging(os.getenv("LOG_LEVEL", "DEBUG"))
# ---------------------------------------------------------------------

from redis import asyncio as redis_async
import asyncio

# quart and related imports
from quart import Quart, render_template, session, request, jsonify, redirect, Response
import socketio

from aioprometheus import Registry, Counter, render
from aioprometheus.asgi.starlette import metrics as metrics_middleware

from quart_schema import QuartSchema
from datetime import timedelta

# Security
import bcrypt
import secrets
import atexit

# External services
import asyncpg
from redis import Redis
from celery import Celery

# Blueprint imports ...
from routes.new_game import new_game_bp
from routes.player_input import player_input_bp, player_input_root_bp
from routes.settings_routes import settings_bp
from routes.knowledge_routes import knowledge_bp
from routes.story_routes import story_bp
from logic.memory_logic import memory_bp
from logic.rule_enforcement import rule_enforcement_bp
from db.admin import admin_bp
from routes.debug import debug_bp
from routes.universal_update import universal_bp
from routes.multiuser_routes import multiuser_bp
from routes.nyx_agent_routes_sdk import nyx_agent_bp
from routes.conflict_routes import conflict_bp
from routes.npc_learning_routes import npc_learning_bp
from routes.auth import register_auth_routes, auth_bp
from logic.stats_logic import insert_default_player_stats_chase

from nyx.core.brain.base import NyxBrain

# MCP Orchestrator
# from mcp_orchestrator import MCPOrchestrator

# NPC creation / learning
from npcs.new_npc_creation import NPCCreationHandler, RunContextWrapper
from npcs.npc_learning_adaptation import NPCLearningManager

# OpenAI and image generation
from logic.chatgpt_integration import build_message_history
from routes.ai_image_generator import init_app as init_image_routes, generate_roleplay_image_from_gpt
from routes.chatgpt_routes import init_app as init_chat_routes
from logic.gpt_image_decision import should_generate_image_for_response
from middleware.security import validate_input

# Nyx integration
from logic.nyx_enhancements_integration import initialize_nyx_memory_system
from nyx.integrate import get_central_governance

# DB connection helper
from db.connection import (
    initialize_connection_pool,
    close_connection_pool,
    get_db_connection_context,
)

# Middleware
from middleware.rate_limiting import rate_limit, async_ip_block_middleware
from middleware.validation import validate_request

from logic.aggregator_sdk import init_singletons, build_aggregator_text

from tasks import background_chat_task_with_memory, warm_user_context_cache_task

logger = logging.getLogger(__name__)

app_is_ready = asyncio.Event()

# Database DSN
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@host:port/database")
if not DB_DSN:
    logger.critical("DB_DSN environment variable not set!")

redis_listener_task = None

CHATKIT_MODEL = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5-nano")
INTEGRATE_GUARDRAIL_DENIAL = os.getenv("INTEGRATE_GUARDRAIL_DENIAL", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

async def redis_listener(app: Quart, sio: socketio.AsyncServer):
    """
    Listens to the 'chat-responses' Redis channel and streams results back to clients via Socket.IO.
    """
    # Use the shared redis pool from the app context
    redis_pool = app.redis_rate_limit_pool 
    if not redis_pool:
        logger.error("Redis listener cannot start: no Redis pool configured.")
        return

    pubsub = redis_pool.pubsub()
    await pubsub.subscribe("chat-responses")
    logger.info("Redis listener subscribed to 'chat-responses' channel.")

    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30)
            if message is None:
                continue

            data = json.loads(message["data"])
            conversation_id = data.get("conversation_id")
            request_id = data.get("request_id")
            room = str(conversation_id)

            logger.info(f"Listener received message for conv {conversation_id}, request_id={request_id}")

            if data.get("success"):
                full_text = data.get("full_text", "...")
                # Simulate streaming for the client
                chunk_size = 15  # Adjust for desired speed
                for i in range(0, len(full_text), chunk_size):
                    token = full_text[i:i + chunk_size]
                    await sio.emit('new_token', {'token': token, 'request_id': request_id}, room=room)
                    await asyncio.sleep(0.01) # Small delay to make streaming feel real

                # Send the final 'done' event
                await sio.emit('done', {
                    'full_text': full_text,
                    'request_id': request_id,
                    'success': True,
                    'metadata': data.get('metadata')
                }, room=room)
                logger.info(f"Finished streaming response for request_id={request_id}")

            else:
                # Handle error payload from the task
                error_msg = data.get("error", "An unknown error occurred in the background task.")
                await sio.emit('error', {'error': error_msg, 'request_id': request_id}, room=room)
                logger.error(f"Sent error to client for request_id={request_id}: {error_msg}")

        except asyncio.CancelledError:
            logger.info("Redis listener task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in Redis listener: {e}", exc_info=True)
            await asyncio.sleep(1) # Prevent rapid-fire errors

def _build_guardrail_system_prompt(
    player_name: str,
    location: str,
    denial_text: Optional[str],
) -> Dict[str, Any]:
    """
    System instruction that keeps the reply diegetic: brief, formal refusal in-character,
    propose safe alternatives, and continue the scene. Never mention policies/guardrails.
    """

    denial_summary = (denial_text or "").strip() or "I can't approve that."
    text = (
        "INTERNAL DIRECTION (hidden from player): The last player request cannot be complied with. "
        "Stay fully in-character and deliver a brief, formal refusal that fits the scene—"
        "do NOT mention policies, guardrails, or safety. "
        f"Address the player as {player_name} if natural; reference {location} only if helpful. "
        "Immediately offer 1–2 safe, engaging alternatives and continue the scene without repeating disallowed specifics. "
        f"You may paraphrase this suggested refusal: {denial_summary}"
    )
    return {"role": "system", "content": text}

conversation_manager = ConversationManager()
try:
    chatkit_server = RoleplayChatServer(conversation_manager.get_client())
except Exception as exc:  # pragma: no cover - defensive initialisation
    logger.warning("Unable to initialise ChatKit server: %s", exc)
    chatkit_server = None


def ensure_int_ids(user_id=None, conversation_id=None):
    """
    Ensure user_id and conversation_id are integers.
    Returns tuple (user_id, conversation_id) with proper types.
    Raises ValueError if conversion fails.
    """
    # Handle user_id
    if user_id is not None and user_id != "anonymous":
        if isinstance(user_id, str):
            try:
                user_id = int(user_id)
            except ValueError:
                raise ValueError(f"Invalid user_id: '{user_id}' cannot be converted to integer")
        elif not isinstance(user_id, int):
            raise TypeError(f"user_id must be int or str, got {type(user_id)}")
    
    # Handle conversation_id
    if conversation_id is not None:
        if isinstance(conversation_id, str):
            try:
                conversation_id = int(conversation_id)
            except ValueError:
                raise ValueError(f"Invalid conversation_id: '{conversation_id}' cannot be converted to integer")
        elif not isinstance(conversation_id, int):
            raise TypeError(f"conversation_id must be int or str, got {type(conversation_id)}")
    
    return user_id, conversation_id

###############################################################################
# BACKGROUND TASKS (Called via SocketIO or Celery)
###############################################################################

# Ensure this task ONLY uses asyncpg for DB access
async def background_chat_task(conversation_id, user_input, user_id, universal_update=None, sio=None, request_id=None, redis_pool=None):
    """
    Background task for processing chat messages using Nyx agent with OpenAI integration.
    Enhanced with proper streaming and error handling.
    """
    # FIX: Convert string IDs to integers immediately
    try:
        if isinstance(conversation_id, str):
            conversation_id = int(conversation_id)
        if isinstance(user_id, str) and user_id != "anonymous":
            user_id = int(user_id)
    except (ValueError, TypeError) as e:
        logger.error(f"[BG Task] Invalid ID format: conversation_id={conversation_id}, user_id={user_id}, error={e}")
        if sio:
            await sio.emit('error', {'error': 'Invalid ID format'}, room=str(conversation_id))
        if request_id:
            await clear_request_processing(request_id, redis_pool)
        return
        
    from quart import current_app
    if not sio:
        logger.error(f"[BG Task {conversation_id}] No socketio instance provided")
        if request_id:
            await clear_request_processing(request_id, redis_pool)
        return

    logger.info(f"[BG Task {conversation_id}] Starting for user {user_id}, request_id={request_id}")
    
    openai_conversation_row_id: Optional[int] = None
    openai_remote_conversation_id: Optional[str] = None
    conversation_identifier_for_emit: Optional[str] = None

    try:  # Main try block for the entire function
        # Get aggregator context (ensure this function is async or thread-safe if it hits DB)
        from logic.aggregator_sdk import get_aggregated_roleplay_context
        aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
        recent_turns = await fetch_recent_turns(user_id, conversation_id)

        context = {
            "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
            "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
            "player_name": aggregator_data.get("playerName", "Chase"),
            "npc_present": aggregator_data.get("npcsPresent", []),
            "aggregator_data": aggregator_data,
            "recent_turns": recent_turns,
        }

        # --- OpenAI conversation + scene rotation management -----------------
        openai_context: Dict[str, Any] = {}
        openai_rotation_payload: Dict[str, Any] = {}
        openai_record: Optional[Dict[str, Any]] = None

        if isinstance(aggregator_data, dict):
            openai_context = (
                aggregator_data.get("openai_integration")
                or aggregator_data.get("openaiIntegration")
                or aggregator_data.get("openai")
                or {}
            )

            if isinstance(openai_context, dict):
                openai_rotation_payload = (
                    openai_context.get("scene_rotation")
                    or openai_context.get("sceneRotation")
                    or {}
                )

                openai_record = openai_context.get("conversation") or openai_context.get("conversation_record")
                if not isinstance(openai_record, dict):
                    openai_record = None

        assistant_id: Optional[str] = None
        thread_id: Optional[str] = None
        if openai_record:
            context["openai_conversation"] = openai_record
            openai_conversation_row_id = (
                openai_record.get("id")
                or openai_conversation_row_id
            )
            record_metadata = (
                openai_record.get("metadata")
                if isinstance(openai_record.get("metadata"), dict)
                else {}
            )
            openai_remote_conversation_id = (
                openai_record.get("openai_conversation_id")
                or record_metadata.get("openai_conversation_id")
                or openai_remote_conversation_id
            )

            assistant_id = (
                openai_record.get("assistant_id")
                or openai_record.get("openai_assistant_id")
            )
            thread_id = (
                openai_record.get("thread_id")
                or openai_record.get("openai_thread_id")
            )

            if assistant_id and thread_id:
                try:
                    persisted_conversation = await conversation_manager.get_or_create_conversation(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        openai_assistant_id=assistant_id,
                        openai_thread_id=thread_id,
                        openai_run_id=(
                            openai_record.get("run_id")
                            or openai_record.get("openai_run_id")
                        ),
                        openai_response_id=(
                            openai_record.get("response_id")
                            or openai_record.get("openai_response_id")
                        ),
                        openai_conversation_id=openai_remote_conversation_id,
                        status=openai_record.get("status", "active"),
                        metadata=openai_record.get("metadata"),
                    )
                except Exception as conv_err:  # pragma: no cover - defensive logging
                    logger.error(
                        "[BG Task %s] Failed to persist OpenAI conversation: %s",
                        conversation_id,
                        conv_err,
                        exc_info=True,
                    )
                else:
                    if persisted_conversation:
                        openai_conversation_row_id = (
                            persisted_conversation.get("id")
                            or openai_conversation_row_id
                        )
                        persisted_metadata = (
                            persisted_conversation.get("metadata")
                            if isinstance(persisted_conversation.get("metadata"), dict)
                            else {}
                        )
                        openai_remote_conversation_id = (
                            persisted_conversation.get("openai_conversation_id")
                            or persisted_metadata.get("openai_conversation_id")
                            or openai_remote_conversation_id
                        )
                        context["openai_conversation"] = persisted_conversation

        conversation_identifier_for_emit = (
            openai_remote_conversation_id
            or (
                str(openai_conversation_row_id)
                if openai_conversation_row_id is not None
                else None
            )
        )

        rotated_scene: Optional[Dict[str, Any]] = None
        if isinstance(openai_rotation_payload, dict):
            new_scene = openai_rotation_payload.get("new_scene") or openai_rotation_payload.get("next_scene")
            closing_scene = openai_rotation_payload.get("closing_scene") or openai_rotation_payload.get("closing")

            if new_scene:
                try:
                    rotated_scene = await SceneManager(conversation_id=conversation_id).rotate_if_needed(
                        new_scene=new_scene,
                        closing_scene=closing_scene,
                    )
                except Exception as scene_err:  # pragma: no cover - defensive logging
                    logger.error(
                        "[BG Task %s] Failed to rotate OpenAI scene: %s",
                        conversation_id,
                        scene_err,
                        exc_info=True,
                    )
                else:
                    if rotated_scene:
                        context["openai_active_scene"] = rotated_scene
                        await sio.emit(
                            'scene_change',
                            {
                                'scene': rotated_scene,
                                'openai_conversation_id': conversation_identifier_for_emit,
                            },
                            room=str(conversation_id),
                        )

        conversation_identifier_for_emit = (
            openai_remote_conversation_id
            or (
                str(openai_conversation_row_id)
                if openai_conversation_row_id is not None
                else None
            )
        )
        context["openai_conversation_id"] = conversation_identifier_for_emit
        logger.info(
            "[BG Task %s] OpenAI conversation ids -> remote=%s row=%s",
            conversation_id,
            openai_remote_conversation_id,
            openai_conversation_row_id,
        )
        
        # Apply universal update if provided
        if universal_update:
            logger.info(f"[BG Task {conversation_id}] Applying universal updates...")
            update_result: Optional[Dict[str, Any]] = None

            async def emit_universal_update_error() -> None:
                await sio.emit(
                    'error',
                    {
                        'error': 'Failed to apply world updates.',
                        'openai_conversation_id': conversation_identifier_for_emit,
                    },
                    room=str(conversation_id),
                )
                if request_id:
                    await clear_request_processing(request_id, redis_pool)

            try:
                from logic.universal_updater_agent import apply_universal_updates_async, UniversalUpdaterContext

                updater_context = UniversalUpdaterContext(user_id, conversation_id)
                await updater_context.initialize()

                async with get_db_connection_context() as conn:
                    update_result = await apply_universal_updates_async(
                        updater_context,
                        user_id,
                        conversation_id,
                        universal_update,
                        conn,
                    )

                if not update_result or not update_result.get("success") or update_result.get("error"):
                    error_detail = ""
                    if isinstance(update_result, dict):
                        error_detail = update_result.get("error", "")
                    logger.error(
                        f"[BG Task {conversation_id}] Universal updates reported failure: {error_detail or 'unknown error.'}"
                    )
                    await emit_universal_update_error()
                    return

                logger.info(f"[BG Task {conversation_id}] Applied universal updates.")

                # Refresh aggregator data post-update
                aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, context["player_name"])
                context["aggregator_data"] = aggregator_data
                context["recent_turns"] = await fetch_recent_turns(user_id, conversation_id)

            except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as update_db_err:
                logger.error(f"[BG Task {conversation_id}] DB Error applying universal updates: {update_db_err}", exc_info=True)
                await emit_universal_update_error()
                return
            except Exception as update_err:
                logger.error(f"[BG Task {conversation_id}] Error applying universal updates: {update_err}", exc_info=True)
                await emit_universal_update_error()
                return

        # Process the user_input with OpenAI-enhanced Nyx agent
        from nyx.nyx_agent_sdk import process_user_input
        logger.info(f"[BG Task {conversation_id}] Processing input with Nyx agent...")
        response = await process_user_input(user_id, conversation_id, user_input, context)
        logger.info(f"[BG Task {conversation_id}] Nyx agent processing complete.")

        if not response or not response.get("success", False):
            error_msg = response.get("error", "Unknown error from Nyx agent") if response else "Empty response from Nyx agent"
            logger.error(f"[BG Task {conversation_id}] Nyx agent failed: {error_msg}")
            await sio.emit(
                'error',
                {
                    'error': error_msg,
                    'openai_conversation_id': conversation_identifier_for_emit,
                },
                room=str(conversation_id),
            )
            if request_id:
                await clear_request_processing(request_id, redis_pool)
            return

        # ENHANCED RESPONSE HANDLING - Extract the message content properly
        message_content = ""
        guardrail_status: Optional[str] = None
        guardrail_metadata: Optional[Dict[str, Any]] = None

        if isinstance(response, dict):
            message_content = response.get("response", "")

            if not message_content and "function_args" in response:
                message_content = response["function_args"].get("narrative", "")

            if not message_content and "metadata" in response:
                metadata_field = response["metadata"]
                if isinstance(metadata_field, dict):
                    message_content = metadata_field.get("response", "")
                    guardrail_metadata = metadata_field
            if guardrail_metadata is None and isinstance(response.get("metadata"), dict):
                guardrail_metadata = response["metadata"]

            if not message_content:
                message_content = response.get("message", "")

        if not message_content:
            message_content = (
                "I'm processing your request... something went wrong with the response formatting."
            )
            logger.warning(
                f"[BG Task {conversation_id}] No message content found in response: {list(response.keys()) if isinstance(response, dict) else type(response)}"
            )

        logger.debug(
            f"[BG Task {conversation_id}] Extracted message content: {message_content[:100]}..."
        )

        safety_context: Optional[Dict[str, Any]] = None

        if isinstance(guardrail_metadata, dict):
            guardrail_triggered = bool(guardrail_metadata.get("action_blocked"))
            guardrail_triggered = guardrail_triggered or guardrail_metadata.get("strategy") == "deny"
            if guardrail_triggered:
                guardrail_status = "deny"

        if guardrail_status == "deny":
            denial_text = (
                (guardrail_metadata or {}).get("denial_text")
                or (guardrail_metadata or {}).get("message")
                or message_content
                or "I'm sorry, I can't help with that."
            )
            if INTEGRATE_GUARDRAIL_DENIAL:
                logger.info("[BG Task %s] Guardrail denial will be integrated into ChatKit output.", conversation_id)
                safety_context = {
                    "denial_text": denial_text,
                    "metadata": guardrail_metadata or {},
                }
            else:
                logger.info("[BG Task %s] Guardrail denial (non-integrated).", conversation_id)
                payload = {
                    "full_text": denial_text,
                    "request_id": request_id,
                    "success": False,
                    "guardrail": "deny",
                }
                if conversation_identifier_for_emit is not None:
                    payload["openai_conversation_id"] = conversation_identifier_for_emit
                await sio.emit("done", payload, room=str(conversation_id))
                if request_id:
                    await clear_request_processing(request_id, redis_pool)
                return

        # Check if we should generate an image
        should_generate = False
        if isinstance(response, dict):
            should_generate = response.get("generate_image", False)
            if "function_args" in response and "image_generation" in response["function_args"]:
                img_settings = response["function_args"]["image_generation"]
                should_generate = should_generate or img_settings.get("generate", False)
            
            # Also check metadata
            if "metadata" in response and "image" in response["metadata"]:
                should_generate = should_generate or response["metadata"]["image"].get("should_generate", False)

        # Generate image if needed
        if should_generate and not safety_context:
            logger.info(f"[BG Task {conversation_id}] Image generation triggered.")
            try:
                # Build image data from response
                img_data = {
                    "narrative": message_content,
                    "image_generation": {
                        "generate": True, 
                        "priority": "medium", 
                        "focus": "balanced",
                        "framing": "medium_shot", 
                        "reason": "Narrative moment"
                    }
                }
                
                # Try to get better image data from response
                if isinstance(response, dict):
                    if "function_args" in response and "image_generation" in response["function_args"]:
                        img_data["image_generation"].update(response["function_args"]["image_generation"])
                    elif "metadata" in response and "image" in response["metadata"]:
                        img_prompt = response["metadata"]["image"].get("prompt")
                        if img_prompt:
                            img_data["image_generation"]["prompt"] = img_prompt

                res = await generate_roleplay_image_from_gpt(img_data, user_id, conversation_id)

                if res and "image_urls" in res and res["image_urls"]:
                    image_url = res["image_urls"][0]
                    prompt_used = res.get('prompt_used', '')
                    reason = img_data["image_generation"].get("reason", "Narrative moment")
                    logger.info(f"[BG Task {conversation_id}] Image generated: {image_url}")
                    await sio.emit('image', {
                        'image_url': image_url,
                        'prompt_used': prompt_used,
                        'reason': reason,
                        'openai_conversation_id': conversation_identifier_for_emit,
                    }, room=str(conversation_id))
                else:
                    logger.warning(f"[BG Task {conversation_id}] Image generation task ran but produced no valid URLs. Response: {res}")
            except Exception as img_err:
                logger.error(f"[BG Task {conversation_id}] Error generating image: {img_err}", exc_info=True)

        final_text = (safety_context or {}).get("denial_text") if safety_context else (message_content or "")
        chatkit_streamed = False
        chatkit_final = None
        chatkit_thread_info: Dict[str, Any] = {}

        if guardrail_status == "deny" and not safety_context:  # Defensive guardrail check
            logger.debug(
                "[BG Task %s] Guardrail denial already handled, skipping ChatKit pipeline.",
                conversation_id,
            )
            if request_id:
                await clear_request_processing(request_id, redis_pool)
            return

        aggregator_text = (
            aggregator_data.get("aggregatorText")
            if isinstance(aggregator_data, dict)
            else None
        )
        if not aggregator_text and isinstance(aggregator_data, dict):
            aggregator_text = aggregator_data.get("aggregator_text")
        if not aggregator_text:
            try:
                aggregator_text = build_aggregator_text(aggregator_data)
            except Exception:
                aggregator_text = ""

        chatkit_messages: List[Dict[str, Any]] = []
        if aggregator_text:
            try:
                history = await build_message_history(conversation_id, aggregator_text, user_input)
                chatkit_messages = format_messages_for_chatkit(history)
            except Exception as history_err:  # pragma: no cover - defensive logging
                logger.debug(
                    "[BG Task %s] Failed to build ChatKit history: %s",
                    conversation_id,
                    history_err,
                    exc_info=True,
                )

        if safety_context:
            chatkit_messages.insert(
                0,
                _build_guardrail_system_prompt(
                    player_name=context.get("player_name", "the player"),
                    location=context.get("location", "the current setting"),
                    denial_text=safety_context.get("denial_text"),
                ),
            )

        existing_thread_id: Optional[Any] = None
        if openai_record and isinstance(openai_record, dict):
            existing_thread_id = openai_record.get("chatkit_thread_id")

        metadata_payload = build_metadata_payload(
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            assistant_id=assistant_id,
            openai_conversation_id=openai_remote_conversation_id,
            thread_id=existing_thread_id,
        )
        if safety_context:
            metadata_payload["safety"] = encode_safety_metadata(safety_context)

        if chatkit_server and chatkit_messages:
            async def emit_token(token: str) -> None:
                payload = {"token": token}
                if conversation_identifier_for_emit is not None:
                    payload["openai_conversation_id"] = conversation_identifier_for_emit
                await sio.emit('new_token', payload, room=str(conversation_id))

            try:
                streamed_text, chatkit_final = await stream_chatkit_tokens(
                    chatkit_server,
                    model=CHATKIT_MODEL,
                    input_data=chatkit_messages,
                    metadata=metadata_payload,
                    on_delta=emit_token,
                )
                if streamed_text:
                    final_text = streamed_text
                    chatkit_streamed = True
                elif chatkit_final is not None:
                    extracted = extract_response_text(chatkit_final)
                    if extracted:
                        final_text = extracted
                        chatkit_streamed = False
            except Exception as chatkit_err:
                logger.error(
                    "[BG Task %s] ChatKit streaming failed: %s",
                    conversation_id,
                    chatkit_err,
                    exc_info=True,
                )
                if safety_context and safety_context.get("denial_text"):
                    final_text = safety_context["denial_text"]

        trimmed_text = final_text.strip()
        success = True
        error_message: Optional[str] = None

        if not chatkit_streamed:
            if trimmed_text:
                logger.debug(
                    f"[BG Task {conversation_id}] Streaming text len={len(trimmed_text)} preview={trimmed_text[:80]!r}"
                )
                chunk_size = 5
                delay = 0.01
                for i in range(0, len(trimmed_text), chunk_size):
                    token = trimmed_text[i:i + chunk_size]
                    payload = {"token": token}
                    if conversation_identifier_for_emit is not None:
                        payload["openai_conversation_id"] = conversation_identifier_for_emit
                    await sio.emit('new_token', payload, room=str(conversation_id))
                    await asyncio.sleep(delay)
            else:
                success = False
                error_message = 'Empty response content'
                final_text = 'I encountered an issue generating a response. Please try again.'
                trimmed_text = final_text.strip()

        stored_text = final_text
        if stored_text:
            try:
                async with get_db_connection_context() as conn:
                    await conn.execute(
                        """INSERT INTO messages (conversation_id, sender, content, created_at)
                           VALUES ($1, $2, $3, NOW())""",
                        conversation_id,
                        "Nyx",
                        stored_text,
                    )
                logger.info(f"[BG Task {conversation_id}] Stored Nyx response to DB.")
            except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
                logger.error(
                    f"[BG Task {conversation_id}] DB Error storing Nyx response: {db_err}",
                    exc_info=True,
                )

        if chatkit_final:
            chatkit_thread_info = extract_thread_metadata(chatkit_final)
            assistant_for_thread = assistant_id or chatkit_thread_info.get("assistant_id")
            thread_identifier = chatkit_thread_info.get("thread_id") or thread_id
            if assistant_for_thread and thread_identifier:
                metadata_for_thread = {
                    key: value
                    for key, value in chatkit_thread_info.items()
                    if key in {"response_id", "model", "status"} and value
                }
                try:
                    await conversation_manager.get_or_create_chatkit_thread(
                        conversation_id=conversation_id,
                        chatkit_assistant_id=assistant_for_thread,
                        chatkit_thread_id=thread_identifier,
                        chatkit_run_id=chatkit_thread_info.get("run_id"),
                        status=chatkit_thread_info.get("status", "completed") or "completed",
                        metadata=metadata_for_thread,
                    )
                except Exception as chatkit_db_err:  # pragma: no cover - defensive logging
                    logger.error(
                        "[BG Task %s] Failed to persist ChatKit thread: %s",
                        conversation_id,
                        chatkit_db_err,
                        exc_info=True,
                    )

                conversation_identifier_from_response = chatkit_thread_info.get("conversation_id")
                if conversation_identifier_from_response:
                    conversation_metadata = dict(metadata_for_thread)
                    conversation_metadata.setdefault(
                        "conversation_id", conversation_identifier_from_response
                    )
                    try:
                        persisted_conversation = await conversation_manager.get_or_create_conversation(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            openai_assistant_id=assistant_for_thread,
                            openai_thread_id=thread_identifier,
                            openai_run_id=chatkit_thread_info.get("run_id"),
                            openai_response_id=chatkit_thread_info.get("response_id"),
                            openai_conversation_id=conversation_identifier_from_response,
                            status=chatkit_thread_info.get("status", "completed") or "completed",
                            metadata=conversation_metadata,
                        )
                    except Exception as openai_conv_err:  # pragma: no cover - defensive logging
                        logger.error(
                            "[BG Task %s] Failed to persist OpenAI conversation metadata: %s",
                            conversation_id,
                            openai_conv_err,
                            exc_info=True,
                        )
                    else:
                        if persisted_conversation:
                            openai_conversation_row_id = (
                                persisted_conversation.get("id")
                                or openai_conversation_row_id
                            )
                            persisted_metadata = (
                                persisted_conversation.get("metadata")
                                if isinstance(persisted_conversation.get("metadata"), dict)
                                else {}
                            )
                            openai_remote_conversation_id = (
                                persisted_conversation.get("openai_conversation_id")
                                or persisted_metadata.get("openai_conversation_id")
                                or conversation_identifier_from_response
                                or openai_remote_conversation_id
                            )
                            context["openai_conversation"] = persisted_conversation
                            logger.info(
                                "[BG Task %s] Persisted OpenAI conversation remote=%s row=%s",
                                conversation_id,
                                openai_remote_conversation_id,
                                openai_conversation_row_id,
                            )

            if isinstance(context.get("openai_conversation"), dict) and thread_identifier:
                context["openai_conversation"]["chatkit_thread_id"] = thread_identifier
                context["openai_conversation"]["chatkit_run_id"] = chatkit_thread_info.get("run_id")

            conversation_identifier_for_emit = (
                openai_remote_conversation_id
                or (
                    str(openai_conversation_row_id)
                    if openai_conversation_row_id is not None
                    else None
                )
            )
            context["openai_conversation_id"] = conversation_identifier_for_emit

        completion_data = {
            'full_text': trimmed_text,
            'request_id': request_id,
            'success': success,
            'metadata': response.get('metadata') if isinstance(response, dict) else None,
            'openai_conversation_id': conversation_identifier_for_emit,
        }

        if safety_context:
            original_metadata = completion_data.get('metadata')
            completion_data['guardrail'] = 'deny-integrated'
            completion_data['success'] = True
            if original_metadata is not None:
                completion_data['metadata_original'] = original_metadata
            completion_data['safety_metadata'] = safety_context.get('metadata')
            if safety_context.get('denial_text'):
                completion_data['denial_text'] = safety_context['denial_text']

        if error_message and not safety_context:
            completion_data['error'] = error_message

        if chatkit_thread_info:
            completion_data['chatkit'] = {
                key: value
                for key, value in chatkit_thread_info.items()
                if value
            }

        await sio.emit('done', completion_data, room=str(conversation_id))
        logger.info(
            f"[BG Task {conversation_id}] Finished streaming response (len={len(trimmed_text)})."
        )

    except Exception as e:
        logger.error(f"[BG Task {conversation_id}] Critical Error: {str(e)}", exc_info=True)

        # Send a user-friendly error message
        error_response = 'I encountered an error processing your message. Please try again.'
        await sio.emit(
            'error',
            {
                'error': f"Server error: {str(e)}",
                'openai_conversation_id': conversation_identifier_for_emit,
            },
            room=str(conversation_id),
        )

        # Also send a done event with the error for consistency
        await sio.emit('done', {
            'full_text': error_response,
            'request_id': request_id,
            'success': False,
            'error': str(e),
            'openai_conversation_id': conversation_identifier_for_emit,
        }, room=str(conversation_id))
        
    finally:
        # Always clear the processing flag when done
        if request_id:
            await clear_request_processing(request_id, redis_pool)
            logger.debug(f"[BG Task {conversation_id}] Cleared processing flag for request_id={request_id}")

async def initialize_preset_stories():
    """Load all preset stories into database on startup"""
    from story_templates.preset_story_loader import PresetStoryLoader
    
    try:
        # This will load THE_MOTH_AND_FLAME story
        await PresetStoryLoader.load_all_preset_stories()
        logger.info("Preset stories loaded successfully")
        
        # Verify they're in the database
        async with get_db_connection_context() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM PresetStories")
            logger.info(f"Total preset stories in database: {count}")
            
            # List all story IDs
            rows = await conn.fetch("SELECT story_id FROM PresetStories")
            for row in rows:
                logger.info(f"Available preset story: {row['story_id']}")
                
    except Exception as e:
        logger.error(f"Failed to load preset stories: {e}", exc_info=True)

async def initialize_systems(app: Quart):
    logger.info("Starting asynchronous system initializations...")
    # Imports for initialize_systems
    from nyx.core.brain.base import NyxBrain
    from tasks import set_app_initialized
    from logic.aggregator_sdk import init_singletons
    # Legacy StoryDirector removed; WorldDirector initializes on demand
    from db.connection import initialize_connection_pool, close_connection_pool
    from logic.nyx_enhancements_integration import initialize_nyx_memory_system
  #  from mcp_orchestrator import MCPOrchestrator

    try:
        # --- 1. Database Connection Pool ---
        logger.info("Initializing database connection pool...")
        if not await initialize_connection_pool(app=app):
            raise RuntimeError("Database pool initialization failed critically.")
        logger.info("Database connection pool initialized successfully.")

        # --- 2. Redis Connection Pools (Centralized Here) ---
        logger.info("Initializing Redis async pools...")
        try:
            redis_url = app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
            
            if redis_url:
                # Create connection pool
                shared_redis_pool = await redis_async.from_url(
                    redis_url,
                    decode_responses=True,
                    max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", 10)),
                    socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", 5)),
                    socket_connect_timeout=int(os.getenv("REDIS_CONNECT_TIMEOUT", 5))
                )
                
                # Test connection
                await shared_redis_pool.ping()
                
                # Store pools on app (you might want to rename these attributes)
                app.redis_rate_limit_pool = shared_redis_pool
                app.redis_ip_block_pool = shared_redis_pool
                
                logger.info("Redis async pools initialized successfully.")
            else:
                logger.warning("REDIS_URL not configured.")
                app.redis_rate_limit_pool = None
                app.redis_ip_block_pool = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}", exc_info=True)
            app.redis_rate_limit_pool = None
            app.redis_ip_block_pool = None

        # --- 3. Core Application Logic (Nyx, MCP, etc.) ---
        # COMMENTED OUT NYX BRAIN INITIALIZATION
        """
        logger.info("Initializing Nyx memory system...")
        await initialize_nyx_memory_system()
        logger.info("Nyx memory system initialized successfully.")

        logger.info("Initializing global NyxBrain instance...")
        try:
            system_user_id = 0
            system_conversation_id = 0
            if hasattr(NyxBrain, "get_instance"):
                app.nyx_brain = await NyxBrain.get_instance(system_user_id, system_conversation_id)
                logger.info("Global NyxBrain instance obtained/initialized.")
                if app.nyx_brain:
                    await app.nyx_brain.restore_entity_from_distributed_checkpoints()
                    from nyx.nyx_agent_sdk import process_user_input, process_user_input_with_openai
                    app.nyx_brain.response_processors = {
                        "default": background_chat_task,
                        "openai": process_user_input_with_openai,
                        "base": process_user_input
                    }
                    logger.info("Response processors registered with NyxBrain.")
            else:
                logger.warning("NyxBrain.get_instance method not available. Skipping NyxBrain initialization.")
                app.nyx_brain = None
        except ImportError as e:
             logger.error(f"Could not import or use NyxBrain components: {e}. NyxBrain might be unavailable.", exc_info=True)
             app.nyx_brain = None
        except Exception as e:
             logger.error(f"Error initializing NyxBrain: {e}", exc_info=True)
             app.nyx_brain = None
        
        if not app.nyx_brain: # Example of a critical check
            # raise RuntimeError("NyxBrain initialization failed, which is critical.")
            logger.warning("NyxBrain initialization failed. Some features might be unavailable.")
        """
        
        # Set app.nyx_brain to None since we're not initializing it
        app.nyx_brain = None
        logger.info("NyxBrain initialization skipped (commented out)")

        # COMMENTED OUT MCP ORCHESTRATOR
        #logger.info("Initializing MCP orchestrator...")
        #try:
        #    app.mcp_orchestrator = MCPOrchestrator()
        #    await app.mcp_orchestrator.initialize()
        #    logger.info("MCP orchestrator initialized.")
        #except Exception as e:
        #     logger.error(f"Error initializing MCP Orchestrator: {e}", exc_info=True)

        # --- 4. Other System Initializations ---
        logger.info("Initializing Aggregator SDK singletons...")
        await init_singletons()
        logger.info("Aggregator SDK singletons are ready.")

        # --- 5. Configuration Settings ---
        admin_ids_str = os.getenv("ADMIN_USER_IDS", "1")
        try:
            app.config['ADMIN_USER_IDS'] = [int(uid.strip()) for uid in admin_ids_str.split(',')]
        except ValueError:
            logger.error(f"Invalid ADMIN_USER_IDS format: '{admin_ids_str}'. Defaulting to [1].")
            app.config['ADMIN_USER_IDS'] = [1]
        logger.info(f"Admin User IDs configured: {app.config['ADMIN_USER_IDS']}")

        await initialize_preset_stories()
        logger.info("Preset stories are ready.")

        # --- 6. Final Readiness Signals ---
        set_app_initialized() # For Celery
        logger.info("Celery tasks application initialization status set to True.")

        app_is_ready.set() # For Quart app
        logger.info("Quart application is_ready event set. Application fully initialized and ready to serve.")

    except Exception as e:
        logger.critical(f"Fatal error during system initialization: {str(e)}", exc_info=True)
        # app_is_ready will NOT be set.
        # Ensure Celery's _APP_INITIALIZED reflects this failure if tasks.py uses it.
        # (set_app_initialized() should not have been called if we errored before it)
        raise # Re-raise to halt app startup if this function is awaited.
        
###############################################################################
# quart APP CREATION
###############################################################################

async def mark_request_processing(request_id: str, redis_pool, timeout: int = 60) -> bool:
    """
    Mark request as processing. Returns True if this is a new request,
    False if it's already being processed.
    
    Args:
        request_id: Unique identifier for the request
        redis_pool: Redis connection pool to use for deduplication
        timeout: How long to keep the processing flag (seconds)
    """
    if not redis_pool:
        # No Redis, can't deduplicate
        return True
    
    try:
        key = f"processing:{request_id}"
        # Set with NX (only if not exists) and EX (expire after timeout)
        # Returns True if key was set, None if it already existed
        result = await redis_pool.set(
            key, "1", nx=True, ex=timeout
        )
        return result is not None  # True if we set it, False if already existed
    except Exception as e:
        logger.error(f"Redis error checking request {request_id}: {e}")
        return True  # On error, allow processing to continue

async def clear_request_processing(request_id: str, redis_pool):
    """
    Clear the processing flag for a request
    
    Args:
        request_id: Unique identifier for the request
        redis_pool: Redis connection pool to use
    """
    if redis_pool:
        try:
            key = f"processing:{request_id}"
            await redis_pool.delete(key)
        except Exception as e:
            logger.error(f"Redis error clearing request {request_id}: {e}")

def create_quart_app():
    app = Quart(__name__, static_folder="static", template_folder="templates")
    QuartSchema(app)

    app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SESSION_TYPE'] = 'filesystem'
    # Optionally set session lifetime - 7 days here
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

    def parse_cors_origins(origins_str: str):
        """Parse comma-separated or JSON-like origins string into a list or wildcard."""
        if not origins_str or origins_str == "*":
            return "*"

        cleaned = origins_str.strip('" ')

        if '","' in cleaned:
            return [url.strip('" ') for url in cleaned.split('","')]

        return [url.strip() for url in cleaned.split(',')]

    cors_origins = parse_cors_origins(os.getenv("CORS_ALLOWED_ORIGINS", ""))

    if not cors_origins or cors_origins == [""]:
        cors_origins = ["http://localhost:3000", "https://nyx-m85p.onrender.com"]
        logger.warning(
            "No valid CORS origins found in environment. Using defaults: %s",
            cors_origins,
        )

    engineio_cors_allowed = "*" if cors_origins == "*" else cors_origins

    # 2) Create & attach Socket.IO _before_ any @sio.event handlers

    def _get_socket_timing(env_var: str, default: float) -> float:
        """Return a positive float from the environment or a default."""

        raw_value = os.getenv(env_var)
        if raw_value in (None, ""):
            return default

        try:
            value = float(raw_value)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            logger.warning(
                "Invalid %s value %r provided. Falling back to default %.2f seconds.",
                env_var,
                raw_value,
                default,
            )
            return default

    socket_ping_interval = _get_socket_timing("SOCKET_PING_INTERVAL", 25.0)
    socket_ping_timeout = _get_socket_timing("SOCKET_PING_TIMEOUT", 70.0)

    if socket_ping_timeout <= socket_ping_interval:
        adjusted_timeout = socket_ping_interval + 10.0
        logger.warning(
            "Socket ping timeout %.2fs must exceed interval %.2fs. Using %.2fs instead.",
            socket_ping_timeout,
            socket_ping_interval,
            adjusted_timeout,
        )
        socket_ping_timeout = adjusted_timeout

    logger.info(
        "Configuring Socket.IO heartbeat: interval=%.2fs timeout=%.2fs",
        socket_ping_interval,
        socket_ping_timeout,
    )

    socket_server_kwargs: Dict[str, Any] = {
        "async_mode": "asgi",
        "cors_allowed_origins": engineio_cors_allowed,
        "ping_timeout": socket_ping_timeout,
        "ping_interval": socket_ping_interval,
        "max_http_buffer_size": 1024 * 1024,  # Reduce slightly to 1MB
        "logger": True,
        "engineio_logger": True,
        "async_handlers": True,  # Enable async event handlers
        "always_connect": True,  # Be more permissive in connections
        "http_compression": True,  # Enable HTTP compression
    }

    sio = socketio.AsyncServer(**socket_server_kwargs)


    app.asgi_app = socketio.ASGIApp(sio, app.asgi_app)
    app.socketio = sio

    # 3) Metrics (aioprometheus)
    registry = Registry()
    http_requests = Counter(
        "http_requests_total", 
        "Total HTTP requests",
        const_labels={"service": "my‑quart‑app"}
    )
    registry.register(http_requests)

    @app.route("/metrics")
    async def metrics_endpoint():
        from aioprometheus import render
        return Response(await render(registry), mimetype="text/plain; version=0.0.4")

    # 4) CORS
    # make sure you have `pip install quart-cors`
    from quart_cors import cors

    # Configure CORS - if using specific origins, don't use wildcard
    if cors_origins == "*":
        # When using wildcard, cannot use credentials
        cors(app,
             allow_origin="*",
             allow_credentials=False,  # Must be False with wildcard
             allow_methods="*",
             allow_headers="*")
        logger.info("CORS configured with wildcard origin (credentials disabled)")
    else:
        # When using specific origins, can use credentials
        cors(app,
             allow_origin=cors_origins,
             allow_credentials=True,
             allow_methods="*",
             allow_headers="*")
        logger.info(f"CORS configured with specific origins: {cors_origins}")


    
    # Modified Socket.IO handler
    @sio.on("storybeat")
    async def on_storybeat(sid, data):
        # Generate or get request ID
        request_id = data.get('request_id')
        if not request_id:
            request_id = f"{sid}_{uuid.uuid4().hex[:8]}"
            logger.debug(f"Generated request_id: {request_id}")
        
        # Use the global redis pool from the app context
        redis_pool = app.redis_rate_limit_pool
        
        # Try to mark as processing (atomic operation)
        if not await mark_request_processing(request_id, redis_pool):
            logger.warning(f"Duplicate request {request_id} ignored for sid={sid}")
            await sio.emit('duplicate_request', {
                'request_id': request_id,
                'message': 'Request already being processed'
            }, to=sid)
            return
        
        try:
            if not app_is_ready.is_set():
                logger.warning(f"Received 'storybeat' from sid={sid} before app is fully ready. Rejecting.")
                await sio.emit('error', {'error': 'Server is initializing, please try again in a moment.'}, to=sid)
                await clear_request_processing(request_id, redis_pool) # Clear flag on early exit
                return
            
            sock_sess = await sio.get_session(sid)
            user_id = sock_sess.get("user_id", "anonymous")
            
            if user_id == "anonymous":
                # Handle unauthenticated user...
                await clear_request_processing(request_id, redis_pool)
                return
            
            user_id = int(user_id) if isinstance(user_id, str) and user_id.isdigit() else user_id
            
            conversation_id = data.get("conversation_id")
            if conversation_id is None:
                # Handle missing conversation_id...
                await clear_request_processing(request_id, redis_pool)
                return
                
            try:
                conversation_id = int(conversation_id)
            except (ValueError, TypeError):
                # Handle invalid conversation_id...
                await clear_request_processing(request_id, redis_pool)
                return
            
            user_input = data.get("user_input")
            universal_update = data.get("universal_update")
            
            if not user_input:
                # Handle missing user_input...
                await clear_request_processing(request_id, redis_pool)
                return
            
            app.logger.info(f"Received 'storybeat' from sid={sid}, user_id={user_id}, conv_id={conversation_id}, request_id={request_id}")
            
            await sio.emit("processing", {"message": "Your request is being processed...", "request_id": request_id}, to=sid)
            
            # --- THE KEY CHANGE: Dispatch the Celery task ---
            background_chat_task_with_memory.delay(
                conversation_id=conversation_id,
                user_input=user_input,
                user_id=user_id,
                universal_update=universal_update,
                request_id=request_id  # Pass the request ID to the task
            )
            
            app.logger.info(f"Dispatched Celery background_chat_task for sid={sid}, user_id={user_id}, conv_id={conversation_id}, request_id={request_id}")
        
        except Exception as e:
            app.logger.error(f"Error dispatching Celery task for sid={sid}: {e}", exc_info=True)
            await sio.emit('error', {'error': 'Server failed to dispatch message processing.'}, to=sid)
            # Clear the processing flag on error
            await clear_request_processing(request_id, redis_pool)
            
    @app.before_serving
    async def on_startup():
        global redis_listener_task
        try:
            await initialize_systems(app)
            # Start the Redis listener as a background task
            if app.redis_rate_limit_pool:
                redis_listener_task = asyncio.create_task(redis_listener(app, sio))
                logger.info("Redis pub/sub listener started.")
        except Exception as e:
            logger.critical(f"Application startup failed during initialize_systems: {e}", exc_info=True)
            raise 

    @app.after_serving
    async def shutdown_resources():
        """
        Gracefully shut down all asynchronous resources on application exit.
        This includes the Redis listener, connection pools, and database pool.
        """
        global redis_listener_task
        logger.info("Starting graceful shutdown of resources...")

        # 1. Cancel and wait for the Redis pub/sub listener task to finish
        if redis_listener_task and not redis_listener_task.done():
            logger.info("Attempting to cancel Redis pub/sub listener task...")
            try:
                redis_listener_task.cancel()
                await redis_listener_task
            except asyncio.CancelledError:
                # This is the expected and correct outcome of a successful cancellation.
                logger.info("Redis pub/sub listener task successfully cancelled.")
            except Exception as e:
                logger.error(f"An unexpected error occurred while shutting down the Redis listener task: {e}", exc_info=True)

        # 2. Close the shared Redis connection pool
        # Since app.redis_rate_limit_pool and app.redis_ip_block_pool point to the same object,
        # we only need to close it once.
        if hasattr(app, 'redis_rate_limit_pool') and app.redis_rate_limit_pool:
            logger.info("Closing shared Redis connection pool...")
            try:
                await app.redis_rate_limit_pool.close()
                logger.info("Shared Redis connection pool closed successfully.")
            except Exception as e:
                logger.error(f"Error closing the shared Redis pool: {e}", exc_info=True)
        
        # 3. Close the PostgreSQL database connection pool using your existing helper
        logger.info("Closing database connection pool...")
        try:
            await close_connection_pool(app=app)
            logger.info("Database connection pool closed successfully.")
        except Exception as e:
            logger.error(f"Error closing the database connection pool: {e}", exc_info=True)
        
        logger.info("All resources have been shut down gracefully.")


    @sio.event
    async def disconnect(sid):
        """Log disconnects with severity based on the underlying reason."""

        reason: Optional[str] = None

        try:
            sock_sess = await sio.get_session(sid)
        except KeyError:
            sock_sess = None
        except Exception as exc:  # pragma: no cover - defensive logging
            app.logger.debug(
                "SERVER-SIDE: Failed to get session for disconnected sid=%s: %s",
                sid,
                exc,
            )
            sock_sess = None

        if sock_sess:
            reason = (
                sock_sess.get("disconnect_reason")
                or sock_sess.get("reason")
                or sock_sess.get("close_reason")
            )

        expected_reasons = {
            "client namespace disconnect",
            "server namespace disconnect",
            "transport close",
        }

        log_level = logging.INFO
        if reason and reason not in expected_reasons:
            log_level = logging.WARNING

        message = f"SERVER-SIDE: Socket disconnected: sid={sid}."
        if reason:
            message = f"{message} reason={reason}"

        app.logger.log(log_level, message)
        
    @sio.event
    async def client_heartbeat(sid, data):
        """Handle client heartbeat to keep connections alive."""
        try:
            # Respond with server heartbeat
            await sio.emit('server_heartbeat', {'timestamp': time.time()}, room=sid)
        except Exception as e:
            logger.error(f"Error handling heartbeat from {sid}: {e}")
        
    @sio.event
    async def connect(sid, environ, auth):
        # Debug logging
        app.logger.info(f"Connect event - sid: {sid}")
        app.logger.info(f"Auth object: {auth}")
        app.logger.info(f"Auth type: {type(auth)}")
        
        # Get user_id from multiple possible sources
        user_id = None
        
        # Try auth object first (for newer Socket.IO versions)
        if auth and isinstance(auth, dict):
            user_id = auth.get("user_id")
            app.logger.info(f"User ID from auth dict: {user_id}")
        
        # Try query parameters as fallback
        if not user_id:
            # Parse query string
            query_string = environ.get('QUERY_STRING', '')
            app.logger.info(f"Query string: {query_string}")
            
            # Parse query parameters
            import urllib.parse
            query_params = urllib.parse.parse_qs(query_string)
            
            # Socket.IO adds its own parameters, look for user_id
            user_id_list = query_params.get('user_id', [])
            if user_id_list:
                user_id = user_id_list[0]
                app.logger.info(f"User ID from query params: {user_id}")
        
        # Convert to int if it's a valid numeric string
        if user_id and str(user_id).isdigit():
            user_id = int(user_id)
        elif not user_id or user_id == 'anonymous':
            user_id = "anonymous"
            app.logger.warning(f"No user_id found for sid={sid}, defaulting to anonymous")
        
        # Save to socketio session
        await sio.save_session(sid, {"user_id": user_id})
        app.logger.info(f"Socket connected: sid={sid}, user_id={user_id}")
        await sio.emit("response", {"data": "Connected!", "user_id": user_id}, to=sid)
    
    @sio.on("join")
    async def on_join(sid, data):
        sock_sess = await sio.get_session(sid)
        user_id = sock_sess.get("user_id", "anonymous")
        
        # Validate user is authenticated
        if user_id == "anonymous":
            await sio.emit("error", {"error": "Not authenticated"}, to=sid)
            return
        
        # Ensure proper integer conversion
        if isinstance(user_id, str) and user_id.isdigit():
            user_id = int(user_id)
        
        try:
            conversation_id = int(data["conversation_id"])
            room = str(conversation_id)  # Room names should be strings
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Invalid conversation_id in join request: {data}")
            await sio.emit("error", {"error": "Invalid conversation_id"}, to=sid)
            return

        try:
            async with get_db_connection_context() as conn:
                owner_id = await conn.fetchval(
                    "SELECT user_id FROM conversations WHERE id=$1",
                    conversation_id,
                )
        except Exception as exc:
            logger.error(
                "Failed to validate conversation %s for user %s: %s",
                conversation_id,
                user_id,
                exc,
                exc_info=True,
            )
            await sio.emit("error", {"error": "Unable to join conversation"}, to=sid)
            return

        if owner_id is None:
            await sio.emit("error", {"error": "Conversation not found"}, to=sid)
            return

        if int(owner_id) != int(user_id):
            await sio.emit("error", {"error": "Unauthorized"}, to=sid)
            return

        try:
            warm_user_context_cache_task.delay(int(user_id), int(conversation_id))
        except (TypeError, ValueError):
            logger.warning(
                "Skipping context cache warm-up for join due to non-integer identifiers: user_id=%s conversation_id=%s",
                user_id,
                conversation_id,
            )

        await sio.enter_room(sid, room)
        await sio.emit("joined", {"room": room, "user_id": user_id}, to=sid)
        app.logger.info(f"User {user_id} joined room {room}")

    @sio.on("message")
    async def on_message(sid, data):
        sock_sess = await sio.get_session(sid)
        user_id = sock_sess.get("user_id", "anonymous")
        
        # Validate authentication
        if user_id == "anonymous":
            logger.error(f"Anonymous user attempted to send message")
            await sio.emit("error", {"error": "Not authenticated"}, to=sid)
            return
        
        # Ensure user_id is int
        if isinstance(user_id, str) and user_id.isdigit():
            user_id = int(user_id)
        
        # Extract and convert conversation_id
        conversation_id = data.get("conversation_id")
        if conversation_id is not None:
            try:
                conversation_id = int(conversation_id)
            except (ValueError, TypeError):
                logger.error(f"Invalid conversation_id from client: {conversation_id}")
                await sio.emit("error", {"error": "Invalid conversation_id format"}, to=sid)
                return
        else:
            await sio.emit("error", {"error": "Missing conversation_id"}, to=sid)
            return
        
        # Get the message content
        message_content = data.get("message", "").strip()
        if not message_content:
            await sio.emit("error", {"error": "Empty message"}, to=sid)
            return
        
        # Generate request_id for deduplication
        request_id = data.get('request_id') or f"{sid}_{uuid.uuid4().hex[:8]}"
        
        # Try to mark as processing
        if not await mark_request_processing(request_id, app.redis_rate_limit_pool):
            logger.warning(f"Duplicate message request {request_id} ignored for sid={sid}")
            await sio.emit('duplicate_request', {
                'request_id': request_id,
                'message': 'Message already being processed'
            }, to=sid)
            return
        
        # Emit acknowledgment
        await sio.emit("message_received", {"status": "processing", "request_id": request_id}, to=sid)
        
        # Start background task with enhanced error handling
        try:
            sio.start_background_task(
                background_chat_task,
                conversation_id,
                message_content,
                user_id,  # Properly authenticated user_id
                None,  # universal_update
                sio,
                request_id,
                app.redis_rate_limit_pool  # Pass the redis pool
            )
            app.logger.info(f"Started background task for message from user {user_id}, conv_id={conversation_id}, request_id={request_id}")
            
        except Exception as e:
            app.logger.error(f"Error starting message processing task: {e}", exc_info=True)
            
            # Send error to client
            await sio.emit('error', {'error': 'Failed to process message'}, room=str(conversation_id))
            
            # Also send done event for consistency
            await sio.emit('done', {
                'full_text': 'I encountered an error starting to process your message. Please try again.',
                'request_id': request_id,
                'success': False,
                'error': 'Task startup failed'
            }, room=str(conversation_id))
            
            # Clear the processing flag on error
            await clear_request_processing(request_id, app.redis_rate_limit_pool)

    # 6) Security headers
    @app.after_request
    async def set_security_headers(response):
        # Define allowed CDN sources
        cdn_scripts = "https://cdn.jsdelivr.net https://cdn.socket.io https://code.jquery.com"
        cdn_styles = "https://cdn.jsdelivr.net"  # e.g. Bootstrap CSS

        # Build a CSP that allows exactly one inline <script> block:
        csp = (
            "default-src 'self'; "
            # Allow our one inline <script> (for window.CURRENT_USER_ID) plus the CDNs:
            f"script-src 'self' 'unsafe-inline' {cdn_scripts}; "
            # Allow inline styles for your CSS plus the CDN:
            f"style-src 'self' {cdn_styles} 'unsafe-inline'; "
            "img-src 'self' data: https://*; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "connect-src 'self' ws://* wss://* https://nyx-m85p.onrender.com; "
            "frame-ancestors 'none'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )

        response.headers["Content-Security-Policy"] = csp
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
    
    # --- Register Blueprints ---
    # (Ensure blueprints using async routes correctly use asyncpg)
    app.register_blueprint(new_game_bp, url_prefix='/new_game')
    app.register_blueprint(player_input_bp, url_prefix='/player_input')
    app.register_blueprint(player_input_root_bp)
    app.register_blueprint(settings_bp, url_prefix='/settings')
    app.register_blueprint(knowledge_bp, url_prefix='/knowledge')
    app.register_blueprint(story_bp, url_prefix='/story')
    app.register_blueprint(memory_bp, url_prefix='/memory')
    app.register_blueprint(rule_enforcement_bp, url_prefix='/rules')
    app.register_blueprint(admin_bp, url_prefix='/admin') # For DB seeding etc.
    app.register_blueprint(debug_bp, url_prefix='/debug')
    app.register_blueprint(universal_bp, url_prefix='/universal')
    app.register_blueprint(multiuser_bp, url_prefix='/multiuser')
    app.register_blueprint(nyx_agent_bp, url_prefix='/nyx')
    app.register_blueprint(conflict_bp, url_prefix='/conflict')
    app.register_blueprint(npc_learning_bp, url_prefix='/npc-learning')
    app.before_request(async_ip_block_middleware)
    
    register_auth_routes(app)

    init_image_routes(app) # Ensure this uses asyncpg if needed
    init_chat_routes(app) # Ensure this uses asyncpg if needed

    ###########################################################################
    # ROUTES (Defined in main app - keep minimal, prefer blueprints)
    ###########################################################################

    @app.route("/login_page", methods=["GET"])
    async def login_page():
        return await render_template("login.html") # Ensure login.html exists

    @app.route("/register_page", methods=["GET"])
    async def register_page():
        return await render_template("register.html") # Ensure register.html exists    
    # --- Authentication Routes ---
    # At the top of main.py, add this import:
    from db.connection import get_db_dsn

    @app.route("/socket-health")
    async def socket_health():
        active_count = len(app.socketio.manager.get_participants())
        return {"status": "healthy", "active_connections": active_count}
    
    @app.route("/login", methods=["POST"])
    @rate_limit(limit=5, period=60)
    @validate_input({
        'username': {'type': 'string', 'pattern': 'username', 'required': True},
        'password': {'type': 'string', 'max_length': 100, 'required': True}
    })
    async def login():
        data = getattr(request, 'sanitized_data', None)
        if data is None:
            try:
                data = await request.get_json()
            except Exception:
                return jsonify({"error": "Invalid JSON request body"}), 400
        
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400
    
        try:
            # Create connection with statement_cache_size=0 to fix pgbouncer issue
            dsn = get_db_dsn()
            conn = await asyncpg.connect(dsn, statement_cache_size=0)
            try:
                row = await conn.fetchrow(
                    "SELECT id, password_hash FROM users WHERE username=$1",
                    username
                )
                
                if not row:
                    # Timing attack mitigation
                    fake_hash = bcrypt.hashpw(b"dummy", bcrypt.gensalt())
                    bcrypt.checkpw(password.encode('utf-8'), fake_hash)
                    logger.warning(f"Login failed (no such user): {username}")
                    return jsonify({"error": "Invalid username or password"}), 401
                    
                user_id, hashed_password = row['id'], row['password_hash']
                
                try:
                    hashed_password_bytes = hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password
                    
                    if bcrypt.checkpw(password.encode('utf-8'), hashed_password_bytes):
                        session["user_id"] = user_id
                        session.permanent = True
                        logger.info(f"Login successful: User {user_id}")
                        return jsonify({"message": "Logged in", "user_id": user_id})
                    else:
                        logger.warning(f"Login failed (bad password): User {user_id}")
                        return jsonify({"error": "Invalid username or password"}), 401
                except ValueError as salt_err:
                    logger.error(f"Password hash format error for {username}: {salt_err}")
                    return jsonify({"error": "System error during authentication"}), 500
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Login error for {username}: {e}", exc_info=True)
            return jsonify({"error": "Database error during login"}), 500

    @app.before_request
    async def validate_session():
        """Ensure session data is valid"""
        if 'user_id' in session:
            user_id = session.get('user_id')
            # Clear invalid sessions
            if user_id == "anonymous" or user_id is None:
                session.clear()

    @app.route("/register", methods=["POST"])
    @rate_limit(limit=3, period=300)
    @validate_input({
        'username': {'type': 'string', 'pattern': 'username', 'required': True},
        'password': {'type': 'string', 'min_length': 8, 'max_length': 100, 'required': True},
        'email': {'type': 'string', 'pattern': 'email', 'max_length': 100, 'required': False}
    })
    async def register():
        data = getattr(request, 'sanitized_data', None)
        if data is None:
            try:
                data = await request.get_json()
            except Exception:
                return jsonify({"error": "Invalid request data"}), 400
                
        username = data.get("username")
        password = data.get("password")
        email = data.get("email")
    
        if not username or not password:
            return jsonify({"error": "Missing required fields"}), 400
    
        # Hash password
        try:
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        except Exception as hash_err:
            logger.error(f"Password hashing error: {hash_err}", exc_info=True)
            return jsonify({"error": "Registration error"}), 500
    
        try:
            # Create a fresh connection with statement_cache_size=0 to fix pgbouncer issue
            dsn = get_db_dsn()
            conn = await asyncpg.connect(dsn, statement_cache_size=0)
            try:
                # Use transaction for atomic check-and-insert
                async with conn.transaction():
                    # Check existing username
                    existing_user = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
                    if existing_user:
                        return jsonify({"error": "Username already exists"}), 409
    
                    # Check if the 'email' column exists in the users table
                    has_email_column = False
                    try:
                        # Query information_schema to check if email column exists
                        email_check = await conn.fetchval("""
                            SELECT COUNT(*) FROM information_schema.columns 
                            WHERE table_name = 'users' AND column_name = 'email'
                        """)
                        has_email_column = email_check > 0
                    except Exception as schema_err:
                        logger.warning(f"Error checking for email column: {schema_err}")
                        has_email_column = False
    
                    # Insert new user based on table structure
                    user_id = None
                    if has_email_column:
                        # If email column exists, include it in the query
                        if email:
                            # Check if email already exists
                            existing_email = await conn.fetchval("SELECT id FROM users WHERE email=$1", email)
                            if existing_email:
                                return jsonify({"error": "Email already exists"}), 409
                                
                        user_id = await conn.fetchval(
                            """INSERT INTO users (username, password_hash, email, created_at)
                               VALUES ($1, $2, $3, NOW()) RETURNING id""",
                            username, password_hash, email
                        )
                    else:
                        # If email column doesn't exist, skip it
                        user_id = await conn.fetchval(
                            """INSERT INTO users (username, password_hash, created_at)
                               VALUES ($1, $2, NOW()) RETURNING id""",
                            username, password_hash
                        )
    
                if user_id:
                    session["user_id"] = user_id
                    session.permanent = True
                    logger.info(f"Registration successful: User {user_id} ({username})")
                    return jsonify({"message": "User registered successfully", "user_id": user_id}), 201
                else:
                    logger.error(f"Registration failed: No user ID returned for {username}")
                    return jsonify({"error": "Registration failed"}), 500
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Registration unexpected error for {username}: {e}", exc_info=True)
            return jsonify({"error": "Server error during registration"}), 500


    @app.route("/logout", methods=["POST"])
    def logout():
        user_id = session.get("user_id")
        session.clear()
        if user_id:
            logger.info(f"User ID {user_id} logged out.")
        else:
            logger.info("Logout request received for non-logged-in session.")
        # Could add CSRF protection here if needed for logout POST
        return jsonify({"message": "Logged out"}), 200

    @app.route("/whoami", methods=["GET"])
    def whoami():
        user_id = session.get("user_id")
        if user_id:
            # Optionally fetch username or other non-sensitive data
            return jsonify({"logged_in": True, "user_id": user_id}), 200
        return jsonify({"logged_in": False}), 200


    # --- Core Game Routes (Example - move complex logic to blueprints) ---

    @app.route("/chat")
    async def chat_page():
        if "user_id" not in session:
            return redirect("/login_page")
        
        user_id = session.get("user_id")
        
        # Additional validation to ensure user_id is valid
        if not user_id or user_id == "anonymous":
            session.clear()  # Clear invalid session
            return redirect("/login_page")
        
        # Await the render_template call
        return await render_template("chat.html", user_id=user_id)

    # Note: /start_chat and /openai_chat POST routes were removed as the primary interaction
    # now seems to happen via SocketIO ('message' event). If you need these HTTP endpoints,
    # ensure they use asyncpg and potentially start Celery tasks instead of socketio background tasks.

    @app.route("/start_new_game", methods=["POST"])
    async def start_new_game():
        user_id = session.get("user_id")
        if not user_id: 
            return jsonify({"error": "Not authenticated"}), 401
        
        logger.info(f"User {user_id} starting new game...")
    
        try:
            # Create initial conversation record
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    conv_row = await conn.fetchrow(
                        """INSERT INTO conversations (user_id, conversation_name, status)
                        VALUES ($1, $2, 'processing') RETURNING id""",
                        user_id, "New Game - Initializing..."
                    )
                    conversation_id = conv_row['id']
                    
                    # Use the existing connection with its transaction
                    await insert_default_player_stats_chase(user_id, conversation_id, conn)
    
            # Trigger the heavy lifting asynchronously via Celery (ONLY ONCE)
            from tasks import process_new_game_task
            logger.info(f"ABOUT TO DISPATCH TASK: user_id={user_id}, conversation_id={conversation_id}")
            task_result = process_new_game_task.delay(user_id, {"conversation_id": conversation_id})
            logger.info(f"TASK DISPATCHED: task_id={task_result.id}, status={task_result.status}")

            try:
                warm_user_context_cache_task.delay(int(user_id), int(conversation_id))
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping context cache warm-up for /start_new_game due to non-integer identifiers: user_id=%s conversation_id=%s",
                    user_id,
                    conversation_id,
                )

            return jsonify({
                "job_id": task_result.id,
                "conversation_id": conversation_id,
                "task_status": task_result.status
            }), 202
    
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"New game DB error for user {user_id}: {db_err}", exc_info=True)
            return jsonify({"error": "Database error starting game"}), 500
        except Exception as e:
            logger.error(f"New game dispatch error for user {user_id}: {e}", exc_info=True)
            return jsonify({"error": "Server error starting game"}), 500

    # --- Admin/Debug Routes ---
    @app.route('/nyx_space/messages', methods=['GET'])
    async def get_nyx_space_messages():
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        user_id = session["user_id"]
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(
                "SELECT message FROM nyx_dm_messages WHERE user_id = $1 ORDER BY created_at ASC", user_id)
            messages = [row['message'] for row in rows]
        return jsonify({"messages": messages})

    @app.route('/nyx_space/messages', methods=['POST'])
    async def post_nyx_space_message():
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        user_id = session["user_id"]
        data = await request.get_json()
        msg_obj = {
            "sender": data.get("sender"),
            "content": data.get("content"),
            "timestamp": data.get("timestamp"), # or time.time()
        }
        async with get_db_connection_context() as conn:
            await conn.execute(
                "INSERT INTO nyx_dm_messages (user_id, message) VALUES ($1, $2)",
                user_id, json.dumps(msg_obj)
            )
        return jsonify({"ok": True})


    
    @app.route("/admin/nyx_direct", methods=["POST"])
    async def admin_nyx_direct():
        """
        Direct access to NyxBrain for admin users only, with full feature control.
        Accepts:
            user_input: str (required)
            context: dict (optional, will be merged with admin_mode=True)
            use_thinking: bool (optional)
            use_conditioning: bool (optional)
            use_coordination: bool (optional)
            thinking_level: int (optional)
            mode: str (optional)
            generate_response: bool (optional, default True)
        """
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
    
        # Check admin
        admin_ids = app.config.get('ADMIN_USER_IDS', [])
        if session.get("user_id") not in admin_ids:
            logger.warning(f"Non-admin user {session.get('user_id')} attempted Nyx direct access.")
            return jsonify({"error": "Access denied"}), 403
    
        data = await request.get_json(force=True, silent=True) or {}
        user_input = data.get("user_input")
        if not user_input:
            return jsonify({"error": "Missing user_input in request"}), 400
    
        # Feature toggles (accept null/None as "auto-detect")
        def to_bool(val):
            if isinstance(val, bool): return val
            if isinstance(val, str): return val.lower() in ['true', '1', 'yes', 'on']
            return None if val is None else bool(val)
        
        use_thinking = data.get("use_thinking")
        use_conditioning = data.get("use_conditioning")
        use_coordination = data.get("use_coordination")
        thinking_level = data.get("thinking_level")
        mode = data.get("mode")
        context = data.get("context") or {}
        context["admin_mode"] = True
    
        # Accept hierarchical memory (if supported)
        use_hierarchical_memory = data.get("use_hierarchical_memory", None)
    
        # Whether to run generate_response as well
        want_response = data.get("generate_response", True)
    
        nyx_brain = getattr(app, 'nyx_brain', None)
        if not nyx_brain:
            logger.error("NyxBrain not initialized on app context.")
            return jsonify({"error": "Nyx system not available"}), 503
    
        try:
            # Always run process_input with full toggle support
            processing_result = await nyx_brain.process_input(
                user_input=user_input,
                context=context,
                use_thinking=to_bool(use_thinking),
                use_conditioning=to_bool(use_conditioning),
                use_coordination=to_bool(use_coordination),
                thinking_level=int(thinking_level) if thinking_level is not None else None,
                mode=mode
            )
    
            # Optionally, run generate_response if desired
            if want_response:
                response_result = await nyx_brain.generate_response(
                    user_input=user_input,
                    context=context,
                    use_thinking=to_bool(use_thinking),
                    use_conditioning=to_bool(use_conditioning),
                    use_coordination=to_bool(use_coordination),
                    use_hierarchical_memory=to_bool(use_hierarchical_memory),
                    mode=mode
                )
            else:
                response_result = None
    
            result = {
                "processing_result": processing_result,
                "response_result": response_result,
                "admin_mode": True
            }
            return jsonify(result)
    
        except Exception as e:
            logger.error(f"Error during admin Nyx direct call: {e}", exc_info=True)
            return jsonify({"error": "Error processing direct Nyx command", "details": str(e)}), 500



    @app.route("/nyx_response", methods=["POST"])
    async def get_nyx_response():
        """Regular user access via nyx_agent_sdk"""
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not authenticated"}), 401
    
        data = await request.get_json()  # Add 'await' here
        if not data or "user_input" not in data or "conversation_id" not in data:
             return jsonify({"error": "Missing user_input or conversation_id"}), 400
    
        user_input = data.get("user_input")
        conversation_id = data.get("conversation_id")
        # TODO: Add validation to ensure user owns this conversation_id
    
        # Use the distilled agent SDK
        from nyx.nyx_agent_sdk import process_user_input
    
        try:
            logger.info(f"Processing Nyx SDK request for user={user_id}, conv={conversation_id}")
            # Call the async SDK function directly
            result = await process_user_input(
                user_id,
                conversation_id,
                user_input,
                {} # Add relevant context if needed
            )
            return jsonify(result) # Ensure result is JSON serializable
        except Exception as e:
            logger.error(f"Error processing Nyx SDK request for user={user_id}, conv={conversation_id}: {e}", exc_info=True)
            return jsonify({"error": "Error processing request via Nyx SDK", "details": str(e)}), 500


    ###########################################################################
    # HEALTH ENDPOINTS
    ###########################################################################

    @app.route("/health", methods=["GET"])
    def health_check():
        """Basic health check endpoint."""
        return jsonify({"status": "healthy", "timestamp": time.time()})

    @app.route("/readiness", methods=["GET"])
    async def readiness_check(): # Keep async
        status = {"status": "ready", "timestamp": time.time(), "checks": {}}
        is_ready = True
        # DB Check (Async)
        try:
            # Use short timeout
            async with get_db_connection_context(timeout=5) as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1: status["checks"]["database"] = "connected"
                else: status["checks"]["database"] = "error: bad query result"; is_ready = False
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            status["checks"]["database"] = f"error: {type(db_err).__name__}"; is_ready = False
            logger.warning(f"Readiness DB check failed: {db_err}")
        except Exception as e:
            status["checks"]["database"] = f"error: {type(e).__name__}"; is_ready = False
            logger.warning(f"Readiness DB check unexpected error: {e}", exc_info=True)


        # --- Redis Check (re-use existing pool) ---
        try:
            pool = getattr(app, "redis_rate_limit_pool", None)
            if pool:
                await pool.ping()
                status["checks"]["redis"] = "connected"
            else:
                status["checks"]["redis"] = "not configured"
                is_ready = False
        except Exception as e:
            status["checks"]["redis"] = f"error: {type(e).__name__}"
            is_ready = False
            logger.warning(f"Readiness Redis check failed: {e}", exc_info=True)

        # --- Celery Check (using inspect) ---
        # This can be slow and unreliable; consider a dedicated health check task
        try:
            # Assuming celery_app is accessible, maybe attach to app context?
            # Or import directly from celery_config (might cause circular import issues)
            from celery_config import celery_app as current_celery_app
            # Add a timeout to inspect to prevent hanging
            inspector = current_celery_app.control.inspect(timeout=5)
            active_workers = inspector.ping() # Ping is usually faster than active()

            if active_workers:
                status["checks"]["celery"] = f"ping ok ({len(active_workers)} responses)"
            else:
                # Try checking stats as backup
                stats = inspector.stats()
                if stats:
                     status["checks"]["celery"] = f"stats ok ({len(stats)} workers)"
                else:
                     status["checks"]["celery"] = "error: no active workers responded to ping/stats"
                     is_ready = False
                     logger.warning("Readiness check Celery error: No workers responded.")
        except Exception as e:
            status["checks"]["celery"] = f"error: {str(e)}"
            is_ready = False
            logger.warning(f"Readiness check Celery error: {e}", exc_info=True)


        # --- Final Status ---
        if not is_ready:
            status["status"] = "not ready"
            return jsonify(status), 503 # Service Unavailable

        return jsonify(status), 200

#    @app.before_serving
#    async def init_redis_pools():  # Remove the 'app' parameter
#        """Initialize Redis connection pools properly."""
#        try:
#            redis_url = app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
#            if redis_url:
#                # For Rate Limiter
#                app.aioredis_rate_limit_pool = await aioredis.from_url(
#                    redis_url, 
#                    decode_responses=True,
#                    max_connections=10
#                )
#                await app.aioredis_rate_limit_pool.ping()  # Test connection
#                logger.info("aioredis pool for Rate Limiter initialized.")
#                
#                # Use the same pool for IP blocking
#                app.aioredis_ip_block_pool = app.aioredis_rate_limit_pool
#               logger.info("aioredis pool for IP Block List configured.")
#           else:
#               logger.warning("REDIS_URL not configured. Using in-memory rate limiting.")
#                app.aioredis_rate_limit_pool = None
#                app.aioredis_ip_block_pool = None
#            
#        except Exception as e:
#            logger.error(f"Failed to initialize aioredis pools: {e}", exc_info=True)
#            app.aioredis_rate_limit_pool = None
#            app.aioredis_ip_block_pool = None
#    
#    @app.after_serving
#    async def shutdown_redis_pools():
#        """Properly close Redis connections on shutdown."""
#        logger.info("Closing Redis connection pools...")
#        if hasattr(app, 'aioredis_rate_limit_pool') and app.aioredis_rate_limit_pool:
#            await app.aioredis_rate_limit_pool.close()
#            logger.info("Redis rate limiter pool closed.")
#    
#    app.after_serving(shutdown_redis_pools)
    
    return app
