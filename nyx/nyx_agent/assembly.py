# nyx/nyx_agent/assembly.py

"""Response assembly functions for Nyx Agent SDK"""

import json
import logging
from typing import Dict, List, Any, Optional

from .utils import _tool_output, _extract_last_assistant_text

logger = logging.getLogger(__name__)

def assemble_nyx_response(resp: list) -> Dict[str, Any]:
    """Assemble final response from agent run with full narrator integration.

    Prefers the concrete narrator output (tool_narrate_slice_of_life_scene) if available.
    Falls back to orchestrate_slice_scene packaging or other tools/dialogue.
    """
    # --- Pull tool outputs ----------------------------------------------------
    # Primary scene sources
    narrator_scene = _tool_output(resp, "tool_narrate_slice_of_life_scene")
    packaged_scene = _tool_output(resp, "orchestrate_slice_scene")

    # Other contributors
    dialogue = _tool_output(resp, "generate_npc_dialogue")
    world = _tool_output(resp, "check_world_state")
    npc = _tool_output(resp, "simulate_npc_autonomy")
    img = _tool_output(resp, "decide_image_generation")
    updates = _tool_output(resp, "generate_universal_updates")
    emergent = _tool_output(resp, "generate_emergent_event")
    power = _tool_output(resp, "narrate_power_exchange")
    routine = _tool_output(resp, "narrate_daily_routine")
    ambient = _tool_output(resp, "generate_ambient_narration")

    # Prefer the actual narrator-produced scene
    scene = None
    if isinstance(narrator_scene, dict) and not narrator_scene.get("narrator_request"):
        scene = narrator_scene
    elif isinstance(packaged_scene, dict):
        scene = packaged_scene

    # --- 1) Narrative text ----------------------------------------------------
    narrative = ""
    atmosphere = ""
    nyx_commentary = None
    governance_approved = True
    context_aware = False

    if scene and isinstance(scene, dict):
        narrative = scene.get("narrative") or scene.get("scene_description") or ""
        atmosphere = scene.get("atmosphere", "") or ""
        nyx_commentary = scene.get("nyx_commentary")
        governance_approved = scene.get("governance_approved", True)
        context_aware = scene.get("context_aware", False)

        # If still nothing, try inner scene structure
        if not narrative and "scene" in scene and isinstance(scene["scene"], dict):
            narrative = scene["scene"].get("description", "") or narrative

    # Dialogue fallback
    if (not narrative or not narrative.strip()) and isinstance(dialogue, dict):
        npc_dialogue = dialogue.get("dialogue", "")
        npc_name = dialogue.get("npc_name", "Someone")
        if npc_dialogue:
            narrative = f'{npc_name}: "{npc_dialogue}"'
            if dialogue.get("subtext"):
                narrative += f"\n*{dialogue['subtext']}*"

    # Power exchange fallback
    if (not narrative or not narrative.strip()) and isinstance(power, dict):
        narrative = power.get("narrative") or power.get("moment", "") or narrative

    # Routine fallback
    if (not narrative or not narrative.strip()) and isinstance(routine, dict):
        narrative = routine.get("description") or routine.get("routine_with_dynamics", "") or narrative

    # Final assistant-text fallback
    if not narrative or not narrative.strip():
        narrative = _extract_last_assistant_text(resp) or "The moment unfolds in the simulation."

    # --- 2) World metadata ----------------------------------------------------
    world_mood = "neutral"
    time_of_day = "Unknown"
    tension_level = 3

    if scene and isinstance(scene, dict):
        world_mood = scene.get("world_mood") or world_mood
        time_of_day = scene.get("time_of_day") or time_of_day
        tension_level = scene.get("tension_level", tension_level)

    if isinstance(world, dict):
        world_mood = world.get("world_mood") or world_mood
        time_of_day = world.get("time_of_day") or time_of_day
        tensions = world.get("tensions")
        if isinstance(tensions, dict) and tensions:
            try:
                avg_tension = sum(float(v) for v in tensions.values()) / max(1, len(tensions))
                tension_level = int(round(avg_tension * 10))
            except Exception:
                pass

    # --- 3) Choices / available activities -----------------------------------
    available: list[str] = []

    if scene and isinstance(scene, dict):
        available = list(scene.get("available_activities") or [])
        if not available:
            # some narrators return 'choices'
            ch = scene.get("choices") or []
            if isinstance(ch, list):
                # Normalize choice texts if they are dicts
                normalized = []
                for c in ch:
                    if isinstance(c, dict):
                        txt = c.get("text")
                        if isinstance(txt, str) and txt.strip():
                            normalized.append(txt)
                    elif isinstance(c, str):
                        normalized.append(c)
                available = normalized

    if not available and isinstance(power, dict):
        opts = power.get("player_response_options")
        if isinstance(opts, list):
            available = [str(x) for x in opts if isinstance(x, (str, int))]

    if not available:
        available = ["observe", "browse", "leave", "interact"]

    # --- 4) Emergent opportunities -------------------------------------------
    emergent_opportunities: list[str] = []
    if scene and isinstance(scene, dict):
        eo = scene.get("emergent_opportunities") or []
        if isinstance(eo, list):
            emergent_opportunities = [str(x) for x in eo]

    if isinstance(emergent, dict):
        evt = emergent.get("event")
        if isinstance(evt, dict):
            emergent_opportunities.append(evt.get("title", "Emergent event"))

    # --- 5) Power undertones --------------------------------------------------
    power_undertones: list[str] = []
    if scene and isinstance(scene, dict):
        power_undertones = scene.get("power_undertones") or scene.get("power_dynamic_hints") or []

    if isinstance(power, dict) and power.get("exchange_type"):
        power_undertones.append(f"Active {power['exchange_type']}")

    if not power_undertones:
        power_undertones = ["subtle control dynamics"]

    # --- 6) Tension type safety ----------------------------------------------
    if not isinstance(tension_level, (int, float)):
        try:
            tension_level = int(tension_level)
        except Exception:
            tension_level = 3

    # --- 7) Image decision ----------------------------------------------------
    should_image = False
    image_prompt = None

    if isinstance(img, dict):
        should_image = bool(img.get("should_generate") or img.get("generate_image", False))
        image_prompt = img.get("image_prompt")

    if not should_image and scene and isinstance(scene, dict):
        should_image = bool(scene.get("generate_image", False))
        if not image_prompt:
            image_prompt = scene.get("image_prompt") or scene.get("image_description")

    # --- 8) NPC-specific data -------------------------------------------------
    npc_data: Dict[str, Any] = {}

    if isinstance(dialogue, dict):
        npc_data["last_dialogue"] = {
            "npc_id": dialogue.get("npc_id"),
            "text": dialogue.get("dialogue"),
            "tone": dialogue.get("tone"),
            "requires_response": dialogue.get("requires_response", False),
        }

    if isinstance(npc, dict):
        npc_data["autonomy"] = {
            "actions": npc.get("npc_actions", []),
            "observation": npc.get("nyx_observation", ""),
        }

    # --- 9) Ambient narration merge (optional) --------------------------------
    if isinstance(ambient, dict):
        amb_text = ambient.get("atmosphere") or ambient.get("ambient") or ambient.get("description")
        if amb_text and isinstance(amb_text, str):
            atmosphere = (atmosphere + "\n" + amb_text).strip() if atmosphere else amb_text

    # --- 10) Build final response --------------------------------------------
    response: Dict[str, Any] = {
        "narrative": narrative,
        "world": {
            "mood": world_mood,
            "time_of_day": time_of_day,
            "tension_level": tension_level,
            "atmosphere": atmosphere,
        },
        "choices": available,
        "emergent": emergent_opportunities,
        "undertones": power_undertones,
        "image": {
            "should_generate": should_image,
            "prompt": image_prompt,
        },
        "universal_updates": bool(updates.get("updates_generated")) if isinstance(updates, dict) else False,
        "nyx_commentary": nyx_commentary,
        "governance": {
            "approved": bool(governance_approved),
            "context_aware": bool(context_aware),
        },
        "npc": npc_data if npc_data else None,
        "telemetry": {
            "tools_called": sorted({
                c.get("name") for c in resp
                if isinstance(c, dict) and c.get("type") == "function_call"
            }),
            "tools_with_output": sorted({
                c.get("name") for c in resp
                if isinstance(c, dict) and c.get("type") == "function_call_output"
            }),
            "narrator_used": bool(scene and context_aware),
        },
    }

    # --- 11) Conflicts / intersections / memory context ----------------------
    if scene and isinstance(scene, dict):
        conflict_manifestations = scene.get("conflict_manifestations", [])
        if conflict_manifestations:
            response["conflicts"] = {
                "active": True,
                "manifestations": conflict_manifestations,
            }

        system_triggers = scene.get("system_triggers", [])
        if system_triggers:
            response["system_intersections"] = system_triggers

        relevant_memories = scene.get("relevant_memories", [])
        if relevant_memories:
            response["memory_context"] = relevant_memories[:3]

    return response

async def resolve_scene_requests(resp: list, app_ctx) -> list:
    """
    Look for orchestrate_slice_scene outputs that contain a narrator_request,
    call the narrator tool directly (no LLM), and append the result into resp.
    """
    try:
        scene_out = _tool_output(resp, "orchestrate_slice_scene")
    except Exception:
        scene_out = None

    if not (scene_out and isinstance(scene_out, dict)):
        return resp

    req = scene_out.get("narrator_request")
    if not (req and isinstance(req, dict) and req.get("tool") == "tool_narrate_slice_of_life_scene"):
        return resp

    payload = req.get("payload") or {}
    # Ensure narrator exists
    narrator = getattr(app_ctx, "slice_of_life_narrator", None)
    if not narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(app_ctx.user_id, app_ctx.conversation_id)
        await app_ctx.slice_of_life_narrator.initialize()
        narrator = app_ctx.slice_of_life_narrator

    # Call the tool function *directly* (no Runner.run here)
    try:
        from story_agent.slice_of_life_narrator import narrate_slice_of_life_scene as tool_narrate_slice_of_life_scene
        from .utils import _json_safe, _default_json_encoder
        
        tool_raw = await tool_narrate_slice_of_life_scene(
            narrator.context,
            payload=_json_safe(payload),
        )
        tool_output = tool_raw if isinstance(tool_raw, str) else json.dumps(
            tool_raw, ensure_ascii=False, default=_default_json_encoder
        )
        resp = list(resp)  # avoid mutating caller's list in place
        resp.append({
            "type": "function_call_output",
            "name": "tool_narrate_slice_of_life_scene",
            "output": tool_output,
        })
        return resp
    except Exception as e:
        resp = list(resp)
        resp.append({
            "type": "function_call_output",
            "name": "tool_narrate_slice_of_life_scene",
            "output": json.dumps({"error": "narration_failed", "detail": str(e)}, ensure_ascii=False),
        })
        return resp
