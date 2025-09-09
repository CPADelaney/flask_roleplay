# nyx/nyx_agent/feasibility.py
import json
from typing import Dict, Any
from agents import Agent, Runner
from nyx.nyx_agent.context import NyxContext
from db.connection import get_db_connection_context
from logic.action_parser import parse_action_intents

FEASIBILITY_AGENT = Agent(
    name="FeasibilityChecker",
    instructions="""
    Decide feasibility of multiple action intents in the current setting.
    Input: {"intents":[...],"capabilities":{...},"hard_rules":[...],"soft_rules":[...],
            "setting_kind":"...","reality_context":"normal|dream|vr"}
    Rules:
    - If any intent violates a hard rule, mark it infeasible with strategy="deny".
    - If categories imply impossibility vs capabilities (e.g., magic in modern_realistic), mark "deny".
    - strategy "reinterpret" allowed for near-misses (e.g., use tools/gear instead).
    - strategy "dream" allowed ONLY if reality_context in ("dream","vr") or setting_kind=="surrealist".
    Output ONLY JSON:
      {"overall":{"feasible":bool,"strategy":"allow|deny|reinterpret|dream"},
       "per_intent":[
         {"feasible":bool,"strategy":"allow|deny|reinterpret|dream",
          "violations":[{"rule":"...", "reason":"..."}],
          "narrator_guidance":"...", "suggested_alternatives":["..."],
          "categories":["..."]}
       ]}
    """,
    model="gpt-5-nano"
)

async def _load_rules_and_caps(nyx_ctx: NyxContext):
    capabilities, hard_rules, soft_rules, setting_kind, reality_context = {}, [], [], "modern_realistic", "normal"
    async with get_db_connection_context() as conn:
        cap = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='SettingCapabilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        if cap and cap['value']:
            capabilities = json.loads(cap['value'])
        sk = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='SettingKind'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        if sk and sk["value"]:
            setting_kind = sk["value"]
        rc = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='RealityContext'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        if rc and rc["value"]:
            reality_context = rc["value"]
        rows = await conn.fetch("""SELECT rule_name, condition, effect FROM GameRules""")
        for r in rows:
            item = {"rule_name": r["rule_name"], "condition": r["condition"], "effect": r["effect"]}
            if (r["effect"] or "").lower().startswith("prohibit"):
                hard_rules.append(item)
            else:
                soft_rules.append(item)
    return capabilities, hard_rules, soft_rules, setting_kind, reality_context

async def assess_action_feasibility(nyx_ctx: NyxContext, user_input: str) -> Dict[str, Any]:
    intents = await parse_action_intents(user_input)
    caps, hard_rules, soft_rules, setting_kind, reality_context = await _load_rules_and_caps(nyx_ctx)
    payload = {
        "intents": intents,
        "capabilities": caps,
        "hard_rules": hard_rules,
        "soft_rules": soft_rules,
        "setting_kind": setting_kind,
        "reality_context": reality_context
    }
    run = await Runner.run(FEASIBILITY_AGENT, json.dumps(payload))
    try:
        raw = json.loads(getattr(run, "final_output", "{}"))
        if "overall" not in raw:
            raw["overall"] = {}
        raw["overall"]["feasible"] = all(
            it.get("feasible") or it.get("strategy") in ("reinterpret","dream")
            for it in raw.get("per_intent", [])
        )
        return raw
    except Exception:
        mundane = all("mundane_action" in (i.get("categories") or []) for i in intents)
        return {"overall":{"feasible":mundane,"strategy":"allow" if mundane else "deny"},
                "per_intent":[{"feasible":mundane}]}
