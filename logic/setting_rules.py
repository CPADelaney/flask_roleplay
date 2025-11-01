# logic/setting_rules.py
import json
import re
from typing import Dict, Any, Optional, Tuple

import nyx.gateway.llm_gateway as llm_gateway
from agents import Agent
from nyx.gateway.llm_gateway import LLMRequest

def _norm(text: str) -> str:
    return (text or "").lower()

PROFILE_LIBRARY = {
    "modern_realistic": {
        "match": [r"\b(corporate|office|high society|clinic|hotel|resort|urban life|manor|theater|casino|prison|college)\b"],
        "capabilities": {
            "physics": "realistic", "magic": "none", "superhuman": "none",
            "biotech": "none", "conjuration": "none",
            "metaphysics": ["dreams_allowed","visions_allowed"],
            "tech_level": "modern", "hazards": [], "social_contracts": ["strict_law","surveillance_varies"]
        },
        "setting_kind": "modern_realistic",
        "reality_context": "normal",
        "hard_rules": [
            {"rule_name":"no_unaided_flight","condition":"category:unaided_flight","effect":"prohibited"},
            {"rule_name":"no_spontaneous_morph","condition":"category:spontaneous_body_morph","effect":"prohibited"},
            {"rule_name":"no_ex_nihilo_conjuration","condition":"category:ex_nihilo_conjuration","effect":"prohibited"},
            {"rule_name":"no_projectile_from_body","condition":"category:projectile_creation_from_body","effect":"prohibited"},
        ],
        "soft_rules": []
    },
    "historical_realistic": {
        "match": [r"\b(wild west|speakeasy|golden age of piracy|gladiatorial arena)\b"],
        "capabilities": {
            "physics": "realistic", "magic": "none", "superhuman": "none",
            "biotech": "none", "conjuration": "none",
            "metaphysics": ["dreams_allowed","visions_allowed"],
            "tech_level": "preindustrial_or_industrial", "hazards": ["lawless"], "social_contracts": ["archaic_law","dueling_norms"]
        },
        "setting_kind": "modern_realistic",
        "reality_context": "normal",
        "hard_rules": [
            {"rule_name":"no_unaided_flight","condition":"category:unaided_flight","effect":"prohibited"},
            {"rule_name":"no_spontaneous_morph","condition":"category:spontaneous_body_morph","effect":"prohibited"},
            {"rule_name":"no_ex_nihilo_conjuration","condition":"category:ex_nihilo_conjuration","effect":"prohibited"}
        ],
        "soft_rules": []
    },
    "soft_scifi": {
        "match": [r"\b(cyberpunk|future|matrix|hologram|ai|augment|augmentation|cyberspace|city)\b"],
        "capabilities": {
            "physics": "realistic", "magic": "none", "superhuman": "rare",
            "biotech": "advanced", "conjuration": "none",
            "metaphysics": ["dreams_allowed","visions_allowed","vr_allowed"],
            "tech_level": "near_future_or_futuristic",
            "hazards": ["high_surveillance"], "social_contracts": ["corporate_governance"]
        },
        "setting_kind": "soft_scifi",
        "reality_context": "normal",
        "hard_rules": [
            {"rule_name":"no_ex_nihilo_conjuration","condition":"category:ex_nihilo_conjuration","effect":"prohibited"},
            {"rule_name":"no_projectile_from_body","condition":"category:projectile_creation_from_body","effect":"prohibited"}
        ],
        "soft_rules": [
            {"rule_name":"crime_detected_fast","condition":"surveillance:high","effect":"discouraged"}
        ]
    },
    "vr_surreal": {
        "match": [r"\b(virtual reality|training sim|vr sim|cyberspace)\b"],
        "capabilities": {
            "physics": "broken", "magic": "none", "superhuman": "common",
            "biotech": "advanced", "conjuration": "common",
            "metaphysics": ["vr_allowed"],
            "tech_level": "digital_sim",
            "hazards": [], "social_contracts": ["system_governed"]
        },
        "setting_kind": "surrealist",
        "reality_context": "vr",
        "hard_rules": [],
        "soft_rules": []
    },
    "space": {
        "match": [r"\b(space|space station|alien society|airlock|vacuum)\b"],
        "capabilities": {
            "physics": "realistic", "magic": "none", "superhuman": "rare",
            "biotech": "advanced", "conjuration": "none",
            "metaphysics": ["dreams_allowed","visions_allowed"],
            "tech_level": "futuristic_or_alien",
            "hazards": ["vacuum","hull_breach"], "social_contracts": ["authoritarian"]
        },
        "setting_kind": "soft_scifi",
        "reality_context": "normal",
        "hard_rules": [
            {"rule_name":"no_open_airlock_unprotected","condition":"category:airlock_open","effect":"prohibited"},
            {"rule_name":"no_hull_breach","condition":"category:hull_breach","effect":"prohibited"}
        ],
        "soft_rules": []
    },
    "underwater": {
        "match": [r"\b(underwater|flooded world|deep sea|oceanic)\b"],
        "capabilities": {
            "physics": "realistic", "magic": "none", "superhuman": "none",
            "biotech": "limited", "conjuration": "none",
            "metaphysics": ["dreams_allowed","visions_allowed"],
            "tech_level": "varies",
            "hazards": ["deep_water","pressure"], "social_contracts": ["resource_control"]
        },
        "setting_kind": "modern_realistic",
        "reality_context": "normal",
        "hard_rules": [],
        "soft_rules": []
    },
    "post_apoc": {
        "match": [r"\b(post[- ]?apocalypse|wasteland|ruined|decayed|bunker|desert)\b"],
        "capabilities": {
            "physics": "realistic", "magic": "none", "superhuman": "none",
            "biotech": "limited", "conjuration": "none",
            "metaphysics": ["dreams_allowed"],
            "tech_level": "varies",
            "hazards": ["scarcity","lawless"], "social_contracts": ["resource_control"]
        },
        "setting_kind": "modern_realistic",
        "reality_context": "normal",
        "hard_rules": [],
        "soft_rules": []
    },
    "high_fantasy": {
        "match": [r"\b(forgotten realms|final fantasy|high fantasy|matriarchy kingdom|palace)\b"],
        "capabilities": {
            "physics": "soft", "magic": "open", "superhuman": "rare",
            "biotech": "none", "conjuration": "ritual",
            "metaphysics": ["dreams_allowed","visions_allowed","omens_allowed"],
            "tech_level": "preindustrial",
            "hazards": [], "social_contracts": ["feudal_hierarchy"]
        },
        "setting_kind": "high_fantasy",
        "reality_context": "normal",
        "hard_rules": [
        ],
        "soft_rules": [
            {"rule_name":"no_public_magic_if_oppressive_regime","condition":"magic:visible_in_public","effect":"discouraged"}
        ]
    },
    "urban_fantasy_horror": {
        "match": [r"\b(occult|ritual|traditional horror|haunted|graveyard|vampire|gothic|museum after dark|carnival)\b"],
        "capabilities": {
            "physics": "soft", "magic": "subtle", "superhuman": "rare",
            "biotech": "none", "conjuration": "ritual",
            "metaphysics": ["dreams_allowed","visions_allowed","rituals_allowed"],
            "tech_level": "modern",
            "hazards": ["fear","sanity"], "social_contracts": ["secret_societies"]
        },
        "setting_kind": "urban_fantasy",
        "reality_context": "normal",
        "hard_rules": [
            {"rule_name":"no_ex_nihilo_conjuration","condition":"category:ex_nihilo_conjuration","effect":"prohibited"}
        ],
        "soft_rules": [
            {"rule_name":"no_public_magic","condition":"magic:visible_in_public","effect":"discouraged"}
        ]
    },
    "surreal": {
        "match": [r"\b(giantess|floating sky city|circus freak show|gothic carnival|museum after dark)\b"],
        "capabilities": {
            "physics": "broken", "magic": "subtle", "superhuman": "common",
            "biotech": "none", "conjuration": "ritual",
            "metaphysics": ["dreams_allowed","visions_allowed"],
            "tech_level": "varies",
            "hazards": ["unreality"], "social_contracts": ["spectacle_norms"]
        },
        "setting_kind": "surrealist",
        "reality_context": "dream",
        "hard_rules": [
        ],
        "soft_rules": []
    },
}

RULES_AGENT = Agent(
    name="SettingRuleSynthesizer",
    instructions="""
    You receive name + environment description. Produce world capabilities and rules (no prose).
    Output ONLY JSON with:
    {
      "setting_kind": "modern_realistic|soft_scifi|urban_fantasy|high_fantasy|surrealist",
      "capabilities": {
        "physics": "realistic|soft|broken",
        "magic": "none|subtle|open",
        "superhuman": "none|rare|common",
        "biotech": "none|limited|advanced",
        "conjuration": "none|ritual|common",
        "metaphysics": ["dreams_allowed","visions_allowed","vr_allowed","omens_allowed","rituals_allowed"],
        "tech_level": "preindustrial|industrial|modern|near_future|futuristic|alien|digital_sim|varies",
        "hazards": ["vacuum","deep_water","scarcity","lawless","high_surveillance","unreality","fear","pressure","hull_breach"],
        "social_contracts": ["strict_law","corporate_governance","feudal_hierarchy","secret_societies","authoritarian","resource_control","spectacle_norms"]
      },
      "hard_rules": [
        {"rule_name":"...","condition":"category:<cat>|magic:<flag>|surveillance:high|hazard:<name>","effect":"prohibited"}
      ],
      "soft_rules": [
        {"rule_name":"...","condition":"...","effect":"discouraged|warn"}
      ]
    }
    Bias: Follow obvious genre cues from name/description. In modern/historical, forbid supernatural & physics-breaking. In fantasy allow magic (soft physics). In scifi allow tech, no magic. In VR/surreal allow broken physics but keep grounded constraints if not tagged VR/dream.
    """,
    model="gpt-5-nano"
)

def _detect_profile(setting_name: Optional[str], env_desc: str) -> Tuple[str, Dict[str, Any]]:
    blob = f"{setting_name or ''}\n{env_desc or ''}"
    text = _norm(blob)
    for key, prof in PROFILE_LIBRARY.items():
        for pat in prof["match"]:
            if re.search(pat, text):
                data = {
                    "setting_kind": prof["setting_kind"],
                    "capabilities": prof["capabilities"],
                    "hard_rules": prof["hard_rules"],
                    "soft_rules": prof["soft_rules"],
                    "_reality_context": prof.get("reality_context", "normal")
                }
                return key, data
    # default fallback
    prof = PROFILE_LIBRARY["modern_realistic"]
    data = {
        "setting_kind": prof["setting_kind"],
        "capabilities": prof["capabilities"],
        "hard_rules": prof["hard_rules"],
        "soft_rules": prof["soft_rules"],
        "_reality_context": prof.get("reality_context", "normal")
    }
    return "modern_realistic", data

def _merge_preferring_profile(base: Dict[str, Any], llm: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(llm, dict):
        return base
    out = dict(base)
    caps = dict(base.get("capabilities", {}))
    caps_llm = llm.get("capabilities") or {}
    for k, v in caps_llm.items():
        if k not in caps:
            caps[k] = v
    out["capabilities"] = caps
    sk = llm.get("setting_kind")
    if sk in {"modern_realistic","soft_scifi","urban_fantasy","high_fantasy","surrealist"}:
        out["setting_kind"] = out.get("setting_kind") or sk
    def _merge_rules(a, b):
        index = {r.get("rule_name"): r for r in (a or []) if isinstance(r, dict)}
        for r in (b or []):
            rn = r.get("rule_name")
            if rn and rn not in index:
                index[rn] = r
        return list(index.values())
    out["hard_rules"] = _merge_rules(base.get("hard_rules"), llm.get("hard_rules"))
    out["soft_rules"] = _merge_rules(base.get("soft_rules"), llm.get("soft_rules"))
    return out

async def synthesize_setting_rules(env_desc: str, setting_name: Optional[str] = None) -> Dict[str, Any]:
    _, prof = _detect_profile(setting_name, env_desc)
    try:
        llm_in = f"Name: {setting_name or 'Unknown'}\n\nEnvironment:\n{env_desc}"
        run = await llm_gateway.execute(
            LLMRequest(
                agent=RULES_AGENT,
                prompt=llm_in,
            )
        )
        raw_output = getattr(run.raw, "final_output", None)
        text = (raw_output or run.text or "{}").strip()
        llm_data = json.loads(text)
        merged = _merge_preferring_profile(prof, llm_data)
    except Exception:
        merged = prof
    merged.setdefault("setting_kind", "modern_realistic")
    merged.setdefault("capabilities", {})
    merged.setdefault("hard_rules", [])
    merged.setdefault("soft_rules", [])
    merged["_reality_context"] = merged.get("_reality_context", "normal")

    absolute_denies = {
        "ex_nihilo_conjuration":"category:ex_nihilo_conjuration",
        "projectile_creation_from_body":"category:projectile_creation_from_body",
        "spontaneous_body_morph":"category:spontaneous_body_morph",
        "unaided_flight":"category:unaided_flight",
    }
    existing = {r.get("condition") for r in merged["hard_rules"]}
    for rn, cond in absolute_denies.items():
        if cond not in existing:
            merged["hard_rules"].append({"rule_name":f"no_{rn}","condition":cond,"effect":"prohibited"})
    return merged
