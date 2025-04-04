# nyx/tools/claim_validation.py
"""
Nyx's social media claim validation filter:
- Uses CUA to research the claim
- Filters out known bullshit sources
- Escalates to Chase if ambiguous
"""

import datetime
import logging

logger = logging.getLogger(__name__)

BLACKLISTED_SOURCES = [
    "foxnews", "breitbart", "infowars", "naturalnews",
    "truthsocial", "bitchute", "rumble", "gatewaypundit",
    "zerohedge", "sputnik", "theblaze"
]

def is_blacklisted_source(text: str) -> bool:
    return any(source in text.lower() for source in BLACKLISTED_SOURCES)

async def validate_social_claim(self, text: str, source: str) -> dict:
    """
    Validates a claim by searching for supporting evidence using CUA.
    Filters out bad sources and escalates uncertain cases.
    """
    try:
        response = await self.creative_system.computer_user.run_task(
            url="https://www.google.com",
            prompt=f"Fact check this claim: '{text}'. Source: {source}. "
                   "Scan the first page of results. Are they from credible sources? Return 'true', 'false', or 'unverified' and explain why.",
            width=1024,
            height=768
        )

        explanation = response.strip()
        lower = explanation.lower()

        if is_blacklisted_source(lower):
            return {
                "verdict": "false",
                "explanation": f"Sources include unreliable domains (e.g. {', '.join([s for s in BLACKLISTED_SOURCES if s in lower])})."
            }

        if "true" in lower and "false" not in lower:
            return {"verdict": "true", "explanation": explanation}
        elif "false" in lower and "true" not in lower:
            return {"verdict": "false", "explanation": explanation}
        else:
            # Escalate to Chase
            await self.creative_system.logger.log_thought(
                title="🧠 Escalation: Unverifiable Claim Needs Review",
                content=explanation,
                metadata={
                    "original_claim": text,
                    "source": source,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action_required": "user_review"
                }
            )
            return {
                "verdict": "unverified",
                "explanation": "Claim could not be reliably validated. Escalated to user."
            }

    except Exception as e:
        logger.error(f"Error during claim validation: {e}")
        return {
            "verdict": "unverified",
            "explanation": "Validation failed due to system error."
        }
