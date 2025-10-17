from dataclasses import dataclass
from typing import Optional

from .models.interface import Model

ROLEPLAY_DIRECTIVE = (
    "Stay completely in-world during roleplay. Do not provide real-world "
    "suggestions, resources, or assistance (such as store hours, coupons, "
    "mobile apps, or phone numbers). Redirect or refuse any out-of-game "
    "requests and keep every response diegetic to the fiction."
)


@dataclass
class Agent:
    name: str
    instructions: str
    model: Optional[Model] = None

    def __post_init__(self) -> None:
        base_text = (self.instructions or "").rstrip()
        if ROLEPLAY_DIRECTIVE not in base_text:
            if base_text:
                base_text = f"{base_text}\n\n{ROLEPLAY_DIRECTIVE}"
            else:
                base_text = ROLEPLAY_DIRECTIVE
        self.instructions = base_text + "\n"
