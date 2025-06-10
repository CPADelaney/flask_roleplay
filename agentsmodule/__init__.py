from dataclasses import dataclass
from typing import Optional

from .models.interface import Model

@dataclass
class Agent:
    name: str
    instructions: str
    model: Optional[Model] = None
