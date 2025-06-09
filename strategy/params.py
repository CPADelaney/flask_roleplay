from pydantic import BaseModel, Field

class StrategyParams(BaseModel):
    """Parameters influencing agent strategy."""

    exploration_rate: float = 0.2
    precision_focus: float = 0.5
    risk_tolerance: float = 0.4
    creativity: float = 0.6
    tool_biases: dict[str, float] = Field(default_factory=dict)
