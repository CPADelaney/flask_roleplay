# nyx/core/types/kv.py
from pydantic import BaseModel
from typing import Union

JsonScalar = Union[str, float, int, bool, None]

class KVPair(BaseModel):
    """One explicit key/value item (avoids open Dict fields)."""
    key: str
    value: JsonScalar
