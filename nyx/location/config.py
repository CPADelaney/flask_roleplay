# nyx/location/config.py
from dataclasses import dataclass

@dataclass
class LocationSettings:
    allow_alias_fallbacks: bool = False   # keep false in prod
    allow_brand_fixups: bool = False      # keep false in prod
    nominatim_radius_km: float = 3.0
    widen_radius_km: float = 12.0
