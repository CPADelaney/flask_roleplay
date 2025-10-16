# nyx/location/config.py
from dataclasses import dataclass

MILES_TO_KM = 1.609344

@dataclass
class LocationSettings:
    # Toggle-y helpers (keep strict in prod)
    allow_alias_fallbacks: bool = False
    allow_brand_fixups: bool = False

    # Core radii
    search_radius_km: float = 5.0 * MILES_TO_KM      # ~8.05 km
    widen_radius_km: float  = 7.5 * MILES_TO_KM      # ~12.07 km

    # Overpass controls
    overpass_timeout_s: int = 25
    overpass_limit: int = 24

    # Nominatim controls
    nominatim_limit: int = 12

# Global default; import and use where needed
DEFAULT_LOCATION_SETTINGS = LocationSettings()
