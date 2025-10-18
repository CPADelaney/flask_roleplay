import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from nyx.location.anchors import derive_anchor_from_hierarchy
from nyx.location.types import Location


def test_derive_anchor_from_hierarchy_real_location_full():
    location = Location(
        user_id=1,
        conversation_id=2,
        location_name="Prime Spire",
        district="Downtown",
        city="Metropolis",
        region="Central Province",
        country="Freedonia",
    )

    assert derive_anchor_from_hierarchy(location) == "Downtown, Metropolis, Central Province, Freedonia"


def test_derive_anchor_from_hierarchy_partial_real_location():
    location = Location(
        user_id=3,
        conversation_id=4,
        location_name="Seaside Harbor",
        city="Oceanview",
        country="Freedonia",
    )

    assert derive_anchor_from_hierarchy(location) == "Oceanview, Freedonia"


def test_derive_anchor_from_hierarchy_fictional_location():
    location = Location(
        user_id=5,
        conversation_id=6,
        location_name="Fictional Realm",
        city="Dreamvale",
        country="Fantasia",
        is_fictional=True,
    )

    assert derive_anchor_from_hierarchy(location) is None


def test_derive_anchor_from_hierarchy_non_location():
    class NotALocation:
        is_fictional = False
        city = "Nowhere"
        country = "Freedonia"

    assert derive_anchor_from_hierarchy(NotALocation()) is None
