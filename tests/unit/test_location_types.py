import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from nyx.location.types import Location


def test_location_from_record_accepts_location_id_alias():
    record = {
        "location_id": 777,
        "user_id": 1,
        "conversation_id": 2,
        "location_name": "Skyhold Spire",
    }

    location = Location.from_record(record)

    assert location.id == 777
    assert location.location_id == 777


def test_location_from_record_prefers_override_id():
    record = {
        "location_id": 123,
        "user_id": 5,
        "conversation_id": 9,
        "location_name": "Twilight Market",
    }

    location = Location.from_record(record, id=456)

    assert location.id == 456
    assert location.location_id == 456
