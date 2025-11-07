import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from nyx.location.types import Location, Place


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


def test_place_parses_in_chain():
    place = Place(name="Adventureland in Disneyland Park in Anaheim", level="venue")

    assert place.name == "Adventureland"
    assert place.meta["display_name"] == "Adventureland"
    assert place.meta["parents"] == ["Disneyland Park", "Anaheim"]


def test_place_parses_within_chain_from_meta():
    place = Place(
        name="Mystic Cavern",
        level="venue",
        meta={"display_name": "Mystic Cavern within Hidden Valley within Eldoria"},
    )

    assert place.name == "Mystic Cavern"
    assert place.meta["display_name"] == "Mystic Cavern"
    assert place.meta["parents"] == ["Hidden Valley", "Eldoria"]


def test_place_without_delimiter_leaves_metadata_unchanged():
    place = Place(name="Skyhold Spire", level="venue")

    assert place.name == "Skyhold Spire"
    assert place.meta["display_name"] == "Skyhold Spire"
    assert "parents" not in place.meta
