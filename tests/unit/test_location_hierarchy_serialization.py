import asyncio
import json
import sys
from pathlib import Path

import asyncpg

sys.path.append(str(Path(__file__).resolve().parents[2]))

from nyx.location.hierarchy import get_or_create_location
from nyx.location.types import Candidate, Place


class _FakeConnection:
    """Capture SQL parameters for Places/Locations inserts."""

    def __init__(self):
        self._place_id = 0
        self._location_id = 100
        self.admin_path_payloads = []
        self.place_meta_payloads = []
        self.edge_meta_payloads = []
        self.place_key_payloads = []

    async def fetchrow(self, query, *args):
        normalized = " ".join(query.split())

        if normalized.startswith("INSERT INTO Places"):
            admin_path_arg = args[5]
            self.admin_path_payloads.append(admin_path_arg)

            if not isinstance(admin_path_arg, str):
                raise AssertionError("admin_path should be serialized as a JSON string")

            meta_arg = args[8]
            self.place_meta_payloads.append(meta_arg)

            if not isinstance(meta_arg, str):
                raise asyncpg.DataError("meta should be serialized as a JSON string")

            place_key_arg = args[1]
            self.place_key_payloads.append(place_key_arg)

            self._place_id += 1
            return {"id": self._place_id, "place_key": args[1]}

        if normalized.startswith("SELECT * FROM Locations"):
            return None

        if normalized.startswith("SELECT city, region, country"):
            return None

        if normalized.startswith("INSERT INTO Locations"):
            self._location_id += 1
            (
                user_id,
                conversation_id,
                location_name,
                description,
                location_type,
                parent_location,
                room,
                building,
                district,
                district_type,
                city,
                region,
                country,
                planet,
                galaxy,
                realm,
                lat,
                lon,
                is_fictional,
                open_hours,
                controlling_faction,
                cultural_significance,
                economic_importance,
                strategic_value,
                population_density,
                notable_features,
                hidden_aspects,
                access_restrictions,
                local_customs,
                embedding,
            ) = args

            return {
                "id": self._location_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "location_name": location_name,
                "description": description,
                "location_type": location_type,
                "parent_location": parent_location,
                "room": room,
                "building": building,
                "district": district,
                "district_type": district_type,
                "city": city,
                "region": region,
                "country": country,
                "planet": planet,
                "galaxy": galaxy,
                "realm": realm,
                "lat": lat,
                "lon": lon,
                "is_fictional": is_fictional,
                "open_hours": open_hours,
                "controlling_faction": controlling_faction,
                "cultural_significance": cultural_significance,
                "economic_importance": economic_importance,
                "strategic_value": strategic_value,
                "population_density": population_density,
                "notable_features": notable_features,
                "hidden_aspects": hidden_aspects,
                "access_restrictions": access_restrictions,
                "local_customs": local_customs,
                "embedding": embedding,
            }

        raise AssertionError(f"Unexpected query: {query}")

    async def execute(self, query, *args):
        normalized = " ".join(query.split())

        if normalized.startswith("INSERT INTO PlaceEdges"):
            meta_arg = args[4]
            self.edge_meta_payloads.append(meta_arg)

            if not isinstance(meta_arg, str):
                raise asyncpg.DataError("edge meta should be serialized as a JSON string")

        return None


def test_get_or_create_location_serializes_admin_path():
    conn = _FakeConnection()

    candidate = Candidate(
        place=Place(
            name="Crystal Garden",
            level="venue",
            address={
                "_normalized_admin_path": {
                    "country": "Wonderland",
                    "city": "Heart City",
                }
            },
            meta={"world_name": "Fictionland"},
        ),
        confidence=0.9,
    )

    location = asyncio.run(
        get_or_create_location(
            conn,
            user_id=42,
            conversation_id=13,
            candidate=candidate,
            scope="fictional",
        )
    )

    assert location.location_name == "crystal garden"

    # All admin_path payloads should be serialized JSON strings that round-trip cleanly.
    assert conn.admin_path_payloads, "expected at least one place insert"
    parsed_paths = [json.loads(payload) for payload in conn.admin_path_payloads]

    final_path = parsed_paths[-1]
    assert final_path["world"] == "Fictionland"
    assert final_path["country"] == "Wonderland"
    assert final_path["city"] == "Heart City"
    assert final_path["venue"] == "Crystal Garden"


def test_get_or_create_location_serializes_place_and_edge_meta():
    conn = _FakeConnection()

    candidate = Candidate(
        place=Place(
            name="Arcadia Spire",
            level="venue",
            key="ext-arcadia-spire",
            address={
                "_normalized_admin_path": {
                    "country": "Arcadia",
                    "city": "Skyhold",
                }
            },
            meta={
                "world_name": "Clockwork Realm",
                "classification": {"tier": "prime"},
            },
        ),
        confidence=0.95,
    )

    location = asyncio.run(
        get_or_create_location(
            conn,
            user_id=100,
            conversation_id=200,
            candidate=candidate,
            scope="fictional",
        )
    )

    assert location.location_name == "arcadia spire"

    assert conn.place_meta_payloads, "expected place metadata writes"
    parsed_place_meta = [json.loads(payload) for payload in conn.place_meta_payloads]
    final_meta = parsed_place_meta[-1]
    assert final_meta["world_name"] == "Clockwork Realm"
    assert final_meta["classification"] == {"tier": "prime"}
    assert final_meta["place_key"] == "ext-arcadia-spire"

    assert conn.edge_meta_payloads, "expected edge metadata writes"
    parsed_edge_meta = [json.loads(payload) for payload in conn.edge_meta_payloads]
    assert all(meta_dict.get("scope") == "fictional" for meta_dict in parsed_edge_meta)

    assert conn.place_key_payloads[-1] == "ext-arcadia-spire"
    assert candidate.place.meta["place_key"] == "ext-arcadia-spire"
    assert candidate.place.key == "ext-arcadia-spire"
