import random
from collections import deque
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from nyx.location import hierarchy as location_hierarchy
from nyx.location.anchors import GeoAnchor
from nyx.location.types import Anchor, Candidate, Place, DEFAULT_REALM


class StubConnection:
    def __init__(self, *, city_rows=None, fallback_rows=None):
        self.city_rows = deque(city_rows or [])
        self.fallback_rows = deque(fallback_rows or [])
        self.city_query_args = []
        self.fallback_query_args = []
        self.inserted_rows = []
        self.next_location_id = 1

    async def fetchrow(self, query, *args):
        normalized = " ".join(query.split())
        if "LOWER(city)" in normalized:
            self.city_query_args.append(args)
            if self.city_rows:
                return self.city_rows.popleft()
            return None
        if normalized.startswith("INSERT INTO Locations"):
            record = self._make_location_record(args)
            self.inserted_rows.append(record)
            return record
        if "FROM Locations" in normalized:
            self.fallback_query_args.append(args)
            if self.fallback_rows:
                return self.fallback_rows.popleft()
            return None
        raise AssertionError(f"Unexpected query: {normalized}")

    async def execute(self, _query, *_args):
        return None

    def _make_location_record(self, args):
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
        record = {
            "id": self.next_location_id,
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
        self.next_location_id += 1
        return record


def _install_fake_assign_and_location(monkeypatch):
    async def fake_assign_hierarchy(conn, candidate, **kwargs):
        candidate.place.meta.setdefault("place_key", f"place::{candidate.place.name}")
        return {"chain": [], "leaf": {"id": 101}, "world_name": "Earth"}

    monkeypatch.setattr(location_hierarchy, "assign_hierarchy", fake_assign_hierarchy)

    def fake_from_record(cls, record, **overrides):
        data = {
            "user_id": record["user_id"],
            "conversation_id": record["conversation_id"],
            "location_name": record["location_name"],
            "id": record.get("id"),
            "description": record.get("description"),
            "location_type": record.get("location_type"),
            "parent_location": record.get("parent_location"),
            "room": record.get("room"),
            "building": record.get("building"),
            "district": record.get("district"),
            "district_type": record.get("district_type"),
            "city": record.get("city"),
            "region": record.get("region"),
            "country": record.get("country"),
            "planet": record.get("planet") or "Earth",
            "galaxy": record.get("galaxy") or "Milky Way",
            "realm": record.get("realm") or DEFAULT_REALM,
            "lat": record.get("lat"),
            "lon": record.get("lon"),
            "is_fictional": record.get("is_fictional") or False,
            "open_hours": record.get("open_hours"),
            "controlling_faction": record.get("controlling_faction"),
        }
        data.update(overrides)
        return cls(**data)

    monkeypatch.setattr(location_hierarchy.Location, "from_record", classmethod(fake_from_record))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_region_isolated_between_cities(monkeypatch):
    connection = StubConnection(
        city_rows=deque([None, None]),
        fallback_rows=deque(
            [
                None,
                {
                    "city": "Alpha City",
                    "region": "North Province",
                    "country": "Freedonia",
                    "planet": "Earth",
                    "galaxy": "Milky Way",
                    "realm": DEFAULT_REALM,
                },
            ]
        ),
    )
    _install_fake_assign_and_location(monkeypatch)

    candidate_alpha = Candidate(
        place=Place(
            name="Alpha Park",
            level="venue",
            address={
                "_normalized_admin_path": {
                    "city": "Alpha City",
                    "region": "North Province",
                    "country": "Freedonia",
                }
            },
            meta={"display_name": "Alpha Park"},
        )
    )

    first_location = await location_hierarchy.generate_and_persist_hierarchy(
        connection,
        user_id=7,
        conversation_id=42,
        candidate=candidate_alpha,
        scope="real",
    )

    assert first_location.city == "Alpha City"
    assert first_location.region == "North Province"
    assert first_location.country == "Freedonia"

    candidate_beta = Candidate(
        place=Place(
            name="Beta Gardens",
            level="venue",
            address={"_normalized_admin_path": {"city": "Beta City"}},
            meta={"display_name": "Beta Gardens"},
        )
    )

    second_location = await location_hierarchy.generate_and_persist_hierarchy(
        connection,
        user_id=7,
        conversation_id=42,
        candidate=candidate_beta,
        scope="real",
    )

    assert second_location.city == "Beta City"
    assert second_location.region is None
    assert second_location.country is None
    assert connection.inserted_rows[0]["region"] == "North Province"


@pytest.mark.anyio
async def test_district_matches_anchor_gets_seeded_offset(monkeypatch):
    _install_fake_assign_and_location(monkeypatch)

    connection = StubConnection(city_rows=deque([None, None]), fallback_rows=deque([None, None]))

    anchor_geo = GeoAnchor(lat=37.7749, lon=-122.4194, neighborhood="Downtown", city="San Francisco")
    anchor = Anchor(
        scope="real",
        lat=37.7749,
        lon=-122.4194,
        primary_city="San Francisco",
        hints={"geo_anchor": anchor_geo},
    )

    candidate = Candidate(
        place=Place(
            name="Downtown Cafe",
            level="venue",
            address={"_normalized_admin_path": {"city": "San Francisco", "district": "Downtown"}},
            meta={"display_name": "Downtown Cafe"},
        )
    )

    location = await location_hierarchy.generate_and_persist_hierarchy(
        connection,
        user_id=5,
        conversation_id=99,
        candidate=candidate,
        scope="real",
        anchor=anchor,
    )

    normalized_name = location_hierarchy._normalize_location_name(location.location_name)
    rng = random.Random(f"99:{normalized_name}:offset")
    lat_offset = rng.uniform(-0.001, 0.001)
    lon_offset = rng.uniform(-0.001, 0.001)
    expected_lat = round(anchor.lat + lat_offset, 6)
    expected_lon = round(anchor.lon + lon_offset, 6)

    assert connection.inserted_rows[-1]["lat"] == expected_lat
    assert connection.inserted_rows[-1]["lon"] == expected_lon
    assert candidate.place.meta.get("district_center") == {
        "lat": round(anchor.lat, 6),
        "lon": round(anchor.lon, 6),
    }


@pytest.mark.anyio
async def test_new_district_mints_center_from_anchor(monkeypatch):
    _install_fake_assign_and_location(monkeypatch)

    connection = StubConnection(
        city_rows=deque([None, None, None, None]),
        fallback_rows=deque([None, None]),
    )

    anchor_geo = GeoAnchor(lat=34.0522, lon=-118.2437, city="Los Angeles")
    anchor = Anchor(
        scope="real",
        lat=34.0522,
        lon=-118.2437,
        primary_city="Los Angeles",
        hints={"geo_anchor": anchor_geo},
    )

    display_name = "Harbor Market"
    district_name = "Harbor District"
    candidate = Candidate(
        place=Place(
            name=display_name,
            level="venue",
            address={
                "_normalized_admin_path": {
                    "city": "Los Angeles",
                    "district": district_name,
                }
            },
            meta={"display_name": display_name},
        )
    )

    location = await location_hierarchy.generate_and_persist_hierarchy(
        connection,
        user_id=8,
        conversation_id=200,
        candidate=candidate,
        scope="real",
        anchor=anchor,
    )

    normalized_district = location_hierarchy._normalize_location_name(district_name)
    center_rng = random.Random(f"200:{normalized_district}:center")
    lat_center_offset = center_rng.uniform(-0.005, 0.005)
    lon_center_offset = center_rng.uniform(-0.005, 0.005)
    expected_center_lat = anchor.lat + lat_center_offset
    expected_center_lon = anchor.lon + lon_center_offset

    normalized_name = location_hierarchy._normalize_location_name(location.location_name)
    offset_rng = random.Random(f"200:{normalized_name}:offset")
    lat_offset = offset_rng.uniform(-0.001, 0.001)
    lon_offset = offset_rng.uniform(-0.001, 0.001)
    expected_lat = round(expected_center_lat + lat_offset, 6)
    expected_lon = round(expected_center_lon + lon_offset, 6)

    assert connection.inserted_rows[-1]["lat"] == expected_lat
    assert connection.inserted_rows[-1]["lon"] == expected_lon
    assert candidate.place.meta.get("district_center") == {
        "lat": round(expected_center_lat, 6),
        "lon": round(expected_center_lon, 6),
    }

    # Stable results on repeated invocation with the same identifiers.
    candidate_repeat = Candidate(
        place=Place(
            name=display_name,
            level="venue",
            address={
                "_normalized_admin_path": {
                    "city": "Los Angeles",
                    "district": district_name,
                }
            },
            meta={"display_name": display_name},
        )
    )

    await location_hierarchy.generate_and_persist_hierarchy(
        connection,
        user_id=8,
        conversation_id=200,
        candidate=candidate_repeat,
        scope="real",
        anchor=anchor,
    )

    assert connection.inserted_rows[-1]["lat"] == expected_lat
    assert connection.inserted_rows[-1]["lon"] == expected_lon
    assert connection.inserted_rows[1]["region"] is None
    # ensure we attempted to reuse context but did not propagate hierarchy data
    assert len(connection.city_query_args) == 2
    assert len(connection.fallback_query_args) == 2
