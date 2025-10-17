from collections import deque
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from nyx.location import hierarchy as location_hierarchy
from nyx.location.types import Candidate, Place, DEFAULT_REALM


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
    assert connection.inserted_rows[1]["region"] is None
    # ensure we attempted to reuse context but did not propagate hierarchy data
    assert len(connection.city_query_args) == 2
    assert len(connection.fallback_query_args) == 2
