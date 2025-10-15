import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nyx.geo import toponym


class DummyResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


class DummyHTTPClient:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self._status_code = status_code
        self.calls = 0

    async def get(self, *args, **kwargs):
        self.calls += 1
        return DummyResponse(self._payload, status_code=self._status_code)

    async def aclose(self):
        return None


class FakeGeoConnection:
    def __init__(self):
        self.geo_cache = {}
        self.world_locations = {}

    async def fetchrow(self, query, *args):
        if "FROM geo_cache" in query:
            provider, normalized = args
            entry = self.geo_cache.get((provider, normalized))
            if entry is None:
                return None
            return entry
        if "FROM world_locations" in query:
            (normalized,) = args
            entry = self.world_locations.get(normalized)
            if entry is None:
                return None
            return entry
        return None

    async def execute(self, query, *args):
        if "INSERT INTO geo_cache" in query:
            provider, original_query, normalized, payload, confidence, expires_at = args
            self.geo_cache[(provider, normalized)] = {
                "response": payload,
                "confidence": confidence,
                "expires_at": expires_at,
            }
            return "INSERT 0 1"
        if "INSERT INTO world_locations" in query:
            (
                name,
                normalized_name,
                country_code,
                admin1,
                admin2,
                latitude,
                longitude,
                feature_class,
                feature_code,
                data_source,
                confidence,
            ) = args
            self.world_locations[normalized_name] = {
                "name": name,
                "normalized_name": normalized_name,
                "country_code": country_code,
                "admin1": admin1,
                "admin2": admin2,
                "latitude": latitude,
                "longitude": longitude,
                "feature_class": feature_class,
                "feature_code": feature_code,
                "data_source": data_source,
                "confidence": confidence,
            }
            return "INSERT 0 1"
        return "OK"


def _future(seconds: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def test_geocode_caches_results(monkeypatch):
    fake_conn = FakeGeoConnection()

    @asynccontextmanager
    async def fake_db_context(*_, **__):
        yield fake_conn

    monkeypatch.setattr(toponym, "get_db_connection_context", fake_db_context)

    payload = [
        {
            "name": "Pier 39",
            "display_name": "Pier 39, San Francisco, California, USA",
            "lat": "37.80867",
            "lon": "-122.40982",
            "importance": 0.92,
            "class": "tourism",
            "type": "attraction",
            "address": {
                "country_code": "us",
                "state": "California",
                "county": "San Francisco County",
            },
        }
    ]
    http_client = DummyHTTPClient(payload)

    async def _run():
        result = await toponym.geocode("Pier 39", http_client=http_client)
        assert result is not None
        assert result.confidence == pytest.approx(0.92)
        assert http_client.calls == 1
        assert (toponym.DEFAULT_PROVIDER, "pier 39") in fake_conn.geo_cache
        assert "pier 39" in fake_conn.world_locations

        http_client_second = DummyHTTPClient(payload)
        cached = await toponym.geocode("Pier 39", http_client=http_client_second)
        assert cached is not None
        assert cached.confidence == pytest.approx(0.92)
        assert http_client_second.calls == 0

    asyncio.run(_run())


def test_plausibility_score_prefers_world_locations(monkeypatch):
    fake_conn = FakeGeoConnection()
    fake_conn.world_locations["pier 39"] = {"confidence": 0.95}

    @asynccontextmanager
    async def fake_db_context(*_, **__):
        yield fake_conn

    monkeypatch.setattr(toponym, "get_db_connection_context", fake_db_context)

    http_client = DummyHTTPClient([])

    async def _run():
        score = await toponym.plausibility_score("Pier 39", http_client=http_client)
        assert score == pytest.approx(0.95)
        assert http_client.calls == 0

    asyncio.run(_run())


def test_plausibility_uses_cached_confidence(monkeypatch):
    fake_conn = FakeGeoConnection()
    fake_conn.geo_cache[(toponym.DEFAULT_PROVIDER, "golden gate park")] = {
        "response": {"importance": 0.87},
        "confidence": 0.87,
        "expires_at": _future(3600),
    }

    @asynccontextmanager
    async def fake_db_context(*_, **__):
        yield fake_conn

    monkeypatch.setattr(toponym, "get_db_connection_context", fake_db_context)

    http_client = DummyHTTPClient([])

    async def _run():
        score = await toponym.plausibility_score("Golden Gate Park", http_client=http_client)
        assert score == pytest.approx(0.87)
        assert http_client.calls == 0

    asyncio.run(_run())
