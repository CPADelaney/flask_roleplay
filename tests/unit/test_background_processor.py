import asyncio
import sys
import types
from enum import Enum
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class _StubDBContext:
    async def __aenter__(self):  # pragma: no cover - safety net
        raise AssertionError('test should patch get_db_connection_context')

    async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - safety net
        return False


stub_db_connection = types.ModuleType('db.connection')
stub_db_connection.get_db_connection_context = lambda *args, **kwargs: _StubDBContext()
sys.modules['db.connection'] = stub_db_connection

stub_time_cycle = types.ModuleType('logic.time_cycle')


async def _stub_current_day(*_args, **_kwargs):  # pragma: no cover - safety net
    return 0


stub_time_cycle.get_current_game_day = _stub_current_day
sys.modules['logic.time_cycle'] = stub_time_cycle

stub_grand_conflicts = types.ModuleType('logic.conflict_system.background_grand_conflicts')


class _BackgroundIntensity(Enum):
    DISTANT_RUMOR = 'distant_rumor'
    OCCASIONAL_NEWS = 'occasional_news'
    REGULAR_TOPIC = 'regular_topic'
    AMBIENT_TENSION = 'ambient_tension'
    VISIBLE_EFFECTS = 'visible_effects'


stub_grand_conflicts.BackgroundIntensity = _BackgroundIntensity
stub_grand_conflicts.INTENSITY_TO_FLOAT = {
    _BackgroundIntensity.DISTANT_RUMOR: 0.2,
    _BackgroundIntensity.OCCASIONAL_NEWS: 0.4,
    _BackgroundIntensity.REGULAR_TOPIC: 0.6,
    _BackgroundIntensity.AMBIENT_TENSION: 0.8,
    _BackgroundIntensity.VISIBLE_EFFECTS: 1.0,
}
sys.modules['logic.conflict_system.background_grand_conflicts'] = stub_grand_conflicts

from logic.conflict_system.background_processor import (  # noqa: E402,E501
    BackgroundConflictProcessor,
    get_high_intensity_threshold,
)


class _FakeConnection:
    def __init__(self, fetch_response=None, fetchval_response=None):
        self.fetch_response = fetch_response or []
        self.fetchval_response = fetchval_response
        self.last_fetch_args = None

    async def fetch(self, _query: str, *args):
        self.last_fetch_args = args
        return self.fetch_response

    async def fetchval(self, _query: str, *_args):
        return self.fetchval_response


class _FakeDBContext:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_process_scene_updates_handles_numeric_intensity(monkeypatch):
    threshold = get_high_intensity_threshold()

    conflict_row = {
        'id': 42,
        'intensity': threshold,
        'metadata': '{}',
        'locations': '["tavern"]',
        'figures': '[]',
        'name': 'Charged tavern unrest',
    }

    primary_connection = _FakeConnection(fetch_response=[conflict_row])
    ambient_connection = _FakeConnection(fetchval_response='Crackling energy fills the air')

    connections = [primary_connection, ambient_connection]

    def fake_get_db_connection_context():
        if not connections:
            raise AssertionError('No test connections available')
        return _FakeDBContext(connections.pop(0))

    monkeypatch.setattr(
        'logic.conflict_system.background_processor.get_db_connection_context',
        fake_get_db_connection_context,
    )

    processor = BackgroundConflictProcessor(user_id=1, conversation_id=2)

    scene_context = {
        'location': 'Tavern',
        'npcs': [],
        'conversation_topics': [],
    }

    async def _run():
        return await processor.process_scene_relevant_updates(scene_context)

    result = asyncio.run(_run())

    assert result['ambient_effects'] == ['Crackling energy fills the air']
    assert primary_connection.last_fetch_args == (1, 2, threshold)

