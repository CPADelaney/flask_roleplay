import pathlib
import sys
import types

import asyncio
import asyncpg
import numpy as np


class _DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), self._dim), dtype=float)


class _DummyTransformer:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def get_word_embedding_dimension(self):
        return self._dim


class _DummyPooling:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def get_sentence_embedding_dimension(self):
        return self._dim


dummy_sentence_transformers = types.SimpleNamespace(
    SentenceTransformer=_DummySentenceTransformer,
    models=types.SimpleNamespace(
        Transformer=_DummyTransformer,
        Pooling=lambda *args, **kwargs: _DummyPooling(),
    ),
)

sys.modules.setdefault("sentence_transformers", dummy_sentence_transformers)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import logic.rule_enforcement as rule_enforcement
from logic.rule_enforcement import evaluate_condition, parse_condition


def _sample_metadata():
    return {
        "feasibility": {
            "per_intent": [
                {
                    "categories": ["unaided_flight", "public_magic"],
                    "violations": [{"rule": "category:unaided_flight", "reason": "banned"}],
                }
            ],
            "overall": {
                "violations": [{"rule": "hazard:vacuum", "reason": "danger"}],
            },
            "capabilities": {
                "magic": "limited",
                "hazards": ["vacuum", "radiation"],
            },
        }
    }


def test_parse_condition_numeric_rule():
    logic_op, parsed = parse_condition("obedience >= 70")
    assert logic_op == "SINGLE"
    assert parsed == [
        {
            "type": "numeric",
            "stat": "Obedience",
            "operator": ">=",
            "threshold": 70,
        }
    ]


def test_parse_condition_token_bundle():
    logic_op, parsed = parse_condition("category:unaided_flight|magic:visible_in_public")
    assert logic_op == "SINGLE"
    assert parsed == [
        {
            "type": "token_bundle",
            "tokens": [
                {"kind": "category", "value": "unaided_flight"},
                {"kind": "magic", "value": "visible_in_public"},
            ],
        }
    ]


def test_evaluate_condition_numeric_true():
    logic_op, parsed = parse_condition("obedience >= 70")
    stats = {"Obedience": 72}
    assert evaluate_condition(logic_op, parsed, stats, metadata=_sample_metadata()) is True


def test_evaluate_condition_token_bundle_matches_category():
    logic_op, parsed = parse_condition("category:unaided_flight|magic:visible_in_public")
    assert evaluate_condition(logic_op, parsed, {}, metadata=_sample_metadata()) is True


def test_evaluate_condition_combined_and_rule():
    logic_op, parsed = parse_condition("obedience >= 70 and hazard:vacuum")
    stats = {"Obedience": 71}
    assert evaluate_condition(logic_op, parsed, stats, metadata=_sample_metadata()) is True


def test_evaluate_condition_token_bundle_false_when_missing():
    logic_op, parsed = parse_condition("category:teleportation")
    assert (
        evaluate_condition(logic_op, parsed, {}, metadata=_sample_metadata())
        is False
    )


class _FakeDBContext:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ScopedRuleConnection:
    def __init__(self):
        self.fetch_calls = []

    async def fetch(self, query, *args):
        normalized = " ".join(query.split())
        self.fetch_calls.append((normalized, args))
        if "user_id=$1" in normalized:
            return [
                {"condition": "obedience >= 60", "effect": "{\"apply\": \"obedience\"}"},
                {"condition": "corruption >= 10", "effect": "{\"apply\": \"global\"}"},
            ]
        raise AssertionError(f"Unexpected query: {query}")


class _LegacyRulesConnection:
    def __init__(self):
        self.fetch_attempts = []

    async def fetch(self, query, *args):
        normalized = " ".join(query.split())
        self.fetch_attempts.append((normalized, args))
        if "enabled" in normalized:
            raise asyncpg.UndefinedColumnError("enabled")
        if "user_id=$1" in normalized:
            raise asyncpg.UndefinedColumnError("user_id")
        return [
            {"condition": "obedience >= 50", "effect": "{\"apply\": \"legacy\"}"},
        ]


class _LargeDatasetConnection:
    def __init__(self, target_scope, other_scopes):
        self.target_scope = target_scope
        self.other_scopes = other_scopes
        self.fetch_calls = []
        self.global_rules = [
            {"condition": "corruption >= 10", "effect": "{\"apply\": \"global\"}"}
        ]
        self.rules_by_scope = {
            target_scope: [
                {"condition": "obedience >= 10", "effect": "{\"apply\": \"target\"}"}
            ]
        }
        for scope in other_scopes:
            self.rules_by_scope[scope] = [
                {"condition": f"other-scope:{scope[0]}-{scope[1]}", "effect": "{\"apply\": \"other\"}"}
            ]

    async def fetch(self, query, *args):
        normalized = " ".join(query.split())
        self.fetch_calls.append((normalized, args))
        if "user_id=$1" in normalized:
            uid, cid = args
            assert (uid, cid) == self.target_scope
            scoped = self.rules_by_scope.get((uid, cid), [])
            return scoped + self.global_rules
        raise AssertionError("Scoped query expected for large dataset test")


async def _fake_player_stats(*args, **kwargs):
    return {"Obedience": 95, "Corruption": 20, "Dependency": 15}


async def _fake_apply_effect(effect, *args, **kwargs):
    return {"effect": effect}


def test_enforce_rules_scoped_query_prefers_active_conversation(monkeypatch):
    async def _run():
        conn = _ScopedRuleConnection()
        monkeypatch.setattr(
            rule_enforcement, "get_db_connection_context", lambda: _FakeDBContext(conn)
        )
        monkeypatch.setattr(rule_enforcement, "get_player_stats", _fake_player_stats)
        monkeypatch.setattr(rule_enforcement, "apply_effect", _fake_apply_effect)

        result = await rule_enforcement.enforce_all_rules_on_player(
            player_name="Chase",
            user_id=42,
            conversation_id=99,
            metadata={"scene_tags": ["punishment"], "turn_index": 7},
        )

        rule_enforcement.purge_punishment_gate_state(42, 99)

        assert result["tier"] == "major"
        conditions = [item["condition"] for item in result["triggered"]]
        assert conditions == ["obedience >= 60", "corruption >= 10"]
        assert len(conn.fetch_calls) == 1
        recorded_query, params = conn.fetch_calls[0]
        assert "user_id=$1" in recorded_query
        assert "OR (user_id IS NULL AND conversation_id IS NULL)" in recorded_query
        assert params == (42, 99)

    asyncio.run(_run())


def test_enforce_rules_falls_back_when_columns_missing(monkeypatch):
    async def _run():
        conn = _LegacyRulesConnection()
        monkeypatch.setattr(
            rule_enforcement, "get_db_connection_context", lambda: _FakeDBContext(conn)
        )
        monkeypatch.setattr(rule_enforcement, "get_player_stats", _fake_player_stats)
        monkeypatch.setattr(rule_enforcement, "apply_effect", _fake_apply_effect)

        result = await rule_enforcement.enforce_all_rules_on_player(
            player_name="Chase",
            user_id=7,
            conversation_id=11,
            metadata={"scene_tags": ["discipline"], "turn_index": 5},
        )

        rule_enforcement.purge_punishment_gate_state(7, 11)

        assert result["triggered"]
        assert any(item["condition"] == "obedience >= 50" for item in result["triggered"])
        # Ensure we ultimately fell back to the unscoped legacy query.
        assert conn.fetch_attempts[-1][0].strip().lower().startswith(
            "select condition, effect from gamerules"
        )

    asyncio.run(_run())


def test_enforce_rules_large_dataset_reads_only_target_scope(monkeypatch):
    async def _run():
        target_scope = (101, 202)
        other_scopes = [(200 + i, 300 + i) for i in range(150)]
        conn = _LargeDatasetConnection(target_scope, other_scopes)
        monkeypatch.setattr(
            rule_enforcement, "get_db_connection_context", lambda: _FakeDBContext(conn)
        )
        monkeypatch.setattr(rule_enforcement, "get_player_stats", _fake_player_stats)
        monkeypatch.setattr(rule_enforcement, "apply_effect", _fake_apply_effect)

        result = await rule_enforcement.enforce_all_rules_on_player(
            player_name="Chase",
            user_id=target_scope[0],
            conversation_id=target_scope[1],
            metadata={"scene_tags": ["punishment"], "turn_index": 15},
        )

        rule_enforcement.purge_punishment_gate_state(*target_scope)

        assert len(conn.fetch_calls) == 1
        conditions = {item["condition"] for item in result["triggered"]}
        assert conditions == {"obedience >= 10", "corruption >= 10"}
        # Ensure no unrelated conversation rules leaked into the evaluation path.
        assert all(not cond.startswith("other-scope") for cond in conditions)

    asyncio.run(_run())
