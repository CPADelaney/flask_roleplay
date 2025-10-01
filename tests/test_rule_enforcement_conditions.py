import pathlib
import sys
import types

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

from logic.rule_enforcement import parse_condition, evaluate_condition


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
