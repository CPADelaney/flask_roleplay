from datetime import datetime, timedelta, timezone

import pytest

from nyx.telemetry.metrics import (
    QUEUE_DELAY_SECONDS,
    record_queue_delay,
    record_queue_delay_from_context,
)


def _histogram_sum(metric, **labels):
    for family in metric.collect():
        for sample in family.samples:
            if sample.name.endswith("_sum") and all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return 0.0


def test_record_queue_delay_from_context_observes_histogram():
    labels = {"queue": "test"}
    start = _histogram_sum(QUEUE_DELAY_SECONDS, **labels)
    now = 100.0
    context = {"enqueued_at": now - 3.25}

    delay = record_queue_delay_from_context(context, queue="test", now=now)

    end = _histogram_sum(QUEUE_DELAY_SECONDS, **labels)
    assert delay == pytest.approx(3.25)
    assert end - start == pytest.approx(3.25)


def test_record_queue_delay_parses_iso_timestamp():
    labels = {"queue": "iso"}
    start = _histogram_sum(QUEUE_DELAY_SECONDS, **labels)
    enqueued = datetime(2024, 1, 1, tzinfo=timezone.utc)
    now = enqueued + timedelta(seconds=1.5)

    delay = record_queue_delay(enqueued.isoformat(), queue="iso", now=now.timestamp())

    end = _histogram_sum(QUEUE_DELAY_SECONDS, **labels)
    assert delay == pytest.approx(1.5, abs=1e-6)
    assert end - start == pytest.approx(delay, abs=1e-6)
