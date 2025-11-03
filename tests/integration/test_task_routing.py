from nyx.tasks.queues import QUEUES, ROUTES


def test_routes_cover_expected_tasks():
    assert ROUTES["nyx.tasks.realtime.post_turn.dispatch"]["queue"] == "realtime"
    assert ROUTES["nyx.tasks.background.world_tasks.apply_universal"]["queue"] == "background"
    assert ROUTES["nyx.tasks.heavy.memory_tasks.add_and_embed"]["queue"] == "heavy"
    assert ROUTES["tasks.background_chat_task_with_memory"]["queue"] == "realtime"


def test_priority_queue_configuration():
    queue_args = {queue.name: queue.queue_arguments for queue in QUEUES}
    for name in ("realtime", "background", "heavy"):
        assert name in queue_args
        assert queue_args[name].get("x-max-priority") == 10

    for config in ROUTES.values():
        if config.get("priority") is not None:
            assert 0 <= config["priority"] <= 7
