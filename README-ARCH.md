# Nyx Architecture Rollout Flags

To safely roll out the latest Nyx orchestration architecture we gate each major subsystem behind environment flags. All flags default to `on` so production deployments automatically use the modern stack, and operators can toggle individual systems back to their legacy behaviour by exporting the flag with the value `off`, `false`, or `0`.

| Flag | Default | Controls | Legacy behaviour when disabled |
| ---- | ------- | -------- | -------------------------------- |
| `NYX_FLAG_LLM_GATEWAY` | `on` | Routes LLM invocations through the consolidated gateway with retries and metadata. | Falls back to direct `Runner.run` calls for agent execution. |
| `NYX_FLAG_OUTBOX` | `on` | Persists post-turn side effects in the transactional outbox before dispatching Celery workers. | Sends side effects directly to Celery tasks without touching the outbox. |
| `NYX_FLAG_VERSIONED_CACHE` | `on` | Reads and writes versioned conversation snapshots for cache invalidation. | Keeps per-process snapshots only and skips canonical snapshot persistence. |
| `NYX_FLAG_CONFLICT_FSM` | `on` | Emits conflict resolution events that drive the conflict FSM pipeline. | Skips emitting conflict FSM events; downstream processors retain legacy behaviour. |
| `NYX_FLAG_DOMAIN_EVENTS` | `on` | Publishes memory, world, NPC, lore, and conflict events to downstream subscribers. | Suppresses domain events and leaves only synchronous orchestrator updates. |
| `NYX_FLAG_OUTPUT_EVALS` | `on` | Enables post-turn automated evaluations (conflict drafts, quality checks). | Skips automated evaluation tasks. |

## Operations

* **Toggling a flag:** Export the variable before launching the web app or Celery worker. Example: `export NYX_FLAG_OUTBOX=off`.
* **Rolling back multiple systems:** Set any combination of flags to `off` to revert those systems to their pre-gateway behaviour without redeploying code.
* **Verification:** Run `pytest tests/integration/test_full_turn.py` to exercise the SDK end-to-end with the currently configured flags. The test also covers the legacy path by disabling the outbox flag at runtime.

All flags are read at call-time which makes them safe to change between restarts. Always restart workers after toggling to guarantee a consistent state across processes.
