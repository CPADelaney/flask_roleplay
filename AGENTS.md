AGENTS.md

This repo powers a full-stack interactive roleplay platform: Quart (async Flask) + Socket.IO for realtime UX, Celery for heavy/async work, and an in-house Nyx agent framework for story orchestration, memory, NPCs, and lore.

Quick start
Requirements

Python: 3.12

PostgreSQL with pgvector (vector extension)

Redis 6+ (rate limiting, Socket.IO, Celery broker/backend)

Optional: Qdrant/Chroma if you run external vector stores

Install
# venv
python -m venv .venv && source .venv/bin/activate

# deps (prefer constraints for reproducibility)
pip install -r requirements.txt --constraint constraints.txt

Environment

Copy .env.example if present. Minimum:

OPENAI_API_KEY – required at import time (see config_startup.py)

DB_DSN – e.g. postgresql://user:pass@host:5432/dbname

REDIS_URL – e.g. redis://localhost:6379/0

SECRET_KEY – sessions

Common knobs: LOG_LEVEL, CELERY_BROKER_URL, USE_RABBITMQ, NYX_*, QDRANT_URL, CORS_ALLOWED_ORIGINS, SOCKET_PING_INTERVAL, SOCKET_PING_TIMEOUT.

Run (local)
# Web (ASGI)
hypercorn wsgi:app --bind 0.0.0.0:8080 --reload

# Celery workers (queues: realtime, background, heavy)
celery -A tasks.celery_app worker \
  --loglevel=INFO --queues realtime,background,heavy \
  --pool=prefork --concurrency=1


Docker: docker-compose up --build brings up Quart, Redis, and a Celery worker. entrypoint.sh switches roles via SERVICE_TYPE=web|worker.

Runtime architecture
Ingress & app lifecycle

main.create_quart_app() registers blueprints, Socket.IO, rate limiting, auth/session middleware, and Prometheus metrics.

wsgi.py builds the ASGI app and can start Nyx’s long-lived coroutines (nightly rollups, reflections) outside the request cycle.

Readiness: startup initializes the asyncpg pool, primes caches, warm-loads story/preset catalogs, then flips an “app ready” signal used by Socket handlers and tasks.

Request→response (happy path)

HTTP/Socket.IO: routes under routes/ and the storybeat Socket event receive input, dedupe via Redis, and validate auth.

Context build: logic.aggregator_sdk.get_aggregated_roleplay_context() merges DB world state, cache layers, and memory into a single context bundle.

Narrative: Nyx + story modules (story_agent/, logic/roleplay_engine.py) call LLMs to produce narrative + structured updates.

Immediate sync writes (tiny): a minimal ConversationSnapshot (location/time/participants/conflict/world_version) is saved for the next prompt.

Post-turn fan-out (Celery): the SDK packages side-effects (world deltas, memory event, conflict/NPC/lore hints) and enqueues nyx.tasks.realtime.post_turn.dispatch, which fans out to background/heavy queues.

Latency-first design (what’s sync vs async)

Synchronous (prompt-critical)

Prompt assembly (recent messages + tiny snapshot)

LLM call & output shaping

Minimal snapshot write (for immediate coherence)

Asynchronous (Celery)

World updates (world.apply_universal)

Memory persistence + embeddings (memory.add_and_embed + periodic consolidate_decay)

Conflict processing (conflict.process_events)

NPC adaptation (npc.run_adaptation_cycle)

Lore bundle warmups (lore.precompute_scene_bundle)

Celery queues:

realtime → lightweight dispatcher

background → world/conflict/NPC/lore

heavy → embeddings/consolidation

Roleplay loop (turn-by-turn)

Context build – aggregator composes current scene, NPC presence, quests/events, and preset metadata (cached singletons initialized at startup).

Memory recall – memory.wrapper.MemorySystem routes to emotional/flashback/mask layers and vector retrieval.

Narrative generation – RoleplayEngine.generate_turn() or the Nyx story orchestrator calls GPT for narrative + JSON updates.

World/application – universal updater applies structured updates; caches/context service refresh for subsequent turns.

Fan-out – post-turn side-effects are sent to Celery (see above); the next user input benefits from the context window + snapshot while durable state catches up.

Nyx & Agents SDK

Nyx governs permissions, directives, and inter-agent delegation (see nyx/ governance helpers, directive handlers).

Agents SDK (vendored) provides guardrails (InputGuardrail, OutputGuardrail), tools (agents.function_tool), tracing, and run contexts.

When adding agents/flows, use Nyx governance decorators to keep audits intact and tracing consistent.

Background workers

Celery app: celery_config.py (merges native tasks + nyx.tasks routes/queues; supports Redis or RabbitMQ broker).

Start:

celery -A tasks.celery_app worker --loglevel=INFO \
  --queues realtime,background,heavy --pool=prefork --concurrency=1


Beat: periodic jobs for memory consolidation, NPC cycles, lore refresh.

Observability: optional Prometheus counters/histograms (enable by installing prometheus_client).

Persistence & caching

Always use db.connection.get_db_connection_context() for Postgres (asyncpg pool, pgvector registration, Celery lifecycle hooks).

Redis backs rate limits, Socket.IO, Celery, and the ConversationSnapshotStore (with in-proc fallback).

Prefer utils.cache_manager.CacheManager for TTL caches (NPC/context), not ad-hoc dicts.

Directory guide

main.py – Quart factory, Socket handlers (storybeat), request dedupe, logging/metrics bootstrap, Nyx init.

routes/ – blueprints (new game, player input, auth, knowledge, NPC learning, settings, conflicts, admin).

logic/ – aggregator SDK, roleplay engine, GPT integration, universal updater, calendar/time, relationship/conflict systems.

story_agent/ – orchestrators, simulation agents, daily schedulers, progressive summarization.

memory/ – unified memory system (vector stores, consolidation, emotional tagging) via MemorySystem.

context/ – context service/cache, vector retrieval, perf monitors.

nyx/ – Nyx brain, governance, agent SDK integration, scene/state managers, sync daemon, Celery task packages (nyx/tasks/...).

npcs/ – canonical NPC creation, learning/adaptation, perception; dynamic relationships.

story_templates/ – preset stories, constraint prompts, validation guides.

db/ – async pool, schema helpers/migrations, admin.

monitoring/ – Prometheus metrics & perf config.

utils/, middleware/ – cache helpers, embedding service, validation/security, rate limiting.

tests/ – unit/integration/perf suites with markers.

Development workflow

Let initialize_systems own pool/Redis/Nyx init; don’t spin ad-hoc clients/pools.

Reuse get_db_connection_context() and the memory/NPC wrappers (keeps caches, schemas, and governance consistent).

Route world changes through the universal updater or story orchestrators (avoid bespoke SQL).

Follow existing logging/metrics patterns (monitoring/metrics.py).

Format/lint/typecheck:

black . && isort . && flake8 && mypy .

Testing
pytest                       # full suite with coverage
pytest -m unit               # or: integration, npc, memory, performance


Some tests require Postgres/Redis. Export DB_DSN, REDIS_URL, and seed data beforehand.

Observability & ops

/metrics exposes Prometheus; decorate long-running handlers/tasks via monitoring/metrics.py.

Socket dedupe/rate limit uses Redis—set REDIS_URL or expect local fallback warnings.

Celery uses prioritized queues and can dead-letter on worker loss (configurable in celery_config.py).

Troubleshooting

“DB_DSN not set”: export it (web and worker).

Anonymous Socket.IO rejected: front-end must pass user_id in the auth payload.

Tasks stuck: verify worker can reach Redis/Postgres, and startup called the app init that flips readiness.

Nyx modules “missing”: many import lazily; check logs and ensure NYX_* env is consistent with initialization.

Further reading

README.md – features, setup, deploy.

nyx/Nyx_Doc.txt – deep architecture.

consolidation_implementation_plan.md, migration_plan.txt, data_access_layer_implementation.md – refactor history.

strategy/, lore/, story_templates/ – emergent storytelling & canonical content.

Appendix: Post-turn fan-out (what’s actually queued)

The SDK emits typed side-effects and enqueues nyx.tasks.realtime.post_turn.dispatch:

World → nyx.tasks.background.world_tasks.apply_universal

Memory → nyx.tasks.heavy.memory_tasks.add_and_embed (+ periodic consolidate_decay)

Conflict → nyx.tasks.background.conflict_tasks.process_events

NPCs → nyx.tasks.background.npc_tasks.run_adaptation_cycle

Lore → nyx.tasks.background.lore_tasks.precompute_scene_bundle

All tasks are idempotent, versioned where relevant, and safe to skip under load (optional throttle flags).
