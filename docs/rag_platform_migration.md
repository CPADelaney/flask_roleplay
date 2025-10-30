# RAG Platform Migration Guide

This guide documents the knobs required to switch between the legacy in-house
vector store and the hosted OpenAI Agents vector store. It also explains how to
load historical memories into the hosted platform and how to run the associated
integration tests.

## Hosted vector store loader

Use [`scripts/load_vector_store.py`](../scripts/load_vector_store.py) to push
JSON or JSON Lines documents into the configured OpenAI vector store.

```bash
python scripts/load_vector_store.py path/to/memories.json \
  --collection memory_embeddings \
  --metadata component=backfill \
  --metadata import_batch=2024-05-10
```

Key options:

- `--vector-store-id` &mdash; override the destination vector store. When omitted,
  the loader uses `OPENAI_VECTOR_STORE_NAME` (or the first ID from
  `AGENTS_VECTOR_STORE_IDS`).
- `--collection` &mdash; set the `collection` metadata field on every document.
- `--metadata KEY=VALUE` &mdash; attach additional metadata entries. You can repeat
  this option multiple times.
- `--dry-run` &mdash; validate the payload without calling the OpenAI API.

Each document must contain a `text` field and may optionally provide
`id`/`memory_id`, `filename`, and a `metadata` object. The loader merges the
per-document metadata with the extra CLI-supplied values before uploading. Any
errors during parsing or feature flag validation abort the run with a non-zero
exit code.

## Feature flags and environment variables

Hosted vector store uploads and retrievals are gated by the following
environment variables (all supported in `.env` files):

| Variable | Purpose | Default |
| --- | --- | --- |
| `RAG_BACKEND` | Choose `agents`, `legacy`, or `auto` fallback logic. | `auto` |
| `OPENAI_VECTOR_STORE_NAME` | Friendly identifier for the hosted vector store. | *(empty)* |
| `AGENTS_VECTOR_STORE_IDS` | Comma-separated list of vector store IDs. | *(empty)* |
| `ENABLE_LEGACY_VECTOR_STORE` | Force the legacy pgvector backend. | `0`/`false` |
| `DISABLE_AGENTS_RAG` | Disable the Agents-backed retrieval flow. | `0`/`false` |
| `DISABLE_AGENTS_VECTOR_STORE` | Disable Agents vector store uploads/search. | `0`/`false` |

Set `ENABLE_LEGACY_VECTOR_STORE=1` or `RAG_BACKEND=legacy` to fall back to the
previous pgvector implementation. Setting either `DISABLE_AGENTS_RAG` or
`DISABLE_AGENTS_VECTOR_STORE` also bypasses the hosted experience; the loader
script exits early when those guards are active.

## Integration tests

The integration tests that exercise the hosted stack are **opt-in** and require
both an API key and an explicit environment toggle:

```bash
export OPENAI_API_KEY=sk-your-key
export RUN_RAG_VECTOR_TESTS=1
pytest tests/test_rag_vector_store_integration.py
```

Without both variables, the tests are skipped during collection. The suite
includes:

1. `test_loader_script_upserts_documents` &mdash; validates the loader CLI against
   fixture payloads and ensures it calls `upsert_hosted_vector_documents`.
2. `test_agents_retrieval_integration` &mdash; ensures `rag.backend.ask` routes
   retrievals through the Agents shim when requested.

Refer back to `.env.example` for a quick-start template covering the required
configuration flags.
