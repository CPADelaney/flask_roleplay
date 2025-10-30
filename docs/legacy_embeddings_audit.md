# Legacy Embeddings Audit

The `scripts/ci/forbid_legacy_embeddings.py` scan surfaced several remaining
references to the deprecated `embedding.vector_store` helpers. Each item below
tracks a group of files that still rely on the legacy implementations along with
the recommended follow-up.

- [ ] **Lore managers and lore generator**  \
  Files: `lore/managers/base_manager.py`, `lore/managers/local_lore.py`,
  `lore/managers/geopolitical.py`, `lore/managers/education.py`,
  `lore/managers/religion.py`, `lore/matriarchal_lore_system.py`,
  `lore/lore_generator.py`, `lore/core/canon.py`, `lore/data_access.py`, and
  supporting systems such as `lore/systems/dynamics.py` and
  `lore/systems/regional_culture.py`.  \
  **Action:** Replace direct calls to `generate_embedding` with
  `rag.ask.ask(..., mode="embedding")` (or `utils.embedding_service.get_embedding`
  for cached access) and remove the legacy imports.

- [ ] **Story templates and initialisers**  \
  Files: `story_templates/moth/story_initializer.py` and other template helpers
  that stub embeddings during seeding.  \
  **Action:** Switch fixture generation to call the loader script (for static
  data) or to mock `rag.ask.ask` in tests instead of instantiating legacy
  embeddings.

- [ ] **Logic modules and ad-hoc embedding fallbacks**  \
  Files: `logic/conflict_system/conflict_canon.py`,
  `logic/artifact_system/artifact_manager.py`, and
  `logic/chatgpt_integration.py`.  \
  **Action:** Route these call sites through `rag.ask.ask` or the central
  embedding service so they honour the hosted vector store configuration.

- [ ] **Tests relying on the legacy module**  \
  Files: `tests/test_canon_quests.py`, `tests/unit/test_matriarchal_framework.py`,
  `tests/unit/test_lore_context_embeddings.py`, and similar fixtures that create
  in-memory stubs of `embedding.vector_store`.  \
  **Action:** Update tests to stub `rag.ask.ask` or `utils.embedding_service`
  instead of importing the legacy module, ensuring CI coverage matches production
  paths.

- [ ] **Package re-export**  \
  File: `embedding/__init__.py` still re-exports the legacy helpers.  \
  **Action:** Replace these exports with thin wrappers around the new RAG APIs or
  remove the module entirely once callers have been migrated.

Re-running the audit (`python scripts/ci/forbid_legacy_embeddings.py`) after
addressing each checkbox should return a clean report.
