"""Hot-path helpers for Nyx memory dispatching.

This module intentionally exposes only cache/dispatch entrypoints. Add new
functions with prefixes `dispatch_`, `enqueue_`, or `get_*_from_cache` when the
memory subsystem adopts the split worker model.
"""

__all__: list[str] = []
