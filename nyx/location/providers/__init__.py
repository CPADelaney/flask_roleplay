"""Provider adapters for Nyx location services."""

from .gazetteer import candidate_from_nominatim

__all__ = ["candidate_from_nominatim"]
