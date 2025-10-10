"""Helpers for managing OpenAI conversation scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .conversations import rotate_conversation_scene


class SceneManager:
    """Utility wrapper that coordinates scene rotation persistence."""

    def __init__(self, conversation_id: int) -> None:
        self._conversation_id = conversation_id
        self._active_scene: Optional[Dict[str, Any]] = None

    @property
    def current_scene(self) -> Optional[Dict[str, Any]]:
        """Return the last scene record persisted by :meth:`rotate_if_needed`."""

        return self._active_scene

    async def rotate_if_needed(
        self,
        *,
        new_scene: Optional[Mapping[str, Any]] = None,
        closing_scene: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Rotate the active scene if ``new_scene`` data is provided.

        Parameters
        ----------
        new_scene:
            Mapping describing the scene that should become active.
        closing_scene:
            Optional metadata that should be merged into the scene being closed
            before the new scene is inserted.
        """

        if not new_scene:
            return None

        persisted_scene = await rotate_conversation_scene(
            conversation_id=self._conversation_id,
            new_scene=new_scene,
            closing_scene=closing_scene,
        )

        if persisted_scene is not None:
            self._active_scene = dict(persisted_scene)

        return persisted_scene
