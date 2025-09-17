# logic/stimuli_bus.py
from __future__ import annotations
import time
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

# Optional: if you placed StimuliExtractor elsewhere, adjust the import.
from logic.stimuli_extractor import get_default_extractor, StimuliExtractor, CANON_TO_ADDICTION

@dataclass(frozen=True)
class StimulusEvent:
    """
    A lightweight signal that something relevant was observed.
    - source: 'vision' | 'ner' | 'asr' | 'text' | 'rule' | 'custom'
    - label: raw label as seen by the detector (we'll canonicalize)
    - confidence: 0..1 (best effort)
    - turn_index: optional; attach if the publisher knows the current turn
    - extra: anything else (bbox, speaker, device id...)
    """
    source: str
    label: str
    confidence: float = 1.0
    turn_index: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None
    ts: float = time.time()


class StimuliEventBus:
    """Ultra-simple pub/sub for stimulus events."""
    def __init__(self) -> None:
        self._subs: Set[Callable[[StimulusEvent], None]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, fn: Callable[[StimulusEvent], None]) -> None:
        async with self._lock:
            self._subs.add(fn)

    async def unsubscribe(self, fn: Callable[[StimulusEvent], None]) -> None:
        async with self._lock:
            self._subs.discard(fn)

    async def publish(self, ev: StimulusEvent) -> None:
        # fire-and-forget; keep it non-blocking
        async with self._lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(ev)
            except Exception:
                # never let a subscriber kill the bus
                pass


class StimuliAggregator:
    """
    Collects StimulusEvents across sources, de-dupes, and exposes a per-turn snapshot.
    Also merges in text/NER signals via StimuliExtractor so the orchestrator doesn’t
    hand-curate stimuli.

    Usage:
        bus = StimuliEventBus()
        agg = StimuliAggregator(bus=bus)
        await agg.start()

        # elsewhere (vision thread):
        await bus.publish(StimulusEvent(source='vision', label='sandals', confidence=0.92, turn_index=42))

        # orchestrator, each turn:
        stimuli = agg.snapshot_for_turn(
            turn_index=42,
            scene_text=..., ner_entities=..., addiction_levels=...
        )
    """
    def __init__(
        self,
        bus: Optional[StimuliEventBus] = None,
        extractor: Optional[StimuliExtractor] = None,
        retain_seconds: float = 120.0,   # keep raw events this long
        max_tags_per_turn: int = 6,
        min_conf: float = 0.35,
    ) -> None:
        self._bus = bus
        self._extractor = extractor or get_default_extractor()
        self._retain_seconds = retain_seconds
        self._max_tags = max_tags_per_turn
        self._min_conf = min_conf

        # Raw event buffer (append-only; pruned on read)
        self._events: List[StimulusEvent] = []
        self._lock = asyncio.Lock()

        # Per-turn cache: turn -> (tags, ts)
        self._cache: Dict[int, Tuple[Set[str], float]] = {}

        # subscriber handle
        self._subscribed = False

    # ---- lifecycle --------------------------------------------------

    async def start(self) -> None:
        if self._bus and not self._subscribed:
            await self._bus.subscribe(self._on_event)
            self._subscribed = True

    async def stop(self) -> None:
        if self._bus and self._subscribed:
            await self._bus.unsubscribe(self._on_event)
            self._subscribed = False

    # ---- event intake -----------------------------------------------

    def _on_event(self, ev: StimulusEvent) -> None:
        # Called from bus (sync). Keep it very cheap.
        self._events.append(ev)

    # ---- snapshot ---------------------------------------------------

    def snapshot_for_turn(
        self,
        turn_index: int,
        scene_text: Optional[str] = None,
        ner_entities: Optional[List[Tuple[str, str]]] = None,
        object_detections: Optional[List[Dict[str, str]]] = None,
        addiction_levels: Optional[Dict[str, int]] = None,
    ) -> Set[str]:
        """
        Returns a de-duped set of canonical stimuli tags for this turn, merging:
        - queued events (recent + matching turn if provided)
        - text/NER/object signals via StimuliExtractor
        Caches result keyed by turn_index.
        """
        now = time.time()

        # cache hit?
        cached = self._cache.get(turn_index)
        if cached and (now - cached[1]) < 5.0:  # tiny grace window
            return set(cached[0])

        # 1) Fold queued events (recent) with mild scoring
        #    We prefer events that either match this turn_index OR are recent enough.
        horizon = self._retain_seconds
        by_canon: Dict[str, float] = {}  # canon -> score

        fresh: List[StimulusEvent] = []
        cutoff = now - horizon
        for ev in self._events:
            if ev.ts >= cutoff:
                if (ev.turn_index is None) or (ev.turn_index == turn_index):
                    fresh.append(ev)

        # De-dupe & score
        for ev in fresh:
            canon = self._extractor._to_canon(ev.label)  # canonical tag or None
            if not canon:
                continue
            # base weights by source
            if ev.source == "vision":
                base = 0.80
            elif ev.source == "ner":
                base = 0.50
            elif ev.source == "asr":
                base = 0.45
            elif ev.source == "text":
                base = 0.60
            else:
                base = 0.40
            score = min(1.0, base * 0.7 + ev.confidence * 0.3)

            # tiny bias by addiction severity
            add_type = CANON_TO_ADDICTION.get(canon)
            lvl = int((addiction_levels or {}).get(add_type, 0))
            score = min(1.0, score + 0.06 * max(0, lvl - 1))

            by_canon[canon] = max(by_canon.get(canon, 0.0), score)

        # 2) Fold text/NER/object via extractor (same as your previous flow)
        text_tags = self._extractor.derive_observed_tags(
            scene_text=scene_text,
            ner_entities=ner_entities,
            object_detections=object_detections,
            addiction_levels=addiction_levels,
            min_conf=self._min_conf,
            max_tags=self._max_tags,
        )
        for canon in text_tags:
            by_canon[canon] = max(by_canon.get(canon, 0.0), 0.60)

        # 3) Pick winners, trim, and cache
        winners = {c for c, sc in sorted(by_canon.items(), key=lambda kv: kv[1], reverse=True)
                   if sc >= self._min_conf}
        winners = set(list(winners)[: self._max_tags])

        # prune old events opportunistically
        self._events = [ev for ev in self._events if ev.ts >= cutoff]

        self._cache[turn_index] = (set(winners), now)
        return winners
