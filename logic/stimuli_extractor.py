# logic/stimuli_extractor.py
"""
Stimuli extractor for slice-of-life scenes.

Goal:
- From neutral scene text and optional detections, derive lightweight `stimuli`
  tags (e.g., "sandals", "barefoot", "perfume") to hint micro-cravings.

No hard deps. Optionally uses spaCy if present. Falls back to regex + difflib.

Returns a set of short tokens that match AddictionTriggerConfig.stimuli_affinity.

Example:
    extractor = get_default_extractor()
    stimuli = extractor.derive_observed_tags(
        scene_text="You pay the cashier. Her sandals slap tile; a sweet perfume trails.",
        object_detections=[{"label": "person"}, {"label": "sandals"}],
        addiction_levels={"feet": 2, "scent": 1},
    )
    # => {"sandals", "perfume"}
"""

from __future__ import annotations
import re
import difflib
from typing import Dict, Iterable, List, Optional, Set, Tuple

_WORD = re.compile(r"[A-Za-z][A-Za-z'_-]+")
_LOWER = lambda s: (s or "").strip().lower()

# ---- Canonical vocabulary that maps directly to AddictionTriggerConfig.stimuli_affinity
VOCAB: Dict[str, Set[str]] = {
    # feet-adjacent
    "sandals": {"sandal", "sandals", "slides", "flipflops", "flip-flops"},
    "flipflops": {"flipflop", "flipflops", "flip-flops"},
    "heels": {"heels", "stilettos", "pumps", "platforms"},
    "barefoot": {"barefoot", "bare foot", "bare-feet", "bare feet"},
    "toes": {"toe", "toes", "painted toes", "polished toes"},
    "ankle": {"ankle", "ankles", "anklet"},
    # socks/stockings
    "socks": {"sock", "socks", "ankle socks", "crew socks"},
    "knee_highs": {"knee-highs", "knee highs", "knee-high"},
    "thigh_highs": {"thigh-highs", "thigh highs", "overknees"},
    "stockings": {"stocking", "stockings", "nylons", "pantyhose"},
    # scent-adjacent
    "perfume": {"perfume", "eau de parfum", "eau de toilette", "cologne", "fragrance", "scent"},
    "musk": {"musk", "musky"},
    "sweat": {"sweat", "sweaty", "perspiration"},
    "locker": {"locker room", "locker", "gym locker"},
    "gym": {"gym", "workout", "treadmill", "weight room"},
    "laundry": {"laundry", "hamper", "dirty clothes"},
    # ass/humiliation/submission cues (lightweight)
    "hips": {"hips", "hip"},
    "ass": {"ass", "butt", "rear", "booty"},
    "shorts": {"shorts", "hotpants"},
    "tight_skirt": {"tight skirt", "pencil skirt", "bodycon"},
    "snicker": {"snicker", "snickering", "snide"},
    "laugh": {"laugh", "laughs", "laughing", "giggle", "giggles"},
    "eye_roll": {"eye roll", "eye-roll", "rolls her eyes"},
    "dismissive": {"dismissive", "dismisses you", "scornful", "derisive"},
    "order": {"order", "orders", "command", "commands"},
    "command": {"command", "commands"},
    "kneel": {"kneel", "kneels"},
    "obedience": {"obedience", "obedient", "obey", "obeys"},
}

# Inverse index for quick lookup (token -> canonical)
INV: Dict[str, str] = {}
for canon, variants in VOCAB.items():
    for v in variants:
        INV[_LOWER(v)] = canon
# Provide direct canonical self-maps too
for canon in list(VOCAB.keys()):
    INV.setdefault(canon, canon)

# Simple, readable thresholds
TEXT_HARD_HIT = 0.60         # exact/regex match
TEXT_FUZZY_HIT = 0.42        # fuzzy match via difflib
VISION_HIT = 0.80            # object detection label match
NER_BONUS = 0.10             # minor boost if tagged entity aligns (when provided)
SEVERITY_BONUS = 0.08        # per level >= 2 for the mapped addiction type (mild bias)

# Map canonical stimuli -> target addiction type(s)
# (should mirror AddictionTriggerConfig.stimuli_affinity keys you already use)
CANON_TO_ADDICTION: Dict[str, str] = {
    "sandals": "feet", "flipflops": "feet", "heels": "feet", "barefoot": "feet", "toes": "feet", "ankle": "feet",
    "socks": "socks", "knee_highs": "socks", "thigh_highs": "socks", "stockings": "socks",
    "perfume": "scent", "musk": "scent", "sweat": "scent", "locker": "scent", "gym": "scent", "laundry": "scent",
    "hips": "ass", "ass": "ass", "shorts": "ass", "tight_skirt": "ass",
    "snicker": "humiliation", "laugh": "humiliation", "eye_roll": "humiliation", "dismissive": "humiliation",
    "order": "submission", "command": "submission", "kneel": "submission", "obedience": "submission",
}


def _tokenize(text: str) -> List[str]:
    return [_LOWER(m.group(0)) for m in _WORD.finditer(text or "")]


def _score_hit(canon: str, source: str, base: float, addiction_levels: Optional[Dict[str, int]]) -> float:
    """Apply small bias toward higher current addiction severity."""
    add_type = CANON_TO_ADDICTION.get(canon)
    lvl = int((addiction_levels or {}).get(add_type, 0))
    bonus = SEVERITY_BONUS * max(0, lvl - 1)  # levels 2..4 add up to +0.24 max
    return min(1.0, base + bonus)


class StimuliExtractor:
    def __init__(self, vocab: Dict[str, Set[str]] = VOCAB):
        self.vocab = vocab
        self.inv = INV

        # Optional spaCy (graceful if missing)
        self._nlp = None
        try:
            import spacy  # type: ignore
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = spacy.blank("en")
                if "lemmatizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("lemmatizer", config={"mode": "rule"})
        except Exception:
            self._nlp = None

    # ---- public -----------------------------------------------------

    def derive_observed_tags(
        self,
        scene_text: Optional[str] = None,
        ner_entities: Optional[List[Tuple[str, str]]] = None,  # [(text, label)]
        object_detections: Optional[List[Dict[str, str]]] = None,  # [{"label": "sandals"}]
        addiction_levels: Optional[Dict[str, int]] = None,
        min_conf: float = 0.35,
        max_tags: int = 6,
    ) -> Set[str]:
        """
        Returns a set of canonical stimuli tags, e.g. {"sandals","perfume"}.
        """
        scores: Dict[str, float] = {}

        # 1) Text scan
        if scene_text:
            self._accumulate_text(scene_text, scores, addiction_levels)

        # 2) NER hints (tiny bump)
        if ner_entities:
            for text, label in ner_entities:
                canon = self._to_canon(text)
                if canon:
                    scores[canon] = max(scores.get(canon, 0.0), _score_hit(canon, "ner", TEXT_HARD_HIT - 0.15 + NER_BONUS, addiction_levels))

        # 3) Vision / object detections
        if object_detections:
            for det in object_detections:
                canon = self._to_canon(det.get("label", ""))
                if canon:
                    scores[canon] = max(scores.get(canon, 0.0), _score_hit(canon, "vision", VISION_HIT, addiction_levels))

        # Filter by min_conf and trim
        picked = [c for c, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if sc >= min_conf]
        return set(picked[:max_tags])

    # ---- internals --------------------------------------------------

    def _accumulate_text(self, text: str, scores: Dict[str, float], addiction_levels: Optional[Dict[str, int]]):
        toks = _tokenize(text)

        # exact/phrase hits
        joined = " " .join(toks)
        for canon, variants in self.vocab.items():
            for phrase in variants:
                p = _LOWER(phrase)
                # word-boundary regex for multiword phrases
                if " " in p or "-" in p:
                    pat = r"\b" + re.escape(p).replace(r"\ ", r"\s+") + r"\b"
                    if re.search(pat, joined):
                        scores[canon] = max(scores.get(canon, 0.0), _score_hit(canon, "text", TEXT_HARD_HIT, addiction_levels))
                else:
                    if p in toks:
                        scores[canon] = max(scores.get(canon, 0.0), _score_hit(canon, "text", TEXT_HARD_HIT, addiction_levels))

        # fuzzy single-token matches for near-misses (“flipflop” vs “flipflops”)
        flat_variants = list(self.inv.keys())
        for t in toks:
            near = difflib.get_close_matches(t, flat_variants, n=1, cutoff=0.88)
            if near:
                canon = self.inv.get(near[0])
                if canon:
                    scores[canon] = max(scores.get(canon, 0.0), _score_hit(canon, "fuzzy", TEXT_FUZZY_HIT, addiction_levels))

    def _to_canon(self, text: str) -> Optional[str]:
        t = _LOWER(text)
        if not t:
            return None
        if t in self.inv:
            return self.inv[t]
        # minimal fuzzy fallback
        near = difflib.get_close_matches(t, list(self.inv.keys()), n=1, cutoff=0.88)
        return self.inv.get(near[0]) if near else None


# convenience factory
def get_default_extractor() -> StimuliExtractor:
    return StimuliExtractor()
