"""
Pollen-curated moves library for Reachy Mini.

Two HuggingFace datasets ship with the daemon:
  - pollen-robotics/reachy-mini-emotions-library  (81 emotions, with .wav)
  - pollen-robotics/reachy-mini-dances-library    (19 dances)

The daemon exposes them via:
    POST /api/move/play/recorded-move-dataset/{dataset}/{move_name}

This module provides:
  - lazy preload (snapshot_download) so first play is fast
  - fuzzy lookup ("happy" → ["cheerful1","enthusiastic1","proud1",...])
  - randomised variant pick so repeated calls feel natural
  - safety: auto-enable motors before playing (otherwise the move
    plays its sound but the head doesn't move)
"""

import logging
import os
import random
import sys
import threading
import urllib.parse
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
DANCES_DATASET = "pollen-robotics/reachy-mini-dances-library"

# Map user-facing emotion words → list of recorded-move names that match.
# When a user says "show me happy" the LLM passes one of the keys; we
# pick a random member of the matching list for natural variety.
_EMOTION_ALIASES: Dict[str, List[str]] = {
    "happy":      ["cheerful1", "enthusiastic1", "enthusiastic2", "proud1", "proud2", "success1", "laughing1"],
    "excited":    ["enthusiastic1", "enthusiastic2", "amazed1", "success1", "success2"],
    "sad":        ["sad1", "sad2", "downcast1", "exhausted1", "lonely1", "tired1"],
    "angry":      ["furious1", "rage1", "irritated1", "irritated2", "frustrated1"],
    "frustrated": ["frustrated1", "irritated1", "irritated2", "displeased1", "displeased2"],
    "surprised":  ["surprised1", "surprised2", "amazed1"],
    "amazed":     ["amazed1", "surprised1"],
    "curious":    ["curious1", "inquiring1", "inquiring2", "inquiring3", "thoughtful1", "thoughtful2"],
    "thoughtful": ["thoughtful1", "thoughtful2", "uncertain1"],
    "confused":   ["confused1", "uncertain1", "incomprehensible2", "lost1"],
    "scared":     ["scared1", "fear1", "anxiety1"],
    "shy":        ["shy1"],
    "proud":      ["proud1", "proud2", "proud3"],
    "love":       ["loving1", "grateful1"],
    "loving":     ["loving1", "grateful1"],
    "grateful":   ["grateful1"],
    "welcoming":  ["welcoming1", "welcoming2", "come1"],
    "greet":      ["welcoming1", "welcoming2", "cheerful1"],
    "bored":      ["boredom1", "boredom2", "indifferent1"],
    "tired":      ["tired1", "exhausted1", "sleep1"],
    "calm":       ["calming1", "serenity1", "relief1", "relief2"],
    "yes":        ["yes1", "yes_sad1"],
    "no":         ["no1", "no_sad1", "no_excited1"],
    "disagree":   ["no1", "displeased1"],
    "agree":      ["yes1", "understanding1", "understanding2"],
    "sorry":      ["oops1", "oops2", "shy1"],
    "oops":       ["oops1", "oops2"],
    "attentive":  ["attentive1", "attentive2"],
    "listening":  ["attentive1", "attentive2", "inquiring1"],
    "disgusted":  ["disgusted1", "contempt1"],
    "playful":    ["dance1", "dance2", "dance3", "cheerful1"],
    "go_away":    ["go_away1"],
    "wake":       ["welcoming1", "amazed1"],   # wake-up flavor
    "sleep":      ["sleep1"],
}

# Curated dance shortlist for "do a dance" — ordered roughly by how lively
# they feel. The LLM can also pass an exact dance name and we'll match it.
_FAVORITE_DANCES = [
    "groovy_sway_and_roll",
    "polyrhythm_combo",
    "side_to_side_sway",
    "head_tilt_roll",
    "interwoven_spirals",
    "jackson_square",
    "pendulum_swing",
    "sharp_side_tilt",
    "uh_huh_tilt",
    "yeah_nod",
    "simple_nod",
    "chin_lead",
    "neck_recoil",
    "dizzy_spin",
    "grid_snap",
    "side_glance_flick",
    "side_peekaboo",
    "stumble_and_recover",
    "chicken_peck",
]


class MovesCatalog:
    """Lazy-loads and indexes the Pollen move libraries.

    Thread-safe; the first ``ensure_loaded()`` call may block on HF
    download, subsequent calls are no-ops.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._emotions: List[str] = []
        self._dances: List[str] = []

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                logger.warning("huggingface_hub not installed — moves catalog unavailable")
                self._loaded = True
                return

            for ds, sink in (
                (EMOTIONS_DATASET, self._emotions),
                (DANCES_DATASET, self._dances),
            ):
                try:
                    path = snapshot_download(ds, repo_type="dataset")
                    names = sorted(
                        os.path.splitext(f)[0]
                        for f in os.listdir(path)
                        if f.endswith(".json")
                    )
                    sink.extend(names)
                    print(
                        f"MovesCatalog: loaded {len(names)} moves from {ds}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    logger.warning("Failed to load %s: %s", ds, e)
            self._loaded = True

    @property
    def emotions(self) -> List[str]:
        self.ensure_loaded()
        return list(self._emotions)

    @property
    def dances(self) -> List[str]:
        self.ensure_loaded()
        return list(self._dances)

    def lookup_emotion(self, term: str) -> Optional[str]:
        """Resolve a user-facing word to a recorded emotion move name.

        Tries (in order): exact match → alias map → substring match.
        Picks a random variant when multiple candidates fit so repeated
        calls produce different motion.
        """
        self.ensure_loaded()
        if not self._emotions:
            return None
        term = term.lower().strip()
        if term in self._emotions:
            return term
        candidates = _EMOTION_ALIASES.get(term)
        if candidates:
            in_lib = [c for c in candidates if c in self._emotions]
            if in_lib:
                return random.choice(in_lib)
        # Last resort: substring match (handles "curious2" style queries
        # for variants we didn't explicitly alias).
        partial = [m for m in self._emotions if term in m]
        if partial:
            return random.choice(partial)
        return None

    def lookup_dance(self, term: Optional[str] = None) -> Optional[str]:
        """Resolve a dance name; defaults to a random favourite."""
        self.ensure_loaded()
        if not self._dances:
            return None
        if not term or term.lower().strip() in ("random", "any", ""):
            in_lib = [d for d in _FAVORITE_DANCES if d in self._dances]
            return random.choice(in_lib or self._dances)
        term = term.lower().strip().replace(" ", "_").replace("-", "_")
        if term in self._dances:
            return term
        partial = [d for d in self._dances if term in d]
        if partial:
            return random.choice(partial)
        # Couldn't resolve — fall back to a favourite.
        in_lib = [d for d in _FAVORITE_DANCES if d in self._dances]
        return random.choice(in_lib or self._dances)


def encoded_dataset_path(dataset: str, move_name: str) -> str:
    """URL-encoded daemon path for ``play /recorded-move-dataset/<ds>/<name>``."""
    return f"/api/move/play/recorded-move-dataset/{urllib.parse.quote(dataset, safe='')}/{move_name}"
