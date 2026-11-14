"""CTC decoding: turn the recognizer's per-column scores into text.

The recognizer looks at a line image as a sequence of thin vertical slices, read
right to left. For every slice it gives a score for each symbol in the charset, plus
the special "blank" (id 0) meaning "no new character here".

Greedy decoding (this file):
1. pick the highest-scoring symbol for every slice;
2. merge repeats ("سسسـ" over 3 slices is one س);
3. drop blanks. A blank between two equal symbols keeps them apart, which is how
   doubled letters survive step 2: "ل ل" is [ل, blank, ل].

The result is in *reading order* (right to left, so numbers come out reversed);
`arabic_ocr.bidi.logical_order` turns it into normal logical text.

Pure NumPy: the shipped runtime never imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arabic_ocr import charset
from arabic_ocr.bidi import logical_order


@dataclass
class Decoded:
    text: str  # logical Unicode order
    confidence: float  # 0-1: average probability of the emitted symbols
    ids: list[int]  # symbol ids in reading order (useful for debugging)
    probs: list[float] = field(default_factory=list)  # probability of each id
    frames: list[int] = field(default_factory=list)  # slice where each id was emitted


def greedy_path(scores: np.ndarray) -> tuple[list[int], list[float], list[int]]:
    """Best path through one line's scores, shape (T, num_classes).

    Returns the symbol ids, their probabilities, and the slice (time step) where
    each was first emitted. Slice 0 is the right edge of the line; each slice is
    `width_stride` pixels of the (64 px high) line image.

    `scores` may be log-probabilities or probabilities; only the ranking matters for
    the path, but confidences assume log-probabilities (natural log).
    """
    best = scores.argmax(axis=1)
    ids: list[int] = []
    probs: list[float] = []
    frames: list[int] = []
    prev = -1
    for t, i in enumerate(best):
        i = int(i)
        if i != prev and i != charset.BLANK_ID:
            ids.append(i)
            probs.append(float(np.exp(scores[t, i])))
            frames.append(t)
        prev = i
    return ids, probs, frames


def greedy_ids(scores: np.ndarray) -> tuple[list[int], list[float]]:
    """Like greedy_path, without the slice positions."""
    ids, probs, _ = greedy_path(scores)
    return ids, probs


def greedy_decode(scores: np.ndarray) -> Decoded:
    """Decode one line. `scores`: (T, num_classes) log-probabilities."""
    ids, probs, frames = greedy_path(scores)
    text = logical_order(charset.decode(ids))
    confidence = float(np.mean(probs)) if probs else 0.0
    return Decoded(text=text, confidence=confidence, ids=ids, probs=probs, frames=frames)


def greedy_decode_batch(scores: np.ndarray, lengths: np.ndarray | None = None) -> list[Decoded]:
    """Decode a batch. `scores`: (T, B, num_classes); `lengths`: valid T per line."""
    out = []
    for b in range(scores.shape[1]):
        t = scores.shape[0] if lengths is None else int(lengths[b])
        out.append(greedy_decode(scores[:t, b]))
    return out
