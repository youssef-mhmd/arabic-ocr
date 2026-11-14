"""Reading order: put detected lines in the order a person reads them.

Arabic pages are read right to left, top to bottom; columns right to left.

How columns are found:

1. **Find the gutter.** For every x position across the page, count how many
   lines cross it. In a two-column page, the gap between the columns is crossed by
   almost no lines (only by the odd heading, footnote or page number). We take the
   widest band where that count is lowest, and accept it as a gutter if:
   - few lines cross it (at most 15% of them),
   - it is at least half a line height wide,
   - there is text on both sides of it, *side by side* (many lines on one side
     are at the same height as a line on the other).
   Very short items such as page numbers are ignored while looking for it.
2. **Read around it.** Lines that cross the gutter ("spanning" lines: a heading,
   footnotes) are read where they are, top to bottom. Between two spanning lines,
   read everything right of the gutter, then everything left of it. Each column
   is checked for a gutter of its own (three-column pages).
3. **Inside a single column**, read top to bottom; lines side by side on the same
   row (rare, e.g. a short label next to text) right to left.

Consecutive lines with only a normal line gap between them form a **block**
(roughly a paragraph); a big gap or a column change starts a new block.

Works on axis-aligned boxes (x0, y0, x1, y1); pages are deskewed first.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Box = tuple[float, float, float, float]  # x0, y0, x1, y1

#: A gutter may be crossed by at most this share of the lines.
MAX_CROSSING = 0.15
#: ... and must be at least this many line heights wide.
MIN_GUTTER = 0.5
#: Share of lines on the smaller side that must have a neighbour across the gutter.
MIN_BESIDE = 0.3


def _rows(boxes: Sequence[Box], idx: list[int]) -> list[int]:
    """Top to bottom; lines that share a row (overlap vertically by more than half
    the smaller height) right to left."""
    idx = sorted(idx, key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows: list[list[int]] = []
    for i in idx:
        y0, y1 = boxes[i][1], boxes[i][3]
        if rows:
            last = rows[-1]
            ly0, ly1 = min(boxes[j][1] for j in last), max(boxes[j][3] for j in last)
            if min(y1, ly1) - max(y0, ly0) > 0.5 * min(y1 - y0, ly1 - ly0):
                last.append(i)
                continue
        rows.append([i])
    return [i for row in rows for i in sorted(row, key=lambda i: -boxes[i][2])]


def _split_blocks(boxes: Sequence[Box], ordered: list[int], h: float) -> list[list[int]]:
    """Start a new block wherever the vertical gap to the previous line is big."""
    blocks: list[list[int]] = []
    for i in ordered:
        if blocks and boxes[i][1] - boxes[blocks[-1][-1]][3] < 1.0 * h:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    return blocks


def find_gutter(boxes: Sequence[Box], idx: list[int], h: float) -> float | None:
    """x of the gutter between two columns of these lines, or None.

    Very short items (page numbers, bullets: under two line heights wide) are
    ignored here, since one can sit right in the middle of a gutter.
    """
    idx = [i for i in idx if boxes[i][2] - boxes[i][0] >= 2 * h]
    if len(idx) < 2:
        return None
    x0 = np.array([boxes[i][0] for i in idx])
    x1 = np.array([boxes[i][2] for i in idx])
    lo, hi = int(np.floor(x0.min())), int(np.ceil(x1.max()))
    if hi - lo < 3:
        return None
    # crossing[x - lo] = number of lines covering pixel column x
    diff = np.zeros(hi - lo + 2, np.int32)
    np.add.at(diff, np.floor(x0).astype(int) - lo, 1)
    np.add.at(diff, np.ceil(x1).astype(int) - lo, -1)
    crossing = np.cumsum(diff)[: hi - lo]

    # Only positions with text entirely on both sides can be gutters.
    xs = np.arange(lo, hi)
    has_right = np.array([(x0 >= x).any() for x in xs])
    has_left = np.array([(x1 <= x).any() for x in xs])
    inside = has_left & has_right
    if not inside.any():
        return None
    best = crossing[inside].min()
    if best > MAX_CROSSING * len(idx):
        return None

    # Widest run of positions at that lowest count.
    ok = inside & (crossing == best)
    runs, start = [], None
    for k, v in enumerate([*ok, False]):
        if v and start is None:
            start = k
        elif not v and start is not None:
            runs.append((start, k))
            start = None
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if b - a < MIN_GUTTER * h:
        return None
    x = lo + (a + b) / 2

    # The two sides must be side by side: a good share of the lines on the
    # smaller side are at the same height as some line on the other side.
    right = [i for i in idx if boxes[i][0] >= x]
    left = [i for i in idx if boxes[i][2] <= x]
    small, big = sorted((right, left), key=len)
    beside = sum(
        any(min(boxes[i][3], boxes[j][3]) > max(boxes[i][1], boxes[j][1]) for j in big)
        for i in small
    )
    if beside < MIN_BESIDE * len(small):
        return None
    return x


def _order(boxes: Sequence[Box], idx: list[int], h: float) -> list[list[int]]:
    if not idx:
        return []
    x = find_gutter(boxes, idx, h)
    if x is None:
        return _split_blocks(boxes, _rows(boxes, idx), h)

    spanning = sorted(
        (i for i in idx if boxes[i][0] < x < boxes[i][2]),
        key=lambda i: (boxes[i][1] + boxes[i][3]) / 2,
    )
    out: list[list[int]] = []
    pending: list[int] = []  # spanning lines waiting to be output (as blocks)
    top = -np.inf
    for cut in [*spanning, None]:
        bottom = np.inf if cut is None else (boxes[cut][1] + boxes[cut][3]) / 2
        section = [
            i
            for i in idx
            if not (boxes[i][0] < x < boxes[i][2])
            and top <= (boxes[i][1] + boxes[i][3]) / 2 < bottom
        ]
        if section:
            if pending:
                out += _split_blocks(boxes, pending, h)
                pending = []
            right = [i for i in section if boxes[i][0] >= x]
            left = [i for i in section if boxes[i][0] < x]
            out += _order(boxes, right, h) + _order(boxes, left, h)
        if cut is not None:
            pending.append(cut)
            top = bottom
    if pending:
        out += _split_blocks(boxes, pending, h)
    return out


def reading_order(boxes: Sequence[Box]) -> list[list[int]]:
    """Group line boxes into blocks and order them for reading.

    Returns a list of blocks; each block is a list of indices into `boxes`, in
    reading order. Every index appears exactly once.
    """
    if not boxes:
        return []
    idx = list(range(len(boxes)))
    h = float(np.median([b[3] - b[1] for b in boxes]))
    return _order(boxes, idx, h)
