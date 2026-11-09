"""Text direction (bidi) for Arabic-script lines.

Arabic is written right to left, but numbers inside it run left to right. Two things
in this project depend on exactly which characters are part of a left-to-right number:

- the training renderer, which must lay lines out the way browsers and word
  processors do (training/render.py);
- the recognizer, which reads a line image from right to left and therefore meets
  the digits of a number in reverse (see `reading_order`).

Text is always *stored and output* in logical order (the order it is typed). Nothing
here changes that; `reading_order` is only an internal training/decoding target and
is converted straight back.
"""

from __future__ import annotations

import unicodedata

# This is the Unicode Bidirectional Algorithm (UAX #9), reduced to what our charset
# can contain: a right-to-left paragraph, Arabic letters (type AL), no Latin letters
# (no type L), and no explicit direction controls. Under those conditions the full
# algorithm gives exactly this result:
#   - numbers end up left-to-right (level 2); everything else right-to-left (level 1).
# The W-rules below decide which characters count as part of a number. Checked
# against the python-bidi reference implementation on 20,000 random strings.


def _bidi_type(ch: str) -> str:
    return unicodedata.bidirectional(ch) or "ON"


def ltr_mask(text: str) -> list[bool]:
    """For each character, True if it is drawn left-to-right (part of a number)."""
    types = [_bidi_type(c) for c in text]
    n = len(types)

    # W1: a combining mark takes the type of the character before it.
    for i, t in enumerate(types):
        if t == "NSM":
            types[i] = types[i - 1] if i > 0 else "R"

    # W2: a European number after Arabic letters becomes an Arabic number.
    # W3: Arabic letters (AL) are then treated as plain right-to-left (R).
    last_strong = "R"  # start of a right-to-left paragraph
    for i, t in enumerate(types):
        if t in ("R", "L", "AL"):
            last_strong = t
        elif t == "EN" and last_strong == "AL":
            types[i] = "AN"
    types = ["R" if t == "AL" else t for t in types]

    # W4: one separator between two numbers of the same kind joins them.
    #     + and - (ES) join only European numbers; , . : / (CS) join either kind.
    for i in range(1, n - 1):
        left, mid, right = types[i - 1], types[i], types[i + 1]
        if mid == "ES" and left == right == "EN":
            types[i] = "EN"
        elif mid == "CS" and left == right and left in ("EN", "AN"):
            types[i] = left

    # W5: % signs (ET) next to a European number become part of it.
    i = 0
    while i < n:
        if types[i] == "ET":
            j = i
            while j < n and types[j] == "ET":
                j += 1
            if (i > 0 and types[i - 1] == "EN") or (j < n and types[j] == "EN"):
                types[i:j] = ["EN"] * (j - i)
            i = j
        else:
            i += 1

    # W6, W7, N1, N2, I2: with no left-to-right letters anywhere, every remaining
    # neutral resolves to right-to-left, and numbers (EN/AN) get level 2 (LTR).
    return [t in ("EN", "AN") for t in types]


def reading_order(text: str) -> str:
    """The characters in the order you meet them scanning the line **right to left**.

    That is logical order, except that each left-to-right number is reversed:
    "سنة 2026" -> "سنة 6202". It equals the reverse of the display order.

    The recognizer reads images right to left, so it learns to output text in this
    order. For real text the function is its own inverse (reversing a number twice
    restores it), so the recognizer's output is turned back into logical text with
    this same function:

        reading_order(reading_order(text)) == text

    This held for all 1,020,000 labels of the v1 dataset. It can fail only for
    unnatural strings, e.g. Persian and Arabic digits mixed inside one number or a
    diacritic placed on a digit, where reversing the number changes its neighbours.
    """
    out: list[str] = []
    run: list[str] = []
    for ch, ltr in zip(text, ltr_mask(text), strict=True):
        if ltr:
            run.append(ch)
        else:
            out.extend(reversed(run))
            run = []
            out.append(ch)
    out.extend(reversed(run))
    return "".join(out)


#: Alias that reads better when decoding model output.
logical_order = reading_order
