"""JSON output.

Schema (version 1):

    {
      "schema": "arabic-ocr/1",
      "file": "scan.png",
      "pages": [
        {
          "index": 0, "width": 2480, "height": 3508, "angle": -0.7, "seconds": 1.2,
          "text": "...",
          "blocks": [
            {
              "box": [[x, y], [x, y], [x, y], [x, y]],
              "text": "...",
              "lines": [
                {
                  "box": [[x, y], ...], "text": "...", "confidence": 0.98,
                  "words": [{"box": [[x, y], ...], "text": "...", "confidence": 0.99}]
                }
              ]
            }
          ]
        }
      ]
    }

- Every box is 4 corner points in pixels of the input image, in the order
  top-left, top-right, bottom-right, bottom-left (it may be rotated a little).
- Blocks and lines are in reading order. Words are in reading order too (right to
  left); every text value is in logical Unicode order.
- `confidence` is 0-1 (average probability of the recognized characters).
- `angle` is the deskew rotation the engine applied, in degrees.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from arabic_ocr.pipeline import Page

SCHEMA = "arabic-ocr/1"


def _box(quad: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in quad]


def page_to_dict(page: Page, index: int = 0) -> dict:
    return {
        "index": index,
        "width": page.width,
        "height": page.height,
        "angle": page.angle,
        "seconds": round(page.seconds, 3),
        "text": page.text,
        "blocks": [
            {
                "box": _box(block.box),
                "text": block.text,
                "lines": [
                    {
                        "box": _box(line.box),
                        "text": line.text,
                        "confidence": round(line.confidence, 4),
                        "words": [
                            {
                                "box": _box(w.box),
                                "text": w.text,
                                "confidence": round(w.confidence, 4),
                            }
                            for w in line.words
                        ],
                    }
                    for line in block.lines
                ],
            }
            for block in page.blocks
        ],
    }


def document_to_json(pages: list[Page], file: str | Path | None = None) -> str:
    doc = {
        "schema": SCHEMA,
        "file": str(file) if file is not None else None,
        "pages": [page_to_dict(p, i) for i, p in enumerate(pages)],
    }
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    # Keep each [x, y] point on one line instead of three.
    return _POINT.sub(lambda m: f"[{m.group(1)}, {m.group(2)}]", text)


_POINT = re.compile(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]")
