# 05: From page to text: the pipeline

Notes 03 and 04 cover the two neural networks. This note covers the plain code around them. It turns "an image file" into "text in reading order, with boxes". None of it is machine learning, but every step affects the final accuracy.

Code: `src/arabic_ocr/pipeline.py` and the modules it calls.

---

## 1. The steps

```
image file
  -> preprocess   (preprocess.py)    grayscale, straighten the page
  -> detect       (detect/)          one box per text line      [neural network 1]
  -> crop         (pipeline.py)      cut each line out, rotated flat
  -> recognize    (recognize/)       read each line with CTC    [neural network 2]
  -> order        (layout.py)        blocks and lines in reading order
  -> postprocess  (pipeline.py)      optional: drop diacritics, convert digits
  -> output       (export/, cli.py)  plain text or JSON
```

---

## 2. Preprocessing: straightening a tilted page

A scanned page is often rotated by a degree or two. To measure the tilt we use a **projection profile**:

1. Mark every dark ("ink") pixel.
2. Count the ink in every pixel row. That list of counts is the profile.
3. On a straight page, rows are either inside a line of text (lots of ink) or in the gap between lines (almost none), so the profile jumps up and down sharply. On a tilted page every row cuts across several lines and the profile is smeared and flat.
4. Try rotating by every angle from −10° to +10° and keep the angle with the sharpest profile. First try every whole degree, then every 0.1° around the best one.

Tilts under 0.2° are left alone, because rotating an image blurs it slightly.

Denoising (a median filter) and binarizing (Otsu's black/white threshold) exist but are **off** by default. The networks were trained on grayscale images with realistic noise, so these steps usually remove information the networks could have used. Turn them on (`--denoise`, `--binarize`) only for very poor scans.

---

## 3. Cropping lines: how much margin?

The detector gives a box that hugs the ink of a line. Before reading, we cut that box out of the full-resolution page, rotated so the line is horizontal, plus a small margin.

How big should the margin be? We measured it on 1,327 lines from synthetic test pages, using the *true* boxes so that only the margin changed:

| margin above/below (fraction of line height) | CER   | CER without diacritics |
|---|---|---|
| 0.00 | 1.13% | 0.94% |
| **0.05** (chosen) | **1.16%** | **0.90%** |
| 0.15 | 1.57% | 1.16% |
| 0.35 | 2.95% | 2.34% |

Bigger margins make things *worse*. On a real page the lines are close together, so a big margin pulls in the tops and bottoms of the neighbouring lines, and the recognizer tries to read them. The side margin (0.1 to 0.6 line heights) made almost no difference, so it is set to 0.2.

This is a typical example of **tuning on measurements, not on intuition**. The recognizer's docs said "add about a third of the height". That was true for isolated line images, but not for lines cut from a crowded page.

---

## 4. Where are the words? Using CTC positions

The recognizer reads a line as a sequence of thin vertical slices (note 03). Greedy CTC decoding emits each character at a specific slice. So we know *roughly* where each character is: slice number × 4 pixels (at the model's 64 px height), counted from the right edge.

To make word boxes, we split the line at the space characters. Each word's box runs from the space before it to the space after it. These boxes are approximate (CTC spikes are not perfectly centred on letters), but they are good enough for highlighting and search.

---

## 5. Reading order: finding the gutter

Arabic is read right to left, top to bottom; columns right to left.

1. **Find the gutter.** For every x position across the page, count how many lines cross it. Between two columns there is a band that almost no line crosses. We take the widest such band and accept it as a gutter if at most 15% of the lines cross it, it is at least half a line height wide, and there are lines on both sides *at the same height* (side by side, not one above the other). Tiny items like page numbers are ignored while looking, because a page number often sits right in the middle of the gutter.
2. **Read around it.** Lines that cross the gutter (a heading over both columns, footnotes) are read where they are, top to bottom. Between them, read the right column, then the left. Each column is checked for its own gutter, which handles three-column pages.
3. **Inside a single column**, read top to bottom.

A big vertical gap (more than one line height) starts a new **block**, which is roughly a paragraph. In plain-text output, blocks are separated by an empty line.

The first version cut the page into horizontal strips and joined strips that shared a gap. It failed on 62% of two-column pages, *even with perfect boxes*. When the paragraph breaks in the two columns fall at different heights, some strips hold one line from one column only, which breaks the chain. Measuring the layout step on its own with the true boxes, separately from the detector, found this quickly. The current version orders 495 of 500 test pages exactly right with true boxes; the 5 misses are two neighbouring lines swapped.

It does not understand tables, side notes, or poetry printed in two halves per row. Smarter layout analysis is M4 work.

---

## 6. End-to-end results (synthetic pages)

`training/eval_pages.py` runs the whole pipeline on synthetic test pages and compares with the true text. Missed lines count as all-wrong; invented lines count as extra errors. Results on 200 `pages_v2` test pages, detector-v1 + recognizer-v3:

| page kind | CER | CER without diacritics | reading order exactly right |
|---|---|---|---|
| book | 1.86% | 1.62% | 96.8% |
| two columns | 1.11% | 0.88% | 95.5% |
| screen | 1.62% | 1.54% | 96.8% |
| **all** | **1.58%** | **1.37%** | **96.5%** |

With the detector's boxes, margins measured again on 100 pages: 0.00 → 1.82% CER, 0.05 → 1.93%, 0.10 → 2.11%, 0.15 → 2.37%. 0.05 stays for now (a little room for real scans); to be re-checked on real pages.

The M3 exit criteria are measured on **real** pages, which still need to be collected and labelled.

---

## 6. Speed

`training/benchmark_speed.py` times every step. Synthetic A4 pages at 300 DPI (2480×3500 px), 4 CPU threads, detector-v1 + recognizer-v3:

- detection: ~0.33 s per page;
- recognition: 0.8–3.8 s, depending on the amount of text;
- **median 2.44 s per page** (target ≤ 3 s: pass); a dense two-column page with 119 lines took 4.3 s.

Almost all the time is the recognizer. Two ways to speed it up were measured:
- **Batching** (padding lines of similar width so several are read in one call): no accuracy change, but only ~10% faster. The work is pure computation, not per-call overhead.
- **Parallel lines** (each model call on one thread, 4 lines read at once): ~20% faster with identical results. This is what the recognizer now does.

The owner's laptop limited to 4 threads stands in for "a 4-core CPU". An older 4-core machine will be slower, so the benchmark should be repeated on one before release.
