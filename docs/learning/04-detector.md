# 04: Finding the lines: the text detector

The recognizer (note 03) reads one line at a time. Before it can, something must find where every line is on a page. That is the **text detector**, the first neural network in the pipeline.

*Status: detector-v1 trained 2026-11-14 (results in §6).*

---

## 1. The job

Input: a whole page (a scan, a photo of a printout, a screenshot).
Output: one box per text line, in page coordinates.

The boxes are 4-point polygons, not just upright rectangles, so a slightly rotated scan still gets tight boxes. The page is also straightened first ("deskew", in preprocessing), so lines are close to horizontal.

---

## 2. Painting instead of boxing

There are two families of detectors:
- **box regressors** (YOLO, Faster R-CNN): the network outputs box coordinates directly;
- **segmentation detectors** (DBNet, PSENet, CRAFT): the network paints a map, "is this pixel text?", and boxes are traced around the painted regions.

Text lines are long, thin and touch each other, which box regressors handle badly. Almost every modern text detector is a segmentation detector; ours is **DBNet-style**.

### The core trick

If the network painted whole lines, two lines whose diacritics touch would come out as one blob. So it paints only each line's **core**: a thin band through the middle, 40% of the line's height (`KERNEL_FRACTION` in `src/arabic_ocr/detect/boxes.py`). Cores of neighbouring lines never touch. After the network has painted, each separate core region is traced (OpenCV's `findContours` + `minAreaRect`) and grown back to the full line size, the exact inverse of how the core was shrunk.

```
 full line box     ┌────────────────────────────┐
                   │   ·  ِ   ·        ّ     ·     │
 core (40%)        │ ══════════════════════════ │   <- what the network paints
                   │       ٍ           ·          │
                   └────────────────────────────┘
```

---

## 3. The network

`training/detector_model.py`, about 3.6 million weights (14 MB; PRD budget 25 MB).

1. **Backbone**: residual CNN blocks (as in ResNet) shrink the page to 1/4, 1/8, 1/16 and 1/32 of its size. Small scales see fine detail (dots, strokes); coarse scales see a whole line or column.
2. **Feature pyramid (FPN)**: mixes the scales back together, so each pixel's decision uses detail *and* context.
3. **Head**: upsamples back to full page size and gives one score per pixel.

It is trained **from scratch**. Most detectors start from a backbone pretrained on ImageNet (photos), but those weights come with their own license terms, and our synthetic pages are plentiful.

---

## 4. Training data: synthetic pages

`training/page.py` + `training/make_pages.py` lay out real corpus text with our renderer like a word processor would:
- **book** pages: justified paragraphs, headers, footnotes under a rule, page numbers;
- **two-column** pages (the first column on the right);
- **screen** pages: titles, menus and paragraphs in coloured or dark-mode blocks.

Every line's box and text are recorded, so labels are free and exact. Print pages then get paper texture, shadows, blur, noise, black-and-white thresholding, JPEG and a slight rotation.

Each training example is a random 640×640 window of a page, zoomed randomly so the detector sees text from tiny to large. Lines too small to read are marked "ignore": the loss doesn't count them either way.

### The loss
- **Balanced BCE**: for each pixel, "core or not?". Most pixels are background, so only the 3 hardest background pixels per core pixel are counted ("hard negative mining"). Otherwise predicting "no text anywhere" would already score well.
- **Dice**: measures how well the predicted core *regions* overlap the true ones as a whole, which keeps thin cores from fading away.

---

## 5. Measuring it

`training/detect_eval.py`:
- **IoU** (intersection over union): overlap of a predicted box and a true box, from 0 to 1. A line counts as found when IoU ≥ 0.5.
- **Recall**: the share of true lines that were found. M3 target: ≥ 95%. A missed line is text lost forever.
- **Precision**: the share of predicted boxes that are real lines. A false box usually just produces a little garbage text.
- **F1**: one number combining both.

---

## 6. Results: detector-v1

One training run: `training/configs/detector_v1.yaml`, 12 passes over 30,000 synthetic pages (720,000 crops of 640×640), about 2.2 hours on the RTX 5060 at ~93 crops/s. Exported to `models/detector-v1.onnx` (13.9 MB, under the 25 MB target); ONNX and PyTorch outputs agree to 2e-6.

The validation score barely moved after the 3rd pass (F1 97.8% → 97.9%), so a shorter run would have done almost as well. Worth remembering for the next detector run.

Test pages (500, ONNX on the CPU):

| page kind | precision | recall | F1 |
|---|---|---|---|
| book | 99.5% | 99.6% | 99.6% |
| two columns | 99.9% | 99.9% | 99.9% |
| screen | 99.0% | 97.8% | 98.4% |
| **all** | **99.6%** | **99.4%** | **99.5%** |

Recall target ≥ 95%: **pass** (on synthetic pages; real pages still to come).

### A lesson: check the labels, not just the model

The first evaluation said precision was only 97% and the end-to-end error rate was 3.7%. Looking at one page with its true boxes (green) and the predicted boxes (red) drawn on it showed the reason. Many lines at the bottom of columns had **no true box at all**, although they were clearly printed. The page generator drew a whole paragraph and *afterwards* deleted the labels of lines that went past the bottom margin. The ink stayed on the page, so on ~80% of book pages some real text was unlabelled.

That did two things:
1. **Evaluation** counted the detector's correct boxes on those lines as false alarms.
2. **Training** told the detector "this is not text" for some real text. It learned well anyway, because most lines were labelled, but it sometimes broke such lines into pieces.

After fixing the generator (`training/page.py`: a line that would cross the bottom is never drawn), the test pages were re-made as `pages_v2`, and the same detector-v1 scored 99.6% precision and 99.4% recall. Most of the "errors" had been in the labels.

General rule: when a score looks wrong, **draw the labels and predictions on a few images and look at them** before changing the model.
