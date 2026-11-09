# 03: The line recognizer and CTC

This note explains the model that reads one line image and outputs its text: how it's built, how CTC lets it learn without knowing where each letter is, and what happens during training.

*Status: written during M2. Results will be added after the first full training run.*

**First test run** (300,000 images, 11 minutes on the RTX 5060 laptop): validation CER went 16.6% -> 8.6% -> 5.0% (4.3% with diacritics ignored), and 61% of lines were exactly right. Training speed was about 550 images/second.

---

## 1. The job

Input: a grayscale image of one text line, 64 pixels tall, any width.
Output: the text, in logical Unicode order.

We never cut the line into letters. Arabic letters join, overlap and change shape, so "where does one letter end?" often has no clear answer. Instead the model reads the whole line and learns the alignment itself, using CTC.

---

## 2. The model: CNN + BiLSTM + CTC head ("CRNN")

`training/model.py`, about 4.5 million weights (18 MB).

```
image 64 x W  ──►  CNN  ──►  sequence of W/4 feature vectors  ──►  BiLSTM  ──►  scores per slice
```

**1. CNN (convolutional neural network).** A stack of small filters slid across the image. Early layers respond to edges and strokes; deeper layers combine those into dots, loops and letter parts. Pooling layers shrink the image step by step: the height goes from 64 to 1, and the width shrinks by 4. The result is one feature vector (256 numbers) for every 4-pixel-wide vertical **slice** of the line.

**2. BiLSTM (bidirectional long short-term memory).** A recurrent network that reads the slices in order while keeping a memory, once in each direction. After it, each slice "knows" what is around it. This matters for Arabic: the same stroke can be part of different letters depending on its neighbours, and a diacritic belongs to the letter beneath it.

**3. Linear head.** Turns each slice's features into **122 scores**: one for each of our 121 symbols, plus the **blank**.

### Reading right to left
The model mirrors the image internally, so slice 1 is the *right* edge of the line, the way you read Arabic. Its target text is therefore in **reading order**: logical order, except that numbers come out reversed, because scanning right to left you meet `2026` as `6 2 0 2`. `arabic_ocr.bidi.reading_order` produces these targets, and the same function turns the model's output back into normal logical text. (It round-trips exactly for all 1,020,000 labels in our dataset.) The stored and returned text is always in logical order.

---

## 3. CTC: learning without letter positions

We know the text of each line, but not which slices each letter covers. **CTC (Connectionist Temporal Classification)** solves this.

The model outputs one symbol (or blank) per slice. A line of 70 slices might come out as:

```
_ _ س س _ ل ل ل _ _ ا ا _ م _ _ ...      (_ = blank)
```

CTC's rule for turning that into text: **merge repeated symbols, then remove blanks.** That gives `سلام`. Many different slice sequences collapse to the same text: the letter might cover 2 slices or 5, with blanks before or after.

**The CTC loss** adds up the probability of *every* slice sequence that collapses to the correct text, and training pushes that total up. The model is never told where `س` is; it finds out, because putting `س` on the right slices is the only way to make the correct text likely.

**Why the blank matters:** doubled letters. To output `لل`, the model must emit `ل _ ل`; without a blank between them, `ل ل` would merge into one `ل`. So a line needs at least one slice per character, plus one per doubled letter. The data loader skips any image too narrow for its text.

**Decoding** (`arabic_ocr/recognize/ctc.py`): take the best symbol per slice, merge repeats, drop blanks. This "greedy" decoding is simple and usually almost as good as smarter search ("beam search"), which may come later.

---

## 4. Training: what happens each step

`training/train_recognizer.py`, settings in `training/configs/recognizer_v1.yaml`.

1. **Batch:** about 100 line images of similar width (padding everything to the widest line wastes less work when widths are similar).
2. **Forward pass:** the model produces scores for every slice.
3. **CTC loss:** how unlikely the correct text is. At the start, about 3.5; a well-trained model gets close to 0.
4. **Backward pass + AdamW:** compute how each weight should change, and nudge it.

Settings worth knowing:
- **Learning rate schedule:** starts tiny, ramps up over the first 200,000 images ("warm-up"; large updates on a random model are unstable), then slowly decays along a cosine curve so the final steps are fine adjustments.
- **Mixed precision (bf16):** most math runs in 16-bit numbers, about twice as fast on RTX GPUs; the loss itself stays in 32-bit for accuracy.
- **Gradient clipping:** if a batch asks for a huge change, it's scaled down to a maximum size, so one strange batch can't wreck the model.
- **Skipping non-finite steps:** in our first test run, one batch produced a "not a number" (NaN) value, and because nothing stopped it, every weight became NaN and the model was destroyed. Now any step whose loss or gradient isn't a finite number is skipped and counted, and the first bad batch is saved for investigation. In the next test it happened in 2 of about 2,650 steps (0.08%). Replaying the saved batch gave normal, finite results, so it's a rare numerical hiccup of the GPU math at one particular moment, not bad data. Skipping those steps is the standard remedy.
- **Validation:** every 250,000 images, measure CER on the 10,000 validation lines (never trained on) and keep the best model as `best.pt`.
- **Checkpoints and resume:** `last.pt` holds the weights, optimizer state and position. Stop with Ctrl+C at any time; the same command resumes.

**Reading the progress:**
- loss going down and validation CER going down: learning;
- training loss down but validation CER stuck or rising: overfitting (unlikely with 1M varied synthetic images);
- loss stuck near 3.5: not learning; check the data and the learning rate.
