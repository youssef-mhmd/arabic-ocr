# 02: Data — characters, labels, and synthetic images

A recognizer learns from pairs of **(image, label)**: a picture of a text line and the exact text it shows. This note explains how we define the possible characters, how we make labels consistent, and (as M1 continues) how we generate millions of images.

*Status: complete for M1.*

---

## 1. The character set: the model's alphabet

The recognizer is a **classifier**: at each position along the line it picks one symbol from a fixed list. That list is the **charset**, defined in `src/arabic_ocr/charset.py`. Today it has 121 symbols:

| Group | Examples |
|---|---|
| Arabic letters, including hamza forms | ء آ أ ؤ إ ئ ا ب … ي, plus ٱ |
| Persian / Urdu letters | پ چ ژ ک گ ی ٹ ڈ ڑ ں ھ ہ ے … |
| Diacritics (tashkeel) | ـَ ـُ ـِ ـً ـٌ ـٍ ـّ ـْ ـٰ, plus combining hamza ـٔ ـٕ |
| Tatweel | ـ |
| Digits | ٠–٩, ۰–۹, 0–9 |
| Punctuation | ، ؛ ؟ ٪ ٫ ٬ ۔ and . : ! " ( ) « » - / % … |
| Space | " " |

Each symbol gets an **id** (an integer). The model outputs ids and we convert them back to text. Turning text into ids is called **encoding**; the reverse is **decoding**.

**Id 0 is special: the "blank".** It isn't a character. The CTC method (note 03) needs an extra "nothing new here" option. So the model has 122 outputs (121 symbols + blank).

**Why the order of the list matters:** a trained model's output #47 *means* whatever symbol was #47 when it was trained. If we reorder the list later, an old model would print the wrong letters. So we only ever **append** new symbols.

**Why no Latin letters?** The PRD says Latin text is out of scope. A model can only output symbols in its charset, so Latin words in an image will come out as garbage or nothing. That's expected.

### Diacritics are separate symbols
In Unicode, `بَ` is two characters: `ب` + fatha. We keep it that way. The alternative, one symbol for every letter+mark combination, would explode the charset into thousands of rare classes the model barely sees. Separate symbols mean the model learns "fatha" once and reuses it on every letter.

### Logical order, never visual
Arabic is written right to left, but the text is **stored** in the order it's typed: the first letter typed is the first character in the string. This is called **logical order**. Screens draw it right to left automatically. We never reverse strings; if Arabic looks backwards in a terminal, the terminal is wrong, not the data.

---

## 2. Normalization: one look, one label

The same visible text can be typed with different Unicode characters. If the labels don't agree, the model gets contradictory lessons ("this picture is X" … "the same picture is Y") and learns slower and worse. **Normalization** rewrites text into one standard form. `charset.normalize` does four things:

1. **Presentation forms → base letters.** Unicode has legacy "pre-shaped" characters, like `ﻻ` (lam-alef ligature) or `ﺑ` (beh, initial form). Old PDFs and websites still contain them. We turn them into the normal letters (`ل` + `ا`, `ب`). The *shape* is the font's job, not the label's.
2. **Remove invisible characters.** Direction marks, zero-width joiners, and the byte-order mark don't appear in an image, so the model can never "see" them. We remove them from labels. (One trade-off: the Persian half-space ZWNJ, as in `کتاب‌ها`, is dropped too.)
3. **Unicode NFC.** This does two useful things:
   - It composes letters that were typed in two pieces: `ا` + combining hamza → `أ`.
   - It puts stacked diacritics in one fixed order. Typing shadda then fatha or fatha then shadda looks identical, and after NFC both become the same string.
4. **Whitespace.** Tabs, newlines, non-breaking spaces and runs of spaces all become one normal space.

What normalization **never** does: reverse text, remove diacritics, or swap one letter for another. `أ` stays `أ`; `ي` stays `ي`. Those would be lossy changes, and the PRD makes diacritic stripping an *output option*, not a training default.

A known ambiguity: Farsi yeh `ی` and alef maksura `ى` look identical at the end of a word, and Arabic `ك` vs Persian `ک` can look alike in some fonts. The model can only guess between them from context. We may add an evaluation option that treats them as equal later.

### Output options (applied after recognition)
- `strip_diacritics`: remove tashkeel. Also used to compute the "diacritics ignored" error rate.
- `strip_tatweel`: remove stretching characters.
- `convert_digits`: output digits as Western (0–9) or Arabic-Indic (٠–٩).

---

## 3. From corpus text to training lines

A **corpus** is a big collection of text (e.g. all of Arabic Wikipedia). Raw corpus text is messy, so `training/text.py` processes it in two steps.

**Cleaning (`clean_text`).** Every unsupported character (a Latin letter, an emoji) acts as a cut point. We keep the stretches in between only if they're mostly Arabic script. Diacritics that aren't sitting on a letter are removed, because they can't be drawn sensibly.

**Sampling (`LineSampler`).** We cut clean text into lines whose lengths vary from 1 to 60 characters. A model trained only on long lines does badly on a page number or a one-word heading, and the reverse is also true. The sampler also:
- strips diacritics from about half the lines, so the model sees both diacritized and plain text;
- sometimes inserts a number in Western, Arabic-Indic, or Persian digits;
- sometimes makes a very short fragment (1–5 characters).

**Seeds.** The sampler uses its own random generator with a fixed **seed**. The same seed and the same input always give exactly the same lines, so a dataset can be regenerated and a training result reproduced.

### Which corpora, and why licenses decide

The text we train on shapes what the model is good at, and its license decides what we may do with the model.

| Corpus | What it gives us | License |
|---|---|---|
| Arabic Wikipedia | modern standard Arabic, names, numbers | CC BY-SA 4.0 |
| Persian and Urdu Wikipedia | the extra letters پ چ ژ گ ک ی ٹ ڈ ڑ ں ے … | CC BY-SA 4.0 |
| Arabic Wikisource | classical prose and poetry, often with full tashkeel | CC BY-SA 4.0 |
| Tanzil Quran text | fully diacritized text | CC BY 3.0 + Tanzil terms |

Two well-known corpora were **left out**:
- **Tashkeela** (GPLv2): it's unclear whether a model trained on GPL text must itself be GPL.
- **OpenITI** (non-commercial): a model trained on it couldn't be offered for every use.

Since arabic-ocr is open source and anyone may use it, we only train on text whose license clearly allows that. ML projects often get this wrong and discover too late that their model can't be released.

**Attribution.** CC BY and CC BY-SA require credit. When we release a model, its notes must credit Wikipedia/Wikisource contributors and the Tanzil Project. "ShareAlike" may also mean the weights should be released under CC BY-SA 4.0; PRD §11 already allows that.

**How much text?** 1M training lines of at most 60 characters is about 35 MB of text. We download a few hundred MB per language, which is plenty of variety, instead of the whole multi-GB dumps.

**Preparing a corpus** (`uv run python -m training.corpora prepare`):
1. clean every paragraph;
2. drop exact duplicates (Wikipedia has many repeated boilerplate lines);
3. assign each paragraph to train/val/test by hash (section 7);
4. write one file per split.

The generator then reads random lines from those files without loading them into memory. Each corpus has a **weight** that says how often it's picked. Weight isn't size: the Quran is small but gets 10% of lines, because fully diacritized text is rare and the model needs lots of tashkeel to learn it.

---

## 4. Fonts: variety is the point

A model trained on one font learns *that font*, not Arabic. Show it 50 fonts and it has to learn what all the shapes of ب have in common. That's why the font list (`training/configs/fonts.yaml`) mixes styles:

| Style | What it is | Why we need it |
|---|---|---|
| Naskh | the standard book/newspaper style | most printed Arabic |
| Sans / modern | clean UI fonts (Noto Sans, Vazirmatn, Cairo…) | screenshots, websites, modern documents |
| Kufi | geometric, angular | headings, signs, logos |
| Ruqaa | everyday handwriting style | some print and posters |
| Nastaliq | Urdu/Persian calligraphic style, slanted words | Urdu text (not a v1 accuracy target) |
| Display | decorative headline fonts | robustness; sampled rarely |

Each font has a **sampling weight**: common, realistic fonts are picked more often than decorative ones. A family's weight is split across its files, so a family with 8 weights (Thin to Black) isn't picked 8× as often.

**Licenses.** All 50 families come from the Google Fonts repository and are under the SIL Open Font License. The downloader refuses any font whose license isn't on our allowed list. Fonts are only used to *make* training images; they're never shipped in the app.

**Coverage.** Not every font can draw every character. Many fonts lack Urdu letters like ٹ ڈ ڑ, or the Arabic decimal separator ٫. The manifest (`training/fonts.json`) records, for each font file, which of our symbols it can't draw, and the generator only picks a font that can draw every character in the line. If it drew a missing character anyway, the image would show an empty box (".notdef") while the label says ٹ.

**A trap found by looking: fonts that lie about digits.** During visual inspection, the font Zain drew `٤٠٠` (Arabic-Indic digits) with the *same shapes* as `400`. The label said one thing and the image showed another. Training on that would teach the model to read `400` as `٤٠٠`. The downloader now renders each Arabic digit next to its Western twin and, if they're pixel-identical, marks both as "can't draw" for that font. This is the kind of bug only visual inspection finds, which is why M1 requires a manual check of 200 samples.

**Known ambiguity: ١ vs ۱.** Arabic-Indic and Persian digits look identical for 0, 1, 2, 3, 7, 8 and 9; only 4, 5 and 6 differ. The model has to use context (Persian letters nearby suggest Persian digits). This is unavoidable, and evaluation may treat them as equal.

---

## 5. Rendering: from text to image, correctly shaped

Arabic letters change shape depending on their neighbours: ع looks different at the start, middle, and end of a word. Letters join, some pairs form ligatures (ل + ا → لا), and diacritics must sit above or below the right letter. Doing this is called **shaping**, and it's hard. We use **HarfBuzz**, the shaping engine inside Chrome, Firefox, and Android. Never draw Arabic without it: drawing the letters one by one gives disconnected, wrong shapes.

The renderer (`training/render.py`) does three steps:

1. **Direction runs (bidi).** Arabic is right-to-left, but numbers inside it read left-to-right: `سنة 2026` must show `2026`, not `6202`. The Unicode Bidirectional Algorithm decides which characters belong to a left-to-right number. It has surprising rules. For example, after Arabic text, `50%` is displayed as `%50`, because the percent sign stops being part of the number. We implement the algorithm exactly for the characters in our charset, and we checked it against an independent reference implementation on 20,000 random strings with zero differences. Getting this right matters: real documents were displayed by programs that follow these rules, so our synthetic images must too.
2. **Shaping (HarfBuzz).** Each run becomes a list of glyphs (font shapes) with exact positions.
3. **Rasterizing (FreeType).** Each glyph is drawn as a grayscale bitmap and pasted at its position.

The output is an **ink image**: 0 = no ink, 255 = full ink. Turning that into realistic paper, screens, blur and noise is the job of augmentation (next section, coming soon).

**Checking by eye.** `uv run python -m training.inspect_samples` writes `data/inspect/index.html`. Under every rendered image it shows the label as text drawn by your browser, which is a completely separate shaping engine. If the two disagree on letters, diacritics, word order, or numbers, something is wrong. Automated tests catch crashes; only eyes catch "this looks plausible but says the wrong thing".

---

## 6. Augmentation: teaching the model about messy reality

Our rendered lines are perfect: pure black on pure white, razor-sharp. Real inputs aren't: scans are tilted and grainy, photocopies are faded, screenshots are JPEG-compressed, dark mode inverts the colours. A model that has only ever seen perfect images learns details that exist only in perfect images, then fails on real ones. This gap between training data and real data is the **domain gap**, the biggest risk in this project (PRD §14).

**Augmentation** closes the gap. Every training image gets a random mix of realistic damage, so the model is forced to learn what stays the same through all of it: the letter shapes. Because the damage is random, the same text line looks different every time, which also fights **overfitting** (memorizing).

`training/augment.py` picks one of two modes per image:

| Mode | Share | Effects |
|---|---|---|
| **Print / scan** | 65% | paper texture and stains; thicker (ink bleed) or thinner strokes; slight rotation, slant and perspective; faded ink; uneven lighting (shadows) |
| **Screen** | 35% | flat or gradient background; 25% dark mode (light text on dark); table/input border lines; little noise |

Then both modes may get: low resolution (shrink and re-enlarge), blur, noise, and JPEG compression. Extra random margins are added around the text, because the text detector's crops won't be pixel-exact either.

### The golden rule: never break the label
Augmentation is only useful if the label is still true for the damaged image. So:

- **Never flip or mirror.** A mirrored Arabic word is not the same word.
- **Never cut into the text.** Rotation, slant and perspective enlarge the canvas instead.
- **Keep enough contrast.** Text and background are always at least 90 gray levels apart, about a 3:1 contrast ratio, the minimum for readable text in accessibility guidelines.
- **Don't destroy dots and diacritics.** Looking at samples showed that stacking *every* effect on small text turned `السعر` into a smudge. In Arabic, dots are the only difference between ب ت ث ن ي, so an image where they've vanished teaches the model to guess. The fixes:
  - at most 3 lighting/quality effects per image;
  - blur strength scales with the text size;
  - low resolution never shrinks text below 20 pixels tall.

Tuning augmentation is a balance. Too weak and the model is fragile on real images; too strong and it trains on images even a human couldn't read, which wastes effort and teaches guessing. The only way to judge is to **look**:

```
uv run python -m training.inspect_samples -n 200 --augment
```

---

## 7. Dataset files and splits

Training reads millions of images, often on a rented GPU machine that must first copy them over. So *how* they're stored matters.

**Shards.** We pack samples into `.tar` files ("shards") of about 5,000 samples each. Every sample is three files with the same name: the image (`.png`), the label (`.txt`), and extra info like the font (`.json`). This is the **WebDataset** layout. Copying 200 big files is far faster than copying a million small ones, training can read a shard from start to end without jumping around the disk, and any archive tool can open a shard to look inside.

Images are stored at a fixed **height of 64 pixels** (width follows the line length) as lossless PNG, about 8.5 KB each. 1M images is about 8–9 GB.

**Splits, done at the right level.** Note 01 explained train/validation/test. The trap is *leaks*: if the sentence "اَلْعِلْمُ نُورٌ" is in training and a different crop of it is in the test set, the test score measures memory, not reading. So we split the **source sentences** first, then cut lines from each split's own sentences. Each sentence's split is decided by a **hash** of its text (with diacritics removed, so the diacritized and plain versions go together). A hash is a fixed "fingerprint" of the text: the same sentence always gets the same split, on any computer, without storing a list.

**Reproducible and resumable.** Every shard gets its own random seed, computed from the dataset's seed plus the split name and shard number. So:
- shards can be made **in parallel**, one per CPU core;
- if generation crashes or you press Ctrl+C, just run the command again: **finished shards are kept**, and missing ones are made again identically;
- a shard is written under a temporary name and renamed only when complete, so a half-written shard can never be mistaken for a finished one.

**The manifest.** `manifest.json` records the full config, the number of sentences and samples per split, the fonts commit, and a **charset fingerprint**: a hash of the symbol list. A model trained on this data only works with the exact same charset, so training will check this fingerprint.

```
uv run python -m training.make_dataset training/configs/data_sample.yaml
```

Speed on a normal PC: about 200 images/second per CPU core, so 1M images take roughly 10 minutes on 8+ cores.
