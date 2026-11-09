# 02: Data — characters, labels, and synthetic images

A recognizer learns from pairs of **(image, label)**: a picture of a text line and the exact text it shows. This note explains how we define the possible characters, how we make labels consistent, and (as M1 continues) how we generate millions of images.

*Status: sections 1–3 are written. Fonts, rendering, augmentation, and dataset files will be added as those parts of M1 land.*

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

---

*Next sections (coming in M1): fonts and licenses, shaping with HarfBuzz, augmentation, and dataset files.*
