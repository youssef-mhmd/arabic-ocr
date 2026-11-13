"""Read text lines on the CPU with the exported ONNX recognizer.

    from arabic_ocr.recognize import LineRecognizer
    rec = LineRecognizer("models/recognizer-v1.onnx")
    print(rec.recognize(line_image).text)

What happens to each line image:
1. **Prepare**: convert to grayscale and scale it to the model's input height (64 px),
   keeping its proportions (a wide line stays wide).
2. **Pad and batch**: the width is padded up to a multiple of 4 px (one model step),
   on the left (the end of the line, since the model reads right to left). Only
   lines of exactly the same width share a batch. Why: training tells the LSTM
   where each line ends ("packing"), so padding has no effect there. ONNX cannot
   do that, so here every bit of padding is read as part of the line, and more
   padding can change the result. Keeping padding under one step avoids that.
3. **Run** the ONNX model: per-slice scores for every symbol.
4. **Decode** with greedy CTC (`ctc.py`) into logical-order text.

ONNX Runtime only: the shipped runtime never imports torch.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from arabic_ocr import charset
from arabic_ocr.recognize.ctc import Decoded, greedy_decode

#: Narrower lines are padded up to this width (same as training).
MIN_WIDTH = 32


def to_gray(image: np.ndarray) -> np.ndarray:
    """Any OpenCV-style image (gray, BGR or BGRA) -> 2-D uint8 grayscale."""
    if image.ndim == 3 and image.shape[2] == 4:
        # Put transparent pixels on white, then drop the alpha channel.
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        image = bgr.astype(np.uint8)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def prepare_line(image: np.ndarray, height: int) -> np.ndarray:
    """Grayscale and resize a line image to `height`, keeping its aspect ratio."""
    gray = to_gray(image)
    h, w = gray.shape
    if h != height:
        new_w = max(1, round(w * height / h))
        shrink = h > height
        gray = cv2.resize(
            gray, (new_w, height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC
        )
    return gray


class LineRecognizer:
    """Runs the ONNX line recognizer with ONNX Runtime on the CPU."""

    def __init__(self, model_path: str | Path, threads: int | None = None) -> None:
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        meta = self.session.get_modelmeta().custom_metadata_map
        self.metadata = dict(meta)
        model_fp = meta.get("charset_fingerprint")
        if model_fp is not None and model_fp != charset.fingerprint():
            raise ValueError(
                f"{model_path} was trained with a different charset "
                f"({model_fp}, this version has {charset.fingerprint()})."
            )
        self.height = int(meta.get("input_height", 64))
        self.stride = int(meta.get("width_stride", 4))
        self.input_name = self.session.get_inputs()[0].name

    def run(self, batch: np.ndarray) -> np.ndarray:
        """Raw model call: uint8 (B, 1, H, W) -> log-probabilities (T, B, classes)."""
        return self.session.run(None, {self.input_name: batch})[0]

    def recognize(self, image: np.ndarray) -> Decoded:
        """Read one line image."""
        return self.recognize_batch([image])[0]

    def recognize_batch(self, images: Sequence[np.ndarray], max_batch: int = 16) -> list[Decoded]:
        """Read several line images; results come back in the same order.

        Lines are batched only with lines of exactly the same (padded) width, so a
        line's result never depends on which other lines it was read with.
        """
        lines = [self._pad(prepare_line(img, self.height)) for img in images]
        by_width: dict[int, list[int]] = defaultdict(list)
        for i, line in enumerate(lines):
            by_width[line.shape[1]].append(i)

        results: list[Decoded | None] = [None] * len(lines)
        for width, idx in by_width.items():
            for k in range(0, len(idx), max_batch):
                chunk = idx[k : k + max_batch]
                scores = self.run(np.stack([lines[i] for i in chunk])[:, None])
                for b, i in enumerate(chunk):
                    results[i] = greedy_decode(scores[: width // self.stride, b])
        return results  # type: ignore[return-value]

    def _pad(self, line: np.ndarray) -> np.ndarray:
        """Pad on the LEFT (the end of the line for a right-to-left reader) by
        repeating the edge pixels, up to a multiple of the stride and at least
        MIN_WIDTH. That is less than one output step of padding for normal lines."""
        width = max(line.shape[1], MIN_WIDTH)
        width = -(-width // self.stride) * self.stride
        return np.pad(line, ((0, 0), (width - line.shape[1], 0)), mode="edge")
