"""Find text lines on a page with the exported ONNX detector (CPU, ONNX Runtime).

    from arabic_ocr.detect.detector import LineDetector
    det = LineDetector("models/detector-v1.onnx")
    for d in det.detect(page_image):
        print(d.quad, d.score)

Steps: grayscale -> scale so the longer side is at most `max_side` pixels (the
detector was trained on pages of roughly that size) -> pad to a multiple of 32 ->
run the model -> core map -> line boxes (boxes.py) -> scale boxes back to the
original image.

ONNX Runtime only: the shipped runtime never imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from arabic_ocr.detect.boxes import Detection, map_to_boxes
from arabic_ocr.recognize.recognizer import to_gray

#: Default working size (longer page side, pixels). A4 at 300 DPI is 3508 px tall,
#: so its text is shrunk ~2.5x: body text ends up ~15-25 px tall, as in training.
MAX_SIDE = 1536


@dataclass
class PreparedPage:
    image: np.ndarray  # uint8 (H, W), H and W multiples of 32
    scale: float  # prepared = original * scale
    height: int  # valid (unpadded) size
    width: int


def prepare_page(image: np.ndarray, max_side: int = MAX_SIDE) -> PreparedPage:
    """Grayscale, shrink if needed, and pad (with the border colour) to multiples of 32."""
    gray = to_gray(image)
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    h2, w2 = gray.shape
    ph, pw = -(-h2 // 32) * 32, -(-w2 // 32) * 32
    border = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    padded = cv2.copyMakeBorder(gray, 0, ph - h2, 0, pw - w2, cv2.BORDER_CONSTANT, value=border)
    return PreparedPage(padded, scale, h2, w2)


def boxes_from_prob(prob: np.ndarray, page: PreparedPage, **kw) -> list[Detection]:
    """Core map of a prepared page -> boxes in original-image coordinates."""
    dets = map_to_boxes(prob[: page.height, : page.width], **kw)
    for d in dets:
        d.quad = (d.quad / page.scale).astype(np.float32)
    return dets


class LineDetector:
    def __init__(self, model_path: str | Path, threads: int | None = None) -> None:
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.max_side = int(self.metadata.get("max_side", MAX_SIDE))
        self.input_name = self.session.get_inputs()[0].name

    def probability_map(self, page: PreparedPage) -> np.ndarray:
        """Raw model output for a prepared page: (H, W) probabilities in [0, 1]."""
        return self.session.run(None, {self.input_name: page.image[None, None]})[0][0, 0]

    def detect(self, image: np.ndarray, **kw) -> list[Detection]:
        """Line boxes on a page image (any size, gray or colour)."""
        page = prepare_page(image, self.max_side)
        return boxes_from_prob(self.probability_map(page), page, **kw)
