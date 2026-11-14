"""Make faster int8 versions of the shipped ONNX models (CPU speed, M4).

    uv run --group train python -m training.quantize recognizer models/recognizer-v4.onnx
    uv run --group train python -m training.quantize detector models/detector-v2.onnx

Writes <name>-int8.onnx next to the input. What it does (plain language):

Quantization stores numbers as 8-bit integers instead of 32-bit floats. CPUs can
multiply 8-bit integers much faster, at the cost of a little precision. Only the
*convolution* layers are converted (static quantization, QDQ format): they take
~74% of the recognizer's time. The recognizer's LSTM stays in float: ONNX Runtime's
int8 LSTM was 14x *slower* on this CPU.

"Static" means the value range of every layer's input is measured beforehand on
real examples (calibration): 300 validation line images for the recognizer, 20
validation pages for the detector.

Measured for recognizer-v4 (A4 pages, 4 threads, same process): 3.81 -> 2.16 s per
page; CER unchanged within +-0.03 points on every real and synthetic line set.
Always re-check accuracy before shipping a quantized model.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def recognizer_samples(n: int = 300) -> list[np.ndarray]:
    from arabic_ocr.recognize.recognizer import prepare_line
    from training.dataset import read_split

    out = []
    for i, s in enumerate(read_split(ROOT / "data/datasets/v4", "val")):
        if i >= n:
            break
        img = prepare_line(s.image, 64)
        w = -(-max(img.shape[1], 32) // 4) * 4
        out.append(np.pad(img, ((0, 0), (w - img.shape[1], 0)), mode="edge")[None, None])
    return out


def detector_samples(n: int = 20) -> list[np.ndarray]:
    from arabic_ocr.detect.detector import prepare_page
    from training.make_pages import read_pages

    out = []
    for shard in sorted((ROOT / "data/datasets/pages_v3/val").glob("*.tar")):
        for _key, img, _meta in read_pages(shard):
            out.append(prepare_page(img).image[None, None])
            if len(out) >= n:
                return out
    return out


def quantize(kind: str, src: Path, out: Path) -> None:
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    samples = recognizer_samples() if kind == "recognizer" else detector_samples()
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        name = onnx.load(str(pre)).graph.input[0].name

        class Reader(CalibrationDataReader):
            def __init__(self) -> None:
                self.it = iter(samples)

            def get_next(self):
                x = next(self.it, None)
                return None if x is None else {name: x}

        quantize_static(
            str(pre), str(out), Reader(),
            quant_format=QuantFormat.QDQ, op_types_to_quantize=["Conv"], per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        )  # fmt: skip
    print(f"Wrote {out} ({out.stat().st_size / 2**20:.1f} MB) from {len(samples)} samples")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="int8-quantize a shipped ONNX model.")
    parser.add_argument("kind", choices=["recognizer", "detector"])
    parser.add_argument("model", type=Path)
    args = parser.parse_args(argv)
    out = args.model.with_name(args.model.stem + "-int8.onnx")
    quantize(args.kind, args.model, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
