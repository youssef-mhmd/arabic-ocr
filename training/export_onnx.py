"""Export a trained recognizer checkpoint to ONNX, and check it gives the same answers.

Usage:
    uv run --group train python -m training.export_onnx \
        runs/recognizer_v1/best.pt models/recognizer-v1.onnx
    ... --check-samples 500     # how many test lines to compare (default 300)

What is ONNX? A standard file format for trained models. PyTorch is a big training
library (~3 GB with CUDA); ONNX Runtime is a small, fast engine that only *runs*
models (~15 MB, CPU, every OS). Exporting records the model's calculation steps and
weights into one .onnx file that ONNX Runtime can execute without PyTorch.

The check (CLAUDE.md: "verify the ONNX output matches PyTorch output"):
1. run the same test lines through PyTorch (CPU, float32) and ONNX Runtime;
2. compare the raw scores: the largest difference must be tiny (< 1e-3);
3. compare the decoded text: it must be identical for every line.
The export fails (exit code 1) if either check fails.

The .onnx file also stores metadata the runtime checks: the charset fingerprint (a
model only works with the symbol list it was trained on), input height and stride.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from arabic_ocr import charset
from arabic_ocr.recognize.ctc import greedy_decode
from arabic_ocr.recognize.recognizer import LineRecognizer
from training.dataset import read_split
from training.model import INPUT_HEIGHT, WIDTH_STRIDE, Recognizer

ROOT = Path(__file__).resolve().parent.parent
MAX_ABS_DIFF = 1e-3


class ExportWrapper(nn.Module):
    """The recognizer without the optional `lengths` argument (training-only)."""

    def __init__(self, model: Recognizer) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def load_model(ckpt_path: Path) -> tuple[Recognizer, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt["charset"] != charset.fingerprint():
        raise SystemExit("This checkpoint was trained with a different charset.")
    model = Recognizer(num_classes=ckpt["num_classes"], **ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def export(model: Recognizer, ckpt: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    example = torch.full((2, 1, INPUT_HEIGHT, 256), 255, dtype=torch.uint8)
    batch = torch.export.Dim("batch", min=1, max=1024)
    # Width in units of the stride: the real width is always a multiple of 4.
    width = 4 * torch.export.Dim("width_steps", min=8, max=2048)
    program = torch.onnx.export(
        ExportWrapper(model),
        (example,),
        dynamo=True,
        input_names=["images"],
        output_names=["log_probs"],
        dynamic_shapes={"images": {0: batch, 3: width}},
        optimize=True,
    )
    program.save(str(out))

    proto = onnx.load(str(out))
    state = ckpt["state"]
    meta = {
        "charset_fingerprint": charset.fingerprint(),
        "num_classes": str(ckpt["num_classes"]),
        "input_height": str(INPUT_HEIGHT),
        "width_stride": str(WIDTH_STRIDE),
        "input": "uint8 grayscale (batch, 1, 64, width), width a multiple of 4, white=255",
        "output": "log-probabilities (width/4, batch, classes), right-to-left reading order",
        "run_name": str(ckpt["config"]["name"]),
        "samples_seen": str(state["samples_seen"]),
        "val_cer": f"{state['best_cer']:.5f}",
    }
    del proto.metadata_props[:]
    for k, v in meta.items():
        proto.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out))


def check(model: Recognizer, onnx_path: Path, dataset_dir: Path, n: int) -> bool:
    """Compare PyTorch and ONNX Runtime on `n` test lines, one by one and batched."""
    rec = LineRecognizer(onnx_path)
    samples = []
    for s in read_split(dataset_dir, "test"):
        samples.append(s.image)
        if len(samples) >= n:
            break

    max_diff, text_mismatch = 0.0, 0
    with torch.no_grad():
        for img in samples:
            x = rec._pad(img)[None, None]  # same padding as the runtime
            pt = model(torch.from_numpy(x)).numpy()
            ox = rec.run(x)
            max_diff = max(max_diff, float(np.abs(pt - ox).max()))
            if greedy_decode(pt[:, 0]).text != greedy_decode(ox[:, 0]).text:
                text_mismatch += 1

    # Batched path of the runtime wrapper vs one line at a time.
    one_by_one = [rec.recognize(img).text for img in samples]
    batched = [d.text for d in rec.recognize_batch(samples)]
    batch_mismatch = sum(a != b for a, b in zip(one_by_one, batched, strict=True))

    print(f"Checked {len(samples)} test lines:")
    print(f"  largest score difference PyTorch vs ONNX: {max_diff:.2e} (limit {MAX_ABS_DIFF})")
    print(f"  lines whose text differs PyTorch vs ONNX: {text_mismatch}")
    print(f"  lines whose text differs batched vs one-by-one (ONNX): {batch_mismatch}")
    return max_diff < MAX_ABS_DIFF and text_mismatch == 0 and batch_mismatch == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the recognizer to ONNX and check it.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/datasets/v1")
    parser.add_argument("--check-samples", type=int, default=300)
    args = parser.parse_args(argv)

    model, ckpt = load_model(args.checkpoint)
    # PyTorch's exporter prints many internal deprecation warnings; they are not ours.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    export(model, ckpt, args.output)
    size_mb = args.output.stat().st_size / 2**20
    print(f"Wrote {args.output} ({size_mb:.1f} MB)")
    ok = check(model, args.output, args.dataset, args.check_samples)
    print("PASS: ONNX matches PyTorch." if ok else "FAIL: ONNX does not match PyTorch.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
