"""Export a trained detector checkpoint to ONNX, and check it gives the same answers.

Usage:
    uv run --group train python -m training.export_detector \
        runs/detector_v1/best.pt models/detector-v1.onnx

The ONNX model takes a uint8 grayscale page (1, 1, H, W), H and W multiples of 32,
and returns the core *probability* map (1, 1, H, W) (sigmoid already applied).

The check (CLAUDE.md): run validation pages through PyTorch (CPU, float32) and
ONNX Runtime; the probability maps must agree within 1e-3 and the detected boxes
must be the same.
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

from arabic_ocr.detect.boxes import KERNEL_FRACTION
from arabic_ocr.detect.detector import LineDetector, boxes_from_prob, prepare_page
from training.detector_model import Detector
from training.make_pages import read_pages

ROOT = Path(__file__).resolve().parent.parent


class WithSigmoid(nn.Module):
    def __init__(self, model: Detector) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(images))


def same_boxes(a, b, tolerance: float = 1.0) -> bool:
    """Same number of boxes, each corner within `tolerance` pixels."""
    if len(a) != len(b):
        return False
    key = lambda d: (round(float(d.quad[0][1])), round(float(d.quad[0][0])))  # noqa: E731
    return all(
        np.abs(x.quad - y.quad).max() <= tolerance
        for x, y in zip(sorted(a, key=key), sorted(b, key=key), strict=True)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the detector to ONNX and check it.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check-pages", type=int, default=20)
    args = parser.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = Detector(**cfg.get("model", {}))
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapped = WithSigmoid(model).eval()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    example = torch.full((1, 1, 640, 480), 255, dtype=torch.uint8)
    h = 32 * torch.export.Dim("h_blocks", min=2, max=128)
    w = 32 * torch.export.Dim("w_blocks", min=2, max=128)
    program = torch.onnx.export(
        wrapped,
        (example,),
        dynamo=True,
        input_names=["page"],
        output_names=["core_prob"],
        dynamic_shapes={"images": {2: h, 3: w}},
        optimize=True,
        verbose=False,  # its progress lines contain emoji the Windows console can't print
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(args.output))
    proto = onnx.load(str(args.output))
    meta = {
        "max_side": str(cfg["max_side"]),
        "kernel_fraction": str(KERNEL_FRACTION),
        "run_name": str(cfg["name"]),
        "val_f1": f"{ckpt['state']['best_f1']:.4f}",
        "input": "uint8 grayscale (1, 1, H, W), H and W multiples of 32",
        "output": "line-core probability (1, 1, H, W)",
    }
    del proto.metadata_props[:]
    for k, v in meta.items():
        proto.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(args.output))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MB)")

    det = LineDetector(args.output)
    max_diff, box_mismatch, n = 0.0, 0, 0
    for shard in sorted((ROOT / cfg["dataset"] / "val").glob("shard-*.tar")):
        for _k, img, _meta in read_pages(shard):
            page = prepare_page(img, det.max_side)
            with torch.no_grad():
                pt = wrapped(torch.from_numpy(page.image)[None, None])[0, 0].numpy()
            ox = det.probability_map(page)
            max_diff = max(max_diff, float(np.abs(pt - ox).max()))
            if not same_boxes(boxes_from_prob(pt, page), boxes_from_prob(ox, page)):
                box_mismatch += 1
            n += 1
            if n >= args.check_pages:
                break
        if n >= args.check_pages:
            break
    print(f"Checked {n} pages: largest probability difference {max_diff:.2e}")
    print(f"  pages whose boxes differ: {box_mismatch}")
    ok = max_diff < 1e-3 and box_mismatch == 0
    print("PASS: ONNX matches PyTorch." if ok else "FAIL: ONNX does not match PyTorch.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
