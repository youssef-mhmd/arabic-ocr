"""Text-line detection: core maps -> line boxes (see boxes.py)."""

from arabic_ocr.detect.boxes import Detection, map_to_boxes, quad_iou

__all__ = ["Detection", "map_to_boxes", "quad_iou"]
