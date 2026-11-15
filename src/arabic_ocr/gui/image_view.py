"""The page image with the detected line boxes drawn on top.

- Mouse wheel scrolls; Ctrl + wheel zooms; "Fit" shows the whole page.
- Clicking a line selects it (the text view jumps to it); the selected line is
  drawn stronger.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsScene, QGraphicsView

BOX = QColor(0, 120, 215)
SELECTED = QColor(255, 140, 0)


def to_qimage(image: np.ndarray) -> QImage:
    """A NumPy image (gray, BGR or BGRA, uint8) as a QImage (copied)."""
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    if image.ndim == 2:
        fmt, data = QImage.Format.Format_Grayscale8, image
    elif image.shape[2] == 4:
        data = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        fmt = QImage.Format.Format_RGBA8888
    else:
        data = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        fmt = QImage.Format.Format_RGB888
    return QImage(data.data, w, h, data.strides[0], fmt).copy()


def from_qimage(image: QImage) -> np.ndarray:
    """A QImage (e.g. from the clipboard) as a BGR NumPy image."""
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = image.width(), image.height(), image.bytesPerLine()
    data = np.frombuffer(image.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)
    rgb = data[:, : w * 3].reshape(h, w, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class LineItem(QGraphicsPolygonItem):
    def __init__(self, index: int, box: np.ndarray, width: float) -> None:
        super().__init__(QPolygonF([QPointF(float(x), float(y)) for x, y in box]))
        self.index = index
        self.width = width
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_selected(False)

    def set_selected(self, on: bool) -> None:
        color = SELECTED if on else BOX
        pen = QPen(color, self.width)
        pen.setCosmetic(False)
        fill = QColor(color)
        fill.setAlpha(60 if on else 25)
        self.setPen(pen)
        self.setBrush(QBrush(fill))
        self.setZValue(2 if on else 1)


class ImageView(QGraphicsView):
    line_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        hints = QPainter.RenderHint
        self.setRenderHints(hints.Antialiasing | hints.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(128, 128, 128, 40))
        # Images are never mirrored, even when the window is right-to-left.
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self._items: list[LineItem] = []
        self._selected = -1
        self._fit = True

    def show_page(self, image: np.ndarray | None, boxes: list[np.ndarray]) -> None:
        scene = self.scene()
        scene.clear()
        self._items, self._selected = [], -1
        if image is None:
            return
        scene.addPixmap(QPixmap.fromImage(to_qimage(image)))
        h, w = image.shape[:2]
        pen_width = max(1.0, min(w, h) / 600)
        for i, box in enumerate(boxes):
            item = LineItem(i, box, pen_width)
            scene.addItem(item)
            self._items.append(item)
        scene.setSceneRect(0, 0, w, h)
        self._fit = True
        self.fit()

    def select(self, index: int, scroll: bool = True) -> None:
        if self._selected == index:
            return
        if 0 <= self._selected < len(self._items):
            self._items[self._selected].set_selected(False)
        self._selected = index
        if 0 <= index < len(self._items):
            self._items[index].set_selected(True)
            if scroll:
                self.ensureVisible(self._items[index], 40, 40)

    def fit(self) -> None:
        self._fit = True
        if not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit:
            self.fit()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._fit = False
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, LineItem):
            self.select(item.index, scroll=False)
            self.line_clicked.emit(item.index)
        super().mousePressEvent(event)
