"""OCR in a background thread, so the window never freezes.

The window sends jobs (a file, or a pasted image) to the Worker; the worker reads
them one at a time, in order, on its own thread and reports back with signals:

    started(job_id, pages)          the job began; it has `pages` pages
    progress(job_id, page, pages)   page `page` (1-based) is done
    finished(job_id, Result)        all done
    failed(job_id, message)         something went wrong (e.g. an unreadable file)

Qt delivers signals between threads safely: the window's slots run on the
window's thread. Jobs wait in the thread's event queue, so a batch of 100 files
is just 100 queued `run` calls.

The models are loaded once, on the first job (about a second).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from arabic_ocr.gui.model import Result

#: Resolution PDF pages are rendered at.
DPI = 300


@dataclass
class Job:
    id: int
    name: str
    path: Path | None = None  # a file ...
    image: np.ndarray | None = None  # ... or an image (pasted)


class Cancelled(Exception):
    pass


class Worker(QObject):
    models_loading = Signal()
    models_failed = Signal(str)
    started = Signal(int, int)
    progress = Signal(int, int, int)
    finished = Signal(int, object)
    failed = Signal(int, str)
    submit = Signal(object)  # the window emits this; `run` receives it on the worker thread

    def __init__(self) -> None:
        super().__init__()
        self._ocr = None
        self._cancelled: set[int] = set()
        self._lock = threading.Lock()
        self.submit.connect(self.run)

    def cancel(self, job_ids) -> None:
        """Skip these jobs (a job that is running stops after its current page).
        Safe to call from any thread."""
        with self._lock:
            self._cancelled.update(job_ids)

    def _is_cancelled(self, job_id: int) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def _load(self):
        if self._ocr is None:
            self.models_loading.emit()
            from arabic_ocr.pipeline import OCR

            self._ocr = OCR()
        return self._ocr

    @Slot(object)
    def run(self, job: Job) -> None:
        if self._is_cancelled(job.id):
            return
        try:
            ocr = self._load()
        except Exception as e:  # models missing or broken: every job fails
            self.models_failed.emit(str(e))
            self.failed.emit(job.id, str(e))
            return
        try:
            self.finished.emit(job.id, self._read(ocr, job))
        except Cancelled:
            pass
        except Exception as e:  # unreadable file etc.: report it and go on with the queue
            self.failed.emit(job.id, str(e) or type(e).__name__)

    def _read(self, ocr, job: Job) -> Result:
        # The raw OCR: options (diacritics, digits...) are applied when shown/exported.
        if job.image is not None:
            self.started.emit(job.id, 1)
            page = ocr.read_image(job.image)
            self.progress.emit(job.id, 1, 1)
            return Result(job.name, None, DPI, [page], images=[job.image])

        from arabic_ocr.inputs import load_pages, page_count

        n = page_count(job.path)
        self.started.emit(job.id, n)
        pages = []
        for i, image in enumerate(load_pages(job.path, DPI), 1):
            if self._is_cancelled(job.id):
                raise Cancelled
            pages.append(ocr.read_image(image))
            self.progress.emit(job.id, i, n)
        return Result(job.name, job.path, DPI, pages)


def start_worker() -> tuple[Worker, QThread]:
    """A Worker living on its own thread. Stop it with stop_worker."""
    thread = QThread()
    thread.setObjectName("ocr-worker")
    worker = Worker()
    worker.moveToThread(thread)
    thread.start()
    return worker, thread


def stop_worker(worker: Worker, thread: QThread, wait_ms: int = 10_000) -> None:
    thread.quit()  # stops after the job that is running now
    thread.wait(wait_ms)
