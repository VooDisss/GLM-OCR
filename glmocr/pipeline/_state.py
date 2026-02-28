"""Shared mutable state for the three-stage async pipeline.

This object is created once per ``Pipeline.process()`` call and passed to
all three worker threads.  It holds the inter-thread queues, accumulated
results, and the (optional) UnitTracker reference.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List, Optional

from glmocr.pipeline._unit_tracker import UnitTracker


class PipelineState:
    """Thread-safe container shared by loader / layout / recognition workers.

    Queues (dict messages flow through these):
        page_queue   — Stage 1 → Stage 2
        region_queue — Stage 2 → Stage 3

    Accumulated results (list, not a queue — main thread needs random access):
        recognition_results — Stage 3 appends, main thread snapshots
    """

    def __init__(
        self,
        page_maxsize: int = 100,
        region_maxsize: int = 800,
    ):
        # ── Inter-thread queues ──────────────────────────────────────
        self.page_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=page_maxsize)
        self.region_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=region_maxsize)

        # ── Per-page data (stage 1 & 2 write, main thread reads) ─────
        self.images_dict: Dict[int, Any] = {}
        self.layout_results_dict: Dict[int, List] = {}

        # ── Counters (stage 1 writes, main thread reads after join) ──
        self.num_images_loaded: List[int] = [0]
        self.unit_indices_holder: List[Optional[List[int]]] = [None]

        # ── Recognition results (stage 3 appends, main thread reads) ─
        self._recognition_results: List[Dict[str, Any]] = []
        self._results_lock = threading.Lock()

        # ── UnitTracker (set by main thread after stage 1 & 2 join) ──
        self._tracker: Optional[UnitTracker] = None

        # ── Exception collection ─────────────────────────────────────
        self._exceptions: List[Dict[str, Any]] = []
        self._exception_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recognition results
    # ------------------------------------------------------------------

    def add_recognition_result(self, page_idx: int, region: Dict) -> None:
        """Append a completed region result and notify the tracker."""
        result = {"page_idx": page_idx, "region": region}
        with self._results_lock:
            self._recognition_results.append(result)
        tracker = self._tracker
        if tracker is not None:
            tracker.on_region_done(page_idx)

    def snapshot_recognition_results(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of all results accumulated so far."""
        with self._results_lock:
            return list(self._recognition_results)

    # ------------------------------------------------------------------
    # UnitTracker lifecycle
    # ------------------------------------------------------------------

    def set_tracker(self, tracker: UnitTracker) -> None:
        self._tracker = tracker

    # ------------------------------------------------------------------
    # Exception handling
    # ------------------------------------------------------------------

    def record_exception(self, source: str, exc: Exception) -> None:
        with self._exception_lock:
            self._exceptions.append({"source": source, "exception": exc})

    def raise_if_exceptions(self) -> None:
        with self._exception_lock:
            if self._exceptions:
                raise RuntimeError(
                    "; ".join(
                        f"{e['source']}: {e['exception']}" for e in self._exceptions
                    )
                )
