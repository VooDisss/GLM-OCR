"""UnitTracker — tracks per-unit (per input URL) region completion.

One "unit" corresponds to a single input URL (one image file or one PDF).
A PDF unit may span multiple pages, each page having multiple regions.
This tracker counts completed regions and notifies the main thread when
all regions for a unit are done.

Thread safety:
  - ``on_region_done()`` is called from Thread 3 (recognition worker).
  - ``wait_next_ready_unit()`` / ``iter_ready_units()`` are called from the main thread.
"""

from __future__ import annotations

import queue
import threading
from typing import Dict, List


class UnitTracker:
    """Tracks region-level completion for each input unit."""

    def __init__(
        self,
        num_units: int,
        unit_image_indices: List[List[int]],
        unit_region_count: List[int],
    ):
        self._num_units = num_units
        self._unit_image_indices = unit_image_indices
        self._unit_region_count = unit_region_count

        self._unit_for_image: Dict[int, int] = {
            img_idx: u
            for u in range(num_units)
            for img_idx in unit_image_indices[u]
        }
        self._done_count: List[int] = [0] * num_units
        self._ready_queue: queue.Queue[int] = queue.Queue()
        self._notified: set = set()

        self._lock = threading.Lock()
        self._notify_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Backfill: handle regions that completed *before* tracker existed
    # ------------------------------------------------------------------

    def backfill(self, done_page_indices: List[int]) -> None:
        """Account for regions finished before the tracker was initialised.

        Args:
            done_page_indices: ``page_idx`` values of already-completed regions.

        Called once from the main thread right after construction.
        """
        for page_idx in done_page_indices:
            u = self._unit_for_image.get(page_idx)
            if u is not None:
                self._done_count[u] += 1

        for u in range(self._num_units):
            if self._done_count[u] >= self._unit_region_count[u]:
                self._ready_queue.put(u)
                self._notified.add(u)

    # ------------------------------------------------------------------
    # Runtime: called from Thread 3 after each region completes
    # ------------------------------------------------------------------

    def on_region_done(self, page_idx: int) -> None:
        """Increment the counter for the unit owning *page_idx*.

        O(1). If the unit reaches its target count, enqueue it.
        """
        u = self._unit_for_image.get(page_idx)
        if u is None:
            return
        with self._lock:
            self._done_count[u] += 1
            ready = self._done_count[u] >= self._unit_region_count[u]
        if ready:
            with self._notify_lock:
                if u not in self._notified:
                    self._ready_queue.put(u)
                    self._notified.add(u)

    # ------------------------------------------------------------------
    # Consumption: called from the main thread
    # ------------------------------------------------------------------

    def wait_next_ready_unit(self) -> int:
        """Block until the next unit is ready and return its index."""
        return self._ready_queue.get()

    @property
    def num_units(self) -> int:
        return self._num_units

    @property
    def unit_image_indices(self) -> List[List[int]]:
        return self._unit_image_indices

    @property
    def unit_region_count(self) -> List[int]:
        return self._unit_region_count
