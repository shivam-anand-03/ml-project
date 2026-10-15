"""Background execution of training runs so prediction requests are never blocked."""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from app.config import get_settings
from app.db import get_session_factory
from app.services.training import execute_training_run


class TrainingJobRunner:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="training")
        self._futures: dict[int, Future] = {}
        self._lock = threading.Lock()

    def submit(self, run_id: int) -> None:
        settings = get_settings()
        if settings.training_sync:
            execute_training_run(get_session_factory(), run_id, settings)
            return
        fut = self._executor.submit(execute_training_run, get_session_factory(), run_id, settings)
        with self._lock:
            self._futures[run_id] = fut

    def is_active(self, run_id: int) -> bool:
        with self._lock:
            fut = self._futures.get(run_id)
        return bool(fut and not fut.done())

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for f in self._futures.values() if not f.done())

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


_runner: TrainingJobRunner | None = None


def get_runner() -> TrainingJobRunner:
    global _runner
    if _runner is None:
        _runner = TrainingJobRunner(max_workers=get_settings().training_workers)
    return _runner
