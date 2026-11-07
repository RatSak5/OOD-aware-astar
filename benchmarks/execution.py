"""Persistent isolated search worker with a pilot-frozen wall-clock timeout."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
from time import perf_counter
from typing import Any

from benchmarks.search_worker import worker_main


@dataclass(frozen=True, slots=True)
class SearchExecution:
    """One worker response plus separately reported harness round-trip time."""

    response: dict[str, Any]
    worker_roundtrip_seconds: float
    worker_restarted: bool


class SearchWorker:
    """Keep search calls serial while hard timeouts can terminate a stuck run."""

    def __init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._process: mp.Process | None = None
        self._connection: Connection | None = None
        self._last_run_timed_out = False
        self.worker_startup_seconds = 0.0
        self.worker_restart_count = 0
        self._start_worker()

    def _start_worker(self) -> None:
        started = perf_counter()
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(target=worker_main, args=(child,))
        process.daemon = True
        process.start()
        child.close()
        if not parent.poll(120.0):
            process.terminate()
            process.join(timeout=5.0)
            parent.close()
            raise TimeoutError("Search worker did not become ready within 120 seconds.")
        message = parent.recv()
        if message.get("status") != "ready":
            process.terminate()
            process.join(timeout=5.0)
            parent.close()
            raise RuntimeError(f"Search worker startup failed: {message!r}")
        self._process = process
        self._connection = parent
        self.worker_startup_seconds += perf_counter() - started
        self._last_run_timed_out = False

    def _stop_worker(self, *, terminate: bool) -> None:
        process, connection = self._process, self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            if not terminate:
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            connection.close()
        if process is not None:
            if terminate and process.is_alive():
                process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)

    def run(
        self,
        task: dict[str, Any],
        *,
        timeout_seconds: float | None,
    ) -> SearchExecution:
        """Execute one task or terminate the worker when its frozen limit expires."""

        restarted = False
        if self._process is None or self._connection is None or not self._process.is_alive():
            self._start_worker()
            restarted = True
        assert self._connection is not None
        started = perf_counter()
        try:
            self._connection.send(task)
            ready = (
                self._connection.poll()
                if timeout_seconds is None
                else self._connection.poll(float(timeout_seconds))
            )
            if not ready:
                elapsed = perf_counter() - started
                self._stop_worker(terminate=True)
                self.worker_restart_count += 1
                self._last_run_timed_out = True
                return SearchExecution(
                    response={
                        "status": "timeout",
                        "error_type": "TimeoutError",
                        "error_message": f"Search exceeded {timeout_seconds:g} seconds.",
                    },
                    worker_roundtrip_seconds=float(elapsed),
                    worker_restarted=True,
                )
            response = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError) as error:
            elapsed = perf_counter() - started
            self._stop_worker(terminate=True)
            self.worker_restart_count += 1
            self._last_run_timed_out = False
            return SearchExecution(
                response={
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                worker_roundtrip_seconds=float(elapsed),
                worker_restarted=True,
            )
        elapsed = perf_counter() - started
        self._last_run_timed_out = False
        return SearchExecution(
            response=response,
            worker_roundtrip_seconds=float(elapsed),
            worker_restarted=restarted,
        )

    def close(self) -> None:
        """Stop the persistent worker cleanly."""

        self._stop_worker(terminate=False)

    def __enter__(self) -> "SearchWorker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
