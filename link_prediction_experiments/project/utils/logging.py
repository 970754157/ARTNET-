"""Plain stdout logging helpers for long-running training jobs."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


def log_event(prefix: str, message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{prefix} {ts} {message}", flush=True)


@contextmanager
def log_stage(name: str, prefix: str = "[STAGE]") -> Iterator[None]:
    start = time.perf_counter()
    log_event(prefix, f"START {name}")
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        log_event("[ERROR]", f"{name} failed after {elapsed:.2f}s: {exc}")
        raise
    else:
        elapsed = time.perf_counter() - start
        log_event(prefix, f"DONE {name} ({elapsed:.2f}s)")


@dataclass
class ProgressTicker:
    """Emit throttled plain-text progress updates."""

    label: str
    every_sec: float
    prefix: str = "[STAGE]"
    _last_emit: float = field(default_factory=time.perf_counter)

    def maybe(self, message: str) -> None:
        now = time.perf_counter()
        if now - self._last_emit >= float(self.every_sec):
            log_event(self.prefix, f"{self.label}: {message}")
            self._last_emit = now

    def force(self, message: str) -> None:
        self._last_emit = time.perf_counter()
        log_event(self.prefix, f"{self.label}: {message}")


def _format_fields(fields: Dict[str, object]) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


class StepPhaseContext:
    def __init__(
        self,
        reporter: "StepProgressReporter",
        phase: str,
        start_fields: Optional[Dict[str, object]] = None,
    ):
        self.reporter = reporter
        self.phase = phase
        self.fields: Dict[str, object] = dict(start_fields or {})
        self.started_at = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "StepPhaseContext":
        self.started_at = time.perf_counter()
        self.reporter._emit(self.phase, "start", self.fields)
        if self.reporter.log_every_sec > 0:
            self._thread = threading.Thread(target=self._progress_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)
        elapsed = time.perf_counter() - self.started_at
        done_fields = dict(self.fields)
        done_fields.setdefault("elapsed_sec", f"{elapsed:.2f}")
        if exc is None:
            self.reporter._emit(self.phase, "done", done_fields)
        else:
            done_fields["error"] = str(exc)
            self.reporter._emit(self.phase, "error", done_fields)
        return False

    def update(self, **fields: object) -> None:
        self.fields.update({k: v for k, v in fields.items() if v is not None})

    def _progress_loop(self) -> None:
        while not self._stop_event.wait(self.reporter.log_every_sec):
            elapsed = time.perf_counter() - self.started_at
            progress_fields = dict(self.fields)
            progress_fields.setdefault("elapsed_sec", f"{elapsed:.2f}")
            self.reporter._emit(self.phase, "progress", progress_fields)


@dataclass
class StepProgressReporter:
    step: int
    epoch: int
    active_chunk: int
    log_every_sec: float
    enable_phase_logs: bool = True
    min_items: int = 1
    _last_progress_at: Dict[str, float] = field(default_factory=dict)

    def phase(self, phase: str, **start_fields: object) -> StepPhaseContext:
        return StepPhaseContext(self, phase, dict(start_fields))

    def emit(self, phase: str, status: str, **fields: object) -> None:
        self._emit(phase, status, fields)

    def progress(self, phase: str, **fields: object) -> None:
        now = time.perf_counter()
        last = self._last_progress_at.get(phase, 0.0)
        if now - last < self.log_every_sec:
            return
        self._last_progress_at[phase] = now
        self._emit(phase, "progress", fields)

    def _emit(self, phase: str, status: str, fields: Dict[str, object]) -> None:
        if not self.enable_phase_logs:
            return
        base = {
            "step": self.step,
            "epoch": self.epoch,
            "active_chunk": self.active_chunk,
            "phase": phase,
            "status": status,
        }
        base.update(fields)
        log_event("[STEP]", _format_fields(base))
