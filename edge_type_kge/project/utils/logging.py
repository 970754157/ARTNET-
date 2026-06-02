"""Plain stdout logging helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


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

