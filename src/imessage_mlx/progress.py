from __future__ import annotations

import sys
from typing import TextIO


class ProgressBar:
    """Small progress bar that remains readable in terminals and captured logs."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        enabled: bool = True,
        stream: TextIO | None = None,
        width: int = 24,
        log_step_percent: int = 5,
    ) -> None:
        self.label = label
        self.total = max(0, total)
        self.enabled = enabled and self.total > 0
        self.stream = stream or sys.stderr
        self.width = width
        self.log_step_percent = max(1, log_step_percent)
        self.current = 0
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_log_bucket = -1
        if self.enabled:
            self._render()

    def advance(self, amount: int = 1) -> None:
        if not self.enabled:
            return
        self.current = min(self.total, self.current + amount)
        self._render()

    def close(self) -> None:
        if not self.enabled:
            return
        self.current = self.total
        self._render(force=True)
        if self._is_tty:
            self.stream.write("\n")
            self.stream.flush()
        self.enabled = False

    def _render(self, *, force: bool = False) -> None:
        ratio = self.current / self.total
        percent = int(ratio * 100)
        bucket = percent // self.log_step_percent
        if not self._is_tty and not force and bucket <= self._last_log_bucket:
            return
        self._last_log_bucket = bucket
        filled = min(self.width, int(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        text = f"{self.label} [{bar}] {self.current}/{self.total} ({percent:3d}%)"
        ending = "\r" if self._is_tty else "\n"
        self.stream.write(text + ending)
        self.stream.flush()
