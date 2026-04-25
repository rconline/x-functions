"""Audit sinks — §8.

One event per Pandas UDF batch (NOT per row). Fields per §8 schema.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Callable, Iterable, Optional

AuditEvent = dict[str, Any]


class AuditSink(ABC):
    @abstractmethod
    def emit(self, event: AuditEvent) -> None: ...


class StdoutAuditSink(AuditSink):
    """Default standalone sink — one JSON object per line to stdout."""

    def __init__(self, stream=None):
        self._stream = stream or sys.stdout

    def emit(self, event: AuditEvent) -> None:
        try:
            self._stream.write(json.dumps(event, default=str) + "\n")
            self._stream.flush()
        except Exception:
            # Audit never fails the call. Swallow and carry on.
            pass


class InMemoryAuditSink(AuditSink):
    """Thread-safe in-memory buffer — used by tests via `assert_audit_event`."""

    def __init__(self):
        self._events: list[AuditEvent] = []
        self._lock = Lock()

    def emit(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(dict(event))

    # --- test helpers ---

    @property
    def events(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def find(self, **kwargs) -> list[AuditEvent]:
        with self._lock:
            return [e for e in self._events if _matches(e, kwargs)]

    def only(self, **kwargs) -> AuditEvent:
        matches = self.find(**kwargs)
        if len(matches) != 1:
            raise AssertionError(
                f"expected exactly one audit event matching {kwargs}, found {len(matches)}: "
                f"{json.dumps(matches, default=str, indent=2)}"
            )
        return matches[0]


class CompositeAuditSink(AuditSink):
    def __init__(self, *sinks: AuditSink):
        self._sinks = sinks

    def emit(self, event: AuditEvent) -> None:
        for s in self._sinks:
            try:
                s.emit(event)
            except Exception:
                pass


class CallbackAuditSink(AuditSink):
    """Wrap an arbitrary callable as a sink (useful for Kafka/HTTP shippers)."""

    def __init__(self, fn: Callable[[AuditEvent], None]):
        self._fn = fn

    def emit(self, event: AuditEvent) -> None:
        try:
            self._fn(event)
        except Exception:
            pass


def _matches(event: AuditEvent, criteria: dict[str, Any]) -> bool:
    for k, v in criteria.items():
        if event.get(k) != v:
            return False
    return True
