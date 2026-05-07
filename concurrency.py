"""
Concurrency gates — bound heavy ML operations so concurrent users don't crash
the runtime or contend on shared TensorFlow state.

Pattern borrowed from Menu_Engineering_App's solver_gate (which caps active
CP-SAT solves with a bounded semaphore + per-key locks). Same idea here, but
guarding TF prediction + training instead of constraint solves.

Why this is needed:
  - Streamlit Cloud runs ONE Python process. Every user session is a thread
    inside that process, so they share `tf` state and the `_C` model cache.
  - TF inference under unbounded parallelism saturates a 1 CPU / 1 GB tier
    fast and the kernel OOM-kills Streamlit.
  - Two cold-start sessions could both call `train_model(...)` for the same
    client and clobber each other's artifacts mid-write.

What we expose:
  predict_gate()      — bounded semaphore; caps concurrent predict loops.
  train_gate(ck)      — per-client lock; only one trainer per client at a time.
  cache_lock()        — guards ml_core._C against TOCTOU on first model load.
  GateBusy            — raised on timeout; UI shows a clean retry message.

Tunable via env:
  MAX_CONCURRENT_PREDICTS  default 2   (per Streamlit process)
  PREDICT_TIMEOUT_SEC      default 30
  TRAIN_TIMEOUT_SEC        default 600
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator


MAX_CONCURRENT_PREDICTS = int(os.environ.get("MAX_CONCURRENT_PREDICTS", 2))
PREDICT_TIMEOUT_SEC     = float(os.environ.get("PREDICT_TIMEOUT_SEC", 30.0))
TRAIN_TIMEOUT_SEC       = float(os.environ.get("TRAIN_TIMEOUT_SEC", 600.0))


class GateBusy(Exception):
    """Raised when a gate could not be acquired within timeout."""


# ── Module-level state. Lifetime = lifetime of the Streamlit process. ──
_PREDICT_SEM       = threading.BoundedSemaphore(value=MAX_CONCURRENT_PREDICTS)
_TRAIN_LOCKS: dict[str, threading.Lock] = {}
_TRAIN_LOCKS_GUARD = threading.Lock()
_CACHE_LOCK        = threading.Lock()


def _train_lock_for(ck: str) -> threading.Lock:
    """Return (and lazily create) the per-client training lock."""
    with _TRAIN_LOCKS_GUARD:
        lock = _TRAIN_LOCKS.get(ck)
        if lock is None:
            lock = threading.Lock()
            _TRAIN_LOCKS[ck] = lock
        return lock


@contextmanager
def predict_gate(timeout: float | None = None) -> Iterator[None]:
    """
    Cap concurrent predict-loop executions across all sessions.
    Excess users queue until a slot frees, or raise GateBusy after timeout.
    """
    t = PREDICT_TIMEOUT_SEC if timeout is None else timeout
    if not _PREDICT_SEM.acquire(timeout=t):
        raise GateBusy(
            f"Server is already serving {MAX_CONCURRENT_PREDICTS} predictions. "
            "Please retry in a few seconds."
        )
    try:
        yield
    finally:
        _PREDICT_SEM.release()


@contextmanager
def train_gate(ck: str, timeout: float | None = None) -> Iterator[None]:
    """
    Per-client training lock. If two sessions hit a cold start at the same
    moment, only one trains; the other waits, then sees the artifacts
    already exist and skips.
    """
    t = TRAIN_TIMEOUT_SEC if timeout is None else timeout
    lock = _train_lock_for(ck)
    if not lock.acquire(timeout=t):
        raise GateBusy(
            f"Training for '{ck}' is taking longer than expected (>{t:.0f}s). "
            "Try again shortly."
        )
    try:
        yield
    finally:
        lock.release()


@contextmanager
def cache_lock() -> Iterator[None]:
    """
    Guard the ml_core._C dict against TOCTOU on first load.
    Two threads checking `if k not in _C` simultaneously must not both call fn().
    """
    with _CACHE_LOCK:
        yield


def predict_slots_in_use() -> int:
    """Best-effort: how many predict slots are currently held (informational)."""
    try:
        return MAX_CONCURRENT_PREDICTS - _PREDICT_SEM._value  # type: ignore[attr-defined]
    except Exception:
        return 0
