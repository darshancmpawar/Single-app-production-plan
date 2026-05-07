"""
Helpers to persist client configuration into a single massive dump file.

Concurrency model: rare writes, many reads.
  - One in-process RLock guards the cache + write path.
  - Reads always return a deep copy so callers can't mutate the cache.
  - Writes build a new dict off the cache, atomically replace the file via
    os.replace, then swap the cache pointer in one assignment. A reader
    therefore always sees either the full old state or the full new state,
    never a half-applied dict.
  - File writes go through a tempfile + fsync + os.replace so a reader during
    the write never sees truncated JSON.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from typing import Any

DIR = os.path.dirname(__file__)
MASSIVE_DUMP_PATH = os.path.join(DIR, "client_massive_dump.json")

_LOCK = threading.RLock()
_CACHE: dict[str, Any] | None = None  # canonical in-memory copy; never returned by reference


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write to a temp file in the same directory, fsync, then os.replace."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".massive_dump.", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_locked() -> dict[str, Any]:
    """Caller must hold _LOCK. Lazily fills _CACHE."""
    global _CACHE
    if _CACHE is None:
        data = _read_json(MASSIVE_DUMP_PATH)
        if "clients" not in data:
            data["clients"] = {}
        _CACHE = data
    return _CACHE


def load_massive_dump() -> dict[str, Any]:
    """Public read API. Always returns a deep copy."""
    with _LOCK:
        return copy.deepcopy(_load_locked())


def _clear_override_caches() -> None:
    global _CACHE
    with _LOCK:
        _CACHE = None


def load_logic_overrides() -> dict[str, Any]:
    clients = load_massive_dump().get("clients", {})
    return {ck: (cfg.get("logic") or {}) for ck, cfg in clients.items()}


def load_db_overrides() -> dict[str, Any]:
    clients = load_massive_dump().get("clients", {})
    return {ck: (cfg.get("database") or {}) for ck, cfg in clients.items()}


def save_client_configuration(client_key: str, config: dict[str, Any]) -> None:
    """Build new dump off-cache, atomically replace the file, then swap cache."""
    ck = client_key.strip().lower()
    with _LOCK:
        new_dump = copy.deepcopy(_load_locked())
        clients = new_dump.setdefault("clients", {})
        clients.setdefault(ck, {})
        clients[ck]["logic"] = {
            "menu_categories":          config.get("menu_categories", []),
            "star_categories":          config.get("star_categories", []),
            "nonveg_mode":              config.get("nonveg_mode", "Optional"),
            "slab_adjustments":         config.get("slab_adjustments", []),
            "additional_requirements":  config.get("additional_requirements", ""),
        }
        clients[ck]["database"] = {
            "name": config.get("client_name", ck.title()),
        }
        _atomic_write_json(MASSIVE_DUMP_PATH, new_dump)
        # Swap cache only after the file is durably renamed. If _atomic_write_json
        # raised, _CACHE keeps pointing at the prior state.
        global _CACHE
        _CACHE = new_dump
